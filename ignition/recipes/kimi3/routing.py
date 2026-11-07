# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 decode routing: latent-down and router shards gathered with expert selection.

Each rank projects its rows of the routed-expert down projection and its experts' router logits.
One b12x exchange gathers both shards; a single decode row also selects Kimi's 16 experts inside
that exchange, and two to eight rows select them with b12x's prepared top-16 afterwards. This
replaces the router all-gather, the separate top-k, and the latent all-reduce of the general path.
"""

import itertools
import weakref

import torch
from b12x.preparation import CollectiveRequirement, PreparedCall

from ignition.preparation.tuning import declare

MAX_ROWS = 8
_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


class PreparedLatentShardGather:
    """Gather latent-output column partitions and FP32 router logits in chunks.

    Each rank owns output columns of both projections. Router bytes travel in
    BF16 views without conversion. Serial layers reuse the supplied send and
    receive buffers; each chunk is consumed before the next overwrites them.
    """

    def __init__(self, local_down, local_router, latent, logits, *, runtime,
                 session, coordinator, name, buffers, allocator=None):
        from ignition.components.collectives import PreparedAllGather

        rows, self.down_columns = local_down.shape
        self.world = runtime.world_size
        self.router_columns = local_router.shape[1]
        self.columns = self.down_columns + 2 * self.router_columns
        if (local_down.dtype != torch.bfloat16 or local_router.dtype != torch.float32
                or latent.shape != (rows, self.world * self.down_columns)
                or logits.shape != (rows, self.world * self.router_columns)
                or latent.dtype != torch.bfloat16 or logits.dtype != torch.float32
                or local_router.shape[0] != rows
                or not all(value.is_contiguous() and value.device == local_down.device
                           for value in (local_down, local_router, latent, logits))):
            raise ValueError("latent routing requires matching contiguous BF16 latent and FP32 router partitions")
        self.local_down, self.local_router = local_down, local_router
        self.latent, self.logits = latent, logits
        chunk_rows = min(rows, runtime.max_batch_size)

        def allocate(field, columns):
            if field not in buffers or buffers[field].shape[0] < chunk_rows:
                shape = (chunk_rows, columns)
                buffers[field] = (torch.zeros(shape, dtype=torch.bfloat16, device=local_down.device)
                                  if allocator is None else allocator.zeros(shape, torch.bfloat16))
            return buffers[field][:chunk_rows]

        self.send = allocate("send", self.columns)
        self.received = allocate("received", self.columns * self.world)
        self.bounds = [(begin, min(rows, begin + chunk_rows)) for begin in range(0, rows, chunk_rows)]
        self.parts = [PreparedAllGather(self.send[:end-begin], self.received[:end-begin],
            runtime=runtime, session=session, coordinator=coordinator, name=f"{name}.{index}")
            for index, (begin, end) in enumerate(self.bounds)]

    def __call__(self):
        for (begin, end), part in zip(self.bounds, self.parts, strict=True):
            count = end - begin
            self.send[:count, :self.down_columns].copy_(self.local_down[begin:end])
            self.send[:count, self.down_columns:].copy_(self.local_router[begin:end].view(torch.bfloat16))
            part(self.send[:count])
            received = self.received[:count].view(count, self.world, self.columns)
            self.latent[begin:end].view(count, self.world, self.down_columns).copy_(
                received[:, :, :self.down_columns])
            self.logits[begin:end].view(count, self.world, self.router_columns).view(torch.bfloat16).copy_(
                received[:, :, self.down_columns:])
        return self.latent


def create_route_runtime(group, device, *, ipc=None):
    """Allocate the b12x exchange that carries 1-8 rows of K3 latent and router shards."""
    import torch.distributed as dist
    from b12x.comm.pcie import DcpAllToAll

    world_size = dist.get_world_size(group)
    if world_size < 2 or 3584 % world_size or 896 % world_size:
        raise ValueError("K3 route exchange requires an even latent and expert partition")
    return DcpAllToAll.from_exchange_group(exchange_group=group, device=device, max_batch_size=MAX_ROWS,
        total_heads=world_size, head_dim=8, query_head_dim=(3584 * 2 + 896 * 4) // world_size,
        stream_affine=False, ipc=ipc)


class PreparedRouteGather:
    """Gather latent and FP32 router shards into full rows, then select 16 experts per row.

    The latent, expert weights and expert IDs are caller-owned outputs read by the routed experts.
    Every rank must declare the same operation names and replay them serially.
    """

    def __init__(self, local_down, local_router, correction_bias, latent, weights, ids, *, runtime, session,
                 coordinator, name):
        import torch.distributed as dist
        from b12x.comm import pcie

        rows = local_down.shape[0]
        world = runtime.world_size
        expected = ((local_down, (rows, 3584 // world), torch.bfloat16), (local_router, (rows, 896 // world), torch.float32),
                    (correction_bias, (896,), torch.float32), (latent, (rows, 3584), torch.bfloat16),
                    (weights, (rows, 16), torch.float32), (ids, (rows, 16), torch.int32))
        wrong = [(tuple(t.shape), t.dtype, t.is_contiguous(), shape, dtype) for t, shape, dtype in expected
                 if t.shape != shape or t.dtype != dtype or not t.is_contiguous()]
        if not 1 <= rows <= MAX_ROWS or wrong:
            raise ValueError(f"K3 route gather requires contiguous 1-8 row latent, router and route tensors: {wrong}")
        self.runtime, self.rows = runtime, rows
        self.local_down, self.local_router, self.correction_bias = local_down, local_router, correction_bias
        self.latent, self.weights, self.ids = latent, weights, ids
        ranks = tuple(dist.get_process_group_ranks(runtime.exchange_group))
        requests = []
        if rows == 1:
            self.fused = dict(local_down=local_down, local_router=local_router, correction_bias=correction_bias,
                              out_down=latent, topk_weights=weights, topk_ids=ids)
            self.plan = pcie.plan(pcie.query_from_runtime(runtime, surface="DcpAllToAll.all_gather_pair_kimi_topk",
                                                          call=self.fused), runtime=runtime)
            requests.append(self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(**self.fused), output=latent, owners=(runtime, latent, weights, ids)),
                collective=CollectiveRequirement(key=name, ranks=ranks)))
        else:
            self.logits = torch.zeros((rows, 896), dtype=torch.float32, device=latent.device)
            self.pair = dict(local_first=local_down, local_second=local_router, out_first=latent,
                             out_second=self.logits, threads=512)
            self.select = dict(router_logits=self.logits, correction_bias=correction_bias,
                               output_weights=weights, output_ids=ids)
            self.plan = pcie.plan(pcie.query_from_runtime(runtime, surface="DcpAllToAll.all_gather_pair",
                                                          call=self.pair), runtime=runtime)
            self.select_plan = pcie.plan(pcie.query_from_runtime(runtime, surface="DcpAllToAll.kimi_topk16",
                                                                 call=self.select), runtime=runtime)
            requests.append(self.plan.request(name=name + "/gather", prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(**self.pair), output=latent, owners=(runtime, latent, self.logits)),
                collective=CollectiveRequirement(key=name + "/gather", ranks=ranks)))
            requests.append(self.select_plan.request(name=name + "/select", prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(**self.select), output=weights, owners=(runtime, self.logits, weights, ids))))
        declare(session, tuple(requests), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def __call__(self):
        _route_gather(self.local_down, self.local_router, self.latent, self.weights, self.ids, self.handle)
        return self.latent


@torch.library.custom_op("flare::kimi3_route_gather", mutates_args=("latent", "weights", "ids"))
def _route_gather(local_down: torch.Tensor, local_router: torch.Tensor, latent: torch.Tensor,
                  weights: torch.Tensor, ids: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    runtime = owner.runtime
    if owner.rows == 1:
        runtime.all_gather_pair_kimi_topk(local_down, local_router, owner.correction_bias, latent, weights, ids,
                                          plan=owner.plan)
        return
    runtime.all_gather_pair(local_down, local_router, latent, owner.logits, plan=owner.plan, threads=512)
    runtime.kimi_topk16(owner.logits, owner.correction_bias, weights, ids, plan=owner.select_plan)


@_route_gather.register_fake
def _fake_route_gather(local_down, local_router, latent, weights, ids, handle):
    pass
