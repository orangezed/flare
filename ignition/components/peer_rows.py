# SPDX-License-Identifier: Apache-2.0

"""Prepared paired gathers and owner-row exchange with explicit peer lifetimes."""

import torch
import torch.distributed as dist
from b12x.comm.pcie import pair_copy
from b12x.comm.pcie.pcie_dcp_topk import _tensor_from_cuda_pointer
from b12x.preparation import PreparedCall

from ignition.components.collectives import PreparedAllGather
from ignition.preparation.collectives import create_all_gather_runtime
from ignition.preparation.tuning import declare


class SharedPeerRows:
    """Mapped staging retained through graph drain; a small gather orders producers."""

    def __init__(self, group, device, *, rows, row_bytes, ipc):
        self.group, self.device, self.ipc = group, torch.device(device), ipc
        self.world, self.owner = dist.get_world_size(group), dist.get_rank(group)
        self.address = ipc.cudaMalloc(rows * row_bytes)
        handles = [None] * self.world
        dist.all_gather_object(
            handles, ipc.cudaIpcGetMemHandleBytes(self.address), group=group
        )
        self.peers = [
            self.address
            if rank == self.owner
            else ipc.cudaIpcOpenMemHandleBytes(handle)
            for rank, handle in enumerate(handles)
        ]
        self.staging = _tensor_from_cuda_pointer(
            self.address, (rows, row_bytes), dtype=torch.uint8, device=self.device
        )
        self.staging.zero_()
        self.addresses = torch.tensor(self.peers, dtype=torch.int64, device=self.device)
        self.rank = torch.tensor([self.owner], dtype=torch.int64, device=self.device)
        self.ready = create_all_gather_runtime(
            group, self.device, max_rows=1, columns=self.world * 64, ipc=ipc
        )
        self.closed = False

    def close(self):
        if self.closed:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self.ready.close()
        for rank, pointer in enumerate(self.peers):
            if rank != self.owner:
                self.ipc.cudaIpcCloseMemHandle(pointer)
        self.ipc.cudaFree(self.address)
        self.closed = True


def _declare_copy(session, query, arguments, name):
    plan = pair_copy.plan(query)
    declare(
        session,
        (
            plan.request(
                name=name,
                prepare_call=lambda state: PreparedCall(
                    run=lambda: state.run(*arguments[:4], **arguments[4]),
                    owners=arguments[:4],
                ),
            ),
        ),
    )
    return plan


class PreparedPairedGather:
    """Pack two independent fields into one existing column-gather operation."""

    def __init__(
        self,
        first,
        second,
        out_first,
        out_second,
        *,
        runtime,
        session,
        coordinator,
        name,
        buffers,
    ):
        rows, world = first.shape[0], runtime.world_size
        first_bytes, second_bytes = first.nbytes // rows, second.nbytes // rows
        self.packed = buffers.zeros((rows, first_bytes + second_bytes), torch.uint8)
        self.received = buffers.zeros(
            (rows, world * (first_bytes + second_bytes)), torch.uint8
        )
        self.out_first, self.out_second = out_first, out_second
        self.pack = _declare_copy(
            session,
            pair_copy.Query("pack", rows, first_bytes, second_bytes, world),
            (first, second, self.packed, None, {}),
            name + ".pack",
        )
        self.gather = PreparedAllGather(
            self.packed.view(torch.bfloat16),
            self.received.view(torch.bfloat16),
            runtime=runtime,
            session=session,
            coordinator=coordinator,
            name=name + ".gather",
        )
        self.unpack = _declare_copy(
            session,
            pair_copy.Query("unpack", rows, first_bytes, second_bytes, world),
            (self.received, None, out_first, out_second, {}),
            name + ".unpack",
        )

    def __call__(self, first, second):
        pair_copy.run(first, second, self.packed, plan=self.pack)
        self.gather(self.packed.view(torch.bfloat16))
        pair_copy.run(
            self.received, None, self.out_first, self.out_second, plan=self.unpack
        )
        return self.out_first, self.out_second


class PreparedOwnerRows:
    """Each rank receives all column shards for its strided query rows.

    The caller must execute a collective after consuming the result and before
    reusing shared staging. Sparse index selection's position gather provides
    that join. No other graph may reuse this channel concurrently.
    """

    def __init__(
        self,
        first,
        second,
        out_first,
        out_second,
        *,
        shared,
        session,
        coordinator,
        name,
        buffers,
    ):
        rows, world = first.shape[0], shared.world
        first_bytes, second_bytes = first.nbytes // rows, second.nbytes // rows
        self.shared = shared
        self.staging = shared.staging[:rows]
        self.out_first, self.out_second = out_first, out_second
        if (
            self.staging.shape != (rows, first_bytes + second_bytes)
            or first_bytes + second_bytes < 128
        ):
            raise ValueError(
                "owner-row staging must match both fields and contain a readiness row"
            )
        self.pack = _declare_copy(
            session,
            pair_copy.Query("pack", rows, first_bytes, second_bytes, world),
            (first, second, self.staging, None, {}),
            name + ".pack",
        )
        self.ready_source = self.staging[:1, :128].view(torch.bfloat16)
        self.ready_output = buffers.zeros((1, world * 64), torch.bfloat16)
        self.barrier = PreparedAllGather(
            self.ready_source,
            self.ready_output,
            runtime=shared.ready,
            session=session,
            coordinator=coordinator,
            name=name + ".ready",
        )
        self.pull = _declare_copy(
            session,
            pair_copy.Query("pull", rows, first_bytes, second_bytes, world),
            (
                self.staging,
                None,
                out_first,
                out_second,
                dict(peers=shared.addresses, rank=shared.rank, ready=self.ready_output),
            ),
            name + ".pull",
        )

    def __call__(self, first, second):
        pair_copy.run(first, second, self.staging, plan=self.pack)
        ready = self.barrier(self.ready_source)
        pair_copy.run(
            self.staging,
            None,
            self.out_first,
            self.out_second,
            plan=self.pull,
            peers=self.shared.addresses,
            rank=self.shared.rank,
            ready=ready,
        )
        return self.out_first, self.out_second
