# SPDX-License-Identifier: Apache-2.0

"""Context-gathered sparse prefill against the distributed path and independent math."""

from dataclasses import replace
from datetime import timedelta
import json
import hashlib
import os
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_TP_CUDA_TESTS") != "1", reason="requires assigned TP GPUs")


def _rank(rank, rendezvous, world, rows, compiled_case, destination):
    from ignition.components.fp8_indexer import _write_keys
    from ignition.components.paged_batch import Metadata
    from ignition.preparation.collectives import CollectivePreparation, create_all_gather_runtime, create_head_exchange
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.dcp import ContextParallel, ContextShards, write_records
    from ignition.recipes.kimi3.dsa import PreparedDSA

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=15))
    torch.manual_seed(3189)
    maximum, heads, topk = 1048576, 96, 2048
    block_count = maximum // world // 128
    def random(*shape):
        return (torch.randn(shape, device=device) * .2).bfloat16()
    weights = dict(index_query=random(4096, 1536), index_norm=torch.ones(128, device=device, dtype=torch.bfloat16),
                   index_bias=torch.zeros(128, device=device, dtype=torch.bfloat16))
    residual, key_input, head_weights = random(rows, 1536), random(rows, 128), random(rows, 32)
    all_query, all_latent, all_keys = random(rows, heads, 576), random(maximum, 576), random(maximum, 128)
    local_heads = heads // world
    query = all_query[:, rank * local_heads:(rank + 1) * local_heads].contiguous()
    current_latent = random(rows, 576)
    slots = torch.full((rows,), -1, device=device, dtype=torch.int64)
    metadata = Metadata.allocate(rows, 1, block_count, device)
    physical_pages = torch.arange(block_count - 1, -1, -1, device=device, dtype=torch.int32)
    metadata.page_table[0].copy_(physical_pages)
    shards = ContextShards.on(rank, world, device)
    outcomes = []
    with PackageStorage(rank=rank, world=world, device=rank) as storage:
        latent = storage.device_allocate("latent", (block_count, 128, 576), torch.float8_e4m3fn,
                                        group="KV_PAGES", maximum=block_count, units=1)
        index = storage.device_allocate("index", (block_count * 2, 64 * 132), torch.uint8,
                                       group="KV_INDEX", maximum=block_count * 2, units=1)
        storage.grow("KV_PAGES", block_count)
        storage.grow("KV_INDEX", block_count * 2)
        regions = {x.address: x for x in storage.dynamic_regions}
        pointer_sets = storage.peer_storage.share_regions(dist.group.WORLD,
            [regions[latent.data_ptr()], regions[index.data_ptr()]])
        peers, index_peers = [torch.tensor(p, dtype=torch.int64, device=device) for p in pointer_sets]
        coordinator = CollectivePreparation(dist.group.WORLD)
        candidates = create_all_gather_runtime(dist.group.WORLD, device, max_rows=32,
            columns=4 * topk * world, ipc=storage.peer_storage)
        selections = create_all_gather_runtime(dist.group.WORLD, device, max_rows=min(32, (rows+world-1)//world),
            columns=2 * topk * world, ipc=storage.peer_storage)
        exchange = create_head_exchange(dist.group.WORLD, device, max_rows=32, heads=heads,
            head_dim=512, query_head_dim=576, ipc=storage.peer_storage)
        slab = torch.zeros((maximum, 576), dtype=torch.float8_e4m3fn, device=device)
        index_slab = torch.zeros((maximum // 64, 64 * 132), dtype=torch.uint8, device=device)
        context = ContextParallel(shards, exchange, coordinator, {}, slab, candidates=candidates,
                                  index_slab=index_slab, selections=selections)
        with RecordedPreparationSession(device=device, checkpoint=f"kimi-gathered-tp{world}",
                workload_schema=f"sparse-prefill-{rows}", numerical_policy="fp8", coordinator=coordinator,
                compile_workers=16) as session:
            arena = ScratchArena(device)
            session.register_scratch(arena)
            ops = {}
            for layout in ("dcp", "query_split"):
                output = torch.zeros((rows, local_heads, 512), dtype=torch.bfloat16, device=device)
                op = PreparedDSA(weights, residual, key_input, head_weights, query, output, slots, metadata,
                    latent, index, context=replace(context, prefill_layout=layout), session=session,
                    coordinator=coordinator, name="dsa." + layout, workspace=arena, buffers={},
                    mode="prefill", latent=current_latent, peers=peers, index_peers=index_peers)
                ops[layout] = (op, output)
            session.tune(show_progress=False)
            local_positions = torch.arange(maximum // world, device=device)
            local_slots = physical_pages.long()[local_positions // 128] * 128 + local_positions % 128
            def set_inputs(length, count):
                latent.zero_()
                index.zero_()
                write_records(all_latent[rank::world].contiguous(), local_slots, latent)
                torch.library.wrap_triton(_write_keys)[(maximum//world,)](all_keys[rank::world].contiguous(),
                    index.view(torch.float8_e4m3fn), index.view(torch.float32), local_slots,
                    index.stride(0), False, num_warps=4)
                metadata.starts.fill_(length - count)
                metadata.lengths.fill_(count)
                metadata.query_start_loc[0] = 0
                metadata.query_start_loc[1] = count
                metadata.positions.fill_(-1)
                metadata.positions[:count].copy_(torch.arange(length-count, length, device=device))
                slots.copy_(shards.token_slots(metadata.positions, metadata.token_page_table))
                write_records(current_latent, slots, latent)
                torch.cuda.synchronize()
                dist.barrier()
            set_inputs(32768, rows)
            functions = {layout: item[0] for layout, item in ops.items()}
            signatures = {name: GraphSignature("prefill", 1, rows, ()) for name in functions}
            graphs, retained_forwards = capture_forwards(functions, signatures,
                session=session, device=device, stream=torch.cuda.Stream(device=device),
                pool=storage.graph_pool.id, coordinator=coordinator,
                compile_decoder=compiled_case, warmup_storage=storage,
                progress=(lambda text: print(text, flush=True)) if rank == 0 else None)
            session.freeze()
            for length, count in ((4096, rows), (32768, rows), (126976, rows), (maximum, rows), (2051, min(rows, 17)), (0, 0)):
                set_inputs(length, count)
                outputs, measurements = {}, {}
                for layout in ("dcp", "query_split", "query_split", "dcp"):
                    slab.fill_(float("nan"))
                    index_slab.fill_(255)
                    ops[layout][1].fill_(float("nan"))
                    dist.barrier()
                    before = torch.cuda.memory_stats()["allocation.all.allocated"]
                    graphs[layout].replay()
                    torch.cuda.synchronize()
                    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                    actual = ops[layout][1].float().clone()
                    assert torch.isfinite(actual).all()
                    assert not actual[count:].count_nonzero()
                    outputs[layout] = actual
                    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    dist.barrier()
                    start.record()
                    for _ in range(3):
                        graphs[layout].replay()
                    stop.record()
                    stop.synchronize()
                    samples = [None] * world
                    dist.all_gather_object(samples, start.elapsed_time(stop) / 3)
                    measurements.setdefault(layout, []).append(max(samples))
                if count:
                    for layout in ("query_split",):
                        torch.testing.assert_close(outputs[layout], outputs["dcp"], rtol=.03, atol=.002,
                            msg=lambda s: f"rank={rank} length={length} rows={count} layout={layout}: {s}")
                    # Independent index scores and FP32 softmax for two rows exercise
                    # causal visibility and the current chunk, not only mutual agreement.
                    norm_key = torch.nn.functional.layer_norm(key_input.float(), (128,),
                        weights["index_norm"].float(), weights["index_bias"].float(), 1e-5).bfloat16()
                    keys = all_keys[:length].clone()
                    keys[length-count:] = norm_key[:count]
                    values = all_latent[:length].clone()
                    values[length-count:] = current_latent[:count]
                    def quantize(value, floor):
                        v = value.float()
                        scale = torch.exp2(torch.ceil(torch.log2(v.abs().amax(-1).clamp_min(floor)/448)))
                        return (v / scale[..., None]).to(torch.float8_e4m3fn).float(), scale
                    qi = torch.nn.functional.linear(residual, weights["index_query"]).view(rows,32,128)
                    q, qs = quantize(qi, 1e-10)
                    k, ks = quantize(keys, 1e-4)
                    for row in sorted({0, count-1}):
                        visible = length - count + row + 1
                        scores = torch.einsum("hd,kd->hk", q[row], k[:visible]*ks[:visible,None]).relu()
                        scores = (scores*(head_weights[row].float()*qs[row]*(128*32)**-.5)[:,None]).sum(0)
                        ids = scores.topk(min(topk,visible)).indices
                        kv = values.to(torch.float8_e4m3fn).float()[ids]
                        logits = torch.einsum("hd,kd->hk", query[row].to(torch.float8_e4m3fn).float(), kv)*192**-.5
                        expected = logits.softmax(-1) @ kv[:,:512]
                        torch.testing.assert_close(outputs["query_split"][row], expected, rtol=.03, atol=.002)
                result = dict(context=length, active_rows=count, capacity=rows, compiled=compiled_case,
                              milliseconds=measurements)
                outcomes.append(result)
                if rank == 0:
                    print(json.dumps(result), flush=True)
            set_inputs(32768, rows)
            head_weights.zero_()
            hashes = set()
            for _ in range(10):
                graphs["query_split"].replay()
                torch.cuda.synchronize()
                selected = ops["query_split"][0].prefill.gathered.view(-1, topk)
                ordered = selected.sort(1).values
                assert (ordered >= 0).all() and (ordered < 32768).all()
                assert (ordered[:, 1:] != ordered[:, :-1]).all()
                digest = hashlib.sha256(selected.cpu().numpy().tobytes()).hexdigest()
                peer_hashes = [None] * world
                dist.all_gather_object(peer_hashes, digest)
                assert len(set(peer_hashes)) == 1, "Ranks must consume the same selected IDs even at score ties"
                hashes.add(digest)
            if rank == 0:
                print(json.dumps(dict(tied_score_replays=10, capacity=rows, compiled=compiled_case,
                                      distinct_tie_selections=len(hashes), rank_agreement=True)), flush=True)
            for graph in graphs.values():
                graph.reset()
        selections.close()
        candidates.close()
        exchange.close()
    dist.destroy_process_group()
    if rank == 0:
        destination.write_text(json.dumps(outcomes, indent=2))


@pytest.mark.parametrize("rows,compiled_case", [(128, False), (1024, False), (4096, False), (128, True), (4096, True)],
                         ids=["eager128", "eager1024", "eager4096", "compiled128", "compiled4096"])
def test_gathered_sparse_prefill(tmp_path, rows, compiled_case):
    world = torch.cuda.device_count()
    assert world in (4, 8, 16)
    mp.spawn(_rank, args=((tmp_path/"gloo").as_uri(), world, rows, compiled_case, tmp_path/"timings.json"),
             nprocs=world, join=True)
