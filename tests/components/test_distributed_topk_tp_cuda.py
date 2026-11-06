"""All context ranks must select the same token set, including cutoff ties."""
from datetime import timedelta
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(
    os.getenv("FLARE_RUN_TP_CUDA_TESTS") != "1", reason="requires sixteen assigned CUDA GPUs")


def _rank(rank, world, rendezvous):
    from ignition.components.distributed_topk import PreparedDistributedTopK
    from ignition.preparation.collectives import CollectivePreparation, create_all_gather_runtime
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device, topk = torch.device("cuda", rank), 2048
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=10))
    with PackageStorage(rank=rank, world=world, device=rank) as storage:
        coordinator = CollectivePreparation(dist.group.WORLD)
        runtime = create_all_gather_runtime(dist.group.WORLD, device, max_rows=32,
            columns=4*topk*world, ipc=storage.peer_storage)
        with RecordedPreparationSession(device=device, checkpoint="distributed-candidate-fixture",
                workload_schema="exact-global-tied-sets", numerical_policy="fp32",
                coordinator=coordinator, compile_workers=16) as session:
            workspace, buffers = ScratchArena(device), GraphBufferPool(device, chunk_bytes=2 << 20)
            session.register_scratch(workspace)
            rank_tensor = torch.tensor([rank], dtype=torch.int64, device=device)
            calls = []
            for rows in (1, 4, 32):
                indices = torch.full((rows, topk), -1, dtype=torch.int32, device=device)
                scores = torch.zeros((rows, topk), dtype=torch.float32, device=device)
                output = torch.empty_like(indices)
                operation = PreparedDistributedTopK(indices, scores, output, world=world,
                    rank=rank_tensor, runtime=runtime, session=session, coordinator=coordinator,
                    name=f"global_topk_rows{rows}", buffers=buffers.cursor(), workspace=workspace)
                calls.append((indices, scores, output, operation))
            session.tune(show_progress=False)
            graphs = []
            for _, _, _, operation in calls:
                compiled = compile_forward(operation)
                compiled()
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    compiled()
                graphs.append(graph)
            session.freeze()
            for turn in range(2):
                for slot in ((0, 1, 2) if turn == 0 else (2, 0, 1)):
                    indices, scores, output, operation = calls[slot]
                    rows = indices.shape[0]
                    for total in (0, 221, 2048, 2049, 4096, 32768):
                        for tied in (True, False):
                            indices.fill_(-1)
                            scores.fill_(-torch.inf)
                            expected = torch.full((rows, topk), -1, dtype=torch.int32)
                            for row in range(rows):
                                length = max(0, total-row*3)
                                count = min(topk, max(0, (length+world-1-rank)//world))
                                indices[row, :count] = torch.arange(count, device=device)
                                global_ids = torch.arange(count, device=device)*world+rank
                                scores[row, :count] = 1. if tied else (global_ids % 7).float()
                                if rank == 0:
                                    ids = torch.cat([torch.arange(peer, length, world)
                                        if peer < length else torch.empty(0, dtype=torch.int64)
                                        for peer in range(world)])
                                    values = torch.ones_like(ids) if tied else ids % 7
                                    chosen = values.argsort(descending=True, stable=True)[:topk]
                                    expected[row, :chosen.numel()] = ids[chosen].int()
                            dist.broadcast(expected, src=0)
                            operation.merge.scratch.fill_(255)
                            output.fill_(-777)
                            dist.barrier()
                            before = torch.cuda.memory_stats()["allocation.all.allocated"]
                            for _ in range(3):
                                graphs[slot].replay()
                            torch.cuda.synchronize()
                            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                            torch.testing.assert_close(output.cpu().sort(1).values,
                                expected.sort(1).values, atol=0, rtol=0,
                                msg=lambda message: f"rank={rank} rows={rows} total={total}: {message}")
                    if rank == 0:
                        print(f"TP{world} rows={rows} turn={turn}: exact global tie sets passed", flush=True)
            for graph in graphs:
                graph.reset()
        dist.barrier()
        runtime.close()
    dist.destroy_process_group()


def test_identical_global_tie_sets_across_context_ranks(tmp_path):
    world = torch.cuda.device_count()
    assert world == 16
    mp.spawn(_rank, args=(world, (tmp_path / "rendezvous").as_uri()), nprocs=world, join=True)
