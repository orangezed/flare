"""Exact latent/router transport through prepared TP16 prefill graphs."""

from datetime import timedelta
import os
from pathlib import Path
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_TP_CUDA_TESTS") != "1",
                               reason="requires sixteen assigned CUDA devices")


def _fields(rows, rank, variation, device):
    latent = (torch.arange(rows * 224, device=device, dtype=torch.int32)
              + 257 * rank + 101 * variation).to(torch.int16).view(rows, 224).view(torch.bfloat16)
    router = (torch.arange(rows * 56, device=device, dtype=torch.int32)
              * 1_048_583 + 1_069_547_520 + 193 * rank + variation).view(rows, 56).view(torch.float32)
    return latent, router


def _worker(rank, rendezvous):
    from ignition.preparation.collectives import CollectivePreparation, create_head_exchange
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.routing import PreparedLatentShardGather

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=16, init_method=rendezvous,
                            timeout=timedelta(minutes=10))
    runtime = create_head_exchange(dist.group.WORLD, device, max_rows=1024, heads=16, head_dim=336)
    coordinator = CollectivePreparation(dist.group.WORLD)
    session = RecordedPreparationSession(device=device, checkpoint="latent-router-transport",
        workload_schema="kimi3-prefill-gather", numerical_policy="default", compile_workers=16,
        coordinator=coordinator)
    buffers, cases, graphs, compiled = {}, [], [], []
    try:
        for rows in (4096, 1031, 1024, 128):
            local_down, local_router = _fields(rows, rank, 0, device)
            latent = torch.empty((rows, 3584), dtype=torch.bfloat16, device=device)
            logits = torch.empty((rows, 896), dtype=torch.float32, device=device)
            operation = PreparedLatentShardGather(local_down, local_router, latent, logits,
                runtime=runtime, session=session, coordinator=coordinator,
                name=f"gather-{rows}", buffers=buffers)
            cases.append((rows, local_down, local_router, latent, logits, operation))
        session.tune(show_progress=False)
        for *_, operation in cases:
            operation()
        torch.cuda.synchronize()
        for *_, operation in cases:
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with session.capture(), torch.cuda.graph(graph):
                operation()
            graphs.append(graph)
        for index in (0, 3):
            operation = compile_forward(cases[index][-1])
            compiled.append(operation)
            operation()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with session.capture(), torch.cuda.graph(graph):
                operation()
            graphs.append(graph)
        case_indices = (0, 1, 2, 3, 0, 3)
        # All variants share one bounded 1024-row staging pair.
        assert len({operation.send.data_ptr() for *_, operation in cases}) == 1
        assert len({operation.received.data_ptr() for *_, operation in cases}) == 1
        assert buffers["send"].numel() * 2 + buffers["received"].numel() * 2 == 11_698_176
        for variation in range(3):
            for index in (0, 3, 1, 2, 0, 4, 5, 4):
                rows, local_down, local_router, latent, logits, operation = cases[case_indices[index]]
                expected_down = torch.cat([_fields(rows, peer, variation, device)[0]
                                           for peer in range(16)], dim=1)
                expected_router = torch.cat([_fields(rows, peer, variation, device)[1]
                                             for peer in range(16)], dim=1)
                down, router = _fields(rows, rank, variation, device)
                local_down.copy_(down)
                local_router.copy_(router)
                latent.fill_(float("nan"))
                logits.fill_(float("nan"))
                operation.send.fill_(float("nan"))
                operation.received.fill_(float("nan"))
                before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                for _ in range(3):
                    graphs[index].replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
                assert torch.equal(latent.view(torch.int16), expected_down.view(torch.int16))
                assert torch.equal(logits.view(torch.int32), expected_router.view(torch.int32))
                assert torch.equal(local_down.view(torch.int16), down.view(torch.int16))
                assert torch.equal(local_router.view(torch.int32), router.view(torch.int32))
        dist.barrier()
        if rank == 0:
            print("TP16 latent/router gather: exact fields, partial chunks and shared-scratch replay passed", flush=True)
    finally:
        for graph in graphs:
            graph.reset()
        torch.cuda.synchronize()
        session.close()
        runtime.close()
        dist.destroy_process_group()


def test_latent_router_prefill_gather():
    if torch.cuda.device_count() != 16:
        pytest.skip("requires exactly sixteen visible GPUs")
    with tempfile.TemporaryDirectory(prefix="kimi-latent-gather-") as directory:
        mp.spawn(_worker, args=(Path(directory, "gloo").as_uri(),), nprocs=16, join=True)
