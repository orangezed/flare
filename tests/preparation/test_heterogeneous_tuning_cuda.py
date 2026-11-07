# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Common GEMMs share TP tuning while unequal attention heads remain local."""

import multiprocessing
import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires two explicitly assigned CUDA GPUs")


def _rank(directory, rank, connection):
    from datetime import timedelta
    import torch
    import torch.distributed as dist
    from b12x.preparation.session import PreparationJob
    from ignition.components.collectives import PreparedAllReduce
    from ignition.components.linear import UnquantizedProjection
    from ignition.components.varlen_attention import PreparedVarlenAttention
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=(Path(directory) / "group").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    coordinator = CollectivePreparation(dist.group.WORLD)
    runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=4, hidden_size=512)
    generator = torch.Generator().manual_seed(4187)
    source = torch.randn((4, 768), generator=generator, dtype=torch.bfloat16).to(device)
    weight = torch.randn((512, 768), generator=generator, dtype=torch.bfloat16).to(device)
    output = torch.empty((4, 512), dtype=torch.bfloat16, device=device)
    q, k, v = [torch.randn((128, rank + 1, 128), generator=generator,
                           dtype=torch.bfloat16).to(device) for _ in range(3)]
    attended = torch.empty_like(q)
    cu = torch.tensor([0, 128], dtype=torch.int32, device=device)
    reduced_source = torch.full((4, 512), rank + 1, dtype=torch.bfloat16, device=device)
    reduced = torch.empty_like(reduced_source)
    trials = []
    original = PreparationJob._trial

    def record_trial(job, obligation, index, assignment, config):
        trials.append((obligation.request.name, index, job.session._tuning_ranks))
        return (yield from original(job, obligation, index, assignment, config))

    PreparationJob._trial = record_trial
    reports = []
    try:
        for repetition in range(2):
            session = RecordedPreparationSession(device=device, checkpoint="heterogeneous-tp2",
                workload_schema="common-gemm-local-heads", numerical_policy="bf16",
                namespace={"case": directory}, coordinator=coordinator, compile_workers=4)
            session.cache_dir = Path(directory) / "selections"
            arena = ScratchArena(device)
            session.register_scratch(arena)
            attention = PreparedVarlenAttention(q, k, v, cu, attended, session=session,
                                               workspace=arena, name="local_attention")
            projection = UnquantizedProjection(source, weight, output,
                                               session=session, name="common_gemm")
            reduction = PreparedAllReduce(reduced_source, reduced, runtime=runtime,
                                          session=session, coordinator=coordinator, name="reduce")
            session.tune(show_progress=False)
            assert session._tuning_ranks == (rank,)
            chosen = {item["name"]: item for item in session.tuning_report["selections"]}
            reports.append(dict(
                common=chosen["common_gemm"], local=chosen["local_attention"],
                local_query=attention.plan.contract.encode_query(attention.plan.query),
                trial_count=len(trials),
            ))
            projection(source)
            attention()
            reduction(reduced_source)
            torch.cuda.synchronize()
            torch.testing.assert_close(output.float(), source.float() @ weight.float().T,
                                       rtol=.01, atol=.25)
            torch.testing.assert_close(reduced, torch.full_like(reduced, 3), rtol=0, atol=0)
            expected = output.clone(), attended.clone()
            graph = torch.cuda.CUDAGraph()
            with runtime.capture(), session.capture(), torch.cuda.graph(graph):
                projection(source)
                attention()
                reduction(reduced_source)
            session.freeze()
            for _ in range(4):
                arena.tensor.fill_(255)
                for destination in (output, attended, reduced):
                    destination.fill_(torch.nan)
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
                torch.testing.assert_close(output, expected[0], rtol=0, atol=0)
                torch.testing.assert_close(attended, expected[1], rtol=0, atol=0)
                torch.testing.assert_close(reduced, torch.full_like(reduced, 3), rtol=0, atol=0)
            graph.reset()
            session.close()
        connection.send(dict(reports=reports, trials=trials))
    finally:
        PreparationJob._trial = original
        runtime.close()
        dist.destroy_process_group()
        connection.close()


def test_common_gemm_and_unequal_heads_prepare_and_replay_on_two_gpus(tmp_path):
    import torch

    if torch.cuda.device_count() != 2:
        pytest.skip("expose exactly the two assigned CUDA devices")
    context = multiprocessing.get_context("spawn")
    processes, connections = [], []
    try:
        for rank in range(2):
            root, peer = context.Pipe(duplex=False)
            process = context.Process(target=_rank, args=(str(tmp_path), rank, peer))
            process.start()
            peer.close()
            processes.append(process)
            connections.append(root)
        for process in processes:
            process.join(timeout=180)
        assert [process.exitcode for process in processes] == [0, 0]
        outcomes = [connection.recv() for connection in connections]
        for rank, outcome in enumerate(outcomes):
            common = [item for item in outcome["trials"] if item[0] == "common_gemm"]
            local = [item for item in outcome["trials"] if item[0] == "local_attention"]
            assert common and local
            assert all(item[2] == (0, 1) for item in common)
            assert all(item[2] == (rank,) for item in local)
            assert outcome["reports"][0]["trial_count"] == outcome["reports"][1]["trial_count"]
            assert outcome["reports"][1]["common"]["source"] == "cached"
            assert outcome["reports"][1]["local"]["source"] == "cached"
        for repetition in range(2):
            first, second = (item["reports"][repetition] for item in outcomes)
            assert first["common"]["config"] == second["common"]["config"]
            assert first["local_query"] != second["local_query"]
        print("heterogeneous TP2 selections", outcomes, flush=True)
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
        for connection in connections:
            connection.close()
