# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collective warmup waits for every rank to finish the preceding capture."""

import multiprocessing
import pytest


def capture_rank(rank, world, store, finished, outcome, retain):
    from contextlib import contextmanager, nullcontext
    from datetime import timedelta
    from types import SimpleNamespace
    import time
    import traceback

    import torch.distributed as dist

    from ignition.preparation import capture
    from ignition.preparation.collectives import CollectivePreparation

    try:
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
        coordinator = CollectivePreparation(dist.group.WORLD, timeout=timedelta(seconds=10))
        created = []

        class Graph:
            def __init__(self, **kwargs):
                self.index = len(created)
                created.append(self)

            def instantiate(self):
                assert not retain, "export must defer executable creation"
                if rank == world - 1 and self.index == 0:
                    time.sleep(.3)
                finished[rank * 2 + self.index] = 1

        @contextmanager
        def capture_context(graph, **kwargs):
            yield
            if retain:
                if rank == world - 1 and graph.index == 0:
                    time.sleep(.3)
                finished[rank * 2 + graph.index] = 1
            else:
                graph.instantiate()

        def first():
            pass

        def second():
            assert all(finished[peer * 2] for peer in range(world))

        stream = SimpleNamespace(wait_stream=lambda other: None, synchronize=lambda: None)
        capture.torch.cuda.current_stream = lambda device: stream
        capture.torch.cuda.stream = lambda stream: nullcontext()
        capture.torch.cuda.synchronize = lambda device: None
        capture.torch.cuda.CUDAGraph = Graph
        capture.torch.cuda.graph = capture_context
        functions = {"first": first, "second": second}
        graphs, _ = capture.capture_forwards(functions,
            {name: SimpleNamespace(kind="decode") for name in functions},
            session=SimpleNamespace(capture=nullcontext), device="cpu", stream=stream, pool=None,
            coordinator=coordinator, compile_decoder=False, retain_graphs=retain)
        outcome.put((rank, list(graphs)))
    except Exception:
        outcome.put((rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("retain", [False, True])
def test_warmup_waits_for_slow_peer_capture_finalization(tmp_path, retain):
    context = multiprocessing.get_context("fork")
    world = 3
    finished = context.Array("i", world * 2, lock=False)
    outcome = context.Queue()
    processes = [context.Process(target=capture_rank,
        args=(rank, world, tmp_path / "store", finished, outcome, retain)) for rank in range(world)]
    for process in processes:
        process.start()
    results = dict(outcome.get(timeout=30) for _ in processes)
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    assert results == {rank: ["first", "second"] for rank in range(world)}
    assert all(finished)
