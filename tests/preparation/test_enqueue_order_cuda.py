# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compiled native calls retain their asynchronous enqueue order and source storage."""

import os

import pytest
import torch

from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")
_calls = []


@torch.library.custom_op("flare_test::enqueue_fork", mutates_args=("output",))
def _fork(source: torch.Tensor, output: torch.Tensor, stream: int) -> None:
    _calls.append(("fork", source.data_ptr(), output.data_ptr()))
    auxiliary = torch.cuda.ExternalStream(stream, device=source.device)
    auxiliary.wait_stream(torch.cuda.current_stream(source.device))
    with torch.cuda.stream(auxiliary):
        torch.cuda._sleep(200000)
        output.copy_(source)


@_fork.register_fake
def _fake_fork(source, output, stream):
    pass


@torch.library.custom_op("flare_test::enqueue_work", mutates_args=("output",))
def _work(source: torch.Tensor, output: torch.Tensor) -> None:
    _calls.append(("work", source.data_ptr(), output.data_ptr()))
    torch.mul(source, source, out=output)


@_work.register_fake
def _fake_work(source, output):
    pass


@torch.library.custom_op("flare_test::enqueue_join", mutates_args=("output",))
def _join(source: torch.Tensor, output: torch.Tensor, dependency: torch.Tensor, stream: int) -> None:
    _calls.append(("join", source.data_ptr(), output.data_ptr()))
    torch.cuda.current_stream(source.device).wait_stream(torch.cuda.ExternalStream(stream, device=source.device))


@_join.register_fake
def _fake_join(source, output, dependency, stream):
    pass


@torch.no_grad()
def test_compiled_fork_precedes_independent_work_and_retains_its_source():
    device = torch.device("cuda:0")
    source = torch.randn(1024, device=device)
    main_input = torch.randn_like(source)
    quantized, side_output, main_output = [torch.empty_like(source) for _ in range(3)]
    arena = ScratchArena(device, overlap=True)
    stream = arena.auxiliary_stream.cuda_stream

    def forward():
        quantized.copy_(source.sin())
        _fork(quantized, side_output, stream)
        _work(main_input.cos(), main_output)
        _join(quantized, side_output, main_output, stream)
        return side_output + main_output

    compiled = compile_forward(forward)
    _calls.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = compiled()
    assert [entry[0] for entry in _calls] == ["fork", "work", "join"]
    # A compiler-owned temporary is valid if the join retains the same source
    # that the asynchronous reader received, through all intervening work.
    assert _calls[0][1] == _calls[2][1]
    assert _calls[0][2] == _calls[2][2], _calls
    for _ in range(3):
        source.normal_()
        main_input.normal_()
        quantized.fill_(float("nan"))
        side_output.fill_(float("nan"))
        main_output.fill_(float("nan"))
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        torch.testing.assert_close(output, source.sin() + main_input.cos().square())
    graph.reset()
