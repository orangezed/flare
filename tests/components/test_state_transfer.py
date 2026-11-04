# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordinary execution preserves unused draft bytes during continuation transfers."""

import os

import pytest
import torch
from triton.runtime.interpreter import InterpretedFunction

from ignition.components.state_transfer import _copy_rows


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an explicitly assigned CUDA device"))])
def test_ordinary_state_copy_and_reset_leave_draft_regions_untouched(device):
    backing = torch.full((4, 8192), 231, dtype=torch.uint8, device=device)
    backing[1].fill_(17)
    source = torch.tensor([1, -1], device=device)
    destination = torch.tensor([2, 3], device=device)
    operation = torch.tensor([0, 1], dtype=torch.int32, device=device)
    count = torch.tensor([2], dtype=torch.int32, device=device)
    errors = torch.zeros(2, dtype=torch.int32, device=device)
    args = (backing, source, destination, operation, count, errors, 8192, 1024, ((768, 1792), (4096, 5120)))
    kernel = _copy_rows if device == "cuda" else InterpretedFunction(_copy_rows.fn)
    kernel[(2, 8)](*args)
    if device == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kernel[(2, 8)](*args)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        graph.reset()
    wanted = torch.full((4, 8192), 231, dtype=torch.uint8)
    wanted[1:3].fill_(17)
    wanted[3].zero_()
    for start, end in args[-1]:
        wanted[2:, start:end].fill_(231)
    assert torch.equal(backing.cpu(), wanted)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")
@torch.no_grad()
def test_excluded_partial_pages_survive_snapshot_and_restore():
    from ignition.components.paged_state import PagedStateStorage
    from ignition.components.sampling import DeviceRequestState
    from ignition.components.state import StateLayout
    from ignition.components.state_transfer import PreparedStateTransfer, StateCommands

    device = torch.device("cuda:0")
    specs = [("target", (128,)), ("draft", (128,))]
    layout = StateLayout.from_specs([(name, shape, torch.uint8) for name, shape in specs]
                                   + [("consumed_tokens", (1,), torch.int64)])
    pool = PagedStateStorage.allocate(layout, specs, 8, 2, device)
    commands = StateCommands.allocate(device, capacity=4)
    tables = torch.zeros((4, 1), device=device, dtype=torch.int32)
    tables[1] = 1
    consumed = pool.region("consumed_tokens")
    requests = DeviceRequestState.allocate(device, vocab_size=128, capacity=4, consumed_tokens=consumed[1:5, 0])
    state_rows = torch.arange(1, 5, device=device)[:, None].expand(4, 8).contiguous()
    transfer = PreparedStateTransfer(pool, commands, tables, pages=pool.partial_pages(), page_tokens=128,
        state_rows=state_rows, requests=requests, excluded_regions=("draft",))
    transfer.prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        transfer.validate()
        transfer.apply()
    pool.backing[5].fill_(73)
    consumed[1] = 1
    pool.pages["target"][0].fill_(21)
    pool.pages["draft"][0].fill_(22)
    commands.num_ops.fill_(1)
    commands.source_slots[0], commands.destination_slots[0], commands.request_slots[0] = 1, 5, 0
    commands.expected_consumed[0] = 1
    graph.replay()
    torch.cuda.synchronize()
    assert not commands.errors.count_nonzero()
    assert (pool.region("target")[5] == 21).all()
    assert (pool.region("draft")[5] == 73).all()
    pool.region("draft")[2].fill_(91)
    pool.pages["draft"][1].fill_(92)
    commands.source_slots[0], commands.destination_slots[0], commands.request_slots[0] = 5, 2, 1
    graph.replay()
    torch.cuda.synchronize()
    assert not commands.errors.count_nonzero()
    assert (pool.region("target")[2] == 21).all()
    assert (pool.pages["target"][1] == 21).all()
    assert (pool.region("draft")[2] == 91).all()
    assert (pool.pages["draft"][1] == 92).all()
    graph.reset()
