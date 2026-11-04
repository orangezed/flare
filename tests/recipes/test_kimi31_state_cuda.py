# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Snapshot/restore preserves KDA, index, latent and full-context draft history."""

from dataclasses import replace
import json
import os
from pathlib import Path

import pytest
import torch

from ignition.components.sampling import DeviceRequestState
from ignition.components.state_transfer import PreparedStateTransfer, StateCommands
from ignition.recipes.kimi3.config import DSparkConfig, ModelConfig
from ignition.recipes.kimi3.pool import ContinuationPool

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")


@pytest.mark.parametrize("endpoint", [1, 1023, 1024, 2047, 2048, 2049])
@pytest.mark.parametrize("drafting", [False, True])
@torch.no_grad()
def test_complete_context_page_boundaries(endpoint, drafting):
    root = Path(__file__).parents[1] / "fixtures/kimi31"
    config = ModelConfig.from_dict(json.loads((root / "config.json").read_text()))
    config = replace(config, draft=DSparkConfig.from_dict(json.loads((root / "draft-config.json").read_text()), config.text))
    device = torch.device("cuda:0")
    pool = ContinuationPool.for_model(config, 16, 8, 5, device, speculative_tokens=7)
    commands = StateCommands.allocate(device, capacity=4)
    tables = torch.full((4, 3), -1, dtype=torch.int32, device=device)
    consumed = pool.region("consumed_tokens")
    requests = DeviceRequestState.allocate(device, vocab_size=128, capacity=4, consumed_tokens=consumed[1:5, 0])
    rows = torch.arange(1, 5, device=device)[:, None].expand(4, 8).contiguous()
    excluded = () if drafting else tuple(r.name for r in pool.layout.regions if r.name.startswith("draft"))
    operation = PreparedStateTransfer(pool, commands, tables, pages=pool.partial_pages(), page_tokens=2048,
        state_rows=rows, requests=requests, excluded_regions=excluded)
    operation.prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation.validate()
        operation.apply()
    def replay(items):
        commands.num_ops.fill_(len(items))
        for i, (source, destination, slot) in enumerate(items):
            commands.source_slots[i], commands.destination_slots[i], commands.request_slots[i] = source, destination, slot
            commands.operation[i] = 0
            commands.expected_consumed[i] = endpoint
        graph.replay()
        torch.cuda.synchronize()
        assert not commands.errors.count_nonzero()
    pool.backing[1].random_(0, 256)
    consumed[1] = endpoint
    tables[0].copy_(torch.tensor([3, 0, 4], device=device))
    tables[1].fill_(2)
    tables[2].fill_(1)
    for cache in pool.pages.values():
        cache.random_(0, 256)
    tail = int(tables[0, endpoint // 2048]) if endpoint % 2048 else None
    expected = pool.backing[1].clone()
    expected_pages = {}
    for name, cache in pool.pages.items():
        region = pool.layout.region(name)
        if name in excluded:
            expected[region.offset:region.offset + region.nbytes].zero_()
        elif tail is not None:
            expected_pages[name] = cache[tail].clone()
            expected[region.offset:region.offset + region.nbytes].copy_(cache[tail])
    replay([(1, 5, 0)])
    assert torch.equal(pool.backing[5], expected)
    for cache in pool.pages.values():
        cache.zero_()
    replay([(5, 2, 1), (5, 3, 2)])
    for row in (2, 3, 5):
        assert torch.equal(pool.backing[row], expected)
    for name, cache in pool.pages.items():
        if tail is not None and name not in excluded:
            assert torch.equal(cache[1], expected_pages[name])
            assert torch.equal(cache[2], expected_pages[name])
        else:
            assert not cache.count_nonzero()
    graph.reset()
