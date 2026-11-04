# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Selected Kimi latent records retain FP8 storage and causal selection boundaries."""

import os

import pytest
import torch
from b12x.attention import dense_mla

from ignition.components.dense_mla import PreparedDenseMLA
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")


@pytest.mark.parametrize("rows,heads", [(1, 6), (8, 96)])
@torch.no_grad()
def test_selected_records_match_explicit_attention(rows, heads):
    device = torch.device("cuda:0")
    torch.manual_seed(310)
    records, selected = 4096, 2048
    cache = (torch.randn((records, 1, 576), device=device) * .25).to(torch.float8_e4m3fn)
    query = (torch.randn((rows, heads, 576), device=device) * .25).to(torch.bfloat16)
    output = torch.empty((rows, heads, 512), device=device, dtype=torch.bfloat16)
    lse = torch.empty((rows, heads), device=device)
    table = torch.zeros((rows, selected), device=device, dtype=torch.int32)
    lengths = torch.zeros(rows, device=device, dtype=torch.int32)
    starts = torch.arange(rows + 1, device=device, dtype=torch.int32)
    caps = dense_mla.Caps(device=device, mode="decode", kv_dtype=cache.dtype, q_dtype=query.dtype,
        num_q_heads=heads, page_size=1, max_total_q=rows, max_batch=rows,
        max_cache_tokens=selected, max_page_table_width=selected, num_cache_pages=records,
        partial_dtype=torch.float32, use_cuda_graph=True)
    with RecordedPreparationSession(device=device, checkpoint="kimi-selected-records",
            workload_schema="selected-fp8-576", numerical_policy="unit-scale", compile_workers=16) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        attention = PreparedDenseMLA(query, cache, output, table, lengths, starts, caps=caps,
            scale=192**-.5, session=session, name="selected", workspace=arena, lse=lse)
        session.tune(show_progress=False)
        table.copy_(torch.stack([torch.randperm(records, device=device)[:selected] for _ in range(rows)]))
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            attention()
        session.freeze()
        for count in (1, 65, 2048, 0):
            lengths.fill_(count)
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            if count == 0:
                assert not output.count_nonzero()
                assert torch.isneginf(lse).all()
                continue
            kv = cache.float()[:, 0][table[:, :count].long()]
            q = query.to(torch.float8_e4m3fn).float()
            scores = torch.einsum("rhd,rkd->rhk", q, kv) * 192**-.5
            expected = torch.einsum("rhk,rkd->rhd", scores.softmax(-1), kv[:, :, :512])
            torch.testing.assert_close(output.float(), expected, rtol=.025, atol=.002)
            torch.testing.assert_close(lse, scores.logsumexp(-1), rtol=2e-4, atol=2e-4)
        graph.reset()
