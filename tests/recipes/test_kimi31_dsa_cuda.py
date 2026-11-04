# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Learned sparse selection and latent attention under compiled graph replay."""

import os

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.kimi3.dsa import PreparedDSA

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")


@pytest.mark.parametrize("rows", [1, 33, 4096])
@torch.no_grad()
def test_dsa_selection_and_attention_replay(rows):
    device = torch.device("cuda:0")
    torch.manual_seed(3101)
    records, heads, residual_width = 4096, 6, 64
    bf16, fp8 = torch.bfloat16, torch.float8_e4m3fn
    def random(*shape, scale=.2):
        return (torch.randn(shape, device=device) * scale).bfloat16()
    weights = dict(index_query=random(4096, residual_width),
        index_norm=torch.ones(128, device=device, dtype=bf16), index_bias=torch.zeros(128, device=device, dtype=bf16))
    residual, key_input, head_weights = random(rows, residual_width), random(rows, 128), random(rows, 32)
    query, output = random(rows, heads, 576), torch.empty((rows, heads, 512), device=device, dtype=bf16)
    latent = random(records, 576).to(fp8).view(-1, 128, 576)
    keys = random(records, 128)
    index_cache = torch.zeros((records // 64, 64 * 132), dtype=torch.uint8, device=device)
    slots = torch.full((rows,), -1, dtype=torch.int64, device=device)
    m = Metadata.allocate(rows, 1, records // 128, device)
    m.page_table.copy_(torch.arange(records // 128, device=device, dtype=torch.int32)[None])
    m.positions.copy_(torch.arange(records - rows, records, device=device))
    arena, buffers = ScratchArena(device), GraphBufferPool(device)
    with RecordedPreparationSession(device=device, checkpoint="kimi-dsa-test", workload_schema="dsa-top2048",
            numerical_policy="fp8-index-unit-latent", compile_workers=16) as session:
        session.register_scratch(arena)
        op = PreparedDSA(weights, residual, key_input, head_weights, query, output, slots, m, latent,
            index_cache, context=None, session=session, coordinator=None, name="dsa", workspace=arena,
            buffers={}, allocator=buffers.cursor())
        session.tune(show_progress=False)
        op.stages[0][2].write_keys(keys, torch.arange(records, device=device, dtype=torch.int64))
        compiled = compile_forward(op)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()

        def quantized(x, floor):
            value = x.float()
            scale = torch.exp2(torch.ceil(torch.log2(value.abs().amax(-1).clamp_min(floor) / 448)))
            return (value / scale[..., None]).to(fp8).float(), scale
        k, ks = quantized(keys, 1e-4)
        q, qs = quantized(torch.nn.functional.linear(residual, weights["index_query"]).view(rows, 32, 128), 1e-10)
        scores = torch.einsum("rhd,kd->rhk", q, k * ks[:, None]).relu()
        scores = (scores * (head_weights.float() * qs * (128 * 32)**-.5)[:, :, None]).sum(1)
        for shift in (0, -63):
            m.positions.add_(shift)
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            causal = torch.arange(records, device=device)[None] <= m.positions[:, None]
            selected = scores.masked_fill(~causal, -torch.inf).topk(2048, dim=1).indices.sort(1).values
            kv = latent.float().view(records, 576)[selected]
            attention_scores = torch.einsum("rhd,rkd->rhk", query.to(fp8).float(), kv) * 192**-.5
            # topk pads short prefixes with masked candidates; those records must remain invisible to attention.
            attention_scores.masked_fill_(selected[:, None] > m.positions[:, None, None], -torch.inf)
            expected = torch.einsum("rhk,rkd->rhd", attention_scores.softmax(-1).nan_to_num(0), kv[:, :, :512])
            torch.testing.assert_close(output.float(), expected, rtol=.03, atol=.002)
        visible_positions = m.positions.clone()
        m.positions.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        assert not output.count_nonzero()
        m.positions.copy_(visible_positions)
        output.fill_(float("nan"))
        torch.cuda.synchronize()
        bundle = PreparedGraphBundle(device=0, graphs={"decode": graph},
            buffers={"output": output, "positions": m.positions, "cache": latent, "index": index_cache},
            resources=(op, compiled, arena, buffers, session))
        worker = bundle.start_worker()
        try:
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            ticket = worker.submit(0, repetitions=3)
            completion = worker.poll(timeout_ms=120000)
            assert completion is not None and completion.ticket == ticket and completion.error is None
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            torch.testing.assert_close(output.float(), expected, rtol=.03, atol=.002)
        finally:
            worker.close()
        graph.reset()
