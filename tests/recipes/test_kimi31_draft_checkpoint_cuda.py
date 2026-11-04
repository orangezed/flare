# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-backed full-history draft feedback and causal MLA under native graph replay."""

from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.kimi3.config import ModelConfig, DSparkConfig
from ignition.recipes.kimi3.latent_dspark import DraftWeights, DraftScratch, DraftAttention, PreparedContext

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")


@torch.no_grad()
def test_checkpoint_draft_context_and_attention():
    checkpoint = os.getenv("FLARE_KIMI31_DRAFT_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("set FLARE_KIMI31_DRAFT_CHECKPOINT to the MLA DSpark checkpoint")
    fixture = Path(__file__).parents[1] / "fixtures/kimi31/config.json"
    config = ModelConfig.from_dict(json.loads(fixture.read_text()))
    config = replace(config, draft=DSparkConfig.load(checkpoint, config))
    device, shard = torch.device("cuda:0"), TensorParallelShard(0, 1)
    draft = DraftWeights.load(checkpoint, config, device, shard, target_embedding=None, target_head=None)
    weights = SimpleNamespace(config=config, shard=shard, draft=draft)
    history, block = 65, 7
    cache = torch.zeros((5, 1, 128, 576), dtype=torch.float8_e4m3fn, device=device)
    pool = SimpleNamespace(draft_attention=lambda index: cache[index])
    context_metadata = Metadata.allocate(history, 1, 1, device)
    context_metadata.page_table.zero_()
    context_metadata.positions.copy_(torch.arange(history, device=device))
    context_metadata.lengths.fill_(history)
    context_metadata.query_start_loc[-1] = history
    metadata = Metadata.allocate(block, 1, 1, device)
    metadata.page_table.zero_()
    metadata.positions.copy_(torch.arange(history, history + block, device=device))
    metadata.starts.fill_(history)
    metadata.lengths.fill_(block)
    metadata.query_start_loc[-1] = block
    torch.manual_seed(3107)
    auxiliary = torch.randn((history, 5 * config.hidden), dtype=torch.bfloat16, device=device) * .2
    source = torch.randn((block, config.hidden), dtype=torch.bfloat16, device=device) * .2
    scratch = DraftScratch.allocate(config, shard, block, block, device)
    eps = config.draft.text["rms_norm_eps"]
    def norm(x, w):
        # vLLM's ir.ops.layernorm.rms_norm and canonical Kimi RMSNorm both round the
        # normalized value to the BF16 weight dtype before the learned multiplication.
        normalized = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)).to(w.dtype)
        return (normalized * w).to(x.dtype)
    with RecordedPreparationSession(device=device, checkpoint=str(checkpoint), workload_schema="kimi-latent-draft",
            numerical_policy="checkpoint-bf16", compile_workers=16) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        context = PreparedContext(weights, auxiliary, context_metadata, pool, session=session, name="context",
            workspace=arena, runtime=None, coordinator=None, buffers={})
        attention = DraftAttention(0, draft.layers[0], config, scratch, metadata, pool, block,
            runtime=None, coordinator=None, session=session, name="draft", workspace=arena,
            latent_runtimes=(), context=None)
        session.tune(show_progress=False)
        committed = context_metadata.lengths
        context_forward = compile_forward(lambda: context(auxiliary, committed))
        context_forward()
        previous = cache[:, :, :history].clone()
        forward = compile_forward(lambda: attention(source)[0])
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            actual = forward()
        session.freeze()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(cache[:, :, :history].view(torch.uint8), previous.view(torch.uint8), atol=0, rtol=0)
        rotary = attention.attention.rotary
        def rotate(x, positions):
            angle = positions.float()[:, None] * rotary.inverse[None]
            shape = (positions.numel(),) + (1,) * (x.ndim - 2) + (32,)
            cos = angle.cos().view(shape) * rotary.scale
            sin = angle.sin().view(shape) * rotary.scale
            pairs = x[..., -64:].float().reshape(*x.shape[:-1], 32, 2)
            tail = torch.stack((pairs[..., 0] * cos - pairs[..., 1] * sin,
                                pairs[..., 1] * cos + pairs[..., 0] * sin), -1).flatten(-2).bfloat16()
            return torch.cat((x[..., :-64], tail), -1)
        feedback = norm(torch.nn.functional.linear(auxiliary, draft.projection), draft.hidden_norm)
        projected_context = torch.nn.functional.linear(feedback, draft.context_kv).view(history, 5, 576)
        w = draft.layers[0].attention
        prior = projected_context[:, 0]
        prior = rotate(torch.cat((norm(prior[:, :512], w["kv_norm"]), prior[:, 512:]), -1), context_metadata.positions)
        projected = torch.nn.functional.linear(source, w["latent_gate"])
        q_residual, kv_residual = projected.split((1536, 576), -1)
        query = torch.nn.functional.linear(norm(q_residual, w["query_norm"]), w["q_b"]).view(block, 96, 192)
        query = rotate(query, metadata.positions)
        absorbed = torch.cat((torch.bmm(query[:, :, :128].transpose(0, 1), w["key"]).transpose(0, 1), query[:, :, 128:]), -1)
        added = rotate(torch.cat((norm(kv_residual[:, :512], w["kv_norm"]), kv_residual[:, 512:]), -1), metadata.positions)
        records = torch.cat((prior, added)).float().clamp(-448, 448).to(torch.float8_e4m3fn).float()
        q = absorbed.float().clamp(-448, 448).to(torch.float8_e4m3fn).float()
        scores = torch.einsum("rhd,kd->rhk", q, records) * 192**-.5
        live = torch.arange(history + block, device=device)[None] <= metadata.positions[:, None]
        scores.masked_fill_(~live[:, None], -torch.inf)
        latent = torch.einsum("rhk,kd->rhd", scores.softmax(-1), records[:, :512]).bfloat16()
        expanded = torch.bmm(latent.transpose(0, 1), w["value"]).transpose(0, 1).flatten(1)
        expected = torch.nn.functional.linear(expanded, w["output"])
        error = (actual.float() - expected.float()).square().mean().sqrt() / expected.float().square().mean().sqrt()
        def relative(a, b):
            return float((a.float() - b.float()).square().mean().sqrt() / b.float().square().mean().sqrt().clamp_min(1e-10))
        if error >= .015:
            saved = actual.clone()
            eager = attention(source)[0].clone()
            a = attention.attention
            diagnostic = dict(total=float(error), compiled_vs_eager=relative(saved, eager),
                projected=relative(a.input, projected), query_norm=relative(a.query_norm, norm(q_residual, w["query_norm"])),
                query=relative(a.query, query), absorbed=relative(a.absorbed, absorbed), added=relative(a.latent, added),
                prior=relative(cache[0, 0, :history].float(), prior.to(torch.float8_e4m3fn).float()),
                attended=relative(a.attended, latent), expanded=relative(a.expanded, expanded))
            pytest.fail(str(diagnostic))
        # Feedback commits only accepted target positions; rejected proposal slots remain outside visibility.
        committed.fill_(history - 2)
        context_forward()
        graph.reset()


@torch.no_grad()
def test_complete_checkpoint_draft_survives_scratch_poisoning():
    import ctypes
    from ignition.preparation.disposable import storage_ranges
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.preparation.runtime_package import _Cuda
    from ignition.recipes.kimi3.dspark import PreparedDraft

    checkpoint = os.getenv("FLARE_KIMI31_DRAFT_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("set FLARE_KIMI31_DRAFT_CHECKPOINT to the MLA DSpark checkpoint")
    fixture = Path(__file__).parents[1] / "fixtures/kimi31/config.json"
    config = ModelConfig.from_dict(json.loads(fixture.read_text()))
    config = replace(config, draft=DSparkConfig.load(checkpoint, config))
    device, shard = torch.device("cuda:0"), TensorParallelShard(0, 1)
    # Target embedding/head values are inputs to this component test. The five-layer
    # draft and its context/Markov projections come from the actual checkpoint.
    embedding = torch.zeros((config.vocabulary, config.hidden), device=device, dtype=torch.bfloat16)
    head = torch.zeros_like(embedding)
    torch.manual_seed(3117)
    for token in (42, config.draft.noise_token):
        embedding[token].normal_(std=.2)
    head[:64].normal_(std=.01)
    draft = DraftWeights.load(checkpoint, config, device, shard, target_embedding=embedding, target_head=head)
    weights = SimpleNamespace(config=config, shard=shard, draft=draft, head=head)
    history, block = 65, 7
    cache = torch.zeros((5, 1, 128, 576), dtype=torch.float8_e4m3fn, device=device)
    pool = SimpleNamespace(draft_attention=lambda index: cache[index])
    context_metadata = Metadata.allocate(history, 1, 1, device)
    context_metadata.page_table.zero_()
    context_metadata.positions.copy_(torch.arange(history, device=device))
    context_metadata.lengths.fill_(history)
    context_metadata.query_start_loc[-1] = history
    metadata = Metadata.allocate(block, 1, 1, device, samples=block)
    metadata.page_table.zero_()
    metadata.positions.copy_(torch.arange(history, history + block, device=device))
    metadata.starts.fill_(history)
    metadata.lengths.fill_(block)
    metadata.query_start_loc[-1] = block
    metadata.sample_indices.copy_(torch.arange(block, device=device))
    metadata.live_counts.copy_(torch.tensor([block, 1], device=device, dtype=torch.int32))
    metadata.token_ids.fill_(config.draft.noise_token)
    metadata.token_ids[0] = 42
    auxiliary = torch.randn((history, 5 * config.hidden), dtype=torch.bfloat16, device=device) * .2
    scratch = DraftScratch.allocate(config, shard, block, block, device)
    with RecordedPreparationSession(device=device, checkpoint=str(checkpoint), workload_schema="kimi-latent-draft-complete",
            numerical_policy="checkpoint-bf16", compile_workers=16) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        context = PreparedContext(weights, auxiliary, context_metadata, pool, session=session, name="context",
            workspace=arena, runtime=None, coordinator=None, buffers={})
        model = PreparedDraft(weights, scratch, metadata, pool, block=block, runtime=None,
            vocabulary_runtime=None, embedding_runtime=None, coordinator=None,
            session=session, name="draft", workspace=arena)
        session.tune(show_progress=False)
        update = compile_forward(lambda: context(auxiliary, context_metadata.lengths))
        update()
        before = cache.clone()
        compiled = compile_forward(model)
        compiled()
        expected = scratch.logits.clone()
        assert torch.isfinite(expected).all() and expected.abs().max() > 0
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        cache.copy_(before)
        torch.cuda.synchronize()
        cuda = _Cuda()
        for address, size in storage_ranges(scratch.scratch_tensors()):
            cuda.call("cuMemsetD8_v2", ctypes.c_uint64(address), ctypes.c_ubyte(255), ctypes.c_size_t(size))
        bundle = PreparedGraphBundle(device=0, graphs={"draft": graph}, buffers={"logits": scratch.logits, "cache": cache},
            resources=(model, context, compiled, update, arena, session, weights, scratch))
        worker = bundle.start_worker()
        try:
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            ticket = worker.submit(0, repetitions=3)
            completion = worker.poll(timeout_ms=120000)
            assert completion is not None and completion.ticket == ticket and completion.error is None
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            torch.testing.assert_close(scratch.logits, expected, rtol=.002, atol=.002)
            torch.testing.assert_close(cache[:, :, :history].view(torch.uint8), before[:, :, :history].view(torch.uint8),
                                       rtol=0, atol=0)
        finally:
            worker.close()
        graph.reset()
