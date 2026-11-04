# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-backed gated sparse attention, including learned index and input projections."""

import os

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.kimi3.attention import PreparedMLA
from ignition.recipes.kimi3.weights import KimiWeightLoader

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")


@torch.no_grad()
def test_checkpoint_gated_dsa_matches_explicit_math():
    checkpoint = os.getenv("FLARE_KIMI31_ATTENTION_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("set FLARE_KIMI31_ATTENTION_CHECKPOINT to the target checkpoint or its layer-3 shard subset")
    device = torch.device("cuda:0")
    with KimiWeightLoader(inspect_indexed_checkpoint(checkpoint), device, shard=TensorParallelShard(0, 1),
            prefixes=("language_model.model.layers.3.self_attn.",)) as loader:
        w = loader.load_attention(3)
        eps = loader.config.text["rms_norm_eps"]
    history, rows, pages = 4096, 7, 33
    torch.manual_seed(3131)
    source = torch.randn((rows, 7168), device=device, dtype=torch.bfloat16) * .2
    prior_input = torch.randn((history, 7168), device=device, dtype=torch.bfloat16) * .2
    cache = torch.zeros((pages, 128, 576), device=device, dtype=torch.float8_e4m3fn)
    index_cache = torch.zeros((pages * 2, 64 * 132), device=device, dtype=torch.uint8)
    metadata = Metadata.allocate(rows, 1, pages, device)
    metadata.page_table.copy_(torch.arange(pages, device=device, dtype=torch.int32)[None])
    metadata.positions.copy_(torch.arange(history, history + rows, device=device))
    metadata.starts.fill_(history)
    metadata.lengths.fill_(rows)
    metadata.query_start_loc[-1] = rows
    def norm(x, weight):
        normalized = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)).to(weight.dtype)
        return (normalized * weight).bfloat16()
    def key_norm(x):
        return torch.nn.functional.layer_norm(x.float(), (128,), w["index_norm"].float(), w["index_bias"].float(), eps).bfloat16()
    def quantized(x, floor):
        x = x.float()
        scale = torch.exp2(torch.ceil(torch.log2(x.abs().amax(-1).clamp_min(floor) / 448)))
        return (x / scale[..., None]).to(torch.float8_e4m3fn).float(), scale
    # Prefix records have the same learned projections and normalization as the query rows.
    prior_kv = torch.nn.functional.linear(prior_input, w["latent_gate"][1536:2112])
    prior_latent = torch.cat((norm(prior_kv[:, :512], w["kv_norm"]), prior_kv[:, 512:]), -1)
    prior_key = key_norm(torch.nn.functional.linear(prior_input, w["latent_gate"][14400:14528]))
    with RecordedPreparationSession(device=device, checkpoint=str(checkpoint), workload_schema="kimi-gated-dsa-checkpoint",
            numerical_policy="bf16-and-unit-fp8-cache", compile_workers=16) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        op = PreparedMLA(w, source, metadata, cache, mode="verify", session=session, name="attention", workspace=arena,
            buffers={}, index_cache=index_cache, eps=eps)
        session.tune(show_progress=False)
        cache.view(-1, 576)[:history].copy_(prior_latent.float().clamp(-448, 448).to(torch.float8_e4m3fn))
        op.sparse.stages[0][2].write_keys(prior_key, torch.arange(history, device=device, dtype=torch.int64))
        compiled = compile_forward(lambda: op(source))
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            actual = compiled()
        session.freeze()
        graph.replay()
        torch.cuda.synchronize()
        projected = torch.nn.functional.linear(source, w["latent_gate"])
        query_residual = norm(projected[:, :1536], w["query_norm"])
        query = torch.nn.functional.linear(query_residual, w["q_b"]).view(rows, 96, 192)
        absorbed = torch.cat((torch.bmm(query[:, :, :128].transpose(0, 1), w["key"]).transpose(0, 1), query[:, :, 128:]), -1)
        added = torch.cat((norm(projected[:, 1536:2048], w["kv_norm"]), projected[:, 2048:2112]), -1)
        records = torch.cat((prior_latent, added)).float().clamp(-448, 448).to(torch.float8_e4m3fn).float()
        keys = torch.cat((prior_key, key_norm(projected[:, 14400:14528])))
        index_query = torch.nn.functional.linear(query_residual, w["index_query"]).view(rows, 32, 128)
        q, qs = quantized(index_query, 1e-10)
        k, ks = quantized(keys, 1e-4)
        score = torch.einsum("rhd,kd->rhk", q, k * ks[:, None]).relu()
        score = (score * (projected[:, 14528:14560].float() * qs / 64)[:, :, None]).sum(1)
        score.masked_fill_(torch.arange(history + rows, device=device)[None] > metadata.positions[:, None], -torch.inf)
        selected = score.topk(2048, dim=1).indices.sort(1).values
        kv = records[selected]
        logits = torch.einsum("rhd,rkd->rhk", absorbed.float().clamp(-448, 448).to(torch.float8_e4m3fn).float(), kv) * 192**-.5
        attended = torch.einsum("rhk,rkd->rhd", logits.softmax(-1), kv[:, :, :512]).bfloat16()
        expanded = torch.bmm(attended.transpose(0, 1), w["value"]).transpose(0, 1).flatten(1)
        gated = (expanded.float() * projected[:, 2112:14400].float().sigmoid()).bfloat16()
        expected = torch.nn.functional.linear(gated, w["output"])
        relative = (actual.float() - expected.float()).square().mean().sqrt() / expected.float().square().mean().sqrt()
        assert relative < .015, float(relative)
        graph.reset()
