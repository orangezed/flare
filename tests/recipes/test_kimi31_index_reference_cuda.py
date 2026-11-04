# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare index quantization against the published vLLM K3.1 serving kernels."""

import ast
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch
import triton
import triton.language as tl

from ignition.components.fp8_indexer import PreparedFP8Indexer
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")


def reference():
    root = os.getenv("FLARE_KIMI_REFERENCE")
    if root is None:
        pytest.skip("set FLARE_KIMI_REFERENCE to the Python source tree of vllm/vllm-openai:kimi-k31")
    path = Path(root) / "vllm/models/kimi_k3/nvidia/ops/dsa_indexer.py"
    source = ast.parse(path.read_text(), filename=str(path))
    names = {"_ue8m0_scale", "_fill_topk", "_index_k_insert", "_indexer_k_norm_quant_insert_kernel",
        "indexer_k_norm_quant_insert", "_indexer_q_quant_fold_weights_kernel", "indexer_q_quant_fold_weights"}
    # Execute the upstream functions directly; unrelated engine imports are unnecessary for these kernels.
    selected = ast.Module(body=[n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names],
                          type_ignores=[])
    module = ModuleType("kimi31_serving_reference")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    scope = module.__dict__
    scope.update(torch=torch, triton=triton, tl=tl, _FP8_DTYPE=torch.float8_e4m3fn,
                 _FP8_MAX=448., _USE_FNUZ=False, mqa_logits_use_mx_scales=lambda: False)
    exec(compile(selected, str(path), "exec"), scope)
    return scope


@pytest.mark.parametrize("rows", [1, 65])
@torch.no_grad()
def test_unrotated_index_inputs_match_serving_reference(rows):
    ref = reference()
    device = torch.device("cuda:0")
    torch.manual_seed(318)
    q = torch.randn((rows, 32, 128), device=device, dtype=torch.bfloat16)
    w = torch.randn((rows, 32), device=device, dtype=torch.bfloat16)
    q[0, 0].zero_()
    q[0, 1].fill_(1e-7)
    keys = torch.randn((rows, 128), device=device, dtype=torch.bfloat16)
    norm_w = torch.randn(128, device=device, dtype=torch.bfloat16)
    norm_b = torch.randn(128, device=device, dtype=torch.bfloat16)
    cache = torch.zeros((2, 64 * 132), dtype=torch.uint8, device=device)
    slots = torch.arange(rows, device=device, dtype=torch.int64)
    slots[-1] = -1
    table = torch.tensor([[0, 1]], dtype=torch.int32, device=device).expand(rows, -1)
    lengths = torch.full((rows,), 128, dtype=torch.int32, device=device)
    out = torch.empty((rows, 2048), dtype=torch.int32, device=device)
    pool, arena = GraphBufferPool(device), ScratchArena(device)
    with RecordedPreparationSession(device=device, checkpoint="kimi-index-reference", workload_schema="fp8-plain-index",
            numerical_policy="unrotated-ue8m0", compile_workers=8) as session:
        session.register_scratch(arena)
        index = PreparedFP8Indexer(q, w, cache, table, lengths, None, out, mode="decode", session=session,
            name="index", workspace=arena, buffers=pool.cursor(), rotation="none")
        session.tune(show_progress=False)
        index()
        expected_q, expected_w = ref["indexer_q_quant_fold_weights"](q, w, 1 / 64, 32)
        torch.testing.assert_close(index.quantized.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(index.scaled_weights, expected_w, rtol=0, atol=0)
        normalized = torch.nn.functional.layer_norm(keys.float(), (128,), norm_w.float(), norm_b.float(), 1e-5).bfloat16()
        index.write_keys(normalized, slots)
        expected_cache = torch.zeros((2, 64, 132), dtype=torch.uint8, device=device)
        ref["indexer_k_norm_quant_insert"](keys, norm_w.float(), norm_b.float(), 1e-5, slots,
                                          expected_cache, torch.empty_like(out), rows)
        torch.testing.assert_close(cache, expected_cache.view_as(cache), rtol=0, atol=0)
