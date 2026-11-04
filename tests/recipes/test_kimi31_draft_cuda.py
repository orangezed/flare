# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Full-context draft rotary arithmetic at the prepared one-million-token limit."""

from dataclasses import replace
import json
import os
from pathlib import Path

import pytest
import torch

from ignition.recipes.kimi3.config import DSparkConfig, ModelConfig
from ignition.recipes.kimi3.latent_dspark import YarnRotary
from ignition.preparation.compilation import compile_forward

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")


@torch.no_grad()
def test_yarn_rotation_at_full_context():
    root = Path(__file__).parents[1] / "fixtures/kimi31"
    config = ModelConfig.from_dict(json.loads((root / "config.json").read_text()))
    config = replace(config, draft=DSparkConfig.from_dict(json.loads((root / "draft-config.json").read_text()), config.text))
    rotary = YarnRotary(config, "cuda:0")
    assert rotary.scale == 1.3465735902799727
    positions = torch.tensor([0, 1, 32767, 32768, 1048575], device="cuda", dtype=torch.int64)
    values = torch.randn((5, 6, 192), device="cuda", dtype=torch.bfloat16)
    original = values.clone()
    compiled = compile_forward(lambda: rotary.apply(values, positions))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    for _ in range(3):
        original.normal_()
        values.copy_(original)
        graph.replay()
        torch.cuda.synchronize()
        angles = positions.float()[:, None] * rotary.inverse[None]
        cos, sin = angles.cos()[:, None] * rotary.scale, angles.sin()[:, None] * rotary.scale
        pairs = original[:, :, -64:].float().view(5, 6, 32, 2)
        expected = torch.stack((pairs[..., 0] * cos - pairs[..., 1] * sin,
                                pairs[..., 1] * cos + pairs[..., 0] * sin), -1).flatten(-2).bfloat16()
        torch.testing.assert_close(values[:, :, :128], original[:, :, :128], rtol=0, atol=0)
        torch.testing.assert_close(values[:, :, -64:], expected, rtol=0, atol=0)
    graph.reset()
