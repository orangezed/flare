# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3.1 checkpoint identity, attention partition and complete history storage."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.kimi3.config import ModelConfig, DSparkConfig, LatentDSparkConfig
from ignition.recipes.kimi3.pool import continuation_layout, page_specs, page_bytes
from ignition.recipes.kimi31.recipe import RECIPE

FIXTURES = Path(__file__).parents[1] / "fixtures" / "kimi31"


@pytest.fixture
def raw():
    # Published configuration at moonshotai/Kimi-K3.1-Pro revision 6d0a0202956796543364d666ab19c6ad1a494ee1.
    return json.loads((FIXTURES / "config.json").read_text())


def test_layer_mapping_and_cache_accounting(raw):
    config = ModelConfig.from_dict(raw)
    config.validate_shard(TensorParallelShard(0, 16))
    assert config.layers.count("kda") == 69
    assert config.layers.count("dsa") == 24
    assert {i + 1 for i, kind in enumerate(config.layers) if kind == "dsa"} == set(raw["text_config"]["dsa_config"]["dsa_layers"])
    plain = RECIPE.plain_ranks[16]
    layout = continuation_layout(config, 16)
    assert page_bytes(config) == plain.page_bytes
    assert layout.row_bytes == plain.state_row_bytes
    for name, shape in page_specs(config):
        assert layout.region(name).shape == shape
    draft = DSparkConfig.from_dict(json.loads((FIXTURES / "draft-config.json").read_text()), config.text)
    assert isinstance(draft, LatentDSparkConfig)
    config = replace(config, draft=draft)
    draft.validate_shard(TensorParallelShard(0, 16))
    assert draft.auxiliary_layers == (75, 79, 83, 87, 91)
    assert draft.auxiliary_source == "residual_output"
    layout = continuation_layout(config, 16, speculative_tokens=7)
    assert layout.row_bytes == RECIPE.ranks[16].state_row_bytes
    assert page_bytes(config) == RECIPE.ranks[16].page_bytes
    assert len([name for name, _ in page_specs(config) if name.startswith("draft")]) == 5
    for name, shape in page_specs(config):
        assert layout.region(name).shape == shape


@pytest.mark.parametrize("field,value", [("index_topk", 512), ("index_rope_dim", 64),
    ("index_k_norm_type", "rmsnorm"), ("index_topk_freq", 2), ("dsa_layers", [4])])
def test_rejects_incompatible_sparse_attention(raw, field, value):
    raw["text_config"]["dsa_config"][field] = value
    with pytest.raises(ValueError, match="DSA|K3.1"):
        ModelConfig.from_dict(raw)


def test_dense_k3_cannot_silently_consume_sparse_checkpoint(raw):
    raw.update(model_type="kimi_k3", architectures=["KimiK3ForConditionalGeneration"])
    with pytest.raises(ValueError, match="DSA configuration"):
        ModelConfig.from_dict(raw)
    del raw["text_config"]["dsa_config"]
    config = ModelConfig.from_dict(raw)
    assert config.layers.count("mla") == 24
    assert not any(name.endswith(".index") for name, _ in page_specs(config))


@pytest.mark.parametrize("field,value", [("full_attention_causal", False), ("target_layer_ids", [93]),
    ("layer_types", ["sliding_attention"] * 5), ("vocab_size", 1)])
def test_draft_geometry_and_target_identity(raw, field, value):
    draft = json.loads((FIXTURES / "draft-config.json").read_text())
    draft[field] = value
    with pytest.raises(ValueError, match="DSpark"):
        DSparkConfig.from_dict(draft, raw["text_config"])


def test_recipe_preserves_native_weights_and_sampling():
    assert not RECIPE.online_mxfp8
    assert not RECIPE.language_model_only
    assert RECIPE.sampling_defaults.top_p == .95
    assert RECIPE.sampling_defaults.temperature == 1.
    assert RECIPE.sampling_defaults.top_k == -1
    assert RECIPE.max_draft_tokens == 7
