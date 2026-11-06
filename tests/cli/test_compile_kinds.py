# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compilation selection retains complete graph capture and prepared-weight contracts."""

from dataclasses import replace

import pytest

from ignition.cli import _parser
from ignition.cli.config import Configuration, Setting
from ignition.cli.preparation import Host, derive
from ignition.errors import UsageError
from ignition.recipes import ResolvedModel, recipes
from tests.cli.test_planning import gpu


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU selection tests must not initialize CUDA"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


@pytest.mark.parametrize("command", ["export", "debug-serve", "bench"])
def test_cli_default_selection_and_conflict(command):
    parser = _parser()
    assert parser.parse_args([command, "kimi-k3.1-pro"]).compile_kinds is None
    args = parser.parse_args([command, "kimi-k3.1-pro", "--compile-kinds", "decode,resume"])
    assert args.compile_kinds == ("decode", "resume") and not args.no_compile
    with pytest.raises(SystemExit):
        parser.parse_args([command, "kimi-k3.1-pro", "--compile-kinds", "decode", "--no-compile"])


@pytest.mark.parametrize("text", ["", ",", "decode,", "decode,decode", "unknown", "decode,unknown"])
def test_cli_rejects_invalid_lists(text):
    with pytest.raises(SystemExit):
        _parser().parse_args(["export", "kimi-k3.1-pro", "--compile-kinds", text])


def kimi_plan(tmp_path, monkeypatch, name="kimi31", kinds=("decode", "resume")):
    recipe = next(r for r in recipes() if r.name == name)
    model = ResolvedModel(recipe, recipe.checkpoints[0])
    config = Configuration(tmp_path / "config.toml", {}, False)
    host = Host(tuple(replace(gpu(i, 4096), total_bytes=4096 << 30) for i in range(16)), None)
    return model, derive(config, model, host, gpus="0-15", concurrency=Setting(4, "flag"),
                         speculation=None, compile_kinds=kinds)


@pytest.mark.parametrize("name", ["kimi3", "kimi31"])
def test_plan_recipe_and_rebuild_roundtrip(tmp_path, monkeypatch, name):
    from google.protobuf.json_format import ParseDict
    from ignition.preparation import runtime_package_pb2 as proto
    from ignition.preparation import source_identity
    from ignition.preparation.rebuild_manifest import from_plan
    from ignition.preparation import rank_group
    from ignition.recipes.kimi3.recipe import prepare_variant
    from ignition.recipes.kimi3 import bootstrap

    model, plan = kimi_plan(tmp_path, monkeypatch, name)
    assert plan.compile_kinds == ("decode", "resume")
    assert plan.to_dict()["compile_kinds"] == ["decode", "resume"]
    default = replace(plan, compile_kinds=None)
    assert "compile_kinds" not in default.to_dict()
    assert {k: v for k, v in plan.to_dict().items() if k != "compile_kinds"} == default.to_dict()
    monkeypatch.setattr(source_identity, "source_identity", lambda: "same-source")
    manifest = ParseDict(from_plan(model, plan, compiled_forwards=True), proto.RebuildManifest())
    restored = proto.RebuildManifest.FromString(manifest.SerializeToString())
    assert list(restored.compile_kinds) == ["decode", "resume"]
    assert proto.RebuildManifest.DESCRIPTOR.fields_by_name["compile_kinds"].number == 29
    assert not ParseDict(from_plan(model, default, compiled_forwards=True), proto.RebuildManifest()).compile_kinds
    assert not proto.RebuildManifest.FromString(b"\x08\x01").compile_kinds
    captured = {}
    def group(snapshot, config, prepare_rank, **kwargs):
        captured["config"] = config
        return config
    monkeypatch.setattr(rank_group, "prepare_group", group)
    prepared = prepare_variant(tmp_path, plan, bootstrap=bootstrap, draft_repository="unused", draft_revision="unused",
                               compile_forwards=True, kernel_cache=tmp_path, progress=None)
    assert prepared.compile_kinds == plan.compile_kinds
    assert prepared.max_active == 4 and prepared.prefill_chunk_size == 4096 and prepared.vision
    assert captured["config"] is prepared


def test_only_declared_recipes_accept_override(tmp_path, monkeypatch):
    model, plan = kimi_plan(tmp_path, monkeypatch)
    for recipe in recipes():
        if recipe.name not in ("kimi3", "kimi31"):
            with pytest.raises(UsageError, match="does not declare selective compilation"):
                replace(plan, recipe=recipe)
    with pytest.raises(UsageError, match="must select from"):
        replace(plan, compile_kinds=("mixed",))
    for invalid in ((), ["decode"], ("decode", "decode"), (1,)):
        with pytest.raises(UsageError):
            replace(plan, compile_kinds=invalid)


def test_reuse_weights_remains_compatible_with_compilation_selection():
    args = _parser().parse_args(["export", "kimi-k3.1-pro", "--reuse-weights", "--compile-kinds", "decode,resume"])
    assert args.reuse_weights and args.compile_kinds == ("decode", "resume")
