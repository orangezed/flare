# SPDX-License-Identifier: Apache-2.0
"""FLARE_KIMI3_PREFILL_LATENT_GATHER: switch parsing, channel sizing, call sites, and the layout/numerics contract
(CPU only)."""

import ast
from pathlib import Path

import pytest
import torch

from ignition.recipes.kimi3 import decoder
from ignition.recipes.kimi3.decoder import (LATENT_GATHER_ENV, LATENT_GATHER_LAUNCH_ENV, LATENT_GATHER_ROWS_ENV,
    latent_gather_enabled, latent_gather_launch, latent_gather_rows)

ROOT = Path(decoder.__file__).resolve().parent


@pytest.mark.parametrize("value, expected", [(None, False), ("", False), ("0", False), ("1", True), (" 1 ", True)])
def test_switch_default_off(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(LATENT_GATHER_ENV, raising=False)
    else:
        monkeypatch.setenv(LATENT_GATHER_ENV, value)
    assert latent_gather_enabled(8, 3584) is expected


def test_switch_requires_64_column_partitions(monkeypatch):
    monkeypatch.setenv(LATENT_GATHER_ENV, "1")
    assert latent_gather_enabled(8, 3584) and latent_gather_enabled(4, 3584) and latent_gather_enabled(2, 3584)
    assert not latent_gather_enabled(1, 3584)
    assert not latent_gather_enabled(16, 3584)  # 224 columns per rank is not 64-aligned
    assert not latent_gather_enabled(3, 3584)


def test_channel_rows(monkeypatch):
    monkeypatch.delenv(LATENT_GATHER_ROWS_ENV, raising=False)
    assert latent_gather_rows(1024, 3072) == 1024
    monkeypatch.setenv(LATENT_GATHER_ROWS_ENV, "3072")
    assert latent_gather_rows(1024, 3072) == 3072
    monkeypatch.setenv(LATENT_GATHER_ROWS_ENV, "8192")
    assert latent_gather_rows(1024, 3072) == 3072
    monkeypatch.setenv(LATENT_GATHER_ROWS_ENV, "-1")
    with pytest.raises(ValueError):
        latent_gather_rows(1024, 3072)


def test_launch_geometry(monkeypatch):
    monkeypatch.delenv(LATENT_GATHER_LAUNCH_ENV, raising=False)
    assert latent_gather_launch() == (512, 32)
    monkeypatch.setenv(LATENT_GATHER_LAUNCH_ENV, "256:16")
    assert latent_gather_launch() == (256, 16)
    monkeypatch.setenv(LATENT_GATHER_LAUNCH_ENV, "default")
    assert latent_gather_launch() is None
    for bad in ("1024:32", "512", "512:0", "512:65", "x:y"):
        monkeypatch.setenv(LATENT_GATHER_LAUNCH_ENV, bad)
        with pytest.raises(ValueError):
            latent_gather_launch()


def _calls(path, func):
    tree = ast.parse(path.read_text())
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == func]


def test_only_prefill_decoders_receive_the_channel():
    # PreparedDecoder forwards the channel to the MoE only in prefill mode ...
    (moe,) = _calls(ROOT / "decoder.py", "PreparedLatentMoE")
    kw = {k.arg: ast.unparse(k.value) for k in moe.keywords}
    assert kw["latent_gather_runtime"] == "latent_gather_runtime if mode == 'prefill' else None"
    # ... and the bootstrap hands it only to the main prefill decoders, never through the shared collectives dict
    # that speculation's decoders (decode / verify / draft) receive.
    src = (ROOT / "bootstrap.py").read_text()
    assert "latent_gather_runtime=latent_gather_collective if mode == \"prefill\" else None" in src
    collectives = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign)
                   and any(getattr(t, "id", None) == "collectives" for t in n.targets)]
    assert collectives and all("latent_gather" not in ast.unparse(n.value) for n in collectives)


def test_row_sharded_projection_gather_equals_full_projection():
    # Rank r computes latent columns [r*w, (r+1)*w) over the full hidden row; a rank-major column gather
    # rebuilds exactly the unsharded BF16 projection (one rounding per value, no reduction).
    torch.manual_seed(0)
    world, rows, hidden, latent = 8, 5, 512, 1024
    x = torch.randn(rows, hidden).to(torch.bfloat16)
    w = torch.randn(latent, hidden).to(torch.bfloat16)
    full = (x.double() @ w.double().T).float()
    width = latent // world
    shards = [(x.double() @ w[r * width:(r + 1) * width].double().T).float().to(torch.bfloat16) for r in range(world)]
    gathered = torch.stack(shards, 1).reshape(rows, latent)  # out[row, rank, column] layout of all_gather_heads
    assert torch.equal(gathered, full.to(torch.bfloat16))


def test_today_rounds_twice_gather_rounds_once():
    # The reduction path rounds each rank's partial to BF16 before the FP32 sum; the gather path rounds once.
    torch.manual_seed(1)
    world, rows, hidden, latent = 8, 64, 1024, 256
    x = torch.randn(rows, hidden).to(torch.bfloat16).double()
    w = torch.randn(hidden, latent).to(torch.bfloat16).double()
    ref = x @ w
    k = hidden // world
    partials = [(x[:, r * k:(r + 1) * k] @ w[r * k:(r + 1) * k]).float().to(torch.bfloat16) for r in range(world)]
    today = (sum(p.float() for p in partials[:4]) + sum(p.float() for p in partials[4:])).to(torch.bfloat16)
    once = ref.float().to(torch.bfloat16)
    err = lambda o: float((o.double() - ref).pow(2).mean().sqrt())
    assert err(once) <= err(today)
