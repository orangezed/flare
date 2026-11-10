# SPDX-License-Identifier: Apache-2.0
"""CPU tests: the copy-engine prefill ring switch (IGNITION_KIMI_PREFILL_DMA_ROWS / _BF16) stays prefill-only.

Hand-port of tunamitom/flare-local-inference@bc818b1 + @2f0acab without their island / fp8 commits.
"""

import inspect

import pytest

from ignition.components.capacity import prefill_capacities
from ignition.recipes.kimi3 import bootstrap, recipe
from ignition.recipes.kimi3.bootstrap import PreparationConfig, prefill_reduction_rows, reduction_runtime


def test_switch_defaults_off():
    config = PreparationConfig(devices=tuple(range(8)))
    assert config.prefill_dma_ring_rows == 0 and config.prefill_dma_ring_bf16 is False


@pytest.mark.parametrize("rows", [-1, 1.5, "3072"])
def test_switch_rejects_bad_rows(rows):
    with pytest.raises(ValueError):
        PreparationConfig(devices=tuple(range(8)), prefill_dma_ring_rows=rows)


def test_off_keeps_every_prefill_capacity_on_the_exact_channel():
    assert prefill_capacities(3072) == (128, 256, 512, 1024, 1536, 2048, 2560, 3072)
    assert prefill_reduction_rows(8, 3072, 0) == (0, 3072)


@pytest.mark.parametrize("threshold, expected", [(3072, (3072, 2560)), (2048, (2048, 1536)), (2049, (2049, 2048)),
                                                 (1024, (1024, 512)), (128, (128, 0))])
def test_threshold_splits_capacities(threshold, expected):
    assert prefill_reduction_rows(8, 3072, threshold) == expected


@pytest.mark.parametrize("world, chunk, threshold", [(4, 3072, 3072), (16, 3072, 3072), (8, 3072, 4096), (8, 3072, 0)])
def test_ring_only_at_world_eight_within_the_chunk(world, chunk, threshold):
    assert prefill_reduction_rows(world, chunk, threshold) == (0, chunk)


def test_decode_verify_and_draft_never_select_the_ring():
    collective, exact, ring = object(), object(), object()
    for mode in ("decode", "verify", "draft", "dspark"):
        for capacity in (1, 4, 10, 32, 3072, 1 << 20):
            assert reduction_runtime(mode, capacity, collective=collective, prefill_collective=exact,
                                     prefill_dma=ring, dma_rows=1) is collective


@pytest.mark.parametrize("threshold", [0, 2048, 3072])
def test_prefill_selection_matches_threshold(threshold):
    collective, exact, ring = object(), object(), object()
    dma_rows, small_rows = prefill_reduction_rows(8, 3072, threshold)
    dma = ring if dma_rows else None
    for capacity in prefill_capacities(3072):
        chosen = reduction_runtime("prefill", capacity, collective=collective, prefill_collective=exact,
                                   prefill_dma=dma, dma_rows=dma_rows)
        if dma_rows and capacity >= dma_rows:
            assert chosen is ring
        else:
            assert chosen is exact and capacity <= small_rows


def test_prefill_without_exact_channel_falls_back_to_collective():
    collective, ring = object(), object()
    assert reduction_runtime("prefill", 128, collective=collective, prefill_collective=None, prefill_dma=None,
                             dma_rows=0) is collective
    assert reduction_runtime("prefill", 3072, collective=collective, prefill_collective=None, prefill_dma=ring,
                             dma_rows=128) is ring


def test_decoder_construction_uses_the_selector_only():
    source = inspect.getsource(bootstrap.prepare_rank)
    assert source.count("PreparedDecoder(") == 1
    assert "runtime=reduction_runtime(mode, capacity, collective=collective," in source
    # the ring is captured, closed and retained like the exact prefill channel
    assert "for channel in (prefill_collective, prefill_dma):" in source
    assert "*(() if prefill_dma is None else (prefill_dma,))" in source
    assert 'wire_mode="off"' in source and "B12X_PCIE_DMA_PIECES" in source


def test_recipe_reads_the_export_env():
    source = inspect.getsource(recipe.prepare_variant)
    assert 'os.environ.get("IGNITION_KIMI_PREFILL_DMA_ROWS", "0")' in source
    assert 'os.environ.get("IGNITION_KIMI_PREFILL_DMA_BF16", "0") == "1"' in source


def test_decoder_ring_branches_reject_non_prefill_modes():
    from ignition.recipes.kimi3 import decoder

    source = inspect.getsource(decoder.PreparedDecoder.__init__)
    head, rest = source.split('getattr(runtime, "kimi_bf16_ring", False)', 1)
    bf16_branch, owner_branch = rest.split("elif isinstance(runtime, DmaAllReduce):", 1)
    assert 'if mode != "prefill":' in bf16_branch and "raise ValueError" in bf16_branch
    assert 'if mode != "prefill":' in owner_branch.split("count =", 1)[0]
