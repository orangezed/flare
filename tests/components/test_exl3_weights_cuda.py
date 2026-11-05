# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""EXL3 extent loading, persistent scale ownership, and allocation-free graph replay."""

import os

import pytest
import torch

from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import IndexedCheckpoint

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("first,last", [(0, 256), (256, 512)])
@pytest.mark.parametrize("tokens", [1, 17])
@torch.no_grad()
def test_exl3_loader_matches_direct_preparation_and_survives_source_release(tmp_path, first, last, tokens):
    from b12x.moe import fused_moe as moe
    from b12x.moe._shared.kernels.w4a16.exl3 import prepare_exl3_moe_weights
    from b12x.moe._shared.kernels.w4a16.exl3_synth import Exl3SynthConfig, write_exl3_checkpoint
    from b12x.moe.checkpoints.exl3 import read_exl3_layer
    from ignition.components.moe import PreparedRoutedExperts
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    hidden = 512
    config = Exl3SynthConfig(codebook="lut_e4m3", num_experts=3, hidden_size=hidden,
        intermediate_size=512, moe_layer_indices=(1,), bits=2, intermediate_hadamard=True,
        pre_block=512, post_block=128,
        extent_alignment_slots=4, extent_barriers=(8,), seed=41)
    manifest = write_exl3_checkpoint(tmp_path, config)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    checkpoint = IndexedCheckpoint(tmp_path, {}, ())
    with DirectWeightLoader(checkpoint, device) as loader:
        experts, side_tables = loader.load_exl3_experts(1, num_experts=3, hidden_size=hidden,
            intermediate_size=512, intermediate_range=(first, last), activation="situ")
    layer = read_exl3_layer(tmp_path, manifest, 1, first_slot=first // 32, slot_count=(last - first) // 32)
    direct = prepare_exl3_moe_weights(layer, activation="situ", device=device)
    for name in ("w13", "w2", "gate_suh", "up_suh", "intermediate_rotations", "down_svh"):
        actual = getattr(experts._impl.representation.value, name)
        torch.testing.assert_close(actual, getattr(direct, name), rtol=0, atol=0)
    assert side_tables is experts._impl.representation.value.trellis
    assert experts.plan.activation.mode == "a16"
    # CUDA experts must own all numeric tables after closing and deleting the CPU source.
    for tensor in (layer.codes, layer.rotations, layer.gate_suh, layer.up_suh, layer.down_svh):
        tensor.zero_()
    generator = torch.Generator(device=device).manual_seed(92 + tokens)
    source = .05 * torch.randn((tokens, hidden), generator=generator, device=device, dtype=torch.bfloat16)
    ids = torch.stack((torch.arange(tokens, device=device) % 3,
                       (torch.arange(tokens, device=device) + 1) % 3), -1).to(torch.int32)
    probabilities = torch.full((tokens, 2), .5, device=device)
    with RecordedPreparationSession(device=device, checkpoint="synthetic-exl3-lut-e4m3-2bit",
            workload_schema="exl3-extent-replay", numerical_policy="a16-situ",
            autotune_effort="off", compile_workers=1) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        output = torch.empty_like(source)
        operation = PreparedRoutedExperts(source, experts, ids, probabilities, output,
            session=session, name="exl3", workspace=arena,
            routing=moe.RoutingSpec(deterministic_output=True))
        session.tune(show_progress=False)
        operation()
        expected = output.clone()
        assert torch.isfinite(expected).all() and torch.count_nonzero(expected) > 0
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operation()
        output.fill_(float("nan"))
        allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize(device)
        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
