# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Projection trials release completed expert fixtures before allocating their inputs."""

import os
import weakref

import pytest
import torch

from ignition.components.moe import RouterScratch
from ignition.preparation.fixtures.fused_moe import ExpertBenchmarkInputs
from ignition.preparation.fixtures.projection import input_factory
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an assigned, otherwise unused GPU")


@torch.no_grad()
def test_projection_can_follow_expert_tuning_under_memory_pressure():
    device = torch.device("cuda:0")
    source = torch.empty((4096, 7168), dtype=torch.bfloat16, device=device)
    output = torch.empty((4096, 768), dtype=torch.bfloat16, device=device)
    expert_source = torch.empty((4096, 3584), dtype=torch.bfloat16, device=device)
    with RecordedPreparationSession(device=device, checkpoint="fixture-memory",
            workload_schema="expert-then-projection", numerical_policy="bf16") as session:
        # Isolate the real fixture lifecycle without loading model weights or racing kernels.
        session._tune_status = "tuning"
        previous = weakref.ref(session.benchmark_resource("expert", lambda: ExpertBenchmarkInputs(
            expert_source, experts=896, top_k=16, allocate_router=RouterScratch.allocate)))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info(device)[0]
        assert free > 512 << 20, "assign a GPU with at least 512 MiB free"
        reserve = torch.empty(free - (40 << 20), dtype=torch.uint8, device=device)
        try:
            # The 56 MiB input cannot coexist with the completed expert's cached fixture.
            with pytest.raises(torch.OutOfMemoryError):
                input_factory(source, output)()
            operands = input_factory(source, output, session=session)()
            assert previous() is None
            operands.produce()
            torch.cuda.synchronize()
        finally:
            del reserve
            torch.cuda.empty_cache()
        torch.testing.assert_close(operands.source, operands.seed, rtol=0, atol=0)
