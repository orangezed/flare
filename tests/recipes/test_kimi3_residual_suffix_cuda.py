# SPDX-License-Identifier: Apache-2.0

"""Bounded auxiliary stores preserve the complete target residual computation."""

import os

import pytest
import torch

from ignition.recipes.kimi3.residual import ADD, REPLACE, ResidualStream

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an assigned CUDA device",
)


@pytest.mark.parametrize(
    "capacity,alignment", [(1024, 1), (3072, 1), (3072, 256), (4096, 256)]
)
@pytest.mark.parametrize(
    "sources,mode,write", [(0, REPLACE, 0), (1, ADD, -1), (4, ADD, 4), (8, REPLACE, -1)]
)
@torch.no_grad()
def test_compact_mixture_preserves_target_and_suffix(
    capacity, alignment, sources, mode, write
):
    device = torch.device("cuda:0")
    rows, hidden, keep = capacity, 7168, 1024
    retained_capacity = keep if alignment == 1 else keep + alignment
    generator = torch.Generator(device=device).manual_seed(251)
    prefix = torch.randn(
        (rows, hidden), device=device, dtype=torch.bfloat16, generator=generator
    )
    blocks = torch.randn(
        (rows, 8, hidden), device=device, dtype=torch.bfloat16, generator=generator
    )
    delta = torch.randn(
        prefix.shape, device=device, dtype=torch.bfloat16, generator=generator
    )
    score = torch.randn(hidden, device=device, generator=generator) * 0.01
    norm = torch.randn(hidden, device=device, dtype=torch.bfloat16, generator=generator)
    complete = ResidualStream(
        torch.empty_like(prefix), torch.empty_like(blocks), torch.empty_like(prefix)
    )
    bounded = ResidualStream(
        torch.empty_like(prefix), torch.empty_like(blocks), torch.empty_like(prefix)
    )
    full = torch.empty_like(prefix)
    # A column view matches the six-tap feature tensor used by the decoder.
    backing = torch.full(
        (retained_capacity, hidden * 2),
        float("nan"),
        device=device,
        dtype=torch.bfloat16,
    )
    suffix = backing[:, hidden:]
    count = torch.zeros(1, device=device, dtype=torch.int32)
    kwargs = dict(
        sources=sources,
        mode=mode,
        write=write,
        score=score,
        norm=norm,
        eps=1e-5,
        output_eps=1e-5,
    )

    def run():
        for stream in (complete, bounded):
            stream.prefix.copy_(prefix)
            stream.blocks.copy_(blocks)
        complete.mix(delta, mixture=full, **kwargs)
        bounded.mix(
            delta,
            mixture=suffix,
            mixture_count=count,
            mixture_window=keep,
            mixture_alignment=alignment,
            **kwargs,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for live in sorted({0, 1, 513, 1023, 1024, capacity - 1, capacity}):
        count.fill_(live)
        suffix.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(complete.output, bounded.output)
        assert torch.equal(complete.prefix, bounded.prefix)
        assert torch.equal(complete.blocks, bounded.blocks)
        origin = max(live - keep, 0) // alignment * alignment
        retained = live - origin
        expected = torch.zeros_like(suffix)
        expected[:retained].copy_(full[origin:live])
        assert torch.equal(suffix.view(torch.int16), expected.view(torch.int16))
        assert bool(torch.isnan(backing[:, :hidden]).all())
        prefix.mul_(-1)
    graph.reset()
