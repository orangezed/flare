# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from ignition.components.collectives import PreparedPromotedAllReduce


def test_promoted_reduction_accepts_disjoint_views_of_shared_storage():
    storage = torch.zeros(512, dtype=torch.float32)
    source = torch.randn(4, 16).bfloat16()
    output = torch.empty_like(source)
    op = PreparedPromotedAllReduce(source, output, accumulation_buffers=(storage[:256], storage[256:]),
        runtime=None, session=None, coordinator=None, name="single")
    for sign in (1, -1, 1):
        source.mul_(sign)
        storage.fill_(float("nan"))
        assert torch.equal(op(source), source)


def test_promoted_reduction_rejects_overlapping_workspace():
    storage = torch.zeros(128, dtype=torch.float32)
    source = torch.randn(4, 16).bfloat16()
    with pytest.raises(ValueError, match="overlap"):
        PreparedPromotedAllReduce(source, torch.empty_like(source),
            accumulation_buffers=(storage[:64], storage[32:96]),
            runtime=None, session=None, coordinator=None, name="invalid")


def test_promoted_reduction_rejects_insufficient_capacity():
    source = torch.randn(4, 16).bfloat16()
    with pytest.raises(ValueError, match="capacity"):
        PreparedPromotedAllReduce(source, torch.empty_like(source),
            accumulation_buffers=(torch.zeros(63), torch.zeros(64)),
            runtime=None, session=None, coordinator=None, name="invalid")


def test_promoted_reduction_rejects_cross_device_output():
    source = torch.randn(4, 16).bfloat16()
    with pytest.raises(ValueError, match="BF16 activations"):
        PreparedPromotedAllReduce(source, torch.empty_like(source, device="meta"),
            accumulation_buffers=(torch.zeros(64), torch.zeros(64)),
            runtime=None, session=None, coordinator=None, name="invalid")


def test_promoted_reduction_rejects_empty_shape():
    source = torch.empty((0, 16), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="BF16 activations"):
        PreparedPromotedAllReduce(source, torch.empty_like(source),
            accumulation_buffers=(torch.zeros(64), torch.zeros(64)),
            runtime=None, session=None, coordinator=None, name="invalid")
