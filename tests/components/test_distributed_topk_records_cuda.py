"""Sorted logical records retain ownership, visibility, and page-table semantics."""
import os

import pytest
import torch

from ignition.components.distributed_topk import selected_local_records

pytestmark = pytest.mark.skipif(
    os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires one assigned CUDA GPU")


def reference(selected, positions, pages, world, rank, page_size):
    records = torch.full_like(selected, -1, device="cpu")
    counts = torch.zeros(selected.shape[0], dtype=torch.int32)
    for row, values in enumerate(selected.cpu().tolist()):
        chosen = sorted(value for value in values
                        if 0 <= value <= positions[row] and value % world == rank)
        counts[row] = len(chosen)
        for slot, value in enumerate(chosen):
            local = value // world
            page = int(pages[row, min(local // page_size, pages.shape[1]-1)])
            if page >= 0:
                records[row, slot] = page*page_size + local % page_size
    return records, counts


@pytest.mark.parametrize("world,tensor_rank", [(1, False), (1, True), (4, True), (8, True), (16, True)])
def test_changed_owners_visibility_and_page_order(world, tensor_rank):
    selected = torch.empty((4, 4096), dtype=torch.int32, device="cuda")[:, ::2]
    positions = torch.empty((8,), dtype=torch.int64, device="cuda")[::2]
    pages = torch.empty((4, 1024), dtype=torch.int32, device="cuda")[:, ::2]
    rank = torch.zeros(1, dtype=torch.int64, device="cuda") if tensor_rank else 0
    selected.fill_(-1)
    positions.zero_()
    pages.zero_()
    compiled = torch.compile(selected_local_records, fullgraph=True, dynamic=False)
    kwargs = dict(world=world, rank=rank, page_size=128)
    for _ in range(3):
        compiled(selected, positions, pages, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = compiled(selected, positions, pages, **kwargs)
    for owner in range(world):
        if tensor_rank:
            rank.fill_(owner)
        torch.manual_seed(801+owner)
        for mode in range(4):
            selected.random_(-100, 1000000)
            selected[:, ::11] = -1
            if mode == 1:
                selected.copy_(torch.arange(2048, device="cuda")[None]*world+owner)
            if mode == 2:
                selected.fill_(2147483647)
            if mode == 3:
                selected.fill_(-1)
            positions.copy_(torch.tensor([-1, 128, 999999, 2147483647], device="cuda"))
            pages.copy_(torch.randperm(512, dtype=torch.int32, device="cuda")[None])
            pages[:, ::17] = -1
            expected = reference(selected, positions.cpu(), pages.cpu(), world, owner, 128)
            for value in output:
                value.fill_(-777)
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            for actual, wanted in zip(output, expected):
                torch.testing.assert_close(actual.cpu(), wanted, atol=0, rtol=0)
