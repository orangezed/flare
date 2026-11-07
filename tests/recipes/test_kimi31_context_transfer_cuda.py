# SPDX-License-Identifier: Apache-2.0

"""Check staged history bytes and interleaving without allocating target weights.

Rank buffers reside on one GPU here. The TP prefill test separately qualifies
cross-device synchronization and the prepared reductions.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")


@pytest.mark.parametrize("world,maximum", [(1, 65536), (4, 65536), (16, 1048576)])
@torch.no_grad()
def test_staged_context_bytes(world, maximum):
    from ignition.recipes.kimi3.dsa_context import _pack_history, _unpack_history, RECORD_WORDS
    from ignition.recipes.kimi3.dcp import _launch

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(842)
    rows = maximum // world
    pages = rows // 128
    table = torch.randperm(pages, device=device).int()
    caches = [torch.empty((pages, 128, 576), dtype=torch.uint8, device=device).random_(0, 256) for _ in range(world)]
    indices = [torch.empty((2*pages, 64*132), dtype=torch.uint8, device=device).random_(0, 256) for _ in range(world)]
    stage = torch.empty((world, rows, RECORD_WORDS), dtype=torch.int32, device=device)
    peers = torch.tensor([stage[rank].data_ptr() for rank in range(world)], dtype=torch.int64, device=device)
    owners = [torch.tensor([rank], dtype=torch.int64, device=device) for rank in range(world)]
    end = torch.tensor([maximum], dtype=torch.int64, device=device)
    dependency = torch.zeros(8, dtype=torch.bfloat16, device=device)
    latent = torch.empty((maximum, 144), dtype=torch.int32, device=device)
    index = torch.empty((maximum//64, 2112), dtype=torch.int32, device=device)

    def transfer():
        for rank in range(world):
            _launch(_pack_history, ((rows+3)//4,),
                (caches[rank].view(torch.int32), indices[rank].view(torch.int32), table, end,
                 owners[rank], dependency, stage[rank], pages),
                WORLD=world, PAGE=128, ROWS=4, STRIDE=RECORD_WORDS, num_warps=4)
        _launch(_unpack_history, ((maximum+7)//8,), (peers, end, dependency, latent, index),
            WORLD=world, ROWS=8, STRIDE=RECORD_WORDS, num_warps=4)

    transfer()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        transfer()
    positions = torch.arange(rows, device=device)
    physical = table.long()[positions//128]*128 + positions%128
    index_page = (physical//128)*2 + (physical%128)//64
    key_offsets = (physical%64)[:, None]*32 + torch.arange(32, device=device)[None]
    for count in (0, 1, 63, 64, 127, 128, 129, 2051, maximum, 129, 0):
        for cache, keys in zip(caches, indices, strict=True):
            cache.random_(0, 256)
            keys.random_(0, 256)
        expected_latent = torch.stack([cache.view(torch.int32).view(-1, 144)[physical] for cache in caches], dim=1).reshape(maximum, 144)
        expected_keys = torch.stack([keys.view(torch.int32)[index_page[:, None], key_offsets]
                                    for keys in indices], dim=1).reshape(maximum, 32)
        expected_scales = torch.stack([keys.view(torch.int32)[index_page, 2048+physical%64]
                                      for keys in indices], dim=1).reshape(maximum)
        stage.fill_(12345)
        latent.fill_(-1)
        index.fill_(-1)
        end.fill_(count)
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        actual_keys = index[:, :2048].reshape(maximum, 32)
        actual_scales = index[:, 2048:].reshape(maximum)
        assert torch.equal(latent[:count], expected_latent[:count])
        assert torch.equal(actual_keys[:count], expected_keys[:count])
        assert torch.equal(actual_scales[:count], expected_scales[:count])
        assert (latent[count:] == -1).all()
        assert (actual_keys[count:] == -1).all()
        assert (actual_scales[count:] == -1).all()
    graph.reset()
