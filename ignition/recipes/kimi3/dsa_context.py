# SPDX-License-Identifier: Apache-2.0

"""Transfer admitted sparse-attention history through one shared staging buffer.

KV allocations remain private, including their continuation-row aliases. Each
rank packs its interleaved history locally; captured reductions bracket peer
reads so a following layer cannot overwrite data still being consumed.
"""

import torch
import triton
import triton.language as tl

from ignition.components.collectives import PreparedAllReduce
from ignition.preparation.peer_storage import SharedDeviceBuffer

from .config import PAGE_TOKENS
from .dcp import _launch

# 576 latent bytes, 128 index-key bytes and one FP32 scale; pad to 16 bytes.
RECORD_WORDS = 180


class ContextStaging:
    def __init__(self, group, peer_storage, device, maximum, world, runtime):
        self.runtime = runtime
        self.rows = triton.cdiv(maximum, world)
        self.owner = SharedDeviceBuffer(group, peer_storage, device, self.rows * RECORD_WORDS * 4)
        backing = self.owner.view(self.owner.rank, 0, (self.owner.extent,), torch.uint8)
        self.local = backing[:self.rows * RECORD_WORDS * 4].view(torch.int32).view(self.rows, RECORD_WORDS)
        self.peers = torch.tensor(self.owner.bases, dtype=torch.int64, device=device)
        self.zero = torch.zeros((1, 8), dtype=torch.bfloat16, device=device)
        self.ready, self.completed = torch.zeros_like(self.zero), torch.zeros_like(self.zero)

    def scratch_tensors(self):
        return self.local, self.zero, self.ready, self.completed

    def close(self):
        self.owner.close()


@triton.jit
def _pack_history(Cache, Index, Table, End, Owner, Completed, Destination, columns,
                  WORLD: tl.constexpr, PAGE: tl.constexpr, ROWS: tl.constexpr, STRIDE: tl.constexpr):
    local = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    words = tl.arange(0, 256)
    end = tl.load(End).to(tl.int64)
    rank = tl.load(Owner).to(tl.int64)
    live = (local * WORLD + rank < end) & (tl.load(Completed) == 0)
    block = local // PAGE
    page = tl.load(Table + tl.minimum(block, columns - 1), mask=live, other=-1).to(tl.int64)
    live &= (block < columns) & (page >= 0)
    offset = local % PAGE
    latent = tl.load(Cache + (page[:, None] * PAGE + offset[:, None]) * 144 + words[None, :],
                     mask=live[:, None] & (words[None, :] < 144), other=0)
    index_page = page * 2 + offset // 64
    key = tl.load(Index + index_page[:, None] * 2112 + (offset[:, None] % 64) * 32 + words[None, :] - 144,
                  mask=live[:, None] & (words[None, :] >= 144) & (words[None, :] < 176), other=0)
    scale = tl.load(Index + index_page * 2112 + 2048 + offset % 64, mask=live, other=0)
    values = tl.where(words[None, :] < 144, latent,
                      tl.where(words[None, :] < 176, key, scale[:, None]))
    tl.store(Destination + local[:, None] * STRIDE + words[None, :], values,
             mask=live[:, None] & (words[None, :] < 177))


@triton.jit
def _unpack_history(Addresses, End, Ready, Latents, Index, WORLD: tl.constexpr, ROWS: tl.constexpr, STRIDE: tl.constexpr):
    tokens = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    words = tl.arange(0, 256)
    live = (tokens < tl.load(End).to(tl.int64)) & (tl.load(Ready) == 0)
    source = tl.load(Addresses + tokens % WORLD).to(tl.pointer_type(tl.int32))
    offsets = tokens // WORLD * STRIDE
    values = tl.load(source[:, None] + offsets[:, None] + words[None, :],
                     mask=live[:, None] & (words[None, :] < 176), other=0)
    tl.store(Latents + tokens[:, None] * 144 + words[None, :], values,
             mask=live[:, None] & (words[None, :] < 144))
    tl.store(Index + (tokens[:, None] // 64) * 2112 + (tokens[:, None] % 64) * 32 + words[None, :] - 144,
             values, mask=live[:, None] & (words[None, :] >= 144) & (words[None, :] < 176))
    scale = tl.load(source + offsets + 176, mask=live, other=0)
    tl.store(Index + tokens // 64 * 2112 + 2048 + tokens % 64, scale, mask=live)


class PreparedContextTransfer:
    def __init__(self, cache, index_cache, metadata, end, *, context, session, coordinator, name):
        if (cache.dtype != torch.float8_e4m3fn or cache.ndim != 3 or cache.shape[1:] != (128, 576)
                or index_cache.dtype != torch.uint8 or index_cache.shape != (cache.shape[0] * 2, 64 * 132)
                or not cache.is_contiguous() or not index_cache.is_contiguous()):
            raise ValueError("sparse history transfer requires contiguous FP8 latent and planar index pages")
        self.cache, self.index_cache, self.metadata, self.end = cache, index_cache, metadata, end
        self.context, self.staging = context, context.staging
        self.ready = PreparedAllReduce(self.staging.zero, self.staging.ready,
            runtime=self.staging.runtime, session=session, coordinator=coordinator, name=name + ".ready")
        self.consumed = PreparedAllReduce(self.staging.zero, self.staging.completed,
            runtime=self.staging.runtime, session=session, coordinator=coordinator, name=name + ".consumed")

    def __call__(self):
        c, s, m = self.context, self.staging, self.metadata
        _launch(_pack_history, (triton.cdiv(s.rows, 4),),
            (self.cache.view(torch.int32), self.index_cache.view(torch.int32), m.page_table[0],
             self.end, c.shards.owner, s.completed.view(-1), s.local, m.page_table.shape[1]),
            WORLD=c.shards.world, PAGE=PAGE_TOKENS, ROWS=4, STRIDE=RECORD_WORDS, num_warps=4)
        self.ready(s.zero)
        _launch(_unpack_history, (triton.cdiv(c.slab.shape[0], 8),),
            (s.peers, self.end, s.ready.view(-1), c.slab.view(torch.int32), c.index_slab.view(torch.int32)),
            WORLD=c.shards.world, ROWS=8, STRIDE=RECORD_WORDS, num_warps=4)
        self.consumed(s.zero)
