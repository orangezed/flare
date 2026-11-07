# SPDX-License-Identifier: Apache-2.0

"""Sparse prefill with gathered FP8 context and query-row-sharded index selection."""

import torch
from b12x.attention import dense_mla

from ignition.components.collectives import PreparedAllGather
from ignition.components.dense_mla import PreparedDenseMLA
from ignition.components.fp8_indexer import PreparedFP8Indexer
from ignition.preparation.scratch import GraphBufferPool

from .config import PAGE_TOKENS
from .dcp import RECORD_BYTES, write_records
from .dsa_context import PreparedContextTransfer
from .projections import prepare_projection


class PreparedGatheredDSA:
    """Each rank selects a disjoint strided subset of query rows over the whole context.

    Only final selected positions cross ranks. Every rank then attends its original
    heads to the gathered FP8 latents. Past context is immutable; current-chunk keys
    and latents are available on every rank and written directly into the slab.
    Selection buffers are reused between slices and graph capacities.
    """

    def __init__(self, weights, residual, key_input, head_weights, query, output, slots,
                 metadata, cache, index_cache, *, context, session, coordinator, name,
                 workspace, buffers, latent, eps):
        if (metadata.starts.numel() != 1 or latent is None or context.staging is None
                or context.slab is None or context.index_slab is None or context.selections is None):
            raise ValueError("query-sharded sparse prefill requires one request and shared index/latent context")
        self.metadata, self.context, self.output = metadata, context, output
        self.latent, self.slots = latent, slots
        self.query, self.residual, self.key_input, self.head_weights = query, residual, key_input, head_weights
        self.norm_weight, self.norm_bias, self.eps = weights["index_norm"], weights["index_bias"], eps
        rows, heads, _ = query.shape
        world = context.shards.world
        self.rows, self.world = rows, world
        self.local_rows = min(32, (rows + world - 1) // world)
        self.step = self.local_rows * world
        device = query.device
        # A separate cursor namespace prevents aliasing buffers retained by the DCP decode path.
        if "gathered_prefill_pool" not in buffers:
            buffers["gathered_prefill_pool"] = GraphBufferPool(device, chunk_bytes=2 << 20)
        cursor = buffers["gathered_prefill_pool"].cursor()
        def zeros(shape, dtype=torch.bfloat16):
            return cursor.zeros(shape, dtype)
        self.index_query = zeros((rows, 32, 128))
        self.key = zeros((rows, 128))
        self.project = prepare_projection(residual, weights["index_query"], self.index_query.view(rows, -1),
            session=session, name=name + ".index_query", workspace=workspace)
        self.row_ids = torch.arange(self.local_rows, dtype=torch.int64, device=device) * world
        self.global_ids = torch.arange(self.step, dtype=torch.int64, device=device)
        self.selected_rows = zeros((self.local_rows,), torch.int64)
        self.local_query = zeros((self.local_rows, 32, 128))
        self.local_weights = zeros((self.local_rows, 32))
        self.lengths = zeros((self.local_rows,), torch.int32)
        self.active = zeros((1,), torch.int32)
        self.pages = zeros((self.local_rows, context.index_slab.shape[0]), torch.int32)
        self.page_ids = torch.arange(context.index_slab.shape[0], dtype=torch.int32, device=device)
        self.selected = zeros((self.local_rows, 2048), torch.int32)
        self.gathered = zeros((self.local_rows, world * 2048), torch.int32)
        self.window = zeros((2,), torch.int64)
        self.transfer = PreparedContextTransfer(cache, index_cache, metadata, self.window[1:],
            context=context, session=session, coordinator=coordinator, name=name + ".context")
        self.selector = PreparedFP8Indexer(self.local_query, self.local_weights, context.index_slab,
            self.pages, self.lengths, self.active, self.selected, mode="decode", session=session,
            name=name + ".index", workspace=workspace, buffers=cursor, rotation="none")
        # The cache writer is also used on the committed sharded index pool.
        self.index_cache = index_cache
        self.gather = PreparedAllGather(self.selected.view(torch.float32), self.gathered.view(torch.float32),
            runtime=context.selections, session=session, coordinator=coordinator, name=name + ".selected_positions")
        self.physical = zeros((self.step, 2048), torch.int32)
        self.counts = zeros((self.step,), torch.int32)
        self.offsets = torch.arange(self.step + 1, dtype=torch.int32, device=device)
        self.quantized = zeros((self.step, heads, RECORD_BYTES), torch.float8_e4m3fn)
        self.attended = zeros((self.step, heads, 512))
        caps = dense_mla.Caps(device=device, mode="decode", q_dtype=self.quantized.dtype,
            kv_dtype=context.slab.dtype, num_q_heads=heads, max_total_q=self.step, max_batch=self.step,
            page_size=1, max_cache_tokens=2048, max_page_table_width=2048, num_cache_pages=context.slab.shape[0],
            use_cuda_graph=True, partial_dtype=torch.float32)
        self.attention = PreparedDenseMLA(self.quantized, context.slab.view(-1, 1, RECORD_BYTES), self.attended,
            self.physical, self.counts, self.offsets, caps=caps, scale=192**-.5, session=session,
            name=name + ".sparse_attention", workspace=workspace)

    def __call__(self):
        from ignition.components.fp8_indexer import _write_keys

        m, c = self.metadata, self.context
        self.project(self.residual)
        self.key.copy_(torch.nn.functional.layer_norm(self.key_input.float(), (128,),
            self.norm_weight.float(), self.norm_bias.float(), self.eps).to(torch.bfloat16))
        torch.library.wrap_triton(_write_keys)[(self.rows,)](self.key,
            self.index_cache.view(torch.float8_e4m3fn), self.index_cache.view(torch.float32),
            self.slots, self.index_cache.stride(0), False, num_warps=4)
        self.window[0:1].zero_()
        self.window[1:].copy_(m.starts * (m.lengths > 0))
        self.transfer()
        write_records(self.latent, m.positions, c.slab)
        self.selector.write_keys(self.key, m.positions)
        self.pages.copy_(self.page_ids[None])
        for begin in range(0, self.rows, self.step):
            self.selected_rows.copy_(begin + self.row_ids + c.shards.owner)
            safe = self.selected_rows.clamp_max(self.rows - 1)
            self.local_query.copy_(self.index_query[safe])
            self.local_weights.copy_(self.head_weights[safe])
            self.lengths.copy_(torch.where(self.selected_rows < self.rows, (m.positions[safe] + 1).clamp_min(0), 0))
            self.active.copy_(self.lengths.max().reshape(1))
            self.selector()
            self.gather(self.selected.view(torch.float32))
            end = min(begin + self.step, self.rows)
            count = end - begin
            # All-gather concatenates ranks inside each local row: the resulting rows
            # already have the original token order for strided ownership.
            selected = self.gathered.view(self.step, 2048)
            positions = m.positions[(begin + self.global_ids).clamp_max(self.rows - 1)]
            live = (selected >= 0) & (selected <= positions[:, None]) & (self.global_ids[:, None] < count)
            ordered = torch.sort(torch.where(live, selected, torch.iinfo(torch.int32).max), dim=1).values
            self.physical.copy_(torch.where(ordered != torch.iinfo(torch.int32).max, ordered, -1))
            self.counts.copy_(live.sum(1).to(torch.int32))
            self.quantized.zero_()
            self.quantized[:count].copy_(self.query[begin:end].float().clamp(-448., 448.).to(torch.float8_e4m3fn))
            self.attention()
            self.output[begin:end].copy_(self.attended[:count])
        return self.output
