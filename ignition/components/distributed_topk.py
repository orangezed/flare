# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Exact selection from context shards, with bounded candidate communication."""

import torch
from b12x.attention.dsa_indexer.tiled_topk import merge_tiled_topk_candidates

from .collectives import PreparedAllGather


class PreparedDistributedTopK:
    """Merge each rank's local top-k into global logical positions.

    Context position ``p`` belongs to rank ``p % world`` at local offset ``p // world``.
    Every rank contributes k candidates: keeping only k/world could discard global winners.
    Candidate IDs travel as integer bits alongside FP32 scores; the collective transports bytes.
    Missing local positions have score -inf and ID -1. All ranks receive identical candidates.
    """

    def __init__(self, indices, scores, output, *, world, rank, runtime, session, coordinator,
                 name, buffers):
        if (indices.ndim != 2 or indices.dtype != torch.int32 or scores.shape != indices.shape
                or scores.dtype != torch.float32 or output.shape != indices.shape or output.dtype != indices.dtype
                or world < 2 or rank.shape != (1,) or rank.dtype != torch.int64):
            raise ValueError("distributed top-k requires matching int32 positions, FP32 scores and a rank tensor")
        rows, self.topk = indices.shape
        self.indices, self.scores, self.output = indices, scores, output
        self.world, self.rank = world, rank
        self.local = buffers.zeros((rows, self.topk, 2), torch.float32)
        self.gathered = buffers.zeros((rows, world * self.topk, 2), torch.float32)
        self.values = buffers.zeros(indices.shape, torch.float32)
        self.merge_positions = buffers.zeros(indices.shape, torch.int64)
        self.collective = PreparedAllGather(self.local.view(torch.bfloat16).view(rows, -1),
            self.gathered.view(torch.bfloat16).view(rows, -1), runtime=runtime, session=session,
            coordinator=coordinator, name=name)

    def __call__(self):
        live = self.indices >= 0
        self.local[:, :, 0].copy_(torch.where(live, self.scores, -torch.inf))
        self.local.view(torch.int32)[:, :, 1].copy_(
            torch.where(live, self.indices.long() * self.world + self.rank, -1))
        self.collective(self.local.view(torch.bfloat16).view(self.indices.shape[0], -1))
        rows = self.indices.shape[0]
        values = self.gathered[:, :, 0].view(rows, self.world, self.topk).permute(1, 0, 2)
        indices = self.gathered.view(torch.int32)[:, :, 1].view(rows, self.world, self.topk).permute(1, 0, 2)
        merge_tiled_topk_candidates(candidate_values=values, candidate_indices=indices, topk=self.topk,
            output_values=self.values, output_indices=self.output, merge_positions=self.merge_positions)
        return self.output


def selected_local_records(selected, positions, pages, *, world, rank, page_size):
    """Compact selected global positions into this rank's physical records.

    ``pages`` is a model-page table per query row. The selected positions are independently
    bounded by that row's causal position; negative positions denote inactive rows.
    """
    live = (selected >= 0) & (selected <= positions[:, None]) & (selected % world == rank)
    logical = torch.sort(torch.where(live, selected, torch.iinfo(torch.int32).max), dim=1).values
    count = live.sum(1).to(torch.int32)
    local = logical.long() // world
    page = pages.gather(1, (local // page_size).clamp_max(pages.shape[1] - 1))
    present = torch.arange(selected.shape[1], device=selected.device)[None] < count[:, None]
    physical = torch.where(present & (page >= 0), page.long() * page_size + local % page_size, -1)
    return physical.to(torch.int32), count
