# SPDX-License-Identifier: Apache-2.0

"""Bounded FP32 DMA reduction retaining the full tensor's ring-owner order."""

import torch
from b12x.comm.pcie import DmaAllReduce

from .collectives import PreparedAllReduce


class PreparedOwnerPacketAllReduce:
    """Promote, reduce in place, and round once without a full-sized FP32 copy.

    Each packet takes the same interval from every original owner shard.
    Reducing consecutive token chunks instead would change the ring's FP32
    addition order. Input/output may be exact aliases or disjoint; scratch is
    caller-owned and disjoint from both. Calls sharing the channel serialize.
    """

    def __init__(
        self, source, output, *, workspace, runtime, session, coordinator, name
    ):
        if not isinstance(runtime, DmaAllReduce):
            raise ValueError("owner-packet reduction requires a prepared DMA ring")
        if (
            source.ndim != 2
            or source.numel() == 0
            or source.dtype != torch.bfloat16
            or output.shape != source.shape
            or output.dtype != source.dtype
            or source.device != output.device
            or not source.is_contiguous()
            or not output.is_contiguous()
        ):
            raise ValueError(
                "owner-packet reduction requires matching contiguous BF16 matrices"
            )
        if (
            workspace.device != source.device
            or workspace.dtype != torch.float32
            or not workspace.is_contiguous()
        ):
            raise ValueError(
                "owner-packet reduction requires caller-owned contiguous FP32 scratch"
            )

        def interval(tensor):
            return tensor.data_ptr(), tensor.data_ptr() + tensor.nbytes

        def overlaps(a, b):
            return a[0] < b[1] and b[0] < a[1]

        source_range, output_range, scratch_range = map(
            interval, (source, output, workspace)
        )
        if (
            overlaps(scratch_range, source_range)
            or overlaps(scratch_range, output_range)
            or source_range != output_range
            and overlaps(source_range, output_range)
        ):
            raise ValueError(
                "owner-packet scratch and partial input/output overlaps are not supported"
            )
        self.world = runtime.world_size
        if source.numel() % (self.world * 8):
            raise ValueError(
                "ring owner shards must contain complete 32-byte FP32 groups"
            )
        self.extent = source.numel() // self.world
        capacity = min(workspace.numel(), runtime.max_bytes // 4) // self.world // 8 * 8
        if capacity == 0:
            raise ValueError(
                "owner-packet scratch cannot hold one aligned element group per rank"
            )
        self.output, self.parts = output, []
        self.workspace = workspace
        for first in range(0, self.extent, capacity):
            end = min(first + capacity, self.extent)
            packet = workspace.view(-1)[: self.world * (end - first)].view(
                self.world, end - first
            )
            reduce = PreparedAllReduce(
                packet,
                packet,
                runtime=runtime,
                session=session,
                coordinator=coordinator,
                name=f"{name}.owner_packet{len(self.parts)}",
            )
            self.parts.append((first, end, packet, reduce))

    def __call__(self, source):
        source = source.view(self.world, self.extent)
        output = self.output.view(self.world, self.extent)
        for first, end, packet, reduce in self.parts:
            packet.copy_(source[:, first:end])
            reduce(packet)
            output[:, first:end].copy_(packet)
        return self.output
