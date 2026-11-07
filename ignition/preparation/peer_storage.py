# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Symmetric VMM mappings for b12x collectives during rank preparation."""

import ctypes as C
import os
import pickle
from math import prod

try:
    from multiprocessing.reduction import DupFd
except ImportError:
    # The forward pickler imports this module on Windows; descriptor passing is POSIX-only.
    DupFd = None

from b12x.comm.pcie._cuda_ipc import CudaRTLibrary


class _Location(C.Structure):
    _fields_ = [("kind", C.c_int), ("device", C.c_int)]


class _Access(C.Structure):
    _fields_ = [("location", _Location), ("flags", C.c_uint64)]


class PeerStorage(CudaRTLibrary):
    """Use rank arenas and descriptor passing for the b12x shared-buffer API."""

    def __init__(self, storage):
        super().__init__()
        self.storage = storage
        self.owned = {}
        self.imported = {}
        self.imported_mappings = {}
        self.dynamic = {}
        storage.library.flare_package_allocate.restype = C.c_void_p

    def cudaMalloc(self, size):
        pointer = self.storage.library.flare_package_allocate(
            C.c_size_t(size), C.c_int(self.storage.device), None)
        if pointer is None:
            raise self.storage.error or MemoryError("collective allocation failed")
        self.owned[pointer] = self.storage.mappings[pointer].nbytes
        return pointer

    def cudaFree(self, pointer):
        self.storage.library.flare_package_free(C.c_void_p(pointer), C.c_size_t(0),
                                               C.c_int(self.storage.device), None)
        if self.storage.error:
            raise self.storage.error
        del self.owned[pointer]

    def cudaIpcGetMemHandleBytes(self, pointer):
        if DupFd is None:
            raise RuntimeError("peer storage requires POSIX descriptor passing")
        cuda = self.storage.cuda
        region = self.dynamic.get(pointer)
        mappings = [(pointer, self.owned[pointer])] if region is None else region.mappings
        parts = []
        for address, size in mappings:
            physical, descriptor = C.c_uint64(), C.c_int()
            cuda.call("cuMemRetainAllocationHandle", C.byref(physical), C.c_void_p(address))
            try:
                cuda.call("cuMemExportToShareableHandle", C.byref(descriptor), physical,
                          C.c_int(1), C.c_uint64(0))
            finally:
                cuda.call("cuMemRelease", physical)
            try:
                descriptors = [None if rank == self.storage.rank else DupFd(descriptor.value)
                               for rank in range(self.storage.world)]
                parts.append((address, size, descriptors))
            finally:
                os.close(descriptor.value)
        return pickle.dumps((pointer, self.owned[pointer], parts))

    def cudaIpcOpenMemHandleBytes(self, data):
        base, extent, parts = pickle.loads(data)
        cuda = self.storage.cuda
        mapped = []
        try:
            for address, size, descriptors in parts:
                descriptor = descriptors[self.storage.rank].detach()
                physical = C.c_uint64()
                try:
                    cuda.call("cuMemImportFromShareableHandle", C.byref(physical),
                              C.c_void_p(descriptor), C.c_int(1))
                finally:
                    os.close(descriptor)
                try:
                    cuda.call("cuMemMap", C.c_uint64(address), C.c_size_t(size), C.c_size_t(0),
                              physical, C.c_uint64(0))
                finally:
                    cuda.call("cuMemRelease", physical)
                mapped.append((address, size))
                access = _Access(_Location(1, self.storage.device), 3)
                cuda.call("cuMemSetAccess", C.c_uint64(address), C.c_size_t(size), C.byref(access), C.c_size_t(1))
        except BaseException:
            for address, size in reversed(mapped):
                cuda.call("cuMemUnmap", C.c_uint64(address), C.c_size_t(size))
            raise
        self.imported[base] = extent
        self.imported_mappings[base] = mapped
        return base

    def cudaIpcCloseMemHandle(self, pointer):
        for address, size in self.imported_mappings.pop(pointer):
            self.storage.cuda.call("cuMemUnmap", C.c_uint64(address), C.c_size_t(size))
        del self.imported[pointer]

    def share_regions(self, group, regions):
        """Map every rank's backed cache prefixes, retaining their full virtual extents for export."""
        import torch.distributed as dist

        payloads = []
        for region in regions:
            self.dynamic[region.address] = region
            self.owned[region.address] = region.extent
            payloads.append(self.cudaIpcGetMemHandleBytes(region.address))
        ranks = [None] * self.storage.world
        dist.all_gather_object(ranks, payloads, group=group)
        addresses = [[region.address] * self.storage.world for region in regions]
        for rank, payloads in enumerate(ranks):
            if len(payloads) != len(regions):
                raise ValueError("ranks must share the same cache regions")
            if rank != self.storage.rank:
                for index, payload in enumerate(payloads):
                    addresses[index][rank] = self.cudaIpcOpenMemHandleBytes(payload)
        return addresses

    def reset(self):
        """Reset all local signals and generation-tagged payloads after group drain."""
        import flare_heliopause

        context, _ = flare_heliopause.current_context()
        ranges = [(address, size) for address, size in self.owned.items() if address not in self.dynamic]
        flare_heliopause.export_reset_peer_storage(context, ranges, self)


class SharedDeviceBuffer:
    """One contiguous peer-visible allocation per rank, retained through graph drain.

    Views borrow this object's storage. Synchronization between producers and
    consumers during replay belongs in their captured graphs; the host barriers
    here are used only for preparation and destruction.
    """

    def __init__(self, group, peer_storage, device, capacity):
        import torch
        import torch.distributed as dist

        if type(capacity) is not int or capacity <= 0:
            raise ValueError("shared device buffer capacity must be a positive byte count")
        self.group, self.peer_storage, self.device = group, peer_storage, torch.device(device)
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        self.capacity = capacity
        self.pointer = peer_storage.cudaMalloc(capacity)
        self.extent = peer_storage.owned[self.pointer]
        peer_storage.cudaMemset(self.pointer, 0, self.extent)
        handles = [None] * self.world
        dist.all_gather_object(handles, peer_storage.cudaIpcGetMemHandleBytes(self.pointer), group=group)
        self.bases = [self.pointer if rank == self.rank else peer_storage.cudaIpcOpenMemHandleBytes(handle)
                      for rank, handle in enumerate(handles)]

    def view(self, rank, offset, shape, dtype):
        import torch

        nbytes = prod(shape) * dtype.itemsize
        if offset < 0 or offset % dtype.itemsize or offset + nbytes > self.extent:
            raise ValueError("shared tensor view exceeds its aligned allocation")
        storage = torch._C._construct_storage_from_data_pointer(self.bases[rank] + offset, self.device, nbytes)
        strides = tuple(prod(shape[index + 1:]) for index in range(len(shape)))
        return torch.empty(0, dtype=dtype, device=self.device).set_(storage, 0, shape, strides)

    def exchange(self):
        import torch
        import torch.distributed as dist

        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)

    def close(self):
        self.exchange()
        for rank, base in enumerate(self.bases):
            if rank != self.rank:
                self.peer_storage.cudaIpcCloseMemHandle(base)
        self.exchange()
        self.peer_storage.cudaFree(self.pointer)
