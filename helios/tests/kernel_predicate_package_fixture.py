"""Export ordinary/cooperative launch alternatives for native startup selection.

Run this script on one assigned GPU, then run the ignored Rust test
``startup_kernel_predicates_replay_only_the_selected_launch`` with
``FLARE_KERNEL_PREDICATE_TEST_PACKAGE`` pointing to its output directory.
"""

import ctypes as C
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

from cuda_graph_fixture import KernelParams
from fixture_frontend import server_config
from ignition.preparation.runtime_package import Allocation, Argument, Kernel, Reference, Relocation, RuntimePackage, _Cuda
from signing import publisher


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    cuda = _Cuda()
    cuda.call("cuInit", C.c_uint(0))
    device, context = C.c_int(), C.c_void_p()
    cuda.call("cuDeviceGet", C.byref(device), C.c_int(0))
    cuda.call("cuCtxCreate_v2", C.byref(context), C.c_uint(0), device)
    capability = []
    for attribute in (75, 76):
        value = C.c_int()
        cuda.call("cuDeviceGetAttribute", C.byref(value), C.c_int(attribute), device)
        capability.append(value.value)
    source = root / "alternatives.cu"
    source.write_text('''
#include <cooperative_groups.h>
extern "C" __global__ void choose(int *selector, int *status, const int *source) {
    *selector = source[0]; *status = source[1];
}
extern "C" __global__ void ordinary(int *selector, int *output) {
    if (blockIdx.x == 0 && threadIdx.x == 0) { output[0] += 1; output[2] = gridDim.x; }
}
extern "C" __global__ void cooperative(int *selector, int *output) {
    cooperative_groups::this_grid().sync();
    if (blockIdx.x == 0 && threadIdx.x == 0) { output[1] += 1; output[3] = gridDim.x; }
}
''')
    cubin = root / "alternatives.cubin"
    subprocess.run(["nvcc", "--cubin", "-std=c++17", f"-arch=sm_{capability[0]}{capability[1]}",
                    str(source), "-o", str(cubin)], check=True)
    module = C.c_void_p()
    cuda.call("cuModuleLoad", C.byref(module), C.c_char_p(str(cubin).encode()))
    functions = {}
    for name in ("choose", "ordinary", "cooperative"):
        function = C.c_void_p()
        cuda.call("cuModuleGetFunction", C.byref(function), module, C.c_char_p(name.encode()))
        functions[name] = function
    pointers, allocations, buffers = {}, [], []
    for name, dtype, shape, nbytes in (
        ("frame_index", "int32", [1], 4), ("submission_sequence", "int64", [1], 8),
        ("completion_records", "int64", [2, 1, 24], 384), ("selector", "int32", [1], 4),
        ("status", "int32", [1], 4), ("choice_source", "int32", [2], 8), ("output", "int32", [4], 16),
    ):
        pointer = C.c_uint64()
        cuda.call("cuMemAlloc_v2", C.byref(pointer), C.c_size_t(nbytes))
        cuda.call("cuMemsetD8_v2", pointer, C.c_ubyte(0), C.c_size_t(nbytes))
        pointers[name] = pointer
        allocations.append(Allocation(name, pointer.value, nbytes, initialize="zero"))
        strides = [24, 24, 1] if len(shape) == 3 else [1]
        buffers.append(dict(name=name, address=pointer.value, nbytes=nbytes, dtype=dtype, shape=shape, strides=strides))
    graphs, registrations = {}, []
    for name in ("choose", "decode1", "decode2"):
        graph, executable = C.c_void_p(), C.c_void_p()
        cuda.call("cuGraphCreate", C.byref(graph), C.c_uint(0))
        previous = None
        for kernel in (("cooperative", "choose") if name == "choose" else ("ordinary", "cooperative")):
            fields = ("selector", "status", "choice_source") if kernel == "choose" else ("selector", "output")
            arguments = (C.c_void_p * len(fields))(*(C.addressof(pointers[field]) for field in fields))
            blocks = 1 if name == "choose" else (1 if kernel == "ordinary" else 2) * (1 if name == "decode1" else 2)
            node = C.c_void_p()
            params = KernelParams(func=functions[kernel].value, grid=(C.c_uint * 3)(blocks, 1, 1),
                                  block=(C.c_uint * 3)(32, 1, 1), params=arguments)
            dependencies = None if previous is None else (C.c_void_p * 1)(previous.value)
            cuda.call("cuGraphAddKernelNode_v2", C.byref(node), graph, dependencies,
                      C.c_size_t(previous is not None), C.byref(params))
            if kernel == "cooperative":
                attribute = (C.c_uint64 * 8)(1)
                cuda.call("cuGraphKernelNodeSetAttribute", node, C.c_int(2), C.byref(attribute))
            attribute = (C.c_uint64 * 8)()
            cuda.call("cuGraphKernelNodeGetAttribute", node, C.c_int(2), C.byref(attribute))
            assert attribute[0] == (kernel == "cooperative")
            if name != "choose":
                registrations.append(dict(graph=name, function=functions[kernel].value,
                    argument_index=0, argument_offset=0, selector_address=pointers["selector"].value,
                    predicate=dict(buffer="selector", allowed_values=[list(struct.pack("=i", value)) for value in (1, 2)],
                                   enabled_values=[list(struct.pack("=i", 1 if kernel == "ordinary" else 2))])))
            previous = node
        cuda.call("cuGraphInstantiateWithFlags", C.byref(executable), graph, C.c_uint64(0))
        graphs[name] = SimpleNamespace(raw_cuda_graph=lambda graph=graph: graph.value,
                                      raw_cuda_graph_exec=lambda executable=executable: executable.value)
    bindings = [dict(buffer_name="frame_index", host_offset=0, nbytes=4),
                dict(buffer_name="submission_sequence", host_offset=8, nbytes=8)]
    execution = dict(abi_version=6, max_active=1, input_frame_bytes=16, buffers=buffers,
        graphs=[dict(name=name, kind="decode", sampling_mode="general", batch_capacity=1, token_capacity=1,
            input_bindings=bindings, tuned_bindings=[dict(buffer="selector", tuning_graph="choose")])
            for name in ("decode1", "decode2")],
        tuning_graphs=[dict(name="choose", outputs=["selector"], status_buffer="status", choices=[])])
    def arguments(*names):
        return tuple(Argument(8, (Relocation(0, Reference(name)),)) for name in names)
    package = RuntimePackage(allocations=allocations, modules={"alternatives": cubin}, kernels=[
        Kernel(functions["choose"].value, "alternatives", arguments("selector", "status", "choice_source")),
        *(Kernel(functions[name].value, "alternatives", arguments("selector", "output"))
          for name in ("ordinary", "cooperative")),
    ], owners=(graphs, pointers, module, context))
    package.write(root / "package", graphs, execution=execution, server=server_config(root),
                  kernel_predicates=registrations, publisher=publisher())


if __name__ == "__main__":
    prepare(Path(sys.argv[1]).resolve())
