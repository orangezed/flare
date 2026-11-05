# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare Inductor in-place decisions with a supplied reference scheduler source.

Run with the candidate Torch interpreter:
python qualify/torch_compiler/scheduler_membership.py --baseline-source /path/to/torch/_inductor/scheduler.py
Use --output PATH to save the JSON result as well as printing it. The fixtures
exercise real scheduler decisions and kernel arguments without CUDA or weights.
"""

import argparse
import ast
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import sympy
import torch
from torch._inductor import config, dependencies, ir, scheduler
from torch._inductor.codegen.common import KernelArgs
from torch._inductor.codegen.simd import SIMDKernel
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.graph import GraphLowering
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


def reference_method(source):
    tree = ast.parse(source.read_text())
    base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'BaseSchedulerNode')
    method = next(node for node in base.body if isinstance(node, ast.FunctionDef) and node.name == 'decide_inplace_update')
    namespace = dict(vars(scheduler))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['decide_inplace_update']


CANDIDATE = scheduler.BaseSchedulerNode.decide_inplace_update

class AllocatedBuffer(ir.Buffer):
    def should_allocate(self):
        return getattr(self, 'allocate', True)

    def get_inputs_that_alias_output(self):
        return getattr(self, 'aliases', ())

    def get_mutation_names(self):
        return getattr(self, 'mutation_names', ())


class CountedSet(OrderedSet):
    """Count union work while preserving the installed OrderedSet operators."""

    counts = Counter()

    def __or__(self, other):
        type(self).counts['union_calls'] += 1
        type(self).counts['union_input_items'] += len(self) + len(other)
        return type(self)(super().__or__(other))

    def __contains__(self, key):
        type(self).counts['membership_checks'] += 1
        return super().__contains__(key)


def fixture(case='reuse', set_size=0, counted=False):
    owner = scheduler.Scheduler.__new__(scheduler.Scheduler)
    node = scheduler.SchedulerNode.__new__(scheduler.SchedulerNode)
    node.scheduler = owner
    node.node = SimpleNamespace(get_operation_name=lambda: 'operation', get_device=lambda: torch.device('cpu'))
    sets = CountedSet if counted else OrderedSet
    node.ancestors = sets(['ancestor', 'shared'] + [f'ancestor_{i}' for i in range(set_size)])
    owner.completed_operations = sets(['completed', 'shared'] + [f'completed_{i}' for i in range(set_size)])
    removed = sets(['removed', 'shared'] + [f'removed_{i}' for i in range(set_size)])
    index = sympy.Symbol('i', integer=True)
    read = dependencies.MemoryDep('input', index, (index,), (sympy.Integer(16),))
    node.read_writes = dependencies.ReadWrites(OrderedSet([read]), OrderedSet(), OrderedSet())
    layout = ir.FixedLayout(torch.device('cpu'), torch.float32, [sympy.Integer(16)], [sympy.Integer(1)])
    input_ir = AllocatedBuffer(name='input', layout=layout)
    output_ir = AllocatedBuffer(name='output', layout=layout)
    input_buf = scheduler.SchedulerBuffer(owner, input_ir, None)
    output_buf = scheduler.SchedulerBuffer(owner, output_ir, node)
    node.outputs = [output_buf]
    node.outputs_by_name = {'output': output_buf}
    input_buf.users = [
        scheduler.NodeUser(SimpleNamespace(get_name=lambda name=name: name))
        for name in ('ancestor', 'removed', 'completed', 'shared')
    ] + [scheduler.NodeUser(node, can_inplace=True)]
    owner.name_to_buf = {'input': input_buf}
    owner.name_to_donated_buffer = {}
    owner.name_to_fused_node = {'operation': node}
    owner.node_to_stream = {node: 0}
    owner.buff_to_stream = {'input': 0}
    owner._multi_stream_nodes = False
    owner.node_to_mempool = {node: None}
    owner.buff_to_mempool = {'input': None}
    owner.mutation_renames = {}
    wrapper = PythonWrapperCodegen.__new__(PythonWrapperCodegen)
    wrapper.freed = OrderedSet()
    graph = GraphLowering.__new__(GraphLowering)
    graph.removed_operations = removed
    graph.removed_buffers = OrderedSet()
    graph.graph_inputs = {}
    graph.graph_inputs_original = {}
    graph.constants = {}
    graph.torchbind_constants = {}
    graph.never_reuse_buffers = OrderedSet()
    graph.unaligned_buffers = OrderedSet(['input'])
    graph.buffer_to_padded_size = {}
    graph.sizevars = SizeVarAllocator()
    graph.wrapper_code = wrapper
    graph.has_feature = lambda device, feature: case != 'backend_without_inplace'
    graph.scheduler = owner
    kernel = SimpleNamespace(args=KernelArgs(), inplace_update_buffers={})

    if case == 'extra_live_user':
        input_buf.users.append(scheduler.NodeUser(SimpleNamespace(get_name=lambda: 'live')))
    elif case == 'current_node_excluded':
        node.ancestors.add('operation')
    elif case == 'current_node_removed':
        removed.add('operation')
    elif case == 'current_node_completed':
        owner.completed_operations.add('operation')
    elif case == 'cannot_inplace':
        input_buf.users[-1].can_inplace = False
    elif case == 'wrong_last_user':
        input_buf.users[-1].node = SimpleNamespace(get_name=lambda: 'other')
    elif case == 'cross_stream':
        owner._multi_stream_nodes = True
        owner.buff_to_stream['input'] = 1
    elif case == 'cross_mempool':
        owner.buff_to_mempool['input'] = (7, 9)
    elif case == 'input_removed':
        graph.removed_buffers.add('input')
    elif case == 'output_removed':
        graph.removed_buffers.add('output')
    elif case == 'graph_input':
        graph.graph_inputs['input'] = input_ir
        graph.graph_inputs_original['input'] = input_ir
    elif case == 'constant':
        graph.constants['input'] = object()
    elif case == 'never_reuse':
        graph.never_reuse_buffers.add('input')
    elif case == 'freed':
        wrapper.freed.add('input')
    elif case == 'output_unallocated':
        output_ir.allocate = False
    elif case == 'output_alias':
        output_ir.aliases = ('alias',)
    elif case == 'output_mutation':
        output_ir.mutation_names = ('mutation',)
    elif case in ('input_p2p', 'output_p2p'):
        target = input_ir if case == 'input_p2p' else output_ir
        target.layout = ir.CommBufferLayout(layout, ir.CommBufferType.SYMM_MEM, 'group')
    elif case == 'input_none_layout':
        input_ir.layout = ir.NoneLayout(device=torch.device('cpu'))
    elif case == 'input_multioutput_layout':
        input_ir.layout = ir.MultiOutputLayout.__new__(ir.MultiOutputLayout)
    elif case == 'input_mutation_layout':
        input_ir.layout = ir.MutationLayoutSHOULDREMOVE.__new__(ir.MutationLayoutSHOULDREMOVE)
    elif case == 'fallback_alias':
        input_ir.aliases = ('alias',)
        producer = scheduler.ExternKernelSchedulerNode.__new__(scheduler.ExternKernelSchedulerNode)
        producer.node = ir.FallbackKernel.__new__(ir.FallbackKernel)
        input_buf.defining_op = producer
    elif case == 'multioutput_alias':
        input_ir.aliases = ('alias',)
        producer = scheduler.ExternKernelSchedulerNode.__new__(scheduler.ExternKernelSchedulerNode)
        producer.node = ir.MultiOutput.__new__(ir.MultiOutput)
        input_buf.defining_op = producer
    elif case == 'nop_producer':
        input_buf.defining_op = scheduler.NopKernelSchedulerNode.__new__(scheduler.NopKernelSchedulerNode)
    elif case == 'size_mismatch':
        output_ir.layout = ir.FixedLayout(torch.device('cpu'), torch.float32, [sympy.Integer(32)], [sympy.Integer(1)])
    elif case == 'dtype_mismatch':
        output_ir.layout = ir.FixedLayout(torch.device('cpu'), torch.int64, [sympy.Integer(16)], [sympy.Integer(1)])
    elif case == 'device_mismatch':
        output_ir.layout = ir.FixedLayout(torch.device('meta'), torch.float32, [sympy.Integer(16)], [sympy.Integer(1)])
    elif case == 'fused_index_mismatch':
        other = scheduler.SchedulerNode.__new__(scheduler.SchedulerNode)
        other.node = SimpleNamespace(get_operation_name=lambda: 'fused_other')
        other.read_writes = dependencies.ReadWrites(
            OrderedSet([dependencies.MemoryDep('input', index + 1, (index,), (sympy.Integer(16),))]),
            OrderedSet(), OrderedSet())
        owner.name_to_fused_node['fused_other'] = node
        node.ancestors.add('fused_other')
        input_buf.users.append(scheduler.NodeUser(other))
    elif case == 'donated':
        owner.name_to_donated_buffer['input'] = input_buf
        owner.name_to_buf.clear()
    elif case in ('simd', 'simd_without_mutations'):
        kernel = SIMDKernel.__new__(SIMDKernel)
        kernel.args = KernelArgs()
        kernel.inplace_update_buffers = {}
        kernel.mutations = OrderedSet() if case == 'simd' else None
    elif case == 'no_kernel_args':
        kernel = SimpleNamespace()
    elif case == 'first_read_rejected':
        bad = dependencies.MemoryDep('constant_input', index, (index,), (sympy.Integer(16),))
        node.read_writes.reads = OrderedSet([bad, read])
        owner.name_to_buf['constant_input'] = scheduler.SchedulerBuffer(owner, AllocatedBuffer(name='constant_input', layout=layout), None)
        graph.constants['constant_input'] = object()
    elif case == 'two_outputs':
        second = scheduler.SchedulerBuffer(owner, AllocatedBuffer(name='output2', layout=layout), node)
        node.outputs.append(second)
        node.outputs_by_name['output2'] = second
    elif case == 'no_outputs':
        node.outputs.clear()
    elif case == 'no_reads':
        node.read_writes.reads.clear()
    return node, graph, kernel


def run(method, case='reuse', set_size=0, counted=False):
    node, graph, kernel = fixture(case, set_size, counted)
    sets = (node.ancestors, graph.removed_operations, node.scheduler.completed_operations)
    before = tuple(tuple(s) for s in sets)
    with ExitStack() as stack:
        stack.enter_context(V.set_graph_handler(graph))
        stack.enter_context(V.set_kernel_handler(kernel))
        stack.enter_context(config.patch(inplace_buffers=case != 'inplace_disabled'))
        method(node)
    assert before == tuple(tuple(s) for s in sets), case
    args = getattr(kernel, 'args', None)
    return {
        'decisions': getattr(kernel, 'inplace_update_buffers', {}),
        'inplace_args': {} if args is None else {name: (buf.inner_name, tuple(buf.other_names)) for name, buf in args.inplace_buffers.items()},
        'mutations': tuple(getattr(kernel, 'mutations', ()) or ()),
        'unaligned': tuple(graph.unaligned_buffers),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source', required=True, type=Path,
                        help='Torch scheduler.py containing the reference decide_inplace_update')
    parser.add_argument('--output', type=Path, help='Optional JSON result file')
    args = parser.parse_args()
    ORIGINAL = reference_method(args.baseline_source)
    accepted = ['reuse', 'donated', 'simd', 'first_read_rejected', 'two_outputs']
    rejected = [
        'backend_without_inplace', 'inplace_disabled', 'no_kernel_args',
        'simd_without_mutations', 'extra_live_user', 'current_node_excluded',
        'current_node_removed', 'current_node_completed',
        'cannot_inplace', 'wrong_last_user', 'cross_stream', 'cross_mempool',
        'input_removed', 'output_removed', 'graph_input', 'constant', 'never_reuse',
        'freed', 'output_unallocated', 'output_alias', 'output_mutation',
        'input_p2p', 'output_p2p', 'input_none_layout', 'input_multioutput_layout',
        'input_mutation_layout', 'fallback_alias', 'multioutput_alias', 'nop_producer',
        'size_mismatch', 'dtype_mismatch', 'device_mismatch', 'fused_index_mismatch',
        'no_outputs', 'no_reads',
    ]
    for case in accepted + rejected:
        expected, actual = run(ORIGINAL, case), run(CANDIDATE, case)
        assert actual == expected, (case, expected, actual)
        assert bool(actual['decisions']) == (case in accepted), (case, actual)
        if case in accepted:
            outputs = ['output', 'output2'] if case == 'two_outputs' else ['output']
            assert actual['decisions'] == dict.fromkeys(outputs, 'input')
            assert actual['inplace_args']['input'] == ('in_out_ptr0', tuple(['input'] + outputs))
            assert actual['mutations'] == (('input', 'output') if case == 'simd' else ())
    operation_counts = {}
    for name, method in [('original', ORIGINAL), ('candidate', CANDIDATE)]:
        CountedSet.counts.clear()
        run(method, set_size=50000, counted=True)
        operation_counts[name] = dict(CountedSet.counts)
    # Time only the actual method on reusable fixtures, excluding setup, imports,
    # dependency construction, and global-handler installation.
    timings = {}
    for size in (0, 10000, 50000):
        timings[str(size)] = {}
        for name, method in [('original', ORIGINAL), ('candidate', CANDIDATE)]:
            node, graph, kernel = fixture(set_size=size)
            samples = []
            with V.set_graph_handler(graph), V.set_kernel_handler(kernel), config.patch(inplace_buffers=True):
                for repeat in range(7):
                    start = time.perf_counter()
                    for _ in range(50):
                        kernel.args = KernelArgs()
                        kernel.inplace_update_buffers.clear()
                        method(node)
                    samples.append((time.perf_counter() - start) / 50)
            timings[str(size)][name] = {'median_seconds': statistics.median(samples), 'samples_seconds': samples}
    result = {
        'torch': torch.__version__, 'cases': len(accepted + rejected),
        'accepted_cases': accepted, 'rejected_cases': rejected,
        'all_decisions_and_kernel_args_equal': True,
        'operation_counts_50000_names_per_set': operation_counts,
        'timing': timings,
        'cuda_initialized': torch.cuda.is_initialized(),
    }
    assert not result['cuda_initialized']
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
