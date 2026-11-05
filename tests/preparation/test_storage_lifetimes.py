# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Storage snapshots preserve reads across aliased writes and last-use boundaries."""

import operator
import random

import pytest
import torch

from ignition.preparation import compilation


def reference_reads_across_mutations(nodes, keys, mutations):
    """Compute snapshot requirements by scanning live values."""
    positions = {node: index for index, node in enumerate(nodes)}
    last_use = {node: max((positions[user] for user in node.users), default=-1) for node in nodes}
    for node in reversed(nodes):
        if (isinstance(node.target, torch._ops.OpOverload)
                and node.target.namespace == 'aten' and not node.target._schema.is_mutable):
            for operand in node.all_input_nodes:
                last_use[operand] = max(last_use[operand], last_use[node])
    mutations = {node: {keys[operand] for operand in operands}
                 for node, operands in mutations.items() if operands}
    mutable = set().union(*mutations.values()) if mutations else set()
    live, fresh, required = {}, set(), {}
    for index, node in enumerate(nodes):
        operands = node.all_input_nodes
        reads = set().union(*(live.get(arg, set()) for arg in operands)) if operands else set()
        reads.update(keys[arg] for arg in operands if keys.get(arg) in mutable)
        if node in mutations:
            for previous, pending in live.items():
                if previous in fresh and last_use[previous] > index and pending & mutations[node]:
                    required.setdefault(previous, []).append(node)
                    pending.clear()
        live = {value: pending for value, pending in live.items() if pending and last_use[value] > index}
        if (node in keys and node.op == 'call_function'
                and isinstance(node.target, torch._ops.OpOverload) and node.target.namespace == 'aten'):
            if keys[node] not in {keys[arg] for arg in operands if arg in keys}:
                fresh.add(node)
            live[node] = reads
        elif node in keys and keys[node] in mutable:
            live[node] = {keys[node]}
    return {node: required[node] for node in nodes if node in required}


def assert_matches_reference(graph, keys, mutations):
    graph.lint()
    nodes = list(graph.nodes)
    expected = reference_reads_across_mutations(nodes, keys, mutations)
    actual = compilation._reads_across_mutations(nodes, keys, mutations)
    assert list(actual.items()) == list(expected.items())
    return actual


def random_storage_graph(seed, size=160):
    rng = random.Random(seed)
    graph = torch.fx.Graph()
    values = [graph.placeholder(f'input_{index}') for index in range(8)]
    keys = {value: index // 2 for index, value in enumerate(values)}
    mutations = {}
    next_key = 4
    for _ in range(size):
        left, right, destination = rng.choices(values, k=3)
        kind = rng.randrange(8)
        if kind == 0:
            node = graph.call_function(torch.ops.aten.sin.default, (left,))
        elif kind == 1:
            node = graph.call_function(torch.ops.aten.add.Tensor, (left, right))
        elif kind == 2:
            node = graph.call_function(torch.ops.aten.view.default, (left, [8]))
            keys[node] = keys[left]
        elif kind == 3:
            node = graph.call_function(torch.ops.aten.copy_.default, (destination, left))
            keys[node] = keys[destination]
            mutations[node] = [destination]
        elif kind == 4:
            node = graph.call_function(torch.ops.aten.add.out, (left, right), {'out': destination})
            keys[node] = keys[destination]
            mutations[node] = [destination]
        elif kind == 5:
            node = graph.call_function(torch.ops.aten._foreach_add_.List, ([left, right], [right, left]))
            mutations[node] = [left, right]
            node = graph.call_function(operator.getitem, (node, 0))
            keys[node] = keys[left]
        elif kind == 6:
            node = graph.call_function(torch.ops.aten.mul.Tensor, (left, right))
        else:
            node = graph.call_function(torch.ops.aten.neg.default, (left,))
        if node not in keys:
            keys[node] = next_key
            next_key += 1
        values.append(node)
    graph.output(tuple(rng.sample(values, 16)))
    return graph, keys, mutations


@pytest.mark.parametrize('seed', range(200))
def test_random_aliases_mutations_and_out_arguments(seed):
    graph, keys, mutations = random_storage_graph(seed)
    assert_matches_reference(graph, keys, mutations)


def test_reads_precede_mutation_and_last_use_expiry():
    graph = torch.fx.Graph()
    source = graph.placeholder('source')
    saved = graph.call_function(torch.ops.aten.sin.default, (source,))
    write = graph.call_function(torch.ops.aten.copy_.default, (source, saved))
    graph.output(source)
    keys = {source: 0, saved: 1, write: 0}
    assert assert_matches_reference(graph, keys, {write: [source]}) == {}


def test_snapshot_clears_all_pending_storage_dependencies():
    graph = torch.fx.Graph()
    left, right = graph.placeholder('left'), graph.placeholder('right')
    saved = graph.call_function(torch.ops.aten.add.Tensor, (left, right))
    first = graph.call_function(torch.ops.aten.zero_.default, (left,))
    second = graph.call_function(torch.ops.aten.zero_.default, (right,))
    consumed = graph.call_function(torch.ops.aten.sin.default, (saved,))
    graph.output(consumed)
    keys = {left: 0, right: 1, saved: 2, first: 0, second: 1, consumed: 3}
    actual = assert_matches_reference(graph, keys, {first: [left], second: [right]})
    assert actual == {saved: [first]}


def test_mutation_reads_propagate_before_snapshot_clears():
    graph = torch.fx.Graph()
    source = graph.placeholder('source')
    saved = graph.call_function(torch.ops.aten.sin.default, (source,))
    first = graph.call_function(torch.ops.aten.add_.Tensor, (source, saved))
    derived = graph.call_function(torch.ops.aten.cos.default, (first,))
    second = graph.call_function(torch.ops.aten.zero_.default, (source,))
    graph.output((saved, derived))
    keys = {source: 0, saved: 1, first: 0, derived: 2, second: 0}
    actual = assert_matches_reference(graph, keys, {first: [source], second: [source]})
    assert actual == {saved: [first], derived: [second]}


def test_immutable_dependency_chain_extends_last_use():
    graph = torch.fx.Graph()
    source = graph.placeholder('source')
    first = graph.call_function(torch.ops.aten.sin.default, (source,))
    second = graph.call_function(torch.ops.aten.cos.default, (first,))
    write = graph.call_function(torch.ops.aten.zero_.default, (source,))
    graph.output(second)
    keys = {source: 0, first: 1, second: 2, write: 0}
    actual = assert_matches_reference(graph, keys, {write: [source]})
    assert list(actual) == [first, second]
    assert all(consumers == [write] for consumers in actual.values())


def test_empty_pending_dead_nodes_and_no_mutations():
    graph = torch.fx.Graph()
    source = graph.placeholder('source')
    dead = graph.call_function(torch.ops.aten.sin.default, (source,))
    empty = graph.call_function(torch.ops.aten.ones.default, ([8],))
    graph.output(source)
    assert assert_matches_reference(graph, {source: 0, dead: 1, empty: 2}, {}) == {}
