# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preserve the host enqueue order of prepared native operations through Inductor."""

import torch
from torch._inductor.custom_graph_pass import CustomPassBase, CustomSchedulerPass, get_hash_for_files
from torch._inductor.dependencies import WeakDep


def compilation_options_identity(options):
    """Include compiler-pass implementation identities in the outer forward cache."""
    def identity(value):
        if isinstance(value, CustomPassBase):
            version = value.uuid()
            if version is None:
                raise ValueError("cached prepared forwards require compiler passes with stable identities")
            return type(value).__module__, type(value).__qualname__, version
        if isinstance(value, (list, tuple)):
            return type(value)(identity(item) for item in value)
        return value
    return {key: identity(value) for key, value in options.items()}


class NativeEnqueueOrder(CustomSchedulerPass):
    """Keep stream forks before independent native work, and joins after it.

    Storage-call handles are assigned in source order before functionalization.
    Tensor dependencies alone allow a scheduler to sink an asynchronous fork
    until its join, eliminating the intended overlap. Ordering-only dependencies
    constrain host enqueue order without adding GPU work or synchronization.
    Buffer lifetimes remain the responsibility of each operation's tensor operands.
    """

    def uuid(self):
        return get_hash_for_files((__file__,))

    def __call__(self, nodes):
        from torch._inductor.scheduler import NodeUser

        calls = []
        for node in nodes:
            operation = node.node
            if getattr(operation, "op_overload", None) is not torch.ops.flare.storage_call.default:
                continue
            arguments, keywords = operation.unflatten_args(operation.inputs, operation.constant_args)
            handle = arguments[2] if len(arguments) > 2 else keywords["handle"]
            if type(handle) is not int:
                raise ValueError("prepared storage-call ordering requires a constant invocation index")
            calls.append((handle, node))
        if len(calls) < 2:
            return nodes
        if len({handle for handle, _ in calls}) != len(calls):
            raise ValueError("prepared storage-call invocation indices must be unique")
        calls.sort(key=lambda item: item[0])
        for (_, previous), (_, node) in zip(calls, calls[1:]):
            outputs = previous.get_outputs()
            if not outputs:
                raise ValueError("prepared native operations require mutation outputs for enqueue ordering")
            dependency = outputs[0]
            node.add_fake_dep(WeakDep(dependency.get_name(), node.get_name(), is_fake=True))
            dependency.set_users([*dependency.users, NodeUser(node, can_inplace=False, is_weak=True)])
        scheduler = calls[0][1].scheduler
        scheduler.nodes = scheduler.topological_sort_schedule(nodes)
        scheduler.compute_ancestors()
        scheduler.compute_input_distances()
        return scheduler.nodes
