# SPDX-License-Identifier: Apache-2.0

"""BF16 output shards with the unpartitioned projection's selected arithmetic."""

from dataclasses import replace
from types import SimpleNamespace

import torch
from b12x.gemm import bf16_gemv
from b12x.gemm.bf16_gemv import partition
from b12x.preparation import PreparedCall

from ignition.preparation.fixtures.projection import input_factory
from ignition.preparation.scratch import ScratchArena, prime_owners, scratch_storage
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer


class PreparedOutputPartitionProjection:
    def __init__(self, source, weight, output, *, session, name, workspace):
        if (
            weight.reference_cpu is None
            or weight.reference_cpu.device.type != "cpu"
            or weight.reference_cpu.shape
            != (weight.logical_out_features, weight.in_features)
            or weight.reference_cpu.dtype != torch.bfloat16
            or workspace is None
        ):
            raise ValueError(
                "BF16 output shards require a full CPU tuning reference and caller-owned workspace"
            )
        self.weight, self.output, self.workspace = weight.values, output, workspace
        record_numeric_tensor(self.weight)
        query = replace(
            bf16_gemv.query_from_call(source, self.weight, out=output),
            out_features=weight.logical_out_features,
        )
        self.reference = bf16_gemv.plan(query)
        reference_name = name + ".logical_geometry"
        self.plan = partition.plan(
            self.reference, weight.out_features, dependency=reference_name
        )
        full_output = SimpleNamespace(
            shape=(source.shape[0], weight.logical_out_features),
            stride=lambda: (weight.logical_out_features, 1),
            dtype=output.dtype,
            device=output.device,
        )
        inputs = input_factory(
            source,
            full_output,
            storage=getattr(session, "benchmark_storage", None),
            session=session,
        )
        reference_cpu = weight.reference_cpu

        def benchmark(state):
            operands = inputs()
            full_weight = benchmark_resource(
                session,
                ("bf16_reference_weight", id(reference_cpu)),
                lambda: reference_cpu.to(source.device),
            )
            return PreparedCall(
                run=lambda: state.run(
                    operands.source, full_weight, out=operands.output
                ),
                produce=operands.produce,
                output=operands.output,
                owners=(operands, full_weight),
            )

        # The parent only selects arithmetic. Its benchmark executes the full
        # matrix; no full-width weight is retained for graph execution.
        parent = self.reference.request(
            name=reference_name,
            prepare_call=lambda state: PreparedCall(run=lambda: None),
            benchmark_call=benchmark,
        )

        def prepare(state):
            storage = (
                workspace.tensor if isinstance(workspace, ScratchArena) else workspace
            )
            return PreparedCall(
                run=lambda: state.run(
                    source, self.weight, out=output, workspace=storage
                ),
                owners=prime_owners(workspace, storage),
            )

        child = self.plan.request(name=name, prepare_call=prepare)
        declare(session, (parent, child))
        register_finalizer(session, self.refresh, workspace=workspace)

    def refresh(self):
        self.storage = scratch_storage(
            self.workspace,
            nbytes=self.plan.memory_requirements().scratch_nbytes,
            device=self.output.device,
        )

    def __call__(self, source):
        return partition.mm(
            source, self.weight, out=self.output, workspace=self.storage, plan=self.plan
        )
