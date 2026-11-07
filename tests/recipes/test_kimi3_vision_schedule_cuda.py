"""Head partitions inherit the normally tuned full-head attention arithmetic."""

from functools import partial
import gc
import os
from types import SimpleNamespace
import weakref

import pytest
import torch

from ignition.components.varlen_attention import PreparedVarlenAttention
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.kimi3.vision import _declare_reference_attention, resolve_reference_attention


pytestmark = pytest.mark.skipif(os.getenv('FLARE_RUN_CUDA_TESTS') != '1',
                               reason='requires an explicitly assigned CUDA GPU')


@torch.no_grad()
def test_reference_attention_releases_tuning_operands(monkeypatch, tmp_path):
    from ignition.preparation.fixtures import varlen_attention as fixtures

    allocate = fixtures.attention_benchmark_inputs
    operands = []

    def observe(arguments):
        inputs = allocate(arguments)
        operands.extend(weakref.ref(getattr(inputs, name)) for name in ('q', 'k', 'v', 'output'))
        return inputs

    monkeypatch.setattr(fixtures, 'attention_benchmark_inputs', observe)
    device = torch.device('cuda:0')
    with RecordedPreparationSession(device=device, checkpoint='moonvit-head-partition',
            workload_schema='reference-operand-lifetime', numerical_policy='bf16',
            namespace={'fixture': str(tmp_path)}, compile_workers=2) as session:
        workspace = ScratchArena(device)
        session.register_scratch(workspace)
        _declare_reference_attention(256, device, session=session, workspace=workspace, name='canonical')
        session.tune(show_progress=False)
        torch.cuda.synchronize()
        gc.collect()
        assert operands, 'the isolated tuning namespace must exercise real benchmark operands'
        assert all(reference() is None for reference in operands)
        assert session.tuning_report['benchmarked_candidates'] > 0


@pytest.mark.parametrize('rows,heads', [(64, 1), (256, 2)])
@torch.no_grad()
def test_partitioned_attention_matches_full_head_schedule(rows, heads):
    device = torch.device('cuda:0')
    full_q, full_k, full_v = [torch.randn((rows, 12, 128), dtype=torch.bfloat16, device=device)
                             for _ in range(3)]
    local_q, local_k, local_v = [value[:, :heads].contiguous() for value in (full_q, full_k, full_v)]
    full_out, local_out = torch.empty_like(full_q), torch.empty_like(local_q)
    cu = torch.tensor([0, rows], dtype=torch.int32, device=device)
    with RecordedPreparationSession(device=device, checkpoint='moonvit-head-partition',
            workload_schema='unpartitioned-attention-schedule', numerical_policy='bf16',
            compile_workers=2) as session:
        workspace = ScratchArena(device)
        session.register_scratch(workspace)
        reference = PreparedVarlenAttention(full_q, full_k, full_v, cu, full_out,
            session=session, workspace=workspace, name='reference')
        partition = PreparedVarlenAttention(local_q, local_k, local_v, cu, local_out,
            session=session, workspace=workspace, name='partition')
        before_declaration = torch.cuda.memory_allocated()
        _declare_reference_attention(rows, device, session=session, workspace=workspace, name='canonical')
        assert torch.cuda.memory_allocated() == before_declaration
        encoder = SimpleNamespace(reference_attention_name='canonical', attention_schedule_ready=False,
            blocks=[SimpleNamespace(attn=SimpleNamespace(attend=partition, attention_name='partition'))])
        session.tune(show_progress=False, contextual=partial(resolve_reference_attention, encoders=[encoder]),
                     finalist_policies={})
        assert encoder.attention_schedule_ready and encoder.reference_attention_name is None
        assert partition.plan.selection.config == reference.plan.selection.config
        assert partition.plan.selection.source == 'override'
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            reference()
            partition()
        session.freeze()
        for _ in range(4):
            for full, local in zip((full_q, full_k, full_v), (local_q, local_k, local_v)):
                full.normal_()
                local.copy_(full[:, :heads])
            workspace.tensor.fill_(255)
            full_out.fill_(torch.nan)
            local_out.fill_(torch.nan)
            allocated = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
            torch.testing.assert_close(local_out, full_out[:, :heads], rtol=0, atol=0)
        graph.reset()


@pytest.mark.parametrize('rows,columns', [(64, 1024), (512, 4096)])
@torch.no_grad()
def test_packed_mxfp8_a16_projection_keeps_exact_dyadic_rows(rows, columns):
    from ignition.components.linear import QuantizedProjection
    from ignition.components.weights import quantize_bf16_linear_as_mxfp8

    device = torch.device('cuda:0')
    source = (torch.arange(rows * columns, device=device).view(rows, columns).remainder(13) - 6).bfloat16() / 8
    weight = (torch.arange(1024 * columns, device=device).view(1024, columns).remainder(17) - 8).bfloat16() / 8
    full, part = quantize_bf16_linear_as_mxfp8(weight), quantize_bf16_linear_as_mxfp8(weight[:128])
    output = torch.empty((rows, 1024), dtype=torch.bfloat16, device=device)
    selected = torch.empty((rows, 128), dtype=torch.bfloat16, device=device)
    with RecordedPreparationSession(device=device, checkpoint='moonvit-packed-projection',
            workload_schema='output-row-partitions', numerical_policy='a16', compile_workers=2) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        reference = QuantizedProjection(source, full, output, session=session, workspace=arena,
                                        name='reference', activation_mode='a16')
        candidate = QuantizedProjection(source, part, selected, session=session, workspace=arena,
                                        name='partition', activation_mode='a16', retain_packed_weights=True)
        session.tune(show_progress=False)
        assert candidate.plan.selection.config.a16_backend == 'packed'
        assert candidate.plan.selection.config.mode == 'a16'
        # Ordinary Ignition warms complete forwards before entering the frozen
        # capture phase; fixed packed plans load their retained objects here.
        reference(source)
        candidate(source)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            reference(source)
            candidate(source)
        session.freeze()
        for scale in (1, -1, .5):
            source.mul_(scale)
            arena.tensor.fill_(255)
            output.fill_(torch.nan)
            selected.fill_(torch.nan)
            before = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == before
            torch.testing.assert_close(selected, output[:, :128], atol=0, rtol=0)
        graph.reset()


@pytest.mark.parametrize('rows,columns,global_rows,first,local_rows', [
    (128, 1536, 1024, 128, 128),
    (256, 4096, 1024, 384, 128),
    (256, 4096, 7168, 2688, 896),
])
@torch.no_grad()
def test_output_partition_preserves_unpartitioned_gemm_geometry(rows, columns, global_rows, first, local_rows):
    from ignition.components.linear import QuantizedProjection
    from ignition.components.weights import quantize_bf16_linear_as_mxfp8

    device = torch.device('cuda:0')
    generator = torch.Generator(device=device).manual_seed(7148)
    source = torch.randn((rows, columns), generator=generator, dtype=torch.bfloat16, device=device)
    weight = torch.randn((global_rows, columns), generator=generator, dtype=torch.bfloat16, device=device)
    full = quantize_bf16_linear_as_mxfp8(weight)
    part = quantize_bf16_linear_as_mxfp8(weight[first:first + local_rows])
    output = torch.empty((rows, global_rows), dtype=torch.bfloat16, device=device)
    selected = torch.empty((rows, local_rows), dtype=torch.bfloat16, device=device)
    with RecordedPreparationSession(device=device, checkpoint='moonvit-logical-output-extent',
            workload_schema='output-row-partitions', numerical_policy='a16', compile_workers=2) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        reference = QuantizedProjection(source, full, output, session=session, workspace=arena,
                                        name='reference', activation_mode='a16')
        candidate = QuantizedProjection(source, part, selected, session=session, workspace=arena,
            name='partition', activation_mode='a16',
            output_partition=(global_rows, first, first + local_rows))
        session.tune(show_progress=False)
        reference(source)
        candidate(source)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            reference(source)
            candidate(source)
        session.freeze()
        for _ in range(4):
            source.normal_(generator=generator)
            arena.tensor.fill_(255)
            output.fill_(torch.nan)
            selected.fill_(torch.nan)
            before = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == before
            torch.testing.assert_close(selected, output[:, first:first + local_rows], atol=0, rtol=0)
        graph.reset()
