# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qualify event-triggered root retracing against the installed legacy checker.

Run with CUDA_VISIBLE_DEVICES= and the respective Torch interpreters:
CUDA_VISIBLE_DEVICES= python qualify/torch_compiler/events_cpu.py baseline --output baseline.json
CUDA_VISIBLE_DEVICES= python qualify/torch_compiler/events_cpu.py candidate --baseline-result baseline.json
Use --output PATH to save JSON as well as printing it. CPU traces use
an explicit synthetic stream identity, with actual Dynamo mutation hooks and
actual event-consumer validation. No new tensor operation is registered.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch._dynamo import exc
from torch._dynamo.output_graph import OutputGraph, SubgraphTracer
from torch._dynamo.source import LocalSource
from torch._dynamo.symbolic_convert import InstructionTranslator, SpeculationLog
from torch._dynamo.variables.streams import EventVariable, StreamVariable
from torch._guards import TracingContext, tracing
from torch._subclasses.fake_tensor import FakeTensorMode

def fixture():
    output = OutputGraph.__new__(OutputGraph)
    output.timestamp = 0
    output.bytecode_tracing_timings = SimpleNamespace(create_proxy_ns=0)
    output._current_tx = []
    output.root_tx = SimpleNamespace(f_code=fixture.__code__, nn_module_stack={}, is_co_filename_from_nn_modules=lambda: True, symbolic_stream_state=SimpleNamespace(in_stream_context=lambda: True, cur_stream_id=lambda: 7))
    output._last_checked_input_versions = None
    output._input_mutation_streams = {}
    tracer = SubgraphTracer(output)
    output.tracers = [tracer]
    return (output, tracer)

def add(tracer, name, value, before=False):
    return tracer.create_graph_input(name, type(value), value, before=before, source=LocalSource(name))


parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=('baseline', 'candidate'))
parser.add_argument('--output', type=Path, help='Optional JSON result file')
parser.add_argument('--baseline-result', type=Path, help='Baseline JSON required for candidate comparison')
args = parser.parse_args()
IS_CANDIDATE = args.mode == 'candidate'
if IS_CANDIDATE and args.baseline_result is None:
    parser.error('candidate mode requires --baseline-result PATH')
assert hasattr(SpeculationLog(), 'requires_input_mutation_stream_checks') == IS_CANDIDATE

EVENT_TRACES = []
SCAN_TRACES = []
ATTEMPTS = []
CHECK = OutputGraph.check_input_mutation_on_current_stream
CONSUME = OutputGraph.check_event_record_after_input_mutation
INIT = InstructionTranslator.__init__


@torch._dynamo.assume_constant_result
def event_consumer():
    tx = InstructionTranslator.current_tx()
    EVENT_TRACES.append((id(tx.speculation_log), len(tx.output.tracers)))
    tx.output.check_event_record_after_input_mutation(7)
    return None


def check_on_synthetic_stream(self, tx):
    synthetic = SimpleNamespace(speculation_log=tx.speculation_log,
        symbolic_stream_state=SimpleNamespace(in_stream_context=lambda: True, cur_stream_id=lambda: 7))
    result = CHECK(self, synthetic)
    SCAN_TRACES.append({'attempt': len(ATTEMPTS),
        'initialized': self._last_checked_input_versions is not None,
        'input_versions_at_beginning': list(self.root_tracer._input_versions_at_beginning),
        'versions': None if self._last_checked_input_versions is None else {
            str(index): value for index, value in self._last_checked_input_versions.items()}})
    return result


def initialize_output(self, *positional, **keywords):
    INIT(self, *positional, **keywords)
    ATTEMPTS.append({'log_id': id(self.speculation_log),
        'required': getattr(self.speculation_log, 'requires_input_mutation_stream_checks', True)})


def no_event(source, other, effects):
    effects.append('once')
    source.add_(1)
    return source * 2 + other


def event_before_mutation(source, other, effects):
    effects.append('once')
    event_consumer()
    source.add_(1)
    return source * 2 + other


def mutation_before_event(source, other, effects):
    effects.append('once')
    source.add_(1)
    event_consumer()
    return source * 2 + other


def inline_event(source):
    event_consumer()
    return source.sin()


def event_in_inlined_function(source, other, effects):
    effects.append('once')
    source.add_(1)
    return inline_event(source) + other


@torch.compiler.nested_compile_region
def nested_event(source):
    event_consumer()
    return source.sin()


def event_in_nested_region(source, other, effects):
    effects.append('once')
    source.add_(1)
    return nested_event(source) + other


def nested_event_before_mutation(source, other, effects):
    effects.append('once')
    result = nested_event(source)
    source.add_(1)
    return result + other


def alias_event(source, other, effects):
    effects.append('once')
    source.view_as(source).add_(1)
    event_consumer()
    return other + source


def normalize_error(error):
    text = str(error)
    position = text.find('An event was recorded on a stream where a graph input was previously mutated.')
    if position >= 0:
        return {'category': 'input_mutation_event', 'message': text[position:].split('Input mutation occurred here:')[0].strip()}
    return {'category': type(error).__name__, 'message': text}


def trace_case(model):
    torch._dynamo.reset()
    EVENT_TRACES.clear()
    SCAN_TRACES.clear()
    ATTEMPTS.clear()
    source, other, effects = torch.ones(2), torch.full((2,), .25), []
    calls = []

    def backend(module, examples):
        calls.append(module.code)
        return module.forward

    with (torch.no_grad(), patch.object(OutputGraph, 'check_input_mutation_on_current_stream', check_on_synthetic_stream),
          patch.object(InstructionTranslator, '__init__', initialize_output)):
        try:
            compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
            result = {'output': compiled(source, other, effects).tolist()}
        except Exception as error:
            result = {'error': normalize_error(error)}
    result.update(source=source.tolist(), other=other.tolist(), effects=effects,
                  backend_calls=len(calls), fx_code=calls)
    evidence = {'attempts': ATTEMPTS.copy(), 'event_traces': EVENT_TRACES.copy(),
                'scan_traces': SCAN_TRACES.copy()}
    return result, evidence


def direct_check_sequences():
    results = {}
    fixed_stack = traceback.StackSummary.from_list([('stream_fixture.py', 1, 'fixture', 'source.add_(1)')])
    with tracing(TracingContext(FakeTensorMode())), patch.object(TracingContext, 'extract_stack', return_value=fixed_stack):
        for case in ('metadata_replacement', 'alias', 'reorder', 'new_input', 'direct_stream', 'other_stream'):
            output, tracer = fixture()
            log = SpeculationLog()
            if IS_CANDIDATE:
                log.requires_input_mutation_stream_checks = True
            output.root_tx.speculation_log = log
            current_stream = [7]
            in_context = [True]
            tx = SimpleNamespace(speculation_log=log, symbolic_stream_state=SimpleNamespace(
                in_stream_context=lambda: in_context[0], cur_stream_id=lambda: current_stream[0]))
            source, other = torch.zeros(1), torch.zeros(1)
            psource = add(tracer, 'source', source)
            pother = add(tracer, 'other', source.view_as(source) if case == 'alias' else other)
            CHECK(output, tx)
            if case == 'metadata_replacement':
                other.add_(1)
                psource.node.meta['example_value'] = other
            elif case == 'reorder':
                other.add_(1)
                psource.node.prepend(pother.node)
            elif case == 'new_input':
                add(tracer, 'later', torch.zeros(1), before=True)
                source.add_(1)
            elif case == 'direct_stream':
                in_context[0] = False
                source.add_(1)
            elif case == 'other_stream':
                current_stream[0] = 9
                other.add_(1)
            else:
                source.add_(1)
            CHECK(output, tx)
            errors = {}
            for stream in (7, 9):
                try:
                    CONSUME(output, stream)
                except Exception as error:
                    errors[str(stream)] = normalize_error(error)
            results[case] = {'versions': {str(index): value for index, value in (output._last_checked_input_versions or {}).items()},
                             'mutation_streams': sorted(output._input_mutation_streams), 'event_errors': errors}
    return results


def consumer_deoptimization():
    if not IS_CANDIDATE:
        return {'not_applicable': True}
    result = {}
    for name, method in [('stream_record_event', StreamVariable.record_event), ('event_record', EventVariable.record)]:
        output = OutputGraph.__new__(OutputGraph)
        log = SpeculationLog()
        output.root_tx = SimpleNamespace(speculation_log=log)
        output._input_mutation_streams = {}
        tx = SimpleNamespace(output=output)
        emitted = []
        output.create_proxy = lambda *values: emitted.append(values)
        stream = StreamVariable.__new__(StreamVariable)
        stream.value = SimpleNamespace(record_event=lambda: emitted.append('native_record'))
        stream.user_object_index = 2
        event = EventVariable.__new__(EventVariable)
        event.value = object()
        event.user_object_index = 3
        with patch.object(EventVariable, '_get_stream_arg', return_value=(stream, 2)):
            try:
                method(stream if name == 'stream_record_event' else event, tx, [], {})
            except exc.SpeculationRestartAnalysis:
                pass
            else:
                raise AssertionError('event consumer must restart speculative tracing')
        assert emitted == []
        assert log.requires_input_mutation_stream_checks
        log.restart()
        log.clear()
        assert log.requires_input_mutation_stream_checks
        assert not SpeculationLog().requires_input_mutation_stream_checks
        result[name] = {'deoptimized_before_any_record_or_emission': True, 'flag_survives_restart_and_clear': True}
    return result


def main():
    results = {'mode': args.mode, 'torch': torch.__version__, 'traces': {}, 'evidence': {}}
    models = (no_event, event_before_mutation, mutation_before_event, event_in_inlined_function,
              event_in_nested_region, nested_event_before_mutation, alias_event)
    with patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU qualification attempted CUDA initialization')):
        for model in models:
            result, evidence = trace_case(model)
            results['traces'][model.__name__] = result
            results['evidence'][model.__name__] = evidence
        results['direct_sequences'] = direct_check_sequences()
        results['consumer_deoptimization'] = consumer_deoptimization()
    results['cuda_initialized'] = torch.cuda.is_initialized()
    assert not results['cuda_initialized']
    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))
    assert results['traces']['no_event']['effects'] == ['once']
    assert results['traces']['event_before_mutation']['effects'] == ['once']
    assert results['traces']['event_before_mutation']['output'] == [4.25, 4.25]
    for name in ('mutation_before_event', 'event_in_inlined_function', 'event_in_nested_region', 'alias_event'):
        assert results['traces'][name]['error']['category'] == 'input_mutation_event', (name, results['traces'][name])
        assert results['traces'][name]['source'] == [1., 1.]
        assert results['traces'][name]['effects'] == []
    if IS_CANDIDATE:
        baseline = json.loads(args.baseline_result.read_text())
        assert results['traces'] == baseline['traces']
        assert results['direct_sequences'] == baseline['direct_sequences']
        for name, evidence in results['evidence'].items():
            if name == 'no_event':
                assert len(evidence['attempts']) == 1
                assert not any(row['initialized'] for row in evidence['scan_traces'])
            else:
                assert len(evidence['attempts']) == 2, (name, evidence)
                assert evidence['attempts'][0]['required'] is False
                assert evidence['attempts'][1]['required'] is True
                assert evidence['attempts'][0]['log_id'] == evidence['attempts'][1]['log_id']
                legacy_scans = baseline['evidence'][name]['scan_traces']
                retraced_scans = [row for row in evidence['scan_traces'] if row['attempt'] == 2]
                def without_attempt(rows):
                    return [{key: value for key, value in row.items() if key != 'attempt'} for row in rows]
                assert without_attempt(retraced_scans) == without_attempt(legacy_scans), name
        assert any(depth > 1 for _, depth in results['evidence']['event_in_nested_region']['event_traces'])


if __name__ == '__main__':
    main()
