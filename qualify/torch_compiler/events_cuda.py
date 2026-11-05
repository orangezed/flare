# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare actual CUDA event tracing with legacy and event-triggered checks.

Observers delegate to the installed checker and translator implementations.
The FX backend executes its real CUDA tensor and event operations unchanged.
Select an unused GPU with CUDA_VISIBLE_DEVICES, then run this file with
baseline or candidate as its sole argument. It prints one JSON result.
"""

import argparse
import json

import torch
from torch._dynamo import exc
from torch._dynamo.output_graph import OutputGraph
from torch._dynamo.symbolic_convert import InstructionTranslator, SpeculationLog

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=('baseline', 'candidate'))
args = parser.parse_args()
candidate = args.mode == 'candidate'
assert hasattr(SpeculationLog(), 'requires_input_mutation_stream_checks') == candidate
torch.cuda.set_device(0)
torch.set_num_threads(4)
original_init = InstructionTranslator.__init__
original_scan = OutputGraph.check_input_mutation_on_current_stream
original_consume = OutputGraph.check_event_record_after_input_mutation
attempts, scans, consumers = [], [], []


def observe_init(self, *values, **keywords):
    original_init(self, *values, **keywords)
    attempts.append({'log': id(self.speculation_log),
        'required': getattr(self.speculation_log, 'requires_input_mutation_stream_checks', True)})


def observe_scan(self, tx):
    required = getattr(tx.speculation_log, 'requires_input_mutation_stream_checks', True)
    result = original_scan(self, tx)
    scans.append({'attempt': len(attempts), 'required': required,
        'initialized': self._last_checked_input_versions is not None,
        'beginning_versions': list(self.root_tracer._input_versions_at_beginning)})
    return result


def observe_consumer(self, stream):
    row = {'attempt': len(attempts), 'required': getattr(self.root_tx.speculation_log,
        'requires_input_mutation_stream_checks', True)}
    consumers.append(row)
    try:
        result = original_consume(self, stream)
    except exc.SpeculationRestartAnalysis:
        row['restart'] = True
        raise
    except RuntimeError:
        row['legacy_rejection'] = True
        raise
    else:
        row['accepted'] = True
        return result


InstructionTranslator.__init__ = observe_init
OutputGraph.check_input_mutation_on_current_stream = observe_scan
OutputGraph.check_event_record_after_input_mutation = observe_consumer


def model_for(kind, consumer):
    def record(event, stream):
        if consumer == 'event':
            event.record(stream)
        elif consumer == 'stream':
            stream.record_event(event)
        else:
            stream.record_event()

    def model(source, alias, effects, stream, other_stream, event, initial_stream):
        effects.append('once')
        if kind == 'direct':
            torch.cuda.set_stream(stream)
            source.add_(1)
            record(event, stream)
            torch.cuda.set_stream(initial_stream)
        else:
            with torch.cuda.stream(stream):
                if kind in ('before', 'repeated_before', 'nested_before'):
                    record(event, stream)
                if kind == 'repeated_before':
                    record(event, stream)
                if kind == 'nested_before':
                    with torch.cuda.stream(other_stream):
                        record(event, other_stream)
                if kind == 'alias_after':
                    alias.add_(1)
                else:
                    source.add_(1)
                if kind in ('after', 'alias_after'):
                    record(event, stream)
                if kind == 'nested_after':
                    with torch.cuda.stream(other_stream):
                        record(event, stream)
        return source * 2 + alias
    return model


def normalize_error(error):
    message = str(error)
    marker = 'An event was recorded on a stream where a graph input was previously mutated.'
    if marker in message:
        return {'category': 'input_mutation_event',
                'message': message[message.index(marker):].split('Input mutation occurred here:')[0].strip()}
    return {'category': type(error).__name__, 'message': message}


results = {}
for consumer in ('event', 'stream', 'implicit_stream'):
    for kind in ('noevent', 'before', 'after', 'alias_after', 'repeated_before', 'nested_before', 'nested_after', 'direct'):
        if consumer == 'implicit_stream' and kind not in ('before', 'after'):
            continue
        name = f'{consumer}_{kind}'
        torch._dynamo.reset()
        attempts.clear(); scans.clear(); consumers.clear()
        source = torch.ones(4, device='cuda')
        alias = source.view_as(source)
        effects = []
        stream, other_stream = torch.cuda.Stream(), torch.cuda.Stream()
        event = torch.cuda.Event()
        initial_stream = torch.cuda.current_stream()
        torch.cuda.synchronize()
        graphs = []

        def backend(module, examples):
            graphs.append({'code': module.code,
                'streams': [node.meta.get('custom', {}).get('stream') for node in module.graph.nodes]})
            return module.forward

        try:
            with torch.no_grad():
                compiled = torch.compile(model_for(kind, consumer), backend=backend, fullgraph=True, dynamic=False)
                output = compiled(source, alias, effects, stream, other_stream, event, initial_stream)
                torch.cuda.synchronize()
                result = {'output': output.cpu().tolist()}
                first_attempts = len(attempts)
                repeated_effects = []
                repeated_output = compiled(source, alias, repeated_effects, stream, other_stream, event, initial_stream)
                torch.cuda.synchronize()
                result.update(repeated_output=repeated_output.cpu().tolist(), repeated_effects=repeated_effects,
                              repeated_call_retraced=len(attempts) != first_attempts)
                assert len(attempts) == first_attempts, (name, attempts)
        except Exception as error:
            result = {'error': normalize_error(error)}
        finally:
            torch.cuda.set_stream(initial_stream)
            torch.cuda.synchronize()
        result.update(source=source.cpu().tolist(), alias=alias.cpu().tolist(), effects=effects,
                      graphs=graphs, explicit_event_initialized=event.cuda_event != 0)
        evidence = {'attempts': [dict(item) for item in attempts], 'scans': [dict(item) for item in scans],
                    'consumers': [dict(item) for item in consumers]}
        if candidate:
            expected_attempts = 1 if kind == 'noevent' else 2
            assert len(attempts) == expected_attempts, (name, evidence, result)
            assert len({item['log'] for item in attempts}) == 1, (name, evidence)
            assert sum(item.get('restart', False) for item in consumers) == (kind != 'noevent'), (name, evidence)
            if kind == 'noevent':
                assert scans and not any(item['initialized'] for item in scans), (name, evidence)
            if kind in ('after', 'alias_after', 'nested_after'):
                assert result.get('error', {}).get('category') == 'input_mutation_event', (name, result)
                assert not graphs and not result['explicit_event_initialized'], (name, result)
                assert result['source'] == [1.0] * 4 and effects == [], (name, result)
        for item in evidence['attempts']:
            item.pop('log')
        results[name] = {'result': result, 'evidence': evidence}

print(json.dumps({'mode': args.mode, 'torch': torch.__version__,
                  'gpu': torch.cuda.get_device_name(0), 'cases': results}, indent=2), flush=True)
