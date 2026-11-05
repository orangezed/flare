# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qualify unbacked integer handle inputs through CPU Inductor AOT artifacts."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import torch
from torch._inductor import config
from torch._inductor.fx_passes import post_grad
from torch._inductor.utils import fresh_cache

torch.set_num_threads(1)

@torch.library.custom_op("handle_abi_study::fill", mutates_args=("output",))
def fill(output: torch.Tensor, handle: int) -> None:
    output.fill_(handle + 0.25)

@fill.register_fake
def _(output, handle):
    return None

class HandleForward:
    def __init__(self, handle):
        self._handle = handle
        self.output = torch.empty(8)
    def __call__(self):
        fill(self.output, self._handle)
        return self.output, self.output[1:]

def execute(forward):
    return forward()

def geometry(outputs, forward):
    return dict(values=[value.tolist() for value in outputs],
                shapes=[list(value.shape) for value in outputs],
                strides=[list(value.stride()) for value in outputs],
                aliases_output=[torch._C._is_alias_of(value, forward.output) for value in outputs])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-direct", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    original = getattr(post_grad, "_auto_functionalized_v2_no_clone_trace", None)
    factory = original
    decisions = []
    binding_observations = []
    decompose = post_grad.decompose_auto_functionalized
    def observe_bindings(graph):
        before = {node: (str(node.meta["val"]),
                        repr(node.meta.get("unbacked_bindings")))
                  for node in graph.nodes
                  if isinstance(node.meta.get("val"), torch.SymInt)}
        result = decompose(graph)
        after = {node: (str(node.meta["val"]),
                       repr(node.meta.get("unbacked_bindings")))
                 for node in graph.nodes
                 if isinstance(node.meta.get("val"), torch.SymInt)}
        binding_observations.append(dict(
            names=[node.name for node in before],
            existing_nodes_and_metadata_preserved=before == after,
            added_symbol_nodes=[node.name for node in after if node not in before],
            bindings=[dict(name=node.name, value=data[0], bindings=data[1])
                      for node, data in after.items()]))
        assert before == after
        return result
    def observed(*values, **kwargs):
        result = factory(*values, **kwargs)
        positional, named = torch.utils._pytree.tree_unflatten(values[1], values[2])
        decisions.append(dict(op=str(values[0]), direct=result is not None,
            clones=values[3],
            inputs=[dict(name=value.name, type=type(value.meta.get("val")).__name__,
                         value=str(value.meta.get("val")) if isinstance(value.meta.get("val"), torch.SymInt) else None)
                    for value in values[1] if isinstance(value, torch.fx.Node)]))
        return result
    sources = r'.*\.(?:_?handle|stream_handle|launch_order|plans\[\d+\])$'
    with torch.no_grad(), patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")), fresh_cache(dir=str(args.output.parent), delete=False), config.patch(compile_threads=1), torch.compiler.config.patch(unbacked_sources=sources):
        forward = HandleForward(5)
        with patch.object(post_grad, "decompose_auto_functionalized", observe_bindings), (
            patch.object(post_grad, "_auto_functionalized_v2_no_clone_trace", observed, create=True)
            if factory else patch.dict({}, {})
        ):
            artifact = torch.compile(execute, fullgraph=True, dynamic=False).aot_compile(((forward,), {}))
        path = args.output.parent / "handle.bin"
        artifact.save_compiled_function(str(path))
        with path.open("rb") as file:
            loaded = torch.compiler.load_compiled_function(file)
        rows = []
        for handle in (5, 127, 1, 0, -7):
            for kind, callable in (("compiled", artifact), ("loaded", loaded)):
                bound = HandleForward(handle)
                accepted = callable.guard_check(bound)
                assert accepted == (handle >= 0), (kind, handle, accepted)
                if not accepted:
                    rows.append(dict(handle=handle, callable=kind, guard_accepted=False))
                    continue
                outputs = callable(bound)
                expected = torch.full((8,), handle + 0.25)
                assert torch.equal(outputs[0], expected)
                assert torch.equal(outputs[1], expected[1:])
                assert outputs[0] is bound.output
                assert torch._C._is_alias_of(outputs[1], bound.output)
                rows.append(dict(handle=handle, callable=kind, result=geometry(outputs, bound)))
    assert not torch.cuda.is_initialized()
    result = dict(decisions=decisions, rows=rows, binding_observations=binding_observations,
                  cuda_initialized=False,
                  direct_expected=args.expect_direct,
                  torch=torch.__version__, git=torch.version.git_version)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(decisions=decisions, replay_count=len(rows)), indent=2))
    if args.expect_direct:
        assert any(row["direct"] for row in decisions), decisions

if __name__ == "__main__":
    main()
