# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compile, serialize, reload, and replay no-clone custom operators on CPU."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import torch
from torch._inductor import config
from torch._inductor.fx_passes import post_grad
from torch._inductor.utils import fresh_cache
from torch._higher_order_ops.auto_functionalize import auto_functionalized_v2
import no_clone_fx_cpu as suite

torch.set_num_threads(1)
def tensor_model(source, delta):
    local = source + 0.25
    suite.touch(local, delta)
    source.add_(delta * 0.5)
    return local, local[1:], source, source[1:]

def list_and_view_model(source, delta):
    first, second = source + 0.25, delta * 2.0
    suite.touch_list([delta], [first, second], 3, [1, 2])
    updated = auto_functionalized_v2(
        suite.TOUCH, delta=delta[1:], gain=1.0,
        _all_bases=[source], _value_base_index=0,
        _value_slice_dim=0, _value_slice_start=1, _value_slice_end=8,
    )[1]
    return first, second, updated, updated[1:], source, source[1:]

def run(name, model, root):
    torch._dynamo.reset()
    decisions = []
    helper = getattr(post_grad, "_auto_functionalized_v2_no_clone_trace", None)
    def observed(*args, **kwargs):
        result = helper(*args, **kwargs)
        positional, keywords = suite.pt.tree_unflatten(args[1], args[2])
        decisions.append(dict(op=str(args[0]), direct=result is not None,
            clone_selection=args[3], keywords=list(keywords),
            literal_types=[type(value).__name__ for value in args[1]
                           if not isinstance(value, torch.fx.Node)]))
        return result
    with fresh_cache(dir=str(root), delete=False), config.patch(compile_threads=1):
        with patch.object(post_grad, "_auto_functionalized_v2_no_clone_trace",
                          observed, create=True) if helper else patch.dict({}, {}):
            artifact = torch.compile(model, fullgraph=True, dynamic=False).aot_compile(
                ((torch.arange(8, dtype=torch.float32), torch.ones(8)), {})
            )
        path = root / (name + ".bin")
        artifact.save_compiled_function(str(path))
        with path.open("rb") as file:
            reloaded = torch.compiler.load_compiled_function(file)
        results = []
        for start in (0.0, 3.0, -7.0):
            expected_inputs = (torch.arange(8, dtype=torch.float32) + start,
                               torch.full((8,), start + 1.0))
            expected = suite.q.describe_outputs((*model(*expected_inputs), *expected_inputs))
            for kind, compiled in (("compiled", artifact), ("reloaded", reloaded)):
                inputs = (torch.arange(8, dtype=torch.float32) + start,
                          torch.full((8,), start + 1.0))
                result = suite.q.describe_outputs((*compiled(*inputs), *inputs))
                assert result == expected, (name, start, kind)
                results.append(dict(start=start, callable=kind, result=result))
    if helper:
        assert any(item["direct"] for item in decisions), (name, decisions)
        if name == "list_and_view":
            assert any(not item["direct"] for item in decisions), decisions
    return dict(name=name, decisions=decisions, results=results, artifact=str(path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), patch("torch.cuda._lazy_init",
                               side_effect=AssertionError("CUDA initialization forbidden")):
        rows = [run(name, model, args.output.parent) for name, model in
                (("tensor", tensor_model), ("list_and_view", list_and_view_model))]
    assert not torch.cuda.is_initialized()
    result = dict(torch=torch.__version__, git=torch.version.git_version,
                  cuda_initialized=False, rows=rows)
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        measured = json.loads(json.dumps([row["results"] for row in result["rows"]]))
        assert measured == [row["results"] for row in baseline["rows"]]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([dict(name=row["name"], decisions=row["decisions"],
                          replays=len(row["results"])) for row in rows], indent=2))

if __name__ == "__main__":
    main()
