"""Bounded child-graph and large-context checks for mutation-region refresh."""

import argparse
import json
import time
from pathlib import Path

import cases as q


def run_pass(gm, mode):
    scans = []
    original = q.pm.compute_mutation_region_ids
    def counted(graph):
        scans.append(graph._len)
        return original(graph)
    q.pm.compute_mutation_region_ids = counted
    started = time.perf_counter()
    try:
        with q.V.set_fake_mode(mode):
            q.post_grad.decompose_auto_functionalized(gm.graph)
    finally:
        elapsed = time.perf_counter() - started
        q.pm.compute_mutation_region_ids = original
    descriptions = {}
    for name, module in gm.named_modules():
        if not isinstance(module, q.torch.fx.GraphModule):
            continue
        module.recompile()
        module.graph.lint()
        expected = 0
        regions, custom = [], []
        for node in module.graph.nodes:
            assert node.target not in q.HOPS
            expected += int(q.pm.is_mutation_op(node))
            assert node.meta.get("mutation_region_id") == expected
            regions.append(expected)
            custom.append(node.meta.get("custom"))
        descriptions[name] = dict(code=module.code, regions=regions, custom=custom)
    return dict(wall_s=elapsed, scans=len(scans), node_visits=sum(scans), graphs=descriptions)


def nested(version):
    args = q.inputs()
    fn = q.workload(version, 3, owned=True)
    child = q.make_fx(fn, tracing_mode="fake")(*args)
    for node in child.graph.nodes:
        if node.target in q.HOPS:
            node.meta["only_clone_these_tensors"] = []
            node.meta["custom"] = dict(stream=9, mempool=23, mempool_device=0)
    mode = q.detect_fake_mode([node.meta.get("val") for node in child.graph.nodes])
    for _ in range(2):
        graph = q.torch.fx.Graph()
        x, delta = graph.placeholder("x"), graph.placeholder("delta")
        attr = graph.get_attr("child")
        call = graph.call_function(q.post_grad.control_deps, ((), attr, x, delta))
        graph.output(call)
        child = q.torch.fx.GraphModule({"child": child}, graph)
    result = run_pass(child, mode)
    actual = q.describe_outputs(child(*q.copy_inputs(args)))
    assert actual == q.describe_outputs(fn(*q.copy_inputs(args)))
    result["output"] = actual
    return result


def large_context():
    args = q.inputs()
    fn = q.workload(2, 32)
    gm = q.make_fx(fn, tracing_mode="fake")(*args)
    first = next(node for node in gm.graph.nodes if node.target in q.HOPS)
    value = first.kwargs["_all_bases"][0]
    # A used alias chain gives a large valid FX graph with only32rewrites.
    # No tensors or numerical work are enlarged by this context.
    with gm.graph.inserting_before(first):
        for _ in range(32768):
            alias = gm.graph.call_function(q.torch.ops.aten.alias.default, (value,))
            alias.meta["val"] = value.meta["val"]
            value = alias
    first.kwargs = dict(first.kwargs, _all_bases=[value])
    mode = q.detect_fake_mode([node.meta.get("val") for node in gm.graph.nodes])
    result = run_pass(gm, mode)
    result["output"] = q.describe_outputs(gm(*q.copy_inputs(args)))
    assert result["output"] == q.describe_outputs(fn(*q.copy_inputs(args)))
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
q.qualify("warmup", count=1)
results = {"nested_v1": nested(1), "nested_v2": nested(2), "large_context_32": large_context()}
assert not q.torch.cuda.is_initialized()
args.output.write_text(json.dumps(results) + "\n")
for name, result in results.items():
    print(json.dumps(dict(name=name, **{key: result[key] for key in ("wall_s", "scans", "node_visits")})), flush=True)
