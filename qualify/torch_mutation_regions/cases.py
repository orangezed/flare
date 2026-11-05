"""CPU comparison of the installed auto-functionalized decomposition pass."""

import argparse
import collections
import hashlib
import inspect
import json
import operator
import os
from pathlib import Path
import resource
import time

import torch

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
torch.set_num_threads(1)


def reject_cuda(*args, **kwargs):
    raise AssertionError("CUDA initialization is forbidden")


torch.cuda._lazy_init = reject_cuda
from torch._guards import detect_fake_mode
from torch._higher_order_ops.auto_functionalize import auto_functionalized, auto_functionalized_v2
from torch._inductor import pattern_matcher as pm
from torch._inductor.fx_passes import post_grad
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx

lib = torch.library.Library("mutation_region_study", "DEF")
lib.define("touch_(Tensor(a!) x, Tensor delta) -> ()")
lib.define("pair_(Tensor(a!) x, Tensor(b!) y) -> ()")


def touch(x, delta):
    x.add_(delta)


def pair(x, y):
    x.add_(0.25)
    y.mul_(0.5)


lib.impl("touch_", touch, "CPU")
lib.impl("pair_", pair, "CPU")
torch.library.register_fake("mutation_region_study::touch_")(lambda x, delta: None)
torch.library.register_fake("mutation_region_study::pair_")(lambda x, y: None)
touch_op = torch.ops.mutation_region_study.touch_.default
pair_op = torch.ops.mutation_region_study.pair_.default
HOPS = (auto_functionalized, auto_functionalized_v2)


def inputs(layout="plain", dtype=torch.float32):
    if layout == "empty":
        x = torch.empty((0,), dtype=dtype)
    elif layout == "strided":
        x = torch.arange(32, dtype=dtype).reshape(4, 8)[:, ::2]
    else:
        x = torch.arange(8, dtype=dtype)
    return x, torch.ones_like(x)


def workload(version, count, owned=False, aliases=False, ballast=0):
    def run(x, delta):
        original = x
        x = x.clone() if owned else x
        barrier = delta.clone()
        for _ in range(count):
            barrier.add_(0)
            if aliases:
                x = auto_functionalized_v2(pair_op, _all_bases=[x],
                    _x_base_index=0, _x_slice_dim=0, _x_slice_start=0, _x_slice_end=6,
                    _y_base_index=0, _y_slice_dim=0, _y_slice_start=2, _y_slice_end=8)[1]
            elif version == 1:
                x = auto_functionalized(touch_op, x=x, delta=barrier)[1]
            else:
                x = auto_functionalized_v2(touch_op, delta=barrier,
                                           _all_bases=[x], _x_base_index=0)[1]
            for _ in range(ballast):
                x = x + 0
        return x, x[2:], original, barrier
    return run


def describe_outputs(values):
    return {
        "values": [{"dtype": str(x.dtype), "shape": tuple(x.shape), "stride": tuple(x.stride()),
                    "offset": x.storage_offset(), "bytes": x.contiguous().view(torch.uint8).flatten().tolist()}
                   for x in values],
        "aliases": [[torch._C._is_alias_of(a, b) for b in values] for a in values],
    }


def copy_inputs(values):
    return tuple(torch.empty_strided(x.shape, x.stride(), dtype=x.dtype).copy_(x) for x in values)


def qualify(name, version=2, count=4, owned=False, aliases=False, layout="plain",
            dtype=torch.float32, initial_ids="absent", ballast=0):
    args = inputs(layout, dtype)
    run = workload(version, count, owned, aliases, ballast)
    expected_args = copy_inputs(args)
    expected = describe_outputs(run(*expected_args))
    gm = make_fx(run, tracing_mode="fake")(*args)
    roots = [node for node in gm.graph.nodes if node.target in HOPS]
    assert len(roots) == count
    for index, node in enumerate(roots):
        if owned:
            node.meta["only_clone_these_tensors"] = []
        node.meta["custom"] = {"stream": index % 2 + 3, "mempool": 17, "mempool_device": 0}
        match = pm.CallFunctionVarArgs(node.target).match(node)
        assert pm.is_match(match) and match.nodes == [node]
    if initial_ids == "stale":
        for node in gm.graph.nodes:
            node.meta["mutation_region_id"] = 999
    elif initial_ids == "valid":
        pm.compute_mutation_region_ids(gm.graph)
    scans = []
    compute = pm.compute_mutation_region_ids

    def counted(graph):
        scans.append(graph._len)
        return compute(graph)

    mode = detect_fake_mode([node.meta.get("val") for node in gm.graph.nodes])
    pm.compute_mutation_region_ids = counted
    start, cpu = time.perf_counter(), time.process_time()
    try:
        with V.set_fake_mode(mode):
            post_grad.decompose_auto_functionalized(gm.graph)
    finally:
        elapsed, cpu_elapsed = time.perf_counter() - start, time.process_time() - cpu
        pm.compute_mutation_region_ids = compute
    gm.recompile()
    gm.graph.lint()
    assert not any(node.target in HOPS for node in gm.graph.nodes)
    regions, expected_region, custom = [], 0, []
    for node in gm.graph.nodes:
        expected_region += int(pm.is_mutation_op(node))
        assert node.meta.get("mutation_region_id") == expected_region
        regions.append(expected_region)
        custom.append(node.meta.get("custom"))
    actual_args = copy_inputs(args)
    actual = describe_outputs(gm(*actual_args))
    assert actual == expected, name
    assert all(torch.equal(a, b) for a, b in zip(actual_args, args))
    side_effect_contexts = [node.meta.get("custom") for node in gm.graph.nodes
                            if node.target in (touch_op, pair_op)]
    assert len(side_effect_contexts) == count
    assert all(c and c["stream"] in (3, 4) and c["mempool"] == 17 and c["mempool_device"] == 0
               for c in side_effect_contexts)
    row = dict(name=name, version=version, count=count, scans=len(scans), node_visits=sum(scans),
               wall_s=elapsed, cpu_s=cpu_elapsed, nodes=len(regions), code=gm.code,
               regions=regions, custom=custom, output=actual)
    print(json.dumps({k: v for k, v in row.items() if k not in ("code", "regions", "custom", "output")}), flush=True)
    return row


def default_barrier_check():
    results = []
    for mutation in (False, True):
        def fn(x):
            y = torch.neg(x)
            if mutation:
                x.add_(1)
            return torch.relu(y)
        gm = make_fx(fn, tracing_mode="fake")(torch.arange(8, dtype=torch.float32))
        graph_pass = pm.PatternMatcherPass()
        pattern = pm.CallFunction(torch.ops.aten.relu.default,
            pm.CallFunction(torch.ops.aten.neg.default, pm.KeywordArg("x")))
        seen = []
        @pm.register_graph_pattern(pattern, pass_dict=graph_pass)
        def handler(match, **kwargs):
            seen.append(len(match.nodes))
        count = graph_pass.apply(gm.graph)
        assert count == (0 if mutation else 1)
        results.append(count)
        if "defer_mutation_region_refresh" in inspect.signature(graph_pass.apply).parameters:
            try:
                graph_pass.apply(gm.graph, defer_mutation_region_refresh=True)
            except AssertionError as error:
                assert "singleton" in str(error)
            else:
                raise AssertionError("deferred mode accepted a multi-node match")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", default="32,128,384")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    os.nice(15)
    rows = []
    # Exclude initial import/decomposition setup from scaling measurements.
    qualify("warmup", count=1)
    cases = [dict(name="v1_clone", version=1), dict(name="v1_owned_reinplace", version=1, owned=True),
             dict(name="v1_strided", version=1, layout="strided"), dict(name="v1_bf16", version=1, dtype=torch.bfloat16),
             dict(name="v1_empty", version=1, layout="empty"), dict(name="v2_clone"),
             dict(name="v2_owned_reinplace", owned=True), dict(name="v2_alias_clone", aliases=True),
             dict(name="v2_alias_reinplace", aliases=True, owned=True),
             dict(name="v2_stale_regions", initial_ids="stale"), dict(name="v2_valid_regions", initial_ids="valid")]
    for case in cases:
        rows.append(qualify(**case))
    barrier = default_barrier_check()
    for count in (int(value) for value in args.sizes.split(",")):
        for repeat in range(args.repeats):
            rows.append(qualify(f"scale_{count}_{repeat}", count=count, ballast=4))
    root = Path(torch.__file__).parent
    versions = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in ("_inductor/pattern_matcher.py", "_inductor/fx_passes/post_grad.py")}
    result = dict(torch_version=torch.__version__, torch_git=torch.version.git_version, torch_path=str(root),
                  sources=versions, default_barrier=barrier, cuda_initialized=torch.cuda.is_initialized(),
                  maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, rows=rows)
    assert not result["cuda_initialized"]
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
