# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU parity and timing for no-clone custom-operator decomposition.

Run with CUDA_VISIBLE_DEVICES= in reference and patched Torch environments.
The script never initializes CUDA.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch_mutation_regions"))
import cases as q
from torch.utils import _pytree as pt

torch = q.torch
FAKE_CALLS = []

@torch.library.custom_op("static_abi_study::touch_", mutates_args=("value",))
def touch(value: torch.Tensor, delta: torch.Tensor, gain: float = 1.0) -> None:
    value.add_(delta * gain)

@touch.register_fake
def _(value, delta, gain=1.0):
    FAKE_CALLS.append('touch')
    return None

@torch.library.custom_op("static_abi_study::list_", mutates_args=("writes",))
def touch_list(reads: list[torch.Tensor], writes: list[torch.Tensor], handle: int,
               integers: list[int] | None = None) -> None:
    for value in writes:
        value.add_(reads[0] * handle)

@touch_list.register_fake
def _(reads, writes, handle, integers=None):
    FAKE_CALLS.append('touch_list')
    return None

@torch.library.custom_op("static_abi_study::optional_", mutates_args=("value", "sink"))
def touch_optional(value: torch.Tensor | None, sink: torch.Tensor,
                   delta: torch.Tensor) -> None:
    sink.add_(delta)
    if value is not None:
        value.mul_(2)

@touch_optional.register_fake
def _(value, sink, delta):
    FAKE_CALLS.append('touch_optional')
    return None

@torch.library.custom_op("static_abi_study::keyword_", mutates_args=("value",))
def touch_keyword(value: torch.Tensor, delta: torch.Tensor, *, gain: float = 1.0) -> None:
    value.add_(delta * gain)

@touch_keyword.register_fake
def _(value, delta, *, gain=1.0):
    FAKE_CALLS.append('touch_keyword')
    return None

@torch.library.custom_op("static_abi_study::returning_", mutates_args=("value",))
def returning(value: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    value.add_(delta)
    return delta.clone()

@returning.register_fake
def _(value, delta):
    FAKE_CALLS.append('returning')
    return torch.empty_like(delta)

@torch.library.custom_op("static_abi_study::typed_", mutates_args=("value",))
def touch_typed(value: torch.Tensor, delta: torch.Tensor, dtype: torch.dtype,
                device: torch.device, label: str = "typed", enabled: bool = True,
                gain: float = 1.0, *, bias: int = 0) -> None:
    if enabled:
        value.add_(delta.to(dtype=dtype, device=device) * gain + bias)

@touch_typed.register_fake
def _(value, delta, dtype, device, label="typed", enabled=True, gain=1.0, *, bias=0):
    FAKE_CALLS.append("touch_typed")
    return None

TYPED = torch.ops.static_abi_study.typed_.default
TOUCH = torch.ops.static_abi_study.touch_.default
LIST = torch.ops.static_abi_study.list_.default
OPTIONAL = torch.ops.static_abi_study.optional_.default
KEYWORD = torch.ops.static_abi_study.keyword_.default
RETURNING = torch.ops.static_abi_study.returning_.default
HOPS = q.HOPS

def workload(case, count):
    def run(x, delta):
        a, b = x.clone(), delta.clone()
        for _ in range(count):
            if case == "list":
                result = q.auto_functionalized_v2(LIST, reads=[delta], handle=3,
                    integers=[1, 2], _all_bases=[a, b], _writes_length=2,
                    _writes_0_base_index=0, _writes_1_base_index=1)
                a, b = result[1], result[2]
            elif case.startswith("optional"):
                present = case == "optional_present"
                result = q.auto_functionalized_v2(OPTIONAL, delta=delta,
                    _all_bases=[a, b], _value_base_index=0 if present else None,
                    _sink_base_index=1)
                a, b = result[1], result[2]
            elif case.startswith("typed"):
                extra = dict(label="typed", enabled=True, gain=1.0, bias=0)
                if case == "typed_mixed":
                    extra.update(label="mixed", enabled=True, gain=2.5, bias=3)
                elif case == "typed_disabled":
                    extra.update(label="disabled", enabled=False, gain=2.5, bias=3)
                a = q.auto_functionalized_v2(TYPED, delta=delta,
                    dtype=torch.float32, device=torch.device("cpu"),
                    _all_bases=[a], _value_base_index=0, **extra)[1]
            elif case == "manual_registration":
                a = q.auto_functionalized_v2(q.touch_op, delta=delta,
                    _all_bases=[a], _x_base_index=0)[1]
            elif case == "v1":
                a = q.auto_functionalized(TOUCH, value=a, delta=delta, gain=1.0)[1]
            elif case == "returning":
                result = q.auto_functionalized_v2(RETURNING, delta=delta,
                    _all_bases=[a], _value_base_index=0)
                a = result[1]
            elif case == "keyword":
                a = q.auto_functionalized_v2(KEYWORD, delta=delta, gain=1.0,
                    _all_bases=[a], _value_base_index=0)[1]
            elif case in ("alias", "slice", "strided_view"):
                view = dict(_value_base_index=0)
                if case == "alias":
                    view["_value_alias"] = True
                elif case == "slice":
                    view.update(_value_slice_dim=0, _value_slice_start=0,
                                _value_slice_end=x.shape[0])
                else:
                    view.update(_value_size=tuple(x.shape), _value_stride=tuple(x.stride()),
                                _value_storage_offset=0)
                a = q.auto_functionalized_v2(TOUCH, delta=delta, gain=1.0,
                    _all_bases=[a], **view)[1]
            else:
                a = q.auto_functionalized_v2(TOUCH, delta=delta, gain=1.0,
                    _all_bases=[a], _value_base_index=0)[1]
        return a, b, x, delta
    return run

def geometry(value):
    if isinstance(value, torch.Tensor):
        return dict(dtype=str(value.dtype), device=str(value.device),
                    shape=tuple(value.shape), stride=tuple(value.stride()),
                    offset=value.storage_offset())
    return value

def normalize_meta(node):
    return dict(target=str(node.target), custom=node.meta.get("custom"),
        mutation_region_id=node.meta.get("mutation_region_id"),
        eager=pt.tree_map(geometry, node.meta.get("eager_input_vals")))

def run_case(name, case="tensor", count=4, layout="plain", dtype=torch.float32,
             nested=False, cloning=False, decompose=False):
    inputs = q.inputs(layout, dtype)
    fn = workload(case, count)
    gm = q.make_fx(fn, tracing_mode="fake")(*inputs)
    for node in gm.graph.nodes:
        if node.target in HOPS:
            if not cloning:
                node.meta["only_clone_these_tensors"] = []
            node.meta["custom"] = dict(stream=7, mempool=17, mempool_device=0)
    mode = q.detect_fake_mode([node.meta.get("val") for node in gm.graph.nodes])
    if nested:
        for _ in range(2):
            graph = torch.fx.Graph()
            x, delta = graph.placeholder("x"), graph.placeholder("delta")
            child = graph.get_attr("child")
            out = graph.call_function(q.post_grad.control_deps, ((), child, x, delta))
            graph.output(out)
            gm = torch.fx.GraphModule({"child": gm}, graph)
    decisions = []
    helper = getattr(q.post_grad, "_auto_functionalized_v2_no_clone_trace", None)
    def observed(*args, **kwargs):
        result = helper(*args, **kwargs)
        decisions.append(result is not None)
        return result
    def touch_decomp(value, delta, gain=1.0):
        torch.ops.aten.add_.Tensor(value, delta * gain)
        return None
    with q.V.set_fake_mode(mode):
        table = q.pm.select_decomp_table()
        updates = {TOUCH: touch_decomp} if decompose else {}
        with patch.dict(table, updates):
            with patch.object(q.post_grad, "_auto_functionalized_v2_no_clone_trace",
                              observed, create=True) if helper else patch.dict({}, {}):
                FAKE_CALLS.clear()
                started = time.perf_counter()
                q.post_grad.decompose_auto_functionalized(gm.graph)
                elapsed = time.perf_counter() - started
    modules = []
    for module_name, module in gm.named_modules():
        if isinstance(module, torch.fx.GraphModule):
            module.graph.lint()
            module.recompile()
            modules.append(dict(name=module_name, code=module.code,
                metadata=[normalize_meta(node) for node in module.graph.nodes]))
    result = q.describe_outputs(gm(*q.copy_inputs(inputs)))
    expected = q.describe_outputs(fn(*q.copy_inputs(inputs)))
    assert result == expected, name
    fast_expected = not cloning and not decompose and case in (
        "tensor", "list", "optional_present", "optional_none", "keyword",
        "typed_defaults", "typed_mixed", "typed_disabled",
    )
    if helper and case != "v1":
        assert decisions and all(choice == fast_expected for choice in decisions), (name, decisions)
    return dict(name=name, wall_s=elapsed, fast_decisions=decisions, fake_calls=FAKE_CALLS.copy(),
                output=result, modules=modules)

def literal_exclusions():
    helper = getattr(q.post_grad, "_auto_functionalized_v2_no_clone_trace", None)
    if helper is None:
        return None
    inputs = q.inputs()
    gm = q.make_fx(workload("tensor", 1), tracing_mode="fake")(*inputs)
    node = next(node for node in gm.graph.nodes if node.target in HOPS)
    results = {}
    for name, value in (
        ("tensor", torch.ones(8)),
        ("generator", torch.Generator()),
        ("object", object()),
    ):
        kwargs = dict(node.kwargs, delta=value)
        flat, spec = pt.tree_flatten((tuple(node.args), kwargs))
        selected = helper(TOUCH, flat, spec, ())
        assert selected is None, name
        results[name] = "canonical_trace"
    return results

def symbolic_input_selection():
    helper = getattr(q.post_grad, "_auto_functionalized_v2_no_clone_trace", None)
    if helper is None:
        return None
    from torch._dynamo.source import ConstantSource
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    env = ShapeEnv()
    source = ConstantSource("static_abi_study_symbol")
    expr = env.create_symbol(2, source)
    symbol = env.create_symintnode(expr, hint=2, source=source)
    with FakeTensorMode(shape_env=env):
        tensors = {
            "size": torch.empty((symbol,)),
            "stride": torch.empty_strided((2,), (symbol,)),
            "offset": torch.as_strided(torch.empty(32), (2,), (1,), symbol),
        }
    gm = q.make_fx(workload("tensor", 1), tracing_mode="fake")(*q.inputs())
    node = next(node for node in gm.graph.nodes if node.target in HOPS)
    base = node.kwargs["_all_bases"][0]
    initial = base.meta["val"]
    results = {}
    for name, value in tensors.items():
        base.meta["val"] = value
        flat, spec = pt.tree_flatten((tuple(node.args), dict(node.kwargs)))
        assert helper(TOUCH, flat, spec, ()) is None, name
        results[name] = "canonical_trace"
    base.meta["val"] = initial
    for name, value in (
        ("scalar_int", symbol),
        ("scalar_bool", symbol % 2 == 0),
        ("scalar_float", symbol / 2.0),
    ):
        scalar = gm.graph.placeholder(name)
        scalar.meta["val"] = value
        kwargs = dict(reads=[node.kwargs["delta"]], handle=scalar, integers=None,
                      _all_bases=[base], _writes_length=1, _writes_0_base_index=0)
        flat, spec = pt.tree_flatten(((LIST,), kwargs))
        selected = helper(LIST, flat, spec, ())
        assert (selected is not None) == (name == "scalar_int"), name
        results[name] = "direct_trace" if selected is not None else "canonical_trace"
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    run_case("warmup", count=1)
    cases = [
        dict(name="tensor"),
        dict(name="tensor_strided", layout="strided"),
        dict(name="tensor_empty", layout="empty"),
        dict(name="tensor_bf16", dtype=torch.bfloat16),
        dict(name="list", case="list"),
        dict(name="optional_present", case="optional_present"),
        dict(name="optional_none", case="optional_none"),
        dict(name="keyword", case="keyword"),
        dict(name="typed_defaults", case="typed_defaults"),
        dict(name="typed_mixed", case="typed_mixed"),
        dict(name="typed_disabled", case="typed_disabled"),
        dict(name="list_control_deps", case="list", nested=True),
        dict(name="optional_control_deps", case="optional_none", nested=True),
        dict(name="cloning", cloning=True),
        dict(name="cloning_list", case="list", cloning=True),
        dict(name="alias", case="alias"),
        dict(name="slice", case="slice"),
        dict(name="strided_view", case="strided_view"),
        dict(name="returning", case="returning"),
        dict(name="v1", case="v1"),
        dict(name="manual_registration", case="manual_registration"),
        dict(name="registered_decomposition", decompose=True),
        dict(name="scale_32", count=32),
        dict(name="scale_128", count=128),
    ]
    rows = [run_case(**case) for case in cases]
    result = dict(torch=torch.__version__, git=torch.version.git_version,
        source=str(Path(q.post_grad.__file__)), rows=rows,
        cuda_initialized=torch.cuda.is_initialized(), literal_exclusions=literal_exclusions(),
        symbolic_inputs=symbolic_input_selection())
    assert not result["cuda_initialized"]
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        def behavior(data):
            return [{key: value for key, value in row.items()
                     if key not in ("wall_s", "fast_decisions")} for row in data["rows"]]
        assert json.loads(json.dumps(behavior(result))) == behavior(baseline)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([dict(name=row["name"], wall_s=row["wall_s"],
                          fast=sum(row["fast_decisions"])) for row in rows], indent=2))

if __name__ == "__main__":
    main()
