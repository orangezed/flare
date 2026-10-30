"""Offline ladders retain bounded launch variants at inclusive payload boundaries."""

from copy import deepcopy
from pathlib import Path

import pytest

from ignition.errors import UsageError
from ignition.preparation.collective_ladder import load, resolve, select, validate

LADDER = Path(__file__).resolve().parents[2] / "ignition/preparation/ladders/gb202_pcie_tp4.json"
LADDERS = ((LADDER, "gb202_pcie_tp4"),
           (LADDER.with_name("gb202_pcie_tp4_host_bridge.json"), "gb202_pcie_tp4_host_bridge_candidates"))


def choose(data, size, **changes):
    return select(data, size, **(dict(world_size=4, capability=(12, 0), inplace=False) | changes))


@pytest.mark.parametrize("path,name", LADDERS)
def test_supplied_ladder_covers_every_interval(path, name):
    data = load(path)
    assert data["name"] == name
    assert len(data["buckets"]) == 210
    assert data["min_bytes"] == 16
    assert data["buckets"][-1]["max_bytes"] == 8 << 20
    lower = data["min_bytes"]
    for row in data["buckets"]:
        assert choose(data, lower) is row
        assert choose(data, row["max_bytes"]) is row
        assert sum(len(counts) for variant in row["variants"] for counts in variant["profiles"].values()) <= 4
        lower = row["max_bytes"] + 16


@pytest.mark.parametrize("path", [path for path, _ in LADDERS])
def test_packaged_measured_buckets_preserve_both_launch_kinds(path):
    for row in load(path)["buckets"][1:]:
        assert {variant["cooperative"] for variant in row["variants"]} == {False, True}
        assert sum(len(counts) for variant in row["variants"] for counts in variant["profiles"].values()) == 4


@pytest.mark.parametrize("size", [0, 8, 1025, (8 << 20) + 16, True])
def test_no_extrapolation_or_unaligned_payload(size):
    with pytest.raises(UsageError, match="no bucket"):
        choose(load(LADDER), size)


@pytest.mark.parametrize("changes", [dict(world_size=2), dict(capability=(12, 1)), dict(inplace=True)])
def test_rejects_incompatible_reduction(changes):
    with pytest.raises(UsageError, match="does not match"):
        choose(load(LADDER), 4096, **changes)


@pytest.mark.parametrize("changes", [dict(version=3), dict(extra=1), dict(world_size=1),
    dict(dtype="float16"), dict(inplace=0), dict(min_bytes=1025), dict(buckets=[]), dict(cache_policy="unknown")])
def test_invalid_contract(changes):
    with pytest.raises(UsageError, match="collective ladder"):
        validate(load(LADDER) | changes)


@pytest.mark.parametrize("changes", [dict(max_bytes=0), dict(variants=[]), dict(variants={})])
def test_invalid_bucket(changes):
    data = deepcopy(load(LADDER))
    data["buckets"][0].update(changes)
    with pytest.raises(UsageError, match="collective ladder"):
        validate(data)


@pytest.mark.parametrize("changes", [dict(implementation="oneshot"), dict(cooperative=1),
    dict(threads=96), dict(blocks=0), dict(profiles={}), dict(profiles={"full": [0]}),
    dict(profiles={"full": [65]}), dict(profiles={"full": [1, 1]}), dict(profiles={"bulk_push": [1]})])
def test_invalid_variant(changes):
    data = deepcopy(load(LADDER))
    data["buckets"][0]["variants"][0].update(changes)
    with pytest.raises(UsageError, match="collective ladder"):
        validate(data)


def test_unsupported_schema_requires_regeneration():
    with pytest.raises(UsageError, match="regenerate.*version 2"):
        validate(load(LADDER) | dict(version=1))


def test_variants_have_distinct_geometries_and_a_shared_candidate_budget():
    data = deepcopy(load(LADDER))
    row = data["buckets"][0]
    row["variants"].append(deepcopy(row["variants"][0]))
    with pytest.raises(UsageError, match="duplicate launch geometry"):
        validate(data)
    row["variants"][1]["threads"] = 128
    row["variants"][1]["profiles"] = {"full": [1, 2, 4, 8]}
    with pytest.raises(UsageError, match="at most four runtime candidates"):
        validate(data)
    row["variants"][1]["profiles"] = {"full": [1, 2, 4]}
    assert validate(data) is data


def test_runtime_requires_file_and_static_rejects_it(tmp_path):
    with pytest.raises(UsageError, match="requires --collective-ladder"):
        resolve("runtime", None)
    with pytest.raises(UsageError, match="requires --collective-tuning runtime"):
        resolve("static", LADDER)
    with pytest.raises(UsageError, match="cannot read"):
        resolve("runtime", tmp_path / "missing.json")
    assert resolve("static", None) is None
    assert resolve("runtime", LADDER) == load(LADDER)
    bad = tmp_path / "bad.json"
    bad.write_text("{broken")
    with pytest.raises(UsageError, match="cannot read"):
        load(bad)


@pytest.mark.parametrize("command", ["export", "debug-serve", "bench"])
def test_preparation_cli_accepts_ladder(command):
    from ignition.cli import _parser

    args = _parser().parse_args([command, "muse_spark", "--collective-tuning", "runtime", "--collective-ladder", str(LADDER)])
    assert args.collective_tuning == "runtime"
    assert args.collective_ladder == LADDER


@pytest.mark.parametrize("path", [path for path, _ in LADDERS])
@pytest.mark.parametrize("effort", ["off", "default", "audit"])
def test_b12x_accepts_pinned_geometry_at_both_ends_of_every_bucket(effort, path):
    from b12x.comm.pcie._collective_tuning import CollectiveConfig, CollectiveQuery, TUNING
    from b12x.comm.pcie._joint_profile import selected_profile
    from b12x.preparation import FrozenMapping, autotune_effort

    data = load(path)
    lower = data["min_bytes"]
    with autotune_effort(effort):
        for row in data["buckets"]:
            for variant in row["variants"]:
                family = variant["implementation"]
                mode, counts = next(iter(variant["profiles"].items()))
                config = CollectiveConfig(implementation=family, threads=variant["threads"], blocks=variant["blocks"],
                                          submode=mode, active_blocks=counts[0] if variant["cooperative"] else 1,
                                          cooperative=variant["cooperative"])
                for size in (lower, row["max_bytes"]):
                    query = CollectiveQuery(operation="all_reduce", world_size=4, rank=0,
                        alternatives=FrozenMapping({family: dict(surface="OneshotAllReduce.all_reduce",
                            world_size=4, rank=0, topology="pcie_ipc", setup={},
                            call=dict(threads=256, blocks=64, submode=config.submode,
                                      active_blocks=config.active_blocks, cooperative=config.cooperative,
                                      size_packs=size // 16, stage_input=True, dtype="bfloat16"))}),
                        default=config)
                    configured = TUNING.configure(query, device=None, override=config, search=effort != "off")
                    assert configured.pinned == config
                for mode, counts in variant["profiles"].items():
                    for count in counts:
                        assert selected_profile(family, variant["blocks"], mode, count,
                                                cooperative=variant["cooperative"]) & 0xffff == count
            lower = row["max_bytes"] + 16


def test_native_export_preserves_ladder_contents():
    import json
    from ignition.preparation.package_proto import serving
    from ignition.preparation.runtime_package_pb2 import ModelIndex
    from tests.preparation.test_frontend_metadata import server

    data = load(LADDER)
    data["buckets"][0]["variants"].append(dict(implementation="joint_bandwidth", threads=256,
        blocks=8, cooperative=True, profiles={"bulk_push_phased": [8]}))
    validate(data)
    source = server()
    source["rebuild_manifest"] = dict(version=1, collective_tuning="runtime",
                                      collective_ladder_json=json.dumps(data))
    index = ModelIndex()
    serving(index, source)
    restored = ModelIndex.FromString(index.SerializeToString())
    assert restored.rebuild.collective_tuning == "runtime"
    assert json.loads(restored.rebuild.collective_ladder_json) == data


@pytest.mark.parametrize("family,mode", [("joint_latency", "full_phased"),
    ("joint_latency", "payload_rsag_phased"), ("joint_bandwidth", "payload_rsag_phased"),
    ("joint_bandwidth", "bulk_push_phased")])
def test_phased_profile_admission_uses_b12x(family, mode):
    data = deepcopy(load(LADDER))
    row = data["buckets"][0]["variants"][0]
    row.update(implementation=family, profiles={mode: [1]})
    assert validate(data) is data
    row["profiles"][mode] = [2]
    with pytest.raises(UsageError, match="exactly one active CTA"):
        validate(data)
    row["profiles"][mode] = [1]
    data["world_size"] = 2
    with pytest.raises(UsageError, match="TP size"):
        validate(data)


@pytest.mark.parametrize("family,mode", [("joint_latency", "full_phased"),
    ("joint_latency", "payload_rsag_phased"), ("joint_bandwidth", "payload_rsag_phased"),
    ("joint_bandwidth", "bulk_push_phased")])
def test_cooperative_variants_require_phased_profiles_and_exact_physical_grids(family, mode):
    data = deepcopy(load(LADDER))
    row = data["buckets"][0]["variants"][0]
    row.update(implementation=family, blocks=8, cooperative=True, profiles={mode: [8]})
    assert validate(data) is data
    row["profiles"] = {mode: [4]}
    with pytest.raises(UsageError, match="physical and active CTA counts must match"):
        validate(data)
    row["profiles"] = {"pull": [8]}
    with pytest.raises(UsageError, match="require a phased protocol"):
        validate(data)
    row["profiles"] = {mode: [8]}
    data["world_size"] = 2
    with pytest.raises(UsageError, match="TP size"):
        validate(data)


def test_ladder_validation_does_not_import_cuda_libraries():
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c",
        "import sys; from ignition.preparation.collective_ladder import load; "
        "load(sys.argv[1]); assert 'torch' not in sys.modules; assert 'cuda' not in sys.modules",
        str(LADDER)], check=True)
