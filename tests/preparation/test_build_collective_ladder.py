"""Shared geometry preserves per-report performance objectives and measured coverage."""

import json
from pathlib import Path

import pytest

from ignition.preparation.build_collective_ladder import Cohort, combine, mode_finalists, profile_menu, read_cohort

LATENCY = ("joint_latency", 64, 64)
BANDWIDTH = ("joint_bandwidth", 256, 64)
CONTRACT = dict(compute_capability=[12, 0], world_size=4, dtype="bfloat16", inplace=False, cache_policy="cold")


def cohort(name, switched, timings):
    return Cohort(name, switched, Path(name), CONTRACT, (1024,), {1024: timings})


def test_switched_tolerance_then_bridge_regret_and_distinct_runtime_winners():
    switched = cohort("switched", True, {
        LATENCY: {("full", 1): 1.0}, BANDWIDTH: {("payload_rsag", 8): 1.01}})
    bridge = cohort("bridge", False, {
        LATENCY: {("pull", 1): 15.0}, BANDWIDTH: {("bulk_push_phased", 1): 3.0}})
    ladder, results = combine([switched, bridge], name="measured")
    assert ladder["buckets"][0]["max_bytes"] == 1008
    assert ladder["buckets"][0]["profiles"] == {"full": [1]}
    assert ladder["buckets"][1]["implementation"] == "joint_bandwidth"
    assert ladder["buckets"][1]["profiles"] == {"bulk_push_phased": [1], "payload_rsag": [8]}
    assert results[0]["cohorts"]["bridge"]["menu_best_us"] == 3.0


def test_no_geometry_within_tolerance_minimizes_worst_switched_penalty():
    cohorts = [
        cohort("switched-a", True, {LATENCY: {("full", 1): 1}, BANDWIDTH: {("pull", 1): 1.10}}),
        cohort("switched-b", True, {LATENCY: {("full", 1): 1.20}, BANDWIDTH: {("pull", 1): 1}}),
        cohort("bridge", False, {LATENCY: {("full", 1): 1}, BANDWIDTH: {("pull", 1): 10}}),
    ]
    ladder, _ = combine(cohorts, name="measured")
    assert ladder["buckets"][1]["implementation"] == "joint_bandwidth"


def test_menu_uses_measured_profiles_and_relaxes_only_to_five_percent():
    rows = [{("full", i + 1): 1, ("pull", 1): 1.03} for i in range(5)]
    menu, tolerance = profile_menu(rows)
    assert menu == (("pull", 1),)
    assert tolerance == 0.05
    with pytest.raises(ValueError, match="at most 4"):
        profile_menu([{("full", i + 1): 1} for i in range(5)])


def test_geometry_requires_measurements_in_every_cohort():
    with pytest.raises(ValueError, match="1024 bytes: no compiled geometry"):
        combine([cohort("switched", True, {LATENCY: {("full", 1): 1}}),
                 cohort("bridge", False, {BANDWIDTH: {("pull", 1): 1}})], name="missing")


@pytest.mark.parametrize("switched", (False, True))
def test_single_fabric_selects_its_best_geometry_without_a_second_cohort(switched):
    source = cohort("measured-fabric", switched, {
        LATENCY: {("full", 1): 3.0}, BANDWIDTH: {("bulk_push_phased", 1): 1.0}})
    ladder, measurements = combine([source], name="single-fabric")
    assert ladder["buckets"][1]["implementation"] == "joint_bandwidth"
    assert ladder["buckets"][1]["profiles"] == {"bulk_push_phased": [1]}
    assert set(measurements[0]["cohorts"]) == {"measured-fabric"}


@pytest.mark.parametrize("switched", (False, True))
def test_single_fabric_multiple_cohorts_minimize_worst_normalized_cost(switched):
    sources = [
        cohort("fabric-a", switched, {LATENCY: {("full", 1): 1}, BANDWIDTH: {("pull", 1): 1.1}}),
        cohort("fabric-b", switched, {LATENCY: {("full", 1): 120}, BANDWIDTH: {("pull", 1): 100}}),
    ]
    ladder, _ = combine(sources, name="single-fabric")
    assert ladder["buckets"][1]["implementation"] == "joint_bandwidth"


def test_combiner_requires_a_real_cohort():
    with pytest.raises(ValueError, match="at least one measured report"):
        combine([], name="empty")


def test_mode_finalists_preserve_coverage_and_add_distinct_modes():
    rows = [{("full", 1): 1.0, ("full", 2): 0.995, ("pull", 4): 1.2,
             ("payload_rsag", 8): 1.3, ("full_phased", 1): 1.4, ("payload_rsag_phased", 1): 2.0}]
    cover, tolerance = profile_menu(rows)
    menu = mode_finalists(cover, rows)
    assert cover == (("full", 1),)
    assert menu[:len(cover)] == cover
    assert menu == (("full", 1), ("pull", 4), ("payload_rsag", 8), ("full_phased", 1))
    assert len({mode for mode, _ in menu}) == 4
    assert min(rows[0][profile] for profile in menu) <= min(rows[0].values()) * (1 + tolerance)


def test_mode_finalists_rank_normalized_cost_and_require_measurements_in_every_cohort():
    rows = [
        {("full", 1): 1, ("pull", 1): 1.1, ("pull", 2): 1.3, ("full_phased", 1): 1.01},
        {("full", 1): 100, ("pull", 1): 300, ("pull", 2): 140},
    ]
    assert mode_finalists((("full", 1),), rows) == (("full", 1), ("pull", 2))


def test_retained_finalists_do_not_force_slower_phased_modes():
    source = cohort("host-bridge", False, {BANDWIDTH: {
        ("payload_rsag", 1): 1, ("bulk_push", 4): 1.1, ("bulk_pull", 4): 1.2,
        ("pull", 8): 1.3, ("payload_rsag_phased", 1): 10, ("bulk_push_phased", 1): 20}})
    default, _ = combine([source], name="measured")
    retained, _ = combine([source], name="measured", retain_mode_finalists=True)
    assert default["buckets"][1]["profiles"] == {"payload_rsag": [1]}
    assert retained["buckets"][1]["profiles"] == {
        "payload_rsag": [1], "bulk_push": [4], "bulk_pull": [4], "pull": [8]}


def test_mode_finalists_do_not_expand_a_full_coverage_menu():
    cover = tuple(("full", active) for active in (1, 2, 4, 8))
    rows = [{**dict.fromkeys(cover, 1), ("pull", 1): 0.9}]
    assert mode_finalists(cover, rows) == cover


def report_rows():
    from b12x.tools._collective_ladder_sampling import SAMPLES_PER_ROUND

    config = dict(implementation="joint_latency", threads=64, blocks=64, submode="full", active_blocks=1)
    metadata = dict(dtype="bfloat16", alias=False, cache_policy="cold", sizes=[1024],
        catalogs=[dict(bytes=1024, configs=[config, config | {"active_blocks": 2}])],
        hardware=dict(devices=[dict(compute_capability=[12, 0])] * 4))
    rows = [dict(kind="metadata", schema_version=1, metadata=metadata)]
    for active, samples in [(1, [1, 9]), (1, [5]), (2, [10])]:
        rows.append(dict(kind="measurement", bytes=1024, config=config | {"active_blocks": active},
                         samples_us=samples, correctness="passed", phase="coverage"))
    for active, latency in [(1, 100), (2, 1)]:
        rows.append(dict(kind="measurement", bytes=1024, config=config | {"active_blocks": active},
            samples_us=[latency] * SAMPLES_PER_ROUND, correctness="passed",
            phase="final", batch=1, batch_size=2, rounds=1))
    rows.append(dict(kind="summary", summary=dict(status="complete", missing=[], failures=[])))
    return rows


def test_read_report_uses_screening_medians_and_keeps_final_batch_separate(tmp_path):
    path = tmp_path / "cohort.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in report_rows()))
    result = read_cohort("report", path, switched=True)
    assert result.timings == {1024: {LATENCY: {("full", 1): 5, ("full", 2): 10}}}


@pytest.mark.parametrize("mutation", ("failure", "incomplete_confirmation", "wrong_catalog"))
def test_declared_complete_summary_requires_matching_measurements(tmp_path, mutation):
    path = tmp_path / "cohort.jsonl"
    rows = report_rows()
    if mutation == "failure":
        rows[1]["correctness"] = False
    elif mutation == "incomplete_confirmation":
        del rows[-2]
    else:
        rows[0]["metadata"]["catalogs"][0]["bytes"] = 2048
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="coverage|catalog"):
        read_cohort("report", path, switched=True)


def test_report_cohorts_are_not_pooled_across_timing_scales():
    a = cohort("switched", True, {LATENCY: {("full", 1): 1}, BANDWIDTH: {("pull", 1): 1.01}})
    b = cohort("bridge", False, {LATENCY: {("full", 1): 1000}, BANDWIDTH: {("pull", 1): 500}})
    ladder, results = combine([a, b], name="separate")
    assert ladder["buckets"][1]["implementation"] == "joint_bandwidth"
    assert results[0]["cohorts"]["switched"]["best_us"] == 1
    assert results[0]["cohorts"]["bridge"]["best_us"] == 500
