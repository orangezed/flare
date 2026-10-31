"""Measured launch menus preserve cohort coverage across ordinary and cooperative grids."""

import json
from pathlib import Path

import pytest

from ignition.preparation.build_collective_ladder import Cohort, combine, mode_finalists, profile_menu, read_cohort

LATENCY = ("joint_latency", 64, 64, False)
BANDWIDTH = ("joint_bandwidth", 256, 64, False)
COOPERATIVE = ("joint_bandwidth", 256, 8, True)
CONTRACT = dict(compute_capability=[12, 0], world_size=4, dtype="bfloat16", inplace=False, cache_policy="cold")


def cohort(name, switched, timings):
    return Cohort(name, switched, Path(name), CONTRACT, (1024,), {1024: timings})


def candidates(bucket):
    return {(*[variant[field] for field in ("implementation", "threads", "blocks", "cooperative")], mode, active)
            for variant in bucket["variants"] for mode, counts in variant["profiles"].items() for active in counts}


def test_distinct_fabrics_retain_their_fastest_geometry():
    switched = cohort("switched", True, {
        LATENCY: {("full", 1): 1.0}, BANDWIDTH: {("payload_rsag", 8): 2.0}})
    bridge = cohort("bridge", False, {
        LATENCY: {("full", 1): 15.0}, BANDWIDTH: {("payload_rsag", 8): 3.0}})
    ladder, results = combine([switched, bridge], name="measured")
    assert ladder["version"] == 2
    assert ladder["buckets"][0]["max_bytes"] == 1008
    assert candidates(ladder["buckets"][0]) == {(*LATENCY, "full", 1)}
    assert candidates(ladder["buckets"][1]) == {(*LATENCY, "full", 1), (*BANDWIDTH, "payload_rsag", 8)}
    assert results[0]["cohorts"]["bridge"]["menu_best_us"] == 3.0
    assert results[0]["cohorts"]["switched"]["menu_best_us"] == 1.0


def test_best_ordinary_and_cooperative_choices_survive_for_both_fabrics():
    other_cooperative = ("joint_latency", 128, 4, True)
    sources = [
        cohort("switched", True, {
            LATENCY: {("full", 1): 1}, BANDWIDTH: {("pull", 8): 2},
            COOPERATIVE: {("bulk_push_phased", 8): 1.5}, other_cooperative: {("full_phased", 4): 4}}),
        cohort("bridge", False, {
            LATENCY: {("full", 1): 10}, BANDWIDTH: {("pull", 8): 3},
            COOPERATIVE: {("bulk_push_phased", 8): 8}, other_cooperative: {("full_phased", 4): 1}}),
    ]
    ladder, results = combine(sources, name="portable")
    assert candidates(ladder["buckets"][1]) == {
        (*LATENCY, "full", 1), (*BANDWIDTH, "pull", 8),
        (*COOPERATIVE, "bulk_push_phased", 8), (*other_cooperative, "full_phased", 4)}
    assert all(row["menu_best_us"] == row["best_us"] for row in results[0]["cohorts"].values())


def test_missing_measurements_do_not_establish_candidate_coverage():
    ladder, results = combine([
        cohort("switched", True, {LATENCY: {("full", 1): 1}}),
        cohort("bridge", False, {COOPERATIVE: {("bulk_push_phased", 8): 2}})], name="measured")
    assert candidates(ladder["buckets"][1]) == {(*LATENCY, "full", 1), (*COOPERATIVE, "bulk_push_phased", 8)}
    for row in results[0]["cohorts"].values():
        assert sum(candidate["median_us"] is None for candidate in row["candidate_latencies"]) == 1
        assert row["menu_best_us"] == row["best_us"]


def test_menu_relaxes_only_to_five_percent_and_rejects_uncoverable_cohorts():
    rows = [{("full", i + 1): 1, ("pull", 1): 1.03} for i in range(5)]
    menu, tolerance = profile_menu(rows)
    assert menu == (("pull", 1),)
    assert tolerance == 0.05
    with pytest.raises(ValueError, match="at most 4"):
        profile_menu([{("full", i + 1): 1} for i in range(5)])


def test_four_candidate_limit_does_not_drop_required_cohort_winners():
    sources = [cohort(f"fabric-{i}", True, {LATENCY: {("full", i + 1): 1},
        COOPERATIVE: {("bulk_push_phased", 8): 3}}) for i in range(4)]
    ladder, results = combine(sources, name="coverage")
    assert candidates(ladder["buckets"][1]) == {(*LATENCY, "full", active) for active in range(1, 5)}
    assert all(row["menu_best_us"] == 1 for row in results[0]["cohorts"].values())
    sources.append(cohort("fabric-4", False, {LATENCY: {("full", 5): 1}}))
    with pytest.raises(ValueError, match="1024 bytes: no set of at most 4"):
        combine(sources, name="uncoverable")


@pytest.mark.parametrize("switched", (False, True))
def test_single_fabric_keeps_ordinary_choice_when_cooperative_is_faster(switched):
    source = cohort("measured-fabric", switched, {
        LATENCY: {("full", 1): 3.0}, COOPERATIVE: {("bulk_push_phased", 8): 1.0}})
    ladder, measurements = combine([source], name="single-fabric")
    assert candidates(ladder["buckets"][1]) == {(*LATENCY, "full", 1), (*COOPERATIVE, "bulk_push_phased", 8)}
    assert measurements[0]["cohorts"]["measured-fabric"]["menu_best_us"] == 1


def test_combiner_requires_compatible_distinct_cohorts():
    source = cohort("fabric", True, {LATENCY: {("full", 1): 1}})
    with pytest.raises(ValueError, match="at least one measured report"):
        combine([], name="empty")
    with pytest.raises(ValueError, match="names must be distinct"):
        combine([source, source], name="duplicate")
    other = Cohort("other", False, Path("other"), CONTRACT | dict(cache_policy="warm"), source.sizes, source.timings)
    with pytest.raises(ValueError, match="reports must agree"):
        combine([source, other], name="incompatible")


def test_sparse_reports_preserve_their_measured_endpoints():
    source = cohort("fabric", True, {LATENCY: {("full", 1): 1}})
    source = Cohort(source.name, source.switched, source.path, source.contract, (1024, 8192),
        {1024: source.timings[1024], 8192: source.timings[1024]})
    ladder, measurements = combine([source], name="sparse")
    assert [bucket["max_bytes"] for bucket in ladder["buckets"]] == [1008, 1024, 8192]
    assert [row["bytes"] for row in measurements] == [1024, 8192]


def test_mode_finalists_preserve_coverage_and_add_distinct_modes():
    rows = [{("full", 1): 1.0, ("full", 2): 0.995, ("pull", 4): 1.2,
             ("payload_rsag", 8): 1.3, ("full_phased", 1): 1.4, ("payload_rsag_phased", 1): 2.0}]
    cover, tolerance = profile_menu(rows)
    menu = mode_finalists(cover, rows)
    assert cover == (("full", 1),)
    assert menu == (("full", 1), ("pull", 4), ("payload_rsag", 8), ("full_phased", 1))
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
    assert candidates(default["buckets"][1]) == {(*BANDWIDTH, "payload_rsag", 1)}
    assert candidates(retained["buckets"][1]) == {
        (*BANDWIDTH, "payload_rsag", 1), (*BANDWIDTH, "bulk_push", 4),
        (*BANDWIDTH, "bulk_pull", 4), (*BANDWIDTH, "pull", 8)}


def test_mode_finalists_do_not_expand_a_full_coverage_menu():
    cover = tuple(("full", active) for active in (1, 2, 4, 8))
    assert mode_finalists(cover, [{**dict.fromkeys(cover, 1), ("pull", 1): 0.9}]) == cover


def report_rows(*, cooperative=False):
    from b12x.tools._collective_ladder_sampling import SAMPLES_PER_ROUND

    config = dict(implementation="joint_latency", threads=64, blocks=64, submode="full", active_blocks=1)
    configs = [config, config | {"active_blocks": 2}]
    if cooperative:
        configs.append(config | dict(blocks=8, cooperative=True, submode="full_phased", active_blocks=8))
    metadata = dict(dtype="bfloat16", alias=False, cache_policy="cold", sizes=[1024],
        catalogs=[dict(bytes=1024, configs=configs)],
        hardware=dict(devices=[dict(compute_capability=[12, 0])] * 4))
    rows = [dict(kind="metadata", schema_version=1, metadata=metadata)]
    for candidate, samples in [(configs[0], [1, 9]), (configs[0], [5]), (configs[1], [10])]:
        rows.append(dict(kind="measurement", bytes=1024, config=candidate,
                         samples_us=samples, correctness="passed", phase="coverage"))
    if cooperative:
        rows.append(dict(kind="measurement", bytes=1024, config=configs[2],
                         samples_us=[3], correctness="passed", phase="coverage"))
    for candidate, latency in zip(configs, [100, 1, 50]):
        rows.append(dict(kind="measurement", bytes=1024, config=candidate,
            samples_us=[latency] * SAMPLES_PER_ROUND, correctness="passed",
            phase="final", batch=1, batch_size=len(configs), rounds=1))
    rows.append(dict(kind="summary", summary=dict(status="complete", missing=[], failures=[])))
    return rows


def write_report(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_read_report_uses_screening_medians_and_keeps_final_batch_separate(tmp_path):
    path = write_report(tmp_path / "cohort.jsonl", report_rows())
    result = read_cohort("report", path, switched=True)
    assert result.timings == {1024: {LATENCY: {("full", 1): 5, ("full", 2): 10}}}


def test_read_report_preserves_cooperative_geometry(tmp_path):
    path = write_report(tmp_path / "cooperative.jsonl", report_rows(cooperative=True))
    result = read_cohort("report", path, switched=False)
    assert result.timings[1024][("joint_latency", 64, 8, True)] == {("full_phased", 8): 3}
    assert result.timings[1024][LATENCY] == {("full", 1): 5, ("full", 2): 10}


@pytest.mark.parametrize("mutation", ("failure", "incomplete_confirmation", "wrong_catalog", "invalid_launch"))
def test_declared_complete_summary_requires_valid_matching_measurements(tmp_path, mutation):
    rows = report_rows()
    if mutation == "failure":
        rows[1]["correctness"] = False
    elif mutation == "incomplete_confirmation":
        del rows[-2]
    elif mutation == "wrong_catalog":
        rows[0]["metadata"]["catalogs"][0]["bytes"] = 2048
    else:
        for config in rows[0]["metadata"]["catalogs"][0]["configs"]:
            config["cooperative"] = 1
    path = write_report(tmp_path / "cohort.jsonl", rows)
    with pytest.raises(ValueError, match="coverage|catalog|cooperative"):
        read_cohort("report", path, switched=True)


def test_report_cohorts_are_not_pooled_across_timing_scales():
    a = cohort("switched", True, {LATENCY: {("full", 1): 1}, BANDWIDTH: {("pull", 1): 3}})
    b = cohort("bridge", False, {LATENCY: {("full", 1): 1000}, BANDWIDTH: {("pull", 1): 500}})
    ladder, results = combine([a, b], name="separate")
    assert len(candidates(ladder["buckets"][1])) == 2
    assert results[0]["cohorts"]["switched"]["best_us"] == 1
    assert results[0]["cohorts"]["bridge"]["best_us"] == 500
