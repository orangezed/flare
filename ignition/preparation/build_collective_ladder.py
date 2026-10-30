"""Combine measured PCIe cohorts into a fixed-geometry runtime profile ladder."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from ignition.errors import UsageError

from .collective_ladder import validate


@dataclass(frozen=True)
class Cohort:
    name: str
    switched: bool
    path: Path
    contract: dict
    sizes: tuple[int, ...]
    timings: dict


def read_cohort(name, path, *, switched):
    """Restore b12x screening medians without pooling confirmation batches."""
    from b12x.comm.pcie.joint_profiles import selected_profile
    from b12x.tools._collective_ladder_search import Search, read_report

    metadata, records, summaries = read_report(path)
    if (not summaries or summaries[-1].get("status") != "complete"
            or summaries[-1].get("missing") or summaries[-1].get("missing_profiles")
            or summaries[-1].get("failures")):
        raise ValueError(f"{name}: report requires a completed summary without missing sizes or failures")
    devices = metadata["hardware"]["devices"]
    capabilities = {tuple(device["compute_capability"]) for device in devices}
    if len(capabilities) != 1:
        raise ValueError(f"{name}: all devices must have the same compute capability")
    contract = dict(compute_capability=list(capabilities.pop()), world_size=len(devices),
                    dtype=metadata["dtype"], inplace=metadata["alias"], cache_policy=metadata["cache_policy"])
    sizes = tuple(sorted(metadata["sizes"]))
    if not sizes or len(sizes) != len(set(sizes)) or sizes[0] != 1024:
        raise ValueError(f"{name}: distinct measured sizes must start at 1024 bytes")
    catalogs = {row["bytes"]: row["configs"] for row in metadata["catalogs"]}
    if tuple(sorted(catalogs)) != sizes or len(catalogs) != len(metadata["catalogs"]):
        raise ValueError(f"{name}: the measurement catalog must cover every declared payload exactly once")
    search = Search(catalogs, deadline=0, records=records)
    if search.summary()["status"] != "complete":
        raise ValueError(f"{name}: measurements do not establish complete screening and confirmation coverage")
    timings = {}
    for (size, key), latency in search.latencies.items():
        config = search.configs[size, key]
        geometry = (config["implementation"], config["threads"], config["blocks"])
        profile = (config["submode"], config["active_blocks"])
        selected_profile(geometry[0], geometry[2], *profile, world=contract["world_size"])
        timings.setdefault(size, {}).setdefault(geometry, {})[profile] = latency
    if any(size not in timings for size in sizes):
        raise ValueError(f"{name}: every declared size requires correct measured configurations")
    return Cohort(name, switched, Path(path), contract, sizes, timings)


def profile_menu(timings, *, limit=4):
    """Find the smallest measured profile set covering all cohorts at 1% or 5%."""
    best = [min(rows.values()) for rows in timings]
    profiles = sorted(set().union(*(rows.keys() for rows in timings)))
    full = (1 << len(timings)) - 1
    for tolerance in (0.01, 0.05):
        masks = {}
        for profile in profiles:
            mask = sum(1 << index for index, rows in enumerate(timings)
                       if rows.get(profile, float("inf")) <= best[index] * (1 + tolerance))
            if mask:
                masks[profile] = mask
        covers = {0: ()}
        for profile, mask in masks.items():
            for covered, menu in tuple(covers.items()):
                candidate = (*menu, profile)
                if len(candidate) > limit:
                    continue
                combined = covered | mask
                existing = covers.get(combined)
                if existing is None or (len(candidate), candidate) < (len(existing), existing):
                    covers[combined] = candidate
        if full in covers:
            return covers[full], tolerance
    raise ValueError(f"no set of at most {limit} measured profiles covers every cohort within 5%")


def mode_finalists(menu, timings, *, limit=4):
    """Fill unused slots with measured representatives of distinct additional modes."""
    covered = {mode for mode, _ in menu}
    best = [min(rows.values()) for rows in timings]
    shared = set.intersection(*(set(rows) for rows in timings))
    winners = {}
    for profile in shared:
        mode, _ = profile
        if mode in covered:
            continue
        score = max(rows[profile] / fastest for rows, fastest in zip(timings, best))
        candidate = (score, profile)
        if mode not in winners or candidate < winners[mode]:
            winners[mode] = candidate
    additions = tuple(profile for _, profile in sorted(winners.values()))[:max(0, limit - len(menu))]
    return (*menu, *additions)


def combine(cohorts, *, name, retain_mode_finalists=False):
    """Select shared geometry with switched-first or single-fabric minimax scoring."""
    if not cohorts:
        raise ValueError("at least one measured report is required")
    if len({c.name for c in cohorts}) != len(cohorts):
        raise ValueError("cohort names must be distinct")
    first = cohorts[0]
    both_fabrics = len({cohort.switched for cohort in cohorts}) == 2
    if any(c.contract != first.contract or c.sizes != first.sizes for c in cohorts[1:]):
        raise ValueError("reports must agree on GPU capability, TP size, dtype, aliasing, cache policy, and payload sizes")
    buckets = [dict(max_bytes=1008, implementation="joint_latency", threads=64, blocks=64,
                    profiles={"full": [1]})]
    measurements = []
    for size in first.sizes:
        rows = [c.timings[size] for c in cohorts]
        shared = set.intersection(*(set(row) for row in rows))
        if not shared:
            raise ValueError(f"{size} bytes: no compiled geometry was measured in every cohort")
        best = [min(value for profiles in row.values() for value in profiles.values()) for row in rows]
        ratios = {geometry: [min(row[geometry].values()) / fastest for row, fastest in zip(rows, best)]
                  for geometry in shared}
        if both_fabrics:
            switched_penalty = lambda geometry: max(ratio for c, ratio in zip(cohorts, ratios[geometry]) if c.switched)
            bridge_penalty = lambda geometry: max(ratio for c, ratio in zip(cohorts, ratios[geometry]) if not c.switched)
            within = [geometry for geometry in shared if switched_penalty(geometry) <= 1.02]
            if within:
                geometry = min(within, key=lambda g: (bridge_penalty(g), switched_penalty(g), g))
            else:
                geometry = min(shared, key=lambda g: (switched_penalty(g), bridge_penalty(g), g))
        else:
            geometry = min(shared, key=lambda g: (max(ratios[g]), g))
        try:
            menu, tolerance = profile_menu([row[geometry] for row in rows])
        except ValueError as error:
            raise ValueError(f"{size} bytes, {geometry}: {error}") from error
        if retain_mode_finalists:
            menu = mode_finalists(menu, [row[geometry] for row in rows])
        profiles = {}
        for mode, active in menu:
            profiles.setdefault(mode, []).append(active)
        bucket = dict(max_bytes=size, implementation=geometry[0], threads=geometry[1], blocks=geometry[2], profiles=profiles)
        buckets.append(bucket)
        measurements.append(dict(bytes=size, profile_tolerance=tolerance, geometry=bucket,
            cohorts={cohort.name: dict(best_us=fastest, geometry_best_us=min(row[geometry].values()),
                menu_best_us=min(row[geometry].get(profile, float("inf")) for profile in menu))
                for cohort, row, fastest in zip(cohorts, rows, best)}))
    ladder = validate(dict(version=1, name=name, **first.contract, min_bytes=16, buckets=buckets))
    return ladder, measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--switched", metavar="NAME=REPORT", action="append", default=[])
    parser.add_argument("--host-bridge", metavar="NAME=REPORT", action="append", default=[])
    parser.add_argument("--name", required=True)
    parser.add_argument("--retain-mode-finalists", action="store_true",
                        help="fill unused profile slots with measured winners of other modes, up to four profiles")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True, help="JSON comparison of selected geometry and menu against each report")
    args = parser.parse_args()
    if args.output.resolve() == args.measurements.resolve() or args.output.exists() or args.measurements.exists():
        parser.error("output paths must be distinct and unused")
    try:
        cohorts = []
        for switched, reports in ((True, args.switched), (False, args.host_bridge)):
            for value in reports:
                name, separator, path = value.partition("=")
                if not separator or not name or not path:
                    raise ValueError("report arguments require NAME=REPORT")
                cohorts.append(read_cohort(name, Path(path), switched=switched))
        ladder, measurements = combine(cohorts, name=args.name, retain_mode_finalists=args.retain_mode_finalists)
    except (KeyError, OSError, ValueError, UsageError) as error:
        parser.error(str(error))
    comparison = dict(
        aggregation="b12x screening medians; successful confirmation and final batches remain separate",
        geometry_policy=("switched-first" if len({c.switched for c in cohorts}) == 2 else "single-fabric minimax"),
        retain_mode_finalists=args.retain_mode_finalists,
        cohorts=[dict(name=c.name, switched=c.switched, report=str(c.path)) for c in cohorts],
        control_bucket=dict(min_bytes=16, max_bytes=1008, performance_measured=False),
        buckets=measurements)
    for path, content in ((args.output, ladder), (args.measurements, comparison)):
        with path.open("x") as stream:
            stream.write(json.dumps(content, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
