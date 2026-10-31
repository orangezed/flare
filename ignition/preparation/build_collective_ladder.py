"""Combine measured PCIe cohorts into bounded runtime launch-variant ladders."""

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
    from b12x.comm.pcie.joint_profiles import JointLaunchVariant
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
    if not isinstance(metadata["sizes"], list) or any(
            type(size) is not int or size < 1024 or size % 16 for size in metadata["sizes"]):
        raise ValueError(f"{name}: measured sizes must be integers of at least 1024 bytes, aligned to 16 bytes")
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
        geometry = (config["implementation"], config["threads"], config["blocks"], config.get("cooperative", False))
        profile = (config["submode"], config["active_blocks"])
        JointLaunchVariant(*geometry[:3], profiles=(profile,), cooperative=geometry[3]).validate(contract["world_size"])
        profiles = timings.setdefault(size, {}).setdefault(geometry, {})
        if profile in profiles:
            raise ValueError(f"{name}: duplicate measured launch configuration at {size} bytes")
        profiles[profile] = latency
    if any(size not in timings for size in sizes):
        raise ValueError(f"{name}: every declared size requires correct measured configurations")
    return Cohort(name, switched, Path(path), contract, sizes, timings)


def profile_menu(timings, *, limit=4):
    """Find the smallest measured candidate set covering all cohorts at 1% or 5%."""
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
    raise ValueError(f"no set of at most {limit} measured candidates covers every cohort within 5%")


def launch_menu(timings, *, limit=4):
    """Cover all cohorts, retaining both launch classes when the budget permits."""
    menu, tolerance = profile_menu(timings, limit=limit)
    classes = [
        {candidate: latency for candidate, latency in row.items() if candidate[3] == cooperative}
        for row in timings for cooperative in (False, True)
        if any(candidate[3] == cooperative for candidate in row)
    ]
    try:
        expanded, _ = profile_menu(classes, limit=limit)
    except ValueError:
        expanded = ()
    if expanded and all(min(row.get(candidate, float("inf")) for candidate in expanded)
                        <= min(row.values()) * (1 + tolerance) for row in timings):
        return expanded, tolerance
    preferred = {}
    for row in classes:
        candidate = min(row, key=lambda candidate: (row[candidate], candidate))
        preferred[candidate] = preferred.get(candidate, 0) + 1
    additions = sorted(set(preferred) - set(menu), key=lambda candidate: (-preferred[candidate], candidate))
    return (*menu, *additions[:max(0, limit - len(menu))]), tolerance


def mode_finalists(menu, timings, *, limit=4):
    """Fill unused slots with measured representatives of distinct additional modes."""
    covered = {candidate[-2] for candidate in menu}
    best = [min(rows.values()) for rows in timings]
    shared = set.intersection(*(set(rows) for rows in timings))
    winners = {}
    for profile in shared:
        mode = profile[-2]
        if mode in covered:
            continue
        score = max(rows[profile] / fastest for rows, fastest in zip(timings, best))
        candidate = (score, profile)
        if mode not in winners or candidate < winners[mode]:
            winners[mode] = candidate
    additions = tuple(profile for _, profile in sorted(winners.values()))[:max(0, limit - len(menu))]
    return (*menu, *additions)


def combine(cohorts, *, name, retain_mode_finalists=False):
    """Retain measured launch candidates covering each supplied timing cohort."""
    if not cohorts:
        raise ValueError("at least one measured report is required")
    if len({c.name for c in cohorts}) != len(cohorts):
        raise ValueError("cohort names must be distinct")
    first = cohorts[0]
    if any(c.contract != first.contract or c.sizes != first.sizes for c in cohorts[1:]):
        raise ValueError("reports must agree on GPU capability, TP size, dtype, aliasing, cache policy, and payload sizes")
    buckets = [dict(max_bytes=1008, variants=[dict(implementation="joint_latency", threads=64,
                    blocks=64, cooperative=False, profiles={"full": [1]})])]
    measurements = []
    for size in first.sizes:
        rows = [{(*geometry, *profile): latency
                 for geometry, profiles in cohort.timings[size].items() for profile, latency in profiles.items()}
                for cohort in cohorts]
        best = [min(row.values()) for row in rows]
        try:
            menu, tolerance = launch_menu(rows)
        except ValueError as error:
            raise ValueError(f"{size} bytes: {error}") from error
        if retain_mode_finalists:
            menu = mode_finalists(menu, rows)
        variants = {}
        for family, threads, blocks, cooperative, mode, active in menu:
            variant = variants.setdefault((family, threads, blocks, cooperative), dict(
                implementation=family, threads=threads, blocks=blocks, cooperative=cooperative, profiles={}))
            variant["profiles"].setdefault(mode, []).append(active)
        bucket = dict(max_bytes=size, variants=list(variants.values()))
        buckets.append(bucket)
        measurements.append(dict(bytes=size, profile_tolerance=tolerance, variants=bucket["variants"],
            cohorts={cohort.name: dict(best_us=fastest,
                menu_best_us=min(row.get(candidate, float("inf")) for candidate in menu),
                candidate_latencies=[dict(implementation=candidate[0], threads=candidate[1],
                    blocks=candidate[2], cooperative=candidate[3], submode=candidate[4],
                    active_blocks=candidate[5], median_us=row.get(candidate)) for candidate in menu])
                for cohort, row, fastest in zip(cohorts, rows, best)}))
    ladder = validate(dict(version=2, name=name, **first.contract, min_bytes=16, buckets=buckets))
    return ladder, measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--switched", metavar="NAME=REPORT", action="append", default=[])
    parser.add_argument("--host-bridge", metavar="NAME=REPORT", action="append", default=[])
    parser.add_argument("--name", required=True)
    parser.add_argument("--retain-mode-finalists", action="store_true",
                        help="fill unused candidate slots with measured winners of other modes, up to four candidates")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True, help="JSON comparison of selected launch variants against each report")
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
        candidate_policy="cohort coverage with ordinary and cooperative finalists",
        retain_mode_finalists=args.retain_mode_finalists,
        cohorts=[dict(name=c.name, switched=c.switched, report=str(c.path)) for c in cohorts],
        control_bucket=dict(min_bytes=16, max_bytes=1008, performance_measured=False),
        buckets=measurements)
    for path, content in ((args.output, ladder), (args.measurements, comparison)):
        with path.open("x") as stream:
            stream.write(json.dumps(content, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
