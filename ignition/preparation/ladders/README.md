# Offline collective ladders

`--collective-tuning runtime --collective-ladder PATH` selects a joint PCIe
kernel's compiled geometry from a JSON file. Preparation compiles that geometry
without racing alternatives, regardless of `--autotune-effort`. Captured tuning
graphs select among the file's submode and active-CTA profiles when the package
loads. Serving replays the same compiled kernel with the selected profile.

From the repository root, add these options to `ignition export`:

```sh
--collective-tuning runtime \
--collective-ladder ignition/preparation/ladders/gb202_pcie_tp4.json
```

The option is required whenever the resolved collective policy is `runtime`,
including configuration-file and recipe defaults. Static policy rejects the
option. File contents travel with the preparation plan and package rebuild
settings; workers do not reopen the path.

## Version 1 format

```json
{
  "version": 1,
  "name": "example_tp4",
  "compute_capability": [12, 0],
  "world_size": 4,
  "dtype": "bfloat16",
  "inplace": false,
  "cache_policy": "cold",
  "min_bytes": 1024,
  "buckets": [
    {"max_bytes": 1048576, "implementation": "joint_bandwidth", "threads": 512, "blocks": 64,
     "profiles": {"bulk_push": [32], "bulk_pull": [8]}}
  ]
}
```

Payload bytes count the full per-rank input before reduction. Bounds must be
16-byte aligned. The first bucket includes `min_bytes` through `max_bytes`;
subsequent buckets include payloads above the preceding maximum through their
own maximum. Selection does not extrapolate outside those bounds.

Each bucket declares one kernel family, thread count, and physical CTA grid.
`joint_latency` permits `full`, `payload_rsag`, `pull`, `full_phased`, and
`payload_rsag_phased`; `joint_bandwidth` permits `payload_rsag`, `bulk_push`,
`bulk_pull`, `pull`, `payload_rsag_phased`, and `bulk_push_phased`. Phased modes
order peer destinations to limit simultaneous writes through a host bridge.
They require exactly one active CTA and at least three ranks. Other profiles
permit active CTA counts between one and the physical grid size. Several counts
may belong to the same submode. Ignition validates admission through b12x's
CPU-only `joint_profiles` catalog. `cache_policy` controls the captured profile
race's cache treatment and accepts `cold` or `warm`.

The ladder applies to eligible plain BF16 PCIe reductions. TP size, compute
capability, and input/output aliasing must match. Other collective transports
and fused operations retain their preparation policy. Runtime profile selection
cannot change the compiled family, threads, or physical grid.

## Regenerate from measured reports

Run the b12x ladder sweep separately on each physical fabric, with competing GPU
work and PCIe monitoring stopped. Keep each JSONL report as an independent timing
cohort. From an environment containing Ignition and b12x, combine completed reports:

```sh
python -m ignition.preparation.build_collective_ladder \
  --switched switched=/data/collectives/switched.jsonl \
  --host-bridge host-bridge=/data/collectives/host-bridge.jsonl \
  --name gb202_pcie_tp4 \
  --output /data/collectives/gb202_pcie_tp4.json \
  --measurements /data/collectives/geometry-comparison.json
```

Repeat `--switched NAME=REPORT` or `--host-bridge NAME=REPORT` to retain additional
cohorts. At least one report is required. Both output paths must be unused.
Reports must agree on payload sizes, GPU capability, rank count, dtype, buffer
aliasing, and cache policy. The tool
restores b12x's per-configuration screening medians, pooling repeated screening
samples only within one report and exact configuration. Successful confirmation
and final batches remain separate; they establish the report's confirmed winner
and are not mixed into screening medians. Geometry and menu comparisons use
screening medians because the final batches do not cover every geometry.
Timings from different reports are never pooled.

To prepare a ladder for one fabric class, supply only that class's reports:

```sh
python -m ignition.preparation.build_collective_ladder \
  --host-bridge host-bridge=/data/collectives/host-bridge.jsonl \
  --name gb202_pcie_tp4_host_bridge \
  --output /data/collectives/gb202_pcie_tp4_host_bridge.json \
  --measurements /data/collectives/host-bridge-comparison.json
```

With only one fabric class, geometry minimizes the worst relative latency across
the supplied cohorts. A single report therefore selects its best measured
geometry, with one profile per bucket. Such a ladder has no measured basis for
adaptation between switched and host-bridge paths. Select it explicitly with
`--collective-ladder`; generating it does not change any packaged default.

`--retain-mode-finalists` is disabled by default. Enable it when runtime tuning
should compare additional protocols within the selected geometry:

```sh
python -m ignition.preparation.build_collective_ladder \
  --host-bridge host-bridge=/data/collectives/host-bridge.jsonl \
  --retain-mode-finalists --name gb202_pcie_tp4_host_bridge_candidates \
  --output /data/collectives/gb202_pcie_tp4_host_bridge_candidates.json \
  --measurements /data/collectives/host-bridge-candidates-comparison.json
```

This option retains the minimum coverage menu and fills unused slots, up to four
profiles, with the best measured representatives of other modes. Additional
profiles must have measurements in every supplied cohort. Representatives are
ranked by their worst latency relative to each cohort's best profile in the
selected geometry; mode names receive no preference. The added runtime choices
do not change the compiled geometry or establish coverage of an unmeasured
fabric. For a host-bridge-only input, these are host-bridge geometry choices with
runtime protocol alternatives, without switched-fabric qualification.

At every measured payload, the tool considers only geometries measured in all
cohorts. When both fabric classes are supplied, it first keeps geometries within
2% of every switched cohort's best latency, then minimizes the worst host-bridge
relative latency. If no geometry
meets 2%, it minimizes the worst switched penalty before the host-bridge penalty.
Each selected geometry gets the smallest menu covering every cohort within 1%
of its best measured profile; the tolerance becomes 5% only if four profiles
cannot meet 1%. Missing measurements never establish profile coverage. A bucket
requiring more than four profiles at 5% fails with its payload and geometry.

The generated 16–1008-byte control bucket uses one active CTA of the 64-thread
latency kernel in `full` mode. Its inclusion provides control-reduction coverage,
without a performance ranking. All other recommendations concern measured
endpoints; bucket interiors require separate validation. The comparison JSON
records each report's best latency and the selected geometry/menu latencies.

## GB202 PCIe TP4 recommendations

Both packaged ladders cover out-of-place BF16 reductions on four SM120 GPUs,
with 210 buckets from 16 bytes through 8 MiB. Choose the file explicitly:

- `gb202_pcie_tp4.json` shares geometry across switched and host-bridge reports,
  using the switched-first policy above. Each measured bucket has one or two
  runtime profiles. This prioritizes measured switched latency at a potentially
  substantial host-bridge cost.
- `gb202_pcie_tp4_host_bridge.json` chooses geometry from the host-bridge report
  and retains four measured mode representatives per measured bucket. Startup
  tuning chooses among those profiles within the fixed host-bridge geometry.
  The file does not establish switched-fabric performance or adapt its geometry
  when moved to a switched deployment.

For a four-rank host-bridge deployment, export with the host-bridge ladder:

```sh
.venv/bin/ignition export deepseek-v4.1-flash --gpus 0,4,8,12 \
  --max-concurrency 32 --mtp-draft-tokens 7 \
  --speculative-token-capacity 128 --speculative-graph-budget 64 \
  --prefill-chunk-size 4096 --expert-scale-compression csf --compile-workers 24 \
  --collective-tuning runtime \
  --collective-ladder ignition/preparation/ladders/gb202_pcie_tp4_host_bridge.json
```

The 16–1008-byte control bucket in both files uses one active CTA of the
64-thread latency kernel in `full` mode. It has correctness coverage without
measured performance ranking. The remaining 209 buckets cover 1 KiB through
8 MiB at 16 intervals per doubling.

The input measurements were collected on 2026-10-30 on a host named `frank2`,
using RTX PRO 6000 Blackwell Workstation GPUs: the physically switched group
12,13,14,15 and the host-bridge group 0,4,8,12. The switched classification follows
physical wiring; the operating system's topology report does not describe that
wiring correctly. Both runs used out-of-place BF16, cold-cache timing protocol 3,
CUDA 13.2, PyTorch 2.14.1, and driver 615.71.10. The measured kernel implementation
is FlashInfer `c4e1749c9bf8f9c5f5fa6f1bc6449b2c421ea479` (joint program version 218).
GPU memory offsets were +6000 MHz, with 16,365 MHz observed under load;
SM clocks were not locked. The raw directory contains
clock snapshots, source configuration, logs, and plots:

```text
frank2:/data/collective-ladder-frank2-phased-20261030/
  ladder-frank2-hb-0-4-8-12.jsonl
  ladder-frank2-sw-12-13-14-15-confirmed.jsonl
```

The host-bridge input completed its 600-second sweep. The switched input includes
supplemental confirmation after its 600-second sweep exhausted the final-check
budget; the original `ladder-frank2-sw-12-13-14-15.jsonl` is retained separately.
Both selected inputs have complete coverage and confirmation for all 209 sizes,
without failed configurations. Screening medians determine the combined
geometries and menus; the final confirmation batches remain separate as described
above.

At measured endpoints, the menu's screening penalty against each cohort's best
measured configuration stays below these bounds:

| Ladder | Switched | Host bridge |
|---|---:|---:|
| Shared geometry | 2.22% | 33.21% |
| Host-bridge geometry with mode finalists | Not qualified | 0.90% |

These comparisons concern measured endpoints, not complete-model performance.
Bucket interiors, other hosts, and other topologies have no measured performance
bound. The host-bridge penalty in the shared ladder is principally the cost of
sharing a geometry; its menus still cover each cohort within 1% of that selected
geometry's best profile.

The 2026-10-28 Frank2 reports are excluded because `pcie-watch` was active and
inflated peer-read tails. The selected reports were collected with `pcie-watch`
and `gpu-switch-exporter` disabled and a read-stall probe reporting `NONE` before
collection. Pulsar reports are not inputs to these files, so neither ladder
claims Pulsar qualification.
