# Offline collective ladders

`--collective-tuning runtime --collective-ladder PATH` supplies prepared PCIe
collective launch variants from a JSON file. Preparation compiles the declared
variants without racing alternatives, regardless of `--autotune-effort`.
Captured tuning graphs select a launch variant and protocol when the package
loads. Serving retains that selection for graph replay.

From the repository root, add these options to `ignition export`:

```sh
--collective-tuning runtime \
--collective-ladder ignition/preparation/ladders/gb202_pcie_tp4.json
```

The option is required whenever the resolved collective policy is `runtime`,
including configuration-file and recipe defaults. Static policy rejects the
option. File contents travel with the preparation plan and package rebuild
settings; workers do not reopen the path.

## Version 2 format

```json
{
  "version": 2,
  "name": "example_tp4",
  "compute_capability": [12, 0],
  "world_size": 4,
  "dtype": "bfloat16",
  "inplace": false,
  "cache_policy": "cold",
  "min_bytes": 1024,
  "buckets": [
    {"max_bytes": 1048576, "variants": [
      {"implementation": "joint_bandwidth", "threads": 512, "blocks": 64,
       "cooperative": false, "profiles": {"bulk_push": [32], "bulk_pull": [8]}},
      {"implementation": "joint_bandwidth", "threads": 256, "blocks": 8,
       "cooperative": true, "profiles": {"bulk_push_phased": [8]}}
    ]}
  ]
}
```

Payload bytes count the full per-rank input before reduction. Bounds must be
16-byte aligned. The first bucket includes `min_bytes` through `max_bytes`;
subsequent buckets include payloads above the preceding maximum through their
own maximum. Selection does not extrapolate outside those bounds. Version 1
files require regeneration; they are rejected before preparation.

Each variant declares one kernel family, thread count, physical CTA grid, and
launch kind. Geometries within a bucket must be distinct, counting `cooperative`
as part of the geometry. A bucket permits at most four protocol/active-CTA
candidates in total across its variants. Several active counts may belong to one
protocol in an ordinary variant.

`joint_latency` permits `full`, `payload_rsag`, `pull`, `full_phased`, and
`payload_rsag_phased`; `joint_bandwidth` permits `payload_rsag`, `bulk_push`,
`bulk_pull`, `pull`, `payload_rsag_phased`, and `bulk_push_phased`. Phased modes
order peer destinations to limit simultaneous writes through a host bridge and
require at least three ranks. An ordinary phased launch requires exactly one
active CTA. Ordinary nonphased profiles permit active counts between one and the
physical grid size. A cooperative variant admits only phased protocols and its
active count must equal its physical grid, from one through 64 CTAs. Ignition
validates these constraints through b12x's CPU-only `joint_profiles` catalog.
Cooperative launch admission also requires the grid to fit resident capacity.
`cache_policy` controls the captured race's cache treatment and accepts `cold`
or `warm`.

The ladder applies to eligible plain BF16 PCIe reductions. TP size, compute
capability, and input/output aliasing must match. Other collective transports
and fused operations retain their preparation policy. Ordinary and cooperative
choices compete by measured startup latency; protocol names and fabric labels
do not select runtime policy. No kernels are compiled or recaptured at load time.

## Regenerate from measured reports

Run the b12x ladder sweep separately on each physical fabric, with competing GPU
work and PCIe monitoring stopped. Include `--cooperative-phased` in the b12x sweep
to measure cooperative grids alongside the ordinary candidates. Keep each JSONL
report as an independent timing cohort. From an environment containing Ignition
and b12x, combine completed reports:

```sh
python -m ignition.preparation.build_collective_ladder \
  --switched switched=/data/collectives/switched.jsonl \
  --host-bridge host-bridge=/data/collectives/host-bridge.jsonl \
  --name gb202_pcie_tp4 \
  --output /data/collectives/gb202_pcie_tp4.json \
  --measurements /data/collectives/launch-comparison.json
```

Repeat `--switched NAME=REPORT` or `--host-bridge NAME=REPORT` to retain additional
cohorts. At least one report is required. Both output paths must be unused.
Reports must agree on payload sizes, GPU capability, rank count, dtype, buffer
aliasing, and cache policy. The tool requires a complete summary and verifies
screening and confirmation coverage from the raw measurements. Reports with
failed configurations are rejected.

The tool restores b12x's per-configuration screening medians, pooling repeated
screening samples only within one report and exact configuration. Cooperative
launch kind is part of that identity. Successful confirmation and final batches
remain separate; they establish the report's confirmed winner and are not mixed
into screening medians. Candidate comparisons use screening medians because
final batches do not cover every candidate. Timings from different reports are
never pooled.

At every measured payload, the tool selects the smallest candidate menu covering
every cohort within 1% of its best measured latency across all geometries. It
relaxes to 5% only when four candidates cannot meet 1%. Within that coverage
budget it retains ordinary and cooperative winners for each cohort when their
combined menu fits. Missing measurements never establish coverage. If four
candidates cannot cover every cohort within 5%, combination fails for that
payload. A candidate measured in only one cohort can cover that cohort; the
comparison file records absent measurements as `null` for other cohorts.

`--retain-mode-finalists` is disabled by default. Enable it to fill otherwise
unused slots, up to four total candidates, with measured representatives of
additional protocols. These additions must have measurements in every supplied
cohort. Representatives are ranked by their worst latency relative to each
cohort's best candidate; protocol names receive no preference.

Supply only `--host-bridge` or only `--switched` reports to prepare for one fabric
class. Such a ladder has no measured basis for performance claims about the
omitted class. A single report can retain both ordinary and cooperative launch
choices when it measures both. Select the generated file explicitly with
`--collective-ladder`; generating it does not change any packaged default.

The generated 16–1008-byte control bucket uses one active CTA of the ordinary
64-thread latency kernel in `full` mode. Its inclusion provides control-reduction
coverage without a performance ranking. Other buckets correspond exactly to
the supplied measured endpoints. Sparse size reports do not establish the
209-endpoint sweep or performance bounds for bucket interiors. The comparison
JSON records each cohort's best latency, retained candidate measurements, and
best available menu latency.

## GB202 PCIe TP4 recommendations

Use `gb202_pcie_tp4.json` when a package may serve on switched or host-bridge
PCIe groups. It retains measured choices from both fabrics, and startup tuning
selects among them on the deployment GPUs. Export with the same file for either
group:

```sh
.venv/bin/ignition export deepseek-v4.1-flash --gpus 0,1,2,3 \
  --max-concurrency 32 --mtp-draft-tokens 7 \
  --speculative-token-capacity 128 --speculative-graph-budget 64 \
  --prefill-chunk-size 4096 --expert-scale-compression csf --compile-workers 24 \
  --collective-tuning runtime \
  --collective-ladder ignition/preparation/ladders/gb202_pcie_tp4.json
```

Set `--gpus 0,4,8,12` for the measured host-bridge group on `frank2`. GPU indices
identify devices on a particular host; they do not select collective policy.
`gb202_pcie_tp4_host_bridge.json` retains candidates selected solely from the
host-bridge report. Use it only when that narrower measurement basis is intended;
it does not establish switched-fabric performance.

Both files cover out-of-place BF16 reductions on four SM120 GPUs. They contain
210 buckets from 16 bytes through 8 MiB. The 16–1008-byte control bucket uses one
active CTA of the ordinary 64-thread latency kernel in `full` mode, without a
performance ranking. The remaining 209 buckets cover measured endpoints from
1 KiB through 8 MiB at 16 intervals per doubling. Every measured bucket retains
exactly four candidates, including ordinary and cooperative launches.

The common ladder preserves these choices at three DeepSeek V4.1 Flash payloads:

| Payload | Selected bucket maximum | Ordinary candidates | Cooperative candidates |
|---|---:|---:|---:|
| 80 KiB | 84,992 bytes | 3 | 1 |
| 640 KiB | 679,920 bytes | 2 | 2 |
| 2.5 MiB | 2,719,664 bytes | 2 | 2 |

These payloads lie inside their selected buckets. The sweep measurements apply
to the bucket maxima, not to those three interior payloads. Startup tuning
measures each actual prepared reduction size. In the common ladder, 109 measured
buckets retain three ordinary and one cooperative candidate; the other 100
retain two of each.

The measurement cohorts were collected on 2026-10-30 with NVIDIA RTX PRO 6000
Blackwell Workstation GPUs: the physically switched group 0,1,2,3 on `frank1`
and the host-bridge group 0,4,8,12 on `frank2`. Fabric classification follows
physical wiring; the operating system's topology labels do not fully describe
that wiring. Both cohorts use out-of-place BF16, cold-cache timing protocol 3,
CUDA 13.2, PyTorch 2.14.1+cu132, driver 615.71.10, and a verified +6000 MHz
memory-clock offset. The measurement commands applied no SM clock lock. The
measured joint PCIe collective program is `comm.pcie.joint` version 219.

Both JSONL inputs have complete screening and confirmation coverage for all
209 declared sizes, with no failed configurations or missing required profiles.
The retained candidate menus use per-configuration screening medians; final
confirmation batches remain separate. At the measured endpoints, their largest
observed screening penalties against each cohort's best measured configuration
are bounded as follows:

| Ladder | Switched cohort | Host-bridge cohort |
|---|---:|---:|
| Common switched and host-bridge candidates | 1.00% | 1.00% |
| Host-bridge candidates | Not measured | 1.00% |

These comparisons concern isolated all-reduce calls and do not establish
complete-model serving performance. Each cohort has a measured candidate within
the stated bound in every bucket. The common menu includes 56 candidate/cohort
pairs without an individual timing in that cohort; those entries are `null` in
the comparison and do not establish coverage. Bucket interiors, other hardware,
and other GPU groups have no measured sweep performance bound.

Raw reports and comparisons are retained under
`/data/flare-cooperative-integration-20261030/results/`:

```text
ladder-frank1-sw-confirmed.jsonl
ladder-frank2-hb-confirmed.jsonl
ladder-comparison.json
ladder-host-bridge-comparison.json
```

Each JSONL report includes its legal candidate catalog, hardware and software
metadata, raw timing samples, and complete summary. The comparison files record
the exact retained launch geometries, protocol/active-CTA profiles, available
per-cohort timings, and menu penalties. Incomplete sweep reports are not inputs
to either packaged ladder.
