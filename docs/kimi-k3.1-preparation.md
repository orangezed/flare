# Kimi K3.1 preparation

Use Flare's `k31` branch with the FlashInfer revision pinned in `pyproject.toml`
and `uv.lock`. The matching kernel source is published in
[`local-inference-lab/flashinfer`, branch `k31`](https://github.com/local-inference-lab/flashinfer/tree/k31).
The [model preparation guide](model-preparation.md#kimi-k31-pro) provides the
complete TP16/C4/K7 export and native launch commands. Preparation needs sixteen
SM120 GPUs with 96 GB each and the pinned target checkpoint. Speculative
preparation additionally needs the draft checkpoint; `--no-mtp` omits it.

## Projection conversion scope

`--online-mxfp8` enables conversion of eligible BF16 projection weights. Its
scope defaults to `all`, preserving the complete set listed in the model guide.
Activations remain BF16. To convert only the fused KDA input projection, use
`--online-mxfp8 --online-mxfp8-scope kda-input` in the export command. That
projection contains query, key, value, output gate, forget-gate input and beta.
The forget-gate output, attention output, MLA/DSA projections, shared experts,
latent MoE projections and draft projections retain checkpoint precision.

The scope is recorded in the preparation plan and package rebuild metadata.
Changing it requires preparing the corresponding weight representation; do not
use `--reuse-weights` with a package prepared under a different scope. Omitted
scope metadata in an existing package means `all`.

## C1 query-exchange qualification

The `astra/k31-query-overlap` Flare branch uses the companion FlashInfer branch
`astra/k31-recursive-ready` (`762ada1fea4d9d2e544b768e3935da0906693cd1`). The
repository lockfile still selects the earlier dependency baseline; install or
explicitly select the companion checkout when reproducing this configuration.
The compiler patches described below are also required.

Implemented: single-row decode exchanges the absorbed query on an auxiliary
stream while sparse selection runs on the main stream. Native-call enqueue
ordering survives whole-forward compilation; selection and query exchange join
before attention. Sparse candidate selection uses a stable cutoff tie order
across ranks. No scheduler or target-head precision change is involved.

Qualified conditions: sixteen RTX PRO 6000 Blackwell GPUs, 600 W limits,
memory-clock offset +6000, TP16/DCP16, text only, C1, no MTP, 4096-token prefill
capacity, lossless CSF expert scales, and `--online-mxfp8-scope kda-input`.
Three requests used 221 prompt tokens and 4096 completion tokens at temperature
1, top-p 0.95, and disabled top-k filtering. The native package was
`gb202-tp16-202611062201`.

| Measurement | Result |
|---|---:|
| C1 decode, three runs | 57.756 / 57.714 / 57.678 tok/s |
| Mean C1 decode | 57.716 tok/s |
| KV capacity | 5,066,752 tokens |
| Uncached 32K prefill, HTTP completion including one output token | 960.36 tok/s |
| Native text, queued-request and 32K recall checks | 8 passed |

Compiled TP16 component tests preserve exact outputs, empty-context behavior,
source lifetimes, and allocation-stable replay with delayed ranks. With a
16K prepared history capacity, query overlap reduces the DSA component time by
12.8–14.0%. This component measurement does not establish the same reduction
for the complete model's 1M history capacity: its fused indexer launches 188
CTAs, compared with 16 in that component test.

Research-only limits: concurrent-collective residency is checked during
preparation but is not encoded as a native package admission requirement.
The component test covers one row and one selection slice. Long-context
indexer residency with concurrent query exchange, larger rows, and multiple
slices require further qualification before general deployment. The target
of approximately 64 tok/s has not been achieved.

## Implementation and qualification

| Area | Implemented behavior | Qualification and limit |
|---|---|---|
| Target and draft | Gated DSA, unrotated FP8 index queries, full-context MLA draft history, MoonViT images and shared cache transfers | Qualified at component level, including context-parallel selection and history transfers. |
| Native package | Bounded preparation fixtures, deferred graph executables, compressed weight catalogs and parallel payload publication | Qualified through a TP16/C4/K7 BF16 package with 1024-token prefill: text, images, four concurrent images, concurrent arithmetic and 32K recall. |
| Expert storage and dense precision | Lossless CSF expert scales; explicit `--online-mxfp8` conversion of the projections listed in the model guide | Target-only C1 qualification is recorded above. Compiled prefill and larger compiled model batches remain research-only. Target head, routers and vision retain checkpoint precision. |
| TP16 communication | Ordinary preparation races DCP, phased exchange and fixed-neighbor ring gathers; selected geometry reaches capture | Qualified on sixteen RTX PRO 6000 Blackwell GPUs: ordinary tuner races, bank reuse, exact bytes, mixed replay, guards and counter rollover. Native C1 measurements above include these candidates; that combined result does not isolate each candidate's contribution. |

The ring uses the existing channel storage and argument ABI. It relays one
neighbor's byte payload per hop and preserves rank-ordered output placement.
The tuner retains the ordinary transport for shapes where it wins. Kimi K3.1
releases the inherited gather-thread override while declaring the existing
512-thread/eight-block LSE geometry explicitly.

The BF16/1024 native reference measured C1 at 39.27 tok/s and C4 at 76.51 aggregate
tok/s on sixteen RTX PRO 6000 Blackwell GPUs, with temperature 1, top-p 0.95
and low reasoning effort. Clocks were unlocked; active frequencies were not
recorded for this reference. An uncached 32K
raw-input request generated one token in 259.84 seconds. These measurements
precede the MXFP8 and gather changes and do not describe their serving speed.

## Forward compilation

The storage-lifetime analysis in Ignition indexes pending reads by storage and
retires values at their last-use positions. It preserves the order of reads,
writes and required snapshots without rescanning every live value at each node.
Implemented and qualified: 205 CPU alias/mutation cases and 12 compiled CUDA
cases pass. On a captured 475,656-node Kimi prefill graph, the direct-scan
analysis took 1444.04 seconds and the indexed analysis took 1.20 seconds;
ordered results were identical. This measures one preparation function, not
model latency. The reference used one CPU with low scheduling priority; clocks
and scheduling were not normalized. The captured inputs and raw comparison are
on Frank1 at `/data/kimi31-support/compile-study/storage-lifetime-real-comparison.json`.

Sparse-attention stages also reuse identical cache and selection view objects
for each row count. Each stage retains its own plans, cursor and dependencies.
This changes preparation identities without changing tensor regions or GPU
arithmetic. Six CPU cases and TP1/TP4 CUDA component cases cover serialization,
partial/inactive rows, poisoned scratch and compiled/native replay. The two
source changes pass 211 CPU cases together. Full-model serving with the shared
view identities remains unqualified.

### Torch 2.14.1 preparation patches

The native server does not use Torch. The following patches change only the
`2.14.1+cu132` preparation environment; CUDA libraries and native Torch binaries
are unchanged. They are supplied as source patches rather than host-specific
installed files.

| Patch | Resulting behavior | Qualification |
|---|---|---|
| [Mutation regions](../tools/compiler-patches/torch-2.14.1-mutation-regions.patch) | Rebuild mutation-region IDs once after singleton auto-functionalization rewrites; other passes retain their checks. | Qualified: 20 CPU cases preserve FX, regions, outputs, aliases and stream/mempool metadata. |
| [Scheduler membership](../tools/compiler-patches/torch-2.14.1-scheduler-membership.patch) | Query the existing ancestor, removed-operation and completed-operation sets directly during in-place reuse decisions. | Qualified: 40 CPU cases preserve decisions and real kernel argument bindings. |
| [Event-triggered mutation checks](../tools/compiler-patches/torch-2.14.1-lazy-input-mutation-events.patch) | Defer input-version bookkeeping until event recording requires it; restart root tracing before recording and retain the complete existing checks on that trace. | Qualified: CPU mutation/restart cases and 18 CUDA event cases preserve FX, outputs, errors, side effects and event ordering. |
| [No-clone custom-op decomposition](../tools/compiler-patches/torch-2.14.1-no-clone-custom-op-decomposition.patch) | Build the one-call replacement graph directly for eligible returnless custom operations; retain canonical fake execution, metadata and replacement handling. Tensor geometry must be static; existing FX integer operands remain runtime bindings. Views, clones, unsupported constants and symbol-producing cases retain canonical tracing. | Qualified for complete target-only C1 decode/resume compilation and native text serving. Compiled prefill and larger model batches remain unqualified. |

The no-clone patch covers runtime integer handles, not just constant operands.
CPU and CUDA checks compile, save, reload and rebind handles while preserving
caller mutations and physical alias coherence through writes to overlapping
views. Canonical fake execution preserves layout checks and shape guards.
Its full-model compilation-time benefit remains unqualified.

Apply all four patches in order after environment setup,
while that environment has no running preparation process:

```sh
k31_torch_site=$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
for k31_patch in \
  torch-2.14.1-mutation-regions.patch \
  torch-2.14.1-scheduler-membership.patch \
  torch-2.14.1-lazy-input-mutation-events.patch \
  torch-2.14.1-no-clone-custom-op-decomposition.patch
do
  patch --fuzz=0 --directory "$k31_torch_site" -p1 \
    < "tools/compiler-patches/$k31_patch"
done
```

These four patches are additional to the compatibility patches applied automatically
by Ignition; the preparation command does not apply them itself.

Keep a separate unpatched environment. Representative checks, run from the
repository root, are:

```sh
python3 qualify/torch_mutation_regions/validate.py \
  --baseline-python /path/to/unpatched-venv/bin/python \
  --patched-python .venv/bin/python --output /tmp/k31-mutation-regions

CUDA_VISIBLE_DEVICES= .venv/bin/python qualify/torch_compiler/scheduler_membership.py \
  --baseline-source /path/to/unpatched-venv/lib/python3.12/site-packages/torch/_inductor/scheduler.py

CUDA_VISIBLE_DEVICES= /path/to/unpatched-venv/bin/python \
  qualify/torch_compiler/events_cpu.py baseline --output /tmp/k31-events-baseline.json
CUDA_VISIBLE_DEVICES= .venv/bin/python qualify/torch_compiler/events_cpu.py candidate \
  --baseline-result /tmp/k31-events-baseline.json --output /tmp/k31-events-candidate.json

CUDA_VISIBLE_DEVICES= .venv/bin/python qualify/torch_compiler/unbacked_handle_aot_cpu.py \
  --expect-direct --output /tmp/k31-runtime-handles/result.json
```

The runtime-handle check requires the fourth patch. CUDA event checks use
`qualify/torch_compiler/events_cuda.py baseline|candidate` on an explicitly
selected GPU and write one JSON result to stdout. Use unused output paths.
Torch source keys invalidate forward and Inductor caches normally. Existing
b12x kernel selections remain subject to their query identities and guards;
changing Torch source does not authorize forward-cache migration.

## Gather checks without model weights

The harness uses package-owned CUDA IPC storage, verifies arbitrary byte
patterns and measures maximum rank-local CUDA duration. It needs a built
Heliopause extension and sixteen free GPUs; it does not load model weights.

```sh
.venv/bin/python qualify/pcie_gather.py --gpus 0-15 \
  --ring --model-families --epoch-check --graph-base-slot 1 \
  --samples 3 --compile-workers 16 --output /tmp/k31-gather-bank1

.venv/bin/python qualify/pcie_gather.py --gpus 0-15 \
  --ring --model-families --epoch-check --graph-base-slot 0 \
  --samples 3 --compile-workers 16 --output /tmp/k31-gather-bank0
```

Add `--dry-run` to print shapes without importing CUDA libraries. Both commands
require unused output directories. On sixteen RTX PRO 6000 Blackwell GPUs with
unlocked clocks (active frequencies were not recorded), the 32-row 512 KiB-per-rank candidate gather measured
0.322/0.328 ms warm/cold with the ring, versus approximately 7 ms for ordinary
512-thread/eight-block DCP. These are component timings, not model throughput.

## Full-model validation and host continuation

The MXFP8/4096 configuration is implemented but its serving speed and native KV
capacity remain unqualified. A TP16/C4/K7 diagnostic export uses the ordinary
Ignition command with `--online-mxfp8`, `--prefill-chunk-size 4096`, four image
slots and `--no-compile`. This flag skips complete-forward compilation;
kernels are still prepared and CUDA graphs are captured for native serving.
Flare serves the prepared CUDA graphs natively.
The export captured all 104 graphs and wrote weight payloads. Package combination
then failed because its envelope was 166,123,939 bytes, exceeding the existing
134,217,728-byte limit. No MXFP8/4096 package was published or served. The limit
remains unchanged. The producer now coalesces nearby InitialState ranges in
fixed ordinary device allocations, explicitly storing short zero gaps. It
excludes managed, host and peer allocations, weight-owner ranges, relocations
and conflicting globals. Source payloads remain fully hash-checked. On the
qualified BF16/1024 package, a CPU wire-size replay reduced 68,084,668 to
49,553,379 bytes and 327,834 to 47,868 initializers. It wrote 3.86 GB of state
payloads including 184.55 MB of zero padding; weights were untouched.
The compaction export succeeded with 41,627,807 metadata bytes (39.70 MiB),
below the unchanged limit. Its cache entry is
`/data/kimi31-support/package-cache/models/moonshotai_2FKimi-K3.1-Pro/gb202-tp16-202611052232`.
The initial unrestricted-catalog native load ran out of GPU memory during
graph instantiation. With the explicit sparse prefill selection described
below, native target-only serving passed all 13 functional checks. Decode,
prefill and draft comparisons remain in measurement.

Failed-export cleanup removed the per-rank source records and staging indexes.
Immutable payload blobs remain in the cache, but they do not contain the graph
and placement indexes needed to recombine a package. A CPU-only retry from those
blobs is unsupported without the deleted source records. Preserve the blobs
until weight-cache reuse is established; they are not a loadable runtime.

The failed diagnostic export used Flare `d2e66491` and FlashInfer `981798fd`,
with three Torch patches and distinct DSA view objects. The compaction
validation export uses Flare `9b4c7257`, the same FlashInfer, shared DSA views
and all four Torch patches; it retains `--no-compile` to isolate package
publication from complete-forward compilation.
Its controller is configured to start native Flare on port 8170 only after
successful publication, then check text, images, concurrent requests and 32K
recall before C1/C4 decode, uncached
32K prefill and Tetris. No serving-performance result from that configuration
has been qualified yet.

```sh
systemctl status astra-kimi31-state-compaction-export.service
tail -n 30 /data/kimi31-support/export-mxfp8-prefill4096-state-compaction.log
cat /data/kimi31-support/execution-state.json
```

The controller `/data/kimi31-support/after-mxfp8-state-compaction-export.py`
finished export and stopped after the native startup failure. Its export source
is the detached worktree
`/data/kimi31-support/flare-envelope-validation` at `9b4c7257`; its native
extension is in `/data/kimi31-support/state-compaction-native-python`.
The unrestricted-catalog native process used
`--mtp-draft-tokens 0`; the draft was not executing. It loaded 1385.8 GiB
weights in 241.4 seconds and 7360 kernels in 2.4 seconds, then failed at
`cuGraphInstantiateWithFlags` with CUDA out-of-memory. KV/continuation physical
cache backing occurs after graph preparation and was not the cause.
An instrumentation-only retry identified accumulation of prefill executables:
the 4096-row variant succeeds and consumes 852 MiB per rank, the 3584-row
variant succeeds and adds 746 MiB, then the 3072-row variant fails with about
315 MiB free. All five vision executables together consume only 4 MiB.
Building the raw 319,681-node template consumes no observed GPU memory;
module loading adds 152 MiB. These are measured driver-allocation deltas,
not inference timings. The generic load-time `--prefill-graph-capacities 128,4096` selector preserves
the maximum chunk size and omits unused intermediate prefill executables.
Omission retains the complete catalog. CPU selection, scored-family, peer-rank
and padding/live-page tests pass; the same Kimi package loads successfully
with this flag. Graph preparation took 29.7 seconds, followed by 524,288 KV
tokens and a 0.81–0.88 GiB cache budget per GPU. Thirteen native functional
checks pass, including text, images, four concurrent images, conversation
continuation, long-context recall and concurrent arithmetic.
The package remains intact for native-only diagnosis; no re-export is needed
to inspect its selected graphs and fixed allocations.
Preparation has an 850 GiB service memory limit; an independent watcher stops
only its owned controller below 128 GiB host `MemAvailable`. The first native launch selects `--mtp-draft-tokens 0` from the K7 package.
It validates target-only text/images and measures C1/C4 decode, 32K prefill and
Tetris. The separate `astra-kimi31-state-compaction-k7-followup` unit then loads
the same package with seven drafts, repeats checks and benchmarks, and leaves
K7 serving. Its script is `/data/kimi31-support/after-target-first-benchmarks.py`.
The followup stopped on the initial native failure without taking GPU action.
The replacement target-only service is `astra-kimi31-sparse-prefill-d0-c4`,
PID 2860184 at publication, on port 8170. Its binary is
`/data/kimi31-support/compile-study/native-prefill-selection-target/release/flare`;
it uses the retained package with `--max-concurrency 4 --mtp-draft-tokens 0
--no-mtp-adaptive-verification --prefill-graph-capacities 128,4096`.
`astra-kimi31-sparse-target-validation` runs checks and C1/C4 decode, 32K prefill and Tetris
measurements; inspect `/data/kimi31-support/execution-state.json` and
`/data/kimi31-support/sparse-target-validation.log`. K7 has not yet been
revalidated with the selector. Loading with zero drafts retains
the package's prepared weights and cache layout; a package exported without
drafts is required to measure the storage saved by excluding them.

Full-forward compiler qualification uses the serialized 3584-token prefill
metadata in `/data/kimi31-support/compile-study/full-forward-storage-input/`;
it loads no checkpoint weights. The three-patch compiler passed storage
analysis in 1.589 seconds inside that full pipeline, but reached a 20-minute
bound during per-operation auto-functionalization decomposition before code
generation. The fourth patch addresses that measured phase. Full artifact
completion and subsequent compiled native-package validation remain pending.

The retained BF16/1024 package is
`/data/kimi31-support/package-cache/models/moonshotai_2FKimi-K3.1-Pro/gb202-tp16-202611050522`.
It is the qualified functional baseline; it does not contain the MXFP8 dense
conversion or fixed-neighbor gather selections. Retain it until a replacement
package passes native validation.

The four-patch full metadata compile on Frank2 reached postgrad decomposition
with 49,514 direct replacements and one canonical replacement, then exhausted
its 40-minute bound in `Scheduler.compute_ancestors`. Peak memory was 62.55 GiB;
there was no OOM. That full artifact remains unqualified. Logs are on Frank2
in `/data/kimi31-support/full-forward-four-patch-ready-replay-20261105`.
A scheduler-local compact-ancestry prototype is research-only and has not been
added to the preparation patches. It must preserve ordered-set behavior and
fusion decisions before complete-forward qualification.

## Retain publisher inputs for CPU diagnosis

The optional [publisher-input recorder](../qualify/runtime_package/capture_sources.py)
runs the ordinary Ignition export, then hardlinks the publisher's disk sources
and referenced cached payloads before combination. It records peer-memory
arguments and rewrites only copied FileSource paths. Model operators, graph
choices and the arguments passed to the ordinary publisher remain unchanged.
It omits signing requests and signing material. Use an unused capture directory
on the same filesystem as the source files; a recorder failure is reported and
the ordinary export continues.

```sh
.venv/bin/python qualify/runtime_package/capture_sources.py \
  --capture-dir /data/kimi31-publisher-inputs -- export kimi-k3.1-pro \
  --gpus 0-15 --max-concurrency 4 --image-concurrency 4 \
  --prefill-chunk-size 4096 --mtp-draft-tokens 7 \
  --speculative-token-capacity 128 --speculative-graph-budget 64 \
  --online-mxfp8 --collective-tuning static --autotune-effort default \
  --compile-workers 16 --no-compile --verbose
```

The validation host records those inputs at
`/data/kimi31-support/state-compaction-export-inputs`. The recorded
`combine-inputs.json` identifies copied rank records, payload pins and the cache
root. An explicit CPU retry must supply its own publisher and verify that the
referenced cache objects remain available. This recorder is a diagnosis tool,
not an independently loadable model or a serving path.

## Draft checkpoint compatibility

The draft is `Inferact/Kimi-K3.1-Pro-DSpark-preview`, revision
`5e654fbe920278f228a89519b06b2680aa56480f`. Its [model card](https://huggingface.co/Inferact/Kimi-K3.1-Pro-DSpark-preview)
declares Kimi K3.1 Pro as its target and five full-causal MLA layers. Full MLA
is intentional even though the target uses Gated DSA. The source architecture
identifier `K3DSparkModel` names this checkpoint implementation.

A read-only reference audit found matching zero-based target output taps
75/79/83/87/91, context projection/normalization, full paged history, causal
lengths, shared embeddings/head and sequential Markov conditioning. vLLM's
76/80/84/88/92 capture slots account for its embedding entry. Extracted YaRN
frequencies and scaling matched bitwise. This is a source-contract check;
K3.1 native acceptance and end-to-end speed remain unqualified for MXFP8/4096.

## DSA staging investigation

The recipe currently slices selection/attention into 32 query rows. A 4096-row
prefill therefore contains 128 stages for each of 24 DSA layers, or 3072 stage
bodies; this is a scratch/channel policy, not a model-math requirement. A
128-row capacity would use 768 stage bodies. The isolated prototype is at
`/data/kimi31-support/compile-study/dsa-selection-cap-worktree`; it is not
part of this branch's default preparation. It carries an ordinary
`--dsa-selection-rows` configuration through planning and preparation-only
rebuild metadata. CPU tests pass; GPU qualification is pending.

The captured 32-row shared arena is 519.374 MiB, dominated by MoE, while selected
DSA dense attention uses 2 splits. At 128 rows, that same split choice fits within
the existing arena; declared additional pools/channels require about 113.57 MiB
per rank. Unrestricted tuning can consider 32 splits and require an additional
250.17 MiB arena. These are allocation-plan estimates; captured temporaries,
CUDA granularity, actual plan choices and native graph-executable memory still
require GPU measurement. The native failed package's largest prefill graph has
319,681 nodes and is identical when loading concurrency 1 or 4, so concurrency
reduction alone is not established as a remedy.
