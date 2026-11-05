# Kimi K3.1 preparation

Use Flare's `k31` branch with the FlashInfer revision pinned in `pyproject.toml`
and `uv.lock`. The matching kernel source is published in
[`local-inference-lab/flashinfer`, branch `k31`](https://github.com/local-inference-lab/flashinfer/tree/k31).
The [model preparation guide](model-preparation.md#kimi-k31-pro) provides the
complete TP16/C4/K7 export and native launch commands. Preparation needs sixteen
SM120 GPUs with 96 GB each and the pinned target and draft checkpoints.

## Implementation and qualification

| Area | Implemented behavior | Qualification and limit |
|---|---|---|
| Target and draft | Gated DSA, unrotated FP8 index queries, full-context MLA draft history, MoonViT images and shared cache transfers | Qualified at component level, including context-parallel selection and history transfers. |
| Native package | Bounded preparation fixtures, deferred graph executables, compressed weight catalogs and parallel payload publication | Qualified through a TP16/C4/K7 BF16 package with 1024-token prefill: text, images, four concurrent images, concurrent arithmetic and 32K recall. |
| Expert storage and dense precision | Lossless CSF expert scales; explicit `--online-mxfp8` conversion of the projections listed in the model guide | CSF is used by the native package. The MXFP8/4096 full-model configuration remains research-only until export and service tests complete. Target head, routers and vision retain checkpoint precision. |
| TP16 communication | Ordinary preparation races DCP, phased exchange and fixed-neighbor ring gathers; selected geometry reaches capture | Qualified on sixteen RTX PRO 6000 Blackwell GPUs: nine ordinary tuner races and both bank origins across seven payload families, exact bytes, mixed replay, guards and counter rollover. Full-model performance with these changes is not yet qualified. |

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
| [No-clone custom-op decomposition](../tools/compiler-patches/torch-2.14.1-no-clone-custom-op-decomposition.patch) | Build the one-call replacement graph directly for eligible returnless custom operations; retain canonical fake execution, metadata and replacement handling. Tensor geometry must be static; existing FX integer operands remain runtime bindings. Views, clones, unsupported constants and symbol-producing cases retain canonical tracing. | Research-only for the complete model. Focused CPU and CUDA artifact tests pass; complete Kimi compilation has not completed with this patch. |

The no-clone patch covers runtime integer handles, not just constant operands.
CPU and CUDA checks compile, save, reload and rebind handles while preserving
caller mutations and physical alias coherence through writes to overlapping
views. Canonical fake execution preserves layout checks and shape guards.
Its full-model compilation-time benefit remains unqualified.

Apply the three component-qualified patches in order after environment setup,
while that environment has no running preparation process:

```sh
k31_torch_site=$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
for k31_patch in \
  torch-2.14.1-mutation-regions.patch \
  torch-2.14.1-scheduler-membership.patch \
  torch-2.14.1-lazy-input-mutation-events.patch
do
  patch --fuzz=0 --directory "$k31_torch_site" -p1 \
    < "tools/compiler-patches/$k31_patch"
done
```

The fourth patch is separate for preparation qualification:

```sh
patch --fuzz=0 --directory "$k31_torch_site" -p1 \
  < tools/compiler-patches/torch-2.14.1-no-clone-custom-op-decomposition.patch
```

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
has not been raised; envelope compaction and recoverability are under investigation.

Failed-export cleanup removed the per-rank source records and staging indexes.
Immutable payload blobs remain in the cache, but they do not contain the graph
and placement indexes needed to recombine a package. A CPU-only retry from those
blobs has not been established. Preserve the blobs while resolving the envelope
size; do not assume that they constitute a loadable runtime.

The diagnostic export uses Flare `d2e66491` and FlashInfer `981798fd`, with the
first three Torch patches. It does not include the shared DSA view identities
added to this branch. Record this distinction when comparing preparation times.
Its controller is configured to start native Flare on port 8170 only after
successful publication, then check text, images, concurrent requests and 32K
recall before C1/C4 decode, uncached
32K prefill and Tetris. No serving-performance result from that configuration
has been qualified yet.

```sh
systemctl status astra-kimi31-mxfp8-uncompiled-export.service
tail -n 30 /data/kimi31-support/export-mxfp8-prefill4096-uncompiled.log
cat /data/kimi31-support/execution-state.json
```

The controller is `/data/kimi31-support/after-mxfp8-uncompiled-export.py`; it
exited on the envelope failure. No Kimi server is running on port 8170.
Preparation has an 850 GiB service memory limit; an independent watcher stops
only its owned controller below 128 GiB host `MemAvailable`. The target-only
comparison and restoration of K7 serving remain pending native validation.

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

For the measured compiler phases, complete metadata qualification is being
continued on Frank2 with the recorded TP16/3584 forward.
The helper uses shared GPU 8 only for correctness and compiler execution, with
no weights or GPU-performance claims. Its directory is
`/data/kimi31-support/full-forward-four-patch-ready-replay-20261105`; normal
metadata restoration passed before tracing started. Dependency setup includes
the CuTe Python bootstrap and prepared sampling module. The running unit is
`flare-four-patch-metadata-ready-20261105` (PID 3230572 at publication).
Compilation is bounded to 40 minutes and 128 GiB; only its owned helper stops below 128 GiB host available
memory. Completion of the full artifact remains pending.
