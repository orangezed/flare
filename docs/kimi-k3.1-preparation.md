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

## Preparation compiler patch

The development preparation environment uses Torch `2.14.1+cu132` with
[a Python compiler patch](../tools/compiler-patches/torch-2.14.1-mutation-regions.patch).
It defers mutation-region recomputation only in the singleton
`decompose_auto_functionalized` pass. Other pattern passes retain their checks;
the pass rebuilds region IDs before returning. This patch changes preparation
time. CUDA libraries and Torch binaries are unchanged, and the native server
has no Torch dependency.

Apply it after environment installation, while that environment has no running
preparation process:

```sh
k31_torch_site=$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
patch --fuzz=0 --directory "$k31_torch_site" -p1 \
  < tools/compiler-patches/torch-2.14.1-mutation-regions.patch
```

The patch is CPU-qualified on 20 cases covering v1/v2 functionalization,
reinplacing, overlapping views, BF16, empty and strided tensors, nested
`control_deps`, stream/mempool metadata and ordinary mutation barriers.
Original and patched environments produced identical FX code, region IDs,
outputs and aliases. For 32 rewrites with 32768 alias nodes, the complete pass
measured 0.346 seconds versus 0.185 seconds; full-region scans fell from 33 to 1.
This is a compiler-pass result, not a full-model inference gain.

Keep an unpatched environment for the comparison, then run:

```sh
python3 qualify/torch_mutation_regions/validate.py \
  --baseline-python /path/to/unpatched-venv/bin/python \
  --patched-python .venv/bin/python \
  --output /tmp/k31-compiler-check
```

The driver masks CUDA and launches separate interpreters. Choose an unused
output directory. Torch's source key invalidates forward and Inductor caches
normally. Kernel selections remain subject to their ordinary identity and guard
checks. No forward-cache migration is required.

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

## Work remaining on the development host

The MXFP8/4096 preparation on Frank1 exhausted host RAM during concurrent
forward compilation. Linux recorded almost all of the 1.2 TB host memory in
use; individual compiler processes reached about 108 GiB RSS. The fixed
2 GiB-per-worker estimate in `ignition/preparation/host_memory.py` is inadequate
for these graphs. Reducing `--compile-workers` does not itself provide a
per-process RAM bound.

A sampled compiler stack also showed Dynamo scanning input tensor versions in
`OutputGraph.check_input_mutation_on_current_stream`. A proposed stream-tracking
shortcut was not integrated because correctness across direct stream setters,
new placeholders and multiple devices was not established.

The resumed job uses `--compile-workers 16`, an 850 GiB service memory limit
and a separate watcher that stops its owned preparation process below 128 GiB
host `MemAvailable`. Eighty-two compiled artifacts survived the failed run;
the remaining 3584- and 4096-token prefill variants are needed with and without
draft feedback. Full export, MXFP8 serving speed, native KV capacity and the
target-only comparison remain unqualified.

For continuation on that host:

```sh
systemctl status astra-kimi31-mxfp8-bounded-export.service
tail -n 30 /data/kimi31-support/export-mxfp8-prefill4096-compiler-bounded.log
cat /data/kimi31-support/execution-state.json
```

The controller `/data/kimi31-support/after-mxfp8-compiler-bounded-export.py`
starts native Flare on port 8170 after export, then runs text/image checks,
C1/C4 decode, 32K prefill and Tetris. The separate
`astra-kimi31-target-only-followup.service` waits for those tests, measures the
same package with zero drafts and restores K7 serving. These host scripts are
orchestration; model preparation uses the ordinary Ignition command above.
