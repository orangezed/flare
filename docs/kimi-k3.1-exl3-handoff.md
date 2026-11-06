# Kimi K3.1 EXL3 handoff

Branch `k31exl3` prepares and serves the two-bit checkpoint
[`local-inference-lab/Kimi-K3.1-Pro-EXL3-2bit`](https://huggingface.co/local-inference-lab/Kimi-K3.1-Pro-EXL3-2bit)
through native Flare on eight SM120 GPUs. This document explains how to build
the Ignition package from that checkpoint, how to serve it, and what remains:
speculative decoding (MTP), vision, prefill speed and long context.

It starts from the published checkpoint. How the checkpoint was encoded is
deliberately not documented here; do not add that procedure to this repository.

Read [AGENTS.md](../AGENTS.md) and the [design review](design-review.md) before
changing source. [Kimi K3.1 preparation](kimi-k3.1-preparation.md) describes the
official checkpoint at TP16, including the Torch patches used below.

## Current state

| Area | State |
|---|---|
| Serving | Native `flare run`, TP8 with context parallelism (DCP8), one concurrent request, 128-token prefill chunk, online MXFP8 projections, no speculation, no vision |
| Text | Short arithmetic and coding requests complete with coherent output |
| Decode speed | 46.10 and 46.27 tokens/s on 1,733- and 3,692-token answers |
| Prefill speed | Not measured |
| MTP | Not prepared; memory estimate only |
| Vision | Not prepared, so requests with images are rejected; memory estimate only |
| Context | 129,024 tokens per request; 2,591,744 KV tokens shared across cached histories |
| TP16 | Not supported for this checkpoint. The loader requires each rank to own complete 128-channel intermediate blocks, and 3,072 channels do not divide into sixteen such extents |
| Quality | Model KLD, task quality and long generations have not been evaluated |

The decode figures come from eight RTX PRO 6000 Blackwell Workstation Edition
GPUs, short coding prompts, recipe sampling defaults and `reasoning_effort=low`.
Decode is `(N - 1) / (last token timestamp - first token timestamp)`, which
excludes prefill. Clocks were not recorded, so this is not a clock-controlled
comparison.

## Sources

| Component | Revision |
|---|---|
| Flare | This branch: `a12381ca` from `k31`, then `ffa56338` (EXL3 loading), then this document |
| FlashInfer | `981798fd`, unmodified. `pyproject.toml` and `uv.lock` pin it; branch [`k31exl3`](https://github.com/local-inference-lab/flashinfer/tree/k31exl3) points at it |
| Checkpoint | Hugging Face revision `38fbbe2ba5d46141eaa4d662c128bdf8d810746b` |
| MTP draft, when enabled | `Inferact/Kimi-K3.1-Pro-DSpark-preview@5e654fbe`, pinned in `ignition/recipes/kimi3/config.py` |

Commit `ffa56338` adds:

- EXL3 detection that keeps the checkpoint's BF16 projections as the retained
  precision.
- CPU loading of EXL3 experts and bounded GPU preparation through the existing
  A16 SiTU kernels. It adds no CUDA kernels, serving callbacks or Helios graph
  families.
- Persistent expert side tables that survive native export byte for byte.
- Complete EXL3 byte and cache accounting, with a conservative savings estimate
  applied only with explicit `--online-mxfp8`.
- TP8 geometry for K3.1 and rejection of shards that split intermediate blocks.
  The official MXFP4 path stays as it was.

Flare's `k31` branch has moved past this base. These commits are not included:

| Commit | Change | Relevance |
|---|---|---|
| `bb7f56af` | `flare run --prefill-graph-capacities` loads a sparse set of captured prefill graphs | Needed for larger prefill chunks; see [prefill speed](#prefill-speed) |
| `9b4c7257` | Packs initial state to shrink signed package metadata | Needed if a larger package exceeds the 128 MiB envelope; this package's metadata is 25.4 MB |
| `6552aecc`, `0997910c` | Export diagnostics and documentation | Records the TP16 MXFP8/4096 graph memory failure |

This branch merges with `k31` at `bb7f56af` without textual conflicts. The merged
result has not been built or tested.

## Prepare the environment

These commands restate the setup that produced the reference package. They have
not been replayed from a fresh clone of this branch.

The reference host has eight RTX PRO 6000 Blackwell Workstation Edition GPUs
(97,887 MiB each), driver 615.71.10, CUDA toolkit 13.2, Python 3.12 and 1.1 TiB
of RAM. The checkpoint needs 742.2 GiB of disk and the package cache another
703 GiB. Build dependencies are listed under
[environment setup](model-preparation.md#environment-setup).

```sh
git clone --branch k31exl3 https://github.com/voipmonitor/flare.git
cd flare

uv sync --locked --extra prepare --extra test
uv pip install --python .venv/bin/python --no-deps ./heliopause
.venv/bin/python -m ignition.patches

k31_torch_site=$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
for k31_patch in \
  torch-2.14.1-mutation-regions.patch \
  torch-2.14.1-scheduler-membership.patch \
  torch-2.14.1-lazy-input-mutation-events.patch
do
  patch --fuzz=0 --directory "$k31_torch_site" -p1 \
    < "tools/compiler-patches/$k31_patch"
done

cargo build --locked --release --manifest-path flare/Cargo.toml
```

- The reference package used these three Torch patches, not
  `torch-2.14.1-no-clone-custom-op-decomposition.patch`.
- Reinstalling Torch removes the patches, so apply them again afterwards.
- Avoid `scripts/setup_preparation.py` here. It reinstalls dependencies and can
  refresh the FlashInfer lock.
- Serve with the ordinary Heliopause build. Its `qualification` Cargo feature is
  for `ignition debug-serve` and tests.

## Download the checkpoint

```sh
hf download local-inference-lab/Kimi-K3.1-Pro-EXL3-2bit \
  --revision 38fbbe2ba5d46141eaa4d662c128bdf8d810746b \
  --local-dir /models/Kimi-K3.1-Pro-EXL3-2bit
```

The download is 742.2 GiB in 188 weight files. Export from a local directory as
shown, since that is the path that has been exercised. Tensors outside the routed
experts keep the official BF16 and FP32 precision. Online MXFP8 is applied during
preparation and never changes the checkpoint.

## Configure

Put the configuration in its own directory. With no publishing identity
configured, export creates `development-signing-key.pem` and
`development-publisher.json` beside it. Keep the private key on the host.

```toml
# $K31_HOME/flare-tp8.toml
[server]
host = "0.0.0.0"
port = 8012

[memory]
reserve = "512M"

[paths]
hub_cache = "/root/.cache/huggingface/hub"
kernel_cache = "/root/.cache/flare/k31-exl3/kernels"
package_cache = "/root/.cache/flare/k31-exl3/packages"
export_staging = "/root/.cache/flare/k31-exl3/staging"
prefix_cache = "/root/.cache/flare/k31-exl3/prefix"
cuda = "/usr/local/cuda-13.2"

[prefix_cache]
memory = "4G"
disk_limit = "0"
```

Startup reports a 7.0 GiB host prefix cache with this configuration.

## Export the package

Run from the repository root with eight idle GPUs. Select them by UUID so the
order stays stable.

```sh
ulimit -n 65536
export CUDA_VISIBLE_DEVICES=GPU-...,GPU-...   # exactly eight UUIDs

.venv/bin/python -m ignition export /models/Kimi-K3.1-Pro-EXL3-2bit \
  --config "$K31_HOME/flare-tp8.toml" \
  --repo-id local-inference-lab/Kimi-K3.1-Pro-EXL3-2bit \
  --gpus 0-7 --max-concurrency 1 --prefill-chunk-size 128 \
  --no-mtp --language-model-only --online-mxfp8 \
  --autotune-effort off --compile-workers 16
```

Export prints `Exported cache entry: PATH`. Use that path as `CACHE_ENTRY` below.

| Flag | Effect on this package |
|---|---|
| `--gpus 0-7` | TP8. The indices refer to the UUID mask. Context parallelism is on by default |
| `--max-concurrency 1` | One active request |
| `--prefill-chunk-size 128` | Captures a single 128-token prefill graph |
| `--no-mtp` | Loads no draft weights and keeps no draft history |
| `--language-model-only` | Prepares no vision encoder, weights or image graphs |
| `--online-mxfp8` | Converts eligible BF16 projections while loading; see [the model guide](model-preparation.md#kimi-k31-pro) for the exact set |
| `--autotune-effort off` | Uses heuristic kernel choices without tuning races. Reassess this during speed work |
| `--repo-id` | Records the package identity; nothing is uploaded |

The resulting catalog has six graphs: `prefill` (width 128), `decode`, `resume`,
`init`, `restore` and `snapshot`.

| Stage | Cold export | Cached re-export |
|---|---:|---:|
| Load weights | 146.5 s | 50.3 s |
| Construct graph catalog | 47.1 s | 1.1 s |
| Prepare kernels | 42.9 s | 4.4 s |
| Compile forwards | 143.6 s | 2.8 s |
| Materialize weight payloads | 325.5 s | 140.5 s |

Export needs `RLIMIT_NOFILE` of at least 65536. With the common soft limit of
1024 it fails late with `native package export failed: Too many open files`. Raise
the limit and export again; cached weights are reused.

A package refers to payloads in the shared blob cache, so copying only its entry
directory is not a backup. Keep the whole cache directory, or use Flare's package
conversion described in the [CLI design](cli-design.md).

## Serve the package

```sh
ulimit -n 65536
export CUDA_VISIBLE_DEVICES=GPU-...,GPU-...   # same UUIDs in the same order

profile="$K31_HOME/development-publisher.json"
fingerprint=$(python3 -c 'import hashlib, json, sys; print(hashlib.sha256(bytes(json.load(open(sys.argv[1]))[0]["public_key"])).hexdigest())' "$profile")

flare/target/release/flare run "$CACHE_ENTRY" \
  --config "$K31_HOME/flare-tp8.toml" --gpus 0,1,2,3,4,5,6,7 \
  --publisher-keys "$profile" \
  --dangerously-approve-key "$fingerprint" \
  --dangerously-accept-publisher local-development \
  --trust-store "$K31_HOME/trust.toml" \
  --host 0.0.0.0 --port 8012
```

The two `--dangerously-*` flags approve your own development key without the
first-use prompt, which an unattended launch needs. Use them only for a key
generated on this host. In a terminal you can omit them and accept the key when
prompted. `--host 0.0.0.0` exposes the API on every interface; use `127.0.0.1` on
an untrusted network.

A healthy start loads 702.1 GiB of weights in about two minutes and reports:

```text
Plan cache.      per GPU: 88.59–88.66 GiB loaded · 5.28–5.34 GiB cache budget · reserve 512MiB
Allocate cache.  42.1 GiB · 2591744 KV tokens · 76 shared snapshot slots · limited by memory
Start server.    http://HOST:8012 · 129024 tokens/request
```

Check it with a single request:

```sh
model=$(curl -s http://127.0.0.1:8012/v1/models | python3 -c 'import json, sys; print(json.load(sys.stdin)["data"][0]["id"])')
curl -s http://127.0.0.1:8012/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 * 19?\"}],\"max_tokens\":256,\"reasoning_effort\":\"low\"}"
```

Stop the server with Ctrl-C or SIGTERM so native replay drains before CUDA
resources are released. Serving does not use Python or Torch.

## Remaining work

Rules that apply to everything below:

- Keep the working package. A change to capacities, vision or speculation needs a
  separate export, because native flags cannot add graphs or weights excluded
  during preparation.
- Measure speed through `ignition export` and `flare run`, following
  [GPU performance work](performance-work.md). `ignition debug-serve` timings are
  diagnostic only.
- AGENTS.md forbids Claude Opus 5.5 from designing or implementing low-level CUDA
  kernels and changes to model fusion, projection packing, fused arithmetic, stream
  overlap or scratch aliasing. Claude Opus 5.5 wrote this document, so treat the
  options below as starting points rather than designs, and do not assign those
  mechanisms to that model.

### Memory budget

MTP, vision, larger prefill graphs and long context all draw on the same memory.
These are per-GPU figures at TP8 and one concurrent request. Except for the cache
budget, they are arithmetic from source declarations and safetensors shapes, not
measurements.

| Item | GiB per GPU |
|---|---:|
| Cache and state budget of the current package | 5.28–5.34 |
| Target KV and index pages for 1,048,576 tokens | 2.07 |
| Five FP8 draft histories at 1,048,576 tokens | 0.35 |
| Draft weights, online MXFP8 or BF16 | 0.52 or 0.80 |
| Nine recurrent rows for seven draft tokens | 0.53 |
| Vision at 65,536 patches | at least 2.26 |
| Vision at 16,384 patches | at least 1.21 |
| Graphs for larger prefill chunks | Unmeasured at TP8 |

With online MXFP8 drafts, 1M context, MTP and full vision sum to 5.73 GiB and do
not fit. Lowering vision to 16,384 patches gives 4.68 GiB, which leaves about
0.6 GiB per GPU for additional graphs, buffers and alignment. Those increments
are unmeasured, so this is a candidate rather than a guarantee.

`--online-mxfp8` controls target and draft projections together. No independent
BF16-draft, MXFP8-target switch exists; the BF16 row is an allocation alternative.

### MTP

The package was exported with `--no-mtp`. No speculative K3.1 EXL3 package has
been exported or served.

| Candidate | Inspected revision | Weights | Notes |
|---|---|---:|---|
| [Inferact/Kimi-K3.1-Pro-DSpark-preview](https://huggingface.co/Inferact/Kimi-K3.1-Pro-DSpark-preview) | `6a2ef20a` | 5.00 GiB | Five full-context MLA layers. Flare has its architecture and loader. Preferred first candidate |
| [RadixArk/Kimi-K3.1-Dflash2-Preview](https://huggingface.co/RadixArk/Kimi-K3.1-Dflash2-Preview) | `d0d8bc2e` | 5.19 GiB | Five MLA layers, sliding window of 1,024, block convolutions and a candidate selector. This recipe would need those DFlash2 operations integrated |
| [LithosAI/Kimi-K3.1-Pro-DSpark](https://huggingface.co/LithosAI/Kimi-K3.1-Pro-DSpark) | `d791d217` | 4.19 GiB | Five GQA layers with 16 KV heads and a materially larger draft KV. Not a drop-in replacement |

Use a draft trained for K3.1, not a K3 draft whose tensor shapes happen to match.

1. Compare the pinned Inferact revision `5e654fbe` with the newer revision above
   before changing the pin. The two have not been shown to be interchangeable.
2. Exercise the EXL3 target with latent DSpark and KDA rollback at the intended
   speculative widths. Starting points are `tests/recipes/test_kimi31_draft_cuda.py`,
   `test_kimi31_state_cuda.py` and `test_kimi3_speculation_cuda.py`. None has been
   run with the EXL3 target.
3. Export a separate package. This command has not been run:

   ```sh
   .venv/bin/python -m ignition export /models/Kimi-K3.1-Pro-EXL3-2bit \
     --config "$K31_HOME/flare-tp8.toml" \
     --repo-id local-inference-lab/Kimi-K3.1-Pro-EXL3-2bit \
     --gpus 0-7 --max-concurrency 1 --prefill-chunk-size 128 \
     --mtp-draft-tokens 7 --speculative-token-capacity 128 \
     --speculative-graph-budget 64 \
     --language-model-only --online-mxfp8 \
     --autotune-effort off --compile-workers 16
   ```

4. Check native memory headroom, then measure decode speed and acceptance on this
   two-bit target. Published acceptance on official weights does not carry over.
   Loading the same package with `--mtp-draft-tokens 0` gives a baseline with the
   same layout.

Speculation can commit several tokens with one step timestamp, so the
strictly increasing timestamps assumed by no-speculation measurements do not
apply. Record accepted output tokens, proposal and verification counts and the
end-to-end decode duration. `flare_speculative_rounds_total` and
`flare_speculative_gpu_nanoseconds_total` give graph rounds per second.

Page and row sizes behind the budget: target pages are 2,174,976 bytes per 1,024
global tokens, target and draft pages are 2,543,616 bytes, and a speculative row
is 63,181,056 bytes. Seven draft tokens at one request need nine rows.

### Vision

The package was exported with `--language-model-only`. The MoonViT encoder has
447,358,976 parameters and 894,717,952 bytes of BF16 weights (0.83 GiB). The
checkpoint carries them unchanged.

Native K3.1 image serving has only been exercised at TP16. The TP8 path needs its
own validation. Start from `tests/recipes/test_kimi3_vision_cuda.py` and
`tests/frontend/test_kimi3_images.py`, then export a separate package. This command
has not been run:

```sh
.venv/bin/python -m ignition export /models/Kimi-K3.1-Pro-EXL3-2bit \
  --config "$K31_HOME/flare-tp8.toml" \
  --repo-id local-inference-lab/Kimi-K3.1-Pro-EXL3-2bit \
  --gpus 0-7 --max-concurrency 1 --image-concurrency 1 --prefill-chunk-size 128 \
  --no-mtp --online-mxfp8 \
  --autotune-effort off --compile-workers 16
```

Each rank loads the whole encoder (`ignition/recipes/kimi3/vision.py`), runs it on
identical input and keeps its own 896 of 7,168 output columns
(`ignition/recipes/kimi3/vision_state.py`). The graph therefore needs no collective,
but eight GPUs compute the same result and image encoding is no faster than on one
GPU. The source does not state why the encoder is replicated.

Declared allocations per GPU, with one image slot:

| Item | GiB |
|---|---:|
| Encoder weights, BF16 on every rank; online MXFP8 does not apply | 0.83 |
| Scratch for 65,536 patches | 1.25 |
| Input buffers | 0.15 |
| Image feature arena | 0.03 |
| Total | 2.26 |

This is a lower bound because it excludes temporaries in the compiled graph. The
learned position interpolation alone can reach about 1 GiB at full capacity. On the
TP16 package, all five vision executables together measured only 4 MiB, so the cost
is in storage rather than executables.

Options if this does not fit:

- **Lower the patch capacity.** 16,384 patches (about 1,792 × 1,792 pixels) reduces
  the total to about 1.21 GiB and also shrinks the temporaries. `PATCH_CAPACITIES`
  is a constant shared with K3, and the official processor allows 65,536 patches
  (about 3,584 × 3,584 pixels). Images that need more patches than the prepared
  capacity are rejected at admission unless the resize limit changes too.
- **Shard the encoder.** MiMo, MiniMax and Step-5 already shard theirs with
  all-reduce. For MoonViT this would cut weights to about 0.13 GiB and the total to
  roughly 0.9–1.45 GiB, because every rank must still hold the full-width hidden
  state. It adds 55 all-reduces per image, about 7 GiB of reduced data for the
  largest image. The 12 attention heads also do not divide evenly across TP8 or
  TP16, so `partition()` rejects them; `padded_partition()` would give two heads per
  rank at TP8 and leave the last two ranks' heads empty. This reaches restricted
  mechanisms and needs a design that respects AGENTS.md.
- **Give up KV capacity.** Without changing the encoder, vision at full capacity
  reduces shared KV from 2.59M to at most about 1.5M tokens.

### Prefill speed

Prefill has not been measured on this package. The only related data are two
first-token latencies from the decode runs: 1.12 s with 52 uncached tokens and
1.08 s with 57, each with 68 tokens reused from the prefix cache. Each should
correspond to one 128-token graph plus prefix restore and first-token sampling.

If every chunk cost that much, throughput would be about 115 tokens/s, an uncached
32K prompt would take about 4.7 minutes, and a prompt at the 129,024-token limit
about 18 minutes. Treat this as a prediction to test, since attention cost may grow
with context. For comparison, the official checkpoint at TP16/C4/K7 with BF16
projections and 1,024-token chunks took 259.84 s for an uncached 32K request. That
is a different topology and package, not a controlled baseline.

Measure first, because this needs no new export:

1. Send prompts of about 4K, 16K, 32K, 64K and 128K tokens to `flare run` on idle
   GPUs. `ignition bench` supports only Qwen3.8-27B, so it cannot measure this model.
2. Start each prompt with unique text so that
   `usage.prompt_tokens_details.cached_tokens` stays near zero.
3. Set `return_token_ids` and `return_token_timestamps`, then compute
   `(token_timestamps_ns[0] - performance.ingress_ns) / 1e9` as first-token time and
   `(prompt_tokens - cached_tokens) / first-token time` as prefill throughput.
4. Repeat each size at least three times and record GPU clocks and power alongside
   the raw responses.

Then change one thing at a time and compare on the same workload:

1. **Autotuning.** This package used `--autotune-effort off`. Re-export with
   `default` before attributing any gap to the model. This can affect decode too.
2. **Chunk size.** At 128 tokens, a 32K prompt needs 256 graph replays. Try 1,024 and
   4,096. Each chunk size captures every capacity from `prefill_capacities()`: four
   graphs at 1,024 and ten at 4,096. On the TP16 package the 4,096-token executable
   took 852 MiB per rank and the full set ran out of memory, which
   `--prefill-graph-capacities 128,4096` from `bb7f56af` solved. That flag requires
   merging `k31`. Large forwards have also needed more than 100 GiB of compiler RAM
   per process, and the TP8 graph memory cost is unmeasured.
3. **Attribution.** Before changing execution code, use `qualify/gpu_replay.py` and
   kernel timelines to split time across the 69 KDA layers, 24 sparse layers, experts
   and collectives, as [GPU performance work](performance-work.md) requires.

Optimization beyond capacities and tuning is likely to touch the restricted
mechanisms listed above.

### Context above 129,024 tokens

The per-request limit is the prepared `Plan.context`: `min(recipe.context, pages *
page_tokens)`, where `pages = (supply_bytes - fixed_bytes) // page_bytes`. Here that
is 126 pages of 1,024 tokens. `bootstrap.py` exports it as `max_sequence_tokens`.

The estimate is conservative. `fixed_bytes` includes `graph_bytes = 9 << 29`
(4.5 GiB) from the K3.1 TP8 recipe costs, while this package loads about 0.9 GiB
beyond its weights (88.6 GiB loaded against 702.1 / 8 = 87.8 GiB). Native loading
then allocates far more shared pages than one request may use, so raising KV capacity
or a frontend limit does not enable 1M attention.

To serve 1M, correct and qualify the estimate for the chosen graph catalog, prepare a
separate package and run real long-context generation. The TP8 DSA tests at 16K and
1M in `tests/recipes/test_kimi31_dsa_tp_cuda.py` are operator tests, not full-model
qualification.

## Validation already run

- **CPU.** 501 focused tests passed with 51 skipped. A wider run had 773 passes and
  10 failures, all of which reproduce on the unchanged `a12381ca`: stale fixtures,
  an occupied port 8000, a missing CSF directory and an obsolete peer fixture. Do not
  report the whole suite as passing.
- **GPU components.** `tests/components/test_exl3_weights_cuda.py` passed. Helpers
  outside this repository also checked real layer 1 at both TP8 boundary extents,
  dense, KDA and DSA loading in BF16 and MXFP8, and DSA replay at 1, 33 and 4,096
  rows. These do not establish full-model behavior.
- **Native serving.** `17 * 19` returned 323, a Python `is_prime` request produced a
  coherent function, and two coding requests finished with `finish_reason=stop`.
- **Not run.** KLD, task quality, long generations, images, speculation, concurrency
  above one, and any test of the merged `k31` result.

## Source map

| Concern | Location |
|---|---|
| K3.1 recipe and plan costs | `ignition/recipes/kimi31/recipe.py` |
| Checkpoint parsing and draft pin | `ignition/recipes/kimi3/config.py` |
| Weight loading | `ignition/recipes/kimi3/weights.py`, `ignition/components/weights.py` |
| Draft loader and MLA draft | `ignition/recipes/kimi3/latent_dspark.py` |
| Pages and recurrent rows | `ignition/recipes/kimi3/pool.py` |
| Graph catalog and scratch | `ignition/recipes/kimi3/bootstrap.py` |
| Vision | `ignition/recipes/kimi3/vision.py`, `vision_state.py`, `contract.py` |
| Rank partitions and capacities | `ignition/components/parallel.py`, `capacity.py` |
| Preparation planning | `ignition/planning.py` |
| Native cache fitting | `helios/src/runtime_package/memory.rs` |
