<p align="center">
  <img src="docs/images/flare-wordmark.png" alt="Flare — ASCII wordmark above a solar flare" width="912">
</p>

> **Coding agents:** skip this README and read [AGENTS.md](AGENTS.md) directly.

## Build the cooperative-collective runtime

The `astra/cooperative-collective-preparation` branch contains the native loader for the
[DeepSeek V4.1 `phased-cooperative` package](https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-1017-FLARE/tree/phased-cooperative).
It requires execution ABI 6 and prepared contract 3 and removes unselected
kernel nodes after startup tuning while preserving their dependencies.

With Rust and the native build dependencies installed, build from this repository:

```sh
cargo build --release --locked --manifest-path flare/Cargo.toml
./flare/target/release/flare --help
```

The repository Dockerfile also builds the server from source:

```sh
docker build -t flare:phased-cooperative .
```

Building this native server and serving the published package does not require
Python, PyTorch, or a FlashInfer checkout: the package already contains its CUDA
kernels. Follow the linked model card for download and serving flags, substituting
the executable or image built here. Preparing another model export uses the
companion FlashInfer revision pinned in `pyproject.toml` and `uv.lock`.
The native loader remains independent of the preparation environment.

With the source checkpoint available and four free SM120 GPUs, prepare the
DeepSeek V4.1 configuration used by the published CSF package:

```sh
uv sync --locked --extra prepare --extra test
.venv/bin/python -m ignition export deepseek-v4.1-flash \
  --gpus 0,1,2,3 --max-concurrency 32 --image-concurrency 1 \
  --mtp-draft-tokens 7 --speculative-token-capacity 128 \
  --speculative-graph-budget 64 --prefill-chunk-size 4096 \
  --mla-compute-mode bf16 --expert-scale-compression csf \
  --autotune-effort default --compile-workers 24 \
  --collective-tuning runtime \
  --collective-ladder ignition/preparation/ladders/gb202_pcie_tp4.json \
  --draft-head-quantization nvfp4 --markov-head-quantization fp8 \
  --vocabulary-parallel-sampling --verbose
```

Use `--expert-scale-compression native` for uncompressed expert scales.
`--compile-workers` controls preparation CPU resources; it does not change serving
capacity. Add `--reuse-weights` only when the selected cached package has matching
weight layouts, including the scale and draft-head formats. Automatic tuning on
another host can select different kernel plans. The published measurements apply
to the tested packages, not every subsequent export.

Flare is a local inference server for NVIDIA SM120/SM121 GPUs, built for
long conversations, coding agents, and repeated tool use. Its goal is to make
the memory, bandwidth, and compute in workstations and compact systems such
as DGX Spark go further, while keeping generation responsive.

This is a research prototype, and is made available via open-source to enable
technical collaboration and collect early feedback.

The design rests on a few principles:

- **Embrace the standard PyTorch ecosystem at build time; serve from Rust.**
  Ignition uses PyTorch, Transformers, and FlashInfer/b12x to load models,
  tune kernels, compile forwards, and capture CUDA graphs into a runtime package.
  Helios, the Rust runtime, serves that package without Python. Compilation and
  graph capture finish before serving.

- **Build on the model ecosystem.** Reuse canonical model implementations
  wherever their dataflow fits, and specialize the parts that need different
  layouts or execution. Model development keeps the tools and libraries that
  make the Python ecosystem productive.

- **Make memory and state explicit.** Plan buffers and execution capacities
  ahead of time. Treat conversation state—including recurrent state alongside
  KV caches—as a whole, so reuse, branching, and resuming preserve it correctly.

- **Keep the runtime small and shared.** Model recipes own model computation;
  the runtime owns scheduling, cache storage, and request handling. A focused
  hardware and kernel stack keeps those boundaries manageable.

[vLLM](https://docs.vllm.ai/en/latest/) covers a much broader range of models,
hardware, and deployment patterns. Flare trades that breadth for a focused
serving contract around selected models and local workloads. Flare is not, and
will never, be a replacement for production inference englines like vLLM.

Think of Flare as a specialized option when its hardware and model coverage
fit your machine.

Start with [model preparation and serving](docs/model-preparation.md).
See the [CLI guide](docs/cli-design.md) for configuration and the
[core design](docs/core-design.md) for the architectural reasoning.
