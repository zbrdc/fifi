<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/fifi-logo-dark.svg">
    <img alt="fifi" src="assets/fifi-logo.svg" width="254">
  </picture>
</p>

# fifi

fifi is a Rust inference engine for local coding agents.
It runs on Apple Silicon (MLX and Metal) and NVIDIA CUDA.
The `enginesrv` binary serves an OpenAI-compatible `/v1/chat/completions` route with streaming and tool calls.
One process can hold several engines. The shared memory plan and prefix reuse are in progress.

## Current status

The primary target remains `Qwen/Qwen3.8-Flash-Next` on a 36 GB Mac Studio, with a planned, bounded SSD tier.
Task-branch work below is separate from `main`.

| Model and backend | Implementation status | Remaining work |
| --- | --- | --- |
| `Qwen/Qwen3.8-Flash-Next`, Apple Silicon | Work is paused. SSD expert streaming remains planned. | Feasibility and end-to-end validation remain incomplete. |
| `Qwen3.6-35B-A3B`, Apple Silicon | `main` contains resident MLX execution and HTTP serving. | Full-context and end-to-end validation remain incomplete. |
| `Qwen3.6-35B-A3B`, CUDA | `main` contains Q4_K_M loading and batched prefill through serving. | Correctness, native execution, and full-context validation continue. |
| `Mellum2.1-12B-A2.5B`, CUDA | Decode-chain checks use stand-in weights. Real-checkpoint family execution is not integrated into `main`. | Real-checkpoint execution and end-to-end validation remain open. |
| `Qwen3.8-27B` | `main` contains an oQ6 dense-projection loading proof. Dense loading and decode work continue on `task/qwen38-dense-compile`. | Native compilation, resource admission, and source checks remain open. Full-model validation remains incomplete. |
| Gemma4 12B Humanizer, CUDA | Narrow generation checks and numerical checks at the first position passed on a task branch. | Integration remains on hold for a shared 8-bit quantization policy regression. |
| Gemma4 26B-A4B, CUDA | A task branch is adding text-only execution, configuration, loading, and mixture-of-experts routing using the selected official Google Q4_0 artifact. | Configuration checks do not qualify model execution or the Q4_0 runtime. |
| Nemotron3.5, CUDA | A task branch repairs staged weight projection and resource cleanup. | Memory-limit enforcement, native execution, staged transfers, and real-model checks remain open. |

### Weight formats

These formats are separate from model-level validation.

- MLX affine and oQ safetensors, with the quantization read from the checkpoint metadata.
- GGUF F16, Q4_K, Q6_K, and Q8_0. Q5_0 is in progress and is not in `main`.

### Cache work

The supplied Qwen3.6 Mac Studio configuration retains an 8-bit key-value (KV) cache.

A separate CPU-only prototype routes Q8 cache bytes. It passed behavior and fault-detection checks and independent review, with notes. It is not integrated into the engine or HTTP serving.

Native operation ordering, physical memory peaks, attention math, and useful performance remain unvalidated.

## Build

The crate is in `engine/`. The default build is portable.

```sh
cd engine
cargo build --release --bin enginesrv
cargo test
```

Two features select a backend.

- `native-mlx` builds the MLX adapter. It needs Apple Silicon macOS and the Metal toolchain.
  `native-ane` adds the optional ANE decision head.
- `native-cuda` builds the CUDA backend with `nvcc`. It needs Linux.

Start the server with `enginesrv --config <path>`.

## HTTP contract

- [api/openapi.yaml](api/openapi.yaml) is the HTTP contract that `enginesrv` serves.

## License

The Apache License 2.0 covers fifi. See [LICENSE](LICENSE).
Third-party notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Source code

The repository is public.

```sh
git clone git@github.com:zbrdc/fifi.git
```

## Credits

[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists the copied and derived third-party code that it covers.
The vendored `mlx-sys` and `safetensors` crates keep their own licence files in `engine/vendor/`.

The crate is still named `inference-engine`. The product name `fifi` arrived later, and the crate keeps its old name for now.
