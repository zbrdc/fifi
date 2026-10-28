<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/fifi-logo-dark.svg">
    <img alt="fifi" src="assets/fifi-logo.svg" width="254">
  </picture>
</p>

<p align="center">
  <b>LLM inference in Rust.</b><br>
  Apple Silicon (MLX/Metal) · NVIDIA CUDA
</p>

<p align="center">
  <a href="#build">Build</a> ·
  <a href="#run">Run</a> ·
  <a href="doc/status.md">Status</a> ·
  <a href="#documentation">Docs</a> ·
  <a href="api/openapi.yaml">API</a>
</p>

---

fifi is an LLM inference engine written in Rust.
It runs models on Apple Silicon through MLX and Metal, and on NVIDIA GPUs through CUDA.
Its server, `enginesrv`, provides OpenAI- and Anthropic-compatible APIs (`/v1/chat/completions`, `/v1/messages`) with streaming and tool calls.
One process can host several models.

See [doc/status.md](doc/status.md) for supported models and benchmarks.

## Build

```sh
cd engine
cargo build --release --bin enginesrv --features native-mlx   # Apple Silicon
cargo build --release --bin enginesrv --features native-cuda  # NVIDIA, Linux
```

- `native-mlx` needs macOS on Apple Silicon and the Metal toolchain.
- `native-cuda` needs Linux and `nvcc`.
- Without a backend feature, the build has no GPU backend; it is for development and tests.

Run the tests with `cargo test`.

## Run

```sh
export ENGINESRV_API_KEY=<your key>
target/release/enginesrv --config config/qwen36-mac-studio-36gb.json
```

Set `model_dir` and `tokenizer_dir` in the config first.
[engine/config](engine/config/README.md) has ready-made configs for the Mac Studio and the RTX 3090.

## Documentation

- [Status and benchmarks](doc/status.md)
- [Configuration](engine/config/README.md)
- [Qwen3.6 on CUDA](doc/cuda.md)
- [Flint](doc/flint.md)
- [Nemotron 3.5 Super](doc/nemotron.md)
- [HTTP API](api/openapi.yaml)

## License

fifi is licensed under the [Apache License 2.0](LICENSE).
Third-party code and its licences are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Source code

```sh
git clone https://github.com/zbrdc/fifi.git
```

The Rust crate is still named `inference-engine`.
