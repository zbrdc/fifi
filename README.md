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
  <a href="doc/getting-started.md">Getting started</a> ·
  <a href="#documentation">Docs</a> ·
  <a href="doc/reference/models.md">Models</a> ·
  <a href="doc/reference/http-api.md">API</a>
</p>

---

fifi is an LLM inference engine written in Rust.
It runs models on Apple Silicon through MLX and Metal, and on NVIDIA GPUs through CUDA.
Its server, `enginesrv`, speaks the OpenAI chat completions API and the Anthropic Messages API (`/v1/chat/completions`, `/v1/messages`), with streaming and tool calls.
One process can host several models.

## Models

| Model | Apple Silicon | NVIDIA CUDA |
| --- | --- | --- |
| Qwen3.6-35B-A3B | yes | yes |
| Flint | yes | pilot |
| Nemotron 3.5 Super | | yes |

See [Models](doc/reference/models.md) for checkpoints and context limits.

## Quick start

New to fifi? The [tutorial](doc/getting-started.md) runs the server with a test engine, without a GPU or a model download.

Build with the backend for your machine:

```sh
git clone https://github.com/zbrdc/fifi.git
cd fifi/engine
cargo build --release --bin enginesrv --features native-mlx   # Apple Silicon
cargo build --release --bin enginesrv --features native-cuda  # NVIDIA GPU, Linux
```

Pick a file from [engine/config](engine/config/README.md), set its `model_dir` and `tokenizer_dir`, and start the server:

```sh
export ENGINESRV_API_KEY=my-secret-key
target/release/enginesrv --config config/qwen36-mac-studio-36gb.json
```

It listens on `http://127.0.0.1:18791`. Send the key as `Authorization: Bearer <key>`:

```sh
curl http://127.0.0.1:18791/v1/chat/completions \
  -H "Authorization: Bearer $ENGINESRV_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "fifi-q36", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Run the tests with `cargo test`.

## Documentation

**Tutorial**

- [Getting started](doc/getting-started.md)

**How-to guides**

- [Run Qwen3.6](doc/how-to/qwen36.md)
- [Run Flint](doc/how-to/flint.md)
- [Run Nemotron 3.5 Super](doc/how-to/nemotron.md)
- [Connect a client](doc/how-to/clients.md)

**Reference**

- [Configuration file](doc/reference/configuration.md)
- [Environment variables](doc/reference/environment.md)
- [HTTP API](doc/reference/http-api.md) and the [OpenAPI contract](api/openapi.yaml)
- [Models](doc/reference/models.md)

**Explanation**

- [How fifi plans memory](doc/explanation/memory.md)
- [How conversation reuse works](doc/explanation/prefix-reuse.md)
- [Benchmarks](doc/explanation/benchmarks.md)

## License

fifi is licensed under the [Apache License 2.0](LICENSE).
Third-party code and its licences are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The Rust crate is still named `inference-engine`.
