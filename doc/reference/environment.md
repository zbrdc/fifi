# Environment variables

Set serving variables before starting `enginesrv`, unless an entry says otherwise.
The API key variable comes from `api_key_env` in the [configuration file](configuration.md).

Two kinds of values appear below:

- An unset **switch** keeps its default. A set switch is on only for `1`, `true`, or `on` (ignoring spaces and case). Any other value turns it off.
- A variable with **accepted values** refuses startup when set to anything else, unless the table says otherwise.

For internal diagnostics, tracing, and experimental kernel flags, see [Internal and experimental environment variables](internal-environment.md).

## Build

| Variable | Default | Meaning |
| --- | --- | --- |
| `INFERENCE_CUDA_ARCHS` | `75,86,89` (`72` on aarch64) | GPU architecture targets of CUDA kernels for `cargo build --features native-cuda`. The default covers RTX 20, 30, and 40 series. A single target such as `86` builds faster and smaller. Native `120` for RTX 50 series needs CUDA 12.8 or newer. |
| `FIFI_CUDA_LINEINFO` | unset (off) | Adds `-lineinfo` to `nvcc` flags during `cargo build --features native-cuda` for `compute-sanitizer` and profilers. Leave unset for standard builds. Rebuild to change. |

NVIDIA documents [GPU compute capabilities](https://developer.nvidia.com/cuda/gpus), [PTX compatibility rules](https://docs.nvidia.com/cuda/archive/12.8.1/blackwell-compatibility-guide/index.html), and [CUDA 12.8 compiler support for SM 120](https://docs.nvidia.com/cuda/archive/12.8.2/cuda-features-archive/index.html).

## Server

| Variable | Default | Meaning |
| --- | --- | --- |
| `ENGINESRV_WARM_TURNS` | on (switch) | With `prefix_reuse`, remembers the last conversation's tokens so the next turn only tokenizes new messages. |
| `ENGINESRV_SEED_RESENDS` | on (switch) | With warm turns on, keeps token IDs of a freshly rendered prompt so an identical retry skips rendering and tokenization. |
| `ENGINESRV_RESEND_EXTRA_BYTES` | `0` | With seeded resends on, keeps one displaced conversation's token result if its allocation fits this per-engine byte budget. |
| `ENGINESRV_PREFIX_BLOCK_MATCH` | on (switch) | With `prefix_reuse`, saves one block checkpoint at a multiple of `ENGINESRV_PREFIX_BLOCK_TOKENS` before the conversation end. |
| `ENGINESRV_PREFIX_BLOCK_TOKENS` | 4,096 | Spacing of block cut points, in tokens. Values below 1 select the default. |
| `ENGINESRV_CACHE_USAGE` | on | Report reused prompt tokens in API usage fields. Only `1`, `true`, and `on` keep it on when set. |
| `ENGINESRV_THINK_REPLAY` | on (switch) | Qwen3.6 only: keep reasoning that clients resend with earlier assistant turns in the prompt. |
| `ENGINESRV_NOTICE_TOOLWRAP` | on (switch) | Qwen3.6 only, with think replay on: render system messages after an assistant turn as tool responses. |

## Model stores and the hub

Read by `enginesrv serve` when `--model` is a name, and by the model page of the web console. See [Serve a model without a config file](../how-to/serve-a-model.md).

| Variable | Default | Meaning |
| --- | --- | --- |
| `FIFI_MODELS_DIR` | `~/models` | The models directory, searched first for a model named without a path. |
| `OLLAMA_MODELS` | `~/.ollama/models` | The Ollama store, searched second. |
| `FIFI_HF_HUB` | unset | Location of the Hugging Face hub cache. Wins over `HF_HUB_CACHE`, which wins over `HF_HOME/hub`. |
| `FIFI_HF_ENDPOINT` | unset | Hub address for a mirror. Wins over `HF_ENDPOINT`. |
| `HF_TOKEN` | unset | Access token for gated or private repositories. Reads `HUGGING_FACE_HUB_TOKEN` when unset. |

## Apple Silicon

| Variable | Default | Meaning |
| --- | --- | --- |
| `FIFI_MLX_AUTO_BUDGET` | off, on for Inkling | `1` sizes prefill from Metal's recommended working set at startup for one MLX model per server. `0` opts out for Inkling unless a ceiling is set. |
| `FIFI_MLX_WORKING_SET_MIB` | unset | Positive integer ceiling for an MLX memory budget, in MiB. Caps the budget at the lowest of host budget, Metal working set, and this value. |
| `FIFI_INKLING2_MINI_MLX` | off | Set to `1` for experimental [Inkling 2 Mini text serving](../how-to/inkling.md) on Apple Silicon. |
| `FIFI_Q36_MLX_DISK_EXPERTS` | automatic | Where Qwen3.6 keeps routed experts (`1` forces disk streaming, `0` forces residency, unset chooses automatically from the budget). |
| `Q36_CACHE_LIMIT` | none | Fallback for `cache_limit_bytes`, in bytes. |
| `Q36_FUSED_PREFILL` | on | Fused kernels for prompt processing. Only `0` turns them off. |
| `Q36_FLASH_PREFILL` | on | Fused flash-prefill kernel for composite prefix attention. `0` turns it off. |
| `Q36_FLASH_DELTA` | on | Flash-delta chunked WY-form kernel for prefill chunks >= 16 rows. `0` turns it off. |

## NVIDIA CUDA (Qwen3.6, Flash-Next, Bolt)

For disk prefix-cache variables (`FIFI_KV_CACHE_*`), see [Prefix cache on disk](configuration.md#prefix-cache-on-disk-cuda).

| Variable | Default | Accepted values | Meaning |
| --- | --- | --- | --- |
| `Q36_VERIFY_GGUF_SHA256` | `0` | `0`, `1` | `1` checks the built-in GGUF SHA-256 at load. Explicit `qwen36.gguf_sha256` overrides this. |
| `Q36_ALLOW_UNPINNED_GGUF` | `1` | `0`, `1` | `0` serves only the pinned file. `1` or unset serves the directory's largest valid `.gguf` when the pinned file is absent. |
| `Q36_CUDA_CHUNK_PREFILL` | `1` | `0`, `1` | `0` processes the prompt one token at a time instead of in chunks. |
| `Q36_CUDA_CHUNK_ROWS` | `1024` | `16`, `128`, `512`, `1024`, `2048` | Largest prompt chunk for the `q8` cache when chunking is on. |
| `Q36_CUDA_PINNED_PREFIX` | on | only `1` is on | Any other value keeps saved prefixes in pageable host memory and turns off the GPU copy. |
| `Q36_CUDA_PINNED_PREFIX_BYTES` | smaller of 9 slots and half of available host memory | byte count | Size of the pinned host pool for saved prefixes (up to 8 retained prefixes). `0` keeps none. |
| `Q36_CUDA_PAGED_DEVICE_FIRST` | `0` | `0`, `1` | With `kv_offload_host_bytes`: keep as much of the KV cache on the GPU as fits after a 1 GiB reserve. |
| `FLASH_NEXT_ARTIFACT_MANIFEST` | unset | path | Path of the Flash-Next artifact manifest, read once at engine start. |
| `FLASH_NEXT_CACHE_LIMIT` | unset | byte count | Flash-Next cache limit in bytes, read once at engine start. |
| `FIFI_FLASH_NEXT_BANK_MODE` | `auto` | `auto`, `inclusive`, `exclusive`, `disk` | Expert-bank store mode (`auto` picks inclusive, else exclusive, else `disk` ring). |
| `INFERENCE_FLASH_NEXT_PREFIX_RESUME` | on | switch | Prefix reuse for Flash-Next CUDA (`0`/`false`/`off` opt out). Needs `prefix_reuse` enabled or unset. |
| `FIFI_BOLT_ATTN_SPLIT` | on | switch | Split decode attention for Bolt (`0` or `off` restores single-pass kernel). |
| `FIFI_BOLT_ATTN_SPLITS` | device-derived | `1` to `128` | Overrides Bolt decode attention split count. Changes prefix keys. |

## Other families

| Variable | Default | Accepted values | Meaning |
| --- | --- | --- | --- |
| `FIFI_SENSENOVA_CUDA` | off | only `1` is on | Serves SenseNova-Loop from one CUDA device instead of the CPU (needs `native-cuda`). |
| `FIFI_FLINT_MLX_CONTEXT_LIMIT` | `32768` | `32768`, `8192` | Flint context limit on Apple Silicon. Any other value selects 8,192. |
| `FIFI_FLINT_MLX_BATCH_PREFILL` | `1024` | `1024`, `512`, `0` | Flint prompt chunk size (`0` = one token at a time). Any other value selects 512. |
| `FIFI_FLINT_PREFIX_REUSE` | off | switch | Experimental: keep Flint prefix snapshots between turns. Also needs `"prefix_reuse": true`. |
| `FIFI_FLINT_PREFIX_BUDGET_BYTES` | physical memory minus weights and reserve | positive byte count | Memory cap for Flint prefix snapshots. `0` or an unreadable value refuses startup. |
| `FIFI_NEM35_CPU_NEON` | off | `1` | Uses Arm NEON NVFP4 dot product in the mapped CPU expert pool for Nemotron 3.5 on Apple Silicon. |
