# Configuration file

`enginesrv --config <file>` reads one JSON file.
Startup refuses unknown keys and values outside the ranges below.
When a model does not fit in memory, startup refuses the load; it does not lower a setting to make it fit.

Example files are in [`engine/config`](../../engine/config/README.md).

## Top level

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `listen` | yes | | Address and port, for example `127.0.0.1:18791`. The address must be `127.0.0.1` or `::1`. Port 8788 is refused. |
| `api_key_env` | yes | | Name of the environment variable that holds the API key. Startup fails if the variable is unset or empty. |
| `prefix_reuse` | no | `false` | Keep finished conversation prefixes so that a follow-up turn only processes its new tokens. Applies to every engine that supports it. See [How conversation reuse works](../explanation/prefix-reuse.md). |
| `engines` | yes | | The engines to serve, one object each. |

## Engine entry

| Key | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Engine name. Clients can select the engine with it (`x-engine`). |
| `family` | yes | The model family; see [Models](models.md). |
| `model_dir` | yes | Directory of the checkpoint. |
| `tokenizer_dir` | yes | Directory with `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and `generation_config.json`. For Qwen3.6, Flint and Nemotron, startup checks each file against the hashes of the tested release. |
| `model` | no | The name clients send in the `model` field. Defaults to the family name. |
| `qwen36` | no | [Qwen3.6 options](#qwen36). |
| `flint` | CUDA only | [Flint options](#flint). |
| `nemotron` | `nemotron35` only, required | [Nemotron options](#nemotron). |
| `fake` | no | [Test engine options](#fake). |
| `context_limit` | no | `fake` only: context limit in tokens. Default 4,096. |
| `flash_next` | no | Options of `qwen3_8_flash_next`, which is not validated end to end. |
| `think_replay` | no | `false` drops the reasoning that clients resend with earlier assistant turns from the prompt, for this engine only. `true` is refused; leave the key out for the default. |
| `notice_toolwrap` | no | `false` renders system messages that follow an assistant turn as plain system text, for this engine only. `true` is refused. See `ENGINESRV_NOTICE_TOOLWRAP`. |

## `qwen36`

Options for `qwen3_6_35b_a3b`. "MLX" means Apple Silicon.

| Key | Backend | Default | Meaning |
| --- | --- | --- | --- |
| `context_limit` | both | MLX 262,144; CUDA 32,768 | Context limit in tokens, from 2. |
| `kv_dtype` | both | MLX `q8`; CUDA `f32` | KV cache format: `q8` (8-bit), `f32`, or `bf16` (MLX only). On CUDA, `f32` stores the cache in FP16. |
| `activation_dtype` | MLX | `bf16` | Decode activations: `bf16` or `f32`. CUDA accepts the key and ignores it. |
| `cache_limit_bytes` | MLX | none | MLX allocator cache limit. Falls back to `Q36_CACHE_LIMIT`; on MLX one of the two is required. |
| `session_cap_bytes` | MLX | 4 GiB | Memory for the live session and saved prefixes. |
| `prefill_budget_bytes` | MLX | off | Memory budget for processing the prompt in chunks. Without it, the prompt is processed one token per step. |
| `prefill_session_reserve_bytes` | MLX | none | Memory that prompt processing leaves free. Required with `prefill_budget_bytes`, refused without it. |
| `prefill_chunk` | MLX | 256 | Largest prompt chunk, 1 to 2,048 rows. The engine picks the largest chunk that fits the budget. |
| `compile` | MLX | `false` | Compile the decode step with `mlx_compile`: 9.6 to 13% faster decoding. Logits differ from the uncompiled path by at most 5e-5. |
| `max_seqs` | MLX | 1 | Sequences decoded at once, 1 to 8. Above 1 it needs `compile: true`, a `prefill_chunk` of at least 4, and a `prefill_budget_bytes` larger than both `prefill_session_reserve_bytes` and `session_cap_bytes`. |
| `prefix_store_entries` | MLX | 6 | Saved prefixes kept with `prefix_reuse` on, 1 to 4,096. |
| `capture` | CUDA | `false` | Run each decode step as a CUDA graph. |
| `kv_offload_host_bytes` | CUDA | off | Keep the KV cache in host memory, up to this many bytes. See [below](#host-memory-kv-cache-cuda). |
| `tiered` | CUDA | off | Keep only some experts on the GPU. See [below](#tiered-experts-cuda). |

CUDA refuses `prefill_budget_bytes` and `prefill_session_reserve_bytes`.
It ignores `cache_limit_bytes`, `session_cap_bytes`, `prefill_chunk`, `compile` and `prefix_store_entries`, and logs one line for each.
MLX ignores `capture` with a log line, and refuses `tiered`.
`prefix_store_entries` is ignored with a log line when `prefix_reuse` is off.

### Host-memory KV cache (CUDA)

`kv_offload_host_bytes` keeps the KV cache in host memory for contexts that do not fit on the GPU.
It needs `"kv_dtype": "q8"`, `"capture": false`, `"prefix_reuse": false` and `Q36_CUDA_CHUNK_PREFILL=0`.
It reloads the cache from host memory for every token, so it is slow: an 8K-token replay decoded at 34.95 tokens/s, against 147.14 tokens/s with the cache on the GPU.
`Q36_CUDA_PAGED_DEVICE_FIRST=1` keeps as much of the cache on the GPU as fits after a 1 GiB reserve.

### Tiered experts (CUDA)

`tiered` keeps a fixed number of expert slots per layer on the GPU and runs the other experts on the CPU from the memory-mapped model file.
Do not modify or replace the model file while the server runs.

| Key | Default | Meaning |
| --- | --- | --- |
| `slots` | required | Expert slots per layer on the GPU, 8 to 256. |
| `threads` | 6 | CPU worker threads, 1 to 8. |
| `timeout_ms` | 10,000 | Longest wait for one layer's CPU experts, 1 to 600,000. |
| `preload` | `false` | `true` fills each layer's slots with its lowest-numbered experts before the first token. By default the slots start empty and fill as experts are used. |

Tiered mode refuses `capture: true` and `kv_dtype: q8`.
It processes prompts one token per step and, by default, keeps no saved prefixes in host memory.

## `flint`

Required for Flint on CUDA; refused on Apple Silicon.

| Key | Required | Values | Meaning |
| --- | --- | --- | --- |
| `context_limit` | yes | 1 to 262,144 | Context limit in tokens. |
| `kv_profile` | yes | `"f16"`, `"q8"` | KV cache format. `q8` (int8 in groups of 32 with an FP16 scale) is experimental; its output quality is not measured. |
| `prefill_chunk_size` | no | 1 to 8, default 8 | Prompt rows per step. `1` processes one token at a time. |

## `nemotron`

Required for `nemotron35`. All four keys are required.

| Key | Values | Meaning |
| --- | --- | --- |
| `backend` | `"cuda"`, `"cpu_reference"` | `cuda` needs a Linux build with `--features native-cuda` and accepts greedy decoding only. `cpu_reference` runs on the CPU and is slow. |
| `context_limit` | 1 or more | Context limit in tokens. |
| `cache_entries` | 1 or more | Expert weight tensors kept decoded in host memory. |
| `prefill_chunk` | 1 to `context_limit` | Prompt tokens per step. |

## `fake`

The `fake` family replies with fixed text and loads no model. It is for tests and for trying the server.

| Key | Default | Meaning |
| --- | --- | --- |
| `reply` | `Hello from the fake engine.` | The text of every reply. |
| `token_delay_ms` | 0 | Pause before each token, in milliseconds. |
