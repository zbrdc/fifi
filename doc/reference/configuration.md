# Configuration file

`enginesrv --config <file>` reads one JSON file.
Startup refuses unknown keys, out-of-range values, and models that do not fit in memory.

Example files live in [`engine/config`](../../engine/config/README.md).

## Top level

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `listen` | yes | | Loopback address and port, for example `127.0.0.1:18791` (`127.0.0.1` or `::1` only). Startup refuses port 8788. |
| `api_key_env` | yes | | Name of the environment variable holding the API key. Startup fails if unset or empty. |
| `prefix_reuse` | no | unset | Reuses finished conversation prefixes. Unset turns on for Flash-Next, Qwen3.6, and Bolt on CUDA, and for Qwen3.6 on Apple Silicon when host memory fits. Off elsewhere. `true` or `false` overrides all engines. See [How conversation reuse works](../explanation/prefix-reuse.md). |
| `web_dir` | no | | Path to built web frontend (`web/dist`). Serves `GET /` (`index.html`) and `GET /assets/...`. Requires `index.html`. |
| `engines` | yes | | Array of engine objects to serve. |

## Engine entry

| Key | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Unique engine name. Clients select it with `x-engine`. |
| `family` | yes | Model family name. See [Models](models.md). |
| `model_dir` | yes | Checkpoint directory on local storage. See [Local storage](#local-storage). |
| `tokenizer_dir` | yes | Directory holding `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, and `generation_config.json`. |
| `model` | no | Public model name in `/v1/models` and requests. Defaults to `family`. |
| `qwen36` | no | [Qwen3.6 options](#qwen36). |
| `nemotron` | `nemotron35` only | [Nemotron options](#nemotron). |
| `fake` | no | [Test engine options](#fake). |
| `context_limit` | no | `fake` only: context limit in tokens (default 4,096). |
| `flash_next` | no | [Flash-Next options](#flash_next), including [MLX prefill](#flash-next-mlx-prefill). |
| `think_replay` | no | `false` drops prior assistant reasoning from the prompt for this engine. `true` fails startup. |
| `notice_toolwrap` | no | `false` renders system messages after assistant turns as plain system text. `true` fails startup. |

### Local storage

Model weights must reside on local storage.
Startup refuses any `model_dir` on a network filesystem (NFS, SMB, AFP, WebDAV, sshfs, Ceph), an empty automount, or a symlink pointing to one.
If startup cannot prove local storage, the engine fails to load.
Startup skips this check for the tokenizer directory because it holds only small text files.

### CUDA FP16 prefill attention

FP16 prefill attention uses the default PTX path.
Set [`FIFI_CUDA_ATTN_PREFILL_ADAPT=1`](internal-environment.md#build-and-cuda-diagnostics) to pick tensor-core or compact SIMT kernels from device shared-memory and thread limits.
Requires compatible PTX via [`INFERENCE_CUDA_ARCHS`](environment.md#build). Tested on RTX 2080 Ti and RTX 3080 only.

## Flash-Next MLX prefill

Set `flash_next.prefill_budget_bytes` to enable chunked prompt processing on MLX (omitting it processes one token at a time).
Startup chooses the largest fitting chunk up to `flash_next.prefill_chunk` (default 16, range 1–2,048).
For wave-cap, prefix-budget, and decode-cache tuning flags (`FIFI_FLASH_NEXT_MLX_*`), see [Internal and experimental environment variables](internal-environment.md).

## `qwen36`

Options for `qwen3_6_35b_a3b`. "MLX" means Apple Silicon.

| Key | Backend | Default | Meaning |
| --- | --- | --- | --- |
| `context_limit` | both | MLX 262,144, or CUDA GGUF auto up to 262,144 | Context limit in tokens (minimum 2). Single-card CUDA GGUF auto-sizes to the largest multiple of 1,024 that fits free VRAM. Explicit values stay fixed. |
| `kv_dtype` | both | MLX `q8`, or CUDA GGUF auto | KV cache format: `q8`, `f32` (stored as FP16 on CUDA), or `bf16` (MLX only). Unset on CUDA GGUF picks `f32` when context >= 131,072, else `q8`. |
| `activation_dtype` | MLX | `bf16` | Decode activations: `bf16` or `f32`. Ignored on CUDA. |
| `cache_limit_bytes` | MLX | none | MLX allocator cache limit in bytes. Falls back to `Q36_CACHE_LIMIT`. MLX requires one of the two. |
| `session_cap_bytes` | MLX | 4 GiB | Memory cap for live session and saved prefixes. |
| `prefill_budget_bytes` | MLX | off | Memory budget for chunked prefill. Without it, prefill runs one token per step. |
| `prefill_session_reserve_bytes` | MLX | derived | Free memory reserved during prefill. Requires `prefill_budget_bytes`. |
| `prefill_chunk` | MLX | 256 | Maximum prompt chunk size, 1 to 2,048 rows. |
| `compile` | MLX | `false` | Compile the decode step with `mlx_compile`. |
| `compile_program_limit` | MLX | 8 | Compiled program key budget per adapter (minimum 1). Refused on CUDA. |
| `max_seqs` | MLX | 4 (MoE compiled) or 1 | Concurrent decode sequences, 1 to 8. Values above 1 require `compile: true`, `prefill_chunk` >= 4, and funded budgets. |
| `prefix_store_entries` | MLX | 6 | Saved prefixes kept when `prefix_reuse` is on (1 to 4,096). |
| `weight_format` | CUDA | `gguf` | Checkpoint format (`gguf` only). Startup refuses `exl3` and all other values. |
| `gguf_sha256` | CUDA GGUF | off | Expected 64-hex-digit SHA-256 of the GGUF file. Refused on MLX. See [content pinning](#gguf-content-pinning-cuda). |
| `capture` | CUDA | `false` | Run each decode step as a CUDA graph. |
| `cuda_devices` | CUDA | `[0]` | Ascending CUDA device indices (at most 4). Multi-card requires identical compute capability and peer access, and disables `tiered`, `kv_offload_host_bytes`, `capture`, and speculation. |
| `kv_offload_host_bytes` | CUDA | off | Keep the KV cache in host RAM up to this byte cap. See [Host-memory KV cache](#host-memory-kv-cache-cuda). |
| `kv_cache` | CUDA | on (user cache dir) | Disk prefix cache config (`dir`, `quota_bytes`, `enabled`). See [Prefix cache on disk](#prefix-cache-on-disk-cuda). |
| `tiered` | CUDA | off | Keep selected expert slots in VRAM and run remaining experts on CPU. See [Tiered experts](#tiered-experts-cuda). |
| `draft_mode` | MLX | `disabled` | Speculative draft source on MLX GGUF: `disabled`, `ngram`, or `mtp` (not usable yet). Refused on CUDA. |
| `num_draft_tokens` | MLX | 2 | Maximum draft tokens per check window (1 to 16). |
| `dynamic_draft_tokens` | MLX | `false` | Shorten check windows dynamically from MTP confidence (`ngram` ignores it). |
| `draft_confidence` | MLX | 0.4 | Minimum cumulative confidence product (between 0 and 1) when `dynamic_draft_tokens` is true. |
| `draft_max_rows` | MLX | 8 | Maximum rows per check window, including pending token (2 to 64, max 8 when `draft_mode` is active). |
| `hybrid_draft_tokens` | MLX | 0 | Draft tokens copied from repeated context passages (`0`, or `ngram_match_min` to 256). |
| `ngram_match_min` | MLX | 2 | Minimum repeated tail length for n-gram drafting (1 to 64). |
| `mtp_model_dir` | MLX | none | Pack directory holding `mtp.*` tensors. Required when `draft_mode` is `mtp`, refused otherwise. |
| `draft_rollback` | MLX | `host` | Rollback method (`host` only). |
| `draft_verify` | MLX | `eager` | Draft check mode (`eager` only). |

CUDA refuses `prefill_budget_bytes`, `prefill_session_reserve_bytes`, and all `draft_*` / `num_draft_tokens` / `hybrid_draft_tokens` / `ngram_match_min` / `mtp_model_dir` keys.
MLX ignores `capture` and refuses `tiered`.

### Prefix cache on disk (CUDA)

`kv_cache` persists saved prefixes to local disk for single-card Qwen3.6 GGUF and Bolt engines.
You can also set `qwen36.kv_cache` or `gemma4.kv_cache`, but not both.

```json
"kv_cache": { "dir": "/var/cache/fifi/kv", "quota_bytes": 21474836480 }
```

| Member | Default | Meaning |
| --- | --- | --- |
| `dir` | `$XDG_CACHE_HOME/fifi/kv/<family>` | Absolute owner-only (`0700`) directory supporting `O_DIRECT`. Startup creates it if missing and tests direct I/O. |
| `quota_bytes` | 25% of free space (4 GiB min, 64 GiB max) | Maximum disk usage, at least 1 GiB. Startup evicts oldest entries if usage exceeds `quota_bytes`. |
| `enabled` | `true` | `false` disables disk writes unless `FIFI_KV_CACHE=1` overrides it. |

Environment overrides: `FIFI_KV_CACHE` (`0`/`1`), `FIFI_KV_CACHE_DIR`, `FIFI_KV_CACHE_QUOTA_BYTES`, and `FIFI_KV_CACHE_HOST_BYTES` (host-memory spill threshold).
Legacy `Q36_KV_CACHE_*` names also work.
Run `enginesrv --config <file> --purge-kv-cache` to clear existing cache directories before starting.

Entries hold exact tokens and state, verified on read with CRC32C, SHA-256, and load-identity digests.
Writes occur only under host memory pressure or on graceful shutdown (`SIGINT`/`SIGTERM`), skipping entries under 1,024 positions and keeping 512 MiB free on disk.

### GGUF content pinning (CUDA)

Set `qwen36.gguf_sha256` to a 64-hex-digit SHA-256 digest to bind startup to exact GGUF file bytes.
Startup verifies the digest before loading weights and refuses mismatches.
This overrides `Q36_VERIFY_GGUF_SHA256`.

### Host-memory KV cache (CUDA)

`kv_offload_host_bytes` keeps the KV cache in host RAM for contexts exceeding VRAM.
Requires `"kv_dtype": "q8"`, `"capture": false`, `"prefix_reuse": false`, and `Q36_CUDA_CHUNK_PREFILL=0`.
Set `Q36_CUDA_PAGED_DEVICE_FIRST=1` to keep as much KV cache in VRAM as fits after a 1 GiB reserve.

### Tiered experts (CUDA)

`tiered` keeps a fixed number of expert slots per layer in VRAM and runs other experts on CPU from the memory-mapped model file.
Do not change or replace the model file while running.
Tiered mode refuses `capture: true` and `kv_dtype: q8`, processes prompts one token per step, and keeps no saved prefixes by default.

| Key | Default | Meaning |
| --- | --- | --- |
| `slots` | required | Exact count (`8` to `256`) or `"auto"` (largest count leaving 1 GiB free VRAM). |
| `threads` | 6 | CPU worker threads, 1 to 8. |
| `timeout_ms` | 10,000 | Maximum wait per layer's CPU experts in milliseconds (1 to 600,000). |
| `preload` | `true` | `true` fills slots before the first token. `false` starts empty and fills on demand. |
| `swaps_milli` | 250 | Bank swaps per token in thousandths (0 to 8,000). |

## `flash_next`

Options for `qwen3_8_flash_next` (not checked end to end).

| Key | Default | Meaning |
| --- | --- | --- |
| `cuda_decode` | `true` on `native-cuda` Linux unless MLX keys exist, else `false` | Serves the model on CUDA (experimental). |
| `prefill_chunk` | 16 | Rows per chunked-prefill piece, 1 to 2,048. |
| `prefill_rows` | 256 | Rows per device prefill step, 1 to 2,048 (`1` selects per-token prefill). |
| `context` | auto | K/V positions, a multiple of 4 from 4 to 262,144. |
| `kv_format` | `f16` | K/V storage on card (`f16` only, as `int8` fails). |

## `nemotron`

`nemotron35` requires `backend`, `context_limit`, and `prefill_chunk`. You may omit `cache_entries`.

| Key | Values | Meaning |
| --- | --- | --- |
| `backend` | `"cuda"`, `"cpu_reference"` | `cuda` requires `--features native-cuda` on Linux and accepts greedy decoding only. `cpu_reference` runs on CPU. |
| `context_limit` | 1 or more | Context limit in tokens. |
| `cache_entries` | 1 or more, or omitted | Decoded expert tensors in host RAM. Omitting it auto-sizes to 85% of available host budget and trims the 12 GiB CUDA cache request to fit VRAM. |
| `prefill_chunk` | 1 to `context_limit` | Prompt tokens per step. |

## `fake`

Test engine that replies with fixed text and loads no weights.

| Key | Default | Meaning |
| --- | --- | --- |
| `reply` | `Hello from the fake engine.` | Text returned in every reply. |
| `token_delay_ms` | 0 | Delay before each token, in milliseconds. |
