# Environment variables

Set these before starting `enginesrv`; the server reads them once at startup.
The API key variable is not listed here: its name comes from `api_key_env` in the [configuration file](configuration.md).

Two kinds of values appear below:

- A **switch** that is unset keeps its default. A switch that is set is on only for `1`, `true` or `on` (surrounding spaces and letter case are ignored); any other value turns it off.
- A variable with **accepted values** refuses startup when set to anything else, unless the table says otherwise.

## Build

| Variable | Default | Meaning |
| --- | --- | --- |
| `INFERENCE_CUDA_ARCHS` | `86` (`72` on aarch64) | Comma-separated GPU architectures for the CUDA kernels, for example `89` for RTX 40-series cards or `86,89` for both. Read by `cargo build --features native-cuda`. |

## Server

| Variable | Default | Meaning |
| --- | --- | --- |
| `ENGINESRV_WARM_TURNS` | on (switch) | With `prefix_reuse`, remember the last conversation's tokens so that the next turn only tokenizes its new messages. A request from another conversation replaces the memory. |
| `ENGINESRV_PREFIX_BLOCK_MATCH` | on (switch) | With `prefix_reuse`, also save a prefix every `ENGINESRV_PREFIX_BLOCK_TOKENS` tokens, so a prompt that changes only near its end can still reuse most of the conversation. Used by engines that keep at least two saved prefixes. |
| `ENGINESRV_PREFIX_BLOCK_TOKENS` | 4,096 | Spacing of those saves, in tokens. Values below 1 select the default. |
| `ENGINESRV_CACHE_USAGE` | on | Report reused prompt tokens in the API usage fields. Once set, only the exact values `1`, `true` and `on` keep it on. |
| `ENGINESRV_THINK_REPLAY` | on (switch) | Qwen3.6 only: keep the reasoning that clients resend with earlier assistant turns in the prompt. Off removes it for every engine. |
| `ENGINESRV_NOTICE_TOOLWRAP` | on (switch) | Qwen3.6 only, with think replay on: render system messages that follow an assistant turn as tool responses, so that the chat template keeps the reasoning of earlier tool-calling turns. |

## Qwen3.6 on Apple Silicon

| Variable | Default | Meaning |
| --- | --- | --- |
| `Q36_CACHE_LIMIT` | none | Fallback for `cache_limit_bytes`, in bytes. |
| `Q36_FUSED_PREFILL` | on | Fused kernels for prompt processing. Only `0` turns them off. |

## Qwen3.6 on NVIDIA CUDA

| Variable | Default | Accepted values | Meaning |
| --- | --- | --- | --- |
| `Q36_VERIFY_GGUF_SHA256` | `0` | `0`, `1` | `1` checks the model file's SHA-256 while it loads. |
| `Q36_CUDA_CHUNK_PREFILL` | `1` with `kv_dtype: q8`, else `0` | `0`, `1` | `0` processes the prompt one token at a time. |
| `Q36_CUDA_CHUNK_ROWS` | `1024` | `16`, `128`, `512`, `1024` | Largest prompt chunk when chunking is on. |
| `Q36_CUDA_KV8_STAGE_CONTEXT` | `1` | `0`, `1` | `0` limits the FP16 attention staging buffer to 65,536 rows. |
| `Q36_CUDA_KV8_FIXED_ADMISSION` | `1` | `0`, `1` | `0` returns to the older way of choosing chunk sizes, which may pick smaller chunks or one token at a time. On a 24 GB card, use the 131K profile with it. |
| `Q36_CUDA_CPF_STAGE_128K` | `1` | `0`, `1` | Above 131,072 tokens, attention stages a 131,072-row buffer (256 MiB) when it fits. `0` always uses the 32,768-row buffer (64 MiB). No effect at or below 131,072 tokens. |
| `Q36_CUDA_PINNED_PREFIX` | on | only `1` is on | Any other value keeps saved prefixes in pageable host memory and turns off the GPU copy. |
| `Q36_CUDA_STATE_ONLY_PREFIX` | on | only `1` is on | Any other value turns off the GPU copy of the newest prefix and keeps the pinned host copies. |
| `Q36_CUDA_PINNED_PREFIX_BYTES` | smaller of 3 slots and 8 GiB | byte count | Size of the pinned host pool for saved prefixes. `0` keeps none. |
| `Q36_CUDA_PAGED_DEVICE_FIRST` | `0` | `0`, `1` | With `kv_offload_host_bytes`: keep as much of the KV cache on the GPU as fits after a 1 GiB reserve. |
| `INFERENCE_CUDA_KV8_ATTN_V_PREFETCH` | `0` | only `1` is on | Experimental register-V decode attention: use the prefetch kernel variant, which holds the next key's code words in registers ahead of the dot chain. Read once at startup; only the exact value `1` turns it on, any other value or unset keeps the default kernel. Head dimension 256 only, and it needs the register-V KV8 decode attention path (the default with `kv_dtype: q8`); outputs are bitwise-equal to it. |

`Q36_CUDA_CHUNK_PREFILL=0` and `Q36_CUDA_KV8_STAGE_CONTEXT=0` also turn off `Q36_CUDA_KV8_FIXED_ADMISSION`.
With `tiered`, the pinned prefix pool is off unless `Q36_CUDA_PINNED_PREFIX=1` is set.

## Flint

On Apple Silicon:

| Variable | Default | Accepted values | Meaning |
| --- | --- | --- | --- |
| `FIFI_FLINT_MLX_CONTEXT_LIMIT` | `32768` | `32768`, `8192` | Context limit. Any other value selects 8,192. |
| `FIFI_FLINT_MLX_BATCH_PREFILL` | `1024` | `1024`, `512`, `0` | Prompt chunk size. `0` processes one token at a time, 50 to 70 times slower. Any other value selects 512. |
| `FIFI_FLINT_MLX_ASYNC_CHAIN` | on | switch | Evaluate decode work in chains of 4 layers instead of synchronizing after every layer. |
| `FIFI_FLINT_MLX_COALESCE_RESIDUAL` | on | switch | Merge the residual updates of a decode step. |
| `FIFI_FLINT_MLX_DECODE_ATTN` | on | switch | Use the fused MLX attention kernel for one-token decoding. Startup then refuses a checkpoint whose attention-sink values are not exact in BF16, and names the layer, head and value. |
| `FIFI_FLINT_MLX_RING_VIEW` | on | switch | Read the full-attention KV cache as one ring view instead of concatenating it each step. Needs `FIFI_FLINT_MLX_ASYNC_CHAIN`. |
| `FIFI_FLINT_PREFIX_REUSE` | off | switch | Experimental: keep prefix snapshots between turns. Also needs `"prefix_reuse": true`. |
| `FIFI_FLINT_PREFIX_BUDGET_BYTES` | physical memory minus weights and a system reserve | positive byte count | Memory cap for prefix snapshots. `0` or an unreadable value refuses startup. |

On NVIDIA CUDA:

| Variable | Default | Meaning |
| --- | --- | --- |
| `FIFI_FLINT_CUDA_ALLOC_CACHE` | on (switch) | Reuse small GPU allocations (at most 1 MiB each, 16 MiB and 256 entries in total). The cache is emptied after each request. |
| `FIFI_FLINT_LATE_HEALTH` | on (switch) | Check the model output for non-finite values on the GPU. |
| `FIFI_FLINT_CUDA_PACKED_POINTWISE` | off (switch) | Experimental packed BF16 element-wise kernels. |
| `FIFI_FLINT_PREFIX_REUSE` | off (switch) | Experimental: keep prefix snapshots between turns. Also needs `"prefix_reuse": true` and `"kv_profile": "f16"`; with `q8` the engine logs a note and serves without reuse. |
| `FIFI_FLINT_CUDA_GROUPED_GATHER` | on (switch) | Per-warp shared-table gather for grouped EXL3 expert projections; bit-identical to the shuffle gather by sweep and served text parity, and faster end to end (TTFT -7.3/-6.9 pct and decode -4.0/-3.5 pct at 512/4000 prompts on the RTX 3090). Any other set value, including empty and non-UTF-8, rolls back to the shuffle gather. Device coverage is rows 1 through 8 per request (decode and prefill); longer contexts reuse the same row kernels. |

The startup line reports `alloc_cache=` and `late_health=`.
