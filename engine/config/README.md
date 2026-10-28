# Configuration

`enginesrv --config <file>` reads one JSON file.
Startup refuses unknown keys and invalid values; it never substitutes a smaller setting on its own.

## Shipped configs

| File | Hardware | Model | Context | KV cache |
| --- | --- | --- | ---: | --- |
| `qwen36-mac-studio-36gb.json` | Mac Studio, M4 Max, 36 GB | Qwen3.6-35B-A3B, MLX INT4 | 262,144 | 8-bit |
| `qwen36-cuda-3090-kv8-prefix.json` | RTX 3090, 24 GB | Qwen3.6-35B-A3B, GGUF Q4_K_M | 180,224 | 8-bit |
| `qwen36-cuda-3090-kv8-prefix-131k.json` | RTX 3090, 24 GB | same | 131,072 | 8-bit |
| `qwen36-cuda-3090-kv8-prefix-262k.json` | RTX 3090, 24 GB | same | 262,144 | 8-bit |
| `qwen36-rtx3090-24gb-fp16kv.json` | RTX 3090, 24 GB | same | 98,304 | 16-bit |

Set `model_dir` and `tokenizer_dir` before use. All files read the API key from `ENGINESRV_API_KEY`.
[doc/cuda.md](../../doc/cuda.md) explains how to choose between the RTX 3090 files.

## File layout

```json
{
  "listen": "127.0.0.1:18791",
  "api_key_env": "ENGINESRV_API_KEY",
  "prefix_reuse": false,
  "engines": [
    {
      "id": "q36",
      "family": "qwen3_6_35b_a3b",
      "model": "fifi-q36",
      "model_dir": "/path/to/model",
      "tokenizer_dir": "/path/to/model",
      "qwen36": {"context_limit": 262144}
    }
  ]
}
```

### Top level

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `listen` | yes | | Address and port to listen on. Port 8788 is refused. |
| `api_key_env` | yes | | Name of the environment variable that holds the API key. Startup fails if it is unset or empty. |
| `prefix_reuse` | no | `false` | Keep conversation prefixes between requests so a follow-up turn only prefills its new tokens. Applies to every engine that supports it. |
| `engines` | yes | | One entry per engine. |

### Engine entry

| Key | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Engine name. |
| `family` | yes | `qwen3_6_35b_a3b`, `qwen3_8_27b_dense`, `qwen3_8_flash_next`, `flint`, `nemotron35` or `fake`. |
| `model_dir` | yes | Checkpoint directory. |
| `tokenizer_dir` | yes | Directory with `tokenizer.json` and the chat template. |
| `model` | no | Name clients pass in the `model` field. Defaults to the family name. |
| `qwen36` | no | Options for the two Qwen families (below). |
| `flint` | CUDA only | Options for Flint on CUDA; see [doc/flint.md](../../doc/flint.md). |
| `nemotron` | for `nemotron35` | See [doc/nemotron.md](../../doc/nemotron.md). |
| `flash_next` | no | Options for `qwen3_8_flash_next`. |
| `fake` | no | Test engine that replies with fixed text: `reply`, `token_delay_ms`. It also reads the entry's `context_limit`. |
| `think_replay` | no | `false` stops replaying earlier-turn reasoning for this engine (see `ENGINESRV_THINK_REPLAY`). `true` is refused. |
| `notice_toolwrap` | no | `false` turns off the notice wrap for this engine (see `ENGINESRV_NOTICE_TOOLWRAP`). `true` is refused. |

## `qwen36` options

These apply to `qwen3_6_35b_a3b` and `qwen3_8_27b_dense`. "MLX" means Apple Silicon.

| Key | Backend | Default | Meaning |
| --- | --- | --- | --- |
| `context_limit` | both | MLX 262,144; CUDA 32,768 | Context limit in tokens, from 2. |
| `kv_dtype` | both | MLX `q8`; CUDA `f32` | KV cache format: `q8` (8-bit), `f32`, or `bf16` (MLX only). On CUDA, `f32` stores the cache in FP16. |
| `activation_dtype` | MLX | `bf16` | Decode activations: `bf16` or `f32`. CUDA accepts the key and ignores it. |
| `session_cap_bytes` | MLX | 4 GiB | Memory for the session state. `qwen3_8_27b_dense` needs it set explicitly. |
| `cache_limit_bytes` | MLX | none | MLX allocator cache limit. Falls back to `Q36_CACHE_LIMIT`; one of the two is required. |
| `prefill_budget_bytes` | MLX | off | Memory budget for chunked prefill. Without it, prefill runs one token per step. |
| `prefill_session_reserve_bytes` | MLX | none | Memory the prefill planner leaves free. Required with `prefill_budget_bytes`. |
| `prefill_chunk` | MLX | 256 | Largest prefill chunk, 1 to 2,048 rows. The planner picks the largest chunk that fits the budget. |
| `compile` | MLX | `false` | Compile the decode blocks with `mlx_compile`: 9.6 to 13% faster decode; logits differ from the uncompiled path by at most 5e-5. |
| `max_seqs` | MLX | 1 | Concurrent sequences, 1 to 8. Above 1 it needs `compile: true`, `prefill_budget_bytes` and a `prefill_chunk` of at least 4; not available for `qwen3_8_27b_dense`. |
| `prefix_store_entries` | MLX | 6 | Number of saved prefixes when `prefix_reuse` is on, 1 to 4,096. |
| `capture` | CUDA | `false` | Run each decode step as a CUDA graph. |
| `kv_offload_host_bytes` | CUDA | off | Keep the KV cache in host memory, up to this many bytes. See [doc/cuda.md](../../doc/cuda.md#host-paged-kv-cache-opt-in). |
| `tiered` | CUDA | off | Keep only some routed experts on the GPU; see below. |

### KV cache size (Qwen3.6)

| `kv_dtype` | Activations | Bytes per position |
| --- | --- | ---: |
| `f32` | any | 40,960 |
| `q8` | `bf16` | 10,880 |
| `q8` | `f32` | 11,520 |

Each session also holds 65,863,680 bytes of recurrent (DeltaNet) state.
At 262,144 positions with `q8` and `bf16`, the KV cache takes 2,852,126,720 bytes.

With `prefix_reuse` on, an MLX engine needs `session_cap_bytes` of at least twice the KV cache plus the recurrent state, because one saved prefix and the live session share the cap.
At 262,144 positions with `q8` and `bf16`, that is 5,770,117,120 bytes, more than the 4 GiB default.

### The Mac Studio file

`qwen36-mac-studio-36gb.json` sets a 30 GiB prefill budget (`32212254720`) and a reserve of `8286699520` bytes.
The reserve is the KV cache at 262,144 positions (2,852,126,720), the recurrent state (65,863,680), the cache limit (4 GiB) and a 1 GiB margin.
Recompute it if you change `context_limit`, `kv_dtype`, `activation_dtype` or `cache_limit_bytes`.
The file does not turn on `prefix_reuse`.

### Tiered experts (CUDA)

`tiered` keeps a fixed number of expert slots per layer on the GPU and runs the other routed experts on the CPU from the memory-mapped model file.
Do not modify or replace the model file while the server runs.

| Key | Default | Meaning |
| --- | --- | --- |
| `slots` | required | Expert slots per layer on the GPU, 8 to 256. |
| `threads` | 6 | CPU worker threads, 1 to 8. |
| `timeout_ms` | 10,000 | Longest wait for one layer's CPU experts, 1 to 600,000. |

## Environment variables

Set these before startup.

A switch that is unset keeps its default. A switch that is set is on only for `1`, `true` or `on` (surrounding whitespace and letter case are ignored); any other value turns it off.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ENGINESRV_PREFIX_BLOCK_MATCH` | on (switch) | With `prefix_reuse`, also save a prefix every `ENGINESRV_PREFIX_BLOCK_TOKENS` tokens, so a prompt that changes only its tail can restore most of the conversation. Used by engines that keep at least two saved prefixes. |
| `ENGINESRV_PREFIX_BLOCK_TOKENS` | 4,096 | Spacing of those saves. Values below 1 select the default. |
| `ENGINESRV_WARM_TURNS` | on (switch) | With `prefix_reuse`, remember the last conversation's tokens so the next turn tokenizes only its new messages. A request from another conversation replaces the memory. |
| `ENGINESRV_CACHE_USAGE` | on | Report reused prefix tokens in the API usage fields. Only the exact values `1`, `true` and `on` keep it on once set. |
| `ENGINESRV_THINK_REPLAY` | on (switch) | Qwen3.6 chat template only: keep the reasoning that clients resend with earlier turns in the prompt. |
| `ENGINESRV_NOTICE_TOOLWRAP` | on (switch) | With think replay: render system notices that follow an assistant turn as tool responses, so the template keeps the reasoning of tool loops. |
| `Q36_FUSED_PREFILL` | on | MLX Qwen3.6: fused prefill kernels. Only `0` turns them off. |
| `Q36_CACHE_LIMIT` | none | Fallback for `cache_limit_bytes`. |

Family-specific variables are listed in [doc/cuda.md](../../doc/cuda.md), [doc/flint.md](../../doc/flint.md) and [doc/nemotron.md](../../doc/nemotron.md).
