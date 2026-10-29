# How fifi plans memory

fifi works out how much memory a model needs before it loads it.
If the plan does not fit, the engine does not load, and the server log says which part did not fit.
fifi never lowers the context limit or changes the cache format to make a model fit; you choose those in the [configuration file](../reference/configuration.md).

The server starts listening before the engines load, so a refused load shows up as `/ready` returning `503` while `/health` still answers.

## What takes memory

An engine needs room for:

- **Weights.** Roughly the size of the checkpoint on disk.
- **The KV cache.** Attention keys and values for every position up to the context limit. This is what grows with the context.
- **Recurrent state.** Qwen3.6 mixes attention layers with recurrent (DeltaNet) layers, whose state has a fixed size: 65,863,680 bytes per session, whatever the context.
- **Working buffers** for processing prompts in chunks and for decoding.
- **Saved prefixes** when [prefix reuse](prefix-reuse.md) is on.

## Qwen3.6 KV cache

| Backend | `kv_dtype` | Bytes per position |
| --- | --- | ---: |
| Apple Silicon | `f32` | 40,960 |
| Apple Silicon | `q8`, `bf16` activations | 10,880 |
| Apple Silicon | `q8`, `f32` activations | 11,520 |
| CUDA | `f32` (stored as FP16) | 20,480 |
| CUDA | `q8` | 10,880 |

At 262,144 positions with the 8-bit cache, the KV cache takes 2,852,126,720 bytes (2.66 GiB).

### On Apple Silicon

Unified memory holds everything, so the plan is a budget split:

- `session_cap_bytes` (default 4 GiB) holds the live session: its KV cache and recurrent state. With prefix reuse on, it also holds one saved prefix, so it must be at least twice the KV cache plus the recurrent state. At 262,144 positions that is 5,770,117,120 bytes.
- `prefill_budget_bytes` is the memory prompt processing may use. The engine picks the largest chunk (up to `prefill_chunk` rows) whose buffers fit the budget after the weights and `prefill_session_reserve_bytes`.

The shipped Mac file sets a 30 GiB budget and a reserve of 8,286,699,520 bytes: the KV cache at 262,144 positions, the recurrent state, the 4 GiB allocator cache limit and a 1 GiB margin.
Recompute the reserve if you change `context_limit`, `kv_dtype`, `activation_dtype` or `cache_limit_bytes`.

### On NVIDIA CUDA

Startup measures free GPU memory after loading the weights and refuses the KV cache and prompt buffers if they do not fit.
Other programs count: the desktop alone often takes several hundred MiB.
On a 24 GB RTX 3090, the 180,224-token profile fits with up to 1,536 MiB in use by other programs, and the 262,144-token profile with up to 744 MiB.

Saved prefixes live in pinned host memory. The pool holds up to three slots (two saved prefixes and one staging slot), each the size of the full KV cache plus the recurrent state, and is capped at 8 GiB.
At 180,224 positions with the 8-bit cache, three slots take about 6.08 GB.
The engine also wants 1 GiB of free host memory beyond the pool; when that is short it keeps one saved prefix, and then none.

## Flint

On Apple Silicon, the full-attention KV cache grows with use: a request allocates its prompt length rounded up to 4,096 rows, and the buffers double as the conversation grows, up to the 32,768-token limit.
Sliding-window layers keep a 513-token window.

On CUDA the plan is fixed at startup:

| Item | Size |
| --- | --- |
| KV cache, per layer and position | 2,048 bytes (`f16`), 1,088 bytes (`q8`) |
| Workspace | 512 MiB, plus 16 MiB per prompt row beyond the first (624 MiB at 8 rows) |
| Allocation cache | 16 MiB |
| Output health checks | 2,394,612 bytes |

## Nemotron 3.5 Super

The checkpoint stays on disk and is read as needed.
Startup plans host memory for the dense weights, the session state, `cache_entries` decoded expert tensors and working buffers, and checks GPU memory for the CUDA buffers.
