# How fifi plans memory

fifi computes how much memory a model needs before loading it.
If the plan does not fit, the engine refuses to load, and the server log names the part that failed.
fifi never lowers the context limit or changes the cache format to make a model fit. You choose those in the [configuration file](../reference/configuration.md).

The server starts listening before the engines load, but a refused load does not leave it waiting.
If every engine fails to load, the process exits with a non-zero code and prints the refusal on standard error.
The authenticated `/v1/models` error details are then no longer reachable, because the server is gone.
With several engines, the server keeps running while at least one engine is ready or busy, and `/v1/models` reports the failed engine's error.

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

The shipped Mac file sets a 30 GiB budget and a reserve of 8,286,699,520 bytes. That reserve covers the 262,144-token KV cache, recurrent state, 4 GiB allocator cache, and a 1 GiB margin.
An explicit reserve stays yours: recompute it if you change `context_limit`, `kv_dtype`, `activation_dtype`, or `cache_limit_bytes`. Omitting the member derives this exact sum from the resolved settings, so the shipped file resolves to the same bytes either way.

### On NVIDIA CUDA

Startup measures free GPU memory after loading weights and refuses the KV cache and prompt buffers if they do not fit.
Other programs count: the desktop alone often takes several hundred MiB.
On a 24 GB RTX 3090, the 180,224-token profile fits with up to 1,536 MiB in use by other programs. The 262,144-token profile fits with up to 744 MiB.

Saved prefixes live in pinned host memory. The pool holds up to nine slots (eight saved prefixes and one staging slot), each matching the full KV cache plus recurrent state. By default it takes at most half of available host memory. `Q36_CUDA_PINNED_PREFIX_BYTES` sets its size.
At 180,224 positions with the 8-bit cache, one slot takes about 2.03 GB, so nine slots take about 18.2 GB.
The engine also needs 1 GiB of free host memory beyond the pool. When that runs short, it keeps fewer saved prefixes, down to none.

## Bolt

Bolt keeps saved prefixes in host memory. A saved prefix copies one session's cache below its position. Four full-attention layers keep every position (8,704 bytes each), and twenty sliding layers keep their last 512 positions (10.6 MiB total). A 32,768-position prefix takes about 283 MiB, and a 4,000-token prefix takes about 46 MB. Restoring one writes it back into whichever session serves the next request.

Startup keeps one saved prefix per concurrent session, up to the host's byte budget. The budget equals machine memory minus the OS floor and process memory. The disk tier writes the oldest prefixes to disk when the store grows past it. Every save checks free memory again (respecting cgroup limits on Linux). If space is short, the engine skips the save and processes the next turn whole. Busy hosts start saving once memory frees up, and hosts that tighten later stop saving.

The startup line reports the decision:

- `prefix reuse is on: ... N kept; the host may hold X MiB of them; each save checks the free memory again`.
- `prefix reuse is on but saving waits: the host budget of X MiB cannot fund a saved prefix of up to Z MiB; ...`, when the budget is under one entry. Saves start once the host can fund one.
- `prefix reuse is on: ... the host memory could not be read, so the saves are not sized against it`, when machine memory is unknown. The store then keeps one entry.
- `prefix reuse is off (the configuration sets prefix_reuse to false)`, when the configuration turned it off.

## Flint

On Apple Silicon, the full-attention KV cache grows with use. A request allocates its prompt length rounded up to 4,096 rows, and buffers double as the conversation grows, up to 32,768 tokens.
Sliding-window layers keep a 513-token window.

On CUDA, the plan fixes sizes at startup:

| Item | Size |
| --- | --- |
| KV cache, per layer and position | 2,048 bytes (`f16`), 1,088 bytes (`q8`) |
| Workspace | 512 MiB, plus 16 MiB per prompt row beyond the first (624 MiB at 8 rows) |
| Allocation cache | 16 MiB |
| Output health checks | 2,394,612 bytes |

## Nemotron 3.5 Super

The checkpoint stays on disk, and the loader reads it as needed.
Startup plans host memory for dense weights, session state, `cache_entries` decoded expert tensors, and working buffers. It also checks GPU memory for CUDA buffers.
When you omit `cache_entries`, the loader fits the count to measured available host memory (using the cgroup-aware budget on Linux, falling back to machine memory). One log line states the chosen count.
