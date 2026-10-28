# Flint

The `flint` family serves Flint, a 39.8B-parameter mixture-of-experts model with 2.5B active parameters.
It runs on Apple Silicon through MLX and, as a pilot, on NVIDIA GPUs through CUDA.

| Backend | Build | Checkpoint |
| --- | --- | --- |
| Apple Silicon (MLX) | `--features native-mlx` | `RecursiveSI/flint-int4` (MLX INT4) |
| NVIDIA CUDA (pilot) | `--features native-cuda`, Linux | A complete Flint EXL3 pack, tested with `0xBakeer/TandemLLM-Flint-EXL3-4.0` |

Run Flint in a process of its own.
Flint sets the MLX allocator cache limit to zero for the whole process, which overrides the cache limit of any other MLX engine in it.
A process that mixes Flint with other families is not tested.

## Apple Silicon

An MLX entry has no `flint` object; startup refuses one.

```json
{
  "id": "flint",
  "family": "flint",
  "model_dir": "/models/flint-int4",
  "tokenizer_dir": "/models/flint-int4"
}
```

The context limit is 32,768 tokens.
Full-attention KV buffers grow on demand: a request allocates its prompt length rounded up to 4,096 rows, and the buffers double as the sequence grows, up to the limit.
Sliding-window layers keep a 513-token window.

Prefill runs in pieces of up to 1,024 tokens. Decode runs one token at a time.

### Environment variables

Set these before startup.

| Variable | Default | Accepted values | Effect |
| --- | --- | --- | --- |
| `FIFI_FLINT_MLX_CONTEXT_LIMIT` | `32768` | `32768`, `8192` | Context limit. Any other value selects 8,192. |
| `FIFI_FLINT_MLX_BATCH_PREFILL` | `1024` | `1024`, `512`, `0` | Prefill piece size. `0` prefills one token at a time, 50 to 70 times slower. Any other value selects 512. |
| `FIFI_FLINT_MLX_ASYNC_CHAIN` | on | switch | Evaluates decode work in 4-layer chains instead of one synchronization point per layer. |
| `FIFI_FLINT_MLX_COALESCE_RESIDUAL` | on | switch | Merges the residual updates of a decode step. |
| `FIFI_FLINT_MLX_DECODE_ATTN` | on | switch | Uses the fused MLX attention kernel for one-token decode. |
| `FIFI_FLINT_MLX_RING_VIEW` | on | switch | Reads full-attention KV as one ring view instead of concatenating it each step. Needs the async chain. |
| `FIFI_FLINT_PREFIX_REUSE` | off | switch | Experimental: keeps prefix snapshots between turns. Also needs `"prefix_reuse": true` in the config. |
| `FIFI_FLINT_PREFIX_BUDGET_BYTES` | derived | positive byte count | Memory cap for prefix snapshots. The default is physical memory minus the weights and a reserve for the system. `0` or an unreadable value refuses startup. |

A switch that is unset keeps its default.
A switch that is set is on only for `1`, `true` or `on` (surrounding whitespace and letter case are ignored); any other value turns it off.
To return the whole decode path to its earlier form, set both `FIFI_FLINT_MLX_RING_VIEW=0` and `FIFI_FLINT_MLX_DECODE_ATTN=0`.

With `FIFI_FLINT_MLX_DECODE_ATTN` on, startup refuses a checkpoint whose attention-sink values are not exactly representable in BF16, and names the layer, head and value.
The values of `RecursiveSI/flint-int4` are all exact.

## NVIDIA CUDA (pilot)

The CUDA entry needs a `flint` object:

```json
{
  "listen": "127.0.0.1:8100",
  "api_key_env": "ENGINESRV_API_KEY",
  "engines": [{
    "id": "flint-cuda",
    "family": "flint",
    "model_dir": "/models/flint-exl3",
    "tokenizer_dir": "/models/flint-tokenizer",
    "flint": {"context_limit": 4096, "kv_profile": "f16"}
  }]
}
```

| Key | Required | Values | Meaning |
| --- | --- | --- | --- |
| `context_limit` | yes | 1 to 262,144 | Context limit in tokens. |
| `kv_profile` | yes | `"f16"`, `"q8"` | KV cache format. `q8` (int8 in groups of 32 with an F16 scale) is experimental; its output quality is not measured. |
| `prefill_chunk_size` | no | 1 to 8, default 8 | Prompt rows per prefill step. `1` prefills one token at a time. |

Startup checks free GPU memory for the weights, the KV cache, the workspace and a safety margin, and refuses the load if they do not fit.
It never falls back to a smaller context or another KV format.

| Item | Size |
| --- | --- |
| KV cache, per layer and position | 2,048 bytes (`f16`), 1,088 bytes (`q8`) |
| Workspace | 512 MiB, plus 16 MiB for each prefill row beyond the first (624 MiB at 8 rows) |
| Allocation cache | 16 MiB |
| Health checks | 2,394,612 bytes |

Prefix reuse is off on CUDA.
Greedy decoding and top-k sampling with k from 1 to 20 run on the GPU; other sampling settings run on the host.

### Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `FIFI_FLINT_CUDA_ALLOC_CACHE` | on | Reuses small device allocations (at most 1 MiB each, 16 MiB and 256 entries in total). The cache is emptied after each request. |
| `FIFI_FLINT_LATE_HEALTH` | on | Checks the model outputs for non-finite values on the device. |
| `FIFI_FLINT_CUDA_PACKED_POINTWISE` | off | Experimental packed BF16 element-wise kernels. |

These follow the same switch rules as above. The startup line reports `alloc_cache=` and `late_health=`.

### Errors

A non-finite value in the model output fails that request with HTTP 500 and the error code `backend_error`.
The engine resets the session and keeps serving.
A CUDA runtime or launch error stops the engine; restart the server.
