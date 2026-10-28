# Qwen3.6 on NVIDIA CUDA

The `qwen3_6_35b_a3b` family runs `Qwen3.6-35B-A3B` on NVIDIA GPUs.
It is tested on an RTX 3090 (24 GB).

## Requirements

- A Linux build with `--features native-cuda`.
- `model_dir` must hold `Qwen3.6-35B-A3B-Q4_K_M.gguf`, exactly 20,419,565,568 bytes. Startup refuses any other file.
- Set `Q36_VERIFY_GGUF_SHA256=1` to also check the file's SHA-256 at startup.

## Profiles

[engine/config](../engine/config/README.md) ships four RTX 3090 profiles.
Set `model_dir` and `tokenizer_dir` in the one you use.

| Profile | Context | KV cache | Fits with other GPU use up to |
| --- | ---: | --- | --- |
| [`qwen36-cuda-3090-kv8-prefix.json`](../engine/config/qwen36-cuda-3090-kv8-prefix.json) | 180,224 | 8-bit | 1,536 MiB |
| [`qwen36-cuda-3090-kv8-prefix-131k.json`](../engine/config/qwen36-cuda-3090-kv8-prefix-131k.json) | 131,072 | 8-bit | — |
| [`qwen36-cuda-3090-kv8-prefix-262k.json`](../engine/config/qwen36-cuda-3090-kv8-prefix-262k.json) | 262,144 | 8-bit | 744 MiB |
| [`qwen36-rtx3090-24gb-fp16kv.json`](../engine/config/qwen36-rtx3090-24gb-fp16kv.json) | 98,304 | 16-bit | — |

All four turn on prefix reuse and CUDA graph capture.
The 131K profile reserves smaller buffers and prefills cold prompts faster than the larger ones.
"Other GPU use" is memory already taken by the desktop and other programs.
Startup measures free GPU memory and refuses a load that does not fit; it never shrinks the context or changes the cache format on its own.

Without a profile, the CUDA defaults are a 16-bit KV cache, a 32,768-token context, no graph capture and no prefix reuse.

## Prefill

With the 8-bit KV cache, prompts are prefilled in chunks.
Chunks get smaller deeper into the prompt: up to 1,024 rows at first, at most 512 rows past 65,536 positions, and at most 256 rows past 155,648 positions.
With a context above 131,072 tokens, the largest chunk is 512 rows.
Above 131,072 tokens, attention stages a 131,072-row FP16 buffer (256 MiB) when it fits, and a 32,768-row buffer (64 MiB) otherwise.

## Prefix reuse

With `"prefix_reuse": true`, the engine keeps finished conversation prefixes so the next turn only prefills new tokens.

- The most recent prefix stays on the GPU until a later request overwrites it; restoring it from there needs no copy.
- Up to two prefixes are also kept in pinned host memory, plus one staging slot. Each slot holds the full configured KV cache plus 65,863,680 bytes of recurrent state.
- The pinned pool is the smaller of three slots and 8 GiB. At a 180,224-token context with the 8-bit cache, three slots take about 6.08 GB of host RAM.
- The engine also needs 1 GiB of free host memory beyond the pool. If two prefixes do not fit, it keeps one; if one does not fit, it keeps none.

## Environment variables

Set these before startup.

| Variable | Default | Accepted values | Effect |
| --- | --- | --- | --- |
| `Q36_CUDA_CHUNK_PREFILL` | `1` with the 8-bit cache | `0`, `1` | `0` prefills one token at a time. |
| `Q36_CUDA_CHUNK_ROWS` | `1024` | `16`, `128`, `512`, `1024` | Largest prefill chunk. |
| `Q36_CUDA_KV8_STAGE_CONTEXT` | `1` | `0`, `1` | `0` limits the FP16 attention staging to 65,536 rows. |
| `Q36_CUDA_KV8_FIXED_ADMISSION` | `1` | `0`, `1` | `0` restores the older admission, which may pick smaller chunks or per-token prefill. On a 24 GB card, use the 131K profile with it. |
| `Q36_CUDA_CPF_STAGE_128K` | `1` | `0`, `1` | `0` always uses the 32,768-row staging buffer. |
| `Q36_CUDA_PINNED_PREFIX` | on | `1` is on | Any other value keeps prefixes in pageable memory and turns off the GPU copy. |
| `Q36_CUDA_STATE_ONLY_PREFIX` | on | `1` is on | Any other value turns off the GPU copy and keeps the pinned host copies. |
| `Q36_CUDA_PINNED_PREFIX_BYTES` | smaller of 3 slots and 8 GiB | byte count | Size of the pinned prefix pool. `0` keeps no prefixes. |
| `Q36_VERIFY_GGUF_SHA256` | `0` | `0`, `1` | `1` checks the model file's SHA-256 at startup. |

Startup refuses values outside the accepted list where the list is given.
`Q36_CUDA_CHUNK_PREFILL=0` and `Q36_CUDA_KV8_STAGE_CONTEXT=0` also turn off fixed admission.

## Host-paged KV cache (opt-in)

For contexts that do not fit in GPU memory, the KV cache can live in host memory.
Add `"kv_offload_host_bytes": <bytes>` to the `qwen36` object to set the host budget.

This mode needs `"kv_dtype": "q8"`, `"capture": false`, `"prefix_reuse": false` and `Q36_CUDA_CHUNK_PREFILL=0`.
It reloads the KV cache from host memory for every token, so it is slow: one 8K-token replay decoded at 34.95 tokens/s, against 147.14 tokens/s with the cache on the GPU.

Set `Q36_CUDA_PAGED_DEVICE_FIRST=1` to keep as much of the cache on the GPU as fits after a 1 GiB reserve, and send only the rest to host memory.
