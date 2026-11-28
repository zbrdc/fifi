# Current status

The project goal is 128K to 256K tokens of context for every model, with GGUF as the weight format on CUDA.
All measurements below come from 20 Nov 2026 on `main`. Runs used greedy sampling, 256 generated tokens, fresh server boots, and three runs per prompt size (ranges list all three).

- **Default arm**: server started with only engine and model files configured.
- **Tuned arm**: shipped profile for that host with paths and listen address swapped.
- **Units**: GB is 10^9 bytes. MiB and GiB are binary (2^20 and 2^30 bytes).
- For reference-engine comparisons (`llama.cpp`, `MLX-LM`), see [Benchmarks](benchmarks.md).

## Qwen3.6-35B-A3B, GGUF Q4_K_M, on an RTX 3090 (24 GB, CUDA)

Default arm (context 32,768, KV fp16, graph capture on, 20.69 GB resident weights, 3.63 GB free):

| Prompt tokens | Prefill ms | Decode tok/s | Conditions |
| ---: | --- | --- | --- |
| 510 | 144.5 / 145.4 / 146.1 | 165.28 / 164.95 / 164.71 | three fresh boots |
| 4,010 | 691.6 / 687.3 / 688.2 | 159.51 / 159.32 / 160.10 | same |
| 32,010 | 7,632.2 / 7,585.7 / 7,636.0 | 133.55 / 134.07 / 130.54 | same |
| 128,010 | refused | refused | HTTP 409 `context_limit`: `"prompt is too long: 128010 tokens > 32767 maximum"` |

Tuned arm (`engine/config/qwen36-cuda-3090-kv8-prefix.json`: prefix reuse on, context 180,224, `kv_dtype` q8, capture on, 3.58 GB free):

| Prompt tokens | Prefill ms | Decode tok/s | Conditions |
| ---: | --- | --- | --- |
| 510 | 164.7 / 150.6 / 145.8 | 153.58 / 158.87 / 161.01 | three fresh boots |
| 4,010 | 943.9 / 941.1 / 945.0 | 161.07 / 161.17 / 161.06 | same |
| 32,010 | 8,697.6 / 9,073.4 / 9,116.9 | 136.23 / 131.83 / 129.37 | same |
| 128,010 | 54,824.9 / 54,598.0 / 55,390.3 | 85.42 / 86.39 / 83.50 | ~2,330 tok/s prefill |

- **Memory caps (default arm)**: 6, 8, 12, and 20 GiB container caps all serve (warm prefill 144–148 ms, decode ~174 tok/s).
- **Small card refusal**: With 5,836 MiB VRAM free, startup refuses (`cuda admission: need 22153406976 bytes ..., device has 5805441024 free`). Weights do not spill to host RAM.

## Qwen3.8 Flash-Next, EXL3 3.05 bpw, on an RTX 3090

Default arm (context 4,096, `FLASH_NEXT_ARTIFACT_MANIFEST` set):

| Prompt tokens | Prefill ms | Decode tok/s | Conditions |
| ---: | --- | --- | --- |
| 572 | 8,858.0 / 8,207.0 / 8,621.3 | 28.48 / 29.16 / 29.94 | three fresh boots |
| 4,071 / 4,072 | 61,769.1 / 61,094.8 / 61,136.3 | 23.74 / 22.73 / 23.36 | 4,064-of-4,096 decodes at 26.57 tok/s |

- **Tuned arm** (ring budget 1,155,302,400, EXL3 rows max 2,048): 571/572 prefill 7,539.9–7,838.5 ms (decode 27.57–28.49 tok/s). 4,072 prefill takes 56,516.5–56,986.2 ms (decode 22.48–24.91 tok/s).
- **Prefix reuse**: A 3,216-token second turn reuses 2,895 tokens. TTFT drops from 44.0 s to 5.1 s (default) and from 39.6 s to 4.7 s (tuned).
- **Memory caps (default arm)**: 6, 8, and 12 GiB refuse (EXL3 banks need 14.18 GiB). 20 GiB serves on the disk tier (ready in 20.1 s, warm decode 20.9–22.9 tok/s).

## Qwen3.6-35B-A3B, MLX 4-bit, on Apple Silicon (36 GB)

- **Default arm**: refused at startup (`cache_limit_bytes is missing and Q36_CACHE_LIMIT is not set`).
- **Tuned arm** (`engine/config/qwen36-mac-studio-36gb.json`, peak RSS 16.8 GiB):

| Prompt tokens | Prefill ms (median) | Decode tok/s | Conditions |
| ---: | ---: | --- | --- |
| 526 | 479.5 (478.8 / 479.5 / 782.2) | 86.72 / 86.83 / 86.80 | first hit pays compile tracing |
| 4,022 | 3,230.3 (3,224.7 / 3,230.3 / 3,234.3) | 78.59 / 78.67 / 78.68 | greedy, 256 tokens |
| 32,028 | 34,539.3 (34,493.2 / 34,539.3 / 34,625.8) | 66.58 / 66.79 / 66.83 | same |

Dense Qwen3.8-27B had no MLX pack available on this host.

## Flint, MLX int4 21 GB fixture, on Apple Silicon (36 GB)

Default arm equals tuned arm (context ceiling 32,768, peak RSS 14.1 GiB, ready in ~1 min):

| Prompt tokens | Prefill ms (median) | Decode tok/s | Conditions |
| ---: | ---: | --- | --- |
| 526 | 424 (415.4 / 424.0, with first 2,262 cold) | 52.34 / 52.73 / 52.75 | greedy, 256 tokens |
| 4,022 | 2,944 (2,899.0 / 2,943.7 / 3,240.4) | 48.41 / 48.52 / 48.54 | same |
| 32,028 | 46,986 (46,853 / 46,986 / 46,990) | 26.04 / 26.07 / 26.11 | 128K prompt refuses |

## Qwen3.6-35B-A3B, GGUF Q4_K_M, on an AGX Xavier board

Thermal guard aborts prefill runs at 88 °C.

| Arm | Prompt tokens | Prefill ms | Decode tok/s | Peak temp / notes |
| --- | ---: | --- | --- | --- |
| Default | 486 | 2,911.8–4,296.4 | 20.57–20.69 | 54.5 °C (two runs) |
| Default | 3,993 | 35,755–37,128 | 19.92–19.96 | 60.5–67.0 °C (three runs) |
| Default | 31,650 | 1,681,370 / 1,767,007 | 14.48 / 15.38 | pre-guard (91 °C). Later runs hit 88 °C guard |
| Tuned (KV q8, 16K) | 486 | 2,788.3–3,242.0 | 20.65–20.75 | three runs |
| Tuned (KV q8, 16K) | 3,993 | 36,864.9–43,020.3 | 19.94–20.04 | three runs |
| Tuned (KV q8, 16K) | 15,283 (max) | 401,369.9 | 18.29 | 87.0 °C (two later runs hit 88.0 °C guard) |

## Refused or unserved configurations

- **Qwen3.6 EXL3 on CUDA**: refuses `"weight_format": "exl3"`. CUDA serves GGUF only.
- **Flint on CUDA**: refuses. Flint serves on Apple Silicon only.
- **Bolt e4b on RTX 3090**: refused because the directory held 4 `.gguf` files instead of one.
- **Nemotron 3.5 Super on RTX 3090**: refused because the store held GGUF instead of safetensors.
