# Current status

The goal is 128K to 256K tokens of context for every model, with GGUF as the weight format on CUDA. The context rows show the measured admission limits and boot results.

This page records measurements by model and backend.
It holds measured facts only. Every row gives the date, the conditions and the receipt it comes from, and anything that is an estimate says so.
Numbers move with every merged change, so the date on each row matters more than the order of the rows.

Every row below comes from 20 Nov 2026 on `main` as of that date. We removed earlier sweep figures that this sweep did not reproduce. Where a model could not run that day, the row states that instead of keeping an old number. Workload throughout: greedy sampling, 256 generated tokens, fresh server boot per run, three runs per prompt size. Ranges list all three runs (medians in parentheses).

How to read the rows:

- **Default arm** means the server started with nothing set: a config naming only the engine and model files, with no tuning switches.
- **Tuned arm** means the shipped profile for that host (named in each section) with only paths and listen address swapped.
- **Prefill / first token** is the server's `prefill_ms` / `time_to_first_token_ms` from the request log. **Decode** is `decode_tokens_per_second`. Rows without temperature recording say so.
- **Refusal rows** quote the exact error the server returned. We record refusals as results.
- **Units**: GB is 10^9 bytes. MiB and GiB are binary (2^20 and 2^30 bytes). Rows quoting `nvidia-smi` use MiB.
- Receipts (per-run server logs of the sweep) stay outside this repository. Each row names the receipt kind and date.
- Compare rows only when their conditions match. For comparisons against other engines, see [Benchmarks](benchmarks.md).

## Qwen3.6-35B-A3B, GGUF Q4_K_M, on an RTX 3090 (24 GB, CUDA)

Model file served in place (one GGUF plus tokenizer, 20,419,565,568 bytes of weights). See [Run Qwen3.6](../how-to/qwen36.md).

Default arm, load line: gguf weights resident 20,690,501,632 B, 3,627,941,888 B free after load, context limit 32,768, KV cache fp16, graph capture on.

| Prompt tokens | Prefill ms | Decode tok/s | Date | Conditions |
| ---: | --- | --- | --- | --- |
| 510 | 144.5 / 145.4 / 146.1 | 165.28 / 164.95 / 164.71 | 20 Nov 2026 | three fresh boots; server request log |
| 4,010 | 691.6 / 687.3 / 688.2 | 159.51 / 159.32 / 160.10 | 20 Nov 2026 | same |
| 32,010 | 7,632.2 / 7,585.7 / 7,636.0 | 133.55 / 134.07 / 130.54 | 20 Nov 2026 | same |
| 128,010 | refused | refused | 20 Nov 2026 | HTTP 409 `context_limit`: "prompt is too long: 128010 tokens > 32767 maximum" |

Tuned arm: shipped profile `engine/config/qwen36-cuda-3090-kv8-prefix.json` (prefix reuse on, context limit 180,224, `kv_dtype` q8, capture on, activation fp32). Load line: same resident weights, 3,578,134,528 B free after load, KV cache int8g32.

| Prompt tokens | Prefill ms | Decode tok/s | Date | Conditions |
| ---: | --- | --- | --- | --- |
| 510 | 164.7 / 150.6 / 145.8 | 153.58 / 158.87 / 161.01 | 20 Nov 2026 | three fresh boots; server request log |
| 4,010 | 943.9 / 941.1 / 945.0 | 161.07 / 161.17 / 161.06 | 20 Nov 2026 | same |
| 32,010 | 8,697.6 / 9,073.4 / 9,116.9 | 136.23 / 131.83 / 129.37 | 20 Nov 2026 | same |
| 128,010 | 54,824.9 / 54,598.0 / 55,390.3 | 85.42 / 86.39 / 83.50 | 20 Nov 2026 | the 128K file fits the tuned cap (prefill runs at ~2,330 tok/s) |

Prefix reuse, default arm: after a 4,010-token turn, the save store holds 2 entries and 297,594,880 B. A stateless second request over the same text missed (`prefix_hit miss`, reused 0). First-turn ttft was 709.3 ms, and second-turn ttft was 835.6 ms (4,518 tokens).

Memory caps (default arm, read-only container, `--memory` per start, one start each): 6, 8, 12, and 20 GiB all serve. The tier line stays identical at every cap (weights stream to device, 20.7 GiB resident, 3,474,784,256 B free, context 32,768, KV fp16). Measured 500-token walls (29.0 s at 6 GiB, 22.5 s at 8, 14.0 s at 12, 7.9 s at 20) reflect first-request warm-up. Once warm, prefill runs 144 to 148 ms and decode runs ~174 tok/s at every cap.

Small card: with a holder leaving 5,836 MiB free, default startup refuses with an admission error. The server logs the needed bytes (22153406976) against the free bytes (5805441024). Qwen3.6 does not spill weights to host memory on a small card.

## Qwen3.8 Flash-Next, EXL3 3.05 bpw, on the same RTX 3090

Before the manifest default landed, an unconfigured start refused: "engine config: manifest is missing and FLASH_NEXT_ARTIFACT_MANIFEST is not set". Startup now uses the pack's `download-manifest.json` when neither config nor env names one. Every row below had `FLASH_NEXT_ARTIFACT_MANIFEST` set to that file.

Default arm (context limit 4,096, EXL3 default, server request log, greedy, 256 generated):

| Prompt tokens | Prefill ms | Decode tok/s | Date | Conditions |
| ---: | --- | --- | --- | --- |
| 572 | 8,858.0 / 8,207.0 / 8,621.3 | 28.48 / 29.16 / 29.94 | 20 Nov 2026 | three fresh boots |
| 4,071 / 4,072 | 61,769.1 / 61,094.8 / 61,136.3 | 23.74 / 22.73 / 23.36 | 20 Nov 2026 | the largest admitted 4,064-of-4,096 run decodes at 26.57 tok/s |

Tuned arm (ring budget 1,155,302,400, EXL3 rows max 2,048, batch 1): 500-token (571/572) prefill 7,838.5 / 7,539.9 / 7,602.9 ms, decode 27.57 / 28.46 / 28.49 tok/s. 4,072-token prefill 56,986.2 / 56,516.5 / 56,601.1 ms, decode 22.48 / 24.91 / 23.37 tok/s.

Prefix reuse, both arms: a two-turn conversation with 3,216 tokens on turn two reuses 2,895 of them. TTFT drops to 5.1 s (default) and 4.7 s (tuned) against 44.0 s and 39.6 s cold.

Memory caps (default arm): 6, 8, and 12 GiB refuse with fixed-overhead text because EXL3 banks need 14.18 GiB before any prompt. 20 GiB serves on the disk tier (no pinned store). It reaches ready in 20.1 s, and a 67-token prompt prefills cold in 6,086 ms (11.7 tok/s decode) and warm in 173–200 ms (20.9–22.9 tok/s).

## Qwen3.6-35B-A3B, MLX 4-bit, on Apple Silicon (36 GB)

Model file: the 19G MLX 4-bit pack.

Default arm: refused at startup with `cache_limit_bytes is missing and Q36_CACHE_LIMIT is not set`. An unconfigured start fails on this host class.

Tuned arm: shipped config `engine/config/qwen36-mac-studio-36gb.json` (cache limit 4 GiB, context 262,144, session cap 4 GiB, prefill budget 30 GiB, reserve 8,286,699,520, bf16+q8, compile on). Paths and listen address are swapped. Clean-host rerun, timings from `/v1/recent-requests`, three runs per size:

| Prompt tokens | Prefill ms (median) | Decode tok/s | Date | Conditions |
| ---: | ---: | --- | --- | --- |
| 526 | 479.5 (478.8 / 479.5 / 782.2, first hit pays compile tracing) | 86.72 / 86.83 / 86.80 | 20 Nov 2026 | greedy, 256 tokens |
| 4,022 | 3,230.3 (3,224.7 / 3,230.3 / 3,234.3) | 78.59 / 78.67 / 78.68 | 20 Nov 2026 | same |
| 32,028 | 34,539.3 (34,493.2 / 34,539.3 / 34,625.8) | 66.58 / 66.79 / 66.83 | 20 Nov 2026 | same |

Peak server RSS is 16.8 GiB (unified memory). A third admission of the 4,022-token text prefills in 3,243 ms (median), showing no reuse win on that turn. Against MLX-LM on the same file, see [Benchmarks](benchmarks.md).

We did not measure dense Qwen3.8-27B on this host because no MLX-format pack was available.

## Flint, MLX int4 21 GB fixture, on the same Apple Silicon host

The default arm equals the tuned arm because Flint on MLX has no qualified switch. Config: id flint, family flint, model and tokenizer dirs only. Server-side timings, three runs per size:

| Prompt tokens | Prefill ms (median) | Decode tok/s | Date | Conditions |
| ---: | ---: | --- | --- | --- |
| 526 | 424 (415.4 / 424.0, first 2,262 pays cold) | 52.34 / 52.73 / 52.75 | 20 Nov 2026 | greedy, 256 tokens |
| 4,022 | 2,944 (2,943.7 / 2,899.0 / 3,240.4) | 48.41 / 48.52 / 48.54 | 20 Nov 2026 | same |
| 32,028 | 46,986 (46,853 / 46,986 / 46,990) | 26.04 / 26.07 / 26.11 | 20 Nov 2026 | same |

Peak server RSS is 14.1 GiB. Load to ready takes ~1 min. The context ceiling stays at 32,768. The 32,028+256 prompt fits, while a 128K prompt refuses. A third admission of the 4,022-token text prefills in 3,257 ms (median). On the same host, 32K prefill is 36% slower than Qwen3.6-MLX (46.99 s vs 34.54 s) and decode is 61% slower (26.1 vs 66.8 tok/s).

## Qwen3.6-35B-A3B, GGUF Q4_K_M, on an AGX Xavier board

Same weights file as the 3090 rows. No reference engine runs on this board, so every row is single-engine. The board's thermal guard trips at 88 °C. Guard aborts during prefill appear as guard aborts, and completed runs record peak temperature.

Default arm:

| Prompt tokens | Prefill ms | Decode tok/s | Peak temp | Date | Conditions |
| ---: | --- | --- | --- | --- | --- |
| 486 | 4,296.4 / 2,911.8 | 20.57 / 20.69 | 54.5 °C | 20 Nov 2026 | two runs, temperature recorded |
| 3,993 | 35,755–37,128 | 19.92–19.96 | 60.5–67.0 °C | 20 Nov 2026 | three runs, temperature recorded |
| 31,650 | 1,681,370 / 1,767,007 | 14.48 / 15.38 | not recorded | 20 Nov 2026 | two runs (~28.0 and ~29.5 min prefill) before the thermal guard existed (reached 91 °C). Three later runs aborted at 88 °C during ~13-minute prefill |

Tuned arm (KV q8, context 16,384, capture and prefix reuse on, temperature unrecorded on these runs):

| Prompt tokens | Prefill ms | Decode tok/s | Date | Conditions |
| ---: | --- | --- | --- | --- |
| 486 | 3,242.0 / 2,802.8 / 2,788.3 | 20.75 / 20.72 / 20.65 | 20 Nov 2026 | three runs |
| 3,993 | 43,020.3 / 41,713.5 / 36,864.9 | 20.04 / 19.94 / 20.02 | 20 Nov 2026 | three runs |
| 15,283 (max admitted) | 401,369.9 | 18.29 | 20 Nov 2026 | one completed run (peak 87.0 °C). Two later runs aborted at the 88.0 °C guard |

A 31,650-token prompt over the tuned 16,384 cap refuses with the same `context_limit` 409 as the 3090 default arm. The board sustains ~20 tok/s decode at every admitted size. Its constraint is prefill time and the thermal guard, not decode.

## Models without a current measurement here

- **Qwen3.6 from an EXL3 pack**: refuses. Qwen3.6 on CUDA serves the GGUF checkpoint only, and refuses `"weight_format": "exl3"` before reading any weight.
- **Flint on CUDA**: refuses. Flint serves on Apple Silicon only.
- **Bolt e4b on the 3090**: refused with "gemma4 serve: 4 .gguf files in the model directory, and one expected". The store keeps four quantizations in one directory and the loader accepts one file.
- **Nemotron 3.5 Super on the 3090**: refused. The loader uses safetensors and the store holds only a GGUF conversion ("config.json: No such file" then "open shard: No such file").
