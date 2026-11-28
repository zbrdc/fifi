# Benchmarks

These numbers show what fifi does on tested hardware.
Each table compares runs with the same weights on the same machine, measured in the same window on the same day.
Every fifi figure below comes from 20 Nov 2026 on `main` as of that date.
Workload: greedy sampling, 256 generated tokens, fresh server boot per run, three runs per prompt size. Ranges list all three runs.
Reference engines appear only where a same-window pair exists (llama.cpp on RTX 3090 and MLX-LM on Apple Silicon). Hosts without a reference engine show single-engine figures.

Day-to-day noise is about 4%, so smaller differences are not meaningful.
"Prefill" is the server's own prefill interval. "Decode" is its tokens-per-second counter. Where a reference engine reports differently (prompt-eval and eval intervals), the table names the column.

## Qwen3.6-35B-A3B, GGUF Q4_K_M, on an RTX 3090 (24 GB)

fifi: default arm (nothing set, KV cache fp16, context 32,768). llama.cpp: `llama-server` 0.4.0-dev build 10934, commit `acecd5603`, CUDA build, `-c 32768 -ngl 99 --temp 0`, same GGUF and prompt files, four 32,768-token slots. Same window, card, and host. Figures show fifi `prefill_ms` vs llama.cpp `prompt_eval_ms` (excluding generation) and each engine's decode counter across three runs.

| Prompt tokens | fifi prefill ms | llama.cpp prefill ms | fifi decode tok/s | llama.cpp decode tok/s |
| ---: | --- | --- | --- | --- |
| 510 | 144.5 / 145.4 / 146.1 | 245.10 / 241.71 / 238.07 | 164.71 / 164.95 / 165.28 | 145.73 / 143.55 / 145.51 |
| 4,010 | 687.3 / 688.2 / 691.6 | 1,381.32 / 1,414.53 / 1,390.14 | 159.32 / 159.51 / 160.10 | 141.04 / 143.26 / 142.95 |
| 32,010 | 7,585.7 / 7,632.2 / 7,636.0 | 11,859.71 / 12,114.23 / 12,186.57 | 130.54 / 133.55 / 134.07 | 129.24 / 125.06 / 125.35 |

On this pair, fifi is faster at every size: prefill by 1.6x to 2.0x, decode by 7 to 13 percent. Medians are 165.0 vs 145.5, 159.5 vs 143.0, and 133.6 vs 125.4. fifi's default 32,768 context refuses a 128,010-token prompt. We did not run llama.cpp at that size.

## Qwen3.6-35B-A3B, MLX 4-bit, on Apple Silicon (36 GB)

fifi: shipped 36 GB config (`engine/config/qwen36-mac-studio-36gb.json`) with paths swapped. See [Current status](status.md) for its figure table and switch list. The default arm refuses to start on this host (missing `cache_limit_bytes`), so only the tuned arm pairs.

MLX-LM 0.32.2 (Python 3.12 venv, as 0.29.1 refuses this model type) ran the same model file and prompts in-process with no HTTP layer. It used greedy sampling and 256 generated tokens.

Per-phase, at 526 / 4,022 / 32,028 prompt tokens:

| Metric | fifi | MLX-LM |
| --- | --- | --- |
| Prefill tok/s | 1,098 / 1,246 / 928 | 1,449 / 1,661 / 1,221 |
| Decode tok/s | 86.9 / 78.8 / 66.8 | 117.0 / 115.3 / 95.4 |

**MLX-LM is faster at every size on this host: 24 to 32 percent faster prefill and 26 to 30 percent faster decode.** This is the headline gap for the MLX backend.

Caveats: the MLX-LM figure is in-process (no HTTP, no server), while the fifi figure comes from the server's counters. Both use greedy sampling on the same file. The comparison states its layers honestly, though it is not a like-for-like server pair.

## Single-engine measurements

Flint on Apple Silicon and Qwen3.6 on AGX Xavier have no same-window reference engine pair.
See [Current status](status.md) for their prefill, decode, memory-cap, and thermal-guard measurements.
