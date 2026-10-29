# Benchmarks

These numbers show what fifi does on the hardware it is tested on.
Each table compares runs with the same weights on the same machine.
They are single runs, and day-to-day noise is about 4%, so differences smaller than that are not meaningful.
Tables were measured on different days and builds; compare within a table, not across tables.

Unless a table says otherwise, the workload is a recorded multi-turn chat session whose prompt grows with every turn, and fifi runs with prefix reuse on.
"Time to first token" is how long a turn takes before the first reply token arrives.

## Qwen3.6 on Apple Silicon (M4 Max, 36 GB)

Against llama.cpp (Metal, 8-bit KV cache, `-ub 2048`), 12 turns:

| Metric | fifi | llama.cpp |
| --- | --- | --- |
| Sum of time to first token | 63.1 s | 64.35 s |
| Decode speed | 63.3 tokens/s | 57.4 tokens/s |

A later fifi build measured 57.1 s for the same sum.

Against MLX-LM 0.32.2 (mlx 0.32.4, `--prompt-cache`), the same 12 turns:

| Metric | fifi (8-bit KV) | MLX-LM (16-bit KV) | MLX-LM (`--kv-bits 8`) |
| --- | --- | --- | --- |
| Sum of time to first token | 53.5 s | 44.9 s (11 turns) | 312.3 s |
| First turn, nothing reused | 7.93 s | 7.98 s | |
| Median later turn | 3.87 s | 3.55 s | 26.16 s |
| Decode speed | 62.8 tokens/s | 100.1 tokens/s | 78.8 tokens/s |
| Process memory | 13.0 GB | 17.6 GB | |

The MLX-LM run with its default cache failed on the twelfth turn, so its sum covers 11 turns.
About half of fifi's decode gap to MLX-LM comes from the 8-bit KV cache; the rest is kernel time per step.

## Qwen3.6 on an RTX 3090 (24 GB)

Against llama.cpp (8-bit KV cache, `-ub 1024`), same GGUF file, 131,072-token context:

| Metric | fifi | llama.cpp |
| --- | --- | --- |
| Time to first token, 127,788-token prompt, nothing reused | 50.2 s | 52 s |
| Time to first token, later turn at 128K tokens | 5.40 s | 5.7 s |
| Decode speed | 24 to 68% faster at every measured depth | baseline |

Long sessions:

- With the 180,224-token profile, a session ran 33 turns up to 178,229 tokens with no errors and passed 4 of 4 recall checks.
- With the 262,144-token profile, a session reached 255,966 tokens with no errors and passed 7 of 7 recall checks. A turn at 192,014 tokens took 4.61 s to its first token; repeating the 255,966-token prompt unchanged took 0.474 s.

## Flint on Apple Silicon

| Prompt length | Decode time per token |
| ---: | ---: |
| 22 tokens | 23.1 ms |
| 7,800 tokens | 25.6 ms |
| 26,600 tokens | 40.2 ms |

Prompt processing runs at about 1,000 tokens/s.

Compared with the publisher's own runtime, fifi's next-token distributions have a mean KL divergence of 0.062 and 94.0% top-1 agreement.
The publisher runtime's two modes differ from each other by 0.074 and 92.4%.
