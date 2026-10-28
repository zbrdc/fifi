# Status

## Models

| Family (`family` key) | Model | Backends | State |
| --- | --- | --- | --- |
| `qwen3_6_35b_a3b` | Qwen3.6-35B-A3B | Apple Silicon, NVIDIA CUDA | Serving: streaming, tool calls, cancellation, prefix reuse. 262,144-token context on a 36 GB Mac; up to 262,144 on a 24 GB RTX 3090 ([cuda.md](cuda.md)). |
| `flint` | Flint (39.8B total, 2.5B active) | Apple Silicon; NVIDIA CUDA as a pilot | Serving on Apple Silicon. The CUDA pilot serves short contexts; its long-context quality is not measured ([flint.md](flint.md)). |
| `nemotron35` | Nemotron 3.5 Super (NVFP4) | NVIDIA CUDA with CPU experts | Serving, slowly: prefill runs one token at a time by default ([nemotron.md](nemotron.md)). |
| `qwen3_8_flash_next` | Qwen3.8-Flash-Next | Apple Silicon | In the source tree; not validated end to end. |
| `qwen3_8_27b_dense` | Qwen3.8-27B | Apple Silicon | In the source tree; not validated end to end. |

Qwen3.6 changes are tested against a reference implementation: mean signed dNLL at most 0.0117, top-1 agreement at least 97%, and mean KL at most 0.004.

## Weight formats

- MLX affine and oQ safetensors; the quantization is read from the checkpoint metadata.
- GGUF with F16, Q4_K, Q5_0, Q6_K and Q8_0 tensors (CUDA).
- NVFP4 safetensors (Nemotron 3.5 Super).
- EXL3 (Flint on CUDA).

## Benchmarks

All results are single runs on recorded multi-turn sessions, with real files as tool outputs and prefix reuse on.
Each comparison uses the same weights on the same machine. Day-to-day noise is about 4%.

### Qwen3.6 on Apple Silicon (M4 Max, 36 GB)

Against llama.cpp (Metal, q8_0 KV, `-ub 2048`), 12 turns:

| Metric | fifi | llama.cpp |
| --- | --- | --- |
| Sum of first-token times | 57.1 s | 64.35 s |
| Decode speed | 63.3 tokens/s | 57.4 tokens/s |

The fifi first-token sum is from a later build than the llama.cpp run; on the same day fifi measured 63.1 s.

Against MLX-LM 0.32.2 (mlx 0.32.4, `--prompt-cache`), same 12 turns:

| Metric | fifi (8-bit KV) | MLX-LM (16-bit KV) | MLX-LM (`--kv-bits 8`) |
| --- | --- | --- | --- |
| Sum of first-token times | 53.5 s | 44.9 s (11 turns) | 312.3 s |
| Cold first turn | 7.93 s | 7.98 s | — |
| Median warm turn | 3.87 s | 3.55 s | 26.16 s |
| Decode speed | 62.8 tokens/s | 100.1 tokens/s | 78.8 tokens/s |
| Process memory | 13.0 GB | 17.6 GB | — |

The default MLX-LM run failed on its twelfth turn, so its sum covers 11 turns.
About half of fifi's decode gap to MLX-LM comes from the 8-bit KV cache; the rest is per-step kernel time.

With block prefix matching, a 174,829-token prompt whose last 23 tokens were rewritten reached its first token in 11.8 s, against 40.7 s without it.

### Qwen3.6 on an RTX 3090 (24 GB)

Against llama.cpp (q8_0 KV, `-ub 1024`), same GGUF, 131,072-token capacity:

| Metric | fifi | llama.cpp |
| --- | --- | --- |
| Cold 127,788-token prompt, time to first token | 50.2 s | 52 s |
| Warm 128K-token turn, time to first token | 5.40 s | 5.7 s |
| Decode speed | 24–68% faster at every measured depth | baseline |

Long sessions:

- 180,224-token profile: 33 warm turns up to 178,229 tokens, 4 of 4 recall checks passed, no errors.
- 262,144-token profile: a session reached 255,966 tokens with 7 of 7 recall checks and no errors. A warm turn at 192,014 tokens took 4.61 s to first token; an identical repeat at 255,966 tokens took 0.474 s.

### Flint on Apple Silicon

| Prompt length | Decode time per token |
| ---: | ---: |
| 22 tokens | 23.1 ms |
| 7,800 tokens | 25.6 ms |
| 26,600 tokens | 40.2 ms |

Prefill runs at about 1,000 tokens/s.
fifi's output stays closer to the publisher's runtime than that runtime's two modes stay to each other: KL 0.062 against 0.074, and top-1 agreement 94.0% against 92.4%.
