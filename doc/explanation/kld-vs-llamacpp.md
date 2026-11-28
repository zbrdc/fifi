# fifi and llama.cpp on the same weights: how far apart their next-token distributions are

The record of the run is [kld-vs-llamacpp-20261125.json](../../scripts/benchmarks/kld-vs-llamacpp-20261125.json). The scripts that made it are in [scripts/kld_vs_llamacpp](../../scripts/kld_vs_llamacpp/README.md).

## The question

Three KL figures were being set side by side, and they did not measure the same thing:

- Flint's q8 KV profile against its f16 profile, teacher-forced on 8,448 repository-source positions:
  a mean of 0.254 nats on the legacy kernel and 0.266 on the tiled kernel, which the profile uses by
  default (`doc/reference/environment.md`, the `FIFI_FLINT_CUDA_Q8_TILED` row).
- llama.cpp's q8_0 KV cache against its f16 cache: published by others for this model on other corpora with other scorers.
- the engine against llama.cpp: not measured.

The first two compare an engine with itself under two cache forms. Neither says how far the two engines are from each other. This run measures both kinds in one harness: one GGUF file, one token stream, one metric, both engines, and each engine against itself under a second numerical path.

## What ran

| Arm | What scores the positions |
|---|---|
| `engine-q8` | The engine's `kv_quality_leg` with the int8 (`Int8G32`) KV cache: the cache form of the shipped 3090 profiles, `qwen36-cuda-3090-kv8-prefix.json` among them. |
| `engine-f16` | The same leg with the fp16 KV cache: the control cache. On the CUDA arm a start keeps the cache this way for `kv_dtype: f32`, the member's default (`settings.rs`, `kv_dtype_of`; `start.rs`, `kv_mode_of`), so it is the form of the `qwen36-rtx3090-24gb-fp16kv.json` profile and of a start whose config names no `kv_dtype`. |
| `engine-q8-vs-llama-q8` | The int8 leg once more, with `llama-q8` as its reference: the pair of the two 8-bit caches. |
| `llama-ub1` | A driver on `libllama`: f16 KV cache under flash attention (llama.cpp's `auto` setting, which resolved to on), the prefix in 512-token batches, then one-token decodes for the scored positions. |
| `llama-ub512` | The same driver with the scored positions inside 512-token batches: llama.cpp against itself under a second path. |
| `llama-q8` | The same driver as `llama-ub1`, with llama.cpp's q8_0 KV cache: llama.cpp against itself under a second cache form. |

All arms read the same file (`Qwen3.6-35B-A3B-Q4_K_M.gguf`, SHA-256 in the record) through the same token ids. The ids are the twelve documents of `qualification/flash-next-quality-corpus`, tokenized once; llama.cpp's tokenizer gives the same ids. Each document is one window of 2,402 to 32,768 tokens with 768 scored positions: 9,216 positions in total.

- Engine: `main` at `5ad3798aa`, release build with the `native-cuda` feature.
- llama.cpp: `acecd5603`, the CUDA backend.
- Card: one RTX 3090.

Each row narrows to the sampler vocabulary (248,077 ids) and normalizes in f64. For a pair of arms, `KL(a || b)` sums over the 256 most probable ids of a, and `KL(b || a)` over those of b. Leaving out the remaining terms (~1.4% of probability mass) means neither figure bounds the full-vocabulary KL. For llama.cpp's two cache forms, full-vocabulary KL is 0.0240 and 0.0236 against top-row sums 0.0233 and 0.0232 (2% to 3% higher). Bounds are 95% intervals from a bootstrap over the twelve windows.

Each arm ran at least twice: `llama-ub512`, `llama-ub1`, and `engine-f16` directly and through `scripts/kld_vs_llamacpp/run.sh`. `llama-ub1` and `llama-q8` ran three times each through `run.sh`, producing identical sidecars. We discarded an early `llama-q8` run that omitted the cache type flag and matched `llama-ub1` byte for byte. The int8 arm ran once against each of its three references with identical parameters. On 2026-11-26, `llama-ub1` and `llama-q8` ran again from the committed source, and `score_all.sh` generated the thirteen outputs in this record. Ten score and concentration files from 2026-11-25 match byte for byte (`shape.txt` differs only in five path lines), and all scored numbers match.

## Results

Each engine's cache form against llama.cpp's cache of the same form:

| Pair | KL, first against second | KL, second against first | Same argmax | NLL of the true token, first minus second | Greedy hits on the true token |
|---|---:|---:|---:|---:|---:|
| engine int8 cache, llama.cpp q8_0 cache | 0.0345 | 0.0337 | 96.77% | +0.0093 | 66.96% against 67.08% |
| engine fp16 cache, llama.cpp f16 cache | 0.0366 | 0.0351 | 96.80% | +0.0108 | 67.12% against 67.24% |

The shipped 3090 profile against llama.cpp's default, two different cache forms:

| Pair | KL, first against second | KL, second against first | Same argmax | NLL of the true token, first minus second | Greedy hits on the true token |
|---|---:|---:|---:|---:|---:|
| engine int8 cache, llama.cpp f16 cache | 0.0363 | 0.0364 | 96.65% | +0.0090 | 66.96% against 67.24% |

Each engine against itself under a second path:

| Pair | KL, first against second | KL, second against first | Same argmax | NLL of the true token, first minus second | Greedy hits on the true token |
|---|---:|---:|---:|---:|---:|
| llama.cpp, 512-token batches, 1-token decodes | 0.0330 | | 96.88% | +0.0033 | 67.05% against 67.24% |
| engine int8 cache, engine fp16 cache | | 0.0377 | 96.69% | −0.0019 | 66.96% against 67.12% |
| llama.cpp q8_0 cache, llama.cpp f16 cache | 0.0232 | 0.0233 | 96.70% | −0.0003 | 67.08% against 67.24% |

KL in nats per position, means over 9,216 positions. A blank cell marks a direction the harness omits: `KL(x || y)` over top rows of x needs every logit of y, which only `llama-ub1` and `llama-q8` write unless an engine leg computed it against its reference.

Means of the NLL over all positions: engine 1.5105 with the int8 cache and 1.5124 with the fp16 cache, llama.cpp 1.5016 with the f16 cache and 1.5012 with the q8_0 cache.

## Where the divergence sits

- **Few positions carry the mean.** The median KL is 0.0006 to 0.001 nats in every pair. Only 0.41% to 0.55% of positions exceed one nat, yet they hold 73% to 83% of the sum. Without them, means are 0.006 to 0.007.
- **They are the same positions.** Of the 69 positions where the int8 engine and llama.cpp differ by > 1 nat on true-token log-probability, 35 also exceed 1 nat between llama.cpp's q8_0 and f16 caches, and 39 between its two batching paths. At 8 bits, 43 of 81 overlap llama.cpp's cache forms. At 16 bits, 46 of 78 overlap its batching paths. Chance gives 0.5 to 0.6.
- **Two documents hold most of it.** The 6,774-token window has 0.151 to 0.214 nats in every pair. The second 32,768-token window has 0.059 to 0.145. Together they hold 71% to 81% of each pair's sum. The other ten lie between 0.002 and 0.067.
- **No arm is flatter.** Entropy over the top 256 rows is 1.082 to 1.087 nats in all five arms. Mean probability of the true token is 0.604 to 0.605.

## What the outlier positions are

We traced the outlier positions between the int8 and fp16 caches layer by layer in three windows (6K, 32K, and 80K tokens): 45 outlier positions against 2,259 others.

- **No defect of the int8 cache shows.** Layers 0 to 2 keep no K/V cache and match in both arms. The first difference appears at layer 3 (the first attention layer) at 0.2% of residual norm across all positions and context lengths.
- **The difference grows through expert choices that are nearly tied.** The eighth selected expert leads the best unselected one by 0.02 to 0.05 logits at the median. A tiny residual difference swaps an expert of similar weight, amplifying differences downstream. Each swap multiplies divergence by ~2.6 in layers 3 to 9 across all positions.
- **The outliers are the positions with more swaps, not with closer ties.** The expert set differs in 27% of layers 3 to 9 at outliers vs 23% elsewhere, 62% vs 34% in layers 10 to 19, and 95% vs 46% in layers 20 to 38. From layer 3 to the last layer, divergence grows 354x at outliers vs 36x elsewhere. Their smallest router gaps are not smaller than others.
- **A different expert set alone is harmless.** 734 of 739 other positions in one window differ in some layer with a median KL of 0.0002 to 0.0009 nats.

Host recomputation of the eight selected experts agrees with device picks at all 184,320 pairs of position and layer, and two passes gave identical traces. A smaller error at the first attention layers should shorten this tail by keeping more layers away from a swap. That follows from the figures above, though we did not measure it.

## Reading

1. **Like against like, the engines are as far from each other as each engine is from itself under a numerical change.** Int8 vs llama.cpp q8_0: 0.0345 and 0.0337 nats, 96.8% agreement. Fp16 vs f16: 0.0366 and 0.0351, 96.8%. llama.cpp across batching paths: 0.033, 96.9%. Against llama.cpp's f16 cache, the engine gives 0.0363 and 0.0364 (int8) and 0.0366 and 0.0351 (fp16), within noise intervals. A KL of this size does not show which distribution is nearer to the model.
2. **The true token's NLL is the one measure with an external reference.** Against llama.cpp, engine NLL is 0.009 nats higher (8-bit vs 8-bit and 8-bit vs f16) and 0.011 nats higher (fp16 vs f16), or 0.9% to 1.1% in perplexity. Without outliers, differences drop to 0.0025, 0.0015, and 0.0009. Bootstrap intervals are +0.0004 to +0.0221, +0.0024 to +0.0162, and +0.0009 to +0.0240.
3. **The int8 cache costs about what llama.cpp's q8_0 cache costs, in this harness.** Against its fp16 cache, engine int8 gives 0.038, and llama.cpp q8_0 gives 0.023. Medians (0.0008 and 0.0009) and agreement rates (96.69% and 96.70%) match. The mean is 1.6x larger because 0.50% of positions exceed one nat vs 0.41%. Neither cache moves NLL (−0.0019 engine, −0.0003 llama.cpp, both crossing zero).
4. **Figures of this kind compare only inside one harness.** An independent measurement of llama.cpp's q8_0 cache (top-40 log-probabilities, BF16 weights, ~250,000 tokens in six categories) gives 0.039, and 0.024 for dense Qwen3.6 27B ([localbench, 2026-04-24](https://localbench.substack.com/p/kv-cache-quantization-benchmark)). Our 0.023 matches that scale. A figure one order lower would need a corpus without outliers.
5. **The two 8-bit caches are one quantization scheme.** Engine `Int8G32` and llama.cpp `q8_0` both store 32 signed 8-bit codes with one fp16 scale (`max(abs) / 127`) and round values to nearest integer (`kv8_append.cu` vs `cpy-utils.cuh`). Three details differ: llama.cpp multiplies by reciprocal fp32 scale and rounds ties away from zero, while the engine divides by fp16 scale and rounds ties to even (clamping to ±127). The scale difference leaves ~1/16 code step. Thus the 0.038 vs 0.023 difference in point 3 is not a format difference, and a q8_0 writer would match the existing cache. The proposal to add one is closed.

## Limits

- We did not score either engine against full-precision weights because only the Q4_K_M file was on the host. Every figure compares two runs of the same quantized weights.
- The engine arms run the leg's plain step: prefix through chunked prefill, then one decode step per position. We did not exercise graph capture, prefix reuse, on-demand KV, or contexts beyond 32,768.
- Twelve documents and 9,216 positions: KL intervals are about twice their means wide, and three of six NLL intervals cross zero.
- We did not measure throughput.
- We compared the two 8-bit cache formats by reading the two quantizers, not by running them on the same tensors.
- We traced outlier positions in three of twelve windows without a 32-bit reference, so we do not know how the fp16 cache departs from exact arithmetic there.
