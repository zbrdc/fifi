# fifi and llama.cpp next-token divergence

This report compares next-token distributions between `fifi` and `llama.cpp` on identical weights and tokens.
The full record is [kld-vs-llamacpp-20261125.json](../../scripts/benchmarks/kld-vs-llamacpp-20261125.json), and the harness scripts live in [scripts/kld_vs_llamacpp](../../scripts/kld_vs_llamacpp/README.md).

## Setup

- **Weights**: `Qwen3.6-35B-A3B-Q4_K_M.gguf` (RTX 3090).
- **Corpus**: 12 documents from `qualification/flash-next-quality-corpus` (2,402 to 32,768 tokens each, 768 scored positions per window, 9,216 positions total).
- **Builds**: `fifi` (`5ad3798aa`, `native-cuda`) vs `llama.cpp` (`acecd5603`, CUDA).
- **Metric**: `KL(a || b)` over the top 256 probabilities per position in `f64` (covering ~98.6% of mass), plus true-token NLL and top-1 argmax agreement.

| Arm | Description |
| --- | --- |
| `engine-q8` | `fifi` with int8 (`Int8G32`) KV cache (shipped 3090 profile default). |
| `engine-f16` | `fifi` with fp16 KV cache (`kv_dtype: f32` on CUDA). |
| `llama-ub1` | `llama.cpp` with f16 KV cache, 512-token prefix batches, and 1-token decode steps. |
| `llama-ub512` | `llama.cpp` scoring inside 512-token batches. |
| `llama-q8` | `llama.cpp` with `q8_0` KV cache and 1-token decode steps. |

## Results

Cross-engine comparisons (matched cache formats):

| Pair | KL (A \|\| B) | KL (B \|\| A) | Same argmax | NLL diff (A - B) | True-token hits |
| --- | ---: | ---: | ---: | ---: | --- |
| engine int8 vs llama.cpp q8_0 | 0.0345 | 0.0337 | 96.77% | +0.0093 | 66.96% vs 67.08% |
| engine fp16 vs llama.cpp f16 | 0.0366 | 0.0351 | 96.80% | +0.0108 | 67.12% vs 67.24% |
| engine int8 vs llama.cpp f16 | 0.0363 | 0.0364 | 96.65% | +0.0090 | 66.96% vs 67.24% |

Self-comparisons (cache format or batching path):

| Pair | KL (A \|\| B) | KL (B \|\| A) | Same argmax | NLL diff (A - B) | True-token hits |
| --- | ---: | ---: | ---: | ---: | --- |
| llama.cpp 512-batch vs 1-token | 0.0330 | | 96.88% | +0.0033 | 67.05% vs 67.24% |
| engine int8 vs engine fp16 | | 0.0377 | 96.69% | -0.0019 | 66.96% vs 67.12% |
| llama.cpp q8_0 vs f16 | 0.0232 | 0.0233 | 96.70% | -0.0003 | 67.08% vs 67.24% |

Mean NLL across all positions: `fifi` 1.5105 (int8) and 1.5124 (fp16), and `llama.cpp` 1.5016 (f16) and 1.5012 (q8_0).

## Findings

- **Engine distance matches internal numerical noise.** `fifi` vs `llama.cpp` (0.034–0.037 nats, 96.8% argmax agreement) matches `llama.cpp` against itself across batching paths (0.033 nats, 96.9%) and `fifi` across caches (0.038 nats).
- **A few outlier positions drive the mean.** Median KL is 0.0006 to 0.001 nats. Less than 0.55% of positions exceed 1 nat, carrying 73% to 83% of the total KL. Without those outliers, mean KL is 0.006 to 0.007 nats and NLL gaps drop to 0.0009–0.0025 nats.
- **MoE router flips amplify small residuals.** Tracing 45 outlier positions shows initial layer-3 residual error is uniform (0.2% norm). Because the 8th and 9th routed experts often differ by only 0.02–0.05 logits, small perturbations swap experts downstream. Divergence grows 354x at outliers vs 36x elsewhere.
- **Both 8-bit KV formats use the same block scheme.** `fifi` `Int8G32` and `llama.cpp` `q8_0` both pack 32 signed int8 values per fp16 scale (`max(abs) / 127`). Only rounding ties and scale division order differ (~1/16 step).

## Limits

- Only `Q4_K_M` weights ran (no full-precision baseline).
- Runs used plain chunked prefill and 1-token decode up to 32,768 tokens without graph capture or prefix reuse.
- We did not measure throughput in this harness.
