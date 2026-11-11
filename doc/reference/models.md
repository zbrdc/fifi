# Models

| `family` | Model | Apple Silicon | NVIDIA CUDA | Checkpoint |
| --- | --- | --- | --- | --- |
| `qwen3_6_35b_a3b` | Qwen3.6-35B-A3B | yes | yes | MLX: `mlx-community/Qwen3.6-35B-A3B-4bit`. CUDA: `Qwen3.6-35B-A3B-Q4_K_M.gguf` from `ggml-org/Qwen3.6-35B-A3B-GGUF`, or an [EXL3 pack](configuration.md#exl3-pack-cuda). |
| `flint` | Flint, 39.8B total, 2.5B active | yes | pilot | MLX: `RecursiveSI/flint-int4`. CUDA: `0xBakeer/TandemLLM-Flint-EXL3-4.0`. |
| `nemotron35` | Nemotron 3.5 Super, 120B total, 12B active | no | yes, greedy only | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` at revision `8a115b9d`. |
| `qwen3_8_flash_next` | Qwen3.8-Flash-Next | in the source tree | opt-in | Not validated end to end; CUDA serving requires `flash_next.cuda_decode: true`. |
| `qwen3_8_27b_dense` | Qwen3.8-27B | in the source tree | no | Not validated end to end. |
| `gemma4` | Gemma4 | staged | staged | Registered name only: a config naming it starts no engine ("gemma4 is staged in this build"). |
| `fake` | none | yes | yes | Test engine; replies with fixed text. |

"Pilot" means the model serves short contexts and its long-context quality is not measured.

On CUDA, any single `.gguf` conversion of Qwen3.6-35B-A3B in `model_dir` serves with `Q36_ALLOW_UNPINNED_GGUF=1`; the strict open (architecture metadata and full tensor set) is the gate, and the pinned Q4_K_M file stays the qualified checkpoint and wins whenever it is present.

For CUDA GGUF, `qwen36.gguf_sha256` binds startup to an expected full-file digest, including an alternate checkpoint admitted by that switch.
See [content pinning](configuration.md#gguf-content-pinning-cuda).
A content pin does not qualify another quantization or checkpoint for quality.

## Tokenizer files

Each engine reads `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and `generation_config.json` from its `tokenizer_dir`.
For the three validated families, startup compares the files with the SHA-256 hashes of the tested release and refuses others.
For Qwen3.6 CUDA, use the tokenizer files from `mlx-community/Qwen3.6-35B-A3B-4bit` with either the GGUF file or the EXL3 pack. The other validated checkpoints listed above include matching files.

## Weight formats

| Format | Used by |
| --- | --- |
| MLX affine-quantized safetensors; the quantization is read from `config.json` | Qwen3.6 and Flint on Apple Silicon |
| GGUF with F16, Q4_K, Q5_0, Q6_K and Q8_0 tensors | Qwen3.6 on CUDA |
| EXL3 | Qwen3.6 and Flint on CUDA |
| NVFP4 safetensors | Nemotron 3.5 Super |

## Context limits

| Model and backend | Default | Largest |
| --- | ---: | ---: |
| Qwen3.6, Apple Silicon | 262,144 | 262,144 tested on a 36 GB Mac |
| Qwen3.6, CUDA | 32,768 | 262,144 tested on a 24 GB RTX 3090 |
| Flint, Apple Silicon | 32,768 | 32,768 |
| Flint, CUDA | 131,072 | 262,144 |
| Nemotron 3.5 Super | set in config | |
