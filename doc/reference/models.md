# Models

| `family` | Model | Apple Silicon | NVIDIA CUDA | Checkpoint |
| --- | --- | --- | --- | --- |
| `qwen3_6_35b_a3b` | Qwen3.6-35B-A3B | yes | yes | MLX: `mlx-community/Qwen3.6-35B-A3B-4bit`. CUDA: `Qwen3.6-35B-A3B-Q4_K_M.gguf` from `ggml-org/Qwen3.6-35B-A3B-GGUF`, only this file. |
| `flint` | Flint, 39.8B total, 2.5B active | yes | pilot | MLX: `RecursiveSI/flint-int4`. CUDA: `0xBakeer/TandemLLM-Flint-EXL3-4.0`. |
| `nemotron35` | Nemotron 3.5 Super, 120B total, 12B active | no | yes, greedy only | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` at revision `8a115b9d`. |
| `qwen3_8_flash_next` | Qwen3.8-Flash-Next | in the source tree | no | Not validated end to end. |
| `qwen3_8_27b_dense` | Qwen3.8-27B | in the source tree | no | Not validated end to end. |
| `fake` | none | yes | yes | Test engine; replies with fixed text. |

"Pilot" means the model serves short contexts and its long-context quality is not measured.

## Tokenizer files

Each engine reads `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and `generation_config.json` from its `tokenizer_dir`.
For the three validated families, startup compares the files with the SHA-256 hashes of the tested release and refuses others.
The checkpoints above include matching files, except the Qwen3.6 GGUF: take its tokenizer files from `mlx-community/Qwen3.6-35B-A3B-4bit`.

## Weight formats

| Format | Used by |
| --- | --- |
| MLX affine-quantized safetensors; the quantization is read from `config.json` | Qwen3.6 and Flint on Apple Silicon |
| GGUF with F16, Q4_K, Q5_0, Q6_K and Q8_0 tensors | Qwen3.6 on CUDA |
| EXL3 | Flint on CUDA |
| NVFP4 safetensors | Nemotron 3.5 Super |

## Context limits

| Model and backend | Default | Largest |
| --- | ---: | ---: |
| Qwen3.6, Apple Silicon | 262,144 | 262,144 tested on a 36 GB Mac |
| Qwen3.6, CUDA | 32,768 | 262,144 tested on a 24 GB RTX 3090 |
| Flint, Apple Silicon | 32,768 | 32,768 |
| Flint, CUDA | set in config | 262,144 |
| Nemotron 3.5 Super | set in config | |
