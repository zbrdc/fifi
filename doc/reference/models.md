# Models

| `family` | Model | Apple Silicon | NVIDIA CUDA | Checkpoint |
| --- | --- | --- | --- | --- |
| `qwen3_6_35b_a3b` | Qwen3.6-35B-A3B | yes | yes | MLX: `mlx-community/Qwen3.6-35B-A3B-4bit`. CUDA: `Qwen3.6-35B-A3B-Q4_K_M.gguf` from `ggml-org/Qwen3.6-35B-A3B-GGUF`. CUDA serves only the GGUF format for this model. |
| `flint` | Flint, 39.8B total, 2.5B active | yes | no | `RecursiveSI/flint-int4`. |
| `nemotron35` | Nemotron 3.5 Super, 120B total, 12B active | no | yes, greedy only | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` at revision `8a115b9d`. |
| `qwen3_8_flash_next` | Qwen3.8-Flash-Next | in the source tree | opt-in | Not validated end to end. A `native-cuda` Linux build serves it on the card unless `flash_next.cuda_decode` is false. |
| `qwen3_8_27b_dense` | Qwen3.8-27B | in the source tree | no | Not validated end to end. |
| `gemma4` | Nemotron Bolt E4B | no | Bolt E4B only | The Bolt E4B Q8_0 GGUF or a K-quant conversion of it. The loader detects a bare weights file from `general.architecture` = `gemma4`. Other Gemma4 checkpoints do not serve. |
| `inkling2_mini` | Inkling 2 Mini | experimental, opt-in | no | [Official NVFP4 and MXFP8 safetensors](../how-to/inkling.md), or a text GGUF (mainline `inkling2_loop` or older `inkling2` layout) |
| `inkling2_loop` | Inkling 2 Mini | CPU | yes | Native ModelOpt safetensors and tokenizer files from the [pinned release](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7). |
| `sensenova` | SenseNova-Loop-9B (text) | CPU only | opt-in | Experimental host path for [SenseNova-Loop-9B GGUF](https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF). |
| `raon1` | Raon-1-mini-preview, 28.8B total, 4.3B active | no | experimental | `raon-1-mini-preview-Q4_K_M.gguf` from `KRAFTON/Raon-1-mini-preview-gguf`. The loader reads the tokenizer and chat template from the GGUF file, so the directory needs no tokenizer files. |
| `fake` | none | yes | yes | Test engine. Replies with fixed text. |

"Pilot" means the model serves short contexts and its long-context quality is not measured. "Experimental" and "opt-in" paths are not validated end to end.

See [Serve a model without a config file](../how-to/serve-a-model.md) for starting a model from a path or a name.

A model path that holds only the GGUF weights file starts from its `general.architecture` metadata: `qwen35moe` selects the Qwen3.6 family and `gemma4` selects Bolt E4B. Startup refuses other known architectures with a note: `sensenova_loop` (pass `--family sensenova`), `qwen4exp`, `nemotron_h_moe`, `dflash` and `clip`. A `raon1` weights file serves as the table above describes.

On CUDA, any `.gguf` conversion of Qwen3.6-35B-A3B in `model_dir` serves. When there are several, the loader picks the largest (`Q36_ALLOW_UNPINNED_GGUF=0` serves only the pinned file). The strict open (architecture metadata and full tensor set) is the gate. The pinned Q4_K_M file stays the qualified checkpoint and wins whenever it is present.

For CUDA GGUF, `qwen36.gguf_sha256` binds startup to an expected full-file digest, including an alternate checkpoint admitted by that switch.
See [content pinning](configuration.md#gguf-content-pinning-cuda).
A content pin does not qualify another quantization or checkpoint for quality.

## SenseNova-Loop

With `ENGINESRV_API_KEY` set, run:

```sh
enginesrv serve --model /path/to/SenseNova-Loop-9B-Q4_K_M.gguf --family sensenova
```

The experimental CPU path serves text with a 4,096-token context. It uses the selected
GGUF's embedded tokenizer and chat template. Leave `--tokenizer-dir` unset. In a JSON
config, set it to the same path as `model_dir`. Set `"x-thinking": "off"` in a chat
request to disable thinking. This path does not use a GPU, vision inputs, MTP draft
decoding, or context and KV overrides.

On a `native-cuda` Linux build, `FIFI_SENSENOVA_CUDA=1` serves the same checkpoint from the
card instead. This path is opt-in and not validated end to end.

## Inkling 2 Mini

Inkling 2 Mini serves on Apple Silicon (`inkling2_mini`, see [Run Inkling 2 Mini on a Mac](../how-to/inkling.md)). It also serves on one CUDA device or the CPU (`inkling2_loop`, see [Run Inkling 2 Mini on CPU or CUDA](../how-to/inkling-loop.md)).

## Tokenizer files

A `raon1` engine reads its tokenizer and chat template from the GGUF file instead.
Every other engine reads `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and `generation_config.json` from its `tokenizer_dir`.
For Qwen3.6, Flint and Nemotron, startup compares the files with the SHA-256 hashes of the tested release and refuses others.
For Qwen3.6 CUDA, use the tokenizer files from `mlx-community/Qwen3.6-35B-A3B-4bit` with the GGUF file. The other validated checkpoints listed above include matching files.

## Weight formats

| Format | Used by |
| --- | --- |
| MLX affine-quantized safetensors (quantization in `config.json`) | Qwen3.6, Qwen3.8-27B and Flint on Apple Silicon |
| GGUF with F16, Q4_K, Q5_0, Q6_K and Q8_0 tensors | Qwen3.6 on CUDA |
| GGUF Q8_0, and K-quant conversions of it | Bolt E4B on CUDA |
| GGUF, matrices dequantized on demand | SenseNova-Loop on CPU (the published Q4_K_M file) |
| GGUF, mainline `inkling2_loop` or older `inkling2` layout | Inkling 2 Mini on Apple Silicon, and on CPU or CUDA with family `inkling2_loop` |
| NVFP4 safetensors | Nemotron 3.5 Super |
| Mixed NVFP4 and MXFP8 safetensors | Inkling 2 Mini on Apple Silicon |

## Context limits

| Model and backend | Default | Largest |
| --- | ---: | ---: |
| Qwen3.6, Apple Silicon | 262,144 | 262,144 tested on a 36 GB Mac |
| Qwen3.6, CUDA | 32,768 | 262,144 tested on a 24 GB RTX 3090 |
| Flint, Apple Silicon | 32,768 | 32,768 |
| Nemotron 3.5 Super | set in config | |
| SenseNova-Loop, CPU | the GGUF's declared context, capped at 4,096 | 4,096 |
| Inkling 2 Mini, Apple Silicon | 1,024 | Configurable up to the model limit of 1,048,576, subject to admission |
