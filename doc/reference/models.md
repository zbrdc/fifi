# Models

| `family` | Model | Apple Silicon | NVIDIA CUDA | Checkpoint |
| --- | --- | --- | --- | --- |
| `qwen3_6_35b_a3b` | Qwen3.6-35B-A3B | yes | yes | MLX: `mlx-community/Qwen3.6-35B-A3B-4bit`. CUDA: `Qwen3.6-35B-A3B-Q4_K_M.gguf` from `ggml-org/Qwen3.6-35B-A3B-GGUF`, or an [EXL3 pack](configuration.md#exl3-pack-cuda). |
| `flint` | Flint, 39.8B total, 2.5B active | yes | pilot | MLX: `RecursiveSI/flint-int4`. CUDA: `0xBakeer/TandemLLM-Flint-EXL3-4.0`. |
| `nemotron35` | Nemotron 3.5 Super, 120B total, 12B active | no | yes, greedy only | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` at revision `8a115b9d`. |
| `qwen3_8_flash_next` | Qwen3.8-Flash-Next | in the source tree | opt-in | Not validated end to end; A `native-cuda` Linux build serves it on the card unless `flash_next.cuda_decode` is false. |
| `qwen3_8_27b_dense` | Qwen3.8-27B | in the source tree | no | Not validated end to end. |
| `inkling2_loop` | Inkling 2 Mini | CPU pilot | opt-in | Native ModelOpt safetensors and tokenizer files from the [pinned release](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7). |
| `gemma4` | Gemma4 | staged | staged | Registered name only: a config naming it starts no engine ("gemma4 is staged in this build"). |
| `fake` | none | yes | yes | Test engine; replies with fixed text. |

"Pilot" means the model serves short contexts and its long-context quality is not measured.

On CUDA, any `.gguf` conversion of Qwen3.6-35B-A3B in `model_dir` serves (the largest when there are several; `Q36_ALLOW_UNPINNED_GGUF=0` serves only the pinned file); the strict open (architecture metadata and full tensor set) is the gate, and the pinned Q4_K_M file stays the qualified checkpoint and wins whenever it is present.

For CUDA GGUF, `qwen36.gguf_sha256` binds startup to an expected full-file digest, including an alternate checkpoint admitted by that switch.
See [content pinning](configuration.md#gguf-content-pinning-cuda).
A content pin does not qualify another quantization or checkpoint for quality.

## Inkling 2 Mini

On a 64-bit build, set `FIFI_INKLING_CPU_REFERENCE=1` and choose family `inkling2_loop` in the `enginesrv` configuration. Use the pinned release directory for `model_dir` and `tokenizer_dir`. On a 64-bit Linux build with `native-cuda`, set `FIFI_INKLING_CUDA_REFERENCE=1` instead to run the BF16 and packed expert projections on the first visible CUDA device. Set `CUDA_VISIBLE_DEVICES` to choose the card. Set only one Inkling backend flag. Both pilots support raw completions and plain-text chat, with one active sequence. The default context is 128 tokens including the prompt. Set `FIFI_INKLING_CONTEXT_TOKENS` to opt into a larger context, up to 8,192 tokens on the pinned checkpoint. Routing, attention, KV and convolution state run on the host. On Linux, extended context and prefix storage use FiFi's cgroup-aware host admission. Routing expert page-cache warming receives the RAM budget after the text-state reserve.

`enginesrv check` inspects the safetensors inventory and reports the BF16 KV storage plan for the fixed loop. With a pilot enabled it also checks the pinned tokenizer and the short-context settings. CUDA admission checks available device memory at startup.

For the CUDA pilot, `FIFI_INKLING_CUDA_RESIDENT_EXPERT_MIB` optionally reserves a budget for packed expert banks. The loader places complete banks for the repeated core layers first and includes their bytes in device admission. A budget of zero uses streamed experts. Use `FIFI_INKLING_CUDA_PROFILE=1` to log per-token time by projection and upload phase. Set `FIFI_INKLING_CUDA_DENSE_STREAM=1` to use the experimental vectorized BF16 projections. This path queues all three controller matrices on one stream and copies only the final gain and forget rows back to the host.

The Unsloth Inkling 2 Mini fixed-loop GGUF also works in the text pilot. Point `model_dir` to the GGUF file or a directory with one text GGUF, and use the original Thinking Machines tokenizer files in `tokenizer_dir`. For `enginesrv check`, give the GGUF directory and pass `--tokenizer-dir` separately.

Both backends map the original packed weights from disk and decode or upload only selected expert payloads. By default, the CUDA backend keeps BF16 dense matrices resident for vectorized CUDA projection and reuses a fixed device buffer for streamed BF16 and NVFP4 experts. In this mode, other supported GGUF quantization types project on the host. Set `FIFI_INKLING_CUDA_RESIDENT_EXPERT_MIB` to cap VRAM for compatible NVFP4 expert banks. The shared expert tier keeps complete banks for repeated core layers first and stops at its VRAM headroom. The remaining experts use the file mapping and bounded staging. Set `FIFI_INKLING_CUDA_STREAM_EXPERTS=1` to stage cold BF16 and NVFP4 expert pairs through FiFi's shared pinned CUDA banks. The GPU decodes the original GGUF NVFP4 blocks directly. Both banks are allocated before hot expert placement.

Set `FIFI_INKLING_CUDA_PACKED_GGUF=1` to use CUDA projections with the original GGUF blocks in all ten Unsloth text variants, from UD-IQ1_S and UD-IQ1_M through UD-Q8_K_XL. FiFi selects the projection by each tensor's GGUF type and uses actual packed sizes to place complete expert layers in VRAM. This option enables pinned staging for the remaining experts. Add the RAM and adaptive-cache settings below to warm mapped expert pages and retain frequently selected experts in VRAM.

Set `FIFI_INKLING_CUDA_BATCH_EXPERTS=1` with packed GGUF projection to batch each resident layer's selected experts. The gate/up projection, clipped activation, down projection, and weighted sum run on one CUDA stream. Each resident layer visit reads back one combined activation row.

Set `FIFI_INKLING_CUDA_DIRECT_RESIDENT=1` to keep original GGUF NVFP4 blocks in VRAM and use the direct CUDA kernel for those expert banks.

For VRAM → RAM → SSD expert placement, set `FIFI_INKLING_CUDA_RAM_EXPERT_MIB=auto` or give a positive MiB budget. This enables shared pinned staging and automatic VRAM placement when no explicit VRAM budget is set. After GPU placement, FiFi warms file-backed expert layers within the process's host-memory allowance. Under a smaller allowance, FiFi uses the Qwen expert predictor before attention and asks Linux to read the predicted experts' two GGUF ranges into the page cache. The actual Inkling router determines which experts execute. The budget limits startup warming and advice per visit. Linux can reclaim or retain file pages as memory pressure changes. Set the RAM budget to zero to use ordinary demand paging.

Set `FIFI_INKLING_CUDA_CACHE_EXPERTS=1` to retain frequently routed expert pairs in the remaining VRAM allowance. The shared adaptive policy ranks observed routes per layer. A promotion copies an expert directly from its existing CUDA staging bank. Later hits execute both projections from the cached original bytes. The cache respects the total expert budget and the shared device headroom. With `FIFI_INKLING_CUDA_PROFILE=1`, the server reports hits and avoided host-to-device bytes by layer.

With packed GGUF and adaptive caching enabled, FiFi reserves three quarters of the admitted expert budget for adaptive slots when the full expert weights exceed the budget. The shared device headroom still applies. Set `FIFI_INKLING_CUDA_EXPERT_CACHE_RESERVE_MIB` to choose a fixed quota, or set it to zero to place whole layers first.

Set `"prefix_reuse": true` in the server configuration to reuse saved chat prefixes. Inkling uses the shared scheduler and prefix store with one pinned root and two evictable entries. Each snapshot retains the independent KV and short-convolution state of every visit. KV rows are shared with snapshots while they are in use.

Set `FIFI_INKLING_CUDA_PREFILL_ROWS` from 2 to 16 to group prompt tokens on the GGUF CUDA backend with shared expert streaming. Each attention and convolution state advances in token order. The CUDA staging banks upload each missing expert once per layer visit and prompt chunk. `FIFI_INKLING_CUDA_PROFILE=1` records expert uploads in the shared request log. A value of 1 uses single-token prefill and is the default.

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
| Mixed NVFP4 and MXFP8 safetensors | Inkling 2 Mini |

## Context limits

| Model and backend | Default | Largest |
| --- | ---: | ---: |
| Qwen3.6, Apple Silicon | 262,144 | 262,144 tested on a 36 GB Mac |
| Qwen3.6, CUDA | 32,768 | 262,144 tested on a 24 GB RTX 3090 |
| Flint, Apple Silicon | 32,768 | 32,768 |
| Flint, CUDA | 131,072 | 262,144 |
| Nemotron 3.5 Super | set in config | |
