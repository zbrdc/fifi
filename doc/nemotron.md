# Nemotron 3.5 Super

The `nemotron35` family runs Nemotron 3.5 Super from its NVFP4 safetensors checkpoint.
The tested setup is the CUDA hybrid (`NEM35_CPU_EXPERTS=1`): the routed experts of most MoE layers run on the CPU from memory-mapped weight files, and the rest of the model runs on the GPU.

## Config

```json
{
  "id": "nemotron",
  "family": "nemotron35",
  "model_dir": "/models/nemotron-3.5-super-nvfp4",
  "tokenizer_dir": "/models/nemotron-3.5-super-nvfp4",
  "nemotron": {
    "backend": "cuda",
    "context_limit": 8192,
    "cache_entries": 64,
    "prefill_chunk": 256
  }
}
```

All four `nemotron` keys are required. The values above are an example; no tuned profile ships for this family.

| Key | Values | Meaning |
| --- | --- | --- |
| `backend` | `"cuda"`, `"cpu_reference"` | `cuda` needs a Linux build with `--features native-cuda`. `cpu_reference` runs everything on the CPU and is slow; it exists for checking results. |
| `context_limit` | 1 or more | Context limit in tokens. |
| `cache_entries` | 1 or more | Number of expert weight tensors kept decoded in host memory. |
| `prefill_chunk` | 1 to `context_limit` | Prompt tokens per prefill call. |

Startup checks the tokenizer against the checkpoint and refuses the load if the host does not have enough free memory.

## Environment variables

| Variable | Default | Accepted values | Effect |
| --- | --- | --- | --- |
| `NEM35_CPU_EXPERTS` | `0` | `0`, `1` | `1` runs routed experts on the CPU (the hybrid). |
| `NEM35_GPU_MOE_LAYERS` | `auto` | `auto` or a number | MoE layers whose experts stay on the GPU in the hybrid. |
| `NEM35_CPU_THREADS` | `12` | 1 or more | CPU threads for the hybrid experts. |
| `NEM35_CUDA_CHUNK_PREFILL` | off | a row count | Hybrid only: prefill this many rows at a time instead of one token at a time. `0` turns it off. |

## Performance

On an RTX 3090 with the hybrid, prefill takes 0.55 to 0.6 s per token and decode 0.96 s per token.
llama.cpp decodes the same model at 0.056 s per token on that machine.
Greedy output matches the reference implementation token for token.
