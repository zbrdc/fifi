# Run Inkling 2 Mini on CPU or CUDA

The `inkling2_loop` family runs Inkling 2 Mini on one CUDA device or on the CPU.
It serves raw completions and plain-text chat with one active sequence.
For Apple Silicon, see [Run Inkling 2 Mini on a Mac](inkling.md).

## Start

1. Download the pinned checkpoint (`thinkingmachines/Inkling-2-Mini`, revision `ed40e7da625f594fc9c4990f5f66be24873c22d7`) or a published fixed-loop text GGUF.
2. Point `model_dir` to the safetensors directory, a GGUF file, or a directory with one text GGUF. Point `tokenizer_dir` to the pinned release tokenizer files.
3. Configure an engine with `"family": "inkling2_loop"`, or run `enginesrv serve`:

```sh
export ENGINESRV_API_KEY=my-secret-key
target/release/enginesrv serve --model ~/models/inkling2-mini --family inkling2_loop
```

With no backend variable set, a 64-bit Linux build with `native-cuda` runs on the first visible CUDA device.
Any other 64-bit build runs on the CPU.
Set `CUDA_VISIBLE_DEVICES` to pick a card.

For `enginesrv check` on a GGUF directory, pass `--tokenizer-dir` separately.

## Common options

| Variable | Default | Meaning |
| --- | --- | --- |
| `FIFI_INKLING_CONTEXT_TOKENS` | `8192` | Context limit in tokens (maximum 8,192). |
| `FIFI_INKLING_CPU_REFERENCE` | off | `1` forces CPU execution. Setting both CPU and CUDA flags fails startup. |
| `FIFI_INKLING_CUDA_REFERENCE` | off | `1` forces CUDA execution. |
| `FIFI_INKLING_CUDA_RESIDENT_EXPERT_MIB` | unlimited | Caps VRAM budget for resident packed expert banks (`0` streams all experts). |
| `FIFI_INKLING_CUDA_RAM_EXPERT_MIB` | unset | `auto` or positive MiB warms mapped expert pages in host RAM. `0` uses demand paging. |
| `FIFI_INKLING_CUDA_PREFILL_ROWS` | `512` | Prompt tokens grouped per prefill request (`1` uses single-token prefill). |
| `FIFI_INKLING_DSPARK_GGUF` | unset | Path to the Inkling 2 Mini DSpark sidecar GGUF for speculative decoding. |

Set `"prefix_reuse": true` in the server configuration to reuse saved chat prefixes.

For internal CUDA kernel, DSpark, and profiling flags, see [Internal and experimental environment variables](../reference/internal-environment.md#inkling-2-mini-diagnostics-and-kernel-flags).
