# Run Inkling 2 Mini on Apple Silicon

The experimental native text worker reads the mixed NVFP4/MXFP8 safetensors release or text GGUF files (`inkling2_loop` and older `inkling2` layouts).

## Start with safetensors

1. Download the checkpoint and tokenizer:

```sh
hf download thinkingmachines/Inkling-2-Mini \
  --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 \
  --include config.json model.safetensors.index.json model-00001-of-00001.safetensors \
    tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json \
  --local-dir ~/models/inkling2-mini
```

2. Build with `native-mlx` and start the server from `engine/`:

```sh
export ENGINESRV_API_KEY=my-secret-key
cargo build --release --locked --features native-mlx --bin enginesrv
target/release/enginesrv serve --model ~/models/inkling2-mini \
  --context 128 --listen 127.0.0.1:18795
```

## Start with GGUF

Pass a text GGUF file (including `.gguf_file` suffixes) and the pinned tokenizer directory:

```sh
target/release/enginesrv serve --model /path/to/Inkling-2-Mini-IQ2_XXS.gguf \
  --tokenizer-dir ~/models/inkling2-mini --context 192 --listen 127.0.0.1:18795
```

`enginesrv check` inspects the vision projector (`clip`) and DSpark drafter (`dflash`) metadata, though neither runs on this path.

## Memory and behavior

- The worker serves one active sequence at a time and runs routed experts on Metal.
- With Metal's working set at or below 16 GiB, Inkling defaults to a 5 GiB working set for safetensors and 6 GiB for GGUF.
- Set `FIFI_MLX_WORKING_SET_MIB` to override the ceiling, or `FIFI_MLX_AUTO_BUDGET=1` to use Metal's full recommendation.
- `FIFI_INKLING2_MINI_MLX=0` disables the native family. `FIFI_INKLING2_MINI_GGUF_MLX=0` disables GGUF serving.
- Disabling thinking sets reasoning effort to `0.0`. Inkling can still emit reasoning before the answer, so small token caps may stop inside reasoning.

## Check serving

Send a greedy completion request to check output:

```sh
curl --fail-with-body --silent --show-error http://127.0.0.1:18795/v1/completions \
  -H "Authorization: Bearer ${ENGINESRV_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling","prompt":"The capital of France is","max_tokens":3,"temperature":0}'
```

For full serving checks and benchmarks, run `engine/tests/tools/check_inkling_serving.py` and `engine/tests/tools/bench_inkling.py`.
For internal worker and GGUF tuning flags, see [Internal and experimental environment variables](../reference/internal-environment.md#inkling-2-mini-diagnostics-and-kernel-flags).
