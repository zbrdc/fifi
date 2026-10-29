# Example configs

Ready-made `enginesrv` configuration files for Qwen3.6-35B-A3B.
Set `model_dir` and `tokenizer_dir` before use. All files read the API key from `ENGINESRV_API_KEY`.

| File | Hardware | Context | KV cache | Port |
| --- | --- | ---: | --- | ---: |
| `qwen36-mac-studio-36gb.json` | Mac with Apple Silicon, 36 GB | 262,144 | 8-bit | 18791 |
| `qwen36-cuda-3090-kv8-prefix.json` | RTX 3090, 24 GB | 180,224 | 8-bit | 18791 |
| `qwen36-cuda-3090-kv8-prefix-131k.json` | RTX 3090, 24 GB | 131,072 | 8-bit | 18791 |
| `qwen36-cuda-3090-kv8-prefix-262k.json` | RTX 3090, 24 GB | 262,144 | 8-bit | 18791 |
| `qwen36-rtx3090-24gb-fp16kv.json` | RTX 3090, 24 GB | 98,304 | 16-bit | 18795 |

- How to choose and run one: [Run Qwen3.6](../../doc/how-to/qwen36.md)
- Every key: [Configuration reference](../../doc/reference/configuration.md)
