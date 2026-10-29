# Run Qwen3.6

This guide runs `Qwen3.6-35B-A3B` on a Mac with Apple Silicon or on a PC with an NVIDIA RTX 3090.
If you have not built fifi before, do the [tutorial](../getting-started.md) first.

## On a Mac

You need Apple Silicon with at least 36 GB of memory, and:

- Xcode (from the App Store), then `xcode-select --install` for the command-line tools.
- CMake: `brew install cmake`.
- Network access during the first build, which downloads the MLX sources.
- The Hugging Face CLI: `pip install -U huggingface_hub`.

1. Download the model. It is 20.4 GB.

   ```sh
   hf download mlx-community/Qwen3.6-35B-A3B-4bit --local-dir ~/models/qwen3.6-mlx-4bit
   ```

2. Build with the MLX backend:

   ```sh
   cd fifi/engine
   cargo build --release --bin enginesrv --features native-mlx
   ```

3. Open `config/qwen36-mac-studio-36gb.json` and set `model_dir` and `tokenizer_dir` to the download directory.

4. Start the server:

   ```sh
   export ENGINESRV_API_KEY=my-secret-key
   target/release/enginesrv --config config/qwen36-mac-studio-36gb.json
   ```

5. Wait until `curl http://127.0.0.1:18791/ready` returns `{"status":"ready"}`.

The model name for requests is `fifi-q36`, set by the `model` key of the config.

## On an RTX 3090

You need Linux with an NVIDIA driver, and:

- The CUDA 12 toolkit (`nvcc --version` must report release 12.x).
- The Hugging Face CLI: `pip install -U huggingface_hub`.

1. Download the model file. fifi accepts only this exact file (20,419,565,568 bytes):

   ```sh
   hf download ggml-org/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-Q4_K_M.gguf --local-dir ~/models/qwen3.6
   ```

2. Download the tokenizer files. fifi checks them against fixed hashes; these copies match:

   ```sh
   hf download mlx-community/Qwen3.6-35B-A3B-4bit \
     tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json \
     --local-dir ~/models/qwen3.6
   ```

3. Build with the CUDA backend:

   ```sh
   cd fifi/engine
   cargo build --release --bin enginesrv --features native-cuda
   ```

   The build targets RTX 30-series GPUs (`sm_86`). For another GPU, set `INFERENCE_CUDA_ARCHS`, for example `INFERENCE_CUDA_ARCHS=89`.

4. Choose a profile from `config/`:

   | Profile | Context | Use it when |
   | --- | ---: | --- |
   | `qwen36-cuda-3090-kv8-prefix.json` | 180,224 | Default. Other programs use up to 1.5 GiB of GPU memory. |
   | `qwen36-cuda-3090-kv8-prefix-131k.json` | 131,072 | You want the fastest first reply on long prompts. |
   | `qwen36-cuda-3090-kv8-prefix-262k.json` | 262,144 | You need the longest context and other programs use under 744 MiB of GPU memory. |
   | `qwen36-rtx3090-24gb-fp16kv.json` | 98,304 | You want a 16-bit KV cache. |

   Check current GPU use with `nvidia-smi`. A desktop session alone often uses several hundred MiB.

5. Set `model_dir` and `tokenizer_dir` in the profile to `~/models/qwen3.6` (written out in full), then start the server:

   ```sh
   export ENGINESRV_API_KEY=my-secret-key
   target/release/enginesrv --config config/qwen36-cuda-3090-kv8-prefix.json
   ```

6. Wait for `curl http://127.0.0.1:18791/ready` to return `{"status":"ready"}`. The 16-bit profile listens on port 18795 instead.

To check the model file's SHA-256 while it loads, set `Q36_VERIFY_GGUF_SHA256=1` before starting.

A server process runs one Qwen3.6 CUDA engine.

## Keep long conversations fast

All shipped CUDA profiles set `"prefix_reuse": true`: when a conversation continues, only the new messages are processed.
The Mac profile leaves it off. To turn it on there, set `"prefix_reuse": true` at the top level and raise `session_cap_bytes` in the `qwen36` object to at least `5770117120`; startup refuses a smaller cap.
[How conversation reuse works](../explanation/prefix-reuse.md) explains the memory this takes.

## If the engine does not load

The server starts listening before the model loads.
If loading fails, `/ready` returns `503` with each engine's state, and the server log says why.

| Log message mentions | Cause | Fix |
| --- | --- | --- |
| the GGUF file name or size | `model_dir` does not hold the exact Q4_K_M file | Download it again with the command above. |
| a tokenizer file or its hash | The tokenizer files are missing or differ | Download them from `mlx-community/Qwen3.6-35B-A3B-4bit`. |
| free GPU memory | The profile does not fit next to other GPU use | Close other GPU programs or pick a smaller profile. |
| `session_cap_bytes` | Prefix reuse needs a larger session cap on the Mac | Raise it as described above. |

To change prefill or caching behaviour, see the [environment variables](../reference/environment.md#qwen36-on-nvidia-cuda).
