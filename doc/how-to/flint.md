# Run Flint

Flint is a mixture-of-experts model with 39.8B parameters, 2.5B of them active per token.
fifi serves it on Apple Silicon. On NVIDIA GPUs it runs as a pilot: short contexts work, and long-context quality is not measured.

If you have not built fifi before, do the [tutorial](../getting-started.md) first.
Run Flint in a server process of its own: it sets the MLX allocator cache limit to zero for the whole process.

## On a Mac

You need Apple Silicon, Xcode with its command-line tools, CMake, and network access during the first build.
The checkpoint is 22.6 GB.

1. Download the checkpoint:

   ```sh
   hf download RecursiveSI/flint-int4 --revision 3b93f9053b530488405ae9fca51d4900d6ce21f0 \
     --local-dir ~/models/flint-int4
   ```

2. Build with the MLX backend:

   ```sh
   cd fifi/engine
   cargo build --release --bin enginesrv --features native-mlx
   ```

3. Create `flint.json`. Write the paths out in full:

   ```json
   {
     "listen": "127.0.0.1:18791",
     "api_key_env": "ENGINESRV_API_KEY",
     "engines": [
       {
         "id": "flint",
         "family": "flint",
         "model_dir": "/Users/you/models/flint-int4",
         "tokenizer_dir": "/Users/you/models/flint-int4"
       }
     ]
   }
   ```

   Do not add a `flint` object on the Mac; startup refuses one.

4. Start the server and wait for `/ready`:

   ```sh
   export ENGINESRV_API_KEY=my-secret-key
   target/release/enginesrv --config flint.json
   ```

Send requests with `"model": "flint"`.
The context limit is 32,768 tokens. To use less memory, set `FIFI_FLINT_MLX_CONTEXT_LIMIT=8192` before starting.

## On an NVIDIA GPU (pilot)

You need Linux with the CUDA 12 toolkit.
The weights stay on the GPU; the pack is 20.7 GB.

1. Download the EXL3 pack. It includes the tokenizer files:

   ```sh
   hf download 0xBakeer/TandemLLM-Flint-EXL3-4.0 --revision e7ab11242b0b30a6063f3559dda75fed6b9e12c6 \
     --local-dir ~/models/flint-exl3
   ```

2. Build with the CUDA backend:

   ```sh
   cd fifi/engine
   cargo build --release --bin enginesrv --features native-cuda
   ```

3. Create `flint-cuda.json`. On CUDA the entry needs a `flint` object with the context limit and the KV cache format:

   ```json
   {
     "listen": "127.0.0.1:18791",
     "api_key_env": "ENGINESRV_API_KEY",
     "engines": [
       {
         "id": "flint",
         "family": "flint",
         "model_dir": "/home/you/models/flint-exl3",
         "tokenizer_dir": "/home/you/models/flint-exl3",
         "flint": {"context_limit": 4096, "kv_profile": "f16"}
       }
     ]
   }
   ```

4. Start the server as above and wait for `/ready`.

Startup checks free GPU memory and refuses a load that does not fit; lower `context_limit` if it is refused.
Prefix reuse is experimental on CUDA: set `FIFI_FLINT_PREFIX_REUSE=1` and `"prefix_reuse": true`.
It needs `"kv_profile": "f16"`; with `q8` the engine logs a note and serves without reuse.
Greedy decoding and top-k sampling with k up to 20 run on the GPU; other sampling settings run on the host and are slower.

If the model produces a non-finite value, that request fails with `500 backend_error` and the engine keeps serving.
A CUDA runtime error stops the engine; restart the server.

## See also

- [`flint` options](../reference/configuration.md#flint) and [Flint environment variables](../reference/environment.md#flint)
- [How fifi plans memory](../explanation/memory.md)
