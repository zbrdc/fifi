# Run Flint

Flint is a mixture-of-experts model with 39.8B parameters, 2.5B of them active per token.
fifi serves it on Apple Silicon only.

If you have not built fifi before, follow the [tutorial](../getting-started.md) first.
Run Flint in a server process of its own: it sets the MLX allocator cache limit to 64 MiB for the whole process (rollback: `FIFI_FLINT_MLX_CACHE_LIMIT=0`).

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

4. Start the server and wait for `/ready`:

   ```sh
   export ENGINESRV_API_KEY=my-secret-key
   target/release/enginesrv --config flint.json
   ```

Send requests with `"model": "flint"`.
The context limit is 32,768 tokens. To use less memory, set `FIFI_FLINT_MLX_CONTEXT_LIMIT=8192` before starting.

## On an NVIDIA GPU

Flint does not serve on CUDA in this build. A `native-cuda` build refuses a `flint` entry at startup with "Flint requires --features native-mlx on Apple Silicon".

## See also

- [Flint environment variables](../reference/environment.md#other-families)
- [How fifi plans memory](../explanation/memory.md)
