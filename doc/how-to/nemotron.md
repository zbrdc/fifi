# Run Nemotron 3.5 Super

Nemotron 3.5 Super is a 120B-parameter mixture-of-experts model with 12B active parameters.
fifi runs its NVFP4 checkpoint on Linux with an NVIDIA GPU.
The GPU computes the matrix products and the host runs the rest of each forward pass, so generation is slow.

If you have not built fifi before, do the [tutorial](../getting-started.md) first.

## Steps

You need Linux with the CUDA 12 toolkit and about 76 GB of free disk space.

1. Download the checkpoint at the revision fifi is tested with:

   ```sh
   hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 \
     --revision 8a115b9d966adce8c059a7bcf47f49a1e57957d8 \
     --local-dir ~/models/nemotron-3.5-super
   ```

2. Build with the CUDA backend:

   ```sh
   cd fifi/engine
   cargo build --release --bin enginesrv --features native-cuda
   ```

3. Create `nemotron.json`. Write the paths out in full:

   ```json
   {
     "listen": "127.0.0.1:18791",
     "api_key_env": "ENGINESRV_API_KEY",
     "engines": [
       {
         "id": "nemotron",
         "family": "nemotron35",
         "model_dir": "/home/you/models/nemotron-3.5-super",
         "tokenizer_dir": "/home/you/models/nemotron-3.5-super",
         "nemotron": {
           "backend": "cuda",
           "context_limit": 8192,
           "cache_entries": 64,
           "prefill_chunk": 256
         }
       }
     ]
   }
   ```

   All four `nemotron` keys are required. The values are an example, not a tuned profile.
   `cache_entries` is the number of expert weight tensors kept decoded in host memory.

4. Start the server and wait for `/ready`:

   ```sh
   export ENGINESRV_API_KEY=my-secret-key
   target/release/enginesrv --config nemotron.json
   ```

   Startup checks host and GPU memory and refuses the load if either is short; the log names the part that does not fit.

5. Send requests with `"temperature": 0`:

   ```sh
   curl http://127.0.0.1:18791/v1/chat/completions \
     -H "Authorization: Bearer my-secret-key" \
     -H "Content-Type: application/json" \
     -d '{"model": "nemotron35", "temperature": 0, "messages": [{"role": "user", "content": "Hello!"}]}'
   ```

   On CUDA the engine accepts greedy decoding only. The checkpoint's default temperature is 1.0, so a request without `"temperature": 0` fails.

## Check results on the CPU

`"backend": "cpu_reference"` runs the whole model on the CPU.
It is much slower and exists to check the CUDA results. It needs no CUDA build and accepts any sampling settings.

## See also

- [`nemotron` options](../reference/configuration.md#nemotron)
