# Run Nemotron 3.5 Super

Nemotron 3.5 Super is a 120B-parameter mixture-of-experts model with 12B active parameters.
fifi runs its NVFP4 checkpoint on Linux with an NVIDIA GPU.
The GPU computes the matrix products and the host runs the rest of each forward pass, so generation is slow.

If you have not built fifi before, follow the [tutorial](../getting-started.md) first.

## Steps

You need Linux with the CUDA 12 toolkit and about 76 GB of free disk space.

1. Download the checkpoint at the tested revision:

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

   The three `nemotron` keys besides `cache_entries` are required. The values are an example, not a tuned profile.
   `cache_entries` sets how many expert weight tensors stay decoded in host memory. Admission charges each entry as the largest entry's payload bytes plus the loader's per-entry index (96 bytes on 64-bit targets). If you omit the key, the loader fits the count from 850 permille of the measured host budget minus fixed serving charges. On Linux, this budget is cgroup-aware. A remainder below one entry refuses the load instead of starting with an empty cache. One log line states the chosen count. The loader uses an explicit number verbatim, and a small card refuses the load. Omit the key when you want the fit to trim the 12 GiB device request to what the card holds.

4. Start the server and wait for `/ready`:

   ```sh
   export ENGINESRV_API_KEY=my-secret-key
   target/release/enginesrv --config nemotron.json
   ```

   Startup checks host and GPU memory and refuses the load if either is short. The log names the part that does not fit. On CUDA, the optional device weight cache requests 12 GiB. When the loader fits the cache count, admission trims that request to what the card holds free.

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
It is slower and exists to check CUDA results. It needs no CUDA build and accepts any sampling settings.

On Apple Silicon, `FIFI_NEM35_CPU_NEON=1` selects the Arm NEON NVFP4 dot product in the mapped CPU expert pool instead of the scalar default.

## See also

- [`nemotron` options](../reference/configuration.md#nemotron)
