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
It is much slower and exists to check the CUDA results. It needs no CUDA build and accepts any sampling settings.

### Profile mapped experts on Apple Silicon

The CPU profiler calls the same NVFP4 pool that serves mapped routed experts in the hybrid engine. It can open one layer with the pinned `config.json`, safetensors index and the one shard that contains that layer's routed experts. The layer 1 study uses `model-00002-of-00088.safetensors` from the NVIDIA revision above. Its 512 experts map directly from the 1.63 GB shard. Each call routes 22 experts with 1,024 latent and 2,688 intermediate values.

The 16 GB M4 mini has four performance cores and six efficiency cores. The pool selected the portable scalar kernel. After a sequential pass over the shard, two sweeps used the same deterministic calls in opposite thread-count order, with 31 timed calls per arm. All 12 warmed arms recorded zero process-attributed disk reads and page-ins.

| Threads | Median of per-run p50 | Median of per-run p90 |
| ---: | ---: | ---: |
| 1 | 59.62 ms | 70.95 ms |
| 2 | 35.99 ms | 45.46 ms |
| 4 | 22.87 ms | 34.75 ms |
| 6 | 24.71 ms | 35.64 ms |
| 8 | 27.46 ms | 37.26 ms |
| 10 | 25.93 ms | 38.79 ms |

The fresh six-thread trial took 169.78 ms for its first call and 127.59 ms at p50 over 11 subsequent calls. The trial attributed 710.5 MB of SSD reads and 45,090 page-ins to its process. Four threads gave the lowest warm median in this single-layer scalar study. These measurements cover one routed layer and the tested mix of selected experts.

Build the ignored test on Apple Silicon, then run its separate processes with the driver below. The test prints the runner path after compilation.

```sh
cd engine
cargo test --locked --release --test nemotron35_cpu_expert_profile --no-run
cd ..
python3 scripts/mac_cache_study/nemotron35_cpu_profile.py \
  "$NEM35_PROFILE_RUNNER" "$NEM35_LAYER1_DIR" "$SCRATCH/proc-io" "$SCRATCH/nemotron35-profile"
```

### Compare the NEON NVFP4 kernel

`FIFI_NEM35_CPU_NEON=1` selects a vector dot product for the mapped CPU expert pool on Apple Silicon. It decodes the E2M1 nibbles with a NEON table lookup and accumulates F32 products in four lanes. The existing scalar path remains the default. The numerical suite checks short and production-width matrices, multi-row batches, refusal cases, and reproducibility across thread counts.

The paired run on the same layer 1 shard alternated scalar and NEON with each thread count. Each of the 12 arms ran 31 timed calls. We paused the Flash-Next transfer for this measurement. The table compares paired medians within this run.

| Threads | Scalar p50 | NEON p50 | Scalar p90 | NEON p90 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 36.19 ms | 10.87 ms | 37.35 ms | 11.82 ms |
| 4 | 10.47 ms | 3.56 ms | 11.10 ms | 3.89 ms |
| 6 | 8.60 ms | 3.10 ms | 8.89 ms | 3.45 ms |

At four threads the paired median was 2.94 times faster. All four- and six-thread arms recorded zero SSD reads and zero page-ins. The first one-thread scalar process attributed 28.2 MB of reads across setup and its calls, and the remaining one-thread processes recorded zero. This step measures the CPU expert layer. Full-model quality and a default selection remain open.

Run the profiler in a fresh process with `FIFI_NEM35_CPU_NEON=1` to measure the NEON kernel.

## See also

- [`nemotron` options](../reference/configuration.md#nemotron)
