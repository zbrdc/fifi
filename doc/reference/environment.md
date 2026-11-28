# Environment variables

Set serving variables before starting `enginesrv`, unless an entry says otherwise. The server reads some per call or at engine or session creation.
The API key variable is not here. Its name comes from `api_key_env` in the [configuration file](configuration.md).

Two kinds of values appear below:

- An unset **switch** keeps its default. A set switch is on only for `1`, `true`, or `on` (ignoring spaces and case). Any other value turns it off.
- A variable with **accepted values** refuses startup when set to anything else, unless the table says otherwise.

## Build

| Variable | Default | Meaning |
| --- | --- | --- |
| `INFERENCE_CUDA_ARCHS` | `75,86,89` (`72` on aarch64) | GPU architecture targets of CUDA kernels for `cargo build --features native-cuda`. The default covers RTX 20, 30, and 40 series. A single target such as `86` builds faster and smaller. The build includes PTX for the highest target, which newer drivers can compile for newer GPUs (PTX never runs on older architectures). Native `120` for RTX 50 series needs CUDA 12.8 or newer. Building a target does not qualify it. See [CUDA FP16 prefill attention](configuration.md#cuda-fp16-prefill-attention). |
| `FIFI_CUDA_LINEINFO` | unset (off) | Adds `-lineinfo` to `nvcc` flags during `cargo build --features native-cuda`, embedding source line info without changing optimization. Set it when inspecting kernels with `compute-sanitizer` or an NVIDIA profiler so reports name source lines. Leave it unset for standard builds. Read at build time (`0` stays off). Rebuild to change. |
| `FIFI_CUDA_ATTN_PREFILL_TURING` | off | Set to `1` so prefill attention uses the typed half-precision Turing pilot arm. Any other value keeps the standard path. This pilot targets sm75 devices and is not device-qualified. Read once per process. |

NVIDIA documents the [GPU compute capabilities](https://developer.nvidia.com/cuda/gpus), [PTX compatibility rules](https://docs.nvidia.com/cuda/archive/12.8.1/blackwell-compatibility-guide/index.html), and [CUDA 12.8 compiler support for SM 120](https://docs.nvidia.com/cuda/archive/12.8.2/cuda-features-archive/index.html).

## Server

| Variable | Default | Meaning |
| --- | --- | --- |
| `FIFI_CPU_GREEDY_BLOCKS` | off | Set to `1` to scan host logits in 256-value blocks for greedy sampling. Other values use scalar scan. Both pick the first maximum and reject non-finite logits on host rows of at least 256 values. Restart to change. Our hosts have not verified the contributor's 0.20 to 0.03 ms figure. |
| `FIFI_CPU_Q8G_F16C` | off | Set to `1` so the AVX2 CPU expert routine runs its inner dot product in half precision via F16C when supported. Any other value keeps the reference path. Read once per run. |
| `ENGINESRV_WARM_TURNS` | on (switch) | With `prefix_reuse`, remembers the last conversation's tokens so the next turn only tokenizes new messages. A request from another conversation replaces the memory. |
| `ENGINESRV_SEED_RESENDS` | on (switch) | With warm turns on, keeps token IDs of a freshly rendered prompt so an identical retry skips rendering and tokenization. Measured on Qwen3.6 (256,855 tokens): 1.0 to 1.2 ms with seed vs 270 to 310 ms without. Startup logs `seeded retries:`. Restart to change. |
| `ENGINESRV_RESEND_EXTRA_BYTES` | `0` | With seeded resends on, keeps one displaced conversation's token result if its allocation fits this per-engine byte budget. An exact retry reuses it until the next reseed. |
| `ENGINESRV_PREFIX_BLOCK_MATCH` | on (switch) | With `prefix_reuse`, saves one block checkpoint at a multiple of `ENGINESRV_PREFIX_BLOCK_TOKENS` just before the conversation end. Prompts changing near the end still reuse most tokens. Used by engines with at least two saved prefixes. |
| `ENGINESRV_PREFIX_BLOCK_TOKENS` | 4,096 | Spacing of the cut points, in tokens. Values below 1 select the default. |
| `ENGINESRV_CACHE_USAGE` | on | Report reused prompt tokens in the API usage fields. Once set, only `1`, `true`, and `on` keep it on. |
| `ENGINESRV_THINK_REPLAY` | on (switch) | Qwen3.6 only: keep reasoning that clients resend with earlier assistant turns in the prompt. Off removes it for every engine. |
| `ENGINESRV_NOTICE_TOOLWRAP` | on (switch) | Qwen3.6 only, with think replay on: render system messages after an assistant turn as tool responses so the chat template keeps earlier tool-calling reasoning. |
| `ENGINESRV_SHADOW_BLOCK_LCP` | off (switch) | Read-only observation for the request log: report the longest common prefix the saved block checkpoint would match. Never changes restore or save behavior. |

## Model stores and the hub

Read by `enginesrv serve` when `--model` is a name, and by the model page of the web console. See [Serve a model without a config file](../how-to/serve-a-model.md).

| Variable | Default | Meaning |
| --- | --- | --- |
| `FIFI_MODELS_DIR` | `~/models` | The models directory, searched first for a model named without a path. |
| `OLLAMA_MODELS` | `~/.ollama/models` | The Ollama store, searched second. |
| `FIFI_HF_HUB` | unset | Location of the Hugging Face hub cache. Wins over `HF_HUB_CACHE`, which wins over `HF_HOME/hub`. |
| `FIFI_HF_ENDPOINT` | unset | Hub address, for a mirror. Wins over `HF_ENDPOINT`. |
| `HF_TOKEN` | unset | Access token for gated or private repositories. Reads `HUGGING_FACE_HUB_TOKEN` when unset. |

## Apple Silicon device sizing

| Variable | Default | Meaning |
| --- | --- | --- |
| `FIFI_MLX_AUTO_BUDGET` | off, on for Inkling | `1` sizes Qwen3.6, Qwen3.8 dense, Flash-Next, or Flint prefill from Metal's recommended working set at startup. It applies to one MLX model per server and respects chunk ceilings. Flint and Qwen3.6 stream routed experts when resident weights exceed the budget. Inkling uses this sizing by default (`0` opts out unless a ceiling is set). Without it, oversized resident models size under the host budget. |
| `FIFI_MLX_WORKING_SET_MIB` | unset | Positive integer ceiling for an MLX memory budget, in MiB. The budget is the lowest of host budget (unified memory minus a sixth, at least 4 GiB), Metal's recommended working set, and this ceiling. Every MLX family plans and loads against that number. Setting it turns on automatic sizing for one MLX model per server. |
| `FIFI_INKLING2_MINI_MLX` | off | Set to `1` for experimental [Inkling 2 Mini text serving](../how-to/inkling.md) on Apple Silicon. |
| `FIFI_MLX_REUSE_IDENTITY_RESHAPE` | off | Set to `1` to share a native resident tensor when the requested reshape matches its current dimensions. |
| `FIFI_FLASH_NEXT_MLX_SPARSE_PREFILL` | off | Set to `1` to batch sparse Flash-Next MLX prompt rows when chunked prefill has leftover workspace. Each batch takes up to eight rows, shrinks as the prefix grows, and steps one token when under two rows fit. |
| `FIFI_FLASH_NEXT_MLX_CAPACITY_KV` | off | Set to `1` to append QSA keys, values and index keys to geometrically grown Metal buffers. The session budget charges spare rows. Attention and sparse indexing see only the filled prefix. |
| `FIFI_FLASH_NEXT_MLX_STREAM_TIMING` | off | Set to `1` in an Apple Silicon qualification build to log logical expert-read, page-import, and native MLX wait counters per Flash-Next forward. |
| `FIFI_Q36_MLX_DISK_EXPERTS` | automatic | Where Qwen3.6 keeps routed experts on Apple Silicon. Unset streams from local shards (keeping one layer's pages in memory) when resident weights exceed the budget, else keeps everything resident. `1` forces streaming. `0` forces residency and refuses if it does not fit. Other values log a warning. Streamed mode uses one session. |
| `FIFI_Q36_MLX_DISK_DENSE_FFN` | off | Set to `1` with MLX device sizing enabled to stage Qwen3.8 dense gate, up, and down matrices one layer at a time. Decode and prefill charge the largest layer stage beside resident weights and session memory. |
| `FIFI_Q36_MLX_STREAM_TIMING` | off | Set to `1` in an Apple Silicon qualification build to log one timing record per streamed forward. Qwen3.6 MoE reports host routing, logical reads, page staging, and MLX wait times. Qwen3.8 dense reports logical FFN bytes and slice reads. |
| `FIFI_Q36_MLX_EXPERT_TRACE` | off | Set to `1` in an Apple Silicon qualification build to log sorted checkpoint expert IDs and logical reads for each staged layer of a Qwen3.6 MoE forward. |
| `FIFI_Q36_MLX_FUSE_FINAL_PREFILL` | off | Set to `1` to include the pending prompt token in the last Qwen3.8 dense prefill batch and reuse its logits at step one. Applies to single-sequence disk-staged FFNs when the enlarged batch fits in one forward. |
| `FIFI_MLX_EXPERT_READ_RUNS` | automatic | Group adjacent expert slice reads on multi-row disk-backed paths in Flint, Flash-Next, and Qwen3.6 MoE. Automatic turns on per streaming engine without affecting other engines. `prefill` turns it on, and `0` or other values turn it off. Flint's legacy switches take precedence. |
| `FIFI_MLX_OVERLAP_EXPERT_READS` | automatic | Submit the shared expert to Metal before the CPU reads that layer's routed weights on disk-backed paths in Flint, Flash-Next, and Qwen3.6 MoE. Automatic turns on per streaming engine. `1` turns it on, `0` or other values turn it off. Flint's legacy switch takes precedence. |
| `FIFI_MLX_POSITIONED_READS` | automatic | Read local safetensors shard ranges at explicit offsets in Flint, Flash-Next, and Qwen. Contiguous and vectored reads fill existing buffers. Automatic turns on for any engine under a device budget. `1` turns it on, `0` or other values turn it off. |

## Qwen3.6 on Apple Silicon

| Variable | Default | Meaning |
| --- | --- | --- |
| `Q36_CACHE_LIMIT` | none | Fallback for `cache_limit_bytes`, in bytes. |
| `FIFI_MLX_GROUPED_IMPORT` | off | Set to `1` to send each packed MLX import (payload, scales, biases) through the grouped import operation, synchronizing once instead of three times. Applies to every family on Apple Silicon. Read once at first packed import. |
| `Q36_FUSED_PREFILL` | on | Fused kernels for prompt processing. Only `0` turns them off. |
| `Q36_FLASH_PREFILL` | on | Fused flash-prefill kernel replaces composite prefix attention (default-flip gate passed: dNLL -0.000080, top-1 98.045% at 2,046 positions). `0` turns it off. Validated at D >= 128 with Q8 cache and REP 8. |
| `Q36_FLASH_DELTA` | on | Flash-delta gated rule replaces the recurrent kernel in prefill: chunked WY-form kernel (4.1x) runs for >= 16 rows, recurrent kernel serves decode. `0` turns it off. |
| `Q36_FUSED_DECODE` | off | Opt-in fused decode-middle kernel (conv, SiLU, q/k/v split, head norms, decay/beta gates, and ring-shift). Read per call. Default off until parity gates pass. |
| `Q36_PREFILL_QUERY_SLICE` | none | Opt-in query-block rows for F32, BF16, and Q8 composite prefill attention, read once. Fused flash prefill uses the full eligible chunk. |
| `Q36_KV_GEOMETRIC` | `0` | `1` doubles attention caches up to context limit instead of fixed steps, cutting copy work. Admission and reserve charge allocated capacity. During growth, layers hold old and new buffers (~1.6 GB per layer at 128K to 256K). First row charges a 256-position step. Refused with `Q36_SPEC_STEP_WINDOW=2`. Non-`0`/`1` values fail open. Read once. |
| `Q36_NO_ACTMUL` (and siblings) | unset | Kill switches for individual fused-prefill kernels while `Q36_FUSED_PREFILL` stays on: `Q36_NO_ACTMUL`, `Q36_NO_ATTNGATE`, `Q36_NO_EMBEDDED`, `Q36_NO_GDNPREP`, `Q36_NO_GDNPOST`, `Q36_NO_MOECOMBINE`. Presence alone disables that fusion. |

## CUDA

| Variable | Default | Accepted values | Meaning |
| --- | --- | --- | --- |
| `FIFI_CUDA_ATTN_PREFILL_ADAPT` | `0` | `0`, `1` | Experimental FP16 prefill attention selection for supported geometries. `1` picks a kernel from device shared-memory and thread limits. Unset, `0`, and other values keep the original path. Read once per process. Not qualified across RTX 20, 30, 40, and 50 series. See [CUDA FP16 prefill attention](configuration.md#cuda-fp16-prefill-attention). |
| `FIFI_CUDA_POISON_ALLOC` | off (switch) | `1`, `true`, `on` | Debug fill: fills every device allocation from the allocator or cache with `0xFF` so uninitialized reads see `0xFF` instead of old tenant bytes. Streams in 64 MiB chunks and settles the cache first. Leave off outside sanitizer runs and defect hunts. Read once per process. |

## Qwen3.6 on NVIDIA CUDA

The Flash-Next CUDA step runs one row per start, which is under the 64-row threshold of the two `FIFI_GDN_*` rows below. Their tests cover its geometry.

| Variable | Default | Accepted values | Meaning |
| --- | --- | --- | --- |
| `Q36_TIERED_KV_ON_DEMAND` | off | only `1` is on | A single-card start that plans sessions funds a small initial K/V cell count instead of the full context ceiling. It grows the cache as positions arrive. Starts without session planning ignore it. |
| `FIFI_EXL3_GROUP_SPLIT` | `2` | `2`, `4`, `8` | Sets the K split for host-picked routed and grouped shared-expert EXL3 projections. Flash-Next CUDA uses it for the shared expert without changing device-indexed routed calls. `4` and `8` need more scratch space and change partial-sum order. Invalid values use `2`. Read once per process. |
| `Q36_VERIFY_GGUF_SHA256` | `0` | `0`, `1` | `1` checks the built-in GGUF SHA-256 at load. An explicit `qwen36.gguf_sha256` overrides this and always checks its digest. Unpinned GGUFs have no built-in digest, so startup logs and skips. See [content pinning](configuration.md#gguf-content-pinning-cuda). |
| `Q36_ALLOW_UNPINNED_GGUF` | `1` | `0`, `1` | `0` serves only the pinned file. Unset or `1` serves the directory's largest `.gguf` when pinned `Qwen3.6-35B-A3B-Q4_K_M.gguf` is absent. Model open still gates on architecture and tensor set. The pinned file wins when present. |
| `Q36_CUDA_CHUNK_PREFILL` | on (chunked prefill) | `0`, `1` | `0` processes the prompt one token at a time with `q8` and `f32` (FP16) caches. Unset or `1` keeps chunked prefill on. |
| `Q36_CUDA_CHUNK_ROWS` | `1024` | `16`, `128`, `512`, `1024`, `2048` | Largest prompt chunk for the `q8` cache when chunking is on. Ignored for `f32` (FP16) caches, which use 2,048 rows up to 3,072 positions and 1,024 after. |
| `Q36_P2P_CHUNK` | unset | `0` or a positive row count | Experimental multi-device prompt batching. Unset or `0` keeps the eager path; positive values are capped by the peer transport. Prefix reuse is disabled when the chunk path is active. The path is not yet qualified. |
| `Q36_P2P_CHUNK_STAGED` | unset | `1` enables; other values disable | Experimental persistent FP16 KV staging per full-attention layer for P2P chunks. Copies only newly appended rows, invalidates on reset or prefix import, and admits all layer buffers above the existing free-VRAM floor. Falls back to shared staging when memory is insufficient or the prompt exceeds staging capacity. |
| `Q36_P2P_GREEDY_READBACK` | `0` | only `1` is on | Experimental multi-card GGUF greedy decode. After the first draw records its top-two margin, read the winning score from each visible head shard. A winner in tokenizer padding triggers a full-row read. Unset and restart to restore full-row reads throughout. |
| `FIFI_Q36_SERVE_PHASE` | off | only `1` is on | Diagnostic. Records one timing event per stage of every layer for each prompt chunk and prints one `phase-serve` line per chunk on stderr, with the time shares by layer kind (`attn`, `gdn`, `moe`, `embed`). The served bytes are unchanged. Any value other than `0` or `1` fails the first chunked request and every later one until restart. With the switch on, a failed timing event stops the engine until restart. Read once per process. |
| `Q36_P2P_SKIP_PREFILL_HEAD` | `0` | only `1` is on | Experimental multi-device GGUF path: skip the unused vocabulary projection for prompt-only feeds; logits and non-finite output-head checks still run when a prediction is needed. Set at engine startup. |
| `Q36_P2P_ASYNC_BROADCAST` | `0` | only `1` is on | Experimental multi-device GGUF path: order each broadcast publisher after its consumers with cross-device events. Read at engine startup. |
| `Q36_TIERED_TRACE` | off | only `1` is on | On the chunked P2P path, record gate, up and down routed-projection stream intervals for the first and last layer of each card's first prefill chunk, and report one batch per isolated request. It allocates timing events per card and perturbs the measured chunks; do not use it for benchmarks. Read every chunk. |
| `Q36_P2P_ROUTER_TRACE` | `0` | only `1` is on | For the eager multi-device token path, report the position, layer and finiteness counts at the first router fault. It synchronizes each layer, can stall if CUDA work stalls, and can alter timing or mask races; do not use it for benchmarks. It does not report exact device softmax sums, so finite inputs do not identify the fault predicate. Read once per process. |
| `INFERENCE_CUDA_P2P_RING_MULTIBLOCK` | `0` | only `1` is on | Experimental multi-block peer-ring f32 publishes and f32/FP16 reductions above 2,048 values; FP16 publishes stay single-block. Unset or any value other than `1` keeps the single-block path. Set before startup; changing it requires a restart. This path is not yet qualified. |
| `Q36_CUDA_KV8_STAGE_CONTEXT` | `1` | `0`, `1` | `0` limits the FP16 attention staging buffer to 65,536 rows. |
| `Q36_CUDA_KV8_FIXED_ADMISSION` | `1` | `0`, `1` | `0` returns to the older way of choosing chunk sizes, which may pick smaller chunks or one token at a time. On a 24 GB card, use the 131K profile with it. With the default, a start that is short of card memory shrinks the prefill stage step by step, down to a floor of max(8192, context / 128) rows, before it refuses. Above 131,072 rows of context the shrink frees at most 48 MiB. It does not make the 262,144 profile fit a 24 GB card that also runs a desktop: that start was measured 393.5 MiB short with prefix reuse off and 456.3 MiB short with it on. |
| `Q36_CUDA_CPF_STAGE_128K` | `1` | `0`, `1` | Above 131,072 tokens, attention stages a 131,072-row buffer (256 MiB) when it fits. `0` starts from the 32,768-row buffer (64 MiB) and halves it whenever it is refused (also after a refused wide stage with the flag on), down to the floor max(8192, ceil(context / 128)) — where 8192 is the measured performance floor; halves that cross the floor stop the ladder earlier, and a snapshot too tight for every rung refuses with the original error. No effect at or below 131,072 tokens. |
| `Q36_CUDA_PINNED_PREFIX` | on | only `1` is on | Any other value keeps saved prefixes in pageable host memory and turns off the GPU copy. |
| `Q36_CUDA_STATE_ONLY_PREFIX` | on | only `1` is on | Any other value turns off the GPU copy of the newest prefix and keeps the pinned host copies. |
| `Q36_CUDA_PINNED_PREFIX_BYTES` | smaller of 9 slots and half the available host memory | byte count | Size of the pinned host pool for saved prefixes. One slot stages a save and the rest retain one prefix each, so the default keeps up to 8 prefixes and fewer when half the available host memory (clamped by the cgroup limit) holds fewer slots. The value is an exact cap: it is still limited to 8 retained prefixes and to what the host can hold beside 1 GiB of headroom. `0` keeps none. The startup line `persistent pinned prefix` reports the retained count. |
| `Q36_CUDA_PAGED_DEVICE_FIRST` | `0` | `0`, `1` | With `kv_offload_host_bytes`: keep as much of the KV cache on the GPU as fits after a 1 GiB reserve. |
| `INFERENCE_CUDA_KV8_ATTN_V_PREFETCH` | `0` | only `1` is on | Experimental register-V decode attention: use the prefetch kernel variant, which holds the next key's code words in registers ahead of the dot chain. Read once at startup; only the exact value `1` turns it on, any other value or unset keeps the default kernel. Head dimension 256 only, and it needs the register-V KV8 decode attention path (the default with `kv_dtype: q8`); outputs are bitwise-equal to it. |
| `FIFI_GDN_CONV_PREFILL` | off | `1`, `16`, `32`, `64` (surrounding spaces ignored) | Opt-in for the DeltaNet layers of Qwen3.6 on CUDA: runs the causal-conv step of a prompt chunk of 64 or more rows on a kernel that splits the rows over many blocks. `1` selects 32 rows per thread; `16`, `32` and `64` select that many. Any other value keeps the default kernel, and launches under 64 rows always do. The output is meant to be byte-identical; the device tests that compare it have not run on a device, and the speed is not measured. Read once per process. |
| `FIFI_GDN_SCAN_COLS` | off | `8`, `16`, `32` (surrounding spaces ignored) | Opt-in for the DeltaNet layers of Qwen3.6 prefill on CUDA: runs the gated-delta scan of a launch of 64 rows or more with that many value columns of a head per block instead of 32. `16` gives 256 blocks and `8` gives 512; `32` runs the default shape from a separate kernel copy, for comparison. Any other value keeps the original kernel, and launches under 64 rows always do. The output is meant to be byte-identical; the device tests that compare it have not run on a device, and the speed is not measured. Read once per process. |
| `INFERENCE_QWEN36_SPARSE_ATTN` | `0` | only `1` is on | Arm the sparse decode attention path (bounded admission; unsupported heads fall back to dense). Read once per process. |
| `Q36_CUDA_MIN_FREE_MIB` | admission default | MiB count | Free-after-load floor of the CUDA memory admission, the same bytes for the gate's qualified loop and the server start; parsed once so the checks cannot diverge. An out-of-range value refuses with the accepted range named. |
| `FIFI_CPU_Q8G_FAST_QUANT` | on with AVX2 (x86_64), off elsewhere | `1`, `true`, `on` keep on | Use integer magnitude reduction and signed half comparisons for Q8g CPU expert quantization. Bit-identical to reference. Unset uses fast path on AVX2. Non-enable values fall back to reference. Restart to change. |
| `FIFI_CPU_Q8G_NARROW_SUM` | on with AVX2 | `1`, `true`, `on` keep on | Accumulate eight Q4_K x Q8g products in 16-bit lanes before widening in AVX2 CPU expert dot. Bit-identical integer sums. Unset uses narrow path on AVX2. Non-enable values fall back to widened path. Restart to change. |
| `Q36_TIERED_PREFIX` | `0` | only `1` is on | Tiered sessions may save and restore prefixes. Default off pending device qualification. |
| `Q36_TIERED_PREFIX_EXTRA_BYTES` | unset | unsigned byte count | With `prefix_reuse` and `Q36_TIERED_PREFIX=1`, allows a second pageable checkpoint on single-GPU CUDA. Startup checks full-context state plus 1 GiB headroom, else keeps one entry. Malformed values fail startup. Contributor's 22.4 s to 0.26 s switch figure is unverified here. |
| `Q36_TIERED_CHUNK` | `1` | only `0` is off | Tiered sessions run chunked prefill. `0` keeps per-token prefill. |
| `Q36_TIERED_COLD_LIMIT` | unset | unsigned count | Experimental cap on cold staging per chunk layer, trading PCIe transfers for CPU overflow work. Allocation, admission, and bank depth stay unchanged. |
| `Q36_TIERED_COLD_MIN_ROWS` | `1` | unsigned count, at least 1 | Minimum uses in a chunk layer before earning a PCIe transfer. Picks below it keep the CPU overflow path. |
| `Q36_TIERED_COLD_SHORT_LIMIT` | unset | unsigned count | With chunked prefill, chunks below session width stage at most this many cold experts. The ordinary limit stays the hard ceiling. |
| `Q36_TIERED_PREFILL_4096` | `0` | `0` or `1` | Raises a 2,048-row chunk cap to 4,096 for one session. Requires native tiered KV8, non-fixed admission, and the 2,048-row cap. Other starts refuse. |
| `Q36_TIERED_PREFILL_WARM` | `0` | only `1` is on | Tiered prefill only: after the first >= 64-row chunk commits, moves up to four experts from its last `32 * top_k` picks into layer banks outside swap credit. Failed fills poison the session. Not qualified. |
| `Q36_TIERED_PROBE` | `0` | only `1` is on | Times the per-layer route wait and token device span during decode so their split can decide the TCAP gate. |
| `Q36_TIERED_SPLIT_JOBS` | `0` | only `1` is on | Split each missed expert's job across worker threads (two half-projections) and the joining thread (down pass). Whole-expert path is the default. |
| `Q36_TIERED_SPLIT_MAX_JOBS` | unset | unsigned count | Experimental cap on split-job dispatch. Unset keeps the existing split policy. |
| `Q36_P2P_WATERFILL` | `0` | only `1` is on | Experimental multi-device GGUF path: bandwidth-waterfill placement planner redistributes sliceable roles instead of even splitting. |
| `Q36_P2P_HOST_EXPERT_BYTES` | `0` | nonnegative byte count | With waterfill: per-card sliced mass moved to pinned host RAM at load (`_0` sets card 0 separately). Unparseable values refuse startup. |
| `Q36_P2P_PREFIX` | `0` | only `1` is on | Experimental: prefix save/restore on the card path. Wired but not yet qualified. |
| `Q36_P2P_SYNC_BROADCAST` | `0` | only `1` is on | Cross-card MoE reduction through sync broadcasts instead of the peer ring. |
| `INFERENCE_CUDA_GDN_FUSE_INPROJ` | `0` | only `1` is on | Fuse four Q8_0 DeltaNet input projections (`attn_qkv`, `attn_gate`, `ssm_beta`, `ssm_alpha`) into one upload and call. Bitwise-identical per segment and ~20% faster in isolated A/B bench. Read once per load. |
| `INFERENCE_CUDA_TILED_HEAD` | none | comma-separated tensor names | Decode-only tiled-head hybrid: uploads named tensors in `Tiled` layout for vendored tiled-head matmul beside native ops. Empty or unset disables override. |
| `INFERENCE_CUDA_MMID_COLS_HINT` | `auto` | `auto`, `off`/`0`, integer | Column-block hint for MMID prefill calls. `auto` keeps the shape-driven choice. |
| `Q36_LAYER_TRACE_DIR` | unset (off) | directory | Layer-trace output directory. While tracing, the CUDA decoder also runs EAGER steps. |

`Q36_CUDA_CHUNK_PREFILL=0` and `Q36_CUDA_KV8_STAGE_CONTEXT=0` also turn off `Q36_CUDA_KV8_FIXED_ADMISSION`.
With `tiered`, the pinned prefix pool stays off unless `Q36_CUDA_PINNED_PREFIX=1`.

## Flash-Next on NVIDIA CUDA

The Flash-Next family serves on the card in a `native-cuda` Linux build unless `flash_next.cuda_decode` is `false` (not validated end to end):

| Variable | Default | Meaning |
| --- | --- | --- |
| `FLASH_NEXT_ARTIFACT_MANIFEST` | unset | Path of the artifact manifest, read once at engine start. |
| `FLASH_NEXT_CACHE_LIMIT` | unset | Cache limit in bytes, read once at engine start. |
| `FIFI_FLASH_NEXT_BANK_MODE` | `auto` (strict) | Expert-bank store mode: `inclusive` (all experts in host RAM, ~43 GiB), `exclusive` (evicting), `auto` (inclusive, else exclusive, else `disk` ring), or `disk` (ring only, O_DIRECT probed once, ~0.12 s/cold step). Refuses if memory plan fails. |
| `FIFI_FLASH_NEXT_EXPERT_STATS` | unset | Path of a popularity-profile JSON (`expert_stats` format). Engine `expert_stats` key wins. Malformed files refuse startup. |
| `FIFI_FLASH_NEXT_EXPERT_USAGE_OUT` | unset | Output path for expert-usage dump at shutdown (engine `expert_stats` key wins), warming the next start. |
| `INFERENCE_FLASH_NEXT_PROFILE` | unset | Chunk phase profile (`1`/`true`/`on` arm, `0`/`false`/`off`/unset leave off, else refuse). Sums host and device wait per phase and prints `flash-next feed profile` (serial/bulk row counts and embedding/n-gram/join times). Totals upper-bound copy time and perturbs walls. |
| `INFERENCE_FLASH_NEXT_DRAIN_TICK` | unset | Companion to `INFERENCE_FLASH_NEXT_PROFILE` (`1`/`true`/`on` arm, else refuse). Synchronizes the step stream at each phase boundary to separate host and device times. Serializes execution, so walls are longer. Does nothing if profile is off. |
| `INFERENCE_FLASH_NEXT_DISK_PROFILE` | unset | Disk-tier CUDA expert refill profile (`1`/`true`/`on`). Emits prefill and decode records at turn boundary with demand wall, ring wait, CPU parse/pack, and copy completion times. |
| `INFERENCE_FLASH_NEXT_PREFILL_ROUTE_TRACE` | unset | Disk-tier CUDA diagnostic (`1`/`true`/`on`). Logs each union-admit chunk's initial resident experts, router picks, and counted refills. Output can be large on long prompts. |
| `INFERENCE_FLASH_NEXT_PREFILL_EXACT_GROUPS` | unset | Disk-tier CUDA prefill experiment (`1`/`true`/`on`). Groups consecutive rows while their distinct expert union fits the slot budget. |
| `INFERENCE_FLASH_NEXT_PREFILL_WAVE_LAYER` | unset | Disk-tier CUDA experiment. Set a zero-based layer or `all` to run expert waves: projects routed GPU hits first, then admits cold experts in slot-sized waves. Singletons use the configured planner. |
| `INFERENCE_FLASH_NEXT_PREFIX_RESUME` | unset (on) | Prefix reuse for Flash-Next CUDA (`0`/`false`/`off` opt out, `1`/`true`/`on` arm, else refuse). Needs config `prefix_reuse` enabled (or unset). Saves fixed device state (~110.8 MiB) in a 256 MiB host RAM budget (tip plus 64-token rolling block), restoring matching prefixes and skipping re-prefill. Invalidates on fresh sequences or overwritten boundaries. |
| `FIFI_FLASH_NEXT_GGUF` | unset (off) | Experimental, not qualified. Opt-in for Flash-Next GGUF (`qwen4exp`) on CUDA: reads file in place, one token per step, prefix resume off. Serves only when `1`, else refuses. `INFERENCE_FLASH_NEXT_GGUF_SERVE` takes precedence. |
| `INFERENCE_FLASH_NEXT_GGUF_SERVE` | unset (off) | Experimental, not qualified. Staged GGUF serving path (`1`/`true`/`on` serve, other values refuse). Needs `download-manifest.json` in tokenizer dir (or manifest setting). Ignores `FIFI_FLASH_NEXT_GGUF`. |
| `FIFI_FLASH_GGUF_QSA_CACHE` | unset (`f16`) | QSA cache storage for Flash-Next GGUF decode (`f16`, or `f32` for decode-only up to 16 rows/step using double bytes). Other values refuse. |
| `INFERENCE_FLASH_NEXT_NGRAM_BULK` | unset (on) | Reads and decodes distinct n-gram rows per multi-token chunk via a 32-thread pool while gathering embeddings (`0` forces serial, `1` default, else refuse). Measured 24K prompt: 136.1 s pooled vs 150.4 s serial. |
| `INFERENCE_FLASH_NEXT_PLE_DEVICE` | unset | Set to `1` to project Flash-Next PLE embeddings on CUDA when `prefill_rows` > 1. Borrows stored F16/BF16 matrices and reserves workspace. Other values refuse CUDA startup. |
| `INFERENCE_FLASH_NEXT_PLE_BATCH` | unset | Chunked prefill only (`1` opts in, other values refuse): batches the two PLE host projections (131 MB weights) per chunk, bit-identical to per-token matvecs. Ignored if `prefill_rows` is 1 or PLE device route is on. Read once per process. |
| `INFERENCE_EXL3_ROWS_MAX` | unset (2,048) | Raises Flash-Next CUDA max shared-prefill rows above 2,048 up to 4,096 (unqualified). Digits 1–4096 only, else refuse. Values below 2,048 never lower the ceiling. Read once per process. |
| `INFERENCE_TUNER` | unset (`auto`) | Diagnostic line toggle at Flash-Next CUDA start. `off` prints `mode off (...)` beside in-use values. Any other value prints `auto` suggestions beside in-use values without applying them. Read once per process. |
| `INFERENCE_FLASH_NEXT_RING_BUDGET` | unset (off) | Byte budget for Flash-Next CUDA prefill expert ring (carved from free VRAM before bank sizing, with `0` or unset disabling). Chunks >= `ring_rows` (default 32) stream whole chunks instead of union-admit. Unqualified. A 24K prompt measures ~225 s vs ~382 s on the bank path. |
| `FIFI_EXL3_SINGLE_GATHER` | `wide1` | Kernel for single-row EXL3 projections: `off` (shuffle), `1` (per-warp gather), `wide1`, `wide2`, or `wide4` (16-col tiles per block). Bitwise-equal outputs. Any other value stops startup. Read once per process. |
| `FIFI_EXL3_DP4A` | on | Decodes EXL3 codebook 2 with packed byte dot product in routed-expert calls (`0` turns off). Cannot combine with `wide2` or `wide4`. Read once per process. |
| `FIFI_EXL3_INDEXED_PAIRS` | 20 | Max expert pairs per routed-expert call: `20` or `16`. Read once per process. |
| `FIFI_EXL3_JOBS_REG` | on | `0` runs EXL3 prefill job-table projections on shared-memory kernels instead of register-decode. Same bits. Read once per process. |
| `FIFI_EXL3_JOBS_WARP_ROTATE` | off | `1` rotates job-table input rows with one warp per 128-value group. Byte-identical. Non-`0`/`1` values fail. Read once per process. |
| `FIFI_EXL3_JOBS_PAIR_ROTATE` | off | `1` rotates gate/up input rows of routed experts in one pass. Byte-identical. Untimed on device. Non-`0`/`1` values fail. Read once per process. |
| `FIFI_EXL3_CPU_ISA` | `auto` | CPU ISA for host EXL3 experts: `auto` (AVX-512 VBMI/VNNI, else AVX2 FMA, else scalar), `avx512`, `avx2`, or `scalar`. Missing ISA fails. Read at engine creation. |

## SenseNova-Loop

| Variable | Default | Accepted values | Meaning |
| --- | --- | --- | --- |
| `FIFI_SENSENOVA_CUDA` | off | only `1` is on | Serves SenseNova-Loop from one CUDA device instead of the CPU. Needs a `native-cuda` build, else startup refuses. |
| `FIFI_SENSENOVA_DECODE_GEMM` | unset | `dq` | With the card step, runs decode projections through the kernel that dequantizes each weight block to F32. Any other value keeps default kernels. |
| `FIFI_SENSENOVA_PREFILL_GEMM` | unset | `dq` | With the card step, runs Q4_K and Q6_K prefill projections through the dequantizing matrix product. Any other value keeps default kernels. |

## Flint

Flint serves on Apple Silicon only.

| Variable | Default | Accepted values | Meaning |
| --- | --- | --- | --- |
| `FIFI_FLINT_MLX_CONTEXT_LIMIT` | `32768` | `32768`, `8192` | Context limit. Any other value selects 8,192. |
| `FIFI_FLINT_MLX_BATCH_PREFILL` | `1024` | `1024`, `512`, `0` | Prompt chunk size. `0` processes one token at a time (50 to 70 times slower). Any other value selects 512. |
| `FIFI_FLINT_MLX_REQUEST_PREFILL` | off | `1` | Price prefill attention workspace from the admitted prompt span for a fresh request. Needs the Metal budget. Prefix-marked requests and chained prefill keep startup width. |
| `FIFI_FLINT_MLX_PREFILL_PAIR` | off | `1`, `auto`, `waves64` | Share expert reads across a fresh two-piece Flint prompt (`1` single stage, `auto` single then 64-expert waves, `waves64` waves). Needs request-aware prefill, disk-backed experts, and shared-page pipeline. |
| `FIFI_FLINT_MLX_CANCEL_PREFILL_PAIR` | off | `1` | Set to `1` to stop an admitted Flint pair between layers when the client cancels. Worker discards partial session before next request. |
| `FIFI_FLINT_MLX_COMPACT_KV` | off | `1` | Start Flint full-attention KV planes at 256 rows or first prefill capacity (power of two), doubling up to context limit. Sliding-window planes keep 513 rows. |
| `FIFI_FLINT_MLX_REQUEST_KV` | off | `1` | Size initial full-attention KV from admitted prompt and generation limit. Needs compact KV and Metal budget. Prefix boundaries or unset limits keep geometric growth. |
| `FIFI_FLINT_MLX_CACHE_LIMIT` | `67108864` | `67108864`, `0` | MLX freed-buffer cache in bytes. `0` disables caching (driver round-trip per freed buffer). Any other value selects `0`. |
| `FIFI_FLINT_MLX_ASYNC_CHAIN` | on | switch | Evaluate decode work in chains of 4 layers instead of synchronizing after every layer. |
| `FIFI_FLINT_MLX_COALESCE_RESIDUAL` | on | switch | Merge the residual updates of a decode step. |
| `FIFI_FLINT_MLX_DECODE_ATTN` | on | switch | Use fused MLX attention kernel for one-token decoding. Startup refuses checkpoints with inexact BF16 attention-sink values, naming layer, head, and value. |
| `FIFI_FLINT_MLX_EXPERT_PIPELINE` | off | switch | Disk-backed Flint experts only: submit the shared expert to Metal before the host reads routed expert slices, overlapping read with computation. |
| `FIFI_FLINT_MLX_EXPERT_READ_RUNS` | off | `1`, `prefill`, or off | Disk-backed Flint experts only: combine consecutive expert slices into one read (`1` all stages, `prefill` multi-row forwards). Needs shared-page pipeline. |
| `FIFI_FLINT_EXPERT_TIMING` | off | `1` | Qualification builds only: print `flint_expert_timing` JSON per disk-backed expert forward with wall time, slice-read, and page-import counters. |
| `FIFI_FLINT_EXPERT_TRACE` | off | `1` | Qualification builds only: print `flint_expert_access` JSON per disk-backed expert stage with layer stem, row count, and sorted expert IDs. No prompt text. |
| `FIFI_FLINT_MLX_RING_VIEW` | on | switch | Read full-attention KV cache as one ring view instead of concatenating each step. Needs `FIFI_FLINT_MLX_ASYNC_CHAIN`. |
| `FIFI_FLINT_PREFIX_REUSE` | off | switch | Experimental: keep prefix snapshots between turns. Also needs `"prefix_reuse": true`. |
| `FIFI_FLINT_PREFIX_BUDGET_BYTES` | physical memory minus weights and reserve | positive byte count | Memory cap for prefix snapshots. `0` or an unreadable value refuses startup. |
| `FIFI_FLINT_MLX_COMPILE_LAYER` | off | switch | Run layer tail through one compiled MLX closure. Absent, non-UTF-8, or non-enable values keep eager tail. Read once per process. |
| `FIFI_FLINT_MLX_EXPERT_CACHE_MIB` | off | positive MiB, `0` | Reserve per-layer unified-memory slots for packed experts in one-row disk-backed forwards. Shrinks if budget is tight, refuses below 6 slots/layer or non-integer, and ignores if sizing off or weights fit. |
| `FIFI_FLINT_MLX_PREFILL_CHAIN` | off | switch | Chained multi-row prefill groups (graph-ordered, fence-free). Default off until an A/B proves +10% pp4096 tok/s. Any non-enable value rolls back. |

## Nemotron 3.5 on Apple Silicon

| Variable | Default | Meaning |
| --- | --- | --- |
| `FIFI_NEM35_CPU_NEON` | off | Set to `1` to use the Arm NEON NVFP4 dot product in the mapped CPU expert pool on Apple Silicon. The pool reads it at construction. [Mapped expert profiling](../how-to/nemotron.md#profile-mapped-experts-on-apple-silicon) |

## Nemotron 3.5 on NVIDIA CUDA

Nemotron serves greedy requests only on CUDA (refusing sampled requests). These switches select kernels and diagnostics. All default off unless noted:

| Variable | Default | Meaning |
| --- | --- | --- |
| `NEM35_CHUNK_EXPERTS` | unset | `gpu`: large prompt chunks may stream the layer's routed experts to the device instead of the CPU pool. |
| `NEM35_CUDA_CHUNK_PREFILL` | `0` | With `NEM35_CHUNK_EXPERTS=gpu`: stream the routed experts in row groups of this many rows (capped by the chunk capacity). |
| `NEM35_CUDA_GROUPED_EXPERTS` | off | Grouped NVFP4 GEMM for resident expert batches. |
| `NEM35_NO_BATCH` | off | Kill switch for the legacy GPU batch expert kernel. |
| `NEM35_NO_F32_CACHE` | off | Bypass the F32 dense cache (per-call pinned/pageable upload), for isolating cache issues at production scale. |
| `NEM35_NO_PACKED_CACHE` | off | Bypass packed-weight device residency (upload per call). |
| `NEM35_RESIDENT_DENSE` | off | Startup guard: log the resident dense count and bytes after the first forward and warn if any F32 dense cache remains. |
| `NEM35_DEVICE_CHAIN` | off | The resident guard also checks the device mamba step count against the expected count. |
| `NEM35_DECODED_A_B` | off | Force the decoded route for one measurement run (diagnostic, while default stays the device route). |
| `NEM35_DIAG_TOP2` | off | Log the top-two logits of every step on the CUDA head (diagnostic). |
| `NEM35_PHASE_TIMING` | off | Time host spans per phase. |
| `NEM35_SUBOP_TIMING` | off | Time the sub-operations of the MoE CPU routed path. |
| `NEM35_TOKEN_TIMING` | off | Print each step's timing split to stderr as it closes, so a run cut short keeps the steps it finished. |

## Bolt (Gemma4) on NVIDIA CUDA

| Variable | Default | Accepted values | Meaning |
| --- | --- | --- | --- |
| `FIFI_BOLT_ATTN_SPLIT` | on (switch) | `1`, `on`, `0`, `off` | Split decode attention (partial + combine) instead of single-pass. Matches f64 reference within tolerance. Measured 11.7x faster attention and 6.8 to 43 tok/s decode at 30K depth. `0` or `off` restores single-pass, else refuses. Read once per process. |
| `FIFI_BOLT_ATTN_SPLITS` | device-derived | `1` to `128` | Overrides Bolt decode attention split count. Unset scales 32 by device SM count over 82 (82-SM = 32, 20-SM = 8, unknown = 32). Changes summation order and prefix keys. Non-range values fail. Restart to change. |

## Tests

`cargo test` reads these, not `enginesrv`. Tests needing recorded data check the environment variable first, then `~/.config/fifi/test-paths.json`. Missing locations skip unless `FIFI_TEST_REQUIRE_DATA=1` or the test is `#[ignore]`. Harnesses read further variables (`FLINT_QUAL_*`, `DT_*`, `FIFI_GRAPH_BENCH_*`, `FIFI_FLINT_*_ORACLE*`, `FLINT_PROBE_*`, `*_CHILD`).

| Variable | Default | Meaning |
| --- | --- | --- |
| `FIFI_EXL3_SWEEP_LAYERS` | `12` | Layers of six experts cycled in the ignored `flint_shape_rate_latency` timing test (`1` stays cache resident). Run `cargo test --release --lib flint_shape_rate_latency -- --ignored --nocapture` from `engine/`. |
| `M3_LAYOUT_DIR` | unset | Frozen layout directory for EXL3-vs-GGUF quality comparison. Two default tests skip when unset. Ignored `m3_compare` requires it. |
| `M3_COMPARE` | unset | For ignored `m3_compare`: `P1:<candidate dir>:<reference dir>` or `O1:<candidate dir>:<reference dir>`. Run `cargo test --lib m3_compare -- --ignored --nocapture` from `engine/`. |
| `M3_REGISTRATION` | unset (refused) | For `m3_compare`: `v6` or `v7`, naming the registration version. Unset or unknown values stop the test before reading sidecars (see `qualification/o1-gate-v7/`). |
