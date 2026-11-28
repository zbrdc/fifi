# Internal and experimental environment variables

These variables control diagnostics, tracing, un-qualified kernel experiments, and test harnesses.
For supported user-facing variables, see [Environment variables](environment.md).

## Build and CUDA diagnostics

| Variable | Default | Meaning |
| --- | --- | --- |
| `FIFI_CUDA_ATTN_PREFILL_TURING` | off | `1` enables the typed half-precision Turing pilot arm (`sm75`, not device-qualified). Read once per process. |
| `FIFI_CUDA_ATTN_PREFILL_ADAPT` | `0` | Experimental FP16 prefill attention selection (`0` or `1`). Uses device shared-memory and thread limits. Not qualified across RTX 20–50 series. |
| `FIFI_CUDA_POISON_ALLOC` | off (switch) | Debug fill: fills every device allocation from the allocator or cache with `0xFF` in 64 MiB chunks to catch uninitialized reads. |

## Server and CPU diagnostics

| Variable | Default | Meaning |
| --- | --- | --- |
| `FIFI_CPU_GREEDY_BLOCKS` | off | `1` scans host logits in 256-value blocks for greedy sampling. Other values use scalar scan. |
| `FIFI_CPU_Q8G_F16C` | off | `1` runs the AVX2 CPU expert inner dot product in half precision via F16C when supported. |
| `FIFI_CPU_Q8G_FAST_QUANT` | on with AVX2 | `1`/`true`/`on` keep integer magnitude reduction for Q8g CPU quantization. Other values fall back to reference. |
| `FIFI_CPU_Q8G_NARROW_SUM` | on with AVX2 | `1`/`true`/`on` accumulate eight Q4_K x Q8g products in 16-bit lanes before widening. Other values use widened sums. |
| `ENGINESRV_SHADOW_BLOCK_LCP` | off (switch) | Read-only observation for the request log: reports the longest common prefix the saved block checkpoint would match. |

## Apple Silicon experiments and profiling

| Variable | Default | Meaning |
| --- | --- | --- |
| `FIFI_MLX_REUSE_IDENTITY_RESHAPE` | off | `1` shares a native resident tensor when the requested reshape matches its current dimensions. |
| `FIFI_MLX_GROUPED_IMPORT` | off | `1` sends packed MLX imports (payload, scales, biases) through one grouped import, synchronizing once instead of three times. |
| `FIFI_MLX_EXPERT_READ_RUNS` | automatic | Group adjacent expert slice reads on multi-row disk-backed paths (`prefill` on, `0` off). |
| `FIFI_MLX_OVERLAP_EXPERT_READS` | automatic | Submit shared expert to Metal before CPU reads routed weights (`1` on, `0` off). |
| `FIFI_MLX_POSITIONED_READS` | automatic | Read local safetensors shard ranges at explicit offsets (`1` on, `0` off). |
| `FIFI_FLASH_NEXT_MLX_SPARSE_PREFILL` | off | `1` batches up to 8 sparse Flash-Next MLX prompt rows when chunked prefill has leftover workspace. |
| `FIFI_FLASH_NEXT_MLX_CAPACITY_KV` | off | `1` appends QSA keys, values, and index keys to geometrically grown Metal buffers. |
| `FIFI_FLASH_NEXT_MLX_STREAM_TIMING` | off | `1` logs logical expert-read, page-import, and MLX wait counters per Flash-Next forward in qualification builds. |
| `FIFI_Q36_MLX_DISK_DENSE_FFN` | off | `1` stages Qwen3.8 dense gate, up, and down matrices one layer at a time under MLX device sizing. |
| `FIFI_Q36_MLX_STREAM_TIMING` | off | `1` logs one timing record per streamed forward in qualification builds. |
| `FIFI_Q36_MLX_EXPERT_TRACE` | off | `1` logs sorted checkpoint expert IDs and logical reads per staged layer for Qwen3.6 MoE. |
| `FIFI_Q36_MLX_FUSE_FINAL_PREFILL` | off | `1` includes the pending prompt token in the last Qwen3.8 dense prefill batch. |
| `Q36_FUSED_DECODE` | off | Opt-in fused decode-middle kernel for Qwen3.6 on Apple Silicon. Read per call. |
| `Q36_PREFILL_QUERY_SLICE` | none | Opt-in query-block rows for F32, BF16, and Q8 composite prefill attention. |
| `Q36_KV_GEOMETRIC` | `0` | `1` doubles attention caches up to context limit instead of fixed steps. Refused with `Q36_SPEC_STEP_WINDOW=2`. |
| `Q36_NO_ACTMUL` (and siblings) | unset | Presence disables individual fused-prefill kernels: `Q36_NO_ACTMUL`, `Q36_NO_ATTNGATE`, `Q36_NO_EMBEDDED`, `Q36_NO_GDNPREP`, `Q36_NO_GDNPOST`, `Q36_NO_MOECOMBINE`. |

## Qwen3.6 on NVIDIA CUDA experiments and diagnostics

| Variable | Default | Meaning |
| --- | --- | --- |
| `Q36_TIERED_KV_ON_DEMAND` | off | `1` funds a small initial K/V cell count on single-card session starts and grows on demand. |
| `Q36_CUDA_KV8_STAGE_CONTEXT` | `1` | `0` limits the FP16 attention staging buffer to 65,536 rows. |
| `Q36_CUDA_KV8_FIXED_ADMISSION` | `1` | `0` uses the older chunk-sizing ladder instead of shrinking the prefill stage step by step. |
| `Q36_CUDA_CPF_STAGE_128K` | `1` | Above 131,072 tokens, stages a 131,072-row buffer (256 MiB) when it fits. `0` starts at 32,768 rows and halves down. |
| `Q36_CUDA_STATE_ONLY_PREFIX` | on | Any value other than `1` turns off the GPU copy of the newest prefix and keeps pinned host copies. |
| `Q36_CUDA_MIN_FREE_MIB` | admission default | Free-after-load floor (MiB) shared by qualification and server startup. |
| `INFERENCE_CUDA_KV8_ATTN_V_PREFETCH` | `0` | `1` enables experimental register-V decode attention prefetch kernel (head dim 256 only). |
| `FIFI_GDN_CONV_PREFILL` | off | `1`, `16`, `32`, or `64`: multi-block causal-conv kernel for DeltaNet chunks >= 64 rows. |
| `FIFI_GDN_SCAN_COLS` | off | `8`, `16`, or `32`: value columns per block for DeltaNet prefill scans >= 64 rows. |
| `INFERENCE_QWEN36_SPARSE_ATTN` | `0` | `1` arms sparse decode attention (bounded admission, dense fallback). |
| `INFERENCE_CUDA_GDN_FUSE_INPROJ` | `0` | `1` fuses four Q8_0 DeltaNet input projections into one upload and call. |
| `INFERENCE_CUDA_TILED_HEAD` | none | Comma-separated tensor names uploaded in `Tiled` layout for decode-only tiled-head matmul. |
| `INFERENCE_CUDA_MMID_COLS_HINT` | `auto` | Column-block hint for MMID prefill calls (`auto`, `off`/`0`, or integer). |
| `FIFI_Q36_SERVE_PHASE` | off | `1` records per-stage timing events and prints `phase-serve` lines on stderr. |
| `Q36_LAYER_TRACE_DIR` | unset | Layer-trace output directory. Forces EAGER steps in the CUDA decoder. |
| `Q36_P2P_CHUNK` | unset | Experimental multi-device prompt batching (`0` or positive row count). Disables prefix reuse. |
| `Q36_P2P_CHUNK_STAGED` | unset | `1` enables experimental persistent FP16 KV staging per full-attention layer for P2P chunks. |
| `Q36_P2P_GREEDY_READBACK` | `0` | `1` enables experimental multi-card greedy readback of top-two margins. |
| `Q36_P2P_SKIP_PREFILL_HEAD` | `0` | `1` skips unused vocabulary projection for prompt-only feeds on P2P. |
| `Q36_P2P_ASYNC_BROADCAST` | `0` | `1` orders P2P broadcast publishers after consumers with cross-device events. |
| `Q36_P2P_WATERFILL` | `0` | `1` enables experimental bandwidth-waterfill placement planner on P2P. |
| `Q36_P2P_HOST_EXPERT_BYTES` | `0` | Per-card sliced mass moved to pinned host RAM under waterfill (`_0` sets card 0). |
| `Q36_P2P_PREFIX` | `0` | `1` enables experimental prefix save/restore on the card path. |
| `Q36_P2P_SYNC_BROADCAST` | `0` | `1` uses sync broadcasts instead of the peer ring for cross-card MoE reduction. |
| `INFERENCE_CUDA_P2P_RING_MULTIBLOCK` | `0` | `1` enables experimental multi-block peer-ring f32 publishes and reductions above 2,048 values. |
| `Q36_TIERED_TRACE` | off | `1` records routed-projection stream intervals on chunked P2P. Do not use for benchmarks. |
| `Q36_P2P_ROUTER_TRACE` | `0` | `1` reports position, layer, and finiteness counts at the first router fault. Do not use for benchmarks. |
| `Q36_TIERED_PREFIX` | `0` | `1` allows tiered sessions to save and restore prefixes. |
| `Q36_TIERED_PREFIX_EXTRA_BYTES` | unset | Allows a second pageable checkpoint on single-GPU CUDA when `Q36_TIERED_PREFIX=1`. |
| `Q36_TIERED_CHUNK` | `1` | `0` disables chunked prefill for tiered sessions. |
| `Q36_TIERED_COLD_LIMIT` | unset | Experimental cap on cold staging per chunk layer. |
| `Q36_TIERED_COLD_MIN_ROWS` | `1` | Minimum uses in a chunk layer before earning a PCIe transfer. |
| `Q36_TIERED_COLD_SHORT_LIMIT` | unset | Max cold experts staged for chunks below session width. |
| `Q36_TIERED_PREFILL_4096` | `0` | `1` raises a 2,048-row chunk cap to 4,096 for one session. |
| `Q36_TIERED_PREFILL_WARM` | `0` | `1` pre-warms up to four experts after the first >= 64-row chunk. |
| `Q36_TIERED_PROBE` | `0` | `1` times per-layer route wait and token device span during decode. |
| `Q36_TIERED_SPLIT_JOBS` | `0` | `1` splits missed expert jobs across worker threads. |
| `Q36_TIERED_SPLIT_MAX_JOBS` | unset | Experimental cap on split-job dispatch. |

## Flash-Next on NVIDIA CUDA experiments and diagnostics

| Variable | Default | Meaning |
| --- | --- | --- |
| `FIFI_FLASH_NEXT_EXPERT_STATS` | unset | Path to a popularity-profile JSON (`expert_stats` format). Engine key wins. |
| `FIFI_FLASH_NEXT_EXPERT_USAGE_OUT` | unset | Output path for expert-usage dump at shutdown. |
| `INFERENCE_FLASH_NEXT_PROFILE` | unset | `1`/`true`/`on` prints chunk phase profiles and `flash-next feed profile`. Perturbs walls. |
| `INFERENCE_FLASH_NEXT_DRAIN_TICK` | unset | Companion to `INFERENCE_FLASH_NEXT_PROFILE`: syncs step stream at phase boundaries. |
| `INFERENCE_FLASH_NEXT_DISK_PROFILE` | unset | `1`/`true`/`on` emits disk-tier prefill and decode refill profiles at turn boundaries. |
| `INFERENCE_FLASH_NEXT_PREFILL_ROUTE_TRACE` | unset | `1`/`true`/`on` logs union-admit chunk resident experts and router picks. |
| `INFERENCE_FLASH_NEXT_PREFILL_EXACT_GROUPS` | unset | `1`/`true`/`on` groups consecutive rows while their expert union fits the slot budget. |
| `INFERENCE_FLASH_NEXT_PREFILL_WAVE_LAYER` | unset | Zero-based layer or `all` to run expert waves during multi-row prefill. |
| `FIFI_FLASH_NEXT_GGUF` | unset (off) | `1` opts into experimental Flash-Next GGUF (`qwen4exp`) on CUDA (one token per step). |
| `INFERENCE_FLASH_NEXT_GGUF_SERVE` | unset (off) | `1`/`true`/`on` selects the experimental staged GGUF serving path. |
| `FIFI_FLASH_GGUF_QSA_CACHE` | `f16` | QSA cache storage for Flash-Next GGUF decode (`f16`, or `f32` up to 16 rows/step). |
| `INFERENCE_FLASH_NEXT_NGRAM_BULK` | on | `0` forces serial n-gram row reading instead of the 32-thread pool. |
| `INFERENCE_FLASH_NEXT_PLE_DEVICE` | unset | `1` projects Flash-Next PLE embeddings on CUDA when `prefill_rows` > 1. |
| `INFERENCE_FLASH_NEXT_PLE_BATCH` | unset | `1` batches the two PLE host projections per chunk (chunked prefill only). |
| `INFERENCE_EXL3_ROWS_MAX` | `2048` | Raises max shared-prefill rows up to `4096` (unqualified). |
| `INFERENCE_TUNER` | `auto` | `off` prints `mode off (...)` beside in-use tuner values without applying changes. |
| `INFERENCE_FLASH_NEXT_RING_BUDGET` | unset (off) | Byte budget for Flash-Next CUDA prefill expert ring (carved from free VRAM). |
| `FIFI_EXL3_GROUP_SPLIT` | `2` | K split (`2`, `4`, `8`) for host-picked routed and grouped shared-expert EXL3 projections. |
| `FIFI_EXL3_SINGLE_GATHER` | `wide1` | Kernel for single-row EXL3 projections (`off`, `1`, `wide1`, `wide2`, `wide4`). |
| `FIFI_EXL3_DP4A` | on | `0` disables packed byte dot product for EXL3 codebook 2. |
| `FIFI_EXL3_INDEXED_PAIRS` | `20` | Max expert pairs per routed-expert call (`20` or `16`). |
| `FIFI_EXL3_JOBS_REG` | on | `0` runs EXL3 prefill job-table projections on shared-memory kernels. |
| `FIFI_EXL3_JOBS_WARP_ROTATE` | off | `1` rotates job-table input rows with one warp per 128-value group. |
| `FIFI_EXL3_JOBS_PAIR_ROTATE` | off | `1` rotates gate/up input rows of routed experts in one pass. |
| `FIFI_EXL3_CPU_ISA` | `auto` | CPU ISA for host EXL3 experts (`auto`, `avx512`, `avx2`, `scalar`). |

## SenseNova-Loop, Flint, and Nemotron 3.5 diagnostics

| Variable | Default | Meaning |
| --- | --- | --- |
| `FIFI_SENSENOVA_DECODE_GEMM` | unset | `dq` runs SenseNova-Loop CUDA decode projections through the F32 dequantizing kernel. |
| `FIFI_SENSENOVA_PREFILL_GEMM` | unset | `dq` runs Q4_K and Q6_K prefill projections through the dequantizing matrix product. |
| `FIFI_FLINT_MLX_REQUEST_PREFILL` | off | `1` sizes prefill attention workspace from admitted prompt span. |
| `FIFI_FLINT_MLX_PREFILL_PAIR` | off | `1`, `auto`, or `waves64` shares expert reads across a fresh two-piece Flint prompt. |
| `FIFI_FLINT_MLX_CANCEL_PREFILL_PAIR` | off | `1` stops an admitted Flint pair between layers on client cancel. |
| `FIFI_FLINT_MLX_COMPACT_KV` | off | `1` starts full-attention KV planes at 256 rows (or first prefill capacity) and doubles on demand. |
| `FIFI_FLINT_MLX_REQUEST_KV` | off | `1` sizes initial full-attention KV from admitted prompt and generation limit. |
| `FIFI_FLINT_MLX_CACHE_LIMIT` | `67108864` | MLX freed-buffer cache in bytes (`0` disables). |
| `FIFI_FLINT_MLX_ASYNC_CHAIN` | on | Evaluates Flint decode work in 4-layer chains. |
| `FIFI_FLINT_MLX_COALESCE_RESIDUAL` | on | Merges residual updates of a decode step. |
| `FIFI_FLINT_MLX_DECODE_ATTN` | on | Uses fused MLX attention kernel for one-token decoding. |
| `FIFI_FLINT_MLX_EXPERT_PIPELINE` | off | Submits shared expert to Metal before CPU reads routed expert slices. |
| `FIFI_FLINT_MLX_EXPERT_READ_RUNS` | off | `1` or `prefill` combines consecutive expert slices into one read. |
| `FIFI_FLINT_EXPERT_TIMING` | off | `1` prints `flint_expert_timing` JSON per disk-backed expert forward (qualification builds). |
| `FIFI_FLINT_EXPERT_TRACE` | off | `1` prints `flint_expert_access` JSON per disk-backed expert stage (qualification builds). |
| `FIFI_FLINT_MLX_RING_VIEW` | on | Reads full-attention KV cache as one ring view instead of concatenating each step. |
| `FIFI_FLINT_MLX_COMPILE_LAYER` | off | Runs Flint layer tail through one compiled MLX closure. |
| `FIFI_FLINT_MLX_EXPERT_CACHE_MIB` | off | Reserves per-layer unified-memory slots for packed experts in one-row disk-backed forwards. |
| `FIFI_FLINT_MLX_PREFILL_CHAIN` | off | Chained multi-row prefill groups (graph-ordered, fence-free). |
| `NEM35_CHUNK_EXPERTS` | unset | `gpu` streams routed experts to CUDA on large prompt chunks. |
| `NEM35_CUDA_CHUNK_PREFILL` | `0` | Row group size when `NEM35_CHUNK_EXPERTS=gpu`. |
| `NEM35_CUDA_GROUPED_EXPERTS` | off | Grouped NVFP4 GEMM for resident expert batches. |
| `NEM35_NO_BATCH` | off | Kill switch for the legacy GPU batch expert kernel. |
| `NEM35_NO_F32_CACHE` | off | Bypasses the F32 dense cache (per-call upload). |
| `NEM35_NO_PACKED_CACHE` | off | Bypasses packed-weight device residency (per-call upload). |
| `NEM35_RESIDENT_DENSE` | off | Logs resident dense count and bytes after the first forward. |
| `NEM35_DEVICE_CHAIN` | off | Checks device mamba step count against expected count. |
| `NEM35_DECODED_A_B` | off | Diagnostic: forces decoded route for one measurement run. |
| `NEM35_DIAG_TOP2` | off | Diagnostic: logs top-two logits of every step on the CUDA head. |
| `NEM35_PHASE_TIMING` | off | Times host spans per phase. |
| `NEM35_SUBOP_TIMING` | off | Times sub-operations of the MoE CPU routed path. |
| `NEM35_TOKEN_TIMING` | off | Prints each step's timing split to stderr as it closes. |

## Inkling 2 Mini diagnostics and kernel flags

| Variable | Default | Meaning |
| --- | --- | --- |
| `FIFI_INKLING2_MINI_GGUF_MLX` | on | `0` disables GGUF serving on Apple Silicon native MLX. |
| `FIFI_INKLING_ATTN_CPU_WORKERS` | `4` | Attention projection CPU worker count on Apple Silicon. |
| `FIFI_INKLING_PREFILL_ROWS` | `1` | Prompt prefill rows per step on Apple Silicon safetensors (`1` or `4`). |
| `FIFI_INKLING_MMAP_DENSE` | `1` | `0` disables mmap for dense weights on Apple Silicon benchmarks. |
| `FIFI_INKLING_FILE_CACHE_ONLY` | `1` | `0` disables file-cache-only staging on Apple Silicon benchmarks. |
| `FIFI_INKLING_COMPONENT_TIMING` | off | `1` logs CPU, Metal, and logical dense I/O phase counters. |
| `FIFI_INKLING_ROUTE_TRACE` | off | `1` logs ordered expert selections, plane sizes, and cache counters per visit. |
| `FIFI_INKLING_GGUF_SEGMENTED_LRU` | on | `0` switches Apple Silicon GGUF Metal expert cache from segmented LRU to plain LRU. |
| `FIFI_INKLING_GGUF_PACKED_DOT` | off | `1` selects direct Q6_K and Q8_0 row dots for mapped Apple Silicon GGUF weights. |
| `FIFI_INKLING_GGUF_Q8_ACTIVATIONS` | off | `1` quantizes inputs to mapped Q8_0 projections (requires AArch64 dot product). |
| `FIFI_INKLING_CUDA_PROFILE` | off | `1` logs per-token projection, upload, and expert-cache counters on CUDA. |
| `FIFI_INKLING_CUDA_PACKED_GGUF` | on | `0` disables packed GGUF CUDA projections and keeps BF16 dense matrices resident. |
| `FIFI_INKLING_CUDA_STREAM_EXPERTS` | on | `0` disables pinned staging of cold BF16 and NVFP4 expert pairs. |
| `FIFI_INKLING_CUDA_CACHE_EXPERTS` | on | `0` disables adaptive VRAM caching for routed expert pairs. |
| `FIFI_INKLING_CUDA_EXPERT_CACHE_RESERVE_MIB` | 3/4 of budget | Fixed adaptive expert cache quota in MiB (`0` places whole layers first). |
| `FIFI_INKLING_CUDA_BATCH_EXPERTS` | on | `0` disables batching of resident selected experts per layer. |
| `FIFI_INKLING_CUDA_BATCH_STREAMED_EXPERTS` | off | `1` batches streamed expert projections on CUDA. |
| `FIFI_INKLING_CUDA_DENSE_STREAM` | on | `0` disables vectorized BF16 controller matrix streaming. |
| `FIFI_INKLING_CUDA_ALLOC_CACHE` | on | `0` disables reuse of freed CUDA device buffers and prompt buffers. |
| `FIFI_INKLING_CUDA_DIRECT_RESIDENT` | on | `0` disables direct CUDA kernel for resident GGUF NVFP4 expert banks. |
| `FIFI_INKLING_CUDA_DECODE_Q8_EXPERTS` | on | `0` restores F32 expert products instead of q8_1 activations during decode. |
| `FIFI_INKLING_CUDA_DECODE_EXACT_Q2_DOWN` | off | `1` uses exact Q2 down projection during decode. |
| `FIFI_INKLING_CUDA_IQ2_GATE` | off | `1` enables IQ2 gate kernel on CUDA. |
| `FIFI_INKLING_CUDA_Q8_ACTIVATIONS` | off | `1` enables Q8 activation quantization on CUDA. |
| `FIFI_INKLING_CUDA_PREFILL_TENSOR_CORE_PROJECTIONS` | on | `0` restores exact kernels instead of tensor-core GEMM for prefill > 8 rows. |
| `FIFI_INKLING_CUDA_PREFILL_TENSOR_CORE_EXPERTS` | on | `0` restores dequantizing GEMM for device prompt prefill experts. |
| `FIFI_INKLING_CUDA_PREFILL_WEIGHT_WINDOW` | `64` | Prefill weight window size on CUDA. |
| `FIFI_INKLING_CUDA_PREFILL_ROUTER_INPUT_REUSE` | on | `0` disables gathering resident expert inputs from batched router input. |
| `FIFI_INKLING_CUDA_PREFILL_Q8_ROW_REUSE` | on | `0` disables paired F32 activation rows for resident Q8_0 attention (SM75+). |
| `FIFI_INKLING_CUDA_PREFILL_Q8_ROW_REUSE_REPORT` | off | `1` logs paired and ordinary Q8_0 dispatch counts. |
| `FIFI_INKLING_CUDA_PREFILL_QUERY_TILES` | on | `0` disables query-parallel relative attention for grouped GGUF prefill (SM75+). |
| `FIFI_INKLING_CUDA_PREFILL_RESIDENT_QUERY` | on | `0` disables keeping Q, relative coefficients, sinks, and gates on CUDA during prefill. |
| `FIFI_INKLING_CUDA_BATCH_CONTROLLERS` | on | `0` disables grouping resident Q, K, and V controller projections. |
| `FIFI_INKLING_CUDA_GROUP_PROJECTIONS` | on | `0` disables grouped attention projections. |
| `FIFI_INKLING_CUDA_GROUP_ATTENTION` | on | `0` disables grouped attention execution. |
| `FIFI_INKLING_CUDA_BATCH_ATTENTION_PROJECTIONS` | on | `0` disables batching attention projections across prompt rows. |
| `FIFI_INKLING_CUDA_BATCH_ROUTER` | on | `0` disables batching resident router projections. |
| `FIFI_INKLING_CUDA_BATCH_RELATIVE_ATTENTION` | on | `0` disables batched relative attention. |
| `FIFI_INKLING_CUDA_DEVICE_EXPERT_PREFILL` | on | `0` disables device expert prefill. |
| `FIFI_INKLING_CUDA_RESIDENT_ATTENTION` | on | `0` disables resident target relative attention on CUDA GGUF. |
| `FIFI_INKLING_CUDA_RESIDENT_ATTENTION_RESERVE` | off | `1` reserves resident attention capacity. |
| `FIFI_INKLING_CUDA_CHAIN_ATTENTION_INPUT` | on | `0` disables keeping controller outputs on CUDA through attention input projections. |
| `FIFI_INKLING_CUDA_CHAIN_ATTENTION_OUTPUT` | on | `0` disables keeping attention output on CUDA through its resident output projection. |
| `FIFI_INKLING_CUDA_GROUPED_CHAIN_ATTENTION` | on | `0` disables fused Q/K/V controller normalization for grouped GGUF CUDA prefill. |
| `FIFI_INKLING_CUDA_CHAIN_LOCAL_DECODE` | on | `0` disables chaining local decode attention projections through one readback. |
| `FIFI_INKLING_CUDA_CONTROLLER_CPU_PARITY` | off | `1` uses host-style FP64-rounded `tanh` and ordered sums in grouped Q/K/V normalization. |
| `FIFI_INKLING_DSPARK_CUDA` | off | `1` runs DSpark draft projections and attention on CUDA. |
| `FIFI_INKLING_DSPARK_CUDA_MARKOV` | off | `1` keeps DSpark Markov matrices in VRAM and runs them on CUDA. |
| `FIFI_INKLING_DSPARK_CUDA_HEAD` | off | `1` runs DSpark draft logit correction and selection on CUDA. |
| `FIFI_INKLING_DSPARK_GROUPED_PREFILL` | off | `1` captures target features during grouped prompt prefill. |
| `FIFI_INKLING_CUDA_GROUPED_VERIFY` | off | `1` checks multiple DSpark draft rows per visit on CUDA. |
| `FIFI_INKLING_DSPARK_ADAPTIVE_DRAFT` | off | `1` adjusts DSpark draft width dynamically after rejections or accepts. |

## Test harnesses

`cargo test` reads these, not `enginesrv`. Tests needing recorded data check the environment variable first, then `~/.config/fifi/test-paths.json`. Missing locations skip unless `FIFI_TEST_REQUIRE_DATA=1` or the test is `#[ignore]`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `FIFI_EXL3_SWEEP_LAYERS` | `12` | Layers of six experts cycled in `flint_shape_rate_latency`. |
| `M3_LAYOUT_DIR` | unset | Frozen layout directory for EXL3-vs-GGUF quality comparison (`m3_compare`). |
| `M3_COMPARE` | unset | `P1:<candidate>:<reference>` or `O1:<candidate>:<reference>` for `m3_compare`. |
| `M3_REGISTRATION` | unset (refused) | Registration version (`v6` or `v7`) for `m3_compare` (see `qualification/o1-gate-v7/`). |
