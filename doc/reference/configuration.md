# Configuration file

`enginesrv --config <file>` reads one JSON file.
Startup refuses unknown keys and values outside the ranges below.
When a model does not fit in memory, startup refuses the load rather than lowering a setting to make it fit.

Example files are in [`engine/config`](../../engine/config/README.md).

## Top level

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `listen` | yes | | Address and port, for example `127.0.0.1:18791`. The address must be `127.0.0.1` or `::1`. Port 8788 is refused. |
| `api_key_env` | yes | | Name of the environment variable that holds the API key. Startup fails if the variable is unset or empty. |
| `prefix_reuse` | no | unset | Keep finished conversation prefixes so that a follow-up turn only processes its new tokens. Unset resolves per family at startup: on for the two qualified reuse paths (Flash-Next CUDA, Qwen3.6 CUDA on the CUDA build) and off for every other family, because the boundary marking alone splits the prefill and changes greedy text. An explicit `true` or `false` overrides for all engines. Setting the Flash-Next environment switch to `0` does not revert this key's marking. See [How conversation reuse works](../explanation/prefix-reuse.md). |
| `web_dir` | no | | Directory with the built web frontend (`npm run build` in `web/` writes `web/dist`). When set, `GET /` serves its `index.html` and `GET /assets/...` its hashed assets; startup refuses a directory without `index.html`. The shell is public like `/health`; every API route keeps its key. |
| `engines` | yes | | The engines to serve, one object each. |

## Engine entry

| Key | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Engine name. Clients can select the engine with it (`x-engine`). |
| `family` | yes | The model family; see [Models](models.md). |
| `model_dir` | yes | Directory of the checkpoint. It must be on local storage; see [Local storage](#local-storage). |
| `tokenizer_dir` | yes | Directory with `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and `generation_config.json`. For Qwen3.6, Flint and Nemotron, startup checks each file against the hashes of the tested release. |
| `model` | no | The name clients send in the `model` field. Defaults to the family name. |
| `qwen36` | no | [Qwen3.6 options](#qwen36). |
| `flint` | CUDA only | [Flint options](#flint). |
| `nemotron` | `nemotron35` only, required | [Nemotron options](#nemotron). |
| `fake` | no | [Test engine options](#fake). |
| `context_limit` | no | `fake` only: context limit in tokens. Default 4,096. |
| `flash_next` | no | Options of `qwen3_8_flash_next`, including [MLX prefill admission](#flash-next-mlx-prefill). The family is not validated end to end. |
| `think_replay` | no | `false` drops the reasoning that clients resend with earlier assistant turns from the prompt, for this engine only. `true` is refused; leave the key out for the default. |
| `notice_toolwrap` | no | `false` renders system messages that follow an assistant turn as plain system text, for this engine only. `true` is refused. See `ENGINESRV_NOTICE_TOOLWRAP`. |

### Local storage

Model weights must be on local storage; network storage is not a tier or a fallback. The server refuses to start an engine whose `model_dir` is on a network filesystem (NFS, SMB, AFP, WebDAV, sshfs, Ceph and similar), or on an automount that has mounted nothing. The loaders also refuse a shard file that is a symlink into such a mount, because a symlink counts as where it points. The error reads `model weights must be on local storage, but <path> is on a <type> filesystem`. If the filesystem cannot be determined, the error reads `cannot confirm that model weights at <path> is on local storage` and the engine does not start. The tokenizer directory holds only small text files and is not checked.

### CUDA FP16 prefill attention

FP16 prefill attention retains the existing PTX-based launch path by default. Set [`FIFI_CUDA_ATTN_PREFILL_ADAPT=1`](environment.md#cuda) before engine startup to opt into selection for the current CUDA device and model geometry. The adaptive path uses the loaded kernel's PTX version, the device's per-block shared-memory limit and the device and kernel thread limits. It selects the tensor-core arm when it fits and its loaded code supports it, otherwise the compact SIMT arm when that fits. If neither fits, or a device query or shared-memory opt-in fails, the launch returns an error. Successful adaptive selections are cached per device and geometry.

This selection applies to the FP16 attention part kernel, not every model kernel or the q8 attention path. It does not establish correctness or performance for every GPU and model. Device qualification of the selection across RTX 20, 30, 40 and 50 series remains open. The build must also contain compatible device code; see [`INFERENCE_CUDA_ARCHS`](environment.md#build).

## Flash-Next MLX prefill

An optional positive integer, `flash_next.prefill_budget_bytes`, gives the engine a total byte budget for batched prompt processing. Omit the key to serve one token at a time. With a budget, startup selects the largest fitting chunk up to `flash_next.prefill_chunk` (default 16, range 1–2,048). A worker piece below four rows or insufficient room for a four-row chunk selects the one-token path. The budget applies to chunk admission after model load. This setting is specific to MLX.

The startup line gives the selected path and the bytes charged. The planner reserves resident weights, the decode pool, the QSA cache at the full 32,768-position limit, fixed recurrent state, the configured MLX allocator cache and a 1 GiB pressure margin. With prefix reuse enabled, it also reserves full-sized snapshots for the prefix store's slots by default. Local shard reads have a zero S3 cache charge. The remaining bytes determine which chunk workspace fits.

`FIFI_FLASH_NEXT_MLX_PREFILL_WAVE_CAP` bounds how many routed experts each MLX prefill wave can hold. It accepts an integer from 1 to 512 at startup and defaults to 512. The same cap determines both the workspace charge and the staging pool size. A lower cap can admit a wider chunk within the device budget. Layers whose selected experts exceed the cap run multiple waves and preserve the router's original pair order. Startup logs the wave cap and the selected chunk's workspace charge in bytes.

With prefix reuse enabled, set `FIFI_FLASH_NEXT_MLX_PREFIX_BUDGET_BYTES` to a positive byte count to bound the retained snapshot store. For example, `268435456` sets a 256 MiB cap. Startup charges the smaller of this cap and the default snapshot reservation. A snapshot that exceeds the cap is skipped and previously retained snapshots remain available. The planner continues to charge live state, restore overlap and the allocator cache separately.

`FIFI_FLASH_NEXT_MLX_DECODE_CACHE_SLOTS` sets the number of retained expert records per layer during sampled decode. It defaults to zero and accepts up to four slots, bounded by the model's experts selected per token. A positive setting requires automatic MLX sizing or an explicit `FIFI_MLX_WORKING_SET_MIB` ceiling. Startup checks the page-rounded cache mapping and its metadata against the device budget. The mapping is released at the decode boundary before the next prefill. Only demand reads populate it, after an expert is selected twice in a row. Cached slices are copied into the existing rank-ordered staging set.

## `qwen36`

Options for `qwen3_6_35b_a3b`. "MLX" means Apple Silicon.

For Qwen3.6 MoE or Qwen3.8 dense on Apple Silicon, the matching [disk-staging switch](environment.md#apple-silicon-device-sizing) and an active Metal working-set limit allow an explicit `context_limit` with a host-derived `prefill_budget_bytes`. If the entire checkpoint exceeds the host budget, startup checks resident weights, staging and session memory against that budget and the Metal limit. With an omitted context on a single-sequence start that would exceed the resident host budget, the planner reads the checkpoint headers and sizes context from the resident weights, staging, and session reserve. It preserves the largest prefill width that fits at 4,096 tokens, then grows context within the host and Metal limits in 1,024-token steps. An explicit context or session cap keeps its configured value.

| Key | Backend | Default | Meaning |
| --- | --- | --- | --- |
| `context_limit` | both | MLX 262,144; CUDA 32,768 | Context limit in tokens, from 2. |
| `kv_dtype` | both | MLX `q8`; CUDA `f32` | KV cache format: `q8` (8-bit), `f32`, or `bf16` (MLX only). On CUDA, `f32` stores the cache in FP16. |
| `activation_dtype` | MLX | `bf16` | Decode activations: `bf16` or `f32`. CUDA accepts the key and ignores it. |
| `cache_limit_bytes` | MLX | none | MLX allocator cache limit. Falls back to `Q36_CACHE_LIMIT`; on MLX one of the two is required. |
| `session_cap_bytes` | MLX | 4 GiB | Memory for the live session and saved prefixes. |
| `prefill_budget_bytes` | MLX | off | Memory budget for processing the prompt in chunks. Without it, the prompt is processed one token per step. |
| `prefill_session_reserve_bytes` | MLX | derived | Memory that prompt processing leaves free. With `prefill_budget_bytes` and no reserve member, it is derived: the attention cache at `context_limit`, plus the DeltaNet state, plus `cache_limit_bytes`, plus a 1 GiB pressure margin (the shipped Mac profile's composition). An explicit value always wins. Refused without `prefill_budget_bytes`. |
| `prefill_chunk` | MLX | 256 | Largest prompt chunk, 1 to 2,048 rows. The engine picks the largest chunk that fits the budget. |
| `compile` | MLX | `false` | Compile the decode step with `mlx_compile`: 9.6 to 13% faster decoding. Logits differ from the uncompiled path by at most 5e-5. |
| `compile_program_limit` | MLX | 8 | Budget of distinct compiled program keys per adapter, from 1. Increasing it is opt-in and does not enable `compile`. CUDA refuses an explicit value. |
| `max_seqs` | MLX | 1 | Sequences decoded at once, 1 to 8. Above 1 it needs `compile: true`, a `prefill_chunk` of at least 4, and a `prefill_budget_bytes` larger than both `prefill_session_reserve_bytes` and `session_cap_bytes`. |
| `prefix_store_entries` | MLX | 6 | Saved prefixes kept with `prefix_reuse` on, 1 to 4,096. |
| `weight_format` | CUDA | `gguf` | The format of the checkpoint in `model_dir`: `gguf` (the pinned Q4_K_M file when present, otherwise the directory's largest `.gguf` of the family, validated at open; `Q36_ALLOW_UNPINNED_GGUF=0` serves only the pinned file) or `exl3` (an EXL3 pack directory with `config.json` and the safetensors shards). Any other value, including a different case, is refused. See [below](#exl3-pack-cuda). The MLX arm refuses `exl3`. |
| `gguf_sha256` | CUDA GGUF | off | Expected SHA-256 of the selected GGUF file, exactly 64 hex digits. Startup always verifies this pin before opening the model. MLX and EXL3 refuse the key. See [content pinning](#gguf-content-pinning-cuda). |
| `capture` | CUDA | `true` for `exl3`, `false` for `gguf` | Run each decode step as a CUDA graph. |
| `cuda_devices` | CUDA | `[0]` | The CUDA devices this engine spans, as device numbers in strictly ascending order, at most four. The MLX arm refuses the key. A list other than `[0]` is probed at startup (the devices must exist, share one compute capability, and directly access each other's memory) and serves the whole model across the cards, GGUF checkpoints only, with `tiered`, `kv_offload_host_bytes`, `capture` and speculation off. |
| `kv_offload_host_bytes` | CUDA | off | Keep the KV cache in host memory, up to this many bytes. See [below](#host-memory-kv-cache-cuda). |
| `tiered` | CUDA | off | Keep only some experts on the GPU. See [below](#tiered-experts-cuda). |
| `draft_mode` | CUDA, `exl3` | `disabled` | Where speculative drafts come from: `disabled`, `ngram` (the tokens that followed the latest earlier occurrence of the context's tail) or `mtp` (the model's own MTP head, which needs `mtp_model_dir`; not usable yet). Any other value is refused. |
| `num_draft_tokens` | CUDA, `exl3` | 2 | The most drafts one verify window carries, 1 to 16. |
| `dynamic_draft_tokens` | CUDA, `exl3` | `false` | Shorten each window from the measured chance that its drafts are accepted. Reads the MTP head's confidence, so `ngram` ignores it. |
| `draft_confidence` | CUDA, `exl3` | 0.4 | With `dynamic_draft_tokens`: a draft is dropped when the product of the acceptance chances of the drafts so far falls below this value, above 0 and below 1. |
| `draft_max_rows` | CUDA, `exl3` | 8 | The most rows one window feeds, the pending token included, 2 to 64. It caps `num_draft_tokens` and `hybrid_draft_tokens`. An enabled `draft_mode` refuses more than 8. The default `eager` window feeds one row at a time through the one-row step and uses no multi-row kernel; the opt-in `krow` window (see `draft_verify`) uses the rows kernel for windows of 2 to 8 rows, and from 9 rows the EXL3 projections leave the rows kernel, which is why the cap stays 8. 8 is a cap, not a qualification: whether windows of 2 to 8 rows reproduce one-row decode bit for bit is still to be shown on a device. |
| `hybrid_draft_tokens` | CUDA, `exl3` | 0 (off) | When the context repeats an earlier passage for at least this many tokens, draft that many tokens of the passage in one window. 0, or from `ngram_match_min` to 256. |
| `ngram_match_min` | CUDA, `exl3` | 2 | The shortest repeated tail that an n-gram draft needs, 1 to 64. |
| `mtp_model_dir` | CUDA, `exl3` | none | The pack directory that holds the `mtp.*` tensors. Required by `draft_mode` `mtp`, refused otherwise. |
| `draft_rollback` | CUDA, `exl3` | `host` | How a verify window is undone after a partial accept. `host` copies the whole committed state to the host before the window, writes it back and feeds the kept rows again, so a window costs time proportional to the context. `device` keeps one copy of the DeltaNet state per row but the last in device memory (65,863,680 bytes each, so 62.8 MiB at 2 rows and 439.7 MiB at 8) and rolls the KV cache back by its length; it needs an enabled `draft_mode` and is refused with `kv_offload_host_bytes`. With `device`, start is refused when the slots cannot be allocated or do not fit the memory reserve; there is no fallback to `host`. Any other value is refused. `device` is opt-in and has not been run on a device yet. |
| `draft_verify` | CUDA, `exl3` | `eager` | How a verify window runs its rows. `eager` feeds the rows through the one-row step, one after the other. `krow` is the opt-in K-row window: the rows of a window run as one launch list (or one captured graph per window length) with one synchronization, the DeltaNet states of the rows but the last go straight into the device rollback slots, and the routed experts run one chain per row. The logits of the rows stay on the device: the argmax, the comparison of each row's token with the next draft (the accepted run and the token after it) and a top-20 record of each row are made there, and the host makes one small read of them per window (about 1.5 KB at 8 rows, instead of 8 x 993,280 bytes of logits), re-derives the accepted run from the tokens and fails the window closed if the device disagrees. A greedy draw is answered by the token, a seeded draw by the row's top-20 record (the same selection and the same host draw as the one-row step, so the same token and the same generator state), and a sampler that keeps no limit or more than 20 candidates reads the one row it needs from the device. With `Q36_CUDA_DEVICE_SHORTLIST=0` the top-20 records are not made and every seeded draw reads its row from the device (one row per draw); the greedy path and the accepted run are the same. It runs on the fp16 KV cache, and it is not qualified: the device report (the accepted run, the token after it and the top-20 records) and the draws made from it, as well as the window itself, have not passed all the D1 checks on a device. `krow` needs `draft_rollback` `device` and the EXL3 pack, and start refuses it when `FIFI_EXL3_SINGLE_GATHER` is set to any value (an explicit `wide1` included: unset it) or `FIFI_EXL3_SHARED_IN_ROUTED=1`. A `krow` session runs the reference shared-expert block (the merged block, where it is the default, yields). A one-row window runs the one-row step as before. Any other value is refused. |

CUDA refuses `prefill_budget_bytes` and `prefill_session_reserve_bytes`.
It ignores `cache_limit_bytes`, `session_cap_bytes`, `prefill_chunk`, `compile` and `prefix_store_entries`, and logs one line for each.
MLX ignores `capture` with a log line, and refuses `tiered`.
The `draft_*`, `num_draft_tokens`, `hybrid_draft_tokens`, `ngram_match_min` and `mtp_model_dir` keys are checked at start. A `draft_mode` other than `disabled` is refused on MLX and with GGUF weights, because nothing there can run a window. (EXL3 weights already refuse `kv_offload_host_bytes`, the one KV cache a window cannot export.) `mtp` is refused with EXL3 weights because its draft step does not exist. `ngram` works with EXL3 weights, and the output is the same as with `disabled` for any sampler. A window feeds its rows one at a time and first copies the whole committed session state to the host, so each window costs time proportional to the context and `ngram` is slower than `disabled`: use it to check output, not to serve long contexts.
`prefix_store_entries` is ignored with a log line when `prefix_reuse` is off.

Before saving a new host-backed pageable prefix (the opt-in pageable path; the default pinned-prefix path is unchanged), the CUDA engine checks live host and process-cgroup memory. The check covers the committed KV and recurrent state on every participating card, the largest temporary copy buffer, and 1 GiB of headroom. If that space is unavailable, the engine skips the optional save with one logged line (what was needed, what was available, that the request is unaffected) and keeps the already-cached prefixes available for later requests; if no budget figure can be read at all, the save proceeds as before with one logged line. Reusing an existing checkpoint requires no new allocation and passes without a probe.

### GGUF content pinning (CUDA)

Set `qwen36.gguf_sha256` to a trusted full-file SHA-256 to bind startup to specific checkpoint bytes.
The value accepts uppercase or lowercase hex digits.
A mismatch refuses startup before model parsing or weight upload.
This explicit pin takes precedence over `Q36_VERIFY_GGUF_SHA256`, so startup hashes the file once.
Without the key, the existing filename, size and optional built-in hash checks remain unchanged.

The key also works for an alternate checkpoint filename.
It does not enable unpinned admission or change which file startup selects.
The pinned filename still wins when present, and its size check still applies.
A matching digest establishes content identity, not model quality or source authenticity.
Keep the checkpoint files unchanged while the engine serves them.

### EXL3 pack (CUDA)

`"weight_format": "exl3"` loads the weights from an EXL3 pack in `model_dir` instead of the GGUF file, and serves from it.
Startup admits a pack whose `config.json` names format 1.4.x, 1.5.x or 1.6.x, or exactly version 0.0.29 (the 4.00 bits-per-weight Qwen3.6-35B-A3B pack, which is the one run end to end). Any other version is refused.
Every matrix carries one `mul1` or `mcg` codebook marker, and the whole pack uses one codebook. The routed experts of a projection share one rate.
The tokenizer still comes from `tokenizer_dir`: its files must be the tested release, which the pack's own `tokenizer.json` is not. The pack's file has the same vocabulary and merges and lacks seven unused audio special tokens.
The pack must be Qwen3.6-35B-A3B: startup checks its `config.json` against the model, and the headers of every tensor against the shapes and dtypes the model needs. It refuses a mismatch before the first upload.
All layers, the embedding table and the head go to the GPU: about 14.7 GB for the 4.00 bpw pack, loaded in about 23 s. With `"kv_dtype": "q8"` a 262,144-token context peaked at 21,558 MiB on a 24 GB card.
`exl3` refuses `tiered` and `kv_offload_host_bytes`. `capture` defaults to true for an `exl3` entry (false for GGUF): a captured EXL3 step reads the picked experts on the device (an id-indexed launch), so the graph holds no host read; it supports at most 8 picks per token and a graph that cannot be captured falls back to the eager step. `"capture": false` rolls back to the eager step, and `FIFI_EXL3_SINGLE_GATHER=off` to the shuffle kernel for the single-row projections (an `exl3` entry otherwise runs the wide kernel, see [environment.md](environment.md)); with capture on, `FIFI_EXL3_SINGLE_GATHER=off` alone leaves the routed-expert launches on the id-indexed wide kernel, so set `"capture": false` as well for a full rollback. `INFERENCE_CUDA_SYNC_EACH=1` cannot run with capture and refuses the start of an `exl3` entry; set `"capture": false` with it. `Q36_VERIFY_GGUF_SHA256` has no effect: the pack has no pinned size or hash, and startup identifies it by its `config.json` and by the shape of every matrix.
A pack prefills the prompt in chunks, as the GGUF path does: with `kv_dtype: q8` of up to 1,024 rows, and with the default `f32` cache (FP16 storage) of up to 2,048 rows for the first 3,072 positions and 1,024 after that. `Q36_CUDA_CHUNK_PREFILL=0` rolls either back to one token per step (see [environment.md](environment.md)). A chunk runs every dense projection as a job table of up to 256 rows per job and the experts of a layer as job tables of up to 64 rows per job, on tensor cores with fp16-rounded operands (2.1e-4 relative L2 per projection against fp64); on a device of compute capability 8 or more they run on a kernel that decodes the trellis into the tensor-core operands in registers, and `FIFI_EXL3_JOBS_REG=0` rolls that back to the shared-memory kernels with identical results (see [environment.md](environment.md)); a chunk does not use the single-row projection, so `FIFI_EXL3_SINGLE_GATHER` does not change it (the decode of the last prompt token, a one-row tail and the per-token fallback still honour it), and `FIFI_EXL3_LEGACY`, which selects the Flint midrows path only, has no effect on a Qwen3.6 chunk. Against the per-token path of the same pack, the next-token distribution after a 5,222-token prompt differs by 9.8e-5 nats (KL), and over a 32-token teacher-forced continuation of two prompts the worst step differs by 1.4e-2 nats, at a step that copies the prompt text. In those 66 observed steps the greedy token differs only where the two best logits are within 0.013 of each other (4 steps). The output for a prompt can depend on the chunk size: the same 5,213-token prompt gave different 32-token texts with 1,024-row chunks (context 8,192) and with 512-row chunks (context 262,144). Chunked prefill of an EXL3 pack is not quality-qualified: the registered copy and retrieval quality probes have not been run on it, and `Q36_CUDA_CHUNK_PREFILL=0` rolls back to one token per step. On the per-token path a 5,222-token prompt took 77 s to the first token, against 1.05 s with the GGUF file on the same machine (log `q36x1-gpu-srv1`, sha256 prefix `25c00a15`). Decoding ran at 68.6 tokens/s at an 8,192-token context.
A prefix saved after a chunked prefill and restored from the device tip or the pinned host copy was compared with the cold chunked prefill by the device test `a_restored_prefix_matches_the_cold_chunked_prefill_on_the_real_pack` (3,300-token prompt, 8 cases, the first token and 32 teacher-forced continuation steps; run q36pf-r2, test binary sha8 0de96bbe, built from the tree of commit 1c34c3ea7 plus documentation, per-case printout in the run's receipt q36pf-r2-raw.log). A restore at the 1,024-row chunk boundary reproduced the cold rows bit for bit; at the cuts 1,280 and 2,500 the largest per-step KL divergence was 4.7e-3 nats. One greedy flip occurred, in the two cases that restore at cut 2,500 with a chunked tail (device tip and host copy), at step 12 where the cold run's two best logits were 0.025 apart. The test asserts the exact match at a chunk boundary, a first-token KL below 1e-3, a KL below 5e-2 at every step, and that a greedy flip happens only where the cold run's top-two logit gap is below 0.1. The saved state carries no marker of the chunk partition that built it.

### Host-memory KV cache (CUDA)

`kv_offload_host_bytes` keeps the KV cache in host memory for contexts that do not fit on the GPU.
It needs `"kv_dtype": "q8"`, `"capture": false`, `"prefix_reuse": false` and `Q36_CUDA_CHUNK_PREFILL=0`; startup refuses the combination without `prefix_reuse` false, so on the CUDA build an omitted `prefix_reuse` key (which defaults on for Qwen3.6) also refuses `kv_offload_host_bytes` until the key is set to false explicitly.
It reloads the cache from host memory for every token, so it is slow: an 8K-token replay decoded at 34.95 tokens/s, against 147.14 tokens/s with the cache on the GPU.
`Q36_CUDA_PAGED_DEVICE_FIRST=1` keeps as much of the cache on the GPU as fits after a 1 GiB reserve.

### Tiered experts (CUDA)

`tiered` keeps a fixed number of expert slots per layer on the GPU and runs the other experts on the CPU from the memory-mapped model file.
Do not modify or replace the model file while the server is running.

| Key | Default | Meaning |
| --- | --- | --- |
| `slots` | required | An integer from 8 to 256 requests an exact count. `"auto"` selects the largest whole number that fits every layer after weights, KV cache and step allocations, leaving 1 GiB of free VRAM. Startup reports the selected count and refuses a load that cannot hold one token’s routed experts. |
| `threads` | 6 | CPU worker threads, 1 to 8. |
| `timeout_ms` | 10,000 | Longest wait for one layer's CPU experts, 1 to 600,000. |
| `preload` | `false` | `true` fills each layer's slots with its lowest-numbered experts before the first token. By default the slots start empty and fill as experts are used. |

Tiered mode refuses `capture: true` and `kv_dtype: q8`.
It processes prompts one token per step and, by default, keeps no saved prefixes in host memory.

## `flint`

Optional for Flint on CUDA; refused on Apple Silicon. Without it the engine uses `context_limit` 131,072 and `kv_profile` `"f16"`. When you set it, give both `context_limit` and `kv_profile`.

| Key | Required | Values | Meaning |
| --- | --- | --- | --- |
| `context_limit` | with `flint` | 1 to 262,144 | Context limit in tokens. On a 24 GB card, up to about 60,000 tokens with `f16` and the token-major walk keep every expert on the GPU (the default job-table prefill reserves about 0.65 GB more, so the context that keeps every expert resident is shorter; not measured); a longer context keeps part of them in about 18 GiB of page-locked host memory (see `FIFI_FLINT_CUDA_EXPERT_CACHE` in environment.md). |
| `kv_profile` | with `flint` | `"f16"`, `"q8"`, `"q8k"` | KV cache format. `q8` (int8 in groups of 32 with an FP16 scale) is experimental and materially lossier than f16; see `FIFI_FLINT_CUDA_Q8_TILED` in environment.md. `q8k` quantizes only the K plane and keeps V at f16; it is opt-in with no quality verdict, uses the default prefill enqueue (the q8 tiled route serves uniform `q8` only), and refuses prefix reuse like `q8`. |
| `prefill_chunk_size` | no | 1 to 8, default 8 | Prompt rows per step. `1` processes one token at a time. Any value other than 8 keeps the token-major walk with that chunk instead of the default job-table prefill, which ignores the chunk size (`FIFI_EXL3_LAYER_MAJOR`). |
| `cpu_moe_offload_layers` | no | whole number, default 0 (off); 999 or more means every routed layer | Runs every routed expert of the first N MoE layers on the CPU, from the page-locked host store, instead of on the GPU. Those layers keep no experts on the GPU, so the GPU memory they would have used goes to the other layers' experts or to a longer context. Needs the host store: it turns the expert cache on, and is refused with `FIFI_FLINT_CUDA_EXPERT_CACHE=0` and with an explicit `FIFI_EXL3_LAYER_MAJOR=1` or `jobs` (with the switch unset the engine keeps the token-major walk for CPU experts and says so at start). Exclusive with `cpu_moe_split_experts`. See the trade-off below. |
| `cpu_moe_split_experts` | no | 0 to 255, default 0 (off) | Keeps `256` minus this many experts per layer on the GPU and computes the other experts a token picks on the CPU instead of copying them in. A slow hot-set rule moves an expert into a GPU slot only when it is used clearly more than the least used resident expert. Sets the experts per layer on the GPU, so it is refused together with `FIFI_FLINT_CUDA_EXPERT_SLOTS`; the same refusals as `cpu_moe_offload_layers` apply. Exclusive with `cpu_moe_offload_layers`. |
| `cpu_moe_threads` | no | 0 to 256, default 0 (6) | Threads for the CPU experts, the engine thread included. The CPU expert kernel is limited by memory bandwidth; on a Ryzen 9 7900X 6 threads already read 45 GB/s, and more did not help. |

CPU experts trade decode speed for GPU memory and for PCIe traffic. On an RTX 3090 with the EXL3 4.0 pack, a 4,096-token context, greedy decoding, one run each: with 128 experts per layer on the GPU (6.9 GiB less GPU memory than 224), serving the other picks from the host buffer by copying them in decoded 36.0 tokens/s after a 51.9 s first token and copied 339 GB in the 4,000-token prompt, while `cpu_moe_split_experts` 128 decoded 32.9 tokens/s after a 56.9 s first token and copied nothing. `cpu_moe_offload_layers` 8 freed 2.7 GiB and decoded 37.7 tokens/s against 39.8 with all layers on the GPU (224 experts per layer). The CPU path rounds its intermediate values to bf16 where the GPU kernels publish them, but it does not round each weight to FP16 as they do, so a greedy run can diverge from the GPU run at a close call between two tokens. Prefill with CPU experts is slower because each row-expert pair is computed on its own.

## `flash_next`

The `flash_next` object of one engine entry for `qwen3_8_flash_next`. Every key is optional. The
keys below are the CUDA prefill keys; they need a `native-cuda` Linux build. That build serves on the
card unless the entry sets `cuda_decode: false` (the MLX path, where the CUDA keys are refused).

| Key | Default | Meaning |
| --- | --- | --- |
| `cuda_decode` | `true` in a `native-cuda` Linux build unless `cache_limit_bytes` or `prefill_budget_bytes` is set, `false` elsewhere | Serves the model on a CUDA card. The CUDA decoder is experimental. |
| `prefill_chunk` | 16 | Rows per chunked-prefill piece, 1 to 2,048. A piece wider than `prefill_rows` is streamed in `prefill_rows`-row device chunks (four per 1,024-row piece at the default `prefill_rows` of 256). Wide pieces need the prefill ring (`INFERENCE_FLASH_NEXT_RING_BUDGET`) to be useful. Single measured runs on the RTX 3090 EXL3 pack, a 28,916-token prompt, ring on, `prefill_rows` 256 (the default): prefill took 229 s at `prefill_chunk` 256, 221 s at 1,024 and 219 s at 2,048 - the chunk size barely moves it while the device step stays 256 rows. |
| `prefill_rows` | 256 | Rows of one device prefill step, 1 to 2,048 (the shared supported width; the unset default is 256, and `1` selects the per-token prefill). Measured single runs, RTX 3090 EXL3 pack, the same 28,916-token prompt, ring on (budget 1.3e9), `prefill_chunk` 2,048, no measurement override set: 256 rows prefill 234.9 s / 3,390 GB streamed / 804 misses / 25.08 decode tok/s; 1,024 rows 184.1 s / 943 GB / 861 misses / 23.52 decode tok/s; 2,048 rows 180.1 s / 540 GB / 921 misses / 23.16 decode tok/s (all unwrapped). An earlier 1,024-row run under the nsys profiler measured 162 s prefill and 25.98 decode tok/s - profiler-wrapped walls are comparable only to other wrapped runs; the unwrapped figures above are the served numbers. The ring held 658 slots (1.15 GiB) at 256 rows and 690 (1.21 GiB) at the wider windows. The wider window's arena is card memory the expert bank gives up: resident experts per layer fell 183 (256 rows) to 165 (1,024) to 141 (2,048), and decode pays for it - 25.08 tokens per second at 256 rows against 23.52 at 1,024 and 23.16 at 2,048. A window the card cannot carry next to the trunk refuses at load with the losing budget's byte counts. |
| `context` | from the memory plan | K/V positions, a multiple of 4 from 4 to 262,144. Unset, the start picks the longest context, up to the model maximum, at which the expert bank keeps nine tenths of the slots it has at 4,096, and prints the choice and the slots on one line. A value set here is used as given: the load refuses a context the card cannot hold next to the trunk. |
| `kv_format` | `f16` | K/V storage on the card. `int8` has no attention kernel yet and is refused. |

## `nemotron`

Required for `nemotron35`. `cache_entries` may be omitted; the other three keys are required.

| Key | Values | Meaning |
| --- | --- | --- |
| `backend` | `"cuda"`, `"cpu_reference"` | `cuda` needs a Linux build with `--features native-cuda` and accepts greedy decoding only. `cpu_reference` runs on the CPU and is slow. |
| `context_limit` | 1 or more | Context limit in tokens. |
| `cache_entries` | 1 or more, or omitted | Expert weight tensors kept decoded in host memory. Admission charges each entry as the largest entry's payload bytes plus the loader's per-entry index, 96 bytes per entry on 64-bit targets. Omitting the key fits the count from 850 permille of the measured host budget (the cgroup-aware figure on Linux) minus the fixed serving charges; a remainder below one entry refuses the load, and one log line states the chosen count. An explicit number is honoured verbatim: a card too small for it refuses the load - omit the key, because only the fit trims the 12 GiB device request down to what the card holds. |
| `prefill_chunk` | 1 to `context_limit` | Prompt tokens per step. |

## `fake`

The `fake` family replies with fixed text and loads no model. It is for tests and for trying the server.

| Key | Default | Meaning |
| --- | --- | --- |
| `reply` | `Hello from the fake engine.` | The text of every reply. |
| `token_delay_ms` | 0 | Pause before each token, in milliseconds. |
