# Configuration file

`enginesrv --config <file>` reads one JSON file.
Startup refuses unknown keys and values outside the ranges below.
When a model does not fit in memory, startup refuses the load rather than lowering a setting to make it fit.

Example files are in [`engine/config`](../../engine/config/README.md).

## Top level

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `listen` | yes | | Address and port, for example `127.0.0.1:18791`. The address must be `127.0.0.1` or `::1`. Startup refuses port 8788. |
| `api_key_env` | yes | | Name of the environment variable that holds the API key. Startup fails if the variable is unset or empty. |
| `prefix_reuse` | no | unset | Keeps finished conversation prefixes so a follow-up turn processes only new tokens. When unset, startup picks per family: on for Flash-Next, Qwen3.6, and Bolt on CUDA. On Apple Silicon, it turns on for Qwen3.6 when host memory holds the model, session reserve, and one saved prefix at the start context. It defaults off for every other family. Saving a prefix splits prefill in two, which can change greedy text via floating-point sum order. `true` or `false` overrides all engines. See [How conversation reuse works](../explanation/prefix-reuse.md). |
| `web_dir` | no | | Directory with the built web frontend (`npm run build` in `web/` writes `web/dist`). When set, `GET /` serves `index.html` and `GET /assets/...` serves hashed assets. Startup refuses a directory without `index.html`. The web shell is public like `/health`. Every API route requires its key. |
| `engines` | yes | | The engines to serve, one object each. |

## Engine entry

| Key | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Engine name. Clients can select the engine with it (`x-engine`). |
| `family` | yes | The model family. See [Models](models.md). |
| `model_dir` | yes | Directory of the checkpoint. It must be on local storage. See [Local storage](#local-storage). |
| `tokenizer_dir` | yes | Directory with `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and `generation_config.json`. For Qwen3.6, Flint and Nemotron, startup checks each file against the hashes of the tested release. |
| `model` | no | The name clients send in the `model` field. Defaults to the family name. |
| `qwen36` | no | [Qwen3.6 options](#qwen36). |
| `nemotron` | `nemotron35` only, required | [Nemotron options](#nemotron). |
| `fake` | no | [Test engine options](#fake). |
| `context_limit` | no | `fake` only: context limit in tokens. Default 4,096. |
| `flash_next` | no | Options of `qwen3_8_flash_next`, including [MLX prefill admission](#flash-next-mlx-prefill). Startup does not validate this family end to end. |
| `think_replay` | no | `false` drops reasoning that clients resend with earlier assistant turns from the prompt, for this engine only. Startup refuses `true`. Omit the key for the default. |
| `notice_toolwrap` | no | `false` renders system messages that follow an assistant turn as plain system text, for this engine only. Startup refuses `true`. See `ENGINESRV_NOTICE_TOOLWRAP`. |

### Local storage

Model weights must reside on local storage. Network storage is not a tier or a fallback. The server refuses to start an engine whose `model_dir` is on a network filesystem (NFS, SMB, AFP, WebDAV, sshfs, Ceph and similar), or on an empty automount. Loaders also refuse a shard symlink that points into a network mount. The error reads `model weights must be on local storage, but <path> is on a <type> filesystem`. If the check cannot determine the filesystem, startup logs `cannot confirm that model weights at <path> is on local storage` and stops. Startup does not check the tokenizer directory because it holds only small text files.

### CUDA FP16 prefill attention

FP16 prefill attention keeps the existing PTX-based start path by default. Set [`FIFI_CUDA_ATTN_PREFILL_ADAPT=1`](environment.md#cuda) before engine startup to opt into device-based selection. The adaptive path checks the kernel PTX version, device shared-memory limit, and thread limits. It selects the tensor-core arm when it fits and the code supports it, or the compact SIMT arm when that fits. If neither fits, or if a device query or shared-memory opt-in fails, the launch returns an error. The runtime caches successful selections per device and geometry.

This selection applies to the FP16 attention part kernel, not every model kernel or the q8 attention path. It does not prove correctness or speed for every GPU and model. We tested it on an RTX 2080 Ti and an RTX 3080, not on RTX 40 or 50 series boards. The build must contain compatible device code. See [`INFERENCE_CUDA_ARCHS`](environment.md#build).

## Flash-Next MLX prefill

An optional positive integer, `flash_next.prefill_budget_bytes`, gives the engine a total byte budget for batched prompt processing. Omit the key to serve one token at a time. With a budget, startup selects the largest fitting chunk up to `flash_next.prefill_chunk` (default 16, range 1–2,048). A worker piece below four rows or insufficient room for a four-row chunk selects the one-token path. The budget applies to chunk admission after model load. This setting is specific to MLX.

The startup line gives the selected path and the bytes charged. The planner reserves resident weights, decode pool, QSA cache at 32,768 positions, fixed recurrent state, MLX allocator cache, and a 1 GiB margin. With prefix reuse on, it also reserves full-sized snapshots for prefix store slots. Local shard reads have a zero S3 cache charge. The remaining bytes determine which chunk workspace fits.

`FIFI_FLASH_NEXT_MLX_PREFILL_WAVE_CAP` bounds how many routed experts each MLX prefill wave can hold. It accepts an integer from 1 to 512 at startup and defaults to 512. The same cap determines both the workspace charge and the staging pool size. A lower cap can admit a wider chunk within the device budget. Layers whose selected experts exceed the cap run multiple waves and preserve the router's original pair order. Startup logs the wave cap and the selected chunk's workspace charge in bytes.

With prefix reuse enabled, set `FIFI_FLASH_NEXT_MLX_PREFIX_BUDGET_BYTES` to a positive byte count to bound the retained snapshot store. For example, `268435456` sets a 256 MiB cap. Startup charges the smaller of this cap and the default snapshot reservation. The engine skips a snapshot that exceeds the cap and keeps previously retained snapshots. The planner continues to charge live state, restore overlap, and the allocator cache separately.

`FIFI_FLASH_NEXT_MLX_DECODE_CACHE_SLOTS` sets the number of retained expert records per layer during sampled decode. It defaults to zero and accepts up to four slots, bounded by experts selected per token. A positive setting needs automatic MLX sizing or an explicit `FIFI_MLX_WORKING_SET_MIB` ceiling. Startup checks the page-rounded cache mapping and its metadata against the device budget. The engine releases the mapping at the decode boundary before the next prefill. Only demand reads populate it after an engine picks an expert twice in a row. The engine copies cached slices into the existing rank-ordered staging set.

## `qwen36`

Options for `qwen3_6_35b_a3b`. "MLX" means Apple Silicon.

| Key | Backend | Default | Meaning |
| --- | --- | --- | --- |
| `context_limit` | both | MLX 262,144. CUDA GGUF: largest the free card carries, up to 262,144 | Context limit in tokens, from 2. With no `tiered` or `kv_offload_host_bytes`, a single-card CUDA GGUF start reads free VRAM before upload. It picks the largest multiple of 1,024 whose cache, DeltaNet state, staged rows, and headroom fit beside the weights, then logs it. The engine keeps an explicit value as given. If even the smallest context fails to fit, it keeps 32,768 and takes the tiered rung or load refusal. |
| `kv_dtype` | both | MLX `q8`. CUDA GGUF: chosen from free VRAM | KV cache format: `q8` (8-bit), `f32`, or `bf16` (MLX only). On CUDA, `f32` stores FP16. When omitted on single-card CUDA GGUF, startup picks FP16 if context >= 131,072, else `q8` (which holds nearly double the positions). `q8` changes generated text against FP16. Set `kv_dtype` to fix the format. |
| `activation_dtype` | MLX | `bf16` | Decode activations: `bf16` or `f32`. CUDA accepts the key and ignores it. |
| `cache_limit_bytes` | MLX | none | MLX allocator cache limit. Falls back to `Q36_CACHE_LIMIT`. On MLX, one of the two is required. |
| `session_cap_bytes` | MLX | 4 GiB | Memory for the live session and saved prefixes. |
| `prefill_budget_bytes` | MLX | off | Memory budget for processing the prompt in chunks. Without it, the engine processes the prompt one token per step. |
| `prefill_session_reserve_bytes` | MLX | derived | Memory that prompt processing leaves free. With `prefill_budget_bytes` and no explicit reserve, startup derives it from the attention cache at `context_limit`, DeltaNet state, `cache_limit_bytes`, and a 1 GiB margin. An explicit value always wins. Startup refuses this key without `prefill_budget_bytes`. |
| `prefill_chunk` | MLX | 256 | Largest prompt chunk, 1 to 2,048 rows. The engine picks the largest chunk that fits the budget. |
| `compile` | MLX | `false` | Compile the decode step with `mlx_compile` (9.6 to 13% faster decoding). Logits differ from the uncompiled path by at most 5e-5. |
| `compile_program_limit` | MLX | 8 | Budget of distinct compiled program keys per adapter, from 1. Raising it is opt-in and does not enable `compile`. CUDA refuses an explicit value. |
| `max_seqs` | MLX | 4 on host-derived MoE with compiled decode and prefill budget, else 1 | Sequences decoded at once, 1 to 8. Above 1, it needs `compile: true`, `prefill_chunk` >= 4, and `prefill_budget_bytes` above `prefill_session_reserve_bytes` and `session_cap_bytes`. If the memory plan refuses a derived value, startup falls back to 1 and logs it. Startup never lowers an explicit value. Clients decode together in one window, so greedy text can differ from a solo run. The window meets the batch quality gate (mean KL, worst-row KL, top-1 agreement) against solo runs, not bit equality. |
| `prefix_store_entries` | MLX | 6 | Saved prefixes kept with `prefix_reuse` on, 1 to 4,096. |
| `weight_format` | CUDA | `gguf` | Checkpoint format in `model_dir`. `gguf` serves the pinned Q4_K_M file when present, else the directory's largest valid `.gguf` of the family. `Q36_ALLOW_UNPINNED_GGUF=0` serves only the pinned file. Startup refuses any other value, including `exl3` by name. |
| `gguf_sha256` | CUDA GGUF | off | Expected SHA-256 of the selected GGUF file, exactly 64 hex digits. Startup always checks this pin before opening the model. MLX refuses the key. See [content pinning](#gguf-content-pinning-cuda). |
| `capture` | CUDA | `false` | Run each decode step as a CUDA graph. |
| `cuda_devices` | CUDA | `[0]` | CUDA device indices in strictly ascending order, at most four. MLX refuses this key. A list other than `[0]` probes devices at startup (they must exist, share compute capability, and support peer access) and serves GGUF checkpoints across cards. Multi-card mode forces `tiered`, `kv_offload_host_bytes`, `capture`, and speculation off. |
| `kv_offload_host_bytes` | CUDA | off | Keep the KV cache in host memory, up to this many bytes. See [below](#host-memory-kv-cache-cuda). |
| `kv_cache` | CUDA | on, in the user cache directory | Keeps saved prefixes on local disk so a long conversation survives restarts and pinned-memory eviction. Members: `dir` (absolute path), `quota_bytes` (from 1 GiB), and `enabled` (`false` turns it off). See [below](#prefix-cache-on-disk-cuda). |
| `tiered` | CUDA | off | Keep only some experts on the GPU. See [below](#tiered-experts-cuda). |
| `draft_mode` | MLX | `disabled` | Speculative draft source: `disabled`, `ngram` (tokens after the last match of the context tail), or `mtp` (MTP head using `mtp_model_dir`, not usable yet). Startup refuses any other value. |
| `num_draft_tokens` | MLX | 2 | The most drafts one verify window carries, 1 to 16. |
| `dynamic_draft_tokens` | MLX | `false` | Shorten each window from the measured chance that its drafts succeed. Reads the MTP head's confidence, so `ngram` ignores it. |
| `draft_confidence` | MLX | 0.4 | With `dynamic_draft_tokens`: drops a draft when the cumulative acceptance product falls below this value (between 0 and 1). |
| `draft_max_rows` | MLX | 8 | The most rows one window feeds, including the pending token, 2 to 64. It caps `num_draft_tokens` and `hybrid_draft_tokens`. An enabled `draft_mode` refuses more than 8. |
| `hybrid_draft_tokens` | MLX | 0 (off) | When the context repeats an earlier passage for at least this many tokens, draft that many tokens in one window. 0, or `ngram_match_min` to 256. |
| `ngram_match_min` | MLX | 2 | The shortest repeated tail that an n-gram draft needs, 1 to 64. |
| `mtp_model_dir` | MLX | none | The pack directory that holds the `mtp.*` tensors. Required by `draft_mode` `mtp`. Refused otherwise. |
| `draft_rollback` | MLX | `host` | How a verify window undoes a partial accept: copies committed state to the host, restores it, and feeds the kept rows again. Any other value fails. |
| `draft_verify` | MLX | `eager` | How a verify window runs its rows. `eager` feeds the rows through the one-row step in order. Any other value fails. |

CUDA refuses `prefill_budget_bytes` and `prefill_session_reserve_bytes`.
It ignores `cache_limit_bytes`, `session_cap_bytes`, `prefill_chunk`, `compile` and `prefix_store_entries`, and logs one line for each.
MLX ignores `capture` with a log line, and refuses `tiered`.
Startup checks `draft_*`, `num_draft_tokens`, `hybrid_draft_tokens`, `ngram_match_min`, and `mtp_model_dir`. Setting any of them on CUDA fails with `engine config: Qwen3.6 CUDA speculative verify is removed until the step's many-rows path brings it back; unset the draft options`. On MLX, a `draft_mode` other than `disabled` needs GGUF weights, and `mtp` fails because its draft step does not exist.
Startup ignores `prefix_store_entries` with a log line when `prefix_reuse` is off.

Before saving a host-backed pageable prefix (opt-in pageable path; pinned-prefix path unchanged), the CUDA engine checks live host and cgroup memory. The check covers committed KV and recurrent state on all cards, the largest copy buffer, and 1 GiB headroom. If space lacks, the engine skips the save, logs one line (needed, available, request unaffected), and keeps existing cached prefixes. If no budget can be read, the save proceeds and logs one line. Reusing an existing checkpoint needs no new allocation and passes without a probe.

### Prefix cache on disk (CUDA)

`kv_cache` stores saved prefixes of a CUDA engine in a directory. The next request of a long conversation then reads its state from disk in seconds instead of reprocessing the prompt, even after a restart. This engine-level key works for any family that keeps saved prefixes in host memory (Qwen3.6 GGUF and Bolt). Other families refuse it. You may also set `qwen36.kv_cache` or `gemma4.kv_cache`, but not both.

It is on by default for a single-device Qwen3.6 GGUF engine and for Bolt. It needs the family's host prefix store (`prefix_reuse` and, for Qwen3.6, the pinned prefix pool). When that store is off, on multi-device setups, or when disk space is short, the cache stays off and logs why. The Qwen3.6 cache serves GGUF loads only.

While active, the cache writes served conversations (exact token lists and model state) to `dir`, up to `quota_bytes` (64 GiB max by default). To keep nothing on disk, set `kv_cache.enabled` to `false` or `FIFI_KV_CACHE=0`. `--purge-kv-cache` removes existing files. The start line prints these settings.

```json
"kv_cache": { "dir": "/var/cache/fifi/kv", "quota_bytes": 21474836480 }
```

- `dir`: absolute path. Startup creates it with mode 0700 if missing, and requires an existing directory to be owner-only. One process owns a directory at a time. Without `dir`, it uses `$XDG_CACHE_HOME/fifi/kv/<family>`, or `$HOME/.cache/fifi/kv/<family>` (`qwen36`, `gemma4`), separating families. Two engines of one family need two directories.
- `quota_bytes`: maximum directory size, at least 1 GiB. Without it, startup sets a quarter of free filesystem space plus existing cache bytes (up to 64 GiB). Below 4 GiB, the cache stays off and logs the existing bytes and `--purge-kv-cache`. If `quota_bytes` is below current usage, startup removes the least recently used entries.
- `enabled`: `false` disables the cache regardless of other settings like `FIFI_KV_CACHE_DIR`, unless `FIFI_KV_CACHE=1` overrides it. `true` enables it at the default directory.

Environment variables override the config: `FIFI_KV_CACHE_DIR`, `FIFI_KV_CACHE_QUOTA_BYTES`, `FIFI_KV_CACHE_HOST_BYTES` (host-store prefix bytes before spilling to disk), and `FIFI_KV_CACHE=0` or `1` (except `enabled: false` above). Legacy `Q36_` prefixes also work. A malformed value stops startup. The directory must support direct I/O (not tmpfs or some encrypted/overlay filesystems). Startup writes, syncs, reads back, and removes an 8 MiB scratch file to check and time the disk. If direct I/O fails, it keeps the cache off and logs the reason.

`enginesrv --config <file> --purge-kv-cache` empties the cache directory of every eligible engine in the config before starting.

An entry holds the exact tokens and state of one saved prefix: DeltaNet state and KV rows for Qwen3.6, or KV8 cells of every physical layer for Bolt.

The engine writes to disk only when host memory is short. When available memory drops below the start plan's headroom plus one entry, background threads write the least recently used entries once the store hits its watermark. A host with spare memory writes nothing while serving. If a save needs a slot during a background write, the engine drops and counts that write.

A graceful stop (SIGINT or SIGTERM) writes memory-only tips, newest first. The wait bound is three times the estimated write time plus 10 s, clamped between 15 s and 300 s. The stop line warns when this exceeds systemd's 90 s timeout or Docker's 10 s timeout. A second signal ends the wait at once.

Writes keep 512 MiB free on the filesystem and skip entries under 1,024 positions.

On a host-store miss, the engine reads back an entry with full CRC32C and SHA-256 checks. Each entry binds to its writer's identity: GGUF header digest, tensor samples, file length, mtime, `gguf_sha256` pin, tokenizer/template digests, KV format, and numeric recipe version. A mismatched, truncated, or corrupted entry logs a miss, and the engine processes the prompt normally.

The engine also spills to disk in the background when host-memory prefixes exceed their byte budget, keeping pinned roots last. Written entries stay in memory until eviction. The budget uses the planner's prefix room (Qwen3.6 on Apple Silicon), or machine memory minus the OS floor and resident engine memory. `FIFI_KV_CACHE_HOST_BYTES` overrides this budget, and a malformed value stops startup. The start line prints the budget.

The request log reports `prefix_source` (`device`, `pinned`, or `disk`), disk restore stats (`prefix_disk_bytes`, `prefix_disk_ms`), and session write totals (`prefix_disk_written_entries`, `prefix_disk_written_bytes`, `prefix_disk_dropped`). The start line logs the watermark, and each write logs its reason (`watermark`, `byte budget`, or `stop`). After three consecutive failed writes (full or read-only disk), the cache pauses writes for ten minutes and logs one line.

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

### Host-memory KV cache (CUDA)

`kv_offload_host_bytes` keeps the KV cache in host memory for contexts that do not fit on the GPU.
It needs `"kv_dtype": "q8"`, `"capture": false`, `"prefix_reuse": false`, and `Q36_CUDA_CHUNK_PREFILL=0`. Startup refuses the combination unless `prefix_reuse` is explicitly `false`, because CUDA Qwen3.6 defaults `prefix_reuse` on.
It reloads the cache from host memory for every token, so it is slow. An 8K-token replay decoded at 34.95 tokens/s, against 147.14 tokens/s with the cache on the GPU.
`Q36_CUDA_PAGED_DEVICE_FIRST=1` keeps as much of the cache on the GPU as fits after a 1 GiB reserve.

### Tiered experts (CUDA)

`tiered` keeps a fixed number of expert slots per layer on the GPU and runs the other experts on the CPU from the memory-mapped model file.
Do not change or replace the model file while the server is running.

| Key | Default | Meaning |
| --- | --- | --- |
| `slots` | required | An integer from 8 to 256 sets an exact count. `"auto"` picks the largest count that fits every layer after weights, KV cache, and step allocations, leaving 1 GiB free VRAM. Startup logs the count and refuses if one token's experts fail to fit. |
| `threads` | 6 | CPU worker threads, 1 to 8. |
| `timeout_ms` | 10,000 | Longest wait for one layer's CPU experts, 1 to 600,000. |
| `preload` | `true` | Fills each layer's slots with lowest-numbered experts before the first token so ready starts with a full bank. `false` starts slots empty and fills them on use. |
| `swaps_milli` | 250 | Bank swaps per token, in thousandths, 0 to 8,000. The default 250 buys one promotion per four tokens model-wide, sized for the blocking fill path. |

Tiered mode refuses `capture: true` and `kv_dtype: q8`.
It processes prompts one token per step and, by default, keeps no saved prefixes in host memory.

## `flash_next`

The `flash_next` object of one engine entry for `qwen3_8_flash_next`. Every key is optional. The
keys below are the CUDA prefill keys. They need a `native-cuda` Linux build, which serves on the
card unless `cuda_decode: false` (the MLX path, where CUDA keys fail).

| Key | Default | Meaning |
| --- | --- | --- |
| `cuda_decode` | `true` on `native-cuda` Linux unless `cache_limit_bytes` or `prefill_budget_bytes` is set, else `false` | Serves the model on a CUDA card. The CUDA decoder is experimental. |
| `prefill_chunk` | 16 | Rows per chunked-prefill piece, 1 to 2,048. Pieces wider than `prefill_rows` stream in `prefill_rows`-row chunks (four per 1,024-row piece at default 256). Wide pieces need `INFERENCE_FLASH_NEXT_RING_BUDGET` to help. On an RTX 3090 EXL3 pack (28,916-token prompt, ring on, `prefill_rows` 256), prefill took 229 s at 256, 221 s at 1,024, and 219 s at 2,048. |
| `prefill_rows` | 256 | Rows per device prefill step, 1 to 2,048 (`1` = per-token). On RTX 3090 EXL3 (28,916 tokens, ring 1.3e9, `prefill_chunk` 2,048), unwrapped runs gave: 256 rows = 234.9 s prefill, 3,390 GB streamed, 25.08 tok/s. 1,024 rows = 184.1 s, 943 GB, 23.52 tok/s. 2,048 rows = 180.1 s, 540 GB, 23.16 tok/s. Wider windows take card memory from resident experts (183 to 165 to 141 per layer) and fail load if memory runs short. |
| `context` | from the memory plan | K/V positions, a multiple of 4 from 4 to 262,144. Unset picks the longest context where the expert bank keeps nine tenths of its 4,096-context slots, and logs the choice. Load refuses an explicit value that does not fit. |
| `kv_format` | `f16` | K/V storage on the card. `int8` has no attention kernel yet and fails. |

## `nemotron`

Required for `nemotron35`. You may omit `cache_entries`. The other three keys are required.

| Key | Values | Meaning |
| --- | --- | --- |
| `backend` | `"cuda"`, `"cpu_reference"` | `cuda` needs a Linux build with `--features native-cuda` and accepts greedy decoding only. `cpu_reference` runs on the CPU and is slow. |
| `context_limit` | 1 or more | Context limit in tokens. |
| `cache_entries` | 1 or more, or omitted | Decoded expert tensors in host memory. Admission charges the largest payload plus 96 index bytes per entry (64-bit). Omitting it fits the count from 850 permille of host budget (cgroup-aware on Linux) minus fixed charges, logging the count or refusing if under one entry. Explicit counts use exact values and refuse small cards. Omit to let the fit trim the 12 GiB device request. |
| `prefill_chunk` | 1 to `context_limit` | Prompt tokens per step. |

## `fake`

The `fake` family replies with fixed text and loads no model. It is for tests and for trying the server.

| Key | Default | Meaning |
| --- | --- | --- |
| `reply` | `Hello from the fake engine.` | The text of every reply. |
| `token_delay_ms` | 0 | Pause before each token, in milliseconds. |
