# Engine configs

`qwen36-mac-studio-36gb.json` is the `enginesrv` config for the Qwen3.6-35B-A3B stepping stone on the 36 GB Mac Studio.
It turns chunked prefill on.
The file leaves `prefill_chunk` absent, so the default of 256 applies.

## Numbers

The budget 32,212,254,720 bytes is 30 GiB.
It is the provisional planning budget of decisions 0044 and 0046.
It is the planner input, not a measured fit and not a hard memory cap.

The reserve 6,776,750,080 bytes is a plan sum, not a measurement.
It is 1,342,177,280 (KV cache at 32,768 positions of 40,960 bytes, F32) plus 65,863,680 (DeltaNet state).
It adds 4,294,967,296 (the cache limit of this file) and 1,073,741,824 (the pressure margin of decision 0044 item 6).
The sum is valid only for `context_limit` 32,768 and `cache_limit_bytes` 4,294,967,296.
Recompute the reserve whenever one of them changes.
The reserve has no derived default.

## Prefix reuse

With `prefix_reuse` on in the configuration file, one saved prefix entry and the live session share the engine's `session_cap_bytes` budget: the entry charges its attention caches plus the DeltaNet state (65,863,680 bytes) and the live session can hold the same caches again up to `context_limit`.
The engine start therefore refuses a run whose `session_cap_bytes` is below `2 * 40960 * context_limit + 65863680` (2,750,218,240 bytes at 32,768 positions) and answers with the cap, the requirement, and the hint `raise session_cap_bytes`.
A smaller cap would let a later job fail with `MemoryPlan` although the same job succeeds with `prefix_reuse` off.
The rule is `Q36Settings::check_prefix_reuse_cap` in `engine/src/family/qwen36/settings.rs`; the shipped JSON keeps its current numbers, and the coordinator sets `session_cap_bytes` later.

## Evidence

One X32 smoke trial on the 36 GB Mac Studio admitted 32,010 tokens at chunk 256.
Swap stayed flat against a baseline of about 1.0 GB already in use.
It is one trial and not a bound.

## Use

The API key comes from the variable `ENGINESRV_API_KEY`. The file holds no secret.
The server listens on `127.0.0.1:18791`. The port must never be 8788.
The model path is a Mac path. Change `model_dir` and `tokenizer_dir` together on another machine.
A Q36 result never satisfies Phase 0B or Phase 1B.
