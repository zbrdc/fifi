# Engine configs

`qwen36-mac-studio-36gb.json` is the `enginesrv` config for the Qwen3.6-35B-A3B stepping stone on the 36 GB Mac Studio.
It states `activation_dtype: "bf16"` (the default).
The chunk path of the prefill is dtype aware (slice Q36-BF16-b) and the budget and reserve members below are what its planner needs, and they stay.
TEMPORARY BF16 POLICY: the file leaves `prefill_chunk` absent, and the engine default of `prefill_chunk` is 1 for a BF16 engine (256 for an `f32` engine; `BF16_DEFAULT_PREFILL_CHUNK` in `engine/src/family/qwen36/settings.rs`).
A chunk of 1 plans no chunk: the prompt runs one token at a time, as before slice Q36-BF16-b, and the start-up line says so.
The reason is an open finding (Q36-BF16-SENS): on the 922-token tool prompt the BF16 chunked prefill gives a format-token probability of 0.0426 at chunk 16 and 0.998 at chunk 256 (token by token: 0.9994), and the reference mlx-lm itself varies with the chunk size in BF16 (0.72 to 0.9999 over 13 sizes) but never loses the top token.
To use the chunked prefill with BF16 before the finding is closed, set `prefill_chunk` explicitly (at your own risk for tool calls); the prefill is about 40% faster with the sorted expert products (Q36-PF-SORTED) than before.
Lifting the policy is a one-line change of the constant.

## Activation dtype

The `qwen36.activation_dtype` member (packet Q36-BF16-a, decision 0046 A6) selects the decode
activation dtype: `bf16` or `f32`. The default is `bf16`, and this file states it explicitly.
The member is one lever: `bf16` keeps the stored BF16 scales and biases and converts the layer
norms, `f32` keeps the F32 widening of Q36-G0, and the engine refuses any other pairing at load.
The key-value cache, the DeltaNet state, the router, and the logits stay F32 in both modes.
The chunked prefill runs in either dtype (slice Q36-BF16-b): the chunk blocks cast at the same
boundaries as the decode step, a dtype guard runs after each norm, after each layer, and on every
state tensor, and a `bf16` engine plans its chunk exactly like an `f32` engine when the budget
allows one.
With `bf16` the resident weights are 19,286,549,004 bytes against 21,420,001,804 with `f32`.

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

The E10/E11 numbers are F32 measurements, taken before `activation_dtype` existed: one X32 smoke trial on the 36 GB Mac Studio admitted 32,010 tokens at chunk 256 (decision 0046 item 5c and A4), with swap flat against a baseline of about 1.0 GB already in use.
It is one trial and not a bound, and it predates `activation_dtype`; the chunk numbers of the shipped `bf16` entry are to be re-measured by the Mac gates of slice Q36-BF16-b (the E8 chunk gates and the prompt-speed test, both dtypes).

## Use

The API key comes from the variable `ENGINESRV_API_KEY`. The file holds no secret.
The server listens on `127.0.0.1:18791`. The port must never be 8788.
The paths `model_dir` and `tokenizer_dir` in the file are placeholders. Set both to the directory of your Qwen3.6-35B-A3B checkpoint before you start the server.
A Q36 result never satisfies Phase 0B or Phase 1B.
