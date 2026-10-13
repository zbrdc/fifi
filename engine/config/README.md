# Engine configs

`qwen36-mac-studio-36gb.json` is the `enginesrv` config for the Qwen3.6-35B-A3B stepping stone on the 36 GB Mac Studio.
It states `activation_dtype: "bf16"` (the default).
It also states `kv_dtype: "q8"`, `context_limit: 262144`, and `session_cap_bytes: 4294967296`, which are the defaults of the MLX arm since 2026-10-13 (see "KV cache dtype" and "Numbers").
The chunk path of the prefill is dtype aware (slice Q36-BF16-b) and the budget and reserve members below are what its planner needs, and they stay.
BF16 PREFILL POLICY: the file leaves `prefill_chunk` absent, and the engine default of `prefill_chunk` is 256 for a BF16 engine and for an `f32` engine (`BF16_DEFAULT_PREFILL_CHUNK` in `engine/src/family/qwen36/settings.rs`).
Slice Q36-BF16-b had set the BF16 default to 1 (no chunked prefill) while the finding Q36-BF16-SENS was open. Decision 0046 A6 item 11 (amendment of 2026-10-11) lifted that policy.
The finding: on the 922-token tool prompt the BF16 format-token probability depends on the chunk size, in fifi and in the reference mlx-lm (mlx-lm: 0.72 to 0.9999 over 13 chunk sizes, and the top token never changes).
The serving shapes measure inside the mlx-lm range. Chunk 256 gives 0.9969 (token by token: 0.9994). The tails of 8, 20, 40, 60, 100, 150 and 300 tokens after a 256-token head give 0.9968, 0.9962, 0.9970, 0.9788, 0.9558, 0.9788 and 0.9986. The format token is the top token in each case.
Smaller chunks of 4 to 100 rows give lower values on this prompt. The planner picks them only when the budget leaves no room for 256 rows. The broad tool-call eval (Q36-BF16-EVAL) confirms or reverses the default.
To run BF16 prompts one token at a time, set `prefill_chunk` to 1. A chunk of 1 plans no chunk, and the start-up line says so.

## Activation dtype

The `qwen36.activation_dtype` member (packet Q36-BF16-a, decision 0046 A6) selects the decode
activation dtype: `bf16` or `f32`. The default is `bf16`, and this file states it explicitly.
The member is one lever: `bf16` keeps the stored BF16 scales and biases and converts the layer
norms, `f32` keeps the F32 widening of Q36-G0, and the engine refuses any other pairing at load.
The DeltaNet state, the router, and the logits stay F32 in both modes. The key-value cache is `q8` on the MLX arm unless `kv_dtype` is `f32` (next section).
The chunked prefill runs in either dtype (slice Q36-BF16-b): the chunk blocks cast at the same
boundaries as the decode step, a dtype guard runs after each norm, after each layer, and on every
state tensor, and a `bf16` engine plans its chunk exactly like an `f32` engine when the budget
allows one.
With `bf16` the resident weights are 19,286,549,004 bytes against 21,420,001,804 with `f32`.

## KV cache dtype

The `qwen36.kv_dtype` member (decision 0046 A7) selects the form of the attention cache: `f32` or `q8`.
The default is `q8` on the MLX arm and `f32` on the CUDA arm. The user decided the MLX default on 2026-10-13 after the Mac gate of record Q36-KV8-GATE. The shipped file states `q8` as well.
The `f32` form stays the control on the MLX arm: set `"kv_dtype": "f32"` with an explicit `context_limit` and `session_cap_bytes` for it, because the MLX defaults below belong to the `q8` form.
The `q8` form keeps 8-bit affine codes with one scale and one bias per group of 64 lanes.
The scales and biases have the activation dtype.
On the real model (2 key-value heads of width 256, 10 attention layers), one position takes 40,960 bytes in the `f32` form.
It takes 10,880 bytes in the `q8` form with `bf16` activations and 11,520 bytes with `f32` activations.
The `q8` form needs a head dimension that is a multiple of 64.
Only the MLX arm builds the `q8` form so far. The CUDA arm refuses it and keeps `f32` and 32,768 positions by default.
The MLX arm defaults to `context_limit` 262,144 and `session_cap_bytes` 4,294,967,296 (4 GiB). In the `q8` form with `bf16` activations, 262,144 positions take 2,852,126,720 bytes, and with the DeltaNet state 2,917,990,400 bytes, which the cap admits.
The session admission and the prefix-reuse requirement use the bytes of the chosen form. The reserve below is a plan sum for the `q8` form with `bf16` activations at 262,144 positions: recompute it when one of these changes.
The gate of record Q36-KV8-GATE (in the project ledger) shows the fit on the 36 GB Mac Studio, the retrieval of a needle at 2,048 to 262,000 tokens, the tool-call format, and the window divergence against the `f32` cache, which stays within the schedule noise of the `bf16` prefill. The decode speed at depth is the known price of the composite attention until the fused kernel lands: 42.8 tok/s against 55.0 tok/s of the `f32` cache at 30,000 positions, and about 10 tok/s at 262,000 positions.

## Numbers

The budget 32,212,254,720 bytes is 30 GiB.
It is the provisional planning budget of decisions 0044 and 0046.
It is the planner input, not a measured fit and not a hard memory cap.

The reserve 8,286,699,520 bytes is a plan sum, not a measurement.
It is 2,852,126,720 (KV cache at 262,144 positions of 10,880 bytes, 8-bit with `bf16` scales) plus 65,863,680 (DeltaNet state).
It adds 4,294,967,296 (the cache limit of this file) and 1,073,741,824 (the pressure margin of decision 0044 item 6).
The sum is valid only for `context_limit` 262,144, `kv_dtype` `q8`, `activation_dtype` `bf16`, and `cache_limit_bytes` 4,294,967,296.
Recompute the reserve whenever one of them changes. The `f32` form at 32,768 positions gave 6,776,750,080.
The reserve has no derived default.

## Prefix reuse

With `prefix_reuse` on in the configuration file, one saved prefix entry and the live session share the engine's `session_cap_bytes` budget: the entry charges its attention caches plus the DeltaNet state (65,863,680 bytes) and the live session can hold the same caches again up to `context_limit`.
The engine start therefore refuses a run whose `session_cap_bytes` is below `2 * growth_per_position * context_limit + 65863680` and answers with the cap, the requirement, and the hint `raise session_cap_bytes`.
For the `f32` cache form the growth is 40,960 bytes per position (2,750,218,240 bytes at 32,768 positions). For the default `q8` form with `bf16` activations it is 10,880 bytes per position: 5,770,117,120 bytes at 262,144 positions, more than the default cap of 4 GiB.
A run with `prefix_reuse` on and the default limit must therefore set `session_cap_bytes` of at least 5,770,117,120.
A smaller cap would let a later job fail with `MemoryPlan` although the same job succeeds with `prefix_reuse` off.
The rule is `Q36Settings::check_prefix_reuse_cap` in `engine/src/family/qwen36/settings.rs`; the shipped JSON does not turn `prefix_reuse` on.

## Evidence

The E10/E11 numbers are F32 measurements, taken before `activation_dtype` existed: one X32 smoke trial on the 36 GB Mac Studio admitted 32,010 tokens at chunk 256 (decision 0046 item 5c and A4), with swap flat against a baseline of about 1.0 GB already in use.
It is one trial and not a bound, and it predates `activation_dtype`; the chunk numbers of the shipped `bf16` entry are to be re-measured by the Mac gates of slice Q36-BF16-b (the E8 chunk gates and the prompt-speed test, both dtypes).

## Use

The API key comes from the variable `ENGINESRV_API_KEY`. The file holds no secret.
The server listens on `127.0.0.1:18791`. The port must never be 8788.
The paths `model_dir` and `tokenizer_dir` in the file are placeholders. Set both to the directory of your Qwen3.6-35B-A3B checkpoint before you start the server.
A Q36 result never satisfies Phase 0B or Phase 1B.
