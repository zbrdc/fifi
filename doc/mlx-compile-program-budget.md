# MLX compiled-program budget

`qwen36.compile_program_limit` is an optional positive integer for the MLX
Qwen3.6 and dense Qwen3.8 resident adapter. It defaults to **8**. For example:

```json
{"qwen36": {"compile": true, "compile_program_limit": 12}}
```

The budget counts distinct sets of baked constants, not layers, tapes, or bytes.
Identical keys deduplicate; repeats still succeed when the registry is full. A
new key is refused when the number already admitted equals the budget. Attention
sets need two programs (pre-RoPE and post-core). Supported batch specializations
can produce multiple tapes per key.

The loader configures the native adapter registry before importing the decoder
or warming programs. Both the whole-model load plan and batch-row preparation
read that adapter's budget. Models requiring more programs fail at load before
warmup, rather than at a later layer. Direct adapter users retain default 8.
The registry is bounded, never evicts, and dies with its adapter; configuring a
budget after the first admission is refused rather than clearing its history.

Zero, values exceeding the addressable `Vec<u64>` capacity, invalid JSON integer
values, and any explicit CUDA use are errors. The option does not enable compile
or change any other defaults. A larger budget permits more compiled closures;
it is not a device-memory guarantee. Admission does not qualify a new model or
budget on hardware. No larger-budget Mac hardware qualification is claimed.

Portable regressions cover the registry, whole-model planning, option validation,
effective settings, and a source check of native configuration order.
The executable settings-to-rules test requires the Mac build.
Native type checking and Metal execution remain separate physical checks.
