# Experimental multi-block P2P ring

The peer ring normally publishes and reduces each payload with one CUDA
block. Setting `INFERENCE_CUDA_P2P_RING_MULTIBLOCK=1` before starting the
engine allows larger f32 publishes and f32/FP16 reductions to use multiple blocks.
Payloads of 2,048 values or fewer retain the single-block path. FP16 publishes
also retain one block: their overflow flag lives in peer memory, and not all
P2P links support remote atomics. Only the exact value `1` enables the
experiment; unset the variable and restart to roll back.

The switch is off by default and is not qualified. It does not implement
chunked P2P prefill by itself. Before enabling it for normal serving, compare
the opt-in and default paths on the exact build with two physical peer-enabled
GPUs. Cover f32 and f16 payloads, tails, overflow and timeout status, repeated
mailbox reuse, and complete model logits and replies. Measure prefill and
decode separately using the same binary and configuration, and retain the raw
samples and process/device identities.

An earlier device log and a failed attempt are in the [observation
record](../scripts/benchmarks/p2p-ring-multiblock-observations.json). They do
not identify the exact binaries or supply a matched baseline, so they do not
establish a speedup or qualify this implementation.

## RTX 3060 + RTX 3080 observation

A guarded, same-binary comparison on Qwen3.6 35B A3B Q4_K_M with 512-row
chunks measured the existing opt-in against the default. For a cold 4,274-token
synthetic prompt, median server prefill rates were 474.09 versus 547.93 tokens/s
with the control first, and 472.28 versus 547.68 tokens/s with the candidate
first (three samples per arm and order). The measured interval ends with the
first emitted batch, so it includes first-token generation. The short decode
case changed by less than 1% in both orders. Deterministic answer hashes matched.

The [measurement record](../scripts/benchmarks/qwen36-p2p-ring-performance-results.json)
contains per-sample rates, exact build and device identities, peer correctness
checks, an instrumented launch-grid observation, and restoration receipts. It is
a short synthetic observation, not a 128k/256k coding or recall qualification;
prefix reuse was disabled. The switch remains off by default.

A separate cold-recall run with the same experimental setting completed at
8,312, 32,882 and 131,194 server-counted prompt tokens. All three markers were
recalled at each length; the 131,194-token case reached 493.27 tokens/s to the
first batch. The [recall receipt](../scripts/benchmarks/qwen36-p2p-ring-long-recall.json)
records the device guard, source identities and individual results. Each length
was tested once with a synthetic prompt; this does not qualify multi-turn coding,
prefix reuse or the full 256k context.

A further separate run counted 262,008 prompt tokens, reserved 64 output
positions, and finished normally after 21 output tokens; all three markers were
correct. Its first-batch rate was 442.75 tokens/s. The [near-256k receipt](../scripts/benchmarks/qwen36-p2p-ring-near-256k-recall.json)
records the count, output budget, result and guard. This is one cold synthetic
request rather than a multi-turn coding qualification.

A separate two-turn synthetic coding check passed both the initial Python edit
and its revision with 131,157 and 131,206 prompt tokens respectively. Prefix
reuse was disabled; both turns reported zero reused tokens and each took about
282.2 seconds to the first response batch. The [coding-session receipt](../scripts/benchmarks/qwen36-p2p-ring-coding-session.json)
records both turns and the guard. One small task does not establish general
coding quality or warm-prefix correctness.

The same two-turn coding check also passed near the context limit: the initial
edit and revision counted 261,696 and 261,740 prompt tokens, and each returned
11 tokens with a normal stop. Both turns reused zero tokens and took about
633 seconds to their first batch. The [near-256k coding receipt](../scripts/benchmarks/qwen36-p2p-ring-coding-256-session.json)
records the output budgets, results and guard. This is a single synthetic
session; it does not qualify warm reuse or general coding quality.
