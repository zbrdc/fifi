# Experimental chunked P2P prefill

The `Q36_P2P_CHUNK=<rows>` setting is an experimental, default-off path
for batching prompt processing across the P2P cards. Unset or zero uses the
existing per-token path. A positive row count is capped by the peer ring's
maximum payload. A malformed setting is rejected at startup.

The FP32 peer payload limit permits up to 2,048 rows at hidden width 2,048.
This larger chunk is experimental and can require more VRAM; use 1,024 rows
to return to the earlier limit. FP16 ring exchanges retain their previous
payload limit.

The chunk path is not qualified for production use. Its arena is admitted
only when each card can retain the configured free-VRAM floor; a failed
allocation or construction falls back to the eager path. Prefix reuse is
disabled while chunking is active until its numeric path and turn-chain
behavior have been qualified. The default eager path and its prefix handling
are unchanged.

For Q8 K/V, each card can attach an FP16 attention staging buffer after all
base chunk arenas are allocated. The buffer uses that card's K/V head count
and the same context-based size rule as single-device fixed staging. It is
admitted only if the card can retain its free-VRAM floor; otherwise that card
uses the existing tiled Q8 attention. Staging uses segmented attention beyond
its capacity when the checked partial-buffer plan fits. An allocation race can
also fall back to tiled attention; other CUDA failures refuse the chunk arena.
Tiled attention performs a blocking length update for each query tile.
Neither staged path has yet been qualified. A served comparison with staging
admitted on both physical P2P cards is recorded below.

When this experimental path is active, the card executor advertises the
admitted chunk size to the serving worker. With chunking off, the worker keeps
its existing fallback. The chunk driver leaves one row to the eager path, and
the session keeps one pending token across pieces. Large pieces need physical
device correctness, memory-admission, cancellation and long-context checks;
they are not yet qualified on this port.

## Design

Each card builds a chunk arena and model using its own mixer geometry and
expert-intermediate slices. The embedding and routing run on card 0 and are
broadcast to the others. The cards sum their partial mixer outputs before the
feed-forward normalization, and sum their partial expert outputs before the
next layer. The mailbox is sized to carry the larger of a chunk sum and an
eager decode exchange. Requests larger than the selected chunk size are
split into bounded pieces; a failed piece poisons the session. Each of the
three consecutive routing broadcasts waits for every card to consume it
before the next broadcast. This protects mailbox reuse and has an unmeasured
performance cost.

Both cards compute the selected experts over their intermediate slices. The
existing eager K/V gathers are retained. A physical peer test must check both
single- and multiple-chunk output against per-token execution, using the
intended devices selected through `CUDA_VISIBLE_DEVICES`.

## Measured 512-row comparison

On a 12 GB RTX 3060 and 10 GB RTX 3080, with the Q4_K_M GGUF model, q8 KV,
mapped host experts, and a configured 262,144-token limit, two matched
comparisons used the same experimental staged server binary with
`Q36_P2P_CHUNK=256` and `Q36_P2P_CHUNK=512`. In the 256-first run, median
server job-start-to-first-token rate on three cold 4,274-token prompts per
setting was 304.07 and 471.69 tokens/s. With the order reversed, it was
303.98 and 472.96 tokens/s. Median decode rates for three 128-token replies
per setting were 38.49 and 38.57 tokens/s in the first run, and 38.58 and
38.57 tokens/s in the second. First-token timing includes generating that
token. The 512-row
setting had 1,368 MiB free on the 3060 and 1,475 MiB on the 3080 after
warmup, 32 MiB less per card than the 256-row setting in this run.

The [256-first](../scripts/benchmarks/qwen36-p2p-512-chunk-ab.json) and
[512-first](../scripts/benchmarks/qwen36-p2p-512-chunk-crossover.json)
samples and guarded-run evidence identify the separately tested integration
commit. A [physical-peer test](../scripts/benchmarks/qwen36-p2p-512-parity.json)
also checks a synthetic two-layer fixture against per-token execution at 512
rows. The small fixture compares only the final winner and top-two logits.
Both arms disable prefix reuse, and neither the configured 256k limit nor a
successful short response establishes long-context recall or coding quality.
The setting remains experimental and default off.

## Evidence needed

An earlier, different 3060+3080 prototype produced one short cold sample
of about 106 prompt tokens/s and an eager sample of about 40 tokens/s. Those
observations are not measurements of this port and do not establish a
repeatable speedup or long-context behavior.

Before this path is proposed for use, build the exact commit with native
CUDA, pass the host and physical-peer full and segmented staging tests, and
compare repeated cold served prompts against the eager path with identical
weights and settings.
Record prompt and output token counts, prefix reuse, prefill and decode
rates, VRAM usage, device identities, and failures. Separately validate
128k and near-256k prompts with reserved output and marker recall; synthetic
recall alone is not evidence of coding quality.

## Experimental 1,024-row comparison

On the same RTX 3060 + RTX 3080 pair, a same-binary comparison used the
multi-block FP32 ring in both arms and changed only `Q36_P2P_CHUNK` from 512 to
1,024. With three samples per arm, the median 4,274-token cold prefill rate
rose from 546.02 to 911.92 tokens/s with the 512-row setting first, and from
546.24 to 914.26 tokens/s in the reverse order. The server interval includes
first-batch generation. The 31-token-prompt decode rate stayed near 39 tokens/s,
and the tested deterministic answers matched. After warmup, the larger chunk
used 66 MiB more per card; minimum sampled free VRAM was 1,302 MiB.

The [measurement record](../scripts/benchmarks/qwen36-p2p-chunk1024-results.json)
contains per-sample results, exact build and test identities, the wide-ring and
chunk-parity checks, and restoration receipts. A separate
[long-context record](../scripts/benchmarks/qwen36-p2p-chunk1024-long-recall.json)
shows one cold synthetic recall request at 131,195 prompt tokens with all
three markers correct and 831.52 tokens/s to the first emitted batch. The
subsequent near-256k request was interrupted when the separate 2080 service
restarted; it has no completion result. The pair service was restored.
Neither short synthetic prompts nor a single recall test qualify coding
quality. The chunk and multi-block ring paths remain experimental and off
by default.
