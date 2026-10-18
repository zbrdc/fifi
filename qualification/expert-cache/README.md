# Expert-cache replay (S3 packet T0)

This directory holds the offline replay of the T0 route trace and the T0 prompt set.
The plan is `s3-resident-expert-cache.md` (in the ledger), sections 3.3 to 3.8.

| Path | Content |
| --- | --- |
| `src/` | The `expert-cache` replay simulator. It uses only the Rust standard library. |
| `prompts/` | The `trace-prompts` tokenizer command, `source.json`, and `prompts.proposed.json`. |

## Language

The simulator is a Rust command without dependencies, not a Python script.
The plan permits either. Rust was chosen for three reasons:

- The grid replays about 2.3 million cache accesses for each of about 500 settings. A compiled replay finishes in seconds.
- The planned `SlotTable` of packet T2a is Rust. It can reuse the same synthetic cases and hand-derived counts.
- The project prefers Rust for project tools. Python stays a reference language only.

The simulator does not link the engine. It reads the trace files only.

## Inputs

The ignored engine test `resident_route_trace` (`engine/src/family/flash_next/resident/parity.rs`) writes:

- `trace.tsv`, schema `s3-route-trace/1`. Header lines `# key=value` record the engine commit, the index, config, artifact manifest, and prompts SHA-256 values, the cache limit, `layers`, `experts`, `top_k`, `renorm`, and `expert_bytes` (bytes of one expert per layer, from the checkpoint headers). Each row is one layer of one step: `prompt`, `mode` (`p` or `g`), `pos`, `layer`, the ids `e0..`, the raw probabilities `p0..`, and the gates `g0..`.
- The header field `artifact_manifest_sha256` is the SHA-256 of the file named by `FLASH_NEXT_ARTIFACT_MANIFEST` (for example the `SHA256SUMS` of the artifact directory). Without that variable the value is `unset`, and the coordinator fills it from the `SHA256SUMS` of the artifact when it records the run. The shard hashes come from the same `SHA256SUMS` or the environment record. The repository file `docs/environment/flashnext-oq4e-manifest.json` lists Hugging Face metadata of the candidate, not files of the Mac artifact, and does not replace the value.
- `tokens.tsv`: `prompt`, `pos`, `mode`, `input`, `output`, `seconds`, `read_bytes`, `syncs`.
- `meta.txt`: the prompt list, swap before and after, the run time, and the generated tokens of each prompt.
- `probs.bin` only with `FLASH_NEXT_TRACE_FULL=1`.

## Commands

```sh
export CARGO_TARGET_DIR=<task scratch>/target
cargo test --offline --manifest-path qualification/expert-cache/Cargo.toml
cargo build --offline --release --manifest-path qualification/expert-cache/Cargo.toml
"$CARGO_TARGET_DIR/release/expert-cache" self-test
"$CARGO_TARGET_DIR/release/expert-cache" simulate --trace <dir>/trace.tsv --tokens <dir>/tokens.tsv \
  --out <report dir> --t-fixed <Z1c P2 seconds> \
  --bw t0b-qd1=<rate> --bw t0b-best=<rate> --bw warm-7GiB=7GiB --bw warm-13GiB=13GiB \
  --bw-cold t0b-best
```

A rate is bytes per second, with the optional suffix `GiB` or `GB`.
Without `--bw`, the command uses the two warm rates only and cannot evaluate the gates.
`--t-fixed` defaults to 0.149 s, the estimate of plan section 5. Use the measured Z1c P2 value.
`--bw-page-cache` (default `7GiB`) is the rate of a page-cache hit in the second level.
`--planning-slots` defaults to 193. `--repetition-limit` defaults to 0.5.

## Replay model

- Each layer has its own cache of `S` slots. One step selects `top_k` distinct experts. Hits are the selected experts that the cache holds before the step. Every miss enters the cache, because the device reads experts only from slots. A victim is never an expert of the same step.
- The victim is the evictable resident with the smallest key, ties to the lower expert id:
  - `lru`: the step of the last use.
  - `lfu`: the use count since the start of the prompt. Evicted experts keep their counts.
  - `lfu-h64`, `lfu-h256`, `lfu-h1024`: LFU, and every count halves (integer division) after each `H` admitted steps.
  - `hot-h16`, `hot-h64`, `hot-h256`: decayed hotness with an LRU tie-break. Each use of an expert adds 1 to its hotness, and every hotness shrinks by the factor `2^(-1/H)` after each admitted step (`H` is the half-life in steps). The victim is the evictable resident with the least hotness, ties to the less recently used, then to the lower expert id. Evicted experts keep their hotness. The policy has `S` slots in each layer and no core, like `lru`. The replay stores the hotness scaled by `2^(step / H)` and stops with an error when that exceeds the float range (more than 1,023 half-lives).
  - `static`: a core of `S - top_k` experts that never leave, and an LRU tail of `top_k` slots for the misses.
  - `static-half`: a core of `S / 2` experts and an LRU tail of the rest.
  - `belady`: the furthest next use. It is the bound, not a candidate.
  - `pool`: one cache for all layers. It holds `(layer, expert)` items. Its capacity is the sum of the per-layer slots (`S` times the layer count at a uniform size), so it uses the same memory as the per-layer policies. The victim has the largest staleness, the tokens since the last use. Ties go to the layer that comes up last in the layer cycle (`L, L-1, ..., 0, 47, ..., L+1` for the layer `L` in service), then to the lower expert id. A victim is never an expert of the same step. A warm start holds the ranked experts of each layer, all stale since token -1. The decision value `h*` and the greedy split ignore `pool`: it is another cache design. The per-layer Belady does not bound it.
  - `page-only`: no resident cache. The macOS page cache is modeled as one LRU over the expert byte ranges of the file (one unit per expert, the nine role ranges together; page rounding and the operating system's real eviction rule are not modeled). Its capacity in bytes is the sum over the layers of the slots times the expert bytes, so it uses the same memory as the other policies at the same size. Every selected expert is read through it, and prompt steps fill it too. A hit costs a page-cache copy and a miss costs an SSD read, so `page_cache_bytes_mean` holds the hit bytes and `ssd_bytes_mean` the miss bytes, and `miss_bytes_mean` is every byte read. Compare its `tok_s_*` columns with the resident policies, not its hit rate. It runs with an empty start and admission, without a second level, and `belady-pool` bounds it. It stays out of `h*`.
  - `belady-pool`: the bound of `pool`. It uses the same shared items and capacity. The victim is the item with the furthest next use in time `token * layers + layer`, or never. It is the bound, not a candidate. The check `t0_p3_belady_bound` compares `pool` with `belady-pool` and every other policy with `belady`.
- Each prompt replays alone from its start state. The ranking that sets the static core and the warm start comes from the other prompts (leave one prompt out).
- Start states: `empty` (a static core is still loaded) and `warm` (all slots filled from the ranking; LRU recency follows the rank, LFU counts start at 0).
- Admission: `admit` (prompt steps update the cache) and `bypass` (prompt steps count their hits and change nothing).
- Second level: an LRU page cache of 0, 2, or 4 GiB of whole experts, shared by all layers, in front of the SSD. Only empty starts with admission use it.
- Sizes: 10 to 50 percent of the experts in steps of 5 (floor), and the planning size.
- Greedy split (`split=greedy`): at the planning bytes, each layer starts at `top_k` slots, and each next slot goes to the layer with the largest training gain per byte. The LRU gain comes from the training stack distances of generated steps. The static gain comes from the training counts.
- Read time per generated token: `page-cache bytes / --bw-page-cache + SSD bytes / bw` for each named rate. Tokens per second: `1 / (t_fixed + read time)`.

## Outputs

| File | Content |
| --- | --- |
| `results.tsv` | One row per policy, split, size, start, admission, page-cache size, and scope (each prompt, `all`, and `decision`). Columns: generated-step hit rate, prompt-step hit rate, hit rates of the first 16, 64, and 256 generated steps, misses per generated token (mean, p50, p95, max), miss bytes per token, page-cache and SSD bytes per token, and the read time and tokens per second at each rate. |
| `layers.tsv` | Generated-step hit rate of each layer, scope `decision`, without a page cache. |
| `reuse.tsv` | LRU stack-distance histogram of generated-step accesses, with `cold` for first uses. |
| `distinct.tsv` | Distinct experts per prompt and layer, over all steps and over generated steps. |
| `checks.txt` | T0-P1 (parse checks), T0-P2 (coverage), the repetition share of each prompt, and T0-P3 (self-test, Belady bound, LRU curve from stack distances equal to the direct replay at 3 sizes). |
| `summary.txt` | The decision prompts, the capacity share, the lines `plan_hit_<start>_<policy>_<admit|bypass>` (the decision hit rate of every policy at the planning size), `h*` for each start with its policy, the split gains (Q2 needs 2 percentage points), and T0-G1 and T0-G2 at `--bw-cold`. |

The scope `decision` leaves out prompts whose generated 4-gram repetition passes the limit.
The capacity share is `planning slots / experts` (0.377 at 193 of 512).
The plan text quotes 0.379, the share of the unrounded 9,303 experts.

## Self-test (T0-P3)

`expert-cache self-test` and `cargo test` run the same cases with hand-derived values:

- a pool of 3 slots over 2 layers: 0, 0, 1, 1 hits per token; the experts of the step stay; a warm pool; a bypassed prompt;
- decayed hotness: 6 hits against 5 for LRU; 9 hits against 3 for LFU; a warm tie-break; the overflow stop;
- the page-cache baseline over two layers of 100 and 300 bytes: 0, 1, 1, 1 hits per step and the bytes of each step;
- the same 3 slots under `belady-pool`: 0, 0, 2, 1 hits per token, 3 in all, against 2 for the per-layer Belady;
- a cycle of 4 experts: LRU with 3 slots scores 0, with 4 slots 8; Belady with 3 slots scores 6;
- two alternating experts: 0 hits with 1 slot, 8 with 2;
- a frequent expert between fresh experts: LRU 0, LFU 3, Belady 4;
- LFU decay with a half-life of 2: 3 hits without decay and 9 with it;
- static cores from the other prompt: 3 and 2 hits;
- a 200-byte page cache: 200 page-cache bytes and 300 SSD bytes;
- prompt bypass: 3 generated hits with admission and 1 without;
- a seeded trace: Belady bounds every policy at every size, and the stack-distance LRU equals the direct LRU at 3 sizes;
- the gate formulas, the repetition share, the percentiles, and the trace parser.

## Prompt set

`prompts/source.json` holds the proposed prompts. `trace-prompts` tokenizes them with the engine tokenizer and the pinned template and writes `prompts/prompts.proposed.json`:

```sh
cargo run --offline --release --manifest-path qualification/expert-cache/prompts/Cargo.toml -- \
  --checkpoint <Flash-Next checkpoint dir> --source qualification/expert-cache/prompts/source.json \
  --out qualification/expert-cache/prompts/prompts.proposed.json
```

| Id | Name | Prompt tokens | Generated |
| --- | --- | --- | --- |
| 0 | `c5`, raw `The capital of France is` | 5 | 1,024 |
| 1 | `chat` | 166 | 256 |
| 2 | `code` | 194 | 256 |
| 3 | `tool`, two tool definitions | 581 | 256 |
| 4 | `reasoning` | 150 | 256 |
| 5 | `long-document` | 2,048 | 4 |
| 6 | `long-context-decode` | 1,777 | 275 |

The resident decoder supports only the dense QSA path, which ends at position 2,050 for this checkpoint (`indexer_budget` 2,048, ratio 4).
A prompt of 2,048 tokens can therefore generate only 4 tokens.
Prompt 6 ends at position 2,050 too and measures generated tokens after a long context.
The file is proposed. The coordinator freezes it and records its SHA-256.
