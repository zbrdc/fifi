# Qwen3.6 K/V capacity on a 36 GiB Mac

Where the resident K/V cache stops fitting, per growth law and cache form.
Measured 2026-11-12 on a Mac with 36 GiB of memory (Apple silicon):
qwen3.6-35b-a3b, MLX 4-bit weights (base about 22 GB with the serving
stack), `kv_dtype q8` (about 51 KiB per position summed over the attention
layers), `context_limit` 262,144. Source: `qual-results.jsonl`, the
2026-11-12 measurement campaign (three rounds, both arms, depths 2K to
181K).

## Measured anchors

| arm | positions served | peak footprint |
|-----|-----------------|----------------|
| default (quantum) | 180,838 | 31.0 GB |
| `Q36_KV_GEOMETRIC=1` | 180,838 | 35.0 GB |
| default (quantum) | 114,035 | 29.0 GB |
| `Q36_KV_GEOMETRIC=1` | 114,035 | 30.0 GB |

At 180,838 positions the geometric arm had already landed its full
262,144-position cap (the 131,072 -> 262,144 doubling): the +4 GB over the
default arm is the cap quanta, paid early.

## The fitting law

- **Geometric (q8)**: the cap doubles; once the 262,144 cap is landed the
  footprint is flat to the context edge. It serves the entire 262,144
  context on the machine's 36 GiB with roughly 3.5 GB headroom at
  the edge (extrapolated from the 181K measurement; not measured at the
  edge itself).
- **Default quantum (q8)**: grows about 51 KiB per position to roughly
  34 GB at 262,144 — also fits the full context, tighter at the edge, and
  would keep growing past it if the context limit allowed (ceiling near
  310K).
- **F32**: four times the per-position cost. The geometric cap at 131,072
  alone is about 26 GB of K/V; F32 tops out near 65K depth and cannot
  cover the 128K-256K band. Q8 is the form for deep coding contexts.

## Where it stops fitting

The first position past 262,144. The next geometric doubling
(262,144 -> 524,288) costs about 25.8 GB of q8 K/V alone plus the ~22 GB
base — about 48 GB steady, with the old 12.9 GB cap alive during the copy
transient. That is categorically over a 36 GiB machine in either arm; no
512K context on this hardware in any q8 form.

## Speed context

The geometric win concentrates at depth: under 1 percent prefill gain at
2K, 2 percent at 8K, 7-9 percent at 32K, 14-17 percent at the 114K-181K
cells (figures at 2K-32K are short-prompt figures; the deep cells are the
ones the 128K-256K coding target cares about). Parity: the generated message
was identical between the arms over 128 greedy tokens in all eight cells.
Bit-level agreement is not what the campaign measured; that claim belongs
to the engine's digest tests (the prefill row digests in
`resident/prefill/step.rs` and the geometric-law gate tests in the qwen36
resident suite).
