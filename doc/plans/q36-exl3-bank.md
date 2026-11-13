# q36 EXL3 expert bank — design

Lane `task/q36-exl3-bank`, base `d2015ab85` (2026-11-07). Target: run
Qwen3.6-35B-A3B EXL3 packs on cards that cannot hold the whole routed bank
(first target: RTX 4050 6 GB, 14 GiB RAM — see the lane NOW.md).

## Problem

`family/qwen36/cuda/exl3_load.rs` loads every tensor of an EXL3 pack resident.
A 2.08 bpw pack is 10.58 GB of tensors; the 4050 offers 5.9 GB free. The flint
family has an expert bank (pinned host store + LRU device slots + miss H2D,
`flint/exl3_resident` + `flint/budget.rs`), but q36's EXL3 arm has nothing:
`tiered` is refused for exl3 and the `FIFI_FLINT_*` switches are flint-only.

## Established facts (recon 2026-11-07)

1. The eager EXL3 decode already syncs picks to the host per layer:
   `exl3_moe/merge.rs` — "The host path (picks read back, no graph) queues
   the same groups as plain grouped launches." `read_picks` + grouped
   launches take the router's ids.
2. Expert launches address weights as `id * stride` member banks:
   `project::cuda::exl3::{Exl3Bank, Exl3Io, Exl3Weight}` ("bank Gate,
   ids[s]"; the shared expert rides a 1-member bank whose id word is a zero).
   Nothing in the launch path multiplies by 256 inherently — member count is
   a bank property.
3. q36 already has the exact control-flow precedent in `cuda/moe/tiered.rs`
   (GGUF): router → pinned readback → host maps picks to bank slots →
   **the nine native ops run unchanged** over slot-indexed banks with the
   slot-id buffer standing in for the router ids; captured windows refuse
   the host-sync calls with typed errors (eager only).
4. GGUF tiered serves misses on the CPU. For EXL3 we do not need CPU expert
   kernels: the encoded expert bytes can be copied H2D into a device slot and
   computed by the unchanged device kernels (the flint bank's approach).

## Design

**Bank = per-layer `Exl3Bank`s with `members = S` (slots) instead of 256,
plus a pinned host store of every layer's encoded expert bytes, plus a host
miss resolver at the existing `read_picks` site.**

- **Load** (`exl3_load.rs`, new banked arm): allocate per-layer banks of `S`
  expert stacks (gate/up/down byte ranges per expert as today, `S` members),
  and keep every layer's 256 experts' raw EXL3-encoded bytes in host RAM in a
  **pageable store (plain `Vec`, not page-locked) — v1**; the driver stages
  H2D from pageable memory internally, and 9.9 GB of page-locked memory on a
  14 GiB laptop risks OS instability (reviewer finding 4). A bounded pinned
  staging ring is a measured follow-up, not v1. Admission: host bytes = full
  expert bank (2.08 bpw ≈ 9.9 GB) against a host-RAM budget **with an
  explicit OS margin floor for laptop-class boxes**; device bytes = dense +
  `S`·per-slot + KV + reserve (flint `bank_budget` semantics, per-slot
  audited from the pack, not the 4.00-bpw 72 MiB figure). The q36 arm
  **re-implements the small policy core** (mode/request/budget arithmetic
  with parity tests against flint's `budget.rs` rules) rather than promoting
  `pub(super)` flint internals across families (reviewer finding 5); the
  reader closure comes from the shared `weights::exl3::Checkpoint`, which
  q36 already opens.
- **Decode** (eager only): the resolver hooks the **merged host block
  (`merged_on_host`, the default decode block)** and the reference `routed`
  path alike — both already read picks to the host (`merge.rs:361`,
  `exl3_moe.rs:578`). The host maps each pick `id → slot` through a
  per-layer LRU table; misses enqueue H2D copies of that expert's byte
  ranges into its layer's slot stacks, then the launches receive **slot
  ids** where they today receive router ids. Router weights, kernels,
  buffers and combine are untouched. Readback per layer is ids only
  (32 B); no hidden-row copy (that is the CPU-expert path's need).
- **Refused while banked** (by name, like today's checks): graph capture,
  `draft_mode`, **and `draft_verify = krow` windows** (krow_step runs
  `run_exl3_moe_rows` with device-resident picks under per-window CUDA
  graphs — no host hook exists there; reviewer finding 1).
- **Capture**: a banked engine refuses graph capture (settings force
  `capture: false` with a start line when the bank is on; the typed
  captured-window refusals from tiered are the precedent). This also matches
  the fact that the merged/captured EXL3 step reads picks on the device.
- **Prefill**: v1 routes prefill pieces through the same miss resolver,
  which must also remap the **prefill job-table members** — the job table
  addresses experts directly (`member: member as u32`,
  `project/cuda/exl3_jobs.rs:111`) after a 256-bin host sort (the
  `read_plan` of `family/exl3/prefill_moe.rs`), so the
  remap happens there, host-side (reviewer finding 3). A piece that touches
  all 256 experts of a layer streams them through the slots once per piece.
  Measured cost on the 4050 (PCIe gen4 x8) decides whether we need flint's
  `cache_prefill_floor` analog later. The floor itself is a follow-up, not
  v1.
- **Knobs** (q36 convention — JSON, not env): a `qwen36.expert_bank` object,
  `{ "slots": N, "mode": "auto" | "on" }` — `auto` banks only when the
  resident load would not admit; refused together with `tiered` and
  `kv_offload_host_bytes` (exclusive members, by name, like today's checks).
  `slots` unset = fit (largest bank the device budget leaves, capped 256).
  Mirrors `flint/budget.rs` semantics; no new env vars.
- **Speculation**: `draft_mode` is EXL3-only today; v1 refuses drafts on a
  banked engine (the eager window + miss copies have no draft story yet).

## What is deliberately NOT in v1

CPU expert compute (GGUF-tiered's miss path), overlapped miss copies (a copy
stream + events — the Phase-A audit's whole subject), prefill floors,
multi-device, mmap/NVMe host store. Each is a follow-up with its own gate.

## Test plan

- Unit (host): bank budget arithmetic against the flint rules (parity of
  `plan_bank`/`choose_slots` semantics via the q36 JSON surface); slot table
  LRU behavior; settings refusals (tiered+expert_bank, capture+bank,
  drafts+bank) by name.
- Device (RTX 4050, `--test-threads=1`, single lease): fixture-pack banked step
  vs resident step logits parity (same pack, `slots=256` bank ≡ resident);
  miss-copy correctness with `slots < 256` against a resident reference run;
  LRU eviction order; readback/copy fencing.
- End-to-end (RTX 4050, 2.08 bpw pack): `/ready`, smoke completion, decode
  rate + miss statistics (the flint bank keeps hit/miss counters — carry
  them into the route report), prefill wall time per piece size.

## Milestones

- M1: settings + budget + banked load (host store, slot banks, admission) —
  default-off, unit-green, device load test on the 4050 fixture.
- M2: decode miss resolver + eager banked step + parity tests; capture
  refusal wiring.
- M3: prefill through the resolver; 4050 end-to-end on the 2.08 pack;
  measurements + report; review (zai/glm-5.3).

## Review

M0 review (zai/glm-5.3, 2026-11-07): all six factual claims CONFIRMED with
file:line evidence; five additions folded in above (krow refusal, merged
block as the hook site, prefill job member remap, pageable store, policy
re-implementation). One known check to relax under a flag: `Exl3MoeWeights::new`
rejects `bank.count() != moe.n_expert` (exl3_moe.rs:96) — the banked arm
supplies slot-indexed banks and must carry its own member check.

## Risks / open questions

- `Exl3Bank` member-count invariants: `Exl3MoeWeights::new` rejects
  `bank.count() != moe.n_expert` (exl3_moe.rs:96) — the banked arm needs its
  own `check_bank` (tiered's full-depth-facts trick is the precedent).
- Per-layer sync cost: the eager path already pays the picks readback per
  layer; miss copies add H2D on the same stream. Bound the per-token serial
  copy time on the 4050 before optimizing (M2 measurement).
- Shared expert ("shared in routed") is structurally outside the routed
  banks (three single `CudaWeight::Exl3` matrices,
  `exl3_moe_source.rs:167-176`) — it stays resident and out of the slot
  table unchanged.
- Host store of 9.9 GB on a 14 GiB laptop is resident-but-swappable in v1
  (pageable); the admission leaves an explicit OS margin.
