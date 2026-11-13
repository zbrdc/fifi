//! The step plan of the Flash-Next CUDA decode: which activation tensors one token needs, how big
//! they are, and when each is live. The lifetimes go through `plan_arena`, so the tensors of
//! different phases share addresses and the whole step is ONE `DeviceArena` that never grows.
//!
//! Pure host arithmetic. Every size comes from the geometry of [`FlashGeometry`] (which reads the
//! host reference's `ModelShape`) or from the size function of the kernel wrapper that uses the
//! tensor (`hyper_read_bytes`, `conv_bytes`, `cpf_dn_chunk_bytes`, `QsaAttnShape::part_bytes`,
//! `router_logits_bytes`), so each size has one owner. The pattern is `qwen36/cuda/step_plan.rs`:
//! the lifetimes are in phases of ONE layer, every layer reuses the same offsets, and a DeltaNet
//! layer and a QSA layer run different phases of the same axis, so their tensors may share
//! addresses.
//!
//! # Phases of one token
//!
//! `StepBegin`, `Embed` (the hyper stream is the embedding row in every branch), `Ple` (at the PLE
//! layer only), then per layer `HyperReadA`, `MixerIn` (the in-projections of either mixer), the
//! DeltaNet phases `DeltaConv`, `DeltaRule` or the QSA phases `AttnPrep`, `Index`, `Score`,
//! `Select`, `AttnCore`, then `MixerOut` (the out-projection of either mixer), `HyperApplyA`,
//! `HyperReadM`, `Route`, `GateUp`, `Swiglu`, `Down`, `Combine`, `HyperApplyM`; after the last
//! layer `FinalMix`, `LmHead`, `Argmax`, `Advance`. The EXL3 MoE block runs the routed and shared
//! experts as one op in stream order (`family/exl3/moe.rs`), so the tensors it touches are live
//! from `GateUp` to `Combine`.
//!
//! # Rows
//!
//! A plan is for `rows` in `1..=MAX_STEP_ROWS`: plain decode is 1, a verify window more. Prefill has
//! its own tensor list (F3 of the plan); nothing here is a chunk size.
//!
//! # Context
//!
//! Only `Scores` (`[rows][cap_blocks]`) and `AttnPart` (its `rows` argument) depend on the context.
//! `Chosen` and `AttnPart` are sized for the full indexer budget (512 blocks) at every context. At
//! a context shorter than the budget the select and attention launches must be given
//! `min(budget_blocks, cap_blocks)` (`qsa_select` refuses a larger budget), and the plan then
//! over-allocates, which is harmless.
//! [`footprint`] reports the state and the scratch together for one context, which is what the
//! context budget charges.

use super::state::{FlashGeometry, StateBytes, StatePlan, overflow, product, sum};
use crate::{
    EngError,
    family::{flash_next::moe::context_budget::KvFormat, project::exl3_jobs_plan::MAX_EXL3_ROWS},
    kernels::cuda::{
        ArenaPlan, ArenaTensor, plan_arena,
        prefill::gdn_scan_seq::{cpf_dn_conv_out_bytes, cpf_dn_scan_out_bytes},
        resident::{
            hyper::{HyperShape, hyper_read_bytes},
            qsa_attn::QsaAttnShape,
            router::{RouterShape, router_logits_bytes},
            step_params::ARGMAX_PART_BYTES,
        },
    },
};

const F32: usize = size_of::<f32>();
const I32: usize = size_of::<i32>();

/// Most rows of one decode step. Qwen3.6 caps the merged expert launches at 8 rows and its
/// verify window there; the Flash-Next chunk driver builds its plan at the prefill chunk width,
/// whose shared expert splits its down groups at the kernel's 8-row bound, so the plan takes
/// wider chunks: every doubling halves the per-layer-chunk drain count of a deep prefill.
/// The bound is the shared `MAX_EXL3_ROWS`: `shared_rows` refuses wider windows whatever the
/// caller. The kernel's dense job bound is wider still (`jobs_row_cap`, 256, exl3.cu), so
/// raising this is a planning decision, not a kernel port; see `begin_chunk`.
pub(super) const MAX_STEP_ROWS: usize = MAX_EXL3_ROWS;

/// Splits of the QSA attention in a decode step unless the options say otherwise. The attended
/// positions of a token are the selected blocks (2,048 at most) and the open tail; the part
/// scratch grows linearly with the count (`n_head x splits x 258` f32 per row).
pub(super) const DEFAULT_QSA_SPLITS: usize = 16;

/// Choices fixed for a session that change a size of the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct StepOptions {
    /// `max_splits` of the QSA attention launch, 1 to 128.
    pub(super) qsa_splits: usize,
}

impl Default for StepOptions {
    fn default() -> Self {
        Self {
            qsa_splits: DEFAULT_QSA_SPLITS,
        }
    }
}

/// A hyper-connection site: the read before the attention (or DeltaNet) block, the read before the
/// MoE block, and the final mixer that feeds the head (no inject, so no gate and no block output).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Site {
    Attn,
    Mlp,
    Final,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Slot {
    /// f32 `[rows][branches * hidden]`: the residual stream, updated in place by every apply.
    Stream,
    /// f32 `[rows][hidden]`: the embedding row, written by the feeds before the step.
    EmbedRow,
    /// f32 `[rows][ple_embed]`: the n-gram embedding row of the PLE block, written by the feeds
    /// before the step. It is live for the whole step: the PLE block runs inside the layer loop
    /// (after layer `ple_layer - 1`), so the row has to survive every layer before it, and the
    /// phase axis cannot place the `Ple` phase there. Any feed written ahead of a consumer that
    /// runs mid-layer follows the same rule.
    PleEmbed,
    /// The scratch of a hyper read: f32 `[rows][width]` normed, `[rows][rank]` low, `[rows][width]`
    /// mix.
    HyperNormed(Site),
    HyperLow(Site),
    HyperMix(Site),
    /// f32 `[rows][hidden]`: the read's mixed row, the input of the block (`Attn`, `Mlp`) or of the
    /// head (`Final`).
    Mixed(Site),
    /// f32 `[rows][branches]`: written by the read, consumed by the matching apply (`Attn`, `Mlp`).
    Gate(Site),
    /// f32 `[rows][hidden]`: the block output the apply adds into the stream (`Attn`, `Mlp`).
    BlockOut(Site),
    /// PLE scratch (device PLE only): keys `[rows][width]`, value `[rows][hidden]`, the gated and
    /// normed rows `[rows][width]` and the update `[rows][width]`.
    PleKeys,
    PleValue,
    PleGated,
    PleNormed,
    PleOut,
    /// f32 `[rows][delta proj_ld]`: qkv, z, beta, alpha side by side.
    DeltaProj,
    /// f32 `[rows][conv_dim]`: the conv output the scan reads.
    DeltaConv,
    /// f32 `[rows][value_heads * value_dim]`: the scan output, gated in place.
    DeltaCore,
    /// f32 `[rows][qsa proj_ld]`: per query head `[q | gate]`, then k, v, then the indexer row.
    AttnProj,
    /// f32 `[rows][heads_q][256]`: `attn_prep`'s normed and rotated q.
    AttnQ,
    /// f32 `[rows][idx_heads][idx_dim]`: the indexer queries.
    IndexQueries,
    /// f32 `[rows][cap_blocks]`: the block scores of every token.
    Scores,
    /// i32 `[rows][budget_blocks]`: the chosen blocks, ascending.
    Chosen,
    /// f32 split scratch of the QSA attention.
    AttnPart,
    /// f32 `[rows][heads_q][256]`: the attention output, the input of the out-projection.
    AttnCore,
    /// f32 `[rows][experts + 1]`: router logits, the last column the shared-gate dot.
    RouterLogits,
    TopkId,
    TopkWeight,
    SharedGate,
    /// f32 `[rows][top_k][2 * inter]`: `[gate | up]` of every pick.
    ExpertGu,
    /// f32 `[rows][top_k][inter]`: the SwiGLU of the picks, the down projections' input.
    ExpertAct,
    ExpertDown,
    SharedGu,
    SharedAct,
    SharedOut,
    /// The EXL3 projections' scratch: rotated inputs, the split-K partial, the sticky flag.
    Exl3Xh,
    Exl3Z,
    Exl3Flag,
    /// The device job table of a chunk's job-table projections: one entry (a whole chunk,
    /// whatever its row count within `MAX_STEP_ROWS`), 16 bytes.
    Exl3Jobs,
    /// f32 `[rows][vocab]`.
    Logits,
    ArgmaxPart,
}

/// Phases of one layer plus the prologue and epilogue; the discriminant is the lifetime number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub(super) enum Phase {
    StepBegin = 0,
    Embed = 1,
    Ple = 2,
    HyperReadA = 3,
    /// The in-projections of the mixer: qkv/z/beta/alpha (DeltaNet) or q/k/v/indexer (QSA).
    MixerIn = 4,
    DeltaConv = 5,
    DeltaRule = 6,
    AttnPrep = 7,
    Index = 8,
    Score = 9,
    Select = 10,
    AttnCore = 11,
    /// The out-projection of the mixer into `BlockOut(Attn)`.
    MixerOut = 12,
    HyperApplyA = 13,
    HyperReadM = 14,
    Route = 15,
    GateUp = 16,
    Swiglu = 17,
    Down = 18,
    Combine = 19,
    HyperApplyM = 20,
    FinalMix = 21,
    LmHead = 22,
    Argmax = 23,
    /// `step_advance`, the last kernel of the step.
    Advance = 24,
}

/// The phase table in launch order.
pub(super) const PHASES: [Phase; 25] = [
    Phase::StepBegin,
    Phase::Embed,
    Phase::Ple,
    Phase::HyperReadA,
    Phase::MixerIn,
    Phase::DeltaConv,
    Phase::DeltaRule,
    Phase::AttnPrep,
    Phase::Index,
    Phase::Score,
    Phase::Select,
    Phase::AttnCore,
    Phase::MixerOut,
    Phase::HyperApplyA,
    Phase::HyperReadM,
    Phase::Route,
    Phase::GateUp,
    Phase::Swiglu,
    Phase::Down,
    Phase::Combine,
    Phase::HyperApplyM,
    Phase::FinalMix,
    Phase::LmHead,
    Phase::Argmax,
    Phase::Advance,
];

/// The table is strictly ascending and starts at 0, so a phase number is also its position.
const fn phases_ordered() -> bool {
    let mut at = 0;
    while at < PHASES.len() {
        if PHASES[at] as usize != at {
            return false;
        }
        at += 1;
    }
    true
}

const _: () = assert!(
    phases_ordered(),
    "the phase table must be 0, 1, 2, ... in order"
);

const fn at(phase: Phase) -> usize {
    phase as usize
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PlannedSlot {
    pub(super) slot: Slot,
    pub(super) bytes: usize,
    pub(super) first: usize,
    pub(super) last: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct StepPlan {
    rows: usize,
    positions: usize,
    slots: Vec<PlannedSlot>,
    arena: ArenaPlan,
}

fn refuse(detail: String) -> EngError {
    EngError::BadRequest(format!("Flash-Next step plan: {detail}"))
}

struct SlotList {
    rows: usize,
    slots: Vec<PlannedSlot>,
}

impl SlotList {
    fn add(
        &mut self,
        slot: Slot,
        bytes: Result<usize, EngError>,
        first: Phase,
        last: Phase,
    ) -> Result<(), EngError> {
        self.slots.push(PlannedSlot {
            slot,
            bytes: bytes?,
            first: at(first),
            last: at(last),
        });
        Ok(())
    }

    /// f32 `[rows][width]`.
    fn row(&self, width: usize) -> Result<usize, EngError> {
        product(&[self.rows, width, F32])
    }
}

/// The largest rotated-input and partial sizes of any EXL3 launch of the step: for each launch
/// `(rows, input width, output width)`, the inputs take `rows * k` f32 and the partial `rows * n`.
fn exl3_scratch(geometry: &FlashGeometry, rows: usize) -> Result<(usize, usize), EngError> {
    let (h, moe, delta, qsa) = (
        geometry.hidden,
        (geometry.top_k, geometry.inter, geometry.shared_inter),
        &geometry.delta,
        &geometry.qsa,
    );
    let (top_k, inter, shared_inter) = moe;
    // The routed experts: one row runs the grouped launch of both projections (2 * top_k member
    // rows); a chunk runs them as one job-table launch over its routed pairs, and the pair
    // layout rotates the second projection's input from the partial tail, so the rotation
    // face holds one routed face, not two.
    let (routed_in, routed_out) = if rows == 1 {
        (2 * top_k, top_k)
    } else {
        let routed = rows.checked_mul(top_k).ok_or_else(|| {
            EngError::BadRequest("the chunk's routed pair count overflows".into())
        })?;
        (routed, routed)
    };
    let mut launches = vec![
        // lm_head
        (rows, h, geometry.vocab),
        // the routed job table of the chunked prefill: one launch covers every pair of the
        // chunk, so the rotation and the partials hold the whole chunk's pairs
        (routed_in, h, inter),
        (routed_out, inter, h),
        // shared gate and up as 2 * rows groups, then its down
        (2 * rows, h, shared_inter),
        (rows, shared_inter, h),
        // DeltaNet: qkv, z, out (beta and alpha are dense f32)
        (rows, h, delta.conv_dim),
        (rows, h, delta.value_width()),
        (rows, delta.value_width(), h),
        // QSA: q and gate, k, v, indexer, out
        (rows, h, 2 * qsa.q_width()),
        (rows, h, qsa.kv_width()),
        (rows, h, qsa.idx_proj()),
        (rows, qsa.q_width(), h),
        // the grouped k and v input projections of the QSA mixer: both read the same k and share
        // kv_width, the one mixer pair the grouped launch's single-width contract allows (the
        // tuple counts members as rows, so z covers both members' outputs)
        (2 * rows, h, qsa.kv_width()),
        // PLE: key and value projections of the n-gram row
        (rows, geometry.ple_embed, geometry.width()),
        (rows, geometry.ple_embed, h),
    ];
    launches.retain(|&(r, k, n)| r > 0 && k > 0 && n > 0);
    let mut xh = 0usize;
    let mut z = 0usize;
    for &(r, k, n) in &launches {
        xh = xh.max(product(&[r, k])?);
        z = z.max(product(&[r, n])?);
    }
    Ok((product(&[xh, F32])?, product(&[z, F32])?))
}

fn list_slots(
    geometry: &FlashGeometry,
    rows: usize,
    positions: usize,
    options: &StepOptions,
) -> Result<Vec<PlannedSlot>, EngError> {
    let mut list = SlotList {
        rows,
        slots: Vec::new(),
    };
    let (h, width) = (geometry.hidden, geometry.width());
    let (delta, qsa) = (&geometry.delta, &geometry.qsa);
    let hyper = hyper_read_bytes(HyperShape {
        m: rows,
        branches: geometry.branches,
        hidden: h,
        rank: geometry.rank,
    })?;

    // The prologue and the stream.
    list.add(Slot::EmbedRow, list.row(h), Phase::StepBegin, Phase::Embed)?;
    list.add(
        Slot::PleEmbed,
        list.row(geometry.ple_embed),
        Phase::StepBegin,
        Phase::FinalMix,
    )?;
    list.add(Slot::Stream, list.row(width), Phase::Embed, Phase::FinalMix)?;
    list.add(Slot::PleKeys, list.row(width), Phase::Ple, Phase::Ple)?;
    list.add(Slot::PleValue, list.row(h), Phase::Ple, Phase::Ple)?;
    list.add(Slot::PleGated, list.row(width), Phase::Ple, Phase::Ple)?;
    list.add(Slot::PleNormed, list.row(width), Phase::Ple, Phase::Ple)?;
    list.add(Slot::PleOut, list.row(width), Phase::Ple, Phase::Ple)?;

    // The three hyper reads of a layer and the final mixer. The scratch of a read is live only in
    // its read; the gate lives to the matching apply.
    for (site, read, apply) in [
        (Site::Attn, Phase::HyperReadA, Some(Phase::HyperApplyA)),
        (Site::Mlp, Phase::HyperReadM, Some(Phase::HyperApplyM)),
        (Site::Final, Phase::FinalMix, None),
    ] {
        list.add(Slot::HyperNormed(site), Ok(hyper.normed), read, read)?;
        list.add(Slot::HyperLow(site), Ok(hyper.low), read, read)?;
        list.add(Slot::HyperMix(site), Ok(hyper.mix), read, read)?;
        if let Some(apply) = apply {
            list.add(Slot::Gate(site), Ok(hyper.gate), read, apply)?;
        }
    }
    list.add(
        Slot::Mixed(Site::Attn),
        Ok(hyper.mixed),
        Phase::HyperReadA,
        Phase::MixerIn,
    )?;
    list.add(
        Slot::Mixed(Site::Mlp),
        Ok(hyper.mixed),
        Phase::HyperReadM,
        Phase::Combine,
    )?;
    list.add(
        Slot::Mixed(Site::Final),
        Ok(hyper.mixed),
        Phase::FinalMix,
        Phase::LmHead,
    )?;
    list.add(
        Slot::BlockOut(Site::Attn),
        list.row(h),
        Phase::MixerOut,
        Phase::HyperApplyA,
    )?;
    list.add(
        Slot::BlockOut(Site::Mlp),
        list.row(h),
        Phase::Combine,
        Phase::HyperApplyM,
    )?;

    // DeltaNet.
    list.add(
        Slot::DeltaProj,
        list.row(delta.proj_ld()),
        Phase::MixerIn,
        Phase::DeltaRule,
    )?;
    list.add(
        Slot::DeltaConv,
        cpf_dn_conv_out_bytes(rows, delta.conv_dim),
        Phase::DeltaConv,
        Phase::DeltaRule,
    )?;
    list.add(
        Slot::DeltaCore,
        cpf_dn_scan_out_bytes(rows, delta.value_heads, delta.value_dim),
        Phase::DeltaRule,
        Phase::MixerOut,
    )?;

    // QSA.
    let cap_blocks = qsa.cap_blocks(positions);
    list.add(
        Slot::AttnProj,
        list.row(qsa.proj_ld()),
        Phase::MixerIn,
        Phase::AttnCore,
    )?;
    list.add(
        Slot::AttnQ,
        list.row(qsa.q_width()),
        Phase::AttnPrep,
        Phase::AttnCore,
    )?;
    list.add(
        Slot::IndexQueries,
        list.row(qsa.idx_heads * qsa.idx_dim),
        Phase::Index,
        Phase::Score,
    )?;
    list.add(
        Slot::Scores,
        list.row(cap_blocks),
        Phase::Score,
        Phase::Select,
    )?;
    list.add(
        Slot::Chosen,
        product(&[rows, qsa.budget_blocks(), I32]),
        Phase::Select,
        Phase::AttnCore,
    )?;
    let part = QsaAttnShape {
        m: rows,
        n_head: qsa.heads_q,
        n_kv_head: qsa.heads_kv,
        rows: positions,
        ratio: qsa.ratio,
        budget_blocks: qsa.budget_blocks(),
        max_splits: options.qsa_splits,
    }
    .part_bytes();
    list.add(Slot::AttnPart, part, Phase::AttnCore, Phase::AttnCore)?;
    list.add(
        Slot::AttnCore,
        list.row(qsa.q_width()),
        Phase::AttnCore,
        Phase::MixerOut,
    )?;

    // The MoE block.
    list.add(
        Slot::RouterLogits,
        router_logits_bytes(RouterShape {
            m: rows,
            hidden: h,
            n_expert: geometry.experts,
            top_k: geometry.top_k,
        }),
        Phase::Route,
        Phase::Route,
    )?;
    list.add(
        Slot::TopkId,
        product(&[rows, geometry.top_k, I32]),
        Phase::Route,
        Phase::Combine,
    )?;
    list.add(
        Slot::TopkWeight,
        list.row(geometry.top_k),
        Phase::Route,
        Phase::Combine,
    )?;
    list.add(Slot::SharedGate, list.row(1), Phase::Route, Phase::Combine)?;
    let (top_k, inter, shared) = (geometry.top_k, geometry.inter, geometry.shared_inter);
    for (slot, bytes) in [
        (Slot::ExpertGu, product(&[rows, top_k, 2 * inter, F32])),
        (Slot::ExpertAct, product(&[rows, top_k, inter, F32])),
        (Slot::ExpertDown, product(&[rows, top_k, h, F32])),
        (Slot::SharedGu, list.row(2 * shared)),
        (Slot::SharedAct, list.row(shared)),
        (Slot::SharedOut, list.row(h)),
    ] {
        list.add(slot, bytes, Phase::GateUp, Phase::Combine)?;
    }

    // The EXL3 scratch, the head and the argmax.
    let (xh, z) = exl3_scratch(geometry, rows)?;
    list.add(Slot::Exl3Xh, Ok(xh), Phase::Ple, Phase::LmHead)?;
    list.add(Slot::Exl3Z, Ok(z), Phase::Ple, Phase::LmHead)?;
    // The step clears the flag at StepBegin and folds it into the status just before the argmax.
    list.add(Slot::Exl3Flag, Ok(4), Phase::StepBegin, Phase::Argmax)?;
    list.add(Slot::Exl3Jobs, Ok(16), Phase::StepBegin, Phase::Argmax)?;
    list.add(
        Slot::Logits,
        list.row(geometry.vocab),
        Phase::LmHead,
        Phase::Argmax,
    )?;
    list.add(
        Slot::ArgmaxPart,
        product(&[rows, ARGMAX_PART_BYTES]),
        Phase::Argmax,
        Phase::Argmax,
    )?;
    Ok(list.slots)
}

impl StepPlan {
    /// The plan of one decode step of `rows` rows over a context of `positions`. `positions` only
    /// sizes the block scores and the attention split scratch.
    pub(super) fn new(
        geometry: &FlashGeometry,
        rows: usize,
        positions: usize,
        options: &StepOptions,
    ) -> Result<Self, EngError> {
        if !(1..=MAX_STEP_ROWS).contains(&rows) {
            return Err(refuse(format!(
                "{rows} rows, a decode step takes 1 to {MAX_STEP_ROWS}"
            )));
        }
        if positions == 0 {
            return Err(refuse("a context of zero positions".to_owned()));
        }
        if !(1..=128).contains(&options.qsa_splits) {
            return Err(refuse(format!(
                "{} QSA attention splits, the kernel takes 1 to 128",
                options.qsa_splits
            )));
        }
        let slots = list_slots(geometry, rows, positions, options)?;
        let tensors: Vec<ArenaTensor> = slots
            .iter()
            .map(|planned| ArenaTensor {
                bytes: planned.bytes,
                first: planned.first,
                last: planned.last,
            })
            .collect();
        let arena = plan_arena(&tensors)?;
        Ok(Self {
            rows,
            positions,
            slots,
            arena,
        })
    }

    pub(super) fn rows(&self) -> usize {
        self.rows
    }

    pub(super) fn positions(&self) -> usize {
        self.positions
    }

    pub(super) fn slots(&self) -> &[PlannedSlot] {
        &self.slots
    }

    /// The tensors to hand to `CudaContext::arena`, in slot order.
    pub(super) fn arena_tensors(&self) -> Vec<ArenaTensor> {
        self.slots
            .iter()
            .map(|planned| ArenaTensor {
                bytes: planned.bytes,
                first: planned.first,
                last: planned.last,
            })
            .collect()
    }

    /// The tensor index of `slot` in the arena (`DeviceArena::tensor`).
    pub(super) fn index(&self, slot: Slot) -> Option<usize> {
        self.slots.iter().position(|planned| planned.slot == slot)
    }

    /// `(offset, bytes)` of `slot` in the arena.
    pub(super) fn placement(&self, slot: Slot) -> Option<(usize, usize)> {
        let index = self.index(slot)?;
        Some((self.arena.offsets[index], self.slots[index].bytes))
    }

    /// Device bytes of the step arena.
    pub(super) fn arena_bytes(&self) -> usize {
        self.arena.total
    }

    /// The sum of every tensor, as if none shared an address.
    pub(super) fn unaliased_bytes(&self) -> Result<usize, EngError> {
        sum(&self
            .slots
            .iter()
            .map(|planned| planned.bytes)
            .collect::<Vec<_>>())
    }
}

/// Everything the Flash-Next step puts on the device apart from weights and the expert bank, for
/// one context: the state arena and the step arena. The expert slots take the remainder of the
/// card, so the context budget charges [`Self::total_bytes`] (S0 today: it charges
/// `StateBytes::delta_state_bytes` in `vram.non_expert` and the K/V in its `KvPlan`; F2 charges the
/// whole footprint).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DeviceFootprint {
    pub(super) positions: usize,
    pub(super) rows: usize,
    pub(super) state: StateBytes,
    /// The step arena total.
    pub(super) scratch_bytes: usize,
}

impl DeviceFootprint {
    /// State arena plus step arena.
    pub(super) fn total_bytes(&self) -> Result<usize, EngError> {
        self.state
            .arena_bytes
            .checked_add(self.scratch_bytes)
            .ok_or_else(overflow)
    }

    /// The part that does not grow with the context: fixed state plus the step arena of this
    /// context (the scratch grows by about one byte per position and row).
    pub(super) fn non_context_bytes(&self) -> Result<usize, EngError> {
        self.state
            .fixed_bytes()
            .checked_add(self.scratch_bytes)
            .ok_or_else(overflow)
    }
}

/// The device bytes of the state and one decode step at `positions` context positions.
pub(super) fn footprint(
    geometry: &FlashGeometry,
    positions: usize,
    rows: usize,
    options: &StepOptions,
    format: KvFormat,
) -> Result<DeviceFootprint, EngError> {
    let state = StatePlan::new(geometry, positions, format)?.bytes()?;
    let step = StepPlan::new(geometry, rows, positions, options)?;
    Ok(DeviceFootprint {
        positions,
        rows,
        state,
        scratch_bytes: step.arena_bytes(),
    })
}
