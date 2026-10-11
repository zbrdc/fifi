//! The step plan: which activation tensors one resident decode step needs, how big they are, and
//! when each is live (decision 0047 item 3, packet RD-9a).
//!
//! Pure host arithmetic. Every size comes from the layout record (`HostShape`), from the output
//! rows of a tensor role in the inventory, or from a [`KernelScratch`] that the caller fills with
//! the byte counts of the kernel wrappers; no model number is written here. The lifetimes are in
//! phases of ONE layer. All layers reuse the same offsets, so a layer is a single time axis: a
//! DeltaNet layer runs the phases `DELTA_IN` to `DELTA_OUT`, an attention layer `ATTN_IN` to
//! `ATTN_OUT`, and the tensors of the two mixers may share addresses because a layer runs only one
//! of them. `StepParams`, the result pair and the prefill status word are NOT in this plan: they
//! live in the words block of `state_layout.rs`, so every plan (every row count) and the prefill
//! path see the same addresses. Prefill has its own plan and arena (decision 0049); nothing here
//! is a prefill size.
//!
//! # Rows
//!
//! A plan is built for `rows` in `1..=MAX_STEP_ROWS` (8): plain decode is 1, a verify window
//! (MEL-PLAN 6.5, approved by the user) is 2..=8, and the merged `matmul_expert` takes `m` in
//! `1..=8` (`resident/matmul_expert.rs:36`). The tensors that grow with the row count are the
//! residual and mixer rows, the router and top-k tensors, the expert tensors, `AttnPart` (about
//! 1.06 MB per row), `Logits` (993,280 bytes per row) and, once a per-row argmax exists, the
//! `argmax_part`. The `Xf*` packs are one fixed fragment-order buffer up to 16 rows. A session
//! that runs several row counts keeps one `DeviceArena` sized for its largest plan and reaches the
//! smaller plans through `DeviceArena::view` at their own offsets.
//!
//! # Streams
//!
//! The lifetimes are one time axis and assume stream order. Where work runs on branch streams
//! (`forked_experts`), the branches must join the main stream with events at the phase boundaries
//! the table states, or the aliasing is unsafe (RD-8b.md:68 says the same for `AttnPart`).
//!
//! # Order of the status writers (RD-4d R-C2b, lane C C2)
//!
//! The status word (mask 1 router fault, 2 non-finite logit, 4 invalid step state; a prefill chunk
//! has its own word with mask 8) is written by `step_begin` (first kernel: clears it, then sets 4),
//! by the embedding (RD-4c: ORs 4 for a token id outside the vocabulary), by `attn_prep` (RD-8a:
//! ORs 4 on a refused launch), by the router (ORs 1) and by `argmax` (ORs
//! 2; its `atomicOr` return value is the status half of the result pair). `argmax` MUST be the
//! last kernel that writes status in the step; any later writer breaks the pair.
//!
//! What the compiler checks, and what it does not: [`writes_status`] is an exhaustive `match` over
//! [`Phase`], so adding a phase forces a decision about status; `status_order_ok` (a `const fn`
//! evaluated in a `const` assertion) walks the phase table `PHASES` that the plan's lifetimes are
//! written in and requires the table to be strictly ordered, `StepBegin` to be first and a
//! writer, every writer to be at or before `Argmax`, and `Advance` to be after it and no writer.
//! It cannot see a kernel that is launched but missing from the table: the real order of the
//! launches is RD-9d's, and its order test (RD-4d O5) is the gate. This is a documented rule with
//! a table check, not a proof.
//!
//! The host maps the status before it reads the token: precedence 4 (`step_params`), then 8
//! (`prefill_chunk`, prefill word only), then 1 (`route_experts`), then 2 (`sample`); the adapter
//! sets no latch, the session state is poisoned until reset, and a mask 4 report has two sources:
//! a defect (the host check `ensure_rows` ran first and missed a case), or the embedding gather,
//! which sets mask 4 for a token id outside the vocabulary (RD-4c). The status word cannot tell
//! the two causes apart.
//!
//! Design guidance only, no code copied; the per-component accounting and the plan print follow
//! the idea of MegaCapybara C4 (research/megacapybara-re-lessons.md section 6.1).

use std::fmt;

use super::{
    dims::{StepDims, StepOptions},
    state_layout::align_up,
};
use crate::{
    EngError,
    kernels::cuda::{
        ArenaPlan, ArenaTensor, plan_arena,
        resident::{
            attn_decode::{AttnSplit, attn_part_bytes},
            router::{RouterShape, router_logits_bytes},
            step_params::ARGMAX_PART_BYTES,
        },
        xf_bytes,
    },
};

const F32: usize = size_of::<f32>();
const I32: usize = size_of::<i32>();

/// Most rows of one decode step: the merged `matmul_expert` takes `m` in `1..=8`
/// (`resident/matmul_expert.rs:36`) and the user-approved verify window is capped at 8. RD-8b and
/// the fragment-order `xf` buffer would allow 16; a plan for 9 to 16 rows could never run.
pub(crate) const MAX_STEP_ROWS: usize = 8;

/// Scratch bytes of the stage 1 kernels for one row count. `attn_part` and `argmax_part` come
/// from the merged wrappers (RD-8b `attn_part_bytes` with `AttnSplit::STAGE1`, RD-4d
/// `ARGMAX_PART_BYTES`, RD-5 `router_logits_bytes`), so each size has one owner. [`Self::for_dims`]
/// takes every size from its wrapper; [`Self::new`] takes a router size from the caller and the
/// plan refuses anything below `4 * rows * (experts + 1)` (1,028 at one row).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KernelScratch {
    rows: usize,
    router_logits: usize,
    attn_part: usize,
    argmax_part: usize,
}

impl KernelScratch {
    /// The scratch of `rows` rows with the router size of RD-5's `router_logits_bytes`.
    pub(crate) fn for_dims(dims: &StepDims, rows: usize) -> Result<Self, EngError> {
        let moe = &dims.shape.moe;
        let router = router_logits_bytes(RouterShape {
            m: rows,
            hidden: dims.shape.hidden,
            n_expert: moe.experts,
            top_k: moe.top_k,
        })?;
        Self::new(dims, rows, router)
    }

    pub(crate) fn new(
        dims: &StepDims,
        rows: usize,
        router_logits: usize,
    ) -> Result<Self, EngError> {
        let attn = &dims.shape.attn;
        let attn_part = attn_part_bytes(
            rows,
            attn.heads_kv,
            attn.head_dim,
            AttnSplit::STAGE1.max_splits,
        )?;
        Ok(Self {
            rows,
            router_logits,
            attn_part,
            argmax_part: ARGMAX_PART_BYTES,
        })
    }

    /// Refuses a scratch made for another row count or too small for the shape.
    fn check(&self, dims: &StepDims, rows: usize) -> Result<(), EngError> {
        let router_min = product(&[rows, dims.shape.moe.experts + 1, F32])?;
        if self.rows != rows || self.router_logits < router_min {
            return Err(EngError::BadRequest(format!(
                "step plan: the kernel scratch was made for {} rows with {} router bytes, the plan needs {rows} rows and at least {router_min}",
                self.rows, self.router_logits
            )));
        }
        Ok(())
    }
}

/// One activation tensor of the step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Slot {
    ResidA,
    ResidB,
    /// fp16 pack feeding the mixer in-projections (`k` = hidden).
    XfMix,
    /// f32 `[rows][delta_proj]`: qkv, z, beta, alpha side by side.
    DeltaProj,
    DeltaConv,
    DeltaCore,
    XfDeltaOut,
    /// f32 `[rows][attn_proj]`: per query head `[q | gate]`, then k, then v.
    AttnProj,
    AttnQ,
    /// RD-8b split-KV partials; written by the part kernel and read by the combine kernel, both
    /// in `ATTN_CORE`.
    AttnPart,
    AttnCore,
    XfAttnOut,
    /// Mixer output `[rows][hidden]`, added into the residual by the next norm.
    MixOut,
    XfMoe,
    /// f32 normalized row, the router input.
    NormRow,
    /// RD-5 logits `[rows][experts + 1]`; the last column is the shared-gate dot.
    RouterLogits,
    TopkId,
    TopkWeight,
    SharedGate,
    ExpertGu,
    /// Pack feeding the expert down projection; one per routed expert when forked.
    XfDown(usize),
    ExpertDown,
    SharedGu,
    XfShared,
    SharedOut,
    XfFinal,
    Logits,
    /// RD-4d argmax partials, live only in `ARGMAX`.
    ArgmaxPart,
}

/// Phases of one layer plus the prologue and epilogue. A layer runs `MixPack`, then `DeltaIn..=
/// DeltaOut` or `AttnIn..=AttnOut`, then `AddNorm` onward. The discriminant is the lifetime number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub(super) enum Phase {
    StepBegin = 0,
    Embed = 1,
    MixPack = 2,
    DeltaIn = 3,
    DeltaConv = 4,
    DeltaRule = 5,
    DeltaPack = 6,
    DeltaOut = 7,
    AttnIn = 8,
    AttnPrep = 9,
    AttnCore = 10,
    AttnPack = 11,
    AttnOut = 12,
    AddNorm = 13,
    Route = 14,
    GateUp = 15,
    Swiglu = 16,
    Down = 17,
    Combine = 19,
    FinalNorm = 21,
    LmHead = 22,
    Argmax = 23,
    /// `step_advance`: the last kernel of the step. It writes `pos`, `kv_len` and `kv_slot`,
    /// never `status`.
    Advance = 24,
}

/// The phase table the plan's lifetimes are written in, in launch order.
pub(super) const PHASES: [Phase; 23] = [
    Phase::StepBegin,
    Phase::Embed,
    Phase::MixPack,
    Phase::DeltaIn,
    Phase::DeltaConv,
    Phase::DeltaRule,
    Phase::DeltaPack,
    Phase::DeltaOut,
    Phase::AttnIn,
    Phase::AttnPrep,
    Phase::AttnCore,
    Phase::AttnPack,
    Phase::AttnOut,
    Phase::AddNorm,
    Phase::Route,
    Phase::GateUp,
    Phase::Swiglu,
    Phase::Down,
    Phase::Combine,
    Phase::FinalNorm,
    Phase::LmHead,
    Phase::Argmax,
    Phase::Advance,
];

/// Whether the kernel of `phase` writes the status word. Exhaustive: a new phase needs an answer.
pub(super) const fn writes_status(phase: Phase) -> bool {
    match phase {
        Phase::StepBegin | Phase::Embed | Phase::AttnPrep | Phase::Route | Phase::Argmax => true,
        Phase::MixPack
        | Phase::DeltaIn
        | Phase::DeltaConv
        | Phase::DeltaRule
        | Phase::DeltaPack
        | Phase::DeltaOut
        | Phase::AttnIn
        | Phase::AttnCore
        | Phase::AttnPack
        | Phase::AttnOut
        | Phase::AddNorm
        | Phase::GateUp
        | Phase::Swiglu
        | Phase::Down
        | Phase::Combine
        | Phase::FinalNorm
        | Phase::LmHead
        | Phase::Advance => false,
    }
}

/// The documented status rule over the phase table (see the module doc for what it cannot see).
const fn status_order_ok() -> bool {
    let argmax = Phase::Argmax as usize;
    if PHASES[0] as usize != 0 || !writes_status(PHASES[0]) {
        return false;
    }
    let mut at = 1;
    while at < PHASES.len() {
        let (phase, before) = (PHASES[at] as usize, PHASES[at - 1] as usize);
        if phase <= before || (writes_status(PHASES[at]) && phase > argmax) {
            return false;
        }
        at += 1;
    }
    (Phase::Advance as usize) > argmax
}

const _: () = assert!(
    status_order_ok(),
    "argmax must be the last kernel that writes status in a step"
);

const EMBED: usize = Phase::Embed as usize;
const MIX_PACK: usize = Phase::MixPack as usize;
const DELTA_IN: usize = Phase::DeltaIn as usize;
const DELTA_CONV: usize = Phase::DeltaConv as usize;
const DELTA_RULE: usize = Phase::DeltaRule as usize;
const DELTA_PACK: usize = Phase::DeltaPack as usize;
const DELTA_OUT: usize = Phase::DeltaOut as usize;
const ATTN_IN: usize = Phase::AttnIn as usize;
const ATTN_PREP: usize = Phase::AttnPrep as usize;
const ATTN_CORE: usize = Phase::AttnCore as usize;
const ATTN_PACK: usize = Phase::AttnPack as usize;
const ATTN_OUT: usize = Phase::AttnOut as usize;
const ADD_NORM: usize = Phase::AddNorm as usize;
const ROUTE: usize = Phase::Route as usize;
const GATE_UP: usize = Phase::GateUp as usize;
const SWIGLU: usize = Phase::Swiglu as usize;
const DOWN: usize = Phase::Down as usize;
const COMBINE: usize = Phase::Combine as usize;
const FINAL_NORM: usize = Phase::FinalNorm as usize;
const LM_HEAD: usize = Phase::LmHead as usize;
const ARGMAX: usize = Phase::Argmax as usize;

/// A slot with its size and closed lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlannedSlot {
    pub(crate) slot: Slot,
    pub(crate) bytes: usize,
    pub(crate) first: usize,
    pub(crate) last: usize,
}

/// The activation arena of one step shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StepPlan {
    rows: usize,
    slots: Vec<PlannedSlot>,
    arena: ArenaPlan,
    unaliased: usize,
}

fn product(parts: &[usize]) -> Result<usize, EngError> {
    parts.iter().try_fold(1usize, |total, &part| {
        total
            .checked_mul(part)
            .ok_or_else(|| EngError::MemoryPlan("step plan: a tensor size overflows".to_owned()))
    })
}

impl StepPlan {
    /// Lists the tensors of a step of `rows` rows, places them with `plan_arena`, and records the
    /// size without aliasing for the report.
    pub(crate) fn new(
        dims: &StepDims,
        rows: usize,
        options: &StepOptions,
        scratch: &KernelScratch,
    ) -> Result<Self, EngError> {
        if !(1..=MAX_STEP_ROWS).contains(&rows) {
            return Err(EngError::BadRequest(format!(
                "step plan: {rows} rows, a decode step takes 1 to {MAX_STEP_ROWS}"
            )));
        }
        scratch.check(dims, rows)?;
        let slots = list_slots(dims, rows, options, scratch)?;
        let arena = plan_arena(&arena_tensors(&slots))?;
        let unaliased = slots
            .iter()
            .try_fold(0usize, |sum, s| {
                align_up(s.bytes).and_then(|bytes| sum.checked_add(bytes))
            })
            .ok_or_else(|| EngError::MemoryPlan("step plan: sizes overflow".to_owned()))?;
        Ok(Self {
            rows,
            slots,
            arena,
            unaliased,
        })
    }

    pub(crate) fn tensors(&self) -> &[PlannedSlot] {
        &self.slots
    }

    pub(crate) fn plan(&self) -> &ArenaPlan {
        &self.arena
    }

    /// The list to give `CudaContext::arena`: the device arena then places every tensor at the
    /// offset of this plan (same list, same order, same planner), and `DeviceArena::tensor(index)`
    /// is the range of [`Self::index`].
    pub(crate) fn arena_tensors(&self) -> Vec<ArenaTensor> {
        arena_tensors(&self.slots)
    }

    /// Position of `slot` in [`Self::arena_tensors`], or `None` when the plan has no such slot.
    pub(crate) fn index(&self, slot: Slot) -> Option<usize> {
        self.slots.iter().position(|s| s.slot == slot)
    }

    /// Byte offset of `slot` in the arena, or `None` when the plan has no such slot.
    pub(crate) fn offset(&self, slot: Slot) -> Option<usize> {
        self.arena.offsets.get(self.index(slot)?).copied()
    }

    pub(crate) fn arena_bytes(&self) -> usize {
        self.arena.total
    }

    pub(crate) fn unaliased_bytes(&self) -> usize {
        self.unaliased
    }
}

fn arena_tensors(slots: &[PlannedSlot]) -> Vec<ArenaTensor> {
    slots
        .iter()
        .map(|s| ArenaTensor {
            bytes: s.bytes,
            first: s.first,
            last: s.last,
        })
        .collect()
}

impl fmt::Display for StepPlan {
    /// The arena-plan report line of the qualification run (decision 0047 item 8).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "arena-plan rows={} tensors={} arena_bytes={} unaliased_bytes={}",
            self.rows,
            self.slots.len(),
            self.arena.total,
            self.unaliased
        )
    }
}

/// The slots of one step under construction, with the helpers that size them.
struct SlotList {
    rows: usize,
    slots: Vec<PlannedSlot>,
}

impl SlotList {
    fn add(
        &mut self,
        slot: Slot,
        bytes: Result<usize, EngError>,
        first: usize,
        last: usize,
    ) -> Result<(), EngError> {
        self.slots.push(PlannedSlot {
            slot,
            bytes: bytes?,
            first,
            last,
        });
        Ok(())
    }

    /// f32 values of `width` per row.
    fn row(&self, width: usize) -> Result<usize, EngError> {
        product(&[self.rows, width, F32])
    }

    /// The fp16 pack buffer for `k` values per row.
    fn xf(&self, k: usize) -> Result<usize, EngError> {
        xf_bytes(self.rows, k)
            .ok_or_else(|| EngError::MemoryPlan("step plan: an xf size overflows".to_owned()))
    }
}

fn list_slots(
    dims: &StepDims,
    rows: usize,
    options: &StepOptions,
    scratch: &KernelScratch,
) -> Result<Vec<PlannedSlot>, EngError> {
    let mut list = SlotList {
        rows,
        slots: Vec::new(),
    };
    list_mixers(&mut list, dims, scratch)?;
    list_moe(&mut list, dims, options, scratch)?;
    let h = dims.shape.hidden;
    list.add(Slot::XfFinal, list.xf(h), FINAL_NORM, LM_HEAD)?;
    list.add(Slot::Logits, list.row(dims.shape.vocab), LM_HEAD, ARGMAX)?;
    list.add(Slot::ArgmaxPart, Ok(scratch.argmax_part), ARGMAX, ARGMAX)?;
    Ok(list.slots)
}

/// The residual stream and the tensors of the two mixers.
fn list_mixers(
    list: &mut SlotList,
    dims: &StepDims,
    scratch: &KernelScratch,
) -> Result<(), EngError> {
    let (h, d, a) = (dims.shape.hidden, &dims.shape.delta, &dims.shape.attn);
    let (value_width, q_width) = (d.value_heads * d.value_dim, a.heads_q * a.head_dim);
    list.add(Slot::ResidA, list.row(h), EMBED, ARGMAX)?;
    list.add(Slot::ResidB, list.row(h), EMBED, ARGMAX)?;
    list.add(Slot::XfMix, list.xf(h), MIX_PACK, ATTN_IN)?;
    list.add(
        Slot::DeltaProj,
        list.row(dims.delta_proj),
        DELTA_IN,
        DELTA_RULE,
    )?;
    list.add(
        Slot::DeltaConv,
        list.row(d.conv_dim()),
        DELTA_CONV,
        DELTA_RULE,
    )?;
    list.add(
        Slot::DeltaCore,
        list.row(value_width),
        DELTA_RULE,
        DELTA_PACK,
    )?;
    list.add(
        Slot::XfDeltaOut,
        list.xf(value_width),
        DELTA_PACK,
        DELTA_OUT,
    )?;
    list.add(Slot::AttnProj, list.row(dims.attn_proj), ATTN_IN, ATTN_CORE)?;
    list.add(Slot::AttnQ, list.row(q_width), ATTN_PREP, ATTN_CORE)?;
    list.add(Slot::AttnPart, Ok(scratch.attn_part), ATTN_CORE, ATTN_CORE)?;
    list.add(Slot::AttnCore, list.row(q_width), ATTN_CORE, ATTN_PACK)?;
    list.add(Slot::XfAttnOut, list.xf(q_width), ATTN_PACK, ATTN_OUT)?;
    list.add(Slot::MixOut, list.row(h), DELTA_OUT, ADD_NORM)
}

/// The MoE block: router, routed experts, shared expert.
///
/// The launch order is RD-9c's contract (lane C, phase order, one stream): (1) the router; (2) all
/// expert gate/up matmuls (every slot, gate and up) and the shared gate/up; (3) the shared swiglu;
/// (4) for each slot s, `swiglu_pack(s)` into the one `XfDown`, then `down(s)`; (5) the shared down;
/// (6) the combine. So `XfMoe` is last read in `GateUp`; `ExpertGu` is read by every slot's swiglu
/// inside the `Down` phase and `SharedGu` is kept to `Down` with it; `XfDown` is rewritten per slot
/// (stream order makes swiglu(s), down(s), swiglu(s + 1) safe) and covers the whole `Down` phase.
fn list_moe(
    list: &mut SlotList,
    dims: &StepDims,
    options: &StepOptions,
    scratch: &KernelScratch,
) -> Result<(), EngError> {
    let (h, e) = (dims.shape.hidden, &dims.shape.moe);
    let m = list.rows;
    list.add(Slot::XfMoe, list.xf(h), ADD_NORM, GATE_UP)?;
    list.add(Slot::NormRow, list.row(h), ADD_NORM, ROUTE)?;
    list.add(Slot::RouterLogits, Ok(scratch.router_logits), ROUTE, ROUTE)?;
    list.add(Slot::TopkId, product(&[m, e.top_k, I32]), ROUTE, COMBINE)?;
    list.add(Slot::TopkWeight, list.row(e.top_k), ROUTE, COMBINE)?;
    list.add(Slot::SharedGate, list.row(1), ROUTE, COMBINE)?;
    let gate_up = product(&[m, e.top_k, 2 * e.inter, F32]);
    list.add(Slot::ExpertGu, gate_up, GATE_UP, DOWN)?;
    let branches = if options.forked_experts { e.top_k } else { 1 };
    for branch in 0..branches {
        list.add(Slot::XfDown(branch), list.xf(e.inter), SWIGLU, DOWN)?;
    }
    let down = product(&[m, e.top_k, h, F32]);
    list.add(Slot::ExpertDown, down, DOWN, COMBINE)?;
    list.add(Slot::SharedGu, list.row(2 * e.shared_inter), GATE_UP, DOWN)?;
    list.add(Slot::XfShared, list.xf(e.shared_inter), SWIGLU, DOWN)?;
    list.add(Slot::SharedOut, list.row(h), DOWN, COMBINE)
}
