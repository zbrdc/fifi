//! The pure plan of chunked prefill (decision 0049, packet CPF-1 v1): the two-term chunk
//! policy and the chunk list, the admission rule, and the arena tensor list of one chunk.
//!
//! Host arithmetic only: no CUDA call. It sits in `family::qwen36::cuda`, so it compiles with
//! that module (`native-cuda` on Linux) and not in the default feature set; it calls the
//! `kernels::cuda` sizing functions, and `kernels::cuda` never calls back. Every tensor size is
//! the output of the function that owns the formula (the sister wrappers, `mmq_size`, the
//! router and norm wrappers, `mixer_geometry`); the offsets come from `plan_arena`. Lifetimes
//! are phases of one layer, as in `step_plan`: a layer runs the DeltaNet block (`D*`) or the
//! attention block (`A*`), so their tensors may alias.
//!
//! Admission charges the computed `plan_arena` total plus a flat `part` (the P-b rows of 0049
//! are inputs of the reproduction tests only). The attention `part` buffer is its own
//! allocation outside the arena; its real size is reported in [`ChunkCost`] beside the flat
//! 64 MiB charge, which CPF-1 keeps: lane M decides on the difference (0049 A6 item 8).
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "staged for CPF-7a/7b, the prefill driver (decision 0049)"
    )
)]

use super::{
    dims::{KvMode, StepDims, StepOptions},
    mixer_geometry::MixerGeometry,
    state_layout::{KvPool, StateLayout},
};
use crate::{
    EngError,
    family::project::cuda::WeightLayout,
    kernels::cuda::{
        ArenaTensor, plan_arena,
        prefill::{
            attn_prefill::{CPF_ATTN_PART_CAP, CPF_ATTN_Q_TILE, cpf_attn_workspace_bytes_for_sm},
            gdn_scan_seq::{cpf_dn_conv_out_bytes, cpf_dn_scan_out_bytes},
            mmq_size::{
                check_ncols_domain, expert_bounds_ints, fixup_floats_max, ids_dst_ints,
                ids_src1_ints, q8_1_bytes,
            },
        },
        resident::{
            add_norm_pack::row_bytes,
            attn_prep::q_bytes,
            router::{RouterShape, router_logits_bytes},
        },
        xf_bytes,
    },
};

/// The chunk sizes a session arena may be built for, largest first (0049 item 9). The largest
/// is also the default of the `qwen36.prefill_chunk` cap: absent means no cap.
pub(crate) const M_LADDER: [usize; 5] = [2048, 1024, 512, 256, 128];
/// Where `M_pos` drops to 1,024 and to 512 rows: 0049's "3K" and "32K" read as 3 * 1,024 and
/// 32 * 1,024 positions (both readings give the same chunk list for the 16K and 128K prompts).
const TIER_1024_FROM: usize = 3 * 1024;
const TIER_512_FROM: usize = 32 * 1024;
/// A chunk size is admitted only if at least this many device bytes stay spare (0049 item 9).
pub(crate) const SPARE_MIN: usize = 64 << 20;
/// The flat device charge for the attention `part` buffer in every admission row (0049 item 9).
pub(crate) const PART_CHARGE: usize = CPF_ATTN_PART_CAP;

fn overflow() -> EngError {
    EngError::MemoryPlan("prefill plan: a size overflows".to_owned())
}

/// Bytes of `parts` 4-byte words multiplied together (f32 and i32 tensors).
fn word_bytes(parts: &[usize]) -> Result<usize, EngError> {
    parts
        .iter()
        .try_fold(4usize, |total, &part| total.checked_mul(part))
        .ok_or_else(overflow)
}

/// `M_pos(p0)`: the chunk size the position asks for, 2,048 below 3K, 1,024 from 3K to 32K,
/// 512 beyond (floor 512). `p0` is the position of the chunk's first row.
pub(crate) fn m_pos(p0: usize) -> usize {
    if p0 < TIER_1024_FROM {
        2048
    } else if p0 < TIER_512_FROM {
        1024
    } else {
        512
    }
}

/// One step of a prefill run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrefillStep {
    /// `rows >= 2` rows from position `p0` through the chunk layer loop.
    Chunk { p0: usize, rows: usize },
    /// A run of exactly one row at `p0` goes through `feed`, not a chunk (the Flash-Next
    /// precedent), so a 1-row tail never needs a chunk plan.
    Feed { p0: usize },
}

/// The steps that feed `n_rows` rows from position `p0` (`feed_rows` gets tokens `0..n-1`; the
/// last prompt token goes through decode). Each chunk has `min(M_pos(p0), m_sess, rows left)`
/// rows, `m_sess` being the session's arena size (already capped by `qwen36.prefill_chunk`).
pub(crate) fn chunk_list(
    p0: usize,
    n_rows: usize,
    m_sess: usize,
) -> Result<Vec<PrefillStep>, EngError> {
    if !M_LADDER.contains(&m_sess) {
        return Err(EngError::BadRequest(format!(
            "prefill plan: chunk size {m_sess} is not one of {M_LADDER:?}"
        )));
    }
    let end = p0.checked_add(n_rows).ok_or_else(overflow)?;
    let mut steps = Vec::new();
    let mut at = p0;
    while at < end {
        let rows = m_pos(at).min(m_sess).min(end - at);
        steps.push(match rows {
            1 => PrefillStep::Feed { p0: at },
            _ => PrefillStep::Chunk { p0: at, rows },
        });
        at += rows;
    }
    Ok(steps)
}

/// What chunk size `m` costs the device beyond KV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChunkCost {
    pub(crate) m: usize,
    /// The `plan_arena` total of the chunk's tensor list.
    pub(crate) arena: usize,
    /// Bytes of the attention `part` buffer of the `M_sess` split plan, reported for lane M's
    /// decision on the flat charge (0049 A6 item 8). The driver allocates the buffer at
    /// [`PART_CHARGE`] (= `CPF_ATTN_PART_CAP`), never at this figure: a smaller chunk
    /// launched with its own split plan can need more than the `M_sess` plan (209 rows in a
    /// 256-row session: 14,266,368 B against 12,681,216; 641 rows in a 1,024-row session:
    /// 21,399,552 against 16,908,288), and the flat cap covers both.
    pub(crate) part: usize,
}

/// One extra tensor of the admission sum (seam I4): `fixed + bytes_per_token * M` bytes, such
/// as the tiered `partial [M][2048]` f32 (8,192 B per token) or the expert bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Extra {
    pub(crate) name: &'static str,
    pub(crate) fixed: usize,
    pub(crate) bytes_per_token: usize,
}

fn extras_bytes(extras: &[Extra], m: usize) -> Result<usize, EngError> {
    let mut total = 0usize;
    for extra in extras {
        total = extra
            .bytes_per_token
            .checked_mul(m)
            .and_then(|bytes| bytes.checked_add(extra.fixed))
            .and_then(|bytes| bytes.checked_add(total))
            .ok_or_else(|| {
                EngError::MemoryPlan(format!("prefill plan: extra {} overflows", extra.name))
            })?;
    }
    Ok(total)
}

/// The admission inputs that do not depend on the chunk size or the window (0049 item 9).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Admission {
    /// Device bytes for the resident state ([`state_bytes`]), the decode arena, the prefill
    /// arena and `part`: free VRAM measured ONCE by the caller right after the weight load
    /// and BEFORE the resident state (the DeltaNet pool, the KV and the words, which
    /// `ResidentState::allocate` takes together) and the decode-arena allocations, less the
    /// headroom (never a literal here); the state and the decode arena are then charged as
    /// their own inputs (lane C ruling, 2026-10-14 00:18, option (b): a figure from before
    /// the KV is from before the pool too). The snapshot must follow the loading of every
    /// kernel module: until KV8-B B4 sets CUDA_MODULE_LOADING=EAGER (or one warm-up launch
    /// per entry) at start, a lazily loaded module can take VRAM after the snapshot, and the
    /// headroom has to cover it (lane C ruling, 2026-10-13 23:55). The step's small session
    /// resources — the stream, the event, the pinned word and the kernel scratch (`step.rs`)
    /// — also stay inside that headroom; the decode arena keeps its own separate charge
    /// (real memory, not headroom).
    pub(crate) device_budget: usize,
    /// `DeviceArena::bytes()` of the decode plan, read at run time (0049 A6 item 6).
    pub(crate) decode_arena: usize,
    /// The flat `part` charge, [`PART_CHARGE`] in every row.
    pub(crate) part_charge: usize,
    /// Fallback buffers when one is active; 0 for Qwen3.6 Native (A4).
    pub(crate) fallback: usize,
    pub(crate) device_extras: Vec<Extra>,
    /// Pinned host bytes available, a separate budget: host extras never enter the device sum.
    pub(crate) host_budget: usize,
    pub(crate) host_extras: Vec<Extra>,
}

impl Admission {
    /// The rule with the flat `part` charge and no decode arena, fallback or extras.
    pub(crate) fn new(device_budget: usize) -> Self {
        Self {
            device_budget,
            decode_arena: 0,
            part_charge: PART_CHARGE,
            fallback: 0,
            device_extras: Vec::new(),
            host_budget: 0,
            host_extras: Vec::new(),
        }
    }

    /// Whether chunk size `cost.m` is admitted with `state` bytes of resident state
    /// ([`state_bytes`]): the device sum (the decode arena, the state, the arena, the larger
    /// of the flat and the real `part`, the fallback and the device extras) leaves at least
    /// [`SPARE_MIN`] spare and the pinned host rows fit the host budget.
    pub(crate) fn passes(&self, state: usize, cost: &ChunkCost) -> Result<bool, EngError> {
        // The part term is the larger of the flat charge and the chunk's real part: the flat
        // charge is the floor, so a later lowering of `part_charge` (lane M, A6 item 8)
        // cannot undercharge a chunk whose own split plan needs more.
        let part = self.part_charge.max(cost.part);
        let parts = [
            self.decode_arena,
            state,
            cost.arena,
            part,
            self.fallback,
            extras_bytes(&self.device_extras, cost.m)?,
        ];
        let charged = parts
            .iter()
            .try_fold(0usize, |sum, &part| sum.checked_add(part));
        let spare = self
            .device_budget
            .checked_sub(charged.ok_or_else(overflow)?);
        let host = extras_bytes(&self.host_extras, cost.m)?;
        Ok(spare.is_some_and(|spare| spare >= SPARE_MIN) && host <= self.host_budget)
    }

    /// `M_sess`: the largest of `costs` that is admitted and at most `cap` (the
    /// `qwen36.prefill_chunk` member; `None` is no cap, 2,048). `None` means no size passes
    /// and the session keeps the per-token path. The budget is the caller's snapshot of free
    /// VRAM after the weight load and before the resident state and the decode-arena
    /// allocations, less the headroom (`device_budget`), and that snapshot must follow the
    /// loading of every kernel module: until KV8-B B4 loads eagerly, a lazily loaded module
    /// can take VRAM after it and the headroom has to cover that too (lane C ruling,
    /// 2026-10-13 23:55).
    pub(crate) fn admit(
        &self,
        state: usize,
        costs: &[ChunkCost],
        cap: Option<usize>,
    ) -> Result<Option<usize>, EngError> {
        let cap = cap.unwrap_or(M_LADDER[0]);
        if !M_LADDER.contains(&cap) {
            return Err(EngError::BadRequest(format!(
                "prefill plan: prefill_chunk {cap} is not one of {M_LADDER:?}"
            )));
        }
        let mut best = None;
        for cost in costs.iter().filter(|cost| cost.m <= cap) {
            if self.passes(state, cost)? {
                best = best.max(Some(cost.m));
            }
        }
        Ok(best)
    }
}

/// KV bytes of a session of `window` positions: the pool's exact bytes per position, without
/// `KvPool`'s per-plane `STATE_ALIGN` rounding (at most 255 B a plane, 40 planes at
/// `Int8G32` — about 10 KiB; zero at `Fp16` on this model, whose planes are whole multiples
/// of the alignment). The admission state term is the whole layout ([`state_bytes`]); this
/// exact figure is the 0049-reproduction input.
pub(crate) fn kv_bytes(dims: &StepDims, mode: KvMode, window: usize) -> Result<usize, EngError> {
    KvPool::bytes_per_token(dims, mode)?
        .checked_mul(window)
        .ok_or_else(overflow)
}

/// The state term of the admission sum: [`StateLayout::bytes`] of a session of `window`
/// positions and `delta_slots` DeltaNet slots — the pool, the KV and the persistent words —
/// because `ResidentState::allocate` allocates all three together after the budget snapshot
/// (lane C ruling, 2026-10-14 00:18, option (b)). The KV mode flows through the layout, so
/// `Fp16` and `Int8G32` keep their own figures; the decode arena, the workspace and a
/// prefill arena are separate terms (the [`StateLayout::bytes`] doc). The options fields no
/// pool reads (`forked_experts`, `tiered`, `layout`) sit at the stage-1 defaults.
pub(crate) fn state_bytes(
    dims: &StepDims,
    mode: KvMode,
    window: usize,
    delta_slots: usize,
) -> Result<usize, EngError> {
    let options = StepOptions {
        kv_mode: mode,
        context_tokens: window,
        delta_slots,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Tiled,
    };
    StateLayout::new(dims, &options)?.bytes()
}

/// Phases of one layer of a chunk, in launch order (the `step_plan` style): the lifetime
/// numbers of the tensors. `D*` is the DeltaNet block, `A*` the attention block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    Embed,
    NormA,
    QuantA,
    DProj,
    DConv,
    DScan,
    DQuant,
    DOut,
    AProj,
    APrep,
    AFlash,
    AQuant,
    AOut,
    NormF,
    Route,
    Shared,
    Ids,
    Scatter,
    GateUp,
    Swiglu,
    QuantAct,
    Down,
    Combine,
}

/// The arena tensors of one chunk (plan section C.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChunkSlot {
    Hidden,
    IdsTok,
    X,
    X2,
    XfA,
    XfF,
    Q8X,
    Q8X2,
    ProjD,
    ConvOut,
    OOut,
    Q8OD,
    Q8OA,
    ProjA,
    Q,
    AttnOut,
    Delta,
    Logits,
    TopkId,
    TopkWeight,
    SharedGate,
    GuSh,
    ActSh,
    Q8Ash,
    ShOut,
    IdsG,
    IdsSrc1,
    IdsDst,
    ExpertBounds,
    Q8S,
    Gu,
    Act,
    Q8Act,
    Down,
    Fixup,
}

type Row = (ChunkSlot, usize, Phase, Phase);

/// The tensors that live across blocks: the residual stream, the norm rows and their `xf`
/// sinks (written, never read in prefill), the first quantize and the stream-k fixup range.
fn common_rows(g: &MixerGeometry, m: usize, nsm: usize) -> Result<Vec<Row>, EngError> {
    use {ChunkSlot as S, Phase::*};
    let hidden = row_bytes(m, g.hidden).ok_or_else(overflow)?;
    // `xf_bytes` is not monotone in m: a chunk of at most 16 rows packs the whole fixed
    // fragment (65,537 * 16 B, `native.rs`), so each sink planned at `m` rows also carries
    // the 1-row figure, or a smaller chunk of the session would not fit its slot (M1).
    // `xf_bytes` is not monotone in m: a chunk of at most 16 rows packs the whole fixed
    // fragment (65,537 * 16 B, `native.rs`), so each sink planned at `m` rows also carries
    // the 1-row figure, or a smaller chunk of the session would not fit its slot (M1).
    let xf = xf_bytes(m, g.hidden)
        .zip(xf_bytes(1, g.hidden))
        .map(|(bytes, fragment)| bytes.max(fragment))
        .ok_or_else(overflow)?;
    Ok(vec![
        (S::Hidden, hidden, Embed, Combine),
        (S::IdsTok, word_bytes(&[m])?, Embed, Embed),
        (S::X, hidden, NormA, QuantA),
        (S::XfA, xf, NormA, NormA),
        (S::Q8X, q8_1_bytes(g.hidden, m)?, QuantA, AProj),
        (S::Delta, hidden, DOut, NormF),
        (S::X2, hidden, NormF, Scatter),
        (S::XfF, xf, NormF, NormF),
        (
            S::Fixup,
            word_bytes(&[fixup_floats_max(nsm)?])?,
            Embed,
            Combine,
        ),
    ])
}

/// The two mixer blocks, on disjoint phase ranges.
fn mixer_rows(g: &MixerGeometry, m: usize) -> Result<Vec<Row>, EngError> {
    use {ChunkSlot as S, Phase::*};
    let (d, a) = (&g.delta, &g.attn);
    Ok(vec![
        (S::ProjD, word_bytes(&[m, d.ld])?, DProj, DScan),
        (
            S::ConvOut,
            cpf_dn_conv_out_bytes(m, d.channels)?,
            DConv,
            DScan,
        ),
        (
            S::OOut,
            cpf_dn_scan_out_bytes(m, d.value_heads, d.value_dim)?,
            DScan,
            DQuant,
        ),
        (S::Q8OD, q8_1_bytes(d.value_width, m)?, DQuant, DOut),
        (S::ProjA, word_bytes(&[m, a.ld])?, AProj, AFlash),
        (S::Q, q_bytes(m, a.heads_q), APrep, AFlash),
        (S::AttnOut, word_bytes(&[m, a.q_width])?, AFlash, AQuant),
        (S::Q8OA, q8_1_bytes(a.q_width, m)?, AQuant, AOut),
    ])
}

/// The Resident MoE block: router, shared expert, ids, scatter quantize, gate/up, down.
/// `routed` is the grouped column count `M * top_k`.
fn moe_rows(dims: &StepDims, m: usize) -> Result<Vec<Row>, EngError> {
    use {ChunkSlot as S, Phase::*};
    let (h, e) = (dims.shape.hidden, &dims.shape.moe);
    let routed = m.checked_mul(e.top_k).ok_or_else(overflow)?;
    let picks = word_bytes(&[m, e.top_k])?;
    let router = RouterShape {
        m,
        hidden: h,
        n_expert: e.experts,
        top_k: e.top_k,
    };
    Ok(vec![
        (S::Logits, router_logits_bytes(router)?, Route, Route),
        (S::TopkId, picks, Route, Combine),
        (S::TopkWeight, picks, Route, Combine),
        (S::SharedGate, word_bytes(&[m])?, Route, Combine),
        (S::Q8X2, q8_1_bytes(h, m)?, Shared, Shared),
        (
            S::GuSh,
            word_bytes(&[m, 2 * e.shared_inter])?,
            Shared,
            Shared,
        ),
        (S::ActSh, word_bytes(&[m, e.shared_inter])?, Shared, Shared),
        (S::Q8Ash, q8_1_bytes(e.shared_inter, m)?, Shared, Shared),
        (S::ShOut, word_bytes(&[m, h])?, Shared, Combine),
        (S::IdsG, picks, Ids, Ids),
        (
            S::IdsSrc1,
            word_bytes(&[ids_src1_ints(routed)])?,
            Ids,
            Scatter,
        ),
        (
            S::IdsDst,
            word_bytes(&[ids_dst_ints(routed, m)?])?,
            Ids,
            Down,
        ),
        (
            S::ExpertBounds,
            word_bytes(&[expert_bounds_ints(e.experts)?])?,
            Ids,
            Down,
        ),
        (S::Q8S, q8_1_bytes(h, routed)?, Scatter, GateUp),
        (S::Gu, word_bytes(&[routed, 2 * e.inter])?, GateUp, Swiglu),
        (S::Act, word_bytes(&[routed, e.inter])?, Swiglu, QuantAct),
        (S::Q8Act, q8_1_bytes(e.inter, routed)?, QuantAct, Down),
        (S::Down, word_bytes(&[routed, h])?, Down, Combine),
    ])
}

/// The tensors of a chunk of `m` rows, in the order of the plan table, for a device with `nsm`
/// SMs (the runtime device fact that sizes the fixup range; never a literal here). `m` is in
/// `1..=2048` (the chunk cap of `mmq_size`). Smaller chunks of a session use prefixes of the
/// tensors planned for its `M_sess`: every slot grows with `m` except the `xf` sinks, whose
/// planned size carries the 1-row fragment floor because `xf_bytes` is not monotone in `m`.
pub(crate) fn chunk_tensors(
    dims: &StepDims,
    m: usize,
    nsm: usize,
) -> Result<Vec<(ChunkSlot, ArenaTensor)>, EngError> {
    check_ncols_domain(m)?;
    if nsm == 0 {
        return Err(EngError::BadRequest(format!(
            "prefill plan: nsm {nsm} is below 1; the fixup range needs the device SM count"
        )));
    }
    let g = MixerGeometry::new(dims)?;
    let rows = [
        common_rows(&g, m, nsm)?,
        mixer_rows(&g, m)?,
        moe_rows(dims, m)?,
    ];
    Ok(rows
        .into_iter()
        .flatten()
        .map(|(slot, bytes, first, last)| {
            let (first, last) = (first as usize, last as usize);
            (slot, ArenaTensor { bytes, first, last })
        })
        .collect())
}

/// The cost of one chunk size: the `plan_arena` total of [`chunk_tensors`] and the real bytes of
/// the attention `part` buffer, whose split plan depends on `nsm` and not on the key count.
pub(crate) fn chunk_cost(dims: &StepDims, m: usize, nsm: usize) -> Result<ChunkCost, EngError> {
    let tensors: Vec<ArenaTensor> = chunk_tensors(dims, m, nsm)?
        .into_iter()
        .map(|(_, tensor)| tensor)
        .collect();
    let a = &dims.shape.attn;
    let heads = (a.heads_q, a.heads_kv);
    Ok(ChunkCost {
        m,
        arena: plan_arena(&tensors)?.total,
        part: cpf_attn_workspace_bytes_for_sm(m, heads, a.head_dim, m, CPF_ATTN_Q_TILE, nsm)?,
    })
}

/// The costs of every size of [`M_LADDER`]: the candidates [`Admission::admit`] chooses from.
pub(crate) fn chunk_costs(dims: &StepDims, nsm: usize) -> Result<Vec<ChunkCost>, EngError> {
    M_LADDER.iter().map(|&m| chunk_cost(dims, m, nsm)).collect()
}
