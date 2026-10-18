//! The chunk layer loop of decision 0049 (CPF-7a part 2): plan §C.1 stages P1 and 1-17 of one
//! prefill chunk, ENQUEUE ONLY, on the state's owning stream.
//!
//! CPF-7b's driver admits the chunk, allocates the arena from
//! [`super::prefill_plan::chunk_tensors`], fills the attention `part` buffer, uploads the ids
//! and owns the §C.4 protocol around [`run_chunk_body`]. This module holds the slot views, the
//! checked chunk rows, the pure call-shape builders, the stage lists, the constructor checks
//! and the stage functions themselves. No function here allocates device memory, copies to or
//! from the host, reads a device word, records an event or waits for anything (the one host
//! guard of the body is [`super::state::ResidentState::check_chunk`]).
//!
//! Design guidance only, no code copied.

use super::{
    dims::{KvMode, StepDims},
    layer_table::LayerTable,
    mixer::tail_extent,
    mixer_geometry::MixerGeometry,
    mixer_weights::{DenseWeight, LayerWeights, MixerWeights},
    moe::{Launch, MoeDims, MoeLayerWeights, StackWeights},
    prefill_plan::{ChunkSlot, Phase},
    state::ResidentState,
    state_layout::{Plane, Word},
    tiered_step::TieredRuntime,
};
use crate::family::spec::AttentionKind;
use crate::{
    EngError,
    family::gguf_model::Role,
    kernels::cuda::{
        ArenaTensor, DeviceArena, DeviceBuffer, DeviceRange, NativeWeight, plan_arena,
        prefill::{
            attn_prefill::{
                AttnPrefillCall, AttnPrefillShape, CPF_ATTN_PART_CAP, CPF_ATTN_Q_TILE,
                CpfAttnGeometry, CpfAttnSpan, CpfAttnSplit, cpf_attn_launch_plan,
            },
            attn_prep_chunk::{AttnPrepChunkCall, AttnPrepChunkShape, PrepChunkMode},
            gdn_scan_seq::{GDN_SCAN_EPS, GdnScanSeqCall, ScanShape, ScanWeights},
            mmq_dense::{MmqDenseCall, MmqDenseInputCall},
            mmq_facts::MmqFacts,
            mmq_mmid::{
                IdsGuardCall, MmidIdsOut, MmidLaunchCall, MmidQuantCall, MmidShape, MmidWeight,
                mmq_type_of,
            },
            mmq_size::{NCOLS_LIMIT, fixup_floats_max},
            swiglu_rows::{SwigluRowsCall, SwigluRowsShape},
        },
        resident::{
            add_norm_pack::AddNormCall,
            attn_prep::RopeParams,
            embed::EmbedNativeCall,
            gdn_conv::{ConvCall, ConvShape},
            matvec_q::QuantType,
            moe_combine::{CombineCall, CombineOut, CombineShape, SharedTerm},
            router::{RouterCall, RouterOut, RouterShape, RouterWeights},
        },
    },
};

/// Every slot of the chunk arena exactly once, in `chunk_tensors` order (B.2): the view checks
/// take the arena as the plan of exactly this set, so a slot added to the plan without this
/// module fails [`super::prefill_tests`].
pub(super) const ALL_SLOTS: [ChunkSlot; 35] = [
    ChunkSlot::Hidden,
    ChunkSlot::IdsTok,
    ChunkSlot::X,
    ChunkSlot::XfA,
    ChunkSlot::Q8X,
    ChunkSlot::Delta,
    ChunkSlot::X2,
    ChunkSlot::XfF,
    ChunkSlot::Fixup,
    ChunkSlot::ProjD,
    ChunkSlot::ConvOut,
    ChunkSlot::OOut,
    ChunkSlot::Q8OD,
    ChunkSlot::ProjA,
    ChunkSlot::Q,
    ChunkSlot::AttnOut,
    ChunkSlot::Q8OA,
    ChunkSlot::Logits,
    ChunkSlot::TopkId,
    ChunkSlot::TopkWeight,
    ChunkSlot::SharedGate,
    ChunkSlot::Q8X2,
    ChunkSlot::GuSh,
    ChunkSlot::ActSh,
    ChunkSlot::Q8Ash,
    ChunkSlot::ShOut,
    ChunkSlot::IdsG,
    ChunkSlot::IdsSrc1,
    ChunkSlot::IdsDst,
    ChunkSlot::ExpertBounds,
    ChunkSlot::Q8S,
    ChunkSlot::Gu,
    ChunkSlot::Act,
    ChunkSlot::Q8Act,
    ChunkSlot::Down,
];

/// The rows of one chunk: `m` rows at the positions `p0..p0 + m` of a session whose arena was
/// planned for `m_sess` rows (B.3). Constructed against the caller's `cap` and `m_sess`; the
/// body re-checks the state's own cap (a chunk built against a larger cap than the state's).
#[derive(Debug)]
pub(super) struct ChunkRows {
    p0: usize,
    m: usize,
}

impl ChunkRows {
    /// Guards, in order: (R1) `m` in `1..=m_sess`; (R2) `p0 + m` must not overflow; (R3)
    /// `p0 + m <= cap`, else `ContextLimit`.
    pub(super) fn new(p0: usize, m: usize, cap: usize, m_sess: usize) -> Result<Self, EngError> {
        if m == 0 || m > m_sess {
            return Err(EngError::BadRequest(format!(
                "prefill chunk: rows: {m} rows of a session planned for {m_sess}"
            )));
        }
        let end = p0.checked_add(m).ok_or_else(|| {
            EngError::BadRequest("prefill chunk: rows: p0 + m overflows".to_owned())
        })?;
        if end > cap {
            return Err(EngError::ContextLimit(format!(
                "prefill chunk: rows: p0 {p0} + m {m} passes the context cap {cap}"
            )));
        }
        Ok(Self { p0, m })
    }

    /// The chunk's first position.
    pub(super) fn p0(&self) -> usize {
        self.p0
    }

    /// The chunk's rows.
    pub(super) fn m(&self) -> usize {
        self.m
    }

    /// One past the chunk's last position.
    pub(super) fn end(&self) -> usize {
        self.p0 + self.m
    }
}

/// The index of `slot` in `slots`, or the `is not a view` refusal (B.2).
pub(super) fn slot_position(slots: &[ChunkSlot], slot: ChunkSlot) -> Result<usize, EngError> {
    slots
        .iter()
        .position(|known| *known == slot)
        .ok_or_else(|| not_a_view(slot))
}

/// The refusal of a slot the views do not hold: one text for the list lookup and the view lookup.
fn not_a_view(slot: ChunkSlot) -> EngError {
    EngError::BadRequest(format!("prefill chunk: slots: {slot:?} is not a view"))
}

/// (V1) the arena holds exactly the list's tensors; (V2) no slot twice; (V3) every slot of
/// [`ALL_SLOTS`] present. The order is binding: every refusal names its guard.
pub(super) fn check_view_list(arena_len: usize, slots: &[ChunkSlot]) -> Result<(), EngError> {
    if arena_len != slots.len() {
        return Err(EngError::BadRequest(format!(
            "prefill chunk: slots: the arena holds {arena_len} tensors, the list {}",
            slots.len()
        )));
    }
    for (index, slot) in slots.iter().enumerate() {
        if slots[..index].contains(slot) {
            return Err(EngError::BadRequest(format!(
                "prefill chunk: slots: {slot:?} appears twice"
            )));
        }
    }
    for slot in ALL_SLOTS {
        if !slots.contains(&slot) {
            return Err(EngError::BadRequest(format!(
                "prefill chunk: slots: {slot:?} is missing"
            )));
        }
    }
    Ok(())
}

/// (V4) after V1-V3: each view sits exactly where `plan_arena` puts the list's tensors, so the
/// arena IS the plan of this list and the lifetimes the plan honoured are the list's. A bytes
/// check alone cannot see two same-size slots in swapped order; the offset half can.
pub(super) fn check_view_layout(
    list: &[(ChunkSlot, ArenaTensor)],
    actual: &[(usize, usize)],
) -> Result<(), EngError> {
    let tensors: Vec<ArenaTensor> = list.iter().map(|(_, tensor)| *tensor).collect();
    let plan = plan_arena(&tensors)?;
    for (index, (slot, tensor)) in list.iter().enumerate() {
        let (planned, bytes) = (plan.offsets[index], tensor.bytes);
        let (offset, view_bytes) = actual[index];
        if (planned, bytes) != (offset, view_bytes) {
            return Err(EngError::BadRequest(format!(
                "prefill chunk: slots: {slot:?} is planned at {planned} with {bytes} bytes, \
                 the arena view is at {offset} with {view_bytes} bytes"
            )));
        }
    }
    Ok(())
}

/// The arena views of one chunk's slots (B.2): built from the same `chunk_tensors` list the
/// arena was planned from, slots and tensors together.
pub(super) struct ChunkViews<'a> {
    views: Vec<(ChunkSlot, DeviceRange<'a>)>,
}

impl<'a> ChunkViews<'a> {
    /// Runs [`check_view_list`], reads every view `arena.tensor(i)` and runs
    /// [`check_view_layout`] against the plan of the list itself.
    pub(super) fn new(
        list: &[(ChunkSlot, ArenaTensor)],
        arena: &'a DeviceArena,
    ) -> Result<Self, EngError> {
        let slots: Vec<ChunkSlot> = list.iter().map(|(slot, _)| *slot).collect();
        check_view_list(arena.len(), &slots)?;
        let mut views = Vec::with_capacity(list.len());
        let mut actual = Vec::with_capacity(list.len());
        for (index, (slot, _)) in list.iter().enumerate() {
            let view = arena.tensor(index)?;
            actual.push((view.offset(), view.bytes()));
            views.push((*slot, view));
        }
        check_view_layout(list, &actual)?;
        Ok(Self { views })
    }

    /// The arena view of `slot`, with the refusal of [`slot_position`] but without a list of the
    /// slots: the stage functions call this a few thousand times per chunk.
    pub(super) fn get(&self, slot: ChunkSlot) -> Result<DeviceRange<'a>, EngError> {
        self.views
            .iter()
            .find(|(held, _)| *held == slot)
            .map(|(_, view)| *view)
            .ok_or_else(|| not_a_view(slot))
    }

    /// The rig seam (the `MixerWeights::from_layers` precedent): views without a list or
    /// layout check, so a device rig can pass only the slots it uses.
    #[cfg(test)]
    pub(super) fn from_ranges(views: Vec<(ChunkSlot, DeviceRange<'a>)>) -> Self {
        Self { views }
    }
}

/// The stages of one mixer layer of a chunk (plan §C.1), in launch order; `NormA`, `QuantA`,
/// `NormF`, `Route` and `Moe` are shared between the two kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Stage {
    NormA,
    QuantA,
    DeltaProj,
    Conv,
    Scan,
    QuantDOut,
    SsmOut,
    AttnProj,
    Prep,
    Flash,
    QuantAOut,
    AttnOut,
    NormF,
    Route,
    Moe,
}

/// The stages of the Resident MoE arm (10-17), in launch order; the order is binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MoeStage {
    SharedQuant,
    SharedGate,
    SharedUp,
    SharedSwiglu,
    SharedDownQuant,
    SharedDown,
    Ids,
    Scatter,
    Gate,
    Up,
    Swiglu,
    GatherQuant,
    Down,
    Combine,
}

/// The stage list of one layer of `kind`; any other kind refuses `prefill chunk: kind`.
pub(super) fn layer_stages(kind: AttentionKind) -> Result<Vec<Stage>, EngError> {
    match kind {
        AttentionKind::DeltaNet => Ok(vec![
            Stage::NormA,
            Stage::QuantA,
            Stage::DeltaProj,
            Stage::Conv,
            Stage::Scan,
            Stage::QuantDOut,
            Stage::SsmOut,
            Stage::NormF,
            Stage::Route,
            Stage::Moe,
        ]),
        AttentionKind::Full => Ok(vec![
            Stage::NormA,
            Stage::QuantA,
            Stage::AttnProj,
            Stage::Prep,
            Stage::Flash,
            Stage::QuantAOut,
            Stage::AttnOut,
            Stage::NormF,
            Stage::Route,
            Stage::Moe,
        ]),
        AttentionKind::Qsa => Err(EngError::BadRequest(
            "prefill chunk: kind: the Qsa mixer has no chunk stage list".to_owned(),
        )),
    }
}

/// The stage list of the Resident MoE arm (10a-f, 11-17).
pub(super) fn resident_moe_stages() -> [MoeStage; 14] {
    [
        MoeStage::SharedQuant,
        MoeStage::SharedGate,
        MoeStage::SharedUp,
        MoeStage::SharedSwiglu,
        MoeStage::SharedDownQuant,
        MoeStage::SharedDown,
        MoeStage::Ids,
        MoeStage::Scatter,
        MoeStage::Gate,
        MoeStage::Up,
        MoeStage::Swiglu,
        MoeStage::GatherQuant,
        MoeStage::Down,
        MoeStage::Combine,
    ]
}

/// The phase and slot columns of stage P1 (the embedding, before layer 0).
pub(super) const EMBED_PHASE: Phase = Phase::Embed;
/// Reads `IdsTok`, writes `Hidden` (B.11).
pub(super) const EMBED_SLOTS: (&[ChunkSlot], &[ChunkSlot]) =
    (&[ChunkSlot::IdsTok], &[ChunkSlot::Hidden]);

/// The phase a stage runs at (B.11). `Moe` runs at the block's first phase, `Shared`.
pub(super) fn stage_phase(stage: Stage) -> Phase {
    match stage {
        Stage::NormA => Phase::NormA,
        Stage::QuantA => Phase::QuantA,
        Stage::DeltaProj => Phase::DProj,
        Stage::Conv => Phase::DConv,
        Stage::Scan => Phase::DScan,
        Stage::QuantDOut => Phase::DQuant,
        Stage::SsmOut => Phase::DOut,
        Stage::AttnProj => Phase::AProj,
        Stage::Prep => Phase::APrep,
        Stage::Flash => Phase::AFlash,
        Stage::QuantAOut => Phase::AQuant,
        Stage::AttnOut => Phase::AOut,
        Stage::NormF => Phase::NormF,
        Stage::Route => Phase::Route,
        Stage::Moe => Phase::Shared,
    }
}

/// The phase a MoE stage runs at (B.11).
pub(super) fn moe_stage_phase(stage: MoeStage) -> Phase {
    match stage {
        MoeStage::SharedQuant
        | MoeStage::SharedGate
        | MoeStage::SharedUp
        | MoeStage::SharedSwiglu
        | MoeStage::SharedDownQuant
        | MoeStage::SharedDown => Phase::Shared,
        MoeStage::Ids => Phase::Ids,
        MoeStage::Scatter => Phase::Scatter,
        MoeStage::Gate | MoeStage::Up => Phase::GateUp,
        MoeStage::Swiglu => Phase::Swiglu,
        MoeStage::GatherQuant => Phase::QuantAct,
        MoeStage::Down => Phase::Down,
        MoeStage::Combine => Phase::Combine,
    }
}

/// The arena slots a stage reads and writes (B.11); the persistent state, the weights and
/// `part` are not arena slots.
pub(super) fn stage_slots(stage: Stage) -> (&'static [ChunkSlot], &'static [ChunkSlot]) {
    match stage {
        Stage::NormA => (
            &[ChunkSlot::Hidden],
            &[ChunkSlot::Hidden, ChunkSlot::XfA, ChunkSlot::X],
        ),
        Stage::QuantA => (&[ChunkSlot::X], &[ChunkSlot::Q8X]),
        Stage::DeltaProj => (&[ChunkSlot::Q8X], &[ChunkSlot::ProjD, ChunkSlot::Fixup]),
        Stage::Conv => (&[ChunkSlot::ProjD], &[ChunkSlot::ConvOut]),
        Stage::Scan => (&[ChunkSlot::ConvOut, ChunkSlot::ProjD], &[ChunkSlot::OOut]),
        Stage::QuantDOut => (&[ChunkSlot::OOut], &[ChunkSlot::Q8OD]),
        Stage::SsmOut => (&[ChunkSlot::Q8OD], &[ChunkSlot::Delta, ChunkSlot::Fixup]),
        Stage::AttnProj => (&[ChunkSlot::Q8X], &[ChunkSlot::ProjA, ChunkSlot::Fixup]),
        Stage::Prep => (&[ChunkSlot::ProjA], &[ChunkSlot::Q]),
        Stage::Flash => (&[ChunkSlot::Q, ChunkSlot::ProjA], &[ChunkSlot::AttnOut]),
        Stage::QuantAOut => (&[ChunkSlot::AttnOut], &[ChunkSlot::Q8OA]),
        Stage::AttnOut => (&[ChunkSlot::Q8OA], &[ChunkSlot::Delta, ChunkSlot::Fixup]),
        Stage::NormF => (
            &[ChunkSlot::Hidden, ChunkSlot::Delta],
            &[ChunkSlot::Hidden, ChunkSlot::XfF, ChunkSlot::X2],
        ),
        Stage::Route => (
            &[ChunkSlot::X2],
            &[
                ChunkSlot::Logits,
                ChunkSlot::TopkId,
                ChunkSlot::TopkWeight,
                ChunkSlot::SharedGate,
            ],
        ),
        Stage::Moe => (
            &[
                ChunkSlot::X2,
                ChunkSlot::TopkId,
                ChunkSlot::TopkWeight,
                ChunkSlot::SharedGate,
                ChunkSlot::Hidden,
            ],
            &[ChunkSlot::Hidden],
        ),
    }
}

/// The arena slots a MoE stage reads and writes (B.11).
pub(super) fn moe_stage_slots(stage: MoeStage) -> (&'static [ChunkSlot], &'static [ChunkSlot]) {
    match stage {
        MoeStage::SharedQuant => (&[ChunkSlot::X2], &[ChunkSlot::Q8X2]),
        MoeStage::SharedGate | MoeStage::SharedUp => {
            (&[ChunkSlot::Q8X2], &[ChunkSlot::GuSh, ChunkSlot::Fixup])
        }
        MoeStage::SharedSwiglu => (&[ChunkSlot::GuSh], &[ChunkSlot::ActSh]),
        MoeStage::SharedDownQuant => (&[ChunkSlot::ActSh], &[ChunkSlot::Q8Ash]),
        MoeStage::SharedDown => (&[ChunkSlot::Q8Ash], &[ChunkSlot::ShOut, ChunkSlot::Fixup]),
        MoeStage::Ids => (
            &[ChunkSlot::TopkId],
            &[
                ChunkSlot::IdsG,
                ChunkSlot::IdsSrc1,
                ChunkSlot::IdsDst,
                ChunkSlot::ExpertBounds,
            ],
        ),
        MoeStage::Scatter => (&[ChunkSlot::X2, ChunkSlot::IdsSrc1], &[ChunkSlot::Q8S]),
        MoeStage::Gate | MoeStage::Up => (
            &[ChunkSlot::Q8S, ChunkSlot::IdsDst, ChunkSlot::ExpertBounds],
            &[ChunkSlot::Gu, ChunkSlot::Fixup],
        ),
        MoeStage::Swiglu => (&[ChunkSlot::Gu], &[ChunkSlot::Act]),
        MoeStage::GatherQuant => (&[ChunkSlot::Act, ChunkSlot::IdsDst], &[ChunkSlot::Q8Act]),
        MoeStage::Down => (
            &[ChunkSlot::Q8Act, ChunkSlot::IdsDst, ChunkSlot::ExpertBounds],
            &[ChunkSlot::Down, ChunkSlot::Fixup],
        ),
        MoeStage::Combine => (
            &[
                ChunkSlot::Hidden,
                ChunkSlot::Down,
                ChunkSlot::TopkWeight,
                ChunkSlot::SharedGate,
                ChunkSlot::ShOut,
            ],
            &[ChunkSlot::Hidden],
        ),
    }
}

/// The q8 layout rule (lane C APPROVED): one q8 buffer is read by several GEMMs, and a GEMM
/// of the other scale layout reads wrong values, not a fault. Refuses `prefill chunk: q8
/// layout` for a group whose members differ in [`Q8_1Layout`]; the group's first member is
/// the `mmq_dense_input` representative of the buffer.
pub(super) fn check_q8_layouts(groups: &[(ChunkSlot, &[QuantType])]) -> Result<(), EngError> {
    for (slot, group) in groups {
        let mut layouts = group.iter().map(|quant| mmq_type_of(*quant).q8_1_layout());
        let Some(first) = layouts.next() else {
            return Err(EngError::BadRequest(format!(
                "prefill chunk: q8 layout: {slot:?} holds no projection"
            )));
        };
        if layouts.any(|layout| layout != first) {
            return Err(EngError::BadRequest(format!(
                "prefill chunk: q8 layout: {slot:?} is quantized once but read at mixed layouts"
            )));
        }
    }
    Ok(())
}

/// The SwiGLU shape of the routed stacks (stage 14): `m * top_k` rows, the up half at column
/// `inter` of a `2 * inter` row.
pub(super) fn routed_swiglu(m: usize, dims: &MoeDims) -> SwigluRowsShape {
    SwigluRowsShape {
        rows: m * dims.top_k,
        n_ff: dims.inter,
        up_off: dims.inter,
        ld_in: 2 * dims.inter,
        ld_out: dims.inter,
    }
}

/// The `(offset, length)` of the up half's sub-view of a `Gu` slot of `gu_bytes` bytes
/// (stage 13b): the grouped launch has no column offset, so the up launch writes the tail
/// that starts at column `dims.inter` — the same constant that sets [`routed_swiglu`]'s
/// `up_off` and half its `ld_in`, never a literal.
pub(super) fn routed_up_view(gu_bytes: usize, dims: &MoeDims) -> Result<(usize, usize), EngError> {
    tail_extent(gu_bytes, dims.inter)
}

// ---- the pure call-shape builders (B.10; the `mixer_shapes` pattern) -------------------------

/// The four DeltaNet in-projections over one `Q8X` buffer: role, columns `n`, and the column
/// offset in the `ProjD` row (B.5 stage 3d).
pub(super) fn delta_proj_cols(g: &MixerGeometry) -> [(Role, usize, usize); 4] {
    let d = &g.delta;
    [
        (Role::AttnQkv, d.channels, 0),
        (Role::AttnGate, d.z_rows, d.z_off),
        (Role::SsmBeta, d.beta_rows, d.beta_off),
        (Role::SsmAlpha, d.alpha_rows, d.alpha_off),
    ]
}

/// The three attention projections over one `Q8X` buffer (B.6 stage 3a).
pub(super) fn attn_proj_cols(g: &MixerGeometry) -> [(Role, usize, usize); 3] {
    let a = &g.attn;
    [
        (Role::AttnQ, a.q_rows, 0),
        (Role::AttnK, a.kv_rows, a.k_off),
        (Role::AttnV, a.kv_rows, a.v_off),
    ]
}

/// The conv shape of a chunk (B.5 stage 4d).
pub(super) fn chunk_conv_shape(g: &MixerGeometry, m: usize) -> ConvShape {
    ConvShape {
        m,
        channels: g.delta.channels,
        ld: g.delta.ld,
    }
}

/// The scan shape of a chunk of `rows` in a `cap`-row context (B.5 stage 5d).
pub(super) fn chunk_scan_shape(g: &MixerGeometry, rows: &ChunkRows, cap: usize) -> ScanShape {
    let d = &g.delta;
    ScanShape {
        m: rows.m(),
        p0: rows.p0() as i64,
        ctx_tokens: cap,
        n_k_heads: d.key_heads,
        n_v_heads: d.value_heads,
        head_dim: d.value_dim,
        conv_ld: d.channels,
        ld: d.ld,
        z_off: d.z_off,
        beta_off: d.beta_off,
        alpha_off: d.alpha_off,
    }
}

/// The chunk preparation shape (B.6 stage 4a).
pub(super) fn chunk_prep_shape(
    g: &MixerGeometry,
    rows: &ChunkRows,
    cap: usize,
) -> AttnPrepChunkShape {
    let a = &g.attn;
    AttnPrepChunkShape {
        m: rows.m(),
        n_head: a.heads_q,
        n_kv_head: a.heads_kv,
        head_dim: a.head_dim,
        n_rot: a.n_rot,
        rows: cap,
        ld: a.ld,
        p0: rows.p0() as i64,
    }
}

/// The prefill attention shape (B.6 stage 5a).
pub(super) fn chunk_attn_shape(
    g: &MixerGeometry,
    rows: &ChunkRows,
    cap: usize,
) -> AttnPrefillShape {
    let a = &g.attn;
    AttnPrefillShape {
        m: rows.m(),
        n_head: a.heads_q,
        n_kv_head: a.heads_kv,
        head_dim: a.head_dim,
        cache_rows: cap,
        ld: a.ld,
        p0: rows.p0() as i64,
    }
}

/// The geometry the launch plan of stage 5a takes.
pub(super) fn chunk_attn_geometry(g: &MixerGeometry, m: usize) -> CpfAttnGeometry {
    CpfAttnGeometry {
        m,
        n_head: g.attn.heads_q,
        n_kv_head: g.attn.heads_kv,
        head_dim: g.attn.head_dim,
    }
}

/// The router shape (stage 9).
pub(super) fn chunk_router_shape(dims: &MoeDims, m: usize) -> RouterShape {
    RouterShape {
        m,
        hidden: dims.hidden,
        n_expert: dims.n_expert,
        top_k: dims.top_k,
    }
}

/// The combine shape (stage 17).
pub(super) fn chunk_combine_shape(dims: &MoeDims, m: usize) -> CombineShape {
    CombineShape {
        m,
        n: dims.hidden,
        top_k: dims.top_k,
        ld_down: dims.hidden,
    }
}

/// The grouped-launch shape of the MoE block (stage 11).
pub(super) fn chunk_mmid_shape(dims: &MoeDims, m: usize) -> MmidShape {
    MmidShape {
        n_experts: dims.n_expert,
        m,
        top_k: dims.top_k,
    }
}

/// The shared expert's two GEMM column pairs `(ld, col_off)` (stages 10b, 10c).
pub(super) fn shared_cols(dims: &MoeDims) -> [(usize, usize); 2] {
    [
        (2 * dims.shared_inter, 0),
        (2 * dims.shared_inter, dims.shared_inter),
    ]
}

/// The shared expert's SwiGLU shape (stage 10d).
pub(super) fn shared_swiglu(m: usize, dims: &MoeDims) -> SwigluRowsShape {
    SwigluRowsShape {
        rows: m,
        n_ff: dims.shared_inter,
        up_off: dims.shared_inter,
        ld_in: 2 * dims.shared_inter,
        ld_out: dims.shared_inter,
    }
}

/// The `MmidWeight` of one routed stack (stages 13a, 13b, 16): the whole stack buffer, its
/// type, `k` values per expert row, the expert's `n` rows, and the per-expert byte stride
/// derived from the type.
pub(super) fn routed_weight<'a>(
    stack: &'a NativeWeight,
    _dims: &MoeDims,
    k: usize,
    n: usize,
) -> Result<MmidWeight<'a>, EngError> {
    let quant = stack.quant();
    let stride_bytes = quant.weight_bytes(n, k).ok_or_else(|| {
        EngError::BadRequest(format!(
            "prefill chunk: the {quant:?} stride of {n} rows of {k} values overflows"
        ))
    })?;
    Ok(MmidWeight {
        base: whole(stack.buffer())?,
        quant,
        k,
        n,
        stride_bytes,
    })
}

/// The whole allocation behind a weight buffer, as its weight range (B.7).
fn whole(buffer: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(0, buffer.bytes())
}

// ---- ChunkBufs and its check (B.2, B.10) -----------------------------------------------------

/// The plain facts `check_bufs` guards (B.10).
pub(super) struct BufFacts {
    pub(super) m_sess: usize,
    pub(super) part_bytes: usize,
    pub(super) part_offset: usize,
    pub(super) nsm: usize,
    pub(super) fixup_bytes: usize,
}

/// Guards, in order: (B1) `m_sess` in `1..=2048` (the chunk cap, `NCOLS_LIMIT`); (B2) `part`
/// at least the flat attention cap; (B3) `part` at a 16-byte-aligned offset; (B4) `nsm >= 1`;
/// (B5) `Fixup` at least the fixup floor of `nsm` SMs.
pub(super) fn check_bufs(f: &BufFacts) -> Result<(), EngError> {
    if f.m_sess == 0 || f.m_sess > NCOLS_LIMIT {
        return Err(EngError::BadRequest(format!(
            "prefill chunk: m_sess {} is outside 1..={NCOLS_LIMIT}",
            f.m_sess
        )));
    }
    if f.part_bytes < CPF_ATTN_PART_CAP {
        return Err(EngError::BadRequest(format!(
            "prefill chunk: part: {} bytes, below the {}-byte cap",
            f.part_bytes, CPF_ATTN_PART_CAP
        )));
    }
    if !f.part_offset.is_multiple_of(16) {
        return Err(EngError::BadRequest(format!(
            "prefill chunk: part: offset {} is not 16-byte aligned",
            f.part_offset
        )));
    }
    if f.nsm == 0 {
        return Err(EngError::BadRequest(
            "prefill chunk: nsm 0 is below 1".to_owned(),
        ));
    }
    let floor = fixup_floats_max(f.nsm)? * 4;
    if f.fixup_bytes < floor {
        return Err(EngError::BadRequest(format!(
            "prefill chunk: fixup: {} bytes, below the {}-byte floor of nsm {}",
            f.fixup_bytes, floor, f.nsm
        )));
    }
    Ok(())
}

/// The buffers the body runs on: the slot views, the attention `part` range, and the two
/// facts the stages take from the device (B.2). The `part` range is CPF-7b's allocation,
/// never sized from `ChunkCost.part`.
pub(super) struct ChunkBufs<'a> {
    views: ChunkViews<'a>,
    part: DeviceRange<'a>,
    /// `MmqFacts.nsm` of the constructor's facts; stage 5a's `sm_count`.
    nsm: usize,
    /// The rows the arena was planned for.
    m_sess: usize,
}

impl<'a> ChunkBufs<'a> {
    /// Runs [`check_bufs`] over the views, `part` and facts.
    pub(super) fn new(
        views: ChunkViews<'a>,
        part: DeviceRange<'a>,
        facts: MmqFacts,
        m_sess: usize,
    ) -> Result<Self, EngError> {
        check_bufs(&BufFacts {
            m_sess,
            part_bytes: part.bytes(),
            part_offset: part.offset(),
            nsm: facts.nsm,
            fixup_bytes: views.get(ChunkSlot::Fixup)?.bytes(),
        })?;
        Ok(Self {
            views,
            part,
            nsm: facts.nsm,
            m_sess,
        })
    }
}

// ---- the model, its facts and its check (B.10; A1) --------------------------------------------

/// The plain facts of one routed expert stack (B.10): `None` is "not loaded Native".
#[derive(Clone)]
pub(super) struct StackFacts {
    pub(super) role: Role,
    pub(super) quant: Option<QuantType>,
    pub(super) experts: Option<usize>,
    pub(super) n_rows: usize,
    pub(super) k: usize,
}

/// The plain facts of one layer.
#[derive(Clone)]
pub(super) struct LayerFacts {
    pub(super) kind: AttentionKind,
    pub(super) dense: Vec<(Role, Option<QuantType>)>,
    pub(super) stacks: [StackFacts; 3],
    pub(super) shared: [(Role, Option<QuantType>); 3],
}

/// The plain facts `check_model` guards; the dims ride along for D8.
#[derive(Clone)]
pub(super) struct ModelFacts {
    pub(super) eps_bits: u32,
    pub(super) hidden: usize,
    pub(super) inter: usize,
    pub(super) experts: usize,
    pub(super) layers: Vec<LayerFacts>,
}

/// Guards, in order: (D1) the layer count; (D2) the scan epsilon bit for bit; then per layer,
/// in layer order, (D3) the kind, (D4) every dense role Native, (D5) every stack Native,
/// (D6) every shared projection Native, (D7) the stack depth, (D8) the stack shapes,
/// (D9) the q8 layout groups.
pub(super) fn check_model(f: &ModelFacts, expected_layers: usize) -> Result<(), EngError> {
    if f.layers.len() != expected_layers {
        return Err(EngError::BadRequest(format!(
            "prefill chunk: layers: the model holds {} layers, expected {expected_layers}",
            f.layers.len()
        )));
    }
    if f.eps_bits != GDN_SCAN_EPS.to_bits() {
        return Err(EngError::BadRequest(
            "prefill chunk: eps: the geometry's epsilon is not the scan's".to_owned(),
        ));
    }
    for (layer, facts) in f.layers.iter().enumerate() {
        if !matches!(facts.kind, AttentionKind::DeltaNet | AttentionKind::Full) {
            return Err(EngError::BadRequest(format!(
                "prefill chunk: kind: layer {layer} is a {:?} mixer",
                facts.kind
            )));
        }
        for (role, quant) in &facts.dense {
            if quant.is_none() {
                return Err(EngError::BadRequest(format!(
                    "prefill chunk: layout: layer {layer} {role:?} is not loaded native"
                )));
            }
        }
        for stack in &facts.stacks {
            if stack.quant.is_none() {
                return Err(EngError::BadRequest(format!(
                    "prefill chunk: layout: layer {layer} stack {:?} is not loaded native",
                    stack.role
                )));
            }
        }
        for (role, quant) in &facts.shared {
            if quant.is_none() {
                return Err(EngError::BadRequest(format!(
                    "prefill chunk: layout: layer {layer} shared {role:?} is not loaded native"
                )));
            }
        }
        for stack in &facts.stacks {
            if stack.experts != Some(f.experts) {
                return Err(EngError::BadRequest(format!(
                    "prefill chunk: experts: layer {layer} stack {:?} holds {:?}, the model has {}",
                    stack.role, stack.experts, f.experts
                )));
            }
            let (rows, k) = match stack.role {
                Role::FfnGateExps | Role::FfnUpExps => (f.inter, f.hidden),
                _ => (f.hidden, f.inter),
            };
            let experts = stack.experts.unwrap_or(0);
            if stack.n_rows != experts * rows || stack.k != k {
                return Err(EngError::BadRequest(format!(
                    "prefill chunk: stack shape: layer {layer} {:?} is {} rows of {}, the launch needs {} rows of {}",
                    stack.role,
                    stack.n_rows,
                    stack.k,
                    experts * rows,
                    k
                )));
            }
        }
        let of = |entries: &[(Role, Option<QuantType>)]| {
            entries
                .iter()
                .filter_map(|(_, quant)| *quant)
                .collect::<Vec<_>>()
        };
        let dense_of = || {
            facts
                .dense
                .iter()
                .filter(|(role, quant)| {
                    quant.is_some() && !matches!(role, Role::SsmOut | Role::AttnOutput)
                })
                .filter_map(|(_, quant)| *quant)
                .collect::<Vec<_>>()
        };
        let routed_stacks = || {
            facts
                .stacks
                .iter()
                .filter(|stack| stack.role != Role::FfnDownExps)
                .filter_map(|stack| stack.quant)
                .collect::<Vec<_>>()
        };
        check_q8_layouts(&[
            (ChunkSlot::Q8X, &dense_of()),
            (ChunkSlot::Q8X2, &of(&facts.shared[..2])),
            (ChunkSlot::Q8S, &routed_stacks()),
        ])?;
    }
    Ok(())
}

/// Extracts the facts of the first `layers` layers of `weights` (B.10): a Tiled or absent
/// weight is `None`, and `check_model` is what refuses it.
fn model_facts(
    table: &LayerTable,
    geometry: &MixerGeometry,
    dims: &StepDims,
    weights: &ChunkWeights<'_, '_>,
    layers: usize,
) -> Result<ModelFacts, EngError> {
    let mut facts = Vec::with_capacity(layers);
    for layer in 0..layers {
        let kind = table.slot(layer)?.kind;
        let lw = weights.mixer.layer_weights(layer)?;
        let dense = geometry
            .dense_specs(kind)?
            .into_iter()
            .map(|spec| {
                let quant = lw.native(spec.role).ok().map(|weight| weight.quant());
                (spec.role, quant)
            })
            .collect();
        let stack = |role, stack: &StackWeights<'_>| match stack {
            StackWeights::Native(weight) => StackFacts {
                role,
                quant: Some(weight.quant()),
                experts: weight.experts(),
                n_rows: weight.n_rows(),
                k: weight.k(),
            },
            StackWeights::Tiled(_) => StackFacts {
                role,
                quant: None,
                experts: None,
                n_rows: 0,
                k: 0,
            },
        };
        let mw = &weights.moe[layer];
        let stacks = [
            stack(Role::FfnGateExps, &mw.gate),
            stack(Role::FfnUpExps, &mw.up),
            stack(Role::FfnDownExps, &mw.down),
        ];
        let shared = |role, weight: &DenseWeight<'_>| {
            let quant = match weight {
                DenseWeight::Native(native) => Some(native.quant()),
                DenseWeight::Tiled(_) => None,
            };
            (role, quant)
        };
        let shared = [
            shared(Role::FfnGateShexp, &mw.shared_gate),
            shared(Role::FfnUpShexp, &mw.shared_up),
            shared(Role::FfnDownShexp, &mw.shared_down),
        ];
        facts.push(LayerFacts {
            kind,
            dense,
            stacks,
            shared,
        });
    }
    Ok(ModelFacts {
        eps_bits: geometry.eps.to_bits(),
        hidden: dims.shape.hidden,
        inter: dims.shape.moe.inter,
        experts: dims.shape.moe.experts,
        layers: facts,
    })
}

/// The weights of the chunk body (B.1): the mixer weights, the MoE weights per layer and the
/// native token embedding.
pub(super) struct ChunkWeights<'a, 'w> {
    mixer: &'a MixerWeights<'w>,
    moe: &'a [MoeLayerWeights<'w>],
    embed: &'w NativeWeight,
}

/// The checked model of the chunk body (B.1).
pub(super) struct ChunkModel<'a, 'w> {
    table: &'a LayerTable,
    geometry: &'a MixerGeometry,
    moe_dims: MoeDims,
    weights: ChunkWeights<'a, 'w>,
    layers: usize,
}

impl<'a, 'w> ChunkModel<'a, 'w> {
    /// `check_model` over every layer of the table.
    pub(super) fn new(
        table: &'a LayerTable,
        geometry: &'a MixerGeometry,
        dims: &StepDims,
        weights: ChunkWeights<'a, 'w>,
    ) -> Result<Self, EngError> {
        let facts = model_facts(table, geometry, dims, &weights, table.len())?;
        check_model(&facts, table.len())?;
        Ok(Self {
            table,
            geometry,
            moe_dims: MoeDims::new(dims, 1),
            weights,
            layers: table.len(),
        })
    }

    /// The same checks over the first `n` layers (the rig seam).
    #[cfg(test)]
    pub(super) fn leading(
        table: &'a LayerTable,
        geometry: &'a MixerGeometry,
        dims: &StepDims,
        weights: ChunkWeights<'a, 'w>,
        n: usize,
    ) -> Result<Self, EngError> {
        let facts = model_facts(table, geometry, dims, &weights, n)?;
        check_model(&facts, n)?;
        Ok(Self {
            table,
            geometry,
            moe_dims: MoeDims::new(dims, 1),
            weights,
            layers: n,
        })
    }

    /// The MoE dims at `m` token rows.
    fn dims_at(&self, m: usize) -> MoeDims {
        MoeDims {
            rows: m,
            ..self.moe_dims
        }
    }
}
impl<'a, 'w> ChunkWeights<'a, 'w> {
    /// The production constructor (A1): the driver and the rigs build the weights only
    /// through this.
    pub(super) fn new(
        mixer: &'a MixerWeights<'w>,
        moe: &'a [MoeLayerWeights<'w>],
        embed: &'w NativeWeight,
    ) -> Self {
        Self { mixer, moe, embed }
    }
}

/// The plain facts `check_run` guards (A1).
pub(super) struct RunFacts {
    pub(super) m: usize,
    pub(super) m_sess: usize,
    pub(super) model_layers: usize,
    pub(super) state_layers: usize,
}

/// The guards of [`ChunkRun::new`] (A1): the chunk fits the buffers, and the model holds no
/// more layers than the state (a `leading(n)` rig model is a valid prefix; A1's exact-count
/// reading would refuse the rigs the packet itself builds).
pub(super) fn check_run(f: &RunFacts) -> Result<(), EngError> {
    if f.m == 0 || f.m > f.m_sess {
        return Err(EngError::BadRequest(format!(
            "prefill chunk: rows: {} rows of buffers planned for {}",
            f.m, f.m_sess
        )));
    }
    if f.model_layers > f.state_layers {
        return Err(EngError::BadRequest(format!(
            "prefill chunk: layers: the model holds {} layers, the state {}",
            f.model_layers, f.state_layers
        )));
    }
    Ok(())
}

/// One run of the body: the state, the checked model, the buffers and the rows (B.1).
pub(super) struct ChunkRun<'r, 'w> {
    state: &'r ResidentState,
    model: &'r ChunkModel<'r, 'w>,
    bufs: &'r ChunkBufs<'r>,
    rows: ChunkRows,
}

impl<'r, 'w> ChunkRun<'r, 'w> {
    /// The production constructor (A1): checks the rows against the buffers and the model
    /// against the state.
    pub(super) fn new(
        state: &'r ResidentState,
        model: &'r ChunkModel<'r, 'w>,
        bufs: &'r ChunkBufs<'r>,
        rows: ChunkRows,
    ) -> Result<Self, EngError> {
        check_run(&RunFacts {
            m: rows.m(),
            m_sess: bufs.m_sess,
            model_layers: model.layers,
            state_layers: state.layout().delta.layers + state.layout().kv.layers,
        })?;
        Ok(Self {
            state,
            model,
            bufs,
            rows,
        })
    }
}

// ---- the stage functions (B.4-B.6): P1 and the two mixer blocks ------------------------------

/// What every stage of one layer needs: the launch, the weights, the ordinal, the index,
/// the kind and the prefill word.
struct LayerCtx<'a, 'w> {
    launch: &'a Launch<'a>,
    weights: &'a LayerWeights<'w>,
    moe: &'a MoeLayerWeights<'w>,
    ord: usize,
    layer: usize,
    kind: AttentionKind,
    status: DeviceRange<'a>,
}

impl ChunkRun<'_, '_> {
    fn v(&self, slot: ChunkSlot) -> Result<DeviceRange<'_>, EngError> {
        self.bufs.views.get(slot)
    }
}

/// P1, the embedding (B.4).
fn chunk_embed(
    launch: &Launch<'_>,
    run: &ChunkRun<'_, '_>,
    status: DeviceRange<'_>,
) -> Result<(), EngError> {
    launch.ctx.embed_native(
        launch.stream,
        EmbedNativeCall {
            weight: run.model.weights.embed,
            ids: run.v(ChunkSlot::IdsTok)?,
            x: run.v(ChunkSlot::Hidden)?,
            status,
            m: run.rows.m(),
        },
    )
}

/// Stage 1, `NormA`.
fn norm_a(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let launch = ctx.launch;
    let g = run.model.geometry;
    launch.ctx.add_norm_pack(
        launch.stream,
        AddNormCall {
            m: run.rows.m(),
            n: g.hidden,
            eps: g.eps,
            h: run.v(ChunkSlot::Hidden)?,
            delta: None,
            weight: ctx.weights.raw(Role::AttnNorm)?,
            xf: run.v(ChunkSlot::XfA)?,
            norm_out: Some(run.v(ChunkSlot::X)?),
        },
    )
}

/// Stage 2, `QuantA`: one quantize on the group representative role of the kind.
fn quant_a(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let launch = ctx.launch;
    let rep = match ctx.kind {
        AttentionKind::DeltaNet => Role::AttnQkv,
        AttentionKind::Full => Role::AttnQ,
        AttentionKind::Qsa => unreachable!("layer_stages refused the kind"),
    };
    launch.ctx.mmq_dense_input(
        launch.stream,
        MmqDenseInputCall {
            packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::CpuReference,
            weight: ctx.weights.native(rep)?,
            m: run.rows.m(),
            row_stride: run.model.geometry.hidden,
            x: run.v(ChunkSlot::X)?,
            y: run.v(ChunkSlot::Q8X)?,
        },
    )
}

/// Stage 3d, the four DeltaNet in-projections (B.5).
fn delta_proj(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let launch = ctx.launch;
    let (m, g) = (run.rows.m(), run.model.geometry);
    for (role, _n, col_off) in delta_proj_cols(g) {
        launch.ctx.mmq_dense(
            launch.stream,
            MmqDenseCall {
                weight: ctx.weights.native(role)?,
                m,
                q8_1: run.v(ChunkSlot::Q8X)?,
                dst: run.v(ChunkSlot::ProjD)?,
                ld: g.delta.ld,
                col_off,
                fixup: run.v(ChunkSlot::Fixup)?,
            },
        )?;
    }
    Ok(())
}

/// Stage 4d, the causal conv: the ring state of the layer's own ordinal.
fn delta_conv(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let launch = ctx.launch;
    launch.ctx.gdn_conv(
        launch.stream,
        ConvCall {
            shape: chunk_conv_shape(run.model.geometry, run.rows.m()),
            proj: run.v(ChunkSlot::ProjD)?,
            weights: ctx.weights.raw(Role::SsmConv1d)?,
            state: run.state.delta_conv(0, ctx.ord)?,
            out: run.v(ChunkSlot::ConvOut)?,
        },
    )
}

/// Stage 5d, the scan: the recurrent state of the layer's own ordinal, in place.
fn delta_scan(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let launch = ctx.launch;
    let status = ctx.status;
    launch.ctx.gdn_scan_seq(
        launch.stream,
        GdnScanSeqCall {
            shape: chunk_scan_shape(
                run.model.geometry,
                &run.rows,
                run.state.layout().kv.capacity,
            ),
            eps: run.model.geometry.eps,
            conv: run.v(ChunkSlot::ConvOut)?,
            proj: run.v(ChunkSlot::ProjD)?,
            weights: ScanWeights {
                a: ctx.weights.raw(Role::SsmA)?,
                dt_bias: ctx.weights.raw(Role::SsmDtBias)?,
                norm_w: ctx.weights.raw(Role::SsmNorm)?,
            },
            state: run.state.delta_rec(0, ctx.ord)?,
            o_out: run.v(ChunkSlot::OOut)?,
            status: Some(status),
        },
    )
}

/// Stage 6d, `QuantDOut`.
fn quant_d_out(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let launch = ctx.launch;
    launch.ctx.mmq_dense_input(
        launch.stream,
        MmqDenseInputCall {
            packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::CpuReference,
            weight: ctx.weights.native(Role::SsmOut)?,
            m: run.rows.m(),
            row_stride: run.model.geometry.delta.value_width,
            x: run.v(ChunkSlot::OOut)?,
            y: run.v(ChunkSlot::Q8OD)?,
        },
    )
}

/// Stage 7d, `SsmOut`.
fn ssm_out(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let launch = ctx.launch;
    launch.ctx.mmq_dense(
        launch.stream,
        MmqDenseCall {
            weight: ctx.weights.native(Role::SsmOut)?,
            m: run.rows.m(),
            q8_1: run.v(ChunkSlot::Q8OD)?,
            dst: run.v(ChunkSlot::Delta)?,
            ld: run.model.geometry.hidden,
            col_off: 0,
            fixup: run.v(ChunkSlot::Fixup)?,
        },
    )
}
/// Stage 3a, the three attention projections.
fn attn_proj(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let launch = ctx.launch;
    let (m, g) = (run.rows.m(), run.model.geometry);
    for (role, _n, col_off) in attn_proj_cols(g) {
        launch.ctx.mmq_dense(
            launch.stream,
            MmqDenseCall {
                weight: ctx.weights.native(role)?,
                m,
                q8_1: run.v(ChunkSlot::Q8X)?,
                dst: run.v(ChunkSlot::ProjA)?,
                ld: g.attn.ld,
                col_off,
                fixup: run.v(ChunkSlot::Fixup)?,
            },
        )?;
    }
    Ok(())
}

/// Stage 4a, the chunk preparation and the fp16 KV append of the layer's ordinal.
fn attn_prep(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let launch = ctx.launch;
    let (g, status) = (run.model.geometry, ctx.status);
    launch.ctx.attn_prep_chunk(
        launch.stream,
        AttnPrepChunkCall {
            shape: chunk_prep_shape(g, &run.rows, run.state.layout().kv.capacity),
            rope: RopeParams {
                eps: g.eps,
                rope_base: g.attn.rope_base,
            },
            proj: run.v(ChunkSlot::ProjA)?,
            q_norm: ctx.weights.raw(Role::AttnQNorm)?,
            k_norm: ctx.weights.raw(Role::AttnKNorm)?,
            q: run.v(ChunkSlot::Q)?,
            mode: PrepChunkMode::AppendFp16 {
                k_cache: run.state.kv_plane(ctx.ord, Plane::K)?,
                v_cache: run.state.kv_plane(ctx.ord, Plane::V)?,
            },
            status: Some(status),
        },
    )
}

/// Stage 5a, the flash attention: the launch plan's split, the flat-capped `part`.
fn attn_flash(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let launch = ctx.launch;
    let (g, m, status) = (run.model.geometry, run.rows.m(), ctx.status);
    let plan = cpf_attn_launch_plan(chunk_attn_geometry(g, m), run.rows.end(), 0, run.bufs.nsm)?;
    launch.ctx.attn_prefill(
        launch.stream,
        AttnPrefillCall {
            shape: chunk_attn_shape(g, &run.rows, run.state.layout().kv.capacity),
            span: CpfAttnSpan::FULL,
            split: CpfAttnSplit {
                max_splits: plan.max_splits,
                min_keys: plan.min_keys,
            },
            q_tile: CPF_ATTN_Q_TILE,
            scale: g.attn.scale,
            q: run.v(ChunkSlot::Q)?,
            proj: Some(run.v(ChunkSlot::ProjA)?),
            k_cache: run.state.kv_plane(ctx.ord, Plane::K)?,
            v_cache: run.state.kv_plane(ctx.ord, Plane::V)?,
            part: run.bufs.part,
            out: run.v(ChunkSlot::AttnOut)?,
            status: Some(status),
        },
    )
}

/// Stage 6a, `QuantAOut`.
fn quant_a_out(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let launch = ctx.launch;
    launch.ctx.mmq_dense_input(
        launch.stream,
        MmqDenseInputCall {
            packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::CpuReference,
            weight: ctx.weights.native(Role::AttnOutput)?,
            m: run.rows.m(),
            row_stride: run.model.geometry.attn.q_width,
            x: run.v(ChunkSlot::AttnOut)?,
            y: run.v(ChunkSlot::Q8OA)?,
        },
    )
}

/// Stage 7a, `AttnOut`.
fn attn_out(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let launch = ctx.launch;
    launch.ctx.mmq_dense(
        launch.stream,
        MmqDenseCall {
            weight: ctx.weights.native(Role::AttnOutput)?,
            m: run.rows.m(),
            q8_1: run.v(ChunkSlot::Q8OA)?,
            dst: run.v(ChunkSlot::Delta)?,
            ld: run.model.geometry.hidden,
            col_off: 0,
            fixup: run.v(ChunkSlot::Fixup)?,
        },
    )
}
/// Stage 8, `NormF` (B.7).
fn norm_f(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let launch = ctx.launch;
    let g = run.model.geometry;
    launch.ctx.add_norm_pack(
        launch.stream,
        AddNormCall {
            m: run.rows.m(),
            n: g.hidden,
            eps: g.eps,
            h: run.v(ChunkSlot::Hidden)?,
            delta: Some(run.v(ChunkSlot::Delta)?),
            weight: ctx.weights.raw(Role::PostAttentionNorm)?,
            xf: run.v(ChunkSlot::XfF)?,
            norm_out: Some(run.v(ChunkSlot::X2)?),
        },
    )
}

// ---- the MoE block and the router (B.7, B.8) --------------------------------------------------

/// The native blocks of a MoE projection or stack; `check_model` refused every other load
/// (D5, D6), so this only fails on a model that skipped its check.
fn native_dense<'b>(weight: &'b DenseWeight<'_>) -> Result<&'b NativeWeight, EngError> {
    match weight {
        DenseWeight::Native(weight) => Ok(weight),
        DenseWeight::Tiled(_) => Err(EngError::BadRequest(
            "prefill chunk: layout: the projection is not loaded native".to_owned(),
        )),
    }
}

/// The native blocks of an expert stack (B.8).
fn native_stack<'b>(stack: &'b StackWeights<'_>) -> Result<&'b NativeWeight, EngError> {
    match stack {
        StackWeights::Native(weight) => Ok(weight),
        StackWeights::Tiled(_) => Err(EngError::BadRequest(
            "prefill chunk: layout: the expert stack is not loaded native".to_owned(),
        )),
    }
}

/// The one in/out contract of the MoE block (B.8): in, `x2` f32 `[m][hidden]` read only, the
/// router's `ids` i32 `[m][top_k]` read only (seam I1), `weights` f32 `[m][top_k]`,
/// `shared_gate` f32 `[m]` and `h` f32 `[m][hidden]`; out, `h = h + Σ_r weights[t][r]·down_r
/// + shared_gate[t]·shared_t` in place, plus status bits into `status`.
pub(super) struct MoeChunkIo<'b> {
    pub(super) m: usize,
    pub(super) dims: MoeDims,
    pub(super) x2: DeviceRange<'b>,
    pub(super) ids: DeviceRange<'b>,
    pub(super) weights: DeviceRange<'b>,
    pub(super) shared_gate: DeviceRange<'b>,
    pub(super) h: DeviceRange<'b>,
    pub(super) status: DeviceRange<'b>,
    pub(super) scratch: &'b ChunkViews<'b>,
}

/// The MoE dispatch (B.8): the Resident arm; the Tiered arm is OFF-7's seam I5.
pub(super) enum ChunkMoe<'a, 'w> {
    Resident(&'a MoeLayerWeights<'w>),
    #[cfg_attr(
        test,
        expect(
            dead_code,
            reason = "v1's refusal reads neither field; OFF-7 reads them (seam I5)"
        )
    )]
    Tiered {
        runtime: &'a mut TieredRuntime<'w>,
        layer: usize,
    },
}

/// Stages 10-17 (B.8), ENQUEUE ONLY.
pub(super) fn run_chunk_moe(
    launch: &Launch<'_>,
    moe: ChunkMoe<'_, '_>,
    io: &MoeChunkIo<'_>,
) -> Result<(), EngError> {
    let mw = match moe {
        ChunkMoe::Resident(mw) => mw,
        ChunkMoe::Tiered { .. } => {
            return Err(EngError::BadRequest(
                "tiered chunk prefill: OFF-7".to_owned(),
            ));
        }
    };
    let (ctx, stream, m, dims, d) = (launch.ctx, launch.stream, io.m, &io.dims, io.scratch);
    // 10a-f, the shared expert.
    let (gate_w, up_w, down_w) = (
        native_dense(&mw.shared_gate)?,
        native_dense(&mw.shared_up)?,
        native_dense(&mw.shared_down)?,
    );
    ctx.mmq_dense_input(
        stream,
        MmqDenseInputCall {
            packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::CpuReference,
            weight: gate_w,
            m,
            row_stride: dims.hidden,
            x: io.x2,
            y: d.get(ChunkSlot::Q8X2)?,
        },
    )?;
    let cols = shared_cols(dims);
    for (weight, (_, col_off)) in [(gate_w, cols[0]), (up_w, cols[1])] {
        ctx.mmq_dense(
            stream,
            MmqDenseCall {
                weight,
                m,
                q8_1: d.get(ChunkSlot::Q8X2)?,
                dst: d.get(ChunkSlot::GuSh)?,
                ld: 2 * dims.shared_inter,
                col_off,
                fixup: d.get(ChunkSlot::Fixup)?,
            },
        )?;
    }
    ctx.swiglu_rows(
        stream,
        SwigluRowsCall {
            shape: shared_swiglu(m, dims),
            gu: d.get(ChunkSlot::GuSh)?,
            act: d.get(ChunkSlot::ActSh)?,
        },
    )?;
    ctx.mmq_dense_input(
        stream,
        MmqDenseInputCall {
            packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::CpuReference,
            weight: down_w,
            m,
            row_stride: dims.shared_inter,
            x: d.get(ChunkSlot::ActSh)?,
            y: d.get(ChunkSlot::Q8Ash)?,
        },
    )?;
    ctx.mmq_dense(
        stream,
        MmqDenseCall {
            weight: down_w,
            m,
            q8_1: d.get(ChunkSlot::Q8Ash)?,
            dst: d.get(ChunkSlot::ShOut)?,
            ld: dims.hidden,
            col_off: 0,
            fixup: d.get(ChunkSlot::Fixup)?,
        },
    )?;
    // 11, the ids guard and builder.
    let ids = ctx.mmq_build_ids(
        stream,
        IdsGuardCall {
            shape: chunk_mmid_shape(dims, m),
            ids: io.ids,
            ids_g: d.get(ChunkSlot::IdsG)?,
            status: Some(io.status),
        },
        MmidIdsOut {
            ids_src1: d.get(ChunkSlot::IdsSrc1)?,
            ids_dst: d.get(ChunkSlot::IdsDst)?,
            expert_bounds: d.get(ChunkSlot::ExpertBounds)?,
        },
    )?;
    // 12, the scatter quantize of the gate/up input.
    let gate = native_stack(&mw.gate)?;
    ctx.mmq_quantize_scatter_ids(
        stream,
        &ids,
        MmidQuantCall {
            packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::CpuReference,
            quant: gate.quant(),
            k: dims.hidden,
            row_stride: dims.hidden,
            x: io.x2,
            y: d.get(ChunkSlot::Q8S)?,
        },
    )?;
    // 13a and 13b, the gate and up launches; the up half through the derived sub-view.
    let gu = d.get(ChunkSlot::Gu)?;
    let (up_off, up_len) = routed_up_view(gu.bytes(), dims)?;
    let up_dst = gu.buffer().range(gu.offset() + up_off, up_len)?;
    let launch_mmid =
        |weight: &NativeWeight, y: DeviceRange<'_>, ld: usize, dst: DeviceRange<'_>| {
            let n = weight.n_rows() / weight.experts().unwrap_or(1);
            ctx.mmq_mmid_launch(
                stream,
                MmidLaunchCall {
                    shape: ids.shape(),
                    weight: routed_weight(weight, dims, weight.k(), n)?,
                    ids: &ids,
                    y,
                    dst,
                    ld,
                    fixup: d.get(ChunkSlot::Fixup)?,
                },
            )
        };
    let q8s = d.get(ChunkSlot::Q8S)?;
    launch_mmid(gate, q8s, 2 * dims.inter, gu)?;
    launch_mmid(native_stack(&mw.up)?, q8s, 2 * dims.inter, up_dst)?;
    // 14-16, the SwiGLU, the gather quantize and the down launch.
    ctx.swiglu_rows(
        stream,
        SwigluRowsCall {
            shape: routed_swiglu(m, dims),
            gu: d.get(ChunkSlot::Gu)?,
            act: d.get(ChunkSlot::Act)?,
        },
    )?;
    let down = native_stack(&mw.down)?;
    ctx.mmq_quantize_gather_ids(
        stream,
        &ids,
        MmidQuantCall {
            packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::CpuReference,
            quant: down.quant(),
            k: dims.inter,
            row_stride: dims.inter,
            x: d.get(ChunkSlot::Act)?,
            y: d.get(ChunkSlot::Q8Act)?,
        },
    )?;
    launch_mmid(
        down,
        d.get(ChunkSlot::Q8Act)?,
        dims.hidden,
        d.get(ChunkSlot::Down)?,
    )?;
    // 17, the combine.
    ctx.moe_combine(
        stream,
        CombineCall {
            shape: chunk_combine_shape(dims, m),
            res: io.h,
            down: d.get(ChunkSlot::Down)?,
            weights: io.weights,
            shared: Some(SharedTerm {
                gate: io.shared_gate,
                row: d.get(ChunkSlot::ShOut)?,
            }),
            out: CombineOut::InPlace,
        },
    )
}
/// The router call's buffers (B.7): stage 9 binds the two raw router buffers itself, never
/// `&MoeLayerWeights`, so OFF-7's `ChunkModel::router(layer)` changes one line.
pub(super) struct RouteIo<'b> {
    pub(super) m: usize,
    pub(super) dims: MoeDims,
    pub(super) x2: DeviceRange<'b>,
    pub(super) logits: DeviceRange<'b>,
    pub(super) ids: DeviceRange<'b>,
    pub(super) weights: DeviceRange<'b>,
    pub(super) shared_gate: DeviceRange<'b>,
    pub(super) status: DeviceRange<'b>,
    pub(super) router: &'b DeviceBuffer,
    pub(super) router_shared: &'b DeviceBuffer,
}

/// Stage 9, the router (B.7).
pub(super) fn run_chunk_route(launch: &Launch<'_>, io: &RouteIo<'_>) -> Result<(), EngError> {
    launch.ctx.router(
        launch.stream,
        RouterCall {
            shape: chunk_router_shape(&io.dims, io.m),
            x: io.x2,
            weights: RouterWeights {
                w: whole(io.router)?,
                w_shared: Some(whole(io.router_shared)?),
            },
            logits: io.logits,
            out: RouterOut {
                ids: io.ids,
                weights: io.weights,
                shared_gate: Some(io.shared_gate),
                status: io.status,
            },
        },
    )
}

// ---- the layer walk and the body (B.1) --------------------------------------------------------

/// Stage 9 inside the layer walk.
fn chunk_route(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let launch = ctx.launch;
    let (router, router_shared) = (ctx.moe.router, ctx.moe.router_shared);
    run_chunk_route(
        launch,
        &RouteIo {
            m: run.rows.m(),
            dims: run.model.dims_at(run.rows.m()),
            x2: run.v(ChunkSlot::X2)?,
            logits: run.v(ChunkSlot::Logits)?,
            ids: run.v(ChunkSlot::TopkId)?,
            weights: run.v(ChunkSlot::TopkWeight)?,
            shared_gate: run.v(ChunkSlot::SharedGate)?,
            status: ctx.status,
            router,
            router_shared,
        },
    )
}

/// The MoE dispatch inside the layer walk: v1 always the Resident arm (tiered sessions take
/// the per-token path, `chunk_ready`).
fn chunk_moe(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let launch = ctx.launch;
    run_chunk_moe(
        launch,
        ChunkMoe::Resident(ctx.moe),
        &MoeChunkIo {
            m: run.rows.m(),
            dims: run.model.dims_at(run.rows.m()),
            x2: run.v(ChunkSlot::X2)?,
            ids: run.v(ChunkSlot::TopkId)?,
            weights: run.v(ChunkSlot::TopkWeight)?,
            shared_gate: run.v(ChunkSlot::SharedGate)?,
            h: run.v(ChunkSlot::Hidden)?,
            status: ctx.status,
            scratch: &run.bufs.views,
        },
    )
}

/// One layer of the chunk (B.1): the state ranges by the layer's `ordinal`, the weights by
/// `layer` itself, every stage in `layer_stages` order.
pub(super) fn run_chunk_layer(
    launch: &Launch<'_>,
    run: &ChunkRun<'_, '_>,
    status: DeviceRange<'_>,
    layer: usize,
) -> Result<(), EngError> {
    let slot = run.model.table.slot(layer)?;
    let ctx = LayerCtx {
        launch,
        weights: run.model.weights.mixer.layer_weights(layer)?,
        moe: &run.model.weights.moe[layer],
        ord: slot.ordinal,
        layer,
        kind: slot.kind,
        status,
    };
    for stage in layer_stages(slot.kind)? {
        match stage {
            Stage::NormA => norm_a(run, &ctx)?,
            Stage::QuantA => quant_a(run, &ctx)?,
            Stage::DeltaProj => delta_proj(run, &ctx)?,
            Stage::Conv => delta_conv(run, &ctx)?,
            Stage::Scan => delta_scan(run, &ctx)?,
            Stage::QuantDOut => quant_d_out(run, &ctx)?,
            Stage::SsmOut => ssm_out(run, &ctx)?,
            Stage::AttnProj => attn_proj(run, &ctx)?,
            Stage::Prep => attn_prep(run, &ctx)?,
            Stage::Flash => attn_flash(run, &ctx)?,
            Stage::QuantAOut => quant_a_out(run, &ctx)?,
            Stage::AttnOut => attn_out(run, &ctx)?,
            Stage::NormF => norm_f(run, &ctx)?,
            Stage::Route => chunk_route(run, &ctx)?,
            Stage::Moe => chunk_moe(run, &ctx)?,
        }
    }
    Ok(())
}

/// The chunk layer loop (B.1 order): every host guard before the first launch, then the
/// embedding and every layer, ENQUEUE ONLY on the state's owning stream. Resolves the
/// prefill word ONCE; every stage takes it as a range.
pub(super) fn run_chunk_body(launch: &Launch<'_>, run: &ChunkRun<'_, '_>) -> Result<(), EngError> {
    run.state.check_chunk(launch.stream)?;
    if run.rows.end() > run.state.layout().kv.capacity {
        return Err(EngError::ContextLimit(format!(
            "prefill chunk: rows: p0 {} + m {} passes the context cap {}",
            run.rows.p0(),
            run.rows.m(),
            run.state.layout().kv.capacity
        )));
    }
    if run.state.layout().kv.mode != KvMode::Fp16 {
        return Err(EngError::BadRequest(
            "prefill chunk: kv mode: the chunk path writes fp16 KV".to_owned(),
        ));
    }
    let status = run.state.word(Word::PrefillStatus)?;
    chunk_embed(launch, run, status)?;
    for layer in 0..run.model.layers {
        run_chunk_layer(launch, run, status, layer)?;
    }
    Ok(())
}
