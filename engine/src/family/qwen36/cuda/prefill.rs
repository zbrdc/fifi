//! The chunk layer loop of decision 0049 (CPF-7a part 2): plan §C.1 stages P1 and 1-17 of one
//! prefill chunk on the state's owning stream. FP16 and staged KV8 attention are enqueue-only.
//! Unstaged KV8 uses blocking causal-length writes before each native attention tile.
//!
//! CPF-7b's driver admits the chunk, allocates the arena from
//! [`super::prefill_plan::chunk_tensors`], fills the attention `part` buffer, uploads the ids
//! and owns the §C.4 protocol around [`run_chunk_body`]. This module holds the slot views, the
//! checked chunk rows, the pure call-shape builders, the stage lists, the constructor checks
//! and the stage functions themselves. No function here allocates device memory, reads a device
//! word or records an event. KV8 attention tiles update one device word with `write_at`, which
//! waits for the owning stream. The body also calls [`super::state::ResidentState::check_chunk`].

use super::{
    dims::{KvMode, StepDims},
    layer_table::LayerTable,
    mixer::tail_extent,
    mixer_geometry::MixerGeometry,
    mixer_shapes::{attn_q8_shape, kv8_shape},
    mixer_weights::{DenseWeight, LayerWeights, MixerWeights},
    moe::{Launch, MoeDims, MoeLayerWeights, StackWeights},
    prefill_driver::Kv8Staging,
    prefill_plan::{ChunkSlot, Phase},
    state::ResidentState,
    state_layout::{Plane, Word},
    tiered_step::TieredRuntime,
};
use crate::family::project::cuda::WeightLayout;
use crate::family::spec::AttentionKind;
use crate::{
    EngError,
    family::gguf_model::Role,
    kernels::cuda::{
        ArenaTensor, DeviceArena, DeviceBuffer, DeviceRange, NativeWeight, plan_arena,
        prefill::{
            attn_prefill::{
                AttnKvStorage, AttnPrefillCall, AttnPrefillSegment, AttnPrefillShape,
                CPF_ATTN_MAX_SPLITS, CPF_ATTN_MIN_KEYS, CPF_ATTN_PART_CAP, CPF_ATTN_Q_TILE,
                CpfAttnGeometry, CpfAttnPlan, CpfAttnSpan, CpfAttnSplit, cpf_attn_launch_plan,
                cpf_attn_part_bytes,
            },
            attn_prep_chunk::{AttnPrepChunkCall, AttnPrepChunkShape, PrepChunkMode},
            gdn_scan_seq::{GDN_SCAN_EPS, GdnScanSeqCall, ScanShape, ScanWeights},
            kv8_dequant::{Kv8DequantCall, Kv8DequantShape},
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
            attn_decode::{AttnSpan, AttnSplit},
            attn_decode_q8::{AttnDecodeQ8Call, attn_q8_part_bytes},
            attn_prep::RopeParams,
            embed::EmbedNativeCall,
            gdn_conv::{ConvCall, ConvShape},
            kv8_append::Kv8Data,
            matvec_q::QuantType,
            moe_combine::{CombineCall, CombineOut, CombineShape, SharedTerm},
            router::{RouterCall, RouterOut, RouterShape, RouterWeights},
        },
    },
};
#[cfg(test)]
use kernel_diagnostics::{
    diagnostic_dense, diagnostic_dense_input, diagnostic_routed, diagnostic_shared,
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
    kv8_staging: Option<Kv8Staging<'a>>,
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
            kv8_staging: None,
            nsm: facts.nsm,
            m_sess,
        })
    }

    pub(super) fn with_kv8_staging(mut self, staging: Option<Kv8Staging<'a>>) -> Self {
        self.kv8_staging = staging;
        self
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
            .dense_specs(WeightLayout::Native, kind)?
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
                DenseWeight::Exl3(_) => None,
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

/// Per-run projection diagnostics only. Chunk normalization and nonlinear operations stay
/// unchanged, so neither MMVQ mode is a production per-token numerical oracle.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum ChunkKernelClass {
    #[default]
    ProductionMmq,
    DenseMmvq,
    AllMmvq,
}

/// One run of the body: the state, the checked model, the buffers and the rows (B.1).
pub(super) struct ChunkRun<'r, 'w> {
    state: &'r ResidentState,
    model: &'r ChunkModel<'r, 'w>,
    bufs: &'r ChunkBufs<'r>,
    rows: ChunkRows,
    segmented_parts: std::cell::Cell<usize>,
    #[cfg(test)]
    kernel_class: ChunkKernelClass,
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
            segmented_parts: std::cell::Cell::new(0),
            #[cfg(test)]
            kernel_class: ChunkKernelClass::ProductionMmq,
        })
    }

    pub(super) fn segmented_parts(&self) -> usize {
        self.segmented_parts.get()
    }

    /// Selects this run's diagnostic projections without changing another caller.
    /// `DenseMmvq` includes shared-expert projections; `AllMmvq` also replaces routed GEMMs.
    #[cfg(test)]
    pub(super) fn with_kernel_class(mut self, kernel_class: ChunkKernelClass) -> Self {
        self.kernel_class = kernel_class;
        self
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

    fn dense_input(
        &self,
        launch: &Launch<'_>,
        call: MmqDenseInputCall<'_>,
    ) -> Result<(), EngError> {
        #[cfg(test)]
        if self.kernel_class != ChunkKernelClass::ProductionMmq {
            return diagnostic_dense_input(launch, call);
        }
        launch.ctx.mmq_dense_input(launch.stream, call)
    }

    fn dense(&self, launch: &Launch<'_>, call: MmqDenseCall<'_>) -> Result<(), EngError> {
        #[cfg(test)]
        if self.kernel_class != ChunkKernelClass::ProductionMmq {
            return diagnostic_dense(launch, call);
        }
        launch.ctx.mmq_dense(launch.stream, call)
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
    run.dense_input(
        launch,
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

/// Stage 3d, the four DeltaNet in-projections (B.5). With the fusion flag on (the same
/// `INFERENCE_CUDA_GDN_FUSE_INPROJ` the native loader fuses under) the four weights are ONE
/// wide qkv matrix, so every launch reads its SEGMENT of that one upload: the vendored MMQ
/// schedules its stream-k blocks from (n, m, k), and one wide launch is NOT bitwise the
/// four narrow ones (the GDN fusion divergence, 2026-10-27). Same-shape launches over the
/// same bytes are, so the prefill keeps four launches and the fusion keeps its one upload.
fn call_weight_k(weight: &crate::kernels::cuda::NativeWeight) -> usize {
    weight.k()
}

fn delta_proj(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let launch = ctx.launch;
    let (m, g) = (run.rows.m(), run.model.geometry);
    // The wide upload concatenates the siblings in DeltaProj column order, so a segment's
    // weight-row offset IS its destination column offset.
    let wide = if crate::family::project::cuda::gdn_fuse_inproj() {
        Some(ctx.weights.native(Role::AttnQkv)?)
    } else {
        None
    };
    for &(role, n, col_off) in &delta_proj_cols(g) {
        let (weight, w_seg) = match wide {
            Some(w) => (w, Some((col_off, n))),
            None => (ctx.weights.native(role)?, None),
        };
        // Test-only trace of the projection launch (env-gated, the TTFT-trace pattern):
        // the launch shape AND the hash of the weight bytes this launch actually reads.
        if std::env::var("FIFI_DELTA_TRACE").is_ok() {
            let row_bytes = weight
                .quant()
                .weight_bytes(1, call_weight_k(weight))
                .unwrap_or(0);
            let w_off = w_seg.map(|(off, _)| off).unwrap_or(0);
            let w_bytes = n * row_bytes;
            let mut wb = vec![0u8; w_bytes];
            let range = weight.buffer().range(w_off * row_bytes, w_bytes);
            let mut w_hash = 0u64;
            if let Ok(range) = range
                && range
                    .buffer()
                    .read_at(launch.stream, range.offset(), &mut wb)
                    .is_ok()
            {
                for byte in &wb {
                    w_hash ^= *byte as u64;
                    w_hash = w_hash.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
            eprintln!(
                "DELTA_TRACE ord {} role {:?} n {n} col_off {col_off} ld {} m {m} w_rows {} w_seg {w_seg:?} w_hash {w_hash:016x}",
                ctx.ord,
                role,
                g.delta.ld,
                weight.n_rows(),
            );
        }
        let call = MmqDenseCall {
            weight,
            w_seg,
            m,
            q8_1: run.v(ChunkSlot::Q8X)?,
            dst: run.v(ChunkSlot::ProjD)?,
            ld: g.delta.ld,
            col_off,
            fixup: run.v(ChunkSlot::Fixup)?,
        };
        run.dense(launch, call)?;
        // Test-only ProjD hash after the launch, to bisect which launch first diverges.
        if std::env::var("FIFI_DELTA_TRACE").is_ok() {
            let dst = run.v(ChunkSlot::ProjD)?;
            let floats = (m - 1) * g.delta.ld + col_off + n;
            let mut bytes = vec![0u8; floats * 4];
            dst.buffer()
                .read_at(launch.stream, dst.offset(), &mut bytes)?;
            let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
            for byte in &bytes {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            eprintln!(
                "DELTA_TRACE ord {} {role:?} ProjD hash {hash:016x}",
                ctx.ord
            );
        }
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
    run.dense_input(
        launch,
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
    run.dense(
        launch,
        MmqDenseCall {
            weight: ctx.weights.native(Role::SsmOut)?,
            w_seg: None,
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
        run.dense(
            launch,
            MmqDenseCall {
                weight: ctx.weights.native(role)?,
                w_seg: None,
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

// Independent of StepParams. Every KV8 attention tile sets this word on the owning stream.
const KV8_LENGTH_OFFSET: usize = CPF_ATTN_PART_CAP - size_of::<i32>();
const KV8_ATTN_TILE_ROWS: usize = 16;
const KV8_BULK_ROWS: usize = 1024;

/// A native attention tile keeps its absolute first query position, not the chunk end.
/// Returns `(tile rows, first query's key count)` after checking the complete chunk end.
pub(super) fn kv8_attention_tile(
    p0: usize,
    m: usize,
    row: usize,
) -> Result<(usize, i32), EngError> {
    if !(1..=KV8_BULK_ROWS).contains(&m) || row >= m {
        return Err(EngError::BadRequest(format!(
            "prefill chunk: KV8 tile requires 1..={KV8_BULK_ROWS} bulk rows and row < m"
        )));
    }
    let end = p0
        .checked_add(m)
        .ok_or_else(|| EngError::BadRequest("prefill chunk: KV8 position overflows".to_owned()))?;
    i32::try_from(end)
        .map_err(|_| EngError::BadRequest("prefill chunk: KV8 length exceeds i32".to_owned()))?;
    // row < m and the complete end was checked above.
    let length = i32::try_from(p0 + row + 1)
        .map_err(|_| EngError::BadRequest("prefill chunk: KV8 length exceeds i32".to_owned()))?;
    Ok(((m - row).min(KV8_ATTN_TILE_ROWS), length))
}

/// Byte extent of strided live rows, relative to their slot rather than its backing arena.
pub(super) fn kv8_row_extent(
    capacity: usize,
    row: usize,
    m: usize,
    stride: usize,
    live: usize,
) -> Result<(usize, usize), EngError> {
    let extent = || {
        let offset = row.checked_mul(stride)?;
        let bytes = m.checked_sub(1)?.checked_mul(stride)?.checked_add(live)?;
        let end = offset.checked_add(bytes)?;
        (live > 0 && live <= stride && end <= capacity).then_some((offset, bytes))
    };
    extent().ok_or_else(|| {
        EngError::MemoryPlan("prefill chunk: KV8 row extent does not fit".to_owned())
    })
}

/// `(attention bytes, K offset, V offset, one row-block's bytes)`. Attention scratch is
/// reused across tiles; all-M K/V and the independent word remain disjoint from it.
pub(super) fn kv8_scratch_layout(
    m: usize,
    capacity: usize,
) -> Result<(usize, usize, usize, usize), EngError> {
    kv8_attention_tile(0, m, 0)?;
    let part_bytes =
        attn_q8_part_bytes(m.min(KV8_ATTN_TILE_ROWS), 2, 256, AttnSplit::Q8.max_splits)?;
    let layout = || {
        let rows = m.checked_mul(2)?.checked_mul(256)?.checked_mul(4)?;
        let k = part_bytes;
        let v = k.checked_add(rows)?;
        let end = v.checked_add(rows)?;
        (end <= KV8_LENGTH_OFFSET && capacity >= CPF_ATTN_PART_CAP)
            .then_some((part_bytes, k, v, rows))
    };
    layout().ok_or_else(|| {
        EngError::MemoryPlan(
            "prefill chunk: KV8 scratch overlaps or exceeds part capacity".to_owned(),
        )
    })
}

struct Kv8Parts<'a> {
    k: DeviceRange<'a>,
    v: DeviceRange<'a>,
    part: DeviceRange<'a>,
    length: DeviceRange<'a>,
}

fn part_view<'a>(
    part: DeviceRange<'a>,
    offset: usize,
    bytes: usize,
) -> Result<DeviceRange<'a>, EngError> {
    match (offset.checked_add(bytes), part.offset().checked_add(offset)) {
        (Some(end), Some(start)) if end <= part.bytes() => part.buffer().range(start, bytes),
        _ => Err(EngError::MemoryPlan(
            "prefill chunk: KV8 part subrange does not fit".to_owned(),
        )),
    }
}

/// Q36 bulk work admits 1..=1024 rows. Unstaged KV8 attention uses tiles of at most 16 rows.
fn kv8_parts<'a>(run: &ChunkRun<'a, '_>) -> Result<Kv8Parts<'a>, EngError> {
    let g = run.model.geometry;
    let m = run.rows.m();
    if g.attn.head_dim != 256 || g.attn.heads_q != 16 || g.attn.heads_kv != 2 {
        return Err(EngError::BadRequest(
            "prefill chunk: kv mode: the KV8 body requires Q36 D256".to_owned(),
        ));
    }
    kv8_attention_tile(run.rows.p0(), m, 0)?;
    let (bytes, k, v, rows) = kv8_scratch_layout(m, run.bufs.part.bytes())?;
    // Prove all live rows before the first launch, including strided projection gates.
    kv8_attention_rows(run, 0, m)?;
    Ok(Kv8Parts {
        k: part_view(run.bufs.part, k, rows)?,
        v: part_view(run.bufs.part, v, rows)?,
        part: part_view(run.bufs.part, 0, bytes)?,
        length: part_view(run.bufs.part, KV8_LENGTH_OFFSET, size_of::<i32>())?,
    })
}

/// Stage 4a, preparation and the mode-specific KV append of the layer's ordinal.
fn attn_prep(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let launch = ctx.launch;
    let (g, status) = (run.model.geometry, ctx.status);
    let kv8 = if run.state.layout().kv.mode == KvMode::Int8G32 {
        Some(kv8_parts(run)?)
    } else {
        None
    };
    let mode = if let Some(parts) = &kv8 {
        PrepChunkMode::RowsOut {
            k_rows: parts.k,
            v_rows: parts.v,
        }
    } else {
        PrepChunkMode::AppendFp16 {
            k_cache: run.state.kv_plane(ctx.ord, Plane::K)?,
            v_cache: run.state.kv_plane(ctx.ord, Plane::V)?,
        }
    };
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
            mode,
            status: Some(status),
        },
    )?;
    if let Some(parts) = kv8 {
        let mut shape = kv8_shape(g, run.state.layout().kv.capacity);
        shape.rows = run.rows.m();
        launch.ctx.kv8_chunk(
            launch.stream,
            shape,
            Kv8Data {
                k: parts.k,
                v: parts.v,
                k_codes: run.state.kv_plane(ctx.ord, Plane::K)?,
                k_scales: run.state.kv_plane(ctx.ord, Plane::KScales)?,
                v_codes: run.state.kv_plane(ctx.ord, Plane::V)?,
                v_scales: run.state.kv_plane(ctx.ord, Plane::VScales)?,
            },
            run.rows.p0(),
        )?;
    }
    Ok(())
}

/// One Q8 tile beats deep-prefix staging for the measured short chunks.
pub(super) fn kv8_use_segments(rows: usize) -> bool {
    rows > KV8_ATTN_TILE_ROWS
}

/// Query subtiles share the fixed part buffer without shrinking the admitted chunk.
pub(super) fn kv8_segment_query_rows(
    geometry: CpfAttnGeometry,
    segments: usize,
) -> Result<usize, EngError> {
    if geometry.m == 0 || !(1..=CPF_ATTN_MAX_SPLITS).contains(&segments) {
        return Err(EngError::BadRequest(
            "prefill chunk: invalid segmented query plan".into(),
        ));
    }
    let plan = CpfAttnPlan {
        q_tile: CPF_ATTN_Q_TILE,
        max_splits: segments,
        min_keys: CPF_ATTN_MIN_KEYS,
    };
    let mut query = geometry;
    loop {
        if cpf_attn_part_bytes(query, plan)? <= CPF_ATTN_PART_CAP {
            return Ok(query.m);
        }
        if query.m <= CPF_ATTN_Q_TILE {
            return Err(EngError::MemoryPlan(
                "prefill chunk: segmented attention partials exceed the fixed part cap".into(),
            ));
        }
        query.m = query.m.div_ceil(2);
    }
}

/// Q/output rows are packed; projection gates retain the full fused projection stride.
fn kv8_attention_rows<'a>(
    run: &'a ChunkRun<'_, '_>,
    row: usize,
    m: usize,
) -> Result<[DeviceRange<'a>; 3], EngError> {
    let a = &run.model.geometry.attn;
    let width = a.q_width.checked_mul(4).ok_or_else(|| {
        EngError::MemoryPlan("prefill chunk: KV8 query width overflows".to_owned())
    })?;
    let ld = a.ld.checked_mul(4).ok_or_else(|| {
        EngError::MemoryPlan("prefill chunk: KV8 projection stride overflows".to_owned())
    })?;
    let gate = width.checked_mul(2).ok_or_else(|| {
        EngError::MemoryPlan("prefill chunk: KV8 gate width overflows".to_owned())
    })?;
    let view = |slot, stride, live| {
        let range = run.v(slot)?;
        let (offset, bytes) = kv8_row_extent(range.bytes(), row, m, stride, live)?;
        part_view(range, offset, bytes)
    };
    Ok([
        view(ChunkSlot::Q, width, width)?,
        view(ChunkSlot::ProjA, ld, gate)?,
        view(ChunkSlot::AttnOut, width, width)?,
    ])
}

/// Stage 5a alone is tiled. Prep/StoredHalf append have already written every chunk key;
/// FULL span and the tile-local first length preserve each query's absolute causal prefix.
fn attn_flash_kv8(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let g = run.model.geometry;
    let parts = kv8_parts(run)?;
    for row in (0..run.rows.m()).step_by(KV8_ATTN_TILE_ROWS) {
        let (m, first_length) = kv8_attention_tile(run.rows.p0(), run.rows.m(), row)?;
        let [q, proj, out] = kv8_attention_rows(run, row, m)?;
        // The existing checked copy is stream-ordered but blocking. No fixture initialization
        // or StepParams word is required, including after reset or an FP16 attention plan.
        parts.length.buffer().write_at(
            ctx.launch.stream,
            parts.length.offset(),
            &first_length.to_le_bytes(),
        )?;
        let mut shape = attn_q8_shape(g, run.state.layout().kv.capacity);
        shape.m = m;
        ctx.launch.ctx.attn_decode_q8(
            ctx.launch.stream,
            AttnDecodeQ8Call {
                shape,
                span: AttnSpan::FULL,
                split: AttnSplit::Q8,
                scale: g.attn.scale,
                q,
                proj: Some(proj),
                k_codes: run.state.kv_plane(ctx.ord, Plane::K)?,
                k_scales: run.state.kv_plane(ctx.ord, Plane::KScales)?,
                v_codes: run.state.kv_plane(ctx.ord, Plane::V)?,
                v_scales: run.state.kv_plane(ctx.ord, Plane::VScales)?,
                kv_len: parts.length,
                part: parts.part,
                out,
            },
        )?;
    }
    Ok(())
}

/// Stage 5a, the flash attention: the launch plan's split, the flat-capped `part`.
fn attn_flash(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    if run.state.layout().kv.mode == KvMode::Int8G32 {
        if let Some(staging) = run.bufs.kv8_staging {
            if staging.plan.covers(run.rows.end()) {
                return attn_flash_staged(run, ctx, staging);
            }
            if staging.segmented && kv8_use_segments(run.rows.m()) {
                return attn_flash_segmented(run, ctx, staging);
            }
        }
        return attn_flash_kv8(run, ctx);
    }
    attn_flash_fp16(
        run,
        ctx,
        run.state.layout().kv.capacity,
        [
            run.state.kv_plane(ctx.ord, Plane::K)?,
            run.state.kv_plane(ctx.ord, Plane::V)?,
        ],
    )
}

fn kv8_stage_segment(
    run: &ChunkRun<'_, '_>,
    ctx: &LayerCtx<'_, '_>,
    staging: Kv8Staging<'_>,
    segment: &AttnPrefillSegment,
) -> Result<(), EngError> {
    let a = &run.model.geometry.attn;
    for (codes, scales, out) in [
        (Plane::K, Plane::KScales, staging.planes[0]),
        (Plane::V, Plane::VScales, staging.planes[1]),
    ] {
        ctx.launch.ctx.kv8_dequant_segment(
            ctx.launch.stream,
            Kv8DequantCall {
                shape: Kv8DequantShape {
                    capacity: run.state.layout().kv.capacity,
                    rows: segment.key_rows,
                    n_kv_head: a.heads_kv,
                    head_dim: a.head_dim,
                },
                codes: run.state.kv_plane(ctx.ord, codes)?,
                scales: run.state.kv_plane(ctx.ord, scales)?,
                out,
            },
            segment.key_origin,
        )?;
    }
    Ok(())
}

fn attn_flash_segmented(
    run: &ChunkRun<'_, '_>,
    ctx: &LayerCtx<'_, '_>,
    staging: Kv8Staging<'_>,
) -> Result<(), EngError> {
    let g = run.model.geometry;
    let stage_rows = staging.plan.capacity;
    let query_rows = kv8_segment_query_rows(
        chunk_attn_geometry(g, run.rows.m()),
        run.rows.end().div_ceil(stage_rows),
    )?;
    for row in (0..run.rows.m()).step_by(query_rows) {
        let m = query_rows.min(run.rows.m() - row);
        let rows = ChunkRows::new(
            run.rows.p0() + row,
            m,
            run.state.layout().kv.capacity,
            run.bufs.m_sess,
        )?;
        let [q, proj, out] = kv8_attention_rows(run, row, m)?;
        let call = || AttnPrefillCall {
            storage: AttnKvStorage::F16,
            kv_views: None,
            shape: chunk_attn_shape(g, &rows, stage_rows),
            span: CpfAttnSpan::FULL,
            split: CpfAttnSplit {
                max_splits: rows.end().div_ceil(stage_rows),
                min_keys: CPF_ATTN_MIN_KEYS,
            },
            q_tile: CPF_ATTN_Q_TILE,
            scale: g.attn.scale,
            q,
            proj: Some(proj),
            sinks: None,
            k_cache: staging.planes[0],
            v_cache: staging.planes[1],
            part: run.bufs.part,
            out,
            status: Some(ctx.status),
        };
        // Same-stream order protects stage/part reuse without a host wait.
        for (part_index, key_origin) in (0..rows.end()).step_by(stage_rows).enumerate() {
            let segment = AttnPrefillSegment {
                key_origin,
                key_rows: stage_rows.min(rows.end() - key_origin),
                part_index,
            };
            kv8_stage_segment(run, ctx, staging, &segment)?;
            ctx.launch
                .ctx
                .attn_prefill_part(ctx.launch.stream, call(), segment)?;
            run.segmented_parts
                .set(run.segmented_parts.get().saturating_add(1));
        }
        ctx.launch
            .ctx
            .attn_prefill_combine(ctx.launch.stream, call())?;
    }
    Ok(())
}

fn attn_flash_staged(
    run: &ChunkRun<'_, '_>,
    ctx: &LayerCtx<'_, '_>,
    staging: Kv8Staging<'_>,
) -> Result<(), EngError> {
    let a = &run.model.geometry.attn;
    // KV8 packing precedes this stage on the same stream, including the current chunk.
    for (codes, scales, out) in [
        (Plane::K, Plane::KScales, staging.planes[0]),
        (Plane::V, Plane::VScales, staging.planes[1]),
    ] {
        ctx.launch.ctx.kv8_dequant(
            ctx.launch.stream,
            Kv8DequantCall {
                shape: Kv8DequantShape {
                    capacity: run.state.layout().kv.capacity,
                    rows: run.rows.end(),
                    n_kv_head: a.heads_kv,
                    head_dim: a.head_dim,
                },
                codes: run.state.kv_plane(ctx.ord, codes)?,
                scales: run.state.kv_plane(ctx.ord, scales)?,
                out,
            },
        )?;
    }
    attn_flash_fp16(run, ctx, staging.plan.capacity, staging.planes)
}

fn attn_flash_fp16(
    run: &ChunkRun<'_, '_>,
    ctx: &LayerCtx<'_, '_>,
    capacity: usize,
    [k_cache, v_cache]: [DeviceRange<'_>; 2],
) -> Result<(), EngError> {
    let launch = ctx.launch;
    let (g, m, status) = (run.model.geometry, run.rows.m(), ctx.status);
    let plan = cpf_attn_launch_plan(chunk_attn_geometry(g, m), run.rows.end(), 0, run.bufs.nsm)?;
    launch.ctx.attn_prefill(
        launch.stream,
        AttnPrefillCall {
            storage: AttnKvStorage::F16,
            kv_views: None,
            shape: chunk_attn_shape(g, &run.rows, capacity),
            span: CpfAttnSpan::FULL,
            split: CpfAttnSplit {
                max_splits: plan.max_splits,
                min_keys: plan.min_keys,
            },
            q_tile: CPF_ATTN_Q_TILE,
            scale: g.attn.scale,
            q: run.v(ChunkSlot::Q)?,
            proj: Some(run.v(ChunkSlot::ProjA)?),
            sinks: None,
            k_cache,
            v_cache,
            part: run.bufs.part,
            out: run.v(ChunkSlot::AttnOut)?,
            status: Some(status),
        },
    )
}

/// Stage 6a, `QuantAOut`.
fn quant_a_out(run: &ChunkRun<'_, '_>, ctx: &LayerCtx<'_, '_>) -> Result<(), EngError> {
    let launch = ctx.launch;
    run.dense_input(
        launch,
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
    run.dense(
        launch,
        MmqDenseCall {
            weight: ctx.weights.native(Role::AttnOutput)?,
            w_seg: None,
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
        DenseWeight::Exl3(_) => Err(EngError::BadRequest(
            "prefill chunk: EXL3 is not a native block projection".into(),
        )),
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
    run_chunk_shared(launch, mw, io)?;
    run_chunk_routed(launch, mw, io)?;
    combine_chunk_moe(launch, io)
}

/// Stages 10a-f, the shared expert. Diagnostic runs replace only its projections.
fn run_chunk_shared(
    launch: &Launch<'_>,
    mw: &MoeLayerWeights<'_>,
    io: &MoeChunkIo<'_>,
) -> Result<(), EngError> {
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
                w_seg: None,
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
            w_seg: None,
            m,
            q8_1: d.get(ChunkSlot::Q8Ash)?,
            dst: d.get(ChunkSlot::ShOut)?,
            ld: dims.hidden,
            col_off: 0,
            fixup: d.get(ChunkSlot::Fixup)?,
        },
    )
}

/// Stages 11-16, the grouped routed projections and their existing nonlinear operation.
fn run_chunk_routed(
    launch: &Launch<'_>,
    mw: &MoeLayerWeights<'_>,
    io: &MoeChunkIo<'_>,
) -> Result<(), EngError> {
    let (ctx, stream, m, dims, d) = (launch.ctx, launch.stream, io.m, &io.dims, io.scratch);
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
    )
}

/// Stage 17, unchanged for every diagnostic class.
fn combine_chunk_moe(launch: &Launch<'_>, io: &MoeChunkIo<'_>) -> Result<(), EngError> {
    let (dims, d) = (&io.dims, io.scratch);
    launch.ctx.moe_combine(
        launch.stream,
        CombineCall {
            shape: chunk_combine_shape(dims, io.m),
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
    let io = MoeChunkIo {
        m: run.rows.m(),
        dims: run.model.dims_at(run.rows.m()),
        x2: run.v(ChunkSlot::X2)?,
        ids: run.v(ChunkSlot::TopkId)?,
        weights: run.v(ChunkSlot::TopkWeight)?,
        shared_gate: run.v(ChunkSlot::SharedGate)?,
        h: run.v(ChunkSlot::Hidden)?,
        status: ctx.status,
        scratch: &run.bufs.views,
    };
    #[cfg(test)]
    if run.kernel_class != ChunkKernelClass::ProductionMmq {
        diagnostic_shared(launch, ctx.moe, &io)?;
        if run.kernel_class == ChunkKernelClass::AllMmvq {
            diagnostic_routed(launch, ctx.moe, &io)?;
        } else {
            run_chunk_routed(launch, ctx.moe, &io)?;
        }
        return combine_chunk_moe(launch, &io);
    }
    run_chunk_moe(launch, ChunkMoe::Resident(ctx.moe), &io)
}

/// One layer of the chunk (B.1): the state ranges by the layer's `ordinal`, the weights by
/// `layer` itself, every stage in `layer_stages` order.
pub(super) fn run_chunk_layer(
    launch: &Launch<'_>,
    run: &ChunkRun<'_, '_>,
    status: DeviceRange<'_>,
    layer: usize,
    trace_piece: Option<u32>,
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
            Stage::Moe => {
                // The trace dump of the pre-mixer-out residual: X2 is final here (NormF and
                // Route only read it), and the MoE block is about to consume it.
                if let Some(piece) = trace_piece {
                    // `hidden` from the model dims: the X2 slot is cap-sized, so the
                    // range's byte length is not m rows of hidden.
                    let dims = run.model.dims_at(run.rows.m());
                    let rows = super::trace::prefill_tail(
                        launch,
                        &run.v(ChunkSlot::X2)?,
                        run.rows.m(),
                        dims.hidden,
                    )?;
                    crate::family::qwen36::trace::write_prefill_rows(piece, layer, &rows)?;
                }
                chunk_moe(run, &ctx)?
            }
        }
    }
    Ok(())
}

/// The chunk layer loop (B.1 order): every host guard before the first launch, then the
/// embedding and every layer on the state's owning stream. Resolves the prefill word ONCE;
/// every stage takes it as a range. KV8 accepts bulk M up to 1024. Unstaged attention uses tiles
/// up to 16; each tile initializes its independent part-tail carrier with a blocking write.
/// FP16 and staged KV8 attention are enqueue-only. Normal chunk publication is unchanged.
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
    if run.state.layout().kv.mode == KvMode::Int8G32 {
        // Prove the bounded shapes and disjoint workspace before the first launch.
        kv8_parts(run)?;
    }
    let status = run.state.word(Word::PrefillStatus)?;
    chunk_embed(launch, run, status)?;
    // One prefill-piece index for every layer of the piece (resident_trace's counter).
    let trace_piece = crate::family::qwen36::trace::begin_prefill();
    for layer in 0..run.model.layers {
        run_chunk_layer(launch, run, status, layer, trace_piece)?;
    }
    Ok(())
}

/// Test-only native MMVQ projections over the existing chunk slots. These functions never
/// interpret MMQ-packed bytes as native q8_1. They preserve chunk norms, SwiGLU and routing.
#[cfg(test)]
mod kernel_diagnostics {
    use super::*;
    use crate::kernels::cuda::resident::{
        matvec_expert::ExpertMatvecCall,
        matvec_q::MatvecCall,
        quantize_q8_1::{QuantizeCall, q8_1_bytes},
    };

    fn refusal() -> EngError {
        EngError::BadRequest("prefill chunk: MMVQ diagnostic shape or slot extent".to_owned())
    }

    fn product(a: usize, b: usize) -> Result<usize, EngError> {
        a.checked_mul(b).ok_or_else(refusal)
    }

    /// Byte offsets relative to one slot, not its allocation. The last live row must fit
    /// both its row stride and the actual planned view, even when the arena aliases slots.
    fn row_extent(
        capacity: usize,
        row: usize,
        stride: usize,
        column: usize,
        bytes: usize,
    ) -> Result<(usize, usize), EngError> {
        let reach = column.checked_add(bytes).ok_or_else(refusal)?;
        let offset = row
            .checked_mul(stride)
            .and_then(|base| base.checked_add(column))
            .ok_or_else(refusal)?;
        if reach > stride || offset.checked_add(bytes).is_none_or(|end| end > capacity) {
            return Err(refusal());
        }
        Ok((offset, bytes))
    }

    fn view<'a>(
        range: DeviceRange<'a>,
        extent: (usize, usize),
    ) -> Result<DeviceRange<'a>, EngError> {
        let (offset, bytes) = extent;
        if offset
            .checked_add(bytes)
            .is_none_or(|end| end > range.bytes())
        {
            return Err(refusal());
        }
        let start = range.offset().checked_add(offset).ok_or_else(refusal)?;
        range.buffer().range(start, bytes)
    }

    fn row<'a>(
        range: DeviceRange<'a>,
        index: usize,
        stride: usize,
        column: usize,
        bytes: usize,
    ) -> Result<DeviceRange<'a>, EngError> {
        view(
            range,
            row_extent(range.bytes(), index, stride, column, bytes)?,
        )
    }

    fn last(m: usize) -> Result<usize, EngError> {
        m.checked_sub(1).ok_or_else(refusal)
    }

    pub(super) fn diagnostic_dense_input(
        launch: &Launch<'_>,
        call: MmqDenseInputCall<'_>,
    ) -> Result<(), EngError> {
        let k = call.weight.k();
        let x_stride = product(call.row_stride, 4)?;
        let x_bytes = product(k, 4)?;
        let q_stride = q8_1_bytes(k, 1).ok_or_else(refusal)?;
        // Check the complete active extent before queuing any row.
        row_extent(call.x.bytes(), last(call.m)?, x_stride, 0, x_bytes)?;
        row_extent(call.y.bytes(), last(call.m)?, q_stride, 0, q_stride)?;
        for index in 0..call.m {
            launch.ctx.quantize_q8_1(
                launch.stream,
                QuantizeCall {
                    x: row(call.x, index, x_stride, 0, x_bytes)?,
                    y: row(call.y, index, q_stride, 0, q_stride)?,
                    k,
                    ncols: 1,
                },
            )?;
        }
        Ok(())
    }

    pub(super) fn diagnostic_dense(
        launch: &Launch<'_>,
        call: MmqDenseCall<'_>,
    ) -> Result<(), EngError> {
        if call.weight.experts().is_some() {
            return Err(refusal());
        }
        // A segmented call reads only its rows of the weight, mirroring the production
        // w_seg slice of mmq_dense (the GDN fusion, 2026-10-27).
        let (w_off, n) = call.w_seg.unwrap_or((0, call.weight.n_rows()));
        let k = call.weight.k();
        let q_stride = q8_1_bytes(k, 1).ok_or_else(refusal)?;
        let dst_stride = product(call.ld, 4)?;
        let column = product(call.col_off, 4)?;
        let dst_bytes = product(n, 4)?;
        let row_bytes = call.weight.quant().weight_bytes(1, k).ok_or_else(refusal)?;
        let w_start = w_off.checked_mul(row_bytes).ok_or_else(refusal)?;
        let w_bytes = n.checked_mul(row_bytes).ok_or_else(refusal)?;
        row_extent(call.q8_1.bytes(), last(call.m)?, q_stride, 0, q_stride)?;
        row_extent(
            call.dst.bytes(),
            last(call.m)?,
            dst_stride,
            column,
            dst_bytes,
        )?;
        for index in 0..call.m {
            launch.ctx.matvec_q(
                launch.stream,
                MatvecCall {
                    ty: call.weight.quant(),
                    w: call.weight.buffer().range(w_start, w_bytes)?,
                    x_q8: row(call.q8_1, index, q_stride, 0, q_stride)?,
                    y: row(call.dst, index, dst_stride, column, dst_bytes)?,
                    k,
                    rows: n,
                    ncols: 1,
                },
            )?;
        }
        Ok(())
    }

    pub(super) fn diagnostic_shared(
        launch: &Launch<'_>,
        mw: &MoeLayerWeights<'_>,
        io: &MoeChunkIo<'_>,
    ) -> Result<(), EngError> {
        let (m, dims, d) = (io.m, &io.dims, io.scratch);
        let gate = native_dense(&mw.shared_gate)?;
        let up = native_dense(&mw.shared_up)?;
        let down = native_dense(&mw.shared_down)?;
        let packing = crate::kernels::cuda::prefill::mmq_quant::MmqPacking::CpuReference;
        diagnostic_dense_input(
            launch,
            MmqDenseInputCall {
                packing,
                weight: gate,
                m,
                row_stride: dims.hidden,
                x: io.x2,
                y: d.get(ChunkSlot::Q8X2)?,
            },
        )?;
        for (weight, (ld, col_off)) in [(gate, shared_cols(dims)[0]), (up, shared_cols(dims)[1])] {
            diagnostic_dense(
                launch,
                MmqDenseCall {
                    weight,
                    w_seg: None,
                    m,
                    q8_1: d.get(ChunkSlot::Q8X2)?,
                    dst: d.get(ChunkSlot::GuSh)?,
                    ld,
                    col_off,
                    fixup: d.get(ChunkSlot::Fixup)?,
                },
            )?;
        }
        launch.ctx.swiglu_rows(
            launch.stream,
            SwigluRowsCall {
                shape: shared_swiglu(m, dims),
                gu: d.get(ChunkSlot::GuSh)?,
                act: d.get(ChunkSlot::ActSh)?,
            },
        )?;
        diagnostic_dense_input(
            launch,
            MmqDenseInputCall {
                packing,
                weight: down,
                m,
                row_stride: dims.shared_inter,
                x: d.get(ChunkSlot::ActSh)?,
                y: d.get(ChunkSlot::Q8Ash)?,
            },
        )?;
        diagnostic_dense(
            launch,
            MmqDenseCall {
                weight: down,
                w_seg: None,
                m,
                q8_1: d.get(ChunkSlot::Q8Ash)?,
                dst: d.get(ChunkSlot::ShOut)?,
                ld: dims.hidden,
                col_off: 0,
                fixup: d.get(ChunkSlot::Fixup)?,
            },
        )
    }

    /// One production native MMVQ-id launch per token. Device ids remain in pick order;
    /// only the token base changes. The down launch reads one native q8_1 row per pick.
    fn expert_rows(
        launch: &Launch<'_>,
        call: ExpertMatvecCall<'_>,
        m: usize,
    ) -> Result<(), EngError> {
        let picks = call.n_used;
        let ids_stride = product(picks, 4)?;
        let q_rows = if call.x_per_slot { picks } else { 1 };
        let q_stride = q8_1_bytes(call.k, q_rows).ok_or_else(refusal)?;
        let dst_stride = product(product(picks, call.y_slot_stride)?, 4)?;
        let dst_bytes = picks
            .checked_sub(1)
            .and_then(|slots| slots.checked_mul(call.y_slot_stride))
            .and_then(|lead| lead.checked_add(call.rows))
            .and_then(|floats| floats.checked_mul(4))
            .ok_or_else(refusal)?;
        row_extent(call.ids.bytes(), last(m)?, ids_stride, 0, ids_stride)?;
        row_extent(call.x_q8.bytes(), last(m)?, q_stride, 0, q_stride)?;
        row_extent(call.y.bytes(), last(m)?, dst_stride, 0, dst_bytes)?;
        for index in 0..m {
            launch.ctx.matvec_expert(
                launch.stream,
                ExpertMatvecCall {
                    ids: row(call.ids, index, ids_stride, 0, ids_stride)?,
                    x_q8: row(call.x_q8, index, q_stride, 0, q_stride)?,
                    y: row(call.y, index, dst_stride, 0, dst_bytes)?,
                    ..call
                },
            )?;
        }
        Ok(())
    }

    pub(super) fn diagnostic_routed(
        launch: &Launch<'_>,
        mw: &MoeLayerWeights<'_>,
        io: &MoeChunkIo<'_>,
    ) -> Result<(), EngError> {
        let (m, dims, d) = (io.m, &io.dims, io.scratch);
        let gate = native_stack(&mw.gate)?;
        let up = native_stack(&mw.up)?;
        let down = native_stack(&mw.down)?;
        if !(1..=8).contains(&dims.top_k) || dims.top_k > dims.n_expert {
            return Err(refusal());
        }
        for (weight, k, n) in [
            (gate, dims.hidden, dims.inter),
            (up, dims.hidden, dims.inter),
            (down, dims.inter, dims.hidden),
        ] {
            if weight.experts() != Some(dims.n_expert)
                || weight.k() != k
                || weight.n_rows() != product(dims.n_expert, n)?
            {
                return Err(refusal());
            }
        }
        // Keep chunk status validation. The native expert kernel reads the original ids,
        // like production decode; valid router output has exactly the same pick order.
        // The guarded copy is not read beyond its arena lifetime (the Ids phase).
        launch.ctx.mmq_ids_guard(
            launch.stream,
            IdsGuardCall {
                shape: chunk_mmid_shape(dims, m),
                ids: io.ids,
                ids_g: d.get(ChunkSlot::IdsG)?,
                status: Some(io.status),
            },
        )?;
        let packing = crate::kernels::cuda::prefill::mmq_quant::MmqPacking::CpuReference;
        diagnostic_dense_input(
            launch,
            MmqDenseInputCall {
                packing,
                weight: gate,
                m,
                row_stride: dims.hidden,
                x: io.x2,
                y: d.get(ChunkSlot::Q8S)?,
            },
        )?;
        let gu = d.get(ChunkSlot::Gu)?;
        for (weight, column) in [(gate, 0), (up, dims.inter)] {
            let offset = product(column, 4)?;
            expert_rows(
                launch,
                ExpertMatvecCall {
                    ty: weight.quant(),
                    w_stack: whole(weight.buffer())?,
                    ids: io.ids,
                    x_q8: d.get(ChunkSlot::Q8S)?,
                    x_per_slot: false,
                    y: view(
                        gu,
                        (offset, gu.bytes().checked_sub(offset).ok_or_else(refusal)?),
                    )?,
                    y_slot_stride: product(2, dims.inter)?,
                    k: dims.hidden,
                    rows: dims.inter,
                    n_expert: dims.n_expert,
                    n_used: dims.top_k,
                },
                m,
            )?;
        }
        launch.ctx.swiglu_rows(
            launch.stream,
            SwigluRowsCall {
                shape: routed_swiglu(m, dims),
                gu,
                act: d.get(ChunkSlot::Act)?,
            },
        )?;
        diagnostic_dense_input(
            launch,
            MmqDenseInputCall {
                packing,
                weight: down,
                m: product(m, dims.top_k)?,
                row_stride: dims.inter,
                x: d.get(ChunkSlot::Act)?,
                y: d.get(ChunkSlot::Q8Act)?,
            },
        )?;
        expert_rows(
            launch,
            ExpertMatvecCall {
                ty: down.quant(),
                w_stack: whole(down.buffer())?,
                ids: io.ids,
                x_q8: d.get(ChunkSlot::Q8Act)?,
                x_per_slot: true,
                y: d.get(ChunkSlot::Down)?,
                y_slot_stride: dims.hidden,
                k: dims.inter,
                rows: dims.hidden,
                n_expert: dims.n_expert,
                n_used: dims.top_k,
            },
            m,
        )
    }

    #[test]
    fn diagnostic_rows_keep_padding_columns_and_slot_bounds() {
        // Native q8_1 pads K=768 to 1024 values, independently for each token.
        let q_stride = q8_1_bytes(768, 1).unwrap();
        assert_eq!(q_stride, 1152);
        assert_eq!(
            row_extent(16 * q_stride, 15, q_stride, 0, q_stride).unwrap(),
            (17280, 1152)
        );
        assert!(row_extent(16 * q_stride - 1, 15, q_stride, 0, q_stride).is_err());
        // A projection writes only its columns, not the adjacent projections or next row.
        assert_eq!(
            row_extent(16 * 2048, 15, 2048, 512, 1024).unwrap(),
            (31232, 1024)
        );
        assert!(row_extent(16 * 2048, 0, 2048, 1025, 1024).is_err());
        assert!(row_extent(usize::MAX, usize::MAX, 2, 0, 1).is_err());
        assert!(row_extent(usize::MAX, 0, usize::MAX, usize::MAX, 1).is_err());
        // Eight pick rows per token retain [gate|up] and the router's pick order.
        let (picks, inter, m) = (8, 512, 16);
        let stride = picks * 2 * inter * 4;
        let span = ((picks - 1) * 2 * inter + inter) * 4;
        let up = inter * 4;
        let capacity = m * stride - up;
        let (offset, bytes) = row_extent(capacity, m - 1, stride, 0, span).unwrap();
        assert_eq!(up + offset + bytes, m * stride);
        assert!(row_extent(capacity - 1, m - 1, stride, 0, span).is_err());
    }

    #[test]
    fn diagnostic_native_packs_fit_real_chunk16_slots() {
        use super::super::{common_tests, prefill_plan::chunk_tensors};
        let dims = common_tests::dims();
        let g = MixerGeometry::new(&dims).unwrap();
        let moe = MoeDims::new(&dims, 16);
        // The target diagnostic uses 16 rows in a session planned for 128 rows.
        let list = chunk_tensors(&dims, 128, 82).unwrap();
        for (slot, k, rows) in [
            (ChunkSlot::Q8X, g.hidden, 16),
            (ChunkSlot::Q8OD, g.delta.value_width, 16),
            (ChunkSlot::Q8OA, g.attn.q_width, 16),
            (ChunkSlot::Q8X2, moe.hidden, 16),
            (ChunkSlot::Q8Ash, moe.shared_inter, 16),
            (ChunkSlot::Q8S, moe.hidden, 16),
            (ChunkSlot::Q8Act, moe.inter, 16 * moe.top_k),
        ] {
            let capacity = list
                .iter()
                .find(|(known, _)| *known == slot)
                .unwrap()
                .1
                .bytes;
            let stride = q8_1_bytes(k, 1).unwrap();
            assert!(
                row_extent(capacity, rows - 1, stride, 0, stride).is_ok(),
                "{slot:?}"
            );
        }
    }
}
