//! The launches of one mixer block as DATA (packets RD-9b3a, RD-9b3b): an ordered list of
//! [`BlockOp`] that `Mixers::run_block` executes one by one. Pure host code.
//!
//! The list is the single source of what a block does: the order of the launches, every plan slot
//! each launch reads and writes, the weight roles it names, the `StepParams` words it takes and
//! the column of the shared projection row each projection writes. The executor takes ALL of
//! these from the op it runs (its wrapper arguments that are not slots come from the pure shape
//! builders of `mixer_shapes.rs`); `reads()` and `writes()` are derived from the same fields by a
//! `match`, so there is no second copy. The CPU tests check this list against the step plan
//! (lifetimes, write-before-read, phase order, column tiling), so they check what runs.
//!
//! Design guidance only, no code copied.

use super::{
    dims::KvMode,
    mixer_geometry::MixerGeometry,
    state::StepField,
    state_layout::Plane,
    step_plan::{Phase, Slot},
};
use crate::{
    EngError,
    family::{AttentionKind, gguf_model::Role, project::cuda::WeightLayout},
    kernels::cuda::resident::attn_decode::AttnSplit,
};

/// Rows of every block: the shared projection row exists only at one row (see `mixer_geometry.rs`).
pub(super) const BLOCK_ROWS: usize = 1;

const F32: usize = size_of::<f32>();

/// One launch of a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BlockOp {
    /// RD-4a: `resid += delta` (when given), then `xf = pack(RMSNorm(resid) * weight)` and, when
    /// asked, the f32 normalized row.
    NormPack {
        phase: Phase,
        weight: Role,
        delta: Option<Slot>,
        xf: Slot,
        norm_out: Option<Slot>,
    },
    /// The NATIVE twin of [`BlockOp::NormPack`] (MATVEC-PAR S3b): the same fused add and norm
    /// through `add_rmsnorm_q8`, writing the padded q8_1 row `q8` (the matvec input) beside the
    /// f32 normalized row `norm_out` — the S3a producer always writes both, so every `NormQ8`
    /// names `NormRow` (the router's input at the epilogue, dead scratch at the prologue and the
    /// final norm, which is why the NATIVE plan holds `NormRow` live across the whole layer).
    NormQ8 {
        phase: Phase,
        weight: Role,
        delta: Option<Slot>,
        q8: Slot,
        norm_out: Slot,
    },
    /// One dense projection of the row packed in `xf` into columns `col..` of `y`'s row.
    Project {
        phase: Phase,
        role: Role,
        xf: Slot,
        y: Slot,
        col: usize,
    },
    /// The NATIVE twin of [`BlockOp::Project`] (MATVEC-PAR S3b): one `matvec_q` launch against
    /// the native blocks of `role`, reading the padded q8_1 row `x_q8` and writing `rows` f32
    /// values at column `col` of `y`'s row (one row only, the matvec's `y[j * rows + r]`).
    ProjectQ {
        phase: Phase,
        role: Role,
        x_q8: Slot,
        y: Slot,
        col: usize,
    },
    /// `pack` of the f32 row `x` (`k` values) into `xf`.
    Pack {
        phase: Phase,
        x: Slot,
        xf: Slot,
        k: usize,
    },
    /// The NATIVE twin of [`BlockOp::Pack`] (MATVEC-PAR S3b): the q8_1 quantize of the f32 row
    /// `x` (`k` values) into the padded row `q8`.
    QuantizeQ8 {
        phase: Phase,
        x: Slot,
        q8: Slot,
        k: usize,
    },
    /// RD-7a causal conv: the qkv columns of `proj` -> `out`, with the conv weights `weights`.
    Conv {
        proj: Slot,
        out: Slot,
        weights: Role,
    },
    /// RD-7b gated delta rule: `conv` and the z, beta, alpha columns of `proj` -> `out`.
    Delta {
        conv: Slot,
        proj: Slot,
        out: Slot,
        a: Role,
        dt_bias: Role,
        norm: Role,
    },
    /// RD-8a: `proj` -> `q`, and the K and V rows into the cache at the `pos` word; a refused
    /// position ORs mask 4 into the `status` word.
    AttnPrep {
        proj: Slot,
        q: Slot,
        q_norm: Role,
        k_norm: Role,
        pos: StepField,
        status: StepField,
    },
    /// RD-8b: `q`, the gate columns of `proj` and the cache (up to the `kv_len` word) -> `part`
    /// partials -> `out`.
    AttnDecode {
        q: Slot,
        proj: Slot,
        part: Slot,
        out: Slot,
        kv_len: StepField,
    },
    /// The Int8G32 twin of [`BlockOp::AttnPrep`] (KV8-B B3): the same per-head RMSNorm and
    /// rope through the rows entry of `attn_prep`, but the post-rope k and the unchanged v go
    /// to the f32 rows slots `k_rows`/`v_rows` (indexed by the token) and no fp16 cache byte is
    /// written; a refused position ORs mask 4 into the `status` word.
    AttnPrepRows {
        proj: Slot,
        q: Slot,
        q_norm: Role,
        k_norm: Role,
        k_rows: Slot,
        v_rows: Slot,
        pos: StepField,
        status: StepField,
    },
    /// RD-8k: the KV8 append of the Int8G32 step (KV8-B B3): the f32 `k`/`v` rows of the prep
    /// op, quantized into the four code and scale `planes` of this layer's KV pool at cells
    /// `*pos ..`; a refused position ORs mask 4 into the `status` word.
    Kv8Append {
        k: Slot,
        v: Slot,
        k_codes: Plane,
        k_scales: Plane,
        v_codes: Plane,
        v_scales: Plane,
        pos: StepField,
        status: StepField,
    },
    /// RD-8l: the KV8 split-KV decode attention (KV8-B B3): `q`, the gate columns of `proj` and
    /// the code and scale `planes` (up to the `kv_len` word) -> `part` partials -> `out`, at the
    /// `split` the plan sized the partials with (Q8).
    AttnDecodeQ8 {
        q: Slot,
        proj: Slot,
        k_codes: Plane,
        k_scales: Plane,
        v_codes: Plane,
        v_scales: Plane,
        kv_len: StepField,
        part: Slot,
        out: Slot,
        split: AttnSplit,
    },
}

impl BlockOp {
    pub(super) fn phase(&self) -> Phase {
        match *self {
            Self::NormPack { phase, .. }
            | Self::NormQ8 { phase, .. }
            | Self::Project { phase, .. }
            | Self::ProjectQ { phase, .. }
            | Self::Pack { phase, .. }
            | Self::QuantizeQ8 { phase, .. } => phase,
            Self::Conv { .. } => Phase::DeltaConv,
            Self::Delta { .. } => Phase::DeltaRule,
            Self::AttnPrep { .. } | Self::AttnPrepRows { .. } => Phase::AttnPrep,
            Self::Kv8Append { .. } => Phase::Kv8Append,
            Self::AttnDecode { .. } | Self::AttnDecodeQ8 { .. } => Phase::AttnCore,
        }
    }

    /// The plan slots the launch reads (the residual stream is not a slot).
    pub(super) fn reads(&self) -> Vec<Slot> {
        match *self {
            Self::NormPack { delta, .. } | Self::NormQ8 { delta, .. } => {
                delta.into_iter().collect()
            }
            Self::Project { xf, .. } => vec![xf],
            Self::ProjectQ { x_q8, .. } => vec![x_q8],
            Self::Pack { x, .. } | Self::QuantizeQ8 { x, .. } => vec![x],
            Self::Conv { proj, .. } => vec![proj],
            Self::Delta { conv, proj, .. } => vec![conv, proj],
            Self::AttnPrep { proj, .. } | Self::AttnPrepRows { proj, .. } => vec![proj],
            Self::Kv8Append { k, v, .. } => vec![k, v],
            Self::AttnDecode { q, proj, .. } | Self::AttnDecodeQ8 { q, proj, .. } => {
                vec![q, proj]
            }
        }
    }

    /// The plan slots the launch writes.
    pub(super) fn writes(&self) -> Vec<Slot> {
        match *self {
            Self::NormPack { xf, norm_out, .. } => {
                [Some(xf), norm_out].into_iter().flatten().collect()
            }
            Self::NormQ8 { q8, norm_out, .. } => vec![q8, norm_out],
            Self::Project { y, .. } | Self::ProjectQ { y, .. } => vec![y],
            Self::Pack { xf, .. } => vec![xf],
            Self::QuantizeQ8 { q8, .. } => vec![q8],
            Self::Conv { out, .. } | Self::Delta { out, .. } => vec![out],
            Self::AttnPrep { q, .. } => vec![q],
            Self::AttnPrepRows {
                q, k_rows, v_rows, ..
            } => vec![q, k_rows, v_rows],
            // The quantized cells go to the state's KV planes, which are not plan slots.
            Self::Kv8Append { .. } => Vec::new(),
            Self::AttnDecode { part, out, .. } | Self::AttnDecodeQ8 { part, out, .. } => {
                vec![part, out]
            }
        }
    }
}

/// The byte offset of column `col` inside a fused f32 row, checked for every `ProjectQ` pair of
/// the geometry at block construction (a typed refusal here, never a launch-time wrapper error
/// inside a token): `col * 4` must not overflow and `rows` must fit the fused row's rest. The
/// 4-byte alignment arm was DELETED (MSG-74 item 3): `col * 4` is a multiple of 4 for every
/// integer `col`, the arm was unreachable and its test line a tautology; the wrapper's own
/// alignment check on the (always 4-aligned, because 256-byte arena offsets) slot remains.
pub(super) fn check_native_column(
    role: Role,
    col: usize,
    rows: usize,
    ld: usize,
) -> Result<(), EngError> {
    col.checked_mul(F32).ok_or_else(|| {
        EngError::Checkpoint(format!(
            "mixer ops: the column offset of {role:?} overflows a byte count"
        ))
    })?;
    if col.checked_add(rows).is_none_or(|end| end > ld) {
        return Err(EngError::Checkpoint(format!(
            "mixer ops: {role:?} writes {rows} columns at {col}, past the fused row of {ld}"
        )));
    }
    Ok(())
}

/// The projection twin of one fused-row launch (one role, one column): `Project` against the
/// fp16 pack `xf` or `ProjectQ` against the padded q8_1 row `x_q8` (MATVEC-PAR S3b).
fn project_op(
    phase: Phase,
    role: Role,
    xf: Slot,
    x_q8: Slot,
    y: Slot,
    col: usize,
    layout: WeightLayout,
) -> BlockOp {
    match layout {
        WeightLayout::Tiled => BlockOp::Project {
            phase,
            role,
            xf,
            y,
            col,
        },
        WeightLayout::Native => BlockOp::ProjectQ {
            phase,
            role,
            x_q8,
            y,
            col,
        },
    }
}

/// The pack twin of one core-output launch: `Pack` into the fp16 `xf` or `QuantizeQ8` into the
/// padded q8_1 row `q8` (MATVEC-PAR S3b).
fn pack_op(phase: Phase, x: Slot, xf: Slot, q8: Slot, k: usize, layout: WeightLayout) -> BlockOp {
    match layout {
        WeightLayout::Tiled => BlockOp::Pack { phase, x, xf, k },
        WeightLayout::Native => BlockOp::QuantizeQ8 { phase, x, q8, k },
    }
}

/// The input norm of the mixer: the residual is normalized with the layer's input norm.
fn prologue(layout: WeightLayout) -> BlockOp {
    match layout {
        WeightLayout::Tiled => BlockOp::NormPack {
            phase: Phase::MixPack,
            weight: Role::AttnNorm,
            delta: None,
            xf: Slot::XfMix,
            norm_out: None,
        },
        WeightLayout::Native => BlockOp::NormQ8 {
            phase: Phase::MixPack,
            weight: Role::AttnNorm,
            delta: None,
            q8: Slot::Q8Mix,
            norm_out: Slot::NormRow,
        },
    }
}

/// The residual add of the mixer output and the norm for the MoE block: ends the mixer block.
fn epilogue(layout: WeightLayout) -> BlockOp {
    match layout {
        WeightLayout::Tiled => BlockOp::NormPack {
            phase: Phase::AddNorm,
            weight: Role::PostAttentionNorm,
            delta: Some(Slot::MixOut),
            xf: Slot::XfMoe,
            norm_out: Some(Slot::NormRow),
        },
        WeightLayout::Native => BlockOp::NormQ8 {
            phase: Phase::AddNorm,
            weight: Role::PostAttentionNorm,
            delta: Some(Slot::MixOut),
            q8: Slot::Q8Moe,
            norm_out: Slot::NormRow,
        },
    }
}

fn delta_ops(g: &MixerGeometry, layout: WeightLayout) -> Vec<BlockOp> {
    let d = &g.delta;
    let project = |role, col| {
        project_op(
            Phase::DeltaIn,
            role,
            Slot::XfMix,
            Slot::Q8Mix,
            Slot::DeltaProj,
            col,
            layout,
        )
    };
    vec![
        project(Role::AttnQkv, 0),
        project(Role::AttnGate, d.z_off),
        project(Role::SsmBeta, d.beta_off),
        project(Role::SsmAlpha, d.alpha_off),
        BlockOp::Conv {
            proj: Slot::DeltaProj,
            out: Slot::DeltaConv,
            weights: Role::SsmConv1d,
        },
        BlockOp::Delta {
            conv: Slot::DeltaConv,
            proj: Slot::DeltaProj,
            out: Slot::DeltaCore,
            a: Role::SsmA,
            dt_bias: Role::SsmDtBias,
            norm: Role::SsmNorm,
        },
        pack_op(
            Phase::DeltaPack,
            Slot::DeltaCore,
            Slot::XfDeltaOut,
            Slot::Q8DeltaOut,
            d.value_width,
            layout,
        ),
        project_op(
            Phase::DeltaOut,
            Role::SsmOut,
            Slot::XfDeltaOut,
            Slot::Q8DeltaOut,
            Slot::MixOut,
            0,
            layout,
        ),
    ]
}

fn attn_ops(g: &MixerGeometry, layout: WeightLayout, kv_mode: KvMode) -> Vec<BlockOp> {
    let a = &g.attn;
    let project = |role, col| {
        project_op(
            Phase::AttnIn,
            role,
            Slot::XfMix,
            Slot::Q8Mix,
            Slot::AttnProj,
            col,
            layout,
        )
    };
    // The KV core of the mode (KV8-B B3): the fp16 pair writes the cache rows inside prep and
    // decodes over the fp16 planes; the Int8G32 list hands f32 rows from the rows entry of
    // `attn_prep` to `kv8_append`, which quantizes them into the code and scale planes the q8
    // decode then reads (at the Q8 split, whose min_keys is a multiple of the 64-cell tile).
    let core = match kv_mode {
        KvMode::Fp16 => vec![
            BlockOp::AttnPrep {
                proj: Slot::AttnProj,
                q: Slot::AttnQ,
                q_norm: Role::AttnQNorm,
                k_norm: Role::AttnKNorm,
                pos: StepField::Pos,
                status: StepField::Status,
            },
            BlockOp::AttnDecode {
                q: Slot::AttnQ,
                proj: Slot::AttnProj,
                part: Slot::AttnPart,
                out: Slot::AttnCore,
                kv_len: StepField::KvLen,
            },
        ],
        KvMode::Int8G32 => vec![
            BlockOp::AttnPrepRows {
                proj: Slot::AttnProj,
                q: Slot::AttnQ,
                q_norm: Role::AttnQNorm,
                k_norm: Role::AttnKNorm,
                k_rows: Slot::KRows,
                v_rows: Slot::VRows,
                pos: StepField::Pos,
                status: StepField::Status,
            },
            BlockOp::Kv8Append {
                k: Slot::KRows,
                v: Slot::VRows,
                k_codes: Plane::K,
                k_scales: Plane::KScales,
                v_codes: Plane::V,
                v_scales: Plane::VScales,
                pos: StepField::Pos,
                status: StepField::Status,
            },
            BlockOp::AttnDecodeQ8 {
                q: Slot::AttnQ,
                proj: Slot::AttnProj,
                k_codes: Plane::K,
                k_scales: Plane::KScales,
                v_codes: Plane::V,
                v_scales: Plane::VScales,
                kv_len: StepField::KvLen,
                part: Slot::AttnPart,
                out: Slot::AttnCore,
                split: AttnSplit::Q8,
            },
        ],
    };
    let mut ops = vec![
        project(Role::AttnQ, 0),
        project(Role::AttnK, a.k_off),
        project(Role::AttnV, a.v_off),
    ];
    ops.extend(core);
    ops.push(pack_op(
        Phase::AttnPack,
        Slot::AttnCore,
        Slot::XfAttnOut,
        Slot::Q8AttnOut,
        a.q_width,
        layout,
    ));
    ops.push(project_op(
        Phase::AttnOut,
        Role::AttnOutput,
        Slot::XfAttnOut,
        Slot::Q8AttnOut,
        Slot::MixOut,
        0,
        layout,
    ));
    ops
}

/// Every fused-row column the `ProjectQ` launches of `kind` write, as (role, col, rows, ld): the
/// four (three) in-projections tile `DeltaProj`/`AttnProj` and the output projection writes
/// `MixOut` from column 0.
fn native_columns(g: &MixerGeometry, kind: AttentionKind) -> Vec<(Role, usize, usize, usize)> {
    let (d, a, h) = (&g.delta, &g.attn, g.hidden);
    match kind {
        AttentionKind::DeltaNet => vec![
            (Role::AttnQkv, 0, d.channels, d.ld),
            (Role::AttnGate, d.z_off, d.z_rows, d.ld),
            (Role::SsmBeta, d.beta_off, d.beta_rows, d.ld),
            (Role::SsmAlpha, d.alpha_off, d.alpha_rows, d.ld),
            (Role::SsmOut, 0, h, h),
        ],
        _ => vec![
            (Role::AttnQ, 0, a.q_rows, a.ld),
            (Role::AttnK, a.k_off, a.kv_rows, a.ld),
            (Role::AttnV, a.v_off, a.kv_rows, a.ld),
            (Role::AttnOutput, 0, h, h),
        ],
    }
}

/// The launches of one block for a layer of `kind`, in queue order. With `Tiled` the list is
/// byte-identical to the a4ca4075 lists (pinned by the literal fixture test); with `Native` the
/// `NormPack`/`Project`/`Pack` triples become `NormQ8`/`ProjectQ`/`QuantizeQ8` and the fused-row
/// column offsets are checked first (`check_native_column`), so a bad offset is a typed refusal
/// here at construction, never a launch-time wrapper error inside a token. Under `Int8G32`
/// (Native only, `check_layout_rows`) the attention core becomes the KV8 pair of B3 (the rows
/// entry, the append, the q8 decode).
pub(super) fn block_ops(
    kind: AttentionKind,
    g: &MixerGeometry,
    layout: WeightLayout,
    kv_mode: KvMode,
) -> Result<Vec<BlockOp>, EngError> {
    if layout == WeightLayout::Native {
        for (role, col, rows, ld) in native_columns(g, kind) {
            check_native_column(role, col, rows, ld)?;
        }
    }
    let mut ops = vec![prologue(layout)];
    match kind {
        AttentionKind::DeltaNet => ops.extend(delta_ops(g, layout)),
        AttentionKind::Full => {
            // The plan owns the refusal (`check_layout_rows`); the builder states it too
            // (B2/B3 review item 4): the Int8G32 core's `KRows`/`VRows` slots exist in Native
            // plans only, so a Tiled Int8G32 list could never be planned.
            if kv_mode == KvMode::Int8G32 && layout == WeightLayout::Tiled {
                return Err(EngError::BadRequest(
                    "step ops: the Int8G32 KV mode takes the native layout; Tiled is the fp16 \
                     control path"
                        .to_owned(),
                ));
            }
            ops.extend(attn_ops(g, layout, kv_mode));
        }
        other => {
            return Err(EngError::Checkpoint(format!(
                "block ops: attention kind {other:?} has no resident mixer"
            )));
        }
    }
    ops.push(epilogue(layout));
    Ok(ops)
}
