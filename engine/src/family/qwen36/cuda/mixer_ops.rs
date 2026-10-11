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
    mixer_geometry::MixerGeometry,
    state::StepField,
    step_plan::{Phase, Slot},
};
use crate::{
    EngError,
    family::{AttentionKind, gguf_model::Role},
};

/// Rows of every block: the shared projection row exists only at one row (see `mixer_geometry.rs`).
pub(super) const BLOCK_ROWS: usize = 1;

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
    /// One dense projection of the row packed in `xf` into columns `col..` of `y`'s row.
    Project {
        phase: Phase,
        role: Role,
        xf: Slot,
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
}

impl BlockOp {
    pub(super) fn phase(&self) -> Phase {
        match *self {
            Self::NormPack { phase, .. }
            | Self::Project { phase, .. }
            | Self::Pack { phase, .. } => phase,
            Self::Conv { .. } => Phase::DeltaConv,
            Self::Delta { .. } => Phase::DeltaRule,
            Self::AttnPrep { .. } => Phase::AttnPrep,
            Self::AttnDecode { .. } => Phase::AttnCore,
        }
    }

    /// The plan slots the launch reads (the residual stream is not a slot).
    pub(super) fn reads(&self) -> Vec<Slot> {
        match *self {
            Self::NormPack { delta, .. } => delta.into_iter().collect(),
            Self::Project { xf, .. } => vec![xf],
            Self::Pack { x, .. } => vec![x],
            Self::Conv { proj, .. } => vec![proj],
            Self::Delta { conv, proj, .. } => vec![conv, proj],
            Self::AttnPrep { proj, .. } => vec![proj],
            Self::AttnDecode { q, proj, .. } => vec![q, proj],
        }
    }

    /// The plan slots the launch writes.
    pub(super) fn writes(&self) -> Vec<Slot> {
        match *self {
            Self::NormPack { xf, norm_out, .. } => {
                [Some(xf), norm_out].into_iter().flatten().collect()
            }
            Self::Project { y, .. } => vec![y],
            Self::Pack { xf, .. } => vec![xf],
            Self::Conv { out, .. } | Self::Delta { out, .. } => vec![out],
            Self::AttnPrep { q, .. } => vec![q],
            Self::AttnDecode { part, out, .. } => vec![part, out],
        }
    }
}

/// The input norm of the mixer: the residual is normalized with the layer's input norm.
fn prologue() -> BlockOp {
    BlockOp::NormPack {
        phase: Phase::MixPack,
        weight: Role::AttnNorm,
        delta: None,
        xf: Slot::XfMix,
        norm_out: None,
    }
}

/// The residual add of the mixer output and the norm for the MoE block: ends the mixer block.
fn epilogue() -> BlockOp {
    BlockOp::NormPack {
        phase: Phase::AddNorm,
        weight: Role::PostAttentionNorm,
        delta: Some(Slot::MixOut),
        xf: Slot::XfMoe,
        norm_out: Some(Slot::NormRow),
    }
}

fn delta_ops(g: &MixerGeometry) -> Vec<BlockOp> {
    let d = &g.delta;
    let project = |role, col| BlockOp::Project {
        phase: Phase::DeltaIn,
        role,
        xf: Slot::XfMix,
        y: Slot::DeltaProj,
        col,
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
        BlockOp::Pack {
            phase: Phase::DeltaPack,
            x: Slot::DeltaCore,
            xf: Slot::XfDeltaOut,
            k: d.value_width,
        },
        BlockOp::Project {
            phase: Phase::DeltaOut,
            role: Role::SsmOut,
            xf: Slot::XfDeltaOut,
            y: Slot::MixOut,
            col: 0,
        },
    ]
}

fn attn_ops(g: &MixerGeometry) -> Vec<BlockOp> {
    let a = &g.attn;
    let project = |role, col| BlockOp::Project {
        phase: Phase::AttnIn,
        role,
        xf: Slot::XfMix,
        y: Slot::AttnProj,
        col,
    };
    vec![
        project(Role::AttnQ, 0),
        project(Role::AttnK, a.k_off),
        project(Role::AttnV, a.v_off),
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
        BlockOp::Pack {
            phase: Phase::AttnPack,
            x: Slot::AttnCore,
            xf: Slot::XfAttnOut,
            k: a.q_width,
        },
        BlockOp::Project {
            phase: Phase::AttnOut,
            role: Role::AttnOutput,
            xf: Slot::XfAttnOut,
            y: Slot::MixOut,
            col: 0,
        },
    ]
}

/// The launches of one block for a layer of `kind`, in queue order.
pub(super) fn block_ops(kind: AttentionKind, g: &MixerGeometry) -> Result<Vec<BlockOp>, EngError> {
    let mut ops = vec![prologue()];
    match kind {
        AttentionKind::DeltaNet => ops.extend(delta_ops(g)),
        AttentionKind::Full => ops.extend(attn_ops(g)),
        other => {
            return Err(EngError::Checkpoint(format!(
                "block ops: attention kind {other:?} has no resident mixer"
            )));
        }
    }
    ops.push(epilogue());
    Ok(ops)
}
