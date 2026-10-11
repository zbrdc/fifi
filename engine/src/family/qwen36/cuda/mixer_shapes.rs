//! Pure builders of the shape arguments of the mixer wrappers (packets RD-9b3a, RD-9b3b): plain
//! structs without device ranges, so a CPU test pins every scalar argument of a launch, and the
//! epsilon rule of the DeltaNet kernels. The executor calls these and nothing else for the
//! non-slot arguments of the wrapper calls.
//!
//! Design guidance only, no code copied.

use super::{mixer_geometry::MixerGeometry, mixer_ops::BLOCK_ROWS};
use crate::{
    EngError,
    kernels::cuda::resident::{
        attn_decode::AttnShape,
        attn_prep::{AttnPrepShape, SlotMode},
        gdn_conv::ConvShape,
        gdn_delta::{DeltaShape, GDN_EPS},
    },
};

/// The DeltaNet kernels take exactly this epsilon for the q/k norm and the output norm.
pub(super) fn check_delta_eps(eps: f32) -> Result<(), EngError> {
    if eps.to_bits() != GDN_EPS.to_bits() {
        return Err(EngError::Checkpoint(format!(
            "mixer: the layer norm epsilon is {eps}, the DeltaNet kernels take {GDN_EPS}"
        )));
    }
    Ok(())
}

pub(super) fn conv_shape(g: &MixerGeometry) -> ConvShape {
    ConvShape {
        m: BLOCK_ROWS,
        channels: g.delta.channels,
        ld: g.delta.ld,
    }
}

pub(super) fn delta_shape(g: &MixerGeometry) -> DeltaShape {
    let d = &g.delta;
    DeltaShape {
        m: BLOCK_ROWS,
        n_k_heads: d.key_heads,
        n_v_heads: d.value_heads,
        conv_ld: d.channels,
        ld: d.ld,
        z_off: d.z_off,
        beta_off: d.beta_off,
        alpha_off: d.alpha_off,
    }
}

/// `rows` is the KV capacity of the state.
pub(super) fn attn_prep_shape(g: &MixerGeometry, rows: usize) -> AttnPrepShape {
    let a = &g.attn;
    AttnPrepShape {
        m: BLOCK_ROWS,
        n_head: a.heads_q,
        n_kv_head: a.heads_kv,
        head_dim: a.head_dim,
        n_rot: a.n_rot,
        has_gate: true,
        slot: SlotMode::Position,
        rows,
        ld: a.ld,
    }
}

/// `rows` is the KV capacity of the state.
pub(super) fn attn_decode_shape(g: &MixerGeometry, rows: usize) -> AttnShape {
    let a = &g.attn;
    AttnShape {
        m: BLOCK_ROWS,
        n_head: a.heads_q,
        n_kv_head: a.heads_kv,
        head_dim: a.head_dim,
        cache_rows: rows,
        ld: a.ld,
    }
}
