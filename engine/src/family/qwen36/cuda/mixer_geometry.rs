//! Column layout of the fused projection rows, the scalar arguments of the mixer kernels and the
//! shapes of the mixer weights. Pure host arithmetic, derived from `StepDims`; no model number is
//! written here.
//!
//! One row of `DeltaProj` is `[qkv | z | beta | alpha]` and one row of `AttnProj` is
//! `[per q head (q | gate) | k | v]`: the projections are separate matmuls, each writing its own
//! `rows` columns of the SAME row, which is only possible for ONE row (a matmul writes `y` as
//! `[m][rows]`). Stage 1 therefore runs the mixers at one row; a verify window or chunked prefill
//! needs a matmul with a row stride or one concatenated weight.

use super::dims::StepDims;
use super::shard::ShardTable;
use crate::family::project::cuda::WeightLayout;
use crate::{
    EngError,
    family::{AttentionKind, gguf_model::Role},
};

/// Columns of `DeltaProj` and the arguments of `gdn_conv` and `gdn_delta`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DeltaGeometry {
    pub(super) key_heads: usize,
    pub(super) value_heads: usize,
    /// Conv channels = q, k and v columns = rows of `attn_qkv`.
    pub(super) channels: usize,
    pub(super) z_rows: usize,
    pub(super) beta_rows: usize,
    pub(super) alpha_rows: usize,
    pub(super) z_off: usize,
    pub(super) beta_off: usize,
    pub(super) alpha_off: usize,
    /// Row length of `DeltaProj`.
    pub(super) ld: usize,
    /// Columns of the delta-rule output = k of the `ssm_out` projection.
    pub(super) value_width: usize,
    /// Length of `ssm_norm`.
    pub(super) value_dim: usize,
    /// Taps of the conv weights per channel (the causal conv kernel is fixed at 4).
    pub(super) conv_kernel: usize,
}

/// Columns of `AttnProj` and the arguments of `attn_prep` and `attn_decode`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct AttnGeometry {
    pub(super) heads_q: usize,
    pub(super) heads_kv: usize,
    pub(super) head_dim: usize,
    pub(super) n_rot: usize,
    pub(super) rope_base: f32,
    pub(super) scale: f32,
    pub(super) q_rows: usize,
    pub(super) kv_rows: usize,
    pub(super) k_off: usize,
    pub(super) v_off: usize,
    /// Row length of `AttnProj`.
    pub(super) ld: usize,
    /// Columns of the attention output = k of the `attn_output` projection.
    pub(super) q_width: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct MixerGeometry {
    pub(super) hidden: usize,
    pub(super) eps: f32,
    pub(super) delta: DeltaGeometry,
    pub(super) attn: AttnGeometry,
}

/// Taps of the causal conv kernel (`gdn_conv.h`: weights `[channels][4]`).
const CONV_TAPS: usize = 4;

/// The DeltaNet gates an EXL3 pack keeps as plain matrices (see `MixerGeometry::plain_specs`).
const PLAIN_ROLES: [Role; 2] = [Role::SsmBeta, Role::SsmAlpha];

/// A dense (tiled) mixer weight: its role, output rows and input length.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family) struct DenseSpec {
    pub(in crate::family) role: Role,
    pub(in crate::family) rows: usize,
    pub(in crate::family) k: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RawSpec {
    pub(super) role: Role,
    pub(super) values: usize,
}

impl MixerGeometry {
    pub(super) fn new(dims: &StepDims) -> Result<Self, EngError> {
        let shape = &dims.shape;
        let (d, a) = (&shape.delta, &shape.attn);
        if d.conv_kernel != CONV_TAPS {
            return Err(EngError::Checkpoint(format!(
                "mixer geometry: conv_kernel is {}, the causal conv kernel takes {CONV_TAPS}",
                d.conv_kernel
            )));
        }
        let channels = d.conv_dim();
        let z_rows = d.value_heads * d.value_dim;
        let z_off = channels;
        let beta_off = z_off + z_rows;
        let alpha_off = beta_off + d.value_heads;
        let delta = DeltaGeometry {
            key_heads: d.key_heads,
            value_heads: d.value_heads,
            channels,
            z_rows,
            beta_rows: d.value_heads,
            alpha_rows: d.value_heads,
            z_off,
            beta_off,
            alpha_off,
            ld: alpha_off + d.value_heads,
            value_width: z_rows,
            value_dim: d.value_dim,
            conv_kernel: d.conv_kernel,
        };
        let q_width = a.heads_q * a.head_dim;
        let q_rows = 2 * q_width;
        let kv_rows = a.heads_kv * a.head_dim;
        let attn = AttnGeometry {
            heads_q: a.heads_q,
            heads_kv: a.heads_kv,
            head_dim: a.head_dim,
            n_rot: a.rotary_dim,
            rope_base: a.theta,
            scale: (a.head_dim as f32).sqrt().recip(),
            q_rows,
            kv_rows,
            k_off: q_rows,
            v_off: q_rows + kv_rows,
            ld: q_rows + 2 * kv_rows,
            q_width,
        };
        // The fused widths of `StepPlan` come from the inventory roles; both must agree.
        if delta.ld != dims.delta_proj || attn.ld != dims.attn_proj {
            return Err(EngError::Checkpoint(format!(
                "mixer geometry: projection row lengths {} and {} differ from the plan's {} and {}",
                delta.ld, attn.ld, dims.delta_proj, dims.attn_proj
            )));
        }
        Ok(Self {
            hidden: shape.hidden,
            eps: shape.eps,
            delta,
            attn,
        })
    }

    /// The geometry of `card` of a `table`-sharded model: every field of
    /// [`Self::new`] with the sharded widths replaced by the card's run —
    /// its key/value-head runs, its channel segments of `AttnQkv`, its
    /// query-head run and its K/V projection rows. The replicated widths
    /// stay full: `hidden` (the residual row every card holds), `value_dim`,
    /// `head_dim` and the RoPE facts. `heads_kv` is the CARD's own K/V head
    /// groups (the cache planes are sliced to the card); only the gathered
    /// full row (`Slot::KRows`/`VRows`) keeps the whole shape's width. A
    /// card's projection rows (`ld`) are therefore narrower than
    /// the single-device row, so the plan-width check of `new` does not
    /// apply; the two table-consistency checks below replace it.
    ///
    /// The card's K/V projection rows are its slice of the full row, not a
    /// head multiple: ops that need the absolute row index (the RoPE phase
    /// of a K row) read `table.kv_rows(card)`, not this geometry's slice
    /// length. The full new K/V row still reappears on every card through
    /// the ring gather, but the append writes only the card's heads into
    /// its sliced cache planes.
    pub(super) fn for_card(
        dims: &StepDims,
        table: &ShardTable,
        card: usize,
    ) -> Result<Self, EngError> {
        let shape = &dims.shape;
        let (d, a) = (&shape.delta, &shape.attn);
        if d.conv_kernel != CONV_TAPS {
            return Err(EngError::Checkpoint(format!(
                "mixer geometry: conv_kernel is {}, the causal conv kernel takes {CONV_TAPS}",
                d.conv_kernel
            )));
        }
        let key_heads = table.key_heads(card)?.len();
        let value_heads = table.value_heads(card)?.len();
        let segments = table.delta_qkv_segments(card)?;
        let channels = segments.iter().map(|seg| seg.len()).sum::<usize>();
        let z_rows = value_heads * d.value_dim;
        let z_off = channels;
        let beta_off = z_off + z_rows;
        let alpha_off = beta_off + value_heads;
        let value_width = z_rows;
        if table.delta_out_cols(card)?.len() != value_width {
            return Err(EngError::Checkpoint(format!(
                "shard table: card {card} holds {} out columns for {} value-head columns",
                table.delta_out_cols(card)?.len(),
                value_width
            )));
        }
        let delta = DeltaGeometry {
            key_heads,
            value_heads,
            channels,
            z_rows,
            beta_rows: value_heads,
            alpha_rows: value_heads,
            z_off,
            beta_off,
            alpha_off,
            ld: alpha_off + value_heads,
            value_width,
            value_dim: d.value_dim,
            conv_kernel: d.conv_kernel,
        };
        let heads_q = table.query_heads(card)?.len();
        let q_width = heads_q * a.head_dim;
        if table.attn_o_cols(card)?.len() != q_width {
            return Err(EngError::Checkpoint(format!(
                "shard table: card {card} holds {} output columns for {} query-head columns",
                table.attn_o_cols(card)?.len(),
                q_width
            )));
        }
        let q_rows = 2 * q_width;
        let kv_rows = table.kv_rows(card)?.len();
        if !kv_rows.is_multiple_of(a.head_dim) {
            return Err(EngError::Checkpoint(format!(
                "mixer geometry: card {card} kv rows {kv_rows} are not whole {}-dim heads",
                a.head_dim
            )));
        }
        let attn = AttnGeometry {
            heads_q,
            // The card's own K/V head groups: the cache planes are SLICED to
            // the card (each card stores, appends and decodes only its heads;
            // see ResidentState::allocate_with_kv_heads). The whole-row
            // gather buffers stay at the full shape's width.
            heads_kv: kv_rows / a.head_dim,
            head_dim: a.head_dim,
            n_rot: a.rotary_dim,
            rope_base: a.theta,
            scale: (a.head_dim as f32).sqrt().recip(),
            q_rows,
            kv_rows,
            k_off: q_rows,
            v_off: q_rows + kv_rows,
            ld: q_rows + 2 * kv_rows,
            q_width,
        };
        Ok(Self {
            hidden: shape.hidden,
            eps: shape.eps,
            delta,
            attn,
        })
    }

    /// The dense weights of one layer of `kind`, in launch order. With the GDN input-projection
    /// fusion (default OFF) the four same-input siblings of a DeltaNet layer become ONE spec of
    /// their summed rows under `AttnQkv` - the fused upload of `project::cuda` resolves that
    /// name to the row-concatenated matrix, and the single launch writes the same DeltaProj
    /// columns (contiguous by construction: `ld` is exactly the sum, re-checked here).
    pub(super) fn dense_specs(
        &self,
        layout: WeightLayout,
        kind: AttentionKind,
    ) -> Result<Vec<DenseSpec>, EngError> {
        // Same decision as the loader: fusion exists for Native weights only, so the specs never
        // describe a wide qkv another layout's weights cannot satisfy.
        let fused =
            crate::family::project::cuda::gdn_fuse_inproj() && layout == WeightLayout::Native;
        let mut specs = self.dense_specs_with(fused, kind)?;
        if layout == WeightLayout::Exl3 {
            specs.retain(|spec| !PLAIN_ROLES.contains(&spec.role));
        }
        Ok(specs)
    }

    /// The projections that stay plain F32 matrices beside the EXL3 ones: the pack does not
    /// quantize the 32-row step and decay gates of a DeltaNet layer.
    pub(super) fn plain_specs(&self, layout: WeightLayout, kind: AttentionKind) -> Vec<DenseSpec> {
        if layout != WeightLayout::Exl3 || kind != AttentionKind::DeltaNet {
            return Vec::new();
        }
        let (h, d) = (self.hidden, &self.delta);
        vec![
            DenseSpec {
                role: Role::SsmBeta,
                rows: d.beta_rows,
                k: h,
            },
            DenseSpec {
                role: Role::SsmAlpha,
                rows: d.alpha_rows,
                k: h,
            },
        ]
    }

    /// The `dense_specs` twin with an explicit fusion flag, so the CPU tests pin both arms
    /// without touching the environment.
    pub(super) fn dense_specs_with(
        &self,
        fuse_inproj: bool,
        kind: AttentionKind,
    ) -> Result<Vec<DenseSpec>, EngError> {
        let (h, d, a) = (self.hidden, &self.delta, &self.attn);
        let spec = |role, rows, k| DenseSpec { role, rows, k };
        match kind {
            AttentionKind::DeltaNet if fuse_inproj => {
                let fused_rows = d.channels + d.z_rows + d.beta_rows + d.alpha_rows;
                if fused_rows != d.ld {
                    return Err(EngError::Checkpoint(format!(
                        "mixer geometry: the fused input projection needs contiguous \
                         columns, but {fused_rows} rows do not fill the fused row of {}",
                        d.ld
                    )));
                }
                Ok(vec![
                    spec(Role::AttnQkv, fused_rows, h),
                    spec(Role::SsmOut, h, d.value_width),
                ])
            }
            AttentionKind::DeltaNet => Ok(vec![
                spec(Role::AttnQkv, d.channels, h),
                spec(Role::AttnGate, d.z_rows, h),
                spec(Role::SsmBeta, d.beta_rows, h),
                spec(Role::SsmAlpha, d.alpha_rows, h),
                spec(Role::SsmOut, h, d.value_width),
            ]),
            AttentionKind::Full => Ok(vec![
                spec(Role::AttnQ, a.q_rows, h),
                spec(Role::AttnK, a.kv_rows, h),
                spec(Role::AttnV, a.kv_rows, h),
                spec(Role::AttnOutput, h, a.q_width),
            ]),
            other => Err(no_mixer(other)),
        }
    }

    pub(super) fn raw_specs(&self, kind: AttentionKind) -> Result<Vec<RawSpec>, EngError> {
        let (h, d, a) = (self.hidden, &self.delta, &self.attn);
        let spec = |role, values| RawSpec { role, values };
        let mut specs = vec![spec(Role::AttnNorm, h), spec(Role::PostAttentionNorm, h)];
        match kind {
            AttentionKind::DeltaNet => specs.extend([
                spec(Role::SsmConv1d, d.channels * d.conv_kernel),
                spec(Role::SsmA, d.value_heads),
                spec(Role::SsmDtBias, d.value_heads),
                spec(Role::SsmNorm, d.value_dim),
            ]),
            AttentionKind::Full => specs.extend([
                spec(Role::AttnQNorm, a.head_dim),
                spec(Role::AttnKNorm, a.head_dim),
            ]),
            other => return Err(no_mixer(other)),
        }
        Ok(specs)
    }
}

fn no_mixer(kind: AttentionKind) -> EngError {
    EngError::Checkpoint(format!(
        "mixer geometry: attention kind {kind:?} has no resident mixer"
    ))
}

/// Refuses a NATIVE dense weight that is not the launch's shape: the shape rule of [`check_dense`]
/// plus the MMVQ row-group rule of `matvec_q` (`ROWS_MULTIPLE`), `rows % 4 == 0` — the native
/// loader accepts any row count, so the refusal belongs to the step's construction.
pub(super) fn check_native_dense(
    name: &str,
    have: (usize, usize),
    want: &DenseSpec,
) -> Result<(), EngError> {
    let (rows, k) = have;
    check_dense(name, (rows, k, None), want)?;
    if !rows.is_multiple_of(4) {
        return Err(EngError::Checkpoint(format!(
            "mixer weights: {name} has {rows} rows, the native matvec needs a multiple of 4"
        )));
    }
    Ok(())
}

/// Refuses a dense weight whose shape is not the one the launch needs. `experts` is the stack
/// depth: a dense mixer weight has none.
pub(super) fn check_dense(
    name: &str,
    have: (usize, usize, Option<usize>),
    want: &DenseSpec,
) -> Result<(), EngError> {
    let (rows, k, experts) = have;
    if rows != want.rows || k != want.k || experts.is_some() {
        return Err(EngError::Checkpoint(format!(
            "mixer weights: {name} is {rows} rows of {k} (stack {experts:?}), the step needs {} rows of {} and no stack",
            want.rows, want.k
        )));
    }
    Ok(())
}
