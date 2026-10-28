//! Column layout of the fused projection rows, the scalar arguments of the mixer kernels and the
//! shapes of the mixer weights (packet RD-9b2). Pure host arithmetic, derived from `StepDims`; no
//! model number is written here.
//!
//! One row of `DeltaProj` is `[qkv | z | beta | alpha]` and one row of `AttnProj` is
//! `[per q head (q | gate) | k | v]`: the projections are separate matmuls, each writing its own
//! `rows` columns of the SAME row, which is only possible for ONE row (a matmul writes `y` as
//! `[m][rows]`). Stage 1 therefore runs the mixers at one row (lane C, Q3); a verify window or
//! chunked prefill needs a matmul with a row stride or one concatenated weight (stage 2 / CPF).

use super::dims::StepDims;
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
    /// Taps of the conv weights per channel (RD-7a's kernel is fixed at 4).
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

/// Taps of the RD-7a conv kernel (`gdn_conv.h`: weights `[channels][4]`).
const CONV_TAPS: usize = 4;

/// A dense (tiled) mixer weight: its role, output rows and input length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DenseSpec {
    pub(super) role: Role,
    pub(super) rows: usize,
    pub(super) k: usize,
}

/// A raw f32 vector of a layer: its role and its length in values.
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
                "mixer geometry: conv_kernel is {}, the RD-7a kernel takes {CONV_TAPS}",
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
        // Same decision as the loader: fusion exists for Native weights only, so the specs
        // never describe a wide qkv another layout's weights cannot satisfy (review F4).
        let fused =
            crate::family::project::cuda::gdn_fuse_inproj() && layout == WeightLayout::Native;
        self.dense_specs_with(fused, kind)
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

    /// The raw f32 vectors of one layer of `kind`, the two layer norms included.
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

/// Refuses a NATIVE dense weight that is not the launch's shape (MATVEC-PAR S3b): the shape rule
/// of [`check_dense`] plus the MMVQ row-group rule of `matvec_q` (`ROWS_MULTIPLE`), `rows % 4 == 0`
/// — the native loader accepts any row count, so the refusal belongs to the step's construction.
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
