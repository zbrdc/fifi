//! The decode dq matvec: one f32 activation row `x` through a native Q4_K or Q6_K weight,
//! `y[r] = sum_k x[k] * w[r][k]`, with the weights decoded per value and each block's sum
//! rounded in f32 (not an exact f32 product: the CPU reference holds within
//! `64 * 2^-24 * sum |w x|`, see `tests.rs`). That reference is `reference::weight_value` of
//! `resident/matvec_q/tests.rs`, a transcription of ggml's Q4_K and Q6_K layouts, not the
//! crate's `weights/gguf_dequant.rs`.
//!
//! [CudaContext::matvec_dq] queues ONE kernel on a [DeviceStream] and returns when it is
//! queued. It allocates nothing, copies nothing and never synchronizes.
//!
//! [CudaContext::matvec_dq_native] is the same launch over a [NativeWeight]: the SenseNova
//! decode step's dq route (`FIFI_SENSENOVA_DECODE_GEMM=dq`) takes it for every projection.
//!
//! Every refusal of the conditions below is a BadRequest and happens before any native call:
//! the backend is usable, the type is Q4_K or Q6_K, k is a positive multiple of 256, rows is
//! above zero, w holds the rows' blocks and x k floats at 16-byte alignment, y holds rows
//! floats at 4-byte alignment, and y overlaps no byte of w or x. One refusal comes later and
//! as a Backend error: a k or a rows at or above 2^31 passes these checks and is refused by
//! the native entry, as the native entries of matvec_q, matvec_q5k, matvec_gguf and
//! matvec_q3k refuse it.

use std::sync::OnceLock;

use crate::EngError;
use crate::kernels::cuda::NativeWeight;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::matvec_dq::{MatvecDqRaw, matvec_dq_launch};
use crate::kernels::cuda::resident::matvec_q::{MatvecCall, QuantType};

const F32: usize = size_of::<f32>();

/// Whether `matvec_dq` serves `ty` (`Q4_K`, `Q6_K`, or `Q8_0`). Single source of truth for the
/// kernel, its wrappers, and callers in `family/step`.
pub(crate) fn serves(ty: QuantType) -> bool {
    matches!(ty, QuantType::Q4K | QuantType::Q6K | QuantType::Q80)
}

/// Comma-separated role list (`FIFI_CUDA_DECODE_DQ_ROLES`, default `attn_q,attn_v`; set `0`,
/// `off`, or `none` to disable) selecting 1-row decode projections that bypass `q8_1` activation
/// quantization via [`CudaContext::matvec_dq_native`] (`Q4_K`, `Q6_K`, or `Q8_0`, `k % 256 == 0`).
/// Supported tags include `all`, `attn_q`, `attn_k`, `attn_v`, `attn_output`, `attn_qkv`,
/// `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_out`, `delta_inputs` (`attn_qkv` + `attn_gate`),
/// and `delta_logits` (`ssm_alpha` + `ssm_beta`).
pub(crate) fn decode_dq_role_enabled(role: &str) -> bool {
    if role.is_empty() {
        return false;
    }
    static ROLES: OnceLock<Vec<String>> = OnceLock::new();
    let list = ROLES.get_or_init(|| match std::env::var("FIFI_CUDA_DECODE_DQ_ROLES") {
        Ok(raw) => {
            let trimmed = raw.trim().to_ascii_lowercase();
            if trimmed.is_empty() || matches!(trimmed.as_str(), "0" | "off" | "none") {
                Vec::new()
            } else {
                trimmed
                    .split(',')
                    .map(|part| part.trim().to_owned())
                    .filter(|part| !part.is_empty())
                    .collect()
            }
        }
        Err(_) => vec!["attn_q".to_owned(), "attn_v".to_owned()],
    });
    if list.is_empty() {
        return false;
    }
    list.iter().any(|entry| {
        entry == "all"
            || entry == role
            || (entry == "delta_inputs" && matches!(role, "attn_qkv" | "attn_gate"))
            || (entry == "delta_logits" && matches!(role, "ssm_alpha" | "ssm_beta"))
    })
}

/// One row through one native weight: `y[r] = sum_k x[k] * w[r][k]`.
pub(crate) struct MatvecDqCall<'b> {
    pub(crate) ty: QuantType,
    pub(crate) w: DeviceRange<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
    pub(crate) k: usize,
    pub(crate) rows: usize,
}

impl CudaContext {
    /// `x` (k floats) through `w` (`rows` rows of native blocks) into `y` (rows floats).
    pub(crate) fn matvec_dq(
        &self,
        stream: &DeviceStream,
        call: MatvecDqCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_matvec_dq(&call)?;
        self.launch_on(stream, |stream_raw| matvec_dq_launch(stream_raw, &raw))
    }

    /// One f32 row (k floats) through the resident `weight` into `y` (rows floats): the call
    /// above over the weight's own type, blocks, `k` and rows. A weight the kernel does not
    /// serve (a type other than Q4_K or Q6_K, a `k` that is not a multiple of 256) is the
    /// matvec's own `BadRequest`, before any launch: nothing else runs in its place.
    pub(crate) fn matvec_dq_native(
        &self,
        stream: &DeviceStream,
        weight: &NativeWeight,
        x: DeviceRange<'_>,
        y: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        self.matvec_dq(
            stream,
            MatvecDqCall {
                ty: weight.quant(),
                w: weight.buffer().range(0, weight.buffer().bytes())?,
                x,
                y,
                k: weight.k(),
                rows: weight.n_rows(),
            },
        )
    }

    /// One-row projection helper shared by `qwen36/cuda`: uses `matvec_dq_native` on `x_f32` when
    /// `FIFI_CUDA_DECODE_DQ_ROLES` enables `role_tag` and the weight is served by `matvec_dq`;
    /// otherwise runs the unchanged `matvec_q` path over `x_q8`.
    pub(crate) fn project_1row_native(
        &self,
        stream: &DeviceStream,
        role_tag: &str,
        weight: &NativeWeight,
        x_f32: Option<DeviceRange<'_>>,
        x_q8: DeviceRange<'_>,
        y: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        let ty = weight.quant();
        if decode_dq_role_enabled(role_tag)
            && serves(ty)
            && weight.k().is_multiple_of(256)
            && let Some(x) = x_f32
        {
            return self.matvec_dq_native(stream, weight, x.slice(0, weight.k() * F32)?, y);
        }
        self.matvec_q(
            stream,
            MatvecCall {
                ty,
                w: weight.buffer().range(0, weight.buffer().bytes())?,
                x_q8,
                y,
                k: weight.k(),
                rows: weight.n_rows(),
                ncols: 1,
            },
        )
    }

    fn plan_matvec_dq<'b>(&self, call: &MatvecDqCall<'b>) -> Result<MatvecDqRaw<'b>, EngError> {
        let (k, rows) = (call.k, call.rows);
        let shape = || {
            EngError::BadRequest(
                "matvec_dq needs a Q4_K, Q6_K or Q8_0 weight, k a positive multiple of 256 and rows above zero"
                    .to_owned(),
            )
        };
        if !serves(call.ty) || k == 0 || !k.is_multiple_of(256) || rows == 0 {
            return Err(shape());
        }
        let w_bytes = call.ty.weight_bytes(rows, k).ok_or_else(shape)?;
        let x_bytes = k.checked_mul(F32).ok_or_else(shape)?;
        let y_bytes = rows.checked_mul(F32).ok_or_else(shape)?;
        require_fit("matvec_dq w", &call.w, w_bytes, 16)?;
        require_fit("matvec_dq x", &call.x, x_bytes, 16)?;
        require_fit("matvec_dq y", &call.y, y_bytes, 4)?;
        let y = Claim::of("matvec_dq y", &call.y, y_bytes);
        require_apart(&y, &Claim::of("matvec_dq w", &call.w, w_bytes))?;
        require_apart(&y, &Claim::of("matvec_dq x", &call.x, x_bytes))?;
        Ok(MatvecDqRaw {
            ty: call.ty,
            w: self.raw_range(&call.w)?,
            x: self.raw_range(&call.x)?,
            y: self.raw_range(&call.y)?,
            k,
            rows,
        })
    }
}

#[cfg(test)]
mod tests;
