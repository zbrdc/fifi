//! The fused residual add, RMSNorm and q8_1 quantize of the resident step (MATVEC-PAR S3a K1,
//! decision 0047, stage 1): [`CudaContext::add_rmsnorm_q8`] queues ONE kernel on a
//! [`DeviceStream`] for the step's single row (m = 1) and returns when it is queued. It allocates
//! nothing, copies nothing and never synchronizes.
//!
//! Outputs: `norm_out` holds the f32 normalized row, byte-identical to what `add_norm_pack`
//! writes on the same input (the native kernel restates `add_rmsnorm_pack_k`'s strided sum of
//! squares, its block reduction order, its `rsqrtf` and its expression `row[i] * rstd * w[i]`;
//! the router keeps reading it). `q8` holds the `block_q8_1` quantization of the same values,
//! one warp per 32-value block over the row padded to [`MATRIX_ROW_PADDING`] (512) values, the
//! tail past `n` written as zeros. The quantizer rule is the vendored `quantize_q8_1`'s (`d =
//! amax/127`, `q = amax == 0 ? 0 : roundf(x/d)`, `ds = half2(d, sum x)` with the warp butterfly
//! sum) computed in IEEE arithmetic: the native TU is in the warnings-into-errors group WITHOUT
//! `-use_fast_math` (decision 0049 item 5 names the vendored objects only), so `d` is the IEEE
//! division and the `q8` bytes are bit-checkable against a host reference.
//!
//! Every refusal is a `BadRequest` and happens before any native call. In order: the backend is
//! usable; `n` is a multiple of 256 in `256..=32768` (the bounds of today's `add_norm_pack`); `eps`
//! is finite and above 0; `h`, `delta`, `weight` and `norm_out` hold `n` floats and `q8` holds
//! [`q8_1_bytes`]`(n, 1)` bytes, with `norm_out` 16-byte aligned and the rest 4; the written
//! ranges (`h`, `norm_out`, `q8`) overlap no other range of the call (`delta` and `weight` may
//! overlap each other). A refused call changes nothing.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for RD-9a (decision 0047)")
)]

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::add_rmsnorm_q8::{AddNormQ8Raw, add_rmsnorm_q8_launch};
use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;

/// `n` is a multiple of this (the vendored kernel's unit, as `add_norm_pack`).
pub(crate) const NORM_WIDTH_UNIT: usize = 256;
/// The widest row one call takes (the vendored `kern::kMaxK`).
pub(crate) const MAX_WIDTH: usize = 32_768;

/// One fused add, norm and q8_1 quantize call of the resident step's row. All ranges are device
/// ranges of this context.
///
/// - `h`: f32 `[n]`, 4-byte aligned; read, and written in place with `h + delta` when `delta` is
///   given. It is the new residual.
/// - `delta`: `None`, or f32 `[n]`, 4-byte aligned, read only. It may share bytes with `weight`.
/// - `weight`: f32 `[n]`, 4-byte aligned, read only (the RMSNorm weight).
/// - `norm_out`: f32 `[n]`, 16-byte aligned; the normalized row (the router input).
/// - `q8`: [`q8_1_bytes`]`(n, 1)` bytes, 4-byte aligned; the padded `block_q8_1` row.
/// - `eps`: finite and above 0 (GGUF `layer_norm_rms_epsilon`).
pub(crate) struct AddNormQ8Call<'b> {
    pub(crate) n: usize,
    pub(crate) eps: f32,
    pub(crate) h: DeviceRange<'b>,
    pub(crate) delta: Option<DeviceRange<'b>>,
    pub(crate) weight: DeviceRange<'b>,
    pub(crate) norm_out: DeviceRange<'b>,
    pub(crate) q8: DeviceRange<'b>,
}

/// Bytes of the ranges an `add_rmsnorm_q8` call uses: `n` floats for `h`, `delta`, `weight` and
/// `norm_out`, and the padded `block_q8_1` row for `q8`. `None` on overflow.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct AddNormQ8Needs {
    pub(crate) row: usize,
    pub(crate) weight: usize,
    pub(crate) q8: usize,
}

/// The byte counts the wrapper checks a call against. Pure: no context, no device.
pub(crate) fn add_rmsnorm_q8_needs(n: usize) -> Option<AddNormQ8Needs> {
    Some(AddNormQ8Needs {
        row: n.checked_mul(4)?,
        weight: n.checked_mul(4)?,
        q8: q8_1_bytes(n, 1)?,
    })
}

impl CudaContext {
    /// Queues the fused add, RMSNorm and q8_1 quantize of `call` on `stream`, writing `norm_out`,
    /// `q8` and (when `delta` is given) `h` in place.
    ///
    /// Returns `BadRequest` for a stream or buffer of another adapter, for `n` that is not a
    /// multiple of 256 in `256..=32768`, for `eps` not finite and above 0, for a range that is
    /// short or misaligned, and for a written range that overlaps another range of the call.
    pub(crate) fn add_rmsnorm_q8(
        &self,
        stream: &DeviceStream,
        call: AddNormQ8Call<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_add_rmsnorm_q8(&call)?;
        self.launch_on(stream, |stream_raw| add_rmsnorm_q8_launch(stream_raw, &raw))
    }

    /// Every check of a call, then the native view of its ranges.
    fn plan_add_rmsnorm_q8<'b>(
        &self,
        call: &AddNormQ8Call<'b>,
    ) -> Result<AddNormQ8Raw<'b>, EngError> {
        let h = self.raw_range(&call.h)?;
        let delta = call.delta.as_ref().map(|r| self.raw_range(r)).transpose()?;
        let weight = self.raw_range(&call.weight)?;
        let norm_out = self.raw_range(&call.norm_out)?;
        let q8 = self.raw_range(&call.q8)?;
        if call.n < NORM_WIDTH_UNIT || call.n > MAX_WIDTH {
            return Err(EngError::BadRequest(format!(
                "the CUDA add_rmsnorm_q8 takes n in {NORM_WIDTH_UNIT} to {MAX_WIDTH}, got {}",
                call.n
            )));
        }
        if !call.n.is_multiple_of(NORM_WIDTH_UNIT) {
            return Err(EngError::BadRequest(format!(
                "the CUDA add_rmsnorm_q8 needs n as a multiple of {NORM_WIDTH_UNIT}, got {}",
                call.n
            )));
        }
        if !call.eps.is_finite() || call.eps <= 0.0 {
            return Err(EngError::BadRequest(
                "the CUDA add_rmsnorm_q8 needs eps finite and above 0".to_owned(),
            ));
        }
        let need = add_rmsnorm_q8_needs(call.n).ok_or_else(|| {
            EngError::BadRequest("the CUDA add_rmsnorm_q8 size overflows".to_owned())
        })?;
        require_fit("h", &call.h, need.row, 4)?;
        if let Some(delta) = &call.delta {
            require_fit("delta", delta, need.row, 4)?;
        }
        require_fit("weight", &call.weight, need.weight, 4)?;
        require_fit("norm_out", &call.norm_out, need.row, 16)?;
        require_fit("q8", &call.q8, need.q8, 4)?;
        let written = [
            Claim::of("h", &call.h, need.row),
            Claim::of("norm_out", &call.norm_out, need.row),
            Claim::of("q8", &call.q8, need.q8),
        ];
        let read = [
            call.delta
                .as_ref()
                .map(|delta| Claim::of("delta", delta, need.row)),
            Some(Claim::of("weight", &call.weight, need.weight)),
        ];
        for (i, claim) in written.iter().enumerate() {
            for other in written[i + 1..].iter().chain(read.iter().flatten()) {
                require_apart(claim, other)?;
            }
        }
        Ok(AddNormQ8Raw {
            h,
            delta,
            weight,
            norm_out,
            q8,
            n: call.n,
            eps: call.eps,
        })
    }
}

#[cfg(test)]
pub(in crate::kernels::cuda) mod tests;
