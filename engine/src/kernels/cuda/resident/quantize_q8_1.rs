//! The q8_1 quantize of activation rows for the MMVQ matvec kernels (MATVEC-PAR S1 c3a,
//! decision 0047, stage 1): one launch of the vendored `quantize_q8_1` kernel per call.
//!
//! [`CudaContext::quantize_q8_1`] queues the kernel on a [`DeviceStream`] and returns when it is
//! queued. It allocates nothing, copies nothing and never synchronizes. The output holds one
//! `block_q8_1` row per input row, padded to [`MATRIX_ROW_PADDING`] (512) values; the kernel
//! writes the zero tail of the padding itself, so a later matvec over the padded width is
//! unaffected by stale bytes beyond the live part (test 9 pins this).
//!
//! The block rule is the vendored kernel's: per 32 values `d = amax/127` and `q =
//! roundf(x/d)` (0 when `amax` is 0), stored as halves, and `s` is the raw f32 SUM OF THE INPUTS
//! reduced by the kernel's fixed warp order. The vendored objects are built with
//! `-use_fast_math` (decision 0049 item 5), so the f32 `d` can be up to 3 ulp from the IEEE
//! value and codes can differ at rounding boundaries; the seam-1 test asserts invariants, not
//! bytes, and reports the bit-equal share.
//!
//! Every refusal is a `BadRequest` and happens before any native call. In order: the backend is
//! usable; `k` is a multiple of 32; `ncols` is in `1..=8`; `x` holds `ncols * k` floats and `y`
//! holds [`q8_1_bytes`] bytes, both 4-byte aligned; `y` overlaps no byte of `x`. A refused call
//! changes nothing.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for RD-9a (decision 0047)")
)]

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::quantize_q8_1::{QuantizeRaw, quantize_q8_1_launch};

/// Values per row are padded up to this (the vendored `MATRIX_ROW_PADDING`).
pub(crate) const MATRIX_ROW_PADDING: usize = 512;

/// One quantize call: `ncols` rows of `k` f32 values in `x`, the padded `block_q8_1` rows in `y`.
pub(crate) struct QuantizeCall<'b> {
    pub(crate) x: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
    pub(crate) k: usize,
    pub(crate) ncols: usize,
}

/// Bytes of the `block_q8_1` output of `ncols` rows of `k` values: each row padded to a multiple
/// of [`MATRIX_ROW_PADDING`] values, one 36-byte block per 32 values. Returns `None` on overflow.
pub(crate) fn q8_1_bytes(k: usize, ncols: usize) -> Option<usize> {
    let blocks = k.checked_add(MATRIX_ROW_PADDING - 1)? & !(MATRIX_ROW_PADDING - 1);
    blocks.checked_div(32)?.checked_mul(36)?.checked_mul(ncols)
}

/// Bytes of the two ranges a quantize call uses: `x` holds `ncols` rows of `k` f32 values and `y`
/// the padded `block_q8_1` rows ([`q8_1_bytes`]). `None` on overflow.
fn quantize_needs(k: usize, ncols: usize) -> Option<(usize, usize)> {
    let x = k.checked_mul(ncols)?.checked_mul(4)?;
    Some((x, q8_1_bytes(k, ncols)?))
}

impl CudaContext {
    /// Queues the q8_1 quantize of `call` on `stream`, writing `call.y`.
    ///
    /// Returns `BadRequest` for a stream or buffer of another adapter, for `k` that is not a
    /// positive multiple of 32, for `ncols` outside `1..=8`, for an `x` or `y` range that is
    /// short or misaligned, and for a `y` that overlaps `x`.
    pub(crate) fn quantize_q8_1(
        &self,
        stream: &DeviceStream,
        call: QuantizeCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_quantize(&call)?;
        self.launch_on(stream, |stream_raw| quantize_q8_1_launch(stream_raw, &raw))
    }

    /// Every check of a quantize, then the native view of its ranges.
    fn plan_quantize<'b>(&self, call: &QuantizeCall<'b>) -> Result<QuantizeRaw<'b>, EngError> {
        let (x_raw, y_raw) = (self.raw_range(&call.x)?, self.raw_range(&call.y)?);
        if call.k == 0 || !call.k.is_multiple_of(32) {
            return Err(EngError::BadRequest(format!(
                "the CUDA quantize needs k as a multiple of 32, got {}",
                call.k
            )));
        }
        if !(1..=8).contains(&call.ncols) {
            return Err(EngError::BadRequest(format!(
                "the CUDA quantize takes 1 to 8 rows, got {}",
                call.ncols
            )));
        }
        let (x_need, y_need) = quantize_needs(call.k, call.ncols)
            .ok_or_else(|| EngError::BadRequest("the CUDA quantize size overflows".to_owned()))?;
        require_fit("x", &call.x, x_need, 4)?;
        require_fit("y", &call.y, y_need, 4)?;
        require_apart(
            &Claim::of("y", &call.y, y_need),
            &Claim::of("x", &call.x, x_need),
        )?;
        Ok(QuantizeRaw {
            x: x_raw,
            y: y_raw,
            k: call.k,
            ncols: call.ncols,
        })
    }
}

#[cfg(test)]
mod tests;
