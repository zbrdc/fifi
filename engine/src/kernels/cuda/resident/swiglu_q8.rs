//! The batched SwiGLU + q8_1 quantize of the expert slots: [`CudaContext::swiglu_q8`] queues
//! ONE kernel on a [`DeviceStream`] for all `slots` rows and returns when it is queued. It
//! allocates nothing, copies nothing, memsets nothing and never synchronizes (the old
//! `swiglu_pack` needed tail memsets; this producer quantizes the padded width directly, so its
//! input is read only).
//!
//! `gu` holds the step's `ExpertGu` rows, the strided layout: slot `s` at `gu + s * y_slot_stride`
//! ELEMENTS, gate at columns `0..n_ff` and up at `n_ff..2*n_ff` of that row. `q8` holds `slots`
//! padded `block_q8_1` rows back to back (`q8_1_bytes(n_ff, slots)` bytes); the shared expert is
//! the same call with `slots = 1` on its own buffer. The activation is `silu(gate) * up` with
//! `swiglu_pack_k`'s fast form `x / (1 + __expf(-x))` — `__expf` kept on purpose, the numerics
//! domain of the `swiglu_pack` doc is the stated tolerance basis — and the block rule is the
//! vendored `quantize_q8_1`'s computed in IEEE arithmetic (the native TU is in the
//! warnings-into-errors group WITHOUT `-use_fast_math`), so given the kernel's silu values the `q8`
//! bytes are bit-checkable against a host reference.
//!
//! Every refusal is a `BadRequest` and happens before any native call. In order: the backend is
//! usable; `n_ff` is a multiple of 32 in `32..=32768`; `slots` is in `1..=8`; `y_slot_stride` is at
//! least `2 * n_ff`; `gu` holds the strided footprint and `q8` the padded rows, both 4-byte
//! aligned; `q8` overlaps no byte of `gu`. A refused call changes nothing.

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::swiglu_q8::{SwigluQ8Raw, swiglu_q8_launch};
use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;

/// The most values of one slot half.
pub(crate) const MAX_FF: usize = 32_768;
/// A slot half is a multiple of this (one q8_1 block).
pub(crate) const FF_UNIT: usize = 32;
/// Slots per call (top_k of the step).
pub(crate) const MAX_SLOTS: usize = 8;

/// One batched SwiGLU call. Both ranges are device ranges of this context.
///
/// - `gu`: f32 `[slots][y_slot_stride]` (read only the first `2 * n_ff` columns of each row).
/// - `q8`: [`q8_1_bytes`]`(n_ff, slots)` bytes, 4-byte aligned.
/// - `n_ff`: live values per half, a multiple of 32 in `32..=32768`.
/// - `y_slot_stride`: elements between slot rows, at least `2 * n_ff`.
/// - `slots`: expert slots, `1..=8`.
pub(crate) struct SwigluQ8Call<'b> {
    pub(crate) gu: DeviceRange<'b>,
    pub(crate) q8: DeviceRange<'b>,
    pub(crate) n_ff: usize,
    pub(crate) y_slot_stride: usize,
    pub(crate) slots: usize,
}

/// Bytes of the two ranges a call uses. `None` on overflow or for `slots` 0.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct SwigluQ8Needs {
    pub(crate) gu: usize,
    pub(crate) q8: usize,
}

/// The byte counts the wrapper checks a call against: `gu` the tight strided footprint
/// `(slots - 1) * y_slot_stride + 2 * n_ff` floats, `q8` the padded `block_q8_1` rows. Pure: no
/// context, no device.
pub(crate) fn swiglu_q8_needs(
    n_ff: usize,
    y_slot_stride: usize,
    slots: usize,
) -> Option<SwigluQ8Needs> {
    let gu = y_slot_stride
        .checked_mul(slots.checked_sub(1)?)?
        .checked_add(n_ff.checked_mul(2)?)?
        .checked_mul(4)?;
    Some(SwigluQ8Needs {
        gu,
        q8: q8_1_bytes(n_ff, slots)?,
    })
}

impl CudaContext {
    /// Queues the batched SwiGLU and q8_1 quantize of `call` on `stream`, writing `call.q8`.
    ///
    /// Returns `BadRequest` for a stream or buffer of another adapter, for `n_ff` that is not a
    /// multiple of 32 in `32..=32768`, for `slots` outside `1..=8`, for a `y_slot_stride` below
    /// `2 * n_ff`, for a `gu` or `q8` range that is short or misaligned, and for a `q8` that
    /// overlaps `gu`.
    pub(crate) fn swiglu_q8(
        &self,
        stream: &DeviceStream,
        call: SwigluQ8Call<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_swiglu_q8(&call)?;
        self.launch_on(stream, |stream_raw| swiglu_q8_launch(stream_raw, &raw))
    }

    /// Every check of a call, then the native view of its ranges.
    fn plan_swiglu_q8<'b>(&self, call: &SwigluQ8Call<'b>) -> Result<SwigluQ8Raw<'b>, EngError> {
        let (gu_raw, q8_raw) = (self.raw_range(&call.gu)?, self.raw_range(&call.q8)?);
        if call.n_ff < FF_UNIT || call.n_ff > MAX_FF {
            return Err(EngError::BadRequest(format!(
                "the CUDA swiglu_q8 takes n_ff in {FF_UNIT} to {MAX_FF}, got {}",
                call.n_ff
            )));
        }
        if !call.n_ff.is_multiple_of(FF_UNIT) {
            return Err(EngError::BadRequest(format!(
                "the CUDA swiglu_q8 needs n_ff as a multiple of {FF_UNIT}, got {}",
                call.n_ff
            )));
        }
        if call.slots < 1 || call.slots > MAX_SLOTS {
            return Err(EngError::BadRequest(format!(
                "the CUDA swiglu_q8 takes 1 to {MAX_SLOTS} slots, got {}",
                call.slots
            )));
        }
        if call.y_slot_stride < 2 * call.n_ff {
            return Err(EngError::BadRequest(format!(
                "the CUDA swiglu_q8 y stride {} is below the {} values of a slot",
                call.y_slot_stride,
                2 * call.n_ff
            )));
        }
        let need = swiglu_q8_needs(call.n_ff, call.y_slot_stride, call.slots)
            .ok_or_else(|| EngError::BadRequest("the CUDA swiglu_q8 size overflows".to_owned()))?;
        require_fit("gu", &call.gu, need.gu, 4)?;
        require_fit("q8", &call.q8, need.q8, 4)?;
        require_apart(
            &Claim::of("q8", &call.q8, need.q8),
            &Claim::of("gu", &call.gu, need.gu),
        )?;
        Ok(SwigluQ8Raw {
            gu: gu_raw,
            q8: q8_raw,
            n_ff: call.n_ff,
            y_slot_stride: call.y_slot_stride,
            slots: call.slots,
        })
    }
}

#[cfg(test)]
pub(in crate::kernels::cuda) mod tests;
