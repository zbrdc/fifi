//! Safe launch wrapper of `swiglu_rows` (CPF-7a part 1, CPF-V4-CPF7-PLAN section C.2, N2;
//! decision 0049, seam 1 of item 15): the M-row SwiGLU with f32 output of the chunked-prefill
//! MoE path, `act[r][c] = silu(gu[r][c]) * gu[r][up_off + c]`. The routed experts run it on
//! the 8M token-slot rows of the grouped gate/up GEMM and the shared expert on M rows; `act`
//! then feeds the MMQ q8_1 quantizer. The silu is the upstream helper compiled under the MMQ
//! glue group's fast math (`__expf`, approximate division, subnormals flushed to zero), so
//! the tolerance is the row SW0, not bit equality with an IEEE silu.
//!
//! [`CudaContext::swiglu_rows`] queues one kernel on a [`DeviceStream`] and returns. The call
//! allocates nothing, copies nothing and never synchronizes; a fault in the queued work
//! surfaces at the next synchronize of the stream and latches the backend there.
//!
//! Every argument is checked here, before any native call, and refused as `BadRequest`: the
//! rules mirror `valid_swiglu_rows` of `swiglu_rows.cu`, so the native entry cannot return
//! `INVALID_ARGUMENT` (which would latch the backend). Shape rules carry the tag
//! `swiglu_rows: <rule>`; size, alignment and overlap are refused by `require_fit` and
//! `require_apart` with their own texts.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "no caller yet: the chunk driver does not call this kernel; drop the expect when it does"
    )
)]

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::prefill::swiglu_rows::{
    SwigluRowsInts, SwigluRowsRaw, swiglu_rows_launch,
};

/// The most rows of one call: 8 slot rows per token at the 2,048-row chunk cap.
pub(crate) const MAX_ROWS: usize = 16_384;
/// The most values of one half.
pub(crate) const MAX_FF: usize = 32_768;
/// `n_ff`, `up_off`, `ld_in` and `ld_out` are multiples of this: the kernel moves `float4`.
pub(crate) const QUAD: usize = 4;

/// Geometry of one call: `rows` rows of `n_ff` outputs; the up half starts at column
/// `up_off` of a `gu` row of `ld_in` floats, and an `act` row holds `ld_out` floats.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SwigluRowsShape {
    pub(crate) rows: usize,
    pub(crate) n_ff: usize,
    pub(crate) up_off: usize,
    pub(crate) ld_in: usize,
    pub(crate) ld_out: usize,
}

/// One call. Both ranges are device ranges of this context.
///
/// - `gu`: f32 `[rows][ld_in]`, 16-byte aligned, read only: the gate at columns `0..n_ff`,
///   the up values at `up_off..up_off + n_ff`; other columns are not read.
/// - `act`: f32 `[rows][ld_out]`, 16-byte aligned: columns `0..n_ff` are written, no other
///   byte. Its extent shares no byte with the `gu` extent.
pub(crate) struct SwigluRowsCall<'b> {
    pub(crate) shape: SwigluRowsShape,
    pub(crate) gu: DeviceRange<'b>,
    pub(crate) act: DeviceRange<'b>,
}

/// Bytes of the two extents a call touches.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct SwigluRowsNeeds {
    pub(crate) gu: usize,
    pub(crate) act: usize,
}

/// The extents the wrapper checks a call against: `(rows - 1) * ld_in + up_off + n_ff` floats
/// of `gu` and `(rows - 1) * ld_out + n_ff` floats of `act`. `None` for zero rows or on
/// overflow. Pure: no context, no device.
pub(crate) fn swiglu_rows_needs(shape: &SwigluRowsShape) -> Option<SwigluRowsNeeds> {
    let before = shape.rows.checked_sub(1)?;
    let extent = |ld: usize, tail: usize| before.checked_mul(ld)?.checked_add(tail)?.checked_mul(4);
    Some(SwigluRowsNeeds {
        gu: extent(shape.ld_in, shape.up_off.checked_add(shape.n_ff)?)?,
        act: extent(shape.ld_out, shape.n_ff)?,
    })
}

/// A `BadRequest` that names the rule that fired.
fn rule(tag: &str, detail: &str) -> EngError {
    EngError::BadRequest(format!("swiglu_rows: {tag}: {detail}"))
}

/// `value` as `c_int` for the rule `tag`, or the "above i32::MAX" refusal.
fn to_int(tag: &str, value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value).map_err(|_| rule(tag, &format!("{tag} is {value}, above i32::MAX")))
}

/// The refusal for `tag` unless `value` is a multiple of [`QUAD`].
fn require_quad(tag: &str, value: usize) -> Result<(), EngError> {
    if value.is_multiple_of(QUAD) {
        return Ok(());
    }
    Err(rule(
        tag,
        &format!("{tag} is {value}, not a multiple of {QUAD}"),
    ))
}

/// The refusal for `tag` unless `value >= least`.
fn require_at_least(tag: &str, value: usize, least: usize, what: &str) -> Result<(), EngError> {
    if value >= least {
        return Ok(());
    }
    Err(rule(
        tag,
        &format!("{tag} is {value}, below {what} {least}"),
    ))
}

/// Checks every shape rule of `valid_shape` in `swiglu_rows.cu`, in the order of the tags
/// below, and converts the integers once, so that no conversion can fail inside the launch
/// closure.
pub(in crate::kernels::cuda) fn check_shape(
    shape: &SwigluRowsShape,
) -> Result<SwigluRowsInts, EngError> {
    if !(1..=MAX_ROWS).contains(&shape.rows) {
        let detail = format!("rows is {}, not 1 to {MAX_ROWS}", shape.rows);
        return Err(rule("rows", &detail));
    }
    if !(QUAD..=MAX_FF).contains(&shape.n_ff) {
        let detail = format!("n_ff is {}, not {QUAD} to {MAX_FF}", shape.n_ff);
        return Err(rule("n_ff", &detail));
    }
    require_quad("n_ff", shape.n_ff)?;
    require_quad("up_off", shape.up_off)?;
    require_at_least("up_off", shape.up_off, shape.n_ff, "n_ff")?;
    require_quad("ld_in", shape.ld_in)?;
    let up_end = shape
        .up_off
        .checked_add(shape.n_ff)
        .ok_or_else(|| rule("up_off", "up_off + n_ff overflows"))?;
    require_at_least("ld_in", shape.ld_in, up_end, "up_off + n_ff")?;
    require_quad("ld_out", shape.ld_out)?;
    require_at_least("ld_out", shape.ld_out, shape.n_ff, "n_ff")?;
    Ok(SwigluRowsInts {
        rows: to_int("rows", shape.rows)?,
        n_ff: to_int("n_ff", shape.n_ff)?,
        up_off: to_int("up_off", shape.up_off)?,
        ld_in: to_int("ld_in", shape.ld_in)?,
        ld_out: to_int("ld_out", shape.ld_out)?,
    })
}

impl CudaContext {
    /// Queues the SwiGLU of `call` on `stream`: `act[r][c] = silu(gu[r][c]) * gu[r][up_off +
    /// c]` for every row and every column `c < n_ff`. Returns when the kernel is queued; `act`
    /// is valid after the stream drained.
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a
    /// buffer or stream of another context, a shape the kernel refuses, a range that is too
    /// short or not 16-byte aligned, and an `act` extent that overlaps the `gu` extent.
    pub(crate) fn swiglu_rows(
        &self,
        stream: &DeviceStream,
        call: SwigluRowsCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let gu = self.raw_range(&call.gu)?;
        let act = self.raw_range(&call.act)?;
        let ints = check_shape(&call.shape)?;
        // Unreachable after check_shape on a 64-bit target (every factor is below 2^31); a
        // typed refusal rather than a panic.
        let need = swiglu_rows_needs(&call.shape)
            .ok_or_else(|| rule("size", "the extents overflow usize"))?;
        require_fit("gu", &call.gu, need.gu, 16)?;
        require_fit("act", &call.act, need.act, 16)?;
        require_apart(
            &Claim::of("act", &call.act, need.act),
            &Claim::of("gu", &call.gu, need.gu),
        )?;
        self.launch_on(stream, move |raw| {
            swiglu_rows_launch(raw, SwigluRowsRaw { gu, act, ints })
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
