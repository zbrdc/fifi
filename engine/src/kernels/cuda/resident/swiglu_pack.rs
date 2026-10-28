//! Safe launch wrapper of the SwiGLU + fp16 pack of one expert (RD-4b, decision 0047, seam 1):
//! from the f32 gate and up rows `gu` the fp16 matmul input `xf = fp16(silu(gate) * up)`, in the
//! layout [`CudaContext::pack`] writes (fragment order for `m <= 16`, row-major above), with an
//! optional padded width `k_pad >= n_ff` whose tail is exactly zero (MEL-A1, MEL-C1b).
//!
//! [`CudaContext::swiglu_pack`] and [`CudaContext::swiglu_pack_padded`] queue their work on a
//! [`DeviceStream`] and return. They allocate nothing, copy nothing and never synchronize; a fault
//! in the queued work surfaces at the next synchronize of the stream and latches the backend
//! there. The caller owns the order: `gu` must hold its data before the call, and `xf` is read by
//! the matmul queued behind it.
//!
//! What a call queues, in order: for `n_ff < k_pad` two 2-D memsets that zero columns
//! `n_ff..k_pad` of both halves of every row of `gu`; then the kernel (with `n_ff = k_pad`, so
//! `silu(0) * 0 = +0` fills the tail of `xf`, and with the zero tail entry of a fragment-order
//! `xf` written by the kernel itself, MEMSET-TRIM).
//!
//! Layout of `gu`: f32 `[m][2 * k_pad]`, 4-byte aligned. Row `t` holds the gate in columns
//! `0..k_pad` (live `0..n_ff`) and the up values in columns `k_pad..2 * k_pad` (live
//! `k_pad..k_pad + n_ff`). For `m = 1` the gate and up matmuls write the two halves as two
//! `MatmulIo.y` ranges of one buffer; `m > 1` is a capability of this entry that stage 1 does not
//! call (the matmul wrapper writes `ldy = rows`). **The call overwrites the tails of `gu` with
//! zeros**; the live columns are only read.
//!
//! `xf` is `native::xf_bytes(m, k_pad)` bytes (1,048,592 for `m <= 16`), 16-byte aligned; for
//! `m <= 16` the kernel writes the data lanes of the `m` rows and the zero tail entry the padding
//! lanes read, and the bytes no consumer reads are left untouched (MEMSET-TRIM); above 16 rows it
//! is written in full. `gu` and `xf` must not overlap.
//!
//! Numerics (lane M REPLY-44 class E2, REPLY-73): the fp16 result equals the round to nearest even
//! of the float64 `silu(gate) * up`, or the adjacent value when the reference lies within 2^-20
//! (relative) of the rounding midpoint. The kernel uses the fast `__expf`; the statement is a
//! measured domain, not an analytic bound: any finite `up` for `|gate| <= 12`, and `up` of sd up
//! to 100 for `12 < |gate| <= 219`. The 15 E2 misses at `up` sd 10^4 in `|gate|` [-30, -12]
//! (expf probe, rev 3 evidence) are recorded as the measured edge of the domain, not tuned away.
//!
//! Every argument is checked here, before any native call: shape rules carry the tag
//! `swiglu_pack: <rule>`; sizes and alignment come from `require_fit` and overlap from
//! `require_apart`. On a latched context a bad argument still answers `BadRequest` and a good one
//! answers the latch error (`launch_on` checks the terminal flag itself).
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for RD-9a (decision 0047)")
)]

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{Claim, require_apart, require_fit};
use crate::kernels::cuda::native::resident::swiglu_pack::{
    SwigluInts, SwigluRaw, swiglu_pack_launch,
};
use crate::kernels::cuda::native::xf_bytes;
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream};

/// The most rows (tokens) one call takes.
pub(crate) const MAX_TOKENS: c_int = 32_768;
/// The most columns of one half row (`k_pad`).
pub(crate) const MAX_WIDTH: c_int = 32_768;
/// `k_pad` is a multiple of this (the vendored kernel packs blocks of 256 columns).
const WIDTH_QUANTUM: c_int = 256;

/// One SwiGLU pack call. Both ranges are device ranges of this context.
///
/// - `gu`: f32 `[m][2 * k_pad]`, 4-byte aligned; see the module doc. The tails are overwritten.
/// - `xf`: the packed fp16 output, 16-byte aligned, `xf_bytes(m, k_pad)` bytes at least.
/// - `m`: rows (tokens), `1..=32768`.
/// - `n_ff`: live columns of each half, `1..=k_pad`.
pub(crate) struct SwigluPackCall<'b> {
    pub(crate) gu: DeviceRange<'b>,
    pub(crate) xf: DeviceRange<'b>,
    pub(crate) m: usize,
    pub(crate) n_ff: usize,
}

/// A `BadRequest` that names the rule that fired.
fn rule(tag: &str, detail: &str) -> EngError {
    EngError::BadRequest(format!("swiglu_pack: {tag}: {detail}"))
}

/// `value` as `c_int`, or the "exceeds i32" refusal.
fn to_int(tag: &str, value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value).map_err(|_| rule(tag, &format!("{tag} is {value}, above i32::MAX")))
}

/// `value` as `c_int` when it lies in `low..=high`, else the refusal for `tag`. The range is
/// tested on the converted integer, so the conversion is what refuses a value above `i32::MAX`.
fn ranged(tag: &str, value: usize, low: c_int, high: c_int) -> Result<c_int, EngError> {
    let int = to_int(tag, value)?;
    if !(low..=high).contains(&int) {
        return Err(rule(tag, &format!("{tag} is {value}, not {low} to {high}")));
    }
    Ok(int)
}

/// Checks every shape rule and converts the integers once, so that no conversion can fail inside
/// the launch closure. Each refusal carries the tag `swiglu_pack: <rule>`.
pub(in crate::kernels::cuda) fn check_shape(
    m: usize,
    n_ff: usize,
    k_pad: usize,
) -> Result<SwigluInts, EngError> {
    let m = ranged("m", m, 1, MAX_TOKENS)?;
    let k_pad = ranged("k_pad", k_pad, WIDTH_QUANTUM, MAX_WIDTH)?;
    if k_pad % WIDTH_QUANTUM != 0 {
        return Err(rule(
            "k_pad",
            &format!("k_pad is {k_pad}, not a multiple of {WIDTH_QUANTUM}"),
        ));
    }
    let n_ff = ranged("n_ff", n_ff, 1, k_pad)?;
    Ok(SwigluInts { m, n_ff, k_pad })
}

/// Bytes `gu` must hold: `m` rows of `2 * k_pad` f32. `None` on overflow.
pub(crate) fn gu_bytes(m: usize, k_pad: usize) -> Option<usize> {
    m.checked_mul(2)?.checked_mul(k_pad)?.checked_mul(4)
}

impl CudaContext {
    /// Queues the SwiGLU pack of `call` on `stream` with `k_pad = call.n_ff` (no padding):
    /// `xf = fp16(silu(gate) * up)` for every row of `gu`. Returns when the work is queued.
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a stream
    /// or buffer of another context, a shape the kernel refuses (`n_ff` must be a multiple of 256
    /// up to 32,768), a range that is too short or misaligned, and `gu` and `xf` ranges that
    /// overlap.
    pub(crate) fn swiglu_pack(
        &self,
        stream: &DeviceStream,
        call: SwigluPackCall<'_>,
    ) -> Result<(), EngError> {
        let k_pad = call.n_ff;
        self.launch(stream, call, k_pad)
    }

    /// Queues the SwiGLU pack of `call` on `stream` with the padded width `k_pad >= call.n_ff`,
    /// a multiple of 256 up to 32,768. Columns `n_ff..k_pad` of both halves of every `gu` row are
    /// zeroed first (the call writes them), so the columns `n_ff..k_pad` of `xf` are exactly
    /// `0x0000`. Refusals as for [`Self::swiglu_pack`].
    pub(crate) fn swiglu_pack_padded(
        &self,
        stream: &DeviceStream,
        call: SwigluPackCall<'_>,
        k_pad: usize,
    ) -> Result<(), EngError> {
        self.launch(stream, call, k_pad)
    }

    /// Every check of a call, then the one native launch. The terminal latch is checked by
    /// `launch_on`, after the arguments, so a bad argument is still reported as such.
    fn launch(
        &self,
        stream: &DeviceStream,
        call: SwigluPackCall<'_>,
        k_pad: usize,
    ) -> Result<(), EngError> {
        let ints = check_shape(call.m, call.n_ff, k_pad)?;
        let gu_need = gu_bytes(call.m, k_pad).ok_or_else(|| rule("gu", "size overflows"))?;
        let xf_need = xf_bytes(call.m, k_pad).ok_or_else(|| rule("xf", "size overflows"))?;
        require_fit("gu", &call.gu, gu_need, 4)?;
        require_fit("xf", &call.xf, xf_need, 16)?;
        require_apart(
            &Claim::of("gu", &call.gu, gu_need),
            &Claim::of("xf", &call.xf, xf_need),
        )?;
        let gu = self.raw_range(&call.gu)?;
        let xf = self.raw_range(&call.xf)?;
        self.launch_on(stream, move |raw| {
            swiglu_pack_launch(raw, SwigluRaw { gu, xf, ints })
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
