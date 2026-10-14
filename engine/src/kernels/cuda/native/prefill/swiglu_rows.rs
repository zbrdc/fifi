//! Raw FFI to the M-row SwiGLU entry of `native/prefill/swiglu_rows.cu` (declared in
//! `native/prefill/swiglu_rows.h`), CPF-7a part 1. Child module of `native/prefill.rs`, so it
//! inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this file relies on:
//! - The launch entry is called only through `CudaContext::launch_on`, on the thread that
//!   owns the context and under the dispatch guard (decisions 0023 and 0024). It only
//!   enqueues and returns; the stream is live for the call.
//! - Both ranges are `RawRange`s, proved inside a live allocation by `raw_range`. Their sizes,
//!   their 16-byte alignment and the disjointness of the written `act` extent from the read
//!   `gu` extent (`super::Span`) are proved again here before the call; the C entry repeats
//!   the pointer, alignment, shape and overlap checks on the addresses.
//! - The C argument struct is built in one `unsafe` block that cites those checks.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes
//!   the device before `cudaFree`.
//! - The `extern` declaration, the `repr(C)` mirror and every raw pointer are private to this
//!   file. Nothing is re-exported.

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the prefill driver (decision 0049)")
)]

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Mirror of `inference_cuda_swiglu_rows_args` in `swiglu_rows.h`, field for field.
#[repr(C)]
struct SwigluRowsArgsRaw {
    gu: *const c_void,
    act: *mut c_void,
    rows: c_int,
    n_ff: c_int,
    up_off: c_int,
    ld_in: c_int,
    ld_out: c_int,
}

const _: () = assert!(
    size_of::<SwigluRowsArgsRaw>() == 40
        && offset_of!(SwigluRowsArgsRaw, gu) == 0
        && offset_of!(SwigluRowsArgsRaw, act) == 8
        && offset_of!(SwigluRowsArgsRaw, rows) == 16
        && offset_of!(SwigluRowsArgsRaw, n_ff) == 20
        && offset_of!(SwigluRowsArgsRaw, up_off) == 24
        && offset_of!(SwigluRowsArgsRaw, ld_in) == 28
        && offset_of!(SwigluRowsArgsRaw, ld_out) == 32
);

unsafe extern "C" {
    #[link_name = "inference_cuda_swiglu_rows_launch"]
    fn swiglu_rows_launch_raw(
        args: *const SwigluRowsArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Every integer argument of the launch, already bounded and converted by the wrapper before
/// `launch_on`, so no conversion can fail inside the launch closure.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct SwigluRowsInts {
    pub(in crate::kernels::cuda) rows: c_int,
    pub(in crate::kernels::cuda) n_ff: c_int,
    pub(in crate::kernels::cuda) up_off: c_int,
    pub(in crate::kernels::cuda) ld_in: c_int,
    pub(in crate::kernels::cuda) ld_out: c_int,
}

/// What one launch reads (`gu`) and writes (`act`), and its integers.
pub(in crate::kernels::cuda) struct SwigluRowsRaw<'a> {
    pub(in crate::kernels::cuda) gu: RawRange<'a>,
    pub(in crate::kernels::cuda) act: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: SwigluRowsInts,
}

/// The bytes of the two extents the kernel touches: `(rows - 1) * ld_in + up_off + n_ff`
/// floats of `gu` and `(rows - 1) * ld_out + n_ff` floats of `act`. A negative integer, zero
/// rows or an overflow is an invalid-argument failure.
fn extents(ints: &SwigluRowsInts) -> Result<(usize, usize), DeviceFailure> {
    let size = |n: c_int| usize::try_from(n).map_err(|_| invalid_argument());
    let before = size(ints.rows)?
        .checked_sub(1)
        .ok_or_else(invalid_argument)?;
    let extent = |ld: usize, tail: usize| {
        before
            .checked_mul(ld)
            .and_then(|floats| floats.checked_add(tail))
            .and_then(|floats| floats.checked_mul(4))
            .ok_or_else(invalid_argument)
    };
    let n_ff = size(ints.n_ff)?;
    let up_end = size(ints.up_off)?
        .checked_add(n_ff)
        .ok_or_else(invalid_argument)?;
    Ok((
        extent(size(ints.ld_in)?, up_end)?,
        extent(size(ints.ld_out)?, n_ff)?,
    ))
}

/// Proves a launch before the native call: both ranges hold their extents, start 16-byte
/// aligned, and the written `act` extent shares no byte with the read `gu` extent.
fn check_swiglu_rows(io: &SwigluRowsRaw<'_>) -> Result<(), DeviceFailure> {
    let (gu, act) = extents(&io.ints)?;
    let sized = io.gu.bytes() >= gu && io.act.bytes() >= act;
    let aligned = io.gu.offset().is_multiple_of(16) && io.act.offset().is_multiple_of(16);
    let apart = Span::of(&io.act, act).apart(&Span::of(&io.gu, gu));
    if !(sized && aligned && apart) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the SwiGLU kernel on `stream`: `act[r][c] = silu(gu[r][c]) * gu[r][up_off + c]`
/// for every row and column `c < n_ff`. Returns when the kernel is queued.
pub(in crate::kernels::cuda) fn swiglu_rows_launch(
    stream: &RawStream,
    io: SwigluRowsRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_swiglu_rows(&io)?;
    let args = SwigluRowsArgsRaw {
        gu: io.gu.address().cast_const(),
        act: io.act.address(),
        rows: io.ints.rows,
        n_ff: io.ints.n_ff,
        up_off: io.ints.up_off,
        ld_in: io.ints.ld_in,
        ld_out: io.ints.ld_out,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_swiglu_rows` proved both ranges inside their live allocations, large
    // enough for the extents the kernel touches, 16-byte aligned, and the written `act` extent
    // disjoint from the read `gu` extent, so each pointer in `args` addresses bytes the kernel
    // may touch; the C entry repeats these checks on the addresses. The allocations stay valid
    // until the queued work finishes because `inference_cuda_free` synchronizes the device
    // first. `args` lives across the call and the entry reads it before returning. `stream` is
    // live for the call.
    let status = unsafe { swiglu_rows_launch_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}
