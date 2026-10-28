//! Raw FFI to inference_cuda_gelu_ffn_launch of native/resident/gelu_ffn.cu
//! (declared in native/resident/gelu_ffn.h). Child of native::resident.
//!
//! Safety invariants this module relies on:
//! - RawRange::new proved each range in bounds. This module re-proves that out, gate and
//!   up hold m*n floats each at 4-byte alignment, and that out, which the kernel writes,
//!   shares no byte with gate or up. The C entry repeats the checks on the addresses.
//! - Launch-only: a fault surfaces at the next synchronize; allocations stay valid until
//!   then because inference_cuda_free synchronizes the device first.
//! - Every call runs under the process dispatch guard on the owning thread
//!   (decisions 0023 and 0024). No raw pointer leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

unsafe extern "C" {
    #[link_name = "inference_cuda_gelu_ffn_launch"]
    fn gelu_ffn_launch_raw(
        args: *const GeluFfnArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Mirror of inference_cuda_gelu_ffn_args, field for field.
#[repr(C)]
struct GeluFfnArgsRepr {
    out: *mut c_void,
    gate: *const c_void,
    up: *const c_void,
    m: c_int,
    n: c_int,
}

const _: () = assert!(
    size_of::<GeluFfnArgsRepr>() == 32
        && offset_of!(GeluFfnArgsRepr, out) == 0
        && offset_of!(GeluFfnArgsRepr, gate) == 8
        && offset_of!(GeluFfnArgsRepr, up) == 16
        && offset_of!(GeluFfnArgsRepr, m) == 24
        && offset_of!(GeluFfnArgsRepr, n) == 28
);

/// One gelu_ffn launch: m rows of n values for each of out, gate and up.
pub(in crate::kernels::cuda) struct GeluFfnRaw<'a> {
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) gate: RawRange<'a>,
    pub(in crate::kernels::cuda) up: RawRange<'a>,
    pub(in crate::kernels::cuda) m: usize,
    pub(in crate::kernels::cuda) n: usize,
}

fn check_gelu_ffn(call: &GeluFfnRaw<'_>) -> Result<(), DeviceFailure> {
    let bytes = call
        .m
        .checked_mul(call.n)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    let sized = call.out.bytes >= bytes && call.gate.bytes >= bytes && call.up.bytes >= bytes;
    let aligned = call.out.offset.is_multiple_of(4)
        && call.gate.offset.is_multiple_of(4)
        && call.up.offset.is_multiple_of(4);
    let apart = Span::of(&call.out, bytes).apart(&Span::of(&call.gate, bytes))
        && Span::of(&call.out, bytes).apart(&Span::of(&call.up, bytes));
    let dims = (1..=65_535).contains(&call.m)
        && (32..=65_536).contains(&call.n)
        && call.n.is_multiple_of(32);
    if !(sized && aligned && apart && dims) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the gelu_ffn of call on stream. Returns when the kernel is queued.
pub(in crate::kernels::cuda) fn gelu_ffn_launch(
    stream: &RawStream,
    call: &GeluFfnRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_gelu_ffn(call)?;
    let args = GeluFfnArgsRepr {
        out: call.out.address(),
        gate: call.gate.address().cast_const(),
        up: call.up.address().cast_const(),
        m: c_int::try_from(call.m).map_err(|_| invalid_argument())?,
        n: c_int::try_from(call.n).map_err(|_| invalid_argument())?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: check_gelu_ffn proved the ranges inside their live allocations, large
    // enough, aligned, and disjoint in the write direction; the adapter repeats the
    // checks on the addresses. The allocations stay valid until the queued launch
    // finishes because inference_cuda_free synchronizes the device first.
    let status = unsafe { gelu_ffn_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
