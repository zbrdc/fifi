//! Raw FFI to inference_cuda_row_add_launch of native/resident/row_add.cu
//! (declared in native/resident/row_add.h). Child of native::resident.
//!
//! Safety invariants this module relies on:
//! - RawRange::new proved each range in bounds. This module re-proves that out, x and y
//!   hold n floats each at 4-byte alignment, and that out, which the kernel writes,
//!   shares no byte with x or y. The C entry repeats the checks on the addresses.
//! - Launch-only: a fault surfaces at the next synchronize; allocations stay valid until
//!   then because inference_cuda_free synchronizes the device first.
//! - Every call runs under the process dispatch guard on the owning thread
//!   (decisions 0023 and 0024). No raw pointer leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

unsafe extern "C" {
    #[link_name = "inference_cuda_row_add_launch"]
    fn row_add_launch_raw(
        args: *const RowAddArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Mirror of inference_cuda_row_add_args, field for field.
#[repr(C)]
struct RowAddArgsRepr {
    out: *mut c_void,
    x: *const c_void,
    y: *const c_void,
    n: c_int,
}

const _: () = assert!(
    size_of::<RowAddArgsRepr>() == 32
        && offset_of!(RowAddArgsRepr, out) == 0
        && offset_of!(RowAddArgsRepr, x) == 8
        && offset_of!(RowAddArgsRepr, y) == 16
        && offset_of!(RowAddArgsRepr, n) == 24
);

/// One row_add launch: n values of out, x and y.
pub(in crate::kernels::cuda) struct RowAddRaw<'a> {
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) n: usize,
}

fn check_row_add(call: &RowAddRaw<'_>) -> Result<(), DeviceFailure> {
    let bytes = call.n.checked_mul(4).ok_or_else(invalid_argument)?;
    let sized = call.out.bytes >= bytes && call.x.bytes >= bytes && call.y.bytes >= bytes;
    let aligned = call.out.offset.is_multiple_of(4)
        && call.x.offset.is_multiple_of(4)
        && call.y.offset.is_multiple_of(4);
    let out = Span::of(&call.out, bytes);
    let apart = out.apart(&Span::of(&call.x, bytes)) && out.apart(&Span::of(&call.y, bytes));
    let dims = (32..=4_194_304).contains(&call.n) && call.n.is_multiple_of(32);
    if !(sized && aligned && apart && dims) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the row_add of call on stream. Returns when the kernel is queued.
pub(in crate::kernels::cuda) fn row_add_launch(
    stream: &RawStream,
    call: &RowAddRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_row_add(call)?;
    let args = RowAddArgsRepr {
        out: call.out.address(),
        x: call.x.address().cast_const(),
        y: call.y.address().cast_const(),
        n: c_int::try_from(call.n).map_err(|_| invalid_argument())?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: check_row_add proved the ranges inside their live allocations, large enough,
    // aligned, and disjoint in the write direction; the adapter repeats the checks on the
    // addresses. The allocations stay valid until the queued launch finishes because
    // inference_cuda_free synchronizes the device first.
    let status = unsafe { row_add_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
