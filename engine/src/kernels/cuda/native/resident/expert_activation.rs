//! Raw FFI to the expert-activation entry of native/resident/expert_activation.cu
//! (declared in native/resident/expert_activation.h). Child of native::resident, so it
//! inherits the allow(unsafe_code) of native.rs.
//!
//! Safety invariants this module relies on (as row_scale.rs):
//! - Ranges are proved in bounds by RawRange::new before they get here; this module
//!   re-proves sizes, 4-byte alignment and write disjointness. The C entry
//!   repeats the checks on the addresses.
//! - Launch-only: work queues on the given stream; a fault surfaces at the next
//!   synchronize. Allocations stay valid because inference_cuda_free synchronizes
//!   the whole device before cudaFree.
//! - Every call runs under the process dispatch guard and on the context-owning thread.
//! - No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_longlong, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

unsafe extern "C" {
    #[link_name = "inference_cuda_relu_squared_launch"]
    fn relu_squared_launch_raw(
        args: *const ElementArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Mirror of inference_cuda_element_args (expert_activation.h), field for field.
#[repr(C)]
struct ElementArgsRepr {
    out: *mut c_void,
    x: *const c_void,
    values: c_longlong,
}

const _: () = assert!(
    size_of::<ElementArgsRepr>() == 24
        && offset_of!(ElementArgsRepr, out) == 0
        && offset_of!(ElementArgsRepr, x) == 8
        && offset_of!(ElementArgsRepr, values) == 16
);

/// One elementwise relu-squared: out[i] = max(x[i], 0)^2 over `values` f32.
pub(in crate::kernels::cuda) struct ReluSquaredRaw<'a> {
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) values: usize,
}

/// Queues the elementwise relu-squared on stream. Returns when queued.
pub(in crate::kernels::cuda) fn relu_squared_launch(
    stream: &RawStream,
    call: &ReluSquaredRaw<'_>,
) -> Result<(), DeviceFailure> {
    let bytes = call
        .values
        .checked_mul(4)
        .filter(|bytes| *bytes <= 1usize << 34)
        .ok_or_else(invalid_argument)?;
    if call.values == 0 {
        return Err(invalid_argument());
    }
    let sized = call.out.bytes >= bytes && call.x.bytes >= bytes;
    let aligned = call.out.offset.is_multiple_of(4) && call.x.offset.is_multiple_of(4);
    let apart = Span::of(&call.out, bytes).apart(&Span::of(&call.x, bytes));
    if !(sized && aligned && apart) {
        return Err(invalid_argument());
    }
    let args = ElementArgsRepr {
        out: call.out.address(),
        x: call.x.address().cast_const(),
        values: c_longlong::try_from(call.values).map_err(|_| invalid_argument())?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges are proved live, sized, aligned and disjoint above;
    // the adapter repeats the checks. Launch-only on a live stream.
    let status = unsafe { relu_squared_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
