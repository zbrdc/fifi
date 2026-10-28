//! Raw FFI to inference_cuda_softcap_launch of native/resident/softcap.cu
//! (declared in native/resident/softcap.h). Child of native::resident.
//!
//! Safety invariants this module relies on:
//! - RawRange::new proved each range in bounds. This module re-proves that out and x
//!   hold n floats each at 4-byte alignment, and that out, which the kernel writes,
//!   shares no byte with x. The C entry repeats the checks on the addresses.
//! - Launch-only: a fault surfaces at the next synchronize; allocations stay valid until
//!   then because inference_cuda_free synchronizes the device first.
//! - Every call runs under the process dispatch guard on the owning thread
//!   (decisions 0023 and 0024). No raw pointer leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

unsafe extern "C" {
    #[link_name = "inference_cuda_softcap_launch"]
    fn softcap_launch_raw(
        args: *const SoftcapArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Mirror of inference_cuda_softcap_args, field for field.
#[repr(C)]
struct SoftcapArgsRepr {
    out: *mut c_void,
    x: *const c_void,
    cap: f32,
    n: c_int,
}

const _: () = assert!(
    size_of::<SoftcapArgsRepr>() == 24
        && offset_of!(SoftcapArgsRepr, out) == 0
        && offset_of!(SoftcapArgsRepr, x) == 8
        && offset_of!(SoftcapArgsRepr, cap) == 16
        && offset_of!(SoftcapArgsRepr, n) == 20
);

/// One softcap launch: n values of out and x and the cap.
pub(in crate::kernels::cuda) struct SoftcapRaw<'a> {
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) cap: f32,
    pub(in crate::kernels::cuda) n: usize,
}

fn check_softcap(call: &SoftcapRaw<'_>) -> Result<(), DeviceFailure> {
    let bytes = call.n.checked_mul(4).ok_or_else(invalid_argument)?;
    let sized = call.out.bytes >= bytes && call.x.bytes >= bytes;
    let aligned = call.out.offset.is_multiple_of(4) && call.x.offset.is_multiple_of(4);
    let apart = Span::of(&call.out, bytes).apart(&Span::of(&call.x, bytes));
    let dims = (32..=4_194_304).contains(&call.n)
        && call.n.is_multiple_of(32)
        && call.cap.is_finite()
        && call.cap > 0.0;
    if !(sized && aligned && apart && dims) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the softcap of call on stream. Returns when the kernel is queued.
pub(in crate::kernels::cuda) fn softcap_launch(
    stream: &RawStream,
    call: &SoftcapRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_softcap(call)?;
    let args = SoftcapArgsRepr {
        out: call.out.address(),
        x: call.x.address().cast_const(),
        cap: call.cap,
        n: c_int::try_from(call.n).map_err(|_| invalid_argument())?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: check_softcap proved the ranges inside their live allocations, large
    // enough, aligned, and disjoint in the write direction; the adapter repeats the
    // checks on the addresses. The allocations stay valid until the queued launch
    // finishes because inference_cuda_free synchronizes the device first.
    let status = unsafe { softcap_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
