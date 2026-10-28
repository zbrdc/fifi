//! FFI for the test-only, fixed-size top-five trace.
use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use std::ffi::{c_int, c_void};

#[repr(C)]
struct Args {
    out: *mut c_void,
    x: *const c_void,
    n: c_int,
}
const _: () = assert!(size_of::<Args>() == 24 && std::mem::offset_of!(Args, n) == 16);
unsafe extern "C" {
    fn inference_cuda_top_logits_launch(
        a: *const Args,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) fn launch(
    stream: &RawStream,
    out: RawRange<'_>,
    x: RawRange<'_>,
    n: usize,
) -> Result<(), DeviceFailure> {
    let bytes = n.checked_mul(4).ok_or_else(invalid_argument)?;
    if !(5..=4_194_304).contains(&n)
        || out.bytes < 44
        || x.bytes < bytes
        || !out.offset.is_multiple_of(4)
        || !x.offset.is_multiple_of(4)
        || !Span::of(&out, 44).apart(&Span::of(&x, bytes))
    {
        return Err(invalid_argument());
    }
    let args = Args {
        out: out.address(),
        x: x.address().cast_const(),
        n: n as c_int,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: The checks above prove live, aligned, disjoint input and output extents.
    // The adapter repeats them. Buffer release waits for all queued device work.
    let status = unsafe { inference_cuda_top_logits_launch(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
