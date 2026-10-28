//! NVFP4-only extension of the unchanged dense C argument layout.

use super::{
    DeviceFailure, MmqDenseArgsRaw, MmqDenseRaw, RawRange, RawStream, STAGE_NONE, Span, c_dense,
    check_dense_ranges, checked, invalid_argument, size,
};
use crate::kernels::cuda::prefill::mmq_nvfp4::Nvfp4DenseShape;
use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

#[repr(C)]
struct Nvfp4Args {
    matrix: MmqDenseArgsRaw,
    y_scale: *const f32,
}

const _: () = assert!(size_of::<Nvfp4Args>() == 72 && offset_of!(Nvfp4Args, y_scale) == 64);

unsafe extern "C" {
    fn inference_cuda_mmq_nvfp4_dense(
        args: *const Nvfp4Args,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
}

fn check(io: &MmqDenseRaw<'_>, factors: &RawRange<'_>) -> Result<(), DeviceFailure> {
    let ints = io.ints;
    if ints.ty != 40 || ints.ld != ints.n || ints.col_off != 0 {
        return Err(invalid_argument());
    }
    let shape = Nvfp4DenseShape::new(size(ints.k)?, size(ints.n)?, size(ints.m)?)
        .map_err(|_| invalid_argument())?;
    let need = shape.needs;
    check_dense_ranges(io, need.w, need.y, need.dst)?;
    let scales = Span::of(factors, shape.factor_bytes);
    if factors.bytes() < shape.factor_bytes
        || !factors.offset().is_multiple_of(4)
        || !scales.apart(&Span::of(&io.dst, need.dst))
        || !scales.apart(&Span::of(&io.fixup, io.fixup.bytes()))
    {
        return Err(invalid_argument());
    }
    Ok(())
}

pub(in crate::kernels::cuda) fn mmq_nvfp4_dense_launch(
    stream: &RawStream,
    io: MmqDenseRaw<'_>,
    factors: RawRange<'_>,
) -> Result<(), DeviceFailure> {
    check(&io, &factors)?;
    let args = Nvfp4Args {
        matrix: c_dense(&io),
        y_scale: factors.address().cast_const().cast(),
    };
    let mut stage = STAGE_NONE;
    // SAFETY: check proves live, sized/aligned ranges and disjoint writes; the
    // context owns this stream, and allocation frees wait for queued work.
    let status = unsafe { inference_cuda_mmq_nvfp4_dense(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod tests;
