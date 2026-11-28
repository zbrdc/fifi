use std::ffi::{c_int, c_void};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use crate::kernels::cuda::resident::relative_prep::{
    DepthShape, KvPrepShape, OutputConvShape, PrepMath, PrepShape,
};

unsafe extern "C" {
    fn inference_cuda_effective_taps(
        ranges: *const *mut c_void,
        width: c_int,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_weighted_norm_rows(
        ranges: *const *mut c_void,
        rows: c_int,
        width: c_int,
        eps: f32,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_short_conv_rows(
        ranges: *const *mut c_void,
        rows: c_int,
        width: c_int,
        history: c_int,
        scale: f32,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_relative_prep_rows(
        ranges: *const *mut c_void,
        shape: *const c_int,
        limit: f32,
        eps: f32,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_relative_prep(
        ranges: *const *mut c_void,
        queries: c_int,
        kv: c_int,
        dims: c_int,
        history: c_int,
        qk_limit: f32,
        rel_limit: f32,
        eps: f32,
        pos: *const c_int,
        log_floor: c_int,
        log_alpha: f32,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) fn weighted_norm(
    stream: &RawStream,
    ranges: [RawRange<'_>; 4],
    [rows, width]: [usize; 2],
    eps: f32,
) -> Result<(), DeviceFailure> {
    if !eps.is_finite() || eps <= 0.0 {
        return Err(invalid_argument());
    }
    let [plane, ..] = (DepthShape {
        rows,
        width,
        depth: 1,
    })
    .words()
    .ok_or_else(invalid_argument)?;
    validate_f32_ranges(&ranges, [plane, width, plane, rows])?;
    let pointers = ranges.map(|r| r.address());
    let mut stage = STAGE_NONE;
    // SAFETY: validated bounding geometry and disjoint, aligned regions.
    let status = unsafe {
        inference_cuda_weighted_norm_rows(
            pointers.as_ptr(),
            rows as c_int,
            width as c_int,
            eps,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

fn validate_f32_ranges<const N: usize>(
    ranges: &[RawRange<'_>; N],
    words: [usize; N],
) -> Result<(), DeviceFailure> {
    for (i, (range, words)) in ranges.iter().zip(words).enumerate() {
        let bytes = words * size_of::<f32>();
        if range.bytes() != bytes
            || !range.offset.is_multiple_of(4)
            || ranges[..i]
                .iter()
                .any(|other| !Span::of(range, bytes).apart(&Span::of(other, other.bytes())))
        {
            return Err(invalid_argument());
        }
    }
    Ok(())
}

pub(in crate::kernels::cuda) fn short_conv_rows(
    stream: &RawStream,
    ranges: [RawRange<'_>; 5],
    shape: OutputConvShape,
) -> Result<(), DeviceFailure> {
    let words = shape.words().ok_or_else(invalid_argument)?;
    validate_f32_ranges(&ranges, words)?;
    let pointers = ranges.map(|range| range.address());
    let scale = ((PrepShape::TAPS + 1) as f32).sqrt().recip();
    let mut stage = STAGE_NONE;
    // SAFETY: the spans are aligned, sized for this bounded geometry, and distinct.
    let status = unsafe {
        inference_cuda_short_conv_rows(
            pointers.as_ptr(),
            shape.rows as c_int,
            shape.width as c_int,
            shape.history as c_int,
            scale,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

pub(in crate::kernels::cuda) fn relative_prep_rows(
    stream: &RawStream,
    ranges: [RawRange<'_>; 5],
    shape: KvPrepShape,
) -> Result<(), DeviceFailure> {
    let words = shape.words().ok_or_else(invalid_argument)?;
    validate_f32_ranges(&ranges, words)?;
    let pointers = ranges.map(|range| range.address());
    // words() bounds every count below i32::MAX; only the missing position word maps to -1.
    let geometry = shape.geometry().map(|n| c_int::try_from(n).unwrap_or(-1));
    let mut stage = STAGE_NONE;
    // SAFETY: words() bounds every index the kernel forms for this geometry, and the five
    // spans are exact, aligned and distinct; both arrays outlive the call.
    let status = unsafe {
        inference_cuda_relative_prep_rows(
            pointers.as_ptr(),
            geometry.as_ptr(),
            shape.limit,
            shape.eps,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

pub(in crate::kernels::cuda) fn effective_taps(
    stream: &RawStream,
    ranges: [RawRange<'_>; 3],
    width: usize,
) -> Result<(), DeviceFailure> {
    if !(128..=128 * 128).contains(&width) || !width.is_multiple_of(128) {
        return Err(invalid_argument());
    }
    validate_f32_ranges(&ranges, [4 * width, 4 * (width / 128), 4 * width])?;
    let ptrs = ranges.map(|r| r.address());
    let mut stage = STAGE_NONE;
    // SAFETY: lengths, alignment and disjointness were checked above; ptrs lives across the call.
    let status = unsafe {
        inference_cuda_effective_taps(ptrs.as_ptr(), width as c_int, stream.0.as_ptr(), &mut stage)
    };
    checked(status, stage)
}

pub(in crate::kernels::cuda) fn launch(
    stream: &RawStream,
    ranges: [RawRange<'_>; 15],
    shape: PrepShape,
    history: usize,
    (math, pos): (PrepMath, Option<RawRange<'_>>),
) -> Result<(), DeviceFailure> {
    if math.log.is_some_and(|log| !log.valid()) || math.log.is_some() != pos.is_some() {
        return Err(invalid_argument());
    }
    let (floor, alpha) = math
        .log
        .map_or((0, 0.0), |log| (log.floor as c_int, log.alpha));
    if !(1..=128).contains(&shape.queries)
        || !(1..=128).contains(&shape.kv)
        || !(1..=PrepShape::WIDTH).contains(&shape.dims)
        || history >= PrepShape::TAPS
        || ![math.qk_limit, math.rel_limit, math.eps]
            .into_iter()
            .all(|v| v.is_finite() && v > 0.0)
    {
        return Err(invalid_argument());
    }
    validate_f32_ranges(&ranges, shape.direct_words())?;
    let pos = match pos {
        None => std::ptr::null(),
        Some(pos) => {
            if pos.bytes() < size_of::<c_int>()
                || !pos.offset.is_multiple_of(size_of::<c_int>())
                || ranges.iter().any(|other| {
                    !Span::of(&pos, size_of::<c_int>()).apart(&Span::of(other, other.bytes()))
                })
            {
                return Err(invalid_argument());
            }
            pos.address().cast_const().cast::<c_int>()
        }
    };
    let pointers = ranges.map(|r| r.address());
    let mut stage = STAGE_NONE;
    // SAFETY: the host pointer array lives for the call, and the CUDA spans are
    // aligned, disjoint and large enough for every access by the queued kernel.
    let status = unsafe {
        inference_cuda_relative_prep(
            pointers.as_ptr(),
            shape.queries as c_int,
            shape.kv as c_int,
            shape.dims as c_int,
            history as c_int,
            math.qk_limit,
            math.rel_limit,
            math.eps,
            pos,
            floor,
            alpha,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}
