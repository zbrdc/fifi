//! FFI for the launch-only F32 relative-attention step.
//!
//! RawRange proves allocation bounds. This module checks the precise extents,
//! alignment and write disjointness again before using the device addresses.
//! The context dispatch guard supplies the live stream, and free waits for queued
//! work before freeing a device allocation.

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use crate::kernels::cuda::resident::relative_attention::{
    LogScale, RANGES, RelativeAttentionShape, RelativeAttentionTile, WRITTEN,
};

#[repr(C)]
struct Args {
    ranges: [*mut c_void; RANGES],
    query_heads: c_int,
    kv_heads: c_int,
    relative_dims: c_int,
    relative_extent: c_int,
    capacity: c_int,
}

const _: () = assert!(
    size_of::<Args>() == 120
        && offset_of!(Args, query_heads) == 96
        && offset_of!(Args, capacity) == 112
);

unsafe extern "C" {
    #[link_name = "inference_cuda_relative_attention_launch"]
    fn raw_launch(args: *const Args, stream: *mut c_void, stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_relative_attention_tile_launch"]
    fn raw_tile_launch(
        args: *const Args,
        first: c_int,
        count: c_int,
        input_stride: c_int,
        query_stride: c_int,
        side_stride: c_int,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;

    fn inference_cuda_relative_attention_clip_queries(
        side: *mut c_void,
        scales: *mut c_void,
        status: *mut c_void,
        rows: c_int,
        heads: c_int,
        dims: c_int,
        gap: c_int,
        q_limit: f32,
        rel_limit: f32,
        first: c_int,
        log_floor: c_int,
        log_alpha: f32,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) fn clip_queries(
    stream: &RawStream,
    ranges: [RawRange<'_>; 3],
    shape: RelativeAttentionShape,
    [rows, gap]: [usize; 2],
    limits: [f32; 2],
    log: Option<(LogScale, usize)>,
) -> Result<(), DeviceFailure> {
    let [_, _, _, _, q, _, _, rel, _, sink, gate, _] =
        shape.region_bytes().ok_or_else(invalid_argument)?;
    let (first, floor, alpha) = match log {
        None => (0, 0, 0.0),
        Some((log, first)) if log.valid() && c_int::try_from(first + rows).is_ok() => {
            (first as c_int, log.floor as c_int, log.alpha)
        }
        Some(_) => return Err(invalid_argument()),
    };
    if !(1..=512).contains(&rows)
        || !limits.iter().all(|x| x.is_finite() && *x > 0.0)
        || !gap.is_multiple_of(4)
        || c_int::try_from(gap / 4).is_err()
    {
        return Err(invalid_argument());
    }
    let bytes = [(q + gap + rel + sink + gate) * rows, q + rel, rows * 4];
    let spans = ranges.each_ref().map(|r| Span::of(r, r.bytes()));
    for (i, range) in ranges.iter().enumerate() {
        if range.bytes() != bytes[i]
            || !range.offset().is_multiple_of(4)
            || spans[i + 1..].iter().any(|other| !spans[i].apart(other))
        {
            return Err(invalid_argument());
        }
    }
    let [side, scales, status] = ranges.map(|r| r.address());
    let mut stage = STAGE_NONE;
    // SAFETY: the shape and three disjoint, aligned spans were checked above.
    let code = unsafe {
        inference_cuda_relative_attention_clip_queries(
            side,
            scales,
            status,
            rows as c_int,
            shape.query_heads as c_int,
            shape.relative_dims as c_int,
            (gap / 4) as c_int,
            limits[0],
            limits[1],
            first,
            floor,
            alpha,
            stream.as_ptr(),
            &mut stage,
        )
    };
    checked(code, stage)
}

pub(in crate::kernels::cuda) struct RelativeAttentionRaw<'a> {
    pub(in crate::kernels::cuda) ranges: [RawRange<'a>; RANGES],
    pub(in crate::kernels::cuda) shape: RelativeAttentionShape,
    pub(in crate::kernels::cuda) tile: Option<RelativeAttentionTile>,
}

pub(in crate::kernels::cuda) fn relative_attention_launch(
    stream: &RawStream,
    call: &RelativeAttentionRaw<'_>,
) -> Result<(), DeviceFailure> {
    let bytes = match call.tile {
        Some(tile) => tile.region_bytes(call.shape),
        None => call.shape.region_bytes(),
    }
    .ok_or_else(invalid_argument)?;
    let spans = std::array::from_fn::<_, RANGES, _>(|i| Span::of(&call.ranges[i], bytes[i]));
    for (i, range) in call.ranges.iter().enumerate() {
        if range.bytes() < bytes[i]
            || !range.offset().is_multiple_of(4)
            || (i < WRITTEN && spans[i + 1..].iter().any(|other| !spans[i].apart(other)))
        {
            return Err(invalid_argument());
        }
    }
    let shape = call.shape;
    let args = Args {
        ranges: call.ranges.map(|range| range.address()),
        query_heads: shape.query_heads as c_int,
        kv_heads: shape.kv_heads as c_int,
        relative_dims: shape.relative_dims as c_int,
        relative_extent: shape.relative_extent as c_int,
        capacity: shape.capacity as c_int,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: region_bytes bounded the geometry, interval and strides. The checks
    // above prove all touched ranges are aligned and in bounds, with writes disjoint
    // from all other regions. The adapter copies the args before returning.
    let status = unsafe {
        match call.tile {
            Some(tile) => raw_tile_launch(
                &args,
                tile.first as c_int,
                tile.count as c_int,
                (tile.input_stride / 4) as c_int,
                (tile.query_stride / 4) as c_int,
                (tile.side_stride / 4) as c_int,
                stream.as_ptr(),
                &mut stage,
            ),
            None => raw_launch(&args, stream.as_ptr(), &mut stage),
        }
    };
    checked(status, stage)
}
