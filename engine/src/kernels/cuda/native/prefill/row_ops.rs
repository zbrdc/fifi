//! Private FFI for shared F32 row operations. Checked live ranges and the explicit
//! stream arrive through launch_on; C repeats shape, alignment and overlap checks.

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use crate::kernels::cuda::native::resident::row_scale::row_pow2_geometry;

pub(in crate::kernels::cuda) fn multiply_bytes(
    rows: usize,
    columns: usize,
    rhs_columns: usize,
) -> Option<(usize, usize)> {
    if rhs_columns != 1 && rhs_columns != columns {
        return None;
    }
    Some((
        row_pow2_geometry(rows, columns)?.0,
        row_pow2_geometry(rows, rhs_columns)?.0,
    ))
}

pub(in crate::kernels::cuda) fn gather_bytes(
    rows: usize,
    columns: usize,
    source_rows: usize,
) -> Option<(usize, usize, usize)> {
    let (target, indices) = row_pow2_geometry(rows, columns)?;
    Some((row_pow2_geometry(source_rows, columns)?.0, indices, target))
}

pub(in crate::kernels::cuda) fn rank_sum_bytes(
    rows: usize,
    columns: usize,
    rank: usize,
) -> Option<(usize, usize)> {
    if !(1..=32).contains(&rank) {
        return None;
    }
    Some((
        row_pow2_geometry(rows.checked_mul(rank)?, columns)?.0,
        row_pow2_geometry(rows, columns)?.0,
    ))
}

#[repr(C)]
struct UnaryArgs {
    x: *const f32,
    x_bytes: usize,
    out: *mut f32,
    out_bytes: usize,
    flag: *mut u32,
    flag_bytes: usize,
    rows: usize,
    columns: usize,
}

#[repr(C)]
struct MulArgs {
    x: *const f32,
    x_bytes: usize,
    rhs: *const f32,
    rhs_bytes: usize,
    out: *mut f32,
    out_bytes: usize,
    flag: *mut u32,
    flag_bytes: usize,
    rows: usize,
    columns: usize,
    rhs_columns: usize,
}

#[repr(C)]
struct GatherArgs {
    x: *const f32,
    x_bytes: usize,
    indices: *const i32,
    indices_bytes: usize,
    out: *mut f32,
    out_bytes: usize,
    flag: *mut u32,
    flag_bytes: usize,
    rows: usize,
    columns: usize,
    source_rows: usize,
}

#[repr(C)]
struct SumArgs {
    x: *const f32,
    x_bytes: usize,
    out: *mut f32,
    out_bytes: usize,
    flag: *mut u32,
    flag_bytes: usize,
    rows: usize,
    columns: usize,
    rank: usize,
}

const _: () = assert!(
    size_of::<UnaryArgs>() == 64
        && offset_of!(UnaryArgs, x) == 0
        && offset_of!(UnaryArgs, x_bytes) == 8
        && offset_of!(UnaryArgs, out) == 16
        && offset_of!(UnaryArgs, out_bytes) == 24
        && offset_of!(UnaryArgs, flag) == 32
        && offset_of!(UnaryArgs, flag_bytes) == 40
        && offset_of!(UnaryArgs, rows) == 48
        && offset_of!(UnaryArgs, columns) == 56
);
const _: () = assert!(
    size_of::<MulArgs>() == 88
        && offset_of!(MulArgs, x) == 0
        && offset_of!(MulArgs, x_bytes) == 8
        && offset_of!(MulArgs, rhs) == 16
        && offset_of!(MulArgs, rhs_bytes) == 24
        && offset_of!(MulArgs, out) == 32
        && offset_of!(MulArgs, out_bytes) == 40
        && offset_of!(MulArgs, flag) == 48
        && offset_of!(MulArgs, flag_bytes) == 56
        && offset_of!(MulArgs, rows) == 64
        && offset_of!(MulArgs, columns) == 72
        && offset_of!(MulArgs, rhs_columns) == 80
);
const _: () = assert!(
    size_of::<GatherArgs>() == 88
        && offset_of!(GatherArgs, x) == 0
        && offset_of!(GatherArgs, x_bytes) == 8
        && offset_of!(GatherArgs, indices) == 16
        && offset_of!(GatherArgs, indices_bytes) == 24
        && offset_of!(GatherArgs, out) == 32
        && offset_of!(GatherArgs, out_bytes) == 40
        && offset_of!(GatherArgs, flag) == 48
        && offset_of!(GatherArgs, flag_bytes) == 56
        && offset_of!(GatherArgs, rows) == 64
        && offset_of!(GatherArgs, columns) == 72
        && offset_of!(GatherArgs, source_rows) == 80
);
const _: () = assert!(
    size_of::<SumArgs>() == 72
        && offset_of!(SumArgs, x) == 0
        && offset_of!(SumArgs, x_bytes) == 8
        && offset_of!(SumArgs, out) == 16
        && offset_of!(SumArgs, out_bytes) == 24
        && offset_of!(SumArgs, flag) == 32
        && offset_of!(SumArgs, flag_bytes) == 40
        && offset_of!(SumArgs, rows) == 48
        && offset_of!(SumArgs, columns) == 56
        && offset_of!(SumArgs, rank) == 64
);

unsafe extern "C" {
    fn inference_cuda_row_sigmoid_launch(
        args: *const UnaryArgs,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_row_mul_launch(
        args: *const MulArgs,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_row_gather_launch(
        args: *const GatherArgs,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_row_rank_sum_launch(
        args: *const SumArgs,
        stream: *mut c_void,
        stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) struct RowUnaryRaw<'a> {
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) flag: RawRange<'a>,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) columns: usize,
}

pub(in crate::kernels::cuda) struct RowMulRaw<'a> {
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) rhs: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) flag: RawRange<'a>,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) columns: usize,
    pub(in crate::kernels::cuda) rhs_columns: usize,
}

pub(in crate::kernels::cuda) struct RowGatherRaw<'a> {
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) indices: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) flag: RawRange<'a>,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) columns: usize,
    pub(in crate::kernels::cuda) source_rows: usize,
}

pub(in crate::kernels::cuda) struct RowSumRaw<'a> {
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) flag: RawRange<'a>,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) columns: usize,
    pub(in crate::kernels::cuda) rank: usize,
}

fn check_regions(regions: &[(&RawRange<'_>, usize, bool)]) -> Result<(), DeviceFailure> {
    for (index, (range, needed, written)) in regions.iter().enumerate() {
        if range.bytes() < *needed || !range.offset().is_multiple_of(4) {
            return Err(invalid_argument());
        }
        for (prior, bytes, other_written) in &regions[..index] {
            if (*written || *other_written)
                && !Span::of(range, *needed).apart(&Span::of(prior, *bytes))
            {
                return Err(invalid_argument());
            }
        }
    }
    Ok(())
}

pub(in crate::kernels::cuda) fn row_sigmoid_launch(
    stream: &RawStream,
    call: &RowUnaryRaw<'_>,
) -> Result<(), DeviceFailure> {
    let bytes = row_pow2_geometry(call.rows, call.columns)
        .ok_or_else(invalid_argument)?
        .0;
    check_regions(&[
        (&call.x, bytes, false),
        (&call.out, bytes, true),
        (&call.flag, 4, true),
    ])?;
    let args = UnaryArgs {
        x: call.x.address().cast::<f32>().cast_const(),
        x_bytes: call.x.bytes(),
        out: call.out.address().cast(),
        out_bytes: call.out.bytes(),
        flag: call.flag.address().cast(),
        flag_bytes: call.flag.bytes(),
        rows: call.rows,
        columns: call.columns,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: sized, aligned live ranges, disjoint writes and a leased stream; C repeats checks.
    // Allocation drops drain the device before free, so queued reads/writes retain live storage.
    let status = unsafe { inference_cuda_row_sigmoid_launch(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

pub(in crate::kernels::cuda) fn row_mul_launch(
    stream: &RawStream,
    call: &RowMulRaw<'_>,
) -> Result<(), DeviceFailure> {
    let (bytes, rhs) =
        multiply_bytes(call.rows, call.columns, call.rhs_columns).ok_or_else(invalid_argument)?;
    check_regions(&[
        (&call.x, bytes, false),
        (&call.rhs, rhs, false),
        (&call.out, bytes, true),
        (&call.flag, 4, true),
    ])?;
    let args = MulArgs {
        x: call.x.address().cast::<f32>().cast_const(),
        x_bytes: call.x.bytes(),
        rhs: call.rhs.address().cast::<f32>().cast_const(),
        rhs_bytes: call.rhs.bytes(),
        out: call.out.address().cast(),
        out_bytes: call.out.bytes(),
        flag: call.flag.address().cast(),
        flag_bytes: call.flag.bytes(),
        rows: call.rows,
        columns: call.columns,
        rhs_columns: call.rhs_columns,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: check_regions proves extents/alignment/disjoint writes of live owned ranges.
    // launch_on owns the stream lease; allocation drops synchronize outstanding use.
    let status = unsafe { inference_cuda_row_mul_launch(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

pub(in crate::kernels::cuda) fn row_gather_launch(
    stream: &RawStream,
    call: &RowGatherRaw<'_>,
) -> Result<(), DeviceFailure> {
    let (source, ids, target) =
        gather_bytes(call.rows, call.columns, call.source_rows).ok_or_else(invalid_argument)?;
    check_regions(&[
        (&call.x, source, false),
        (&call.indices, ids, false),
        (&call.out, target, true),
        (&call.flag, 4, true),
    ])?;
    let args = GatherArgs {
        x: call.x.address().cast::<f32>().cast_const(),
        x_bytes: call.x.bytes(),
        indices: call.indices.address().cast::<i32>().cast_const(),
        indices_bytes: call.indices.bytes(),
        out: call.out.address().cast(),
        out_bytes: call.out.bytes(),
        flag: call.flag.address().cast(),
        flag_bytes: call.flag.bytes(),
        rows: call.rows,
        columns: call.columns,
        source_rows: call.source_rows,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: live sized/aligned ranges have disjoint writes, with launch_on's stream lease.
    // C repeats span checks; device IDs are bounded before dereference and owners outlive use.
    let status = unsafe { inference_cuda_row_gather_launch(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

pub(in crate::kernels::cuda) fn row_rank_sum_launch(
    stream: &RawStream,
    call: &RowSumRaw<'_>,
) -> Result<(), DeviceFailure> {
    let (source, target) =
        rank_sum_bytes(call.rows, call.columns, call.rank).ok_or_else(invalid_argument)?;
    check_regions(&[
        (&call.x, source, false),
        (&call.out, target, true),
        (&call.flag, 4, true),
    ])?;
    let args = SumArgs {
        x: call.x.address().cast::<f32>().cast_const(),
        x_bytes: call.x.bytes(),
        out: call.out.address().cast(),
        out_bytes: call.out.bytes(),
        flag: call.flag.address().cast(),
        flag_bytes: call.flag.bytes(),
        rows: call.rows,
        columns: call.columns,
        rank: call.rank,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: checked live extents, disjoint writes and a leased stream; C repeats validation.
    // Every source rank lies inside the proved input extent and drops drain before free.
    let status = unsafe { inference_cuda_row_rank_sum_launch(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod tests;
