//! Raw FFI to inference_cuda_row_scale_launch of native/resident/row_scale.cu
//! (declared in native/resident/row_scale.h). Child of native::resident, so it inherits
//! the allow(unsafe_code) of native.rs.
//!
//! Safety invariants this module relies on:
//! - A device range is proved in bounds by RawRange::new before it gets here. This module
//!   proves, again and on the sizes of the call, that out and x hold m*n floats each, are
//!   4-byte aligned, and that out, which the kernel writes, shares no byte with x. The C
//!   entry repeats these checks on the addresses.
//! - The launch entry only queues work on the stream it is given and returns the launch
//!   status; a fault in the kernel surfaces at the next synchronize.
//! - The allocations stay valid until the queued launch finishes because
//!   inference_cuda_free synchronizes the whole device before cudaFree.
//! - Every call runs under the process dispatch guard and on the thread that owns the
//!   context.
//! - No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

unsafe extern "C" {
    #[link_name = "inference_cuda_row_scale_launch"]
    fn row_scale_launch_raw(
        args: *const RowScaleArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Mirror of inference_cuda_row_scale_args (row_scale.h), field for field.
#[repr(C)]
struct RowScaleArgsRepr {
    out: *mut c_void,
    x: *const c_void,
    scale: f32,
    m: c_int,
    n: c_int,
}

const _: () = assert!(
    size_of::<RowScaleArgsRepr>() == 32
        && offset_of!(RowScaleArgsRepr, out) == 0
        && offset_of!(RowScaleArgsRepr, x) == 8
        && offset_of!(RowScaleArgsRepr, scale) == 16
        && offset_of!(RowScaleArgsRepr, m) == 20
        && offset_of!(RowScaleArgsRepr, n) == 24
);

/// What one row-scale launch reads and writes: m rows of n f32 values and the scale.
pub(in crate::kernels::cuda) struct RowScaleRaw<'a> {
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) scale: f32,
    pub(in crate::kernels::cuda) m: usize,
    pub(in crate::kernels::cuda) n: usize,
}

/// Proves a row-scale launch before the native call: a finite scale, m and n in
/// range, both ranges sized and 4-byte aligned, and out disjoint from x.
fn check_row_scale(call: &RowScaleRaw<'_>) -> Result<(), DeviceFailure> {
    let values = call
        .m
        .checked_mul(call.n)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    let sized = call.out.bytes >= values && call.x.bytes >= values;
    let aligned = call.out.offset.is_multiple_of(4) && call.x.offset.is_multiple_of(4);
    let apart = Span::of(&call.out, values).apart(&Span::of(&call.x, values));
    let dims = (1..=65_536).contains(&call.m)
        && (1..=(1 << 22)).contains(&call.n)
        && values / 4 <= (1usize << 32)
        && call.scale.is_finite();
    if !(sized && aligned && apart && dims) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the row scale of call on stream. Returns when the kernel is queued.
pub(in crate::kernels::cuda) fn row_scale_launch(
    stream: &RawStream,
    call: &RowScaleRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_row_scale(call)?;
    let args = RowScaleArgsRepr {
        out: call.out.address(),
        x: call.x.address().cast_const(),
        scale: call.scale,
        m: c_int::try_from(call.m).map_err(|_| invalid_argument())?,
        n: c_int::try_from(call.n).map_err(|_| invalid_argument())?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: check_row_scale proved the out and x ranges inside their live allocations,
    // large enough, aligned, and disjoint; the adapter repeats the checks on the addresses.
    // The allocations stay valid until the queued launch finishes because
    // inference_cuda_free synchronizes the device first. stream is live for the call and
    // args is a plain value the entry reads.
    let status = unsafe { row_scale_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[repr(C)]
struct RowPow2ArgsRepr {
    x: *const f32,
    x_bytes: usize,
    y: *mut f32,
    y_bytes: usize,
    scales: *mut f32,
    scales_bytes: usize,
    flag: *mut u32,
    flag_bytes: usize,
    rows: usize,
    columns: usize,
    mode: c_int,
    reserved: c_int,
}

/// Shared F32 working-row geometry, independent of the weight encoding.
pub(in crate::kernels::cuda) fn row_pow2_geometry(
    rows: usize,
    columns: usize,
) -> Option<(usize, usize)> {
    if !(1..=65_536).contains(&rows) || !(1..=(1 << 22)).contains(&columns) {
        return None;
    }
    let values = rows.checked_mul(columns)?;
    if values > (1usize << 32) {
        return None;
    }
    Some((values.checked_mul(4)?, rows.checked_mul(4)?))
}

unsafe extern "C" {
    #[link_name = "inference_cuda_row_pow2_launch"]
    fn row_pow2_launch_raw(
        args: *const RowPow2ArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) struct RowPow2Raw<'a> {
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) scales: RawRange<'a>,
    pub(in crate::kernels::cuda) flag: RawRange<'a>,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) columns: usize,
    pub(in crate::kernels::cuda) restore: bool,
}

pub(in crate::kernels::cuda) fn row_pow2_launch(
    stream: &RawStream,
    call: &RowPow2Raw<'_>,
) -> Result<(), DeviceFailure> {
    let (values, scales) =
        row_pow2_geometry(call.rows, call.columns).ok_or_else(invalid_argument)?;
    let ranges = [&call.x, &call.y, &call.scales, &call.flag];
    let need = [values, values, scales, 4];
    for i in 0..ranges.len() {
        if ranges[i].bytes < need[i] || !ranges[i].offset.is_multiple_of(4) {
            return Err(invalid_argument());
        }
        for j in 0..i {
            if !Span::of(ranges[i], need[i]).apart(&Span::of(ranges[j], need[j])) {
                return Err(invalid_argument());
            }
        }
    }
    let args = RowPow2ArgsRepr {
        x: call.x.address().cast::<f32>().cast_const(),
        x_bytes: call.x.bytes,
        y: call.y.address().cast(),
        y_bytes: call.y.bytes,
        scales: call.scales.address().cast(),
        scales_bytes: call.scales.bytes,
        flag: call.flag.address().cast(),
        flag_bytes: call.flag.bytes,
        rows: call.rows,
        columns: call.columns,
        mode: if call.restore { 1 } else { 0 },
        reserved: 0,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: all four live ranges are sized, aligned and disjoint. Context owns
    // the dispatch/stream lease, and allocation drops drain outstanding device use.
    let status = unsafe { row_pow2_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[repr(C)]
struct Bf16PublishArgsRepr {
    x: *const f32,
    x_bytes: usize,
    y: *mut f32,
    y_bytes: usize,
    flag: *mut u32,
    flag_bytes: usize,
    rows: usize,
    columns: usize,
}

unsafe extern "C" {
    #[link_name = "inference_cuda_bf16_publish_rows"]
    fn bf16_publish_raw(
        args: *const Bf16PublishArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) struct Bf16PublishRaw<'a> {
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) flag: RawRange<'a>,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) columns: usize,
}

pub(in crate::kernels::cuda) fn bf16_publish_launch(
    stream: &RawStream,
    call: &Bf16PublishRaw<'_>,
) -> Result<(), DeviceFailure> {
    let (values, _) = row_pow2_geometry(call.rows, call.columns).ok_or_else(invalid_argument)?;
    let ranges = [&call.x, &call.y, &call.flag];
    let need = [values, values, 4];
    for i in 0..ranges.len() {
        if ranges[i].bytes < need[i] || !ranges[i].offset.is_multiple_of(4) {
            return Err(invalid_argument());
        }
        for j in 0..i {
            if !Span::of(ranges[i], need[i]).apart(&Span::of(ranges[j], need[j])) {
                return Err(invalid_argument());
            }
        }
    }
    let args = Bf16PublishArgsRepr {
        x: call.x.address().cast::<f32>().cast_const(),
        x_bytes: call.x.bytes,
        y: call.y.address().cast(),
        y_bytes: call.y.bytes,
        flag: call.flag.address().cast(),
        flag_bytes: call.flag.bytes,
        rows: call.rows,
        columns: call.columns,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: live aligned disjoint spans are sized above; context owns the
    // explicit stream/dispatch lease and allocation drops drain queued work.
    let status = unsafe { bf16_publish_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod row_bf16_device_tests;

#[cfg(test)]
mod row_bf16_tests {
    use super::Bf16PublishArgsRepr;

    #[test]
    fn row_bf16_publication_abi_is_explicit_and_64_bytes() {
        assert_eq!(size_of::<Bf16PublishArgsRepr>(), 64);
        assert_eq!(align_of::<Bf16PublishArgsRepr>(), 8);
        assert_eq!(std::mem::offset_of!(Bf16PublishArgsRepr, x), 0);
        assert_eq!(std::mem::offset_of!(Bf16PublishArgsRepr, x_bytes), 8);
        assert_eq!(std::mem::offset_of!(Bf16PublishArgsRepr, y), 16);
        assert_eq!(std::mem::offset_of!(Bf16PublishArgsRepr, y_bytes), 24);
        assert_eq!(std::mem::offset_of!(Bf16PublishArgsRepr, flag), 32);
        assert_eq!(std::mem::offset_of!(Bf16PublishArgsRepr, flag_bytes), 40);
        assert_eq!(std::mem::offset_of!(Bf16PublishArgsRepr, rows), 48);
        assert_eq!(std::mem::offset_of!(Bf16PublishArgsRepr, columns), 56);
    }
}

#[cfg(test)]
mod row_pow2_device_tests;

#[cfg(test)]
mod row_pow2_tests {
    use super::{RowPow2ArgsRepr, row_pow2_geometry};

    #[test]
    fn batched_pair_geometry_covers_a_chunk_and_its_down_output() {
        assert_eq!(row_pow2_geometry(12, 512), Some((24_576, 48)));
        assert_eq!(row_pow2_geometry(3072, 512), Some((6_291_456, 12_288)));
        assert_eq!(row_pow2_geometry(3072, 2048), Some((25_165_824, 12_288)));
    }

    #[test]
    fn row_pow2_geometry_refuses_empty_overflow_and_grid_overflow() {
        for shape in [(0, 512), (12, 0), (usize::MAX, 512), (12, usize::MAX)] {
            assert!(row_pow2_geometry(shape.0, shape.1).is_none());
        }
        if usize::BITS == 64 {
            assert!(row_pow2_geometry(i32::MAX as usize + 1, 512).is_none());
            assert_eq!(
                row_pow2_geometry(65_536, 65_536),
                Some((1usize << 34, 1usize << 18))
            );
            assert!(row_pow2_geometry(65_536, 65_537).is_none());
        }
    }

    #[test]
    fn row_scale_private_abi_has_explicit_mode_and_reserved_tail() {
        assert_eq!(size_of::<RowPow2ArgsRepr>(), 88);
        assert_eq!(align_of::<RowPow2ArgsRepr>(), 8);
        assert_eq!(std::mem::offset_of!(RowPow2ArgsRepr, x), 0);
        assert_eq!(std::mem::offset_of!(RowPow2ArgsRepr, x_bytes), 8);
        assert_eq!(std::mem::offset_of!(RowPow2ArgsRepr, y), 16);
        assert_eq!(std::mem::offset_of!(RowPow2ArgsRepr, y_bytes), 24);
        assert_eq!(std::mem::offset_of!(RowPow2ArgsRepr, scales), 32);
        assert_eq!(std::mem::offset_of!(RowPow2ArgsRepr, scales_bytes), 40);
        assert_eq!(std::mem::offset_of!(RowPow2ArgsRepr, flag), 48);
        assert_eq!(std::mem::offset_of!(RowPow2ArgsRepr, flag_bytes), 56);
        assert_eq!(std::mem::offset_of!(RowPow2ArgsRepr, rows), 64);
        assert_eq!(std::mem::offset_of!(RowPow2ArgsRepr, columns), 72);
        assert_eq!(std::mem::offset_of!(RowPow2ArgsRepr, mode), 80);
        assert_eq!(std::mem::offset_of!(RowPow2ArgsRepr, reserved), 84);
    }
}
