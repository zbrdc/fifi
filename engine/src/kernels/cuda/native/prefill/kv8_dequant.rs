//! Raw KV8-prefix expansion launch. Raw ranges are live allocation slices; size,
//! alignment and output/read disjointness are checked here and by the C entry.
//! launch_on owns the stream dispatch guard, and allocation free drains queued work.
//! Pointers and unsafe calls never leave this native child.

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

#[repr(C)]
struct Kv8DequantArgsRaw {
    codes: *const c_void,
    scales: *const c_void,
    out: *mut c_void,
    capacity: c_int,
    rows: c_int,
    n_kv_head: c_int,
    head_dim: c_int,
}

const _: () = assert!(
    size_of::<Kv8DequantArgsRaw>() == 40
        && offset_of!(Kv8DequantArgsRaw, codes) == 0
        && offset_of!(Kv8DequantArgsRaw, scales) == 8
        && offset_of!(Kv8DequantArgsRaw, out) == 16
        && offset_of!(Kv8DequantArgsRaw, capacity) == 24
        && offset_of!(Kv8DequantArgsRaw, rows) == 28
        && offset_of!(Kv8DequantArgsRaw, n_kv_head) == 32
        && offset_of!(Kv8DequantArgsRaw, head_dim) == 36
);

unsafe extern "C" {
    #[link_name = "inference_cuda_kv8_dequant_launch"]
    fn kv8_dequant_launch_raw(
        args: *const Kv8DequantArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_kv8_dequant_segment_launch"]
    fn kv8_dequant_segment_launch_raw(
        args: *const Kv8DequantArgsRaw,
        source_row_origin: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct Kv8DequantInts {
    pub(in crate::kernels::cuda) capacity: c_int,
    pub(in crate::kernels::cuda) rows: c_int,
    pub(in crate::kernels::cuda) n_kv_head: c_int,
    pub(in crate::kernels::cuda) head_dim: c_int,
}

pub(in crate::kernels::cuda) struct Kv8DequantRaw<'a> {
    pub(in crate::kernels::cuda) codes: RawRange<'a>,
    pub(in crate::kernels::cuda) scales: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: Kv8DequantInts,
}

fn extents(i: &Kv8DequantInts) -> Result<(usize, usize, usize), DeviceFailure> {
    if !(1..=(1 << 30) - 1).contains(&i.capacity)
        || !(1..=i.capacity).contains(&i.rows)
        || !(1..=64).contains(&i.n_kv_head)
        || !(32..=1024).contains(&i.head_dim)
        || i.head_dim % 32 != 0
    {
        return Err(invalid_argument());
    }
    let product = |rows: c_int| {
        (rows as usize)
            .checked_mul(i.n_kv_head as usize)
            .and_then(|n| n.checked_mul(i.head_dim as usize))
            .ok_or_else(invalid_argument)
    };
    let codes = product(i.capacity)?;
    let scales = (codes / 32).checked_mul(2).ok_or_else(invalid_argument)?;
    let out = product(i.rows)?
        .checked_mul(2)
        .ok_or_else(invalid_argument)?;
    Ok((codes, scales, out))
}

fn check(io: &Kv8DequantRaw<'_>) -> Result<(), DeviceFailure> {
    let (codes, scales, out) = extents(&io.ints)?;
    let fit = io.codes.bytes() >= codes && io.scales.bytes() >= scales && io.out.bytes() >= out;
    let aligned = io.codes.offset().is_multiple_of(4)
        && io.scales.offset().is_multiple_of(4)
        && io.out.offset().is_multiple_of(16);
    let apart = Span::of(&io.out, out).apart(&Span::of(&io.codes, codes))
        && Span::of(&io.out, out).apart(&Span::of(&io.scales, scales));
    if !(fit && aligned && apart) {
        return Err(invalid_argument());
    }
    Ok(())
}

pub(in crate::kernels::cuda) fn kv8_dequant_launch(
    stream: &RawStream,
    io: Kv8DequantRaw<'_>,
    source_row_origin: c_int,
) -> Result<(), DeviceFailure> {
    check(&io)?;
    if source_row_origin < 0
        || i64::from(source_row_origin) + i64::from(io.ints.rows) > i64::from(io.ints.capacity)
    {
        return Err(invalid_argument());
    }
    let args = Kv8DequantArgsRaw {
        codes: io.codes.address().cast_const(),
        scales: io.scales.address().cast_const(),
        out: io.out.address(),
        capacity: io.ints.capacity,
        rows: io.ints.rows,
        n_kv_head: io.ints.n_kv_head,
        head_dim: io.ints.head_dim,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: check proved live extents, alignment and output/read disjointness.
    // The C entry repeats the address checks. Free drains queued work; args and stream
    // remain live across this asynchronous launch call.
    let status = unsafe {
        if source_row_origin == 0 {
            kv8_dequant_launch_raw(&args, stream.as_ptr(), &mut stage)
        } else {
            kv8_dequant_segment_launch_raw(&args, source_row_origin, stream.as_ptr(), &mut stage)
        }
    };
    checked(status, stage)
}
