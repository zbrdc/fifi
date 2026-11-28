//! Raw FFI to `inference_cuda_matvec_dq_launch` of `native/resident/matvec_dq.cu` (declared in
//! `native/resident/matvec_dq.h`). Child of `native::resident`, so it inherits the
//! `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this module relies on (mirrors `matvec_q5k`):
//! - A device range is proved in bounds by `RawRange::new` before it gets here. This module
//!   proves, again and on the sizes of the call, that `w` holds `rows` rows of `k/256` native
//!   blocks of the call's type, that `x` holds `k` floats, that `y` holds `rows` floats, that
//!   `w` and `x` start at 16-byte offsets and `y` at 4, that `y`, which the kernel writes,
//!   shares no byte with `w` or `x`, and that the type is Q4_K or Q6_K. The C entry repeats the
//!   checks on the addresses.
//! - The launch entry only queues work on the stream it is given; a fault in the kernel
//!   surfaces at the next synchronize.
//! - The allocations stay valid until the queued launch finishes because `inference_cuda_free`
//!   synchronizes the whole device before `cudaFree`.
//! - No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use crate::kernels::cuda::resident::matvec_q::QuantType;

/// Mirror of `inference_cuda_matvec_dq_args` (`matvec_dq.h`), field for field.
/// The layout is pinned as well by the `static_assert` of the header.
#[repr(C)]
struct MatvecDqArgsRepr {
    w: *const c_void,
    x: *const c_void,
    y: *mut f32,
    k: c_int,
    rows: c_int,
    ty: c_int,
}

const _: () = assert!(
    size_of::<MatvecDqArgsRepr>() == 40
        && offset_of!(MatvecDqArgsRepr, w) == 0
        && offset_of!(MatvecDqArgsRepr, x) == 8
        && offset_of!(MatvecDqArgsRepr, y) == 16
        && offset_of!(MatvecDqArgsRepr, k) == 24
        && offset_of!(MatvecDqArgsRepr, rows) == 28
        && offset_of!(MatvecDqArgsRepr, ty) == 32
);

unsafe extern "C" {
    #[link_name = "inference_cuda_matvec_dq_launch"]
    fn matvec_dq_launch_raw(
        args: *const MatvecDqArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// The decode dq matvec call: `w` native blocks of Q4_K or Q6_K, `x` f32 with `k` values at a
/// 16-byte offset, `y` f32 out with `rows` values. `rows` may be any positive count: the
/// kernel owns whole rows.
pub(in crate::kernels::cuda) struct MatvecDqRaw<'a> {
    pub(in crate::kernels::cuda) ty: QuantType,
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) rows: usize,
}

fn check_matvec_dq(call: &MatvecDqRaw<'_>) -> Result<(), DeviceFailure> {
    let (k, rows) = (call.k, call.rows);
    if !matches!(call.ty, QuantType::Q4K | QuantType::Q6K | QuantType::Q80) {
        return Err(invalid_argument());
    }
    let need_w = call.ty.weight_bytes(rows, k);
    let need_x = k.checked_mul(4);
    let need_y = rows.checked_mul(4);
    let ints = [c_int::try_from(k).ok(), c_int::try_from(rows).ok()];
    let Some(need_w) = need_w else {
        return Err(invalid_argument());
    };
    let Some((need_x, need_y)) = need_x.zip(need_y) else {
        return Err(invalid_argument());
    };
    let aligned = call.w.offset.is_multiple_of(16)
        && call.x.offset.is_multiple_of(16)
        && call.y.offset.is_multiple_of(4);
    let y = Span::of(&call.y, need_y);
    let apart = y.apart(&Span::of(&call.w, need_w)) && y.apart(&Span::of(&call.x, need_x));
    if ints.iter().any(|int| int.is_none())
        || !aligned
        || !apart
        || call.w.bytes < need_w
        || call.x.bytes < need_x
        || call.y.bytes < need_y
        || k == 0
        || !k.is_multiple_of(256)
        || rows == 0
    {
        return Err(invalid_argument());
    }
    Ok(())
}

pub(in crate::kernels::cuda) fn matvec_dq_launch(
    stream: &RawStream,
    call: &MatvecDqRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_matvec_dq(call)?;
    let mut stage = STAGE_NONE;
    // SAFETY: `check_matvec_dq` proved the `w`, `x` and `y` ranges inside their live
    // allocations, large enough for the call, aligned, and `y` disjoint from the ranges the
    // kernel reads; the adapter repeats the checks on the addresses. The allocations stay
    // valid until the queued launch finishes because `inference_cuda_free` synchronizes the
    // device first. `stream` is live for the call. `ty` is Q4_K or Q6_K, the two types whose
    // `ggml` value is defined.
    let args = MatvecDqArgsRepr {
        w: call.w.address().cast_const(),
        x: call.x.address().cast_const(),
        y: call.y.address().cast::<f32>(),
        k: c_int::try_from(call.k).map_err(|_| invalid_argument())?,
        rows: c_int::try_from(call.rows).map_err(|_| invalid_argument())?,
        ty: call.ty.ggml(),
    };
    let status = unsafe { matvec_dq_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::cuda::native::STATUS_INVALID_ARGUMENT;

    const fn pointer<T>(address: usize) -> *mut T {
        std::ptr::without_provenance_mut(address)
    }

    type Case = (&'static str, fn(&mut MatvecDqArgsRepr));

    /// An address that would launch: every case changes one thing so the entry refuses before
    /// it uses any pointer, so the test needs no device. `w` is 64 rows of k 4096 in Q4_K
    /// (147456 bytes from 0x10_0000), `x` is 16 KB from 0x20_0000, `y` is 256 bytes from
    /// 0x30_0000.
    fn base() -> MatvecDqArgsRepr {
        MatvecDqArgsRepr {
            w: pointer::<c_void>(0x10_0000).cast_const(),
            x: pointer::<c_void>(0x20_0000).cast_const(),
            y: pointer(0x30_0000),
            k: 4096,
            rows: 64,
            ty: QuantType::Q4K.ggml(),
        }
    }

    /// `(status, stage)` of the entry for `args`.
    fn run(edit: fn(&mut MatvecDqArgsRepr)) -> (c_int, c_int) {
        let mut args = base();
        edit(&mut args);
        let mut stage = STAGE_NONE;
        // SAFETY: every case below makes the entry return before it reads `args` or `stream`;
        // `stage` is a live, writable c_int.
        let status =
            unsafe { matvec_dq_launch_raw(&args, pointer::<c_void>(0x40_0000), &mut stage) };
        (status, stage)
    }

    #[test]
    fn the_matvec_dq_entry_refuses_bad_arguments_before_any_device_call() {
        let refused = (STATUS_INVALID_ARGUMENT, STAGE_NONE);
        let cases: [Case; 14] = [
            ("NULL w", |a| a.w = std::ptr::null()),
            ("NULL x", |a| a.x = std::ptr::null()),
            ("NULL y", |a| a.y = std::ptr::null_mut()),
            ("k = 0", |a| a.k = 0),
            ("k = 100 (not a multiple of 256)", |a| a.k = 100),
            ("k below zero", |a| a.k = -256),
            ("no rows", |a| a.rows = 0),
            ("a type the kernel does not serve", |a| a.ty = 6),
            ("w not 16-byte aligned", |a| {
                a.w = pointer::<c_void>(0x10_0008).cast_const()
            }),
            ("x not 16-byte aligned", |a| {
                a.x = pointer::<c_void>(0x20_0008).cast_const()
            }),
            ("y not 4-byte aligned", |a| a.y = pointer::<f32>(0x30_0002)),
            ("y inside w", |a| a.y = pointer::<f32>(0x11_0000)),
            ("y inside x", |a| a.y = pointer::<f32>(0x20_1000)),
            ("y over the end of w", |a| {
                a.y = pointer::<f32>(0x10_0000 + 147456 - 256)
            }),
        ];
        for (name, edit) in cases {
            assert_eq!(run(edit), refused, "{name}");
        }
        let mut stage = STAGE_NONE;
        // SAFETY: the entry tests its pointers before any field, so a NULL args and a NULL
        // stream are refused, not dereferenced; `stage` is a live, writable c_int.
        let null_args = unsafe {
            matvec_dq_launch_raw(
                pointer::<MatvecDqArgsRepr>(0).cast_const(),
                pointer::<c_void>(0x40_0000),
                &mut stage,
            )
        };
        assert_eq!((null_args, stage), refused, "NULL args");
        let null_stream =
            unsafe { matvec_dq_launch_raw(&base(), std::ptr::null_mut(), &mut stage) };
        assert_eq!((null_stream, stage), refused, "NULL stream");
    }
}
