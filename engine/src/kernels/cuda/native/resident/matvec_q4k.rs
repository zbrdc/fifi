//! Raw FFI to `inference_cuda_matvec_q4k_launch` of `native/resident/matvec_q4k.cu`
//! (declared in `native/resident/matvec_q4k.h`). Child of `native::resident`, so it inherits
//! the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this module relies on (mirrors `matvec_q80`):
//! - A device range is proved in bounds by `RawRange::new` before it gets here. This module
//!   proves, again and on the sizes of the call, that `w` holds `rows` rows of `k/256` native
//!   Q4_K blocks (144 bytes each), that `x_q8` holds the padded blocks of `ncols` rows, that
//!   `y` holds `ncols * rows` floats, that `y`, which the kernel writes, shares no byte with
//!   `w` or `x_q8`, and that every range is aligned (2 bytes for the weights, whose f16
//!   scales sit at even offsets; 4 for the rest). The C entry repeats the checks on the
//!   addresses.
//! - The launch entry only queues work on the stream it is given; a fault in the kernel
//!   surfaces at the next synchronize.
//! - The allocations stay valid until the queued launch finishes because `inference_cuda_free`
//!   synchronizes the whole device before `cudaFree`.
//! - No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// The Q4_K block: 256 values, 144 bytes on disk (f16 d, f16 dmin, 12 packed scale bytes, 128
/// code bytes). The inline reference of `resident/matvec_q/tests.rs` is the CPU oracle of the
/// layout.
pub(in crate::kernels::cuda) const Q4K_BLOCK: usize = 144;

/// Mirror of `inference_cuda_matvec_q4k_args` (`matvec_q4k.h`), field for field.
/// The layout is pinned as well by the `static_assert` of the header.
#[repr(C)]
struct MatvecQ4kArgsRepr {
    w: *const c_void,
    x_q8: *const c_void,
    y: *mut f32,
    k: c_int,
    rows: c_int,
    ncols: c_int,
}

const _: () = assert!(
    size_of::<MatvecQ4kArgsRepr>() == 40
        && offset_of!(MatvecQ4kArgsRepr, w) == 0
        && offset_of!(MatvecQ4kArgsRepr, x_q8) == 8
        && offset_of!(MatvecQ4kArgsRepr, y) == 16
        && offset_of!(MatvecQ4kArgsRepr, k) == 24
        && offset_of!(MatvecQ4kArgsRepr, rows) == 28
        && offset_of!(MatvecQ4kArgsRepr, ncols) == 32
);

unsafe extern "C" {
    #[link_name = "inference_cuda_matvec_q4k_launch"]
    fn matvec_q4k_launch_raw(
        args: *const MatvecQ4kArgsRepr,
        x_row_bytes: usize,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// The in-house Q4_K matvec call: `w` native blocks, `x_q8` the padded `quantize_q8_1` rows,
/// `x_row_bytes` one activation row's stride (`q8_1_bytes(k, 1)`), `y` f32 out with column `j`
/// at `y + j * rows`. `rows` may be any positive count: the kernel owns whole rows, unlike
/// the vendored matvec's row groups.
pub(in crate::kernels::cuda) struct MatvecQ4kRaw<'a> {
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) x_q8: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) ncols: usize,
}

pub(in crate::kernels::cuda) fn check_matvec_q4k(
    call: &MatvecQ4kRaw<'_>,
    x_row_bytes: usize,
) -> Result<(), DeviceFailure> {
    let (k, rows, ncols) = (call.k, call.rows, call.ncols);
    let blocks = k / 256;
    let need_w = rows
        .checked_mul(blocks)
        .and_then(|n| n.checked_mul(Q4K_BLOCK));
    let ints = [
        c_int::try_from(k).ok(),
        c_int::try_from(rows).ok(),
        c_int::try_from(ncols).ok(),
    ];
    let Some(need_w) = need_w else {
        return Err(invalid_argument());
    };
    let aligned = call.w.offset.is_multiple_of(2)
        && call.x_q8.offset.is_multiple_of(4)
        && call.y.offset.is_multiple_of(4);
    let y = Span::of(&call.y, rows.saturating_mul(ncols).saturating_mul(4));
    let apart = y.apart(&Span::of(&call.w, need_w)) && y.apart(&Span::of(&call.x_q8, x_row_bytes));
    if ints.iter().any(|int| int.is_none())
        || !aligned
        || !apart
        || x_row_bytes == 0
        || call.x_q8.bytes < x_row_bytes.saturating_mul(ncols)
        || call.y.bytes < rows.saturating_mul(ncols).saturating_mul(4)
        || call.w.bytes < need_w
        || k == 0
        || rows == 0
        || !(1..=8).contains(&ncols)
        || !k.is_multiple_of(256)
    {
        return Err(invalid_argument());
    }
    Ok(())
}

pub(in crate::kernels::cuda) fn matvec_q4k_launch(
    stream: &RawStream,
    call: &MatvecQ4kRaw<'_>,
    x_row_bytes: usize,
) -> Result<(), DeviceFailure> {
    check_matvec_q4k(call, x_row_bytes)?;
    let mut stage = STAGE_NONE;
    // SAFETY: `check_matvec_q4k` proved the `w`, `x_q8` and `y` ranges inside their live
    // allocations, large enough for the call, aligned, and `y` disjoint from the ranges the
    // kernel reads; the adapter repeats the checks on the addresses. The allocations stay
    // valid until the queued launch finishes because `inference_cuda_free` synchronizes the
    // device first. `stream` is live for the call.
    let args = MatvecQ4kArgsRepr {
        w: call.w.address().cast_const(),
        x_q8: call.x_q8.address().cast_const(),
        y: call.y.address().cast::<f32>(),
        k: c_int::try_from(call.k).map_err(|_| invalid_argument())?,
        rows: c_int::try_from(call.rows).map_err(|_| invalid_argument())?,
        ncols: c_int::try_from(call.ncols).map_err(|_| invalid_argument())?,
    };
    let status =
        unsafe { matvec_q4k_launch_raw(&args, x_row_bytes, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
