//! Raw FFI to `inference_cuda_quantize_q8_1_launch` of `native/resident/quantize_q8_1.cu`
//! (declared in `native/resident/quantize_q8_1.h`). Child of `native::resident`, so it inherits
//! the `allow(unsafe_code)` of `native.rs` (decision 0047, child modules of `native.rs`).
//!
//! Safety invariants this module relies on:
//! - A device range is proved in bounds by `RawRange::new` before it gets here. This module
//!   proves, again and on the sizes of the call, that `x` holds `ncols * k` floats and `y` holds
//!   `q8_1_bytes(k, ncols)` bytes, that both are 4-byte aligned, and that `y`, which the kernel
//!   writes, shares no byte with `x`. The C entry repeats these checks on the addresses.
//! - The launch entry only queues work on the stream it is given and returns the launch status;
//!   a fault in the kernel surfaces at the next synchronize.
//! - The allocations stay valid until the queued launch finishes because `inference_cuda_free`
//!   synchronizes the whole device before `cudaFree`.
//! - Every call runs under the process dispatch guard and on the thread that owns the context
//!   (decisions 0023 and 0024).
//! - No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

unsafe extern "C" {
    #[link_name = "inference_cuda_quantize_q8_1_launch"]
    fn quantize_q8_1_launch_raw(
        args: *const QuantizeArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Mirror of `inference_cuda_quantize_q8_1_args` (`quantize_q8_1.h`), field for field.
/// The layout is pinned below and by the `static_assert` of the header.
#[repr(C)]
struct QuantizeArgsRepr {
    x: *const c_void,
    y: *mut c_void,
    k: c_int,
    ncols: c_int,
}

const _: () = assert!(
    size_of::<QuantizeArgsRepr>() == 24
        && offset_of!(QuantizeArgsRepr, x) == 0
        && offset_of!(QuantizeArgsRepr, y) == 8
        && offset_of!(QuantizeArgsRepr, k) == 16
        && offset_of!(QuantizeArgsRepr, ncols) == 20
);

/// What one quantize launch reads and writes: `x` holds `ncols` rows of `k` f32 values, `y`
/// receives the `block_q8_1` rows of the padded width.
pub(in crate::kernels::cuda) struct QuantizeRaw<'a> {
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) ncols: usize,
}

/// Bytes of the `block_q8_1` output of `ncols` rows of `k` values: rows padded to
/// `MATRIX_ROW_PADDING` (512), 32 values and 36 bytes per block. The safe wrapper's
/// `q8_1_bytes` is the same rule; this copy keeps the native child free of safe-tree imports.
fn q8_1_bytes_native(k: usize, ncols: usize) -> Option<usize> {
    let padded = k.checked_add(511)? & !511;
    let blocks = padded.checked_div(32)?;
    blocks.checked_mul(36)?.checked_mul(ncols)
}

/// Proves a quantize launch before the native call: `k` a multiple of 32, `ncols` in `1..=8`,
/// `x` and `y` sized and 4-byte aligned, and `y` disjoint from `x`.
fn check_quantize(call: &QuantizeRaw<'_>) -> Result<(), DeviceFailure> {
    let (k, ncols) = (call.k, call.ncols);
    let x_bytes = k
        .checked_mul(ncols)
        .and_then(|values| values.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    let y_bytes = q8_1_bytes_native(k, ncols).ok_or_else(invalid_argument)?;
    let sized = call.x.bytes >= x_bytes && call.y.bytes >= y_bytes;
    let aligned = call.x.offset.is_multiple_of(4) && call.y.offset.is_multiple_of(4);
    let apart = Span::of(&call.y, y_bytes).apart(&Span::of(&call.x, x_bytes));
    if !(sized && aligned && apart) || k < 32 || !k.is_multiple_of(32) || !(1..=8).contains(&ncols)
    {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the q8_1 quantize of `call` on `stream`. Returns when the kernel is queued.
pub(in crate::kernels::cuda) fn quantize_q8_1_launch(
    stream: &RawStream,
    call: &QuantizeRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_quantize(call)?;
    let args = QuantizeArgsRepr {
        x: call.x.address().cast_const(),
        y: call.y.address(),
        k: c_int::try_from(call.k).map_err(|_| invalid_argument())?,
        ncols: c_int::try_from(call.ncols).map_err(|_| invalid_argument())?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_quantize` proved the `x` and `y` ranges inside their live allocations,
    // large enough for the call, aligned, and disjoint; the adapter repeats the checks on the
    // addresses. The allocations stay valid until the queued launch finishes because
    // `inference_cuda_free` synchronizes the device first. `stream` is live for the call and
    // `args` is a plain value the entry reads.
    let status = unsafe { quantize_q8_1_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod tests {
    use super::super::super::STATUS_INVALID_ARGUMENT;
    use super::super::RawDevice;
    use super::*;

    /// The arguments of one launch as plain numbers; the addresses are made up and apart. Each
    /// case changes one thing so the entry refuses before it reaches the device, and a valid
    /// call is never made (the CPU runner has no driver).
    #[derive(Clone, Copy)]
    struct Args {
        x: usize,
        y: usize,
        k: c_int,
        ncols: c_int,
        stream: usize,
    }

    const GOOD: Args = Args {
        x: 0x10_0000,
        y: 0x30_0000,
        k: 2048,
        ncols: 2,
        stream: 0x1,
    };

    type Case = (&'static str, fn(&mut Args));

    fn pointer<T>(address: usize) -> *mut T {
        std::ptr::without_provenance_mut(address)
    }

    fn run(args: Args) -> c_int {
        let repr = QuantizeArgsRepr {
            x: pointer::<c_void>(args.x).cast_const(),
            y: pointer(args.y),
            k: args.k,
            ncols: args.ncols,
        };
        let mut stage = STAGE_NONE;
        // SAFETY: only called with arguments the entry refuses before it uses any pointer or
        // makes a CUDA call (see `GOOD` and the cases); `stage` is a live, writable c_int.
        unsafe { quantize_q8_1_launch_raw(&repr, pointer(args.stream), &mut stage) }
    }

    /// The ranges of a quantize call inside one unbacked allocation (numbers only): k 2080 pads to
    /// 2560, so two columns need 2 * 80 blocks of 36 bytes.
    #[derive(Clone, Copy)]
    struct Layout {
        x: (usize, usize),
        y: (usize, usize),
        k: usize,
        ncols: usize,
    }

    const X_BYTES: usize = 2 * 2080 * 4;
    const Y_BYTES: usize = 2 * 80 * 36;
    const VALID: Layout = Layout {
        x: (0, X_BYTES),
        y: (32_768, Y_BYTES),
        k: 2080,
        ncols: 2,
    };

    type LayoutCase = (&'static str, fn(&mut Layout));

    fn verdict(device: &RawDevice, layout: &Layout) -> Result<(), DeviceFailure> {
        let range = |(offset, bytes): (usize, usize)| {
            RawRange::new(device, offset, bytes).expect("a range inside the arena")
        };
        check_quantize(&QuantizeRaw {
            x: range(layout.x),
            y: range(layout.y),
            k: layout.k,
            ncols: layout.ncols,
        })
    }

    /// Host only (KUBDAN-MSG-64 items 6 and 10): the native check against the ranges of an unbacked
    /// allocation. The valid layout passes; each case changes one thing and is refused.
    #[test]
    fn check_quantize_refuses_short_unpadded_misaligned_and_overlapping_ranges() {
        let device = RawDevice::unbacked(1 << 20);
        assert_eq!(verdict(&device, &VALID), Ok(()));
        let cases: [LayoutCase; 14] = [
            ("x one float short", |l| l.x.1 = X_BYTES - 4),
            ("y one block short", |l| l.y.1 = Y_BYTES - 36),
            ("y unpadded: 65 blocks per column", |l| l.y.1 = 2 * 65 * 36),
            ("x misaligned by 2", |l| l.x.0 = 2),
            ("y misaligned by 2", |l| l.y.0 += 2),
            ("y over x", |l| l.y.0 = 8_192),
            ("y over the end of x", |l| l.y.0 = X_BYTES - 4),
            ("k 0", |l| l.k = 0),
            ("k 16, below one block", |l| l.k = 16),
            ("k 2090, off the 32 grid", |l| {
                l.k = 2090;
                l.x.1 = 2090 * 2 * 4;
            }),
            ("no columns", |l| l.ncols = 0),
            ("nine columns", |l| {
                l.ncols = 9;
                l.x.1 = 9 * 2080 * 4;
                l.y = (131_072, 9 * 80 * 36);
            }),
            ("k * ncols overflows", |l| l.k = usize::MAX - 31),
            ("k * ncols * 4 overflows", |l| {
                l.k = (usize::MAX / 2) & !31;
                l.ncols = 1;
            }),
        ];
        for (name, edit) in cases {
            let mut layout = VALID;
            edit(&mut layout);
            assert_eq!(
                verdict(&device, &layout),
                Err(invalid_argument()),
                "quantize check: {name}"
            );
        }
    }

    /// Host only: every case is refused by the entry's own checks before any CUDA call.
    #[test]
    fn quantize_arguments_are_checked_before_any_launch() {
        let cases: [Case; 10] = [
            ("NULL x", |a| a.x = 0),
            ("NULL y", |a| a.y = 0),
            ("NULL stream", |a| a.stream = 0),
            ("k below one block", |a| a.k = 16),
            ("k not a multiple of 32", |a| a.k = 2048 + 4),
            ("no rows", |a| a.ncols = 0),
            ("nine rows", |a| a.ncols = 9),
            ("x offset not a multiple of 4", |a| a.x += 2),
            ("y offset not a multiple of 4", |a| a.y += 1),
            ("y over x", |a| a.y = a.x + 512),
        ];
        for (name, edit) in cases {
            let mut args = GOOD;
            edit(&mut args);
            assert_eq!(run(args), STATUS_INVALID_ARGUMENT, "quantize: {name}");
        }
    }
}
