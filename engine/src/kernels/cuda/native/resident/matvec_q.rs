//! Raw FFI to `inference_cuda_matvec_q_launch` of `native/resident/matvec_q.cu` (declared in
//! `native/resident/matvec_q.h`). Child of `native::resident`, so it inherits the
//! `allow(unsafe_code)` of `native.rs` (decision 0047, child modules of `native.rs`).
//!
//! Safety invariants this module relies on:
//! - A device range is proved in bounds by `RawRange::new` before it gets here. This module
//!   proves, again and on the sizes of the call, that `w` holds the native blocks of `rows` rows
//!   of `k` values, that `x_q8` holds the padded blocks of `ncols` rows, that `y` holds
//!   `ncols * rows` floats, that every range is 4-byte aligned, and that `y`, which the kernel
//!   writes, shares no byte with `w` or `x_q8`. The C entry repeats these checks on the
//!   addresses.
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
use crate::kernels::cuda::resident::matvec_q::{QuantType, ROWS_MULTIPLE};

/// Mirror of `inference_cuda_matvec_q_args` (`matvec_q.h`), field for field.
/// The layout is pinned below and by the `static_assert` of the header.
#[repr(C)]
struct MatvecArgsRepr {
    ty: c_int,
    w: *const c_void,
    x_q8: *const c_void,
    y: *mut f32,
    k: c_int,
    rows: c_int,
    ncols: c_int,
}

const _: () = assert!(
    size_of::<MatvecArgsRepr>() == 48
        && offset_of!(MatvecArgsRepr, ty) == 0
        && offset_of!(MatvecArgsRepr, w) == 8
        && offset_of!(MatvecArgsRepr, x_q8) == 16
        && offset_of!(MatvecArgsRepr, y) == 24
        && offset_of!(MatvecArgsRepr, k) == 32
        && offset_of!(MatvecArgsRepr, rows) == 36
        && offset_of!(MatvecArgsRepr, ncols) == 40
);

unsafe extern "C" {
    #[link_name = "inference_cuda_matvec_q_launch"]
    fn matvec_q_launch_raw(
        args: *const MatvecArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// What one dense matvec reads and writes.
pub(in crate::kernels::cuda) struct MatvecRaw<'a> {
    pub(in crate::kernels::cuda) ty: QuantType,
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) x_q8: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) ncols: usize,
}

/// Bytes each range of a call must hold.
struct Sizes {
    w: usize,
    x_q8: usize,
    y: usize,
}

fn call_sizes(ty: QuantType, k: usize, rows: usize, ncols: usize) -> Option<Sizes> {
    let blocks_y = (k.checked_add(511)? & !511) / 32;
    Some(Sizes {
        w: ty.weight_bytes(rows, k)?,
        x_q8: blocks_y.checked_mul(36)?.checked_mul(ncols)?,
        y: rows.checked_mul(ncols)?.checked_mul(4)?,
    })
}

/// Proves a dense matvec before the native call: type, `k` a multiple of the block size, `rows` a
/// positive multiple of 4, `ncols` in `1..=8`, every range sized and 4-byte aligned, and `y`
/// disjoint from `w` and `x_q8`.
pub(in crate::kernels::cuda) fn check_matvec(call: &MatvecRaw<'_>) -> Result<(), DeviceFailure> {
    let (k, rows, ncols) = (call.k, call.rows, call.ncols);
    let need = call_sizes(call.ty, k, rows, ncols).ok_or_else(invalid_argument)?;
    let ints = [
        c_int::try_from(k).ok(),
        c_int::try_from(rows).ok(),
        c_int::try_from(ncols).ok(),
    ];
    let sized = call.w.bytes >= need.w && call.x_q8.bytes >= need.x_q8 && call.y.bytes >= need.y;
    let aligned = call.w.offset.is_multiple_of(4)
        && call.x_q8.offset.is_multiple_of(4)
        && call.y.offset.is_multiple_of(4);
    let y = Span::of(&call.y, need.y);
    let apart = y.apart(&Span::of(&call.w, need.w)) && y.apart(&Span::of(&call.x_q8, need.x_q8));
    if ints.iter().any(|int| int.is_none())
        || !sized
        || !aligned
        || !apart
        || k == 0
        || rows < 1
        || !rows.is_multiple_of(ROWS_MULTIPLE)
        || !(1..=8).contains(&ncols)
        || !k.is_multiple_of(call.ty.block_size())
    {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the dense matvec of `call` on `stream`. Returns when the kernel is queued.
pub(in crate::kernels::cuda) fn matvec_q_launch(
    stream: &RawStream,
    call: &MatvecRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_matvec(call)?;
    let mut stage = STAGE_NONE;
    // SAFETY: `check_matvec` proved the `w`, `x_q8` and `y` ranges inside their live
    // allocations, large enough for the call, aligned, and `y` disjoint from the ranges the
    // kernel reads; the adapter repeats the checks on the addresses. The allocations stay valid
    // until the queued launch finishes because `inference_cuda_free` synchronizes the device
    // first. `stream` is live for the call.
    let args = MatvecArgsRepr {
        ty: call.ty.ggml(),
        w: call.w.address().cast_const(),
        x_q8: call.x_q8.address().cast_const(),
        y: call.y.address().cast::<f32>(),
        k: c_int::try_from(call.k).map_err(|_| invalid_argument())?,
        rows: c_int::try_from(call.rows).map_err(|_| invalid_argument())?,
        ncols: c_int::try_from(call.ncols).map_err(|_| invalid_argument())?,
    };
    let status = unsafe { matvec_q_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Test support of the seam-1 microbench: a raw timing event recorded on a stream. The handle
/// is opaque to Rust and consumed by [`event_elapsed_ms_raw`] and [`event_destroy_raw`].
#[cfg(test)]
pub(in crate::kernels::cuda) fn event_record_raw(
    stream: &RawStream,
) -> Result<*mut c_void, DeviceFailure> {
    let mut event: *mut c_void = std::ptr::null_mut();
    let mut stage = STAGE_NONE;
    // SAFETY: `event` is a live, writable slot for the handle and `stream` is live for the call;
    // the entry creates one timing event and records it, nothing else.
    let status = unsafe { event_record_create_raw(stream.0.as_ptr(), &mut event, &mut stage) };
    checked(status, stage)?;
    Ok(event)
}

/// Test support: the milliseconds between two events of [`event_record_raw`].
#[cfg(test)]
pub(in crate::kernels::cuda) fn event_elapsed_ms_raw(
    start: *mut c_void,
    end: *mut c_void,
) -> Result<f32, DeviceFailure> {
    let mut ms: f32 = 0.0;
    let mut stage = STAGE_NONE;
    // SAFETY: both handles were created by `event_record_raw` and not destroyed yet; `ms` is a
    // live, writable f32.
    let status = unsafe { event_elapsed_ms_raw_decl(start, end, &mut ms, &mut stage) };
    checked(status, stage)?;
    Ok(ms)
}

/// Test support: destroys one event of [`event_record_raw`].
#[cfg(test)]
pub(in crate::kernels::cuda) fn event_destroy_raw(event: *mut c_void) -> Result<(), DeviceFailure> {
    let mut stage = STAGE_NONE;
    // SAFETY: the handle was created by `event_record_raw` and is consumed once here.
    let status = unsafe { event_destroy_test_raw(event, &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
unsafe extern "C" {
    #[link_name = "inference_cuda_event_record_create"]
    fn event_record_create_raw(
        stream: *mut c_void,
        out_event: *mut *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_event_elapsed_ms"]
    fn event_elapsed_ms_raw_decl(
        start: *mut c_void,
        end: *mut c_void,
        out_ms: *mut f32,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_event_destroy_test"]
    fn event_destroy_test_raw(event: *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_mem_info"]
    fn mem_info_raw(out_free: *mut usize, out_total: *mut usize, out_stage: *mut c_int) -> c_int;
}

/// Test support: free device memory in bytes, or `None` when the query fails.
#[cfg(test)]
pub(in crate::kernels::cuda) fn free_bytes_raw() -> Option<usize> {
    let (mut free, mut total) = (0usize, 0usize);
    let mut stage = STAGE_NONE;
    // SAFETY: both slots are live, writable values for the call; the entry only writes them on
    // success.
    let status = unsafe { mem_info_raw(&mut free, &mut total, &mut stage) };
    (status == 0).then_some(free)
}

#[cfg(test)]
mod tests {
    use super::super::super::STATUS_INVALID_ARGUMENT;
    use super::super::RawDevice;
    use super::*;
    use crate::kernels::cuda::resident::matvec_q::QuantType;

    /// The arguments of one launch as plain numbers; the addresses are made up and apart. Each
    /// case changes one thing so the entry refuses before it reaches the device.
    #[derive(Clone, Copy)]
    struct Args {
        ty: c_int,
        w: usize,
        x_q8: usize,
        y: usize,
        k: c_int,
        rows: c_int,
        ncols: c_int,
        stream: usize,
    }

    const GOOD: Args = Args {
        ty: 12,
        w: 0x10_0000,
        x_q8: 0x20_0000,
        y: 0x40_0000,
        k: 2048,
        rows: 8,
        ncols: 1,
        stream: 0x1,
    };

    type Case = (&'static str, fn(&mut Args));

    fn pointer<T>(address: usize) -> *mut T {
        std::ptr::without_provenance_mut(address)
    }

    fn run(args: Args) -> c_int {
        let repr = MatvecArgsRepr {
            ty: args.ty,
            w: pointer::<c_void>(args.w).cast_const(),
            x_q8: pointer::<c_void>(args.x_q8).cast_const(),
            y: pointer(args.y),
            k: args.k,
            rows: args.rows,
            ncols: args.ncols,
        };
        let mut stage = STAGE_NONE;
        // SAFETY: only called with arguments the entry refuses before it uses any pointer or
        // makes a CUDA call (see `GOOD` and the cases); `stage` is a live, writable c_int.
        unsafe { matvec_q_launch_raw(&repr, pointer(args.stream), &mut stage) }
    }

    unsafe extern "C" {
        #[link_name = "inference_cuda_mmvq_runnable_arch"]
        fn mmvq_runnable_arch_raw(cc: c_int, built: *const c_int, n_built: c_int) -> c_int;
    }

    /// The built architecture a device of `cc` runs on for the list `built` (0: none).
    fn runnable(cc: c_int, built: &[c_int]) -> c_int {
        let n_built = c_int::try_from(built.len()).expect("a short list");
        // SAFETY: `built` is a live slice of `n_built` ints; the entry only reads it.
        unsafe { mmvq_runnable_arch_raw(cc, built.as_ptr(), n_built) }
    }

    /// Host only (KUBDAN-MSG-64 item 1): the wrappers hand the vendored objects the built
    /// architecture the device runs on, not the device's own cc. An sm_89 GPU with a 72,86 build
    /// runs the sm_86 cubin; a Turing GPU does not run the sm_72 one; a device below every built
    /// architecture runs nothing.
    #[test]
    fn mmvq_arch_clamps_to_the_cubin_the_device_runs() {
        // (built list, device cc, architecture to use; 0 refuses)
        let table: [(&[c_int], c_int, c_int); 18] = [
            (&[720, 860], 890, 860),
            (&[720, 860], 860, 860),
            (&[720, 860], 870, 860),
            (&[720, 860], 720, 720),
            (&[720, 860], 750, 0),
            (&[720, 860], 710, 0),
            (&[720, 860], 800, 0),
            (&[720, 860], 900, 0),
            (&[720, 860], 0, 0),
            (&[720, 860], -10, 0),
            (&[860, 720], 890, 860),
            (&[860, 890], 890, 890),
            (&[860, 890], 860, 860),
            (&[860, 890], 800, 0),
            (&[860], 890, 860),
            (&[860], 720, 0),
            (&[720], 750, 0),
            (&[], 860, 0),
        ];
        for (built, cc, want) in table {
            assert_eq!(runnable(cc, built), want, "built {built:?}, device cc {cc}");
        }
        // SAFETY: a null list is refused before it is read.
        assert_eq!(
            unsafe { mmvq_runnable_arch_raw(860, std::ptr::null(), 2) },
            0
        );
    }

    /// The ranges of a dense matvec inside one unbacked allocation (numbers only): Q4_K, k 768
    /// (three blocks, padded to 1024 on the activation side), 64 rows, two columns.
    #[derive(Clone, Copy)]
    struct Layout {
        w: (usize, usize),
        x: (usize, usize),
        y: (usize, usize),
        k: usize,
        rows: usize,
        ncols: usize,
    }

    const W_BYTES: usize = 64 * 3 * 144;
    const X_BYTES: usize = 2 * 32 * 36;
    const Y_BYTES: usize = 64 * 2 * 4;
    const VALID: Layout = Layout {
        w: (0, W_BYTES),
        x: (32_768, X_BYTES),
        y: (65_536, Y_BYTES),
        k: 768,
        rows: 64,
        ncols: 2,
    };

    type LayoutCase = (&'static str, fn(&mut Layout));

    fn verdict(device: &RawDevice, layout: &Layout) -> Result<(), DeviceFailure> {
        let range = |(offset, bytes): (usize, usize)| {
            RawRange::new(device, offset, bytes).expect("a range inside the arena")
        };
        check_matvec(&MatvecRaw {
            ty: QuantType::Q4K,
            w: range(layout.w),
            x_q8: range(layout.x),
            y: range(layout.y),
            k: layout.k,
            rows: layout.rows,
            ncols: layout.ncols,
        })
    }

    /// Host only (KUBDAN-MSG-64 items 6 and 10): the native check against the ranges of an unbacked
    /// allocation. The valid layout passes; each case changes one thing and is refused.
    #[test]
    fn check_matvec_refuses_short_unpadded_misaligned_and_overlapping_ranges() {
        let device = RawDevice::unbacked(1 << 20);
        assert_eq!(verdict(&device, &VALID), Ok(()));
        let cases: [LayoutCase; 18] = [
            ("w one block short", |l| l.w.1 = W_BYTES - 144),
            ("x one byte short", |l| l.x.1 = X_BYTES - 1),
            ("x unpadded: 24 blocks per column", |l| l.x.1 = 2 * 24 * 36),
            ("y one float short", |l| l.y.1 = Y_BYTES - 4),
            ("w misaligned by 2", |l| l.w.0 = 2),
            ("x misaligned by 1", |l| l.x.0 += 1),
            ("y misaligned by 2", |l| l.y.0 += 2),
            ("y over w", |l| l.y.0 = 1_024),
            ("y over the end of w", |l| l.y.0 = W_BYTES - 4),
            ("y over x", |l| l.y.0 = l.x.0 + 1_024),
            ("y over the end of x", |l| l.y.0 = l.x.0 + X_BYTES - 4),
            ("k 0", |l| l.k = 0),
            ("k 700, off the 256 grid", |l| l.k = 700),
            ("no rows", |l| l.rows = 0),
            ("rows 62, not a multiple of 4", |l| l.rows = 62),
            ("no columns", |l| l.ncols = 0),
            ("nine columns", |l| {
                l.ncols = 9;
                l.x.1 = 9 * 32 * 36;
                l.y.1 = 64 * 9 * 4;
            }),
            ("weight bytes overflow", |l| l.rows = usize::MAX - 3),
        ];
        for (name, edit) in cases {
            let mut layout = VALID;
            edit(&mut layout);
            assert_eq!(
                verdict(&device, &layout),
                Err(invalid_argument()),
                "matvec check: {name}"
            );
        }
    }

    /// Host only: every case is refused by the entry's own checks before any CUDA call. The
    /// table covers the packet's rows: unknown type, k off the block grid, rows 0, ncols 0 and
    /// 9, misaligned pointers, overlaps, short stream.
    #[test]
    fn matvec_arguments_are_checked_before_any_launch() {
        let cases: [Case; 17] = [
            ("NULL w", |a| a.w = 0),
            ("NULL x_q8", |a| a.x_q8 = 0),
            ("NULL y", |a| a.y = 0),
            ("NULL stream", |a| a.stream = 0),
            ("unknown type", |a| a.ty = 99),
            ("Q4_K k off the 256 grid", |a| a.k = 2048 - 128),
            ("no rows", |a| a.rows = 0),
            ("rows 6 not a multiple of 4", |a| a.rows = 6),
            ("rows 3 not a multiple of 4", |a| a.rows = 3),
            ("no columns", |a| a.ncols = 0),
            ("nine columns", |a| a.ncols = 9),
            ("w offset not a multiple of 4", |a| a.w += 2),
            ("x_q8 offset not a multiple of 4", |a| a.x_q8 += 1),
            ("y offset not a multiple of 4", |a| a.y += 2),
            ("y over x_q8", |a| a.y = a.x_q8 + 1024),
            ("y over w", |a| a.y = a.w + 1024),
            ("negative k", |a| a.k = -2048),
        ];
        for (name, edit) in cases {
            let mut args = GOOD;
            edit(&mut args);
            assert_eq!(run(args), STATUS_INVALID_ARGUMENT, "matvec: {name}");
        }
    }
}
