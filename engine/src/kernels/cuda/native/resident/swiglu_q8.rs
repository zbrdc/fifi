//! Raw FFI to `inference_cuda_swiglu_q8_launch` of `native/resident/swiglu_q8.cu` (declared in
//! `native/resident/swiglu_q8.h`). Child of `native::resident`, so it inherits the
//! `allow(unsafe_code)` of `native.rs` (decision 0047, child modules of `native.rs`).
//!
//! Safety invariants this module relies on:
//! - A device range is proved in bounds by `RawRange::new` before it gets here. This module
//!   proves, again and on the sizes of the call, that `gu` holds the A2 footprint
//!   `(slots - 1) * y_slot_stride + 2 * n_ff` floats, that `q8` holds `slots` padded
//!   `block_q8_1` rows, that both are 4-byte aligned, and that `q8`, which the kernel writes,
//!   shares no byte with `gu`. The C entry repeats these checks on the addresses.
//! - The launch entry only queues one kernel on the stream it is given and returns the launch
//!   status; a fault in the kernel surfaces at the next synchronize.
//! - The allocations stay valid until the queued launch finishes because `inference_cuda_free`
//!   synchronizes the whole device before `cudaFree`.
//! - Every call runs under the process dispatch guard and on the thread that owns the context
//!   (decisions 0023 and 0024).
//! - No raw pointer and no unsafe function leaves this module.

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for RD-9a (decision 0047)")
)]

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

unsafe extern "C" {
    #[link_name = "inference_cuda_swiglu_q8_launch"]
    fn swiglu_q8_launch_raw(
        args: *const SwigluQ8ArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Mirror of `inference_cuda_swiglu_q8_args` (`swiglu_q8.h`), field for field. The layout is
/// pinned below and by the `static_assert` of the header.
#[repr(C)]
struct SwigluQ8ArgsRepr {
    gu: *const c_void,
    q8: *mut c_void,
    n_ff: c_int,
    y_slot_stride: c_int,
    slots: c_int,
}

const _: () = assert!(
    size_of::<SwigluQ8ArgsRepr>() == 32
        && offset_of!(SwigluQ8ArgsRepr, gu) == 0
        && offset_of!(SwigluQ8ArgsRepr, q8) == 8
        && offset_of!(SwigluQ8ArgsRepr, n_ff) == 16
        && offset_of!(SwigluQ8ArgsRepr, y_slot_stride) == 20
        && offset_of!(SwigluQ8ArgsRepr, slots) == 24
);

/// What one launch reads and writes: the strided `[slot][gate|up]` rows of `gu` and the
/// `slots` padded `block_q8_1` rows of `q8`.
pub(in crate::kernels::cuda) struct SwigluQ8Raw<'a> {
    pub(in crate::kernels::cuda) gu: RawRange<'a>,
    pub(in crate::kernels::cuda) q8: RawRange<'a>,
    pub(in crate::kernels::cuda) n_ff: usize,
    pub(in crate::kernels::cuda) y_slot_stride: usize,
    pub(in crate::kernels::cuda) slots: usize,
}

/// Bytes of the `block_q8_1` output of one row of `n_ff` values: padded to 512, one 36-byte
/// block per 32. The safe wrapper's `q8_1_bytes(n_ff, slots)` is the same rule; this copy keeps
/// the native child free of safe-tree imports.
fn q8_bytes_native(n_ff: usize, slots: usize) -> Option<usize> {
    let padded = n_ff.checked_add(511)? & !511;
    padded.checked_div(32)?.checked_mul(36)?.checked_mul(slots)
}

/// Proves a launch before the native call: `n_ff` a multiple of 32 in `32..=32768`, `slots` in
/// `1..=8`, `y_slot_stride >= 2 * n_ff`, `gu` the A2 footprint and `q8` the padded rows, both
/// 4-byte aligned, and `q8` disjoint from `gu`.
fn check_swiglu_q8(call: &SwigluQ8Raw<'_>) -> Result<(), DeviceFailure> {
    let slot_gaps = call.slots.checked_sub(1).ok_or_else(invalid_argument)?;
    let gu_floats = call
        .y_slot_stride
        .checked_mul(slot_gaps)
        .zip(call.n_ff.checked_mul(2))
        .and_then(|(gaps, both_halves)| gaps.checked_add(both_halves))
        .ok_or_else(invalid_argument)?;
    let gu_bytes = gu_floats.checked_mul(4).ok_or_else(invalid_argument)?;
    let q8_need = q8_bytes_native(call.n_ff, call.slots).ok_or_else(invalid_argument)?;
    let ints = [
        c_int::try_from(call.n_ff).ok(),
        c_int::try_from(call.y_slot_stride).ok(),
        c_int::try_from(call.slots).ok(),
    ];
    if ints.iter().any(|int| int.is_none())
        || call.gu.bytes < gu_bytes
        || call.q8.bytes < q8_need
        || !call.gu.offset.is_multiple_of(4)
        || !call.q8.offset.is_multiple_of(4)
        || !Span::of(&call.q8, q8_need).apart(&Span::of(&call.gu, gu_bytes))
        || call.n_ff < 32
        || call.n_ff > 32_768
        || !call.n_ff.is_multiple_of(32)
        || !(1..=8).contains(&call.slots)
        || call.y_slot_stride < 2 * call.n_ff
    {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the batched SwiGLU and q8_1 quantize of `call` on `stream`. Returns when the kernel is
/// queued.
pub(in crate::kernels::cuda) fn swiglu_q8_launch(
    stream: &RawStream,
    call: &SwigluQ8Raw<'_>,
) -> Result<(), DeviceFailure> {
    check_swiglu_q8(call)?;
    let args = SwigluQ8ArgsRepr {
        gu: call.gu.address().cast_const(),
        q8: call.q8.address(),
        n_ff: c_int::try_from(call.n_ff).map_err(|_| invalid_argument())?,
        y_slot_stride: c_int::try_from(call.y_slot_stride).map_err(|_| invalid_argument())?,
        slots: c_int::try_from(call.slots).map_err(|_| invalid_argument())?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_swiglu_q8` proved the `gu` and `q8` ranges inside their live allocations,
    // large enough for the call, aligned, and disjoint; the adapter repeats the checks on the
    // addresses. The allocations stay valid until the queued launch finishes because
    // `inference_cuda_free` synchronizes the device first. `stream` is live for the call and
    // `args` is a plain value the entry reads.
    let status = unsafe { swiglu_q8_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
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
        gu: usize,
        q8: usize,
        n_ff: c_int,
        y_slot_stride: c_int,
        slots: c_int,
        stream: usize,
    }

    /// Bytes the entry sizes for GOOD: 8 slots of stride 1024, so the A2 footprint of gu is
    /// (7 * 1024 + 2 * 512) floats.
    const GU: usize = (7 * 1024 + 2 * 512) * 4;

    const GOOD: Args = Args {
        gu: 0x10_0000,
        q8: 0x40_0000,
        n_ff: 512,
        y_slot_stride: 1024,
        slots: 8,
        stream: 0x1,
    };

    type Case = (&'static str, fn(&mut Args));

    fn pointer<T>(address: usize) -> *mut T {
        std::ptr::without_provenance_mut(address)
    }

    fn run(args: Args) -> c_int {
        let repr = SwigluQ8ArgsRepr {
            gu: pointer::<c_void>(args.gu).cast_const(),
            q8: pointer(args.q8),
            n_ff: args.n_ff,
            y_slot_stride: args.y_slot_stride,
            slots: args.slots,
        };
        let mut stage = STAGE_NONE;
        // SAFETY: only called with arguments the entry refuses before it uses any pointer or
        // makes a CUDA call (see `GOOD` and the cases); `stage` is a live, writable c_int.
        unsafe { swiglu_q8_launch_raw(&repr, pointer(args.stream), &mut stage) }
    }

    /// Host only: every case is refused by the entry's own checks before any CUDA call.
    #[test]
    fn swiglu_q8_arguments_are_checked_before_any_launch() {
        let cases: [Case; 15] = [
            ("NULL gu", |a| a.gu = 0),
            ("NULL q8", |a| a.q8 = 0),
            ("NULL stream", |a| a.stream = 0),
            ("n_ff below one block", |a| a.n_ff = 16),
            ("n_ff zero", |a| a.n_ff = 0),
            ("n_ff off the 32 grid", |a| a.n_ff = 512 + 16),
            ("n_ff above the width bound", |a| a.n_ff = 32_768 + 32),
            ("no slots", |a| a.slots = 0),
            ("nine slots", |a| a.slots = 9),
            ("stride below 2*n_ff", |a| a.y_slot_stride = 1023),
            ("gu offset not a multiple of 4", |a| a.gu += 2),
            ("q8 offset not a multiple of 4", |a| a.q8 += 1),
            ("q8 over gu", |a| a.q8 = a.gu + 512),
            ("q8 over the end of gu", |a| a.q8 = a.gu + GU - 4),
            ("negative stride", |a| a.y_slot_stride = -1024),
        ];
        for (name, edit) in cases {
            let mut args = GOOD;
            edit(&mut args);
            assert_eq!(run(args), STATUS_INVALID_ARGUMENT, "swiglu_q8: {name}");
        }
    }

    /// The ranges of one launch inside one unbacked allocation (numbers only): n_ff 896 (Mellum)
    /// pads to 1024, so each slot writes 32 blocks, not 28.
    #[derive(Clone, Copy)]
    struct Layout {
        gu: (usize, usize),
        q8: (usize, usize),
        n_ff: usize,
        y_slot_stride: usize,
        slots: usize,
    }

    const GU_N: usize = (7 * 2048 + 2 * 896) * 4;
    const Q8_N: usize = 8 * 32 * 36;
    const VALID: Layout = Layout {
        gu: (0, GU_N),
        q8: (1 << 19, Q8_N),
        n_ff: 896,
        y_slot_stride: 2048,
        slots: 8,
    };

    type LayoutCase = (&'static str, fn(&mut Layout));

    fn verdict(device: &RawDevice, layout: &Layout) -> Result<(), DeviceFailure> {
        let range = |(offset, bytes): (usize, usize)| {
            RawRange::new(device, offset, bytes).expect("a range inside the arena")
        };
        check_swiglu_q8(&SwigluQ8Raw {
            gu: range(layout.gu),
            q8: range(layout.q8),
            n_ff: layout.n_ff,
            y_slot_stride: layout.y_slot_stride,
            slots: layout.slots,
        })
    }

    /// Host only (KUBDAN-MSG-64 items 6 and 10): the native check against the ranges of an
    /// unbacked allocation. The valid layout passes (stride 2048 > 2 * 896); each case changes
    /// one thing and is refused, among them the tight A2 footprint of `gu` and the padded
    /// 32-block rows at the Mellum width.
    #[test]
    fn check_swiglu_q8_refuses_short_unpadded_misaligned_and_overlapping_ranges() {
        let device = RawDevice::unbacked(1 << 21);
        assert_eq!(verdict(&device, &VALID), Ok(()));
        let cases: [LayoutCase; 14] = [
            ("gu one float short of the footprint", |l| {
                l.gu.1 = GU_N - 4;
            }),
            ("gu packed to slots * 2*n_ff", |l| l.gu.1 = 8 * 2 * 896 * 4),
            ("q8 one block short", |l| l.q8.1 = Q8_N - 36),
            ("q8 unpadded: 28 blocks per slot", |l| l.q8.1 = 8 * 28 * 36),
            ("gu misaligned by 2", |l| l.gu.0 = 2),
            ("q8 misaligned by 1", |l| l.q8.0 += 1),
            ("q8 over gu", |l| l.q8.0 = 4_096),
            ("q8 over the end of gu", |l| l.q8.0 = GU_N - 4),
            // n_ff 912 needs a gu of (7 * 2048 + 2 * 912) * 4 = 64 640 bytes: the footprint of
            // the OFF-GRID width itself, so the fit passes and only the % 32 guard refuses
            // (with VALID's 64 512 bytes the gu-short guard fired first). n_ff 0 is refused by
            // n_ff < 32 ALONE (0 is on the 32 grid, its footprint 57 344 bytes fits, q8 needs
            // 0), and n_ff 32 800 (stride 65 600 = 2 * n_ff, one slot, the matching 262 400
            // and 37 440 byte ranges) by the width bound alone.
            ("n_ff off the 32 grid, its own footprint", |l| {
                l.n_ff = 896 + 16;
                l.gu.1 = (7 * 2048 + 2 * 912) * 4;
            }),
            ("n_ff zero", |l| l.n_ff = 0),
            ("n_ff 32 800 at the width bound", |l| {
                l.n_ff = 32_800;
                l.y_slot_stride = 65_600;
                l.slots = 1;
                l.gu.1 = 262_400;
                l.q8.1 = 37_440;
            }),
            ("stride below 2*n_ff", |l| l.y_slot_stride = 2 * 896 - 32),
            ("no slots", |l| l.slots = 0),
            ("nine slots", |l| {
                l.slots = 9;
                l.gu.1 = (8 * 2048 + 2 * 896) * 4;
                l.q8.1 = 9 * 32 * 36;
            }),
        ];
        for (name, edit) in cases {
            let mut layout = VALID;
            edit(&mut layout);
            assert_eq!(
                verdict(&device, &layout),
                Err(invalid_argument()),
                "swiglu_q8 check: {name}"
            );
        }
    }
}
