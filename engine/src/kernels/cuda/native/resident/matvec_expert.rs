//! Raw FFI to `inference_cuda_matvec_expert_launch` of `native/resident/matvec_expert.cu`
//! (declared in `native/resident/matvec_expert.h`). Child of `native::resident`, so it inherits
//! the `allow(unsafe_code)` of `native.rs` (decision 0047, child modules of `native.rs`).
//!
//! Safety invariants this module relies on:
//! - A device range is proved in bounds by `RawRange::new` before it gets here. This module
//!   proves, again and on the sizes of the call, that `w_stack` holds `n_expert` experts, that
//!   `ids` holds `n_used` ints, that `x_q8` holds the padded rows the `x_per_slot` mode reads,
//!   that `y` holds `(n_used - 1) * y_slot_stride + rows` floats, that every range is 4-byte
//!   aligned, and that `y`, which the kernel writes, shares no byte with `w_stack`, `x_q8` or
//!   `ids`. The C entry repeats these checks on the addresses.
//! - The ids are read by the kernel when it runs; an id outside `[0, n_expert)` writes the quiet
//!   NaN to that slot's rows and reads no weight (the V-MV1 A1 guard).
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

/// Mirror of `inference_cuda_matvec_expert_args` (`matvec_expert.h`), field for field.
/// The layout is pinned below and by the `static_assert` of the header.
#[repr(C)]
struct ExpertArgsRepr {
    ty: c_int,
    w_stack: *const c_void,
    ids: *const c_void,
    x_q8: *const c_void,
    x_per_slot: c_int,
    y: *mut f32,
    y_slot_stride: c_int,
    k: c_int,
    rows: c_int,
    n_expert: c_int,
    n_used: c_int,
}

const _: () = assert!(
    size_of::<ExpertArgsRepr>() == 72
        && offset_of!(ExpertArgsRepr, ty) == 0
        && offset_of!(ExpertArgsRepr, w_stack) == 8
        && offset_of!(ExpertArgsRepr, ids) == 16
        && offset_of!(ExpertArgsRepr, x_q8) == 24
        && offset_of!(ExpertArgsRepr, x_per_slot) == 32
        && offset_of!(ExpertArgsRepr, y) == 40
        && offset_of!(ExpertArgsRepr, y_slot_stride) == 48
        && offset_of!(ExpertArgsRepr, k) == 52
        && offset_of!(ExpertArgsRepr, rows) == 56
        && offset_of!(ExpertArgsRepr, n_expert) == 60
        && offset_of!(ExpertArgsRepr, n_used) == 64
);

unsafe extern "C" {
    #[link_name = "inference_cuda_matvec_expert_launch"]
    fn matvec_expert_launch_raw(
        args: *const ExpertArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// What one expert matvec reads and writes.
pub(in crate::kernels::cuda) struct ExpertMatvecRaw<'a> {
    pub(in crate::kernels::cuda) ty: QuantType,
    pub(in crate::kernels::cuda) w_stack: RawRange<'a>,
    pub(in crate::kernels::cuda) ids: RawRange<'a>,
    pub(in crate::kernels::cuda) x_q8: RawRange<'a>,
    pub(in crate::kernels::cuda) x_per_slot: bool,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) y_slot_stride: usize,
    pub(in crate::kernels::cuda) k: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) n_expert: usize,
    pub(in crate::kernels::cuda) n_used: usize,
}

/// Bytes each range of a call must hold.
struct Sizes {
    w_stack: usize,
    ids: usize,
    x_q8: usize,
    y: usize,
}

fn call_sizes(call: &ExpertMatvecRaw<'_>) -> Option<Sizes> {
    let blocks_y = (call.k.checked_add(511)? & !511) / 32;
    let x_rows = if call.x_per_slot { call.n_used } else { 1 };
    let y_floats = call
        .y_slot_stride
        .checked_mul(call.n_used.checked_sub(1)?)?
        .checked_add(call.rows)?;
    Some(Sizes {
        w_stack: call
            .ty
            .weight_bytes(call.rows.checked_mul(call.n_expert)?, call.k)?,
        ids: call.n_used.checked_mul(4)?,
        x_q8: blocks_y.checked_mul(36)?.checked_mul(x_rows)?,
        y: y_floats.checked_mul(4)?,
    })
}

/// Proves an expert matvec before the native call: the `matvec_q` shape rules, `n_used` in
/// `1..=8` and at most `n_expert`, `y_slot_stride >= rows`, every range sized and 4-byte
/// aligned, and `y` disjoint from `w_stack`, `x_q8` and `ids`.
fn check_expert(call: &ExpertMatvecRaw<'_>) -> Result<(), DeviceFailure> {
    let need = call_sizes(call).ok_or_else(invalid_argument)?;
    let ints = [
        c_int::try_from(call.k).ok(),
        c_int::try_from(call.rows).ok(),
        c_int::try_from(call.n_expert).ok(),
        c_int::try_from(call.n_used).ok(),
        c_int::try_from(call.y_slot_stride).ok(),
    ];
    let sized = call.w_stack.bytes >= need.w_stack
        && call.ids.bytes >= need.ids
        && call.x_q8.bytes >= need.x_q8
        && call.y.bytes >= need.y;
    let aligned = call.w_stack.offset.is_multiple_of(4)
        && call.ids.offset.is_multiple_of(4)
        && call.x_q8.offset.is_multiple_of(4)
        && call.y.offset.is_multiple_of(4);
    let y = Span::of(&call.y, need.y);
    let apart = y.apart(&Span::of(&call.w_stack, need.w_stack))
        && y.apart(&Span::of(&call.x_q8, need.x_q8))
        && y.apart(&Span::of(&call.ids, need.ids));
    if ints.iter().any(|int| int.is_none())
        || !sized
        || !aligned
        || !apart
        || call.k == 0
        || call.rows < 1
        || !call.rows.is_multiple_of(ROWS_MULTIPLE)
        || call.n_expert < 1
        || !(1..=8).contains(&call.n_used)
        || call.n_used > call.n_expert
        || call.y_slot_stride < call.rows
        || !call.k.is_multiple_of(call.ty.block_size())
    {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the expert matvec of `call` on `stream`. Returns when the kernel is queued.
pub(in crate::kernels::cuda) fn matvec_expert_launch(
    stream: &RawStream,
    call: &ExpertMatvecRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_expert(call)?;
    let args = ExpertArgsRepr {
        ty: call.ty.ggml(),
        w_stack: call.w_stack.address().cast_const(),
        ids: call.ids.address().cast_const(),
        x_q8: call.x_q8.address().cast_const(),
        x_per_slot: i32::from(call.x_per_slot),
        y: call.y.address().cast::<f32>(),
        y_slot_stride: c_int::try_from(call.y_slot_stride).map_err(|_| invalid_argument())?,
        k: c_int::try_from(call.k).map_err(|_| invalid_argument())?,
        rows: c_int::try_from(call.rows).map_err(|_| invalid_argument())?,
        n_expert: c_int::try_from(call.n_expert).map_err(|_| invalid_argument())?,
        n_used: c_int::try_from(call.n_used).map_err(|_| invalid_argument())?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_expert` proved every range inside its live allocation, large enough,
    // aligned, and `y` disjoint from the ranges the kernel reads; the adapter repeats the checks
    // on the addresses. The allocations stay valid until the queued launch finishes because
    // `inference_cuda_free` synchronizes the device first. `stream` is live for the call.
    let status = unsafe { matvec_expert_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod tests {
    use super::super::super::STATUS_INVALID_ARGUMENT;
    use super::super::RawDevice;
    use super::*;

    /// The ranges of an expert matvec inside one unbacked allocation (numbers only): Q4_K, k 768
    /// (three blocks, padded to 1024), 16 experts of 64 rows, 8 used slots, a y stride of 128 (the
    /// A2 [slot][gate|up] layout) and one shared activation row.
    #[derive(Clone, Copy)]
    struct Layout {
        w_stack: (usize, usize),
        ids: (usize, usize),
        x_q8: (usize, usize),
        y: (usize, usize),
        k: usize,
        rows: usize,
        n_expert: usize,
        n_used: usize,
        x_per_slot: bool,
        y_slot_stride: usize,
    }

    const STACK_BYTES: usize = 16 * 64 * 3 * 144;
    const X_ROW_BYTES: usize = 32 * 36;
    const Y_FOOTPRINT: usize = (7 * 128 + 64) * 4;
    const VALID: Layout = Layout {
        w_stack: (0, STACK_BYTES),
        ids: (1 << 20, IDS_BYTES),
        x_q8: (2 << 20, X_ROW_BYTES),
        y: (3 << 20, Y_FOOTPRINT),
        k: 768,
        rows: 64,
        n_expert: 16,
        n_used: 8,
        x_per_slot: false,
        y_slot_stride: 128,
    };

    type LayoutCase = (&'static str, fn(&mut Layout));

    fn verdict(device: &RawDevice, layout: &Layout) -> Result<(), DeviceFailure> {
        let range = |(offset, bytes): (usize, usize)| {
            RawRange::new(device, offset, bytes).expect("a range inside the arena")
        };
        check_expert(&ExpertMatvecRaw {
            ty: QuantType::Q4K,
            w_stack: range(layout.w_stack),
            ids: range(layout.ids),
            x_q8: range(layout.x_q8),
            x_per_slot: layout.x_per_slot,
            y: range(layout.y),
            y_slot_stride: layout.y_slot_stride,
            k: layout.k,
            rows: layout.rows,
            n_expert: layout.n_expert,
            n_used: layout.n_used,
        })
    }

    /// Host only (KUBDAN-MSG-64 items 6 and 10): the native check against the ranges of an unbacked
    /// allocation. The valid layout passes; each case changes one thing and is refused, among them
    /// the A2 footprint of `y` and the checked `rows * n_expert`.
    #[test]
    fn check_expert_refuses_short_unpadded_misaligned_and_overlapping_ranges() {
        let device = RawDevice::unbacked(1 << 23);
        assert_eq!(verdict(&device, &VALID), Ok(()));
        let cases: [LayoutCase; 30] = [
            ("stack one block short", |l| l.w_stack.1 = STACK_BYTES - 144),
            ("ids one int short", |l| l.ids.1 = IDS_BYTES - 4),
            ("x one byte short", |l| l.x_q8.1 = X_ROW_BYTES - 1),
            ("x unpadded: 24 blocks", |l| l.x_q8.1 = 24 * 36),
            ("per-slot x with the shared size", |l| l.x_per_slot = true),
            ("y without the rows of the last slot (A2)", |l| {
                l.y.1 = 7 * 128 * 4;
            }),
            ("y one float short", |l| l.y.1 = Y_FOOTPRINT - 4),
            ("y packed to rows instead of the stride", |l| {
                l.y.1 = 8 * 64 * 4;
            }),
            ("stack misaligned by 2", |l| l.w_stack.0 = 2),
            ("ids misaligned by 2", |l| l.ids.0 += 2),
            ("x misaligned by 1", |l| l.x_q8.0 += 1),
            ("y misaligned by 2", |l| l.y.0 += 2),
            ("y over the stack", |l| l.y.0 = 4_096),
            ("y over the end of the stack", |l| l.y.0 = STACK_BYTES - 4),
            ("y over the ids", |l| l.y.0 = l.ids.0 + 16),
            ("y over the last id word", |l| l.y.0 = l.ids.0 + 28),
            ("y over x", |l| l.y.0 = l.x_q8.0 + 1_024),
            ("y over the seventh per-slot activation row", |l| {
                l.x_per_slot = true;
                l.x_q8.1 = 8 * X_ROW_BYTES;
                l.y.0 = l.x_q8.0 + 7 * X_ROW_BYTES;
            }),
            ("k 0", |l| l.k = 0),
            ("k 700, off the 256 grid", |l| l.k = 700),
            ("no rows", |l| l.rows = 0),
            ("rows 62, not a multiple of 4", |l| l.rows = 62),
            ("y stride below rows", |l| l.y_slot_stride = 63),
            ("no used slots", |l| l.n_used = 0),
            ("nine used slots", |l| {
                l.n_used = 9;
                l.ids.1 = 9 * 4;
                l.y.1 = (8 * 128 + 64) * 4;
            }),
            ("more used slots than experts", |l| l.n_expert = 4),
            ("rows * n_expert overflows (F10)", |l| {
                l.rows = (usize::MAX / 8) & !3;
                l.y_slot_stride = l.rows;
            }),
            ("rows * n_expert wraps to zero (1<<61 x 8)", |l| {
                l.rows = 1 << 61;
                l.n_expert = 8;
                l.n_used = 1;
                l.ids.1 = 4;
                l.y_slot_stride = 1 << 61;
            }),
            ("stack bytes overflow", |l| {
                l.rows = (usize::MAX / 288) & !3;
                l.y_slot_stride = l.rows;
            }),
            ("y stride overflows the footprint", |l| {
                l.y_slot_stride = usize::MAX;
            }),
        ];
        for (name, edit) in cases {
            let mut layout = VALID;
            edit(&mut layout);
            assert_eq!(
                verdict(&device, &layout),
                Err(invalid_argument()),
                "expert check: {name}"
            );
        }
    }

    /// The arguments of one launch as plain numbers; the addresses are made up and apart. Each
    /// case changes one thing so the entry refuses before it reaches the device.
    #[derive(Clone, Copy)]
    struct Args {
        ty: c_int,
        w_stack: usize,
        ids: usize,
        x_q8: usize,
        x_per_slot: c_int,
        y: usize,
        y_slot_stride: c_int,
        k: c_int,
        rows: c_int,
        n_expert: c_int,
        n_used: c_int,
        stream: usize,
    }

    /// GOOD is a call every check of the entry accepts, so each case below changes one thing and is
    /// refused for that thing only. Its four ranges are pairwise apart (see the sizes below); `y`
    /// used to lie inside the 9.4 MB stack range, which made every case fail the overlap check
    /// whatever the check under test did (KUBDAN-MSG-64 item 5).
    const GOOD: Args = Args {
        ty: 12,
        w_stack: 0x10_0000,
        ids: 0xB0_0000,
        x_q8: 0xC0_0000,
        x_per_slot: 0,
        y: 0x100_0000,
        y_slot_stride: 512,
        k: 2048,
        rows: 512,
        n_expert: 16,
        n_used: 8,
        stream: 0x1,
    };

    /// Bytes the entry sizes for GOOD: 16 experts of 512 rows of 8 Q4_K blocks of 144 bytes, 8 ids,
    /// one shared activation row of 64 q8_1 blocks of 36 bytes, and (8 - 1) * 512 + 512 floats of y.
    const W_BYTES: usize = 16 * 512 * 8 * 144;
    const IDS_BYTES: usize = 8 * 4;
    const X_BYTES: usize = 64 * 36;
    const Y_BYTES: usize = (7 * 512 + 512) * 4;

    type Case = (&'static str, fn(&mut Args));

    fn pointer<T>(address: usize) -> *mut T {
        std::ptr::without_provenance_mut(address)
    }

    fn run(args: Args) -> c_int {
        let repr = ExpertArgsRepr {
            ty: args.ty,
            w_stack: pointer::<c_void>(args.w_stack).cast_const(),
            ids: pointer::<c_void>(args.ids).cast_const(),
            x_q8: pointer::<c_void>(args.x_q8).cast_const(),
            x_per_slot: args.x_per_slot,
            y: pointer(args.y),
            y_slot_stride: args.y_slot_stride,
            k: args.k,
            rows: args.rows,
            n_expert: args.n_expert,
            n_used: args.n_used,
        };
        let mut stage = STAGE_NONE;
        // SAFETY: only called with arguments the entry refuses before it uses any pointer or
        // makes a CUDA call (see `GOOD` and the cases); `stage` is a live, writable c_int.
        unsafe { matvec_expert_launch_raw(&repr, pointer(args.stream), &mut stage) }
    }

    /// Host only: the expert rows of the argument table, refused before any CUDA call. Every case
    /// must answer `INVALID_ARGUMENT`; a case that is not refused reaches the CUDA runtime (status
    /// 35 on the CPU runner), so the failure message lists each such case with its status. The
    /// control mutants E1 (the four alignment checks), E2 (the y/x_q8 and y/w_stack overlap checks)
    /// and E3 (the `y_slot_stride < rows` check) each flip named cases of this table, and the
    /// stack-guard mutants S2b (both int64 guards), S2b1 (guard 1 only) and S2b2 (guard 2 only)
    /// each flip the two wrap-to-zero rows.
    #[test]
    fn matvec_expert_arguments_are_checked_before_any_launch() {
        let ranges = [
            ("w_stack", GOOD.w_stack, W_BYTES),
            ("ids", GOOD.ids, IDS_BYTES),
            ("x_q8", GOOD.x_q8, X_BYTES),
            ("y", GOOD.y, Y_BYTES),
        ];
        for (i, (name, start, bytes)) in ranges.iter().enumerate() {
            assert_eq!(start % 4, 0, "GOOD {name} is aligned");
            for (other, other_start, other_bytes) in &ranges[i + 1..] {
                assert!(
                    start + bytes <= *other_start || other_start + other_bytes <= *start,
                    "GOOD {name} overlaps {other}"
                );
            }
        }
        let cases: [Case; 33] = [
            ("NULL w_stack", |a| a.w_stack = 0),
            ("NULL ids", |a| a.ids = 0),
            ("NULL x_q8", |a| a.x_q8 = 0),
            ("NULL y", |a| a.y = 0),
            ("NULL stream", |a| a.stream = 0),
            ("unknown type", |a| a.ty = 99),
            ("Q6_K k off the 256 grid", |a| {
                a.ty = 14;
                a.k = 2048 - 64;
            }),
            ("k 0", |a| a.k = 0),
            ("negative k", |a| a.k = -2048),
            ("no rows", |a| a.rows = 0),
            ("rows 510 not a multiple of 4", |a| a.rows = 510),
            ("rows 3 not a multiple of 4", |a| a.rows = 3),
            ("no experts", |a| a.n_expert = 0),
            ("no used slots", |a| a.n_used = 0),
            ("nine used slots", |a| a.n_used = 9),
            ("n_used over n_expert", |a| {
                a.n_used = 8;
                a.n_expert = 4;
            }),
            ("y_slot_stride below rows", |a| a.y_slot_stride = 511),
            ("x_per_slot not 0 or 1", |a| a.x_per_slot = 2),
            ("w_stack offset not a multiple of 4", |a| a.w_stack += 2),
            ("ids offset not a multiple of 4", |a| a.ids += 2),
            ("x_q8 offset not a multiple of 4", |a| a.x_q8 += 1),
            ("y offset not a multiple of 4", |a| a.y += 2),
            ("y over the ids", |a| a.y = a.ids),
            ("y over x_q8", |a| a.y = a.x_q8 + 1024),
            ("y over w_stack", |a| a.y = a.w_stack + 0x1000),
            ("y one float inside the end of w_stack", |a| {
                a.y = a.w_stack + W_BYTES - 4;
            }),
            ("stack bytes overflow", |a| {
                a.n_expert = c_int::MAX;
                a.rows = 1 << 26;
                a.y_slot_stride = c_int::MAX;
            }),
            // S2b (lane C's S2 review): the two stack-byte overflow guards, reached with every
            // EARLIER check satisfied — the "stack bytes overflow" case above is also caught by
            // the apart guard once the guards are deleted (its n_expert x c_int::MAX stride y
            // footprint ~60 GB and the wrapped stack bytes ~1.8e19 both overlap), so it kills
            // nothing. Both cases below wrap the product to EXACTLY zero without the guards
            // (2^30 x 2^30 x 144 = 9 x 2^64, and 2^52 x 2^20 = 2^72), keep y at 1 << 40 so its
            // 2^35-byte footprint overlaps nothing, and pass rows % 4, k % 256, n_used <= 8,
            // the alignments and the c_int ranges.
            ("stack bytes wrap to zero (guard 2)", |a| {
                a.k = 256;
                a.rows = 1 << 30;
                a.n_expert = 1 << 30;
                a.y_slot_stride = 1 << 30;
                a.y = 1 << 40;
            }),
            ("expert blocks over INT64_MAX/n_expert (guard 1)", |a| {
                a.k = 1 << 30;
                a.rows = 1 << 30;
                a.n_expert = 1 << 20;
                a.y_slot_stride = 1 << 30;
                a.y = 1 << 40;
            }),
            ("y over the last id word", |a| a.y = a.ids + 28),
            ("per-slot x, y over the seventh activation row", |a| {
                a.x_per_slot = 1;
                a.y = a.x_q8 + 7 * X_BYTES;
            }),
            ("Q8_0 k off the 32 grid", |a| {
                a.ty = 8;
                a.k = 2048 - 16;
            }),
            ("Q5_0 k off the 32 grid", |a| {
                a.ty = 6;
                a.k = 2048 + 8;
            }),
        ];
        let mut not_refused = Vec::new();
        for (name, edit) in cases {
            let mut args = GOOD;
            edit(&mut args);
            let status = run(args);
            if status != STATUS_INVALID_ARGUMENT {
                not_refused.push((name, status));
            }
        }
        assert!(
            not_refused.is_empty(),
            "expert: cases not refused with INVALID_ARGUMENT: {not_refused:?}"
        );
    }

    /// S5c (lane C's K2 verdict): no `check_expert` row can tell a wrapping `call_sizes` from
    /// the checked one — the c_int ranges refuse the huge rows before the sizes are computed —
    /// so the wrap is pinned directly. rows 1<<61 x 8 experts wraps to 2^64 == 0 in the
    /// stack-rows product while every other count of this call stays inside usize (y:
    /// 0 * stride + 1<<61 floats, one shared activation row, 16 padded blocks), so a wrapping
    /// form would answer `Some` and this assert fails.
    #[test]
    fn call_sizes_refuses_a_rows_n_expert_that_wraps_to_zero() {
        let device = RawDevice::unbacked(64);
        let range = || RawRange::new(&device, 0, 64).expect("a range inside the arena");
        assert!(
            call_sizes(&ExpertMatvecRaw {
                ty: QuantType::Q4K,
                w_stack: range(),
                ids: range(),
                x_q8: range(),
                x_per_slot: false,
                y: range(),
                y_slot_stride: 1 << 61,
                k: 256,
                rows: 1 << 61,
                n_expert: 8,
                n_used: 1,
            })
            .is_none()
        );
    }
}
