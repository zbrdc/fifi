//! Raw FFI to `inference_cuda_add_rmsnorm_q8_launch` of `native/resident/add_rmsnorm_q8.cu`
//! (declared in `native/resident/add_rmsnorm_q8.h`). Child of `native::resident`, so it inherits
//! the `allow(unsafe_code)` of `native.rs` (decision 0047, child modules of `native.rs`).
//!
//! Safety invariants this module relies on:
//! - A device range is proved in bounds by `RawRange::new` before it gets here. This module
//!   proves, again and on the sizes of the call, that `h`, `delta`, `weight` and `norm_out` hold
//!   `n` floats, that `q8` holds the padded `block_q8_1` row (`pad(n, 512)/32 * 36` bytes), that
//!   the f32 ranges are 4-byte aligned (`norm_out` 16) and `q8` 4, and that the ranges the kernel
//!   writes (`h`, `norm_out`, `q8`) share no byte with any other range. The C entry repeats these
//!   checks on the addresses.
//! - The launch entry only queues one kernel on the stream it is given and returns the launch
//!   status; a fault in the kernel surfaces at the next synchronize.
//! - The allocations stay valid until the queued launch finishes because `inference_cuda_free`
//!   synchronizes the whole device before `cudaFree`.
//! - Every call runs under the process dispatch guard and on the thread that owns the context
//!   (decisions 0023 and 0024).
//! - No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

unsafe extern "C" {
    #[link_name = "inference_cuda_add_rmsnorm_q8_launch"]
    fn add_rmsnorm_q8_launch_raw(
        args: *const AddNormQ8ArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Mirror of `inference_cuda_add_rmsnorm_q8_args` (`add_rmsnorm_q8.h`), field for field.
/// The layout is pinned below and by the `static_assert` of the header.
#[repr(C)]
struct AddNormQ8ArgsRepr {
    h: *mut c_void,
    delta: *const c_void,
    w: *const c_void,
    norm_out: *mut c_void,
    q8: *mut c_void,
    n: c_int,
    eps: f32,
}

const _: () = assert!(
    size_of::<AddNormQ8ArgsRepr>() == 48
        && offset_of!(AddNormQ8ArgsRepr, h) == 0
        && offset_of!(AddNormQ8ArgsRepr, delta) == 8
        && offset_of!(AddNormQ8ArgsRepr, w) == 16
        && offset_of!(AddNormQ8ArgsRepr, norm_out) == 24
        && offset_of!(AddNormQ8ArgsRepr, q8) == 32
        && offset_of!(AddNormQ8ArgsRepr, n) == 40
        && offset_of!(AddNormQ8ArgsRepr, eps) == 44
);

/// What one launch reads and writes: the residual row `h` (updated in place when `delta` is
/// given), the weight, and the two outputs `norm_out` (f32 `[n]`) and `q8` (the padded
/// `block_q8_1` row).
pub(in crate::kernels::cuda) struct AddNormQ8Raw<'a> {
    pub(in crate::kernels::cuda) h: RawRange<'a>,
    pub(in crate::kernels::cuda) delta: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) weight: RawRange<'a>,
    pub(in crate::kernels::cuda) norm_out: RawRange<'a>,
    pub(in crate::kernels::cuda) q8: RawRange<'a>,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) eps: f32,
}

/// Bytes of the `block_q8_1` output of one row of `n` values: padded to 512 values, one 36-byte
/// block per 32. The safe wrapper's `q8_1_bytes(n, 1)` is the same rule; this copy keeps the
/// native child free of safe-tree imports.
fn q8_bytes_native(n: usize) -> Option<usize> {
    let padded = n.checked_add(511)? & !511;
    padded.checked_div(32)?.checked_mul(36)
}

/// Proves a launch before the native call: `n` a multiple of 256 in `256..=32768`, `eps` finite
/// and above 0, every f32 range `n` floats and 4-byte aligned (`norm_out` 16), `q8` the padded
/// row and 4-byte aligned, and the written extents (`h`, `norm_out`, `q8`) disjoint from every
/// other extent (`delta` and `weight` may share bytes with each other).
fn check_add_norm_q8(call: &AddNormQ8Raw<'_>) -> Result<(), DeviceFailure> {
    let n = call.n;
    let row = n.checked_mul(4).ok_or_else(invalid_argument)?;
    let q8_need = q8_bytes_native(n).ok_or_else(invalid_argument)?;
    let fits = |range: &RawRange<'_>, need: usize, align: usize| {
        range.bytes >= need && range.offset.is_multiple_of(align)
    };
    if !fits(&call.h, row, 4)
        || !fits(&call.weight, row, 4)
        || !fits(&call.norm_out, row, 16)
        || !fits(&call.q8, q8_need, 4)
        || call
            .delta
            .as_ref()
            .is_some_and(|delta| !fits(delta, row, 4))
        || !(256..=32_768).contains(&n)
        || !n.is_multiple_of(256)
        || !call.eps.is_finite()
        || call.eps <= 0.0
    {
        return Err(invalid_argument());
    }
    let written = [
        Span::of(&call.h, row),
        Span::of(&call.norm_out, row),
        Span::of(&call.q8, q8_need),
    ];
    let read = [
        call.delta.map(|delta| Span::of(&delta, row)),
        Some(Span::of(&call.weight, row)),
    ];
    for (i, span) in written.iter().enumerate() {
        let apart = written[i + 1..].iter().all(|other| span.apart(other))
            && read.iter().flatten().all(|other| span.apart(other));
        if !apart {
            return Err(invalid_argument());
        }
    }
    Ok(())
}

/// Queues the fused add, RMSNorm and q8_1 quantize of `call` on `stream`. Returns when the kernel
/// is queued.
pub(in crate::kernels::cuda) fn add_rmsnorm_q8_launch(
    stream: &RawStream,
    call: &AddNormQ8Raw<'_>,
) -> Result<(), DeviceFailure> {
    check_add_norm_q8(call)?;
    let args = AddNormQ8ArgsRepr {
        h: call.h.address(),
        delta: call
            .delta
            .as_ref()
            .map_or(std::ptr::null(), |r| r.address().cast_const()),
        w: call.weight.address().cast_const(),
        norm_out: call.norm_out.address(),
        q8: call.q8.address(),
        n: c_int::try_from(call.n).map_err(|_| invalid_argument())?,
        eps: call.eps,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_add_norm_q8` proved every range inside its live allocation, large enough,
    // aligned, and the written extents disjoint from every other extent; the adapter repeats the
    // checks on the addresses. The allocations stay valid until the queued launch finishes
    // because `inference_cuda_free` synchronizes the device first. `stream` is live for the call
    // and `args` is a plain value the entry reads.
    let status = unsafe { add_rmsnorm_q8_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
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
        h: usize,
        delta: usize,
        w: usize,
        norm_out: usize,
        q8: usize,
        n: c_int,
        eps: f32,
        stream: usize,
    }

    /// Bytes the entry sizes for GOOD: n 2048, so every f32 row is 8192 bytes.
    const ROW: usize = 2048 * 4;

    const GOOD: Args = Args {
        h: 0x10_0000,
        delta: 0x30_0000,
        w: 0x50_0000,
        norm_out: 0x70_0000,
        q8: 0x90_0000,
        n: 2048,
        eps: 1e-6,
        stream: 0x1,
    };

    type Case = (&'static str, fn(&mut Args));

    fn pointer<T>(address: usize) -> *mut T {
        std::ptr::without_provenance_mut(address)
    }

    fn run(args: Args) -> c_int {
        let repr = AddNormQ8ArgsRepr {
            h: pointer(args.h),
            delta: pointer::<c_void>(args.delta).cast_const(),
            w: pointer::<c_void>(args.w).cast_const(),
            norm_out: pointer(args.norm_out),
            q8: pointer(args.q8),
            n: args.n,
            eps: args.eps,
        };
        let mut stage = STAGE_NONE;
        // SAFETY: only called with arguments the entry refuses before it uses any pointer or
        // makes a CUDA call (see `GOOD` and the cases); `stage` is a live, writable c_int.
        unsafe { add_rmsnorm_q8_launch_raw(&repr, pointer(args.stream), &mut stage) }
    }

    /// Host only: every case is refused by the entry's own checks before any CUDA call.
    #[test]
    fn add_rmsnorm_q8_arguments_are_checked_before_any_launch() {
        let cases: [Case; 22] = [
            ("NULL h", |a| a.h = 0),
            ("NULL w", |a| a.w = 0),
            ("NULL norm_out", |a| a.norm_out = 0),
            ("NULL q8", |a| a.q8 = 0),
            ("NULL stream", |a| a.stream = 0),
            ("n below one unit", |a| a.n = 128),
            ("n off the 256 grid", |a| a.n = 2048 + 64),
            ("n above the width bound", |a| a.n = 32_768 + 256),
            ("eps zero", |a| a.eps = 0.0),
            ("eps negative", |a| a.eps = -1e-6),
            ("eps NaN", |a| a.eps = f32::NAN),
            ("h offset not a multiple of 4", |a| a.h += 2),
            ("delta offset not a multiple of 4", |a| a.delta += 2),
            ("w offset not a multiple of 4", |a| a.w += 2),
            ("norm_out offset not a multiple of 16", |a| a.norm_out += 8),
            ("q8 offset not a multiple of 4", |a| a.q8 += 2),
            ("q8 over the end of w", |a| a.q8 = a.w + ROW - 4),
            ("norm_out over delta", |a| a.norm_out = a.delta + 512),
            // The remaining apart pairs of the entry's nested loops, one case each: the three
            // written-written pairs (h/norm_out, h/q8, norm_out/q8) and h against the weight
            // (a written-read pair besides the two above).
            ("q8 over h", |a| a.q8 = a.h + 512),
            ("norm_out over h", |a| a.norm_out = a.h + 512),
            ("q8 over norm_out", |a| a.q8 = a.norm_out + 512),
            ("h over w", |a| a.w = a.h + 512),
        ];
        for (name, edit) in cases {
            let mut args = GOOD;
            edit(&mut args);
            assert_eq!(run(args), STATUS_INVALID_ARGUMENT, "add_rmsnorm_q8: {name}");
        }
    }

    /// The ranges of one launch inside one unbacked allocation (numbers only): n 2304 (Mellum)
    /// pads to 2560, so `q8` holds 80 blocks of 36 bytes, not 72.
    #[derive(Clone, Copy)]
    struct Layout {
        h: (usize, usize),
        delta: (usize, usize),
        w: (usize, usize),
        norm_out: (usize, usize),
        q8: (usize, usize),
        n: usize,
    }

    const N: usize = 2304;
    const ROW_N: usize = N * 4;
    const Q8_N: usize = 80 * 36;
    const VALID: Layout = Layout {
        h: (0, ROW_N),
        delta: (3 * ROW_N, ROW_N),
        w: (6 * ROW_N, ROW_N),
        norm_out: (9 * ROW_N, ROW_N),
        q8: (12 * ROW_N, Q8_N),
        n: N,
    };

    type LayoutCase = (&'static str, fn(&mut Layout));

    /// Resizes all five ranges for `n` on the VALID multipliers, so every FIT guard passes for
    /// the new n and only the n guard under test can refuse (the old n rows resized h alone, so
    /// the delta fit guard refused first and the n guards were never reached).
    fn resized(layout: &mut Layout, n: usize) {
        let row = n * 4;
        let q8 = q8_bytes_native(n).expect("the padded q8 row");
        layout.n = n;
        layout.h = (0, row);
        layout.delta = (3 * row, row);
        layout.w = (6 * row, row);
        layout.norm_out = (9 * row, row);
        layout.q8 = (12 * row, q8);
    }

    fn verdict(
        device: &RawDevice,
        layout: &Layout,
        delta: bool,
        eps: f32,
    ) -> Result<(), DeviceFailure> {
        let range = |(offset, bytes): (usize, usize)| {
            RawRange::new(device, offset, bytes).expect("a range inside the arena")
        };
        check_add_norm_q8(&AddNormQ8Raw {
            h: range(layout.h),
            delta: delta.then(|| range(layout.delta)),
            weight: range(layout.w),
            norm_out: range(layout.norm_out),
            q8: range(layout.q8),
            n: layout.n,
            eps,
        })
    }

    /// Host only (KUBDAN-MSG-64 items 6 and 10, lane C's K1 verdict): the native check against
    /// the ranges of an unbacked allocation, with and without `delta`. The VALID ranges sit
    /// 2 * ROW_N apart, so a case that moves or resizes one overlaps exactly the range it
    /// names. The valid layout passes; each row reaches ITS OWN guard with every earlier check
    /// satisfied: the five fit guards (short and misaligned, one row per range), the n guards
    /// (every range resized for the case's n), and the apart pairs incl. the written-written
    /// pairs h/norm_out and norm_out/q8; the eps guards run through `verdict`'s eps parameter.
    #[test]
    fn check_add_rmsnorm_q8_refuses_short_unpadded_misaligned_and_overlapping_ranges() {
        let device = RawDevice::unbacked(1 << 21);
        assert_eq!(verdict(&device, &VALID, true, 1e-6), Ok(()));
        assert_eq!(verdict(&device, &VALID, false, 1e-6), Ok(()));
        let cases: [LayoutCase; 20] = [
            ("h one float short", |l| l.h.1 = ROW_N - 4),
            ("delta one float short", |l| l.delta.1 = ROW_N - 4),
            ("weight one float short", |l| l.w.1 = ROW_N - 4),
            ("norm_out one float short", |l| l.norm_out.1 = ROW_N - 4),
            ("q8 one block short", |l| l.q8.1 = Q8_N - 36),
            ("q8 unpadded: 72 blocks", |l| l.q8.1 = 72 * 36),
            ("h misaligned by 2", |l| l.h.0 = 2),
            ("delta misaligned by 2", |l| l.delta.0 += 2),
            ("weight misaligned by 2", |l| l.w.0 += 2),
            ("norm_out misaligned by 8", |l| l.norm_out.0 += 8),
            ("q8 misaligned by 2", |l| l.q8.0 += 2),
            // n 0 keeps VALID's sizes: an empty range is not representable (RawRange::new
            // refuses one), and with a need of 0 every fit passes on the VALID bytes, so only
            // the width bound refuses.
            ("n zero", |l| l.n = 0),
            ("n below one unit", |l| resized(l, 128)),
            ("n above the width bound", |l| resized(l, 32_768 + 256)),
            ("n off the 256 grid", |l| resized(l, N + 128)),
            ("q8 over h", |l| l.q8.0 = 4_096),
            ("h over delta", |l| l.h.0 = 3 * ROW_N + 512),
            ("norm_out over w", |l| l.norm_out.0 = 6 * ROW_N + 512),
            ("norm_out over the end of h", |l| l.norm_out.0 = ROW_N - 16),
            ("norm_out over q8", |l| l.norm_out.0 = 12 * ROW_N - 16),
        ];
        for (name, edit) in cases {
            let mut layout = VALID;
            edit(&mut layout);
            assert_eq!(
                verdict(&device, &layout, true, 1e-6),
                Err(invalid_argument()),
                "add_rmsnorm_q8 check: {name}"
            );
        }
        // The eps guards of the check (finite and above 0), reached with the valid layout.
        assert_eq!(
            verdict(&device, &VALID, true, 0.0),
            Err(invalid_argument()),
            "add_rmsnorm_q8 check: eps zero"
        );
        assert_eq!(
            verdict(&device, &VALID, true, f32::NAN),
            Err(invalid_argument()),
            "add_rmsnorm_q8 check: eps NaN"
        );
    }
}
