//! Raw FFI to `inference_cuda_moe_combine_launch` of `native/resident/moe_combine.cu` (declared in
//! `native/resident/moe_combine.h`). Child module of `native::resident`, so it inherits the
//! `allow(unsafe_code)` of `native.rs` (decision 0047, child modules of `native.rs`).
//!
//! Design guidance only, no code copied. Ideas from MegaCapybara v1.35 static RE,
//! the MegaCapybara static-RE lessons, C6 and C8.
//!
//! Safety invariants this file relies on:
//! - The launch entry is called only through `CudaContext::launch_on`, on the thread that owns the
//!   context and under the dispatch guard (decisions 0023 and 0024).
//! - Every range is a `RawRange`, which `RawRange::new` proved inside its live allocation.
//! - Shape, sizes, alignment and disjointness are proved again here (`super::Span` for the
//!   overlaps), so a bad call is refused with `invalid_argument` and never reaches the device.
//!   The only range the kernel writes is `out`; it may be exactly `res` (in place) and shares no
//!   byte with any other input.
//! - The C struct is built in one `unsafe` block that cites those checks.
//! - Every allocation stays valid until queued work finishes, because `inference_cuda_free`
//!   synchronizes the whole device before `cudaFree`; the launch is asynchronous.
//! - The extern declaration, the `repr(C)` mirror and every raw pointer stay private to this file;
//!   nothing is re-exported.

use std::ffi::{c_int, c_void};
use std::mem::offset_of;
use std::ptr;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

const MAX_TOKENS: usize = 4096;
const MAX_N: usize = 65536;
const MAX_TOP_K: usize = 32;
const MAX_LD_DOWN: usize = 1 << 26;

/// Field for field the `inference_cuda_moe_combine_args` of `moe_combine.h`.
#[repr(C)]
struct CombineArgsRaw {
    res: *const c_void,
    down: *const c_void,
    weights: *const c_void,
    shared_gate: *const c_void,
    shared: *const c_void,
    out: *mut f32,
    m: c_int,
    n: c_int,
    top_k: c_int,
    ld_down: c_int,
}

// The `static_assert` of the header, repeated: a drift of either side stops the build.
const _: () = {
    assert!(size_of::<CombineArgsRaw>() == 64);
    assert!(offset_of!(CombineArgsRaw, res) == 0);
    assert!(offset_of!(CombineArgsRaw, down) == 8);
    assert!(offset_of!(CombineArgsRaw, weights) == 16);
    assert!(offset_of!(CombineArgsRaw, shared_gate) == 24);
    assert!(offset_of!(CombineArgsRaw, shared) == 32);
    assert!(offset_of!(CombineArgsRaw, out) == 40);
    assert!(offset_of!(CombineArgsRaw, m) == 48);
    assert!(offset_of!(CombineArgsRaw, n) == 52);
    assert!(offset_of!(CombineArgsRaw, top_k) == 56);
    assert!(offset_of!(CombineArgsRaw, ld_down) == 60);
};

unsafe extern "C" {
    #[link_name = "inference_cuda_moe_combine_launch"]
    fn moe_combine_launch_raw(
        args: *const CombineArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// What one combine launch reads and writes. `shared_gate` and `shared` are both given or both
/// absent. The in-place case is `out` at the address of `res`.
pub(in crate::kernels::cuda) struct CombineRaw<'a> {
    pub(in crate::kernels::cuda) res: RawRange<'a>,
    pub(in crate::kernels::cuda) down: RawRange<'a>,
    pub(in crate::kernels::cuda) weights: RawRange<'a>,
    pub(in crate::kernels::cuda) shared_gate: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) shared: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) m: c_int,
    pub(in crate::kernels::cuda) n: c_int,
    pub(in crate::kernels::cuda) top_k: c_int,
    pub(in crate::kernels::cuda) ld_down: c_int,
}

/// The bytes each range of a call must hold.
struct Need {
    rows: usize,
    down: usize,
    weights: usize,
    gate: usize,
}

/// The byte needs of the integers of `call`, or `None` outside the limits of `valid()` in the
/// `.cu` file or when a product overflows.
fn needs(call: &CombineRaw<'_>) -> Option<Need> {
    let [m, n, top_k, ld_down] =
        [call.m, call.n, call.top_k, call.ld_down].map(|value| usize::try_from(value).ok());
    let (m, n, top_k, ld_down) = (m?, n?, top_k?, ld_down?);
    let shape = (1..=MAX_TOKENS).contains(&m)
        && (4..=MAX_N).contains(&n)
        && n.is_multiple_of(4)
        && (1..=MAX_TOP_K).contains(&top_k)
        && (n..=MAX_LD_DOWN).contains(&ld_down)
        && ld_down.is_multiple_of(4);
    if !shape {
        return None;
    }
    let picks = m.checked_mul(top_k)?;
    let floats_down = (picks - 1).checked_mul(ld_down)?.checked_add(n)?;
    Some(Need {
        rows: m.checked_mul(n)?.checked_mul(4)?,
        down: floats_down.checked_mul(4)?,
        weights: picks.checked_mul(4)?,
        gate: m.checked_mul(4)?,
    })
}

/// True when `range` holds `need` bytes and starts at a multiple of `align`.
fn fits(range: &RawRange<'_>, need: usize, align: usize) -> bool {
    range.bytes >= need && range.offset.is_multiple_of(align)
}

/// True when the range is absent, or present and fitting.
fn fits_if_given(range: Option<&RawRange<'_>>, need: usize, align: usize) -> bool {
    range.is_none_or(|range| fits(range, need, align))
}

/// True when `out` shares no byte with an input, except that it may be exactly `res`.
fn out_apart(call: &CombineRaw<'_>, need: &Need) -> bool {
    let out = Span::of(&call.out, need.rows);
    let in_place = ptr::eq(call.out.device, call.res.device) && call.out.offset == call.res.offset;
    let mut inputs = vec![
        Span::of(&call.down, need.down),
        Span::of(&call.weights, need.weights),
    ];
    if !in_place {
        inputs.push(Span::of(&call.res, need.rows));
    }
    if let (Some(gate), Some(shared)) = (&call.shared_gate, &call.shared) {
        inputs.push(Span::of(gate, need.gate));
        inputs.push(Span::of(shared, need.rows));
    }
    inputs.iter().all(|input| out.apart(input))
}

/// Proves a combine launch before the native call: the limits, the optional pair, sizes,
/// alignment (`res`, `down`, `shared` and `out` 16; `weights` and `shared_gate` 4) and that `out`
/// overlaps no input except exactly `res`.
fn check_combine(call: &CombineRaw<'_>) -> Result<(), DeviceFailure> {
    let need = needs(call).ok_or_else(invalid_argument)?;
    let paired = call.shared_gate.is_some() == call.shared.is_some();
    let sized = fits(&call.res, need.rows, 16)
        && fits(&call.down, need.down, 16)
        && fits(&call.weights, need.weights, 4)
        && fits_if_given(call.shared_gate.as_ref(), need.gate, 4)
        && fits_if_given(call.shared.as_ref(), need.rows, 16)
        && fits(&call.out, need.rows, 16);
    if !(paired && sized && out_apart(call, &need)) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the combine of `call` on `stream`: one kernel. Returns when it is queued.
pub(in crate::kernels::cuda) fn moe_combine_launch(
    stream: &RawStream,
    call: &CombineRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_combine(call)?;
    let optional = |range: &Option<RawRange<'_>>| {
        range
            .as_ref()
            .map_or(ptr::null_mut(), |range| range.address())
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_combine` proved every range inside its live allocation (`RawRange::new`),
    // large enough for the call, aligned as the entry requires, and `out` disjoint from every
    // input except exactly `res`; `shared_gate` and `shared` are both given or both null. The
    // entry checks the addresses again. The allocations stay valid until the queued work
    // finishes because `inference_cuda_free` synchronizes the device first, and `args` lives
    // until the entry returns (it copies the struct into the launch parameters). `stream` is live
    // for the call.
    let status = unsafe {
        let args = CombineArgsRaw {
            res: call.res.address().cast_const(),
            down: call.down.address().cast_const(),
            weights: call.weights.address().cast_const(),
            shared_gate: optional(&call.shared_gate).cast_const(),
            shared: optional(&call.shared).cast_const(),
            out: call.out.address().cast::<f32>(),
            m: call.m,
            n: call.n,
            top_k: call.top_k,
            ld_down: call.ld_down,
        };
        moe_combine_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage)
    };
    checked(status, stage)
}

#[cfg(test)]
mod tests {
    use super::super::super::{RawDevice, STATUS_INVALID_ARGUMENT};
    use super::*;

    /// The arguments of one `inference_cuda_moe_combine_launch` call as plain numbers.
    #[derive(Clone, Copy)]
    struct Numbers {
        res: usize,
        down: usize,
        weights: usize,
        shared_gate: usize,
        shared: usize,
        out: usize,
        m: c_int,
        n: c_int,
        top_k: c_int,
        ld_down: c_int,
        stream: usize,
    }

    /// One token of 2048 columns, eight picks, the shared pair. The addresses are made up and 4 GiB
    /// apart, so even 4097 tokens fit between them; each case changes one thing so that the entry
    /// refuses before it reaches the device, and a valid call is never made.
    const BASE: Numbers = Numbers {
        res: 1 << 32,
        down: 2 << 32,
        weights: 3 << 32,
        shared_gate: 4 << 32,
        shared: 5 << 32,
        out: 6 << 32,
        m: 1,
        n: 2048,
        top_k: 8,
        ld_down: 2048,
        stream: 0x1,
    };

    /// A named edit of a valid argument set.
    type Case = (&'static str, fn(&mut Numbers));

    fn pointer<T>(address: usize) -> *mut T {
        ptr::without_provenance_mut(address)
    }

    /// `(status, stage)` of the entry for `numbers`, or for a NULL argument struct.
    fn run_combine(numbers: Numbers, null_args: bool) -> (c_int, c_int) {
        let args = CombineArgsRaw {
            res: pointer::<c_void>(numbers.res).cast_const(),
            down: pointer::<c_void>(numbers.down).cast_const(),
            weights: pointer::<c_void>(numbers.weights).cast_const(),
            shared_gate: pointer::<c_void>(numbers.shared_gate).cast_const(),
            shared: pointer::<c_void>(numbers.shared).cast_const(),
            out: pointer(numbers.out),
            m: numbers.m,
            n: numbers.n,
            top_k: numbers.top_k,
            ld_down: numbers.ld_down,
        };
        let args_ptr = if null_args {
            ptr::null()
        } else {
            &raw const args
        };
        let mut stage = STAGE_NONE;
        // SAFETY: only called with arguments the entry refuses before it uses any pointer or
        // makes a CUDA call (see `BASE`); `args` and `stage` are live for the call.
        let status =
            unsafe { moe_combine_launch_raw(args_ptr, pointer(numbers.stream), &mut stage) };
        (status, stage)
    }

    /// Host only: every case is refused by the entry's own checks before any CUDA call, so the
    /// test needs no device. It covers the C half of the defence in depth: a NULL pointer or
    /// stream, an unpaired shared term, a limit, a misaligned pointer and an overlap, all of which
    /// the Rust wrapper already refuses.
    #[test]
    fn the_native_combine_entry_refuses_bad_arguments_before_any_device_call() {
        let refused = (STATUS_INVALID_ARGUMENT, STAGE_NONE);
        let cases: [Case; 28] = [
            ("NULL stream", |a| a.stream = 0),
            ("NULL res", |a| a.res = 0),
            ("NULL down", |a| a.down = 0),
            ("NULL weights", |a| a.weights = 0),
            ("NULL out", |a| a.out = 0),
            ("gate without shared", |a| a.shared = 0),
            ("shared without gate", |a| a.shared_gate = 0),
            ("m 0", |a| a.m = 0),
            ("m 4097", |a| a.m = 4097),
            ("n 0", |a| a.n = 0),
            ("n 6", |a| a.n = 6),
            ("n 65540", |a| a.n = 65540),
            ("top_k 0", |a| a.top_k = 0),
            ("top_k 33", |a| a.top_k = 33),
            ("ld_down below n", |a| a.ld_down = 2044),
            ("ld_down 66", |a| a.ld_down = 66),
            ("ld_down 2^26 + 4", |a| a.ld_down = (1 << 26) + 4),
            ("res + 4", |a| a.res += 4),
            ("down + 8", |a| a.down += 8),
            ("shared + 4", |a| a.shared += 4),
            ("out + 4", |a| a.out += 4),
            ("weights + 2", |a| a.weights += 2),
            ("gate + 2", |a| a.shared_gate += 2),
            ("out partly over res", |a| a.out = a.res + 16),
            ("out partly over down", |a| a.out = a.down + 16),
            ("out partly over shared", |a| a.out = a.shared + 16),
            ("out partly over weights", |a| a.out = a.weights + 16),
            ("out starting on the gate", |a| a.out = a.shared_gate),
        ];
        assert_eq!(run_combine(BASE, true), refused, "combine: NULL args");
        for (name, edit) in cases {
            let mut numbers = BASE;
            edit(&mut numbers);
            assert_eq!(run_combine(numbers, false), refused, "combine: {name}");
        }
    }

    /// Six unbacked allocations (res, down, weights, gate, shared, out), each far larger than any
    /// range below, so only the term under test can refuse a call.
    fn devices() -> [RawDevice; 6] {
        [(); 6].map(|()| RawDevice::unbacked(1 << 16))
    }

    fn range(device: &RawDevice, offset: usize, bytes: usize) -> RawRange<'_> {
        RawRange::new(device, offset, bytes).expect("a range inside the fixture")
    }

    /// Two tokens of 64 columns, four picks at a stride of 96, the shared pair, every range exact
    /// and aligned, each in its own allocation.
    fn raw_base(d: &[RawDevice; 6]) -> CombineRaw<'_> {
        CombineRaw {
            res: range(&d[0], 0, 512),
            down: range(&d[1], 0, 2944),
            weights: range(&d[2], 0, 32),
            shared_gate: Some(range(&d[3], 0, 8)),
            shared: Some(range(&d[4], 0, 512)),
            out: range(&d[5], 0, 512),
            m: 2,
            n: 64,
            top_k: 4,
            ld_down: 96,
        }
    }

    /// `check_combine` itself, on the host: the Rust half of the defence, which the C entry test
    /// above does not reach. Each case has room for everything, so the overlap or the shared
    /// pairing is the only term that can refuse it.
    #[test]
    fn check_combine_refuses_overlap_and_an_unpaired_shared_term() {
        let d = devices();
        assert!(check_combine(&raw_base(&d)).is_ok(), "the base call");
        let in_place = CombineRaw {
            out: range(&d[0], 0, 512),
            ..raw_base(&d)
        };
        assert!(check_combine(&in_place).is_ok(), "out exactly over res");
        let over = |input: usize, bytes: usize| range(&d[input], 16, bytes);
        let cases: [(&str, CombineRaw<'_>); 8] = [
            (
                "gate without shared",
                CombineRaw {
                    shared: None,
                    ..raw_base(&d)
                },
            ),
            (
                "shared without gate",
                CombineRaw {
                    shared_gate: None,
                    ..raw_base(&d)
                },
            ),
            (
                "out partly over res",
                CombineRaw {
                    out: over(0, 512),
                    ..raw_base(&d)
                },
            ),
            (
                "out partly over down",
                CombineRaw {
                    out: over(1, 512),
                    ..raw_base(&d)
                },
            ),
            (
                "out partly over weights",
                CombineRaw {
                    out: over(2, 512),
                    ..raw_base(&d)
                },
            ),
            (
                "out over the gate",
                CombineRaw {
                    out: range(&d[3], 0, 512),
                    ..raw_base(&d)
                },
            ),
            (
                "out partly over shared",
                CombineRaw {
                    out: over(4, 512),
                    ..raw_base(&d)
                },
            ),
            (
                "out partly before res",
                CombineRaw {
                    out: range(&d[0], 0, 512),
                    res: range(&d[0], 16, 512),
                    ..raw_base(&d)
                },
            ),
        ];
        for (name, call) in cases {
            assert!(check_combine(&call).is_err(), "refuses {name}");
        }
    }
}
