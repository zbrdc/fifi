//! Raw FFI to the SwiGLU + fp16 pack entry of `native/resident/swiglu_pack.cu` (declared in
//! `native/resident/swiglu_pack.h`), RD-4b. Child module of `native/resident.rs`, so it inherits
//! the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this file relies on:
//! - The launch entry is called only through `CudaContext::launch_on`, on the thread that owns the
//!   context and under the dispatch guard (decisions 0023 and 0024). It only enqueues and returns.
//! - Every range is a `RawRange`, proved inside a live allocation by `raw_range`. Sizes, alignment
//!   and disjointness (`super::Span`, `super::pack_bytes`) are proved again here, on the extents
//!   the kernel touches, before the call; the C entry repeats the alignment, shape and overlap
//!   checks on the addresses.
//! - The C argument struct is built in one `unsafe` block that cites those checks.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes the
//!   device before `cudaFree`.
//! - The `extern` declaration, the `repr(C)` mirror and every raw pointer are private to this
//!   file. Nothing is re-exported.

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::{
    DeviceFailure, MAX_DIM, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument,
    pack_bytes,
};

/// The width quantum of the vendored kernel: `k_pad` is a multiple of it.
const WIDTH_QUANTUM: usize = 256;

/// Mirror of `inference_cuda_swiglu_pack_args` in `swiglu_pack.h`, field for field.
#[repr(C)]
struct SwigluPackArgsRaw {
    gu: *mut c_void,
    xf: *mut c_void,
    m: c_int,
    n_ff: c_int,
    k_pad: c_int,
}

const _: () = assert!(
    size_of::<SwigluPackArgsRaw>() == 32
        && offset_of!(SwigluPackArgsRaw, gu) == 0
        && offset_of!(SwigluPackArgsRaw, xf) == 8
        && offset_of!(SwigluPackArgsRaw, m) == 16
        && offset_of!(SwigluPackArgsRaw, n_ff) == 20
        && offset_of!(SwigluPackArgsRaw, k_pad) == 24
);

unsafe extern "C" {
    #[link_name = "inference_cuda_swiglu_pack_launch"]
    fn swiglu_pack_launch_raw(
        args: *const SwigluPackArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// The three integer arguments of the launch, already bounded and converted by the wrapper before
/// `launch_on`, so no conversion can fail inside the launch closure.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct SwigluInts {
    pub(in crate::kernels::cuda) m: c_int,
    pub(in crate::kernels::cuda) n_ff: c_int,
    pub(in crate::kernels::cuda) k_pad: c_int,
}

/// What one SwiGLU pack reads and writes: the converted ranges and the integers.
pub(in crate::kernels::cuda) struct SwigluRaw<'a> {
    pub(in crate::kernels::cuda) gu: RawRange<'a>,
    pub(in crate::kernels::cuda) xf: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: SwigluInts,
}

/// `n` as `usize`, or an invalid-argument failure for a negative value.
fn size(n: c_int) -> Result<usize, DeviceFailure> {
    usize::try_from(n).map_err(|_| invalid_argument())
}

/// Proves a launch before the native call: the shape, the sizes and alignment of both ranges (`gu`
/// 4, `xf` 16) and that `xf`, which the launch writes, shares no byte with `gu`, which it writes
/// as well (the zero tails). The extents compared are the ones the kernel touches.
fn check_swiglu(call: &SwigluRaw<'_>) -> Result<(), DeviceFailure> {
    let (m, n_ff, k_pad) = (
        size(call.ints.m)?,
        size(call.ints.n_ff)?,
        size(call.ints.k_pad)?,
    );
    let shape = (1..=MAX_DIM).contains(&m)
        && (WIDTH_QUANTUM..=MAX_DIM).contains(&k_pad)
        && k_pad.is_multiple_of(WIDTH_QUANTUM)
        && (1..=k_pad).contains(&n_ff);
    let gu_need = m
        .checked_mul(2 * k_pad)
        .and_then(|floats| floats.checked_mul(4))
        .filter(|_| shape)
        .ok_or_else(invalid_argument)?;
    let (_, xf_need) = pack_bytes(m, k_pad).ok_or_else(invalid_argument)?;
    let sized = call.gu.bytes >= gu_need && call.xf.bytes >= xf_need;
    let aligned = call.gu.offset.is_multiple_of(4) && call.xf.offset.is_multiple_of(16);
    let apart = Span::of(&call.gu, gu_need).apart(&Span::of(&call.xf, xf_need));
    if !(sized && aligned && apart) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the SwiGLU + pack on `stream`: the zero tails of `gu` when `n_ff < k_pad`, and the
/// kernel (which writes the shared zero tail of a fragment-order `xf` itself). Returns when it
/// is queued.
pub(in crate::kernels::cuda) fn swiglu_pack_launch(
    stream: &RawStream,
    call: SwigluRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_swiglu(&call)?;
    let ints = call.ints;
    let args = SwigluPackArgsRaw {
        gu: call.gu.address(),
        xf: call.xf.address(),
        m: ints.m,
        n_ff: ints.n_ff,
        k_pad: ints.k_pad,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: (a) `check_swiglu` proved both ranges inside their live allocations and large enough
    // for the call; the allocations outlive the queued work because `inference_cuda_free`
    // synchronizes the device before `cudaFree`. (b) It also proved the alignment (`gu` 4, `xf`
    // 16) and that the extents of `gu` and `xf` share no byte, so the kernel's reads and the
    // writes to `xf` and the `gu` tails cannot alias; the C entry repeats the checks on the
    // addresses. (c) `stream` is borrowed from a live `RawStream`. (d) `args` is a live local that
    // the entry reads only during the call and keeps no pointer to. (e) The dispatch guard is held
    // by `launch_on`, which is the only caller.
    let status = unsafe { swiglu_pack_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod tests {
    use super::super::super::STATUS_INVALID_ARGUMENT;
    use super::*;

    /// A valid argument set of the C entry as plain numbers: one row of 256 columns, `gu` and `xf`
    /// at made-up addresses far apart. Each case changes one thing so the entry refuses before it
    /// reaches the device; a valid call is never made.
    #[derive(Clone, Copy)]
    struct Args {
        gu: usize,
        xf: usize,
        m: c_int,
        n_ff: c_int,
        k_pad: c_int,
        stream: usize,
    }

    const VALID: Args = Args {
        gu: 0x10_0000,
        xf: 0x40_0000,
        m: 1,
        n_ff: 256,
        k_pad: 256,
        stream: 0x1,
    };

    /// A named edit of the valid argument set.
    type Case = (&'static str, fn(&mut Args));

    fn pointer(address: usize) -> *mut c_void {
        std::ptr::without_provenance_mut(address)
    }

    /// `(status, stage)` of the entry for `args`.
    fn run_entry(args: Args) -> (c_int, c_int) {
        let raw = SwigluPackArgsRaw {
            gu: pointer(args.gu),
            xf: pointer(args.xf),
            m: args.m,
            n_ff: args.n_ff,
            k_pad: args.k_pad,
        };
        let mut stage = STAGE_NONE;
        // SAFETY: only called with arguments the entry refuses before it uses any pointer or
        // makes a CUDA call (see `VALID`); `raw` and `stage` are live for the call.
        let status =
            unsafe { swiglu_pack_launch_raw(&raw const raw, pointer(args.stream), &mut stage) };
        (status, stage)
    }

    /// Host only: every case is refused by the entry's own checks before any CUDA call, so the
    /// test needs no device. It covers the C half of the defence in depth: a bad shape, a
    /// misaligned pointer, a NULL stream and an overlap, all of which the Rust wrapper already
    /// refuses.
    #[test]
    fn the_native_swiglu_entry_refuses_bad_arguments_before_any_device_call() {
        let refused = (STATUS_INVALID_ARGUMENT, STAGE_NONE);
        let cases: [Case; 14] = [
            ("m 0", |a| a.m = 0),
            ("m 32769, xf above gu", |a| {
                // gu then spans about 67 MB: keep xf clear of it, so the m limit is what refuses.
                a.m = 32_769;
                a.xf = a.gu + 0x500_0000;
            }),
            ("k_pad 255", |a| a.k_pad = 255),
            ("k_pad 257", |a| a.k_pad = 257),
            ("k_pad 300", |a| a.k_pad = 300),
            ("k_pad 33024", |a| a.k_pad = 33_024),
            ("n_ff 0", |a| a.n_ff = 0),
            ("n_ff above k_pad", |a| a.n_ff = 257),
            ("gu at +2", |a| a.gu += 2),
            ("xf at +8", |a| a.xf += 8),
            ("NULL stream", |a| a.stream = 0),
            ("xf equal to gu", |a| a.xf = a.gu),
            ("xf inside gu", |a| a.xf = a.gu + 16),
            ("gu inside xf", |a| a.gu = a.xf + 64),
        ];
        for (name, edit) in cases {
            let mut args = VALID;
            edit(&mut args);
            assert_eq!(run_entry(args), refused, "{name}");
        }
    }
}
