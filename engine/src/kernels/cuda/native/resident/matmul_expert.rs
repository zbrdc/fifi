//! Raw FFI to `inference_cuda_matmul_expert_launch` and its helper entries of
//! `native/resident/matmul_expert.cu` (declared in `native/resident/matmul_expert.h`). Child of
//! `native::resident`, so it inherits the `allow(unsafe_code)` of `native.rs` (decision 0047,
//! child modules of `native.rs`).
//!
//! Safety invariants this module relies on:
//! - A device range is proved in bounds by `RawRange::new` before it gets here. This module
//!   proves, again and on the sizes of the call, that the stack holds `count` experts, that
//!   `ids` holds `slot + 1` ints, that `xf` and `y` hold what `activation_bytes` says, that every
//!   range is aligned, and that `y`, which the kernel writes, shares no byte with `xf`, the
//!   experts' tiles or the ids. The C entry repeats these checks on the addresses.
//! - The ids are not read by the host: the kernel reads `ids[slot]` when it runs, and an id
//!   outside `[0, count)` makes it write NaN to `y` and read no weight.
//! - The allocations stay valid until the queued launch finishes because `inference_cuda_free`
//!   synchronizes the whole device before `cudaFree`.
//! - The launch entry only queues work on the stream it is given and returns the launch status;
//!   a fault in the kernel surfaces at the next synchronize.
//! - Every call runs under the process dispatch guard and on the thread that owns the context
//!   (decisions 0023 and 0024).
//! - No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_void};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use crate::kernels::cuda::native::{TileVec, Tiled, TiledType, activation_bytes};

unsafe extern "C" {
    #[link_name = "inference_cuda_matmul_expert_launch"]
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the C declaration in matmul_expert.h"
    )]
    fn matmul_expert_launch_raw(
        ggml_type: c_int,
        m: c_int,
        stack: *const c_void,
        ids: *const c_int,
        slot: c_int,
        count: c_int,
        xf: *const c_void,
        y: *mut f32,
        ldy: c_int,
        n_rows: c_int,
        k: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_expert_shape_ok"]
    fn expert_shape_ok_raw(m: c_int, n_rows: c_int, k: c_int) -> c_int;
}

#[cfg(all(test, feature = "cuda-fault-qualification"))]
unsafe extern "C" {
    #[link_name = "inference_cuda_fault_expert_rejections"]
    fn fault_expert_rejections_raw(out_mask: *mut u32, out_stage: *mut c_int) -> c_int;
}

/// True when `m` tokens against an expert of `rows` x `k` are a shape the launch accepts: `m` in
/// `1..=8`, a tiled shape, and no split along k. The split rule is the vendored `split_k`; the
/// adapter asks it, so this crate keeps no copy. Host only, no CUDA call.
pub(in crate::kernels::cuda) fn expert_shape_ok(m: usize, rows: usize, k: usize) -> bool {
    let (Ok(m), Ok(rows), Ok(k)) = (
        c_int::try_from(m),
        c_int::try_from(rows),
        c_int::try_from(k),
    ) else {
        return false;
    };
    // SAFETY: the entry takes three integers, reads no pointer and makes no CUDA call.
    unsafe { expert_shape_ok_raw(m, rows, k) == 0 }
}

/// What one expert launch reads and writes: `stack` holds the tiles of the `count` experts of
/// `rows` x `k` that the call may select, `ids` the expert ids (the kernel reads `ids[slot]`),
/// `xf` the packed input of `m` tokens, and `y` receives `m` rows of `rows` values.
pub(in crate::kernels::cuda) struct ExpertRaw<'a> {
    pub(in crate::kernels::cuda) ty: TiledType,
    pub(in crate::kernels::cuda) stack: RawRange<'a>,
    pub(in crate::kernels::cuda) ids: RawRange<'a>,
    pub(in crate::kernels::cuda) slot: usize,
    pub(in crate::kernels::cuda) count: usize,
    pub(in crate::kernels::cuda) xf: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) m: usize,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) k: usize,
}

/// The integer arguments of a checked call.
struct Args {
    m: c_int,
    rows: c_int,
    k: c_int,
    slot: c_int,
    count: c_int,
}

/// Bytes each range of a call must hold.
struct Sizes {
    stack: usize,
    ids: usize,
    xf: usize,
    y: usize,
}

fn call_args(call: &ExpertRaw<'_>) -> Option<Args> {
    let tiled = Tiled {
        ty: call.ty,
        n_rows: call.rows,
        k: call.k,
    };
    if !tiled.fits() || call.count < 1 || !expert_shape_ok(call.m, call.rows, call.k) {
        return None;
    }
    Some(Args {
        m: c_int::try_from(call.m).ok()?,
        rows: c_int::try_from(call.rows).ok()?,
        k: c_int::try_from(call.k).ok()?,
        slot: c_int::try_from(call.slot).ok()?,
        count: c_int::try_from(call.count).ok()?,
    })
}

fn call_sizes(call: &ExpertRaw<'_>) -> Option<Sizes> {
    let tiled = Tiled {
        ty: call.ty,
        n_rows: call.rows,
        k: call.k,
    };
    let expert_bytes = tiled.tile_vecs()?.checked_mul(size_of::<TileVec>())?;
    let need = activation_bytes(call.rows, call.k, call.m).ok()?;
    Some(Sizes {
        stack: expert_bytes.checked_mul(call.count)?,
        ids: call.slot.checked_add(1)?.checked_mul(size_of::<c_int>())?,
        xf: need.xf,
        y: need.y,
    })
}

/// Proves an expert launch before the native call: shape, sizes, alignment (stack and xf 16, ids
/// and y 4) and that `y` overlaps none of the ranges the kernel reads.
fn check_expert(call: &ExpertRaw<'_>) -> Result<Args, DeviceFailure> {
    let args = call_args(call).ok_or_else(invalid_argument)?;
    let need = call_sizes(call).ok_or_else(invalid_argument)?;
    let sized = call.stack.bytes >= need.stack
        && call.ids.bytes >= need.ids
        && call.xf.bytes >= need.xf
        && call.y.bytes >= need.y;
    let aligned = call.stack.offset.is_multiple_of(16)
        && call.xf.offset.is_multiple_of(16)
        && call.ids.offset.is_multiple_of(4)
        && call.y.offset.is_multiple_of(4);
    let y = Span::of(&call.y, need.y);
    let apart = y.apart(&Span::of(&call.xf, need.xf))
        && y.apart(&Span::of(&call.stack, need.stack))
        && y.apart(&Span::of(&call.ids, need.ids));
    if !(sized && aligned && apart) {
        return Err(invalid_argument());
    }
    Ok(args)
}

/// Queues the expert matmul of `call` on `stream`. Returns when the kernel is queued.
pub(in crate::kernels::cuda) fn matmul_expert_launch(
    stream: &RawStream,
    call: &ExpertRaw<'_>,
) -> Result<(), DeviceFailure> {
    let args = check_expert(call)?;
    let mut stage = STAGE_NONE;
    // SAFETY: `check_expert` proved the stack, ids, xf and y ranges inside their live
    // allocations, large enough for the call, aligned, and `y` disjoint from the other three; the
    // adapter repeats the checks on the addresses. The allocations stay valid until the queued
    // launch finishes because `inference_cuda_free` synchronizes the device first. The kernel
    // reads the ids when it runs and the adapter keeps no pointer. `stream` is live for the call.
    let status = unsafe {
        matmul_expert_launch_raw(
            call.ty.ggml(),
            args.m,
            call.stack.address().cast_const(),
            call.ids.address().cast::<c_int>().cast_const(),
            args.slot,
            args.count,
            call.xf.address().cast_const(),
            call.y.address().cast::<f32>(),
            args.rows,
            args.rows,
            args.k,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// Fault qualification: the bit mask of the fourteen invalid argument sets that the vendored
/// `kern::matmul_expert` rejected without launching.
#[cfg(all(test, feature = "cuda-fault-qualification"))]
pub(in crate::kernels::cuda) fn fault_expert_rejections() -> Result<u32, DeviceFailure> {
    let mut mask = 0u32;
    let mut stage = STAGE_NONE;
    // SAFETY: `mask` and `stage` are live, writable values for the call; the entry launches
    // nothing and passes only null pointers to the vendored function.
    let status = unsafe { fault_expert_rejections_raw(&mut mask, &mut stage) };
    checked(status, stage)?;
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::super::super::STATUS_INVALID_ARGUMENT;
    use super::*;

    /// The arguments of one `inference_cuda_matmul_expert_launch` call as plain numbers.
    #[derive(Clone, Copy)]
    struct ExpertArgs {
        ty: c_int,
        m: c_int,
        stack: usize,
        ids: usize,
        slot: c_int,
        count: c_int,
        xf: usize,
        y: usize,
        ldy: c_int,
        n_rows: c_int,
        k: c_int,
        stream: usize,
    }

    /// Four Q4_K experts of eight rows of 256 values (1152 bytes each), one token, slot 0. The
    /// addresses are made up and apart; each case changes one thing so the entry refuses before
    /// it reaches the device, and a valid call is never made.
    const EXPERT_ARGS: ExpertArgs = ExpertArgs {
        ty: 12,
        m: 1,
        stack: 0x1_0000,
        ids: 0x2_0000,
        slot: 0,
        count: 4,
        xf: 0x10_0000,
        y: 0x30_0000,
        ldy: 8,
        n_rows: 8,
        k: 256,
        stream: 0x1,
    };

    /// A named edit of a valid argument set.
    type Case = (&'static str, fn(&mut ExpertArgs));

    fn pointer<T>(address: usize) -> *mut T {
        std::ptr::without_provenance_mut(address)
    }

    /// `(status, stage)` of the entry for `args`.
    fn run_expert(args: ExpertArgs) -> (c_int, c_int) {
        let mut stage = STAGE_NONE;
        // SAFETY: only called with arguments the entry refuses before it uses any pointer or
        // makes a CUDA call (see `EXPERT_ARGS`); `stage` is a live, writable c_int.
        let status = unsafe {
            matmul_expert_launch_raw(
                args.ty,
                args.m,
                pointer::<c_void>(args.stack).cast_const(),
                pointer::<c_int>(args.ids).cast_const(),
                args.slot,
                args.count,
                pointer::<c_void>(args.xf).cast_const(),
                pointer(args.y),
                args.ldy,
                args.n_rows,
                args.k,
                pointer(args.stream),
                &mut stage,
            )
        };
        (status, stage)
    }

    /// Host only: every case is refused by the entry's own checks before any CUDA call, so the
    /// test needs no device. It covers the C half of the defence in depth: a NULL pointer or
    /// stream, a bad shape, a misaligned pointer, an overlap or a size that wraps, all of which
    /// the Rust wrapper already refuses.
    #[test]
    fn the_native_expert_entry_refuses_bad_arguments_before_any_device_call() {
        let refused = (STATUS_INVALID_ARGUMENT, STAGE_NONE);
        let cases: [Case; 24] = [
            ("NULL stack", |a| a.stack = 0),
            ("NULL ids", |a| a.ids = 0),
            ("NULL xf", |a| a.xf = 0),
            ("NULL y", |a| a.y = 0),
            ("NULL stream", |a| a.stream = 0),
            ("a type that is not tiled", |a| a.ty = 99),
            ("12 rows", |a| {
                a.n_rows = 12;
                a.ldy = 12;
            }),
            ("k = 100", |a| a.k = 100),
            ("k above the longest row", |a| a.k = 32768 + 256),
            ("a shape that splits along k", |a| {
                a.n_rows = 64;
                a.ldy = 64;
                a.k = 4096;
            }),
            ("no tokens", |a| a.m = 0),
            ("more tokens than an expert launch takes", |a| a.m = 9),
            ("slot below zero", |a| a.slot = -1),
            ("no experts", |a| a.count = 0),
            ("ldy below n_rows", |a| a.ldy = 7),
            ("stack offset not a multiple of 16", |a| a.stack += 8),
            ("xf offset not a multiple of 16", |a| a.xf += 4),
            ("ids offset not a multiple of 4", |a| a.ids += 2),
            ("y offset not a multiple of 4", |a| a.y += 2),
            ("y inside xf", |a| a.y = a.xf + 1024),
            ("y inside the stack", |a| a.y = a.stack + 1024),
            ("y over the ids", |a| a.y = a.ids),
            ("count times expert bytes wraps", |a| {
                a.n_rows = 0x7fff_fff8;
                a.ldy = a.n_rows;
                a.k = 32768;
                a.count = c_int::MAX;
            }),
            ("y over the end of xf", |a| a.y = a.xf + 1_048_592 - 16),
        ];
        for (name, edit) in cases {
            let mut args = EXPERT_ARGS;
            edit(&mut args);
            assert_eq!(run_expert(args), refused, "expert: {name}");
        }
    }
}
