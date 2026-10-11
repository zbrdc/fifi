//! Raw FFI to the Q5_0 decode matmul entries of `native/resident/matmul_q5_0.cu` (declared in
//! `native/resident/matmul_q5_0.h`, MEL-C1b). Child of `native::resident`, so it inherits the
//! `allow(unsafe_code)` of `native.rs` (decision 0047, child modules of `native.rs`).
//!
//! Safety invariants this module relies on:
//! - A device range is proved in bounds by `RawRange::new` (or a `DeviceView` check) before it
//!   gets here. This module proves, again and on the sizes of the call, that the weight tiles
//!   (and, for an expert call, the `count` experts and the `slot + 1` ids) lie inside their
//!   allocation, that `xf` and `y` hold what `activation_bytes` says for the 256-padded row
//!   width, that every range is aligned, and that `y`, which the kernel writes, shares no byte
//!   with `xf`, the weight tiles or the ids. The C entries repeat these checks on the addresses.
//! - The ids are not read by the host: the kernel reads `ids[slot]` when it runs, and an id
//!   outside `[0, count)` makes it write NaN to `y` and read no weight.
//! - The kernel never reads the `xf` content beyond `k`, so the zero tail of the activation row is
//!   a convention of the producer, not a memory-safety condition.
//! - The allocations stay valid until the queued launch finishes because `inference_cuda_free`
//!   synchronizes the whole device before `cudaFree`.
//! - The launch entries only queue work on the stream they are given and return the launch status;
//!   a fault in the kernel surfaces at the next synchronize.
//! - Every call runs under the process dispatch guard and on the thread that owns the context
//!   (decisions 0023 and 0024).
//! - No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_void};

use super::matmul_expert::ExpertRaw;
use super::{DeviceFailure, MatmulRaw, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use crate::kernels::cuda::native::{TileVec, Tiled, TiledType, activation_bytes, activation_k};

unsafe extern "C" {
    #[link_name = "inference_cuda_matmul_q5_0_launch"]
    fn matmul_q5_0_launch_raw(
        m: c_int,
        weight: *const c_void,
        xf: *const c_void,
        y: *mut f32,
        ldy: c_int,
        n_rows: c_int,
        k: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_matmul_expert_q5_0_launch"]
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the C declaration in matmul_q5_0.h, which mirrors the RD-0b expert entry"
    )]
    fn matmul_expert_q5_0_launch_raw(
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
    #[link_name = "inference_cuda_q5_0_shape_ok"]
    fn q5_0_shape_ok_raw(m: c_int, n_rows: c_int, k: c_int, expert: c_int) -> c_int;
}

/// Most tokens a dense Q5_0 launch takes (two token groups of 8).
pub(in crate::kernels::cuda) const MAX_DENSE_TOKENS: usize = 16;

/// True when `m` tokens against `rows` x `k` Q5_0 values are a shape the launch accepts: `m` in
/// `1..=8` for an expert call and `1..=16` for a dense one, `rows` a positive multiple of 8, `k` a
/// positive multiple of 128 up to 32768. Host only, no CUDA call.
pub(in crate::kernels::cuda) fn q5_0_shape_ok(
    m: usize,
    rows: usize,
    k: usize,
    expert: bool,
) -> bool {
    let (Ok(m), Ok(rows), Ok(k)) = (
        c_int::try_from(m),
        c_int::try_from(rows),
        c_int::try_from(k),
    ) else {
        return false;
    };
    // SAFETY: the entry takes four integers, reads no pointer and makes no CUDA call.
    unsafe { q5_0_shape_ok_raw(m, rows, k, c_int::from(expert)) == 0 }
}

/// The integer arguments of a checked call.
struct Dims {
    m: c_int,
    rows: c_int,
    k: c_int,
}

/// The shape of a Q5_0 call as C ints, or `None` for another type or a shape the kernel refuses.
fn dims(ty: TiledType, rows: usize, k: usize, m: usize, expert: bool) -> Option<Dims> {
    let tiled = Tiled {
        ty,
        n_rows: rows,
        k,
    };
    if ty != TiledType::Q50 || !tiled.fits() || !q5_0_shape_ok(m, rows, k, expert) {
        return None;
    }
    Some(Dims {
        m: c_int::try_from(m).ok()?,
        rows: c_int::try_from(rows).ok()?,
        k: c_int::try_from(k).ok()?,
    })
}

/// Proves a dense launch before the native call: shape, sizes, the weight tiles inside their
/// allocation, alignment (weight 16, xf 16, y 4) and that `y` overlaps neither the weight nor `xf`.
fn check_dense(call: &MatmulRaw<'_>) -> Result<Dims, DeviceFailure> {
    let tiled = call.tiled;
    let dims = dims(tiled.ty, tiled.n_rows, tiled.k, call.m, false).ok_or_else(invalid_argument)?;
    let k_x = activation_k(tiled.ty, tiled.k);
    let need = activation_bytes(tiled.n_rows, k_x, call.m).map_err(|_| invalid_argument())?;
    let tile_bytes = tiled
        .tile_vecs()
        .and_then(|vecs| vecs.checked_mul(size_of::<TileVec>()))
        .ok_or_else(invalid_argument)?;
    let weight_inside = tile_bytes
        .checked_add(call.weight.offset)
        .is_some_and(|end| end <= call.weight.device.bytes);
    let sized = call.xf.bytes >= need.xf && call.y.bytes >= need.y;
    let aligned = call.weight.offset.is_multiple_of(16)
        && call.xf.offset.is_multiple_of(16)
        && call.y.offset.is_multiple_of(4);
    let weight = Span {
        device: call.weight.device,
        start: call.weight.offset,
        len: tile_bytes,
    };
    let y = Span::of(&call.y, need.y);
    let apart = y.apart(&Span::of(&call.xf, need.xf)) && y.apart(&weight);
    if !(weight_inside && sized && aligned && apart) {
        return Err(invalid_argument());
    }
    Ok(dims)
}

/// Queues the dense Q5_0 matmul of `call` on `stream`. Returns when the kernel is queued.
pub(in crate::kernels::cuda) fn matmul_q5_0_launch(
    stream: &RawStream,
    call: &MatmulRaw<'_>,
) -> Result<(), DeviceFailure> {
    let dims = check_dense(call)?;
    let mut stage = STAGE_NONE;
    // SAFETY: `check_dense` proved the weight tiles and both ranges inside their live
    // allocations, large enough for the call, aligned, and `y` disjoint from the weight and `xf`,
    // so the weight offset stays inside its allocation; the adapter repeats the checks on the
    // addresses. The allocations stay valid until the queued launch finishes because
    // `inference_cuda_free` synchronizes the device first. `stream` is live for the call.
    let status = unsafe {
        let weight = call
            .weight
            .device
            .ptr
            .as_ptr()
            .cast::<u8>()
            .add(call.weight.offset);
        matmul_q5_0_launch_raw(
            dims.m,
            weight.cast::<c_void>().cast_const(),
            call.xf.address().cast_const(),
            call.y.address().cast::<f32>(),
            dims.rows,
            dims.rows,
            dims.k,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// Bytes each range of an expert call must hold.
struct Sizes {
    stack: usize,
    ids: usize,
    xf: usize,
    y: usize,
}

fn expert_sizes(call: &ExpertRaw<'_>) -> Option<Sizes> {
    let one = Tiled {
        ty: call.ty,
        n_rows: call.rows,
        k: call.k,
    };
    let expert_bytes = one.tile_vecs()?.checked_mul(size_of::<TileVec>())?;
    let need = activation_bytes(call.rows, activation_k(call.ty, call.k), call.m).ok()?;
    Some(Sizes {
        stack: expert_bytes.checked_mul(call.count)?,
        ids: call.slot.checked_add(1)?.checked_mul(size_of::<c_int>())?,
        xf: need.xf,
        y: need.y,
    })
}

/// Proves an expert launch before the native call: shape, sizes, alignment (stack and xf 16, ids
/// and y 4) and that `y` overlaps none of the ranges the kernel reads.
fn check_expert(call: &ExpertRaw<'_>) -> Result<(Dims, c_int, c_int), DeviceFailure> {
    let dims = dims(call.ty, call.rows, call.k, call.m, true).ok_or_else(invalid_argument)?;
    if call.count < 1 {
        return Err(invalid_argument());
    }
    let slot = c_int::try_from(call.slot).map_err(|_| invalid_argument())?;
    let count = c_int::try_from(call.count).map_err(|_| invalid_argument())?;
    let need = expert_sizes(call).ok_or_else(invalid_argument)?;
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
    Ok((dims, slot, count))
}

/// Queues the Q5_0 expert matmul of `call` on `stream`. Returns when the kernel is queued.
pub(in crate::kernels::cuda) fn matmul_expert_q5_0_launch(
    stream: &RawStream,
    call: &ExpertRaw<'_>,
) -> Result<(), DeviceFailure> {
    let (dims, slot, count) = check_expert(call)?;
    let mut stage = STAGE_NONE;
    // SAFETY: `check_expert` proved the stack, ids, xf and y ranges inside their live
    // allocations, large enough for the call, aligned, and `y` disjoint from the other three; the
    // adapter repeats the checks on the addresses. The allocations stay valid until the queued
    // launch finishes because `inference_cuda_free` synchronizes the device first. The kernel
    // reads the ids when it runs and the adapter keeps no pointer. `stream` is live for the call.
    let status = unsafe {
        matmul_expert_q5_0_launch_raw(
            dims.m,
            call.stack.address().cast_const(),
            call.ids.address().cast::<c_int>().cast_const(),
            slot,
            count,
            call.xf.address().cast_const(),
            call.y.address().cast::<f32>(),
            dims.rows,
            dims.rows,
            dims.k,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

#[cfg(test)]
mod tests {
    use super::super::super::STATUS_INVALID_ARGUMENT;
    use super::*;

    /// Host only: the shape rule of the C entry, through the same call the wrappers use.
    #[test]
    fn q5_0_native_shape_rules() {
        for (m, rows, k, expert) in [
            (1, 8, 896, false),
            (16, 8, 896, false),
            (8, 8, 896, true),
            (1, 2304, 128, true),
            (1, 2304, 32768, false),
            (1, 8, 640, true),
        ] {
            assert!(
                q5_0_shape_ok(m, rows, k, expert),
                "accepts m={m} rows={rows} k={k} expert={expert}"
            );
        }
        for (m, rows, k, expert, why) in [
            (1, 8, 0, false, "k = 0"),
            (1, 8, 100, false, "k = 100"),
            (1, 8, 864, false, "k = 864 (a multiple of 32, not of 128)"),
            (1, 8, 32896, false, "k above 32768"),
            (1, 12, 896, false, "12 rows"),
            (1, 0, 896, false, "no rows"),
            (0, 8, 896, false, "no tokens"),
            (17, 8, 896, false, "17 tokens, dense"),
            (9, 8, 896, true, "9 tokens, expert"),
        ] {
            assert!(
                !q5_0_shape_ok(m, rows, k, expert),
                "refuses {why}: m={m} rows={rows} k={k} expert={expert}"
            );
        }
    }

    /// Host only: every case is refused by the entry's own checks before any CUDA call, so the
    /// test needs no device.
    #[test]
    fn the_native_q5_0_entries_refuse_bad_arguments_before_any_device_call() {
        let refused = (STATUS_INVALID_ARGUMENT, STAGE_NONE);
        let pointer = |address: usize| std::ptr::without_provenance_mut::<c_void>(address);
        let dense = |m: c_int, weight: usize, ldy: c_int, rows: c_int, k: c_int, stream: usize| {
            let mut stage = STAGE_NONE;
            // SAFETY: only called with arguments the entry refuses before it uses any pointer or
            // makes a CUDA call; `stage` is a live, writable c_int.
            let status = unsafe {
                matmul_q5_0_launch_raw(
                    m,
                    pointer(weight).cast_const(),
                    pointer(0x10_0000).cast_const(),
                    pointer(0x30_0000).cast(),
                    ldy,
                    rows,
                    k,
                    pointer(stream),
                    &mut stage,
                )
            };
            (status, stage)
        };
        assert_eq!(dense(1, 0x1_0000, 8, 8, 100, 1), refused, "k = 100");
        assert_eq!(dense(17, 0x1_0000, 8, 8, 896, 1), refused, "17 tokens");
        assert_eq!(dense(1, 0x1_0000, 7, 8, 896, 1), refused, "ldy below rows");
        assert_eq!(
            dense(1, 0x1_0008, 8, 8, 896, 1),
            refused,
            "weight misaligned"
        );
        assert_eq!(dense(1, 0, 8, 8, 896, 1), refused, "NULL weight");
        assert_eq!(dense(1, 0x1_0000, 8, 8, 896, 0), refused, "NULL stream");
    }
}
