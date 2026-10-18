//! Raw FFI to the dense MMQ GEMM entry of `native/prefill/mmq_dense.cu` (its header holds the
//! full contract), CPF-V4 G2. Child module of `native/prefill.rs`, so it inherits the
//! `allow(unsafe_code)` of `native.rs`; the safety invariants are the parent's list.
//! Addition of this entry: the launch only enqueues; the stream-k fixup range is sized by the
//! safe wrapper (the device's maximum), and a range the vendored launch finds too short for
//! the tile width it picked is refused there with `cudaErrorInvalidValue` before any kernel
//! is enqueued.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "no caller yet: the chunk driver does not call this kernel; drop the expect when it does"
    )
)]

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use crate::kernels::cuda::prefill::mmq_dense::DenseInts;
use crate::kernels::cuda::prefill::mmq_size::q8_1_bytes;
use crate::kernels::cuda::resident::matvec_q::QuantType;

/// Mirror of `inference_cuda_mmq_dense_args`, field for field.
#[repr(C)]
struct MmqDenseArgsRaw {
    w: *const c_void,
    y: *const c_void,
    dst: *mut c_void,
    fixup: *mut c_void,
    fixup_bytes: usize,
    ty: c_int,
    k: c_int,
    n: c_int,
    m: c_int,
    ld: c_int,
    col_off: c_int,
}

const _: () = assert!(
    size_of::<MmqDenseArgsRaw>() == 64
        && offset_of!(MmqDenseArgsRaw, y) == 8
        && offset_of!(MmqDenseArgsRaw, dst) == 16
        && offset_of!(MmqDenseArgsRaw, fixup) == 24
        && offset_of!(MmqDenseArgsRaw, fixup_bytes) == 32
        && offset_of!(MmqDenseArgsRaw, ty) == 40
        && offset_of!(MmqDenseArgsRaw, k) == 44
        && offset_of!(MmqDenseArgsRaw, n) == 48
        && offset_of!(MmqDenseArgsRaw, m) == 52
        && offset_of!(MmqDenseArgsRaw, ld) == 56
        && offset_of!(MmqDenseArgsRaw, col_off) == 60
);

unsafe extern "C" {
    #[link_name = "inference_cuda_mmq_dense"]
    fn mmq_dense_raw(
        args: *const MmqDenseArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// The ranges of one dense call plus its converted integers.
pub(in crate::kernels::cuda) struct MmqDenseRaw<'a> {
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) dst: RawRange<'a>,
    pub(in crate::kernels::cuda) fixup: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: DenseInts,
}

/// `n` as `usize`, or an invalid-argument failure.
fn size(n: c_int) -> Result<usize, DeviceFailure> {
    usize::try_from(n).map_err(|_| invalid_argument())
}

/// The weight type of a `ggml_type` discriminant, for the three dense cases.
fn quant_of(ggml: c_int) -> Result<QuantType, DeviceFailure> {
    [QuantType::Q80, QuantType::Q4K, QuantType::Q6K]
        .into_iter()
        .find(|quant| quant.ggml() == ggml)
        .ok_or_else(invalid_argument)
}

/// Proves a dense call before the native call: the C sizes (weight blocks, `y` with the
/// constant pad, the destination reach), the alignments (w and y 16, dst 4, fixup 16), and
/// that the two written ranges share no byte with another range.
fn check_dense(io: &MmqDenseRaw<'_>) -> Result<(), DeviceFailure> {
    let ints = io.ints;
    let (k, n, m) = (size(ints.k)?, size(ints.n)?, size(ints.m)?);
    let (ld, col_off) = (size(ints.ld)?, size(ints.col_off)?);
    let quant = quant_of(ints.ty)?;
    let w_need = quant.weight_bytes(n, k).ok_or_else(invalid_argument)?;
    let y_need = q8_1_bytes(k, m).map_err(|_| invalid_argument())?;
    let dst_need = m
        .checked_sub(1)
        .and_then(|last| last.checked_mul(ld))
        .and_then(|lead| lead.checked_add(col_off))
        .and_then(|reach| reach.checked_add(n))
        .and_then(|floats| floats.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    let sized = io.w.bytes() >= w_need && io.y.bytes() >= y_need && io.dst.bytes() >= dst_need;
    let aligned = io.w.offset().is_multiple_of(16)
        && io.y.offset().is_multiple_of(16)
        && io.dst.offset().is_multiple_of(4)
        && io.fixup.offset().is_multiple_of(16);
    let (w, y) = (Span::of(&io.w, w_need), Span::of(&io.y, y_need));
    let (dst, fixup) = (
        Span::of(&io.dst, dst_need),
        Span::of(&io.fixup, io.fixup.bytes()),
    );
    let apart =
        dst.apart(&w) && dst.apart(&y) && fixup.apart(&w) && fixup.apart(&y) && fixup.apart(&dst);
    if !(sized && aligned && apart) {
        return Err(invalid_argument());
    }
    Ok(())
}

fn c_dense(io: &MmqDenseRaw<'_>) -> MmqDenseArgsRaw {
    MmqDenseArgsRaw {
        w: io.w.address().cast_const(),
        y: io.y.address().cast_const(),
        dst: io.dst.address(),
        fixup: io.fixup.address(),
        fixup_bytes: io.fixup.bytes(),
        ty: io.ints.ty,
        k: io.ints.k,
        n: io.ints.n,
        m: io.ints.m,
        ld: io.ints.ld,
        col_off: io.ints.col_off,
    }
}

/// Queues the dense GEMM on `stream`.
pub(in crate::kernels::cuda) fn mmq_dense_launch(
    stream: &RawStream,
    io: MmqDenseRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_dense(&io)?;
    let args = c_dense(&io);
    let mut stage = STAGE_NONE;
    // SAFETY: `check_dense` proved every range inside its live allocation, large enough for
    // the call, aligned, and the written ranges (`dst`, `fixup`) disjoint from every other
    // range, so each pointer in `args` addresses bytes the kernels may touch. The C ABI
    // carries no capacities, so the C entry repeats only what addresses can show: the shape
    // rules, the alignments and the overlaps, plus the fixup size it derives from the
    // device. The allocations stay valid until the queued
    // work finishes because `inference_cuda_free` synchronizes the device first. `args`
    // lives across the call and the entry copies it before returning. `stream` is live for
    // the call.
    let status = unsafe { mmq_dense_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod tests;
