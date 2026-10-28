//! Raw FFI to the q8_1 quantize entries of `native/prefill/mmq_quant.cu` (its header holds
//! the full contract), CPF-V4 G4. Child module of `native/prefill.rs`, so it inherits the
//! `allow(unsafe_code)` of `native.rs`; the safety invariants are the parent's list.
//! Additions of this entry: both launches only enqueue; the gather and inverse maps' VALUES
//! are trusted device indices (plan section C.2 N1, the guarded ids builder), so only their
//! capacity is checked here.

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
use crate::kernels::cuda::prefill::mmq_size::q8_1_bytes;

/// Mirror of `inference_cuda_mmq_quantize_args`, field for field.
#[repr(C)]
struct MmqQuantizeArgsRaw {
    x: *const c_void,
    ids: *const c_void,
    y: *mut c_void,
    ty: c_int,
    k: c_int,
    row_stride: c_int,
    ncols: c_int,
    packing: c_int,
}

/// Mirror of `inference_cuda_mmq_quantize_scatter_args`.
#[repr(C)]
struct MmqQuantizeScatterArgsRaw {
    x: *const c_void,
    ids_src1_inv: *const c_void,
    y: *mut c_void,
    ty: c_int,
    k: c_int,
    row_stride: c_int,
    m: c_int,
    top_k: c_int,
    packing: c_int,
}

const _: () = assert!(
    size_of::<MmqQuantizeArgsRaw>() == 48
        && offset_of!(MmqQuantizeArgsRaw, ids) == 8
        && offset_of!(MmqQuantizeArgsRaw, y) == 16
        && offset_of!(MmqQuantizeArgsRaw, ty) == 24
        && offset_of!(MmqQuantizeArgsRaw, k) == 28
        && offset_of!(MmqQuantizeArgsRaw, row_stride) == 32
        && offset_of!(MmqQuantizeArgsRaw, ncols) == 36
        && offset_of!(MmqQuantizeArgsRaw, packing) == 40
);

const _: () = assert!(
    size_of::<MmqQuantizeScatterArgsRaw>() == 48
        && offset_of!(MmqQuantizeScatterArgsRaw, ids_src1_inv) == 8
        && offset_of!(MmqQuantizeScatterArgsRaw, y) == 16
        && offset_of!(MmqQuantizeScatterArgsRaw, ty) == 24
        && offset_of!(MmqQuantizeScatterArgsRaw, k) == 28
        && offset_of!(MmqQuantizeScatterArgsRaw, row_stride) == 32
        && offset_of!(MmqQuantizeScatterArgsRaw, m) == 36
        && offset_of!(MmqQuantizeScatterArgsRaw, top_k) == 40
        && offset_of!(MmqQuantizeScatterArgsRaw, packing) == 44
);

unsafe extern "C" {
    #[link_name = "inference_cuda_mmq_quantize"]
    fn mmq_quantize_raw(
        args: *const MmqQuantizeArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_mmq_quantize_scatter"]
    fn mmq_quantize_scatter_raw(
        args: *const MmqQuantizeScatterArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// The converted integers of one quantize call, bounded and converted by the wrapper's
/// rule checks, so no conversion can fail inside a launch closure.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct QuantInts {
    pub(in crate::kernels::cuda) packing: c_int,
    pub(in crate::kernels::cuda) ty: c_int,
    pub(in crate::kernels::cuda) k: c_int,
    pub(in crate::kernels::cuda) row_stride: c_int,
    pub(in crate::kernels::cuda) ncols: c_int,
}

/// The converted integers of one scatter call.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct ScatterInts {
    pub(in crate::kernels::cuda) packing: c_int,
    pub(in crate::kernels::cuda) ty: c_int,
    pub(in crate::kernels::cuda) k: c_int,
    pub(in crate::kernels::cuda) row_stride: c_int,
    pub(in crate::kernels::cuda) m: c_int,
    pub(in crate::kernels::cuda) top_k: c_int,
}

/// The ranges of one quantize call plus its converted integers.
pub(in crate::kernels::cuda) struct MmqQuantizeRaw<'a> {
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) gather: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: QuantInts,
}

/// The ranges of one scatter call plus its converted integers.
pub(in crate::kernels::cuda) struct MmqQuantizeScatterRaw<'a> {
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) inv: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: ScatterInts,
}

/// `n` as `usize`, or an invalid-argument failure.
fn size(n: c_int) -> Result<usize, DeviceFailure> {
    usize::try_from(n).map_err(|_| invalid_argument())
}

/// Proves a quantize call before the native call: the C sizes (source
/// `(ncols - 1) * row_stride + k` floats, map `ncols` ints, `y` `q8_1_bytes` with the pad),
/// the 16/4-byte alignments, and that `y` shares no byte with an input.
fn check_quantize(io: &MmqQuantizeRaw<'_>) -> Result<(), DeviceFailure> {
    let ints = io.ints;
    let (k, stride, ncols) = (size(ints.k)?, size(ints.row_stride)?, size(ints.ncols)?);
    let x_need = ncols
        .checked_sub(1)
        .and_then(|last| last.checked_mul(stride))
        .and_then(|lead| lead.checked_add(k))
        .and_then(|floats| floats.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    let y_need = q8_1_bytes(k, ncols).map_err(|_| invalid_argument())?;
    let sized = io.x.bytes() >= x_need
        && io.y.bytes() >= y_need
        && io
            .gather
            .as_ref()
            .is_none_or(|map| map.bytes() >= ncols * 4);
    let aligned = io.x.offset().is_multiple_of(16)
        && io.y.offset().is_multiple_of(16)
        && io
            .gather
            .as_ref()
            .is_none_or(|map| map.offset().is_multiple_of(4));
    let out = Span::of(&io.y, y_need);
    let apart = out.apart(&Span::of(&io.x, x_need))
        && io
            .gather
            .as_ref()
            .is_none_or(|map| out.apart(&Span::of(map, ncols * 4)));
    if !(sized && aligned && apart) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Proves a scatter call the same way (inverse map `m * top_k` ints).
fn check_scatter(io: &MmqQuantizeScatterRaw<'_>) -> Result<(), DeviceFailure> {
    let ints = io.ints;
    let (k, stride) = (size(ints.k)?, size(ints.row_stride)?);
    let (m, top_k) = (size(ints.m)?, size(ints.top_k)?);
    let ncols = m.checked_mul(top_k).ok_or_else(invalid_argument)?;
    let x_need = m
        .checked_sub(1)
        .and_then(|last| last.checked_mul(stride))
        .and_then(|lead| lead.checked_add(k))
        .and_then(|floats| floats.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    let y_need = q8_1_bytes(k, ncols).map_err(|_| invalid_argument())?;
    let sized = io.x.bytes() >= x_need && io.inv.bytes() >= ncols * 4 && io.y.bytes() >= y_need;
    let aligned = io.x.offset().is_multiple_of(16)
        && io.y.offset().is_multiple_of(16)
        && io.inv.offset().is_multiple_of(4);
    let out = Span::of(&io.y, y_need);
    let apart = out.apart(&Span::of(&io.x, x_need)) && out.apart(&Span::of(&io.inv, ncols * 4));
    if !(sized && aligned && apart) {
        return Err(invalid_argument());
    }
    Ok(())
}

fn c_quantize(io: &MmqQuantizeRaw<'_>) -> MmqQuantizeArgsRaw {
    MmqQuantizeArgsRaw {
        x: io.x.address().cast_const(),
        ids: io
            .gather
            .as_ref()
            .map_or(std::ptr::null(), |map| map.address().cast_const()),
        y: io.y.address(),
        ty: io.ints.ty,
        k: io.ints.k,
        row_stride: io.ints.row_stride,
        ncols: io.ints.ncols,
        packing: io.ints.packing,
    }
}

fn c_scatter(io: &MmqQuantizeScatterRaw<'_>) -> MmqQuantizeScatterArgsRaw {
    MmqQuantizeScatterArgsRaw {
        x: io.x.address().cast_const(),
        ids_src1_inv: io.inv.address().cast_const(),
        y: io.y.address(),
        ty: io.ints.ty,
        k: io.ints.k,
        row_stride: io.ints.row_stride,
        m: io.ints.m,
        top_k: io.ints.top_k,
        packing: io.ints.packing,
    }
}

/// Queues the (optionally gathered) quantize on `stream`.
pub(in crate::kernels::cuda) fn mmq_quantize_launch(
    stream: &RawStream,
    io: MmqQuantizeRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_quantize(&io)?;
    let args = c_quantize(&io);
    let mut stage = STAGE_NONE;
    // SAFETY: `check_quantize` proved every range inside its live allocation, large enough
    // for the call, aligned, and that `y` is disjoint from the source and the map, so each
    // pointer in `args` addresses bytes the kernel may touch; the C entry repeats these
    // checks on the addresses. The allocations stay valid until the queued work finishes
    // because `inference_cuda_free` synchronizes the device first. `args` lives across the
    // call and the entry copies it before returning. `stream` is live for the call.
    let status = unsafe { mmq_quantize_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Queues the scatter quantize on `stream`.
pub(in crate::kernels::cuda) fn mmq_quantize_scatter_launch(
    stream: &RawStream,
    io: MmqQuantizeScatterRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_scatter(&io)?;
    let args = c_scatter(&io);
    let mut stage = STAGE_NONE;
    // SAFETY: as `mmq_quantize_launch`, with the inverse map in place of the gather.
    let status = unsafe { mmq_quantize_scatter_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}
