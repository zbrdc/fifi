//! Raw FFI to the mmid entries of `native/prefill/mmq_mmid.cu` (declared in
//! `native/prefill/mmq_mmid.h`), CPF-V4 G3. Child module of `native/prefill.rs`, so it
//! inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this file relies on:
//! - Every launch entry is called only through `CudaContext::launch_on`, on the thread that
//!   owns the context and under the dispatch guard (decisions 0023 and 0024). It only enqueues
//!   and returns; the stream is live for the call.
//! - Every range is a `RawRange`, proved inside a live allocation by `raw_range`. Sizes,
//!   alignment and disjointness (`super::Span`) are proved again here, on the extents the
//!   kernels touch, before the call; the C entry repeats the pointer, alignment, shape and
//!   disjointness checks on the addresses.
//! - Each C argument struct is built in one `unsafe` block that cites those checks.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes
//!   the device before `cudaFree`.
//! - The `extern` declarations, the `repr(C)` mirrors and every raw pointer are private to
//!   this file. Nothing is re-exported.

// Test builds reach the whole launch through the device tests of `prefill::mmq_mmid`.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "staged for CPF-7, the prefill driver (decision 0049)"
    )
)]

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use crate::kernels::cuda::prefill::mmq_size::{
    expert_bounds_ints, fixup_floats_max, ids_dst_ints, q8_1_bytes,
};

/// Mirror of `inference_cuda_mmq_ids_guard_args` in `mmq_mmid.h`, field for field.
#[repr(C)]
struct IdsGuardArgsRaw {
    ids: *const c_void,
    ids_g: *mut c_void,
    status: *mut c_void,
    n_experts: c_int,
    m: c_int,
    top_k: c_int,
}

const _: () = assert!(
    size_of::<IdsGuardArgsRaw>() == 40
        && offset_of!(IdsGuardArgsRaw, status) == 16
        && offset_of!(IdsGuardArgsRaw, n_experts) == 24
        && offset_of!(IdsGuardArgsRaw, top_k) == 32
);

/// Mirror of `inference_cuda_mmq_build_ids_args` in `mmq_mmid.h`, field for field.
#[repr(C)]
struct BuildIdsArgsRaw {
    ids: *const c_void,
    ids_g: *mut c_void,
    ids_src1: *mut c_void,
    ids_dst: *mut c_void,
    expert_bounds: *mut c_void,
    status: *mut c_void,
    n_experts: c_int,
    m: c_int,
    top_k: c_int,
}

const _: () = assert!(
    size_of::<BuildIdsArgsRaw>() == 64
        && offset_of!(BuildIdsArgsRaw, expert_bounds) == 32
        && offset_of!(BuildIdsArgsRaw, status) == 40
        && offset_of!(BuildIdsArgsRaw, n_experts) == 48
        && offset_of!(BuildIdsArgsRaw, top_k) == 56
);

/// The vendored `mmq_args` geometry of one grouped launch in its field order (mmq.cuh:1326-
/// 1333), as `fill_mmid_args` fills it: the tail of `inference_cuda_mmq_mmid_args`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::kernels::cuda) struct MmidGeometry {
    pub(in crate::kernels::cuda) type_x: i64,
    pub(in crate::kernels::cuda) ncols_x: i64,
    pub(in crate::kernels::cuda) nrows_x: i64,
    pub(in crate::kernels::cuda) ncols_dst: i64,
    pub(in crate::kernels::cuda) stride_row_x: i64,
    pub(in crate::kernels::cuda) ncols_y: i64,
    pub(in crate::kernels::cuda) nrows_dst: i64,
    pub(in crate::kernels::cuda) nchannels_x: i64,
    pub(in crate::kernels::cuda) nchannels_y: i64,
    pub(in crate::kernels::cuda) stride_channel_x: i64,
    pub(in crate::kernels::cuda) stride_channel_y: i64,
    pub(in crate::kernels::cuda) stride_channel_dst: i64,
    pub(in crate::kernels::cuda) nsamples_x: i64,
    pub(in crate::kernels::cuda) nsamples_y: i64,
    pub(in crate::kernels::cuda) stride_sample_x: i64,
    pub(in crate::kernels::cuda) stride_sample_y: i64,
    pub(in crate::kernels::cuda) stride_sample_dst: i64,
    pub(in crate::kernels::cuda) ncols_max: i64,
    pub(in crate::kernels::cuda) ncols_opt: i64,
}

/// Mirror of `inference_cuda_mmq_mmid_args` in `mmq_mmid.h`: the ranges, their capacities,
/// then the geometry.
#[repr(C)]
struct MmidArgsRaw {
    x: *const c_void,
    y: *const c_void,
    ids_dst: *const c_void,
    expert_bounds: *const c_void,
    dst: *mut c_void,
    fixup: *mut c_void,
    x_bytes: usize,
    y_bytes: usize,
    ids_dst_bytes: usize,
    expert_bounds_bytes: usize,
    dst_bytes: usize,
    fixup_bytes: usize,
    geometry: MmidGeometry,
}

const _: () = assert!(
    size_of::<MmidArgsRaw>() == 248
        && offset_of!(MmidArgsRaw, fixup) == 40
        && offset_of!(MmidArgsRaw, x_bytes) == 48
        && offset_of!(MmidArgsRaw, fixup_bytes) == 88
        && offset_of!(MmidArgsRaw, geometry) == 96
        && offset_of!(MmidGeometry, nchannels_x) == 56
        && offset_of!(MmidGeometry, ncols_max) == 136
);

unsafe extern "C" {
    #[link_name = "inference_cuda_mmq_mmid_launch"]
    fn mmq_mmid_launch_raw(
        args: *const MmidArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_mmq_ids_guard"]
    fn mmq_ids_guard_raw(
        args: *const IdsGuardArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_mmq_build_ids"]
    fn mmq_build_ids_raw(
        args: *const BuildIdsArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// The id shape of a call, already bounded and converted by the wrapper, so no conversion can
/// fail inside the launch closure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::kernels::cuda) struct IdsShapeInts {
    pub(in crate::kernels::cuda) n_experts: c_int,
    pub(in crate::kernels::cuda) m: c_int,
    pub(in crate::kernels::cuda) top_k: c_int,
}

/// What one guard launch reads (`ids`) and writes (`ids_g`, the optional status word).
pub(in crate::kernels::cuda) struct IdsGuardRaw<'a> {
    pub(in crate::kernels::cuda) ids: RawRange<'a>,
    pub(in crate::kernels::cuda) ids_g: RawRange<'a>,
    pub(in crate::kernels::cuda) status: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) shape: IdsShapeInts,
}

/// What one builder launch reads and writes: the guard's ranges, then the three outputs of
/// `mm_ids_helper`.
pub(in crate::kernels::cuda) struct BuildIdsRaw<'a> {
    pub(in crate::kernels::cuda) guard: IdsGuardRaw<'a>,
    pub(in crate::kernels::cuda) ids_src1: RawRange<'a>,
    pub(in crate::kernels::cuda) ids_dst: RawRange<'a>,
    pub(in crate::kernels::cuda) expert_bounds: RawRange<'a>,
}

/// `n` as `usize`, or an invalid-argument failure for a negative value.
fn size(n: c_int) -> Result<usize, DeviceFailure> {
    usize::try_from(n).map_err(|_| invalid_argument())
}

/// Bytes of the `[m][top_k]` i32 id rows.
fn rows_bytes(shape: &IdsShapeInts) -> Result<usize, DeviceFailure> {
    size(shape.m)?
        .checked_mul(size(shape.top_k)?)
        .and_then(|ints| ints.checked_mul(4))
        .ok_or_else(invalid_argument)
}

/// True when every range is 4-byte aligned (allocations start 256-byte aligned, so the offset
/// decides the address).
fn aligned4(ranges: &[&RawRange<'_>]) -> bool {
    ranges.iter().all(|range| range.offset().is_multiple_of(4))
}

/// True when every written span shares no byte with any other span of the call.
fn written_apart(written: &[Span<'_>], read: &[Span<'_>]) -> bool {
    written.iter().enumerate().all(|(i, span)| {
        written[i + 1..].iter().all(|other| span.apart(other))
            && read.iter().all(|other| span.apart(other))
    })
}

/// Proves a guard launch: `ids` and `ids_g` hold the rows, the status word 4 bytes, all are
/// 4-byte aligned, and `ids_g` and the status word share no byte with each other or `ids`.
fn check_guard(io: &IdsGuardRaw<'_>) -> Result<(), DeviceFailure> {
    let rows = rows_bytes(&io.shape)?;
    let sized = io.ids.bytes() >= rows
        && io.ids_g.bytes() >= rows
        && io.status.as_ref().is_none_or(|s| s.bytes() >= 4);
    let aligned =
        aligned4(&[&io.ids, &io.ids_g]) && io.status.as_ref().is_none_or(|s| aligned4(&[s]));
    let mut written = vec![Span::of(&io.ids_g, rows)];
    written.extend(io.status.as_ref().map(|s| Span::of(s, 4)));
    if !(sized && aligned && written_apart(&written, &[Span::of(&io.ids, rows)])) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// The status pointer of the call: NULL when the call has none.
fn status_ptr(status: Option<&RawRange<'_>>) -> *mut c_void {
    status.map_or(std::ptr::null_mut(), RawRange::address)
}

/// Queues the ids guard N1 on `stream`: `ids_g` receives the router rows, every flagged row
/// rewritten to `0..top_k`, and the status word the masks of `mmq_mmid.h`. Returns when the
/// kernel is queued.
pub(in crate::kernels::cuda) fn mmq_ids_guard_launch(
    stream: &RawStream,
    io: IdsGuardRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_guard(&io)?;
    let args = IdsGuardArgsRaw {
        ids: io.ids.address().cast_const(),
        ids_g: io.ids_g.address(),
        status: status_ptr(io.status.as_ref()),
        n_experts: io.shape.n_experts,
        m: io.shape.m,
        top_k: io.shape.top_k,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_guard` proved every range inside its live allocation, large enough for
    // the m * top_k ints (the status word 4 bytes), 4-byte aligned, and that `ids_g` and the
    // status word are disjoint from each other and from `ids`; the C entry repeats these
    // checks on the addresses. The allocations stay valid until the queued work finishes
    // because `inference_cuda_free` synchronizes the device first. `args` lives across the
    // call and the entry copies it before returning. `stream` is live for the call.
    let status = unsafe { mmq_ids_guard_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Proves a builder launch: `ids`, `ids_g`, `ids_src1` and the written part of `ids_dst` hold
/// the m * top_k ints, `expert_bounds` n_experts + 1 ints, the status word 4 bytes; all are
/// 4-byte aligned, and every written range shares no byte with another range of the call.
fn check_build(io: &BuildIdsRaw<'_>) -> Result<(), DeviceFailure> {
    let guard = &io.guard;
    let rows = rows_bytes(&guard.shape)?;
    let bounds = size(guard.shape.n_experts)?
        .checked_add(1)
        .and_then(|ints| ints.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    let sized = [&guard.ids, &guard.ids_g, &io.ids_src1, &io.ids_dst]
        .iter()
        .all(|range| range.bytes() >= rows)
        && io.expert_bounds.bytes() >= bounds
        && guard.status.as_ref().is_none_or(|s| s.bytes() >= 4);
    let aligned = aligned4(&[
        &guard.ids,
        &guard.ids_g,
        &io.ids_src1,
        &io.ids_dst,
        &io.expert_bounds,
    ]) && guard.status.as_ref().is_none_or(|s| aligned4(&[s]));
    let mut written = vec![
        Span::of(&guard.ids_g, rows),
        Span::of(&io.ids_src1, rows),
        Span::of(&io.ids_dst, rows),
        Span::of(&io.expert_bounds, bounds),
    ];
    written.extend(guard.status.as_ref().map(|s| Span::of(s, 4)));
    if !(sized && aligned && written_apart(&written, &[Span::of(&guard.ids, rows)])) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the id builder on `stream`: the guard N1 into `ids_g`, then `mm_ids_helper` on
/// `ids_g` with the inverse map (contract in `mmq_mmid.h`). Returns when both are queued.
pub(in crate::kernels::cuda) fn mmq_build_ids_launch(
    stream: &RawStream,
    io: BuildIdsRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_build(&io)?;
    let guard = &io.guard;
    let args = BuildIdsArgsRaw {
        ids: guard.ids.address().cast_const(),
        ids_g: guard.ids_g.address(),
        ids_src1: io.ids_src1.address(),
        ids_dst: io.ids_dst.address(),
        expert_bounds: io.expert_bounds.address(),
        status: status_ptr(guard.status.as_ref()),
        n_experts: guard.shape.n_experts,
        m: guard.shape.m,
        top_k: guard.shape.top_k,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_build` proved every range inside its live allocation, large enough for
    // what the two kernels touch, 4-byte aligned, and every written range (`ids_g`, the three
    // helper outputs, the status word) disjoint from every other range of the call; the C
    // entry repeats these checks on the addresses. The allocations stay valid until the queued
    // work finishes because `inference_cuda_free` synchronizes the device first. `args` lives
    // across the call and the entry copies it before returning. `stream` is live for the call.
    let status = unsafe { mmq_build_ids_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// What one grouped launch reads (`x`, `y`, the two id ranges) and writes (`dst`, `fixup`), its
/// geometry, and the SM count of the device facts the wrapper read (the fixup range holds the
/// documented maximum of `nsm` slots).
pub(in crate::kernels::cuda) struct MmidLaunchRaw<'a> {
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) y: RawRange<'a>,
    pub(in crate::kernels::cuda) ids_dst: RawRange<'a>,
    pub(in crate::kernels::cuda) expert_bounds: RawRange<'a>,
    pub(in crate::kernels::cuda) dst: RawRange<'a>,
    pub(in crate::kernels::cuda) fixup: RawRange<'a>,
    pub(in crate::kernels::cuda) geometry: MmidGeometry,
    pub(in crate::kernels::cuda) nsm: c_int,
}

/// The bytes of each range of a grouped launch, in the order x, y, ids_dst, expert_bounds,
/// dst, fixup, from the geometry by the rules of `mmq_mmid.h` (CPF-V4 G1's sizing). `None`
/// for a negative field, an unglued type or an overflow.
fn launch_needs(g: &MmidGeometry, nsm: c_int) -> Option<[usize; 6]> {
    let int = |value: i64| usize::try_from(value).ok();
    let block = match g.type_x {
        12 => 144,
        14 => 210,
        _ => return None,
    };
    let (n, e, ne, m) = (
        int(g.nrows_x)?,
        int(g.nchannels_x)?,
        int(g.ncols_dst)?,
        int(g.ncols_max)?,
    );
    let blocks = (e.checked_sub(1)?.checked_mul(int(g.stride_channel_x)?)?)
        .checked_add(n.checked_mul(int(g.stride_row_x)?)?)?;
    let dst = (ne.checked_sub(1)?.checked_mul(int(g.nrows_dst)?)?).checked_add(n)?;
    Some([
        blocks.checked_mul(block)?,
        q8_1_bytes(int(g.ncols_x)?, ne).ok()?,
        ids_dst_ints(ne, m).ok()?.checked_mul(4)?,
        expert_bounds_ints(e).ok()?.checked_mul(4)?,
        dst.checked_mul(4)?,
        fixup_floats_max(int(i64::from(nsm))?)
            .ok()?
            .checked_mul(4)?,
    ])
}

/// Proves a grouped launch: every range holds its need and is aligned (x, y and fixup 16
/// bytes, the rest 4), and the written `dst` and `fixup` share no byte with each other or the
/// four read ranges.
fn check_mmid(io: &MmidLaunchRaw<'_>) -> Result<(), DeviceFailure> {
    let need = launch_needs(&io.geometry, io.nsm).ok_or_else(invalid_argument)?;
    let ranges = [
        (&io.x, 16),
        (&io.y, 16),
        (&io.ids_dst, 4),
        (&io.expert_bounds, 4),
        (&io.dst, 4),
        (&io.fixup, 16),
    ];
    let fits = ranges
        .iter()
        .zip(need)
        .all(|((range, align), bytes)| range.bytes() >= bytes && range.offset() % align == 0);
    let spans: Vec<Span<'_>> = ranges
        .iter()
        .zip(need)
        .map(|((range, _), bytes)| Span::of(range, bytes))
        .collect();
    if !(fits && written_apart(&spans[4..], &spans[..4])) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues one grouped GEMM on `stream`: the vendored `mul_mat_q_case` of the weight type with
/// the stream-k fixup when the launch needs it (contract in `mmq_mmid.h`). Returns when the
/// kernels are queued.
pub(in crate::kernels::cuda) fn mmq_mmid_launch(
    stream: &RawStream,
    io: MmidLaunchRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_mmid(&io)?;
    let args = MmidArgsRaw {
        x: io.x.address().cast_const(),
        y: io.y.address().cast_const(),
        ids_dst: io.ids_dst.address().cast_const(),
        expert_bounds: io.expert_bounds.address().cast_const(),
        dst: io.dst.address(),
        fixup: io.fixup.address(),
        x_bytes: io.x.bytes(),
        y_bytes: io.y.bytes(),
        ids_dst_bytes: io.ids_dst.bytes(),
        expert_bounds_bytes: io.expert_bounds.bytes(),
        dst_bytes: io.dst.bytes(),
        fixup_bytes: io.fixup.bytes(),
        geometry: io.geometry,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_mmid` proved every range inside its live allocation and large enough for
    // what the geometry lets the kernels touch, aligned, and the written `dst` and `fixup`
    // disjoint from each other and from the read ranges; the capacities passed are the range
    // sizes, and the C entry repeats the geometry, capacity, alignment and disjointness
    // checks. The id VALUES are an `MmidIds`'s, whose constructors prove them. The
    // allocations stay valid until the queued work finishes because `inference_cuda_free`
    // synchronizes the device first. `args` lives across the call and the entry copies it
    // before returning. `stream` is live for the call.
    let status = unsafe { mmq_mmid_launch_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
