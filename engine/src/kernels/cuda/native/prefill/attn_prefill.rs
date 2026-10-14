//! Raw FFI to the chunked-prefill attention entry of `native/prefill/attn_prefill.cu`
//! (declared in `native/prefill/attn_prefill.h`), CPF-ATTN-A K2. Child module of
//! `native/prefill.rs`, so it inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this file relies on:
//! - The launch entry is called only through `CudaContext::launch_on`, on the thread that
//!   owns the context and under the dispatch guard (decisions 0023 and 0024). It only
//!   enqueues (two kernels) and returns; the stream is live for the call.
//! - Every range is a `RawRange`, proved inside a live allocation by `raw_range`. Sizes,
//!   alignment and disjointness (`super::Span`) are proved again here, on the extents the
//!   kernels touch, before the call; the C entry repeats the pointer, alignment and shape
//!   checks on the addresses.
//! - The C argument struct is built in one `unsafe` block that cites those checks.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes
//!   the device before `cudaFree`.
//! - The `extern` declaration, the `repr(C)` mirror and every raw pointer are private to this
//!   file. Nothing is re-exported.

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

/// The fp16 KV mode of `attn_prefill.h` (the KV8 loader arrives with IMPL-KV8-A).
pub(in crate::kernels::cuda) const KV_FP16: c_int = 0;

/// Mirror of `inference_cuda_attn_prefill_args` in `attn_prefill.h`, field for field.
#[repr(C)]
struct AttnPrefillArgsRaw {
    q: *const c_void,
    proj: *const c_void,
    k_cache: *const c_void,
    v_cache: *const c_void,
    kv8_k_codes: *const c_void,
    kv8_k_scales: *const c_void,
    kv8_v_codes: *const c_void,
    kv8_v_scales: *const c_void,
    part: *mut c_void,
    out: *mut c_void,
    status: *mut c_void,
    ld: c_int,
    p0: c_int,
    m: c_int,
    n_head: c_int,
    n_kv_head: c_int,
    head_dim: c_int,
    cache_rows: c_int,
    max_splits: c_int,
    min_keys: c_int,
    window: c_int,
    ring: c_int,
    kv_mode: c_int,
    kv8_capacity: c_int,
    scale: f32,
}

const _: () = assert!(
    size_of::<AttnPrefillArgsRaw>() == 144
        && offset_of!(AttnPrefillArgsRaw, k_cache) == 16
        && offset_of!(AttnPrefillArgsRaw, kv8_k_codes) == 32
        && offset_of!(AttnPrefillArgsRaw, part) == 64
        && offset_of!(AttnPrefillArgsRaw, out) == 72
        && offset_of!(AttnPrefillArgsRaw, status) == 80
        && offset_of!(AttnPrefillArgsRaw, ld) == 88
        && offset_of!(AttnPrefillArgsRaw, p0) == 92
        && offset_of!(AttnPrefillArgsRaw, m) == 96
        && offset_of!(AttnPrefillArgsRaw, head_dim) == 108
        && offset_of!(AttnPrefillArgsRaw, cache_rows) == 112
        && offset_of!(AttnPrefillArgsRaw, max_splits) == 116
        && offset_of!(AttnPrefillArgsRaw, window) == 124
        && offset_of!(AttnPrefillArgsRaw, kv_mode) == 132
        && offset_of!(AttnPrefillArgsRaw, scale) == 140
);

unsafe extern "C" {
    #[link_name = "inference_cuda_attn_prefill_part_floats"]
    fn attn_prefill_part_floats_raw(
        m: c_int,
        n_kv_head: c_int,
        head_dim: c_int,
        max_splits: c_int,
    ) -> usize;
    #[link_name = "inference_cuda_attn_prefill_launch"]
    fn attn_prefill_launch_raw(
        args: *const AttnPrefillArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

#[cfg(all(test, feature = "cuda-fault-qualification"))]
unsafe extern "C" {
    #[link_name = "inference_cuda_attn_prefill_fault_bad_bounds_launch"]
    fn attn_prefill_fault_bad_bounds_raw(
        args: *const AttnPrefillArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Every integer argument of the launch, already bounded and converted by the wrapper before
/// `launch_on`, so no conversion can fail inside the launch closure. `ld` is 0 without
/// `proj`.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct AttnPrefillInts {
    pub(in crate::kernels::cuda) ld: c_int,
    pub(in crate::kernels::cuda) p0: c_int,
    pub(in crate::kernels::cuda) m: c_int,
    pub(in crate::kernels::cuda) n_head: c_int,
    pub(in crate::kernels::cuda) n_kv_head: c_int,
    pub(in crate::kernels::cuda) head_dim: c_int,
    pub(in crate::kernels::cuda) cache_rows: c_int,
    pub(in crate::kernels::cuda) max_splits: c_int,
    pub(in crate::kernels::cuda) min_keys: c_int,
    pub(in crate::kernels::cuda) window: c_int,
    pub(in crate::kernels::cuda) ring: c_int,
}

/// What one prefill attention launch reads and writes: the converted ranges, the optional
/// projection (the output gate) and prefill status word, the integers and the score scale.
pub(in crate::kernels::cuda) struct AttnPrefillRaw<'a> {
    pub(in crate::kernels::cuda) q: RawRange<'a>,
    pub(in crate::kernels::cuda) proj: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) k_cache: RawRange<'a>,
    pub(in crate::kernels::cuda) v_cache: RawRange<'a>,
    pub(in crate::kernels::cuda) part: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) status: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) ints: AttnPrefillInts,
    pub(in crate::kernels::cuda) scale: f32,
}

/// The bytes each range must hold and the alignment of its start.
struct Needs {
    q: usize,
    proj: usize,
    cache: usize,
    part: usize,
    out: usize,
}

/// `n` as `usize`, or an invalid-argument failure for a negative value.
fn size(n: c_int) -> Result<usize, DeviceFailure> {
    usize::try_from(n).map_err(|_| invalid_argument())
}

/// Product of the factors, or an invalid-argument failure on overflow.
fn product(factors: &[usize]) -> Result<usize, DeviceFailure> {
    factors
        .iter()
        .try_fold(1usize, |acc, &factor| acc.checked_mul(factor))
        .ok_or_else(invalid_argument)
}

fn needs(io: &AttnPrefillRaw<'_>) -> Result<Needs, DeviceFailure> {
    let ints = &io.ints;
    let (m, heads, kv_heads) = (size(ints.m)?, size(ints.n_head)?, size(ints.n_kv_head)?);
    let (dim, ld, rows) = (size(ints.head_dim)?, size(ints.ld)?, size(ints.cache_rows)?);
    // SAFETY: a pure function of its four integers; it reads no memory and needs no device.
    let part_floats = unsafe {
        attn_prefill_part_floats_raw(ints.m, ints.n_kv_head, ints.head_dim, ints.max_splits)
    };
    let row_floats = product(&[heads, dim])?;
    let proj = if io.proj.is_some() {
        let last_row = product(&[m.saturating_sub(1), ld, 4])?;
        last_row
            .checked_add(product(&[row_floats, 2, 4])?)
            .ok_or_else(invalid_argument)?
    } else {
        0
    };
    Ok(Needs {
        q: product(&[m, row_floats, 4])?,
        proj,
        cache: product(&[rows, kv_heads, dim, 2])?,
        part: product(&[part_floats, 4])?,
        out: product(&[m, row_floats, 4])?,
    })
}

/// Proves a launch before the native call: sizes and alignment of every range (q, the caches,
/// part and out 16-byte; proj and the status word 4-byte) and that `out` and `part`, which
/// the kernels write, share no byte with any other range of the call.
fn check_attn(io: &AttnPrefillRaw<'_>) -> Result<(), DeviceFailure> {
    let need = needs(io)?;
    let sized = io.q.bytes() >= need.q
        && io.k_cache.bytes() >= need.cache
        && io.v_cache.bytes() >= need.cache
        && io.part.bytes() >= need.part
        && io.out.bytes() >= need.out
        && io
            .proj
            .as_ref()
            .is_none_or(|proj| proj.bytes() >= need.proj)
        && io.status.as_ref().is_none_or(|s| s.bytes() >= 4);
    let aligned = io.q.offset().is_multiple_of(16)
        && io.k_cache.offset().is_multiple_of(16)
        && io.v_cache.offset().is_multiple_of(16)
        && io.part.offset().is_multiple_of(16)
        && io.out.offset().is_multiple_of(16)
        && io
            .proj
            .as_ref()
            .is_none_or(|proj| proj.offset().is_multiple_of(4))
        && io
            .status
            .as_ref()
            .is_none_or(|s| s.offset().is_multiple_of(4));
    let out = Span::of(&io.out, need.out);
    let part = Span::of(&io.part, need.part);
    let mut reads = vec![
        Span::of(&io.q, need.q),
        Span::of(&io.k_cache, need.cache),
        Span::of(&io.v_cache, need.cache),
    ];
    if let Some(proj) = &io.proj {
        reads.push(Span::of(proj, need.proj));
    }
    let apart = out.apart(&part) && reads.iter().all(|read| out.apart(read) && part.apart(read));
    if !(sized && aligned && apart) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// The C mirror of `io`, with the kv8 fields NULL and 0 (the fp16 mode of this round).
fn c_args(io: &AttnPrefillRaw<'_>) -> AttnPrefillArgsRaw {
    let ints = &io.ints;
    AttnPrefillArgsRaw {
        q: io.q.address().cast_const(),
        proj: io
            .proj
            .as_ref()
            .map_or(std::ptr::null(), |proj| proj.address().cast_const()),
        k_cache: io.k_cache.address().cast_const(),
        v_cache: io.v_cache.address().cast_const(),
        kv8_k_codes: std::ptr::null(),
        kv8_k_scales: std::ptr::null(),
        kv8_v_codes: std::ptr::null(),
        kv8_v_scales: std::ptr::null(),
        part: io.part.address(),
        out: io.out.address(),
        status: io
            .status
            .as_ref()
            .map_or(std::ptr::null_mut(), |s| s.address()),
        ld: ints.ld,
        p0: ints.p0,
        m: ints.m,
        n_head: ints.n_head,
        n_kv_head: ints.n_kv_head,
        head_dim: ints.head_dim,
        cache_rows: ints.cache_rows,
        max_splits: ints.max_splits,
        min_keys: ints.min_keys,
        window: ints.window,
        ring: ints.ring,
        kv_mode: KV_FP16,
        kv8_capacity: 0,
        scale: io.scale,
    }
}

/// Queues the two prefill attention kernels (partials, then the fixed-order combine) on
/// `stream`. Returns when both are queued.
pub(in crate::kernels::cuda) fn attn_prefill_launch(
    stream: &RawStream,
    io: AttnPrefillRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_attn(&io)?;
    let args = c_args(&io);
    let mut stage = STAGE_NONE;
    // SAFETY: `check_attn` proved every range inside its live allocation, large enough for
    // the call, aligned, and that `out` and `part` are disjoint from every other range, so
    // each pointer in `args` addresses bytes the kernels may touch; the C entry repeats these
    // checks on the addresses. The allocations stay valid until the queued work finishes
    // because `inference_cuda_free` synchronizes the device first. `args` lives across the
    // call and the entry copies it before returning. `stream` is live for the call.
    let status = unsafe { attn_prefill_launch_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Queues the same two kernels with a deliberately violated p0 bound, so the kernels' own
/// device re-checks (decision 0049 item 2: no partial, NaN out, mask 8) are the code under
/// test; every other rule of the call holds. Qualified fault path only.
#[cfg(all(test, feature = "cuda-fault-qualification"))]
pub(in crate::kernels::cuda) fn attn_prefill_fault_bad_bounds(
    stream: &RawStream,
    io: AttnPrefillRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_attn(&io)?;
    let mut args = c_args(&io);
    args.p0 = io.ints.cache_rows; // cache_rows + m > cache_rows: the bound violation
    let mut stage = STAGE_NONE;
    // SAFETY: as `attn_prefill_launch`; the part kernel refuses before any partial write and
    // the combine's NaN write stays inside the proved `out` range.
    let status = unsafe { attn_prefill_fault_bad_bounds_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}
