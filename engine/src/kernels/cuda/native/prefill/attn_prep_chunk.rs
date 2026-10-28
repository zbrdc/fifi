//! Raw FFI to the chunk attention preparation entry of `native/prefill/attn_prep_chunk.cu`
//! (declared in `native/prefill/attn_prep_chunk.h`), CPF-ATTN-A K1. Child module of
//! `native/prefill.rs`, so it inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! The math is a byte-for-byte re-expression of the vendored `attn_prep` with `p0` a launch
//! argument (decision 0049 items 1 and 2); the wrapper's tests pin the bit equality.
//!
//! Safety invariants this file relies on:
//! - The launch entry is called only through `CudaContext::launch_on`, on the thread that
//!   owns the context and under the dispatch guard (decisions 0023 and 0024). It only
//!   enqueues and returns; the stream is live for the call.
//! - Every range is a `RawRange`, proved inside a live allocation by `raw_range`. Sizes,
//!   alignment and disjointness (`super::Span`) are proved again here, on the extents the
//!   kernel touches, before the call; the C entry repeats the pointer, alignment and shape
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

/// Mode of the KV outputs: append the normed and rotated rows to the fp16 cache planes.
pub(in crate::kernels::cuda) const MODE_APPEND_FP16: c_int = 0;
/// Mode of the KV outputs: write the post-rope k rows and the unchanged v rows as f32, the
/// input contract of `kv8_append`'s chunk entry (IMPL-KV8-A).
pub(in crate::kernels::cuda) const MODE_ROWS_OUT: c_int = 1;

/// Mirror of `inference_cuda_attn_prep_chunk_args` in `attn_prep_chunk.h`, field for field.
#[repr(C)]
struct AttnPrepChunkArgsRaw {
    proj: *const c_void,
    q_norm: *const c_void,
    k_norm: *const c_void,
    k_cache: *mut c_void,
    v_cache: *mut c_void,
    q: *mut c_void,
    k_rows: *mut c_void,
    v_rows: *mut c_void,
    status: *mut c_void,
    ld: c_int,
    p0: c_int,
    m: c_int,
    n_head: c_int,
    n_kv_head: c_int,
    head_dim: c_int,
    n_rot: c_int,
    rows: c_int,
    mode: c_int,
    eps: f32,
    rope_base: f32,
}

const _: () = assert!(
    size_of::<AttnPrepChunkArgsRaw>() == 120
        && offset_of!(AttnPrepChunkArgsRaw, proj) == 0
        && offset_of!(AttnPrepChunkArgsRaw, k_cache) == 24
        && offset_of!(AttnPrepChunkArgsRaw, q) == 40
        && offset_of!(AttnPrepChunkArgsRaw, k_rows) == 48
        && offset_of!(AttnPrepChunkArgsRaw, status) == 64
        && offset_of!(AttnPrepChunkArgsRaw, ld) == 72
        && offset_of!(AttnPrepChunkArgsRaw, p0) == 76
        && offset_of!(AttnPrepChunkArgsRaw, m) == 80
        && offset_of!(AttnPrepChunkArgsRaw, head_dim) == 92
        && offset_of!(AttnPrepChunkArgsRaw, rows) == 100
        && offset_of!(AttnPrepChunkArgsRaw, mode) == 104
        && offset_of!(AttnPrepChunkArgsRaw, eps) == 108
);

unsafe extern "C" {
    #[link_name = "inference_cuda_attn_prep_chunk_launch"]
    fn attn_prep_chunk_launch_raw(
        args: *const AttnPrepChunkArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

#[cfg(all(test, feature = "cuda-fault-qualification"))]
unsafe extern "C" {
    #[link_name = "inference_cuda_attn_prep_chunk_fault_bad_bounds_launch"]
    fn attn_prep_chunk_fault_bad_bounds_raw(
        args: *const AttnPrepChunkArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Every integer argument of the launch, already bounded and converted by the wrapper before
/// `launch_on`, so no conversion can fail inside the launch closure.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct AttnPrepChunkInts {
    pub(in crate::kernels::cuda) ld: c_int,
    pub(in crate::kernels::cuda) p0: c_int,
    pub(in crate::kernels::cuda) m: c_int,
    pub(in crate::kernels::cuda) n_head: c_int,
    pub(in crate::kernels::cuda) n_kv_head: c_int,
    pub(in crate::kernels::cuda) head_dim: c_int,
    pub(in crate::kernels::cuda) n_rot: c_int,
    pub(in crate::kernels::cuda) rows: c_int,
    pub(in crate::kernels::cuda) mode: c_int,
}

/// The KV outputs one launch writes, by mode. The enum is the mode/NULL consistency rule: the
/// unused pair is absent, so the pointers handed to C are NULL exactly where the mode leaves
/// them and the C entry's consistency check can never fire through this module.
pub(in crate::kernels::cuda) enum PrepChunkOut<'a> {
    AppendFp16 {
        k_cache: RawRange<'a>,
        v_cache: RawRange<'a>,
    },
    RowsOut {
        k_rows: RawRange<'a>,
        v_rows: RawRange<'a>,
    },
}

/// What one chunk preparation launch reads and writes: the converted ranges, the mode's
/// outputs, the optional prefill status word, the integers and the rope scalars.
pub(in crate::kernels::cuda) struct AttnPrepChunkRaw<'a> {
    pub(in crate::kernels::cuda) proj: RawRange<'a>,
    pub(in crate::kernels::cuda) q_norm: RawRange<'a>,
    pub(in crate::kernels::cuda) k_norm: RawRange<'a>,
    pub(in crate::kernels::cuda) q: RawRange<'a>,
    pub(in crate::kernels::cuda) status: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) out: PrepChunkOut<'a>,
    pub(in crate::kernels::cuda) ints: AttnPrepChunkInts,
    pub(in crate::kernels::cuda) eps: f32,
    pub(in crate::kernels::cuda) rope_base: f32,
}

/// The bytes each range must hold.
struct Needs {
    proj: usize,
    norm: usize,
    q: usize,
    kv: usize,
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

fn needs(io: &AttnPrepChunkRaw<'_>) -> Result<Needs, DeviceFailure> {
    let ints = &io.ints;
    let (m, heads, kv_heads) = (size(ints.m)?, size(ints.n_head)?, size(ints.n_kv_head)?);
    let (dim, ld, rows) = (size(ints.head_dim)?, size(ints.ld)?, size(ints.rows)?);
    let row_width = product(&[heads * 2 + kv_heads * 2, dim])?;
    let proj = product(&[m.saturating_sub(1), ld, 4])?
        .checked_add(product(&[row_width, 4])?)
        .ok_or_else(invalid_argument)?;
    let per_mode = match io.out {
        PrepChunkOut::AppendFp16 { .. } => product(&[rows, kv_heads, dim, 2])?,
        PrepChunkOut::RowsOut { .. } => product(&[m, kv_heads, dim, 4])?,
    };
    Ok(Needs {
        proj,
        norm: product(&[dim, 4])?,
        q: product(&[m, heads, dim, 4])?,
        kv: per_mode,
    })
}

/// Proves a launch before the native call: sizes and alignment of every range (`q` and the
/// fp16 caches 16-byte; `proj`, the norms, the f32 rows of the RowsOut mode and the status
/// word 4-byte) and that every range the kernel writes (`q`, the mode's KV outputs, `status`)
/// shares no byte with any other range of the call.
fn check_prep_chunk(io: &AttnPrepChunkRaw<'_>) -> Result<(), DeviceFailure> {
    let need = needs(io)?;
    let norm = need.norm;
    let (kv_a, kv_b) = match &io.out {
        PrepChunkOut::AppendFp16 { k_cache, v_cache } => (k_cache, v_cache),
        PrepChunkOut::RowsOut { k_rows, v_rows } => (k_rows, v_rows),
    };
    let kv_align = match io.out {
        PrepChunkOut::AppendFp16 { .. } => 16,
        PrepChunkOut::RowsOut { .. } => 4,
    };
    let sized = io.proj.bytes() >= need.proj
        && io.q_norm.bytes() >= norm
        && io.k_norm.bytes() >= norm
        && io.q.bytes() >= need.q
        && kv_a.bytes() >= need.kv
        && kv_b.bytes() >= need.kv
        && io.status.as_ref().is_none_or(|s| s.bytes() >= 4);
    let aligned = io.proj.offset().is_multiple_of(4)
        && io.q_norm.offset().is_multiple_of(4)
        && io.k_norm.offset().is_multiple_of(4)
        && io.q.offset().is_multiple_of(16)
        && kv_a.offset().is_multiple_of(kv_align)
        && kv_b.offset().is_multiple_of(kv_align)
        && io
            .status
            .as_ref()
            .is_none_or(|s| s.offset().is_multiple_of(4));
    // `q`, both KV outputs of the mode and the status word are written: each is apart from
    // every other range of the call; the reads may overlap each other.
    let mut written = vec![
        Span::of(&io.q, need.q),
        Span::of(kv_a, need.kv),
        Span::of(kv_b, need.kv),
    ];
    if let Some(status) = &io.status {
        written.push(Span::of(status, 4));
    }
    let reads = [
        Span::of(&io.proj, need.proj),
        Span::of(&io.q_norm, norm),
        Span::of(&io.k_norm, norm),
    ];
    let apart = written.iter().enumerate().all(|(i, claim)| {
        written[i + 1..].iter().all(|other| claim.apart(other))
            && reads.iter().all(|read| claim.apart(read))
    });
    if !(sized && aligned && apart) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// The C mirror of `io`, with NULL exactly where the mode leaves a pointer unused.
fn c_args(io: &AttnPrepChunkRaw<'_>) -> AttnPrepChunkArgsRaw {
    let (k_cache, v_cache, k_rows, v_rows) = match &io.out {
        PrepChunkOut::AppendFp16 { k_cache, v_cache } => (
            k_cache.address(),
            v_cache.address(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ),
        PrepChunkOut::RowsOut { k_rows, v_rows } => (
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            k_rows.address(),
            v_rows.address(),
        ),
    };
    AttnPrepChunkArgsRaw {
        proj: io.proj.address().cast_const(),
        q_norm: io.q_norm.address().cast_const(),
        k_norm: io.k_norm.address().cast_const(),
        k_cache,
        v_cache,
        q: io.q.address(),
        k_rows,
        v_rows,
        status: io
            .status
            .as_ref()
            .map_or(std::ptr::null_mut(), |s| s.address()),
        ld: io.ints.ld,
        p0: io.ints.p0,
        m: io.ints.m,
        n_head: io.ints.n_head,
        n_kv_head: io.ints.n_kv_head,
        head_dim: io.ints.head_dim,
        n_rot: io.ints.n_rot,
        rows: io.ints.rows,
        mode: io.ints.mode,
        eps: io.eps,
        rope_base: io.rope_base,
    }
}

/// Queues the chunk preparation kernel on `stream`: per-head RMSNorm and rope of the q and k
/// rows at the positions `p0 .. p0 + m`, the f32 `q` output, and the k/v rows written to the
/// fp16 cache planes or to f32 rows by the mode. Returns when the kernel is queued.
pub(in crate::kernels::cuda) fn attn_prep_chunk_launch(
    stream: &RawStream,
    io: AttnPrepChunkRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_prep_chunk(&io)?;
    let args = c_args(&io);
    let mut stage = STAGE_NONE;
    // SAFETY: `check_prep_chunk` proved every range inside its live allocation, large enough
    // for the call, aligned, and that every written range (`q`, the mode's KV outputs, the
    // status word) is disjoint from every other range, so each pointer in `args` addresses
    // bytes the kernel may touch; the C entry repeats these checks on the addresses. The
    // allocations stay valid until the queued work finishes because `inference_cuda_free`
    // synchronizes the device first. `args` lives across the call and the entry copies it
    // before returning. `stream` is live for the call.
    let status = unsafe { attn_prep_chunk_launch_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Queues the same kernel with a deliberately violated p0 bound, so the kernel's own device
/// re-check (decision 0049 item 2) is the code under test; every other rule of the call
/// holds, so the guard's NaN-q write touches valid memory. Qualified fault path only.
#[cfg(all(test, feature = "cuda-fault-qualification"))]
pub(in crate::kernels::cuda) fn attn_prep_chunk_fault_bad_bounds(
    stream: &RawStream,
    io: AttnPrepChunkRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_prep_chunk(&io)?;
    let mut args = c_args(&io);
    args.p0 = io.ints.rows; // rows + m > rows: the bound violation, p0 still a valid int
    let mut stage = STAGE_NONE;
    // SAFETY: as `attn_prep_chunk_launch`; the guard refuses before any KV access, and the
    // NaN-q write stays inside the proved `q` range.
    let status =
        unsafe { attn_prep_chunk_fault_bad_bounds_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}
