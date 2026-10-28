//! Raw FFI to the split-KV decode attention entry of `native/resident/attn_decode.cu` (declared
//! in `native/resident/attn_decode.h`), RD-8b. Child module of `native/resident.rs`, so it
//! inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this file relies on:
//! - The launch entry is called only through `CudaContext::launch_on`, on the thread that owns
//!   the context and under the dispatch guard (decisions 0023 and 0024).
//! - Every range is a `RawRange`, proved inside its live allocation. Sizes, alignment and
//!   disjointness (`super::Span`) are proved again here before the call. The C entry checks
//!   pointer/scalar/alignment conditions, not allocation extents or overlap.
//! - The C struct is built in one `unsafe` block that cites those checks.
//! - Allocations stay valid until a synchronize: `inference_cuda_free` synchronizes the device
//!   first.
//! - The `extern` declarations, the `repr(C)` mirror and every raw pointer are private to this
//!   file. Nothing is re-exported.

use std::ffi::{c_int, c_void};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Mirror of `inference_cuda_attn_decode_args` in `attn_decode.h`, field for field.
#[repr(C)]
struct AttnDecodeArgsRaw {
    q: *const c_void,
    proj: *const c_void,
    k_cache: *const c_void,
    v_cache: *const c_void,
    kv_len: *const c_void,
    part: *mut c_void,
    out: *mut c_void,
    ld: c_int,
    m: c_int,
    n_head: c_int,
    n_kv_head: c_int,
    head_dim: c_int,
    cache_rows: c_int,
    max_splits: c_int,
    min_keys: c_int,
    window: c_int,
    ring: c_int,
    scale: f32,
}

unsafe extern "C" {
    #[link_name = "inference_cuda_attn_decode_part_floats"]
    fn attn_decode_part_floats_raw(
        m: c_int,
        n_kv_head: c_int,
        head_dim: c_int,
        max_splits: c_int,
    ) -> usize;
    #[link_name = "inference_cuda_attn_decode_launch"]
    fn attn_decode_launch_raw(
        args: *const AttnDecodeArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_attn_decode_sinks_launch"]
    fn attn_decode_sinks_launch_raw(
        args: *const AttnDecodeArgsRaw,
        sinks: *const f32,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Every integer argument of the launch, already bounded and converted by the wrapper before
/// `launch_on`, so no conversion can fail inside the launch closure. `ld` is 0 without `proj`.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct AttnInts {
    pub(in crate::kernels::cuda) m: c_int,
    pub(in crate::kernels::cuda) n_head: c_int,
    pub(in crate::kernels::cuda) n_kv_head: c_int,
    pub(in crate::kernels::cuda) head_dim: c_int,
    pub(in crate::kernels::cuda) ld: c_int,
    pub(in crate::kernels::cuda) cache_rows: c_int,
    pub(in crate::kernels::cuda) max_splits: c_int,
    pub(in crate::kernels::cuda) min_keys: c_int,
    pub(in crate::kernels::cuda) window: c_int,
    pub(in crate::kernels::cuda) ring: c_int,
}

/// What one attention launch reads and writes: the converted ranges, the integers and the score
/// scale.
pub(in crate::kernels::cuda) struct AttnDecodeRaw<'a> {
    pub(in crate::kernels::cuda) q: RawRange<'a>,
    pub(in crate::kernels::cuda) proj: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) k_cache: RawRange<'a>,
    pub(in crate::kernels::cuda) v_cache: RawRange<'a>,
    pub(in crate::kernels::cuda) kv_len: RawRange<'a>,
    pub(in crate::kernels::cuda) part: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: AttnInts,
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

fn needs(io: &AttnDecodeRaw<'_>) -> Result<Needs, DeviceFailure> {
    let ints = &io.ints;
    let (m, heads, kv_heads) = (size(ints.m)?, size(ints.n_head)?, size(ints.n_kv_head)?);
    let (dim, ld, rows) = (size(ints.head_dim)?, size(ints.ld)?, size(ints.cache_rows)?);
    // SAFETY: a pure function of its four integers; it reads no memory and needs no device.
    let part_floats = unsafe {
        attn_decode_part_floats_raw(ints.m, ints.n_kv_head, ints.head_dim, ints.max_splits)
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

/// Proves a launch before the native call: sizes and alignment of every range (q 16, proj 4,
/// k and v 16, kv_len 4, part 16, out 4) and that `out` and `part`, which the kernels write,
/// share no byte with any other range.
fn check_attn(io: &AttnDecodeRaw<'_>) -> Result<(), DeviceFailure> {
    let need = needs(io)?;
    let sized = io.q.bytes >= need.q
        && io.k_cache.bytes >= need.cache
        && io.v_cache.bytes >= need.cache
        && io.kv_len.bytes >= 4
        && io.part.bytes >= need.part
        && io.out.bytes >= need.out
        && io.proj.as_ref().is_none_or(|proj| proj.bytes >= need.proj);
    let aligned = io.q.offset.is_multiple_of(16)
        && io.k_cache.offset.is_multiple_of(16)
        && io.v_cache.offset.is_multiple_of(16)
        && io.kv_len.offset.is_multiple_of(4)
        && io.part.offset.is_multiple_of(16)
        && io.out.offset.is_multiple_of(4)
        && io
            .proj
            .as_ref()
            .is_none_or(|proj| proj.offset.is_multiple_of(4));
    let out = Span::of(&io.out, need.out);
    let part = Span::of(&io.part, need.part);
    let mut reads = vec![
        Span::of(&io.q, need.q),
        Span::of(&io.k_cache, need.cache),
        Span::of(&io.v_cache, need.cache),
        Span::of(&io.kv_len, 4),
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

/// Queues the two attention kernels (partials, then the fixed-order combine) on `stream`.
/// Returns when both are queued.
pub(in crate::kernels::cuda) fn attn_decode_launch(
    stream: &RawStream,
    io: AttnDecodeRaw<'_>,
) -> Result<(), DeviceFailure> {
    launch_inner(stream, io, None)
}

pub(in crate::kernels::cuda) fn attn_decode_sinks_launch(
    stream: &RawStream,
    io: AttnDecodeRaw<'_>,
    sinks: RawRange<'_>,
) -> Result<(), DeviceFailure> {
    launch_inner(stream, io, Some(sinks))
}

fn launch_inner(
    stream: &RawStream,
    io: AttnDecodeRaw<'_>,
    sinks: Option<RawRange<'_>>,
) -> Result<(), DeviceFailure> {
    check_attn(&io)?;
    if let Some(sinks) = &sinks {
        let bytes = product(&[size(io.ints.n_head)?, 4])?;
        let need = needs(&io)?;
        let read = Span::of(sinks, bytes);
        if sinks.bytes < bytes
            || !sinks.offset.is_multiple_of(4)
            || !Span::of(&io.out, need.out).apart(&read)
            || !Span::of(&io.part, need.part).apart(&read)
        {
            return Err(invalid_argument());
        }
    }
    let ints = io.ints;
    let args = AttnDecodeArgsRaw {
        q: io.q.address().cast_const(),
        proj: io
            .proj
            .as_ref()
            .map_or(std::ptr::null(), |proj| proj.address().cast_const()),
        k_cache: io.k_cache.address().cast_const(),
        v_cache: io.v_cache.address().cast_const(),
        kv_len: io.kv_len.address().cast_const(),
        part: io.part.address(),
        out: io.out.address(),
        ld: ints.ld,
        m: ints.m,
        n_head: ints.n_head,
        n_kv_head: ints.n_kv_head,
        head_dim: ints.head_dim,
        cache_rows: ints.cache_rows,
        max_splits: ints.max_splits,
        min_keys: ints.min_keys,
        window: ints.window,
        ring: ints.ring,
        scale: io.scale,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_attn` proved every range inside its live allocation, large enough for the
    // call, aligned, and `out` and `part` disjoint from every other range; the sink claims above
    // extend that proof to its separate pointer. The C entry checks pointer/scalar/alignment
    // conditions, not allocation extents or overlap. Allocations stay valid until work finishes because
    // `inference_cuda_free` synchronizes the device first. `args` lives across the call and the
    // entry copies it before returning. `stream` is live for the call.
    let status = unsafe {
        match sinks {
            Some(sinks) => attn_decode_sinks_launch_raw(
                &raw const args,
                sinks.address().cast::<f32>().cast_const(),
                stream.0.as_ptr(),
                &mut stage,
            ),
            None => attn_decode_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage),
        }
    };
    checked(status, stage)
}
