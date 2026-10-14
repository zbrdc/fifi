//! Raw FFI to the KV8 split-KV decode attention entry of `native/resident/attn_decode_q8.cu`
//! (declared in `native/resident/attn_decode_q8.h`), RD-8l. Child module of
//! `native/resident.rs`, so it inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Design guidance only, no code copied (ideas from Strata's laguna_kv8 flash decode and
//! MegaCapybara C10/C11, the packed-int8 layout only).
//!
//! Safety invariants this file relies on:
//! - The launch entry is called only through `CudaContext::launch_on`, on the thread that owns
//!   the context and under the dispatch guard (decisions 0023 and 0024).
//! - Every range is a `RawRange`, proved inside a live allocation. Sizes, alignment and
//!   disjointness (`super::Span`) are proved again here before the call; the C entry repeats
//!   them on the addresses.
//! - The C struct is built in one `unsafe` block that cites those checks.
//! - Allocations stay valid until a synchronize: `inference_cuda_free` synchronizes the device
//!   first.
//! - The `extern` declarations, the `repr(C)` mirror and every raw pointer are private to this
//!   file. Nothing is re-exported.

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Mirror of `inference_cuda_attn_decode_q8_args` in `attn_decode_q8.h`, field for field.
#[repr(C)]
struct AttnDecodeQ8ArgsRaw {
    q: *const c_void,
    proj: *const c_void,
    k_codes: *const c_void,
    k_scales: *const c_void,
    v_codes: *const c_void,
    v_scales: *const c_void,
    kv_len: *const c_void,
    part: *mut c_void,
    out: *mut c_void,
    ld: c_int,
    m: c_int,
    n_head: c_int,
    n_kv_head: c_int,
    head_dim: c_int,
    cells: c_int,
    max_splits: c_int,
    min_keys: c_int,
    window: c_int,
    ring: c_int,
    scale: f32,
}

const _: () = assert!(
    size_of::<AttnDecodeQ8ArgsRaw>() == 120
        && offset_of!(AttnDecodeQ8ArgsRaw, proj) == 8
        && offset_of!(AttnDecodeQ8ArgsRaw, k_scales) == 24
        && offset_of!(AttnDecodeQ8ArgsRaw, kv_len) == 48
        && offset_of!(AttnDecodeQ8ArgsRaw, part) == 56
        && offset_of!(AttnDecodeQ8ArgsRaw, out) == 64
        && offset_of!(AttnDecodeQ8ArgsRaw, ld) == 72
);

unsafe extern "C" {
    #[link_name = "inference_cuda_attn_decode_q8_part_floats"]
    fn attn_decode_q8_part_floats_raw(
        m: c_int,
        n_kv_head: c_int,
        head_dim: c_int,
        max_splits: c_int,
    ) -> usize;
    #[link_name = "inference_cuda_attn_decode_q8_launch"]
    fn attn_decode_q8_launch_raw(
        args: *const AttnDecodeQ8ArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Every integer argument of the launch, already bounded and converted by the wrapper before
/// `launch_on`, so no conversion can fail inside the launch closure. `ld` is 0 without `proj`.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct AttnQ8Ints {
    pub(in crate::kernels::cuda) m: c_int,
    pub(in crate::kernels::cuda) n_head: c_int,
    pub(in crate::kernels::cuda) n_kv_head: c_int,
    pub(in crate::kernels::cuda) head_dim: c_int,
    pub(in crate::kernels::cuda) ld: c_int,
    pub(in crate::kernels::cuda) cells: c_int,
    pub(in crate::kernels::cuda) max_splits: c_int,
    pub(in crate::kernels::cuda) min_keys: c_int,
    pub(in crate::kernels::cuda) window: c_int,
    pub(in crate::kernels::cuda) ring: c_int,
}

/// What one attention launch reads and writes: the converted ranges, the integers and the score
/// scale.
pub(in crate::kernels::cuda) struct AttnDecodeQ8Raw<'a> {
    pub(in crate::kernels::cuda) q: RawRange<'a>,
    pub(in crate::kernels::cuda) proj: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) k_codes: RawRange<'a>,
    pub(in crate::kernels::cuda) k_scales: RawRange<'a>,
    pub(in crate::kernels::cuda) v_codes: RawRange<'a>,
    pub(in crate::kernels::cuda) v_scales: RawRange<'a>,
    pub(in crate::kernels::cuda) kv_len: RawRange<'a>,
    pub(in crate::kernels::cuda) part: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: AttnQ8Ints,
    pub(in crate::kernels::cuda) scale: f32,
}

/// The bytes each range must hold and the alignment of its start.
struct Needs {
    q: usize,
    proj: usize,
    codes: usize,
    scales: usize,
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

fn needs(io: &AttnDecodeQ8Raw<'_>) -> Result<Needs, DeviceFailure> {
    let ints = &io.ints;
    let (m, heads, kv_heads) = (size(ints.m)?, size(ints.n_head)?, size(ints.n_kv_head)?);
    let (dim, ld, cells) = (size(ints.head_dim)?, size(ints.ld)?, size(ints.cells)?);
    // SAFETY: a pure function of its four integers; it reads no memory and needs no device.
    let part_floats = unsafe {
        attn_decode_q8_part_floats_raw(ints.m, ints.n_kv_head, ints.head_dim, ints.max_splits)
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
        codes: product(&[kv_heads, cells, dim])?,
        scales: product(&[kv_heads, cells, dim / 32, 2])?,
        part: product(&[part_floats, 4])?,
        out: product(&[m, row_floats, 4])?,
    })
}

/// Proves a launch before the native call: sizes and alignment of every range (q 16, proj 4,
/// the four planes 4, kv_len 4, part 16, out 4) and that `out` and `part`, which the kernels
/// write, share no byte with any other range.
fn check_attn(io: &AttnDecodeQ8Raw<'_>) -> Result<(), DeviceFailure> {
    let need = needs(io)?;
    let sized = io.q.bytes >= need.q
        && io.k_codes.bytes >= need.codes
        && io.k_scales.bytes >= need.scales
        && io.v_codes.bytes >= need.codes
        && io.v_scales.bytes >= need.scales
        && io.kv_len.bytes >= 4
        && io.part.bytes >= need.part
        && io.out.bytes >= need.out
        && io.proj.as_ref().is_none_or(|proj| proj.bytes >= need.proj);
    let aligned = io.q.offset.is_multiple_of(16)
        && io.k_codes.offset.is_multiple_of(4)
        && io.k_scales.offset.is_multiple_of(4)
        && io.v_codes.offset.is_multiple_of(4)
        && io.v_scales.offset.is_multiple_of(4)
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
        Span::of(&io.k_codes, need.codes),
        Span::of(&io.k_scales, need.scales),
        Span::of(&io.v_codes, need.codes),
        Span::of(&io.v_scales, need.scales),
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
pub(in crate::kernels::cuda) fn attn_decode_q8_launch(
    stream: &RawStream,
    io: AttnDecodeQ8Raw<'_>,
) -> Result<(), DeviceFailure> {
    check_attn(&io)?;
    let ints = io.ints;
    let args = AttnDecodeQ8ArgsRaw {
        q: io.q.address().cast_const(),
        proj: io
            .proj
            .as_ref()
            .map_or(std::ptr::null(), |proj| proj.address().cast_const()),
        k_codes: io.k_codes.address().cast_const(),
        k_scales: io.k_scales.address().cast_const(),
        v_codes: io.v_codes.address().cast_const(),
        v_scales: io.v_scales.address().cast_const(),
        kv_len: io.kv_len.address().cast_const(),
        part: io.part.address(),
        out: io.out.address(),
        ld: ints.ld,
        m: ints.m,
        n_head: ints.n_head,
        n_kv_head: ints.n_kv_head,
        head_dim: ints.head_dim,
        cells: ints.cells,
        max_splits: ints.max_splits,
        min_keys: ints.min_keys,
        window: ints.window,
        ring: ints.ring,
        scale: io.scale,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_attn` proved every range inside its live allocation, large enough for the
    // call, aligned, and `out` and `part` disjoint from every other range, so each pointer in
    // `args` addresses bytes the kernels may touch; the C entry repeats these checks on the
    // addresses. The allocations stay valid until the queued work finishes because
    // `inference_cuda_free` synchronizes the device first. `args` lives across the call and the
    // entry copies it before returning. `stream` is live for the call.
    let status =
        unsafe { attn_decode_q8_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
