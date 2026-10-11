//! Raw FFI to the attention preparation entry of `native/resident/attn_prep.cu` (declared in
//! `native/resident/attn_prep.h`), RD-8a. Child module of `native/resident.rs`, so it inherits the
//! `allow(unsafe_code)` of `native.rs`.
//!
//! Design guidance only, no code copied (idea from MegaCapybara v1.35 static RE, section 6.1 C10,
//! layout only).
//!
//! Safety invariants this file relies on:
//! - The launch entry is called only through `CudaContext::launch_on`, on the thread that owns the
//!   context and under the dispatch guard (decisions 0023 and 0024). It only enqueues and returns.
//! - Every range is a `RawRange`, proved inside a live allocation by `raw_range`. Sizes, alignment
//!   and disjointness (`super::Span`) are proved again here, on the extents the kernel touches,
//!   before the call; the C entry repeats the alignment and shape checks on the addresses.
//! - The C argument struct is built in one `unsafe` block that cites those checks.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes the
//!   device before `cudaFree`.
//! - The `extern` declaration, the `repr(C)` mirror and every raw pointer are private to this
//!   file. Nothing is re-exported.

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// The only head size of the vendored kernel.
const HEAD_DIM: usize = 256;
/// `slot_mode` of `inference_cuda_attn_prep_args`: the row is the position.
const SLOT_POSITION: c_int = 0;

/// Mirror of `inference_cuda_attn_prep_args` in `attn_prep.h`, field for field.
#[repr(C)]
struct AttnPrepArgsRaw {
    proj: *const c_void,
    q_norm: *const c_void,
    k_norm: *const c_void,
    k_cache: *mut c_void,
    v_cache: *mut c_void,
    q: *mut c_void,
    pos: *const c_void,
    status: *mut c_void,
    ld: c_int,
    m: c_int,
    n_head: c_int,
    n_kv_head: c_int,
    head_dim: c_int,
    n_rot: c_int,
    rows: c_int,
    has_gate: c_int,
    slot_mode: c_int,
    slot_ring: c_int,
    eps: f32,
    rope_base: f32,
}

const _: () = assert!(
    size_of::<AttnPrepArgsRaw>() == 112
        && offset_of!(AttnPrepArgsRaw, pos) == 48
        && offset_of!(AttnPrepArgsRaw, status) == 56
        && offset_of!(AttnPrepArgsRaw, ld) == 64
        && offset_of!(AttnPrepArgsRaw, slot_ring) == 100
        && offset_of!(AttnPrepArgsRaw, eps) == 104
        && offset_of!(AttnPrepArgsRaw, rope_base) == 108
);

unsafe extern "C" {
    #[link_name = "inference_cuda_attn_prep_launch"]
    fn attn_prep_launch_raw(
        args: *const AttnPrepArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Every integer argument of the launch, already bounded and converted by the wrapper before
/// `launch_on`, so no conversion can fail inside the launch closure.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct AttnInts {
    pub(in crate::kernels::cuda) m: c_int,
    pub(in crate::kernels::cuda) n_head: c_int,
    pub(in crate::kernels::cuda) n_kv_head: c_int,
    pub(in crate::kernels::cuda) n_rot: c_int,
    pub(in crate::kernels::cuda) rows: c_int,
    pub(in crate::kernels::cuda) ld: c_int,
}

/// What one attention preparation reads and writes: the converted ranges, the integers and the
/// rope scalars.
pub(in crate::kernels::cuda) struct AttnPrepRaw<'a> {
    pub(in crate::kernels::cuda) proj: RawRange<'a>,
    pub(in crate::kernels::cuda) q_norm: RawRange<'a>,
    pub(in crate::kernels::cuda) k_norm: RawRange<'a>,
    pub(in crate::kernels::cuda) k_cache: RawRange<'a>,
    pub(in crate::kernels::cuda) v_cache: RawRange<'a>,
    pub(in crate::kernels::cuda) q: RawRange<'a>,
    pub(in crate::kernels::cuda) pos: RawRange<'a>,
    pub(in crate::kernels::cuda) status: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: AttnInts,
    pub(in crate::kernels::cuda) eps: f32,
    pub(in crate::kernels::cuda) rope_base: f32,
}

/// The bytes each range must hold.
struct Needs {
    proj: usize,
    norm: usize,
    q: usize,
    cache: usize,
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

fn needs(ints: &AttnInts) -> Result<Needs, DeviceFailure> {
    let (m, ld, rows) = (size(ints.m)?, size(ints.ld)?, size(ints.rows)?);
    let (heads, kv_heads) = (size(ints.n_head)?, size(ints.n_kv_head)?);
    let row_w = product(&[heads * 2 + kv_heads * 2, HEAD_DIM])?;
    let proj_floats = product(&[m.saturating_sub(1), ld])?
        .checked_add(row_w)
        .ok_or_else(invalid_argument)?;
    Ok(Needs {
        proj: product(&[proj_floats, 4])?,
        norm: HEAD_DIM * 4,
        q: product(&[m, heads, HEAD_DIM, 4])?,
        cache: product(&[rows, kv_heads, HEAD_DIM, 2])?,
    })
}

/// True when `range` holds `need` bytes and starts at a multiple of `align`.
fn fits(range: &RawRange<'_>, need: usize, align: usize) -> bool {
    range.bytes >= need && range.offset.is_multiple_of(align)
}

/// Sizes and alignment of the eight ranges: `q`, `k_cache` and `v_cache` 16, the rest 4.
fn check_fit(io: &AttnPrepRaw<'_>, need: &Needs) -> bool {
    fits(&io.proj, need.proj, 4)
        && fits(&io.q_norm, need.norm, 4)
        && fits(&io.k_norm, need.norm, 4)
        && fits(&io.q, need.q, 16)
        && fits(&io.k_cache, need.cache, 16)
        && fits(&io.v_cache, need.cache, 16)
        && fits(&io.pos, 4, 4)
        && fits(&io.status, 4, 4)
}

/// Proves a launch before the native call: the shape integers, sizes and alignment of every range
/// and that the four ranges the kernel writes (`q`, both caches and `status`) share no byte with
/// each other or with the four it reads. The touched extents are compared, not the whole ranges,
/// so `pos` and `status` may sit in one block.
fn check_attn_prep(io: &AttnPrepRaw<'_>) -> Result<(), DeviceFailure> {
    let need = needs(&io.ints)?;
    if !check_fit(io, &need) {
        return Err(invalid_argument());
    }
    let written = [
        Span::of(&io.q, need.q),
        Span::of(&io.k_cache, need.cache),
        Span::of(&io.v_cache, need.cache),
        Span::of(&io.status, 4),
    ];
    let read = [
        Span::of(&io.proj, need.proj),
        Span::of(&io.q_norm, need.norm),
        Span::of(&io.k_norm, need.norm),
        Span::of(&io.pos, 4),
    ];
    for (i, a) in written.iter().enumerate() {
        let apart_from_written = written[i + 1..].iter().all(|b| a.apart(b));
        if !(apart_from_written && read.iter().all(|b| a.apart(b))) {
            return Err(invalid_argument());
        }
    }
    Ok(())
}

/// Queues the attention preparation kernel on `stream`: per-head RMSNorm of q and k, rope at the
/// device position, the `q` output and the KV append. Returns when it is queued.
pub(in crate::kernels::cuda) fn attn_prep_launch(
    stream: &RawStream,
    call: AttnPrepRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_attn_prep(&call)?;
    let ints = call.ints;
    let args = AttnPrepArgsRaw {
        proj: call.proj.address().cast_const(),
        q_norm: call.q_norm.address().cast_const(),
        k_norm: call.k_norm.address().cast_const(),
        k_cache: call.k_cache.address(),
        v_cache: call.v_cache.address(),
        q: call.q.address(),
        pos: call.pos.address().cast_const(),
        status: call.status.address(),
        ld: ints.ld,
        m: ints.m,
        n_head: ints.n_head,
        n_kv_head: ints.n_kv_head,
        head_dim: 256,
        n_rot: ints.n_rot,
        rows: ints.rows,
        has_gate: 1,
        slot_mode: SLOT_POSITION,
        slot_ring: 0,
        eps: call.eps,
        rope_base: call.rope_base,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: (a) `check_attn_prep` proved every range inside its live allocation and large enough
    // for the call; the allocations outlive the queued work because `inference_cuda_free`
    // synchronizes the device before `cudaFree`. (b) It also proved the alignment (16 for `q` and
    // the caches, 4 for the rest) and that the touched extents of `q`, both caches and `status`
    // share no byte with any other range, so the kernel's writes cannot alias its reads; the C
    // entry repeats the checks on the addresses. (c) `stream` is borrowed from a live `RawStream`.
    // (d) `args` is a live local that the entry reads only during the call and keeps no pointer to.
    // (e) The dispatch guard is held by `launch_on`, which is the only caller.
    let status = unsafe { attn_prep_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
