//! Raw FFI to the KV8 split-KV decode attention entry of `native/resident/attn_decode_q8.cu`
//! (declared in `native/resident/attn_decode_q8.h`). Child module of `native/resident.rs`, so it
//! inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! The packed-int8 layout follows Strata's laguna_kv8 flash decode.
//!
//! Safety invariants this file relies on:
//! - The launch entry is called only through `CudaContext::launch_on`, on the thread that owns
//!   the context and under the dispatch guard.
//! - Every range is a `RawRange`, proved inside a live allocation. Sizes, alignment and
//!   disjointness (`super::Span`) are proved again here before the call. The C entry checks
//!   its available pointer/scalar/alignment conditions, not allocation extents or overlap.
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
    #[cfg(all(test, feature = "cuda-arms-qualification"))]
    #[link_name = "inference_cuda_attn_decode_q8_set_arms"]
    fn attn_decode_q8_set_arms_raw(prefetch: c_int) -> c_int;
    #[cfg(all(test, feature = "cuda-arms-qualification"))]
    #[link_name = "inference_cuda_attn_decode_q8_arm_attrs"]
    fn attn_decode_q8_arm_attrs_raw(prefetch: c_int, regs: *mut c_int, smem: *mut c_int) -> c_int;
    #[cfg(all(test, feature = "cuda-arms-qualification"))]
    #[link_name = "inference_cuda_attn_decode_q8_dispatched"]
    fn attn_decode_q8_dispatched_raw() -> c_int;
    #[cfg(all(test, feature = "cuda-arms-qualification"))]
    #[link_name = "inference_cuda_attn_decode_q8_kernel_identity_reset"]
    fn attn_decode_q8_kernel_identity_reset_raw() -> c_int;
    #[cfg(all(test, feature = "cuda-arms-qualification"))]
    #[link_name = "inference_cuda_attn_decode_q8_kernel_identity"]
    fn attn_decode_q8_kernel_identity_raw(ran: *mut c_int) -> c_int;
    #[cfg(all(test, feature = "cuda-arms-qualification"))]
    #[link_name = "inference_cuda_attn_decode_q8_split_plan_device"]
    fn attn_decode_q8_split_plan_device_raw(
        n_keys: c_int,
        max_splits: c_int,
        min_keys: c_int,
        chunk: *mut c_int,
        n_active: *mut c_int,
    ) -> c_int;
    #[cfg(all(test, feature = "cuda-arms-qualification"))]
    #[link_name = "inference_cuda_attn_decode_q8_split_plan"]
    fn attn_decode_q8_split_plan_raw(
        n_keys: c_int,
        max_splits: c_int,
        min_keys: c_int,
        chunk: *mut c_int,
        n_active: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_attn_decode_q8_launch"]
    fn attn_decode_q8_launch_raw(
        args: *const AttnDecodeQ8ArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_attn_decode_q8_sinks_launch"]
    fn attn_decode_q8_sinks_launch_raw(
        args: *const AttnDecodeQ8ArgsRaw,
        sinks: *const f32,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Zero the device-written kernel identity before a fixture leg.
#[cfg(all(test, feature = "cuda-arms-qualification"))]
pub(in crate::kernels::cuda) fn attn_decode_q8_kernel_identity_reset() -> Result<(), i32> {
    // Safety: writes one device symbol int through the runtime API.
    let code = unsafe { attn_decode_q8_kernel_identity_reset_raw() };
    if code == 0 { Ok(()) } else { Err(code) }
}

/// Which kernel last wrote its identity ON DEVICE (1 legacy, 2 prefetch): unlike the
/// host dispatch tag, a launch that resolves to the wrong kernel cannot produce this.
#[cfg(all(test, feature = "cuda-arms-qualification"))]
pub(in crate::kernels::cuda) fn attn_decode_q8_kernel_identity() -> Result<i32, i32> {
    let mut ran: c_int = 0;
    // Safety: passes a valid pointer to one int; no device pointers involved.
    let code = unsafe { attn_decode_q8_kernel_identity_raw(&mut ran) };
    if code == 0 { Ok(ran) } else { Err(code) }
}

/// The split plan computed by the DEVICE helper itself (1 thread), for pinning the
/// device copy against the host mirror.
#[cfg(all(test, feature = "cuda-arms-qualification"))]
pub(in crate::kernels::cuda) fn attn_decode_q8_split_plan_device(
    n_keys: i32,
    max_splits: i32,
    min_keys: i32,
) -> Result<(i32, i32), i32> {
    let (mut chunk, mut n_active): (c_int, c_int) = (0, 0);
    // Safety: passes valid pointers to two ints; the FFI allocates and frees its own
    // device buffer and rejects degenerate inputs before launching.
    let code = unsafe {
        attn_decode_q8_split_plan_device_raw(
            n_keys,
            max_splits,
            min_keys,
            &mut chunk,
            &mut n_active,
        )
    };
    if code == 0 {
        Ok((chunk, n_active))
    } else {
        Err(code)
    }
}

#[cfg(all(test, feature = "cuda-arms-qualification"))]
pub(in crate::kernels::cuda) fn attn_decode_q8_dispatched() -> i32 {
    // Safety: a relaxed atomic load; no pointers or device state involved.
    unsafe { attn_decode_q8_dispatched_raw() }
}

/// Explicit arm control for the register-V decode attention:
/// `prefetch` -1 follows `INFERENCE_CUDA_KV8_ATTN_V_PREFETCH` while 0/1 force the
/// legacy/prefetch arm. A test or harness call site uses this instead of mutating the
/// environment (edition 2024 denies `set_var` in this crate).
#[cfg(all(test, feature = "cuda-arms-qualification"))]
pub(in crate::kernels::cuda) fn attn_decode_q8_set_arms(prefetch: i32) -> Result<(), i32> {
    // Safety: writes one thread-local int; no pointers or device state involved.
    let code = unsafe { attn_decode_q8_set_arms_raw(prefetch) };
    if code == 0 { Ok(()) } else { Err(code) }
}

/// The register count and static shared-memory bytes of one D=256 register-V instantiation
/// (test-only): the fixture asserts the legacy and prefetch arms are two DIFFERENT kernels
/// before comparing their outputs, so an arm toggle that silently compares a kernel with
/// itself fails instead of passing vacuously.
#[cfg(all(test, feature = "cuda-arms-qualification"))]
pub(in crate::kernels::cuda) fn attn_decode_q8_arm_attrs(
    prefetch: i32,
) -> Result<(i32, i32), DeviceFailure> {
    let (mut regs, mut smem) = (0_i32, 0_i32);
    // Safety: two writable out-pointers of the declared width; no device state involved.
    let rc = unsafe { attn_decode_q8_arm_attrs_raw(prefetch, &mut regs, &mut smem) };
    if rc != 0 {
        return Err(invalid_argument());
    }
    Ok((regs, smem))
}

/// The exact (chunk, n_active) split plan the kernel will use (test-only): the fixture pins
/// that the caps 64/61/41 give distinct plans at 65,536 keys before any device leg.
#[cfg(all(test, feature = "cuda-arms-qualification"))]
pub(in crate::kernels::cuda) fn attn_decode_q8_split_plan(
    n_keys: i32,
    max_splits: i32,
    min_keys: i32,
) -> Result<(i32, i32), DeviceFailure> {
    let (mut chunk, mut n_active) = (0_i32, 0_i32);
    // Safety: two writable out-pointers of the declared width; pure arithmetic inside.
    let rc = unsafe {
        attn_decode_q8_split_plan_raw(n_keys, max_splits, min_keys, &mut chunk, &mut n_active)
    };
    if rc != 0 {
        return Err(invalid_argument());
    }
    Ok((chunk, n_active))
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

fn needs(ints: &AttnQ8Ints, with_proj: bool) -> Result<Needs, DeviceFailure> {
    let (m, heads, kv_heads) = (size(ints.m)?, size(ints.n_head)?, size(ints.n_kv_head)?);
    let (dim, ld, cells) = (size(ints.head_dim)?, size(ints.ld)?, size(ints.cells)?);
    // SAFETY: a pure function of its four integers; it reads no memory and needs no device.
    let part_floats = unsafe {
        attn_decode_q8_part_floats_raw(ints.m, ints.n_kv_head, ints.head_dim, ints.max_splits)
    };
    let row_floats = product(&[heads, dim])?;
    let proj = if with_proj {
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
    let need = needs(&io.ints, io.proj.is_some())?;
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
    launch_inner(stream, io, None)
}

pub(in crate::kernels::cuda) fn attn_decode_q8_sinks_launch(
    stream: &RawStream,
    io: AttnDecodeQ8Raw<'_>,
    sinks: RawRange<'_>,
) -> Result<(), DeviceFailure> {
    launch_inner(stream, io, Some(sinks))
}

fn launch_inner(
    stream: &RawStream,
    io: AttnDecodeQ8Raw<'_>,
    sinks: Option<RawRange<'_>>,
) -> Result<(), DeviceFailure> {
    check_attn(&io)?;
    if let Some(sinks) = &sinks {
        let bytes = product(&[size(io.ints.n_head)?, 4])?;
        let need = needs(&io.ints, io.proj.is_some())?;
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
    // `args` addresses bytes the kernels may touch; the sink claims above cover its separate
    // pointer. The C entry checks pointer/scalar/alignment
    // conditions, not allocation extents or overlap. Allocations stay valid until work finishes because
    // `inference_cuda_free` synchronizes the device first. `args` lives across the call and the
    // entry copies it before returning. `stream` is live for the call.
    let status = unsafe {
        match sinks {
            Some(sinks) => attn_decode_q8_sinks_launch_raw(
                &raw const args,
                sinks.address().cast::<f32>().cast_const(),
                stream.0.as_ptr(),
                &mut stage,
            ),
            None => attn_decode_q8_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage),
        }
    };
    checked(status, stage)
}

/// Actual read/write ranges of one local-page partial; no unused output or length buffer.
pub(in crate::kernels::cuda) struct AttnQ8PartialRaw<'a> {
    pub(in crate::kernels::cuda) q: RawRange<'a>,
    pub(in crate::kernels::cuda) k_codes: RawRange<'a>,
    pub(in crate::kernels::cuda) k_scales: RawRange<'a>,
    pub(in crate::kernels::cuda) v_codes: RawRange<'a>,
    pub(in crate::kernels::cuda) v_scales: RawRange<'a>,
    pub(in crate::kernels::cuda) part: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: AttnQ8Ints,
    pub(in crate::kernels::cuda) visible: c_int,
    pub(in crate::kernels::cuda) group: c_int,
    pub(in crate::kernels::cuda) scale: f32,
}

/// Actual inputs/output of the final merge; no unused query or KV allocations.
pub(in crate::kernels::cuda) struct AttnQ8CombineRaw<'a> {
    pub(in crate::kernels::cuda) part: RawRange<'a>,
    pub(in crate::kernels::cuda) proj: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: AttnQ8Ints,
    pub(in crate::kernels::cuda) visible: c_int,
}

unsafe extern "C" {
    fn inference_cuda_attn_decode_q8_partial_launch(
        args: *const AttnDecodeQ8ArgsRaw,
        visible: c_int,
        group: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_attn_decode_q8_combine_launch(
        args: *const AttnDecodeQ8ArgsRaw,
        visible: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

fn paged_geometry(i: &AttnQ8Ints, visible: c_int, gate: bool) -> Result<(), DeviceFailure> {
    if i.m != 1
        || !(1..=64).contains(&i.n_kv_head)
        || i.n_head != 8 * i.n_kv_head
        || !matches!(i.head_dim, 128 | 256)
        || i.max_splits != 64
        || !(64..=(1 << 20)).contains(&i.min_keys)
        || i.min_keys % 64 != 0
        || i.cells < 1
        || i.cells > i.min_keys * 64
        || visible < 1
        || visible > i.cells
        || i.window != 0
        || i.ring != 0
    {
        return Err(invalid_argument());
    }
    if gate && i.ld < i.n_head * 2 * i.head_dim {
        return Err(invalid_argument());
    }
    Ok(())
}

// The selective C entries neither require nor dereference these unused null fields.
fn paged_arguments(i: &AttnQ8Ints) -> AttnDecodeQ8ArgsRaw {
    AttnDecodeQ8ArgsRaw {
        q: std::ptr::null(),
        proj: std::ptr::null(),
        k_codes: std::ptr::null(),
        k_scales: std::ptr::null(),
        v_codes: std::ptr::null(),
        v_scales: std::ptr::null(),
        kv_len: std::ptr::null(),
        part: std::ptr::null_mut(),
        out: std::ptr::null_mut(),
        ld: i.ld,
        m: i.m,
        n_head: i.n_head,
        n_kv_head: i.n_kv_head,
        head_dim: i.head_dim,
        cells: i.cells,
        max_splits: i.max_splits,
        min_keys: i.min_keys,
        window: i.window,
        ring: i.ring,
        scale: 1.0,
    }
}

pub(in crate::kernels::cuda) fn attn_q8_partial_launch(
    stream: &RawStream,
    io: AttnQ8PartialRaw<'_>,
) -> Result<(), DeviceFailure> {
    paged_geometry(&io.ints, io.visible, false)?;
    if io.ints.cells != io.ints.min_keys
        || !(0..64).contains(&io.group)
        || !(io.scale.is_finite() && io.scale > 0.0 && io.scale < 1e30)
    {
        return Err(invalid_argument());
    }
    let need = needs(&io.ints, false)?;
    let scratch = Span::of(&io.part, need.part);
    let reads = [
        (&io.q, need.q, 16),
        (&io.k_codes, need.codes, 4),
        (&io.k_scales, need.scales, 4),
        (&io.v_codes, need.codes, 4),
        (&io.v_scales, need.scales, 4),
    ];
    if io.part.bytes < need.part
        || !io.part.offset.is_multiple_of(16)
        || !reads.iter().all(|(r, n, a)| {
            r.bytes >= *n && r.offset.is_multiple_of(*a) && scratch.apart(&Span::of(r, *n))
        })
    {
        return Err(invalid_argument());
    }
    let args = AttnDecodeQ8ArgsRaw {
        q: io.q.address().cast_const(),
        k_codes: io.k_codes.address().cast_const(),
        k_scales: io.k_scales.address().cast_const(),
        v_codes: io.v_codes.address().cast_const(),
        v_scales: io.v_scales.address().cast_const(),
        part: io.part.address(),
        scale: io.scale,
        ..paged_arguments(&io.ints)
    };
    let mut stage = STAGE_NONE;
    // SAFETY: checked live extents/alignment and scratch disjointness; the selective entry
    // copies this ABI-checked struct and never reads its unused null fields. Owners cover work.
    let status = unsafe {
        inference_cuda_attn_decode_q8_partial_launch(
            &raw const args,
            io.visible,
            io.group,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}

pub(in crate::kernels::cuda) fn attn_q8_combine_launch(
    stream: &RawStream,
    io: AttnQ8CombineRaw<'_>,
) -> Result<(), DeviceFailure> {
    paged_geometry(&io.ints, io.visible, io.proj.is_some())?;
    let need = needs(&io.ints, io.proj.is_some())?;
    let output = Span::of(&io.out, need.out);
    let scratch = Span::of(&io.part, need.part);
    let gate_ok = io.proj.as_ref().is_none_or(|r| {
        r.bytes >= need.proj
            && r.offset.is_multiple_of(4)
            && output.apart(&Span::of(r, need.proj))
            && scratch.apart(&Span::of(r, need.proj))
    });
    if io.part.bytes < need.part
        || !io.part.offset.is_multiple_of(16)
        || io.out.bytes < need.out
        || !io.out.offset.is_multiple_of(4)
        || !output.apart(&scratch)
        || !gate_ok
    {
        return Err(invalid_argument());
    }
    let args = AttnDecodeQ8ArgsRaw {
        proj: io
            .proj
            .as_ref()
            .map_or(std::ptr::null(), |r| r.address().cast_const()),
        part: io.part.address(),
        out: io.out.address(),
        ..paged_arguments(&io.ints)
    };
    let mut stage = STAGE_NONE;
    // SAFETY: checked live extents/alignment and output disjointness, ABI mirror, live stream.
    // The selective combine copies args and reads only part/proj/out; unused fields stay null.
    let status = unsafe {
        inference_cuda_attn_decode_q8_combine_launch(
            &raw const args,
            io.visible,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}
