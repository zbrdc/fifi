//! Raw FFI to the sequential scan entry of `native/prefill/gdn_scan_seq.cu` (declared in
//! `native/prefill/gdn_scan_seq.h`), CPF-DN-A. Child module of `native/prefill.rs`, so it
//! inherits the `allow(unsafe_code)` of `native.rs` (decision 0047, child modules of
//! `native.rs`).
//!
//! The kernels are the vendored per-token math (RD-4V, the user's own xavier-qwen code)
//! restated in a value-sliced launch with the q/k L2 norms in the ggml form; the wrapper's
//! tests pin both against an f64 reference and against the decode kernel.
//!
//! Safety invariants this file relies on:
//! - The launch entry is called only through `CudaContext::launch_on`, on the thread that
//!   owns the context and under the process dispatch guard (decisions 0023 and 0024). It
//!   only enqueues the two kernels (the scan, then the out-gate) and returns; the stream is
//!   live for the call.
//! - Every range is a `RawRange`, proved inside a live allocation by `raw_range`. Sizes,
//!   alignment and disjointness (`super::resident::Span`) are proved again here, on the
//!   extents the kernels touch, before the call; the C entry repeats them on the addresses.
//! - The C argument struct is built in one `unsafe` block that cites those checks.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes
//!   the device before `cudaFree`.
//! - The `extern` declaration, the `repr(C)` mirror and every raw pointer are private to
//!   this file. Nothing is re-exported.

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the prefill driver (decision 0049)")
)]

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Mirror of `inference_cuda_gdn_scan_seq_args` in `gdn_scan_seq.h`, field for field.
#[repr(C)]
struct GdnScanSeqArgsRaw {
    conv: *const c_void,
    proj: *const c_void,
    a: *const c_void,
    dt_bias: *const c_void,
    norm_w: *const c_void,
    state: *mut c_void,
    o_out: *mut c_void,
    status: *mut c_void,
    m: c_int,
    p0: c_int,
    ctx_tokens: c_int,
    conv_ld: c_int,
    ld: c_int,
    z_off: c_int,
    beta_off: c_int,
    alpha_off: c_int,
    n_k_heads: c_int,
    n_v_heads: c_int,
    head_dim: c_int,
    eps: f32,
}

const _: () = assert!(
    size_of::<GdnScanSeqArgsRaw>() == 112
        && offset_of!(GdnScanSeqArgsRaw, conv) == 0
        && offset_of!(GdnScanSeqArgsRaw, state) == 40
        && offset_of!(GdnScanSeqArgsRaw, status) == 56
        && offset_of!(GdnScanSeqArgsRaw, m) == 64
        && offset_of!(GdnScanSeqArgsRaw, p0) == 68
        && offset_of!(GdnScanSeqArgsRaw, ctx_tokens) == 72
        && offset_of!(GdnScanSeqArgsRaw, conv_ld) == 76
        && offset_of!(GdnScanSeqArgsRaw, ld) == 80
        && offset_of!(GdnScanSeqArgsRaw, z_off) == 84
        && offset_of!(GdnScanSeqArgsRaw, n_k_heads) == 96
        && offset_of!(GdnScanSeqArgsRaw, head_dim) == 104
        && offset_of!(GdnScanSeqArgsRaw, eps) == 108
);

unsafe extern "C" {
    #[link_name = "inference_cuda_gdn_scan_seq_launch"]
    fn gdn_scan_seq_launch_raw(
        args: *const GdnScanSeqArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

#[cfg(all(test, feature = "cuda-fault-qualification"))]
unsafe extern "C" {
    #[link_name = "inference_cuda_gdn_scan_seq_fault_bad_bounds_launch"]
    fn gdn_scan_seq_fault_bad_bounds_raw(
        args: *const GdnScanSeqArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Every integer argument of the launch, already bounded and converted by the wrapper
/// before `launch_on`, so no conversion can fail inside the launch closure.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct GdnScanSeqInts {
    pub(in crate::kernels::cuda) m: c_int,
    pub(in crate::kernels::cuda) p0: c_int,
    pub(in crate::kernels::cuda) ctx_tokens: c_int,
    pub(in crate::kernels::cuda) conv_ld: c_int,
    pub(in crate::kernels::cuda) ld: c_int,
    pub(in crate::kernels::cuda) z_off: c_int,
    pub(in crate::kernels::cuda) beta_off: c_int,
    pub(in crate::kernels::cuda) alpha_off: c_int,
    pub(in crate::kernels::cuda) n_k_heads: c_int,
    pub(in crate::kernels::cuda) n_v_heads: c_int,
    pub(in crate::kernels::cuda) head_dim: c_int,
}

/// What one scan launch reads and writes: `conv` the K0 output rows, `proj` the projection
/// rows (z, beta, alpha), the per-head weights, the in-place `state`, the `o_out` rows
/// (written pre-norm by K1, rewritten in place by K2), the optional prefill status word and
/// the integers.
pub(in crate::kernels::cuda) struct GdnScanSeqRaw<'a> {
    pub(in crate::kernels::cuda) conv: RawRange<'a>,
    pub(in crate::kernels::cuda) proj: RawRange<'a>,
    pub(in crate::kernels::cuda) a: RawRange<'a>,
    pub(in crate::kernels::cuda) dt_bias: RawRange<'a>,
    pub(in crate::kernels::cuda) norm_w: RawRange<'a>,
    pub(in crate::kernels::cuda) state: RawRange<'a>,
    pub(in crate::kernels::cuda) o_out: RawRange<'a>,
    pub(in crate::kernels::cuda) status: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) ints: GdnScanSeqInts,
    pub(in crate::kernels::cuda) eps: f32,
}

/// The bytes each range must hold.
struct Needs {
    conv: usize,
    proj: usize,
    head_vec: usize,
    norm_w: usize,
    state: usize,
    o_out: usize,
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

fn needs(ints: &GdnScanSeqInts) -> Result<Needs, DeviceFailure> {
    let m = size(ints.m)?;
    if m == 0 {
        return Err(invalid_argument());
    }
    let (nk, nv) = (size(ints.n_k_heads)?, size(ints.n_v_heads)?);
    let head_dim = size(ints.head_dim)?;
    let conv_cols = product(&[nk + nk + nv, head_dim])?;
    let z_end = size(ints.z_off)?
        .checked_add(product(&[nv, head_dim])?)
        .ok_or_else(invalid_argument)?;
    let beta_end = size(ints.beta_off)?
        .checked_add(nv)
        .ok_or_else(invalid_argument)?;
    let alpha_end = size(ints.alpha_off)?
        .checked_add(nv)
        .ok_or_else(invalid_argument)?;
    let proj_cols = z_end.max(beta_end).max(alpha_end);
    let proj = product(&[m - 1, size(ints.ld)?])?
        .checked_add(proj_cols)
        .and_then(|floats| floats.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    Ok(Needs {
        conv: product(&[m, conv_cols, 4])?,
        proj,
        head_vec: product(&[nv, 4])?,
        norm_w: product(&[head_dim, 4])?,
        state: product(&[nv, head_dim, head_dim, 4])?,
        o_out: product(&[m, nv, head_dim, 4])?,
    })
}

/// Proves a launch before the native call: size and 4-byte alignment of every range (the
/// status word when given), and that `state` and `o_out`, which the kernels write, share no
/// byte with any other range or each other, and that the status word, which the guard ORs,
/// shares no byte with any range.
fn check_scan(io: &GdnScanSeqRaw<'_>) -> Result<(), DeviceFailure> {
    let need = needs(&io.ints)?;
    let ranges = [
        (&io.conv, need.conv),
        (&io.proj, need.proj),
        (&io.a, need.head_vec),
        (&io.dt_bias, need.head_vec),
        (&io.norm_w, need.norm_w),
        (&io.state, need.state),
        (&io.o_out, need.o_out),
    ];
    let fitted = ranges
        .iter()
        .all(|(range, bytes)| range.bytes() >= *bytes && range.offset().is_multiple_of(4))
        && io
            .status
            .as_ref()
            .is_none_or(|s| s.bytes() >= 4 && s.offset().is_multiple_of(4));
    let state = Span::of(&io.state, need.state);
    let out = Span::of(&io.o_out, need.o_out);
    let reads = [
        Span::of(&io.conv, need.conv),
        Span::of(&io.proj, need.proj),
        Span::of(&io.a, need.head_vec),
        Span::of(&io.dt_bias, need.head_vec),
        Span::of(&io.norm_w, need.norm_w),
    ];
    let apart = state.apart(&out)
        && reads
            .iter()
            .all(|read| state.apart(read) && out.apart(read))
        && io.status.as_ref().is_none_or(|s| {
            let status = Span::of(s, 4);
            status.apart(&state)
                && status.apart(&out)
                && reads.iter().all(|read| status.apart(read))
        });
    if !(fitted && apart) {
        return Err(invalid_argument());
    }
    Ok(())
}

fn c_args(io: &GdnScanSeqRaw<'_>) -> GdnScanSeqArgsRaw {
    GdnScanSeqArgsRaw {
        conv: io.conv.address().cast_const(),
        proj: io.proj.address().cast_const(),
        a: io.a.address().cast_const(),
        dt_bias: io.dt_bias.address().cast_const(),
        norm_w: io.norm_w.address().cast_const(),
        state: io.state.address(),
        o_out: io.o_out.address(),
        status: io
            .status
            .as_ref()
            .map_or(std::ptr::null_mut(), |s| s.address()),
        m: io.ints.m,
        p0: io.ints.p0,
        ctx_tokens: io.ints.ctx_tokens,
        conv_ld: io.ints.conv_ld,
        ld: io.ints.ld,
        z_off: io.ints.z_off,
        beta_off: io.ints.beta_off,
        alpha_off: io.ints.alpha_off,
        n_k_heads: io.ints.n_k_heads,
        n_v_heads: io.ints.n_v_heads,
        head_dim: io.ints.head_dim,
        eps: io.eps,
    }
}

/// Queues the scan kernels on `stream`: K1 (the value-sliced sequential scan, the state
/// updated in place, the pre-norm rows written) then K2 (the gated output RMSNorm in place
/// on the same rows). Returns when both are queued.
pub(in crate::kernels::cuda) fn gdn_scan_seq_launch(
    stream: &RawStream,
    io: GdnScanSeqRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_scan(&io)?;
    let args = c_args(&io);
    let mut stage = STAGE_NONE;
    // SAFETY: `check_scan` proved every range inside its live allocation, large enough for
    // the call and 4-byte aligned, `state` and `o_out` disjoint from every other range and
    // each other, and the status word disjoint from every range, so each pointer in `args`
    // addresses bytes the kernels may touch; the C entry repeats these checks on the
    // addresses. The allocations stay valid until the queued work finishes because
    // `inference_cuda_free` synchronizes the device first. `args` lives across the call and
    // the entry copies it before returning. `stage` is a live writable `c_int`. `stream` is
    // live for the call.
    let status = unsafe { gdn_scan_seq_launch_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Queues the same two kernels with a deliberately violated p0 bound, so the kernels' own
/// device re-check (decision 0049 item 2: no state load, NaN rows, mask 8) is the code under
/// test; every other rule of the call holds, so the guard's writes touch valid memory.
/// Qualified fault path only.
#[cfg(all(test, feature = "cuda-fault-qualification"))]
pub(in crate::kernels::cuda) fn gdn_scan_seq_fault_bad_bounds(
    stream: &RawStream,
    io: GdnScanSeqRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_scan(&io)?;
    let mut args = c_args(&io);
    args.p0 = io.ints.ctx_tokens; // ctx_tokens + m > ctx_tokens: the violation
    let mut stage = STAGE_NONE;
    // SAFETY: as `gdn_scan_seq_launch`; the guards refuse before any state access, and the
    // NaN writes stay inside the proved `o_out` range.
    let status = unsafe { gdn_scan_seq_fault_bad_bounds_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}
