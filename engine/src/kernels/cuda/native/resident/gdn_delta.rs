//! Raw FFI to the gated DeltaNet launch entry of `native/resident/gdn_delta.cu` (declared in
//! `native/resident/gdn_delta.h`), RD-7b. Child module of `native/resident.rs`, so it inherits the
//! `allow(unsafe_code)` of `native.rs`.
//!
//! Design guidance only, no code copied: no MegaCapybara-derived design. The kernel behind the
//! entry is the user's own xavier-qwen code, restored in RD-4V.
//!
//! Safety invariants this file relies on:
//! - The launch entry is called only through `CudaContext::launch_on`, on the thread that owns
//!   the context and under the dispatch guard (decisions 0023 and 0024).
//! - Every range is a `RawRange`, proved inside its live allocation. Sizes, alignment and
//!   disjointness (`super::Span`) are proved again here before the call; the C entry repeats them
//!   on the addresses.
//! - The C struct is built in one `unsafe` block that cites those checks.
//! - Allocations stay valid until a synchronize: `inference_cuda_free` synchronizes the device
//!   first.
//! - The `extern` declaration, the `repr(C)` mirror and every raw pointer are private to this
//!   file. Nothing is re-exported.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for RD-9a (decision 0047)")
)]

use std::ffi::{c_int, c_void};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Size of the key and value head the vendored kernel is built for (`kern::kStateDim`).
const HEAD_DIM: usize = 128;

/// Mirror of `inference_cuda_gdn_delta_args` in `gdn_delta.h`, field for field.
#[repr(C)]
struct GdnDeltaArgsRaw {
    conv: *const c_void,
    proj: *const c_void,
    a: *const c_void,
    dt_bias: *const c_void,
    norm_w: *const c_void,
    state: *mut c_void,
    out: *mut c_void,
    m: c_int,
    conv_ld: c_int,
    ld: c_int,
    z_off: c_int,
    beta_off: c_int,
    alpha_off: c_int,
    n_k_heads: c_int,
    n_v_heads: c_int,
    eps: f32,
}

unsafe extern "C" {
    #[link_name = "inference_cuda_gdn_delta_launch"]
    fn gdn_delta_launch_raw(
        args: *const GdnDeltaArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Every integer argument of the launch, already bounded and converted by the wrapper before
/// `launch_on`, so no conversion can fail inside the launch closure.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct GdnDeltaInts {
    pub(in crate::kernels::cuda) m: c_int,
    pub(in crate::kernels::cuda) conv_ld: c_int,
    pub(in crate::kernels::cuda) ld: c_int,
    pub(in crate::kernels::cuda) z_off: c_int,
    pub(in crate::kernels::cuda) beta_off: c_int,
    pub(in crate::kernels::cuda) alpha_off: c_int,
    pub(in crate::kernels::cuda) n_k_heads: c_int,
    pub(in crate::kernels::cuda) n_v_heads: c_int,
}

/// What one gated delta launch reads and writes: the converted ranges, the integers and `eps`.
pub(in crate::kernels::cuda) struct GdnDeltaRaw<'a> {
    pub(in crate::kernels::cuda) conv: RawRange<'a>,
    pub(in crate::kernels::cuda) proj: RawRange<'a>,
    pub(in crate::kernels::cuda) a: RawRange<'a>,
    pub(in crate::kernels::cuda) dt_bias: RawRange<'a>,
    pub(in crate::kernels::cuda) norm_w: RawRange<'a>,
    pub(in crate::kernels::cuda) state: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: GdnDeltaInts,
    pub(in crate::kernels::cuda) eps: f32,
}

/// The bytes each range must hold.
struct Needs {
    conv: usize,
    proj: usize,
    head_vec: usize,
    norm_w: usize,
    state: usize,
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

/// Bytes of a `[m][ld]` f32 array whose last row is read up to column `cols`.
fn rows_bytes(m: usize, ld: usize, cols: usize) -> Result<usize, DeviceFailure> {
    product(&[m - 1, ld, 4])?
        .checked_add(product(&[cols, 4])?)
        .ok_or_else(invalid_argument)
}

fn needs(ints: &GdnDeltaInts) -> Result<Needs, DeviceFailure> {
    let m = size(ints.m)?;
    if m == 0 {
        return Err(invalid_argument());
    }
    let (nk, nv) = (size(ints.n_k_heads)?, size(ints.n_v_heads)?);
    let conv_cols = product(&[nk + nk + nv, HEAD_DIM])?;
    let z_end = size(ints.z_off)?
        .checked_add(product(&[nv, HEAD_DIM])?)
        .ok_or_else(invalid_argument)?;
    let beta_end = size(ints.beta_off)?
        .checked_add(nv)
        .ok_or_else(invalid_argument)?;
    let alpha_end = size(ints.alpha_off)?
        .checked_add(nv)
        .ok_or_else(invalid_argument)?;
    let proj_cols = z_end.max(beta_end).max(alpha_end);
    Ok(Needs {
        conv: rows_bytes(m, size(ints.conv_ld)?, conv_cols)?,
        proj: rows_bytes(m, size(ints.ld)?, proj_cols)?,
        head_vec: product(&[nv, 4])?,
        norm_w: product(&[HEAD_DIM, 4])?,
        state: product(&[nv, HEAD_DIM, HEAD_DIM, 4])?,
        out: product(&[m, nv, HEAD_DIM, 4])?,
    })
}

/// Proves a launch before the native call: size and 4-byte alignment of every range, and that
/// `state` and `out`, which the kernel writes, share no byte with any other range.
fn check_delta(io: &GdnDeltaRaw<'_>) -> Result<(), DeviceFailure> {
    let need = needs(&io.ints)?;
    let ranges = [
        (&io.conv, need.conv),
        (&io.proj, need.proj),
        (&io.a, need.head_vec),
        (&io.dt_bias, need.head_vec),
        (&io.norm_w, need.norm_w),
        (&io.state, need.state),
        (&io.out, need.out),
    ];
    let fitted = ranges
        .iter()
        .all(|(range, bytes)| range.bytes >= *bytes && range.offset.is_multiple_of(4));
    let state = Span::of(&io.state, need.state);
    let out = Span::of(&io.out, need.out);
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
            .all(|read| state.apart(read) && out.apart(read));
    if !(fitted && apart) {
        return Err(invalid_argument());
    }
    Ok(())
}

/// Queues the gated delta kernel on `stream`. Returns when it is queued.
pub(in crate::kernels::cuda) fn gdn_delta_launch(
    stream: &RawStream,
    io: GdnDeltaRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_delta(&io)?;
    let ints = io.ints;
    let args = GdnDeltaArgsRaw {
        conv: io.conv.address().cast_const(),
        proj: io.proj.address().cast_const(),
        a: io.a.address().cast_const(),
        dt_bias: io.dt_bias.address().cast_const(),
        norm_w: io.norm_w.address().cast_const(),
        state: io.state.address(),
        out: io.out.address(),
        m: ints.m,
        conv_ld: ints.conv_ld,
        ld: ints.ld,
        z_off: ints.z_off,
        beta_off: ints.beta_off,
        alpha_off: ints.alpha_off,
        n_k_heads: ints.n_k_heads,
        n_v_heads: ints.n_v_heads,
        eps: io.eps,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check_delta` proved every range inside its live allocation, large enough for the
    // call and 4-byte aligned, and `state` and `out` disjoint from every other range, so each
    // pointer in `args` addresses bytes the kernel may touch; the C entry repeats these checks on
    // the addresses. The allocations stay valid until the queued work finishes because
    // `inference_cuda_free` synchronizes the device first. `args` lives across the call and the
    // argument struct is read only during it (the kernel arguments are copied at launch, so
    // nothing keeps the pointer). `stage` is a live writable `c_int`. `stream` is live for the
    // call.
    let status = unsafe { gdn_delta_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
