//! Safe launch wrapper of the gated DeltaNet recurrence step (RD-7b, decision 0047, seam 1).
//!
//! Design guidance only, no code copied: there is no MegaCapybara-derived design here. The kernel
//! (`gated_delta_k`) is the user's own xavier-qwen code, restored in RD-4V; this file and the
//! launch shim of `native/resident/gdn_delta.cu` add no kernel. The ggml comparison of
//! `gated_delta_k` found no shape match (REPLY-38; REPLY-35 condition 3).
//!
//! [`CudaContext::gdn_delta`] queues one kernel on a [`DeviceStream`] and returns. Per value head
//! it L2-normalizes q and k (`x * rsqrt(sum(x^2) + eps)`, q also times `1/sqrt(128)`), forms the
//! decay `exp(softplus(alpha + dt_bias) * a)` and the write strength `sigmoid(beta)`, updates the
//! 128 x 128 state in place (decay, delta-rule correction, rank-one write), reads out with q,
//! applies the gated RMSNorm with `silu(z)` and writes the f32 row that the `ssm_out` projection
//! packs. Key head `kh = h % n_k_heads` (tiled, the GGUF layout). `a` is the GGUF `ssm_a` as
//! stored (`-exp(A_log)`, negative). The call allocates nothing, copies nothing and never
//! synchronizes; a fault in the queued work surfaces at the next synchronize of the stream.
//!
//! What the caller guarantees:
//! - **Sequence start.** The state is zero at the start of a sequence. The call is launch-only and
//!   in place and does not zero it; zeroing is the caller's (RD-9b, RD-2c).
//! - **State layout.** `state` is f32 `[n_v_heads][128 key i][128 value j]`, `j` fastest. ggml keeps
//!   the transpose, `[head][128 value j][128 key i]` with the key index fastest, so a state import
//!   or export transposes. Its owner is RD-9a2 (tag `qwen36-gguf-xq-1` keeps the device order).
//! - `conv` is f32 `[m][conv_ld]` laid out q (`n_k_heads` x 128) | k (`n_k_heads` x 128) |
//!   v (`n_v_heads` x 128), the output of the conv step, written before the call.
//! - `proj` is f32 `[m][ld]` holding `z` (`n_v_heads` x 128 at `z_off`), the beta logits
//!   (`n_v_heads` at `beta_off`) and the alpha logits (`n_v_heads` at `alpha_off`).
//! - A NaN or infinity in the state or in an input stays in the state for the rest of the
//!   sequence; the host cannot see device data, and the non-finite logits flag of the step
//!   catches the consequence.
//!
//! **The eps rule.** One `eps` serves both the L2 norm and the output RMS norm, and the kernel is
//! only validated for the GGUF value, the f32 `1e-6`. The wrapper refuses any other value, bit for
//! bit ([`GDN_EPS`]); the field exists so that the value is asserted at the call.
//!
//! The kernel reads no position, length or slot word, so the wrapper takes no `StepParams`.
//! Every other argument is checked here, before any native call, and refused as `BadRequest`:
//! the rules mirror `valid()` of `gdn_delta.cu`, so the native entry cannot return
//! `INVALID_ARGUMENT` (which would latch the backend). Local rules carry the tag
//! `gdn_delta: <rule>`; size, alignment and overlap are refused by `require_fit` and
//! `require_apart` with their own texts.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for RD-9a (decision 0047)")
)]

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::gdn_delta::{
    GdnDeltaInts, GdnDeltaRaw, gdn_delta_launch,
};

/// The one `eps` the kernel is validated for: the GGUF rms eps, the f32 `1e-6`.
pub(crate) const GDN_EPS: f32 = 1.0e-6_f32;
/// Key and value head size, the vendored constant `kStateDim`.
pub(crate) const GDN_HEAD_DIM: usize = 128;
/// The most tokens one call takes.
pub(crate) const GDN_DELTA_MAX_M: usize = 32768;
/// The most key heads (policy; Qwen3.6 has 16).
pub(crate) const GDN_MAX_K_HEADS: usize = 64;
/// The most value heads (policy; Qwen3.6 has 32).
pub(crate) const GDN_MAX_V_HEADS: usize = 128;
/// The largest `conv_ld`.
const MAX_CONV_LD: usize = 1 << 20;
/// The largest `ld`; the offsets are bounded by it.
const MAX_LD: usize = 1 << 22;

/// Geometry of one call: `m` tokens, `n_k_heads` key heads and `n_v_heads` value heads (a
/// multiple of `n_k_heads`), the row length `conv_ld` of `conv`, and the row length `ld` of `proj`
/// with the float offsets of `z`, the beta logits and the alpha logits inside a row.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DeltaShape {
    pub(crate) m: usize,
    pub(crate) n_k_heads: usize,
    pub(crate) n_v_heads: usize,
    pub(crate) conv_ld: usize,
    pub(crate) ld: usize,
    pub(crate) z_off: usize,
    pub(crate) beta_off: usize,
    pub(crate) alpha_off: usize,
}

/// The bytes each range of a call must hold, from [`delta_bytes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeltaBytes {
    pub(crate) conv: usize,
    pub(crate) proj: usize,
    pub(crate) a: usize,
    pub(crate) dt_bias: usize,
    pub(crate) norm_w: usize,
    pub(crate) state: usize,
    pub(crate) out: usize,
}

/// The per-layer weights: f32 `[n_v_heads]` `a` (GGUF `ssm_a` as stored) and `dt_bias`, and
/// f32 `[128]` `norm_w` (GGUF `ssm_norm.weight`). Each 4-byte aligned.
pub(crate) struct DeltaWeights<'b> {
    pub(crate) a: DeviceRange<'b>,
    pub(crate) dt_bias: DeviceRange<'b>,
    pub(crate) norm_w: DeviceRange<'b>,
}

/// One gated delta call. All ranges are device ranges of this context, 4-byte aligned.
///
/// - `eps`: exactly [`GDN_EPS`], bit for bit.
/// - `conv`, `proj`: see the module doc.
/// - `state`: f32 `[n_v_heads][128][128]`, read and written in place.
/// - `out`: f32 `[m][n_v_heads * 128]`, written; `state` and `out` overlap no other range.
pub(crate) struct DeltaCall<'b> {
    pub(crate) shape: DeltaShape,
    pub(crate) eps: f32,
    pub(crate) conv: DeviceRange<'b>,
    pub(crate) proj: DeviceRange<'b>,
    pub(crate) weights: DeltaWeights<'b>,
    pub(crate) state: DeviceRange<'b>,
    pub(crate) out: DeviceRange<'b>,
}

/// A `BadRequest` that names the rule that fired.
fn rule(tag: &str, detail: &str) -> EngError {
    EngError::BadRequest(format!("gdn_delta: {tag}: {detail}"))
}

/// Product of the factors, or the "size overflow" refusal.
fn product(factors: &[usize]) -> Result<usize, EngError> {
    factors
        .iter()
        .try_fold(1usize, |acc, &factor| acc.checked_mul(factor))
        .ok_or_else(|| rule("size overflow", "a range size overflows"))
}

/// `a + b`, or the "size overflow" refusal.
fn sum(a: usize, b: usize) -> Result<usize, EngError> {
    a.checked_add(b)
        .ok_or_else(|| rule("size overflow", "a range size overflows"))
}

/// The end column of `z`, of the beta logits and of the alpha logits in a `proj` row.
fn proj_ends(shape: &DeltaShape) -> Result<[(&'static str, usize); 3], EngError> {
    let nv = shape.n_v_heads;
    Ok([
        ("z", sum(shape.z_off, product(&[nv, GDN_HEAD_DIM])?)?),
        ("beta", sum(shape.beta_off, nv)?),
        ("alpha", sum(shape.alpha_off, nv)?),
    ])
}

/// The bytes each range of `shape` must hold: `conv` and `proj` reach
/// `(m - 1) * row length + the columns read`, the rest follow from the head counts.
///
/// Returns `BadRequest` ("size overflow") when a product does not fit `usize`, and for `m` 0.
pub(crate) fn delta_bytes(shape: DeltaShape) -> Result<DeltaBytes, EngError> {
    let before_last = shape
        .m
        .checked_sub(1)
        .ok_or_else(|| rule("tokens", "m is 0"))?;
    let (nk, nv) = (shape.n_k_heads, shape.n_v_heads);
    let conv_cols = product(&[sum(nk, sum(nk, nv)?)?, GDN_HEAD_DIM])?;
    let proj_cols = proj_ends(&shape)?
        .iter()
        .map(|&(_, end)| end)
        .max()
        .unwrap_or(0);
    let reach = |ld: usize, cols: usize| -> Result<usize, EngError> {
        product(&[sum(product(&[before_last, ld])?, cols)?, 4])
    };
    Ok(DeltaBytes {
        conv: reach(shape.conv_ld, conv_cols)?,
        proj: reach(shape.ld, proj_cols)?,
        a: product(&[nv, 4])?,
        dt_bias: product(&[nv, 4])?,
        norm_w: product(&[GDN_HEAD_DIM, 4])?,
        state: product(&[nv, GDN_HEAD_DIM, GDN_HEAD_DIM, 4])?,
        out: product(&[shape.m, nv, GDN_HEAD_DIM, 4])?,
    })
}

/// `value` as `c_int`, or the "exceeds i32" refusal naming `name`. Every value is bounded by the
/// limits checked before, so this cannot fail inside the launch closure.
fn to_int(name: &str, value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value).map_err(|_| rule("exceeds i32", &format!("{name} is {value}")))
}

/// Rule (1), head counts, token count and `eps`.
fn check_counts(shape: &DeltaShape, eps: f32) -> Result<(), EngError> {
    if !(1..=GDN_DELTA_MAX_M).contains(&shape.m) {
        return Err(rule("tokens", &format!("m is {}, not 1 to 32768", shape.m)));
    }
    if !(1..=GDN_MAX_K_HEADS).contains(&shape.n_k_heads) {
        return Err(rule(
            "n_k_heads",
            &format!("n_k_heads is {}, not 1 to 64", shape.n_k_heads),
        ));
    }
    if !(1..=GDN_MAX_V_HEADS).contains(&shape.n_v_heads) {
        return Err(rule(
            "n_v_heads",
            &format!("n_v_heads is {}, not 1 to 128", shape.n_v_heads),
        ));
    }
    if shape.n_v_heads.checked_rem(shape.n_k_heads) != Some(0) {
        return Err(rule(
            "heads",
            &format!(
                "n_v_heads {} is not a multiple of n_k_heads {}",
                shape.n_v_heads, shape.n_k_heads
            ),
        ));
    }
    if eps.to_bits() != GDN_EPS.to_bits() {
        return Err(rule("eps", &format!("eps is {eps:e}, not exactly 1e-6")));
    }
    Ok(())
}

/// Rule (1), the row lengths: `conv_ld` holds a whole conv row, `ld` reaches the end of `z`, of
/// the beta logits and of the alpha logits, and both stay under their limits.
fn check_rows(shape: &DeltaShape) -> Result<(), EngError> {
    let conv_cols = product(&[
        sum(shape.n_k_heads, sum(shape.n_k_heads, shape.n_v_heads)?)?,
        GDN_HEAD_DIM,
    ])?;
    if shape.conv_ld < conv_cols || shape.conv_ld > MAX_CONV_LD {
        return Err(rule(
            "conv_ld",
            &format!(
                "conv_ld is {}, not {conv_cols} to {MAX_CONV_LD}",
                shape.conv_ld
            ),
        ));
    }
    for (name, end) in proj_ends(shape)? {
        if shape.ld < end {
            return Err(rule(
                "ld",
                &format!("ld {} is below the {name} end {end}", shape.ld),
            ));
        }
    }
    if shape.ld > MAX_LD {
        return Err(rule("ld", &format!("ld {} is above {MAX_LD}", shape.ld)));
    }
    Ok(())
}

/// Rule (1), every value that goes to C as `c_int`, converted once.
fn convert_ints(shape: &DeltaShape) -> Result<GdnDeltaInts, EngError> {
    Ok(GdnDeltaInts {
        m: to_int("m", shape.m)?,
        conv_ld: to_int("conv_ld", shape.conv_ld)?,
        ld: to_int("ld", shape.ld)?,
        z_off: to_int("z_off", shape.z_off)?,
        beta_off: to_int("beta_off", shape.beta_off)?,
        alpha_off: to_int("alpha_off", shape.alpha_off)?,
        n_k_heads: to_int("n_k_heads", shape.n_k_heads)?,
        n_v_heads: to_int("n_v_heads", shape.n_v_heads)?,
    })
}

/// Rule (2), size and alignment, one `require_fit` per range (4-byte aligned).
fn check_fit(call: &DeltaCall<'_>, need: &DeltaBytes) -> Result<(), EngError> {
    require_fit("conv", &call.conv, need.conv, 4)?;
    require_fit("proj", &call.proj, need.proj, 4)?;
    require_fit("a", &call.weights.a, need.a, 4)?;
    require_fit("dt_bias", &call.weights.dt_bias, need.dt_bias, 4)?;
    require_fit("norm_w", &call.weights.norm_w, need.norm_w, 4)?;
    require_fit("state", &call.state, need.state, 4)?;
    require_fit("out", &call.out, need.out, 4)
}

/// Rule (3), disjointness: `state` and `out` are written, so each is apart from every other range
/// (11 pairs); the five read ranges may share bytes.
fn check_apart(call: &DeltaCall<'_>, need: &DeltaBytes) -> Result<(), EngError> {
    let state = Claim::of("state", &call.state, need.state);
    let out = Claim::of("out", &call.out, need.out);
    let reads = [
        Claim::of("conv", &call.conv, need.conv),
        Claim::of("proj", &call.proj, need.proj),
        Claim::of("a", &call.weights.a, need.a),
        Claim::of("dt_bias", &call.weights.dt_bias, need.dt_bias),
        Claim::of("norm_w", &call.weights.norm_w, need.norm_w),
    ];
    require_apart(&state, &out)?;
    for read in &reads {
        require_apart(&state, read)?;
        require_apart(&out, read)?;
    }
    Ok(())
}

impl CudaContext {
    /// Queues the gated delta step of `call` on `stream`: `m` tokens through all value heads,
    /// the state updated in place and `out` written. Returns when the kernel is queued; the
    /// output is valid after the stream drained.
    ///
    /// The module doc lists what the caller guarantees (zero state at a sequence start, the
    /// state layout and its transpose against ggml, the `conv` and `proj` rows written before
    /// the call) and the eps rule.
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a
    /// stream or buffer of another context, an `eps` other than `1e-6`, a token or head count
    /// the kernel refuses, a row length that does not hold the columns read, a range that is
    /// too short or misaligned, and a `state` or `out` range that overlaps another range of
    /// the call.
    pub(crate) fn gdn_delta(
        &self,
        stream: &DeviceStream,
        call: DeltaCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        check_counts(&call.shape, call.eps)?;
        check_rows(&call.shape)?;
        let ints = convert_ints(&call.shape)?;
        let need = delta_bytes(call.shape)?;
        check_fit(&call, &need)?;
        check_apart(&call, &need)?;
        let conv = self.raw_range(&call.conv)?;
        let proj = self.raw_range(&call.proj)?;
        let a = self.raw_range(&call.weights.a)?;
        let dt_bias = self.raw_range(&call.weights.dt_bias)?;
        let norm_w = self.raw_range(&call.weights.norm_w)?;
        let state = self.raw_range(&call.state)?;
        let out = self.raw_range(&call.out)?;
        let eps = call.eps;
        self.launch_on(stream, move |raw| {
            gdn_delta_launch(
                raw,
                GdnDeltaRaw {
                    conv,
                    proj,
                    a,
                    dt_bias,
                    norm_w,
                    state,
                    out,
                    ints,
                    eps,
                },
            )
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
