//! The chunked-prefill DeltaNet scan (CPF-DN-A, decision 0049 design row 9, P-a): the pure
//! workspace sizing of this file now, and (with the later commits of the packet) the safe
//! launch wrapper of `gdn_scan_seq`, the value-sliced sequential gated-delta scan over one
//! chunk of `m` tokens.
//!
//! What the seam is. Per DeltaNet layer, the driver runs the reused decode conv (RD-7a's
//! [`CudaContext::gdn_conv`], unedited, with `m = M`: its ring handoff across a chunk
//! boundary is bit-identical to `M` decode steps, because the ring update is pure data
//! movement — pinned bit-exactly by the tests), then this packet's scan: the
//! value-sliced sequential kernel over the conv rows, and the gated output RMSNorm in place
//! on the scan output. The chunkwise P-b kernels (CPF-4b..d) replace this packet's entry
//! wholesale later; the conv reuse, the state buffer and layout, the in-place `o_out`
//! contract, the `p0`/`ctx_tokens`/status-word guard rules and these sizing functions carry
//! over unchanged.
//!
//! If a layer launch faults mid-chunk, the in-place state is advanced only up to that
//! layer, so the driver must poison the session, not resume it.
//!
//! The sizing functions are pure host arithmetic, checked throughout (`usize` overflow is
//! a typed refusal, the `conv_bytes` pattern of `resident/gdn_conv.rs`). What they count is
//! the kernel-side scratch of one chunk ONLY — the `conv_out` rows the conv writes and the
//! scan reads, and the `o_out` rows the scan writes pre-norm and the out-gate rewrites in
//! place (ONE buffer, not two). They deliberately do NOT count the projection rows (`qkv`,
//! `z`, `alpha`/`beta`) — those are the DeltaProj GEMM output, CPF-V4's buffers — nor the
//! driver's `q8_1`/`ssm_out` rows, nor the persistent `DeltaPool` state slab, so that no
//! caller can double-count them against the admission budget.

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the prefill driver (decision 0049)")
)]

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::prefill::gdn_scan_seq::{
    GdnScanSeqInts, GdnScanSeqRaw, gdn_scan_seq_launch,
};

/// The kernel-side geometry of one chunk: the conv channel count
/// `(2 * key_heads + valueHeads) * headDim` and the value-head outputs of the scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CpfDnGeometry {
    /// f32 columns of one conv row (`(2 * n_k + n_v) * head_dim`).
    pub(crate) channels: usize,
    /// Value heads of the scan (`n_v_heads`).
    pub(crate) value_heads: usize,
    /// Key/value head dimension (the compiled case is 128; the sizing stays generic).
    pub(crate) head_dim: usize,
}

/// The kernel-side chunk scratch of [`cpf_dn_chunk_bytes`], in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CpfDnChunkBytes {
    /// The conv output rows the scan reads (also its input rows): `m * channels * 4`.
    pub(crate) conv_out: usize,
    /// The scan output rows, written pre-norm and gated in place: `m * value_heads *
    /// head_dim * 4`.
    pub(crate) scan_out: usize,
    /// `conv_out + scan_out`.
    pub(crate) total: usize,
}

/// A `BadRequest` of this module.
fn refuse(detail: &str) -> EngError {
    EngError::BadRequest(format!("cpf_dn: {detail}"))
}

/// Bytes of the conv output rows of one chunk: `m` rows of `channels` f32 values. This is
/// the exact read reach of the scan over `conv` (the K0 output is `[m][channels]` with no
/// row padding), so a shorter `conv` range can never reach the kernel. Pure; `BadRequest`
/// when the byte count overflows `usize`.
pub(crate) fn cpf_dn_conv_out_bytes(m: usize, channels: usize) -> Result<usize, EngError> {
    m.checked_mul(channels)
        .and_then(|floats| floats.checked_mul(4))
        .ok_or_else(|| refuse("the conv-out byte count overflows"))
}

/// Bytes of the scan output rows of one chunk: `m` rows of `value_heads * head_dim` f32
/// values. ONE buffer: the scan writes the pre-norm rows and the out-gate rewrites them in
/// place. Pure; `BadRequest` when the byte count overflows `usize`.
pub(crate) fn cpf_dn_scan_out_bytes(
    m: usize,
    value_heads: usize,
    head_dim: usize,
) -> Result<usize, EngError> {
    m.checked_mul(value_heads)
        .and_then(|floats| floats.checked_mul(head_dim))
        .and_then(|floats| floats.checked_mul(4))
        .ok_or_else(|| refuse("the scan-out byte count overflows"))
}

/// The kernel-side chunk scratch of `geometry` over `m` tokens: the composition of
/// [`cpf_dn_conv_out_bytes`] and [`cpf_dn_scan_out_bytes`]. What is NOT in it, so nobody
/// double-counts, is listed in the module doc. Pure; `BadRequest` on overflow.
pub(crate) fn cpf_dn_chunk_bytes(
    geometry: &CpfDnGeometry,
    m: usize,
) -> Result<CpfDnChunkBytes, EngError> {
    let conv_out = cpf_dn_conv_out_bytes(m, geometry.channels)?;
    let scan_out = cpf_dn_scan_out_bytes(m, geometry.value_heads, geometry.head_dim)?;
    let total = conv_out
        .checked_add(scan_out)
        .ok_or_else(|| refuse("the chunk byte count overflows"))?;
    Ok(CpfDnChunkBytes {
        conv_out,
        scan_out,
        total,
    })
}

/// The one `eps` the kernels are validated for: the GGUF rms eps, the f32 `1e-6`.
pub(crate) const GDN_SCAN_EPS: f32 = 1.0e-6_f32;
/// Head size of the compiled case; any other value is the typed `head_dim` refusal.
pub(crate) const GDN_SCAN_HEAD_DIM: usize = 128;
const MAX_M: usize = 32768;
const MAX_K_HEADS: usize = 64;
const MAX_V_HEADS: usize = 128;
const MAX_LD: usize = 1 << 22;
const MAX_CTX: usize = (1 << 30) - 1;

/// Geometry of one scan call: `m` tokens at position `p0` of `ctx_tokens`, `n_k_heads` key
/// heads tiled over `n_v_heads` value heads of `head_dim`, the conv row length `conv_ld`, and
/// the projection row length `ld` with its `z`/`beta`/`alpha` float offsets.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScanShape {
    pub(crate) m: usize,
    pub(crate) p0: i64,
    pub(crate) ctx_tokens: usize,
    pub(crate) n_k_heads: usize,
    pub(crate) n_v_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) conv_ld: usize,
    pub(crate) ld: usize,
    pub(crate) z_off: usize,
    pub(crate) beta_off: usize,
    pub(crate) alpha_off: usize,
}

/// The per-layer weights: f32 `[n_v_heads]` `a` (ssm_a as stored) and `dt_bias`, and f32
/// `[128]` `norm_w`, each 4-byte aligned.
pub(crate) struct ScanWeights<'b> {
    pub(crate) a: DeviceRange<'b>,
    pub(crate) dt_bias: DeviceRange<'b>,
    pub(crate) norm_w: DeviceRange<'b>,
}

/// One scan call: the K0 output rows `conv`, the projection rows `proj`, the weights, the
/// in-place `state`, the `o_out` rows (pre-norm, then gated in place), the optional prefill
/// status word, `shape` and `eps` (exactly [`GDN_SCAN_EPS`]).
pub(crate) struct GdnScanSeqCall<'b> {
    pub(crate) shape: ScanShape,
    pub(crate) eps: f32,
    pub(crate) conv: DeviceRange<'b>,
    pub(crate) proj: DeviceRange<'b>,
    pub(crate) weights: ScanWeights<'b>,
    pub(crate) state: DeviceRange<'b>,
    pub(crate) o_out: DeviceRange<'b>,
    pub(crate) status: Option<DeviceRange<'b>>,
}

/// The bytes each range of a call must hold, from [`scan_bytes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScanBytes {
    pub(crate) conv: usize,
    pub(crate) proj: usize,
    pub(crate) a: usize,
    pub(crate) dt_bias: usize,
    pub(crate) norm_w: usize,
    pub(crate) state: usize,
    pub(crate) o_out: usize,
}

/// A `BadRequest` that names the rule that fired.
fn rule(tag: &str, detail: &str) -> EngError {
    EngError::BadRequest(format!("gdn_scan_seq: {tag}: {detail}"))
}

fn product(factors: &[usize]) -> Result<usize, EngError> {
    factors
        .iter()
        .try_fold(1usize, |acc, &f| acc.checked_mul(f))
        .ok_or_else(|| rule("size overflow", "a range size overflows"))
}

fn sum(a: usize, b: usize) -> Result<usize, EngError> {
    a.checked_add(b)
        .ok_or_else(|| rule("size overflow", "a range size overflows"))
}

fn to_int(name: &str, value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value).map_err(|_| rule("exceeds i32", &format!("{name} is {value}")))
}

/// The end column of `z`, of the beta logits and of the alpha logits in a `proj` row.
fn proj_ends(shape: &ScanShape) -> Result<[usize; 3], EngError> {
    let nv = shape.n_v_heads;
    Ok([
        sum(shape.z_off, product(&[nv, GDN_SCAN_HEAD_DIM])?)?,
        sum(shape.beta_off, nv)?,
        sum(shape.alpha_off, nv)?,
    ])
}

/// The bytes each range of `shape` must hold: `conv`/`o_out` by the sizing functions (their
/// exact kernel reach), `proj` by `(m - 1) * ld + max end`, the rest by the head counts.
pub(crate) fn scan_bytes(shape: ScanShape) -> Result<ScanBytes, EngError> {
    let before_last = shape.m.checked_sub(1).ok_or_else(|| rule("m", "m is 0"))?;
    let channels = product(&[
        sum(shape.n_k_heads, sum(shape.n_k_heads, shape.n_v_heads)?)?,
        GDN_SCAN_HEAD_DIM,
    ])?;
    let proj_cols = proj_ends(&shape)?.into_iter().max().unwrap_or(0);
    let reach = sum(product(&[before_last, shape.ld])?, proj_cols)?;
    Ok(ScanBytes {
        conv: cpf_dn_conv_out_bytes(shape.m, channels)?,
        proj: product(&[reach, 4])?,
        a: product(&[shape.n_v_heads, 4])?,
        dt_bias: product(&[shape.n_v_heads, 4])?,
        norm_w: product(&[GDN_SCAN_HEAD_DIM, 4])?,
        state: product(&[shape.n_v_heads, GDN_SCAN_HEAD_DIM, GDN_SCAN_HEAD_DIM, 4])?,
        o_out: cpf_dn_scan_out_bytes(shape.m, shape.n_v_heads, GDN_SCAN_HEAD_DIM)?,
    })
}

fn check_shape(shape: &ScanShape, eps: f32) -> Result<(), EngError> {
    if shape.head_dim != GDN_SCAN_HEAD_DIM {
        return Err(rule(
            "head_dim",
            &format!("head_dim is {}, not 128", shape.head_dim),
        ));
    }
    if !(1..=MAX_K_HEADS).contains(&shape.n_k_heads) {
        return Err(rule(
            "heads",
            &format!("n_k_heads is {}, not 1 to 64", shape.n_k_heads),
        ));
    }
    if !(1..=MAX_V_HEADS).contains(&shape.n_v_heads) {
        return Err(rule(
            "heads",
            &format!("n_v_heads is {}, not 1 to 128", shape.n_v_heads),
        ));
    }
    if !shape.n_v_heads.is_multiple_of(shape.n_k_heads) {
        return Err(rule(
            "heads",
            &format!(
                "n_v_heads {} is not a multiple of n_k_heads {}",
                shape.n_v_heads, shape.n_k_heads
            ),
        ));
    }
    if !(1..=MAX_M).contains(&shape.m) {
        return Err(rule("m", &format!("m is {}, not 1 to 32768", shape.m)));
    }
    let conv_cols = (2 * shape.n_k_heads + shape.n_v_heads) * GDN_SCAN_HEAD_DIM;
    if shape.conv_ld != conv_cols {
        return Err(rule(
            "conv_ld",
            &format!(
                "conv_ld is {}, not the row width {conv_cols}",
                shape.conv_ld
            ),
        ));
    }
    let ends = proj_ends(shape)?;
    let max_end = ends.into_iter().max().unwrap_or(0);
    if shape.ld < max_end || shape.ld > MAX_LD {
        return Err(rule(
            "ld",
            &format!("ld is {}, not {max_end} to {MAX_LD}", shape.ld),
        ));
    }
    if eps.to_bits() != GDN_SCAN_EPS.to_bits() {
        return Err(rule("eps", &format!("eps is {eps:e}, not exactly 1e-6")));
    }
    if !(1..=MAX_CTX).contains(&shape.ctx_tokens) {
        return Err(rule(
            "p0",
            &format!("ctx_tokens is {}, not 1 to {MAX_CTX}", shape.ctx_tokens),
        ));
    }
    if shape.p0 < 0 {
        return Err(rule("p0", &format!("p0 is {}, below 0", shape.p0)));
    }
    let span = (shape.p0 as u64)
        .checked_add(shape.m as u64)
        .ok_or_else(|| rule("p0", "p0 + m overflows"))?;
    if span > shape.ctx_tokens as u64 {
        return Err(rule(
            "p0",
            &format!("p0 + m is {span}, above ctx_tokens {}", shape.ctx_tokens),
        ));
    }
    Ok(())
}

/// Converts `shape` once, so no conversion can fail inside the launch closure.
pub(in crate::kernels::cuda) fn scan_ints(shape: &ScanShape) -> Result<GdnScanSeqInts, EngError> {
    Ok(GdnScanSeqInts {
        m: to_int("m", shape.m)?,
        p0: c_int::try_from(shape.p0).map_err(|_| rule("p0", "p0 is above i32::MAX"))?,
        ctx_tokens: to_int("ctx_tokens", shape.ctx_tokens)?,
        conv_ld: to_int("conv_ld", shape.conv_ld)?,
        ld: to_int("ld", shape.ld)?,
        z_off: to_int("z_off", shape.z_off)?,
        beta_off: to_int("beta_off", shape.beta_off)?,
        alpha_off: to_int("alpha_off", shape.alpha_off)?,
        n_k_heads: to_int("n_k_heads", shape.n_k_heads)?,
        n_v_heads: to_int("n_v_heads", shape.n_v_heads)?,
        head_dim: to_int("head_dim", shape.head_dim)?,
    })
}

fn check_fit(call: &GdnScanSeqCall<'_>, need: &ScanBytes) -> Result<(), EngError> {
    require_fit("conv", &call.conv, need.conv, 4)?;
    require_fit("proj", &call.proj, need.proj, 4)?;
    require_fit("a", &call.weights.a, need.a, 4)?;
    require_fit("dt_bias", &call.weights.dt_bias, need.dt_bias, 4)?;
    require_fit("norm_w", &call.weights.norm_w, need.norm_w, 4)?;
    require_fit("state", &call.state, need.state, 4)?;
    require_fit("o_out", &call.o_out, need.o_out, 4)?;
    if let Some(status) = &call.status {
        require_fit("status", status, 4, 4)?;
    }
    Ok(())
}

/// `state` and `o_out` are written: each apart from every other range and each other; the
/// status word, which the guard ORs, apart from every range. Reads may share bytes.
fn check_apart(call: &GdnScanSeqCall<'_>, need: &ScanBytes) -> Result<(), EngError> {
    let state = Claim::of("state", &call.state, need.state);
    let out = Claim::of("o_out", &call.o_out, need.o_out);
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
    if let Some(status) = &call.status {
        let word = Claim::of("status", status, 4);
        require_apart(&word, &state)?;
        require_apart(&word, &out)?;
        for read in &reads {
            require_apart(&word, read)?;
        }
    }
    Ok(())
}

impl CudaContext {
    /// Queues the sequential scan of `call` on `stream`: K1 (the value-sliced scan, the
    /// state updated in place, the pre-norm rows written) then K2 (the gated output RMSNorm
    /// in place on the same rows). No allocation, no copies, no synchronization.
    ///
    /// If a layer launch faults mid-chunk, the in-place state is advanced only up to that
    /// layer, so the driver must poison the session, not resume it.
    ///
    /// Returns `BadRequest`, before any native call, for a stream or buffer of another
    /// context, a shape the kernels refuse, a range too short or misaligned, and a
    /// `state`/`o_out`/status overlap.
    pub(crate) fn gdn_scan_seq(
        &self,
        stream: &DeviceStream,
        call: GdnScanSeqCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        check_shape(&call.shape, call.eps)?;
        let ints = scan_ints(&call.shape)?;
        let need = scan_bytes(call.shape)?;
        check_fit(&call, &need)?;
        check_apart(&call, &need)?;
        let conv = self.raw_range(&call.conv)?;
        let proj = self.raw_range(&call.proj)?;
        let a = self.raw_range(&call.weights.a)?;
        let dt_bias = self.raw_range(&call.weights.dt_bias)?;
        let norm_w = self.raw_range(&call.weights.norm_w)?;
        let state = self.raw_range(&call.state)?;
        let o_out = self.raw_range(&call.o_out)?;
        let status = call
            .status
            .as_ref()
            .map(|r| self.raw_range(r))
            .transpose()?;
        let eps = call.eps;
        self.launch_on(stream, move |raw| {
            gdn_scan_seq_launch(
                raw,
                GdnScanSeqRaw {
                    conv,
                    proj,
                    a,
                    dt_bias,
                    norm_w,
                    state,
                    o_out,
                    status,
                    ints,
                    eps,
                },
            )
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
