//! Q4_K expert kernels on the CPU with activations on the 32-value group grid (the q8_1 grid
//! of the device path), safe Rust (the kernel of the OFF-2 tier).
//!
//! WHY this grid: the Q8_K kernel of [`super::expert_cpu_q8k`] quantizes activations per
//! 256 values, llama.cpp's CPU scheme. Its rounding noise is 1.7 to 3.5 percent of an expert
//! output and, once the layer inputs differ by a hair, independent of the oracle's. The model
//! level gate (b) of OFF-2 failed at 80 percent CPU-served picks with that grid (p95 0.686
//! against the bound 0.651) and passed with this one (0.478), better than the all-device run:
//! 32-value groups bound the noise of an outlier to its own group. A Q4_K sub-block is 32
//! values wide, so each activation group lines up with exactly one weight sub-block.
//!
//! This mirrors the device's GRID (groups of 32, `d = amax / 127`, `q = round(x / d)`), not its bits: the
//! CPU keeps `d` in f32 and the group sums as exact integers, where the device stores both as f16
//! (`ds = half2(d, sum)`); the rule-C bound of the device tests carries that difference.
//!
//! Numerics (binding; the scalar path is the reference, [`DotBackend`] may dispatch to the
//! scoped AVX2 module `expert_cpu_avx2`, which reproduces it bit for bit). An activation row is
//! quantized per group of 32: `amax = max|x_i|`; if `amax == 0` then `d = 0`, every `q = 0`;
//! else `d = amax / 127.0`, `inv = 127.0 / amax` (f32), `q_i = round(x_i * inv)` (ties away
//! from zero) clamped to `-127..=127`; `s` is the exact i32 sum of the group's `q`. For one
//! 256-value weight block with sub-block `(sc_j, m_j)`, group `g = 8 * b + j`:
//! `isum_j = sum of q4_i * q8_i` over the 32 values (exact i32);
//! `p1_j = d_g * ((sc_j * isum_j) as f32)` and `p2_j = d_g * ((m_j * s_g) as f32)` (both
//! integer products stay below 2^24, so the casts are exact; one f32 multiply each);
//! `t1 = tree8(p1)`, `t2 = tree8(p2)` with `tree8(p) = ((p0 + p1) + (p2 + p3)) + ((p4 + p5) +
//! (p6 + p7))`; block value `= d_w * t1 - dmin_w * t2`; the row result is the f32 sum of the
//! block values in block order from +0.0. Only integer parts and the fixed float order are
//! involved, so any correct SIMD arrangement gives the same bits.

use super::expert_cpu::{
    ExpertCpuError, ExpertShape, ExpertSlices, Q4K_BLOCK_BYTES, Q4K_BLOCK_VALUES, f16_bits_to_f32,
    scale_min,
};
use super::expert_cpu_q8k::DotBackend;

/// Activations per quantization group: one Q4_K sub-block.
pub(crate) const GROUP: usize = 32;
/// Groups per Q4_K block.
pub(crate) const GROUPS_PER_BLOCK: usize = Q4K_BLOCK_VALUES / GROUP;

/// One quantized activation row on the 32-value grid: a scale `d` and an exact sum `s` of the
/// `q` values per group, and one signed byte `q` per value in `-127..=127`. Built by
/// [`Q8gRow::zeros`] and filled by [`quantize_q8g`]; nothing else writes them.
pub struct Q8gRow {
    /// Per-group scale `d = amax / 127`.
    pub(crate) d: Vec<f32>,
    /// One quantized activation per input value.
    pub(crate) q: Vec<i8>,
    /// Exact sum of the 32 `q` of each group (`|s| <= 4064`).
    pub(crate) s: Vec<i32>,
}

impl Q8gRow {
    /// Allocates a zeroed row for `k` activations; `k` must be a positive multiple of
    /// [`Q4K_BLOCK_VALUES`] (whole weight blocks), else `BadShape`.
    pub fn zeros(k: usize) -> Result<Self, ExpertCpuError> {
        if k == 0 || !k.is_multiple_of(Q4K_BLOCK_VALUES) {
            return Err(ExpertCpuError::BadShape(format!(
                "k = {k} is not a positive multiple of {Q4K_BLOCK_VALUES}"
            )));
        }
        Ok(row_zeros(k))
    }
}

/// Unvalidated allocation shared by [`Q8gRow::zeros`] and the expert scratch (floor sizes; the
/// forward rejects a shape that is not a block multiple before it reads any scratch byte).
fn row_zeros(k: usize) -> Q8gRow {
    Q8gRow {
        d: vec![0.0; k / GROUP],
        q: vec![0; k],
        s: vec![0; k / GROUP],
    }
}

/// Quantizes `x` into `out` as described in the module doc. `x` must be finite, a positive
/// multiple of [`Q4K_BLOCK_VALUES`] long and match `out`'s lengths; every violation is
/// reported before a byte of `out` is written. Deterministic.
pub fn quantize_q8g(x: &[f32], out: &mut Q8gRow) -> Result<(), ExpertCpuError> {
    if x.is_empty() || !x.len().is_multiple_of(Q4K_BLOCK_VALUES) {
        return Err(ExpertCpuError::BadShape(format!(
            "x length {} is not a positive multiple of {}",
            x.len(),
            Q4K_BLOCK_VALUES
        )));
    }
    let groups = x.len() / GROUP;
    for (what, got, want) in [
        ("out.d", out.d.len(), groups),
        ("out.q", out.q.len(), x.len()),
        ("out.s", out.s.len(), groups),
    ] {
        if got != want {
            return Err(ExpertCpuError::BadLength { what, got, want });
        }
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(ExpertCpuError::BadShape(format!(
            "activation row of length {} has non-finite values",
            x.len()
        )));
    }
    for (g, xs) in x.as_chunks::<GROUP>().0.iter().enumerate() {
        let amax = xs.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let qs = &mut out.q[g * GROUP..][..GROUP];
        if amax == 0.0 {
            out.d[g] = 0.0;
            out.s[g] = 0;
            qs.fill(0);
            continue;
        }
        let (d, inv) = (amax / 127.0, 127.0 / amax);
        out.d[g] = d;
        for (q, &v) in qs.iter_mut().zip(xs) {
            *q = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
        out.s[g] = qs.iter().map(|&q| i32::from(q)).sum();
    }
    Ok(())
}

/// The fixed pairwise sum of eight products: `((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7))`.
/// The AVX2 path produces the same order with two `hadd` steps and one lane add.
pub(crate) fn tree8(p: [f32; GROUPS_PER_BLOCK]) -> f32 {
    ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]))
}

/// The float tail of one block, shared verbatim by the scalar and AVX2 paths: `d_w * t1 -
/// dmin_w * t2`, each product rounded once.
pub(crate) fn q8g_block_value(d_w: f32, dmin_w: f32, t1: f32, t2: f32) -> f32 {
    d_w * t1 - dmin_w * t2
}

/// Integer dot of one Q4_K row (`n` whole [`Q4K_BLOCK_BYTES`] blocks) with a quantized row:
/// the reference of the module doc. Nibbles are indexed exactly as the f32 kernel does:
/// chunk `c = j / 2` of the 32 data bytes, low nibbles are sub-block `2c`, high nibbles `2c + 1`.
/// `row` and `xq` must describe the same value count; the matvec validates that.
pub fn q4k_q8g_row_dot_scalar(row: &[u8], xq: &Q8gRow) -> f32 {
    let mut acc = 0.0f32;
    for (b, blk) in row.as_chunks::<Q4K_BLOCK_BYTES>().0.iter().enumerate() {
        let d_w = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin_w = f16_bits_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let (mut p1, mut p2) = ([0.0f32; GROUPS_PER_BLOCK], [0.0f32; GROUPS_PER_BLOCK]);
        for j in 0..GROUPS_PER_BLOCK {
            let g = GROUPS_PER_BLOCK * b + j;
            let (sc, m) = scale_min(j, blk);
            let (c, high) = (j / 2, j % 2 == 1);
            let mut isum = 0i32;
            for t in 0..GROUP {
                let byte = blk[16 + 32 * c + t];
                let nib = if high { byte >> 4 } else { byte & 0x0f };
                isum += i32::from(nib) * i32::from(xq.q[g * GROUP + t]);
            }
            p1[j] = xq.d[g] * (i32::from(sc) * isum) as f32;
            p2[j] = xq.d[g] * (i32::from(m) * xq.s[g]) as f32;
        }
        acc += q8g_block_value(d_w, dmin_w, tree8(p1), tree8(p2));
    }
    acc
}

/// One row through `backend`; the AVX2 arm re-checks the feature bit inside its module.
fn row_dot(backend: DotBackend, row: &[u8], xq: &Q8gRow) -> f32 {
    if backend == DotBackend::scalar() {
        q4k_q8g_row_dot_scalar(row, xq)
    } else {
        row_dot_simd(row, xq)
    }
}

/// AVX2 hook: enters the scoped intrinsics module on x86_64, which re-checks the feature bit
/// before any intrinsic runs; elsewhere `DotBackend::detect` never reports `Avx2`.
#[cfg(target_arch = "x86_64")]
fn row_dot_simd(row: &[u8], xq: &Q8gRow) -> f32 {
    super::expert_cpu_avx2::q4k_q8g_row_dot(row, xq)
}

/// Non-x86_64 stand-in: the backend type can never select the intrinsics path here.
#[cfg(not(target_arch = "x86_64"))]
fn row_dot_simd(row: &[u8], xq: &Q8gRow) -> f32 {
    q4k_q8g_row_dot_scalar(row, xq)
}

/// `BadLength` for the first `xq` buffer that does not describe exactly `k` values.
fn xq_bad_length(xq: &Q8gRow, k: usize) -> Option<ExpertCpuError> {
    [
        ("xq.d", xq.d.len(), k / GROUP),
        ("xq.q", xq.q.len(), k),
        ("xq.s", xq.s.len(), k / GROUP),
    ]
    .into_iter()
    .find(|(_, got, want)| got != want)
    .map(|(what, got, want)| ExpertCpuError::BadLength { what, got, want })
}

/// The row loop shared by the public matvec and the expert forward once every length check
/// has passed (no allocation, rows in order).
fn matvec_rows(backend: DotBackend, w: &[u8], k: usize, xq: &Q8gRow, out: &mut [f32]) {
    let row_bytes = k / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES;
    for (o, row) in out.iter_mut().zip(w.chunks_exact(row_bytes)) {
        *o = row_dot(backend, row, xq);
    }
}

/// Dots `rows` rows of `k` Q4_K weights with the quantized activation row `xq` through
/// `backend`, writing `rows` f32 results to `out`. Contract of `q4k_q8k_matvec`: `k` a
/// positive multiple of [`Q4K_BLOCK_VALUES`], `rows` nonzero, `w` exactly `rows * k / 256 * 144`
/// bytes, `xq` describing exactly `k` values, `out` exactly `rows` long; every violation is
/// reported before any output byte is touched.
pub fn q4k_q8g_matvec(
    backend: DotBackend,
    w: &[u8],
    rows: usize,
    k: usize,
    xq: &Q8gRow,
    out: &mut [f32],
) -> Result<(), ExpertCpuError> {
    if k == 0 || !k.is_multiple_of(Q4K_BLOCK_VALUES) {
        return Err(ExpertCpuError::BadShape(format!(
            "k = {k} is not a positive multiple of {Q4K_BLOCK_VALUES}"
        )));
    }
    if rows == 0 {
        return Err(ExpertCpuError::BadShape(format!(
            "rows = 0 in a {rows}x{k} Q4_K x Q8 matvec"
        )));
    }
    let want = (k / Q4K_BLOCK_VALUES)
        .checked_mul(Q4K_BLOCK_BYTES)
        .and_then(|row_bytes| rows.checked_mul(row_bytes))
        .ok_or_else(|| ExpertCpuError::BadShape(format!("{rows} rows of {k} values overflow")))?;
    if let Some(err) = xq_bad_length(xq, k) {
        return Err(err);
    }
    for (what, got, want_len) in [("out", out.len(), rows), ("w", w.len(), want)] {
        if got != want_len {
            return Err(ExpertCpuError::BadLength {
                what,
                got,
                want: want_len,
            });
        }
    }
    matvec_rows(backend, w, k, xq, out);
    Ok(())
}

/// Reusable scratch for [`expert_forward_q8g`] so a call allocates nothing: the f32
/// `gate`/`up`/`act` rows (`inter` long) and two quantized rows, `qx` for the hidden input
/// and `qh` for the intermediate activations.
pub struct ExpertScratchQ8g {
    gate: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    qx: Q8gRow,
    qh: Q8gRow,
}

impl ExpertScratchQ8g {
    /// Allocates scratch sized for one expert of `shape` (floor-sized for a shape that is not
    /// a block multiple; the forward rejects such shapes before reading any scratch byte).
    pub fn new(shape: ExpertShape) -> Self {
        Self {
            gate: vec![0.0; shape.inter],
            up: vec![0.0; shape.inter],
            act: vec![0.0; shape.inter],
            qx: row_zeros(shape.hidden),
            qh: row_zeros(shape.inter),
        }
    }
}

/// Runs one expert through the per-32 integer path: quantize `x` (`hidden` values), gate and
/// up rows against it, `h = silu(gate) * up` in f32 with the silu of the f32 reference
/// (`x / (1 + exp(-x))`), quantize `h` (`inter` values), down rows into `out`.
///
/// Validation is the `expert_forward` contract (shape first, then every implied length, `out`
/// untouched on error) plus the quantizer's refusal of non-finite `x`; with matching scratch a
/// call allocates nothing.
pub fn expert_forward_q8g(
    backend: DotBackend,
    shape: ExpertShape,
    s: &ExpertSlices<'_>,
    x: &[f32],
    scratch: &mut ExpertScratchQ8g,
    out: &mut [f32],
) -> Result<(), ExpertCpuError> {
    let ExpertShape { hidden, inter } = shape;
    if hidden == 0
        || inter == 0
        || !hidden.is_multiple_of(Q4K_BLOCK_VALUES)
        || !inter.is_multiple_of(Q4K_BLOCK_VALUES)
    {
        return Err(ExpertCpuError::BadShape(format!(
            "expert shape {hidden}x{inter}: extents must be positive multiples of \
             {Q4K_BLOCK_VALUES}"
        )));
    }
    let slice_len = |rows: usize, cols: usize| {
        rows.saturating_mul(cols / Q4K_BLOCK_VALUES)
            .saturating_mul(Q4K_BLOCK_BYTES)
    };
    let (side, down_len) = (slice_len(inter, hidden), slice_len(hidden, inter));
    let check = |what: &'static str, got: usize, want: usize| -> Result<(), ExpertCpuError> {
        if got == want {
            Ok(())
        } else {
            Err(ExpertCpuError::BadLength { what, got, want })
        }
    };
    check("x", x.len(), hidden)?;
    check("out", out.len(), hidden)?;
    check("gate", s.gate.len(), side)?;
    check("up", s.up.len(), side)?;
    check("down", s.down.len(), down_len)?;
    for (what, got, want) in [
        ("scratch.gate", scratch.gate.len(), inter),
        ("scratch.up", scratch.up.len(), inter),
        ("scratch.act", scratch.act.len(), inter),
    ] {
        check(what, got, want)?;
    }
    for (what, row, k) in [
        ("scratch.qx", &scratch.qx, hidden),
        ("scratch.qh", &scratch.qh, inter),
    ] {
        if let Some(ExpertCpuError::BadLength { got, want, .. }) = xq_bad_length(row, k) {
            return Err(ExpertCpuError::BadLength { what, got, want });
        }
    }
    let ExpertScratchQ8g {
        gate,
        up,
        act,
        qx,
        qh,
    } = scratch;
    quantize_q8g(x, qx)?;
    matvec_rows(backend, s.gate, hidden, qx, gate);
    matvec_rows(backend, s.up, hidden, qx, up);
    for i in 0..inter {
        let g = gate[i];
        act[i] = g / (1.0 + (-g).exp()) * up[i];
    }
    quantize_q8g(act, qh)?;
    matvec_rows(backend, s.down, inter, qh, out);
    Ok(())
}

#[cfg(test)]
mod tests;
