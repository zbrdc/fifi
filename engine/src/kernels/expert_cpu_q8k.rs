//! Q8_K integer-dot expert kernels on the CPU, safe Rust (staged for later
//! integration, OFF-2 C6/C7).
//!
//! The safe f32 kernel in [`super::expert_cpu`] is DRAM-bound at ~2.6 GB/s per thread
//! (lane L, 2026-10-13) because it expands every Q4_K nibble to f32 before the dot.
//! This module keeps the activation in llama.cpp's Q8_K integer form and dots it
//! against the raw nibbles: per 256-value block, `sumi = sum_j sc_j * isum_j` and
//! `summ = sum_j m_j * (bsums[2j] + bsums[2j+1])` are exact i32 sums, then the float
//! tail `d_x * (d_w * sumi - dmin_w * summ)` of [`q8k_block_value`]. That is the same
//! numerics class the GPU path already uses (q8_1 activations). The scalar path here
//! is the reference and the fallback; [`DotBackend`] dispatches to the scoped AVX2
//! module `expert_cpu_avx2` (decision 0017, Addendum 2), which reproduces the exact
//! integers and shares the float tail, so both backends are bitwise identical. The
//! f32 `expert_forward` stays unchanged and remains the reference.

use super::expert_cpu::{
    ExpertCpuError, ExpertShape, ExpertSlices, Q4K_BLOCK_BYTES, Q4K_BLOCK_VALUES, f16_bits_to_f32,
    scale_min,
};

/// One quantized activation row (the llama.cpp Q8_K layout): a scale `d` per
/// 256-value block, one signed byte `q` per value in `-127..=127`, and the exact i16
/// sums `bsums` of the 16 consecutive groups of 16 `q`s per block (the weight minima
/// only ever need those group sums). Rows are built by [`Q8kRow::zeros`] and filled by
/// [`quantize_q8k`]; nothing else writes them.
pub struct Q8kRow {
    /// Per-block scale `d = amax / 127`, dimensioned like the activation values.
    pub(crate) d: Vec<f32>,
    /// One quantized activation per input value, `-127..=127`.
    pub(crate) q: Vec<i8>,
    /// 16 exact group sums per block (i16: `16 * 127` fits with room to spare).
    pub(crate) bsums: Vec<i16>,
}

impl Q8kRow {
    /// Allocates a zeroed row for `k` activations; `k` must be a positive multiple of
    /// [`Q4K_BLOCK_VALUES`] (whole blocks), else `BadShape`.
    pub fn zeros(k: usize) -> Result<Self, ExpertCpuError> {
        if k == 0 || !k.is_multiple_of(Q4K_BLOCK_VALUES) {
            return Err(ExpertCpuError::BadShape(format!(
                "k = {k} is not a positive multiple of {Q4K_BLOCK_VALUES}"
            )));
        }
        Ok(row_zeros(k))
    }
}

/// Unvalidated allocation shared by [`Q8kRow::zeros`] and the expert scratch: buffers
/// sized for `k` values by floor division. A caller passing a non-block multiple
/// rejects that shape in validation long before any scratch byte is read.
fn row_zeros(k: usize) -> Q8kRow {
    let blocks = k / Q4K_BLOCK_VALUES;
    Q8kRow {
        d: vec![0.0; blocks],
        q: vec![0; k],
        bsums: vec![0; 16 * blocks],
    }
}

/// Quantizes `x` into `out`, one 256-value block at a time: `amax = max|x_i|`; an
/// all-zero block stores `d = 0.0` with all-zero `q` and `bsums`; otherwise `d =
/// amax / 127.0`, `inv = 127.0 / amax` (both f32 divisions), `q_i = round(x_i * inv)`
/// (Rust `f32::round`, ties away from zero) clamped to `-127..=127` as i8, and
/// `bsums[j]` is the exact i16 sum of the `q` values `16j..16j+16` of the block.
///
/// `x` must be finite and a positive multiple of [`Q4K_BLOCK_VALUES`] long and must
/// match `out`'s lengths; every violation is reported before a byte of `out` is
/// written. Deterministic: identical inputs give bitwise-identical rows.
pub fn quantize_q8k(x: &[f32], out: &mut Q8kRow) -> Result<(), ExpertCpuError> {
    if x.is_empty() || !x.len().is_multiple_of(Q4K_BLOCK_VALUES) {
        return Err(ExpertCpuError::BadShape(format!(
            "x length {} is not a positive multiple of {}",
            x.len(),
            Q4K_BLOCK_VALUES
        )));
    }
    let blocks = x.len() / Q4K_BLOCK_VALUES;
    for (what, got, want) in [
        ("out.d", out.d.len(), blocks),
        ("out.q", out.q.len(), x.len()),
        ("out.bsums", out.bsums.len(), 16 * blocks),
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
    for (b, xs) in x.as_chunks::<Q4K_BLOCK_VALUES>().0.iter().enumerate() {
        let mut amax = 0.0f32;
        for &v in xs {
            amax = amax.max(v.abs());
        }
        let qs = &mut out.q[Q4K_BLOCK_VALUES * b..][..Q4K_BLOCK_VALUES];
        let sums = &mut out.bsums[16 * b..][..16];
        if amax == 0.0 {
            out.d[b] = 0.0;
            qs.fill(0);
            sums.fill(0);
            continue;
        }
        let d = amax / 127.0;
        let inv = 127.0 / amax;
        out.d[b] = d;
        for (q, &v) in qs.iter_mut().zip(xs) {
            *q = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
        for (j, sum) in sums.iter_mut().enumerate() {
            *sum = qs[16 * j..][..16]
                .iter()
                .map(|&q| i32::from(q))
                .sum::<i32>() as i16;
        }
    }
    Ok(())
}

/// The float tail of one block: `t1 = d_w * sumi`, `t2 = dmin_w * summ`, block value
/// `= d_x * (t1 - t2)`, in exactly that operation order. This is the ONE piece of
/// float arithmetic of the integer dot, shared verbatim by the scalar and AVX2 paths;
/// `sumi`/`summ` are exact integers, so reproducing them reproduces the bits.
pub(crate) fn q8k_block_value(d_x: f32, d_w: f32, dmin_w: f32, sumi: i32, summ: i32) -> f32 {
    let t1 = d_w * sumi as f32;
    let t2 = dmin_w * summ as f32;
    d_x * (t1 - t2)
}

/// Integer dot of one Q4_K row (`n` whole [`Q4K_BLOCK_BYTES`] blocks) with a
/// quantized row: per block, `isum_j` is the exact i32 sum of `q4_i * q8_i` over the
/// 32 values of sub-block `j` (nibbles indexed exactly as the f32 kernel does: chunk
/// `c = j / 2`, low nibbles are sub-block `2c`, high nibbles `2c + 1`),
/// `sumi = sum_j sc_j * isum_j`, `summ = sum_j m_j * (bsums[2j] + bsums[2j + 1])`,
/// then [`q8k_block_value`]; block values accumulate into one f32 accumulator in
/// block order from `+0.0`. Overflow bounds: `|sumi| <= 63 * 15 * 127 * 256` =
/// 30.7M and `|summ| <= 63 * 2 * 16 * 127 * 8` = 2.05M, both far inside i32.
/// `row` and `xq` must describe the same value count; the matvec validates that.
pub fn q4k_q8k_row_dot_scalar(row: &[u8], xq: &Q8kRow) -> f32 {
    let mut acc = 0.0f32;
    for (b, blk) in row.as_chunks::<Q4K_BLOCK_BYTES>().0.iter().enumerate() {
        let d_w = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin_w = f16_bits_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let q8 = &xq.q[Q4K_BLOCK_VALUES * b..][..Q4K_BLOCK_VALUES];
        let bsums = &xq.bsums[16 * b..][..16];
        let (mut sumi, mut summ) = (0i32, 0i32);
        for j in 0..8 {
            let (sc, m) = scale_min(j, blk);
            let (c, high) = (j / 2, j % 2 == 1);
            let mut isum = 0i32;
            for t in 0..32 {
                let byte = blk[16 + 32 * c + t];
                let nib = if high { byte >> 4 } else { byte & 0x0f };
                isum += i32::from(nib) * i32::from(q8[32 * j + t]);
            }
            sumi += i32::from(sc) * isum;
            let msum = i32::from(bsums[2 * j]) + i32::from(bsums[2 * j + 1]);
            summ += i32::from(m) * msum;
        }
        acc += q8k_block_value(xq.d[b], d_w, dmin_w, sumi, summ);
    }
    acc
}

/// Which dot implementation a call uses. The `Avx2` state can only be obtained from
/// [`DotBackend::detect`], which performs the runtime CPU check, so safe code cannot
/// ask for the AVX2 path on a CPU without it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DotBackend(Backend);

/// Private discriminant of [`DotBackend`]: `Avx2` exists on every target so the type
/// is uniform, but `detect()` only ever produces it on x86_64 with the feature bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Scalar,
    #[cfg_attr(
        not(target_arch = "x86_64"),
        allow(
            dead_code,
            reason = "only `detect()` on x86_64 builds it; the other targets keep the match arms"
        )
    )]
    Avx2,
}

impl DotBackend {
    /// Probes the host: `Avx2` iff the target is x86_64 and the CPU reports the AVX2
    /// feature at runtime; `Scalar` otherwise (the standard library caches the probe).
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            return Self(Backend::Avx2);
        }
        Self(Backend::Scalar)
    }

    /// The scalar reference path, on every target and CPU.
    pub fn scalar() -> Self {
        Self(Backend::Scalar)
    }

    /// `"scalar"` or `"avx2"`: the implementation this backend selects.
    pub fn name(&self) -> &'static str {
        match self.0 {
            Backend::Scalar => "scalar",
            Backend::Avx2 => "avx2",
        }
    }
}

/// One row through `backend`; the AVX2 arm re-checks the feature bit inside its
/// module before any intrinsic runs.
fn row_dot(backend: DotBackend, row: &[u8], xq: &Q8kRow) -> f32 {
    match backend {
        DotBackend(Backend::Scalar) => q4k_q8k_row_dot_scalar(row, xq),
        DotBackend(Backend::Avx2) => row_dot_simd(row, xq),
    }
}

/// AVX2 hook: enters the scoped intrinsics module on x86_64, which re-checks the
/// feature bit before any intrinsic runs; on other targets `DotBackend::detect`
/// never reports `Avx2`, so the arm falls back to the scalar reference there.
#[cfg(target_arch = "x86_64")]
fn row_dot_simd(row: &[u8], xq: &Q8kRow) -> f32 {
    super::expert_cpu_avx2::q4k_q8k_row_dot(row, xq)
}

/// Non-x86_64 stand-in for the AVX2 hook (see the x86_64 twin): the backend type
/// can never select the intrinsics path here, so the scalar reference answers.
#[cfg(not(target_arch = "x86_64"))]
fn row_dot_simd(row: &[u8], xq: &Q8kRow) -> f32 {
    q4k_q8k_row_dot_scalar(row, xq)
}

/// `BadLength` for the first `xq` buffer that does not describe exactly `k` values.
fn xq_bad_length(xq: &Q8kRow, k: usize) -> Option<ExpertCpuError> {
    let blocks = k / Q4K_BLOCK_VALUES;
    [
        ("xq.d", xq.d.len(), blocks),
        ("xq.q", xq.q.len(), k),
        ("xq.bsums", xq.bsums.len(), 16 * blocks),
    ]
    .into_iter()
    .find(|(_, got, want)| got != want)
    .map(|(what, got, want)| ExpertCpuError::BadLength { what, got, want })
}

/// The row loop shared by the public matvec and the expert forward once every length
/// check has passed (mirrors `expert_cpu::matvec_rows`; no allocation, rows in order).
fn matvec_rows(backend: DotBackend, w: &[u8], k: usize, xq: &Q8kRow, out: &mut [f32]) {
    let row_bytes = k / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES;
    for (o, row) in out.iter_mut().zip(w.chunks_exact(row_bytes)) {
        *o = row_dot(backend, row, xq);
    }
}

/// Dots `rows` rows of `k` Q4_K weights with the quantized activation row `xq`
/// through `backend`, writing `rows` f32 results to `out`.
///
/// The same contract as `expert_cpu::q4k_matvec`: `k` a positive multiple of
/// [`Q4K_BLOCK_VALUES`], `rows` nonzero, `w` exactly `rows * k / 256 * 144` bytes,
/// `xq` describing exactly `k` values, `out` exactly `rows` long; every violation is
/// reported before any output byte is touched. With a matching row from
/// [`quantize_q8k`] the call allocates nothing.
pub fn q4k_q8k_matvec(
    backend: DotBackend,
    w: &[u8],
    rows: usize,
    k: usize,
    xq: &Q8kRow,
    out: &mut [f32],
) -> Result<(), ExpertCpuError> {
    if k == 0 || !k.is_multiple_of(Q4K_BLOCK_VALUES) {
        return Err(ExpertCpuError::BadShape(format!(
            "k = {k} is not a positive multiple of {Q4K_BLOCK_VALUES}"
        )));
    }
    if rows == 0 {
        return Err(ExpertCpuError::BadShape(format!(
            "rows = 0 in a {rows}x{k} Q4_K x Q8_K matvec"
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

/// Reusable scratch for [`expert_forward_q8k`] so a call allocates nothing: the f32
/// `gate`/`up`/`act` rows (`inter` long) and two quantized rows, `qx` for the hidden
/// input and `qh` for the intermediate activations.
pub struct ExpertScratchQ8k {
    /// Gate projection output, `inter` values.
    gate: Vec<f32>,
    /// Up projection output, `inter` values.
    up: Vec<f32>,
    /// `silu(gate) * up`, `inter` values.
    act: Vec<f32>,
    /// Quantized hidden input (`hidden` values).
    qx: Q8kRow,
    /// Quantized intermediate activations (`inter` values).
    qh: Q8kRow,
}

impl ExpertScratchQ8k {
    /// Allocates scratch sized for one expert of `shape`: value buffers `inter` long
    /// and quantized rows for `hidden` and `inter` inputs. A shape that is not a
    /// positive multiple of 256 still allocates floor-sized buffers; the forward
    /// rejects such shapes before reading any scratch byte.
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

/// Runs one expert through the integer path: quantize `x` (`hidden` values), gate
/// and up rows against it, `h = silu(gate) * up` in f32 with the same silu as the
/// reference `expert_forward` (`x / (1 + exp(-x))`), quantize `h` (`inter` values),
/// down rows into `out`.
///
/// Validation is the `expert_forward` contract verbatim (shape first, then every
/// implied length, `out` untouched on error) plus the quantizer's refusal of
/// non-finite `x`; with matching scratch a call allocates nothing.
pub fn expert_forward_q8k(
    backend: DotBackend,
    shape: ExpertShape,
    s: &ExpertSlices<'_>,
    x: &[f32],
    scratch: &mut ExpertScratchQ8k,
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
    let blocks = |k: usize| k / Q4K_BLOCK_VALUES;
    check("x", x.len(), hidden)?;
    check("out", out.len(), hidden)?;
    check("gate", s.gate.len(), side)?;
    check("up", s.up.len(), side)?;
    check("down", s.down.len(), down_len)?;
    check("scratch.gate", scratch.gate.len(), inter)?;
    check("scratch.up", scratch.up.len(), inter)?;
    check("scratch.act", scratch.act.len(), inter)?;
    check("scratch.qx.d", scratch.qx.d.len(), blocks(hidden))?;
    check("scratch.qx.q", scratch.qx.q.len(), hidden)?;
    check(
        "scratch.qx.bsums",
        scratch.qx.bsums.len(),
        16 * blocks(hidden),
    )?;
    check("scratch.qh.d", scratch.qh.d.len(), blocks(inter))?;
    check("scratch.qh.q", scratch.qh.q.len(), inter)?;
    check(
        "scratch.qh.bsums",
        scratch.qh.bsums.len(),
        16 * blocks(inter),
    )?;
    let ExpertScratchQ8k {
        gate,
        up,
        act,
        qx,
        qh,
    } = scratch;
    quantize_q8k(x, qx)?;
    matvec_rows(backend, s.gate, hidden, qx, gate);
    matvec_rows(backend, s.up, hidden, qx, up);
    for i in 0..inter {
        let g = gate[i];
        act[i] = g / (1.0 + (-g).exp()) * up[i];
    }
    quantize_q8k(act, qh)?;
    matvec_rows(backend, s.down, inter, qh, out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        DotBackend, ExpertScratchQ8k, Q8kRow, expert_forward_q8k, q4k_q8k_matvec, quantize_q8k,
    };
    use crate::kernels::expert_cpu::{ExpertCpuError, Q4K_BLOCK_BYTES, Q4K_BLOCK_VALUES};

    /// Tiny deterministic LCG (Knuth MMIX constants); tests never touch the OS.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            // Knuth MMIX step; an LCG's low bits are weak, so callers take the top.
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 11
        }
        fn f32_between(&mut self, lo: f32, hi: f32) -> f32 {
            // `next()` tops out at 2^53 (top 53 bits of the state), so the scale is
            // 2^53, NOT u64::MAX: dividing by u64::MAX would squeeze every value into
            // 0.05% of the range (a sliver the shared expert_cpu helper sits in).
            lo + (hi - lo) * (self.next() as f32 / (1u64 << 53) as f32)
        }
    }

    #[test]
    fn quantizer_round_trips_within_half_a_step_and_pins_the_edges() {
        // Per element |x - d*q| <= d/2 + 4 ulp of the range top (the three roundings
        // in d, inv and x*inv); the amax element maps to +-127; an all-zero block
        // gives d = 0 and q = 0; bsums are the exact sums; repeats are bitwise
        // identical; bad input is refused with out untouched.
        let mut rng = Lcg(0x8E57_8E57_8E57_8E57);
        let k = 4 * Q4K_BLOCK_VALUES;
        let mut x = vec![0.0f32; k];
        // Block 0 stays all zero; block 1 is constant magnitude (every |x| = amax);
        // block 2 is tiny but normal-scale (inv stays finite); block 3 is random.
        for i in 0..Q4K_BLOCK_VALUES {
            x[Q4K_BLOCK_VALUES + i] = if i % 2 == 0 { 1.0 } else { -1.0 };
            x[2 * Q4K_BLOCK_VALUES + i] = 1e-30 * rng.f32_between(-1.0, 1.0);
            x[3 * Q4K_BLOCK_VALUES + i] = rng.f32_between(-2.0, 3.0);
        }
        let mut a = Q8kRow::zeros(k).unwrap();
        quantize_q8k(&x, &mut a).unwrap();

        // Round trip plus exact bsums, per block.
        let ulp4 = |d: f32| d * 128.0 * f32::EPSILON;
        for b in 0..4 {
            let xs = &x[b * Q4K_BLOCK_VALUES..][..Q4K_BLOCK_VALUES];
            let d = a.d[b];
            if b == 0 {
                assert_eq!(d.to_bits(), 0.0f32.to_bits());
                assert!(a.q[..Q4K_BLOCK_VALUES].iter().all(|&q| q == 0));
                assert!(a.bsums[..16].iter().all(|&s| s == 0));
                continue;
            }
            for (i, (&v, &q)) in xs.iter().zip(&a.q[b * Q4K_BLOCK_VALUES..]).enumerate() {
                let dq = d * f32::from(q);
                assert!(
                    (v - dq).abs() <= d / 2.0 + ulp4(d),
                    "block {b} elem {i}: |{v} - {dq}| > {}",
                    d / 2.0 + ulp4(d)
                );
            }
            let sums = &a.bsums[16 * b..][..16];
            for (j, &s) in sums.iter().enumerate() {
                let want: i32 = a.q[b * Q4K_BLOCK_VALUES + 16 * j..][..16]
                    .iter()
                    .map(|&q| i32::from(q))
                    .sum();
                assert_eq!(i32::from(s), want, "bsums[{j}] of block {b}");
            }
        }
        // Edge pinning: every constant-magnitude element maps to +-127 ...
        for i in 0..Q4K_BLOCK_VALUES {
            let want = if i % 2 == 0 { 127 } else { -127 };
            assert_eq!(a.q[Q4K_BLOCK_VALUES + i], want);
        }
        // ... and the random block's amax element does too.
        let (mut at, mut amax) = (0usize, 0.0f32);
        for (i, &v) in x[3 * Q4K_BLOCK_VALUES..].iter().enumerate() {
            if v.abs() > amax {
                (at, amax) = (i, v.abs());
            }
        }
        let want = if x[3 * Q4K_BLOCK_VALUES + at] < 0.0 {
            -127
        } else {
            127
        };
        assert_eq!(a.q[3 * Q4K_BLOCK_VALUES + at], want);

        // Repeats are bitwise identical even into a dirty row.
        let mut b_row = Q8kRow::zeros(k).unwrap();
        let mut other = vec![9.0f32; k];
        other[0] = 0.0; // forces the zero-block branch to rewrite zeros
        quantize_q8k(&other, &mut b_row).unwrap();
        quantize_q8k(&x, &mut b_row).unwrap();
        assert_eq!(a.d, b_row.d);
        assert_eq!(a.q, b_row.q);
        assert_eq!(a.bsums, b_row.bsums);

        // Bad input is refused with out untouched.
        let mut r = Q8kRow::zeros(k).unwrap();
        r.d[0] = 42.0;
        r.q[0] = 9;
        r.bsums[0] = 9;
        let unchanged = |r: &Q8kRow| {
            r.d[0] == 42.0
                && r.q[0] == 9
                && r.bsums[0] == 9
                && r.d[1..].iter().all(|&v| v == 0.0)
                && r.q[1..].iter().all(|&v| v == 0)
                && r.bsums[1..].iter().all(|&v| v == 0)
        };
        assert!(matches!(Q8kRow::zeros(0), Err(ExpertCpuError::BadShape(_))));
        assert!(matches!(
            Q8kRow::zeros(100),
            Err(ExpertCpuError::BadShape(_))
        ));
        let mut bad_len = x.clone();
        bad_len.pop();
        assert!(matches!(
            quantize_q8k(&bad_len, &mut r),
            Err(ExpertCpuError::BadShape(_))
        ));
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bad = x.clone();
            bad[7] = poison;
            assert!(matches!(
                quantize_q8k(&bad, &mut r),
                Err(ExpertCpuError::BadShape(_))
            ));
        }
        let mut big = Q8kRow::zeros(k + Q4K_BLOCK_VALUES).unwrap();
        assert!(matches!(
            quantize_q8k(&x, &mut big),
            Err(ExpertCpuError::BadLength {
                what: "out.d",
                got: 5,
                want: 4
            })
        ));
        big.d.truncate(4);
        assert!(matches!(
            quantize_q8k(&x, &mut big),
            Err(ExpertCpuError::BadLength {
                what: "out.q",
                got: 1280,
                want: 1024
            })
        ));
        assert!(unchanged(&r), "errors must not touch out");
    }

    /// Finite f16 bit patterns for d/dmin: normals of varied size, negatives,
    /// subnormals (the same spread the f32 kernel's tests use).
    const F16_BITS: [u16; 9] = [
        0x0001, 0x03ff, 0x0400, 0x3500, 0x3800, 0x3c00, 0x4000, 0x7bff, 0xbc00,
    ];

    /// Writes one random Q4_K block: random scale/nibble bytes, finite d/dmin.
    fn random_block(rng: &mut Lcg, out: &mut [u8; Q4K_BLOCK_BYTES], d_bits: &[u16]) {
        *out = [0; Q4K_BLOCK_BYTES];
        for b in out[4..].iter_mut() {
            *b = rng.next() as u8;
        }
        let pick = |rng: &mut Lcg| d_bits[(rng.next() % d_bits.len() as u64) as usize];
        out[0..2].copy_from_slice(&pick(rng).to_le_bytes());
        out[2..4].copy_from_slice(&pick(rng).to_le_bytes());
    }

    /// `rows x k` random Q4_K weights: every block random with finite d/dmin.
    fn random_weights(rng: &mut Lcg, rows: usize, k: usize) -> Vec<u8> {
        let mut w = vec![0u8; rows * k / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES];
        for blk in w.as_chunks_mut::<Q4K_BLOCK_BYTES>().0 {
            let mut block = [0u8; Q4K_BLOCK_BYTES];
            random_block(rng, &mut block, &F16_BITS);
            *blk = block;
        }
        w
    }

    #[test]
    fn scalar_integer_dot_matches_an_independent_reference() {
        // Reference: the dequantized weights (repo Q4K decode) dotted in f64 with the
        // DEQUANTIZED activation d_x * q; |got - ref| <= 1e-5 * sum|terms| + 1e-9.
        use crate::weights::{GgufType, dequantize_into};
        let mut rng = Lcg(0x9A3C_9A3C_0000_0001);
        for (rows, k) in [(1usize, 256usize), (3, 256), (5, 512), (2, 2048), (4, 2048)] {
            let w = random_weights(&mut rng, rows, k);
            let x: Vec<f32> = (0..k).map(|_| rng.f32_between(-1.0, 1.0)).collect();
            let mut xq = Q8kRow::zeros(k).unwrap();
            quantize_q8k(&x, &mut xq).unwrap();
            let mut got = vec![f32::NAN; rows];
            q4k_q8k_matvec(DotBackend::scalar(), &w, rows, k, &xq, &mut got).unwrap();
            let row_bytes = k / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES;
            for (r, out) in got.iter().enumerate() {
                let mut w32 = vec![0.0f32; k];
                dequantize_into(GgufType::Q4K, &w[r * row_bytes..][..row_bytes], &mut w32).unwrap();
                let (mut reference, mut abs) = (0.0f64, 0.0f64);
                for (i, &wi) in w32.iter().enumerate() {
                    let dq = f64::from(xq.d[i / Q4K_BLOCK_VALUES] * f32::from(xq.q[i]));
                    reference += f64::from(wi) * dq;
                    abs += (f64::from(wi) * dq).abs();
                }
                let bound = 1e-5 * abs + 1e-9;
                assert!(
                    (f64::from(*out) - reference).abs() <= bound,
                    "row {r} of {rows}x{k}: got {out}, ref {reference}, bound {bound}"
                );
            }
        }
    }

    #[test]
    fn q8k_error_stays_inside_the_derived_bound_against_the_f32_reference() {
        // Derivation: quantization moves each activation by at most half a step,
        // |d*q - x| <= d/2, so |dot(w, dq) - dot(w, x)| <=
        // sum_b (d_b / 2) * sum_{i in b} |w_i|; the ~1e-7 relative roundings inside d,
        // inv, the products and the sequential f32 tail stay far below the extra
        // 1e-5 * sum|w_i x_i| term.
        use crate::weights::{GgufType, dequantize_into};
        let mut rng = Lcg(0x7BD1_1C0F_0000_0003);
        for (k, rows) in [(2048usize, 64usize), (512, 64)] {
            let w = random_weights(&mut rng, rows, k);
            let x: Vec<f32> = (0..k).map(|_| rng.f32_between(-1.0, 1.0)).collect();
            let mut xq = Q8kRow::zeros(k).unwrap();
            quantize_q8k(&x, &mut xq).unwrap();
            let mut got = vec![f32::NAN; rows];
            q4k_q8k_matvec(DotBackend::scalar(), &w, rows, k, &xq, &mut got).unwrap();
            let row_bytes = k / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES;
            for (r, out) in got.iter().enumerate() {
                let mut w32 = vec![0.0f32; k];
                dequantize_into(GgufType::Q4K, &w[r * row_bytes..][..row_bytes], &mut w32).unwrap();
                let (mut reference, mut abs, mut bound) = (0.0f64, 0.0f64, 0.0f64);
                for (i, &wi) in w32.iter().enumerate() {
                    reference += f64::from(wi) * f64::from(x[i]);
                    abs += (f64::from(wi) * f64::from(x[i])).abs();
                }
                for (b, blk) in w32.as_chunks::<Q4K_BLOCK_VALUES>().0.iter().enumerate() {
                    let block_abs: f64 = blk.iter().map(|&wi| f64::from(wi).abs()).sum();
                    bound += f64::from(xq.d[b] / 2.0) * block_abs;
                }
                bound += 1e-5 * abs;
                assert!(
                    (f64::from(*out) - reference).abs() <= bound,
                    "row {r} of {rows}x{k}: |got - ref| = {} > bound {}",
                    (f64::from(*out) - reference).abs(),
                    bound
                );
            }
        }
    }
    /// Small f16 patterns for the full-expert test: d/dmin at 2^-12..2^-10 keep the
    /// outputs order-1e6 and the Q8_K error comfortably inside the asserted bounds.
    const F16_BITS_SMALL: [u16; 3] = [0x1000, 0x1400, 0x1800];

    #[test]
    fn expert_forward_q8k_is_close_to_the_f32_expert() {
        use crate::kernels::expert_cpu::{
            ExpertScratch, ExpertShape, ExpertSlices, expert_forward,
        };
        // Data note: x is drawn positive (0.25..1.25, order-1 magnitudes like a gated
        // activation), so every projection is mean-dominated; with zero-mean x the
        // Q8_K step error is a fixed ~amax/(440 sigma) of each dot's own noise, which
        // no 3x-headroom bound could hold under the 3e-2 mean-relative cap.
        //
        // Measured on this data (dev profile, this machine): max abs error 2.447e4
        // against outputs of magnitude up to ~2.4e6; mean relative error 2.567e-4,
        // where "relative" divides |got - ref| by max(|ref|, 5% of the largest |ref|)
        // so near-zero outputs stay bounded. Asserted bounds carry 3x headroom; the
        // mean relative bound also sits far under the task's 3e-2 rule.
        let shape = ExpertShape {
            hidden: 2048,
            inter: 512,
        };
        let mut rng = Lcg(0x1A2B_3C4D_5E6F_7081);
        let make = |rng: &mut Lcg, blocks: usize| -> Vec<u8> {
            (0..blocks)
                .flat_map(|_| {
                    let mut block = [0u8; Q4K_BLOCK_BYTES];
                    random_block(rng, &mut block, &F16_BITS_SMALL);
                    block
                })
                .collect()
        };
        let side_blocks = shape.inter * shape.hidden / Q4K_BLOCK_VALUES;
        let gate = make(&mut rng, side_blocks);
        let up = make(&mut rng, side_blocks);
        let down = make(&mut rng, side_blocks);
        let x: Vec<f32> = (0..shape.hidden)
            .map(|_| rng.f32_between(0.25, 1.25))
            .collect();
        let slices = ExpertSlices {
            gate: &gate,
            up: &up,
            down: &down,
        };
        let mut ref_out = vec![f32::NAN; shape.hidden];
        let mut ref_scratch = ExpertScratch::new(shape);
        expert_forward(shape, &slices, &x, &mut ref_scratch, &mut ref_out).unwrap();
        let mut got = vec![f32::NAN; shape.hidden];
        let mut scratch = ExpertScratchQ8k::new(shape);
        expert_forward_q8k(
            DotBackend::scalar(),
            shape,
            &slices,
            &x,
            &mut scratch,
            &mut got,
        )
        .unwrap();
        let max_abs = got
            .iter()
            .zip(&ref_out)
            .map(|(g, r)| f64::from((g - r).abs()))
            .fold(0.0f64, f64::max);
        let scale = f64::from(ref_out.iter().fold(0.0f32, |m, v| m.max(v.abs())));
        let mean_rel = got
            .iter()
            .zip(&ref_out)
            .map(|(g, r)| f64::from((g - r).abs()) / f64::from(r.abs()).max(0.05 * scale))
            .sum::<f64>()
            / got.len() as f64;
        assert!(
            max_abs <= 8e4,
            "max abs error {max_abs} exceeds the 3x-headroom bound 8e4"
        );
        assert!(
            mean_rel <= 8e-4,
            "mean relative error {mean_rel} exceeds the 3x-headroom bound 8e-4"
        );
        assert!(
            mean_rel < 3e-2,
            "mean relative error {mean_rel} exceeds the 3e-2 rule"
        );
    }

    #[test]
    fn expert_forward_q8k_validates_like_the_f32_path() {
        use crate::kernels::expert_cpu::{ExpertShape, ExpertSlices};
        let shape = ExpertShape {
            hidden: 512,
            inter: 256,
        };
        let side = 256 * 512 / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES;
        let down = 512 * 256 / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES;
        let zero = |n: usize| vec![0u8; n];
        let (gate, up, down_w, long_up) = (zero(side), zero(side), zero(down), zero(side + 1));
        let x = vec![0.5f32; shape.hidden];
        let mut out = vec![f32::NAN; shape.hidden];
        let good = ExpertSlices {
            gate: &gate,
            up: &up,
            down: &down_w,
        };
        let all_nan = |v: &[f32]| v.iter().all(|v| v.is_nan());

        // Shape violations first, on otherwise valid arguments.
        for bad in [
            ExpertShape {
                hidden: 100,
                inter: 512,
            },
            ExpertShape {
                hidden: 0,
                inter: 512,
            },
            ExpertShape {
                hidden: 512,
                inter: 0,
            },
        ] {
            let mut scratch = ExpertScratchQ8k::new(bad);
            assert!(matches!(
                expert_forward_q8k(DotBackend::scalar(), bad, &good, &x, &mut scratch, &mut out),
                Err(ExpertCpuError::BadShape(_))
            ));
        }
        assert!(all_nan(&out), "a BadShape call must not touch out");

        // One mutated argument per case; everything else stays valid.
        let bad = |what: &'static str, got: usize, want: usize| ExpertCpuError::BadLength {
            what,
            got,
            want,
        };
        let cases: [(&str, usize, usize, u8); 10] = [
            ("x", 511, 512, 0),
            ("out", 513, 512, 1),
            ("gate", side - 1, side, 2),
            ("up", side + 1, side, 3),
            ("down", down - Q4K_BLOCK_BYTES, down, 4),
            ("scratch.gate", 255, 256, 5),
            ("scratch.act", 255, 256, 6),
            ("scratch.qx.q", 511, 512, 7),
            ("scratch.qh.d", 0, 1, 8),
            ("scratch.qh.bsums", 15, 16, 9),
        ];
        for (what, got, want, case) in cases {
            let slices = match case {
                2 => ExpertSlices {
                    gate: &gate[..side - 1],
                    up: &up,
                    down: &down_w,
                },
                3 => ExpertSlices {
                    gate: &gate,
                    up: &long_up,
                    down: &down_w,
                },
                4 => ExpertSlices {
                    gate: &gate,
                    up: &up,
                    down: &down_w[..down - Q4K_BLOCK_BYTES],
                },
                _ => ExpertSlices {
                    gate: &gate,
                    up: &up,
                    down: &down_w,
                },
            };
            let xx = if case == 0 {
                x[..511].to_vec()
            } else {
                x.clone()
            };
            let mut sc = ExpertScratchQ8k::new(shape);
            match case {
                5 => sc.gate.truncate(255),
                6 => sc.act.truncate(255),
                7 => sc.qx.q.truncate(511),
                8 => sc.qh.d.clear(),
                9 => sc.qh.bsums.truncate(15),
                _ => {}
            }
            let mut oo = if case == 1 {
                vec![f32::NAN; 513]
            } else {
                vec![f32::NAN; 512]
            };
            assert_eq!(
                expert_forward_q8k(DotBackend::scalar(), shape, &slices, &xx, &mut sc, &mut oo),
                Err(bad(what, got, want))
            );
            let checked = if case == 1 { 513 } else { 512 };
            assert!(all_nan(&oo[..checked]), "{what} case touched out");
        }

        // Non-finite x is refused by the quantizer contract, out untouched.
        let mut poison = x.clone();
        poison[3] = f32::NAN;
        let mut scratch = ExpertScratchQ8k::new(shape);
        assert!(matches!(
            expert_forward_q8k(
                DotBackend::scalar(),
                shape,
                &good,
                &poison,
                &mut scratch,
                &mut out
            ),
            Err(ExpertCpuError::BadShape(_))
        ));
        assert!(all_nan(&out));

        // A following valid call is still correct: the scratch is reusable and
        // repeated runs are bitwise identical.
        expert_forward_q8k(
            DotBackend::scalar(),
            shape,
            &good,
            &x,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        let mut again = vec![f32::NAN; shape.hidden];
        expert_forward_q8k(
            DotBackend::scalar(),
            shape,
            &good,
            &x,
            &mut scratch,
            &mut again,
        )
        .unwrap();
        assert_eq!(out, again, "reused scratch changed the result");
        assert!(out.iter().all(|v| !v.is_nan()), "valid call must fill out");
    }

    #[test]
    fn avx2_equals_scalar_bitwise() {
        let backend = DotBackend::detect();
        if backend.name() != "avx2" {
            println!("skipping avx2_equals_scalar_bitwise: no AVX2 on this host");
            return;
        }
        let mut rng = Lcg(0xC0DE_C0DE_0000_0004);
        let run = |w: &[u8], rows: usize, k: usize, x: &[f32]| -> (Vec<f32>, Vec<f32>) {
            let mut xq = Q8kRow::zeros(k).unwrap();
            quantize_q8k(x, &mut xq).unwrap();
            let (mut a, mut b) = (vec![f32::NAN; rows], vec![f32::NAN; rows]);
            q4k_q8k_matvec(DotBackend::scalar(), w, rows, k, &xq, &mut a).unwrap();
            q4k_q8k_matvec(backend, w, rows, k, &xq, &mut b).unwrap();
            (a, b)
        };
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        // Random data across k in {256, 512, 2048} and 3..=64 rows.
        for (rows, k) in [
            (3usize, 256usize),
            (17, 256),
            (64, 256),
            (3, 512),
            (33, 512),
            (64, 512),
            (3, 2048),
            (64, 2048),
        ] {
            let w = random_weights(&mut rng, rows, k);
            let x: Vec<f32> = (0..k).map(|_| rng.f32_between(-1.0, 1.0)).collect();
            let (a, b) = run(&w, rows, k, &x);
            assert_eq!(bits(&a), bits(&b), "{rows}x{k} random data diverged");
        }
        // Extreme 1: every nibble 15 with every scale and minimum 63, moderate d/dmin.
        let mut w = random_weights(&mut rng, 8, 512);
        for blk in w.as_chunks_mut::<Q4K_BLOCK_BYTES>().0 {
            blk[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
            blk[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
            blk[4..16].fill(0xff);
            blk[16..].fill(0xff);
        }
        let x: Vec<f32> = (0..512).map(|_| rng.f32_between(-1.0, 1.0)).collect();
        let (a, b) = run(&w, 8, 512, &x);
        assert_eq!(bits(&a), bits(&b), "all-15 nibbles with scales 63 diverged");
        // Extreme 2: every q8 at +-127 (constant-magnitude activation).
        let mut x = vec![0.0f32; 512];
        for (i, v) in x.iter_mut().enumerate() {
            *v = if i % 2 == 0 { 1.25 } else { -1.25 };
        }
        let (a, b) = run(&w, 8, 512, &x);
        assert_eq!(bits(&a), bits(&b), "every q8 at +-127 diverged");
        // Extreme 3: alternating signs with random magnitudes.
        let mut x = vec![0.0f32; 2048];
        for (i, v) in x.iter_mut().enumerate() {
            *v = if i % 2 == 0 {
                rng.f32_between(0.1, 1.0)
            } else {
                -rng.f32_between(0.1, 1.0)
            };
        }
        let w = random_weights(&mut rng, 5, 2048);
        let (a, b) = run(&w, 5, 2048, &x);
        assert_eq!(bits(&a), bits(&b), "alternating signs diverged");
    }

    #[test]
    fn scalar_and_avx2_expert_forward_are_bitwise_equal() {
        let backend = DotBackend::detect();
        if backend.name() != "avx2" {
            println!("skipping scalar_and_avx2_expert_forward_are_bitwise_equal: no AVX2");
            return;
        }
        use crate::kernels::expert_cpu::ExpertShape;
        let shape = ExpertShape {
            hidden: 2048,
            inter: 512,
        };
        let mut rng = Lcg(0xFEED_FACE_0000_0005);
        let make = |rng: &mut Lcg, blocks: usize| -> Vec<u8> {
            (0..blocks)
                .flat_map(|_| {
                    let mut block = [0u8; Q4K_BLOCK_BYTES];
                    random_block(rng, &mut block, &F16_BITS);
                    block
                })
                .collect()
        };
        let side_blocks = shape.inter * shape.hidden / Q4K_BLOCK_VALUES;
        let gate = make(&mut rng, side_blocks);
        let up = make(&mut rng, side_blocks);
        let down = make(&mut rng, side_blocks);
        let x: Vec<f32> = (0..shape.hidden)
            .map(|_| rng.f32_between(-1.0, 1.0))
            .collect();
        let slices = crate::kernels::expert_cpu::ExpertSlices {
            gate: &gate,
            up: &up,
            down: &down,
        };
        let (mut a, mut b) = (vec![f32::NAN; shape.hidden], vec![f32::NAN; shape.hidden]);
        let (mut sa, mut sb) = (ExpertScratchQ8k::new(shape), ExpertScratchQ8k::new(shape));
        expert_forward_q8k(DotBackend::scalar(), shape, &slices, &x, &mut sa, &mut a).unwrap();
        expert_forward_q8k(backend, shape, &slices, &x, &mut sb, &mut b).unwrap();
        assert_eq!(
            a.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            "scalar and AVX2 expert forwards diverged"
        );
    }
    #[test]
    fn unsafe_is_confined_to_the_avx2_module() {
        // The needle is assembled at run time so this test's own text can never match.
        let needle = ["uns", "afe {"].concat();
        let files = [
            (
                "expert_cpu_avx2.rs",
                include_str!("expert_cpu_avx2.rs"),
                5usize,
            ),
            ("expert_cpu_q8g.rs", include_str!("expert_cpu_q8g.rs"), 0),
            ("expert_cpu_q8k.rs", include_str!("expert_cpu_q8k.rs"), 0),
            ("expert_cpu.rs", include_str!("expert_cpu.rs"), 0),
            (
                "tiered/mod.rs",
                include_str!("../family/qwen36/tiered/mod.rs"),
                0,
            ),
            (
                "tiered/locator.rs",
                include_str!("../family/qwen36/tiered/locator.rs"),
                0,
            ),
            (
                "tiered/policy.rs",
                include_str!("../family/qwen36/tiered/policy.rs"),
                0,
            ),
            (
                "tiered/cpu_pool.rs",
                include_str!("../family/qwen36/tiered/cpu_pool.rs"),
                0,
            ),
            (
                "tiered/budget.rs",
                include_str!("../family/qwen36/tiered/budget.rs"),
                0,
            ),
            (
                "tiered/report.rs",
                include_str!("../family/qwen36/tiered/report.rs"),
                0,
            ),
            (
                "tiered/guess.rs",
                include_str!("../family/qwen36/tiered/guess.rs"),
                0,
            ),
            (
                "tiered/residency.rs",
                include_str!("../family/qwen36/tiered/residency.rs"),
                0,
            ),
        ];
        for (name, text, max) in files {
            let lines: Vec<&str> = text.lines().collect();
            let mut blocks = 0usize;
            for (i, line) in lines.iter().enumerate() {
                if !line.contains(&needle) {
                    continue;
                }
                blocks += 1;
                assert!(
                    blocks <= max,
                    "{name} carries more than {max} raw block(s) of this kind"
                );
                let prev = lines[..i].iter().rev().find(|l| !l.trim().is_empty());
                assert!(
                    prev.is_some_and(|l| l.contains("SAFETY")),
                    "{name} line {}: block lacks a preceding SAFETY comment",
                    i + 1
                );
            }
            if max > 0 {
                assert!(blocks >= 1, "{name} must carry its scoped blocks");
            }
        }
    }
}
