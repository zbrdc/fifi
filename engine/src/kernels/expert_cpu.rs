//! Q4_K expert forward on the CPU, safe Rust (the reference and the f32 pool kernel of the OFF-2 tier).
//!
//! One routed expert of Qwen3.6-35B-A3B is three Q4_K matrices (gate/up `inter x hidden`,
//! down `hidden x inter`) as raw 144-byte ggml blocks; this module computes
//! `out = down(silu(gate(x)) * up(x))` straight from those byte slices, for experts not
//! resident in the VRAM bank. Throughput is DRAM-bound, so the inner loops are written for
//! autovectorization: fixed-size array chunks, no `unsafe`, no intrinsics, no
//! data-dependent branches.
//!
//! Numerics (bit-deterministic, independent of the compiler's vector width):
//!
//! * Weights decode per 256-value block exactly as ggml does: `d` and `dmin` are IEEE
//!   binary16, the 8 sub-blocks of 32 values carry a 6-bit scale `sc` and minimum `m`, and
//!   `w = (d * sc) * q - (dmin * m)` for nibble `q` (the low nibble of a byte first).
//! * A sub-block dot uses the factored form `(d * sc) * sum(q_i * x_i) - (dmin * m) *
//!   sum(x_i)`: each sum fills 8 independent f32 lane accumulators in index order and
//!   reduces as `((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7))`, so the result never depends on
//!   how the compiler vectorizes the loops. The `sum(x_i)` tables are computed once per
//!   matvec and shared by every row (identical values for every row, so no change).
//! * Row results accumulate into 8 lanes keyed by the sub-block index (lane j receives
//!   every block's sub-block j) and reduce in that same fixed order.
//! * All arithmetic is f32; `silu(x) = x / (1 + exp(-x))`.

/// One ggml Q4_K block in bytes: 2 f16 scales, 12 scale bytes, 128 nibble bytes.
pub const Q4K_BLOCK_BYTES: usize = 144;
/// f32 weights in one Q4_K block (4 chunks x 2 sub-blocks x 32 values).
pub const Q4K_BLOCK_VALUES: usize = 256;

/// Validation failure, reported before any output byte is written.
#[derive(Debug, PartialEq, Eq)]
pub enum ExpertCpuError {
    /// Structurally impossible arguments (zero rows, `k` not a positive multiple of 256,
    /// an expert shape whose extents are not multiples of 256).
    BadShape(String),
    /// A byte or value slice has the wrong length for the requested geometry.
    BadLength {
        /// Which slice was rejected ("w", "x", "out", "gate", "up", "down", scratch).
        what: &'static str,
        /// The length that was passed.
        got: usize,
        /// The length the geometry requires.
        want: usize,
    },
}

/// Converts IEEE binary16 bits to `f32`, keeping subnormals; a NaN becomes quiet. Q4_K
/// block scales are binary16 and this module uses no crates, so it decodes them itself.
pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1f);
    let mant = u32::from(bits & 0x03ff);
    let out = match (exp, mant) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal binary16: normalize the mantissa into a normal f32 exponent.
            let shift = mant.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | (((mant << shift) & 0x03ff) << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, _) => sign | 0x7fc0_0000 | (mant << 13),
        _ => sign | ((exp + 112) << 23) | (mant << 13),
    };
    f32::from_bits(out)
}

/// Unpacks the 6-bit scale and minimum of sub-block `j` (0..8) of a Q4_K block.
///
/// The 12 packed scale bytes live at block offset 4..16 in the ggml layout.
pub(crate) fn scale_min(j: usize, blk: &[u8; Q4K_BLOCK_BYTES]) -> (u8, u8) {
    let q = |i: usize| blk[4 + i];
    if j < 4 {
        (q(j) & 63, q(j + 4) & 63)
    } else {
        (
            (q(j + 4) & 0x0f) | ((q(j - 4) >> 6) << 4),
            (q(j + 4) >> 4) | ((q(j) >> 6) << 4),
        )
    }
}

/// Sums the 8 lane accumulators as `((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7))`; that fixed
/// order is the module's determinism contract, so every reduction uses it.
fn reduce8(l: [f32; 8]) -> f32 {
    (l[0] + l[1]) + (l[2] + l[3]) + (l[4] + l[5]) + (l[6] + l[7])
}

/// Fills `sums` with the sub-block sums of `x`: entry `8 * b + j` sums the 32 activations
/// of sub-block `j` of 256-value block `b` in the fixed 8-lane order. Every row dots the
/// same sub-blocks against the per-sub-block minima, so these are computed once per call.
fn x_sub_sums(x: &[f32], sums: &mut [f32]) {
    for (t, xs) in sums
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<Q4K_BLOCK_VALUES>().0)
    {
        for (j, ec) in t.iter_mut().zip(xs.as_chunks::<32>().0) {
            let mut lanes = [0.0f32; 8];
            for g in ec.as_chunks::<8>().0 {
                for l in 0..8 {
                    lanes[l] += g[l];
                }
            }
            *j = reduce8(lanes);
        }
    }
}

/// Dots one row of Q4_K blocks (an exact-length slice) with `x`: `row` is
/// `blocks * Q4K_BLOCK_BYTES` bytes, `x` the matching `blocks * 256` activations and
/// `sums` their precomputed sub-block sums, all re-sliced to fixed-size arrays so the
/// inner loops carry no bounds checks.
fn q4k_row_dot(row: &[u8], x: &[f32], sums: &[f32]) -> f32 {
    let (blocks, _) = row.as_chunks::<Q4K_BLOCK_BYTES>();
    let mut acc = [0.0f32; 8];
    for ((blk, xs), tx) in blocks
        .iter()
        .zip(x.as_chunks::<Q4K_BLOCK_VALUES>().0)
        .zip(sums.as_chunks::<8>().0)
    {
        let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_bits_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let qs: &[[u8; 32]] = blk[16..].as_chunks::<32>().0;
        let (xc, _) = xs.as_chunks::<32>();
        for (c, q) in qs.iter().enumerate() {
            let (sc1, m1) = scale_min(2 * c, blk);
            let (sc2, m2) = scale_min(2 * c + 1, blk);
            let (d1, min1) = (d * f32::from(sc1), dmin * f32::from(m1));
            let (d2, min2) = (d * f32::from(sc2), dmin * f32::from(m2));
            let (e1, e2) = (&xc[2 * c], &xc[2 * c + 1]);
            // One fused loop: both nibble streams of each byte pair dot in one pass.
            let (mut s1, mut s2) = ([0.0f32; 8], [0.0f32; 8]);
            for ((qc, g1), g2) in q
                .as_chunks::<8>()
                .0
                .iter()
                .zip(e1.as_chunks::<8>().0)
                .zip(e2.as_chunks::<8>().0)
            {
                for l in 0..8 {
                    s1[l] += f32::from(qc[l] & 0x0f) * g1[l];
                    s2[l] += f32::from(qc[l] >> 4) * g2[l];
                }
            }
            acc[2 * c] += d1 * reduce8(s1) - min1 * tx[2 * c];
            acc[2 * c + 1] += d2 * reduce8(s2) - min2 * tx[2 * c + 1];
        }
    }
    reduce8(acc)
}

/// The row loop shared by `q4k_matvec` and `expert_forward` once every length check
/// has passed: fills the shared sub-block sums (`sums` holds `k / 32` entries) and dots
/// every row in order.
fn matvec_rows(w: &[u8], k: usize, x: &[f32], sums: &mut [f32], out: &mut [f32]) {
    x_sub_sums(x, sums);
    let row_bytes = (k / Q4K_BLOCK_VALUES) * Q4K_BLOCK_BYTES;
    for (o, row) in out.iter_mut().zip(w.chunks_exact(row_bytes)) {
        *o = q4k_row_dot(row, x, sums);
    }
}

/// Dots `rows` rows of `k` Q4_K weights with the f32 vector `x`, writing `rows` results
/// to `out`.
///
/// `w` is row-major ggml blocks, so `k` must be a positive multiple of
/// [`Q4K_BLOCK_VALUES`] and `w` exactly `rows * k / 256 * 144` bytes; `x` must be `k` long
/// and `out` exactly `rows` long. Every length is checked before any output byte is
/// touched, so on error `out` is unchanged. A row range needs no separate API: pass `rows`
/// and the matching sub-slice of `w`. The call allocates one `k / 32` f32 table for the
/// shared sub-block sums; `expert_forward` reuses its scratch instead.
pub fn q4k_matvec(
    w: &[u8],
    rows: usize,
    k: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), ExpertCpuError> {
    if k == 0 || !k.is_multiple_of(Q4K_BLOCK_VALUES) {
        return Err(ExpertCpuError::BadShape(format!(
            "k = {k} is not a positive multiple of {Q4K_BLOCK_VALUES}"
        )));
    }
    if rows == 0 {
        return Err(ExpertCpuError::BadShape(format!(
            "rows = 0 in a {rows}x{k} Q4_K matvec"
        )));
    }
    let row_bytes = (k / Q4K_BLOCK_VALUES) * Q4K_BLOCK_BYTES;
    let want = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| ExpertCpuError::BadShape(format!("{rows} x {row_bytes} bytes overflows")))?;
    for (what, got, want_len) in [
        ("x", x.len(), k),
        ("out", out.len(), rows),
        ("w", w.len(), want),
    ] {
        if got != want_len {
            return Err(ExpertCpuError::BadLength {
                what,
                got,
                want: want_len,
            });
        }
    }
    let mut sums = vec![0.0; k / Q4K_BLOCK_VALUES * 8];
    matvec_rows(w, k, x, &mut sums, out);
    Ok(())
}

/// Geometry of one routed expert: rows/columns of its three matrices.
///
/// `hidden` is the model hidden size (input of gate/up, rows of down) and `inter` the
/// per-expert intermediate size (rows of gate/up, input of down). Qwen3.6-35B-A3B uses
/// 2048/512; both must be positive multiples of 256 so every row is whole blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertShape {
    /// Model hidden size in values; also the expert output length.
    pub hidden: usize,
    /// Per-expert intermediate size in values; also the gate/up row count.
    pub inter: usize,
}

/// The three raw Q4_K slices of one expert, row-major, as stored in the GGUF.
///
/// Each slice is `rows * columns / 256 * 144` bytes; for the Qwen3.6 shape every slice is
/// exactly 589,824 bytes.
pub struct ExpertSlices<'a> {
    /// Gate projection, `inter` rows over `hidden` inputs.
    pub gate: &'a [u8],
    /// Up projection, `inter` rows over `hidden` inputs.
    pub up: &'a [u8],
    /// Down projection, `hidden` rows over `inter` inputs.
    pub down: &'a [u8],
}

/// Reusable scratch for [`expert_forward`] so a call allocates nothing.
///
/// `gate` and `up` hold the two projections, `act` holds `silu(gate) * up` (all `inter`
/// long) and `sums` the shared activation sub-block sums; see [`x_sub_sums`].
pub struct ExpertScratch {
    gate: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    sums: Vec<f32>,
}

impl ExpertScratch {
    /// Allocates scratch sized for one expert of `shape`: value buffers `inter` long,
    /// `sums` sized for the longer of `hidden` and `inter`.
    ///
    /// The buffers are zeroed so a reused scratch never leaks values of one call into
    /// the next.
    pub fn new(shape: ExpertShape) -> Self {
        Self {
            gate: vec![0.0; shape.inter],
            up: vec![0.0; shape.inter],
            act: vec![0.0; shape.inter],
            sums: vec![0.0; shape.hidden.max(shape.inter) / Q4K_BLOCK_VALUES * 8],
        }
    }
}

/// Runs one expert: `out = down(silu(gate(x)) * up(x))`.
///
/// Every length implied by `shape` (slices, `x`, `out`, scratch) is checked before any
/// computation, so on error `out` is left untouched. The scratch must come from
/// [`ExpertScratch::new`] with the same shape; with it, a call performs no allocation.
pub fn expert_forward(
    shape: ExpertShape,
    s: &ExpertSlices<'_>,
    x: &[f32],
    scratch: &mut ExpertScratch,
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
    // A size that overflows usize can never match a real slice, so clamp it.
    let slice_len = |rows: usize, cols: usize| {
        rows.saturating_mul(cols / Q4K_BLOCK_VALUES)
            .saturating_mul(Q4K_BLOCK_BYTES)
    };
    let (side, down) = (slice_len(inter, hidden), slice_len(hidden, inter));
    let sums_len = hidden.max(inter) / Q4K_BLOCK_VALUES * 8;
    for (what, got, want) in [
        ("x", x.len(), hidden),
        ("out", out.len(), hidden),
        ("gate", s.gate.len(), side),
        ("up", s.up.len(), side),
        ("down", s.down.len(), down),
        ("scratch.gate", scratch.gate.len(), inter),
        ("scratch.up", scratch.up.len(), inter),
        ("scratch.act", scratch.act.len(), inter),
        ("scratch.sums", scratch.sums.len(), sums_len),
    ] {
        if got != want {
            return Err(ExpertCpuError::BadLength { what, got, want });
        }
    }
    let ExpertScratch {
        gate,
        up,
        act,
        sums,
    } = scratch;
    // gate and up share x; recomputing their 64-entry sums is noise against 8192 dots.
    let side_sums = hidden / Q4K_BLOCK_VALUES * 8;
    matvec_rows(s.gate, hidden, x, &mut sums[..side_sums], gate);
    matvec_rows(s.up, hidden, x, &mut sums[..side_sums], up);
    for i in 0..inter {
        let g = gate[i];
        act[i] = g / (1.0 + (-g).exp()) * up[i];
    }
    let inter_sums = inter / Q4K_BLOCK_VALUES * 8;
    matvec_rows(s.down, inter, act, &mut sums[..inter_sums], out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ExpertCpuError, ExpertScratch, ExpertShape, ExpertSlices, Q4K_BLOCK_BYTES,
        Q4K_BLOCK_VALUES, expert_forward, q4k_matvec,
    };
    use crate::weights::{GgufType, dequantize_into};

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
            lo + (hi - lo) * (self.next() as f32 / (1u64 << 53) as f32)
        }
        fn pick<'a, T>(&mut self, set: &'a [T]) -> &'a T {
            &set[(self.next() % set.len() as u64) as usize]
        }
    }

    /// Finite f16 bit patterns for d/dmin: normals of varied size, negatives, subnormals.
    const F16_BITS: [u16; 9] = [
        0x0001, 0x03ff, 0x0400, 0x3500, 0x3800, 0x3c00, 0x4000, 0x7bff, 0xbc00,
    ];

    /// Writes one random Q4_K block: random scale/nibble bytes, finite d/dmin.
    fn random_block(rng: &mut Lcg, out: &mut [u8; 144], d_bits: &[u16]) {
        *out = [0; 144];
        for b in out[4..].iter_mut() {
            *b = rng.next() as u8;
        }
        out[0..2].copy_from_slice(&rng.pick(d_bits).to_le_bytes());
        out[2..4].copy_from_slice(&rng.pick(d_bits).to_le_bytes());
    }

    /// `rows x k` Q4_K weights: block 0 all-zero, then the scales-63/all-15 and the
    /// subnormal-d/dmin corner blocks, then random blocks.
    fn random_weights(rng: &mut Lcg, rows: usize, k: usize) -> Vec<u8> {
        let mut w = vec![0u8; rows * k / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES];
        for (at, blk) in w
            .as_chunks_mut::<Q4K_BLOCK_BYTES>()
            .0
            .iter_mut()
            .enumerate()
        {
            let mut block = [0u8; 144];
            match at {
                0 => {} // block 0 stays all-zero
                1 => {
                    blk[4..16].fill(0xff); // every scale and minimum reads 63
                    blk[16..].fill(0xff); // every nibble reads 15
                    blk[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                    blk[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
                    continue;
                }
                2 => {
                    random_block(rng, &mut block, &F16_BITS);
                    block[0..2].copy_from_slice(&0x0001u16.to_le_bytes()); // subnormal d
                    block[2..4].copy_from_slice(&0x0002u16.to_le_bytes()); // subnormal dmin
                }
                _ => random_block(rng, &mut block, &F16_BITS),
            }
            blk.copy_from_slice(&block);
        }
        w
    }

    /// Reference (dot, sum|w*x|) for one row: the dequant reference, f64 math.
    fn reference_row_dot(row: &[u8], x: &[f32]) -> (f64, f64) {
        let mut w = vec![0.0f32; x.len()];
        dequantize_into(GgufType::Q4K, row, &mut w).unwrap();
        let terms = || {
            w.iter()
                .zip(x)
                .map(|(wi, xi)| f64::from(*wi) * f64::from(*xi))
        };
        (terms().sum::<f64>(), terms().map(f64::abs).sum::<f64>())
    }

    /// `ExpertShape` literal helper: keeps the error-path cases one line each.
    fn shape(hidden: usize, inter: usize) -> ExpertShape {
        ExpertShape { hidden, inter }
    }

    /// `ExpertSlices` literal helper, for the same reason as `shape`.
    fn sl<'a>(gate: &'a [u8], up: &'a [u8], down: &'a [u8]) -> ExpertSlices<'a> {
        ExpertSlices { gate, up, down }
    }

    #[test]
    fn expert_cpu_q4k_matches_the_dequant_reference() {
        // Per-row bound: |got - ref| <= 1e-4 * sum(|w_i * x_i|) + 1e-6, where ref is
        // the dequant reference decode of the row dotted with x in f64.
        let mut rng = Lcg(0xC5C5_C5C5_C5C5_C5C5);
        for (rows, k) in [(1usize, 256usize), (3, 256), (5, 512), (2, 2048), (4, 2048)] {
            let w = random_weights(&mut rng, rows, k);
            let x: Vec<f32> = (0..k).map(|_| rng.f32_between(-1.0, 1.0)).collect();
            let mut got = vec![f32::NAN; rows];
            q4k_matvec(&w, rows, k, &x, &mut got).unwrap();
            let row_bytes = k / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES;
            for (r, out) in got.iter().enumerate() {
                let (reference, abs) = reference_row_dot(&w[r * row_bytes..][..row_bytes], &x);
                let bound = 1e-4 * abs + 1e-6;
                assert!(
                    (f64::from(*out) - reference).abs() <= bound,
                    "row {r} of {rows}x{k}: got {out}, ref {reference}, bound {bound}"
                );
            }
        }
    }

    #[test]
    fn expert_cpu_is_bit_deterministic_and_independent_of_chunking() {
        let mut rng = Lcg(0x5C5C_5C5C_5C5C_5C5C);
        let (rows, k) = (6usize, 1024usize);
        let w = random_weights(&mut rng, rows, k);
        let x: Vec<f32> = (0..k).map(|_| rng.f32_between(-2.0, 2.0)).collect();
        let mut a = vec![0.0f32; rows];
        let mut b = vec![1.0f32; rows];
        q4k_matvec(&w, rows, k, &x, &mut a).unwrap();
        q4k_matvec(&w, rows, k, &x, &mut b).unwrap();
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "same inputs must give bit-identical outputs"
        );
        // Two calls over half the rows each must equal the single call bitwise.
        let half = rows / 2;
        let row_bytes = k / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES;
        let mut split = vec![0.0f32; rows];
        q4k_matvec(&w[..half * row_bytes], half, k, &x, &mut split[..half]).unwrap();
        q4k_matvec(
            &w[half * row_bytes..],
            rows - half,
            k,
            &x,
            &mut split[half..],
        )
        .unwrap();
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            split.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn expert_cpu_refuses_bad_lengths_before_any_work() {
        let mut rng = Lcg(0xFEED_F15C_0000_0001);
        let (rows, k) = (4usize, 512usize);
        let good = random_weights(&mut rng, rows, k);
        let x: Vec<f32> = (0..k).map(|_| rng.f32_between(-1.0, 1.0)).collect();
        let all_nan = |v: &[f32]| v.iter().all(|v| v.is_nan());

        // q4k_matvec: shape violations.
        let mut out = vec![f32::NAN; rows];
        for bad_k in [0usize, 100, 384] {
            assert!(matches!(
                q4k_matvec(&good, rows, bad_k, &x, &mut out),
                Err(ExpertCpuError::BadShape(_))
            ));
        }
        assert!(matches!(
            q4k_matvec(&good, 0, k, &x, &mut out),
            Err(ExpertCpuError::BadShape(_))
        ));
        assert!(all_nan(&out), "a BadShape call must not touch out");

        // q4k_matvec: length violations name their slice and leave out untouched.
        let bad = |what: &'static str, got: usize, want: usize| ExpertCpuError::BadLength {
            what,
            got,
            want,
        };
        let mut bad_x = x.clone();
        bad_x.pop();
        let mut short_out = vec![f32::NAN; rows - 1];
        let long_w = [good.as_slice(), &[0u8; 144][..]].concat();
        assert_eq!(
            q4k_matvec(&good, rows, k, &bad_x, &mut out),
            Err(bad("x", k - 1, k))
        );
        assert_eq!(
            q4k_matvec(&good, rows, k, &x, &mut short_out),
            Err(bad("out", rows - 1, rows))
        );
        assert_eq!(
            q4k_matvec(&good[..good.len() - 1], rows, k, &x, &mut out),
            Err(bad("w", good.len() - 1, good.len()))
        );
        assert_eq!(
            q4k_matvec(&long_w, rows, k, &x, &mut out),
            Err(bad("w", good.len() + 144, good.len()))
        );
        assert!(
            all_nan(&out) && all_nan(&short_out),
            "length errors must not touch out"
        );

        // expert_forward: shape violations first, on otherwise valid arguments.
        for sh in [shape(512, 256), shape(2048, 512)] {
            let side = sh.inter * sh.hidden / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES;
            let down = sh.hidden * sh.inter / Q4K_BLOCK_VALUES * Q4K_BLOCK_BYTES;
            let zero = |n: usize| vec![0u8; n];
            let (gate, up, down_w, long_up) = (zero(side), zero(side), zero(down), zero(side + 1));
            let x = vec![0.5f32; sh.hidden];
            let mut scratch = ExpertScratch::new(sh);
            let mut out = vec![f32::NAN; sh.hidden];
            let good = sl(&gate, &up, &down_w);
            for bad in [shape(100, 512), shape(0, 512), shape(512, 0)] {
                assert!(matches!(
                    expert_forward(bad, &good, &x, &mut scratch, &mut out),
                    Err(ExpertCpuError::BadShape(_))
                ));
            }
            assert!(all_nan(&out), "a BadShape call must not touch out");

            // One mutated input per case; everything else stays valid.
            let sums_want = sh.hidden.max(sh.inter) / Q4K_BLOCK_VALUES * 8;
            let cases: [(&str, usize, usize, u8); 8] = [
                ("x", sh.hidden - 1, sh.hidden, 0),
                ("out", sh.hidden + 1, sh.hidden, 1),
                ("gate", side - 1, side, 2),
                ("up", side + 1, side, 3),
                ("down", down - Q4K_BLOCK_BYTES, down, 4),
                ("scratch.gate", sh.inter - 1, sh.inter, 5),
                ("scratch.act", sh.inter - 1, sh.inter, 6),
                ("scratch.sums", sums_want - 1, sums_want, 7),
            ];
            for (what, got, want, case) in cases {
                let slices = match case {
                    2 => sl(&gate[..side - 1], &up, &down_w),
                    3 => sl(&gate, &long_up, &down_w),
                    4 => sl(&gate, &up, &down_w[..down - Q4K_BLOCK_BYTES]),
                    _ => sl(&gate, &up, &down_w),
                };
                let xx = if case == 0 {
                    x[..sh.hidden - 1].to_vec()
                } else {
                    x.clone()
                };
                let mut sc = ExpertScratch::new(sh);
                match case {
                    5 => sc.gate.truncate(sh.inter - 1),
                    6 => sc.act.truncate(sh.inter - 1),
                    7 => sc.sums.truncate(sums_want - 1),
                    _ => {}
                }
                let mut oo = if case == 1 {
                    vec![f32::NAN; sh.hidden + 1]
                } else {
                    vec![f32::NAN; sh.hidden]
                };
                assert_eq!(
                    expert_forward(sh, &slices, &xx, &mut sc, &mut oo),
                    Err(bad(what, got, want))
                );
                let checked = if case == 1 { got } else { sh.hidden };
                assert!(all_nan(&oo[..checked]), "{what} case touched out");
            }

            // The untouched valid case must succeed and fill every output.
            expert_forward(sh, &good, &x, &mut scratch, &mut out).unwrap();
            assert!(out.iter().all(|v| !v.is_nan()), "valid call must fill out");
        }
    }

    /// Moderate f16 patterns for the full-expert tests, keeping f32 inside tolerance.
    const F16_BITS_MODERATE: [u16; 4] = [0x3500, 0x3c00, 0x4000, 0xbc00];

    #[test]
    fn expert_cpu_expert_forward_matches_a_dequantised_reference() {
        // Tolerance: 1e-3 relative to the largest reference output magnitude.
        let shape = shape(2048, 512);
        let mut rng = Lcg(0x1A2B_3C4D_5E6F_7081);
        let make = |rng: &mut Lcg, blocks: usize| -> Vec<u8> {
            (0..blocks)
                .flat_map(|_| {
                    let mut block = [0u8; 144];
                    random_block(rng, &mut block, &F16_BITS_MODERATE);
                    block
                })
                .collect()
        };
        let gate = make(&mut rng, shape.inter * shape.hidden / Q4K_BLOCK_VALUES);
        let up = make(&mut rng, shape.inter * shape.hidden / Q4K_BLOCK_VALUES);
        let down = make(&mut rng, shape.hidden * shape.inter / Q4K_BLOCK_VALUES);
        let x: Vec<f32> = (0..shape.hidden)
            .map(|_| rng.f32_between(-1.0, 1.0))
            .collect();
        let mut out = vec![f32::NAN; shape.hidden];
        let mut scratch = ExpertScratch::new(shape);
        let slices = ExpertSlices {
            gate: &gate,
            up: &up,
            down: &down,
        };
        expert_forward(shape, &slices, &x, &mut scratch, &mut out).unwrap();

        // Reference: dequantize the three slices, matvec in f64, silu in f64.
        let dequant = |raw: &[u8], values: usize| -> Vec<f32> {
            let mut w = vec![0.0f32; values];
            dequantize_into(GgufType::Q4K, raw, &mut w).unwrap();
            w
        };
        let matvec_f64 = |w: &[f32], rows: usize, k: usize, v: &[f32]| -> Vec<f64> {
            (0..rows)
                .map(|r| {
                    (0..k)
                        .map(|c| f64::from(w[r * k + c]) * f64::from(v[c]))
                        .sum::<f64>()
                })
                .collect()
        };
        let wg = dequant(&gate, shape.inter * shape.hidden);
        let wu = dequant(&up, shape.inter * shape.hidden);
        let wd = dequant(&down, shape.hidden * shape.inter);
        let g = matvec_f64(&wg, shape.inter, shape.hidden, &x);
        let u = matvec_f64(&wu, shape.inter, shape.hidden, &x);
        let act: Vec<f32> = (0..shape.inter)
            .map(|i| {
                let gi = g[i];
                (gi / (1.0 + (-gi).exp()) * u[i]) as f32
            })
            .collect();
        let reference = matvec_f64(&wd, shape.hidden, shape.inter, &act);
        let max = reference.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (got, want) in out.iter().zip(&reference) {
            assert!(
                (f64::from(*got) - want).abs() <= 1e-3 * max + 1e-6,
                "got {got}, ref {want}, max {max}"
            );
        }
    }

    #[test]
    #[ignore = "timing probe: lane L runs it with --ignored --nocapture in a release build"]
    fn expert_cpu_rate_is_recorded() {
        use std::time::Instant;

        let shape = shape(2048, 512);
        let mut rng = Lcg(0xCAFE_BABE_0000_0002);
        let make = |rng: &mut Lcg, blocks: usize| -> Vec<u8> {
            (0..blocks)
                .flat_map(|_| {
                    let mut block = [0u8; 144];
                    random_block(rng, &mut block, &F16_BITS_MODERATE);
                    block
                })
                .collect()
        };
        let gate = make(&mut rng, shape.inter * shape.hidden / Q4K_BLOCK_VALUES);
        let up = make(&mut rng, shape.inter * shape.hidden / Q4K_BLOCK_VALUES);
        let down = make(&mut rng, shape.hidden * shape.inter / Q4K_BLOCK_VALUES);
        let x: Vec<f32> = (0..shape.hidden)
            .map(|_| rng.f32_between(-1.0, 1.0))
            .collect();
        let slices = ExpertSlices {
            gate: &gate,
            up: &up,
            down: &down,
        };
        let mut scratch = ExpertScratch::new(shape);
        let mut out = vec![0.0f32; shape.hidden];
        let iterations = 2000u64;
        let start = Instant::now();
        for _ in 0..iterations {
            expert_forward(shape, &slices, &x, &mut scratch, &mut out).unwrap();
        }
        let elapsed = start.elapsed();
        let bytes = iterations as f64 * 3.0 * 589_824.0;
        println!(
            "expert_cpu_rate_is_recorded: {iterations} experts in {elapsed:?}: {:.2} GB/s of \
             Q4_K weight bytes",
            bytes / elapsed.as_secs_f64() / 1e9
        );
    }
}
