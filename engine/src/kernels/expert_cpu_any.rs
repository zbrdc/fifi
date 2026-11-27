//! One routed expert on the CPU for any GGUF type with a decoder: the kernel of the tier when a
//! file stores an expert stack in a type the Q4_K kernels do not read.
//!
//! The three projections are decoded a row at a time to f32 (`weights::dequantize_into`, the
//! oracle every device kernel is checked against) and dotted with the f32 activation, so the
//! result is the expert evaluated in f32 over the decoded weights. The dot accumulates into eight
//! lanes in index order and reduces in a fixed order, so it does not depend on how the compiler
//! vectorizes it.

use super::expert_cpu::{ExpertCpuError, ExpertShape};
use crate::weights::{GgufType, dequantize_into};

/// The raw slices of one expert, each in its own type's blocks.
pub struct AnyExpertSlices<'a> {
    pub gate: (GgufType, &'a [u8]),
    pub up: (GgufType, &'a [u8]),
    pub down: (GgufType, &'a [u8]),
}

/// Reusable buffers of [`expert_forward_any`]: the two projections, the activation and one decoded
/// row.
pub struct ExpertScratchAny {
    gate: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    row: Vec<f32>,
}

impl ExpertScratchAny {
    pub fn new(shape: ExpertShape) -> Self {
        Self {
            gate: vec![0.0; shape.inter],
            up: vec![0.0; shape.inter],
            act: vec![0.0; shape.inter],
            row: vec![0.0; shape.hidden.max(shape.inter)],
        }
    }
}

/// Bytes of `rows` rows of `k` values in `ty` blocks, `None` when `k` is off the block grid or the
/// size overflows.
pub fn slice_bytes(ty: GgufType, rows: usize, k: usize) -> Option<usize> {
    let (elems, bytes) = (
        usize::try_from(ty.block_elems()).ok()?,
        usize::try_from(ty.block_bytes()).ok()?,
    );
    if elems == 0 || !k.is_multiple_of(elems) {
        return None;
    }
    rows.checked_mul(k / elems)?.checked_mul(bytes)
}

/// Bytes of the gate, up and down slices of one expert of `shape` in `quant`, `None` when a row
/// length is off its type's block grid.
pub fn expert_slice_bytes(shape: ExpertShape, quant: [GgufType; 3]) -> Option<[usize; 3]> {
    Some([
        slice_bytes(quant[0], shape.inter, shape.hidden)?,
        slice_bytes(quant[1], shape.inter, shape.hidden)?,
        slice_bytes(quant[2], shape.hidden, shape.inter)?,
    ])
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let (a8, at) = a.as_chunks::<8>();
    let (b8, bt) = b.as_chunks::<8>();
    for (x, y) in a8.iter().zip(b8) {
        for l in 0..8 {
            lanes[l] += x[l] * y[l];
        }
    }
    let tail: f32 = at.iter().zip(bt).map(|(x, y)| x * y).sum();
    ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
        + tail
}

/// `out[r] = w[r] . x` for the rows of `bytes`.
fn matvec(
    ty: GgufType,
    bytes: &[u8],
    k: usize,
    x: &[f32],
    row: &mut [f32],
    out: &mut [f32],
) -> Result<(), ExpertCpuError> {
    let row_bytes = slice_bytes(ty, 1, k).ok_or_else(|| {
        ExpertCpuError::BadShape(format!(
            "{} rows of {k} values are off the block grid",
            ty.name()
        ))
    })?;
    let row = &mut row[..k];
    for (chunk, y) in bytes.chunks_exact(row_bytes).zip(out.iter_mut()) {
        dequantize_into(ty, chunk, row)
            .map_err(|error| ExpertCpuError::BadShape(format!("{error:?}")))?;
        *y = dot(row, x);
    }
    Ok(())
}

/// [`expert_forward_any`] over the three raw slices of one expert in `quant`, building the
/// scratch in `scratch` on first use.
pub fn expert_forward_typed(
    shape: ExpertShape,
    quant: [GgufType; 3],
    [gate, up, down]: [&[u8]; 3],
    x: &[f32],
    scratch: &mut Option<ExpertScratchAny>,
    out: &mut [f32],
) -> Result<(), ExpertCpuError> {
    let scratch = scratch.get_or_insert_with(|| ExpertScratchAny::new(shape));
    let slices = AnyExpertSlices {
        gate: (quant[0], gate),
        up: (quant[1], up),
        down: (quant[2], down),
    };
    expert_forward_any(shape, &slices, x, scratch, out)
}

/// Runs one expert: `out = down(silu(gate(x)) * up(x))`. Every length implied by `shape` is checked
/// before any computation, so on error `out` is untouched.
pub fn expert_forward_any(
    shape: ExpertShape,
    s: &AnyExpertSlices<'_>,
    x: &[f32],
    scratch: &mut ExpertScratchAny,
    out: &mut [f32],
) -> Result<(), ExpertCpuError> {
    let ExpertShape { hidden, inter } = shape;
    if hidden == 0 || inter == 0 {
        return Err(ExpertCpuError::BadShape(format!(
            "expert shape {hidden}x{inter} has a zero extent"
        )));
    }
    let want = |ty: GgufType, rows: usize, k: usize| {
        slice_bytes(ty, rows, k).ok_or_else(|| {
            ExpertCpuError::BadShape(format!(
                "{} rows of {k} values are off the block grid",
                ty.name()
            ))
        })
    };
    for (what, (ty, bytes), rows, k) in [
        ("gate", s.gate, inter, hidden),
        ("up", s.up, inter, hidden),
        ("down", s.down, hidden, inter),
    ] {
        let wanted = want(ty, rows, k)?;
        if bytes.len() != wanted {
            return Err(ExpertCpuError::BadLength {
                what,
                got: bytes.len(),
                want: wanted,
            });
        }
    }
    for (what, got, wanted) in [("x", x.len(), hidden), ("out", out.len(), hidden)] {
        if got != wanted {
            return Err(ExpertCpuError::BadLength {
                what,
                got,
                want: wanted,
            });
        }
    }
    let ExpertScratchAny { gate, up, act, row } = scratch;
    matvec(s.gate.0, s.gate.1, hidden, x, row, gate)?;
    matvec(s.up.0, s.up.1, hidden, x, row, up)?;
    for ((a, g), u) in act.iter_mut().zip(gate.iter()).zip(up.iter()) {
        *a = g / (1.0 + (-g).exp()) * u;
    }
    matvec(s.down.0, s.down.1, inter, act, row, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::expert_cpu::{ExpertScratch, ExpertSlices, expert_forward};

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 11
        }
        fn unit(&mut self) -> f32 {
            (self.next() % 2001) as f32 / 1000.0 - 1.0
        }
    }

    /// Encodes `values` in `ty` where an encoder exists: Q8_0 (and F32 through it is not used).
    fn q8_0_bytes(rng: &mut Lcg, rows: usize, k: usize) -> Vec<u8> {
        let values: Vec<f32> = (0..rows * k).map(|_| rng.unit()).collect();
        crate::weights::encode_q8_0(&values).expect("encode")
    }

    fn q4k_bytes(rng: &mut Lcg, rows: usize, k: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; rows * k / 256 * 144];
        for block in bytes.as_chunks_mut::<144>().0 {
            block.fill_with(|| (rng.next() & 0xff) as u8);
            // Small finite binary16 scales: 0x2c00 = 0.0625.
            block[0..2].copy_from_slice(&0x2c00u16.to_le_bytes());
            block[2..4].copy_from_slice(&0x2800u16.to_le_bytes());
        }
        bytes
    }

    const SHAPE: ExpertShape = ExpertShape {
        hidden: 512,
        inter: 256,
    };

    #[test]
    fn q4_k_experts_match_the_q4_k_kernel_to_rounding() {
        let mut rng = Lcg(7);
        let gate = q4k_bytes(&mut rng, SHAPE.inter, SHAPE.hidden);
        let up = q4k_bytes(&mut rng, SHAPE.inter, SHAPE.hidden);
        let down = q4k_bytes(&mut rng, SHAPE.hidden, SHAPE.inter);
        let x: Vec<f32> = (0..SHAPE.hidden).map(|_| rng.unit()).collect();
        let mut want = vec![0.0; SHAPE.hidden];
        expert_forward(
            SHAPE,
            &ExpertSlices {
                gate: &gate,
                up: &up,
                down: &down,
            },
            &x,
            &mut ExpertScratch::new(SHAPE),
            &mut want,
        )
        .expect("q4_k kernel");
        let mut got = vec![0.0; SHAPE.hidden];
        expert_forward_any(
            SHAPE,
            &AnyExpertSlices {
                gate: (GgufType::Q4K, &gate),
                up: (GgufType::Q4K, &up),
                down: (GgufType::Q4K, &down),
            },
            &x,
            &mut ExpertScratchAny::new(SHAPE),
            &mut got,
        )
        .expect("any kernel");
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() <= 1e-3 * scale, "{a} vs {b}");
        }
    }

    #[test]
    fn mixed_types_run_and_a_wrong_length_is_refused_before_any_write() {
        let mut rng = Lcg(11);
        let gate = q8_0_bytes(&mut rng, SHAPE.inter, SHAPE.hidden);
        let up = q8_0_bytes(&mut rng, SHAPE.inter, SHAPE.hidden);
        let down = q4k_bytes(&mut rng, SHAPE.hidden, SHAPE.inter);
        let x: Vec<f32> = (0..SHAPE.hidden).map(|_| rng.unit()).collect();
        let slices = AnyExpertSlices {
            gate: (GgufType::Q80, &gate),
            up: (GgufType::Q80, &up),
            down: (GgufType::Q4K, &down),
        };
        let mut out = vec![0.0; SHAPE.hidden];
        expert_forward_any(
            SHAPE,
            &slices,
            &x,
            &mut ExpertScratchAny::new(SHAPE),
            &mut out,
        )
        .expect("mixed");
        assert!(out.iter().all(|v| v.is_finite()) && out.iter().any(|v| *v != 0.0));
        let short = AnyExpertSlices {
            gate: (GgufType::Q80, &gate[..gate.len() - 34]),
            ..slices
        };
        let mut untouched = vec![9.0; SHAPE.hidden];
        let error = expert_forward_any(
            SHAPE,
            &short,
            &x,
            &mut ExpertScratchAny::new(SHAPE),
            &mut untouched,
        )
        .expect_err("short gate");
        assert!(matches!(
            error,
            ExpertCpuError::BadLength { what: "gate", .. }
        ));
        assert!(untouched.iter().all(|v| *v == 9.0));
    }

    #[test]
    fn slice_bytes_follow_the_block_grid() {
        assert_eq!(slice_bytes(GgufType::Q4K, 512, 2048), Some(512 * 8 * 144));
        assert_eq!(slice_bytes(GgufType::Q80, 4, 64), Some(4 * 2 * 34));
        assert_eq!(slice_bytes(GgufType::Q4K, 4, 100), None);
    }

    type Decode<'a> = &'a dyn Fn(&[u8]) -> Vec<f32>;

    // Independent per-block references, written from the block layouts alone: they share
    // nothing with `weights::dequantize_into`, which the kernel under test calls.

    fn f16(bits: u16) -> f32 {
        half::f16::from_bits(bits).to_f32()
    }

    fn put_f16(block: &mut [u8], at: usize, value: f32) {
        block[at..at + 2].copy_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
    }

    fn k4_scale_min(j: usize, q: &[u8]) -> (f32, f32) {
        let (sc, mn) = crate::weights::gguf_dequant::scale_min_k4(j, q);
        (f32::from(sc), f32::from(mn))
    }

    fn q8_0_block(b: &[u8]) -> Vec<f32> {
        let d = f16(u16::from_le_bytes([b[0], b[1]]));
        b[2..34].iter().map(|&q| d * f32::from(q as i8)).collect()
    }

    fn q5_k_block(b: &[u8]) -> Vec<f32> {
        let (d, dmin) = (
            f16(u16::from_le_bytes([b[0], b[1]])),
            f16(u16::from_le_bytes([b[2], b[3]])),
        );
        let (scales, qh, qs) = (&b[4..16], &b[16..48], &b[48..176]);
        let mut y = Vec::with_capacity(256);
        for chunk in 0..4 {
            let (sc1, m1) = k4_scale_min(2 * chunk, scales);
            let (sc2, m2) = k4_scale_min(2 * chunk + 1, scales);
            let (u1, u2) = (1u8 << (2 * chunk), 2u8 << (2 * chunk));
            let ql = &qs[chunk * 32..chunk * 32 + 32];
            for l in 0..32 {
                let hi = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                y.push(d * sc1 * (f32::from(ql[l] & 0xf) + hi) - dmin * m1);
            }
            for l in 0..32 {
                let hi = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                y.push(d * sc2 * (f32::from(ql[l] >> 4) + hi) - dmin * m2);
            }
        }
        y
    }

    fn q6_k_block(b: &[u8]) -> Vec<f32> {
        let d = f16(u16::from_le_bytes([b[208], b[209]]));
        let mut y = vec![0f32; 256];
        for half in 0..2 {
            let ql = &b[half * 64..half * 64 + 64];
            let qh = &b[128 + half * 32..128 + half * 32 + 32];
            let sc = &b[192 + half * 8..192 + half * 8 + 8];
            for l in 0..32 {
                let at = |shift: u32| i32::from((qh[l] >> shift) & 3) << 4;
                let q1 = (i32::from(ql[l] & 0xf) | at(0)) - 32;
                let q2 = (i32::from(ql[l + 32] & 0xf) | at(2)) - 32;
                let q3 = (i32::from(ql[l] >> 4) | at(4)) - 32;
                let q4 = (i32::from(ql[l + 32] >> 4) | at(6)) - 32;
                let s = |i: usize| f32::from(sc[i] as i8);
                let base = half * 128;
                y[base + l] = d * s(l / 16) * q1 as f32;
                y[base + l + 32] = d * s(l / 16 + 2) * q2 as f32;
                y[base + l + 64] = d * s(l / 16 + 4) * q3 as f32;
                y[base + l + 96] = d * s(l / 16 + 6) * q4 as f32;
            }
        }
        y
    }

    const KVALUES_IQ4: [f32; 16] = [
        -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0,
        69.0, 89.0, 113.0,
    ];

    fn iq4_xs_block(b: &[u8]) -> Vec<f32> {
        let d = f16(u16::from_le_bytes([b[0], b[1]]));
        let scales_h = u16::from_le_bytes([b[2], b[3]]);
        let (scales_l, qs) = (&b[4..8], &b[8..136]);
        let mut y = Vec::with_capacity(256);
        for ib in 0..8 {
            let low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf;
            let ls = i32::from(low) | i32::from((scales_h >> (2 * ib)) & 3) << 4;
            let dl = d * (ls - 32) as f32;
            let q = &qs[ib * 16..ib * 16 + 16];
            y.extend(q.iter().map(|&v| dl * KVALUES_IQ4[usize::from(v & 0xf)]));
            y.extend(q.iter().map(|&v| dl * KVALUES_IQ4[usize::from(v >> 4)]));
        }
        y
    }

    /// The sign byte of a 7-bit pattern: the pattern with an eighth bit that makes the parity of
    /// the byte even, which is how the format stores eight signs in seven bits.
    #[cfg(feature = "native-cuda")]
    fn iq3_signs(pattern: u32) -> u8 {
        let low = (pattern & 127) as u8;
        low | (((low.count_ones() & 1) as u8) << 7)
    }

    #[cfg(feature = "native-cuda")]
    fn iq3_xxs_block(b: &[u8]) -> Vec<f32> {
        use crate::weights::IQ3XXS_GRID;
        let d = f16(u16::from_le_bytes([b[0], b[1]]));
        let (qs, sas) = (&b[2..66], &b[66..98]);
        let mut y = Vec::with_capacity(256);
        for ib in 0..8 {
            let aux = u32::from_le_bytes([
                sas[4 * ib],
                sas[4 * ib + 1],
                sas[4 * ib + 2],
                sas[4 * ib + 3],
            ]);
            let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
            for l in 0..4 {
                let signs = iq3_signs(aux >> (7 * l));
                for (half, grid) in [
                    IQ3XXS_GRID[usize::from(qs[ib * 8 + 2 * l])],
                    IQ3XXS_GRID[usize::from(qs[ib * 8 + 2 * l + 1])],
                ]
                .into_iter()
                .enumerate()
                {
                    for j in 0..4 {
                        let sign = if signs & (1 << (j + 4 * half)) != 0 {
                            -1.0
                        } else {
                            1.0
                        };
                        y.push(db * f32::from(grid.to_le_bytes()[j]) * sign);
                    }
                }
            }
        }
        y
    }

    /// Random blocks of `ty` with its f16 scale fields set to small finite values.
    fn random_rows(rng: &mut Lcg, ty: GgufType, rows: usize, k: usize) -> Vec<u8> {
        let (elems, bytes) = (ty.block_elems() as usize, ty.block_bytes() as usize);
        let mut out = vec![0u8; rows * k / elems * bytes];
        for block in out.chunks_exact_mut(bytes) {
            block.fill_with(|| (rng.next() & 0xff) as u8);
            let scales: &[usize] = match ty {
                GgufType::Q5K => &[0, 2],
                GgufType::Q6K => &[208],
                _ => &[0],
            };
            for &at in scales {
                put_f16(block, at, 0.01 + (rng.next() % 100) as f32 * 1e-4);
            }
        }
        out
    }

    /// The expert in f64 over the reference decoder of each stack's type.
    fn reference_expert(
        decode: [Decode<'_>; 3],
        types: [GgufType; 3],
        slices: [&[u8]; 3],
        x: &[f32],
    ) -> Vec<f64> {
        let rows_of = |which: usize, k: usize| -> Vec<Vec<f64>> {
            let bytes = types[which].block_bytes() as usize;
            let row_bytes = k / types[which].block_elems() as usize * bytes;
            slices[which]
                .chunks_exact(row_bytes)
                .map(|row| {
                    row.chunks_exact(bytes)
                        .flat_map(|block| decode[which](block))
                        .map(f64::from)
                        .collect()
                })
                .collect()
        };
        let dot = |row: &[f64], v: &[f64]| row.iter().zip(v).map(|(a, b)| a * b).sum::<f64>();
        let x64: Vec<f64> = x.iter().copied().map(f64::from).collect();
        let (gate, up) = (rows_of(0, SHAPE.hidden), rows_of(1, SHAPE.hidden));
        let act: Vec<f64> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| {
                let (g, u) = (dot(g, &x64), dot(u, &x64));
                g / (1.0 + (-g).exp()) * u
            })
            .collect();
        rows_of(2, SHAPE.inter)
            .iter()
            .map(|row| dot(row, &act))
            .collect()
    }

    fn check_against_reference(types: [GgufType; 3], decode: [Decode<'_>; 3], seed: u64) {
        let mut rng = Lcg(seed);
        let gate = random_rows(&mut rng, types[0], SHAPE.inter, SHAPE.hidden);
        let up = random_rows(&mut rng, types[1], SHAPE.inter, SHAPE.hidden);
        let down = random_rows(&mut rng, types[2], SHAPE.hidden, SHAPE.inter);
        let x: Vec<f32> = (0..SHAPE.hidden).map(|_| rng.unit()).collect();
        let mut got = vec![0.0; SHAPE.hidden];
        expert_forward_typed(SHAPE, types, [&gate, &up, &down], &x, &mut None, &mut got)
            .expect("kernel");
        let want = reference_expert(decode, types, [&gate, &up, &down], &x);
        let scale = want.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-9);
        for (a, b) in got.iter().zip(&want) {
            assert!(
                (f64::from(*a) - b).abs() <= 2e-4 * scale,
                "{types:?}: {a} against the reference {b}"
            );
        }
    }

    #[test]
    fn q5_k_q6_k_and_q8_0_experts_match_the_per_block_reference() {
        check_against_reference(
            [GgufType::Q5K, GgufType::Q5K, GgufType::Q6K],
            [&q5_k_block, &q5_k_block, &q6_k_block],
            3,
        );
        check_against_reference(
            [GgufType::Q80, GgufType::Q6K, GgufType::Q80],
            [&q8_0_block, &q6_k_block, &q8_0_block],
            5,
        );
    }

    #[test]
    fn iq4_xs_experts_match_the_per_block_reference() {
        check_against_reference(
            [GgufType::Iq4Xs, GgufType::Iq4Xs, GgufType::Q80],
            [&iq4_xs_block, &iq4_xs_block, &q8_0_block],
            7,
        );
    }

    /// The crate's one IQ3_XXS table is 256 distinct entries in ascending order whose bytes are
    /// the eight magnitudes of the format, and the sign byte of every 7-bit pattern the reference
    /// derives has even parity.
    #[cfg(feature = "native-cuda")]
    #[test]
    fn the_iq3_xxs_grid_has_the_structure_of_the_format() {
        use crate::weights::IQ3XXS_GRID;
        let mut bytes = std::collections::BTreeSet::new();
        for entry in IQ3XXS_GRID {
            bytes.extend(entry.to_le_bytes());
        }
        assert_eq!(
            bytes.into_iter().collect::<Vec<_>>(),
            [4, 12, 20, 28, 36, 44, 52, 62]
        );
        assert!(IQ3XXS_GRID.windows(2).all(|pair| pair[0] < pair[1]));
        assert!((0..128u32).all(|pattern| iq3_signs(pattern).count_ones().is_multiple_of(2)));
    }

    #[cfg(feature = "native-cuda")]
    #[test]
    fn iq3_xxs_experts_match_the_per_block_reference() {
        check_against_reference(
            [GgufType::Iq3Xxs, GgufType::Iq3Xxs, GgufType::Iq4Xs],
            [&iq3_xxs_block, &iq3_xxs_block, &iq4_xs_block],
            11,
        );
    }
}
