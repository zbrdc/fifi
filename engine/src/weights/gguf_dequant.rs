//! CPU reference dequantizer from GGUF blocks to `f32`.
//!
//! The decoders are written from the block layouts and from the reference
//! behaviour observed through the ggml CPU oracle. Each function decodes one
//! block into its `f32` weights. Float expressions keep a fixed evaluation
//! order and never fuse a multiply with an add, so results are reproducible
//! bit for bit. Block scales that are half-precision values decode exactly; a
//! NaN scale becomes a quiet NaN.
//!
//! Oracle build effects. The fixtures come from the pinned oracle built with
//! `clang -O2 -ffp-contract=off` on arm64. Two effects follow from that build,
//! not from IEEE 754 or from the ggml source:
//!
//! * `clang -O2` folds a multiply by `-1.0` into a negation, which flips the
//!   sign bit of a NaN. At `-O0` the sign would be kept. [`signed`] follows the
//!   optimized behaviour.
//! * A contracted build (`-ffp-contract=on`) differs from the fixtures in 512
//!   values each for Q4_K and Q5_K. Later kernels must not contract `d * q - m`.

use super::gguf::GgufType;
use super::gguf_tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
    KSIGNS_IQ2XS, KVALUES_IQ4NL,
};
use crate::EngError;

fn bad(message: impl Into<String>) -> EngError {
    EngError::Checkpoint(format!("gguf dequant: {}", message.into()))
}

/// Converts IEEE binary16 bits to `f32`. A NaN keeps its payload and becomes quiet.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1f);
    let mant = u32::from(bits & 0x03ff);
    let out = match (exp, mant) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: normalize the mantissa.
            let shift = mant.leading_zeros() - 21;
            let m = (mant << shift) & 0x03ff;
            sign | ((113 - shift) << 23) | (m << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, _) => sign | 0x7fc0_0000 | (mant << 13),
        _ => sign | ((exp + 112) << 23) | (mant << 13),
    };
    f32::from_bits(out)
}

fn rd_f16(b: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[at], b[at + 1]]))
}

fn rd_u16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn rd_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Applies sign bit `j` of `signs` to `v`: negates when the bit is set.
///
/// The rule follows the pinned oracle build, not IEEE semantics. `clang -O2`
/// turns the reference's multiply by `-1.0` into a negation, which also flips
/// the sign bit of a NaN. A multiply would keep that sign (observed at `-O0`).
fn signed(v: f32, signs: u8, j: usize) -> f32 {
    if signs & KMASK_IQ2XS[j] != 0 { -v } else { v }
}

fn q2_0(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    for (j, out) in y.iter_mut().enumerate() {
        let q = i32::from((b[2 + j / 4] >> ((j % 4) * 2)) & 3);
        *out = (q - 1) as f32 * d;
    }
}

fn q8_0(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    for (j, out) in y.iter_mut().enumerate() {
        *out = d * f32::from(b[2 + j] as i8);
    }
}

/// Q5_0: `d`, the 32 high bits `qh` and 16 bytes of low nibbles. Element `j`
/// (0..16) takes the low nibble of `qs[j]`, element `j + 16` the high nibble;
/// bit `i` of `qh` is the fifth bit of element `i`. The quant is `(0..32) - 16`.
///
/// The product is `q * d` in that order, as the oracle writes `x0 * d`: with
/// `d = +0.0` and `q = -16` the result is `-0.0`, and the order fixes the sign
/// of a NaN `d`.
fn q5_0(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qh = rd_u32(b, 2);
    let qs = &b[6..22];
    for (j, &byte) in qs.iter().enumerate() {
        // The masks keep the fifth bit at 0 or 16, so the values fit an i32.
        let low_bit = i32::from((qh >> j) & 1 == 1) << 4;
        let high_bit = i32::from((qh >> (j + 16)) & 1 == 1) << 4;
        let low = i32::from(byte & 0x0f) | low_bit;
        let high = i32::from(byte >> 4) | high_bit;
        y[j] = (low - 16) as f32 * d;
        y[j + 16] = (high - 16) as f32 * d;
    }
}

/// Q4_0: `d`, then 16 code bytes. Value `i` (0..32) is the `i / 16` nibble
/// (low first) of code byte `i % 16`; the quant is `(0..16) - 8`, and the
/// product is `q * d` in that order, matching the ggml reference
/// `dequantize_row_q4_0` (`x0*d`). Each value's address comes from its
/// index; no ggml loop nest is transcribed.
fn q4_0(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    for (i, out) in y.iter_mut().enumerate() {
        let nib = (b[2 + i % 16] >> (4 * (i / 16))) & 0x0f;
        *out = (i32::from(nib) - 8) as f32 * d;
    }
}

/// Unpacks the 6-bit scale and minimum with index `j` (0..8) of a K-quant block.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn q2_k(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 80);
    let dmin = rd_f16(b, 82);
    for (group, out) in y.as_chunks_mut::<16>().0.iter_mut().enumerate() {
        let scale = b[group];
        let step = d * f32::from(scale & 15);
        let min = dmin * f32::from(scale >> 4);
        // Two groups share a shift in each 32-byte packed half-block.
        let offset = 16 + 32 * (group / 8) + 16 * (group % 2);
        let shift = 2 * ((group % 8) / 2);
        for (dst, &packed) in out.iter_mut().zip(&b[offset..][..16]) {
            *dst = step * f32::from((packed >> shift) & 3) - min;
        }
    }
}

/// Q3_K, decoded per value index from the published block layout; no ggml
/// loop nest or register shuffle is transcribed. The 110-byte block is 32
/// `hmask` bytes at +0, 64 `qs` bytes at +32, 12 packed 6-bit sub-scale
/// bytes at +96, and the f16 super-scale `d` LAST at +108.
///
/// Value `i` (0..256): `half = i / 128`, lane `j = (i % 128) / 32`, `r =
/// i % 32`. Its code is bit pair `2 * j` of `qs[32 * half + r]`; its hmask
/// bit is `1 << (4 * half + j)` of `hmask[r]` (set subtracts 0, clear 4).
/// Sub-scale `s = 8 * half + 2 * j + r / 16` unpacks as column `s % 4`, row
/// `s / 4` of the packed layout: the `s / 8` nibble of code byte
/// `ks[(s % 4 + s / 4 * 4) % 8]` in the low four bits, the top two bits
/// `2 * (s / 4)` of `ks[8 + s % 4]`. The value is `(d * (scale - 32)) * q`.
fn q3_k(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 108);
    let hm = &b[0..32];
    let qs = &b[32..96];
    let ks = &b[96..108];
    for (i, out) in y.iter_mut().enumerate() {
        let (half, j, r) = (i / 128, (i % 128) / 32, i % 32);
        let s = 8 * half + 2 * j + r / 16;
        let (row, col) = (s / 4, s % 4);
        let nib = (ks[(col + 4 * row) % 8] >> (4 * (row / 2))) & 0x0f;
        let h2 = (ks[8 + col] >> (2 * row)) & 3;
        let dl = d * (i32::from(nib | (h2 << 4)) - 32) as f32;
        let code = (qs[32 * half + r] >> (2 * j)) & 3;
        let off = if hm[r] & (1 << (4 * half + j)) != 0 {
            0
        } else {
            4
        };
        *out = dl * (i32::from(code) - off) as f32;
    }
}

fn q4_k(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let dmin = rd_f16(b, 2);
    let scales = &b[4..16];
    let qs = &b[16..144];
    for chunk in 0..4 {
        let (sc1, m1) = scale_min_k4(2 * chunk, scales);
        let (sc2, m2) = scale_min_k4(2 * chunk + 1, scales);
        let d1 = d * f32::from(sc1);
        let min1 = dmin * f32::from(m1);
        let d2 = d * f32::from(sc2);
        let min2 = dmin * f32::from(m2);
        let q = &qs[32 * chunk..32 * chunk + 32];
        let out = &mut y[64 * chunk..64 * chunk + 64];
        for l in 0..32 {
            out[l] = d1 * f32::from(q[l] & 0x0f) - min1;
            out[32 + l] = d2 * f32::from(q[l] >> 4) - min2;
        }
    }
}

fn q5_k(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let dmin = rd_f16(b, 2);
    let scales = &b[4..16];
    let qh = &b[16..48];
    let qs = &b[48..176];
    for chunk in 0..4 {
        let (sc1, m1) = scale_min_k4(2 * chunk, scales);
        let (sc2, m2) = scale_min_k4(2 * chunk + 1, scales);
        let d1 = d * f32::from(sc1);
        let min1 = dmin * f32::from(m1);
        let d2 = d * f32::from(sc2);
        let min2 = dmin * f32::from(m2);
        let u1 = 1u8 << (2 * chunk);
        let u2 = 2u8 << (2 * chunk);
        let q = &qs[32 * chunk..32 * chunk + 32];
        let out = &mut y[64 * chunk..64 * chunk + 64];
        for l in 0..32 {
            let hi1 = if qh[l] & u1 != 0 { 16 } else { 0 };
            let hi2 = if qh[l] & u2 != 0 { 16 } else { 0 };
            out[l] = d1 * f32::from((q[l] & 0x0f) + hi1) - min1;
            out[32 + l] = d2 * f32::from((q[l] >> 4) + hi2) - min2;
        }
    }
}

fn q6_k(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 208);
    for half in 0..2 {
        let ql = &b[64 * half..64 * half + 64];
        let qh = &b[128 + 32 * half..128 + 32 * half + 32];
        let sc = &b[192 + 8 * half..192 + 8 * half + 8];
        let out = &mut y[128 * half..128 * half + 128];
        for l in 0..32 {
            let is = l / 16;
            let h = qh[l];
            let q1 = i32::from((ql[l] & 0x0f) | ((h & 3) << 4)) - 32;
            let q2 = i32::from((ql[l + 32] & 0x0f) | (((h >> 2) & 3) << 4)) - 32;
            let q3 = i32::from((ql[l] >> 4) | (((h >> 4) & 3) << 4)) - 32;
            let q4 = i32::from((ql[l + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32;
            out[l] = d * f32::from(sc[is] as i8) * q1 as f32;
            out[l + 32] = d * f32::from(sc[is + 2] as i8) * q2 as f32;
            out[l + 64] = d * f32::from(sc[is + 4] as i8) * q3 as f32;
            out[l + 96] = d * f32::from(sc[is + 6] as i8) * q4 as f32;
        }
    }
}

fn iq1_grid(index: usize, delta: f32, scale: f32, values: &mut [f32; 8]) {
    let grid = IQ1S_GRID[index];
    for (lane, value) in values.iter_mut().enumerate() {
        let code = (grid >> (2 * lane)) & 3;
        *value = scale * (f32::from(code) - 1.0 + delta);
    }
}

fn iq1_s(b: &[u8], y: &mut [f32]) {
    let scale = rd_f16(b, 0);
    for (group, row) in y.as_chunks_mut::<32>().0.iter_mut().enumerate() {
        let high = rd_u16(b, 34 + 2 * group);
        let d = scale * f32::from(2 * ((high >> 12) & 7) + 1);
        let delta = if high & 0x8000 == 0 { 0.125 } else { -0.125 };
        for (segment, values) in row.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let low = usize::from(b[2 + 4 * group + segment]);
            let index = low | (usize::from((high >> (3 * segment)) & 7) << 8);
            iq1_grid(index, delta, d, values);
        }
    }
}

fn iq1_m(b: &[u8], y: &mut [f32]) {
    let scales = [rd_u16(b, 48), rd_u16(b, 50), rd_u16(b, 52), rd_u16(b, 54)];
    // The high nibble of each scale word contributes to the block's FP16 scale.
    let bits = scales.iter().enumerate().fold(0, |bits, (word, &packed)| {
        bits | ((packed >> 12) << (4 * word))
    });
    let scale = f16_to_f32(bits);
    for (group, row) in y.as_chunks_mut::<32>().0.iter_mut().enumerate() {
        let packed_scales = scales[group / 2] >> (6 * (group % 2));
        for (segment, values) in row.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let high = b[32 + 2 * group + segment / 2] >> (4 * (segment % 2));
            let index = usize::from(b[4 * group + segment]) | (usize::from(high & 7) << 8);
            let delta = if high & 8 == 0 { 0.125 } else { -0.125 };
            let multiplier = (packed_scales >> (3 * (segment / 2))) & 7;
            iq1_grid(index, delta, scale * f32::from(2 * multiplier + 1), values);
        }
    }
}

fn iq2_xxs(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    for ib in 0..8 {
        let a0 = rd_u32(b, 2 + 8 * ib);
        let a1 = rd_u32(b, 2 + 8 * ib + 4);
        let db = d * (0.5 + (a1 >> 28) as f32) * 0.25;
        for l in 0..4 {
            let grid = IQ2XXS_GRID[((a0 >> (8 * l)) & 0xff) as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize];
            for j in 0..8 {
                y[32 * ib + 8 * l + j] = signed(db * f32::from(grid[j]), signs, j);
            }
        }
    }
}

fn iq2_xs(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let scales = &b[66..74];
    for ib in 0..8 {
        let db = [
            d * (0.5 + f32::from(scales[ib] & 0x0f)) * 0.25,
            d * (0.5 + f32::from(scales[ib] >> 4)) * 0.25,
        ];
        for l in 0..4 {
            let v = rd_u16(b, 2 + 2 * (4 * ib + l));
            let grid = IQ2XS_GRID[usize::from(v & 511)].to_le_bytes();
            let signs = KSIGNS_IQ2XS[usize::from(v >> 9)];
            for j in 0..8 {
                y[32 * ib + 8 * l + j] = signed(db[l / 2] * f32::from(grid[j]), signs, j);
            }
        }
    }
}

fn iq2_s(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qs = &b[2..34];
    let signs = &b[34..66];
    let qh = &b[66..74];
    let scales = &b[74..82];
    for ib in 0..8 {
        let db = [
            d * (0.5 + f32::from(scales[ib] & 0x0f)) * 0.25,
            d * (0.5 + f32::from(scales[ib] >> 4)) * 0.25,
        ];
        for l in 0..4 {
            let hi = (usize::from(qh[ib]) << (8 - 2 * l)) & 0x300;
            let grid = IQ2S_GRID[usize::from(qs[4 * ib + l]) | hi].to_le_bytes();
            let sg = signs[4 * ib + l];
            for j in 0..8 {
                y[32 * ib + 8 * l + j] = signed(db[l / 2] * f32::from(grid[j]), sg, j);
            }
        }
    }
}

fn iq3_xxs(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qs = &b[2..66];
    for ib in 0..8 {
        let a = rd_u32(b, 66 + 4 * ib);
        let db = d * (0.5 + (a >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((a >> (7 * l)) & 127) as usize];
            let g1 = IQ3XXS_GRID[usize::from(qs[8 * ib + 2 * l])].to_le_bytes();
            let g2 = IQ3XXS_GRID[usize::from(qs[8 * ib + 2 * l + 1])].to_le_bytes();
            for j in 0..4 {
                y[32 * ib + 8 * l + j] = signed(db * f32::from(g1[j]), signs, j);
                y[32 * ib + 8 * l + j + 4] = signed(db * f32::from(g2[j]), signs, j + 4);
            }
        }
    }
}

fn iq3_s(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let qs = &b[2..66];
    let qh = &b[66..74];
    let signs = &b[74..106];
    let scales = &b[106..110];
    for ib in 0..8 {
        let s = scales[ib / 2];
        let nib = if ib % 2 == 0 { s & 0x0f } else { s >> 4 };
        let db = d * (1 + 2 * i32::from(nib)) as f32;
        let h = usize::from(qh[ib]);
        for l in 0..4 {
            let i1 = usize::from(qs[8 * ib + 2 * l]) | ((h << (8 - 2 * l)) & 256);
            let i2 = usize::from(qs[8 * ib + 2 * l + 1]) | ((h << (7 - 2 * l)) & 256);
            let g1 = IQ3S_GRID[i1].to_le_bytes();
            let g2 = IQ3S_GRID[i2].to_le_bytes();
            let sg = signs[4 * ib + l];
            for j in 0..4 {
                y[32 * ib + 8 * l + j] = signed(db * f32::from(g1[j]), sg, j);
                y[32 * ib + 8 * l + j + 4] = signed(db * f32::from(g2[j]), sg, j + 4);
            }
        }
    }
}

fn iq4_nl(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    for j in 0..16 {
        y[j] = d * f32::from(KVALUES_IQ4NL[usize::from(b[2 + j] & 0x0f)]);
        y[j + 16] = d * f32::from(KVALUES_IQ4NL[usize::from(b[2 + j] >> 4)]);
    }
}

fn iq4_xs(b: &[u8], y: &mut [f32]) {
    let d = rd_f16(b, 0);
    let scales_h = rd_u16(b, 2);
    let scales_l = &b[4..8];
    let qs = &b[8..136];
    for ib in 0..8 {
        let lo = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0f;
        let hi = ((scales_h >> (2 * ib)) & 3) as u8;
        let ls = i32::from(lo | (hi << 4));
        let dl = d * (ls - 32) as f32;
        for j in 0..16 {
            let q = qs[16 * ib + j];
            y[32 * ib + j] = dl * f32::from(KVALUES_IQ4NL[usize::from(q & 0x0f)]);
            y[32 * ib + j + 16] = dl * f32::from(KVALUES_IQ4NL[usize::from(q >> 4)]);
        }
    }
}

fn mxfp4(b: &[u8], y: &mut [f32]) {
    // The GGML E2M1 table uses doubled integer values, so halve the E8M0 scale.
    const VALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    let exponent = u32::from(b[0]);
    let scale = f32::from_bits(if exponent < 2 {
        0x0020_0000 << exponent
    } else {
        (exponent - 1) << 23
    });
    let (low, high) = y.split_at_mut(16);
    for ((lo, hi), &packed) in low.iter_mut().zip(high).zip(&b[1..]) {
        *lo = f32::from(VALUES[usize::from(packed & 15)]) * scale;
        *hi = f32::from(VALUES[usize::from(packed >> 4)]) * scale;
    }
}

fn nvfp4(b: &[u8], y: &mut [f32]) {
    // The GGUF integer-value convention makes code 8 positive zero.
    const VALUES: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    for (group, &code) in b[..4].iter().enumerate() {
        // GGUF ignores bit 7, except 0x7f is zero while 0xff is finite 480.
        let scale = match code {
            0x7f => 0.0,
            0xff => 480.0,
            _ => crate::kernels::nvfp4::E4M3_VALUES[usize::from(code & 0x7f)],
        };
        for j in 0..8 {
            let packed = b[4 + group * 8 + j];
            y[group * 16 + j] = VALUES[usize::from(packed & 15)] * scale;
            y[group * 16 + j + 8] = VALUES[usize::from(packed >> 4)] * scale;
        }
    }
}

type Decoder = fn(&[u8], &mut [f32]);

/// Checks the type first, then the source length. Returns the decoder, the
/// block size in bytes, the block size in values, and the number of values.
fn plan(ty: GgufType, src_len: usize) -> Result<(Decoder, usize, usize, usize), EngError> {
    let decode: Decoder = match ty {
        GgufType::Q20 => q2_0,
        GgufType::Q40 => q4_0,
        GgufType::Q2K => q2_k,
        GgufType::Q3K => q3_k,
        GgufType::Q4K => q4_k,
        GgufType::Q50 => q5_0,
        GgufType::Q5K => q5_k,
        GgufType::Q6K => q6_k,
        GgufType::Q80 => q8_0,
        GgufType::Mxfp4 => mxfp4,
        GgufType::Nvfp4 => nvfp4,
        GgufType::Iq1S => iq1_s,
        GgufType::Iq1M => iq1_m,
        GgufType::Iq2Xxs => iq2_xxs,
        GgufType::Iq2Xs => iq2_xs,
        GgufType::Iq2S => iq2_s,
        GgufType::Iq3Xxs => iq3_xxs,
        GgufType::Iq3S => iq3_s,
        GgufType::Iq4Nl => iq4_nl,
        GgufType::Iq4Xs => iq4_xs,
        GgufType::F32 => |b, y| y[0] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        GgufType::F16 => |b, y| y[0] = rd_f16(b, 0),
        GgufType::Bf16 => |b, y| {
            y[0] = f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16);
        },
        other => return Err(bad(format!("unsupported type {}", other.name()))),
    };
    let block_bytes = usize::try_from(ty.block_bytes()).map_err(|_| bad("block size"))?;
    let block_elems = usize::try_from(ty.block_elems()).map_err(|_| bad("block size"))?;
    if !src_len.is_multiple_of(block_bytes) {
        return Err(bad(format!(
            "input length {src_len} is not a multiple of {block_bytes} bytes for {}",
            ty.name()
        )));
    }
    let values = (src_len / block_bytes)
        .checked_mul(block_elems)
        .ok_or_else(|| bad("output length overflows"))?;
    Ok((decode, block_bytes, block_elems, values))
}

/// Dequantizes whole blocks of `ty` from `src` into `dst`.
///
/// `src` must hold a whole number of blocks. A malformed source or an
/// unsupported type returns `EngError::Checkpoint`. A `dst` that does not hold
/// exactly the block values returns `EngError::BadRequest`.
pub fn dequantize_into(ty: GgufType, src: &[u8], dst: &mut [f32]) -> Result<(), EngError> {
    let (decode, block_bytes, block_elems, values) = plan(ty, src.len())?;
    if dst.len() != values {
        return Err(EngError::BadRequest(format!(
            "gguf dequant: output length {} does not match {values} values for {}",
            dst.len(),
            ty.name()
        )));
    }
    for (b, y) in src
        .chunks_exact(block_bytes)
        .zip(dst.chunks_exact_mut(block_elems))
    {
        decode(b, y);
    }
    Ok(())
}

/// Project the first `rows` of a little-endian F32 GGUF matrix. The ordered
/// products are accumulated directly from the mapped row without a decoded copy.
pub(crate) fn project_f32_rows(src: &[u8], x: &[f32], rows: usize) -> Result<Vec<f32>, EngError> {
    let row_bytes = x
        .len()
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| bad("F32 row size"))?;
    if row_bytes == 0 || !src.len().is_multiple_of(row_bytes) || rows > src.len() / row_bytes {
        return Err(bad("invalid F32 matrix extent"));
    }
    Ok(src
        .chunks_exact(row_bytes)
        .take(rows)
        .map(|row| {
            row.as_chunks::<4>()
                .0
                .iter()
                .zip(x)
                .map(|(w, &value)| f32::from_le_bytes(*w) * value)
                .sum()
        })
        .collect())
}

/// Decodes one NVFP4 expert and applies its little-endian F32 tensor scale.
/// Each source expert has exactly `dst.len()` values. The scale count defines
/// the expert count. No weight copy is allocated; on numeric error `dst` may
/// already have been written.
pub fn dequantize_nvfp4_expert_into(
    src: &[u8],
    expert_scales: &[u8],
    expert: usize,
    dst: &mut [f32],
) -> Result<(), EngError> {
    let (scales, rest) = expert_scales.as_chunks::<4>();
    if scales.is_empty() || !rest.is_empty() {
        return Err(bad("NVFP4 expert scales must be nonempty F32 values"));
    }
    let scale = scales
        .get(expert)
        .ok_or_else(|| EngError::BadRequest("NVFP4 expert index is outside the tensor".into()))?;
    let scale = f32::from_le_bytes(*scale);
    if !scale.is_finite() {
        return Err(bad("NVFP4 tensor scale must be finite"));
    }
    if dst.is_empty() || !dst.len().is_multiple_of(64) {
        return Err(EngError::BadRequest(
            "NVFP4 expert output must contain whole nonempty blocks".into(),
        ));
    }
    let bytes = (dst.len() / 64)
        .checked_mul(36)
        .ok_or_else(|| bad("NVFP4 expert size overflows"))?;
    let total = bytes
        .checked_mul(scales.len())
        .ok_or_else(|| bad("NVFP4 source size overflows"))?;
    if src.len() != total {
        return Err(bad("NVFP4 source length differs from its expert geometry"));
    }
    let start = expert * bytes;
    dequantize_into(GgufType::Nvfp4, &src[start..start + bytes], dst)?;
    for value in dst {
        *value *= scale;
        if !value.is_finite() {
            return Err(bad("NVFP4 scaled weight is not finite"));
        }
    }
    Ok(())
}

/// Re-layouts GGUF NVFP4 blocks into ModelOpt adjacent-nibble data and E4M3FN scales.
/// Tensor multipliers remain separate. Unsigned scale 0xff decodes to 480,
/// which E4M3FN cannot represent; refuse it before writing either output.
pub fn repack_nvfp4_into(
    src: &[u8],
    packed: &mut [u8],
    group_scales: &mut [u8],
) -> Result<(), EngError> {
    let (blocks, rest) = src.as_chunks::<36>();
    if !rest.is_empty() {
        return Err(bad("NVFP4 source must contain whole 36-byte blocks"));
    }
    // Both count products are bounded by the borrowed source length.
    if packed.len() != blocks.len() * 32 || group_scales.len() != blocks.len() * 4 {
        return Err(EngError::BadRequest(
            "NVFP4 re-layout output size differs".into(),
        ));
    }
    if blocks.iter().any(|block| block[..4].contains(&0xff)) {
        return Err(bad(
            "NVFP4 unsigned scale 480 cannot be re-laid out as E4M3FN",
        ));
    }
    for (index, block) in blocks.iter().enumerate() {
        for (group, &scale) in block[..4].iter().enumerate() {
            group_scales[index * 4 + group] = if scale == 0x7f { 0 } else { scale & 0x7f };
            let data = &block[4 + group * 8..4 + (group + 1) * 8];
            let code = |column: usize| {
                let value = (data[column % 8] >> (4 * (column / 8))) & 15;
                // GGUF's integer zero must not become ModelOpt negative zero.
                if value == 8 { 0 } else { value }
            };
            for pair in 0..8 {
                packed[index * 32 + group * 8 + pair] = code(2 * pair) | (code(2 * pair + 1) << 4);
            }
        }
    }
    Ok(())
}

pub fn dequantize(ty: GgufType, src: &[u8]) -> Result<Vec<f32>, EngError> {
    let (_, _, _, values) = plan(ty, src.len())?;
    let mut out = vec![0f32; values];
    dequantize_into(ty, src, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn mapped_f32_projection_matches_the_decoded_rows_bit_for_bit() {
        use super::*;
        let x: Vec<_> = (0..23).map(|i| (3.25 - i as f32) / 13.0).collect();
        let bytes: Vec<_> = (0..5 * x.len())
            .flat_map(|i| ((i % 19) as f32 / 7.0 - 1.125).to_le_bytes())
            .collect();
        let want: Vec<_> = bytes
            .chunks_exact(x.len() * 4)
            .take(3)
            .map(|row| {
                let decoded = dequantize(GgufType::F32, row).unwrap();
                x.iter()
                    .zip(decoded)
                    .map(|(&a, b)| a * b)
                    .sum::<f32>()
                    .to_bits()
            })
            .collect();
        let got = project_f32_rows(&bytes, &x, 3).unwrap();
        assert_eq!(got.into_iter().map(f32::to_bits).collect::<Vec<_>>(), want);
        assert!(project_f32_rows(&bytes[..bytes.len() - 1], &x, 3).is_err());
        assert!(project_f32_rows(&bytes, &x, 6).is_err());
    }

    #[test]
    fn iq1_s_matches_the_reference_grid_scales_and_signs() {
        let mut block = std::array::from_fn::<_, 50, _>(|i| (i * 37 + 11) as u8);
        block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        let mut weights = [0.0; 256];
        super::dequantize_into(super::GgufType::Iq1S, &block, &mut weights).unwrap();
        assert_eq!(
            weights[..8],
            [-2.625, 0.375, 3.375, 3.375, 3.375, 0.375, 3.375, 0.375]
        );
        assert_eq!(
            weights[112..120],
            [
                -1.875, -1.875, -16.875, -16.875, -16.875, -16.875, -1.875, -1.875
            ]
        );
        assert_eq!(
            weights[248..],
            [-4.375, 5.625, 0.625, 0.625, -4.375, -4.375, 0.625, -4.375]
        );
    }

    use super::*;

    #[test]
    fn f16_conversion_covers_every_class() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x03ff), 1023.0 * 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x0400), 2f32.powi(-14));
        assert_eq!(f16_to_f32(0x8000).to_bits(), 0x8000_0000);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert_eq!(f16_to_f32(0x7d55).to_bits(), 0x7fea_a000);
    }

    #[test]
    fn q4_0_golden_fixed_block() {
        // One Q4_0 block: d = 1.0, then 16 nibble bytes. Expected values are
        // derived by hand from the ggml reference: element j is the LOW
        // nibble of qs[j] minus 8, element j + 16 the HIGH nibble minus 8.
        let mut b = [0u8; 18];
        b[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        let qs = [
            0x10u8, 0x21, 0xf0, 0x89, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0x55, 0x55, 0x55, 0x55,
        ];
        b[2..18].copy_from_slice(&qs);
        let mut y = [0f32; 32];
        q4_0(&b, &mut y);
        let mut want = [0f32; 32];
        for (j, &byte) in qs.iter().enumerate() {
            want[j] = f32::from(byte & 0x0f) - 8.0;
            want[j + 16] = f32::from(byte >> 4) - 8.0;
        }
        assert_eq!(y, want);
        // Spot values pin the nibble order and the d placement.
        assert_eq!(y[2], -8.0);
        assert_eq!(y[18], 7.0);
        assert_eq!(y[19], 0.0);
    }

    #[test]
    fn q3_k_golden_fixed_block() {
        // One Q3_K block, hand-traced against the ggml reference layout:
        // hmask[0] = 0x03 (bits for lanes 0 and 1 of the first half),
        // qs[0]=0, qs[1]=0, qs[16]=0xFF, qs[32]=0x01, qs[48]=0x02 (all other
        // qs bytes 0), scales = 0x35 for the first 8 bytes and 0 for the last
        // four, d = 1.0 stored LAST at offset 108.
        //
        // The 0x35/0x00 scale pack expands to sub-scale 5 for sub-blocks 0..8
        // (dl = -27) and 3 for sub-blocks 8..16 (dl = -29): the low nibbles
        // 0x5 land in aux lanes 0/1, the high nibbles 0x3 in lanes 2/3.
        // Every untouched value reads q = 0 with its hmask bit clear, so q is
        // -4: y = 108.0 in the first half and 116.0 in the second.
        let mut b = [0u8; 110];
        b[0] = 0x03; // hmask[0]
        b[32 + 16] = 0xff; // qs[16]
        b[32 + 32] = 0x01; // qs[32]
        b[32 + 48] = 0x02; // qs[48]
        for k in 0..8 {
            b[96 + k] = 0x35;
        }
        b[108..110].copy_from_slice(&0x3c00u16.to_le_bytes());
        let mut y = [0f32; 256];
        q3_k(&b, &mut y);

        let mut want = [0f32; 256];
        want.fill(0.0);
        for v in want.iter_mut().take(128) {
            *v = 108.0;
        }
        for v in want.iter_mut().skip(128) {
            *v = 116.0;
        }
        // hm[0] bit 0 set: value 0 keeps q = 0 in lane 0.
        want[0] = 0.0;
        // hm[0] bit 1 set: value 32 (lane 1, same hmask byte) keeps q = 0.
        want[32] = 0.0;
        // qs[16] = 0xFF fills every lane: values 16, 48, 80 and 112 all read
        // 3 in their lane bits, minus 4 for the clear hmask bit.
        want[16] = 27.0;
        want[48] = 27.0;
        want[80] = 27.0;
        want[112] = 27.0;
        // qs[32] = 0x01: value 128 (second half, lane 0) reads 1, q = -3:
        // -29 * -3.
        want[128] = 87.0;
        // qs[48] = 0x02: value 144 (second half, group two, lane 0) reads 2,
        // q = -2: -29 * -2.
        want[144] = 58.0;
        assert_eq!(y, want);
    }

    #[test]
    fn q3_k_golden_second_block_mixed_signs() {
        // A second hand-traced block: qs all zero, so every value is 0 or -4
        // by its hmask bit alone; the 16 sub-scales mix signs across all four
        // aux rows (+17, -3, +1, -30) and every hmask byte carries both set
        // and clear bits in both halves (0x55 on even bytes, 0xAA on odd;
        // bits 2*half + j). Scale pack:
        // ks[0..4] = 0x11 (row 0 low nibble 1, row 2 high nibble 1),
        // ks[4..8] = 0x2D (row 1 low nibble 0xD, row 3 high nibble 2),
        // ks[8..12] = 0x27 (the 2-bit high fields 3, 1, 2 of rows 0, 1, 2).
        let mut b = [0u8; 110];
        for (l, slot) in b.iter_mut().enumerate().take(32) {
            *slot = if l % 2 == 0 { 0x55 } else { 0xaa };
        }
        for k in 0..4 {
            b[96 + k] = 0x11;
            b[100 + k] = 0x2d;
            b[104 + k] = 0x27;
        }
        b[108..110].copy_from_slice(&0x3c00u16.to_le_bytes());
        let mut y = [0f32; 256];
        q3_k(&b, &mut y);

        // dl per (half, j): sub-scale is = 8*half + 2*j + g2 picks rows
        // 0..4 -> +17, 4..8 -> -3, 8..12 -> +1, 12..16 -> -30 (both groups of
        // a lane share one row, so dl depends on (half, j) only).
        const DL: [[f32; 4]; 2] = [[17.0, 17.0, -3.0, -3.0], [1.0, 1.0, -30.0, -30.0]];
        let mut want = [0f32; 256];
        for (half, half_dl) in DL.iter().enumerate() {
            for (j, &dl) in half_dl.iter().enumerate() {
                // The hmask bit of (half, j) is 4*half + j: even bytes
                // (0x55) set the even bits, odd bytes (0xAA) the odd bits,
                // so the bit is clear exactly on l of the parity opposite
                // the bit's, which is j % 2.
                let clear_odd_l = j % 2 == 0;
                for g2 in 0..2 {
                    for l in 0..16 {
                        let l_odd = l % 2 == 1;
                        let clear = l_odd == clear_odd_l;
                        if clear {
                            let v = half * 128 + j * 32 + g2 * 16 + l;
                            want[v] = dl * -4.0;
                        }
                    }
                }
            }
        }
        assert_eq!(y, want);
        // Spot values pin the signs: value 1 (half 0, lane 0, odd l, clear
        // bit) reads -68; value 64 (half 0, lane 2, even l, set bit) reads 0;
        // value 145 (half 1, lane 0, odd l) reads -4; value 240 (half 1,
        // lane 3, even l, clear) reads 120.
        assert_eq!(y[1], -68.0);
        assert_eq!(y[64], 0.0);
        assert_eq!(y[145], -4.0);
        assert_eq!(y[240], 120.0);
    }

    #[test]
    fn q3_k_golden_hmask_bit_is_per_half() {
        // Hand-derived from the layout: every sub-scale unpacks to 49 (dl =
        // 17), every code is 0, and every hmask byte is 0xF0, whose high
        // nibble sets bits 4-7 (half 1, lanes 0-3) while the low nibble
        // clears bits 0-3 (half 0). Half 0 must read -68 everywhere and
        // half 1 zero; a bit index of 2 * half + j would leave half-1 lanes
        // 0 and 1 reading bits 2-3 (clear) and fail this golden.
        let mut b = [0u8; 110];
        b[0..32].fill(0xF0);
        // qs stay 0 (code 0 everywhere).
        for k in 0..8 {
            b[96 + k] = 0x11; // low nibble 1 for rows 0-1, high nibble 1 for rows 2-3
        }
        for k in 8..12 {
            b[96 + k] = 0xFF; // top two bits 3 for every row
        }
        b[108..110].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
        let y = dequantize(GgufType::Q3K, &b).expect("decode");
        let mut want = vec![-68.0f32; 128];
        want.resize(256, 0.0);
        assert_eq!(y, want);
        assert_eq!(y[0], -68.0);
        assert_eq!(y[127], -68.0);
        assert_eq!(y[128], 0.0);
        assert_eq!(y[255], 0.0);
    }

    #[test]
    fn f16_matches_exhaustive_reference() {
        for bits in 0..=u16::MAX {
            let v = f16_to_f32(bits);
            if v.is_nan() {
                assert_eq!(v.to_bits() & 0x7fc0_0000, 0x7fc0_0000);
                continue;
            }
            // Re-encoding through exact arithmetic: sign * mant * 2^exp.
            let exp = i32::from((bits >> 10) & 0x1f);
            let mant = f64::from(bits & 0x3ff);
            let mag = if exp == 0 {
                mant * 2f64.powi(-24)
            } else if exp == 31 {
                f64::INFINITY
            } else {
                (1024.0 + mant) * 2f64.powi(exp - 25)
            };
            let want = if bits & 0x8000 != 0 { -mag } else { mag };
            assert_eq!(f64::from(v).to_bits(), want.to_bits(), "{bits:#06x}");
        }
    }
}
