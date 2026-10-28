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
    IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS,
    KVALUES_IQ4NL,
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
        GgufType::Q4K => q4_k,
        GgufType::Q50 => q5_0,
        GgufType::Q5K => q5_k,
        GgufType::Q6K => q6_k,
        GgufType::Q80 => q8_0,
        GgufType::Nvfp4 => nvfp4,
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

/// Dequantizes whole blocks of `ty` from `src` into a new vector.
pub fn dequantize(ty: GgufType, src: &[u8]) -> Result<Vec<f32>, EngError> {
    let (_, _, _, values) = plan(ty, src.len())?;
    let mut out = vec![0f32; values];
    dequantize_into(ty, src, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
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
