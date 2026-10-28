//! Gather and refusal tests of embed_q8 against a host dequantizer.
use super::*;
use crate::kernels::cuda::{CudaContext, testkit};

/// The gather of two rows equals the host dequantization of the same blocks.
#[test]
fn gathers_rows_against_the_host_dequantizer() {
    if !testkit::device_present("embed_q8 basic") {
        return;
    }
    const ROWS: usize = 8;
    const K: usize = 64;
    let mut table: Vec<u8> = Vec::new();
    for row in 0..ROWS {
        for block in 0..K / 32 {
            let scale = 0.03125_f32 * (row as f32 + 1.0) + block as f32 * 0.0078125;
            table.extend_from_slice(&half_bits(scale)[..]);
            for lane in 0..32 {
                table.push((((row * 7 + block * 3 + lane) % 251) as i8 - 125) as u8);
            }
        }
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let w = testkit::upload_bytes(&ctx, &table);
    let ids = testkit::upload_bytes(&ctx, &[3i32.to_le_bytes(), 6i32.to_le_bytes()].concat());
    let out = testkit::upload_filled(&ctx, 0xff, 2 * K * 4);
    ctx.embed_q8(
        &stream,
        EmbedQ8Call {
            out: out.range(0, 2 * K * 4).expect("out"),
            w: w.range(0, table.len()).expect("w"),
            ids: ids.range(0, 8).expect("ids"),
            m: 2,
            n_rows: ROWS,
            k: K,
        },
    )
    .expect("gather");
    stream.synchronize().expect("drain");
    let got = testkit::floats(&testkit::read_bytes(&ctx, &out, 2 * K * 4));
    for (token, id) in [3usize, 6].into_iter().enumerate() {
        for index in 0..K {
            let block = id * (K / 32) + index / 32;
            let base = block * 34;
            let scale = f16_bits_to_f32(u16::from_le_bytes([table[base], table[base + 1]]));
            let q = table[base + 2 + index % 32] as i8;
            let want = scale * f32::from(q);
            assert_eq!(got[token * K + index], want, "token={token} index={index}");
        }
    }
}

/// The wrapper refuses a row count above the gather limit before any native call.
#[test]
fn refuses_too_many_rows() {
    if !testkit::device_present("embed_q8 row refusal") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let w = testkit::upload_bytes(&ctx, &[0u8; 34 * 8]);
    let ids = testkit::upload_bytes(&ctx, &[0u8; 4]);
    let out = testkit::upload_filled(&ctx, 0xff, 64 * 4);
    let err = ctx
        .embed_q8(
            &stream,
            EmbedQ8Call {
                out: out.range(0, 64 * 4).expect("out"),
                w: w.range(0, 34 * 8).expect("w"),
                ids: ids.range(0, 4).expect("ids"),
                m: 4097,
                n_rows: 8,
                k: 32,
            },
        )
        .expect_err("rows refused");
    assert!(matches!(err, crate::EngError::BadRequest(_)), "{err:?}");
}

fn half_bits(value: f32) -> [u8; 2] {
    // Round to nearest even through the F16 conversion the device performs.
    let bits = f32_to_f16_bits(value);
    bits.to_le_bytes()
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased < -14 {
        return sign;
    }
    let half_exp = ((unbiased + 15) as u16) << 10;
    let half_mant = (mant >> 13) as u16;
    sign | half_exp | half_mant
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x03ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}
