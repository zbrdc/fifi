//! Comparisons against references frozen before the candidate implementation.
use super::*;
use crate::kernels::cuda::{CudaContext, testkit};

const LOCAL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/gemma4/kernels/local-gqa2.bin"
));
const GLOBAL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/gemma4/kernels/global-gqa16.bin"
));

#[test]
fn matches_the_frozen_reference_in_both_geometries() {
    if !testkit::device_present("attn_decode_gemma frozen fixtures") {
        return;
    }
    for (fixture, kv_heads, dim, window) in
        [(LOCAL, 8usize, 256usize, 16usize), (GLOBAL, 1, 512, 0)]
    {
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        let cells = 40;
        let q_bytes = 16 * dim * 4;
        let code_bytes = kv_heads * cells * dim;
        let scale_bytes = kv_heads * cells * (dim / 32) * 2;
        let expected_at = q_bytes + 2 * code_bytes + 2 * scale_bytes;
        assert_eq!(fixture.len(), expected_at + 16 * dim * 8);
        let input = testkit::upload_bytes(&ctx, &fixture[..expected_at]);
        let out = testkit::upload_filled(&ctx, 0xff, q_bytes);
        let len = testkit::upload_bytes(&ctx, &(cells as i32).to_le_bytes());
        let status = testkit::upload_bytes(&ctx, &0u32.to_le_bytes());
        ctx.attn_decode_gemma(
            &stream,
            AttnDecodeGemmaCall {
                out: out.range(0, q_bytes).expect("out"),
                status: Some(status.range(0, 4).expect("status")),
                q: input.range(0, q_bytes).expect("q"),
                k_codes: input.range(q_bytes, code_bytes).expect("k codes"),
                k_scales: input
                    .range(q_bytes + code_bytes, scale_bytes)
                    .expect("k scales"),
                v_codes: input
                    .range(q_bytes + code_bytes + scale_bytes, code_bytes)
                    .expect("v codes"),
                v_scales: input
                    .range(q_bytes + 2 * code_bytes + scale_bytes, scale_bytes)
                    .expect("v scales"),
                kv_len: len.range(0, 4).expect("length"),
                m: 1,
                n_q_head: 16,
                n_kv_head: kv_heads,
                head_dim: dim,
                cells,
                window,
                scale: 1.0,
            },
        )
        .expect("attention");
        stream.synchronize().expect("completion");
        assert_eq!(testkit::read_bytes(&ctx, &status, 4), [0, 0, 0, 0]);
        assert_eq!(
            testkit::read_bytes(&ctx, &input, expected_at),
            fixture[..expected_at]
        );
        let got = testkit::floats(&testkit::read_bytes(&ctx, &out, q_bytes));
        let expected = fixture[expected_at..].as_chunks::<8>().0;
        for (index, (actual, bytes)) in got.iter().zip(expected).enumerate() {
            let reference = f64::from_le_bytes(*bytes);
            assert!(
                (f64::from(*actual) - reference).abs() <= 3e-3 + 3e-3 * reference.abs(),
                "kv_heads={kv_heads} dim={dim} element={index}: {actual} versus {reference}"
            );
        }
    }
}

/// The wrapper refuses a head_dim outside 256 and 512 before any native call.
#[test]
fn refuses_an_unsupported_head_dim() {
    if !testkit::device_present("attn_decode_gemma dim refusal") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let q_dev = testkit::upload_f32(&ctx, &[0.0f32; 16 * 128]);
    let codes_dev = testkit::upload_filled(&ctx, 0, 8 * 4 * 128);
    let scales_dev = testkit::upload_filled(&ctx, 0, 8 * 4 * 2 * 2);
    let kv_len_dev = testkit::upload_bytes(&ctx, &0i32.to_le_bytes());
    let out_dev = testkit::upload_filled(&ctx, 0xff, 16 * 128 * 4);
    let err = ctx
        .attn_decode_gemma(
            &stream,
            AttnDecodeGemmaCall {
                out: out_dev.range(0, 16 * 128 * 4).expect("out"),
                status: None,
                q: q_dev.range(0, 16 * 128 * 4).expect("q"),
                k_codes: codes_dev.range(0, 8 * 4 * 128).expect("k"),
                k_scales: scales_dev.range(0, 8 * 4 * 2 * 2).expect("ks"),
                v_codes: codes_dev.range(0, 8 * 4 * 128).expect("v"),
                v_scales: scales_dev.range(0, 8 * 4 * 2 * 2).expect("vs"),
                kv_len: kv_len_dev.range(0, 4).expect("len"),
                m: 1,
                n_q_head: 16,
                n_kv_head: 8,
                head_dim: 128,
                cells: 4,
                window: 0,
                scale: 1.0,
            },
        )
        .expect_err("head dim refused");
    assert!(matches!(err, crate::EngError::BadRequest(_)), "{err:?}");
}

/// Zero queries make every score equal, so attention is a plain mean of V. Each lane owns
/// its own head values, so a combine that loses the lane dimension returns one value per
/// 32-value group and fails here.
#[test]
fn combines_per_lane_values_across_warps() {
    if !testkit::device_present("attn per-lane combine") {
        return;
    }
    let (heads, dim, kv_heads, cells) = (16usize, 256usize, 8usize, 40usize);
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let q = testkit::upload_f32(&ctx, &vec![0.0f32; heads * dim]);
    let codes = testkit::upload_filled(&ctx, 0, kv_heads * cells * dim);
    let mut v_codes = vec![0u8; kv_heads * cells * dim];
    for h in 0..kv_heads {
        for c in 0..cells {
            for d in 0..dim {
                v_codes[(h * cells + c) * dim + d] = ((d + c) % 7) as u8;
            }
        }
    }
    let vq = testkit::upload_bytes(&ctx, &v_codes);
    let one = [0u8, 0x3c];
    let scales_len = kv_heads * cells * (dim / 32) * 2;
    let scale_bytes: Vec<u8> = one.iter().cycle().take(scales_len).copied().collect();
    let v_scales = testkit::upload_bytes(&ctx, &scale_bytes);
    let out = testkit::upload_filled(&ctx, 0xff, heads * dim * 4);
    let len = testkit::upload_bytes(&ctx, &(cells as i32).to_le_bytes());
    ctx.attn_decode_gemma(
        &stream,
        AttnDecodeGemmaCall {
            out: out.range(0, heads * dim * 4).expect("out"),
            status: None,
            q: q.range(0, heads * dim * 4).expect("q"),
            k_codes: codes.range(0, kv_heads * cells * dim).expect("k"),
            k_scales: v_scales.range(0, scales_len).expect("ks"),
            v_codes: vq.range(0, kv_heads * cells * dim).expect("v"),
            v_scales: v_scales.range(0, scales_len).expect("vs"),
            kv_len: len.range(0, 4).expect("len"),
            m: 1,
            n_q_head: heads,
            n_kv_head: kv_heads,
            head_dim: dim,
            cells,
            window: 0,
            scale: 1.0,
        },
    )
    .expect("attention");
    stream.synchronize().expect("drain");
    let got = testkit::floats(&testkit::read_bytes(&ctx, &out, heads * dim * 4));
    for qh in 0..heads {
        for d in 0..dim {
            let mut want = 0.0f64;
            for c in 0..cells {
                want += ((d + c) % 7) as f64;
            }
            want /= cells as f64;
            let value = f64::from(got[qh * dim + d]);
            assert!(
                (value - want).abs() < 1e-3,
                "head {qh} d {d}: {value} versus {want}"
            );
        }
    }
}
