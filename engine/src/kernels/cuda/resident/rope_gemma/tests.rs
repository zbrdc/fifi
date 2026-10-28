//! Split-half rotations checked against independent frozen F64 outputs.
use super::*;
use crate::kernels::cuda::{CudaContext, testkit};

const LOCAL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/gemma4/kernels/rope-local-8191.bin"
));
const GLOBAL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/gemma4/kernels/rope-global-zero-8191.bin"
));

#[test]
fn matches_the_frozen_reference_at_position_8191() {
    if !testkit::device_present("rope_gemma frozen fixtures") {
        return;
    }
    for (fixture, dim, global, kv_heads) in
        [(LOCAL, 256usize, false, 8usize), (GLOBAL, 512, true, 1)]
    {
        assert_eq!(fixture.len(), dim * 12);
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        let single = &fixture[..dim * 4];
        let q = testkit::upload_bytes(&ctx, &single.repeat(16));
        let k = testkit::upload_bytes(&ctx, &single.repeat(kv_heads));
        let position = testkit::upload_bytes(&ctx, &8191i32.to_le_bytes());
        ctx.rope_gemma(
            &stream,
            RopeGemmaCall {
                q: q.range(0, dim * 16 * 4).expect("q"),
                k: k.range(0, dim * kv_heads * 4).expect("k"),
                pos: position.range(0, 4).expect("position"),
                factors: None,
                rows_q: 1,
                rows_k: 1,
                n_q_head: 16,
                n_kv_head: kv_heads,
                head_dim: dim,
                global_mode: global,
            },
        )
        .expect("rotation");
        stream.synchronize().expect("completion");
        let expected = fixture[dim * 4..].as_chunks::<8>().0;
        for (name, buffer, heads) in [("q", &q, 16), ("k", &k, kv_heads)] {
            let actual = testkit::floats(&testkit::read_bytes(&ctx, buffer, dim * heads * 4));
            for (index, value) in actual.iter().enumerate() {
                let reference = f64::from_le_bytes(expected[index % dim]);
                assert!(
                    (f64::from(*value) - reference).abs() <= 2e-6 + 2e-5 * reference.abs(),
                    "{name} dim={dim} element={index}: {value} versus {reference}"
                );
            }
        }
    }
}

/// The wrapper refuses a head_dim that does not match the mode.
#[test]
fn refuses_a_head_dim_that_does_not_match_the_mode() {
    if !testkit::device_present("rope_gemma dim refusal") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let q_dev = testkit::upload_f32(&ctx, &[0.0f32; 16 * 256]);
    let k_dev = testkit::upload_f32(&ctx, &[0.0f32; 8 * 256]);
    let pos_dev = testkit::upload_bytes(&ctx, &0i32.to_le_bytes());
    let err = ctx
        .rope_gemma(
            &stream,
            RopeGemmaCall {
                q: q_dev.range(0, 16 * 256 * 4).expect("q"),
                k: k_dev.range(0, 8 * 256 * 4).expect("k"),
                pos: pos_dev.range(0, 4).expect("pos"),
                factors: None,
                rows_q: 1,
                rows_k: 1,
                n_q_head: 16,
                n_kv_head: 8,
                head_dim: 256,
                global_mode: true,
            },
        )
        .expect_err("mode mismatch refused");
    assert!(matches!(err, crate::EngError::BadRequest(_)), "{err:?}");
}

#[test]
fn batch_rows_use_consecutive_positions() {
    if !testkit::device_present("rope_gemma consecutive rows") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let rows = 8;
    for (fixture, dim, global, heads) in [(LOCAL, 256usize, false, 8usize), (GLOBAL, 512, true, 1)]
    {
        let input = &fixture[..dim * 4];
        let q = testkit::upload_bytes(&ctx, &input.repeat(16 * rows));
        let k = testkit::upload_bytes(&ctx, &input.repeat(heads * rows));
        let pos = testkit::upload_bytes(&ctx, &8184i32.to_le_bytes());
        ctx.rope_gemma(
            &stream,
            RopeGemmaCall {
                q: q.range(0, q.bytes()).expect("q"),
                k: k.range(0, k.bytes()).expect("k"),
                pos: pos.range(0, 4).expect("pos"),
                factors: None,
                rows_q: rows,
                rows_k: rows,
                n_q_head: 16,
                n_kv_head: heads,
                head_dim: dim,
                global_mode: global,
            },
        )
        .expect("batch rotation");
        stream.synchronize().expect("batch drain");
        let batch_q = testkit::read_bytes(&ctx, &q, q.bytes());
        let batch_k = testkit::read_bytes(&ctx, &k, k.bytes());
        for row in 0..rows {
            let single_q = testkit::upload_bytes(&ctx, &input.repeat(16));
            let single_k = testkit::upload_bytes(&ctx, &input.repeat(heads));
            let pos = testkit::upload_bytes(&ctx, &((8184 + row) as i32).to_le_bytes());
            ctx.rope_gemma(
                &stream,
                RopeGemmaCall {
                    q: single_q.range(0, single_q.bytes()).expect("q"),
                    k: single_k.range(0, single_k.bytes()).expect("k"),
                    pos: pos.range(0, 4).expect("pos"),
                    factors: None,
                    rows_q: 1,
                    rows_k: 1,
                    n_q_head: 16,
                    n_kv_head: heads,
                    head_dim: dim,
                    global_mode: global,
                },
            )
            .expect("single rotation");
            stream.synchronize().expect("single drain");
            for (batch, single) in [(&batch_q, &single_q), (&batch_k, &single_k)] {
                let bytes = single.bytes();
                assert_eq!(
                    &batch[row * bytes..(row + 1) * bytes],
                    testkit::read_bytes(&ctx, single, bytes),
                    "dim {dim} row {row}"
                );
            }
        }
    }
}
