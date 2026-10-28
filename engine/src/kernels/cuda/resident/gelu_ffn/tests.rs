//! GELU-tanh outputs frozen independently before the candidate implementation.
use super::*;
use crate::kernels::cuda::{CudaContext, testkit};

const CASE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/gemma4/kernels/gelu-tanh-gate-up.bin"
));

#[test]
fn matches_the_frozen_reference() {
    if !testkit::device_present("gelu_ffn frozen fixture") {
        return;
    }
    const VALUES: usize = 11;
    assert_eq!(CASE.len(), VALUES * 16);
    let gate = testkit::floats(&CASE[..VALUES * 4]);
    let up = testkit::floats(&CASE[VALUES * 4..VALUES * 8]);
    let input: Vec<f32> = gate.iter().copied().cycle().take(32).collect();
    let multiplier: Vec<f32> = up.iter().copied().cycle().take(32).collect();
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let g = testkit::upload_f32(&ctx, &input);
    let u = testkit::upload_f32(&ctx, &multiplier);
    let out = testkit::upload_filled(&ctx, 0xff, 128);
    ctx.gelu_ffn(
        &stream,
        GeluFfnCall {
            out: out.range(0, 128).expect("out"),
            gate: g.range(0, 128).expect("gate"),
            up: u.range(0, 128).expect("up"),
            m: 1,
            n: 32,
        },
    )
    .expect("GELU");
    stream.synchronize().expect("completion");
    let actual = testkit::floats(&testkit::read_bytes(&ctx, &out, 128));
    let expected = CASE[VALUES * 8..].as_chunks::<8>().0;
    for (index, value) in actual.iter().enumerate() {
        let reference = f64::from_le_bytes(expected[index % VALUES]);
        assert!(
            (f64::from(*value) - reference).abs() <= 2e-6 + 2e-5 * reference.abs(),
            "element={index}: {value} versus {reference}"
        );
    }
}

/// The wrapper refuses a width that is not a multiple of 32 before any native call.
#[test]
fn refuses_a_width_that_is_not_a_multiple_of_32() {
    if !testkit::device_present("gelu_ffn width refusal") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let gate_dev = testkit::upload_f32(&ctx, &[0.0f32; 33]);
    let up_dev = testkit::upload_f32(&ctx, &[0.0f32; 33]);
    let out_dev = testkit::upload_filled(&ctx, 0xff, 33 * 4);
    let err = ctx
        .gelu_ffn(
            &stream,
            GeluFfnCall {
                out: out_dev.range(0, 33 * 4).expect("out"),
                gate: gate_dev.range(0, 33 * 4).expect("gate"),
                up: up_dev.range(0, 33 * 4).expect("up"),
                m: 1,
                n: 33,
            },
        )
        .expect_err("width refused");
    assert!(matches!(err, crate::EngError::BadRequest(_)), "{err:?}");
}
