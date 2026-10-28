//! Softcap outputs frozen independently before the candidate implementation.
use super::*;
use crate::kernels::cuda::{CudaContext, testkit};

const CASE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/gemma4/kernels/softcap-frozen.bin"
));

/// Parses the fixture: u32 case count, then per case f32 cap, u32 count, inputs, expected.
fn cases() -> Vec<(f32, Vec<f32>, Vec<f64>)> {
    let mut rest = CASE;
    let mut count = [0u8; 4];
    count.copy_from_slice(&rest[..4]);
    rest = &rest[4..];
    let mut parsed = Vec::with_capacity(u32::from_le_bytes(count) as usize);
    for _ in 0..u32::from_le_bytes(count) {
        let mut cap = [0u8; 4];
        cap.copy_from_slice(&rest[..4]);
        rest = &rest[4..];
        let mut header = [0u8; 4];
        header.copy_from_slice(&rest[..4]);
        rest = &rest[4..];
        let values = u32::from_le_bytes(header) as usize;
        let inputs: Vec<f32> = rest[..values * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        rest = &rest[values * 4..];
        let expected: Vec<f64> = rest[..values * 8]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect();
        rest = &rest[values * 8..];
        parsed.push((f32::from_le_bytes(cap), inputs, expected));
    }
    assert!(rest.is_empty(), "trailing fixture bytes");
    parsed
}

#[test]
fn matches_the_frozen_reference() {
    if !testkit::device_present("softcap frozen fixture") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    for (cap, inputs, expected) in cases() {
        let input: Vec<f32> = inputs.iter().copied().cycle().take(32).collect();
        let reference: Vec<f64> = expected.iter().copied().cycle().take(32).collect();
        let x = testkit::upload_f32(&ctx, &input);
        let out = testkit::upload_filled(&ctx, 0xff, 128);
        ctx.softcap(
            &stream,
            SoftcapCall {
                out: out.range(0, 128).expect("out"),
                x: x.range(0, 128).expect("x"),
                cap,
                n: 32,
            },
        )
        .expect("softcap");
        stream.synchronize().expect("completion");
        let actual = testkit::floats(&testkit::read_bytes(&ctx, &out, 128));
        for (index, value) in actual.iter().enumerate() {
            let want = reference[index];
            assert!(
                (f64::from(*value) - want).abs() <= 2e-6 + 2e-5 * want.abs(),
                "element={index}: {value} versus {want}"
            );
        }
    }
}

/// The wrapper refuses a cap that is not above zero before any native call.
#[test]
fn refuses_a_cap_that_is_not_above_zero() {
    if !testkit::device_present("softcap cap refusal") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let x_dev = testkit::upload_f32(&ctx, &[1.0f32; 32]);
    let out_dev = testkit::upload_filled(&ctx, 0xff, 128);
    let err = ctx
        .softcap(
            &stream,
            SoftcapCall {
                out: out_dev.range(0, 128).expect("out"),
                x: x_dev.range(0, 128).expect("x"),
                cap: 0.0,
                n: 32,
            },
        )
        .expect_err("cap refused");
    assert!(matches!(err, crate::EngError::BadRequest(_)), "{err:?}");
}

/// The wrapper refuses a width that is not a multiple of 32 before any native call.
#[test]
fn refuses_a_width_that_is_not_a_multiple_of_32() {
    if !testkit::device_present("softcap width refusal") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let x_dev = testkit::upload_f32(&ctx, &[1.0f32; 33]);
    let out_dev = testkit::upload_filled(&ctx, 0xff, 33 * 4);
    let err = ctx
        .softcap(
            &stream,
            SoftcapCall {
                out: out_dev.range(0, 33 * 4).expect("out"),
                x: x_dev.range(0, 33 * 4).expect("x"),
                cap: 30.0,
                n: 33,
            },
        )
        .expect_err("width refused");
    assert!(matches!(err, crate::EngError::BadRequest(_)), "{err:?}");
}
