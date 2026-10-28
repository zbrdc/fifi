//! Sum and refusal tests of row_add. Device runs go through gpu_run.sh only.
use super::*;
use crate::kernels::cuda::{CudaContext, testkit};

/// The sum of two rows matches the host reference bit for bit.
#[test]
fn adds_two_rows_against_the_host_reference() {
    if !testkit::device_present("row_add basic") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let x: Vec<f32> = (0..64).map(|i| i as f32 * 0.5 - 7.0).collect();
    let y: Vec<f32> = (0..64).map(|i| 3.25 - i as f32 * 0.25).collect();
    let x_dev = testkit::upload_f32(&ctx, &x);
    let y_dev = testkit::upload_f32(&ctx, &y);
    let out_dev = testkit::upload_filled(&ctx, 0xff, x.len() * 4);
    ctx.row_add(
        &stream,
        RowAddCall {
            out: out_dev.range(0, x.len() * 4).expect("out"),
            x: x_dev.range(0, x.len() * 4).expect("x"),
            y: y_dev.range(0, x.len() * 4).expect("y"),
            n: x.len(),
        },
    )
    .expect("row add");
    stream.synchronize().expect("drain");
    let got = testkit::floats(&testkit::read_bytes(&ctx, &out_dev, x.len() * 4));
    for (index, value) in got.iter().enumerate() {
        assert_eq!(*value, x[index] + y[index], "element={index}");
    }
}

/// The wrapper refuses an out that overlaps x before any native call.
#[test]
fn refuses_an_out_that_overlaps_an_input() {
    if !testkit::device_present("row_add alias refusal") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let x_dev = testkit::upload_f32(&ctx, &[1.0f32; 32]);
    let y_dev = testkit::upload_f32(&ctx, &[2.0f32; 32]);
    let range = x_dev.range(0, 128).expect("x");
    let err = ctx
        .row_add(
            &stream,
            RowAddCall {
                out: range,
                x: range,
                y: y_dev.range(0, 128).expect("y"),
                n: 32,
            },
        )
        .expect_err("alias refused");
    assert!(matches!(err, crate::EngError::BadRequest(_)), "{err:?}");
}
