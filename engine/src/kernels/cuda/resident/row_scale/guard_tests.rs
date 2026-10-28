use super::*;
use crate::kernels::cuda::testkit;

#[test]
fn row_pow2_normalizes_exponent_boundaries_and_extremes() {
    if !testkit::device_present("row pow2 exponent boundaries") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let maxima = [
        0.0f32,
        1.0,
        f32::from_bits(1024.0f32.to_bits() - 1),
        1024.0,
        f32::from_bits(1024.0f32.to_bits() + 1),
        2048.0,
        f32::from_bits(2048.0f32.to_bits() + 1),
        f32::MAX,
    ];
    let expected = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 4.0, f32::from_bits(245 << 23)];
    let columns = 129;
    let mut values = Vec::new();
    for (row, &maximum) in maxima.iter().enumerate() {
        let mut v: Vec<_> = (0..columns)
            .map(|col| maximum * [0.125, -0.25, 0.5][col % 3])
            .collect();
        v[[0, 1, 64, 127, 128, 65, 17, 128][row]] = maximum;
        v[2] = -0.0;
        values.extend(v);
    }
    let x = testkit::upload_f32(&ctx, &values);
    let y = ctx.allocate(values.len() * 4).unwrap();
    let restored = ctx.allocate(values.len() * 4).unwrap();
    let scales = ctx.allocate(maxima.len() * 4).unwrap();
    let flag = ctx.upload(&8u32.to_ne_bytes()).unwrap();
    for (mode, input, out) in [
        (RowPow2Mode::Normalize, &x, &y),
        (RowPow2Mode::Restore, &y, &restored),
    ] {
        ctx.row_pow2(
            &stream,
            RowPow2Call {
                x: input.range(0, values.len() * 4).unwrap(),
                out: out.range(0, values.len() * 4).unwrap(),
                scales: scales.range(0, maxima.len() * 4).unwrap(),
                flag: flag.range(0, 4).unwrap(),
                rows: maxima.len(),
                columns,
                mode,
            },
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    assert_eq!(
        testkit::floats(&testkit::read_bytes(&ctx, &scales, maxima.len() * 4)),
        expected
    );
    let actual = testkit::floats(&testkit::read_bytes(&ctx, &y, values.len() * 4));
    for (i, &got) in actual.iter().enumerate() {
        let want = (f64::from(values[i]) / f64::from(expected[i / columns])) as f32;
        assert_eq!(got.to_bits(), want.to_bits(), "normalized element {i}");
        assert!(got.abs() <= 1024.0);
    }
    assert_eq!(
        testkit::read_bytes(&ctx, &restored, values.len() * 4),
        values
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>()
    );
    assert_eq!(testkit::read_bytes(&ctx, &flag, 4), 8u32.to_ne_bytes());
}

#[test]
fn row_pow2_restore_refuses_invalid_data_and_overflow() {
    if !testkit::device_present("row pow2 restore refusals") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let max_scale = f32::from_bits(245 << 23);
    let cases = [
        (0.0, [1.0, -1.0, 0.0], [0.0; 3], 9u32),
        (-2.0, [1.0, -1.0, 0.0], [0.0; 3], 9),
        (3.0, [1.0, -1.0, 0.0], [0.0; 3], 9),
        (f32::NAN, [1.0, -1.0, 0.0], [0.0; 3], 9),
        (f32::INFINITY, [1.0, -1.0, 0.0], [0.0; 3], 9),
        (f32::from_bits(246 << 23), [1.0, -1.0, 0.0], [0.0; 3], 9),
        (2.0, [f32::MAX, -f32::MAX, 1.0], [0.0, 0.0, 2.0], 9),
        (
            1.0,
            [f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            [0.0; 3],
            9,
        ),
        (max_scale, [1.0, -1.0, 0.0], [max_scale, -max_scale, 0.0], 8),
        (
            1.0,
            [f32::MIN_POSITIVE, -0.0, 1.0],
            [f32::MIN_POSITIVE, -0.0, 1.0],
            8,
        ),
    ];
    for (factor, values, want, health) in cases {
        let x = testkit::upload_f32(&ctx, &values);
        let y = ctx.allocate(12).unwrap();
        let scales = testkit::upload_f32(&ctx, &[factor]);
        let flag = ctx.upload(&8u32.to_ne_bytes()).unwrap();
        ctx.row_pow2(
            &stream,
            RowPow2Call {
                x: x.range(0, 12).unwrap(),
                out: y.range(0, 12).unwrap(),
                scales: scales.range(0, 4).unwrap(),
                flag: flag.range(0, 4).unwrap(),
                rows: 1,
                columns: 3,
                mode: RowPow2Mode::Restore,
            },
        )
        .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(
            testkit::read_bytes(&ctx, &flag, 4),
            health.to_ne_bytes(),
            "factor {factor:?}"
        );
        assert_eq!(
            testkit::read_bytes(&ctx, &y, 12),
            want.iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
            "factor {factor:?}"
        );
        assert_eq!(testkit::read_bytes(&ctx, &scales, 4), factor.to_ne_bytes());
        ctx.ensure_usable().unwrap();
    }
}

#[test]
fn row_pow2_checked_refusals_preserve_buffers_and_context() {
    if !testkit::device_present("row pow2 checked refusals") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let foreign = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let foreign_stream = foreign.stream().unwrap();
    let x = testkit::upload_f32(&ctx, &[1.0; 8]);
    let y = testkit::upload_filled(&ctx, 0xa5, 36);
    let scales = testkit::upload_f32(&ctx, &[1.0; 2]);
    let flag = ctx.upload(&8u32.to_ne_bytes()).unwrap();
    let foreign_x = testkit::upload_f32(&foreign, &[1.0; 8]);
    for case in 0..10 {
        let mut call = RowPow2Call {
            x: x.range(0, 32).unwrap(),
            out: y.range(0, 32).unwrap(),
            scales: scales.range(0, 8).unwrap(),
            flag: flag.range(0, 4).unwrap(),
            rows: 2,
            columns: 4,
            mode: RowPow2Mode::Normalize,
        };
        match case {
            0 => call.x = x.range(0, 31).unwrap(),
            1 => call.out = y.range(0, 31).unwrap(),
            2 => call.scales = scales.range(0, 7).unwrap(),
            3 => call.flag = flag.range(0, 3).unwrap(),
            4 => call.out = y.range(1, 32).unwrap(),
            5 => call.out = x.range(0, 32).unwrap(),
            6 => call.scales = x.range(0, 8).unwrap(),
            7 => call.flag = scales.range(0, 4).unwrap(),
            8 => call.x = foreign_x.range(0, 32).unwrap(),
            9 => {}
            _ => unreachable!(),
        }
        let selected = if case == 9 { &foreign_stream } else { &stream };
        let err = ctx
            .row_pow2(selected, call)
            .expect_err("checked preflight refusal");
        assert!(
            matches!(err, EngError::BadRequest(_)),
            "case {case}: {err:?}"
        );
        ctx.ensure_usable().unwrap();
    }
    stream.synchronize().unwrap();
    assert_eq!(
        testkit::read_bytes(&ctx, &x, 32),
        [1.0f32; 8]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>()
    );
    assert_eq!(testkit::read_bytes(&ctx, &y, 36), vec![0xa5; 36]);
    assert_eq!(
        testkit::read_bytes(&ctx, &scales, 8),
        [1.0f32; 2]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>()
    );
    assert_eq!(testkit::read_bytes(&ctx, &flag, 4), 8u32.to_ne_bytes());
}
