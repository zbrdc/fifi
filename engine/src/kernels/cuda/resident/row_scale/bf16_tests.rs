use super::*;
use crate::kernels::cuda::testkit;

/// Explicit RNE bit cases: ties-even, signed zero, wide normals and BF16 subnormals.
#[test]
fn row_bf16_publication_matches_rne_without_f16_narrowing() {
    if !testkit::device_present("row BF16 publication RNE") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let baseline = ctx.live_bytes();
    {
        let cases: [(u32, u16); 14] = [
            (0x47808000, 0x4780),
            (0x47818000, 0x4782),
            (0x3f808000, 0x3f80),
            (0x3f818000, 0x3f82),
            (0xbf808000, 0xbf80),
            (0xbf818000, 0xbf82),
            (0x48000000, 0x4800),
            (0xc8000000, 0xc800),
            (0x00010000, 0x0001),
            (0x00008000, 0x0000),
            (0x00018000, 0x0002),
            (0x80000000, 0x8000),
            (0x00000000, 0x0000),
            (0x7f7f0000, 0x7f7f),
        ];
        let (rows, columns) = (37, cases.len());
        let selected: Vec<_> = (0..rows)
            .flat_map(|row| (0..columns).map(move |col| cases[(row + col) % cases.len()]))
            .collect();
        let input: Vec<_> = selected
            .iter()
            .map(|&(bits, _)| f32::from_bits(bits))
            .collect();
        let x = testkit::upload_f32(&ctx, &input);
        let bytes = input.len() * 4;
        let y = testkit::upload_filled(&ctx, 0xa5, bytes + 16);
        let flag = ctx.upload(&8u32.to_ne_bytes()).unwrap();
        ctx.publish_bf16_rows(
            &stream,
            Bf16PublishCall {
                x: x.range(0, bytes).unwrap(),
                out: y.range(0, bytes).unwrap(),
                flag: flag.range(0, 4).unwrap(),
                rows,
                columns,
            },
        )
        .expect("one row-block publication");
        stream.synchronize().expect("completion before health/read");
        assert_eq!(testkit::read_bytes(&ctx, &flag, 4), 8u32.to_ne_bytes());
        let actual = testkit::floats(&testkit::read_bytes(&ctx, &y, bytes));
        for (i, (&got, &(_, want))) in actual.iter().zip(&selected).enumerate() {
            assert_eq!(got.to_bits(), u32::from(want) << 16, "element {i}");
            assert!(got.is_finite());
        }
        let all = testkit::read_bytes(&ctx, &y, bytes + 16);
        assert_eq!(&all[bytes..], &[0xa5; 16]);
        assert_eq!(
            testkit::read_bytes(&ctx, &x, bytes),
            input
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>()
        );
    }
    assert_eq!(ctx.live_bytes(), baseline);
}

#[test]
fn row_bf16_publication_checks_post_cast_overflow_and_preserves_status() {
    if !testkit::device_present("row BF16 publication health") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut input = vec![131072.0f32; 3 * 257];
    let mut expected = input.clone();
    for (index, value) in [
        (0, f32::NAN),
        (256, f32::INFINITY),
        (512, f32::NEG_INFINITY),
        (513, f32::MAX),
        (768, -f32::MAX),
        (769, f32::from_bits(0x7f7f8000)),
    ] {
        input[index] = value;
        expected[index] = 0.0;
    }
    input[100] = -131072.0;
    expected[100] = -131072.0;
    input[101] = f32::from_bits(0x7f7f0000);
    expected[101] = input[101];
    input[770] = -0.0;
    expected[770] = -0.0;
    let x = testkit::upload_f32(&ctx, &input);
    let y = ctx.allocate(input.len() * 4).unwrap();
    let flag = ctx.upload(&8u32.to_ne_bytes()).unwrap();
    ctx.publish_bf16_rows(
        &stream,
        Bf16PublishCall {
            x: x.range(0, input.len() * 4).unwrap(),
            out: y.range(0, input.len() * 4).unwrap(),
            flag: flag.range(0, 4).unwrap(),
            rows: 3,
            columns: 257,
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    assert_eq!(testkit::read_bytes(&ctx, &flag, 4), 9u32.to_ne_bytes());
    assert_eq!(
        testkit::read_bytes(&ctx, &y, expected.len() * 4),
        expected
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>()
    );
    ctx.ensure_usable().unwrap();
}

#[test]
fn row_bf16_publication_finite_overflow_alone_sets_health() {
    if !testkit::device_present("row BF16 finite-only overflow") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let input = [f32::MAX, -f32::MAX, f32::from_bits(0x7f7f8000), 131072.0];
    assert!(input.iter().all(|x| x.is_finite()));
    let x = testkit::upload_f32(&ctx, &input);
    let y = ctx.allocate(16).unwrap();
    let flag = ctx.upload(&8u32.to_ne_bytes()).unwrap();
    ctx.publish_bf16_rows(
        &stream,
        Bf16PublishCall {
            x: x.range(0, 16).unwrap(),
            out: y.range(0, 16).unwrap(),
            flag: flag.range(0, 4).unwrap(),
            rows: 2,
            columns: 2,
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    assert_eq!(testkit::read_bytes(&ctx, &flag, 4), 9u32.to_ne_bytes());
    let expected = [0.0f32, 0.0, 0.0, 131072.0];
    assert_eq!(
        testkit::read_bytes(&ctx, &y, 16),
        expected
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>()
    );
    ctx.ensure_usable().unwrap();
}

#[test]
fn row_bf16_publication_refuses_invalid_regions_without_mutation() {
    if !testkit::device_present("row BF16 publication preflight") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let foreign = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let other_stream = foreign.stream().unwrap();
    let x = testkit::upload_f32(&ctx, &[1.0; 9]);
    let foreign_x = testkit::upload_f32(&foreign, &[1.0; 8]);
    let y = testkit::upload_filled(&ctx, 0xa5, 36);
    let flag = ctx.upload(&8u32.to_ne_bytes()).unwrap();
    for case in 0..11 {
        let mut call = Bf16PublishCall {
            x: x.range(0, 32).unwrap(),
            out: y.range(0, 32).unwrap(),
            flag: flag.range(0, 4).unwrap(),
            rows: 2,
            columns: 4,
        };
        match case {
            0 => call.rows = 0,
            1 => call.columns = 0,
            2 => call.x = x.range(0, 31).unwrap(),
            3 => call.out = y.range(0, 31).unwrap(),
            4 => call.flag = flag.range(0, 3).unwrap(),
            5 => call.x = x.range(1, 32).unwrap(),
            6 => call.out = y.range(1, 32).unwrap(),
            7 => call.out = x.range(0, 32).unwrap(),
            8 => call.flag = x.range(0, 4).unwrap(),
            9 => call.x = foreign_x.range(0, 32).unwrap(),
            10 => {}
            _ => unreachable!(),
        }
        let selected = if case == 10 { &other_stream } else { &stream };
        let error = ctx
            .publish_bf16_rows(selected, call)
            .expect_err("preflight refusal");
        assert!(
            matches!(error, EngError::BadRequest(_)),
            "case {case}: {error:?}"
        );
        ctx.ensure_usable().unwrap();
    }
    stream.synchronize().unwrap();
    assert_eq!(
        testkit::read_bytes(&ctx, &x, 36),
        [1.0f32; 9]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>()
    );
    assert_eq!(testkit::read_bytes(&ctx, &y, 36), vec![0xa5; 36]);
    assert_eq!(testkit::read_bytes(&ctx, &flag, 4), 8u32.to_ne_bytes());
}
