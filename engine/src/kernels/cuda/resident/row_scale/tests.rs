//! Wrapper tests of row_scale. Device runs go through gpu_run.sh only.

use super::*;
use crate::kernels::cuda::CudaContext;
use crate::kernels::cuda::testkit;

/// One row of four values scaled by two matches the host reference exactly.
#[test]
fn scales_one_row_against_the_host_reference() {
    if !testkit::device_present("row_scale basic") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let x: Vec<f32> = vec![1.0, -2.0, 0.5, 4.0];
    let x_dev = testkit::upload_f32(&ctx, &x);
    let out_dev = testkit::upload_filled(&ctx, 0xff, x.len() * 4);
    ctx.row_scale(
        &stream,
        RowScaleCall {
            out: out_dev.range(0, x.len() * 4).expect("out range"),
            x: x_dev.range(0, x.len() * 4).expect("x range"),
            scale: 2.0,
            m: 1,
            n: x.len(),
        },
    )
    .expect("row scale");
    stream.synchronize().expect("drain");
    let got = testkit::floats(&testkit::read_bytes(&ctx, &out_dev, x.len() * 4));
    for (g, w) in got.iter().zip([2.0f32, -4.0, 1.0, 8.0]) {
        assert_eq!(*g, w);
    }
}

fn batch_guard_values(rows: usize, columns: usize) -> Vec<f32> {
    (0..rows)
        .flat_map(|row| {
            (0..columns).map(move |col| match row % 3 {
                0 => 4096.0,
                1 => 65536.0,
                _ if col % 2 == 0 => 4096.0,
                _ => -4096.0,
            })
        })
        .collect()
}

/// One guard and one restore cover all routed rows, preserving earlier status bits.
#[test]
fn row_pow2_round_trip_preserves_a_block_and_earlier_status() {
    if !testkit::device_present("row_pow2 block round trip") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let baseline = ctx.live_bytes();
    {
        let (rows, columns) = (12, 512);
        let values = batch_guard_values(rows, columns);
        let x = testkit::upload_f32(&ctx, &values);
        let scaled = testkit::upload_filled(&ctx, 0xff, values.len() * 4);
        let restored = testkit::upload_filled(&ctx, 0xff, values.len() * 4);
        let scales = testkit::upload_filled(&ctx, 0xff, rows * 4);
        let flag = ctx.upload(&8u32.to_ne_bytes()).expect("flag");
        ctx.row_pow2(
            &stream,
            RowPow2Call {
                x: x.range(0, values.len() * 4).unwrap(),
                out: scaled.range(0, values.len() * 4).unwrap(),
                scales: scales.range(0, rows * 4).unwrap(),
                flag: flag.range(0, 4).unwrap(),
                rows,
                columns,
                mode: RowPow2Mode::Normalize,
            },
        )
        .expect("all-row guard");
        ctx.row_pow2(
            &stream,
            RowPow2Call {
                x: scaled.range(0, values.len() * 4).unwrap(),
                out: restored.range(0, values.len() * 4).unwrap(),
                scales: scales.range(0, rows * 4).unwrap(),
                flag: flag.range(0, 4).unwrap(),
                rows,
                columns,
                mode: RowPow2Mode::Restore,
            },
        )
        .expect("all-row restore");
        stream.synchronize().expect("completion before publication");
        let got_scales = testkit::floats(&testkit::read_bytes(&ctx, &scales, rows * 4));
        let expected: Vec<_> = (0..rows)
            .map(|row| if row % 3 == 1 { 64.0 } else { 4.0 })
            .collect();
        assert_eq!(got_scales, expected);
        let normalized = testkit::floats(&testkit::read_bytes(&ctx, &scaled, values.len() * 4));
        for row in 0..rows {
            for col in 0..columns {
                assert_eq!(
                    normalized[row * columns + col],
                    values[row * columns + col] / expected[row]
                );
            }
        }
        assert_eq!(
            testkit::floats(&testkit::read_bytes(&ctx, &restored, values.len() * 4)),
            values
        );
        assert_eq!(testkit::read_bytes(&ctx, &flag, 4), 8u32.to_ne_bytes());
    }
    assert_eq!(
        ctx.live_bytes(),
        baseline,
        "owners released after completion"
    );
}

/// Seventeen tokens x six assignments, split into ragged expert runs of 37/65.
#[test]
fn row_pow2_grouped_down_processes_a_whole_routed_block() {
    if !testkit::device_present("row_pow2 grouped down block") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let done = ctx.event().expect("completion event");
    let baseline = ctx.live_bytes();
    {
        let (rows, inner, outputs) = (17 * 6, 512, 2048);
        let packed_bytes = inner * outputs / 2;
        let scale_bytes = inner * outputs / 16;
        let a = ctx.upload(&vec![0x11; packed_bytes]).expect("expert A");
        let b = ctx.upload(&vec![0x22; packed_bytes]).expect("expert B");
        let s = ctx.upload(&vec![0x38; scale_bytes]).expect("E4M3 one");
        let wa = ctx
            .nvfp4_weight_range(
                a.range(0, packed_bytes).unwrap(),
                s.range(0, scale_bytes).unwrap(),
                2.0,
                [outputs, inner],
            )
            .expect("A view");
        let wb = ctx
            .nvfp4_weight_range(
                b.range(0, packed_bytes).unwrap(),
                s.range(0, scale_bytes).unwrap(),
                3.0,
                [outputs, inner],
            )
            .expect("B view");
        let views: Vec<_> = (0..rows)
            .map(|row| if row < 37 { wa } else { wb })
            .collect();
        let plan = ctx.nvfp4_grouped_plan(&views).expect("one block plan");
        let values = batch_guard_values(rows, inner);
        let x = testkit::upload_f32(&ctx, &values);
        let normalized = ctx.allocate(rows * inner * 4).expect("normalized block");
        let scales = ctx.allocate(rows * 4).expect("per-assignment factors");
        let projected = ctx.allocate(rows * outputs * 4).expect("scaled down block");
        let restored = ctx.allocate(rows * outputs * 4).expect("F32 down block");
        let flag = ctx.upload(&0u32.to_ne_bytes()).expect("row health");
        let mut status = ctx.nvfp4_launch_status(&stream).expect("matmul health");
        ctx.row_pow2(
            &stream,
            RowPow2Call {
                x: x.range(0, rows * inner * 4).unwrap(),
                out: normalized.range(0, rows * inner * 4).unwrap(),
                scales: scales.range(0, rows * 4).unwrap(),
                flag: flag.range(0, 4).unwrap(),
                rows,
                columns: inner,
                mode: RowPow2Mode::Normalize,
            },
        )
        .expect("one all-row normalization");
        ctx.nvfp4_grouped_enqueue(
            &stream,
            &plan,
            normalized.range(0, rows * inner * 4).unwrap(),
            projected.range(0, rows * outputs * 4).unwrap(),
            &mut status,
        )
        .expect("one grouped down enqueue");
        ctx.row_pow2(
            &stream,
            RowPow2Call {
                x: projected.range(0, rows * outputs * 4).unwrap(),
                out: restored.range(0, rows * outputs * 4).unwrap(),
                scales: scales.range(0, rows * 4).unwrap(),
                flag: flag.range(0, 4).unwrap(),
                rows,
                columns: outputs,
                mode: RowPow2Mode::Restore,
            },
        )
        .expect("one all-row F32 undo");
        stream.record(&done).expect("covers matmul and undo");
        status
            .check(&stream, &done)
            .expect("matmul complete and healthy");
        assert_eq!(testkit::read_bytes(&ctx, &flag, 4), 0u32.to_ne_bytes());
        let actual = testkit::floats(&testkit::read_bytes(&ctx, &restored, rows * outputs * 4));
        for (row, got) in actual.chunks_exact(outputs).enumerate() {
            // E2M1 codes 1/2 are 0.5/1.0; the per-expert multipliers are 2/3.
            let weight = if row < 37 { 1.0 } else { 3.0 };
            let want = (values[row * inner..(row + 1) * inner]
                .iter()
                .map(|&v| f64::from(v))
                .sum::<f64>()
                * weight) as f32;
            assert!(
                got.iter().all(|&v| v.is_finite() && v == want),
                "routed row {row}"
            );
        }
        ctx.ensure_usable()
            .expect("numeric status does not poison context");
    }
    assert_eq!(
        ctx.live_bytes(),
        baseline,
        "block owners released after completion"
    );
}

#[test]
fn row_pow2_rejects_nonfinite_rows_without_clearing_status() {
    if !testkit::device_present("row_pow2 nonfinite rows") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let values = [
        f32::NAN,
        1.0,
        2.0,
        3.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0,
        2.0,
        1024.0,
        -2048.0,
        0.0,
        -0.0,
    ];
    let x = testkit::upload_f32(&ctx, &values);
    let out = ctx.allocate(values.len() * 4).expect("out");
    let scales = ctx.allocate(3 * 4).expect("scales");
    let flag = ctx.upload(&8u32.to_ne_bytes()).expect("earlier status");
    ctx.row_pow2(
        &stream,
        RowPow2Call {
            x: x.range(0, values.len() * 4).unwrap(),
            out: out.range(0, values.len() * 4).unwrap(),
            scales: scales.range(0, 3 * 4).unwrap(),
            flag: flag.range(0, 4).unwrap(),
            rows: 3,
            columns: 4,
            mode: RowPow2Mode::Normalize,
        },
    )
    .expect("enqueue validation of rows");
    stream.synchronize().expect("completion");
    assert_eq!(testkit::read_bytes(&ctx, &flag, 4), 9u32.to_ne_bytes());
    assert_eq!(
        testkit::floats(&testkit::read_bytes(&ctx, &scales, 12)),
        [1.0, 1.0, 2.0]
    );
    let actual = testkit::floats(&testkit::read_bytes(&ctx, &out, values.len() * 4));
    let expected = [
        0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 512.0, -1024.0, 0.0, -0.0,
    ];
    assert_eq!(
        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    ctx.ensure_usable().expect("caller can reject and retry");
}

/// The wrapper refuses a non-finite scale before any native call.
#[test]
fn refuses_a_non_finite_scale() {
    if !testkit::device_present("row_scale nan refusal") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let x_dev = testkit::upload_f32(&ctx, &[1.0f32, 2.0]);
    let out_dev = testkit::upload_filled(&ctx, 0xff, 2 * 4);
    let err = ctx
        .row_scale(
            &stream,
            RowScaleCall {
                out: out_dev.range(0, 2 * 4).expect("out"),
                x: x_dev.range(0, 2 * 4).expect("x"),
                scale: f32::NAN,
                m: 1,
                n: 2,
            },
        )
        .expect_err("nan scale refused");
    assert!(matches!(err, crate::EngError::BadRequest(_)), "{err:?}");
}
