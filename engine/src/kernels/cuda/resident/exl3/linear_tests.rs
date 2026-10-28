use super::Exl3LinearCall;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, testkit};
use crate::tensor::{Dtype, Tensor};
use crate::weights::{
    Tier, WeightSource,
    exl3::{Codebook, Config, Matrix},
};
use half::f16;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

struct Components(HashMap<String, Tensor>);
impl WeightSource for Components {
    fn tensor(&self, key: &str) -> Option<Tensor> {
        self.0.get(key).cloned()
    }
    fn resident_bytes(&self) -> u64 {
        self.0.values().map(Tensor::byte_len).sum()
    }
    fn tier(&self) -> Tier {
        Tier::Resident
    }
}
fn fixture(subnormal: bool, book: Codebook, bits_twice: u8, shape: (usize, usize)) -> Matrix {
    let (k, n) = shape;
    let mut rng = testkit::Rng::new(0x4558_4c33);
    let tensor = |dtype, shape, bytes| Tensor::try_from_host(dtype, shape, bytes).unwrap();
    let words = usize::from(bits_twice) * 8;
    let packed = (0..(k / 16) * (n / 16) * words)
        .flat_map(|_| (rng.next_u64() as u16).to_le_bytes())
        .collect();
    let scales = |count, low: bool| {
        (0..count)
            .flat_map(|i| {
                let value = if low {
                    f16::from_bits(1 + (i % 31) as u16)
                } else {
                    f16::from_f32(0.5 + (i % 7) as f32 / 8.0)
                };
                value.to_bits().to_le_bytes()
            })
            .collect()
    };
    let mut source = Components(
        [
            (
                "m.trellis".into(),
                tensor(
                    Dtype::I16,
                    vec![(k / 16) as u32, (n / 16) as u32, words as u32],
                    packed,
                ),
            ),
            (
                "m.suh".into(),
                tensor(Dtype::F16, vec![k as u32], scales(k, subnormal)),
            ),
            (
                "m.svh".into(),
                tensor(Dtype::F16, vec![n as u32], scales(n, false)),
            ),
        ]
        .into(),
    );
    let marker = match book {
        Codebook::ThreeInst => None,
        Codebook::Mcg => Some(("mcg", 0xcbac_1fedu32)),
        Codebook::Mul1 => Some(("mul1", 0x83dc_d12du32)),
    };
    if let Some((suffix, word)) = marker {
        source.0.insert(
            format!("m.{suffix}"),
            tensor(Dtype::I32, vec![], word.to_le_bytes().to_vec()),
        );
    }
    let config =
        Config::from_json(br#"{"quantization_config":{"quant_method":"exl3","version":"1.6.2"}}"#)
            .unwrap();
    Matrix::load(&source, "m", &config).unwrap()
}
fn upload_scales(ctx: &CudaContext, tensor: &Tensor) -> DeviceBuffer {
    let values: Vec<_> = tensor
        .host_bytes()
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| f16::from_bits(u16::from_le_bytes(*p)).to_f32())
        .collect();
    testkit::upload_f32(ctx, &values)
}

fn reference_rotate(values: &[f64]) -> Vec<f64> {
    let mut output = vec![0.0; values.len()];
    for (input, output) in values
        .as_chunks::<128>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<128>().0.iter_mut())
    {
        for (row, result) in output.iter_mut().enumerate() {
            for (col, value) in input.iter().enumerate() {
                let sign = if (row & col).count_ones().is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };
                *result += sign * value;
            }
            *result /= 128.0f64.sqrt();
        }
    }
    output
}

fn reference_project(
    xh: &[f64],
    weights: &[u16],
    shape: (usize, usize),
    span: std::ops::Range<usize>,
) -> Vec<f64> {
    let (k, n) = shape;
    let mut output = vec![0.0; xh.len() / k * n];
    for (x, y) in xh.chunks_exact(k).zip(output.chunks_exact_mut(n)) {
        for i in span.clone() {
            for (y, bits) in y.iter_mut().zip(&weights[i * n..(i + 1) * n]) {
                *y += x[i] * f16::from_bits(*bits).to_f64();
            }
        }
    }
    output
}

pub(super) fn rounded_mma_inputs(actual: &[u8], expected: &[f64]) -> Vec<f64> {
    assert_eq!(actual.len(), expected.len() * 4);
    assert_stage_close(actual, expected, "R1 independent F32 input rotation");
    actual
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f16::from_f32(f32::from_le_bytes(*bytes)).to_f64())
        .collect()
}

fn reference_output(projection: &[f64], scales: &Tensor) -> Vec<f64> {
    let scales: Vec<_> = scales
        .host_bytes()
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| f16::from_bits(u16::from_le_bytes(*bytes)).to_f64())
        .collect();
    let mut output = reference_rotate(projection);
    for (index, value) in output.iter_mut().enumerate() {
        *value *= scales[index % scales.len()];
    }
    output
}

fn assert_stock_quality(matrix: &Matrix, input: &[f32], expected: &[f64], actual: &[u8]) {
    let input_bytes: Vec<_> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let expected_bytes: Vec<_> = expected
        .iter()
        .cycle()
        .take(actual.len() / 4)
        .flat_map(|v| v.to_le_bytes())
        .collect();
    // R2 constants belong only to the captured first fixture, never to arbitrary inputs.
    for (bytes, hash) in [
        (
            input_bytes.as_slice(),
            "6f87b6b39bb4cb7e33161110fdec094f5aa3fc50df3163e07083754c5e3f0acd",
        ),
        (
            matrix.trellis().host_bytes().unwrap(),
            "3eda98429914367a550b7592dd3ba506c08f074088a6ff55260a3bc977987d26",
        ),
        (
            matrix.suh().host_bytes().unwrap(),
            "c3c83348fadb5df2fb05716f08e96dfaad593b8c1481b118403fb8359613305d",
        ),
        (
            matrix.svh().host_bytes().unwrap(),
            "c3c83348fadb5df2fb05716f08e96dfaad593b8c1481b118403fb8359613305d",
        ),
        (
            expected_bytes.as_slice(),
            "29659a83ed5d6cd13259fe34ad3eab5de509ac0114d826a439c123fd99a62a88",
        ),
    ] {
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            hash,
            "R2 fixture binding"
        );
    }
    let values: Vec<_> = actual
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f64::from(f32::from_le_bytes(*bytes)))
        .collect();
    let mut error = 0.0;
    let mut norm = 0.0;
    let mut worst = 0.0f64;
    for (index, row) in values.as_chunks::<384>().0.iter().enumerate() {
        let target = &expected[(index % 8) * 384..(index % 8 + 1) * 384];
        let row_error = row
            .iter()
            .zip(target)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>();
        let row_norm = target.iter().map(|x| x.powi(2)).sum::<f64>();
        assert!(row_norm > 0.0 && row.iter().all(|x| x.is_finite()));
        error += row_error;
        norm += row_norm;
        worst = worst.max((row_error / row_norm).sqrt());
    }
    // Both fresh stocks repeat; optional F32 output gives the stricter recorded k=1 bound.
    assert!((error / norm).sqrt() <= 0.00063165465646091, "R2 global");
    assert!(worst <= 0.0006745546121254832, "R2 worst row");
}

fn assert_stage_close(actual: &[u8], expected: &[f64], stage: &str) {
    let mut error = 0.0;
    let mut norm = 0.0;
    for (index, bytes) in actual.as_chunks::<4>().0.iter().enumerate() {
        let got = f64::from(f32::from_le_bytes(*bytes));
        let want = expected[index % expected.len()];
        assert!(got.is_finite(), "{stage}/index{index}");
        error += (got - want).powi(2);
        norm += want.powi(2);
    }
    assert!(
        if norm == 0.0 {
            error == 0.0
        } else {
            (error / norm).sqrt() < 1e-5
        },
        "{stage}/error{error}/norm{norm}"
    );
}

fn constant_projection(
    ctx: &CudaContext,
    stream: &DeviceStream,
    value: f32,
    bias: Option<&[f32]>,
    geometry: (usize, u32),
) -> (Vec<f32>, u32, Vec<u8>) {
    let (rows, split_k) = geometry;
    let project = if rows == 1 {
        CudaContext::exl3_linear
    } else {
        CudaContext::exl3_linear_midrows
    };
    let matrix = fixture(false, Codebook::Mul1, 3, (256, 128));
    let bytes = matrix.trellis().host_bytes().unwrap();
    let packed = testkit::upload_bytes(ctx, bytes);
    let suh = upload_scales(ctx, matrix.suh());
    let svh = upload_scales(ctx, matrix.svh());
    let bias = bias.map(|v| testkit::upload_f32(ctx, v));
    let x = testkit::upload_f32(ctx, &vec![value; rows * 256]);
    let xh = testkit::upload_filled(ctx, 0xff, rows * 256 * 4);
    let banks = if rows <= 8 { split_k as usize - 1 } else { 1 };
    let z_bytes = rows * 128 * 4 * banks;
    let z = testkit::upload_filled(ctx, 0xff, z_bytes);
    let out = testkit::upload_filled(ctx, 0xff, rows * 128 * 4);
    let flag = testkit::upload_bytes(ctx, &8u32.to_le_bytes());
    project(
        ctx,
        stream,
        Exl3LinearCall {
            trellis: packed.range(0, bytes.len()).unwrap(),
            suh: suh.range(0, 256 * 4).unwrap(),
            svh: svh.range(0, 128 * 4).unwrap(),
            bias: bias.as_ref().map(|b| b.range(0, 128 * 4).unwrap()),
            x: x.range(0, rows * 256 * 4).unwrap(),
            xh: xh.range(0, rows * 256 * 4).unwrap(),
            z: z.range(0, z_bytes).unwrap(),
            out: out.range(0, rows * 128 * 4).unwrap(),
            flag: flag.range(0, 4).unwrap(),
            rows,
            split_k,
            k: 256,
            n: 128,
            bits_twice: 3,
            book: matrix.codebook(),
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let output = testkit::read_bytes(ctx, &out, rows * 128 * 4)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let flag = u32::from_le_bytes(testkit::read_bytes(ctx, &flag, 4).try_into().unwrap());
    (output, flag, testkit::read_bytes(ctx, &xh, 256 * 4))
}

fn staged_guard_cases(ctx: &CudaContext, stream: &DeviceStream, geometry: (usize, u32)) {
    let (rows, split_k) = geometry;
    let project = if rows == 1 {
        CudaContext::exl3_linear
    } else {
        CudaContext::exl3_linear_midrows
    };
    let packed = testkit::upload_filled(ctx, 0, 6144);
    let suh = testkit::upload_f32(ctx, &[1.0; 256]);
    let xh = testkit::upload_filled(ctx, 0xff, rows * 256 * 4);
    let banks = if rows <= 8 { split_k as usize - 1 } else { 1 };
    let z_bytes = rows * 128 * 4 * banks;
    let z = testkit::upload_filled(ctx, 0xff, z_bytes);
    let out = testkit::upload_filled(ctx, 0xff, rows * 128 * 4);
    // One-hot H128 inputs stay finite; zero trellis yields a constant -3.453125 weight.
    for (stage, value, scale, bias, initial_flag, want_flag) in [
        ("projection", f32::MAX / 4.0, 1.0, 0.0, 8u32, 9u32),
        ("merge", f32::MAX / 60.0, 1.0, 0.0, 8, 9),
        ("scale", 1e20, 1e20, 0.0, 8, 9),
        ("bias", -f32::MAX / 20000.0, 1.0, f32::MAX, 8, 9),
        ("already flagged", 0.0, 1.0, 0.0, 9, 9),
        ("clean", 0.0, 1.0, 0.0, 8, 8),
    ] {
        let mut input = [0.0; 256];
        input[0] = value;
        input[128] = value;
        let x = testkit::upload_f32(ctx, &input.repeat(rows));
        let svh = testkit::upload_f32(ctx, &[scale; 128]);
        let bias = testkit::upload_f32(ctx, &[bias; 128]);
        let flag = testkit::upload_bytes(ctx, &initial_flag.to_le_bytes());
        project(
            ctx,
            stream,
            Exl3LinearCall {
                trellis: packed.range(0, 6144).unwrap(),
                suh: suh.range(0, 1024).unwrap(),
                svh: svh.range(0, 512).unwrap(),
                bias: Some(bias.range(0, 512).unwrap()),
                x: x.range(0, rows * 1024).unwrap(),
                xh: xh.range(0, rows * 1024).unwrap(),
                z: z.range(0, z_bytes).unwrap(),
                out: out.range(0, rows * 512).unwrap(),
                flag: flag.range(0, 4).unwrap(),
                rows,
                split_k,
                k: 256,
                n: 128,
                bits_twice: 3,
                book: Codebook::Mul1,
            },
        )
        .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(
            testkit::read_bytes(ctx, &flag, 4),
            want_flag.to_le_bytes(),
            "{stage}/split{split_k}"
        );
        assert!(
            testkit::read_bytes(ctx, &out, rows * 512)
                .as_chunks::<4>()
                .0
                .iter()
                .all(|b| f32::from_le_bytes(*b).is_finite()),
            "{stage}/split{split_k}"
        );
        let partials = testkit::read_bytes(ctx, &z, z_bytes);
        if stage == "projection" || rows > 8 && stage == "merge" {
            assert!(
                partials
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|b| f32::from_le_bytes(*b) == 0.0)
            );
        } else if stage == "merge" {
            assert!(partials.as_chunks::<4>().0.iter().all(|b| {
                let v = f32::from_le_bytes(*b);
                v.is_finite() && v != 0.0
            }));
        }
    }
}

#[test]
fn projection_preserves_small_f32_bias_after_zero_product() {
    if !testkit::device_present("EXL3 F32 final bias") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let bias: Vec<f32> = (1..=128).map(|i| i as f32 * 1e-10).collect();
    let matrix = fixture(false, Codebook::Mul1, 3, (256, 128));
    let mut expected = vec![0.0; 128];
    matrix
        .project_fp64_into(&[f64::from(1e-5f32); 256], 1, &mut expected)
        .unwrap();
    let scaled: Vec<_> = matrix
        .suh()
        .host_bytes()
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| f64::from(1e-5f32) * f16::from_bits(u16::from_le_bytes(*bytes)).to_f64())
        .collect();
    let xh_ref = reference_rotate(&scaled);
    let mut weights = vec![0u16; 256 * 128];
    matrix
        .codebook()
        .reconstruct_into(matrix.trellis(), &mut weights)
        .unwrap();
    for (rows, split_k) in [1, 9, 144]
        .into_iter()
        .flat_map(|rows| [2, 4, 8].map(|split_k| (rows, split_k)))
    {
        let (actual, flag, _) =
            constant_projection(&ctx, &stream, 0.0, Some(&bias), (rows, split_k));
        assert_eq!(actual, bias.repeat(rows));
        assert_eq!(flag, 8);
        let (actual, flag, xh) =
            constant_projection(&ctx, &stream, 1e-5, Some(&bias), (rows, split_k));
        assert_eq!(flag, 8);
        let reference =
            if !super::midrows_enabled(rows, std::env::var_os("FIFI_EXL3_LEGACY").as_deref()) {
                expected.clone()
            } else {
                let operands = rounded_mma_inputs(&xh, &xh_ref);
                reference_output(
                    &reference_project(&operands, &weights, (256, 128), 0..256),
                    matrix.svh(),
                )
            };
        for (index, value) in actual.iter().enumerate() {
            let want = reference[index % 128] + f64::from(bias[index % 128]);
            assert!(
                (f64::from(*value) - want).abs() < 1e-9,
                "split{split_k}/index{index}"
            );
        }
    }
}

#[test]
fn nonfinite_and_overflowing_arithmetic_set_the_sticky_error_bit() {
    if !testkit::device_present("EXL3 nonfinite F32 guard") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    for (rows, split_k) in [1, 9, 144]
        .into_iter()
        .flat_map(|rows| [2, 4, 8].map(|split_k| (rows, split_k)))
    {
        for value in [f32::NAN, f32::INFINITY, f32::MAX] {
            let (actual, flag, _) =
                constant_projection(&ctx, &stream, value, None, (rows, split_k));
            assert_eq!(flag, 9, "value={value}/split{split_k}");
            assert!(actual.iter().all(|v| v.is_finite()));
        }
        staged_guard_cases(&ctx, &stream, (rows, split_k));
        let (_, flag, _) = constant_projection(&ctx, &stream, 0.0, None, (rows, split_k));
        assert_eq!(flag, 8, "a data error must not poison the context");
    }
}

#[test]
fn midrows_accumulation_exceeds_f16_range_without_overflow() {
    if !testkit::device_present("EXL3 midrows F32 accumulator range") {
        return;
    }
    let matrix = fixture(false, Codebook::Mul1, 3, (256, 128));
    let scaled: Vec<_> = matrix
        .suh()
        .host_bytes()
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| 4096.0 * f16::from_bits(u16::from_le_bytes(*p)).to_f64())
        .collect();
    let xh = reference_rotate(&scaled);
    assert!(xh.iter().all(|x| x.abs() <= 65504.0));
    let mut weights = vec![0u16; 256 * 128];
    matrix
        .codebook()
        .reconstruct_into(matrix.trellis(), &mut weights)
        .unwrap();
    let projection = reference_project(&xh, &weights, (256, 128), 0..256);
    assert!(projection.iter().any(|x| x.abs() > 65504.0));
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    for rows in [9, 144] {
        let (actual, flag, _) = constant_projection(&ctx, &stream, 4096.0, None, (rows, 2));
        assert_eq!(
            flag, 8,
            "rows{rows}: finite operands must not overflow accumulation"
        );
        assert!(actual.iter().all(|x| x.is_finite()));
        assert!(actual.iter().any(|x| x.abs() > 65504.0));
    }
}

#[test]
fn midrows_opt_in_preserves_f32_stages_and_subnormal_scales() {
    if !testkit::device_present("EXL3 explicit midrows F32 projection") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = testkit::Rng::new(0x4d49_4452_4f57);
    let (k, n) = (256, 384);
    for (rows, subnormal) in [(9, false), (9, true), (144, true)] {
        let matrix = fixture(subnormal, Codebook::Mul1, 5, (k, n));
        let input: Vec<f32> = (0..rows * k)
            .map(|_| (rng.below(65537) as f32 - 32768.0) / 65536.0)
            .collect();
        let input_ref: Vec<_> = input.iter().map(|&v| f64::from(v)).collect();
        let mut expected = vec![0.0; rows * n];
        matrix
            .project_fp64_into(&input_ref, rows, &mut expected)
            .unwrap();
        assert!(expected.iter().any(|v| *v != 0.0));
        let suh_ref: Vec<_> = matrix
            .suh()
            .host_bytes()
            .unwrap()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| f16::from_bits(u16::from_le_bytes(*p)).to_f64())
            .collect();
        let scaled: Vec<_> = input_ref
            .iter()
            .enumerate()
            .map(|(i, x)| x * suh_ref[i % k])
            .collect();
        let xh_ref = reference_rotate(&scaled);
        let mut weights = vec![0u16; k * n];
        matrix
            .codebook()
            .reconstruct_into(matrix.trellis(), &mut weights)
            .unwrap();
        let bytes = matrix.trellis().host_bytes().unwrap();
        let mut packed_bytes = vec![0xa5; 2];
        packed_bytes.extend_from_slice(bytes);
        let packed = testkit::upload_bytes(&ctx, &packed_bytes);
        let suh = upload_scales(&ctx, matrix.suh());
        let svh = upload_scales(&ctx, matrix.svh());
        let x = testkit::upload_f32(&ctx, &input);
        let xh = testkit::upload_filled(&ctx, 0xff, rows * k * 4);
        let z_bytes = rows * n * 4;
        let z = testkit::upload_filled(&ctx, 0xa5, z_bytes + 8);
        let out = testkit::upload_filled(&ctx, 0xff, rows * n * 4);
        let flag = testkit::upload_bytes(&ctx, &8u32.to_le_bytes());
        ctx.exl3_linear_midrows(
            &stream,
            Exl3LinearCall {
                trellis: packed.range(2, bytes.len()).unwrap(),
                suh: suh.range(0, k * 4).unwrap(),
                svh: svh.range(0, n * 4).unwrap(),
                bias: None,
                x: x.range(0, rows * k * 4).unwrap(),
                xh: xh.range(0, rows * k * 4).unwrap(),
                z: z.range(4, z_bytes).unwrap(),
                out: out.range(0, rows * n * 4).unwrap(),
                flag: flag.range(0, 4).unwrap(),
                rows,
                split_k: 2,
                k,
                n,
                bits_twice: matrix.bitrate_twice(),
                book: matrix.codebook(),
            },
        )
        .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(testkit::read_bytes(&ctx, &flag, 4), 8u32.to_le_bytes());
        let actual_xh = testkit::read_bytes(&ctx, &xh, rows * k * 4);
        assert_stage_close(&actual_xh, &xh_ref, "input_rotate");
        let operands =
            if super::midrows_enabled(rows, std::env::var_os("FIFI_EXL3_LEGACY").as_deref()) {
                rounded_mma_inputs(&actual_xh, &xh_ref)
            } else {
                xh_ref
            };
        let project_ref = reference_project(&operands, &weights, (k, n), 0..k);
        let expected = reference_output(&project_ref, matrix.svh());
        let projection = testkit::read_bytes(&ctx, &z, z_bytes + 8);
        assert_eq!(&projection[..4], &[0xa5; 4], "leading projection guard");
        assert_eq!(
            &projection[z_bytes + 4..],
            &[0xa5; 4],
            "trailing projection guard"
        );
        assert_stage_close(
            &projection[4..z_bytes + 4],
            &project_ref,
            "midrows projection",
        );
        assert_stage_close(
            &testkit::read_bytes(&ctx, &out, rows * n * 4),
            &expected,
            "midrows complete fp64 equation",
        );
    }
}

#[test]
fn gemv_and_batched_projection_keep_subnormal_scales_in_f32() {
    projection_cases(false);
}

#[test]
fn midrows_all_books_rates_and_row_tails_match_fp64() {
    projection_cases(true);
}

fn projection_cases(midrows: bool) {
    if !testkit::device_present("EXL3 F32 batched projection") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let mut rng = testkit::Rng::new(0x4c49_4e45_4152);
    let cases = if midrows {
        [Codebook::ThreeInst, Codebook::Mcg, Codebook::Mul1]
            .into_iter()
            .flat_map(|book| {
                [2, 4, 6, 8, 10, 12, 14, 16, 3, 5, 7]
                    .into_iter()
                    .filter(move |bits| bits % 2 == 0 || book == Codebook::Mul1)
                    .flat_map(move |bits| {
                        [false, true].map(move |low| (book, bits, low, (384, 384)))
                    })
            })
            .collect::<Vec<_>>()
    } else {
        vec![
            (Codebook::ThreeInst, 2, false, (256, 128)),
            (Codebook::Mcg, 16, false, (256, 128)),
            (Codebook::Mul1, 10, false, (384, 128)),
            (Codebook::Mcg, 12, false, (384, 128)),
            (Codebook::ThreeInst, 14, false, (384, 128)),
            (Codebook::Mul1, 3, false, (256, 128)),
            (Codebook::Mul1, 6, false, (256, 128)),
            (Codebook::Mul1, 3, true, (256, 128)),
            (Codebook::Mul1, 5, false, (256, 384)),
            (Codebook::Mul1, 7, false, (384, 384)),
        ]
    };
    for (book, bits, subnormal, (k, n)) in cases {
        let matrix = fixture(subnormal, book, bits, (k, n));
        let precision = if midrows { 65536 } else { 1024 };
        let prototype: Vec<f64> = (0..8 * k)
            .map(|_| (rng.below(precision + 1) as f64 - (precision / 2) as f64) / precision as f64)
            .collect();
        let mut expected = vec![0.0; 8 * n];
        matrix
            .project_fp64_into(&prototype, 8, &mut expected)
            .unwrap();
        let half_values = |tensor: &Tensor| -> Vec<f64> {
            tensor
                .host_bytes()
                .unwrap()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| f16::from_bits(u16::from_le_bytes(*p)).to_f64())
                .collect()
        };
        let suh_ref = half_values(matrix.suh());
        let svh_ref = half_values(matrix.svh());
        let scaled: Vec<_> = prototype
            .iter()
            .enumerate()
            .map(|(i, x)| x * suh_ref[i % k])
            .collect();
        let xh_ref = reference_rotate(&scaled);
        let mut weights = vec![0u16; k * n];
        book.reconstruct_into(matrix.trellis(), &mut weights)
            .unwrap();
        let mut project_ref = reference_project(&xh_ref, &weights, (k, n), 0..k);
        let mut output_ref = reference_rotate(&project_ref);
        for (i, value) in output_ref.iter_mut().enumerate() {
            *value *= svh_ref[i % n];
        }
        let mut r1_prefix = Vec::new();
        let partial_refs: HashMap<_, _> = [2u32, 4, 8]
            .into_iter()
            .map(|split_k| {
                let banks: Vec<_> = (0..split_k as usize - 1)
                    .map(|bank| {
                        reference_project(
                            &xh_ref,
                            &weights,
                            (k, n),
                            k * bank / split_k as usize..k * (bank + 1) / split_k as usize,
                        )
                    })
                    .collect();
                (split_k, banks)
            })
            .collect();
        let bytes = matrix.trellis().host_bytes().unwrap();
        let offset = if n == 384 { 2 } else { 0 };
        let mut storage = vec![0xa5; offset];
        storage.extend_from_slice(bytes);
        let packed = testkit::upload_bytes(&ctx, &storage);
        let suh = upload_scales(&ctx, matrix.suh());
        let svh = upload_scales(&ctx, matrix.svh());
        let row_counts: Vec<_> = if midrows {
            (9..=144).collect()
        } else {
            vec![1usize, 2, 3, 4, 7, 8, 9, 17, 2048, 4096]
        };
        let splits: &[u32] = if midrows { &[2] } else { &[2, 4, 8] };
        for (rows, split_k) in row_counts
            .into_iter()
            .flat_map(|rows| splits.iter().copied().map(move |split_k| (rows, split_k)))
        {
            let input: Vec<f32> = (0..rows * k)
                .map(|i| prototype[i % (8 * k)] as f32)
                .collect();
            let x = testkit::upload_f32(&ctx, &input);
            let xh = testkit::upload_filled(&ctx, 0xff, rows * k * 4);
            let banks = if rows <= 8 { split_k as usize - 1 } else { 1 };
            let z_bytes = rows * n * 4 * banks;
            let z = testkit::upload_filled(&ctx, 0xa5, z_bytes + 8);
            let out = testkit::upload_filled(&ctx, 0xff, rows * n * 4);
            let flag = testkit::upload_bytes(&ctx, &8u32.to_le_bytes());
            let call = Exl3LinearCall {
                trellis: packed.range(offset, bytes.len()).unwrap(),
                suh: suh.range(0, k * 4).unwrap(),
                svh: svh.range(0, n * 4).unwrap(),
                bias: None,
                x: x.range(0, rows * k * 4).unwrap(),
                xh: xh.range(0, rows * k * 4).unwrap(),
                z: z.range(4, z_bytes).unwrap(),
                out: out.range(0, rows * n * 4).unwrap(),
                flag: flag.range(0, 4).unwrap(),
                rows,
                split_k,
                k,
                n,
                bits_twice: matrix.bitrate_twice(),
                book: matrix.codebook(),
            };
            if midrows {
                ctx.exl3_linear_midrows(&stream, call)
            } else {
                ctx.exl3_linear(&stream, call)
            }
            .unwrap();
            stream.synchronize().unwrap();
            assert_eq!(testkit::read_bytes(&ctx, &flag, 4), 8u32.to_le_bytes());
            let actual_xh = testkit::read_bytes(&ctx, &xh, rows * k * 4);
            assert_stage_close(&actual_xh, &xh_ref, "input_rotate");
            if midrows
                && super::midrows_enabled(rows, std::env::var_os("FIFI_EXL3_LEGACY").as_deref())
            {
                if r1_prefix.is_empty() {
                    r1_prefix = actual_xh[..8 * k * 4].to_vec();
                    let operands = rounded_mma_inputs(&r1_prefix, &xh_ref);
                    project_ref = reference_project(&operands, &weights, (k, n), 0..k);
                    output_ref = reference_output(&project_ref, matrix.svh());
                }
                // Reuse the eight prototype rows only after proving their F32 operands repeat.
                for (index, bytes) in actual_xh.as_chunks::<4>().0.iter().enumerate() {
                    assert_eq!(*bytes, r1_prefix.as_chunks::<4>().0[index % (8 * k)]);
                }
            }
            let z_storage = testkit::read_bytes(&ctx, &z, z_bytes + 8);
            assert_eq!(&z_storage[..4], &[0xa5; 4], "split{split_k} leading guard");
            assert_eq!(
                &z_storage[z_bytes + 4..],
                &[0xa5; 4],
                "split{split_k} trailing guard"
            );
            let references = if rows <= 8 {
                &partial_refs[&split_k][..banks]
            } else {
                std::slice::from_ref(&project_ref)
            };
            for (bank, reference) in references.iter().enumerate() {
                let start = 4 + bank * rows * n * 4;
                assert_stage_close(
                    &z_storage[start..start + rows * n * 4],
                    reference,
                    &format!("project/rows{rows}/split{split_k}/bank{bank}"),
                );
            }
            let actual = testkit::read_bytes(&ctx, &out, rows * n * 4);
            let case =
                format!("{book:?}/bits{bits}/subnormal{subnormal}/rows{rows}/split{split_k}");
            assert_stage_close(&actual, &output_ref, &format!("output_rotate/{case}"));
            assert_stage_close(
                &actual,
                if midrows { &output_ref } else { &expected },
                &format!("complete fp64 equation/{case}"),
            );
            if midrows
                && matches!(rows, 9 | 64)
                && book == Codebook::ThreeInst
                && bits == 2
                && !subnormal
            {
                // Keep the frozen nine-row stock bound on both legacy and MMA dispatch.
                assert_stock_quality(&matrix, &input[..9 * k], &expected, &actual[..9 * n * 4]);
            }
        }
    }
}

#[test]
fn grouped_gather_matches_shfl_gather_bitwise() {
    // Standard gate: skips on a device-less host, FAILS when the lease sets
    // INFERENCE_CUDA_REQUIRE_DEVICE — never a silent pass.
    if !testkit::device_present("EXL3 grouped gather sweep") {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let run = |groups: &[(Codebook, u8, usize)], k: usize| -> Vec<Vec<u8>> {
        let n = 128usize;
        let total: usize = groups.iter().map(|g| g.2).sum();
        let input: Vec<f32> = (0..total * k)
            .map(|i| ((i * 17 % 251) as f32 - 125.0) / 1024.0)
            .collect();
        let x = testkit::upload_f32(&ctx, &input);
        let xh = testkit::upload_filled(&ctx, 0xff, total * k * 4);
        let flag = testkit::upload_bytes(&ctx, &8u32.to_le_bytes());
        (0..2)
            .map(|arm| {
                // Each arm owns its partials workspace: a candidate kernel that
                // skips split stores must not inherit the other arm's partials.
                let z = testkit::upload_filled(&ctx, 0xa5, total * n * 4 + 8);
                let out =
                    testkit::upload_filled(&ctx, if arm == 0 { 0xff } else { 0xa5 }, total * n * 4);
                let mut first = 0usize;
                // Buffers live outside the call list: Exl3LinearCall borrows them.
                let banks: Vec<_> = groups
                    .iter()
                    .map(|&(book, bits, _rows)| {
                        let matrix = fixture(false, book, bits, (k, n));
                        let bytes = matrix.trellis().host_bytes().unwrap();
                        (
                            testkit::upload_bytes(&ctx, bytes),
                            upload_scales(&ctx, matrix.suh()),
                            upload_scales(&ctx, matrix.svh()),
                        )
                    })
                    .collect();
                let calls: Vec<_> = banks
                    .iter()
                    .zip(groups.iter())
                    .map(|((packed, suh, svh), &(book, bits, rows))| {
                        let offset = first;
                        first += rows;
                        Exl3LinearCall {
                            trellis: packed
                                .range(0, k * n / 256 * usize::from(bits) * 16)
                                .unwrap(),
                            suh: suh.range(0, k * 4).unwrap(),
                            svh: svh.range(0, n * 4).unwrap(),
                            bias: None,
                            x: x.range(offset * k * 4, rows * k * 4).unwrap(),
                            xh: xh.range(offset * k * 4, rows * k * 4).unwrap(),
                            z: z.range(4 + offset * n * 4, rows * n * 4).unwrap(),
                            out: out.range(offset * n * 4, rows * n * 4).unwrap(),
                            flag: flag.range(0, 4).unwrap(),
                            rows,
                            split_k: 2,
                            k,
                            n,
                            bits_twice: bits,
                            book,
                        }
                    })
                    .collect();
                // arm 0 = default shuffle gather (reference), arm 1 = shared-table
                // gather (candidate); the flag mapping lives here and nowhere else.
                ctx.exl3_grouped(&stream, &calls, arm == 1).unwrap();
                assert_eq!(
                    testkit::read_bytes(&ctx, &flag, 4),
                    8u32.to_le_bytes(),
                    "flag must stay clear (arm {arm})"
                );
                testkit::read_bytes(&ctx, &out, total * n * 4)
            })
            .collect()
    };
    // Permanent single-call Legacy reference on the identical mixed layout.
    let legacy_run = |groups: &[(Codebook, u8, usize)], k: usize| -> Vec<u8> {
        let n = 128usize;
        let total: usize = groups.iter().map(|g| g.2).sum();
        let input: Vec<f32> = (0..total * k)
            .map(|i| ((i * 17 % 251) as f32 - 125.0) / 1024.0)
            .collect();
        let x = testkit::upload_f32(&ctx, &input);
        let xh = testkit::upload_filled(&ctx, 0xff, total * k * 4);
        let z = testkit::upload_filled(&ctx, 0xa5, total * n * 4 + 8);
        let out = testkit::upload_filled(&ctx, 0xff, total * n * 4);
        let flag = testkit::upload_bytes(&ctx, &8u32.to_le_bytes());
        let mut first = 0usize;
        for &(book, bits, rows) in groups {
            let matrix = fixture(false, book, bits, (k, n));
            let bytes = matrix.trellis().host_bytes().unwrap();
            let trellis = testkit::upload_bytes(&ctx, bytes);
            let suh = upload_scales(&ctx, matrix.suh());
            let svh = upload_scales(&ctx, matrix.svh());
            let call = Exl3LinearCall {
                trellis: trellis
                    .range(0, k * n / 256 * usize::from(bits) * 16)
                    .unwrap(),
                suh: suh.range(0, k * 4).unwrap(),
                svh: svh.range(0, n * 4).unwrap(),
                bias: None,
                x: x.range(first * k * 4, rows * k * 4).unwrap(),
                xh: xh.range(first * k * 4, rows * k * 4).unwrap(),
                z: z.range(4 + first * n * 4, rows * n * 4).unwrap(),
                out: out.range(first * n * 4, rows * n * 4).unwrap(),
                flag: flag.range(0, 4).unwrap(),
                rows,
                split_k: 2,
                k,
                n,
                bits_twice: bits,
                book,
            };
            ctx.exl3_linear(&stream, call).unwrap();
            first += rows;
        }
        stream.synchronize().unwrap();
        assert_eq!(
            testkit::read_bytes(&ctx, &flag, 4),
            8u32.to_le_bytes(),
            "legacy path must leave the flag clear"
        );
        testkit::read_bytes(&ctx, &out, total * n * 4)
    };
    // Even rates for all books, odd rates book Mul1 only (geometry()-valid pairs);
    // k=128 gives zero-tile warps, k=384 the 2-vs-1 trip skew, k=2048 the prod shape.
    let rates: Vec<(Codebook, u8)> = [Codebook::ThreeInst, Codebook::Mcg, Codebook::Mul1]
        .iter()
        .flat_map(|b| [2u8, 4, 6, 8, 10, 12, 14, 16].map(|bits| (*b, bits)))
        .chain([3u8, 5, 7].map(|bits| (Codebook::Mul1, bits)))
        .collect();
    for &(book, bits) in &rates {
        for &k in &[128usize, 384] {
            let outs = run(&[(book, bits, 8)], k);
            assert_eq!(outs[0], outs[1], "gather vs shfl {book:?}/{bits}/{k}/rows8");
        }
    }
    let outs = run(&[(Codebook::Mul1, 8, 8)], 2048);
    assert_eq!(outs[0], outs[1], "gather vs shfl prod shape k=2048");
    // Mixed launch: distinct books, odd+even rates, uneven row counts (incl. 1-row tail).
    let mixed = [
        (Codebook::ThreeInst, 4, 8),
        (Codebook::Mcg, 10, 8),
        (Codebook::Mul1, 3, 1),
        (Codebook::Mul1, 8, 2),
    ];
    let outs = run(&mixed, 384);
    assert_eq!(outs[0], outs[1], "gather vs shfl mixed books/rates/rows");
    // Permanent reference: the single-call Legacy path (project_small) shares no
    // grouped dispatch code with the sweep, so it anchors the arithmetic.
    let legacy = legacy_run(&mixed, 384);
    assert_eq!(outs[0], legacy, "shfl gather vs legacy mixed");
}
