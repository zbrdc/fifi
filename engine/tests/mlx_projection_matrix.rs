//! Released MLX projection metadata and same-shape reshape parity on Apple Silicon.
#![cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]

use std::time::Instant;

use inference_engine::{
    EngError,
    kernels::{MlxKernels, PackedWeight, QuantizationMode, QuantizationSpec},
    tensor::{Dtype, Tensor},
};
use sha2::{Digest, Sha256};

const WIDTH: u32 = 2048;
const BATCHES: [u32; 2] = [1, 32];

fn floats(dtype: Dtype, shape: Vec<u32>, values: impl Iterator<Item = f32>) -> Tensor {
    let bytes = match dtype {
        Dtype::F32 => values.flat_map(f32::to_le_bytes).collect(),
        Dtype::Bf16 => values
            .flat_map(|value| half::bf16::from_f32(value).to_bits().to_le_bytes())
            .collect(),
        _ => panic!("unsupported fixture float dtype {dtype:?}"),
    };
    Tensor::try_from_host(dtype, shape, bytes).unwrap()
}

fn weight(format: QuantizationSpec, dtype: Dtype, width: u32) -> PackedWeight {
    let payload_width = width * format.bits / 32;
    let scale_width = width / format.group_size;
    // Finite, nonzero payload in both the affine and MX floating-point encodings.
    let payload = Tensor::try_from_host(
        Dtype::U32,
        vec![width, payload_width],
        vec![0x35; (width * payload_width * 4) as usize],
    )
    .unwrap();
    let aux = vec![width, scale_width];
    let count = (width * scale_width) as usize;
    let (scales, biases) = if format.mode == QuantizationMode::Affine {
        (
            floats(dtype, aux.clone(), std::iter::repeat_n(0.015625, count)),
            Some(floats(dtype, aux, std::iter::repeat_n(-0.0078125, count))),
        )
    } else {
        (
            Tensor::try_from_host(Dtype::U8, aux, vec![127; count]).unwrap(),
            None,
        )
    };
    PackedWeight::try_new(payload, scales, biases, vec![width, width], format).unwrap()
}

fn formats() -> impl Iterator<Item = (&'static str, QuantizationSpec)> {
    [2, 3, 4, 5, 6, 8]
        .into_iter()
        .flat_map(|bits| {
            [32, 64, 128].into_iter().map(move |group_size| {
                (
                    "affine",
                    QuantizationSpec {
                        mode: QuantizationMode::Affine,
                        bits,
                        group_size,
                    },
                )
            })
        })
        .chain([
            (
                "mxfp4",
                QuantizationSpec {
                    mode: QuantizationMode::Mxfp4,
                    bits: 4,
                    group_size: 32,
                },
            ),
            (
                "mxfp8",
                QuantizationSpec {
                    mode: QuantizationMode::Mxfp8,
                    bits: 8,
                    group_size: 32,
                },
            ),
        ])
}

#[test]
fn reshape_preserves_owner_and_dtype_checks() {
    let mut first = MlxKernels::try_new().unwrap();
    let mut second = MlxKernels::try_new().unwrap();
    let input = floats(Dtype::F32, vec![1, 2], [1.0, -2.0].into_iter());
    let row = first.import_tensor(&input).unwrap();
    assert!(matches!(
        second.reshape(&row, &[1, 2]),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        first.reshape(&row, &[1, 3]),
        Err(EngError::BadRequest(_))
    ));

    first.allow_bf16(true);
    let input = floats(Dtype::Bf16, vec![1, 2], [1.0, -2.0].into_iter());
    let row = first.import_tensor(&input).unwrap();
    let reshaped = first.reshape(&row, row.shape()).unwrap();
    assert_eq!(
        first.readback(&reshaped).unwrap().host_bytes(),
        input.host_bytes()
    );
    first.allow_bf16(false);
    assert!(matches!(
        first.reshape(&row, &[1, 2]),
        Err(EngError::BadRequest(_))
    ));

    let input = Tensor::try_from_host(Dtype::U32, vec![2], vec![3, 0, 0, 0, 4, 0, 0, 0]).unwrap();
    let ids = first.import_tensor(&input).unwrap();
    let reshaped = first.reshape(&ids, &[2]).unwrap();
    assert_eq!(
        first.readback(&reshaped).unwrap().host_bytes(),
        input.host_bytes()
    );
}

#[test]
#[ignore = "explicit Apple Silicon GPU allocation for packed projection timing"]
fn packed_projection_matrix() {
    let samples: usize = std::env::var("FIFI_MLX_PROJECTION_SAMPLES")
        .ok()
        .map_or(12, |value| value.parse().unwrap());
    assert!(samples > 0);
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    k.set_cache_limit(64 << 20).unwrap();
    let large = std::env::var_os("FIFI_MLX_PROJECTION_LARGE").is_some_and(|value| value == "1");
    let (width, batches) = if large {
        (4096, [128, 256])
    } else {
        (WIDTH, BATCHES)
    };
    for (mode, spec) in formats().filter(|(mode, spec)| {
        !large
            || *mode != "affine"
            || ([4, 8].contains(&spec.bits) && [32, 64].contains(&spec.group_size))
    }) {
        for (dtype, name) in [(Dtype::F32, "f32"), (Dtype::Bf16, "bf16")] {
            let packed = k.import_packed(&weight(spec, dtype, width)).unwrap();
            for rows in batches {
                let input = floats(
                    dtype,
                    vec![rows, width],
                    (0..rows * width).map(|index| (1 + index % 17) as f32 / 512.0),
                );
                let input = k.import_tensor(&input).unwrap();
                let view = k.reshape(&input, input.shape()).unwrap();
                let expected = k.qmm(&input, &packed).unwrap();
                let actual = k.qmm(&view, &packed).unwrap();
                let actual = k.readback(&actual).unwrap();
                let expected = k.readback(&expected).unwrap();
                let bytes = actual.host_bytes().unwrap();
                assert_eq!(
                    bytes,
                    expected.host_bytes().unwrap(),
                    "{mode} {spec:?} {name} {rows}"
                );
                assert!(bytes.iter().any(|byte| *byte != 0));
                assert_eq!(actual.shape(), &[rows, width]);
                let digest = format!("{:x}", Sha256::digest(bytes));

                let mut submission = Vec::with_capacity(samples);
                let mut elapsed = Vec::with_capacity(samples);
                for _ in 0..samples {
                    let started = Instant::now();
                    let view = k.reshape(&input, input.shape()).unwrap();
                    let output = k.qmm(&view, &packed).unwrap();
                    submission.push(started.elapsed().as_nanos() as u64);
                    k.evaluate(&[&output]).unwrap();
                    elapsed.push(started.elapsed().as_nanos() as u64);
                }
                println!(
                    "mlx_projection {}",
                    serde_json::json!({
                        "quant": mode, "bits": spec.bits, "group": spec.group_size,
                        "dtype": name, "rows": rows, "width": width, "output_sha256": digest,
                        "submit_ns": submission, "elapsed_ns": elapsed,
                    })
                );
            }
        }
    }
}
