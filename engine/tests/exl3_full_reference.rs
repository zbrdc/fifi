use std::collections::HashMap;

use inference_engine::{
    tensor::{Dtype, Tensor},
    weights::{
        Tier, WeightSource,
        exl3::{Config, Matrix},
    },
};

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

fn tensor(dtype: Dtype, shape: Vec<u32>, bytes: Vec<u8>) -> Tensor {
    Tensor::try_from_host(dtype, shape, bytes).unwrap()
}

fn matrix(k: u32, n: u32, bias: Option<Dtype>) -> Matrix {
    let scales = |length: u32, first: u16, second: u16| {
        let bytes = (0..length)
            .flat_map(|index| {
                (match index {
                    0 => first,
                    128 => second,
                    _ => 0x3c00u16,
                })
                .to_le_bytes()
            })
            .collect();
        tensor(Dtype::F16, vec![length], bytes)
    };
    let mut source = Components(
        [
            (
                "matrix.trellis".into(),
                tensor(
                    Dtype::I16,
                    vec![k / 16, n / 16, 48],
                    vec![0; (k as usize / 16) * (n as usize / 16) * 48 * 2],
                ),
            ),
            ("matrix.suh".into(), scales(k, 0x4000, 0x4200)),
            ("matrix.svh".into(), scales(n, 0x4200, 0x4400)),
            (
                "matrix.mul1".into(),
                tensor(Dtype::I32, vec![], 0x83dc_d12du32.to_le_bytes().to_vec()),
            ),
        ]
        .into(),
    );
    if let Some(dtype) = bias {
        let bytes = match dtype {
            Dtype::F16 => 0x3400u16.to_le_bytes().repeat(n as usize),
            Dtype::F32 => 0.25f32.to_le_bytes().repeat(n as usize),
            _ => panic!("unsupported test bias dtype"),
        };
        source
            .0
            .insert("matrix.bias".into(), tensor(dtype, vec![n], bytes));
    }
    let cfg = Config::from_json(br#"{"quantization_config":{"quant_method":"exl3","version":"1.6.2","codebook":"mul1","bits":4.0,"out_scales":"auto","expert_interm_div":128}}"#).unwrap();
    Matrix::load(&source, "matrix", &cfg).unwrap()
}

#[test]
fn normalized_block_rotations_preserve_both_nonsquare_axes() {
    // Zero trellis gives W'=-221/64. Two normalized H128 transforms give -442 at each block origin.
    for (k, n) in [(128, 256), (256, 128)] {
        let m = matrix(k, n, None);
        let mut actual: Vec<f64> = vec![42.0; k as usize * n as usize];
        m.reconstruct_fp64_into(&mut actual).unwrap();
        for row in 0..k as usize {
            for col in 0..n as usize {
                let expected = if row.is_multiple_of(128) && col.is_multiple_of(128) {
                    -442.0 * (2 + row / 128) as f64 * (3 + col / 128) as f64
                } else {
                    0.0
                };
                assert!(
                    (actual[row * n as usize + col] - expected).abs() < 1e-9,
                    "k={k},n={n},row={row},col={col},got={},expected={expected}",
                    actual[row * n as usize + col]
                );
            }
        }
    }
}

#[test]
fn fp64_projection_is_batched_with_optional_half_or_float_bias() {
    let mut input: Vec<f64> = vec![0.0; 3 * 256];
    for (row, (first, second)) in [(1.0, 0.0), (0.0, 1.0), (2.0, -1.0)]
        .into_iter()
        .enumerate()
    {
        input[row * 256] = first;
        input[row * 256 + 128] = second;
    }
    for bias in [None, Some(Dtype::F16), Some(Dtype::F32)] {
        let m = matrix(256, 128, bias);
        let mut actual: Vec<f64> = vec![42.0; 3 * 128];
        m.project_fp64_into(&input, 3, &mut actual).unwrap();
        let addend = if bias.is_some() { 0.25 } else { 0.0 };
        for (row, first) in [-2652.0, -3978.0, -1326.0].into_iter().enumerate() {
            for col in 0..128 {
                let expected = if col == 0 { first + addend } else { addend };
                assert!(
                    (actual[row * 128 + col] - expected).abs() < 1e-9,
                    "bias={bias:?},row={row},col={col}"
                );
            }
        }
    }
}

#[test]
fn fp64_projection_preserves_bias_after_finite_cancellation() {
    let m = matrix(256, 128, Some(Dtype::F32));
    let mut weights = vec![0.0; 256 * 128];
    m.reconstruct_fp64_into(&mut weights).unwrap();
    let scale = 2.0f64.powi(100);
    let mut input = vec![0.0; 256];
    input[0] = -weights[128 * 128] * scale;
    input[128] = weights[0] * scale;
    assert!(input.iter().all(|x| x.is_finite()));
    assert_eq!(input[0] * weights[0] + input[128] * weights[128 * 128], 0.0);
    let mut output = vec![42.0; 128];
    m.project_fp64_into(&input, 1, &mut output).unwrap();
    assert!((output[0] - 0.25).abs() < 1e-12, "got {}", output[0]);
}

#[test]
fn full_fp64_rejects_wrong_destination_without_partial_writes() {
    let m = matrix(128, 256, None);
    let mut destination: Vec<f64> = vec![42.0; 128 * 256 - 1];
    assert!(m.reconstruct_fp64_into(&mut destination).is_err());
    assert!(destination.iter().all(|value| *value == 42.0));
}

#[test]
fn batched_fp64_rejects_bad_spans_without_partial_writes() {
    let m = matrix(256, 128, Some(Dtype::F32));
    let mut destination: Vec<f64> = vec![42.0; 3 * 128];
    assert!(
        m.project_fp64_into(&vec![0.0; 3 * 256 - 1], 3, &mut destination)
            .is_err()
    );
    assert!(destination.iter().all(|value| *value == 42.0));
    for rows in [0, usize::MAX] {
        assert!(m.project_fp64_into(&[], rows, &mut destination).is_err());
        assert!(destination.iter().all(|value| *value == 42.0));
    }
    destination.pop();
    assert!(
        m.project_fp64_into(&vec![0.0; 3 * 256], 3, &mut destination)
            .is_err()
    );
    assert!(destination.iter().all(|value| *value == 42.0));
    let wide = matrix(128, 256, None);
    assert!(matches!(
        wide.project_fp64_into(&[], usize::MAX / 256 + 1, &mut []),
        Err(inference_engine::EngError::MemoryPlan(_))
    ));
}

fn subnormal_matrix() -> (Matrix, [f64; 128], [f64; 128]) {
    let suh = std::array::from_fn(|i| half::f16::from_bits(155 + (i % 70) as u16).to_f64());
    let svh = std::array::from_fn(|i| 0.75 + (i % 23) as f64 / 32.0);
    let scales = |values: &[f64]| {
        tensor(
            Dtype::F16,
            vec![128],
            values
                .iter()
                .flat_map(|v| half::f16::from_f64(*v).to_le_bytes())
                .collect(),
        )
    };
    let mut state = 0x1357_2468u32;
    let packed = (0..8 * 8 * 64)
        .flat_map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 16) as u16).to_le_bytes()
        })
        .collect();
    let source = Components(
        [
            (
                "matrix.trellis".into(),
                tensor(Dtype::I16, vec![8, 8, 64], packed),
            ),
            ("matrix.suh".into(), scales(&suh)),
            ("matrix.svh".into(), scales(&svh)),
            (
                "matrix.mul1".into(),
                tensor(Dtype::I32, vec![], 0x83dc_d12du32.to_le_bytes().to_vec()),
            ),
        ]
        .into(),
    );
    let cfg = Config::from_json(br#"{"quantization_config":{"quant_method":"exl3","version":"1.6.2","codebook":"mul1","bits":4.0}}"#).unwrap();
    (Matrix::load(&source, "matrix", &cfg).unwrap(), suh, svh)
}

fn direct_hadamard_128(values: &[f64], left: bool) -> Vec<f64> {
    assert_eq!(values.len(), 128 * 128);
    let mut output = vec![0.0; values.len()];
    for row in 0..128 {
        for col in 0..128 {
            let coordinate = if left { row } else { col };
            let sum = (0usize..128)
                .map(|inner| {
                    let value = if left {
                        values[inner * 128 + col]
                    } else {
                        values[row * 128 + inner]
                    };
                    if (coordinate & inner).count_ones().is_multiple_of(2) {
                        value
                    } else {
                        -value
                    }
                })
                .sum::<f64>();
            output[row * 128 + col] = sum / 128.0f64.sqrt();
        }
    }
    output
}

fn staged_half_weights(prime: &[f64], suh: &[f64], svh: &[f64]) -> Vec<f64> {
    let half = |value| half::f16::from_f64(value).to_f64();
    let mut values = direct_hadamard_128(prime, true);
    for (index, value) in values.iter_mut().enumerate() {
        *value = half(half(*value) * suh[index / 128]);
    }
    values = direct_hadamard_128(&values, false);
    for (index, value) in values.iter_mut().enumerate() {
        *value = half(half(*value) * svh[index % 128]);
    }
    values
}

fn relative_l2(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let squared_error = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).powi(2))
        .sum::<f64>();
    (squared_error / expected.iter().map(|v| v * v).sum::<f64>()).sqrt()
}

#[test]
fn subnormal_scales_need_more_precision_than_staged_fp16() {
    let (matrix, suh, svh) = subnormal_matrix();
    assert!(
        suh.iter()
            .all(|value| *value > 0.0 && *value < 2.0f64.powi(-14))
    );
    let mut bits = vec![0; 128 * 128];
    matrix
        .codebook()
        .reconstruct_into(matrix.trellis(), &mut bits)
        .unwrap();
    let prime: Vec<_> = bits
        .into_iter()
        .map(|v| half::f16::from_bits(v).to_f64())
        .collect();
    let mut ideal = direct_hadamard_128(&prime, true);
    for (index, value) in ideal.iter_mut().enumerate() {
        *value *= suh[index / 128];
    }
    ideal = direct_hadamard_128(&ideal, false);
    for (index, value) in ideal.iter_mut().enumerate() {
        *value *= svh[index % 128];
    }
    let mut native = vec![0.0; ideal.len()];
    matrix.reconstruct_fp64_into(&mut native).unwrap();
    assert!(relative_l2(&native, &ideal) < 1e-12);

    // FP16 rounding after each Hadamard and scale loses precision on subnormal expert weights.
    let staged = staged_half_weights(&prime, &suh, &svh);
    let loss = relative_l2(&staged, &ideal);
    assert!((0.002..0.005).contains(&loss), "staged FP16 loss={loss}");
    let normal_suh = suh.map(|value| value * 128.0);
    let restored: Vec<_> = staged_half_weights(&prime, &normal_suh, &svh)
        .into_iter()
        .map(|value| value / 128.0)
        .collect();
    assert!(relative_l2(&restored, &ideal) < 0.0008);
}
