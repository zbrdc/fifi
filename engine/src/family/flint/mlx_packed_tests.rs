//! Mixed-operand smoke tests for the exact native import/projection seams.
//! The expected products below are exact rational dot products, not MLX outputs.
use crate::{
    kernels::{MlxKernels, PackedWeight, QuantizationMode, QuantizationSpec},
    tensor::{Dtype, ExecutionTensor, Tensor},
};

fn import(k: &mut MlxKernels, dtype: Dtype, shape: &[u32], bytes: Vec<u8>) -> ExecutionTensor {
    k.import_tensor(&Tensor::try_from_host(dtype, shape.to_vec(), bytes).unwrap())
        .unwrap()
}

fn read(k: &mut MlxKernels, tensor: &ExecutionTensor) -> Vec<f32> {
    let host = k.readback(tensor).unwrap();
    assert_eq!(host.dtype(), Dtype::F32);
    host.host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn packed(bits: u32, stacked: bool) -> PackedWeight {
    let experts = if stacked { 3 } else { 1 };
    let per_word = 32 / bits;
    let mut payload = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for expert in 0..experts {
        for row in 0..2 {
            for word in 0..64 / per_word {
                let mut value = 0_u32;
                for lane in 0..per_word {
                    let col = word * per_word + lane;
                    let code = if bits == 4 {
                        (col + 3 * row + expert) % 16
                    } else if row == 0 {
                        col
                    } else {
                        255 - col
                    };
                    value |= code << (lane * bits);
                }
                payload.extend(value.to_le_bytes());
            }
            let (scale, bias): (f32, f32) = match (bits, row) {
                (4, 0) => (0.5, -1.0),
                (4, _) => (0.25, 2.0),
                (8, 0) => (1.0 / 128.0, -0.5),
                (8, _) => (1.0 / 256.0, 1.0),
                _ => unreachable!(),
            };
            scales.extend(((scale.to_bits() >> 16) as u16).to_le_bytes());
            biases.extend(((bias.to_bits() >> 16) as u16).to_le_bytes());
        }
    }
    let shape = |last| {
        if stacked {
            vec![3, 2, last]
        } else {
            vec![2, last]
        }
    };
    PackedWeight::try_new(
        Tensor::try_from_host(Dtype::U32, shape(64 / per_word), payload).unwrap(),
        Tensor::try_from_host(Dtype::Bf16, shape(1), scales).unwrap(),
        Some(Tensor::try_from_host(Dtype::Bf16, shape(1), biases).unwrap()),
        shape(64),
        QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits,
            group_size: 64,
        },
    )
    .unwrap()
}

fn activation(k: &mut MlxKernels, shape: &[u32]) -> ExecutionTensor {
    let values: Vec<_> = (0..64).map(|i| ((i % 7) as f32 - 3.0) / 8.0).collect();
    import(
        k,
        Dtype::F32,
        shape,
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            actual.is_finite() && (actual - expected).abs() < 1e-6,
            "{actual} != {expected}"
        );
    }
}

#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn f32_activation_bf16_affine_int4_int8_and_stacked_gather() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let x = activation(&mut k, &[1, 64]);
    for (bits, expected) in [
        (4, [-11.0 / 16.0, -25.0 / 16.0]),
        (8, [255.0 / 1024.0, -399.0 / 512.0]),
    ] {
        let host = packed(bits, false);
        let weight = k.import_packed(&host).unwrap();
        let out = k.qmm(&x, &weight).unwrap();
        assert_eq!(out.shape(), [1, 2]);
        close(&read(&mut k, &out), &expected);
        if bits == 8 {
            // Production embeddings expand only bounded INT8 rows through this seam.
            let dense = k.dequantize(&weight).unwrap();
            let values = read(&mut k, &dense);
            assert_eq!(dense.shape(), [2, 64]);
            close(&values[..3], &[-0.5, -0.4921875, -0.484375]);
            close(
                &values[64..67],
                &[511.0 / 256.0, 510.0 / 256.0, 509.0 / 256.0],
            );
        }
    }
    let weight = k.import_packed(&packed(4, true)).unwrap();
    let x = activation(&mut k, &[1, 1, 64]);
    let ids = import(
        &mut k,
        Dtype::U32,
        &[2],
        [2_u32, 0].iter().flat_map(|v| v.to_le_bytes()).collect(),
    );
    let out = k.gather_qmm(&x, &weight, &ids).unwrap();
    assert_eq!(out.shape(), [2, 1, 2]);
    close(
        &read(&mut k, &out),
        &[31.0 / 16.0, -9.0 / 4.0, -11.0 / 16.0, -25.0 / 16.0],
    );
}
