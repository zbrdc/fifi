#![cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]

use std::sync::Mutex;

use inference_engine::{
    EngError,
    kernels::{MlxKernels, RopeSpec},
    tensor::{Dtype, Tensor},
};

static GPU_TEST: Mutex<()> = Mutex::new(());

fn f32_input(shape: Vec<u32>, values: &[f32]) -> Tensor {
    Tensor::try_from_host(
        Dtype::F32,
        shape,
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
    .unwrap()
}

fn values_of(tensor: &Tensor) -> Vec<f32> {
    tensor
        .host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

// The Metal sin, cos, and exp approximations differ from the double-precision expected
// values by a few F32 ulp of the largest operand; 1e-5 relative covers them.
fn assert_near(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "actual values {actual:?}");
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 1e-5 * b.abs().max(1.0),
            "element {index}: actual {a}, expected {b}"
        );
    }
}

#[test]
fn plain_rms_norm_normalizes_each_row() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = kernels
        .import_tensor(&f32_input(vec![2, 2], &[3.0, 4.0, 1.0, 1.0]))
        .unwrap();

    let output = kernels.rms_norm_plain(&x, 1e-6).unwrap();
    assert_eq!(output.shape(), [2, 2]);
    let read = kernels.readback(&output).unwrap();
    assert_near(
        &values_of(&read),
        &[0.8485281, 1.1313708, 0.9999995, 0.9999995],
    );
}

#[test]
fn scaled_rope_uses_scale_offset_and_dims() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = kernels
        .import_tensor(&f32_input(
            vec![1, 2, 4],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ))
        .unwrap();

    let output = kernels
        .rope_scaled(
            &x,
            &RopeSpec {
                dims: 4,
                base: 10000.0,
                scale: 2.0,
                offset: 1,
            },
        )
        .unwrap();
    assert_eq!(output.shape(), [1, 2, 4]);
    let read = kernels.readback(&output).unwrap();
    assert_near(
        &values_of(&read),
        &[
            -3.1440391, 1.9196053, -0.3391431, 4.0391974, 2.0293994, 5.675286, -8.359518, 8.233537,
        ],
    );

    let y = kernels
        .import_tensor(&f32_input(vec![1, 1, 4], &[1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    let rotated = kernels
        .rope_scaled(
            &y,
            &RopeSpec {
                dims: 2,
                base: 10000.0,
                scale: 1.0,
                offset: 1,
            },
        )
        .unwrap();
    assert_eq!(rotated.shape(), [1, 1, 4]);
    let read_y = kernels.readback(&rotated).unwrap();
    // Only the first `dims` values rotate; the rest pass through unchanged.
    assert_near(&values_of(&read_y), &[-1.1426397, 1.9220756, 3.0, 4.0]);
}

#[test]
fn grouped_attention_shares_key_heads() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let q = kernels
        .import_tensor(&f32_input(vec![1, 2, 1, 2], &[1.0, 0.0, 0.0, 1.0]))
        .unwrap();
    let k = kernels
        .import_tensor(&f32_input(vec![1, 1, 2, 2], &[1.0, 0.0, 0.0, 1.0]))
        .unwrap();
    let v = kernels
        .import_tensor(&f32_input(vec![1, 1, 2, 2], &[10.0, 0.0, 0.0, 20.0]))
        .unwrap();

    let output = kernels.grouped_attention(&q, &k, &v, 1.0, false).unwrap();
    assert_eq!(output.shape(), [1, 2, 1, 2]);
    let read = kernels.readback(&output).unwrap();
    assert_near(
        &values_of(&read),
        &[7.310586, 5.3788284, 2.6894142, 14.621172],
    );
}

#[test]
fn grouped_attention_applies_the_causal_mask() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let q = kernels
        .import_tensor(&f32_input(vec![1, 1, 2, 2], &[1.0, 0.0, 0.0, 1.0]))
        .unwrap();
    let k = kernels
        .import_tensor(&f32_input(vec![1, 1, 2, 2], &[1.0, 0.0, 0.0, 1.0]))
        .unwrap();
    let v = kernels
        .import_tensor(&f32_input(vec![1, 1, 2, 2], &[10.0, 0.0, 0.0, 20.0]))
        .unwrap();

    let output = kernels.grouped_attention(&q, &k, &v, 1.0, true).unwrap();
    assert_eq!(output.shape(), [1, 1, 2, 2]);
    let read = kernels.readback(&output).unwrap();
    // Query 0 sees only key 0, so its row is the first value row unchanged.
    assert_near(&values_of(&read), &[10.0, 0.0, 2.6894142, 14.621172]);
}

#[test]
fn fast_operations_reject_bad_requests() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let mut other = MlxKernels::try_new().unwrap();
    let integer = kernels
        .import_tensor(
            &Tensor::try_from_host(
                Dtype::U32,
                vec![2],
                [0u32, 1]
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            )
            .unwrap(),
        )
        .unwrap();
    let x = kernels
        .import_tensor(&f32_input(vec![2], &[3.0, 4.0]))
        .unwrap();
    let flat = kernels
        .import_tensor(&f32_input(vec![1, 4], &[1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    let block = kernels
        .import_tensor(&f32_input(vec![1, 1, 4], &[1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    let foreign_keys = other
        .import_tensor(&f32_input(
            vec![1, 2, 2, 2],
            &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        ))
        .unwrap();
    let foreign_values = other
        .import_tensor(&f32_input(
            vec![1, 2, 2, 2],
            &[10.0, 0.0, 0.0, 20.0, 20.0, 0.0, 0.0, 10.0],
        ))
        .unwrap();
    let queries = kernels
        .import_tensor(&f32_input(
            vec![1, 3, 1, 2],
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ))
        .unwrap();
    let keys = kernels
        .import_tensor(&f32_input(
            vec![1, 2, 2, 2],
            &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        ))
        .unwrap();
    let values = kernels
        .import_tensor(&f32_input(
            vec![1, 2, 2, 2],
            &[10.0, 0.0, 0.0, 20.0, 20.0, 0.0, 0.0, 10.0],
        ))
        .unwrap();

    assert!(
        matches!(
            kernels.rms_norm_plain(&integer, 1e-6),
            Err(EngError::BadRequest(_))
        ),
        "1: plain RMS norm of a U32 input must be BadRequest"
    );
    assert!(
        matches!(
            kernels.rms_norm_plain(&x, 0.0),
            Err(EngError::BadRequest(_))
        ),
        "2: plain RMS norm with eps zero must be BadRequest"
    );
    assert!(
        matches!(
            kernels.rope_scaled(
                &flat,
                &RopeSpec {
                    dims: 2,
                    base: 10000.0,
                    scale: 1.0,
                    offset: 0
                }
            ),
            Err(EngError::BadRequest(_))
        ),
        "3: scaled RoPE of a rank-two input must be BadRequest"
    );
    assert!(
        matches!(
            kernels.rope_scaled(
                &block,
                &RopeSpec {
                    dims: 2,
                    base: 10000.0,
                    scale: f32::NAN,
                    offset: 0
                }
            ),
            Err(EngError::BadRequest(_))
        ),
        "4: scaled RoPE with a NaN scale must be BadRequest"
    );
    assert!(
        matches!(
            kernels.grouped_attention(&queries, &foreign_keys, &foreign_values, 1.0, false),
            Err(EngError::BadRequest(_))
        ),
        "5: grouped attention with foreign keys must be BadRequest"
    );
    assert!(
        matches!(
            kernels.grouped_attention(&queries, &keys, &values, 1.0, false),
            Err(EngError::BadRequest(_))
        ),
        "6: grouped attention with ungrouped heads must be BadRequest"
    );

    let output = kernels.rms_norm_plain(&x, 1e-6).unwrap();
    assert_eq!(output.shape(), [2]);
    let read = kernels.readback(&output).unwrap();
    assert_eq!(
        values_of(&read).len(),
        2,
        "actual values {:?}",
        values_of(&read)
    );
}

/// Two queries follow two cached positions: 4 keys, `D = 2`, scale 1. Query 0 has position 2 and
/// sees keys 0 to 2. Query 1 has position 3 and sees keys 0 to 3. That is a causal mask aligned to
/// the last key. With `e = exp(1)`, the logits of query 0 are `[1, 0, 1]` and give
/// `(40e, 20 + 30e) / (2e + 1) = (16.892752, 15.776812)`. The logits of query 1 are `[0, 1, 1, 0]`
/// and give `(15 + 30e, 50e + 7) / (2e + 2) = (12.982939, 19.217759)`. A mask aligned to the first
/// key would give `(10, 0)` and `(2.689414, 14.621172)`.
#[test]
fn grouped_attention_aligns_the_causal_mask_to_the_last_key() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let q = kernels
        .import_tensor(&f32_input(vec![1, 1, 2, 2], &[1.0, 0.0, 0.0, 1.0]))
        .unwrap();
    let k = kernels
        .import_tensor(&f32_input(
            vec![1, 1, 4, 2],
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        ))
        .unwrap();
    let v = kernels
        .import_tensor(&f32_input(
            vec![1, 1, 4, 2],
            &[10.0, 0.0, 0.0, 20.0, 30.0, 30.0, 5.0, 7.0],
        ))
        .unwrap();

    let output = kernels.grouped_attention(&q, &k, &v, 1.0, true).unwrap();
    assert_eq!(output.shape(), [1, 1, 2, 2]);
    let read = kernels.readback(&output).unwrap();
    assert_near(
        &values_of(&read),
        &[16.892752, 15.776812, 12.982939, 19.217759],
    );
}
