use super::{read_step, select};
use crate::{
    EngError,
    family::flint::math,
    kernels::MlxKernels,
    tensor::{Dtype, ExecutionTensor, Tensor},
};

fn input(k: &mut MlxKernels, shape: &[u32], values: &[f32]) -> ExecutionTensor {
    let host = Tensor::try_from_host(
        Dtype::F32,
        shape.to_vec(),
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap();
    k.import_tensor(&host).unwrap()
}

fn bytes(k: &mut MlxKernels, x: &ExecutionTensor) -> Vec<u8> {
    k.readback(x).unwrap().host_bytes().unwrap().to_vec()
}

#[test]
fn device_sigmoid_top6_matches_reference_and_saturated_ties() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    let cases: [[f32; 256]; 3] = [
        std::array::from_fn(|i| ((i * 37) % 256) as f32 / 32.0 - 4.0),
        [0.0; 256],
        std::array::from_fn(|i| 20.0 + i as f32),
    ];
    for row in cases {
        let expected = math::route(&row).unwrap();
        let raw = input(&mut k, &[1, 256], &row);
        #[cfg(feature = "native-fault-qualification")]
        k.qualification_reset_timing();
        let route = select(&mut k, &raw).unwrap();
        #[cfg(feature = "native-fault-qualification")]
        {
            let timing = k.qualification_timing();
            assert_eq!(timing.import_calls, 0);
            assert_eq!(timing.readback_calls, 0);
            assert_eq!(timing.evaluate_calls, 0);
        }
        assert_eq!(route.ids.bound(), 256);
        assert_eq!(route.weights.shape(), [1, 6, 1]);
        assert_eq!(route.weights.dtype(), Dtype::F32);
        assert_eq!(route.health.shape(), [2]);
        let ids: Vec<_> = bytes(&mut k, route.ids.ids())
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| u32::from_le_bytes(*v))
            .collect();
        assert_eq!(ids, expected.indices);
        let weights = bytes(&mut k, &route.weights);
        for (value, expected) in weights.as_chunks::<4>().0.iter().zip(expected.weights) {
            let actual = f32::from_le_bytes(*value);
            assert!((actual - expected).abs() < 2e-7, "{actual} != {expected}");
        }
        assert_eq!(read_step(&mut k, None, &[route.health]).unwrap(), None);
    }
}

#[test]
fn raw_nonfinite_and_zero_mass_fail_at_the_step_boundary() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut row = [0.0; 256];
        row[255] = bad;
        let raw = input(&mut k, &[1, 256], &row);
        let route = select(&mut k, &raw).unwrap();
        assert!(matches!(
            read_step(&mut k, None, &[route.health]),
            Err(EngError::Backend { .. })
        ));
    }
    let raw = input(&mut k, &[1, 256], &[-1000.0; 256]);
    let route = select(&mut k, &raw).unwrap();
    assert!(matches!(
        read_step(&mut k, None, &[route.health]),
        Err(EngError::Backend { .. })
    ));
    let short = input(&mut k, &[1, 255], &[0.0; 255]);
    assert!(matches!(
        select(&mut k, &short),
        Err(EngError::BadRequest(_))
    ));
}

#[test]
fn combined_export_keeps_bf16_logits_and_rejects_bad_health() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let values: Vec<_> = (0..150_144)
        .map(|i| (i % 13) as f32 * 0.1 + 0.001)
        .collect();
    let logits = input(&mut k, &[1, 150_144], &values);
    let logits = k.cast(&logits, Dtype::Bf16).unwrap();
    let health = input(&mut k, &[2], &[1.0, 3.0]);
    let second = input(&mut k, &[2], &[1.0, 2.0]);
    let actual = read_step(&mut k, Some(&logits), &[health, second])
        .unwrap()
        .unwrap();
    assert_eq!(actual.len(), values.len());
    for (actual, expected) in actual.into_iter().zip(values) {
        let bits = expected.to_bits();
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
        assert_eq!(actual.to_bits(), rounded);
    }
    for (flag, mass) in [
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
        (1.0, f32::NAN),
        (1.0, f32::INFINITY),
    ] {
        let bad = input(&mut k, &[2], &[flag, mass]);
        assert!(matches!(
            read_step(&mut k, Some(&logits), &[bad]),
            Err(EngError::Backend { .. })
        ));
    }
    let malformed = input(&mut k, &[1], &[1.0]);
    assert!(matches!(
        read_step(&mut k, None, &[malformed]),
        Err(EngError::BadRequest(_))
    ));
}

#[test]
fn batch_sigmoid_routes_keep_each_rows_top6_and_mass() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    let rows: [[f32; 256]; 3] = [
        std::array::from_fn(|i| ((i * 37) % 256) as f32 / 32.0 - 4.0),
        [0.0; 256],
        std::array::from_fn(|i| 20.0 + i as f32),
    ];
    let flat: Vec<_> = rows.iter().flatten().copied().collect();
    let raw = input(&mut k, &[3, 256], &flat);
    #[cfg(feature = "native-fault-qualification")]
    k.qualification_reset_timing();
    let route = select(&mut k, &raw).unwrap();
    #[cfg(feature = "native-fault-qualification")]
    {
        let timing = k.qualification_timing();
        assert_eq!(
            (
                timing.import_calls,
                timing.readback_calls,
                timing.evaluate_calls
            ),
            (0, 0, 0)
        );
    }
    assert_eq!(route.ids.ids().shape(), [18]);
    assert_eq!(route.weights.shape(), [3, 6, 1]);
    assert_eq!(route.health.shape(), [4]);
    let ids = bytes(&mut k, route.ids.ids());
    let weights = bytes(&mut k, &route.weights);
    for (row, (ids, weights)) in rows.iter().zip(
        ids.as_chunks::<24>()
            .0
            .iter()
            .zip(weights.as_chunks::<24>().0),
    ) {
        let expected = math::route(row).unwrap();
        let actual: Vec<_> = ids
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| u32::from_le_bytes(*v))
            .collect();
        assert_eq!(actual, expected.indices);
        for (value, expected) in weights.as_chunks::<4>().0.iter().zip(expected.weights) {
            assert!((f32::from_le_bytes(*value) - expected).abs() < 2e-7);
        }
    }
    let health = bytes(&mut k, &route.health);
    let health: Vec<_> = health
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect();
    assert_eq!(health[0], 1.0);
    assert!(health[1].is_finite() && health[1] > 0.0);
    assert_eq!(&health[2..], &[3.0, 6.0]);
    assert_eq!(read_step(&mut k, None, &[route.health]).unwrap(), None);
}

#[test]
fn batch_health_rejects_a_bad_row_without_exporting_a_partial_block() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut rows = [0.0; 3 * 256];
        rows[256 + 255] = bad;
        let raw = input(&mut k, &[3, 256], &rows);
        let route = select(&mut k, &raw).unwrap();
        assert!(matches!(
            read_step(&mut k, None, &[route.health]),
            Err(EngError::Backend { .. })
        ));
    }
    let mut rows = [0.0; 3 * 256];
    rows[512..].fill(-1000.0);
    let raw = input(&mut k, &[3, 256], &rows);
    let route = select(&mut k, &raw).unwrap();
    assert!(matches!(
        read_step(&mut k, None, &[route.health]),
        Err(EngError::Backend { .. })
    ));
}
