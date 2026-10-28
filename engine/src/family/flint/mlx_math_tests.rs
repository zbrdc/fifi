//! Native tests prepared for the parent's Mac window; not portable qualification.
use super::*;
use crate::{family::flint::reference::AttentionPartial, tensor::Tensor};

fn import(k: &mut MlxKernels, shape: &[u32], values: &[f32]) -> ExecutionTensor {
    let host = Tensor::try_from_host(
        Dtype::F32,
        shape.to_vec(),
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap();
    k.import_tensor(&host).unwrap()
}

fn read(k: &mut MlxKernels, x: &ExecutionTensor) -> Vec<f32> {
    let host = k.readback(x).unwrap();
    host.host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

#[test]
fn native_sinks_gqa_and_window_edges_match_scalar() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    for slots in [512, 513, 514] {
        let mut scores = vec![0.0; 32 * 2 * slots];
        let values: Vec<_> = (0..4 * slots * 128)
            .map(|i| ((i * 17) % 137) as f32 / 137.0 - 0.5)
            .collect();
        let sinks: Vec<_> = (0..32)
            .map(|h| {
                if h == 0 {
                    1000.0
                } else {
                    h as f32 / 16.0 - 1.0
                }
            })
            .collect();
        let mut expected = Vec::new();
        for (h, &sink) in sinks.iter().enumerate() {
            for row in 0..2 {
                let mut p = AttentionPartial::new(128).unwrap();
                for s in 0..slots {
                    let score = ((h * 31 + s * 13 + row * 7) % 53) as f32 / 19.0;
                    // Query absolute positions 512/513, slots start at zero.
                    // Includes a fully masked row to test sink-only output.
                    let visible = h != 1 && s <= 512 + row && 512 + row - s < 513;
                    scores[(h * 2 + row) * slots + s] =
                        if visible { score } else { f32::NEG_INFINITY };
                    if visible {
                        let v: Vec<_> = values
                            [((h / 8) * slots + s) * 128..((h / 8) * slots + s + 1) * 128]
                            .iter()
                            .map(|&v| v as f64)
                            .collect();
                        p.push(score as f64, &v).unwrap();
                    }
                }
                expected.extend(p.finish(sink as f64).unwrap());
            }
        }
        let scores = import(&mut k, &[32, 2, slots as u32], &scores);
        let values = import(&mut k, &[4, slots as u32, 128], &values);
        let sink = import(&mut k, &[32], &sinks);
        let output = sink_attention(&mut k, &scores, &values, &sink).unwrap();
        for (actual, expected) in read(&mut k, &output).iter().zip(expected) {
            assert!(
                (*actual as f64 - expected).abs() < 2e-6,
                "{actual} vs {expected}"
            );
        }
    }
}

#[test]
fn native_three_projection_experts_and_ungated_shared_combine() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    let x = import(&mut k, &[1, 2], &[2.0, -1.0]);
    let gate = import(&mut k, &[2, 2], &[-1.0, 1.0, 0.0, 1.0]);
    let up = import(&mut k, &[2, 2], &[1.0, 3.0, 1.0, -2.0]);
    let down = import(&mut k, &[2, 2], &[1.0, -3.0, 2.0, 1.0]);
    let w = SwiGluWeights {
        gate: Projection::Dense(&gate),
        up: Projection::Dense(&up),
        down: Projection::Dense(&down),
    };
    let shared = expert(&mut k, &w, &x).unwrap();
    let scalar = super::super::reference::swiglu(&[-2.0, 1.0], &[1.0, 8.0]).unwrap();
    let expected = [scalar[0] + 2.0 * scalar[1], -3.0 * scalar[0] + scalar[1]];
    for (actual, want) in read(&mut k, &shared).iter().zip(expected) {
        assert!((actual - want).abs() < 2e-6);
    }
    let weights = import(&mut k, &[1, 1, 1], &[0.25]);
    let total = combine(&mut k, &shared, &weights, &shared).unwrap();
    for (actual, want) in read(&mut k, &total).iter().zip(expected) {
        assert!((actual - want * 1.25).abs() < 3e-6);
    }
}

#[test]
fn native_qk_rmsnorm_rope_and_scores_use_absolute_positions() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    let q = import(&mut k, &[32, 1, 128], &[1.0; 32 * 128]);
    let keys = import(&mut k, &[4, 1, 128], &[2.0; 4 * 128]);
    let gamma = import(&mut k, &[128], &[0.5; 128]);
    for pos in [None, Some(0), Some(512), Some(513), Some(514), Some(8191)] {
        let (qn, kn) = normalize_qk(&mut k, (&q, &keys), (&gamma, &gamma), 1e-6, pos).unwrap();
        // F64 geometry avoids compounding the reused F32 powf oracle's phase rounding.
        let rms = 0.5f64 / (1.0 + f64::from(1e-6_f32)).sqrt();
        // With u=EPSILON/2, budget u for inv_freq and u for pos*inv_freq:
        // |delta_angle| <= 2*u*pos. Rotation amplifies it by |a|+|b|=2*rms;
        // eight EPSILON*rms cover normalization, trig and output rounding.
        let eps = f64::from(f32::EPSILON);
        let tolerance = 2.0 * rms * f64::from(pos.unwrap_or(0)) * eps + 8.0 * eps * rms;
        let mut expected_q = vec![rms; 128];
        if let Some(pos) = pos {
            for index in 0..64 {
                let angle = f64::from(pos) / 10_000.0f64.powf(index as f64 / 64.0);
                let (sin, cos) = angle.sin_cos();
                expected_q[index] = rms * (cos - sin);
                expected_q[index + 64] = rms * (cos + sin);
            }
        }
        for (index, (actual, want)) in read(&mut k, &qn)
            .iter()
            .zip(expected_q.iter().cycle())
            .enumerate()
        {
            let actual = f64::from(*actual);
            assert!(
                (actual - want).abs() < tolerance,
                "QK preparation pos={pos:?} index={index}: actual={actual} expected={want} bound={tolerance}"
            );
        }
        let scale = import(&mut k, &[1], &[1.0 / 128.0f32.sqrt()]);
        let mask = import(&mut k, &[1, 1], &[0.0]);
        let scores = qk_scores(&mut k, &qn, &kn, &scale, &mask).unwrap();
        let expected =
            128.0 * 0.5 / (1.0f32 + 1e-6).sqrt() * 1.0 / (4.0f32 + 1e-6).sqrt() / 128.0f32.sqrt();
        for value in read(&mut k, &scores) {
            assert!((value - expected).abs() < 3e-5);
        }
    }
}
