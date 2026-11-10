//! Test-only scalar references for gated-delta prefill.

#[derive(Clone, Copy)]
pub(crate) enum Norm {
    Ggml,
    AddEps,
    QwenRms,
}

pub(crate) fn normalize(x: &[f32], eps: f64, scale: f64, norm: Norm) -> Vec<f64> {
    let sum: f64 = x.iter().map(|&v| f64::from(v).powi(2)).sum();
    let denominator = match norm {
        Norm::Ggml => sum.max(eps * eps),
        Norm::AddEps => sum + eps,
        Norm::QwenRms => sum + x.len() as f64 * eps,
    }
    .sqrt();
    x.iter()
        .map(|&v| f64::from(v) * scale / denominator)
        .collect()
}

/// State is `[key_dim, value_dim]`, with the value dimension contiguous.
pub(crate) fn step(
    state: &mut [f64],
    q: &[f64],
    k: &[f64],
    v: &[f64],
    decay: f64,
    beta: f64,
) -> Vec<f64> {
    assert_eq!(q.len(), k.len());
    assert_eq!(state.len(), k.len() * v.len());
    let mut sk = vec![0.0; v.len()];
    for i in 0..k.len() {
        for j in 0..v.len() {
            state[i * v.len() + j] *= decay;
            sk[j] += state[i * v.len() + j] * k[i];
        }
    }
    let delta: Vec<f64> = (0..v.len()).map(|j| (v[j] - sk[j]) * beta).collect();
    let mut out = vec![0.0; v.len()];
    for i in 0..k.len() {
        for j in 0..v.len() {
            state[i * v.len() + j] += k[i] * delta[j];
            out[j] += state[i * v.len() + j] * q[i];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{Norm, normalize, step};

    #[test]
    fn normalization_forms_are_distinct_for_sparse_near_zero_input() {
        let mut x = [0.0f32; 128];
        x[0] = 1e-5;
        let expected = [
            (Norm::Ggml, 1.0),
            (Norm::AddEps, 0.0099995),
            (Norm::QwenRms, 0.000883883),
        ];
        for (form, want) in expected {
            let out = normalize(&x, 1e-6, 1.0, form);
            assert!((out[0] - want).abs() < 1e-8);
            assert!(out[1..].iter().all(|&v| v == 0.0));
            assert!(
                normalize(&[0.0; 128], 1e-6, 1.0, form)
                    .iter()
                    .all(|&v| v == 0.0)
            );
        }
    }

    #[test]
    fn recurrence_matches_hand_values_with_nonzero_state() {
        let mut state = [1.0, 3.0, 2.0, 4.0];
        let out = step(&mut state, &[0.0, 1.0], &[1.0, 0.0], &[2.0, 4.0], 0.5, 0.5);
        assert_eq!(state, [1.25, 2.75, 1.0, 2.0]);
        assert_eq!(out, [1.0, 2.0]);
        let out = step(&mut state, &[1.0, 2.0], &[1.0, 1.0], &[1.0, 2.0], 0.25, 0.5);
        assert_eq!(state, [17.0 / 32.0, 35.0 / 32.0, 15.0 / 32.0, 29.0 / 32.0]);
        assert_eq!(out, [47.0 / 32.0, 93.0 / 32.0]);
    }
}
