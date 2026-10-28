//! Host f32 helpers shared by the Flash-Next forward blocks.

use crate::EngError;

/// Logistic function `1 / (1 + exp(-x))`.
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// SiLU `x * sigmoid(x)`.
pub(crate) fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Stable softplus `max(x, 0) + ln(1 + exp(-|x|))`.
pub(crate) fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// Checks that `values` holds exactly `len` elements.
pub(crate) fn expect_len(values: &[f32], len: usize, what: &str) -> Result<(), EngError> {
    if values.len() != len {
        return Err(EngError::BadRequest(format!(
            "{what} has {} values, expected {len}",
            values.len()
        )));
    }
    Ok(())
}

/// Dot product `sum(a[i] * b[i])` for equal-length slices.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> Result<f32, EngError> {
    expect_len(b, a.len(), "dot operand")?;
    Ok(a.iter().zip(b).map(|(x, y)| x * y).sum())
}

/// Zero-centered RMS normalization `x[i] * (1 + weight[i]) / sqrt(mean(x^2) + eps)`, in the oracle operation order.
pub(crate) fn rms0(x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, EngError> {
    if x.is_empty() {
        return Err(EngError::BadRequest("rms0 input is empty".to_owned()));
    }
    expect_len(weight, x.len(), "rms0 weight")?;
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let denom = (mean + eps).sqrt();
    Ok(x.iter()
        .zip(weight)
        .map(|(v, w)| v * (1.0 + w) / denom)
        .collect())
}

/// Half-split RoPE rotating pairs `(i, i + rotary_dim / 2)` by `pos / theta^(2i / rotary_dim)`.
/// The position converts to f32 as the oracle does, so positions above 2^24 round.
pub(crate) fn rope_half(
    values: &mut [f32],
    rotary_dim: usize,
    pos: u32,
    theta: f32,
) -> Result<(), EngError> {
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || rotary_dim > values.len() {
        return Err(EngError::BadRequest(format!(
            "rope rotary dimension {rotary_dim} is invalid for {} values",
            values.len()
        )));
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(EngError::BadRequest(format!(
            "rope theta {theta} is invalid"
        )));
    }
    let half = rotary_dim / 2;
    for i in 0..half {
        let exponent = (2 * i) as f32 / rotary_dim as f32;
        let angle = pos as f32 / theta.powf(exponent);
        let (sin, cos) = angle.sin_cos();
        let a = values[i];
        let b = values[i + half];
        values[i] = a * cos - b * sin;
        values[i + half] = b * cos + a * sin;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-6
    }

    #[test]
    fn sigmoid_matches_known_values() {
        let v0 = sigmoid(0.0);
        assert!(close(v0, 0.5), "{v0} vs 0.5");
        let v2 = sigmoid(2.0);
        assert!(close(v2, 0.880_797_1), "{v2} vs 0.880_797_1");
        assert_eq!(sigmoid(-100.0), 0.0);
        assert_eq!(sigmoid(100.0), 1.0);
    }

    #[test]
    fn silu_matches_known_values() {
        let p = silu(2.0);
        assert!(close(p, 1.761_594_2), "{p} vs 1.761_594_2");
        let n = silu(-1.0);
        assert!(close(n, -0.268_941_4), "{n} vs -0.268_941_4");
    }

    #[test]
    fn softplus_is_stable() {
        let z = softplus(0.0);
        assert!(close(z, std::f32::consts::LN_2), "{z} vs 0.693_147_2");
        let big = softplus(30.0);
        assert!(close(big, 30.0), "{big} vs 30.0");
        let small = softplus(-30.0);
        assert!((0.0..=1e-12).contains(&small), "{small}");
        assert_eq!(softplus(-200.0), 0.0);
        let neg = softplus(-1.0);
        assert!(close(neg, 0.313_261_7), "{neg} vs 0.313_261_7");
        let pos = softplus(1.0);
        assert!(close(pos, 1.313_261_7), "{pos} vs 1.313_261_7");
    }

    #[test]
    fn dot_multiplies_and_sums() {
        let d = dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
        assert!(close(d, 32.0), "{d} vs 32.0");
        assert!(matches!(
            dot(&[1.0], &[1.0, 2.0]),
            Err(EngError::BadRequest(_))
        ));
    }

    #[test]
    fn rms0_with_epsilon() {
        let out = rms0(&[3.0, 4.0], &[0.0, 0.0], 0.5).unwrap();
        let expected = [0.832_050_3f32, 1.109_400_4];
        assert_eq!(out.len(), expected.len());
        for (a, e) in out.iter().zip(expected) {
            assert!(close(*a, e), "{a} vs {e}");
        }
    }

    #[test]
    fn rms0_uses_one_plus_weight() {
        let out = rms0(&[3.0, 4.0], &[0.0, 1.0], 0.0).unwrap();
        let expected = [0.848_528_1f32, 2.262_741_7];
        assert_eq!(out.len(), expected.len());
        for (a, e) in out.iter().zip(expected) {
            assert!(close(*a, e), "{a} vs {e}");
        }
        assert!(matches!(
            rms0(&[3.0, 4.0], &[0.0], 0.0),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(rms0(&[], &[], 0.0), Err(EngError::BadRequest(_))));
    }

    #[test]
    fn rope_half_rotates_pairs_i_and_i_plus_half() {
        let mut v = [1.0f32, 0.0, 0.0, 1.0, 7.0];
        rope_half(&mut v, 4, 1, 10_000.0).unwrap();
        let expected = [0.540_302_3f32, -0.009_999_833, 0.841_471, 0.999_95, 7.0];
        for (a, e) in v.iter().zip(expected) {
            assert!(close(*a, e), "{a} vs {e}");
        }
    }

    #[test]
    fn rope_half_rotates_six_dimensions_at_position_three() {
        let mut v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        rope_half(&mut v, 6, 3, 10_000.0).unwrap();
        let expected = [
            -1.554_472_5f32,
            1.286_650_9,
            2.961_157_8,
            -3.818_85,
            5.229_2,
            6.019_264_5,
            7.0,
        ];
        for (a, e) in v.iter().zip(expected) {
            assert!((a - e).abs() <= 1e-5, "{a} vs {e}");
        }
    }

    #[test]
    fn rope_half_rejects_invalid_theta() {
        let mut v = [0.0f32; 4];
        assert!(matches!(
            rope_half(&mut v, 4, 1, 0.0),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            rope_half(&mut v, 4, 1, f32::NAN),
            Err(EngError::BadRequest(_))
        ));
    }

    #[test]
    fn rope_half_at_position_zero_is_identity() {
        let mut v = [0.25f32, -1.5, 2.0, 3.0];
        rope_half(&mut v, 4, 0, 10_000.0).unwrap();
        assert_eq!(v, [0.25, -1.5, 2.0, 3.0]);
    }

    #[test]
    fn rope_half_rejects_bad_dimensions() {
        let mut v = [0.0f32; 4];
        assert!(matches!(
            rope_half(&mut v, 3, 1, 10_000.0),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            rope_half(&mut v, 6, 1, 10_000.0),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            rope_half(&mut v, 0, 1, 10_000.0),
            Err(EngError::BadRequest(_))
        ));
    }

    #[test]
    fn expect_len_reports_mismatch() {
        assert!(expect_len(&[1.0, 2.0], 2, "x").is_ok());
        assert!(matches!(
            expect_len(&[1.0], 2, "x"),
            Err(EngError::BadRequest(_))
        ));
    }
}
