//! Real-weight qualification is approximate; synthetic BF16 equality is unchanged.

pub(super) fn compare(expected: &[f32], actual: &[f32], width: usize) -> Result<(), String> {
    if width == 0
        || expected.is_empty()
        || expected.len() != actual.len()
        || !expected.len().is_multiple_of(width)
    {
        return Err("invalid real-layer row geometry".into());
    }
    if expected.iter().chain(actual).any(|v| !v.is_finite()) {
        return Err("nonfinite real-layer value".into());
    }
    for (index, (reference, device)) in expected
        .chunks_exact(width)
        .zip(actual.chunks_exact(width))
        .enumerate()
    {
        let mut error = 0.0_f64;
        let mut reference_norm = 0.0_f64;
        let mut device_norm = 0.0_f64;
        let mut dot = 0.0_f64;
        let mut max_abs = 0.0_f64;
        for (&reference, &device) in reference.iter().zip(device) {
            let (reference, device) = (f64::from(reference), f64::from(device));
            error += (device - reference).powi(2);
            reference_norm += reference * reference;
            device_norm += device * device;
            dot += reference * device;
            max_abs = max_abs.max((device - reference).abs());
        }
        let relative_l2 = if reference_norm == 0.0 {
            if device_norm == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            (error / reference_norm).sqrt()
        };
        let cosine = if reference_norm == 0.0 && device_norm == 0.0 {
            1.0
        } else if reference_norm == 0.0 || device_norm == 0.0 {
            0.0
        } else {
            dot / (reference_norm.sqrt() * device_norm.sqrt())
        };
        let diagnostic = format!(
            "row={index} rel_l2={relative_l2:.9e} cosine={cosine:.9e} max_abs={max_abs:.9e}"
        );
        println!("REAL_ROW {diagnostic}");
        if relative_l2 > 0.01 || cosine < 0.9999 {
            return Err(format!("real-layer quality failed: {diagnostic}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collinear_amplitude_error_must_fail_even_with_cosine_one() {
        assert!(
            compare(&[1.0, 2.0], &[2.0, 4.0], 2).is_err(),
            "cosine alone must not accept a relative-L2 error of one"
        );
    }

    #[test]
    fn matching_rows_and_exact_zero_norms_pass() {
        compare(&[1.0, -2.0, 0.0, -0.0], &[1.0, -2.0, -0.0, 0.0], 2).unwrap();
        assert!(compare(&[0.0, 0.0], &[0.0, 1.0], 2).is_err());
        assert!(compare(&[0.0, 1.0], &[0.0, 0.0], 2).is_err());
    }

    #[test]
    fn direction_error_and_a_bad_later_row_fail() {
        let error = compare(&[1.0, 0.0], &[0.0, 1.0], 2).unwrap_err();
        assert!(error.contains("cosine=0.000000000e0"));
        assert!(compare(&[1.0, 2.0, 1.0, 2.0], &[1.0, 2.0, 2.0, 4.0], 2).is_err());
    }

    #[test]
    fn nonfinite_values_in_either_side_fail() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(compare(&[value], &[0.0], 1).is_err());
            assert!(compare(&[0.0], &[value], 1).is_err());
        }
    }

    #[test]
    fn malformed_shapes_fail() {
        assert!(compare(&[], &[], 1).is_err());
        assert!(compare(&[1.0], &[1.0], 0).is_err());
        assert!(compare(&[1.0], &[], 1).is_err());
        assert!(compare(&[1.0], &[1.0], 2).is_err());
    }
}
