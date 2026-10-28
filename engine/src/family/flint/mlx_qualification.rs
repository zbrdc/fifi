//! Test-only direct-forward qualification. Never part of a production binary.

use std::io::{self, Write};

#[path = "mlx_layer_capture.rs"]
mod layer_capture;

fn parse_tokens(raw: &[u8]) -> io::Result<Vec<u32>> {
    if raw.len() > 1024 * 1024 {
        return Err(io::Error::other("token JSON exceeds 1 MiB"));
    }
    let ids: Vec<u32> = serde_json::from_slice(raw)?;
    if ids.len() < 2 {
        return Err(io::Error::other("requires at least 2 raw IDs"));
    }
    let limit = super::factory::context_limit() as usize;
    if ids.len() > limit {
        return Err(io::Error::other(format!(
            "token count {} exceeds the serving context limit {} (FIFI_FLINT_MLX_CONTEXT_LIMIT)",
            ids.len(),
            limit
        )));
    }
    if ids.iter().any(|&id| id >= 150144) {
        return Err(io::Error::other(
            "token ID exceeds the Flint vocabulary (150144)",
        ));
    }
    Ok(ids)
}

fn profile_tokens(raw: &[u8]) -> io::Result<Vec<u32>> {
    let ids = parse_tokens(raw)?;
    if ids.len() < 100 {
        return Err(io::Error::other(
            "profile requires 98 warm tokens, one measured token, and its target",
        ));
    }
    Ok(ids)
}

fn write_row(writer: &mut impl Write, row: &[f32]) -> io::Result<()> {
    if row.len() != 150144 || row.iter().any(|v| !v.is_finite()) {
        return Err(io::Error::other(
            "requires one finite full-vocabulary F32 row",
        ));
    }
    for value in row {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

// KL uses log probabilities so a zero underflowed probability never divides by zero.
pub(super) fn row_kl(batched: &[f32], scalar: &[f32]) -> f64 {
    let normalization = |row: &[f32]| {
        let maximum = f64::from(row.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let log_mass = row
            .iter()
            .map(|&v| (f64::from(v) - maximum).exp())
            .sum::<f64>()
            .ln();
        (maximum, log_mass)
    };
    let (bm, bz) = normalization(batched);
    let (sm, sz) = normalization(scalar);
    batched
        .iter()
        .zip(scalar)
        .map(|(&b, &s)| {
            let log_p = (f64::from(b) - bm) - bz;
            let log_q = (f64::from(s) - sm) - sz;
            log_p.exp() * (log_p - log_q)
        })
        .sum()
}

#[cfg(test)]
mod row_kl_tests {
    use super::row_kl;

    #[test]
    fn identical_rows_are_zero() {
        let row = vec![0.1f32, -2.0, 5.0, -0.5];
        assert!(row_kl(&row, &row).abs() < 1e-12);
    }

    #[test]
    fn two_point_closed_form() {
        // p = [0.7, 0.3], q = [0.5, 0.5]:
        // KL = 0.7*ln(1.4) + 0.3*ln(0.6) = 0.0823...
        let p = vec![0.7f32.ln(), 0.3f32.ln()];
        let q = vec![0.5f32.ln(), 0.5f32.ln()];
        let got = row_kl(&p, &q);
        let want = 0.7 * (0.7_f64 / 0.5).ln() + 0.3 * (0.3_f64 / 0.5).ln();
        assert!((got - want).abs() < 1e-7, "got {got}, want {want}"); // f32 logit precision
    }
}

fn top1_gap(row: &[f32], token: u32) -> f64 {
    let next = row
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != token as usize)
        .map(|(_, &value)| f64::from(value))
        .fold(f64::NEG_INFINITY, f64::max);
    f64::from(row[token as usize]) - next
}

fn identity_metrics(batched: &[Vec<f32>], scalar: &[Vec<f32>]) -> io::Result<serde_json::Value> {
    use crate::family::sampler::Sampler;
    if batched.len() != 33 || scalar.len() != 33 {
        return Err(io::Error::other(
            "identity requires the last prefill row and 32 decode rows",
        ));
    }
    let mut sampler = Sampler::greedy();
    let mut matches = 0;
    let mut first_flip = None;
    let mut divergences = Vec::with_capacity(33);
    for (row, (b, s)) in batched.iter().zip(scalar).enumerate() {
        if b.len() != s.len() {
            return Err(io::Error::other("identity vocabulary widths differ"));
        }
        let err = |e| io::Error::other(format!("{e:?}"));
        let bt = sampler.sample(b).map_err(err)?;
        let st = sampler.sample(s).map_err(err)?;
        if bt != st && first_flip.is_none() {
            first_flip = Some(
                serde_json::json!({"row":row,"predicts_token_position":2048+row,
                "batched_token":bt,"per_token_token":st,
                "batched_top1_gap":top1_gap(b,bt),"per_token_top1_gap":top1_gap(s,st)}),
            );
        }
        matches += usize::from(bt == st);
        divergences.push(row_kl(b, s));
    }
    let mean = divergences.iter().sum::<f64>() / 33.0;
    let maximum = divergences.iter().copied().fold(0.0_f64, f64::max);
    let agreement = matches as f64 / 33.0;
    Ok(
        serde_json::json!({"rows":33,"kl_direction":"batched || per-token",
        "kl_per_row":divergences,"mean_kl":mean,"max_row_kl":maximum,"top1_matches":matches,
        "top1_agreement":agreement,"first_top1_difference":first_flip,
        "passed":mean.is_finite() && mean <= 0.004 && maximum <= 0.05 && agreement >= 0.99}),
    )
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
#[path = "mlx_qualification_native.rs"]
mod native;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(super) use native::{Span, capture, injected_input, wanted};
#[cfg(all(
    feature = "native-mlx",
    feature = "native-fault-qualification",
    target_os = "macos",
    target_arch = "aarch64"
))]
pub(super) use native::{trace_close, trace_open};

#[cfg(test)]
mod tests {
    #[test]
    fn identity_gate_rejects_one_large_same_context_row_despite_small_mean() {
        let control = vec![vec![0.1, 0.0]; 33];
        let mut batched = control.clone();
        batched[3] = vec![0.85, 0.0];
        let metrics = super::identity_metrics(&batched, &control).unwrap();
        assert!(metrics["mean_kl"].as_f64().unwrap() < 0.004);
        assert_eq!(metrics["top1_agreement"], 1.0);
        assert_eq!(metrics["passed"], false);
        assert!(metrics["max_row_kl"].as_f64().unwrap() > 0.05);
    }

    #[test]
    fn identity_gate_has_pinned_direction_offsets_and_stable_tails() {
        let batched = vec![vec![0.75_f32.ln(), 0.25_f32.ln()]; 33];
        let scalar = vec![vec![0.5_f32.ln(), 0.5_f32.ln()]; 33];
        let equal = super::identity_metrics(&batched, &batched).unwrap();
        assert_eq!(equal["mean_kl"], 0.0);
        assert_eq!(equal["max_row_kl"], 0.0);
        assert_eq!(equal["first_top1_difference"], serde_json::Value::Null);
        assert_eq!(equal["passed"], true);
        let drift = super::identity_metrics(&batched, &scalar).unwrap();
        let expected = 0.75 * 1.5_f64.ln() + 0.25 * 0.5_f64.ln();
        assert!((drift["mean_kl"].as_f64().unwrap() - expected).abs() < 1e-7);
        assert_eq!(drift["passed"], false);
        let reverse = super::identity_metrics(&scalar, &batched).unwrap();
        assert!(reverse["mean_kl"].as_f64().unwrap() > expected);
        let tail =
            super::identity_metrics(&vec![vec![0.0, -1000.0]; 33], &vec![vec![-1000.0, 0.0]; 33])
                .unwrap();
        assert_eq!(tail["mean_kl"], 1000.0);
        assert_eq!(tail["top1_agreement"], 0.0);
        assert_eq!(tail["first_top1_difference"]["row"], 0);
        assert_eq!(
            tail["first_top1_difference"]["predicts_token_position"],
            2048
        );
        assert_eq!(tail["first_top1_difference"]["batched_top1_gap"], 1000.0);
        assert_eq!(tail["first_top1_difference"]["per_token_top1_gap"], 1000.0);
        assert!(super::identity_metrics(&batched[..32], &scalar).is_err());
        let mut invalid = scalar;
        invalid[32][1] = f32::NAN;
        assert!(super::identity_metrics(&batched, &invalid).is_err());
        invalid[32] = vec![0.0];
        assert!(super::identity_metrics(&batched, &invalid).is_err());
    }

    #[test]
    fn profile_inputs_require_a_complete_warm_step_and_target() {
        let ids = vec![198_u32; 100];
        assert_eq!(
            super::profile_tokens(&serde_json::to_vec(&ids).unwrap()).unwrap(),
            ids
        );
        assert!(super::profile_tokens(&serde_json::to_vec(&ids[..99]).unwrap()).is_err());
        let mut invalid = ids;
        invalid[98] = 150144;
        assert!(super::profile_tokens(&serde_json::to_vec(&invalid).unwrap()).is_err());
    }

    #[test]
    fn raw_teacher_forcing_ids_and_finite_full_rows_are_checked() {
        assert_eq!(super::parse_tokens(b"[150000,7]").unwrap(), [150000, 7]);
        for invalid in [b"[]".as_slice(), b"[1]", b"[1,150144]", b"[1,-1]"] {
            assert!(super::parse_tokens(invalid).is_err());
        }
        let mut bytes = Vec::new();
        let mut row = vec![0.0_f32; 150144];
        row[0] = 1.25;
        row[150143] = -2.5;
        super::write_row(&mut bytes, &row).unwrap();
        assert_eq!(bytes.len(), 600576);
        assert_eq!(&bytes[..4], &[0, 0, 160, 63]);
        assert_eq!(&bytes[600572..], &[0, 0, 32, 192]);
        assert!(super::write_row(&mut Vec::new(), &[0.0]).is_err());
        row[13] = f32::NAN;
        let mut refused = Vec::new();
        assert!(super::write_row(&mut refused, &row).is_err());
        assert!(refused.is_empty());
    }
}
