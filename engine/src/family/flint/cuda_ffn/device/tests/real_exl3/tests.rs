use super::*;

#[test]
fn cached_head_checks_identity_length_endianness_and_finite_values() {
    let mut bytes = vec![0; VOCAB * 4];
    bytes[4..8].copy_from_slice(&1.0f32.to_le_bytes());
    let sha = format!("{:x}", Sha256::digest(&bytes));
    let logits = cached_head(&bytes, &sha).unwrap();
    assert_eq!(logits.len(), VOCAB);
    assert_eq!(comparison::greedy(&logits).unwrap(), 1);
    assert!(cached_head(&bytes, "wrong SHA").is_err());
    assert!(cached_head(&bytes[..bytes.len() - 1], &sha).is_err());
    for value in [f32::NAN, f32::INFINITY] {
        bytes[..4].copy_from_slice(&value.to_le_bytes());
        let sha = format!("{:x}", Sha256::digest(&bytes));
        assert!(cached_head(&bytes, &sha).is_err());
    }
}

#[test]
fn comparison_uses_distribution_not_a_logit_offset_or_invented_bound() {
    let comparison = comparison::compare(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
    assert!(comparison.first_id_agrees);
    assert_eq!(comparison.kl_gguf_to_exl3, 0.0);
}

#[test]
fn greedy_ties_keep_the_first_token_and_nonfinite_logits_fail() {
    assert_eq!(comparison::greedy(&[3.0, 3.0, 2.0]).unwrap(), 0);
    assert_eq!(comparison::greedy(&[-0.0, 0.0]).unwrap(), 0);
    assert!(comparison::greedy(&[]).is_err());
    assert!(comparison::greedy(&[f32::NAN]).is_err());
    assert!(comparison::greedy(&[f32::INFINITY]).is_err());
}

#[test]
fn comparison_reports_disagreement_and_rejects_bad_logits() {
    let result = comparison::compare(&[0.0, 1.0], &[1.0, 0.0]).unwrap();
    assert!(!result.first_id_agrees);
    assert!(result.kl_gguf_to_exl3.is_finite() && result.kl_gguf_to_exl3 > 0.0);
    assert!(comparison::compare(&[0.0], &[0.0, 1.0]).is_err());
    assert!(comparison::compare(&[0.0], &[f32::NAN]).is_err());
    let huge = comparison::compare(&[1e30, 1e30], &[1e30, 1e30]).unwrap();
    assert_eq!(huge.kl_gguf_to_exl3, 0.0);
}

#[test]
fn divergence_reports_first_changed_id_or_missing_continuation() {
    assert_eq!(
        comparison::first_divergence(&[1, 2, 3], &[1, 4, 3]),
        Some(1)
    );
    assert_eq!(comparison::first_divergence(&[1, 2], &[1]), Some(1));
    assert_eq!(comparison::first_divergence(&[1], &[1, 2]), Some(1));
    assert_eq!(comparison::first_divergence(&[1, 2], &[1, 2]), None);
}
