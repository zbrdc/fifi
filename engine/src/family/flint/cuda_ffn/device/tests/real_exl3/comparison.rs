use super::*;

pub(super) const LABEL: &str = "NOT SAME-PACK ORACLE";

pub(super) struct HeadComparison {
    pub(super) first_id_agrees: bool,
    pub(super) kl_gguf_to_exl3: f64,
}

pub(super) fn greedy(logits: &[f32]) -> Result<u32, EngError> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err(bad("diagnostic logits must be nonempty and finite"));
    }
    Ok(logits
        .iter()
        .enumerate()
        .max_by(|(i, a), (j, b)| a.partial_cmp(b).unwrap().then_with(|| j.cmp(i)))
        .unwrap()
        .0 as u32)
}

pub(super) fn compare(gguf: &[f32], exl3: &[f32]) -> Result<HeadComparison, EngError> {
    if gguf.len() != exl3.len() {
        return Err(bad("GGUF and EXL3 logit vocabularies differ"));
    }
    let first_id_agrees = greedy(gguf)? == greedy(exl3)?;
    let p = log_probabilities(gguf);
    let q = log_probabilities(exl3);
    let kl_gguf_to_exl3 = p.iter().zip(q).map(|(p, q)| p.exp() * (p - q)).sum();
    Ok(HeadComparison {
        first_id_agrees,
        kl_gguf_to_exl3,
    })
}

pub(super) fn first_divergence(gguf: &[u32], exl3: &[u32]) -> Option<usize> {
    gguf.iter()
        .zip(exl3)
        .position(|(a, b)| a != b)
        .or_else(|| (gguf.len() != exl3.len()).then_some(gguf.len().min(exl3.len())))
}

fn log_probabilities(logits: &[f32]) -> Vec<f64> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut shifted = logits
        .iter()
        .map(|v| f64::from(*v) - maximum)
        .collect::<Vec<_>>();
    let log_sum = shifted.iter().map(|v| v.exp()).sum::<f64>().ln();
    for value in &mut shifted {
        *value -= log_sum;
    }
    shifted
}
