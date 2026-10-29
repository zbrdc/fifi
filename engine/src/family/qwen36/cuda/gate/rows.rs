//! Per-row summaries for the stage 1 gate: top-32, FNV-1a 64 hash and log-sum-exp of one logit row.
//!
//! Pure host code: no device type and no `CudaContext`, so the tests below run on a CPU-only host.
#![cfg_attr(not(test), expect(dead_code, reason = "staged for the gate loop"))]

/// Number of (id, logit) pairs kept per step.
pub(super) const TOP_K: usize = 32;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a 64 over the little-endian bytes of the row (the definition of the oracle's row hash).
pub(super) fn fnv1a64(row: &[f32]) -> u64 {
    let mut h = FNV_OFFSET;
    for v in row {
        for b in v.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

/// `ln(sum(exp(x)))` in f64. A NaN anywhere gives NaN, and a +inf gives +inf, so the caller can flag the row.
pub(super) fn log_sum_exp(row: &[f32]) -> f64 {
    let max = row
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| if b > a { b } else { a });
    if row.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    if !max.is_finite() {
        return f64::from(max);
    }
    let max = f64::from(max);
    max + row
        .iter()
        .map(|&v| (f64::from(v) - max).exp())
        .sum::<f64>()
        .ln()
}

/// The `k` largest entries, descending by logit, ties to the lower id. A NaN never ranks. A `k` of zero
/// returns no pairs: the scan below compares against `best[k - 1]`, which would index before the start.
pub(super) fn top_k(row: &[f32], k: usize) -> Vec<(u32, f32)> {
    if k == 0 {
        return Vec::new();
    }
    let mut best: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    for (id, &v) in row.iter().enumerate() {
        if v.is_nan() || (best.len() == k && v <= best[k - 1].1) {
            continue;
        }
        let at = best.iter().position(|&(_, w)| w < v).unwrap_or(best.len());
        best.insert(at, (id as u32, v));
        best.truncate(k);
    }
    best
}

/// Everything the raw file records about one row.
pub(super) struct RowSummary {
    pub argmax: u32,
    pub argmax_logit: f32,
    pub top: Vec<(u32, f32)>,
    pub hash: u64,
    pub lse: f64,
    pub finite: bool,
}

pub(super) fn summarize(row: &[f32]) -> RowSummary {
    let top = top_k(row, TOP_K);
    let (argmax, argmax_logit) = top.first().copied().unwrap_or((0, f32::NAN));
    RowSummary {
        argmax,
        argmax_logit,
        top,
        hash: fnv1a64(row),
        lse: log_sum_exp(row),
        finite: row.iter().all(|v| v.is_finite()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1,000 values from an LCG, with a planted tie at ids 10 and 11. `gen_golden.py` computes the expected values
    /// independently in Python.
    fn golden_row() -> Vec<f32> {
        let mut x: u32 = 12345;
        let mut row: Vec<f32> = (0..1000)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (x as i32) as f32 / 268_435_456.0
            })
            .collect();
        row[10] = 9.0;
        row[11] = 9.0;
        row
    }

    #[test]
    fn hash_matches_the_python_golden_value() {
        assert_eq!(
            format!("{:016x}", fnv1a64(&golden_row())),
            "12f192be653d877a"
        );
    }

    #[test]
    fn hash_of_an_empty_row_is_the_offset_basis() {
        assert_eq!(fnv1a64(&[]), FNV_OFFSET);
    }

    #[test]
    fn one_changed_bit_changes_the_hash() {
        let mut row = golden_row();
        let before = fnv1a64(&row);
        row[500] = f32::from_bits(row[500].to_bits() ^ 1);
        assert_ne!(fnv1a64(&row), before);
    }

    #[test]
    fn lse_matches_the_python_golden_value() {
        let got = log_sum_exp(&golden_row());
        assert!((got - 12.223_441_089_000_945).abs() < 1e-9, "{got}");
    }

    #[test]
    fn lse_of_a_single_value_is_that_value() {
        assert!((log_sum_exp(&[3.5]) - 3.5).abs() < 1e-12);
    }

    #[test]
    fn lse_subtracts_the_max_before_the_exponential() {
        // ln(e^100 + e^100) = 100 + ln 2.
        assert!((log_sum_exp(&[100.0, 100.0]) - (100.0 + std::f64::consts::LN_2)).abs() < 1e-9);
        // Above 88.0, where a plain f32 exp overflows; 1000 is beyond f64's exp domain too, so the
        // unstabilized form returns +inf while the max-subtracted form stays exact.
        let got = log_sum_exp(&[1000.0, 999.0]);
        let want = 1000.0 + (1.0 + std::f64::consts::E.recip()).ln();
        assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
    }

    #[test]
    fn lse_of_non_finite_rows_is_not_finite() {
        assert!(log_sum_exp(&[1.0, f32::NAN]).is_nan());
        assert_eq!(log_sum_exp(&[1.0, f32::INFINITY]), f64::INFINITY);
    }

    #[test]
    fn top_k_matches_the_python_golden_order_and_ties_go_to_the_lower_id() {
        let top = top_k(&golden_row(), 32);
        let ids: Vec<u32> = top.iter().map(|t| t.0).collect();
        assert_eq!(&ids[..8], &[10, 11, 23, 900, 840, 990, 6, 338]);
        assert_eq!(top.len(), 32);
        assert!(top.windows(2).all(|w| w[0].1 >= w[1].1));
    }

    #[test]
    fn top_k_of_zero_is_empty_for_any_row() {
        assert!(top_k(&[1.0, 2.0, 3.0], 0).is_empty());
        assert!(top_k(&[], 0).is_empty());
    }

    #[test]
    fn top_k_never_ranks_a_nan_and_handles_short_rows() {
        assert_eq!(
            top_k(&[f32::NAN, 1.0, f32::NAN, 2.0], 3),
            vec![(3, 2.0), (1, 1.0)]
        );
        assert!(top_k(&[], 5).is_empty());
    }

    #[test]
    fn summary_argmax_is_top_zero_and_flags_non_finite_rows() {
        let s = summarize(&golden_row());
        assert_eq!((s.argmax, s.argmax_logit), (10, 9.0));
        assert!(s.finite && s.top[0] == (s.argmax, s.argmax_logit));
        assert_eq!(s.hash, fnv1a64(&golden_row()));
        assert!((s.lse - log_sum_exp(&golden_row())).abs() < 1e-12);
        let mut bad = golden_row();
        bad[7] = f32::NAN;
        assert!(!summarize(&bad).finite);
        assert!(summarize(&bad).lse.is_nan());
    }
}
