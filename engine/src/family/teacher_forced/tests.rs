use super::*;

/// Hand-computed golden rows: exps and sums small enough to verify by
/// hand, checked against ln identities rather than restating the code.
#[test]
fn golden_rows_match_hand_computed_probabilities() {
    // exps [1, 2, 1], sum 4: P(1) = 1/2, P(0) = 1/4.
    let row = [0.0f32, 2.0f32.ln(), 0.0];
    let lp1 = logprob_true(&row, 1).unwrap();
    let lp0 = logprob_true(&row, 0).unwrap();
    // The input logits are f32 (ln 2 is rounded), so 1e-6, not 1e-12.
    assert!((lp1 - (0.5f64).ln()).abs() < 1e-6);
    assert!((lp0 - (0.25f64).ln()).abs() < 1e-6);
    assert_eq!(argmax_low_id(&row), Ok(1));
    // Probabilities sum to 1 across the row.
    let total: f64 = (0..3).map(|i| logprob_true(&row, i).unwrap().exp()).sum();
    assert!((total - 1.0).abs() < 1e-6);
}

/// Ties break to the LOWER id, and NaN never ranks even at id 0.
#[test]
fn ties_break_low_and_nan_never_ranks() {
    assert_eq!(argmax_low_id(&[1.0, 1.0, 2.0]), Ok(2));
    assert_eq!(argmax_low_id(&[3.0, 3.0, 1.0]), Ok(0));
    assert_eq!(argmax_low_id(&[f32::NAN, 5.0, 5.0]), Ok(1));
    assert!(argmax_low_id(&[f32::NAN, f32::NAN]).is_err());
}

/// The padded-vocab rule: padding rows participate in normalization
/// identically in both arms. exps [e,e,e,e,e,1,1,1] over width 8 with a
/// true vocabulary of 5: P(true) = e / (5e + 3), never renormalized to
/// the 5-row universe.
#[test]
fn padding_participates_in_normalization() {
    let row = [1.0f32; 5]
        .iter()
        .copied()
        .chain([0.0f32; 3])
        .collect::<Vec<_>>();
    let lp = logprob_true(&row, 2).unwrap();
    let want = 1.0f64 - (5.0 * 1.0f64.exp() + 3.0).ln();
    assert!((lp - want).abs() < 1e-12, "got {lp}, want {want}");
    assert_eq!(argmax_low_id(&row), Ok(0));
}

/// A true id outside the row is refused, an empty row is refused, and a
/// NaN row propagates NaN instead of inventing a value.
#[test]
fn layout_and_device_bugs_are_refused() {
    assert!(logprob_true(&[0.0, 0.0], 2).is_err());
    assert!(logprob_true(&[], 0).is_err());
    assert!(argmax_low_id(&[]).is_err());
    let nan = [f32::NAN, 0.0];
    assert!(logprob_true(&nan, 1).unwrap().is_nan());
    assert_eq!(argmax_low_id(&nan), Ok(1));
}

fn logit_rows(positions: usize, width: usize, seed: u32) -> Vec<Vec<f32>> {
    (0..positions)
        .map(|p| {
            (0..width)
                .map(|i| {
                    let x = (i as u32).wrapping_mul(2_654_435_761) ^ ((p as u32 + seed) * 40_503);
                    (x % 1000) as f32 / 100.0
                })
                .collect()
        })
        .collect()
}

fn sidecar(rows: &[Vec<f32>], ids: &[u32], reference: Option<Vec<u8>>) -> Vec<u8> {
    let mut sidecar = Sidecar::new(3, rows.len(), reference).unwrap();
    for (row, &id) in rows.iter().zip(ids) {
        sidecar.push(row, id).unwrap();
    }
    sidecar.finish().unwrap()
}

fn word(bytes: &[u8], at: usize) -> [u8; 4] {
    bytes[at..at + 4].try_into().unwrap()
}

/// The record layout score.py reads: the header, then per position the true id, its
/// log-probability, the argmax, the top ids in falling order with their log-probabilities, the
/// divergence and the tail mass; a run against itself has no divergence.
#[test]
fn a_sidecar_holds_the_records_score_py_reads() {
    let (rows, ids) = (logit_rows(3, 300, 1), [7u32, 0, 299]);
    let alone = sidecar(&rows, &ids, None);
    assert_eq!(alone.len(), HEADER_BYTES + 3 * RECORD_BYTES);
    assert_eq!(&alone[..4], b"KVQ1");
    assert_eq!(
        [1, 2, 3].map(|i| u32::from_le_bytes(word(&alone, 4 * i))),
        [3, 3, 256]
    );
    for (p, row) in rows.iter().enumerate() {
        let at = HEADER_BYTES + p * RECORD_BYTES;
        let lse = log_sum_exp(row);
        assert_eq!(u32::from_le_bytes(word(&alone, at)), ids[p]);
        let lp = f32::from_le_bytes(word(&alone, at + 4));
        assert!((f64::from(lp) - (f64::from(row[ids[p] as usize]) - lse)).abs() < 1e-5);
        let argmax = u32::from_le_bytes(word(&alone, at + 8));
        assert_eq!(argmax, argmax_low_id(row).unwrap());
        let top: Vec<u32> = (0..TOP_ROWS)
            .map(|i| u32::from_le_bytes(word(&alone, at + 12 + 4 * i)))
            .collect();
        assert_eq!(top[0], argmax);
        assert!(top.windows(2).all(|pair| {
            let (a, b) = (row[pair[0] as usize], row[pair[1] as usize]);
            a > b || (a == b && pair[0] < pair[1])
        }));
        let tail = at + 12 + 8 * TOP_ROWS;
        assert_eq!(word(&alone, tail), [0; 4]);
        assert_eq!(word(&alone, tail + 4), [0; 4]);
    }
    let again = sidecar(&rows, &ids, Some(alone.clone()));
    for p in 0..3 {
        let tail = HEADER_BYTES + p * RECORD_BYTES + 12 + 8 * TOP_ROWS;
        let kl = f32::from_le_bytes(word(&again, tail));
        let mass = f32::from_le_bytes(word(&again, tail + 4));
        assert!(kl.abs() < 1e-6, "kl {kl}");
        assert!(mass > 0.0 && mass < 1.0, "tail mass {mass}");
    }
    let other = sidecar(&logit_rows(3, 300, 9), &ids, Some(alone.clone()));
    let tail = HEADER_BYTES + 12 + 8 * TOP_ROWS;
    assert!(f32::from_le_bytes(word(&other, tail)) > 0.0);
}

/// A reference of another shape or another id stream, a non-finite row, and a short or long
/// window are refused.
#[test]
fn a_sidecar_refuses_what_it_cannot_pair() {
    let (rows, ids) = (logit_rows(2, 300, 1), [7u32, 8]);
    let alone = sidecar(&rows, &ids, None);
    assert!(Sidecar::new(3, 3, Some(alone.clone())).is_err());
    assert!(Sidecar::new(4, 2, Some(alone.clone())).is_err());
    let mut other = Sidecar::new(3, 2, Some(alone)).unwrap();
    assert!(other.push(&rows[0], 8).is_err());
    let mut short = Sidecar::new(3, 2, None).unwrap();
    let mut bad_row = rows[0].clone();
    bad_row[5] = f32::INFINITY;
    assert!(short.push(&bad_row, 7).is_err());
    short.push(&rows[0], 7).unwrap();
    assert!(short.finish().is_err());
    let mut long = Sidecar::new(3, 1, None).unwrap();
    long.push(&rows[0], 7).unwrap();
    assert!(long.push(&rows[1], 8).is_err());
}
