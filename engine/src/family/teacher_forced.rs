//! Teacher-forced scoring of a fixed token stream, shared by the quality legs of every family.
//!
//! A leg prefills the tokens before each window's scored region, feeds each true token, and reads
//! the full logits row of the next position. A row is narrowed to the sampler vocabulary and
//! scored in f64: the true-token log-probability, the greedy argmax (ties to the lower id) and the
//! [`TOP_ROWS`] largest entries go to a per-window KVQ1 sidecar. When a reference run's sidecars
//! are named, a sidecar also holds the divergence of the reference distribution from its own over
//! the reference's stored top rows (no bound on the whole-vocabulary KL, the terms outside being
//! signed) and the reference's mass outside them. `scripts/kld_vs_llamacpp/score.py` pairs the
//! sidecars of two runs.
//!
//! `Run::from_env` (CUDA builds, beside the legs) reads the stream and the windows from the
//! environment: `KVQ_IDS` and `KVQ_IDS_SHA` (the little-endian u32 id stream and its SHA-256),
//! `KVQ_LAYOUT` (the windows), `KVQ_WINDOWS` (`lo:hi`, a range of window indices), `KVQ_OUT`,
//! and optionally `KVQ_REF`.

use crate::EngError;
use std::time::Instant;

#[cfg(feature = "native-cuda")]
mod run;
#[cfg(feature = "native-cuda")]
pub(crate) use run::{Run, Window, score_windows};

/// Rows kept per position in the sidecar.
pub(crate) const TOP_ROWS: usize = 256;
/// Bytes of one position record: true id, log-probability of the true id, argmax id, the top
/// ids and their log-probabilities, the divergence over the reference's top rows, and the
/// reference's tail mass.
const RECORD_BYTES: usize = 4 + 4 + 4 + TOP_ROWS * 8 + 4 + 4;
/// Magic, window index, positions, top rows.
const HEADER_BYTES: usize = 16;
const MAGIC: &[u8; 4] = b"KVQ1";

fn bad(text: impl std::fmt::Display) -> EngError {
    EngError::BadRequest(format!("teacher-forced scoring: {text}"))
}

/// `ln(sum(exp(row)))` in f64, stable at the max. An all-NaN or empty row
/// is a caller bug the runner rejects; this returns -inf / NaN faithfully.
pub(crate) fn log_sum_exp(row: &[f32]) -> f64 {
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

/// The log-probability of `true_id` under the row: `logit - lse`. A `true_id`
/// outside the row is a layout bug, refused loudly.
pub(crate) fn logprob_true(row: &[f32], true_id: u32) -> Result<f64, &'static str> {
    let logit = f64::from(
        row.get(true_id as usize)
            .copied()
            .ok_or("the true token id is outside the logits row")?,
    );
    Ok(logit - log_sum_exp(row))
}

/// The greedy argmax, ties to the lower id, NaN never ranks. An empty row or
/// an all-NaN row is refused (the runner treats it as a device failure).
pub(crate) fn argmax_low_id(row: &[f32]) -> Result<u32, &'static str> {
    let mut best: Option<(u32, f32)> = None;
    for (id, &v) in row.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        match best {
            Some((_, b)) if v <= b => {}
            _ => best = Some((id as u32, v)),
        }
    }
    best.map(|(id, _)| id)
        .ok_or("no finite entry in the logits row")
}

/// The `TOP_ROWS` largest entries of `row`, as ids (ties to the lower id) and their
/// log-probabilities under `lse`.
fn top_rows(row: &[f32], lse: f64) -> (Vec<u32>, Vec<f32>) {
    let mut order: Vec<u32> = (0..row.len() as u32).collect();
    let by_value = |a: &u32, b: &u32| {
        row[*b as usize]
            .total_cmp(&row[*a as usize])
            .then_with(|| a.cmp(b))
    };
    order.select_nth_unstable_by(TOP_ROWS, by_value);
    order.truncate(TOP_ROWS);
    order.sort_by(by_value);
    let logprobs = order
        .iter()
        .map(|&i| (f64::from(row[i as usize]) - lse) as f32)
        .collect();
    (order, logprobs)
}

/// The sidecar of one window, filled one scored position at a time.
pub(crate) struct Sidecar {
    window: usize,
    positions: usize,
    scored: usize,
    reference: Option<Vec<u8>>,
    bytes: Vec<u8>,
    started: Instant,
}

impl Sidecar {
    fn new(window: usize, positions: usize, reference: Option<Vec<u8>>) -> Result<Self, EngError> {
        let length = HEADER_BYTES + positions * RECORD_BYTES;
        let mut bytes = Vec::with_capacity(length);
        bytes.extend_from_slice(MAGIC);
        for word in [window, positions, TOP_ROWS] {
            bytes.extend_from_slice(&(word as u32).to_le_bytes());
        }
        if let Some(reference) = &reference
            && (reference.len() != length || reference[..HEADER_BYTES] != bytes[..])
        {
            return Err(bad(format!(
                "window {window}: the reference sidecar is not one of {positions} positions"
            )));
        }
        Ok(Self {
            window,
            positions,
            scored: 0,
            reference,
            bytes,
            started: Instant::now(),
        })
    }

    /// Scores the next position from its logits `row`, narrowed to the sampler vocabulary, and
    /// the true id that follows.
    pub(crate) fn push(&mut self, row: &[f32], true_id: u32) -> Result<(), EngError> {
        let (w, index) = (self.window, self.scored);
        if index == self.positions {
            return Err(bad(format!("window {w}: more rows than scored positions")));
        }
        if row.iter().any(|v| !v.is_finite()) {
            return Err(bad(format!(
                "window {w} position {index}: a non-finite logit"
            )));
        }
        let lse = log_sum_exp(row);
        let true_logprob = logprob_true(row, true_id).map_err(bad)? as f32;
        let argmax = argmax_low_id(row).map_err(bad)?;
        let (top_ids, top_logprobs) = top_rows(row, lse);
        let (divergence, tail) = match &self.reference {
            Some(bytes) => {
                let at = HEADER_BYTES + index * RECORD_BYTES;
                let record = &bytes[at..at + RECORD_BYTES];
                let word = |i: usize| <[u8; 4]>::try_from(&record[4 * i..4 * i + 4]).unwrap();
                if u32::from_le_bytes(word(0)) != true_id {
                    return Err(bad(format!(
                        "window {w} position {index}: the reference scored another id"
                    )));
                }
                let (mut kl, mut mass) = (0f64, 0f64);
                for i in 0..TOP_ROWS {
                    let id = u32::from_le_bytes(word(3 + i)) as usize;
                    let lp = f64::from(f32::from_le_bytes(word(3 + TOP_ROWS + i)));
                    let p_ref = lp.exp();
                    let q_lp = f64::from(*row.get(id).ok_or_else(|| {
                        bad(format!("window {w}: a reference id is outside the row"))
                    })?) - lse;
                    kl += p_ref * (lp - q_lp);
                    mass += p_ref;
                }
                (kl as f32, (1.0 - mass) as f32)
            }
            None => (0.0, 0.0),
        };
        let out = &mut self.bytes;
        out.extend_from_slice(&true_id.to_le_bytes());
        out.extend_from_slice(&true_logprob.to_le_bytes());
        out.extend_from_slice(&argmax.to_le_bytes());
        top_ids
            .iter()
            .for_each(|id| out.extend_from_slice(&id.to_le_bytes()));
        top_logprobs
            .iter()
            .for_each(|lp| out.extend_from_slice(&lp.to_le_bytes()));
        out.extend_from_slice(&divergence.to_le_bytes());
        out.extend_from_slice(&tail.to_le_bytes());
        self.scored += 1;
        if index % 256 == 0 {
            println!(
                "kvq window {w} scored {index}/{} [{:.0}s]",
                self.positions,
                self.started.elapsed().as_secs_f64()
            );
        }
        Ok(())
    }

    fn finish(self) -> Result<Vec<u8>, EngError> {
        if self.scored != self.positions {
            return Err(bad(format!(
                "window {}: {} of {} positions scored",
                self.window, self.scored, self.positions
            )));
        }
        println!(
            "kvq window {} done [{:.0}s]",
            self.window,
            self.started.elapsed().as_secs_f64()
        );
        Ok(self.bytes)
    }
}

#[cfg(test)]
mod tests;
