//! Depth residual blocks are immutable once a layer boundary publishes them.

use std::sync::{Arc, OnceLock};

use super::{MIX_EPS, add, dot, rms_norm};
use crate::kernels::portable_exp::portable_exp;

/// The current residual row and the sealed depth blocks. Each sealed block is stored
/// RMS-normalized once, when its boundary publishes it, and the normalized current row is
/// cached until the current row changes; every mix reads the same bits it would compute
/// from the raw rows.
#[derive(Clone)]
pub(super) struct Stream {
    current: Vec<f32>,
    sealed: Vec<Arc<Vec<f32>>>,
    // A lock rather than a cell so prompt rows can be mixed on several host threads.
    normalized: OnceLock<Arc<Vec<f32>>>,
}

impl Stream {
    pub(super) fn new(current: Vec<f32>) -> Self {
        Self {
            current,
            sealed: Vec::new(),
            normalized: OnceLock::new(),
        }
    }

    /// A stream from its current row and its sealed rows, already normalized, oldest first.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn from_parts(current: Vec<f32>, sealed: Vec<Vec<f32>>) -> Self {
        Self {
            current,
            sealed: sealed.into_iter().map(Arc::new).collect(),
            normalized: OnceLock::new(),
        }
    }

    pub(super) fn block_count(&self) -> usize {
        self.sealed.len()
    }

    pub(super) fn current(&self) -> &[f32] {
        &self.current
    }

    pub(super) fn into_current(self) -> Vec<f32> {
        self.current
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn append_device_state(&self, bytes: &mut Vec<u8>) {
        let at = bytes.len();
        bytes.resize(at + self.device_state_bytes(), 0);
        self.write_device_state(&mut bytes[at..]);
    }

    /// The bytes of the device state: the current row, then the sealed rows in order.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn device_state_bytes(&self) -> usize {
        (self.current.len() + self.sealed.iter().map(|row| row.len()).sum::<usize>())
            * size_of::<f32>()
    }

    /// Writes the device state into `out`, which holds [`Self::device_state_bytes`] bytes.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn write_device_state(&self, out: &mut [u8]) {
        let values = std::iter::once(&self.current[..])
            .chain(self.sealed.iter().map(|row| &row[..]))
            .flatten();
        for (word, value) in out.as_chunks_mut::<4>().0.iter_mut().zip(values) {
            *word = value.to_ne_bytes();
        }
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn publish_resident_current(&mut self, next: Vec<f32>, seal: bool) {
        if seal {
            // Keep the host's sealed blocks canonical for the next host visit.
            self.seal(next);
        } else {
            self.normalized.take();
            self.current = next;
        }
    }

    pub(super) fn add_current(&mut self, row: &[f32]) {
        self.normalized.take();
        add(&mut self.current, row, 1.0);
    }

    pub(super) fn seal(&mut self, next: Vec<f32>) {
        let mut previous = std::mem::replace(&mut self.current, next);
        let normalized = self.normalized.take().unwrap_or_else(|| {
            rms_norm(&mut previous, MIX_EPS);
            Arc::new(previous)
        });
        self.sealed.push(normalized);
    }

    pub(super) fn mix(&self, first_block: usize, query: &[f32]) -> Vec<f32> {
        let last = self
            .normalized
            .get_or_init(|| Arc::new(normalize(&self.current)));
        let rows = self.sealed[first_block..]
            .iter()
            .rev()
            .map(|row| row.as_slice())
            .chain(std::iter::once(last.as_slice()));
        mix_rows(rows, query)
    }
}

fn normalize(row: &[f32]) -> Vec<f32> {
    let mut copy = row.to_vec();
    rms_norm(&mut copy, MIX_EPS);
    copy
}

fn mix_rows<'a>(rows: impl Iterator<Item = &'a [f32]> + Clone, query: &[f32]) -> Vec<f32> {
    let scale = (query.len() as f32).sqrt().recip();
    static BATCH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let scores: Vec<_> = if *BATCH
        .get_or_init(|| std::env::var("FIFI_INKLING_HOST_DEPTH_SCORE4").as_deref() == Ok("1"))
    {
        score_rows4(rows.clone(), query, scale)
    } else {
        rows.clone().map(|row| dot(row, query) * scale).collect()
    };
    let peak = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    weighted_mix(scores.into_iter().zip(rows), query.len(), peak)
}

fn score_rows4<'a>(
    mut rows: impl Iterator<Item = &'a [f32]>,
    query: &[f32],
    scale: f32,
) -> Vec<f32> {
    let mut scores = Vec::with_capacity(rows.size_hint().0);
    loop {
        let group = std::array::from_fn(|_| rows.next());
        let [Some(a), Some(b), Some(c), Some(d)] = group else {
            scores.extend(
                group
                    .into_iter()
                    .flatten()
                    .map(|row| dot(row, query) * scale),
            );
            return scores;
        };
        // Independent row accumulators permit parallel execution without reassociating any dot.
        let mut sums = [-0.0; 4];
        for ((((q, a), b), c), d) in query.iter().zip(a).zip(b).zip(c).zip(d) {
            sums[0] += a * q;
            sums[1] += b * q;
            sums[2] += c * q;
            sums[3] += d * q;
        }
        scores.extend(sums.map(|sum| sum * scale));
    }
}

fn weighted_mix<'a>(
    rows: impl Iterator<Item = (f32, &'a [f32])>,
    width: usize,
    peak: f32,
) -> Vec<f32> {
    let mut mass = 0.0;
    let mut output = vec![0.0; width];
    for (score, row) in rows {
        let weight = portable_exp(score - peak);
        mass += weight;
        add(&mut output, row, weight);
    }
    for value in &mut output {
        *value /= mass;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_mix_can_exclude_the_first_core_boundary() {
        let mut stream = Stream::new(vec![0.0, 2.0]);
        stream.seal(vec![2.0, 0.0]);
        let both = stream.mix(0, &[0.0; 2]);
        let current = stream.mix(1, &[0.0; 2]);
        assert!((both[0] - both[1]).abs() < 1e-6);
        assert!((current[0] - std::f32::consts::SQRT_2).abs() < 1e-6);
        assert_eq!(current[1], 0.0);
    }

    /// The mix from raw rows: normalize every block at each mix, as the stored form must match.
    fn raw_mix(rows: &[Vec<f32>], current: &[f32], first: usize, query: &[f32]) -> Vec<f32> {
        let normalized: Vec<_> = rows[first..]
            .iter()
            .rev()
            .map(|row| normalize(row))
            .chain(std::iter::once(normalize(current)))
            .collect();
        mix_rows(normalized.iter().map(Vec::as_slice), query)
    }

    #[test]
    fn stored_blocks_mix_to_the_same_bits_as_raw_rows_across_branch_and_invalidation() {
        let input = vec![3.0, -6.0, 0.75, -0.0];
        let mut raw = (Vec::new(), input.clone());
        let mut stream = Stream::new(input);
        for new in [[-1.0, -0.0, 0.0, 2.0], [0.0, -4.0, 0.5, 0.0]] {
            same_bits(&raw, &stream);
            raw.0.push(std::mem::replace(&mut raw.1, new.to_vec()));
            stream.seal(new.to_vec());
        }
        same_bits(&raw, &stream);
        let mut raw_fork = raw.clone();
        let mut fork = stream.clone();
        add(&mut raw_fork.1, &[0.0, 2.0, -8.0, 1.0], 1.0);
        fork.add_current(&[0.0, 2.0, -8.0, 1.0]);
        same_bits(&raw_fork, &fork);
        same_bits(&raw, &stream);
        add(&mut raw_fork.1, &[0.0, 3.0, 0.0, -1.0], 1.0);
        fork.add_current(&[0.0, 3.0, 0.0, -1.0]);
        raw_fork
            .0
            .push(std::mem::replace(&mut raw_fork.1, vec![0.0; 4]));
        fork.seal(vec![0.0; 4]);
        same_bits(&raw_fork, &fork);
    }

    #[test]
    fn interleaved_scores_preserve_scalar_reductions_and_tail_order() {
        let q = [0.5, 1.0, -3.0, 1.0, -0.0];
        let rows: Vec<_> = (0..11)
            .map(|i| [1e10, i as f32, -0.25, -5e9, 0.0])
            .collect();
        for n in 0..=rows.len() {
            let iter = rows[..n].iter().rev().map(|r| r.as_slice());
            let actual = score_rows4(iter.clone(), &q, 0.37);
            let expected = iter.map(|r| (dot(r, &q) * 0.37).to_bits());
            assert!(actual.into_iter().map(f32::to_bits).eq(expected));
        }
    }

    fn same_bits((rows, current): &(Vec<Vec<f32>>, Vec<f32>), stream: &Stream) {
        assert_eq!(current.as_slice(), stream.current());
        assert_eq!(rows.len(), stream.block_count());
        for first in 0..=rows.len() {
            for query in [[0.0; 4], [2.0, -1.0, 0.0, 3.0], [-5.0, 4.0, 0.0, 0.0]] {
                let bits = |row: Vec<f32>| row.into_iter().map(f32::to_bits).collect::<Vec<_>>();
                assert_eq!(
                    bits(raw_mix(rows, current, first, &query)),
                    bits(stream.mix(first, &query))
                );
            }
        }
    }
}
