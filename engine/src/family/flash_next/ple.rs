//! Flash-Next PLE n-gram decode step (design section "PLE").

use super::math::{dot, expect_len, rms0, silu};
use crate::EngError;
use crate::family::project::Projector;

#[derive(Debug, Clone, Copy)]
pub(crate) struct PleDims {
    pub(crate) branches: u32,
    pub(crate) hidden: u32,
    /// Only 3 is supported.
    pub(crate) ngram: u32,
    pub(crate) heads_per_ngram: u32,
    pub(crate) head_dim: u32,
    pub(crate) conv_kernel: u32,
    pub(crate) eos: u32,
    pub(crate) eps: f32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PleShape {
    dims: PleDims,
    width: usize,
    heads: usize,
    embed: usize,
    ring_len: usize,
}

impl PleShape {
    pub(crate) fn new(dims: PleDims) -> Result<Self, EngError> {
        if dims.branches == 0 || dims.hidden == 0 || dims.heads_per_ngram == 0 || dims.head_dim == 0
        {
            return Err(EngError::Checkpoint(
                "ple dimensions must be nonzero".to_owned(),
            ));
        }
        if dims.ngram != 3 {
            return Err(EngError::Checkpoint("ple ngram must be 3".to_owned()));
        }
        if dims.conv_kernel < 2 {
            return Err(EngError::Checkpoint(
                "ple conv kernel must be at least 2".to_owned(),
            ));
        }
        if !dims.eps.is_finite() || dims.eps < 0.0 {
            return Err(EngError::Checkpoint(
                "ple epsilon must be finite and nonnegative".to_owned(),
            ));
        }
        let overflow = || EngError::Checkpoint("ple dimensions overflow".to_owned());
        let width = (dims.branches as usize)
            .checked_mul(dims.hidden as usize)
            .ok_or_else(overflow)?;
        let heads = ((dims.ngram - 1) as usize)
            .checked_mul(dims.heads_per_ngram as usize)
            .ok_or_else(overflow)?;
        let embed = heads
            .checked_mul(dims.head_dim as usize)
            .ok_or_else(overflow)?;
        let ring_len = width
            .checked_mul((dims.conv_kernel - 1) as usize)
            .and_then(|v| v.checked_mul(dims.ngram as usize))
            .ok_or_else(overflow)?;
        Ok(Self {
            dims,
            width,
            heads,
            embed,
            ring_len,
        })
    }

    pub(crate) fn dims(&self) -> PleDims {
        self.dims
    }

    /// Width `heads * head_dim` of the gathered n-gram embedding that the projections read.
    pub(crate) fn embed(&self) -> usize {
        self.embed
    }
}

pub(crate) struct PleWeights<'a, W> {
    pub(crate) multipliers: &'a [i64],
    /// Vocabulary size of each head, length `heads`.
    pub(crate) vocab: &'a [i64],
    /// First global row of each head, length `heads`.
    pub(crate) offsets: &'a [i64],
    /// Row tables in global row order, each `[rows, head_dim]`.
    pub(crate) shards: &'a [W],
    /// Row count of each table, same length as `shards`.
    pub(crate) shard_rows: &'a [u64],
    pub(crate) weight_scale: f32,
    /// Key projection `[width, heads * head_dim]`.
    pub(crate) key: &'a W,
    /// Value projection `[hidden, heads * head_dim]`.
    pub(crate) value: &'a W,
    pub(crate) norm_key: &'a [f32],
    pub(crate) norm_query: &'a [f32],
    pub(crate) norm_conv: &'a [f32],
    /// Convolution weights, length `width * conv_kernel`, row `c` at `c * conv_kernel`.
    pub(crate) conv: &'a [f32],
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PleState {
    history: Vec<u32>,
    ring: Vec<f32>,
}

impl PleState {
    pub(crate) fn new(shape: &PleShape) -> Self {
        Self {
            history: Vec::new(),
            ring: vec![0.0; shape.ring_len],
        }
    }

    pub(crate) fn byte_len(&self) -> usize {
        self.history.len() * 4 + self.ring.len() * 4
    }

    /// Installs a saved `(history, ring)` pair, as a prefix restore brings the feed back to a
    /// boundary. Refuses a ring of the wrong length; the history is whatever the boundary held.
    pub(crate) fn resume(&mut self, history: &[u32], ring: &[f32]) -> Result<(), EngError> {
        if ring.len() != self.ring.len() {
            return Err(EngError::BadRequest(format!(
                "the PLE ring holds {} values, not {}",
                self.ring.len(),
                ring.len()
            )));
        }
        self.history.clear();
        self.history.extend_from_slice(history);
        self.ring.copy_from_slice(ring);
        Ok(())
    }

    /// The ring values in host layout: the resume snapshot's host half.
    pub(crate) fn ring_values(&self) -> &[f32] {
        &self.ring
    }

    /// The convolution ring, row `c` at `c * (conv_kernel - 1) * ngram` (test observation only).
    #[cfg(test)]
    pub(crate) fn ring(&self) -> &[f32] {
        &self.ring
    }

    /// The last tokens fed, at most two; empty at the start of a sequence and after end-of-sequence.
    pub(crate) fn history(&self) -> &[u32] {
        &self.history
    }
}

fn hash_rows(
    shape: &PleShape,
    weights_vocab: &[i64],
    offsets: &[i64],
    multipliers: &[i64],
    total_rows: u64,
    history: &[u32],
    token: u32,
) -> Result<Vec<u64>, EngError> {
    let eos = shape.dims.eos;
    let hp = shape.dims.heads_per_ngram as usize;
    let p1 = history.last().copied().unwrap_or(eos);
    let p2 = match history {
        [first, _] => *first,
        _ => eos,
    };
    let mut rows = Vec::with_capacity(weights_vocab.len());
    for (h, (&vocab, &offset)) in weights_vocab.iter().zip(offsets).enumerate() {
        let mut x = u64::from(token).wrapping_mul(multipliers[0] as u64)
            ^ u64::from(p1).wrapping_mul(multipliers[1] as u64);
        if h >= hp {
            x ^= u64::from(p2).wrapping_mul(multipliers[2] as u64);
        }
        if vocab <= 0 {
            return Err(EngError::Checkpoint(
                "ple head vocabulary must exceed zero".to_owned(),
            ));
        }
        let r = (x as i64).rem_euclid(vocab);
        let row = match offset.checked_add(r) {
            Some(row) if row >= 0 => row,
            _ => return Err(EngError::Checkpoint("ple row is invalid".to_owned())),
        };
        let row = row as u64;
        if row >= total_rows {
            return Err(EngError::Checkpoint(
                "ple row is outside the tables".to_owned(),
            ));
        }
        rows.push(row);
    }
    Ok(rows)
}

fn locate_row(shard_rows: &[u64], row: u64) -> Option<(usize, u64)> {
    let mut start = 0u64;
    for (p, &count) in shard_rows.iter().enumerate() {
        let end = start.checked_add(count)?;
        if row < end {
            return Some((p, row - start));
        }
        start = end;
    }
    None
}

/// Hashing stays on the host.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct RowPlan {
    /// `(table index, local row)` for each head, in head order.
    pub(crate) rows: Vec<(usize, u32)>,
    /// The history after `token`: empty after the end-of-sequence token, otherwise the last two tokens.
    pub(crate) history: Vec<u32>,
}

pub(crate) fn plan_rows<W>(
    shape: &PleShape,
    weights: &PleWeights<'_, W>,
    history: &[u32],
    token: u32,
) -> Result<RowPlan, EngError> {
    let dims = shape.dims;
    if weights.multipliers.len() != dims.ngram as usize
        || weights.vocab.len() != shape.heads
        || weights.offsets.len() != shape.heads
        || weights.shards.len() != weights.shard_rows.len()
        || weights.shards.is_empty()
    {
        return Err(EngError::BadRequest(
            "ple table lengths do not match the shape".to_owned(),
        ));
    }
    let total_rows = weights
        .shard_rows
        .iter()
        .try_fold(0u64, |sum, &rows| sum.checked_add(rows))
        .ok_or_else(|| EngError::Checkpoint("ple table rows overflow".to_owned()))?;
    let global = hash_rows(
        shape,
        weights.vocab,
        weights.offsets,
        weights.multipliers,
        total_rows,
        history,
        token,
    )?;
    let mut rows = Vec::with_capacity(global.len());
    for &row in &global {
        let (table, local) = locate_row(weights.shard_rows, row)
            .ok_or_else(|| EngError::Checkpoint("ple row is outside the tables".to_owned()))?;
        let local = u32::try_from(local)
            .map_err(|_| EngError::BadRequest("ple local row exceeds u32".to_owned()))?;
        rows.push((table, local));
    }
    let history = if token == dims.eos {
        Vec::new()
    } else {
        let mut next = history.to_vec();
        next.push(token);
        let keep_from = next.len().saturating_sub(2);
        next.split_off(keep_from)
    };
    Ok(RowPlan { rows, history })
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ChunkPlan {
    /// The `(table index, local row)` pairs of each token, in token order.
    pub(crate) rows: Vec<Vec<(usize, u32)>>,
    pub(crate) history: Vec<u32>,
}

pub(crate) fn plan_chunk_rows<W>(
    shape: &PleShape,
    weights: &PleWeights<'_, W>,
    history: &[u32],
    tokens: &[u32],
) -> Result<ChunkPlan, EngError> {
    if tokens.is_empty() {
        return Err(EngError::BadRequest(
            "ple chunk needs at least one token".to_owned(),
        ));
    }
    let mut rows = Vec::with_capacity(tokens.len());
    let mut current = history.to_vec();
    for &token in tokens {
        let plan = plan_rows(shape, weights, &current, token)?;
        rows.push(plan.rows);
        current = plan.history;
    }
    Ok(ChunkPlan {
        rows,
        history: current,
    })
}

/// Returns the residual update of length `width`.
pub(crate) fn step<P: Projector>(
    projector: &mut P,
    shape: &PleShape,
    weights: &PleWeights<'_, P::Weight>,
    state: &mut PleState,
    token: u32,
    hyper: &[f32],
) -> Result<Vec<f32>, EngError> {
    let dims = shape.dims;
    let width = shape.width;
    let hidden = dims.hidden as usize;
    let branches = dims.branches as usize;
    let ngram = dims.ngram as usize;
    let conv_kernel = dims.conv_kernel as usize;
    let taps = conv_kernel - 1;
    let stride = taps * ngram;
    let head_dim = dims.head_dim as usize;
    let eps = dims.eps;
    let conv_len = width
        .checked_mul(conv_kernel)
        .ok_or_else(|| EngError::Checkpoint("ple conv length overflows".to_owned()))?;

    expect_len(hyper, width, "ple hyper")?;
    expect_len(weights.norm_key, width, "ple key norm")?;
    expect_len(weights.norm_query, width, "ple query norm")?;
    expect_len(weights.norm_conv, width, "ple conv norm")?;
    expect_len(weights.conv, conv_len, "ple conv")?;
    expect_len(&state.ring, shape.ring_len, "ple ring")?;
    if !weights.weight_scale.is_finite() {
        return Err(EngError::BadRequest(
            "ple weight scale must be finite".to_owned(),
        ));
    }
    let profile = step_prof::on();
    let mut t = std::time::Instant::now();
    let mut tick = |slot: usize| {
        if profile {
            step_prof::add(slot, t.elapsed());
            t = std::time::Instant::now();
        }
    };
    let plan = plan_rows(shape, weights, &state.history, token)?;
    let mut emb = Vec::with_capacity(shape.embed);
    for &(table, local) in &plan.rows {
        let values = projector.rows(&weights.shards[table], &[local])?;
        expect_len(&values, head_dim, "ple row")?;
        emb.extend(values.iter().map(|v| v * weights.weight_scale));
    }
    let new_history = plan.history;
    tick(16);

    let keys = projector.linear(weights.key, &emb)?;
    expect_len(&keys, width, "ple key")?;
    tick(17);
    let value = projector.linear(weights.value, &emb)?;
    expect_len(&value, hidden, "ple value")?;
    tick(18);

    let mut gated = vec![0.0f32; width];
    let mut normed = vec![0.0f32; width];
    for b in 0..branches {
        let lo = b * hidden;
        let hi = lo + hidden;
        let kn = rms0(&keys[lo..hi], &weights.norm_key[lo..hi], eps)?;
        let qn = rms0(&hyper[lo..hi], &weights.norm_query[lo..hi], eps)?;
        let d = f64::from(dot(&kn, &qn)? / (hidden as f32).sqrt());
        let shaped = d.abs().max(1e-6).sqrt().copysign(d);
        let g = (1.0 / (1.0 + (-shaped).exp())) as f32;
        for (out, v) in gated[lo..hi].iter_mut().zip(&value) {
            *out = g * v;
        }
        let block = rms0(&gated[lo..hi], &weights.norm_conv[lo..hi], eps)?;
        normed[lo..hi].copy_from_slice(&block);
    }
    tick(19);

    let mut out = vec![0.0f32; width];
    for (c, out_c) in out.iter_mut().enumerate() {
        let kernel = &weights.conv[c * conv_kernel..(c + 1) * conv_kernel];
        let ring = &state.ring[c * stride..(c + 1) * stride];
        let mut a = kernel[taps] * normed[c];
        for (k, w) in kernel[..taps].iter().enumerate() {
            a += w * ring[k * ngram];
        }
        *out_c = gated[c] + silu(a);
    }
    tick(20);

    let mut new_ring = vec![0.0f32; shape.ring_len];
    for (c, &n) in normed.iter().enumerate() {
        let old = &state.ring[c * stride..(c + 1) * stride];
        let next = &mut new_ring[c * stride..(c + 1) * stride];
        next[..stride - 1].copy_from_slice(&old[1..]);
        next[stride - 1] = n;
    }

    state.history = new_history;
    state.ring = new_ring;
    tick(21);
    Ok(out)
}

/// The prefill profiler's slots, when the CUDA driver that dumps them is compiled in.
#[cfg(feature = "native-cuda")]
mod step_prof {
    pub(super) fn on() -> bool {
        crate::family::flash_next::cuda::driver::chunk::prof::on()
    }
    pub(super) fn add(slot: usize, spent: std::time::Duration) {
        crate::family::flash_next::cuda::driver::chunk::prof::add(slot, spent);
    }
}

#[cfg(not(feature = "native-cuda"))]
mod step_prof {
    pub(super) fn on() -> bool {
        false
    }
    pub(super) fn add(_: usize, _: std::time::Duration) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::project::{DenseProjector, DenseWeight};

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5
    }

    fn dims() -> PleDims {
        PleDims {
            branches: 2,
            hidden: 2,
            ngram: 3,
            heads_per_ngram: 1,
            head_dim: 2,
            conv_kernel: 3,
            eos: 0,
            eps: 1e-6,
        }
    }

    const MULTIPLIERS: [i64; 3] = [-1, 3, 5];
    const VOCAB: [i64; 2] = [5, 7];
    const OFFSETS: [i64; 2] = [0, 5];

    struct Tables {
        shards: Vec<DenseWeight>,
        shard_rows: Vec<u64>,
        key: DenseWeight,
        value: DenseWeight,
        norm_key: Vec<f32>,
        norm_query: Vec<f32>,
        norm_conv: Vec<f32>,
        conv: Vec<f32>,
    }

    fn tables() -> Tables {
        let shards = (0..2u64)
            .map(|p| {
                let mut data = Vec::new();
                for l in 0..6u64 {
                    let g = p * 6 + l;
                    data.push(0.1 * (g + 1) as f32);
                    data.push(0.2 * ((g % 3) as f32 - 1.0));
                }
                DenseWeight::matrix(6, 2, data)
            })
            .collect();
        let key = DenseWeight::matrix(
            4,
            4,
            (0..16usize)
                .map(|i| (((i * 3) % 7) as f32 - 3.0) / 4.0)
                .collect(),
        );
        let value = DenseWeight::matrix(
            2,
            4,
            (0..8usize)
                .map(|i| (((i * 5) % 9) as f32 - 4.0) / 5.0)
                .collect(),
        );
        let mut conv = Vec::new();
        for c in 0..4usize {
            conv.push(0.2 * (c + 1) as f32);
            conv.push(-0.3);
            conv.push(1.0);
        }
        Tables {
            shards,
            shard_rows: vec![6, 6],
            key,
            value,
            norm_key: vec![0.1, -0.1, 0.2, 0.0],
            norm_query: vec![0.0, 0.3, -0.2, 0.1],
            norm_conv: vec![0.5, -0.5, 0.0, 0.25],
            conv,
        }
    }

    fn weights(tables: &Tables) -> PleWeights<'_, DenseWeight> {
        PleWeights {
            multipliers: &MULTIPLIERS,
            vocab: &VOCAB,
            offsets: &OFFSETS,
            shards: &tables.shards,
            shard_rows: &tables.shard_rows,
            weight_scale: 2.0,
            key: &tables.key,
            value: &tables.value,
            norm_key: &tables.norm_key,
            norm_query: &tables.norm_query,
            norm_conv: &tables.norm_conv,
            conv: &tables.conv,
        }
    }

    fn hyper_at(t: u32) -> [f32; 4] {
        let t = t as f32;
        [1.0 + 0.1 * t, -0.5, 2.0 - 0.2 * t, 0.3 * t]
    }

    #[test]
    fn shape_rejects_invalid_dimensions() {
        let base = dims();
        assert!(PleShape::new(base).is_ok());
        let cases = [
            PleDims { ngram: 2, ..base },
            PleDims {
                conv_kernel: 1,
                ..base
            },
            PleDims {
                branches: 0,
                ..base
            },
            PleDims {
                eps: f32::NAN,
                ..base
            },
        ];
        for bad in cases {
            assert!(
                matches!(PleShape::new(bad), Err(EngError::Checkpoint(_))),
                "{bad:?} must be a checkpoint error"
            );
        }
    }

    #[test]
    fn hash_rows_follow_history_and_eos() {
        let shape = PleShape::new(dims()).unwrap();
        let rows = |history: &[u32], token: u32| {
            hash_rows(&shape, &VOCAB, &OFFSETS, &MULTIPLIERS, 12, history, token).unwrap()
        };
        assert_eq!(rows(&[], 1), [4, 11]);
        assert_eq!(rows(&[1], 2), [2, 9]);
        assert_eq!(rows(&[1, 2], 0), [1, 8]);
        assert_eq!(rows(&[], 3), [2, 9]);
    }

    #[test]
    fn plan_rows_locates_rows_and_updates_history() {
        let tables = tables();
        let shape = PleShape::new(dims()).unwrap();
        let w = weights(&tables);
        let from_empty = plan_rows(&shape, &w, &[], 1).unwrap();
        assert_eq!(
            from_empty,
            RowPlan {
                rows: vec![(0, 4), (1, 5)],
                history: vec![1]
            },
            "plan after the empty history"
        );
        let from_one = plan_rows(&shape, &w, &[1], 2).unwrap();
        assert_eq!(
            from_one,
            RowPlan {
                rows: vec![(0, 2), (1, 3)],
                history: vec![1, 2]
            },
            "plan after history [1]"
        );
        let at_eos = plan_rows(&shape, &w, &[1, 2], 0).unwrap();
        assert_eq!(
            at_eos,
            RowPlan {
                rows: vec![(0, 1), (1, 2)],
                history: vec![]
            },
            "plan at the end-of-sequence token"
        );
        let after = plan_rows(&shape, &w, &[1, 2], 3).unwrap();
        assert_eq!(after.history, vec![2, 3], "history after token 3");
    }

    /// The fixture has end-of-sequence token 0 and tables of 6 rows each. The single-token plans
    /// above give: token 1 after [] reads rows (0, 4), (1, 5); token 2 after [1] reads (0, 2), (1, 3);
    /// token 0 after [1, 2] reads (0, 1), (1, 2) and clears the history; token 3 after [] reads
    /// global rows 2 and 9, which are (0, 2) and (1, 3) (see `hash_rows_follow_history_and_eos`).
    #[test]
    fn plan_chunk_rows_carries_the_history_and_clears_it_at_eos() {
        let tables = tables();
        let shape = PleShape::new(dims()).unwrap();
        let w = weights(&tables);
        let plan = plan_chunk_rows(&shape, &w, &[], &[1, 2, 0, 3]).unwrap();
        let expected: Vec<Vec<(usize, u32)>> = vec![
            vec![(0, 4), (1, 5)],
            vec![(0, 2), (1, 3)],
            vec![(0, 1), (1, 2)],
            vec![(0, 2), (1, 3)],
        ];
        assert_eq!(plan.rows, expected, "rows per token");
        assert_eq!(plan.history, vec![3], "history after the chunk");
        // Two chunks that carry the history equal one chunk.
        let first = plan_chunk_rows(&shape, &w, &[], &[1, 2]).unwrap();
        assert_eq!(first.history, vec![1, 2], "history after the first chunk");
        let second = plan_chunk_rows(&shape, &w, &first.history, &[0, 3]).unwrap();
        assert_eq!(second.rows, expected[2..], "rows of the second chunk");
        assert_eq!(second.history, vec![3], "history after the second chunk");
    }

    #[test]
    fn plan_chunk_rows_rejects_an_empty_chunk() {
        let tables = tables();
        let shape = PleShape::new(dims()).unwrap();
        let w = weights(&tables);
        assert!(matches!(
            plan_chunk_rows(&shape, &w, &[1], &[]),
            Err(EngError::BadRequest(_))
        ));
    }

    #[test]
    fn resume_installs_the_saved_half_and_refuses_a_wrong_ring() {
        let shape = PleShape::new(dims()).unwrap();
        let mut state = PleState::new(&shape);
        state.history.extend_from_slice(&[7, 8, 9]);
        for (i, v) in state.ring.iter_mut().enumerate() {
            *v = i as f32 + 0.5;
        }
        let (history, ring) = (state.history.clone(), state.ring.clone());
        // A ring of the wrong length is refused and changes nothing.
        let mut short = ring.clone();
        short.pop();
        assert!(state.resume(&[1], &short).is_err());
        assert_eq!(state.history, history, "a refused resume changes nothing");
        // A later sequence's history REPLACES the boundary's, never appends.
        state.history.clear();
        state.history.push(42);
        state.resume(&history, &ring).unwrap();
        assert_eq!(state.history, history);
        assert_eq!(state.ring, ring);
        assert_eq!(
            state.ring_values(),
            &ring[..],
            "the snapshot reads the same values back"
        );
    }

    #[test]
    fn four_steps_match_hand_values() {
        let tables = tables();
        let shape = PleShape::new(dims()).unwrap();
        let mut state = PleState::new(&shape);
        let mut projector = DenseProjector::default();
        let expected = [
            [-1.669_724_8, -0.470_042_5, -0.970_681_8, -0.364_474_5],
            [-1.478_167_9, -0.621_882_5, -0.784_524_9, -0.433_732_4],
            [-1.154_285_7, -0.202_266_1, -0.609_025_3, -0.174_610_1],
            [-1.529_356_7, -0.606_979_3, -0.687_824, -0.371_725_8],
        ];
        let tokens = [1, 2, 0, 3];
        for (index, (token, want)) in tokens.into_iter().zip(expected).enumerate() {
            let t = index as u32 + 1;
            let got = step(
                &mut projector,
                &shape,
                &weights(&tables),
                &mut state,
                token,
                &hyper_at(t),
            )
            .unwrap();
            assert_eq!(got.len(), 4, "step {t} width");
            for (g, w) in got.iter().zip(want) {
                assert!(close(*g, w), "step {t}: {g} vs {w}");
            }
            if t == 3 {
                assert!(state.history.is_empty(), "history after step 3");
            }
        }
        assert_eq!(state.history, [3]);
        let ring_want = [
            0.0,
            0.0,
            -2.048_379_4,
            -1.963_380_5,
            -2.094_252_3,
            -1.963_380_5,
            0.0,
            0.0,
            -0.183_828_9,
            -0.267_733_7,
            -0.112_594_2,
            -0.267_733_7,
            0.0,
            0.0,
            -1.365_584_3,
            -1.308_916_7,
            -1.396_157_4,
            -1.308_914_9,
            0.0,
            0.0,
            -0.459_571_6,
            -0.669_332_4,
            -0.281_483_4,
            -0.669_331_5,
        ];
        assert_eq!(state.ring.len(), ring_want.len());
        for (i, (g, w)) in state.ring.iter().zip(ring_want).enumerate() {
            assert!(close(*g, w), "ring slot {i}: {g} vs {w}");
        }
        assert_eq!(state.byte_len(), 100);
    }

    #[test]
    fn wrong_hyper_length_leaves_state_unchanged() {
        let tables = tables();
        let shape = PleShape::new(dims()).unwrap();
        let mut state = PleState::new(&shape);
        let mut projector = DenseProjector::default();
        let result = step(
            &mut projector,
            &shape,
            &weights(&tables),
            &mut state,
            1,
            &[1.0, -0.5, 2.0],
        );
        assert!(matches!(result, Err(EngError::BadRequest(_))));
        assert_eq!(projector.calls, 0);
        assert_eq!(state, PleState::new(&shape));
        assert_eq!(PleState::new(&shape).byte_len(), 96);
    }

    #[test]
    fn failed_projection_leaves_state_unchanged() {
        let mut tables = tables();
        tables.value = DenseWeight::matrix(2, 3, vec![1.0; 6]);
        let shape = PleShape::new(dims()).unwrap();
        let mut state = PleState::new(&shape);
        let mut projector = DenseProjector::default();
        let result = step(
            &mut projector,
            &shape,
            &weights(&tables),
            &mut state,
            1,
            &hyper_at(1),
        );
        assert!(matches!(result, Err(EngError::BadRequest(_))));
        assert_eq!(state, PleState::new(&shape));
    }

    #[test]
    fn row_outside_the_tables_is_a_checkpoint_error() {
        let mut tables = tables();
        tables.shard_rows = vec![6, 4];
        tables.shards[1] = DenseWeight::matrix(4, 2, vec![0.0; 8]);
        let shape = PleShape::new(dims()).unwrap();
        let mut state = PleState::new(&shape);
        let mut projector = DenseProjector::default();
        let result = step(
            &mut projector,
            &shape,
            &weights(&tables),
            &mut state,
            1,
            &hyper_at(1),
        );
        assert!(matches!(result, Err(EngError::Checkpoint(_))));
        assert_eq!(projector.calls, 0);
    }

    /// The EXL3 pack's hashing tensors are byte-identical to those of the official and MLX
    /// checkpoints (manifest of the fixture). Its rows for this prompt were selected by a
    /// separate Python port of the hashing, and their decoded values track the official BF16
    /// rows, so the row numbers are the model's own.
    #[test]
    fn the_planned_rows_of_a_real_prompt_match_the_exl3_fixture() {
        let manifest: serde_json::Value = serde_json::from_slice(include_bytes!(
            "../../../tests/fixtures/flash_next_exl3_ngram/manifest.json"
        ))
        .unwrap();
        let ints = |name: &str| -> Vec<i64> {
            manifest[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect()
        };
        let (multipliers, vocab, offsets) = (
            ints("layer_multipliers"),
            ints("head_vocab_sizes"),
            ints("head_offsets"),
        );
        let shard_rows = vec![2_500_012u64; 128];
        let shards = vec![(); 128];
        let weights = PleWeights {
            multipliers: &multipliers,
            vocab: &vocab,
            offsets: &offsets,
            shards: &shards,
            shard_rows: &shard_rows,
            weight_scale: 1.0,
            key: &(),
            value: &(),
            norm_key: &[],
            norm_query: &[],
            norm_conv: &[],
            conv: &[],
        };
        let shape = PleShape::new(PleDims {
            branches: 4,
            hidden: 2560,
            ngram: 3,
            heads_per_ngram: 8,
            head_dim: 160,
            conv_kernel: 4,
            eos: 248_044,
            eps: 1e-6,
        })
        .unwrap();
        let prompt: Vec<u32> = ints("prompt").into_iter().map(|t| t as u32).collect();
        let plan = plan_chunk_rows(&shape, &weights, &[], &prompt).unwrap();
        let planned: Vec<i64> = plan
            .rows
            .iter()
            .flatten()
            .map(|&(table, local)| table as i64 * 2_500_012 + i64::from(local))
            .collect();
        assert_eq!(planned, ints("global_rows"));
    }
}
