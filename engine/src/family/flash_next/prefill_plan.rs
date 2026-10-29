//! Pure planning helpers for chunked prefill.
//!
//! These functions hold no tensors and make no native call. The resident prefill step uses them to
//! merge the expert choices of a chunk, to plan positioned reads, and to pick the chunk size that
//! fits the memory plan.

use crate::EngError;

/// Returns a `BadRequest` with `message`.
fn bad(message: &str) -> EngError {
    EngError::BadRequest(message.into())
}

/// Expert page sets lent at the same time during a prefill.
const PREFILL_SETS: u32 = 2;

/// Route ranges in one prefill page set: wave-local rank, token row, gate, inverse permutation.
const ROUTE_RANGES: u64 = 4;

/// Bytes of one route word (`U32` or `F32`).
const ROUTE_WORD_BYTES: u64 = 4;

/// Rejects a wave cap of zero.
fn check_wave_cap(wave_cap: u32) -> Result<(), EngError> {
    if wave_cap == 0 {
        return Err(bad("prefill wave cap must exceed zero"));
    }
    Ok(())
}

/// The routing of one prefill chunk after the per-token expert selection.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ChunkRoute {
    /// The distinct selected experts of the chunk, ascending.
    pub(crate) distinct: Vec<u32>,
    /// For each pair `p = t * top_k + j` (token-major), the rank of its expert in `distinct`.
    pub(crate) rank_idx: Vec<u32>,
    /// For each pair, its gate weight, in the same order.
    pub(crate) gates: Vec<f32>,
    /// Experts selected per token.
    pub(crate) top_k: u32,
}

/// Merges the per-token selections of a chunk (`moe::select` output, one row per token).
///
/// Every row must hold the same number of experts, and every expert must be below `experts`.
/// The result lists the distinct experts in ascending order and gives each pair its rank in that
/// list. Pairs follow the token-major order of the input rows.
pub(crate) fn union_and_remap(
    selections: &[Vec<(u32, f32)>],
    experts: u32,
) -> Result<ChunkRoute, EngError> {
    let top_k = selections
        .first()
        .map(Vec::len)
        .filter(|&len| len > 0)
        .ok_or_else(|| bad("prefill route needs a token with at least one expert"))?;
    let mut distinct = Vec::new();
    for row in selections {
        if row.len() != top_k {
            return Err(bad("prefill route rows differ in length"));
        }
        for &(expert, _) in row {
            if expert >= experts {
                return Err(bad("prefill route expert is out of range"));
            }
            distinct.push(expert);
        }
    }
    distinct.sort_unstable();
    distinct.dedup();
    let pairs = selections
        .len()
        .checked_mul(top_k)
        .ok_or_else(|| bad("prefill route pair count overflows"))?;
    let mut rank_idx = Vec::with_capacity(pairs);
    let mut gates = Vec::with_capacity(pairs);
    for &(expert, gate) in selections.iter().flatten() {
        let rank = distinct
            .binary_search(&expert)
            .map_err(|_| bad("prefill route rank is missing"))?;
        rank_idx.push(u32::try_from(rank).map_err(|_| bad("prefill route rank exceeds u32"))?);
        gates.push(gate);
    }
    Ok(ChunkRoute {
        distinct,
        rank_idx,
        gates,
        top_k: u32::try_from(top_k).map_err(|_| bad("prefill route top-k exceeds u32"))?,
    })
}

/// A run of consecutive expert ids that one positioned read can load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExpertRun {
    /// The first expert id of the run.
    pub(crate) first: u32,
    /// The rank of that expert in the distinct list, which is its slot in the page set.
    pub(crate) rank: u32,
    /// The number of experts in the run.
    pub(crate) len: u32,
}

/// Splits a strictly ascending expert list into runs of consecutive ids.
///
/// A run ends at a gap. The experts of a run are adjacent in the checkpoint stack and in the page
/// set, so one read can fill them.
pub(crate) fn coalesce_runs(distinct: &[u32]) -> Result<Vec<ExpertRun>, EngError> {
    if !distinct.is_sorted_by(|a, b| a < b) {
        return Err(bad("prefill experts must be strictly ascending"));
    }
    let overflow = || bad("prefill run length overflows u32");
    let mut runs: Vec<ExpertRun> = Vec::new();
    for (index, &expert) in distinct.iter().enumerate() {
        let continues = runs
            .last()
            .is_some_and(|run| run.first.checked_add(run.len) == Some(expert));
        if continues {
            if let Some(run) = runs.last_mut() {
                run.len = run.len.checked_add(1).ok_or_else(overflow)?;
            }
        } else {
            let rank = u32::try_from(index).map_err(|_| bad("prefill rank exceeds u32"))?;
            runs.push(ExpertRun {
                first: expert,
                rank,
                len: 1,
            });
        }
    }
    Ok(runs)
}

/// The retire cap of a decode step and the base of the prefill retire cap: 64 MiB.
pub(crate) const RETIRE_CAP_BASE: u64 = 64 << 20;

/// The retire cap of the adapter while a prefill chunk runs: a flat 512 MiB. The adapter returns to
/// its previous cap after the chunk.
pub(crate) const PREFILL_RETIRE_CAP: u64 = 512 << 20;

/// State-sized tensors that the unrolled DeltaNet loop drops per token and layer: the decayed
/// state, the outer product, and the replaced state.
pub(crate) const DELTA_RETIRED_STATES: u64 = 3;

/// Retired bytes of one chunk token in one DeltaNet layer: `3 * VH * KD * VD * 4` (F32 state).
///
/// A zero dimension or an overflow is a `BadRequest`.
pub(crate) fn delta_retire_per_token(
    value_heads: u32,
    key_dim: u32,
    value_dim: u32,
) -> Result<u64, EngError> {
    u64::from(value_heads)
        .checked_mul(u64::from(key_dim))
        .and_then(|n| n.checked_mul(u64::from(value_dim)))
        .and_then(|n| n.checked_mul(4))
        .and_then(|n| n.checked_mul(DELTA_RETIRED_STATES))
        .filter(|&n| n > 0)
        .ok_or_else(|| bad("delta retire bytes are zero or overflow"))
}

/// Tokens per DeltaNet sub-chunk: `floor((cap - 64 MiB) / per_token)`, at least 1 and at most
/// `tokens`. The layer state is evaluated between sub-chunks, so the retired bytes of one layer
/// stay near `cap`. `tokens` and `per_token` must be positive.
pub(crate) fn delta_subchunk(tokens: u32, cap: u64, per_token: u64) -> Result<u32, EngError> {
    if tokens == 0 || per_token == 0 {
        return Err(bad(
            "delta sub-chunk needs positive tokens and bytes per token",
        ));
    }
    let fit = cap.saturating_sub(RETIRE_CAP_BASE) / per_token;
    Ok(u32::try_from(fit).unwrap_or(u32::MAX).clamp(1, tokens))
}

/// The token ranges `(start, stop)` of the DeltaNet sub-chunks of a chunk of `tokens` tokens, in
/// order. Every range is non-empty, and together they cover `0..tokens` exactly. `tokens` and
/// `sub` must be positive, so the caller never starts the DeltaNet chunk with an empty range.
pub(crate) fn delta_ranges(tokens: u32, sub: u32) -> Result<Vec<(u32, u32)>, EngError> {
    if tokens == 0 || sub == 0 {
        return Err(bad("delta sub-chunk needs positive tokens and size"));
    }
    (0..tokens.div_ceil(sub))
        .map(|part| {
            let start = part
                .checked_mul(sub)
                .ok_or_else(|| bad("delta sub-chunk start overflows u32"))?;
            let stop = start.saturating_add(sub).min(tokens);
            Ok((start, stop))
        })
        .collect()
}

/// Planned extra synchronizations of one chunk: `delta_layers * (ceil(tokens / sub) - 1)`.
pub(crate) fn delta_extra_syncs(delta_layers: u32, tokens: u32, sub: u32) -> Result<u64, EngError> {
    if sub == 0 || tokens == 0 {
        return Err(bad("delta sub-chunk count needs positive tokens and size"));
    }
    Ok(u64::from(delta_layers) * u64::from(tokens.div_ceil(sub).saturating_sub(1)))
}

/// Experts that each prefill page set holds for a chunk of `tokens` tokens when the union runs in
/// waves of at most `wave_cap` experts. The pool is sized from `ChunkCost` so that admission and
/// the pool use one rule: `min(experts, wave_cap, tokens * top_k)`.
pub(crate) fn prefill_set_experts(
    experts: u32,
    top_k: u32,
    wave_cap: u32,
    tokens: u32,
) -> Result<u32, EngError> {
    ChunkCost {
        experts,
        top_k,
        expert_bytes: 0,
        sets: PREFILL_SETS,
        per_token: 0,
        fixed: 0,
    }
    .with_wave_cap(wave_cap)?
    .set_experts(tokens)
}

/// The memory terms of one prefill chunk. All sizes are in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChunkCost {
    /// Experts that one page set holds at most: the expert count of a MoE layer when the chunk runs
    /// in one wave, or the wave cap `U_w` after `with_wave_cap`.
    pub(crate) experts: u32,
    /// Experts selected per token.
    pub(crate) top_k: u32,
    /// Bytes of one expert across the gate, up, and down stacks, including page rounding.
    pub(crate) expert_bytes: u64,
    /// Expert page sets lent at the same time (two).
    pub(crate) sets: u32,
    /// Activation workspace bytes per chunk token.
    pub(crate) per_token: u64,
    /// Bytes that do not depend on the chunk size, including the retire cap.
    pub(crate) fixed: u64,
}

impl ChunkCost {
    /// The cost of a chunk that runs its expert union in waves of at most `wave_cap` experts.
    ///
    /// A cap above the layer's expert count changes nothing. A cap of zero is a `BadRequest`.
    pub(crate) fn with_wave_cap(self, wave_cap: u32) -> Result<Self, EngError> {
        check_wave_cap(wave_cap)?;
        Ok(Self {
            experts: self.experts.min(wave_cap),
            ..self
        })
    }

    /// Experts that each page set must hold for a chunk of `tokens` tokens.
    ///
    /// A chunk selects at most `tokens * top_k` distinct experts, and a set holds at most
    /// `experts`. The result is the smaller of the two.
    pub(crate) fn set_experts(&self, tokens: u32) -> Result<u32, EngError> {
        let overflow = || EngError::MemoryPlan("prefill workspace overflows u64".into());
        let pairs = u64::from(tokens)
            .checked_mul(u64::from(self.top_k))
            .ok_or_else(overflow)?;
        u32::try_from(pairs.min(u64::from(self.experts))).map_err(|_| overflow())
    }

    /// Workspace bytes of a chunk of `tokens` tokens.
    ///
    /// A page set must hold every distinct expert the chunk can select in one wave. That count is
    /// `set_experts(tokens)`.
    pub(crate) fn workspace(&self, tokens: u32) -> Result<u64, EngError> {
        let overflow = || EngError::MemoryPlan("prefill workspace overflows u64".into());
        let held = u64::from(self.set_experts(tokens)?);
        let sets = u64::from(self.sets)
            .checked_mul(held)
            .and_then(|n| n.checked_mul(self.expert_bytes))
            .ok_or_else(overflow)?;
        let activations = u64::from(tokens)
            .checked_mul(self.per_token)
            .ok_or_else(overflow)?;
        sets.checked_add(activations)
            .and_then(|n| n.checked_add(self.fixed))
            .ok_or_else(overflow)
    }

    /// Adds the prefill retire cap to the cost: `PREFILL_RETIRE_CAP` (512 MiB) to `fixed`.
    ///
    /// The cap is flat, so `per_token` does not change.
    pub(crate) fn with_retire_cap(self) -> Result<Self, EngError> {
        Ok(Self {
            fixed: self.fixed.checked_add(PREFILL_RETIRE_CAP).ok_or_else(|| {
                EngError::MemoryPlan("prefill retire cap cost overflows u64".into())
            })?,
            ..self
        })
    }

    /// Adds the route pages of the page sets to the cost.
    ///
    /// Each set holds four route ranges of `tokens * top_k` words of 4 bytes (rank, token, gate,
    /// inverse permutation). That is `sets * 16 * top_k` bytes per token in `per_token`. Each range
    /// starts on a page boundary, so `fixed` grows by `sets * 4 * page` for the rounding. The
    /// counts over-estimate by the gate page that `expert_bytes` already holds, which is safe.
    pub(crate) fn with_route_pages(self, page: u64) -> Result<Self, EngError> {
        let overflow = || EngError::MemoryPlan("prefill route page cost overflows u64".into());
        let sets = u64::from(self.sets);
        let per_set_token = u64::from(self.top_k)
            .checked_mul(ROUTE_RANGES * ROUTE_WORD_BYTES)
            .ok_or_else(overflow)?;
        Ok(Self {
            per_token: sets
                .checked_mul(per_set_token)
                .and_then(|n| n.checked_add(self.per_token))
                .ok_or_else(overflow)?,
            fixed: sets
                .checked_mul(ROUTE_RANGES)
                .and_then(|n| n.checked_mul(page))
                .and_then(|n| n.checked_add(self.fixed))
                .ok_or_else(overflow)?,
            ..self
        })
    }
}

/// Returns the largest candidate chunk size whose workspace is at most `remaining`, or `None`.
///
/// The candidates can come in any order. A candidate of zero tokens is a `BadRequest`.
pub(crate) fn largest_chunk(
    cost: &ChunkCost,
    candidates: &[u32],
    remaining: u64,
) -> Result<Option<u32>, EngError> {
    let mut best: Option<u32> = None;
    for &tokens in candidates {
        if tokens == 0 {
            return Err(bad("prefill chunk size must exceed zero"));
        }
        if cost.workspace(tokens)? <= remaining {
            best = best.max(Some(tokens));
        }
    }
    Ok(best)
}

/// The candidate chunk sizes of the admission rule.
pub(crate) const PREFILL_CANDIDATES: [u32; 11] = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 1];

/// Experts per wave, `U_w`. The first release runs one wave: 512 is the expert count of a layer.
pub(crate) const PREFILL_WAVE_CAP: u32 = 512;

/// Activation bytes per chunk token, a planning value.
const PLAN_PER_TOKEN: u64 = 800_000;

/// Chunk bytes that do not depend on the chunk size, a planning value. The retire cap and the route
/// pages come on top.
const PLAN_FIXED: u64 = 100_000_000;

/// The admission cost of a prefill chunk on a model with `experts` experts per layer, `top_k`
/// experts per token, `expert_bytes` per expert (page rounded), and OS page size `page`.
///
/// It holds the planning terms, the wave cap `PREFILL_WAVE_CAP`, the retire cap, and the route
/// pages. `largest_chunk` and the prefill page sets read the same value, so they cannot disagree.
pub(crate) fn admission_cost(
    experts: u32,
    top_k: u32,
    expert_bytes: u64,
    page: u64,
) -> Result<ChunkCost, EngError> {
    ChunkCost {
        experts,
        top_k,
        expert_bytes,
        sets: PREFILL_SETS,
        per_token: PLAN_PER_TOKEN,
        fixed: PLAN_FIXED,
    }
    .with_wave_cap(PREFILL_WAVE_CAP)?
    .with_retire_cap()?
    .with_route_pages(page)
}

/// The bytes left for a prefill chunk (`remaining` of the admission rule): the engine budget minus
/// the resident weights, the decode pool, the S3 cache, and the session reserve. A negative result
/// is zero.
///
/// The admission rule is `B - resident weights - session state - S3 cache - MLX cache limit -
/// pressure margin`. The session reserve holds the last three terms that the caller owns: the
/// session state, the MLX cache limit, and the pressure margin (see `FlashExecutor::plan_prefill`).
/// The decode pool is an extra term: the rule counts it as idle during prefill, and this function
/// charges it anyway (conservative by 83,214,336 B on the real model).
pub(crate) fn prefill_remaining(
    budget: u64,
    weights: u64,
    pool: u64,
    s3_cache: u64,
    session_reserve: u64,
) -> u64 {
    budget
        .saturating_sub(weights)
        .saturating_sub(pool)
        .saturating_sub(s3_cache)
        .saturating_sub(session_reserve)
}

/// The terms that one admission charged, in bytes: the executor records which terms it charged.
/// `Display` gives one `name=bytes` line per term and ends with the resulting `remaining_bytes`, so
/// a report can print it unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrefillTerms {
    /// The engine budget `B`.
    pub(crate) budget: u64,
    /// The resident weights of this executor.
    pub(crate) weights: u64,
    /// The decode pool: an extra term beyond the admission rule.
    pub(crate) decode_pool: u64,
    /// The S3 cache.
    pub(crate) s3_cache: u64,
    /// The session state, the MLX cache limit, and the pressure margin, as the caller summed them.
    pub(crate) session_reserve: u64,
}

impl PrefillTerms {
    /// The bytes left for a chunk after every term (`prefill_remaining`).
    pub(crate) fn remaining(&self) -> u64 {
        prefill_remaining(
            self.budget,
            self.weights,
            self.decode_pool,
            self.s3_cache,
            self.session_reserve,
        )
    }
}

impl std::fmt::Display for PrefillTerms {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(out, "budget_bytes={}", self.budget)?;
        writeln!(out, "weights_bytes={}", self.weights)?;
        writeln!(out, "decode_pool_bytes={}", self.decode_pool)?;
        writeln!(out, "s3_cache_bytes={}", self.s3_cache)?;
        writeln!(out, "session_reserve_bytes={}", self.session_reserve)?;
        writeln!(out, "remaining_bytes={}", self.remaining())
    }
}

/// The result of one admission: the chunk size and the terms that were charged for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrefillPlan {
    /// The chunk size, or `None` when no size above one token fits.
    pub(crate) chunk: Option<u32>,
    /// The terms that were charged.
    pub(crate) terms: PrefillTerms,
}

/// The chunk size that the prefill runs, or `None` when it runs one-token steps. It is the largest
/// candidate that fits `remaining`. A chunk of one token is the one-token path, so a result of 1 is
/// `None`.
pub(crate) fn admitted_chunk(cost: &ChunkCost, remaining: u64) -> Result<Option<u32>, EngError> {
    Ok(largest_chunk(cost, &PREFILL_CANDIDATES, remaining)?.filter(|&tokens| tokens > 1))
}

/// One wave of a chunk: at most `wave_cap` consecutive ranks of the sorted union and the pairs
/// that use them.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Wave {
    /// The experts of the wave, ascending. `coalesce_runs` on this list gives wave-local ranks.
    pub(crate) experts: Vec<u32>,
    /// The pairs `p = t * top_k + j` whose expert is in the wave, ordered by (local rank, pair
    /// index): stable by local rank, so the pair index breaks ties.
    pub(crate) pairs: Vec<u32>,
    /// For each pair of `pairs`, the rank of its expert inside `experts`.
    pub(crate) local_ranks: Vec<u32>,
    /// For each pair of `pairs`, its token row `p / top_k`.
    pub(crate) tokens: Vec<u32>,
    /// For each pair of `pairs`, its gate weight.
    pub(crate) gates: Vec<f32>,
}

/// The waves of one chunk and the permutation that restores pair order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WavePlan {
    /// The waves in rank order. Every wave holds at least one pair.
    pub(crate) waves: Vec<Wave>,
    /// For each pair `p`, its position in the concatenation of the `pairs` lists of all waves.
    /// Taking the concatenated wave rows at these positions restores pair order.
    pub(crate) inverse: Vec<u32>,
}

/// Waves needed for `union` distinct experts when a wave holds at most `wave_cap` experts.
///
/// A cap of zero is a `BadRequest`.
pub(crate) fn wave_count(union: u32, wave_cap: u32) -> Result<u32, EngError> {
    check_wave_cap(wave_cap)?;
    Ok(union.div_ceil(wave_cap))
}

/// Splits the experts of a chunk route into waves of at most `wave_cap` experts.
///
/// Wave `i` takes ranks `i * wave_cap` to `(i + 1) * wave_cap - 1` of `route.distinct`. Each pair
/// joins the wave of its rank, and the pairs of a wave are ordered by (local rank, pair index): a
/// stable sort by rank keeps the pair index ascending inside one rank, so every wave lists
/// non-decreasing local ranks and the concatenation of the wave pair lists holds the pairs in
/// (rank, pair) order, the non-decreasing index order that `gather_qmm_sorted` requires. The route
/// must come from `union_and_remap`: ascending distinct experts, ranks below their count, one gate
/// per pair, a whole number of tokens, and every expert used by at least one pair.
pub(crate) fn split_waves(route: &ChunkRoute, wave_cap: u32) -> Result<WavePlan, EngError> {
    let union =
        u32::try_from(route.distinct.len()).map_err(|_| bad("prefill union exceeds u32"))?;
    let count = wave_count(union, wave_cap)?;
    let top_k = usize::try_from(route.top_k).map_err(|_| bad("prefill top-k exceeds usize"))?;
    let pairs = route.rank_idx.len();
    if union == 0
        || pairs == 0
        || top_k == 0
        || route.gates.len() != pairs
        || !pairs.is_multiple_of(top_k)
    {
        return Err(bad("prefill route is inconsistent"));
    }
    if !route.distinct.is_sorted_by(|a, b| a < b) {
        return Err(bad("prefill experts must be strictly ascending"));
    }
    let cap = usize::try_from(wave_cap).map_err(|_| bad("prefill wave cap exceeds usize"))?;
    let mut waves = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
    for experts in route.distinct.chunks(cap) {
        waves.push(Wave {
            experts: experts.to_vec(),
            pairs: Vec::new(),
            local_ranks: Vec::new(),
            tokens: Vec::new(),
            gates: Vec::new(),
        });
    }
    // The pairs in (rank, pair) order: a stable sort by rank keeps the pair index ascending inside
    // one rank, so each wave receives its pairs already ordered by (local rank, pair index) and the
    // concatenation of the wave pair lists is the (rank, pair) order that the inverse permutation
    // inverts.
    let mut order: Vec<(u32, usize)> = route
        .rank_idx
        .iter()
        .enumerate()
        .map(|(pair, &rank)| (rank, pair))
        .collect();
    order.sort_by_key(|&(rank, _)| rank);
    for &(rank, pair) in &order {
        let rank = usize::try_from(rank).map_err(|_| bad("prefill rank exceeds usize"))?;
        if rank >= route.distinct.len() {
            return Err(bad("prefill route rank is out of range"));
        }
        let wave = waves
            .get_mut(rank / cap)
            .ok_or_else(|| bad("prefill route rank has no wave"))?;
        let gate = route
            .gates
            .get(pair)
            .copied()
            .ok_or_else(|| bad("prefill route is inconsistent"))?;
        wave.pairs
            .push(u32::try_from(pair).map_err(|_| bad("prefill pair exceeds u32"))?);
        wave.local_ranks
            .push(u32::try_from(rank % cap).map_err(|_| bad("prefill rank exceeds u32"))?);
        wave.tokens
            .push(u32::try_from(pair / top_k).map_err(|_| bad("prefill token exceeds u32"))?);
        wave.gates.push(gate);
    }
    let mut inverse = vec![0u32; pairs];
    let mut position = 0u32;
    for wave in &waves {
        if wave.pairs.is_empty() {
            return Err(bad("prefill wave has no pair"));
        }
        for &pair in &wave.pairs {
            let slot = inverse
                .get_mut(usize::try_from(pair).map_err(|_| bad("prefill pair exceeds usize"))?)
                .ok_or_else(|| bad("prefill pair is out of range"))?;
            *slot = position;
            position = position
                .checked_add(1)
                .ok_or_else(|| bad("prefill position exceeds u32"))?;
        }
    }
    Ok(WavePlan { waves, inverse })
}

/// Proves on the host the order that `gather_qmm_sorted` requires of one wave: the wave-local ranks
/// of its pairs must be non-decreasing. The wave import runs this check before it writes any page,
/// because the method cannot read the device indices. An inversion returns a `BadRequest` whose
/// text starts `flash-next prefill:`; an empty slice passes.
pub(crate) fn require_sorted_ranks(ranks: &[u32]) -> Result<(), EngError> {
    if ranks.is_sorted() {
        Ok(())
    } else {
        Err(bad("flash-next prefill: the wave ranks are not sorted"))
    }
}

/// Extra synchronizations that waves add to one chunk.
///
/// A layer with `w` waves adds `w - 1` evaluations. `waves_per_layer` holds the wave count of each
/// MoE layer. Equal counts `w` over 48 layers give `48 * (w - 1)`. A count of zero is a `BadRequest`.
pub(crate) fn wave_syncs(waves_per_layer: &[u32]) -> Result<u64, EngError> {
    waves_per_layer.iter().try_fold(0u64, |total, &waves| {
        let extra = waves
            .checked_sub(1)
            .ok_or_else(|| bad("prefill layer needs at least one wave"))?;
        total
            .checked_add(u64::from(extra))
            .ok_or_else(|| bad("prefill sync count overflows u64"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_bad_request<T: std::fmt::Debug>(result: Result<T, EngError>) -> bool {
        matches!(result, Err(EngError::BadRequest(_)))
    }

    /// Three tokens, three experts each. The distinct experts are 0, 2, 5, 7, 9.
    fn selections() -> Vec<Vec<(u32, f32)>> {
        vec![
            vec![(5, 0.5), (2, 0.3), (9, 0.2)],
            vec![(2, 0.6), (7, 0.3), (5, 0.1)],
            vec![(9, 0.4), (0, 0.35), (7, 0.25)],
        ]
    }

    #[test]
    fn union_orders_the_experts_and_ranks_each_pair() {
        let route = union_and_remap(&selections(), 512).unwrap();
        assert_eq!(route.distinct, [0, 2, 5, 7, 9], "distinct experts");
        assert_eq!(route.top_k, 3, "top-k");
        // Pair p = t * 3 + j: row 0 is 5, 2, 9; row 1 is 2, 7, 5; row 2 is 9, 0, 7.
        assert_eq!(route.rank_idx, [2, 1, 4, 1, 3, 2, 4, 0, 3], "ranks");
        assert_eq!(
            route.gates,
            [0.5, 0.3, 0.2, 0.6, 0.3, 0.1, 0.4, 0.35, 0.25],
            "gates in token-major order"
        );
    }

    #[test]
    fn union_rejects_ragged_empty_and_out_of_range_rows() {
        assert!(is_bad_request(union_and_remap(&[], 512)), "no tokens");
        assert!(
            is_bad_request(union_and_remap(&[vec![]], 512)),
            "token without experts"
        );
        assert!(
            is_bad_request(union_and_remap(
                &[vec![(1, 0.5), (2, 0.5)], vec![(3, 1.0)]],
                512
            )),
            "ragged rows"
        );
        assert!(
            is_bad_request(union_and_remap(&[vec![(512, 1.0)]], 512)),
            "expert equal to the expert count"
        );
        assert!(
            union_and_remap(&[vec![(511, 1.0)]], 512).is_ok(),
            "last valid expert"
        );
    }

    #[test]
    fn coalesce_joins_consecutive_experts() {
        let runs = coalesce_runs(&[3, 4, 5, 7, 8, 20]).unwrap();
        assert_eq!(
            runs,
            [
                ExpertRun {
                    first: 3,
                    rank: 0,
                    len: 3
                },
                ExpertRun {
                    first: 7,
                    rank: 3,
                    len: 2
                },
                ExpertRun {
                    first: 20,
                    rank: 5,
                    len: 1
                },
            ]
        );
        assert!(coalesce_runs(&[]).unwrap().is_empty());
        assert_eq!(
            coalesce_runs(&[9]).unwrap(),
            [ExpertRun {
                first: 9,
                rank: 0,
                len: 1
            }]
        );
    }

    #[test]
    fn coalesce_rejects_unsorted_and_repeated_experts() {
        assert!(is_bad_request(coalesce_runs(&[5, 3])), "descending");
        assert!(is_bad_request(coalesce_runs(&[2, 2])), "repeated");
    }

    fn cost() -> ChunkCost {
        ChunkCost {
            experts: 512,
            top_k: 10,
            expert_bytes: 2_764_800,
            sets: 2,
            per_token: 800_000,
            fixed: 100_000_000,
        }
    }

    /// `sets * min(experts, tokens * top_k) * expert_bytes + tokens * per_token + fixed`:
    /// T = 1 holds 10 experts, T = 16 holds 160, T = 64 and T = 512 hold all 512.
    #[test]
    fn workspace_adds_sets_activations_and_fixed_bytes() {
        let cost = cost();
        assert_eq!(cost.workspace(1).unwrap(), 156_096_000);
        assert_eq!(cost.workspace(16).unwrap(), 997_536_000);
        assert_eq!(cost.workspace(64).unwrap(), 2_982_355_200);
        assert_eq!(cost.workspace(512).unwrap(), 3_340_755_200);
    }

    #[test]
    fn workspace_reports_overflow_as_a_memory_plan_error() {
        let cost = ChunkCost {
            expert_bytes: u64::MAX,
            per_token: 0,
            fixed: 0,
            ..cost()
        };
        assert!(matches!(cost.workspace(1), Err(EngError::MemoryPlan(_))));
    }

    #[test]
    fn largest_chunk_picks_the_biggest_fit() {
        let cost = cost();
        let sizes = [512, 256, 128, 64, 32, 16, 8, 4, 1];
        let pick = |candidates: &[u32], remaining: u64| {
            largest_chunk(&cost, candidates, remaining).unwrap()
        };
        assert_eq!(pick(&sizes, 1_000_000_000), Some(16));
        assert_eq!(pick(&sizes, 997_536_000), Some(16), "exact fit");
        assert_eq!(pick(&sizes, 997_535_999), Some(8), "one byte short");
        assert_eq!(pick(&sizes, 3_000_000_000), Some(64));
        assert_eq!(pick(&sizes, 100_000_000), None, "not even one token fits");
        assert_eq!(
            pick(&[16, 512, 1, 64], 3_000_000_000),
            Some(64),
            "any order"
        );
        assert_eq!(pick(&[], 3_000_000_000), None, "no candidates");
        assert!(is_bad_request(largest_chunk(&cost, &[4, 0], u64::MAX)));
    }

    /// Tokens 0 to 2 select (3, 8), (4, 20), and (5, 7). The union is 3, 4, 5, 7, 8, 20.
    fn run_route() -> ChunkRoute {
        union_and_remap(
            &[
                vec![(3, 0.5), (8, 0.5)],
                vec![(4, 0.25), (20, 0.75)],
                vec![(5, 0.125), (7, 0.875)],
            ],
            512,
        )
        .unwrap()
    }

    /// Cap 2 over the five experts 0, 2, 5, 7, 9 of `selections()` gives the waves {0, 2}, {5, 7},
    /// and {9}. The pair ranks are 2, 1, 4, 1, 3, 2, 4, 0, 3 for p = 0 to 8, with 3 pairs per
    /// token, so the pairs in (rank, pair) order are p7 (rank 0), p1 and p3 (rank 1), p0 and p5
    /// (rank 2), p4 and p8 (rank 3), and p2 and p6 (rank 4).
    #[test]
    fn split_waves_assigns_pairs_local_ranks_and_the_inverse() {
        let route = union_and_remap(&selections(), 512).unwrap();
        let plan = split_waves(&route, 2).unwrap();
        let experts: Vec<&[u32]> = plan.waves.iter().map(|w| w.experts.as_slice()).collect();
        assert_eq!(
            experts,
            [&[0, 2][..], &[5, 7][..], &[9][..]],
            "wave experts"
        );
        // Wave 0 (ranks 0 and 1, local rank = rank) takes p7, then p1 and p3: pairs 7, 1, 3 of
        // tokens 2, 0, 1 with gates 0.35, 0.3, 0.6 and local ranks 0, 1, 1.
        let wave0 = &plan.waves[0];
        assert_eq!(wave0.pairs, [7, 1, 3], "wave 0 pairs");
        assert_eq!(wave0.local_ranks, [0, 1, 1], "wave 0 local ranks");
        assert_eq!(wave0.tokens, [2, 0, 1], "wave 0 tokens");
        assert_eq!(wave0.gates, [0.35, 0.3, 0.6], "wave 0 gates");
        // Wave 1 (ranks 2 and 3) takes p0 and p5 of rank 2, then p4 and p8 of rank 3: pairs
        // 0, 5, 4, 8 of tokens 0, 1, 1, 2 with gates 0.5, 0.1, 0.3, 0.25 and local ranks 0, 0, 1, 1.
        let wave1 = &plan.waves[1];
        assert_eq!(wave1.pairs, [0, 5, 4, 8], "wave 1 pairs");
        assert_eq!(wave1.local_ranks, [0, 0, 1, 1], "wave 1 local ranks");
        assert_eq!(wave1.tokens, [0, 1, 1, 2], "wave 1 tokens");
        assert_eq!(wave1.gates, [0.5, 0.1, 0.3, 0.25], "wave 1 gates");
        // Wave 2 (rank 4) takes p2 and p6 in pair order.
        let wave2 = &plan.waves[2];
        assert_eq!(wave2.pairs, [2, 6], "wave 2 pairs");
        assert_eq!(wave2.local_ranks, [0, 0], "wave 2 local ranks");
        assert_eq!(wave2.tokens, [0, 2], "wave 2 tokens");
        assert_eq!(wave2.gates, [0.2, 0.4], "wave 2 gates");
        // The concatenation of the pair lists is 7, 1, 3, 0, 5, 4, 8, 2, 6, so p0 sits at
        // position 3, p1 at 1, p2 at 7, p3 at 2, p4 at 5, p5 at 4, p6 at 8, p7 at 0, and p8 at 6.
        assert_eq!(
            plan.inverse,
            [3, 1, 7, 2, 5, 4, 8, 0, 6],
            "inverse permutation"
        );
    }

    /// Cap 4 over the union 3, 4, 5, 7, 8, 20 gives {3, 4, 5, 7} and {8, 20}. The pair ranks are
    /// 0, 4, 1, 5, 2, 3 for p = 0 to 5, so the (rank, pair) order is p0, p2, p4, p5, p1, p3: the
    /// rank order happens to coincide with the token-major order for this route, so the expected
    /// lists are the same as before the rank sort. `coalesce_runs` on each wave gives wave-local
    /// ranks.
    #[test]
    fn split_waves_keeps_runs_and_ranks_local_to_each_wave() {
        let route = run_route();
        assert_eq!(route.distinct, [3, 4, 5, 7, 8, 20]);
        assert_eq!(route.rank_idx, [0, 4, 1, 5, 2, 3]);
        let plan = split_waves(&route, 4).unwrap();
        assert_eq!(plan.waves.len(), 2, "two waves");
        let wave0 = &plan.waves[0];
        assert_eq!(wave0.experts, [3, 4, 5, 7]);
        assert_eq!(wave0.pairs, [0, 2, 4, 5]);
        assert_eq!(wave0.local_ranks, [0, 1, 2, 3]);
        assert_eq!(wave0.tokens, [0, 1, 2, 2]);
        assert_eq!(wave0.gates, [0.5, 0.25, 0.125, 0.875]);
        let wave1 = &plan.waves[1];
        assert_eq!(wave1.experts, [8, 20]);
        assert_eq!(wave1.pairs, [1, 3]);
        assert_eq!(wave1.local_ranks, [0, 1]);
        assert_eq!(wave1.tokens, [0, 1]);
        assert_eq!(wave1.gates, [0.5, 0.75]);
        assert_eq!(plan.inverse, [0, 4, 1, 5, 2, 3], "inverse permutation");
        assert_eq!(
            coalesce_runs(&wave0.experts).unwrap(),
            [
                ExpertRun {
                    first: 3,
                    rank: 0,
                    len: 3
                },
                ExpertRun {
                    first: 7,
                    rank: 3,
                    len: 1
                },
            ],
            "wave 0 runs"
        );
        assert_eq!(
            coalesce_runs(&wave1.experts).unwrap(),
            [
                ExpertRun {
                    first: 8,
                    rank: 0,
                    len: 1
                },
                ExpertRun {
                    first: 20,
                    rank: 1,
                    len: 1
                },
            ],
            "wave 1 runs restart at rank 0"
        );
    }

    /// A cap at or above the union (five experts) leaves one wave that holds every pair in
    /// (rank, pair) order: p7 (rank 0), p1 and p3 (rank 1), p0 and p5 (rank 2), p4 and p8
    /// (rank 3), and p2 and p6 (rank 4). The local ranks of that wave are the chunk ranks in that
    /// order, non-decreasing, and the inverse of the concatenation restores pair order.
    #[test]
    fn split_waves_with_a_large_cap_is_one_rank_sorted_wave() {
        let route = union_and_remap(&selections(), 512).unwrap();
        for cap in [5, 6, u32::MAX] {
            let plan = split_waves(&route, cap).unwrap();
            assert_eq!(plan.waves.len(), 1, "cap {cap}");
            let wave = &plan.waves[0];
            assert_eq!(wave.experts, [0, 2, 5, 7, 9], "cap {cap}: experts");
            assert_eq!(
                wave.pairs,
                [7, 1, 3, 0, 5, 4, 8, 2, 6],
                "cap {cap}: pairs in (rank, pair) order"
            );
            assert_eq!(
                wave.local_ranks,
                [0, 1, 1, 2, 2, 3, 3, 4, 4],
                "cap {cap}: local ranks are non-decreasing"
            );
            assert_eq!(
                wave.tokens,
                [2, 0, 1, 0, 1, 1, 2, 0, 2],
                "cap {cap}: tokens"
            );
            assert_eq!(
                wave.gates,
                [0.35, 0.3, 0.6, 0.5, 0.1, 0.3, 0.25, 0.2, 0.4],
                "cap {cap}: gates"
            );
            assert_eq!(
                plan.inverse,
                [3, 1, 7, 2, 5, 4, 8, 0, 6],
                "cap {cap}: inverse"
            );
        }
    }

    /// The order proof that the wave import runs before any device work of the wave. Hand
    /// derivations: `[0, 1, 1, 3, 5]` never falls and the empty list has no adjacent pair, so both
    /// pass; `[0, 2, 1]` falls at the last window (2 > 1), `[2, 1, 3]` falls at the first window
    /// (2 > 1), and the strictly descending `[5, 4, 3, 2, 1]` falls at every window, so all three
    /// fail.
    #[test]
    fn require_sorted_ranks_passes_sorted_and_empty_and_fails_one_inversion() {
        assert!(
            require_sorted_ranks(&[0, 1, 1, 3, 5]).is_ok(),
            "a sorted list passes"
        );
        assert!(require_sorted_ranks(&[]).is_ok(), "the empty list passes");
        for (list, why) in [
            (&[0u32, 2, 1][..], "one inversion at the end"),
            (&[2u32, 1, 3][..], "one inversion at the start"),
            (&[5u32, 4, 3, 2, 1][..], "a strictly descending list"),
        ] {
            let failed = require_sorted_ranks(list);
            assert!(
                matches!(&failed, Err(EngError::BadRequest(m)) if m.starts_with("flash-next prefill:")),
                "{why} ({list:?}) is a bad request that names the flash-next prefill, got {failed:?}"
            );
        }
    }

    /// A hand-written route of the property test: token-major rows of (expert, gate) pairs.
    type Selections = Vec<Vec<(u32, f32)>>;

    /// The hand-written route family of the property test: seven token-major routes over experts
    /// below 64, no RNG.
    fn property_routes() -> Vec<(&'static str, Selections)> {
        vec![
            ("one token one expert", vec![vec![(5, 1.0)]]),
            (
                "one token three experts",
                vec![vec![(9, 0.1), (3, 0.6), (6, 0.3)]],
            ),
            (
                "every token the same two experts",
                vec![
                    vec![(2, 0.5), (4, 0.5)],
                    vec![(4, 0.6), (2, 0.4)],
                    vec![(2, 0.7), (4, 0.3)],
                    vec![(4, 0.8), (2, 0.2)],
                    vec![(2, 0.9), (4, 0.1)],
                ],
            ),
            (
                "the selections family",
                vec![
                    vec![(5, 0.5), (2, 0.3), (9, 0.2)],
                    vec![(2, 0.6), (7, 0.3), (5, 0.1)],
                    vec![(9, 0.4), (0, 0.35), (7, 0.25)],
                ],
            ),
            (
                "descending rows",
                vec![
                    vec![(7, 0.2), (5, 0.3), (3, 0.5)],
                    vec![(6, 0.1), (4, 0.4), (2, 0.5)],
                ],
            ),
            (
                "two experts swapped between tokens",
                vec![
                    vec![(0, 0.6), (8, 0.4)],
                    vec![(8, 0.7), (0, 0.3)],
                    vec![(0, 0.1), (8, 0.9)],
                ],
            ),
            (
                "six tokens of four choices",
                vec![
                    vec![(1, 0.4), (15, 0.3), (2, 0.2), (9, 0.1)],
                    vec![(9, 0.25), (2, 0.25), (1, 0.25), (15, 0.25)],
                    vec![(15, 0.1), (9, 0.2), (2, 0.3), (1, 0.4)],
                    vec![(2, 0.34), (1, 0.33), (9, 0.32), (15, 0.01)],
                    vec![(0, 0.5), (3, 0.2), (6, 0.2), (9, 0.1)],
                    vec![(10, 0.6), (11, 0.2), (12, 0.1), (13, 0.1)],
                ],
            ),
        ]
    }

    /// For every route of the family and every wave cap, each wave lists its pairs with
    /// non-decreasing local ranks that still name the pair's rank, token, and gate in the route,
    /// the concatenated pair lists taken at `inverse` restore the token-major pair order, and the
    /// multiset of (token, gate, rank) triples is unchanged.
    #[test]
    fn split_waves_sorts_ranks_and_keeps_the_inverse_and_the_multiset_over_a_family() {
        for (name, selections) in property_routes() {
            let route = union_and_remap(&selections, 64).unwrap();
            let top_k = route.top_k as usize;
            let pairs = route.rank_idx.len();
            // The token-major multiset of (token, gate, rank).
            let mut want: Vec<(u32, f32, u32)> = route
                .rank_idx
                .iter()
                .zip(&route.gates)
                .enumerate()
                .map(|(pair, (&rank, &gate))| (u32::try_from(pair / top_k).unwrap(), gate, rank))
                .collect();
            want.sort_by(|a, b| a.0.cmp(&b.0).then(a.2.cmp(&b.2)).then(a.1.total_cmp(&b.1)));
            for cap in [1u32, 2, 3, 5, 64] {
                let plan = split_waves(&route, cap).unwrap();
                let mut concatenated: Vec<u32> = Vec::with_capacity(pairs);
                let mut got: Vec<(u32, f32, u32)> = Vec::with_capacity(pairs);
                for (index, wave) in plan.waves.iter().enumerate() {
                    let offset = u32::try_from(index).unwrap() * cap;
                    assert!(
                        wave.local_ranks.is_sorted(),
                        "{name}, cap {cap}: wave {index} lists non-decreasing local ranks"
                    );
                    let rows = wave
                        .pairs
                        .iter()
                        .zip(&wave.local_ranks)
                        .zip(&wave.tokens)
                        .zip(&wave.gates);
                    for (((&pair, &local), &token), &gate) in rows {
                        let pair = pair as usize;
                        assert_eq!(
                            route.rank_idx[pair],
                            offset + local,
                            "{name}, cap {cap}: pair {pair} keeps its rank"
                        );
                        assert_eq!(
                            token,
                            u32::try_from(pair / top_k).unwrap(),
                            "{name}, cap {cap}: pair {pair} keeps its token"
                        );
                        assert_eq!(
                            gate, route.gates[pair],
                            "{name}, cap {cap}: pair {pair} keeps its gate"
                        );
                        concatenated.push(pair as u32);
                        got.push((token, gate, offset + local));
                    }
                }
                assert_eq!(
                    concatenated.len(),
                    pairs,
                    "{name}, cap {cap}: every pair joins exactly one wave"
                );
                for (pair, &slot) in plan.inverse.iter().enumerate() {
                    assert_eq!(
                        concatenated[slot as usize], pair as u32,
                        "{name}, cap {cap}: the inverse restores pair {pair}"
                    );
                }
                got.sort_by(|a, b| a.0.cmp(&b.0).then(a.2.cmp(&b.2)).then(a.1.total_cmp(&b.1)));
                assert_eq!(
                    got, want,
                    "{name}, cap {cap}: the (token, gate, rank) multiset"
                );
            }
        }
    }

    /// A route of more than 20 pairs whose local ranks repeat many times, so the (rank, pair)
    /// order of `split_waves` is pinned at a size that no small-slice luck can pass: two hundred
    /// tokens of two choices over the two experts 0 and 1, where even tokens pick (0, 1) and odd
    /// tokens pick (1, 0). Hand derivation: the distinct experts are 0 and 1, and the pair
    /// `p = t * 2 + j` has rank 0 when `p` is 0 or 3 modulo 4 (the first choice of an even token
    /// and the second of an odd one) and rank 1 when `p` is 1 or 2 modulo 4, so each rank is
    /// shared by the 200 pairs of its two residues, interleaved in pair order. For every wave and
    /// every cap, equal adjacent local ranks must imply an ascending pair index: the stable sort
    /// of `split_waves` gives that order, and an unstable sort of these 400 pairs, above the
    /// 20-element insertion-sort bound of `sort_unstable_by_key`, can swap pairs of one rank.
    #[test]
    fn split_waves_keeps_equal_local_ranks_in_pair_order_over_a_large_route() {
        let selections: Vec<Vec<(u32, f32)>> = (0..200)
            .map(|token| {
                if token % 2 == 0 {
                    vec![(0, 0.75), (1, 0.25)]
                } else {
                    vec![(1, 0.25), (0, 0.75)]
                }
            })
            .collect();
        let route = union_and_remap(&selections, 64).unwrap();
        assert_eq!(route.distinct, [0, 1], "two experts");
        assert_eq!(route.rank_idx.len(), 400, "200 tokens of two pairs");
        for cap in [1u32, 2, 3, 5, 64] {
            let plan = split_waves(&route, cap).unwrap();
            for (index, wave) in plan.waves.iter().enumerate() {
                assert!(
                    wave.local_ranks.is_sorted(),
                    "cap {cap}: wave {index} lists non-decreasing local ranks"
                );
                assert!(
                    wave.local_ranks.windows(2).any(|two| two[0] == two[1]),
                    "cap {cap}: wave {index} repeats a local rank, or the route tests no tie"
                );
                for (pairs, ranks) in wave.pairs.windows(2).zip(wave.local_ranks.windows(2)) {
                    if ranks[0] == ranks[1] {
                        assert!(
                            pairs[0] < pairs[1],
                            "cap {cap}: wave {index}: pair {} follows pair {} inside one rank",
                            pairs[1],
                            pairs[0]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn split_waves_rejects_a_zero_cap_and_an_inconsistent_route() {
        let route = union_and_remap(&selections(), 512).unwrap();
        assert!(is_bad_request(split_waves(&route, 0)), "cap zero");
        let with = |change: &dyn Fn(&mut ChunkRoute)| {
            let mut broken = route.clone();
            change(&mut broken);
            split_waves(&broken, 2)
        };
        assert!(
            is_bad_request(with(&|r| r.rank_idx[0] = 5)),
            "rank equal to the expert count"
        );
        assert!(
            is_bad_request(with(&|r| r.rank_idx[0] = 6)),
            "rank inside the last wave but beyond the experts"
        );
        assert!(
            is_bad_request(with(&|r| {
                r.gates.pop();
            })),
            "short gates"
        );
        assert!(
            is_bad_request(with(&|r| r.top_k = 4)),
            "pairs do not divide by top-k"
        );
        assert!(is_bad_request(with(&|r| r.top_k = 0)), "top-k zero");
        assert!(
            is_bad_request(with(&|r| r.distinct[1] = 0)),
            "repeated expert"
        );
        assert!(is_bad_request(with(&|r| r.distinct.clear())), "no experts");
        // Expert 9 (rank 4) is the only member of wave 2. Moving its pairs to rank 3 leaves an
        // empty wave.
        assert!(
            is_bad_request(with(&|r| {
                for rank in &mut r.rank_idx {
                    if *rank == 4 {
                        *rank = 3;
                    }
                }
            })),
            "a wave without a pair"
        );
    }

    #[test]
    fn wave_count_rounds_up() {
        assert_eq!(wave_count(512, 128).unwrap(), 4);
        assert_eq!(wave_count(513, 128).unwrap(), 5);
        assert_eq!(wave_count(1, 128).unwrap(), 1);
        assert_eq!(wave_count(128, 128).unwrap(), 1);
        assert_eq!(wave_count(0, 128).unwrap(), 0);
        assert!(is_bad_request(wave_count(10, 0)));
    }

    #[test]
    fn wave_syncs_counts_one_evaluate_per_extra_wave_in_each_layer() {
        assert_eq!(
            wave_syncs(&[4; 48]).unwrap(),
            144,
            "four waves in 48 layers"
        );
        assert_eq!(
            wave_syncs(&[3; 48]).unwrap(),
            96,
            "three waves in 48 layers"
        );
        assert_eq!(wave_syncs(&[1; 48]).unwrap(), 0, "one wave adds nothing");
        assert_eq!(wave_syncs(&[1, 2, 3, 4]).unwrap(), 6, "0 + 1 + 2 + 3");
        assert_eq!(wave_syncs(&[]).unwrap(), 0, "no layers");
        assert!(
            is_bad_request(wave_syncs(&[2, 0])),
            "a layer without a wave"
        );
    }

    /// `sets * held * expert_bytes + tokens * per_token + fixed` with `held` capped at 128:
    /// T = 16 holds min(160, 128) = 128 experts, so 2 * 128 * 2,764,800 = 707,788,800.
    #[test]
    fn a_wave_cap_bounds_the_page_sets_of_the_workspace() {
        let capped = cost().with_wave_cap(128).unwrap();
        assert_eq!(capped.experts, 128);
        assert_eq!(
            cost().with_wave_cap(600).unwrap().experts,
            512,
            "a cap above the experts"
        );
        assert!(is_bad_request(cost().with_wave_cap(0)));
        assert_eq!(cost().set_experts(1).unwrap(), 10);
        assert_eq!(cost().set_experts(51).unwrap(), 510);
        assert_eq!(cost().set_experts(52).unwrap(), 512);
        assert_eq!(capped.set_experts(12).unwrap(), 120);
        assert_eq!(capped.set_experts(13).unwrap(), 128);
        assert_eq!(capped.set_experts(512).unwrap(), 128);
        assert_eq!(
            capped.workspace(1).unwrap(),
            156_096_000,
            "10 experts, as without waves"
        );
        assert_eq!(capped.workspace(16).unwrap(), 820_588_800);
        assert_eq!(capped.workspace(512).unwrap(), 1_217_388_800);
        // 1 GB: without waves the largest chunk is 16 tokens. With 128-expert waves it is 128.
        let sizes = [512, 256, 128, 64, 32, 16, 8, 4, 1];
        assert_eq!(
            largest_chunk(&cost(), &sizes, 1_000_000_000).unwrap(),
            Some(16)
        );
        assert_eq!(
            largest_chunk(&capped, &sizes, 1_000_000_000).unwrap(),
            Some(128)
        );
    }

    /// 512 MiB is 536,870,912 B and 64 MiB is 67,108,864 B.
    #[test]
    fn the_prefill_retire_cap_is_a_flat_512_mib_over_the_64_mib_base() {
        assert_eq!(PREFILL_RETIRE_CAP, 536_870_912, "prefill cap");
        assert_eq!(RETIRE_CAP_BASE, 67_108_864, "decode cap");
    }

    /// The cap goes into `fixed` only: 100,000,000 + 536,870,912 = 636,870,912 B. At 512 tokens the
    /// workspace is 2,831,155,200 + 512 * 800,000 + 636,870,912 = 3,877,626,112 B, and at one token
    /// it is 55,296,000 + 800,000 + 636,870,912 = 692,966,912 B.
    #[test]
    fn the_admission_cost_covers_the_prefill_retire_cap() {
        let capped = cost().with_retire_cap().unwrap();
        assert_eq!(capped.fixed, 636_870_912, "fixed bytes");
        assert_eq!(capped.per_token, 800_000, "bytes per token do not change");
        assert_eq!(capped.workspace(1).unwrap(), 692_966_912, "one token");
        assert_eq!(capped.workspace(512).unwrap(), 3_877_626_112, "512 tokens");
    }

    #[test]
    fn the_admission_cost_reports_retire_cap_overflow() {
        let fixed = ChunkCost {
            fixed: u64::MAX,
            ..cost()
        };
        assert!(matches!(
            fixed.with_retire_cap(),
            Err(EngError::MemoryPlan(_))
        ));
    }

    /// Four ranges of `top_k` words per token and set: 2 sets * 4 ranges * 4 bytes * 10 = 320 B per
    /// token. Four page roundings per set: 2 * 4 * 16,384 = 131,072 B. At one token the workspace is
    /// 55,296,000 + 800,320 + 100,131,072 = 156,227,392 B.
    #[test]
    fn the_route_pages_add_four_ranges_to_every_set() {
        let routed = cost().with_route_pages(16_384).unwrap();
        assert_eq!(routed.per_token, 800_320, "bytes per token");
        assert_eq!(routed.fixed, 100_131_072, "fixed bytes");
        assert_eq!(routed.workspace(1).unwrap(), 156_227_392, "one token");
    }

    #[test]
    fn the_route_pages_report_overflow() {
        let per_token = ChunkCost {
            per_token: u64::MAX,
            ..cost()
        };
        assert!(matches!(
            per_token.with_route_pages(16_384),
            Err(EngError::MemoryPlan(_))
        ));
        let fixed = ChunkCost {
            fixed: u64::MAX,
            ..cost()
        };
        assert!(matches!(
            fixed.with_route_pages(16_384),
            Err(EngError::MemoryPlan(_))
        ));
        assert!(matches!(
            cost().with_route_pages(u64::MAX),
            Err(EngError::MemoryPlan(_))
        ));
    }

    /// A set holds `min(experts, wave cap, tokens * top_k)` experts.
    #[test]
    fn the_prefill_sets_follow_the_wave_cap_and_the_token_count() {
        assert_eq!(
            prefill_set_experts(512, 10, 512, 1).unwrap(),
            10,
            "one token"
        );
        assert_eq!(
            prefill_set_experts(512, 10, 512, 64).unwrap(),
            512,
            "all experts"
        );
        assert_eq!(
            prefill_set_experts(512, 10, 128, 64).unwrap(),
            128,
            "wave cap"
        );
        assert_eq!(
            prefill_set_experts(512, 10, 128, 5).unwrap(),
            50,
            "few tokens"
        );
        assert_eq!(
            prefill_set_experts(512, 10, 1000, 100).unwrap(),
            512,
            "cap above"
        );
        assert_eq!(
            prefill_set_experts(512, 10, 128, 64).unwrap(),
            cost().with_wave_cap(128).unwrap().set_experts(64).unwrap(),
            "the cost rule"
        );
        assert!(
            is_bad_request(prefill_set_experts(512, 10, 0, 1)),
            "zero cap"
        );
    }

    /// Real dimensions: 3 * 48 * 128 * 128 * 4 = 9,437,184 B (9 MiB). Fixture: 3 * 2 * 32 * 32 * 4 = 24,576 B.
    #[test]
    fn delta_retire_bytes_follow_the_state_dimensions() {
        assert_eq!(delta_retire_per_token(48, 128, 128).unwrap(), 9_437_184);
        assert_eq!(delta_retire_per_token(2, 32, 32).unwrap(), 24_576);
        assert!(
            is_bad_request(delta_retire_per_token(0, 128, 128)),
            "zero heads"
        );
        assert!(
            is_bad_request(delta_retire_per_token(u32::MAX, u32::MAX, u32::MAX)),
            "overflow"
        );
    }

    /// `floor((cap - 67,108,864) / 9,437,184)`: 469,762,048 / 9,437,184 = 49.78, so 49 under the
    /// prefill cap. A chunk of fewer tokens takes all its tokens in one part. A cap of 64 MiB leaves
    /// no room, so the size is 1. The fixture cap `64 MiB + 3 * 24,576` gives 3.
    #[test]
    fn the_delta_subchunk_fits_the_retire_cap_room() {
        let p = 9_437_184;
        let size = |tokens, cap| delta_subchunk(tokens, cap, p).unwrap();
        assert_eq!(size(512, PREFILL_RETIRE_CAP), 49);
        assert_eq!(size(2048, PREFILL_RETIRE_CAP), 49);
        assert_eq!(size(50, PREFILL_RETIRE_CAP), 49);
        assert_eq!(size(49, PREFILL_RETIRE_CAP), 49);
        assert_eq!(size(16, PREFILL_RETIRE_CAP), 16, "never above the chunk");
        assert_eq!(size(512, RETIRE_CAP_BASE), 1, "no room above 64 MiB");
        assert_eq!(
            size(512, RETIRE_CAP_BASE + p - 1),
            1,
            "room below one token"
        );
        assert_eq!(size(512, RETIRE_CAP_BASE + p), 1, "room for one token");
        assert_eq!(size(512, RETIRE_CAP_BASE + 2 * p), 2, "room for two tokens");
        let fixture = RETIRE_CAP_BASE + 3 * 24_576;
        assert_eq!(delta_subchunk(7, fixture, 24_576).unwrap(), 3);
        assert_eq!(
            delta_subchunk(2, fixture, 24_576).unwrap(),
            2,
            "never above the chunk"
        );
        assert_eq!(
            delta_subchunk(3, 1, 24_576).unwrap(),
            1,
            "a cap below 64 MiB"
        );
        assert!(
            is_bad_request(delta_subchunk(0, fixture, 24_576)),
            "zero tokens"
        );
        assert!(
            is_bad_request(delta_subchunk(7, fixture, 0)),
            "zero bytes per token"
        );
    }

    /// 512 tokens in sub-chunks of 49 make ceil(512 / 49) = 11 parts, so 10 evaluations per layer
    /// over 36 DeltaNet layers: 360. 64 tokens of 49 make 2 parts: 36. 64 tokens of 1 make 63 per
    /// layer: 2,268. The fixture has one DeltaNet layer: 7 tokens in parts of 3 make 3 parts (2
    /// extra), 3 tokens make 1 part, and 4 tokens make 2 parts.
    #[test]
    fn the_planned_delta_evaluations_follow_the_part_count() {
        assert_eq!(delta_extra_syncs(36, 512, 49).unwrap(), 360);
        assert_eq!(delta_extra_syncs(36, 64, 49).unwrap(), 36);
        assert_eq!(delta_extra_syncs(36, 16, 16).unwrap(), 0);
        assert_eq!(delta_extra_syncs(36, 64, 1).unwrap(), 2_268);
        assert_eq!(delta_extra_syncs(1, 7, 3).unwrap(), 2);
        assert_eq!(delta_extra_syncs(1, 3, 3).unwrap(), 0);
        assert_eq!(delta_extra_syncs(1, 4, 3).unwrap(), 1);
        assert!(is_bad_request(delta_extra_syncs(1, 4, 0)), "zero size");
        assert!(is_bad_request(delta_extra_syncs(1, 0, 3)), "zero tokens");
    }

    /// 4 tokens in parts of 3 or 2, 7 in parts of 3, and a part size above the chunk.
    #[test]
    fn the_delta_ranges_cover_the_chunk_without_empty_parts() {
        assert_eq!(delta_ranges(4, 3).unwrap(), [(0, 3), (3, 4)]);
        assert_eq!(delta_ranges(4, 2).unwrap(), [(0, 2), (2, 4)]);
        assert_eq!(delta_ranges(7, 3).unwrap(), [(0, 3), (3, 6), (6, 7)]);
        assert_eq!(delta_ranges(3, 3).unwrap(), [(0, 3)]);
        assert_eq!(delta_ranges(3, 5).unwrap(), [(0, 3)]);
        assert_eq!(delta_ranges(1, 1).unwrap(), [(0, 1)]);
        assert!(is_bad_request(delta_ranges(0, 3)), "zero tokens");
        assert!(is_bad_request(delta_ranges(4, 0)), "zero size");
        for (tokens, sub) in [(512u32, 49u32), (64, 1), (65, 7)] {
            let ranges = delta_ranges(tokens, sub).unwrap();
            let extra = delta_extra_syncs(1, tokens, sub).unwrap();
            assert_eq!(
                ranges.len() as u64 - 1,
                extra,
                "parts for {tokens} in {sub}"
            );
            assert!(ranges.iter().all(|&(a, b)| a < b), "no empty part");
        }
    }

    /// Real dimensions: 512 experts, top-k 10, 2,764,800 B per expert, 16,384 B pages. Hand
    /// arithmetic: `per_token` = 800,000 + 2 sets * 160 B of route words; `fixed` = 100,000,000 +
    /// 536,870,912 (retire cap) + 2 * 4 * 16,384 (route page rounding) = 637,001,984.
    /// `workspace(T)` = 2 * min(10 T, 512) * 2,764,800 + T * 800,320 + 637,001,984.
    #[test]
    fn the_admission_cost_has_the_planning_terms_the_retire_cap_and_the_route_pages() {
        let cost = admission_cost(512, 10, 2_764_800, 16_384).unwrap();
        assert_eq!(
            cost,
            ChunkCost {
                experts: 512,
                top_k: 10,
                expert_bytes: 2_764_800,
                sets: 2,
                per_token: 800_320,
                fixed: 637_001_984,
            }
        );
        assert_eq!(cost.workspace(1).unwrap(), 693_098_304, "T 1");
        assert_eq!(cost.workspace(512).unwrap(), 3_877_921_024, "T 512");
        assert_eq!(cost.workspace(2048).unwrap(), 5_107_212_544, "T 2048");
    }

    #[test]
    fn the_admission_cost_caps_the_experts_of_a_set_at_the_wave_cap() {
        let big = admission_cost(1024, 10, 2_764_800, 16_384).unwrap();
        assert_eq!(big.experts, PREFILL_WAVE_CAP, "1024 experts run in waves");
        let small = admission_cost(128, 10, 2_764_800, 16_384).unwrap();
        assert_eq!(small.experts, 128, "a small layer keeps its expert count");
    }

    #[test]
    fn the_remaining_bytes_are_the_budget_minus_every_term_and_never_negative() {
        assert_eq!(prefill_remaining(100, 10, 20, 30, 5), 35, "all terms");
        assert_eq!(prefill_remaining(100, 10, 20, 30, 40), 0, "reserve exceeds");
        assert_eq!(prefill_remaining(100, 100, 0, 0, 0), 0, "exactly spent");
        assert_eq!(
            prefill_remaining(100, 0, 0, 101, 0),
            0,
            "cache alone exceeds"
        );
        assert_eq!(
            prefill_remaining(u64::MAX, u64::MAX, 1, 0, 0),
            0,
            "no wrap-around"
        );
    }

    /// The terms print in the order of the admission rule with the decode pool as an extra term,
    /// and `remaining_bytes` equals the budget minus every charged term.
    #[test]
    fn the_charged_terms_print_every_byte_value_and_the_remaining_bytes() {
        let terms = PrefillTerms {
            budget: 1_000,
            weights: 100,
            decode_pool: 20,
            s3_cache: 30,
            session_reserve: 400,
        };
        assert_eq!(terms.remaining(), 450, "budget minus the four terms");
        assert_eq!(
            terms.to_string(),
            "budget_bytes=1000\nweights_bytes=100\ndecode_pool_bytes=20\n\
             s3_cache_bytes=30\nsession_reserve_bytes=400\nremaining_bytes=450\n"
        );
        let spent = PrefillTerms {
            session_reserve: 2_000,
            ..terms
        };
        assert_eq!(spent.remaining(), 0, "never negative");
        assert!(
            spent.to_string().ends_with("remaining_bytes=0\n"),
            "printed"
        );
    }

    /// The workspaces of the real-dimension cost: 2048 -> 5,107,212,544; 1024 -> 4,287,684,864;
    /// 512 -> 3,877,921,024; 4 -> 861,387,264; 1 -> 693,098,304.
    #[test]
    fn the_admitted_chunk_is_the_largest_candidate_that_fits_and_one_is_no_chunk() {
        let cost = admission_cost(512, 10, 2_764_800, 16_384).unwrap();
        let admitted = |remaining| admitted_chunk(&cost, remaining).unwrap();
        assert_eq!(admitted(5_107_212_544), Some(2048), "exactly 2048");
        assert_eq!(admitted(5_107_212_543), Some(1024), "one byte less");
        assert_eq!(admitted(4_000_000_000), Some(512), "between 512 and 1024");
        assert_eq!(admitted(861_387_264), Some(4), "exactly 4");
        assert_eq!(
            admitted(861_387_263),
            None,
            "only 1 fits: the one-token path"
        );
        assert_eq!(admitted(693_098_304), None, "exactly 1: the one-token path");
        assert_eq!(admitted(693_098_303), None, "nothing fits");
        assert_eq!(admitted(0), None, "no remaining budget");
    }
}
