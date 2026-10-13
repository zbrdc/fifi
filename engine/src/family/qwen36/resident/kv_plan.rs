//! Portable planning of the attention capacity cache (packet Q36-KV-INTEGRATE) and the
//! depth-dependent prefill chunk of packet KV8-INTEGRATE.
//!
//! The cache of one attention layer is written in place by the consuming `kv_append`: the first
//! `len` positions are valid and the rest is slack. In the F32 form (`KvDtype::F32`, the
//! control) it is a pair of F32 buffers `[1, KVH, cap, D]`; in the Q8 form (`KvDtype::Q8`,
//! decision 0046 A7) it is the 8-bit affine codes `[1, KVH, cap, D/4]` beside scales and biases
//! `[1, KVH, cap, D/64]` of the ACTIVATION dtype (`settings::kv_growth_per_position`, the one
//! byte authority). The arithmetic here — the capacity rounding, the capacity after an append,
//! the logical byte count that the session budget charges, and the score-matrix bound that
//! shrinks a prefill chunk at depth — calls no native code, so the tests run on Linux.
//!
//! The logical charge per layer and position is 4,096 bytes in the F32 form on the real model
//! (KVH 2, D 256; 40,960 over its 10 attention layers), and in the Q8 form 1,088 bytes at BF16
//! scales (10,880 over the 10 layers) or 1,152 at F32 scales (11,520). The session slack stays
//! below one `KV_STEP` of the charge per attention layer: 10,485,760 bytes in the F32 form,
//! 2,785,280 at Q8/BF16, 2,949,120 at Q8/F32 scales.
//!
//! The growth TRANSIENT at a capacity growth — the old buffer, its slice copy, and the new
//! buffer alive together, about 100 MB per layer at depth 8,192 in the F32 form — is NOT part
//! of the slack bound below: the session reserve margin and the retire cap cover it.

use crate::EngError;

/// The growth quantum of the capacity buffers, in positions (packet Q36-KV-INTEGRATE): a buffer
/// grows to a multiple of this many positions, so the slack of one layer stays below one quantum
/// of its growth bytes and the slack of a whole session below one quantum per attention layer
/// (in the F32 form 256 * 4,096 * 10 = 10,485,760 bytes on the real model; at Q8 with BF16
/// scales 256 * 1,088 * 10 = 2,785,280).
pub(super) const KV_STEP: usize = 256;

/// The score-matrix byte budget of one attention layer of one prefill chunk (packet
/// KV8-INTEGRATE item 6): 512 MiB. The F32 score matrix of a chunk of `L` rows at `len`
/// positions is `rep * L * len * 4` bytes (`rep = QH / KVH`, 8 on the real model: 2.1 GB at
/// L = 256 and len = 262,144), and the softmax and the mask add copies on top, so the planner
/// halves the chunk once the matrix alone would pass the budget.
pub(super) const SCORE_BUDGET_BYTES: u64 = 512 << 20;

fn bad(message: &str) -> EngError {
    EngError::BadRequest(format!("q36 kv plan: {message}"))
}

/// The largest power of two at most `requested` whose score matrix at a cache of `len`
/// positions — `rep * rows * len * 4` bytes, the packet's formula — stays within
/// `score_budget`, never below 16 rows and never above `requested` (packet KV8-INTEGRATE
/// item 6). The chunk runs to `len + rows` positions and the next chunk plans from there, so
/// the split of a prompt is a deterministic function of the prompt length, the start position,
/// and the requested setting. With the real shape (`rep` 8) and `SCORE_BUDGET_BYTES` the answer
/// is 256 up to len 65,536 (where `8 * 256 * 65,536 * 4` is exactly the budget) and halves
/// beyond: 128 at 131,072 and 64 at 262,144. A `requested` below 16 is honored as is: the
/// floor bounds the budget's squeeze, not the caller's own request. A smaller chunk changes
/// the BF16 rounding of the prefill (finding Q36-BF16-SENS), so any gate that compares a
/// chunked prefill must compare it at the same depth schedule.
pub(super) fn chunk_rows_for_depth(
    requested: usize,
    len: usize,
    rep: u64,
    score_budget: u64,
) -> usize {
    let requested = requested.max(1);
    let len = len as u64;
    let fits = |rows: u64| {
        rep.checked_mul(rows)
            .and_then(|cells| cells.checked_mul(len))
            .and_then(|cells| cells.checked_mul(4))
            .is_some_and(|bytes| bytes <= score_budget)
    };
    // The largest power of two at most `requested`, halved until the matrix fits.
    let mut rows: u64 = 1;
    while rows
        .checked_mul(2)
        .is_some_and(|doubled| doubled <= requested as u64)
    {
        rows *= 2;
    }
    while rows > 1 && !fits(rows) {
        rows /= 2;
    }
    let floored = rows.max(16);
    usize::try_from(floored)
        .unwrap_or(usize::MAX)
        .min(requested)
}

/// The chunk ranges `(start, stop)` of `rows` prompt rows after a cache of `start_len` valid
/// positions, each of at most the depth-aware chunk of `chunk_rows_for_depth` at its own
/// position (packet KV8-INTEGRATE item 6): the prefill planner's one split, a deterministic
/// function of the prompt length, the start position, and the setting. `rep` is the query
/// repeats per key-value head (`QH / KVH`). Every range is non-empty and together they cover
/// `0..rows` exactly, as the fixed-size split did before the depth rule.
pub(super) fn depth_ranges(
    start_len: u32,
    rows: u32,
    requested: u32,
    rep: u64,
    score_budget: u64,
) -> Result<Vec<(u32, u32)>, EngError> {
    if rows == 0 || requested == 0 {
        return Err(bad("depth ranges need positive rows and a requested chunk"));
    }
    let (requested, mut begin, mut len) = (
        usize::try_from(requested).map_err(|_| bad("the requested chunk exceeds the platform"))?,
        0usize,
        u64::from(start_len),
    );
    let total = usize::try_from(rows).map_err(|_| bad("the prompt rows exceed the platform"))?;
    let mut out = Vec::new();
    while begin < total {
        let len_now = usize::try_from(len).map_err(|_| bad("the cache length exceeds usize"))?;
        let take = chunk_rows_for_depth(requested, len_now, rep, score_budget).min(total - begin);
        let stop = begin
            .checked_add(take)
            .ok_or_else(|| bad("the chunk range overflows"))?;
        let (a, b) = (
            u32::try_from(begin).map_err(|_| bad("the chunk range exceeds u32"))?,
            u32::try_from(stop).map_err(|_| bad("the chunk range exceeds u32"))?,
        );
        out.push((a, b));
        begin = stop;
        len += u64::try_from(take).map_err(|_| bad("the chunk length exceeds u64"))?;
    }
    Ok(out)
}

/// Rounds `positions` up to a multiple of `step`. A zero step or an overflow is `BadRequest`.
pub(super) fn round_up(positions: usize, step: usize) -> Result<usize, EngError> {
    if step == 0 {
        return Err(bad("the cache growth step is zero"));
    }
    positions
        .div_ceil(step)
        .checked_mul(step)
        .ok_or_else(|| bad("the cache capacity overflows usize"))
}

/// The capacity after appending `rows` positions to a buffer of `len` valid positions and `cap`
/// allocated positions: `cap` when the rows fit, else `round_up(len + rows, step)`. An empty
/// cache (`len == 0`, `cap == 0`) allocates `round_up(rows, step)` positions for its first rows.
pub(super) fn next_capacity(
    len: usize,
    rows: usize,
    cap: usize,
    step: usize,
) -> Result<usize, EngError> {
    let needed = len
        .checked_add(rows)
        .ok_or_else(|| bad("the cache length overflows usize"))?;
    if needed <= cap {
        return Ok(cap);
    }
    round_up(needed, step)
}

/// The LOGICAL byte count of `len` positions at `per_position` bytes each: the admission charge
/// of the cache, which the capacity slack never enters (packet Q36-KV-INTEGRATE). An overflow is
/// `MemoryPlan`.
pub(super) fn logical_bytes(len: usize, per_position: u64) -> Result<u64, EngError> {
    u64::try_from(len)
        .ok()
        .and_then(|len| len.checked_mul(per_position))
        .ok_or_else(|| EngError::MemoryPlan("q36 kv plan: the logical bytes overflow u64".into()))
}

/// The LOGICAL byte count of the attention caches of a whole state: `layers` attention layers
/// that each hold `len` valid positions of `per_layer` bytes per position, so the charge counts
/// `len * per_layer` once per layer and equals the position times the growth of one position
/// over ALL layers (`layers * per_layer`). Charging each layer the ALL-layer growth instead
/// multiplies the charge by the layer count — the admission bug of the fix round, which denied
/// every step with `MemoryPlan`. An overflow is `MemoryPlan`.
pub(super) fn session_bytes(layers: usize, len: usize, per_layer: u64) -> Result<u64, EngError> {
    let one_layer = logical_bytes(len, per_layer)?;
    let layers = u64::try_from(layers)
        .map_err(|_| EngError::MemoryPlan("q36 kv plan: the layer count overflows u64".into()))?;
    one_layer.checked_mul(layers).ok_or_else(|| {
        EngError::MemoryPlan("q36 kv plan: the session logical bytes overflow u64".into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_up_moves_to_the_next_multiple_of_the_step() {
        // The real model's quantum: the first position allocates 256.
        assert_eq!(round_up(1, 256).unwrap(), 256);
        assert_eq!(round_up(256, 256).unwrap(), 256, "an exact multiple stays");
        assert_eq!(
            round_up(257, 256).unwrap(),
            512,
            "one over moves a whole step"
        );
        // The tiny-model test quantum.
        assert_eq!(round_up(7, 3).unwrap(), 9);
        assert_eq!(round_up(6, 3).unwrap(), 6);
        // No position needs no buffer.
        assert_eq!(round_up(0, 8).unwrap(), 0);
    }

    #[test]
    fn round_up_rejects_a_zero_step_and_an_overflow() {
        assert!(round_up(5, 0).is_err(), "a zero step has no multiple");
        assert!(
            round_up(usize::MAX, 2).is_err(),
            "usize::MAX rounded up to a multiple of 2 overflows"
        );
    }

    #[test]
    fn next_capacity_keeps_a_fitting_buffer_and_grows_in_steps() {
        // The first rows of an empty cache allocate a whole quantum.
        assert_eq!(next_capacity(0, 1, 0, 256).unwrap(), 256);
        assert_eq!(next_capacity(0, 5, 0, 4).unwrap(), 8);
        // Rows that fit keep the capacity: 2 valid of 3, one more row fits exactly.
        assert_eq!(next_capacity(2, 1, 3, 3).unwrap(), 3);
        assert_eq!(
            next_capacity(3, 1, 4, 3).unwrap(),
            4,
            "an exact fit does not grow"
        );
        // One row past the capacity grows to the next multiple of the step.
        assert_eq!(next_capacity(3, 1, 3, 3).unwrap(), 6);
        assert_eq!(next_capacity(5, 1, 4, 3).unwrap(), 6);
        // A chunk wider than the free space: 2 valid of 4, five rows need 7, so 8.
        assert_eq!(next_capacity(2, 5, 4, 4).unwrap(), 8);
        // A chunk wider than a whole step crosses several quanta at once.
        assert_eq!(next_capacity(10, 512, 16, 256).unwrap(), 512 + 256);
    }

    #[test]
    fn next_capacity_rejects_a_length_overflow() {
        assert!(
            next_capacity(usize::MAX, 1, 0, 256).is_err(),
            "one more position past usize::MAX overflows"
        );
    }

    #[test]
    fn logical_bytes_multiply_positions_by_growth() {
        // The tiny layer: 2 key-value heads of width 32, keys and values, F32: 512 per position.
        assert_eq!(logical_bytes(9, 512).unwrap(), 4_608);
        // The real layer: 2 * 256 * 2 * 4 = 4,096 per position.
        assert_eq!(logical_bytes(100, 4_096).unwrap(), 409_600);
        assert_eq!(logical_bytes(KV_STEP, 4_096).unwrap(), 1_048_576);
        // The session slack bound of the decision: one quantum per attention layer.
        assert_eq!(10 * logical_bytes(KV_STEP, 4_096).unwrap(), 10_485_760);
        // An empty cache charges nothing.
        assert_eq!(logical_bytes(0, 4_096).unwrap(), 0);
    }

    #[test]
    fn logical_bytes_reject_an_overflow() {
        assert!(
            logical_bytes(usize::MAX, u64::MAX).is_err(),
            "usize::MAX positions of u64::MAX bytes overflow"
        );
    }

    /// Packet Q36-KV-INTEGRATE fix1 item 1: the session charge counts the length once per layer
    /// with the PER-LAYER growth, so a state at `pos` charges `pos * kv_growth_bytes(weights)`.
    /// The expectations are hand-derived and would fail under the fix-round bug, which fed the
    /// all-layer growth (`kv_growth_bytes`) into the per-layer formula and charged `n_attn`
    /// times too much.
    #[test]
    fn session_bytes_count_each_layer_once() {
        // The tiny model: two attention layers of 512 bytes per position, 1,024 over both, so
        // three positions charge 3 * 1,024 = 3,072 (not 2 * 3 * 1,024 = 6,144).
        assert_eq!(session_bytes(2, 3, 512).unwrap(), 3_072);
        // The real model: ten attention layers of 4,096 bytes per position, 40,960 over all, so
        // one position charges 40,960 and one quantum of positions 256 * 40,960.
        assert_eq!(session_bytes(10, 1, 4_096).unwrap(), 40_960);
        assert_eq!(
            session_bytes(10, KV_STEP, 4_096).unwrap(),
            KV_STEP as u64 * 40_960
        );
        // A model without attention layers charges nothing, whatever the length.
        assert_eq!(session_bytes(0, 512, 4_096).unwrap(), 0);
    }

    #[test]
    fn session_bytes_reject_an_overflow() {
        assert!(
            session_bytes(2, usize::MAX, u64::MAX).is_err(),
            "two layers of usize::MAX positions of u64::MAX bytes overflow"
        );
        assert!(
            session_bytes(usize::MAX, 1, 2).is_err(),
            "usize::MAX layers of two bytes overflow"
        );
    }

    /// Review B finding 3: the constant itself is pinned, with the slack arithmetic of the four
    /// doc comments that state the session bound (256 * 4,096 * 10 = 10,485,760 on the real
    /// model in the F32 form; 256 * 1,088 * 10 = 2,785,280 at Q8 with BF16 scales). The slack of
    /// one layer stays below one quantum — at most `KV_STEP - 1` positions — so the session
    /// slack stays strictly below each bound.
    #[test]
    fn the_growth_quantum_pins_the_slack_bound_of_the_docs() {
        assert_eq!(KV_STEP, 256, "the documented slack arithmetic assumes 256");
        assert!((KV_STEP - 1) as u64 * 4_096 * 10 < 10_485_760);
        assert_eq!(KV_STEP as u64 * 4_096 * 10, 10_485_760);
        assert!((KV_STEP - 1) as u64 * 1_088 * 10 < 2_785_280);
        assert_eq!(KV_STEP as u64 * 1_088 * 10, 2_785_280);
    }

    /// Packet KV8-INTEGRATE item 6, hand-derived at the real shape (`rep` 8) with
    /// `SCORE_BUDGET_BYTES`: 256 up to len 65,536 — where `8 * 256 * 65,536 * 4` is exactly the
    /// budget — and halving beyond. A `requested` below 16 is the caller's own request and is
    /// honored; the floor of 16 bounds the budget's squeeze; a non-power-of-two `requested`
    /// quantizes down to its power of two.
    #[test]
    fn chunk_rows_for_depth_returns_256_to_len_65k_and_halves_beyond() {
        assert_eq!(SCORE_BUDGET_BYTES, 536_870_912, "512 MiB");
        // The budget arithmetic the table rests on: 2.1 GB is the packet's example.
        assert_eq!(8u64 * 256 * 65_536 * 4, 536_870_912);
        assert_eq!(8u64 * 256 * 262_144 * 4, 2_147_483_648);
        let table = [
            (32_768usize, 256usize),
            (65_536, 256),
            (131_072, 128),
            (262_144, 64),
        ];
        for (len, want) in table {
            assert_eq!(
                chunk_rows_for_depth(256, len, 8, SCORE_BUDGET_BYTES),
                want,
                "the chunk at len {len}"
            );
        }
        // One past the exact fit halves.
        assert_eq!(
            chunk_rows_for_depth(256, 65_537, 8, SCORE_BUDGET_BYTES),
            128
        );
        // The floor of 16: even a len whose matrix at 16 rows passes the budget keeps 16, and a
        // requested below 16 is honored as the caller's request (the tiny-model test knobs).
        assert_eq!(
            chunk_rows_for_depth(256, 1 << 24, 8, SCORE_BUDGET_BYTES),
            16
        );
        assert_eq!(chunk_rows_for_depth(3, 0, 8, SCORE_BUDGET_BYTES), 3);
        assert_eq!(chunk_rows_for_depth(2, 1_000, 2, SCORE_BUDGET_BYTES), 2);
        // A non-power-of-two requested value quantizes down to its power of two.
        assert_eq!(chunk_rows_for_depth(100, 0, 8, SCORE_BUDGET_BYTES), 64);
        // At len 0 every requested power of two fits.
        assert_eq!(chunk_rows_for_depth(2_048, 0, 8, SCORE_BUDGET_BYTES), 2_048);
    }

    /// The prefill split of `depth_ranges`: the fixed-size cover of the old fixed split at a depth
    /// the budget admits everywhere, the depth-aware halving inside one piece at 65,536, and
    /// the tiny-model knobs that keep their own size. Deterministic in the prompt length, the
    /// start position, and the setting.
    #[test]
    fn depth_ranges_split_a_prompt_by_the_depth_schedule() {
        // Shallow: the old fixed split.
        assert_eq!(
            depth_ranges(0, 600, 256, 8, SCORE_BUDGET_BYTES).unwrap(),
            vec![(0, 256), (256, 512), (512, 600)]
        );
        // A prompt after 65,536 cached positions: the first chunk is 256 (the exact fit), the
        // second plans at 65,792 where 256 no longer fits, so it halves to 128, and so does the
        // third.
        assert_eq!(
            depth_ranges(65_536, 512, 256, 8, SCORE_BUDGET_BYTES).unwrap(),
            vec![(0, 256), (256, 384), (384, 512)]
        );
        // The tiny-model knob of three rows keeps three rows at every depth of a test.
        assert_eq!(
            depth_ranges(0, 9, 3, 2, SCORE_BUDGET_BYTES).unwrap(),
            vec![(0, 3), (3, 6), (6, 9)]
        );
        // One row is one range, and the same inputs give the same ranges.
        let once = depth_ranges(5, 1, 256, 8, SCORE_BUDGET_BYTES).unwrap();
        assert_eq!(once, vec![(0, 1)]);
        assert_eq!(
            depth_ranges(5, 1, 256, 8, SCORE_BUDGET_BYTES).unwrap(),
            once,
            "the split is deterministic"
        );
        // Zero rows or a zero requested chunk is refused.
        assert!(depth_ranges(0, 0, 256, 8, SCORE_BUDGET_BYTES).is_err());
        assert!(depth_ranges(0, 4, 0, 8, SCORE_BUDGET_BYTES).is_err());
    }
}
