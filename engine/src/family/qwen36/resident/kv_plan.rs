//! Portable planning of the attention capacity cache (packet Q36-KV-INTEGRATE).
//!
//! The cache of one attention layer is a pair of preallocated F32 capacity buffers
//! `[1, KVH, cap, D]` (keys and values) written in place by the consuming `kv_append`: the first
//! `len` positions are valid and the rest is slack. The arithmetic here — the capacity rounding,
//! the capacity after an append, and the logical byte count that the session budget charges —
//! calls no native code, so the tests run on Linux.
//!
//! The growth TRANSIENT at a capacity growth — the old buffer, its slice copy, and the new
//! buffer alive together, about 100 MB per layer at depth 8,192 — is NOT part of the slack
//! bound below: the session reserve margin and the retire cap cover it.

use crate::EngError;

/// The growth quantum of the capacity buffers, in positions (packet Q36-KV-INTEGRATE): a buffer
/// grows to a multiple of this many positions, so the slack of one layer stays below one quantum
/// of its growth bytes and the slack of a whole session below one quantum per attention layer
/// (256 * 4,096 * 10 = 10,485,760 bytes on the real model).
pub(super) const KV_STEP: usize = 256;

fn bad(message: &str) -> EngError {
    EngError::BadRequest(format!("q36 kv plan: {message}"))
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
    /// model). The slack of one layer stays below one quantum — at most `KV_STEP - 1` positions
    /// — so the session slack stays strictly below the bound.
    #[test]
    fn the_growth_quantum_pins_the_slack_bound_of_the_docs() {
        assert_eq!(KV_STEP, 256, "the documented slack arithmetic assumes 256");
        assert!((KV_STEP - 1) as u64 * 4_096 * 10 < 10_485_760);
        assert_eq!(KV_STEP as u64 * 4_096 * 10, 10_485_760);
    }
}
