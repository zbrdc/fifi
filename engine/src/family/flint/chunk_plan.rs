//! Budget-driven chunk boundaries for a Flint prefill request - the portable
//! half of the chunked-prefill proposal. The batch forward itself is the [T, H]
//! interface spec agreed with the `mlx_forward` owner; this module only decides
//! how many rows each chunk carries, refusing (MemoryPlan) when even one row
//! cannot fit the remaining budget.

// The driver slice (the per-chunk forward_piece loop) wires this; until then the
// planner is constructed only by its tests.
#![allow(dead_code)]

use crate::EngError;
use std::ops::Range;

/// Plans chunk row counts from a byte budget and a per-row cost. The cost is the
/// biggest per-row intermediate the batch forward holds (the caller measures or
/// plans it); the planner itself stays pure and portable.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FlintChunkPlanner {
    row_bytes: u64,
    max_rows: usize,
}

impl FlintChunkPlanner {
    /// `row_bytes` must cover one row of the batch intermediates; `max_rows` is the
    /// engine's serving bound on one chunk (the [T, H] interface's own limit).
    pub(crate) fn new(row_bytes: u64, max_rows: usize) -> Result<Self, EngError> {
        if row_bytes == 0 {
            return Err(EngError::BadRequest(
                "Flint chunk planner needs a positive per-row cost".into(),
            ));
        }
        if max_rows == 0 {
            return Err(EngError::BadRequest(
                "Flint chunk planner needs a positive chunk bound".into(),
            ));
        }
        Ok(Self {
            row_bytes,
            max_rows,
        })
    }

    /// The rows of the next chunk for `remaining` rows under `budget_bytes` free
    /// for prefill: the whole tail when it fits, else the budget-bound largest
    /// count, else a MemoryPlan refusal (a prompt that cannot run one row at a
    /// time cannot run at all).
    pub(crate) fn next_chunk(
        &self,
        remaining: usize,
        budget_bytes: u64,
    ) -> Result<usize, EngError> {
        if remaining == 0 {
            return Ok(0);
        }
        let rows = (budget_bytes / self.row_bytes)
            .try_into()
            .unwrap_or(usize::MAX);
        if rows == 0 {
            return Err(EngError::MemoryPlan(format!(
                "Flint prefill: one row costs {} bytes, the budget holds none",
                self.row_bytes
            )));
        }
        Ok(rows.min(remaining).min(self.max_rows))
    }

    /// The full boundary list for a prompt of `rows` rows under `budget_bytes`.
    pub(crate) fn chunks(
        &self,
        rows: usize,
        budget_bytes: u64,
    ) -> Result<Vec<Range<usize>>, EngError> {
        let mut out = Vec::new();
        let mut start = 0;
        while start < rows {
            let n = self.next_chunk(rows - start, budget_bytes)?;
            out.push(start..start + n);
            start += n;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planner(row_bytes: u64, max_rows: usize) -> FlintChunkPlanner {
        FlintChunkPlanner::new(row_bytes, max_rows).unwrap()
    }

    #[test]
    fn the_tail_fits_whole_and_clamps_to_the_chunk_bound() {
        let p = planner(1_024, 8);
        assert_eq!(p.next_chunk(5, u64::MAX).unwrap(), 5, "the whole tail");
        assert_eq!(p.next_chunk(20, u64::MAX).unwrap(), 8, "the serving bound");
    }

    #[test]
    fn the_budget_bounds_the_count_and_zero_refuses() {
        let p = planner(1_024, 8);
        assert_eq!(p.next_chunk(20, 5_120).unwrap(), 5, "the budget row count");
        let refused = p.next_chunk(20, 1_023).unwrap_err();
        assert!(matches!(refused, EngError::MemoryPlan(_)), "{refused:?}");
    }

    #[test]
    fn an_empty_prompt_yields_no_boundaries() {
        let p = planner(1_024, 8);
        assert_eq!(p.next_chunk(0, 0).unwrap(), 0);
        assert!(p.chunks(0, 0).unwrap().is_empty());
    }

    #[test]
    fn the_boundary_list_covers_exactly_and_ends_on_the_tail() {
        let p = planner(1_024, 8);
        let bounds = p.chunks(21, 8_192).unwrap();
        assert_eq!(bounds.len(), 3, "{bounds:?}");
        assert_eq!(bounds[0], 0..8);
        assert_eq!(bounds[2], 16..21, "the tail carries the remainder");
        let covered: usize = bounds.iter().map(|r| r.len()).sum();
        assert_eq!(covered, 21);
    }

    #[test]
    fn construction_refuses_degenerate_costs() {
        assert!(FlintChunkPlanner::new(0, 8).is_err());
        assert!(FlintChunkPlanner::new(1_024, 0).is_err());
    }
}
