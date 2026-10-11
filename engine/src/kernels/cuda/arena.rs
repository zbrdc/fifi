//! Static device-arena planning (CUDA-RD section 2, packet RD-1).
//!
//! [`plan_arena`] gives every tensor a byte offset inside one allocation.
//! Tensors that are live in the same step never share an address; tensors
//! whose lifetimes do not overlap may. It is the greedy-by-size technique:
//! tensors are placed largest first, each at the lowest [`ARENA_ALIGN`]-aligned
//! offset that no tensor live at the same time already covers.
//!
//! The planner is pure arithmetic and calls no CUDA function, so it runs and
//! is tested on any host. Its owner is `context::workspace`.

use std::cmp::Reverse;

use crate::EngError;

/// Offsets are multiples of this many bytes. `cudaMalloc` returns 256-byte
/// aligned memory, so every offset keeps that alignment in the allocation.
pub(super) const ARENA_ALIGN: usize = 256;

/// One tensor of an arena: its size and the steps in which it is live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ArenaTensor {
    /// Size in bytes; never 0.
    pub(super) bytes: usize,
    /// First step in which the tensor is live.
    pub(super) first: usize,
    /// Last step in which the tensor is live; not before `first`.
    pub(super) last: usize,
}

impl ArenaTensor {
    fn shares_a_step_with(&self, other: &Self) -> bool {
        self.first <= other.last && other.first <= self.last
    }
}

/// Where each tensor lives, in input order, and the bytes the arena needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ArenaPlan {
    pub(super) offsets: Vec<usize>,
    /// The end of the highest tensor; 0 for no tensors.
    pub(super) total: usize,
}

/// Plans `tensors` into one arena.
///
/// Returns `BadRequest` for a tensor of 0 bytes or one that ends before it
/// starts, and `MemoryPlan` when an offset or the total overflows `usize`.
pub(super) fn plan_arena(tensors: &[ArenaTensor]) -> Result<ArenaPlan, EngError> {
    for (index, tensor) in tensors.iter().enumerate() {
        if tensor.bytes == 0 || tensor.first > tensor.last {
            return Err(EngError::BadRequest(format!(
                "cuda arena tensor {index} has {} bytes live from step {} to step {}",
                tensor.bytes, tensor.first, tensor.last
            )));
        }
    }
    // The sort is stable, so tensors of equal size keep their input order.
    let mut order: Vec<usize> = (0..tensors.len()).collect();
    order.sort_by_key(|&index| Reverse(tensors[index].bytes));
    let mut offsets = vec![0usize; tensors.len()];
    let mut total = 0usize;
    for (done, &index) in order.iter().enumerate() {
        let offset = lowest_offset(tensors, &offsets, &order[..done], index)?;
        offsets[index] = offset;
        total = total.max(end_of(offset, tensors[index].bytes)?);
    }
    Ok(ArenaPlan { offsets, total })
}

/// The lowest aligned offset at which tensor `index` avoids every tensor of
/// `placed` that is live in one of its steps.
fn lowest_offset(
    tensors: &[ArenaTensor],
    offsets: &[usize],
    placed: &[usize],
    index: usize,
) -> Result<usize, EngError> {
    let tensor = tensors[index];
    let mut busy = Vec::new();
    for &other in placed {
        if tensor.shares_a_step_with(&tensors[other]) {
            busy.push((
                offsets[other],
                end_of(offsets[other], tensors[other].bytes)?,
            ));
        }
    }
    busy.sort_unstable();
    let mut candidate = 0usize;
    for (start, end) in busy {
        if end_of(candidate, tensor.bytes)? <= start {
            break;
        }
        candidate = candidate.max(align_up(end)?);
    }
    Ok(candidate)
}

fn overflow() -> EngError {
    EngError::MemoryPlan("cuda arena: offsets overflow the address space".to_owned())
}

fn end_of(offset: usize, bytes: usize) -> Result<usize, EngError> {
    offset.checked_add(bytes).ok_or_else(overflow)
}

fn align_up(value: usize) -> Result<usize, EngError> {
    value
        .checked_next_multiple_of(ARENA_ALIGN)
        .ok_or_else(overflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn live(bytes: usize, first: usize, last: usize) -> ArenaTensor {
        ArenaTensor { bytes, first, last }
    }

    fn planned(tensors: &[ArenaTensor]) -> ArenaPlan {
        plan_arena(tensors).expect("plan the arena")
    }

    fn together(a: &ArenaTensor, b: &ArenaTensor) -> bool {
        a.first <= b.last && b.first <= a.last
    }

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// 24 tensors of 1 to 20,000 bytes, each live for 1 to 5 of 8 steps.
    fn random_tensors(seed: u64) -> Vec<ArenaTensor> {
        let mut state = seed;
        let mut below = |limit: u64| usize::try_from(xorshift(&mut state) % limit).expect("small");
        (0..24)
            .map(|_| {
                let bytes = 1 + below(20_000);
                let first = below(8);
                live(bytes, first, first + below(5))
            })
            .collect()
    }

    #[test]
    fn disjoint_lifetimes_share_an_offset() {
        let plan = planned(&[live(1000, 0, 1), live(600, 2, 3)]);
        assert_eq!(plan.offsets, vec![0, 0]);
        assert_eq!(plan.total, 1000);
    }

    #[test]
    fn the_largest_tensor_goes_first_at_the_lowest_aligned_offsets() {
        let tensors = [live(300, 0, 0), live(5000, 0, 0), live(700, 0, 0)];
        let plan = planned(&tensors);
        assert_eq!(plan.offsets, vec![5888, 0, 5120]);
        assert_eq!(plan.total, 6188);
    }

    #[test]
    fn live_together_tensors_never_overlap_and_offsets_are_aligned() {
        for seed in [1u64, 7, 42] {
            let tensors = random_tensors(seed);
            let plan = planned(&tensors);
            for (i, a) in tensors.iter().enumerate() {
                // The literal 256 is the cudaMalloc alignment the planner promises.
                assert_eq!(plan.offsets[i] % 256, 0, "seed {seed} tensor {i}");
                assert!(
                    plan.offsets[i] + a.bytes <= plan.total,
                    "seed {seed} tensor {i}"
                );
                for (j, b) in tensors.iter().enumerate().skip(i + 1) {
                    let apart = plan.offsets[i] + a.bytes <= plan.offsets[j]
                        || plan.offsets[j] + b.bytes <= plan.offsets[i];
                    assert!(!together(a, b) || apart, "seed {seed}: {i} and {j} overlap");
                }
            }
        }
    }

    #[test]
    fn offsets_equal_a_brute_force_lowest_fit() {
        for seed in [3u64, 11, 99] {
            let tensors = random_tensors(seed);
            let plan = planned(&tensors);
            let mut order: Vec<usize> = (0..tensors.len()).collect();
            order.sort_by_key(|&i| Reverse(tensors[i].bytes));
            let mut placed: Vec<(usize, usize)> = Vec::new();
            for &i in &order {
                let lowest = (0usize..)
                    .step_by(256)
                    .find(|&candidate| {
                        placed
                            .iter()
                            .filter(|&&(j, _)| together(&tensors[i], &tensors[j]))
                            .all(|&(j, offset)| {
                                candidate + tensors[i].bytes <= offset
                                    || offset + tensors[j].bytes <= candidate
                            })
                    })
                    .expect("some offset is free");
                assert_eq!(plan.offsets[i], lowest, "seed {seed} tensor {i}");
                placed.push((i, lowest));
            }
        }
    }

    #[test]
    fn a_tensor_fills_the_gap_between_two_live_neighbours() {
        // K lives with H (0..2000) and I (3072..4572), so it goes in between.
        let tensors = [
            live(3000, 0, 0), // G
            live(2000, 1, 1), // H
            live(1500, 0, 1), // I
            live(900, 0, 0),  // J
            live(900, 1, 1),  // K
        ];
        let plan = planned(&tensors);
        assert_eq!(plan.offsets, vec![0, 0, 3072, 4608, 2048]);
        assert_eq!(plan.total, 5508);
    }

    #[test]
    fn zero_bytes_and_inverted_lifetimes_are_bad_requests() {
        for tensors in [[live(0, 0, 0)], [live(8, 3, 2)]] {
            let error = plan_arena(&tensors).expect_err("invalid tensor");
            assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
        }
    }

    #[test]
    fn an_overflowing_plan_is_a_memory_plan_error() {
        // The first tensor ends at 2^64 - 256; the second would end past 2^64.
        let tensors = [live(usize::MAX - 255, 0, 0), live(300, 0, 0)];
        let error = plan_arena(&tensors).expect_err("offsets overflow");
        assert!(matches!(error, EngError::MemoryPlan(_)), "{error:?}");
    }

    #[test]
    fn an_alignment_that_overflows_is_a_memory_plan_error() {
        // The first tensor ends at 2^64 - 101, which still fits, but the next
        // 256-byte boundary is 2^64. Only the alignment step can overflow here.
        let tensors = [live(usize::MAX - 100, 0, 0), live(300, 0, 0)];
        let error = plan_arena(&tensors).expect_err("the aligned offset overflows");
        assert!(matches!(error, EngError::MemoryPlan(_)), "{error:?}");
    }

    #[test]
    fn no_tensors_plan_to_an_empty_arena() {
        let plan = planned(&[]);
        assert_eq!(
            plan,
            ArenaPlan {
                offsets: Vec::new(),
                total: 0
            }
        );
    }
}
