//! Bounded staging for grouped resident controller → attention projections.

use super::{CudaContext, DensePlan, DenseStep, EngError, controller_residency};
use crate::family::inkling2::switches::Switch;

pub(super) const TILE_ROWS: usize = 8;
const MAX_BYTES: usize = 4 << 20;

/// The prompt tile sizes, widest first: the admission below steps down them by the free memory,
/// and the exact projection kernels take a whole tile of one of them. 512 is the most rows one
/// relative attention snapshot serves.
pub(in crate::family::inkling2) const PROMPT_TILES: [usize; 4] = [512, 256, 128, 64];

/// The most rows a prompt tile takes: the widest prompt tile when both wide weight paths are on,
/// else the legacy 8.
pub(super) fn target_rows() -> usize {
    if Switch::PrefillQ8Visit.on() && Switch::PrefillWeightWide.on() {
        PROMPT_TILES[0]
    } else {
        TILE_ROWS
    }
}

pub(super) fn enabled() -> bool {
    Switch::PrefillControllerWorkspace.on()
}

pub(super) fn admit(
    context: &CudaContext,
    [rank, hidden, max_output]: [usize; 3],
    expert_bytes: u64,
) -> Result<Option<DenseStep>, EngError> {
    let shape = [rank, hidden, max_output];
    let Some(floor) = controller_residency::workspace_floor(expert_bytes) else {
        return Ok(None);
    };
    let free = context.memory_info()?.free as u64;
    for rows in PROMPT_TILES.into_iter().chain([TILE_ROWS]) {
        if rows > target_rows() {
            continue;
        }
        let candidate = if rows == TILE_ROWS {
            plan(shape)?
        } else {
            bounded_plan(shape, rows, rows * (512 << 10))?
        };
        let Some(plan) = candidate.filter(|p| fits(free, p.device_bytes, floor)) else {
            continue;
        };
        let step = match DenseStep::new(context, &plan) {
            Ok(step) => step,
            Err(EngError::MemoryPlan(_)) => continue,
            Err(e) => return Err(e),
        };
        if context.memory_info()?.free as u64 >= floor {
            return Ok(Some(step));
        }
        // Drop the oversized arena before considering the next smaller budget.
    }
    Ok(None)
}

fn fits(free: u64, bytes: usize, floor: u64) -> bool {
    free.checked_sub(bytes as u64)
        .is_some_and(|after| after >= floor)
}

fn plan(shape: [usize; 3]) -> Result<Option<DensePlan>, EngError> {
    bounded_plan(shape, TILE_ROWS, MAX_BYTES)
}

fn bounded_plan(
    [rank, hidden, max_output]: [usize; 3],
    rows: usize,
    budget: usize,
) -> Result<Option<DensePlan>, EngError> {
    if !(1..=4096).contains(&hidden) || !(1..=4096).contains(&rank) || max_output == 0 {
        return Ok(None);
    }
    let Some(words) = DenseStep::controlled_attention_row_words([rank, hidden], max_output)
        .and_then(|n| n.checked_mul(rows))
        .filter(|&n| n <= budget / size_of::<f32>())
    else {
        return Ok(None);
    };
    let plan = DensePlan::new([words, 2 * hidden], hidden, rank)?;
    Ok((plan.device_bytes <= budget).then_some(plan))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controller_workspace_bounds_shape_and_reserve() {
        let [rank, hidden, output] = [64, 2304, 8 * 3072];
        let budget = plan([rank, hidden, output]).unwrap().unwrap().device_bytes;
        let words = TILE_ROWS * (24 * hidden + 3 * rank + 3 + output);
        assert!(budget >= 4 * words && budget <= MAX_BYTES);
        for shape in [
            [0, hidden, output],
            [rank, hidden, 0],
            [rank, hidden, usize::MAX],
            [rank, 4097, output],
            [rank, hidden, MAX_BYTES],
        ] {
            assert!(plan(shape).unwrap().is_none(), "{shape:?}");
        }
        let floor = controller_residency::workspace_floor(212459524).unwrap();
        let exact = floor + budget as u64;
        assert!(fits(exact, budget, floor));
        assert!(!fits(exact - 1, budget, floor));
        assert!(!fits(0, budget, floor));
    }
}
