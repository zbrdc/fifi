//! The chunk-step list type of the shared prefill core, ungated: a family's tier plan (a pure
//! host planner, no CUDA call) builds `Vec<PrefillStep>` and hands it to the gated
//! [`crate::family::prefill::arena::drive_steps`], so the enum lives where both sides can name it.

/// One step of the family's chunk list: `rows >= 2` rows from position `p0` through the chunk
/// path, or a run of exactly one row at `p0` through the per-token path, so a 1-row tail never
/// needs a chunk plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrefillStep {
    Chunk { p0: usize, rows: usize },
    Feed { p0: usize },
}
