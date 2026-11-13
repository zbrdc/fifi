//! The size rules of an EXL3 routed job table: pure arithmetic, shared by the host-side scratch
//! planners (which size the sort lists and the job table) and the device-side job-table
//! machinery in [`crate::family::project::cuda::exl3_jobs`].

/// The most rows of one job of a routed expert: the narrowest CTA (64 rows). An expert holds a few
/// dozen rows of a chunk, and a wider CTA would decode no fewer weights and run mostly idle warps.
pub(crate) const EXL3_EXPERT_TILE_ROWS: usize = 64;

/// The widest rows window the shared EXL3 expert machinery plans for: `shared_rows`
/// (`family::exl3::moe`) and the Flash-Next chunk plan (`MAX_STEP_ROWS`) refuse wider
/// windows, and the chunk scratch is sized at this width. A kernel cap it is not — the dense
/// job row cap is `jobs_row_cap` (256, exl3.cu) and the shared down groups split at 8 — this
/// is the planning bound both families agreed on.
pub(crate) const MAX_EXL3_ROWS: usize = 64;

/// i32 words of one job of the device table.
pub(crate) const EXL3_JOB_WORDS: usize = 4;

/// The most jobs of one routed pass over `routed` pairs and `experts` experts: one per expert
/// with rows plus one per full tile beyond the first of an expert.
pub(crate) fn exl3_max_jobs(experts: usize, routed: usize) -> usize {
    experts + routed.div_ceil(EXL3_EXPERT_TILE_ROWS)
}
