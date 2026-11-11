//! The size rules of an EXL3 routed job table: pure arithmetic, shared by the host-side scratch
//! planners (which size the sort lists and the job table) and the device-side job-table
//! machinery in [`crate::family::project::cuda::exl3_jobs`].

/// The most rows of one job of a routed expert: the narrowest CTA (64 rows). An expert holds a few
/// dozen rows of a chunk, and a wider CTA would decode no fewer weights and run mostly idle warps.
pub(crate) const EXL3_EXPERT_TILE_ROWS: usize = 64;

/// i32 words of one job of the device table.
pub(crate) const EXL3_JOB_WORDS: usize = 4;

/// The most jobs of one routed pass over `routed` pairs and `experts` experts: one per expert
/// with rows plus one per full tile beyond the first of an expert.
pub(crate) fn exl3_max_jobs(experts: usize, routed: usize) -> usize {
    experts + routed.div_ceil(EXL3_EXPERT_TILE_ROWS)
}
