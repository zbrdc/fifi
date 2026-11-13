//! The shared prefill core, promoted from the first chunked family: the admission arithmetic
//! every chunked family shares, and the CUDA arena-fill and chunk-driver primitives. Family
//! plans stay with their families; nothing here names one.
//!
//! The family constants are parameters, not imports: the spare minimum, the flat part charge
//! and the ladder of chunk sizes are supplied by the calling family, so the shared arithmetic
//! stays identical for the family whose goldens pin it while another family binds its own
//! values.

// Host (non-CUDA) builds have no production consumer yet: the promoting family's CUDA driver
// re-exports these under native-cuda, and the second consumer imports them ungated later.
#[allow(
    dead_code,
    reason = "the promoting family's CUDA driver consumes this under native-cuda; the second family's ungated planner lands in the shared-pass series"
)]
pub(crate) mod admission;

// Same host-build situation as `admission`: the CUDA-gated arena consumes this, and the second
// family's ungated planner arrives with the shared-pass series.
#[allow(
    dead_code,
    reason = "the gated arena module is its only current consumer; the second family's ungated planner lands in the shared-pass series"
)]
pub(crate) mod steps;

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in crate::family) mod arena;
