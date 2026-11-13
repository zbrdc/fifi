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
#[allow(dead_code)]
pub(crate) mod admission;

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in crate::family) mod arena;
