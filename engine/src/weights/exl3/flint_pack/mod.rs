//! Flint EXL3 pack adapter and audit: CPU only, never serving.
//!
//! The 0xBakeer Flint packs are plain exllamav3 packs with one Flint convention: the routed
//! experts' `up_proj` is stored divided by `quantization_config.expert_interm_div` (128), and the
//! runtime multiplies the normalized routing weights by that factor. [`RoutedScale`] reads the
//! factor from the model config and applies it to an existing `Routed`; the shared router, the
//! shared expert and the other projections are not touched.
//!
//! [`audit`] checks a pack against the schema derived from its own config: every tensor name,
//! dtype and shape, the trellis extent, the signed-I32 mul1 marker, shard extents against file
//! sizes, the 16-bit non-quantized roles, and the scale vectors. It reads only headers and the
//! small scale, marker, norm, sink and router ranges; trellis payloads are read only for a
//! caller-bounded sample admitted through `Matrix::load`.

mod audit;
mod factor;
mod scan;
mod schema;

#[cfg(test)]
mod audit_tests;
#[cfg(test)]
mod factor_tests;
#[cfg(test)]
mod fixture;
#[cfg(test)]
mod real_pack;
#[cfg(test)]
mod scale_tests;

#[cfg(test)]
pub(crate) use audit::Audit;
#[cfg(test)]
use audit::audit_published;
pub(crate) use audit::{Limits, audit};
pub(crate) use factor::RoutedScale;
#[cfg(test)]
pub(crate) use fixture::PackBuilder;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) use schema::Plain;
pub(crate) use schema::{Geometry, Kind, Plan};
