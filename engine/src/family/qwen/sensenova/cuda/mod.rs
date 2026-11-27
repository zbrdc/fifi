//! The CUDA path of SenseNova-Loop: host geometry over the shared device-planning shape and the
//! card serve wiring. The step it runs is the lineage's ([`crate::family::qwen::cuda`]).

pub(crate) mod geometry;
#[cfg(feature = "native-cuda")]
pub(crate) mod serve;
