//! The CUDA path of SenseNova-Loop: host geometry over the shared device-planning
//! shape, the GGUF weight adapter, and the trunk executor.
//!
//! The shared qwen36 CUDA serve path is wired to one model spec; this module rides the
//! shared kernels and planning pieces the way Flash-Next does, owning its step loop.

pub(crate) mod geometry;
pub(crate) mod load;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod quality_run;
#[cfg(feature = "native-cuda")]
pub(crate) mod serve;
#[cfg(feature = "native-cuda")]
pub(crate) mod step;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod step_device_tests;
