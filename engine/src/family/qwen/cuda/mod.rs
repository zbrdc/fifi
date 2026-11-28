//! The CUDA step of the lineage: one executor over the shared step for every model the family's
//! recipe describes, its device load, and the legs that score and check it on a card.

pub(crate) mod load;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod quality_run;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod reuse_device_tests;
#[cfg(feature = "native-cuda")]
pub(crate) mod serve;
#[cfg(feature = "native-cuda")]
pub(crate) mod step;
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod step_device_tests;
