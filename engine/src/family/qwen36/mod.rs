#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[expect(
    dead_code,
    reason = "staged for the resident step's mixers, MoE block and token loop"
)]
pub mod cuda;
#[cfg(test)]
mod exl3_admission;
pub(crate) mod factory;
#[cfg(test)]
mod fixtures;
mod host;
pub(in crate::family) mod host_cpu;
mod host_load;
#[cfg(test)]
mod host_real;
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
mod resident;
pub(crate) mod settings;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod start;
pub mod tiered;
#[cfg(test)]
mod tiny;
mod trace;
