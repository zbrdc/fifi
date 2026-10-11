#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[expect(dead_code, reason = "staged for RD-9b to RD-9d (decision 0047)")]
pub mod cuda;
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
#[cfg(test)]
mod tiny;
