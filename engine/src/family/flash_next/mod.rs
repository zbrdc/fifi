//! Host-orchestrated forward pass for `qwen3_8_flash_next`.
//!
//! Activations stay in host `f32` vectors. Quantized projections go through the
//! [`crate::family::project::Projector`] seam. Decision 0035 and `docs/design/flash-next-executor.md`
//! define the blocks, their order, and the parity gate.

mod artifact;
mod deltanet;
pub(crate) mod factory;
mod hyper;
mod load;
pub(in crate::family) mod math;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod mlx;
mod moe;
mod ple;
mod prefill_plan;
mod qsa;
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
mod resident;
pub(crate) mod settings;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod start;
mod step;
pub(in crate::family) mod stored;
