//! Resident decode of the stepping-stone family `Qwen3.6-35B-A3B` on MLX (decision 0046).
//!
//! All weights except the embedding table, including the 120 expert stacks, stay resident on the
//! engine thread. The embedding table stays in the checkpoint, and the token start reads one row
//! per token. The portable parts (`layout`, `tiny`) run on Linux. The graph builders need the Mac GPU.

mod chunk_plan;
mod layout;

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod attn;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod decoder;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod delta;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(super) mod executor;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod moe;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod ops;
#[cfg(all(
    test,
    feature = "native-mlx",
    target_os = "macos",
    target_arch = "aarch64"
))]
mod parity;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod prefill;
#[cfg(all(
    test,
    feature = "native-mlx",
    target_os = "macos",
    target_arch = "aarch64"
))]
mod real;
#[cfg(all(
    test,
    feature = "native-fault-qualification",
    target_os = "macos",
    target_arch = "aarch64"
))]
mod route_parity;
#[cfg(all(
    test,
    feature = "native-mlx",
    target_os = "macos",
    target_arch = "aarch64"
))]
mod scale_knob;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod state;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod step;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod weights;
