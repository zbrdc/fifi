mod config;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod flint_pack;
mod full;
mod header;
mod reference;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
mod source;

pub use config::Config;
pub use header::Matrix;
pub use reference::Codebook;
#[cfg(test)]
pub(crate) use source::CONFIG_LIMIT;
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) use source::Checkpoint;

/// Stored components, including legacy forms that Matrix::load must explicitly refuse.
pub const COMPONENT_SUFFIXES: [&str; 9] = [
    "trellis", "suh", "svh", "bias", "mcg", "mul1", "su", "sv", "scale",
];

// The marker words also serve as the integer codebook multipliers.
const MCG_MARKER: u32 = 0xcbac_1fed;
const MUL1_MARKER: u32 = 0x83dc_d12d;
