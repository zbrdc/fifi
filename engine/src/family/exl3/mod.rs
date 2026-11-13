//! The EXL3 MoE block shared by the families.
//!
//! Promoted from `qwen36::cuda`: the routed expert block, its
//! merged-pass host path, and the weight source that lays experts out as a
//! bank. The old `qwen36::cuda` paths re-export from here.

pub(in crate::family) mod bank;
pub(in crate::family) mod launch;
pub(in crate::family) mod layer_table;
pub(in crate::family) mod moe;
pub(in crate::family) mod moe_source;
pub(in crate::family) mod shapes;
