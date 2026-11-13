//! The routed MoE block of a prefill chunk: the block itself is family-neutral and lives in
//! [`crate::family::exl3::prefill_moe`]; this module re-exports it for the chunk driver's stage
//! function and keeps the device tests, which build their context from this family's weights.

#[cfg(test)]
pub(super) use crate::family::exl3::prefill_moe::routed_with;
pub(super) use crate::family::exl3::prefill_moe::{Bufs, run, run_banked};

#[cfg(test)]
#[path = "prefill_exl3_moe_device_tests.rs"]
mod device_tests;
