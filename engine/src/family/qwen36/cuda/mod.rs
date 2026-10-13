//! The resident Qwen3.6 CUDA decode step (decision 0047), packet RD-9a: the pure host planning.
//! The state owner (RD-9a2), the mixers, the MoE block and the token loop (RD-9b to RD-9d) are
//! later packets, and chunked prefill (decision 0049) has its own plan beside this one.
//!
//! `dims` holds the checked dimensions and the session options; `step_plan` derives the
//! activation tensors of a decode step and their lifetimes; `state_layout` fixes the byte offsets
//! and the element order of the persistent state and words (read its module doc for the layout,
//! its transposes against llama.cpp and the export contract). No model number is written in this
//! module: everything comes from the layout record (`HostShape`), the inventory roles and the
//! kernel wrappers' own size functions. Design guidance only, no code copied.

mod book;
#[cfg(test)]
mod book_tests;
#[cfg(test)]
mod common_tests;
mod dims;
mod executor;
#[cfg(test)]
mod export_tests;
pub mod gate;
#[cfg(test)]
mod geometry_tests;
mod layer_table;
#[cfg(test)]
mod layer_tests;
mod mixer;
mod mixer_geometry;
mod mixer_ops;
mod mixer_shapes;
#[cfg(test)]
mod mixer_tests;
mod mixer_weights;
mod moe;
#[cfg(test)]
mod native_device_rig;
#[cfg(test)]
mod native_guard_tests;
#[cfg(test)]
mod native_mixer_device_tests;
#[cfg(test)]
mod native_moe_device_tests;
#[cfg(test)]
mod native_ops_tests;
#[cfg(test)]
mod native_plan_tests;
#[cfg(test)]
mod ops_tests;
mod pinned;
#[cfg(test)]
mod r5_fmt;
mod readback;
#[cfg(test)]
mod readback_tests;
mod resources;
#[cfg(test)]
mod resources_tests;
#[cfg(test)]
mod router_tests;
mod rules;
#[cfg(test)]
mod rules_tests;
mod run_report;
#[cfg(test)]
mod run_report_tests;
#[cfg(test)]
mod shape_tests;
pub(crate) mod start;
#[cfg(test)]
mod start_health_tests;
#[cfg(test)]
mod start_pins;
#[cfg(test)]
mod start_tests;
mod state;
mod state_layout;
mod step;
mod step_ops;
#[cfg(test)]
mod step_ops_tests;
mod step_plan;
#[cfg(test)]
mod step_tests;
#[cfg(test)]
mod tests;
mod verdict;
#[cfg(test)]
mod verdict_tests;
