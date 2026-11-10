//! The resident Qwen3.6 CUDA decode step: the pure host planning, the state owner, the mixers, the
//! MoE block and the token loop. Chunked prefill has its own plan beside the decode plan.
//!
//! `dims` holds the checked dimensions and the session options; `step_plan` derives the
//! activation tensors of a decode step and their lifetimes; `state_layout` fixes the byte offsets
//! and the element order of the persistent state and words (read its module doc for the layout,
//! its transposes against llama.cpp and the export contract). No model number is written in this
//! module: everything comes from the layout record (`HostShape`), the inventory roles and the
//! kernel wrappers' own size functions.

#[cfg(all(test, feature = "research-activation"))]
mod activation_device_tests;
pub(in crate::family) mod book;
#[cfg(test)]
mod book_tests;
mod card_attn;
#[cfg(test)]
mod card_attn_device_tests;
mod card_block;
#[cfg(test)]
mod card_block_device_tests;
#[cfg(test)]
mod card_chunk_device_tests;
mod card_load;
#[cfg(test)]
mod card_load_tests;
#[cfg(test)]
mod card_moe_device_tests;
mod card_prefill;
mod card_session;
#[cfg(test)]
mod card_session_device_tests;
mod card_step;
mod card_token;
#[cfg(test)]
mod card_token_device_tests;
mod card_weights;
#[cfg(test)]
mod card_weights_tests;
#[cfg(test)]
mod common_tests;
#[cfg(test)]
mod d1_window_device_tests;
mod dims;
mod eager_window;
#[cfg(test)]
mod eager_window_device_tests;
mod executor;
#[cfg(test)]
mod executor_tests;
mod exl3_bank;
#[cfg(test)]
mod exl3_bank_krow_device_tests;
#[cfg(test)]
mod exl3_bank_step_device_tests;
#[cfg(test)]
mod exl3_bank_timing_probe;
#[cfg(test)]
mod exl3_device_tests;
mod exl3_load;
#[cfg(test)]
mod exl3_load_device_tests;
pub(in crate::family) mod exl3_moe;
#[cfg(test)]
mod exl3_moe_device_tests;
#[cfg(test)]
mod exl3_moe_merge_device_tests;
#[cfg(test)]
mod exl3_moe_rows_device_tests;
mod exl3_moe_source;
pub(in crate::family) mod exl3_source;
#[cfg(test)]
mod exl3_step_device_tests;
#[cfg(test)]
mod exl3_step_merge_device_tests;
#[cfg(test)]
mod export_tests;
pub mod gate;
#[cfg(test)]
mod geometry_tests;
#[cfg(test)]
mod krow_alpha_device_tests;
#[cfg(test)]
mod krow_device_tests;
mod krow_ops;
#[cfg(test)]
mod krow_step_device_tests;
#[cfg(test)]
mod kv8_step_device_tests;
mod layer_table;
#[cfg(test)]
mod layer_tests;
#[cfg(all(test, feature = "native-cuda"))]
#[path = "m3_run.rs"]
mod m3_run;
mod mixer;
mod mixer_geometry;
mod mixer_ops;
mod mixer_shapes;
#[cfg(test)]
mod mixer_tests;
mod mixer_weights;
pub(in crate::family) mod moe;
mod mtp_load;
#[cfg(test)]
mod mtp_load_device_tests;
mod multi;
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
mod native_tiered_device_tests;
#[cfg(test)]
mod ops_tests;
#[cfg(test)]
mod paging_bound_tests;
#[cfg(test)]
mod paging_boundary_tests;
#[cfg(test)]
mod paging_capture_tests;
#[cfg(test)]
mod paging_cow_refusal_tests;
#[cfg(test)]
mod paging_cow_tests;
#[cfg(test)]
mod paging_decode_guard_tests;
#[cfg(test)]
mod paging_decode_tests;
#[cfg(test)]
mod paging_pipeline_tests;
mod paging_plan;
#[cfg(test)]
mod paging_plan_tests;
#[cfg(test)]
mod paging_state_tests;
mod paging_store;
#[cfg(test)]
mod paging_store_tests;
mod pinned;
mod placement;
#[cfg(test)]
mod placement_tests;
mod prefill;
#[cfg(test)]
mod prefill_device_tests;
mod prefill_driver;
#[cfg(test)]
mod prefill_driver_tests;
#[cfg(test)]
mod prefill_exl3_step_device_tests;
mod prefill_plan;
#[cfg(test)]
mod prefill_plan_tests;
#[cfg(test)]
mod prefill_step_device_tests;
#[cfg(test)]
mod prefill_tests;
mod prefix;
#[cfg(test)]
mod prefix_device_tests;
#[cfg(test)]
mod prefix_tests;
#[cfg(test)]
mod r5_fmt;
mod readback;
#[cfg(test)]
mod readback_tests;
mod resources;
#[cfg(test)]
mod resources_tests;
#[cfg(test)]
mod rollback_window_device_tests;
#[cfg(test)]
mod router_tests;
mod rules;
#[cfg(test)]
mod rules_tests;
mod run_report;
#[cfg(test)]
mod run_report_tests;
mod section_timing;
#[cfg(all(test, feature = "native-cuda"))]
#[path = "section_timing_device_tests.rs"]
mod section_timing_device_tests;
#[cfg(test)]
mod section_timing_tests;
#[cfg(test)]
mod shape_tests;
mod shard;
mod sparse_exec;
#[cfg(test)]
mod sparse_step_device_tests;
#[cfg(test)]
mod spec_window_device_tests;
pub(crate) mod start;
#[cfg(test)]
mod start_health_tests;
#[cfg(test)]
mod start_pins;
#[cfg(test)]
mod start_tests;
mod state;
pub(in crate::family) mod state_layout;
mod step;
mod step_ops;
#[cfg(test)]
mod step_ops_tests;
mod step_plan;
#[cfg(test)]
mod step_tests;
#[cfg(test)]
mod tests;
mod tiered_load;
mod tiered_step;
mod trace;
mod verdict;
#[cfg(test)]
mod verdict_tests;
#[cfg(test)]
mod window_bench_device_tests;
