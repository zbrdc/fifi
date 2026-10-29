//! Shared fixtures of the resident-step unit tests: the real Qwen3.6 dimensions, the inventory and
//! a one-row plan. CPU only.
use super::{
    super::host::HostShape,
    dims::{KvMode, StepDims, StepOptions},
    step_plan::{KernelScratch, StepPlan},
};
use crate::family::project::cuda::WeightLayout;
use crate::family::{QWEN3_6_35B_A3B, gguf_model::Inventory};

pub(super) fn inventory() -> Inventory {
    Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory")
}

pub(super) fn dims() -> StepDims {
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, 64).expect("shape");
    StepDims::new(&shape, &inventory()).expect("dims")
}

pub(super) fn plan(d: &StepDims) -> StepPlan {
    plan_of(d, WeightLayout::Tiled)
}

/// The one-row plan of layout `layout` (the NATIVE plan beside the Tiled one).
pub(super) fn plan_of(d: &StepDims, layout: WeightLayout) -> StepPlan {
    let options = StepOptions {
        kv_mode: KvMode::Fp16,
        context_tokens: 1024,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout,
    };
    let scratch = KernelScratch::for_dims(d, 1, KvMode::Fp16).expect("scratch");
    StepPlan::new(d, 1, &options, &scratch).expect("plan")
}
