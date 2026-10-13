//! Unit test of the router scratch size (test seam U13). CPU only.
use super::{
    common_tests::dims,
    step_plan::{KernelScratch, Slot, StepPlan},
};

#[test]
fn the_router_scratch_comes_from_the_router_wrapper() {
    let d = dims();
    let options = super::dims::StepOptions {
        kv_mode: super::dims::KvMode::Fp16,
        context_tokens: 8,
        delta_slots: 1,
        forked_experts: false,
        layout: crate::family::project::cuda::WeightLayout::Tiled,
    };
    for (rows, bytes) in [(1, 1_028), (8, 8_224)] {
        let scratch = KernelScratch::for_dims(&d, rows).expect("scratch");
        let plan = StepPlan::new(&d, rows, &options, &scratch).expect("plan");
        let at = plan.index(Slot::RouterLogits).expect("router logits");
        assert_eq!(plan.tensors()[at].bytes, bytes, "{rows} rows");
    }
}
