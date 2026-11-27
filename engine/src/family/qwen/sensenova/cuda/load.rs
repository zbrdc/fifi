//! The device load of SenseNova-Loop: the trunk name template over the GGUF, uploaded
//! whole through the shared loader in Native layout — Q4_K_M stays 5.5 GB of blocks the
//! MMVQ matvec reads as stored, the small F32 tensors (conv, `ssm_a`, `dt.bias`, every
//! norm, the mHC bias) land as raw F32 copies.
//!
//! No [`Inventory`](crate::family::gguf_model::Inventory): the list is what the layers of the
//! family builder name (`family/qwen/recipe.rs`). `load_sliced` refuses a
//! named tensor the file lacks, which is the whole validation; a file tensor the list does
//! not name is simply never uploaded.
//!
//! The embedding stays off the device (the Flash-Next pattern): one 8 KiB F16 row per
//! token is gathered on the host, and the card spends its bytes on `output.weight`
//! (≈0.5 GB Q4_K_M) which every step needs.

#[cfg(feature = "native-cuda")]
use super::super::LoopPlan;

/// Every tensor the device holds, in load order: what the plan's layers name.
#[cfg(feature = "native-cuda")]
pub(crate) fn tensor_names(plan: &LoopPlan) -> Vec<String> {
    use crate::family::qwen::recipe::{Blocks, layers, tensors};
    let blocks = Blocks {
        split_delta_inputs: false,
        experts: false,
    };
    tensors(&layers(&plan.visits(), blocks).0)
}

/// Uploads every tensor of the template in one all-or-nothing Native load. Admission
/// and reserve are the caller's `options`; the layout must be Native (the resident
/// kernels read stored blocks).
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) fn load(
    loader: &crate::family::project::cuda::BorrowedLoader<'_>,
    weights: &crate::weights::GgufWeights,
    plan: &LoopPlan,
    options: crate::family::project::cuda::LoadOptions<'_>,
) -> Result<crate::family::project::cuda::CudaWeights, crate::EngError> {
    let names = tensor_names(plan);
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    loader.load_sliced(weights, &refs, options, &[], 0)
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests {
    use super::*;

    fn pinned_plan() -> LoopPlan {
        let mut recurrent = vec![true; 56];
        for il in [7, 15, 23, 31, 40] {
            recurrent[il] = false;
        }
        LoopPlan {
            loop_start: 4,
            loop_end: 12,
            loop_count: 4,
            mhc_start: 8,
            mhc_end: 24,
            mhc_iterations: 1,
            mhc_pre_scale: 1.0,
            mhc_fixed_identity: false,
            recurrent,
        }
    }

    #[test]
    fn template_covers_every_physical_layer_once() {
        let plan = pinned_plan();
        let names = tensor_names(&plan);
        // No tensor twice (a revisited block's tensors appear under one stem only).
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len());
        // Head and final norm.
        assert!(names.first().is_some_and(|n| n == "output.weight"));
        assert!(names.last().is_some_and(|n| n == "output_norm.weight"));
        // A looped delta layer keeps its own stem and tensors.
        assert!(names.contains(&"blk.5.attn_qkv_gate.weight".to_owned()));
        // The loop's revisits introduce no second stem: physical layers run over blocks
        // {0..3} head, {4..11} loop, {36..55} tail — nothing at or above 56.
        assert!(names.contains(&"blk.36.attn_qkv_gate.weight".to_owned()));
        assert!(!names.contains(&"blk.56.attn_qkv_gate.weight".to_owned()));
        // The attention layer inside the loop has attention tensors, no GDN ones.
        assert!(names.contains(&"blk.7.attn_q.weight".to_owned()));
        assert!(!names.contains(&"blk.7.ssm_out.weight".to_owned()));
        // Tail physical 16 is the tail attention visit 40's weight set: stem blk.40.
        assert!(names.contains(&"blk.40.attn_q.weight".to_owned()));
        // mHC routers on physical layers 8..24, both parts, weight and bias.
        assert!(names.contains(&"blk.8.hc_attn_fn.weight".to_owned()));
        assert!(names.contains(&"blk.47.hc_ffn_fn.bias".to_owned()));
        assert!(!names.contains(&"blk.7.hc_attn_fn.weight".to_owned()));
        assert!(!names.contains(&"blk.48.hc_ffn_fn.weight".to_owned()));
        // 32 layers: 30 delta (7 tensors) + 2 attn (6 tensors) + 5 common each, 16
        // mhc layers x 4 tensors, head + final norm.
        assert_eq!(names.len(), 2 + 32 * 5 + 30 * 7 + 2 * 6 + 16 * 4);
    }
}
