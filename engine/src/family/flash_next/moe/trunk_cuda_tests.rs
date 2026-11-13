//! The whole Flash-Next trunk through `trunk_weights` on the real 3.05 bpw pack: every one of the
//! 1062 tensors uploads, each under its pack name, and the device holds exactly the plan's trunk
//! bytes (3,872,376,320). Per-tensor read-back checksums are in `cuda/load_tests.rs` (layers 0, 1
//! and 3 and the root); this test closes the sum over all layers.
//!
//! `FIFI_FLASH_NEXT_EXL3_PACK` names the full local pack. Needs a GPU and 4.5 GiB; the test lives
//! here because `FlashLoadPlan::new` is private to `moe`.

use super::{
    load_plan::FlashLoadPlan,
    tier_plan::{TierInputs, TierShape},
};
use crate::{
    family::{
        flash_next::{
            cuda::load::{TrunkList, upload::trunk_weights},
            load::{exl3_pack::inspect_flash_pack, model_shape},
        },
        project::cuda::{CudaProjector, Reserve},
        qwen36::{cuda::exl3_source::PackReader, tiered::budget::KvPlan},
    },
    kernels::cuda::device_present,
    weights::exl3::Checkpoint,
};

#[test]
#[ignore = "needs a GPU (4.5 GiB) and the full pack in FIFI_FLASH_NEXT_EXL3_PACK"]
fn the_whole_trunk_uploads_and_the_device_holds_the_plans_trunk_bytes() {
    assert!(
        device_present("Flash-Next whole trunk"),
        "a GPU is required"
    );
    let dir = std::env::var_os("FIFI_FLASH_NEXT_EXL3_PACK").expect("FIFI_FLASH_NEXT_EXL3_PACK");
    let checkpoint = Checkpoint::open(std::path::Path::new(&dir)).expect("open the pack");
    let report = inspect_flash_pack(&checkpoint.source, &checkpoint.model_config).expect("walk");
    let (shape, kinds) = model_shape(&checkpoint.model_config).expect("model shape");
    let tier = TierShape::from_pack(
        kinds.len(),
        shape.moe.experts(),
        shape.moe.top_k(),
        report.expert_bytes,
    )
    .expect("tier shape");
    let card = TierInputs::cost_model_3090(
        tier,
        (0, 0),
        KvPlan {
            positions: 1,
            bytes_per_position: 1,
        },
    );
    let plan = FlashLoadPlan::new(&report, &checkpoint.model_config, 0, card).expect("plan");
    assert_eq!(plan.trunk_bytes, 3_872_376_320);

    let list = TrunkList::new(&checkpoint.source, &checkpoint.model_config).expect("list");
    let projector = CudaProjector::new().expect("projector");
    let reader = PackReader::new(&projector, &checkpoint, Reserve::default());
    let weights = trunk_weights(&reader, &list, &plan).expect("upload the trunk");
    assert_eq!(weights.len(), list.all().count());
    assert_eq!(list.all().count(), 1062);
    assert_eq!(weights.device_bytes() as u64, plan.trunk_bytes);
    for tensor in list.all() {
        assert!(
            weights.get(&tensor.name).is_some(),
            "{} not loaded",
            tensor.name
        );
    }
    eprintln!(
        "whole trunk: {} tensors, {} device bytes",
        weights.len(),
        weights.device_bytes()
    );
}
