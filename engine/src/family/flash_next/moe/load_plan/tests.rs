//! The load plan from the real Flash-Next config and a header-walk report with the sizes the walk
//! found on the 3.05 bpw pack (2026-11-04: 74,041 matrices, 51.37 GB of device bytes), and from
//! the sparse pack's own headers when `FIFI_FLASH_NEXT_EXL3_PACK` names it.

use super::*;
use crate::family::{
    exl3_pack::{Marker, PackReport},
    flash_next::load::exl3_pack::inspect_flash_pack,
};
use std::collections::BTreeMap;

const CONFIG: &[u8] =
    include_bytes!("../../../../../tests/fixtures/qwen3_8_flash_next_config.json");
const GIB_U: u64 = 1 << 30;
const EXPERT_BYTES: u64 = 1_881_600;

fn report(trunk: u64) -> PackReport {
    let experts = 24_576 * EXPERT_BYTES;
    let embedding = 248_320u64 * 2560 * 2;
    PackReport {
        device_bytes: experts + embedding + trunk,
        expert_bytes: experts,
        matrices: 74_041,
        rates: BTreeMap::new(),
        codebook: Marker::Mul1,
    }
}

fn card() -> TierInputs {
    let shape = TierShape::from_pack(48, 512, 10, 24_576 * EXPERT_BYTES).unwrap();
    TierInputs::cost_model_3090(
        shape,
        (0, 0),
        KvPlan {
            positions: 1,
            bytes_per_position: 1,
        },
    )
}

#[test]
fn the_plan_reads_every_size_from_the_pack_and_the_config() {
    let plan = FlashLoadPlan::new(&report(3_854_000_000), CONFIG, 32_640_183_408, card()).unwrap();
    assert_eq!(
        (plan.shape.layers, plan.shape.experts, plan.shape.top_k),
        (48, 512, 10)
    );
    assert_eq!(plan.shape.expert_bytes, EXPERT_BYTES);
    assert_eq!(plan.trunk_bytes, 3_854_000_000);
    assert_eq!(plan.embedding_bytes, 1_271_398_400);
    // 48 value heads x 128 x 128 recurrent values and a 3 x 10,240 conv ring, f32, in 36 DeltaNet layers.
    assert_eq!(
        plan.delta_state_bytes,
        36 * (48 * 128 * 128 + 3 * 10_240) * 4
    );
    let geometry = plan.budget().geometry();
    assert_eq!(
        (
            geometry.cached_layers,
            geometry.heads_kv,
            geometry.head_dim,
            geometry.idx_dim,
            geometry.ratio
        ),
        (12, 2, 256, 128, 4)
    );
}

#[test]
fn a_served_context_settles_slots_and_names_every_part_in_the_load_line() {
    let plan = FlashLoadPlan::new(&report(3_854_000_000), CONFIG, 32_640_183_408, card()).unwrap();
    let long = plan.at_context(262_144, KvFormat::F16, 100).unwrap();
    let mid = plan.at_context(131_072, KvFormat::F16, 100).unwrap();
    let (long_slots, mid_slots) = (
        long.context.plan.slots_per_layer(),
        mid.context.plan.slots_per_layer(),
    );
    // The real delta state is 117.7 MB (112.2 MiB), a little over the 108 MiB (36 x 3 MiB) stand-in of
    // the tier plan's own tests (4.4 MB, far below the 90 MB of a slot).
    assert!((140..=146).contains(&long_slots), "{long_slots}");
    assert!((178..=184).contains(&mid_slots), "{mid_slots}");
    assert_eq!(long.kv_bytes, 262_144 * 26_112);
    let line = long.to_string();
    for part in [
        "context 262144 positions",
        "K/V F16 6.38 GiB",
        "trunk 3.59 GiB",
        &format!("{long_slots} of 512 experts per layer"),
        "pinned in host RAM",
        "embedding 1.18 GiB host",
        "n-gram table 30.40 GiB local NVMe",
    ] {
        assert!(line.contains(part), "{part:?} missing from {line}");
    }
    // VRAM parts and the left-over account for the card, as in the tier plan's own check.
    let vram = long.trunk_bytes
        + plan.delta_state_bytes
        + long.kv_bytes
        + long.context.plan.bank_bytes()
        + long.context.plan.vram_left();
    assert!(vram < 24 * GIB_U, "{vram}");
    assert!(
        24 * GIB_U - vram < 2 * GIB_U,
        "reserves are about 2 GiB: {vram}"
    );
}

#[test]
fn a_context_that_keeps_too_few_slots_is_refused_with_the_longest_that_works() {
    let plan = FlashLoadPlan::new(&report(3_854_000_000), CONFIG, 0, card()).unwrap();
    let text = plan
        .at_context(262_144, KvFormat::F16, 160)
        .unwrap_err()
        .describe();
    assert!(
        text.contains("the longest context with 160 slots is"),
        "{text}"
    );
    let text = plan
        .at_context(131_072, KvFormat::Int8, 100)
        .unwrap_err()
        .describe();
    assert!(text.contains("no attention kernel yet"), "{text}");
}

#[test]
fn a_report_smaller_than_its_experts_and_embedding_is_refused() {
    let mut small = report(0);
    small.device_bytes = small.expert_bytes;
    let text = FlashLoadPlan::new(&small, CONFIG, 0, card())
        .unwrap_err()
        .describe();
    assert!(text.contains("exceed the pack's device bytes"), "{text}");
}

#[test]
#[ignore = "walks the headers of the pack in FIFI_FLASH_NEXT_EXL3_PACK (no payload)"]
fn the_real_pack_headers_give_the_documented_plan() {
    let (checkpoint, _) = super::super::real_layer_tests::open();
    let report = inspect_flash_pack(&checkpoint.source, &checkpoint.model_config).unwrap();
    let plan =
        FlashLoadPlan::new(&report, &checkpoint.model_config, 32_640_183_408, card()).unwrap();
    assert_eq!(plan.trunk_bytes, 3_872_376_320);
    for (context, lo, hi) in [(32_768, 204, 212), (131_072, 178, 184), (262_144, 140, 146)] {
        let line = plan.at_context(context, KvFormat::F16, 100).unwrap();
        let slots = line.context.plan.slots_per_layer();
        eprintln!("{line}");
        assert!((lo..=hi).contains(&slots), "{context}: {slots}");
    }
}
