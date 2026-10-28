use super::{
    common_tests::dims,
    dims::{KvMode, StepOptions, TieredPlan},
    paging_plan::PagingPlan,
    paging_store::PagedKv,
    state::PagedStatePlan,
    state_layout::StateLayout,
};
use crate::family::project::cuda::WeightLayout;

fn options(capacity: usize) -> StepOptions {
    StepOptions {
        kv_mode: KvMode::Int8G32,
        context_tokens: capacity,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Native,
    }
}

#[test]
fn paged_state_admission_counts_slots_instead_of_logical_kv() {
    let d = dims();
    let o = options(262_144);
    let logical = StateLayout::new(&d, &o).unwrap();
    let pages = PagingPlan::new(&d, o.context_tokens).unwrap();
    let planned = PagedStatePlan::new(&d, &o, usize::MAX).unwrap();
    assert_eq!(
        planned.device_bytes().unwrap(),
        logical.delta.total + logical.words.total + pages.device_bytes,
    );
    assert!(planned.device_bytes().unwrap() < logical.bytes().unwrap());
}

#[test]
fn host_admission_includes_pins_and_metadata_before_allocation() {
    let d = dims();
    let o = options(320);
    let pages = PagingPlan::new(&d, o.context_tokens).unwrap();
    let minimum = PagedKv::minimum_host_bytes(&pages).unwrap();
    assert!(minimum > pages.backing_bytes + pages.pinned_bytes);
    assert!(PagedStatePlan::new(&d, &o, minimum - 1).is_err());
    assert!(PagedStatePlan::new(&d, &o, minimum).is_ok());
    assert!(PagedStatePlan::new(&d, &o, usize::MAX).is_ok());
}

#[test]
fn paged_state_plan_refuses_unsupported_modes_without_a_context() {
    let d = dims();
    let valid = options(320);
    for bad in [
        StepOptions {
            kv_mode: KvMode::Fp16,
            ..valid
        },
        StepOptions {
            layout: WeightLayout::Tiled,
            ..valid
        },
        StepOptions {
            tiered: Some(TieredPlan::new(40, 6)),
            ..valid
        },
        StepOptions {
            forked_experts: true,
            ..valid
        },
        StepOptions {
            delta_slots: 0,
            ..valid
        },
    ] {
        assert!(PagedStatePlan::new(&d, &bad, usize::MAX).is_err());
    }
}

#[test]
fn host_metadata_overflow_is_refused_by_preflight() {
    let mut plan = PagingPlan::new(&dims(), 320).unwrap();
    plan.groups = usize::MAX;
    assert!(matches!(
        PagedKv::minimum_host_bytes(&plan),
        Err(crate::EngError::MemoryPlan(_))
    ));
}

#[test]
fn pages_use_existing_q8_planes_and_sixty_four_partials() {
    let dims = dims();
    let plan = PagingPlan::new(&dims, 262_144).expect("Q8 paging plan");
    assert_eq!(plan.group_cells, 4096);
    assert_eq!(plan.groups, 64);
    assert_eq!(plan.group.total, 4_456_448);
    assert_eq!(plan.slot_bytes, 4_456_452);
    assert_eq!(plan.backing_bytes, 2_852_126_720);
    assert_eq!(plan.device_bytes, 8_912_904);
    assert_eq!(plan.pinned_bytes, 8_912_904);
}

#[test]
fn staging_groups_bill_padding_and_the_partial_tail() {
    let dims = dims();
    for (capacity, cells, groups, backing) in [
        (1, 256, 1, 2_785_280),
        (320, 256, 2, 5_570_560),
        (16_384, 256, 64, 178_257_920),
        (16_385, 512, 33, 183_828_480),
    ] {
        let plan = PagingPlan::new(&dims, capacity).expect("padded Q8 geometry");
        assert_eq!((plan.group_cells, plan.groups), (cells, groups));
        assert_eq!(plan.group.capacity, cells);
        assert_eq!(plan.group.layers, 1);
        assert_eq!(plan.group.mode, super::dims::KvMode::Int8G32);
        assert_eq!(plan.backing_bytes, backing);
        assert_eq!(plan.slot_bytes, plan.group.total + 4);
        assert_eq!(plan.device_bytes, 2 * plan.slot_bytes);
        assert_eq!(plan.pinned_bytes, plan.device_bytes);
    }
}

#[test]
fn largest_native_group_is_valid_metadata() {
    let plan = PagingPlan::new(&dims(), 67_108_864).expect("native upper geometry bound");
    assert_eq!(plan.group_cells, 1_048_576);
    assert_eq!(plan.groups, 64);
}

#[test]
fn zero_and_unsupported_contexts_are_refused() {
    let dims = dims();
    for capacity in [0, 67_108_865, usize::MAX] {
        assert!(
            matches!(
                PagingPlan::new(&dims, capacity),
                Err(crate::EngError::BadRequest(_))
            ),
            "capacity {capacity} must be refused"
        );
    }
}

#[test]
fn a_checkpoint_without_full_attention_has_no_paging_plan() {
    let mut dims = dims();
    dims.shape
        .kinds
        .fill(crate::family::AttentionKind::DeltaNet);
    assert!(matches!(
        PagingPlan::new(&dims, 1024),
        Err(crate::EngError::Checkpoint(_))
    ));
}

#[test]
fn backing_multiplication_overflow_is_refused() {
    let mut dims = dims();
    dims.shape.attn.heads_kv = usize::MAX / 1_000_000;
    dims.shape.attn.head_dim = 128;
    // One group's metadata fits; the aggregate backing does not.
    super::state_layout::KvPool::q8_staging_layer(&dims, 256).expect("one group fits");
    assert!(matches!(
        PagingPlan::new(&dims, 16_384),
        Err(crate::EngError::MemoryPlan(_))
    ));
}
