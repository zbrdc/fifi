//! Unit tests of the step shapes and the name lookups (test seam U13). CPU only. The weight
//! lookups that read `CudaWeights` are covered by the scratch device run.
use super::{
    common_tests::{dims, inventory},
    dims::{KvMode, StepOptions, TieredPlan},
    mixer_geometry::DenseSpec,
    resources::{
        GlobalNames, GlobalSpecs, MoeSpecs, PagedResourcesPlan, STEP_ROWS, StackSpec,
        check_native_stack, check_paged_modes, check_stack, check_stage1_options, f32_bytes,
        moe_dims,
    },
};
use super::{
    mixer::MixerViews,
    mixer_geometry::MixerGeometry,
    state::PagedStatePlan,
    step_plan::{KernelScratch, StepPlan},
};
use crate::family::project::cuda::WeightLayout;
use crate::kernels::cuda::resident::attn_decode_q8::{
    AttnQ8Shape, attn_q8_part_bytes, paged::PagedQ8Scratch,
};
use crate::{EngError, family::gguf_model::Role};

#[test]
fn paged_resource_admission_charges_state_arena_and_one_partial_owner() {
    let d = dims();
    let o = StepOptions {
        kv_mode: KvMode::Int8G32,
        context_tokens: 32_768,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Native,
    };
    let state = PagedStatePlan::new(&d, &o, usize::MAX).unwrap();
    let scratch = KernelScratch::for_dims(&d, 1, o.kv_mode).unwrap();
    let arena = StepPlan::new_paged(&d, &o, &scratch).unwrap();
    let partial = attn_q8_part_bytes(1, d.shape.attn.heads_kv, d.shape.attn.head_dim, 64).unwrap();
    let expected = state.device_bytes().unwrap() + arena.arena_bytes() + partial;
    let planned = PagedResourcesPlan::new(d, &inventory(), &o, usize::MAX).unwrap();
    assert_eq!(planned.device_bytes().unwrap(), expected);
    assert!(PagedResourcesPlan::new(dims(), &inventory(), &o, 1).is_err());
}

#[test]
fn paged_partial_allocation_size_is_checked_without_a_context() {
    let d = dims();
    let shape = AttnQ8Shape {
        m: 1,
        n_head: d.shape.attn.heads_q,
        n_kv_head: d.shape.attn.heads_kv,
        head_dim: d.shape.attn.head_dim,
        cells: 320,
        ld: d.shape.attn.heads_q * 2 * d.shape.attn.head_dim,
    };
    assert_eq!(
        PagedQ8Scratch::allocation_bytes(&shape, 256).unwrap(),
        attn_q8_part_bytes(1, shape.n_kv_head, shape.head_dim, 64).unwrap(),
    );
    assert!(PagedQ8Scratch::allocation_bytes(&AttnQ8Shape { m: 2, ..shape }, 256).is_err());
    assert!(PagedQ8Scratch::allocation_bytes(&shape, 1).is_err());
}

#[test]
fn paged_modes_refuse_graphs_and_chunks_without_affecting_resident_modes() {
    for (paged, capture, chunked, allowed) in [
        (false, false, false, true),
        (false, false, true, true),
        (false, true, false, true),
        (false, true, true, true),
        (true, false, false, true),
        (true, false, true, false),
        (true, true, false, false),
        (true, true, true, false),
    ] {
        assert_eq!(check_paged_modes(paged, capture, chunked).is_ok(), allowed);
    }
}

#[test]
fn paged_mixer_views_use_owned_partials_instead_of_an_arena_slot() {
    let d = dims();
    let geometry = MixerGeometry::new(&d).unwrap();
    let o = StepOptions {
        kv_mode: KvMode::Int8G32,
        context_tokens: 320,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Native,
    };
    let scratch = KernelScratch::for_dims(&d, 1, o.kv_mode).unwrap();
    let resident = StepPlan::new(&d, 1, &o, &scratch).unwrap();
    let paged = StepPlan::new_paged(&d, &o, &scratch).unwrap();
    for plan in [&resident, &paged] {
        let indices = MixerViews::slot_indices(plan, &geometry, o.layout, o.kv_mode).unwrap();
        assert_eq!(
            indices
                .iter()
                .any(|(slot, _)| *slot == super::step_plan::Slot::AttnPart),
            plan.index(super::step_plan::Slot::AttnPart).is_some(),
        );
        for (slot, index) in indices {
            assert_eq!(plan.tensors()[index].slot, slot);
        }
    }
}

#[test]
fn moe_specs_follow_the_layout_record() {
    let d = dims();
    let m = &d.shape.moe;
    let specs = MoeSpecs::new(&d).expect("specs");
    assert_eq!(specs.router_values, m.experts * d.shape.hidden);
    assert_eq!(specs.shexp_values, d.shape.hidden);
    let gate = StackSpec {
        experts: m.experts,
        rows: m.inter,
        k: d.shape.hidden,
    };
    assert_eq!((specs.gate, specs.up), (gate, gate));
    let down = StackSpec {
        experts: m.experts,
        rows: d.shape.hidden,
        k: m.inter,
    };
    assert_eq!(specs.down, down);
    assert_eq!(specs.shared_gate.rows, m.shared_inter);
    assert_eq!(specs.shared_up.k, d.shape.hidden);
    assert_eq!(
        specs.shared_down,
        DenseSpec {
            role: Role::FfnDownShexp,
            rows: d.shape.hidden,
            k: m.shared_inter
        }
    );
}

#[test]
fn the_moe_block_gets_the_dimensions_of_the_layout_record() {
    let d = dims();
    let m = &d.shape.moe;
    let got = moe_dims(&d);
    assert_eq!(
        (
            got.rows,
            got.hidden,
            got.n_expert,
            got.top_k,
            got.inter,
            got.shared_inter
        ),
        (
            STEP_ROWS,
            d.shape.hidden,
            m.experts,
            m.top_k,
            m.inter,
            m.shared_inter
        )
    );
}

/// The NATIVE stack guard (MATVEC-PAR S3b, MSG-70 condition 5): the shape rule passes first,
/// then the per-expert `rows % 4` rule refuses a stack whose per-expert height is ragged (every
/// earlier check satisfied), `experts: None` refuses as "not an expert stack", and the valid
/// multiples-of-4 form passes.
#[test]
fn a_native_stack_with_ragged_per_expert_rows_is_refused() {
    let want = StackSpec {
        experts: 4,
        rows: 8,
        k: 256,
    };
    // The valid native form: 4 experts of 8 rows (8 % 4 == 0).
    assert!(check_native_stack("w", (32, 256, Some(4)), &want).is_ok());
    // Ragged per-expert rows: a shape check_stack ACCEPTS (4 experts of 8 rows... 34 rows over
    // 4 experts would fail the shape rule first, so use a want of 4x9: 36 rows, 9 % 4 != 0).
    let ragged = StackSpec {
        experts: 4,
        rows: 9,
        k: 256,
    };
    match check_native_stack("w", (36, 256, Some(4)), &ragged) {
        Err(EngError::Checkpoint(message)) => {
            assert!(
                message.contains("multiple of 4") && message.contains("9"),
                "{message}"
            );
        }
        other => panic!("a 9-row-per-expert native stack must be refused: {other:?}"),
    }
    // The guard's first conjunct (rows not a multiple of the depth) is UNREACHABLE by
    // construction: check_stack's shape rule pins have.0 == experts * rows, which always
    // divides - stated so nobody adds a dead case for it. `experts: None` is the same: the
    // shape rule refuses it first (pinned), so the native rule's own None arm is
    // defence-in-depth behind it.
    assert!(check_native_stack("w", (32, 256, None), &want).is_err());
}

#[test]
fn a_stacked_weight_must_match_depth_rows_and_length() {
    let want = StackSpec {
        experts: 4,
        rows: 8,
        k: 256,
    };
    assert!(check_stack("w", (32, 256, Some(4)), &want).is_ok());
    for have in [
        (32, 256, None),
        (32, 256, Some(3)),
        (24, 256, Some(4)),
        (32, 512, Some(4)),
    ] {
        assert!(check_stack("w", have, &want).is_err(), "{have:?}");
    }
    let huge = StackSpec {
        experts: usize::MAX,
        rows: 2,
        k: 256,
    };
    assert!(check_stack("w", (0, 256, Some(usize::MAX)), &huge).is_err());
}

#[test]
fn the_embedding_and_the_head_are_one_row_per_token_of_the_vocabulary() {
    let d = dims();
    let specs = GlobalSpecs::new(&d);
    assert_eq!(
        (specs.embed.role, specs.head.role),
        (Role::TokenEmbd, Role::Output)
    );
    for spec in [specs.embed, specs.head] {
        assert_eq!((spec.rows, spec.k), (d.shape.vocab, d.shape.hidden));
    }
    assert_eq!(specs.norm_values, d.shape.hidden);
}

#[test]
fn the_global_names_come_from_the_inventory() {
    let names = GlobalNames::new(&inventory()).expect("names");
    assert_eq!(names.embed, "token_embd.weight");
    assert_eq!(names.head, "output.weight");
    assert_eq!(names.norm, "output_norm.weight");
}

#[test]
fn the_step_runs_one_row() {
    assert_eq!(STEP_ROWS, 1, "the MoE block takes one row (RD-9c)");
}

#[test]
fn f32_bytes_converts_values_and_refuses_an_overflow() {
    assert_eq!(f32_bytes(4).expect("4 values"), 16);
    assert_eq!(f32_bytes(0).expect("0 values"), 0);
    match f32_bytes(usize::MAX) {
        Err(EngError::BadRequest(message)) => {
            assert!(message.contains("do not fit a byte count"), "{message}");
        }
        other => panic!("the byte count of usize::MAX values must be refused: {other:?}"),
    }
}

#[test]
fn forked_experts_are_refused_in_stage_one() {
    let options = |forked_experts| StepOptions {
        kv_mode: KvMode::Fp16,
        context_tokens: 1024,
        delta_slots: 1,
        forked_experts,
        tiered: None,
        layout: WeightLayout::Tiled,
    };
    match check_stage1_options(&options(true)) {
        Err(EngError::BadRequest(message)) => {
            assert!(message.contains("forked_experts"), "{message}");
            assert!(message.contains("stage 1"), "{message}");
            assert!(
                message.contains("per-expert XfDown slots are a stage-2 item"),
                "{message}"
            );
        }
        other => panic!("forked experts must be refused in stage 1: {other:?}"),
    }
    assert!(check_stage1_options(&options(false)).is_ok());
}

#[test]
fn tiered_plan_is_copy_eq_and_validates_its_numbers() {
    let plan = TieredPlan::new(40, 6);
    let copy = plan;
    assert_eq!(plan, copy, "the plan is Copy and Eq");
    assert!(plan.check().is_ok());
    let refused = |bad: TieredPlan, needle: &str| match bad.check() {
        Err(EngError::BadRequest(message)) => {
            assert!(message.contains(needle), "{message}");
        }
        other => panic!("{bad:?} must be refused naming {needle}: {other:?}"),
    };
    refused(
        TieredPlan {
            slots_per_layer: 0,
            ..plan
        },
        "slots_per_layer",
    );
    refused(TieredPlan { threads: 0, ..plan }, "threads 0");
    refused(
        TieredPlan {
            threads: TieredPlan::MAX_THREADS + 1,
            ..plan
        },
        "threads 9",
    );
    refused(
        TieredPlan {
            timeout_ms: 0,
            ..plan
        },
        "timeout_ms",
    );
    refused(
        TieredPlan {
            swaps_milli: 8_001,
            ..plan
        },
        "swaps_milli",
    );
    refused(
        TieredPlan {
            decay_milli: 0,
            ..plan
        },
        "decay_milli",
    );
    refused(
        TieredPlan {
            decay_milli: 1_001,
            ..plan
        },
        "decay_milli",
    );
    assert!(
        TieredPlan {
            threads: TieredPlan::MAX_THREADS,
            decay_milli: 1_000,
            ..plan
        }
        .check()
        .is_ok()
    );
}

#[test]
fn tiered_requires_the_native_layout_and_a_valid_plan_at_the_option_check() {
    let options = |layout, tiered| StepOptions {
        kv_mode: KvMode::Fp16,
        context_tokens: 1024,
        delta_slots: 1,
        forked_experts: false,
        tiered,
        layout,
    };
    let plan = Some(TieredPlan::new(40, 6));
    assert!(check_stage1_options(&options(WeightLayout::Native, plan)).is_ok());
    assert!(check_stage1_options(&options(WeightLayout::Native, None)).is_ok());
    match check_stage1_options(&options(WeightLayout::Tiled, plan)) {
        Err(EngError::Checkpoint(message)) => {
            assert!(message.contains("native layout"), "{message}")
        }
        other => panic!("tiered on the tiled layout must be refused: {other:?}"),
    }
    let broken = Some(TieredPlan::new(40, 0));
    assert!(matches!(
        check_stage1_options(&options(WeightLayout::Native, broken)),
        Err(EngError::BadRequest(_))
    ));
}
