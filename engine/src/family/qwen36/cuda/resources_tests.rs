//! Unit tests of the step shapes and the name lookups (test seam U13). CPU only. The weight
//! lookups that read `CudaWeights` are covered by the scratch device run.
use super::{
    common_tests::{dims, inventory},
    dims::{KvMode, StepOptions},
    mixer_geometry::DenseSpec,
    resources::{
        GlobalNames, GlobalSpecs, MoeSpecs, STEP_ROWS, StackSpec, check_stack,
        check_stage1_options, f32_bytes, moe_dims,
    },
};
use crate::{EngError, family::gguf_model::Role};

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
