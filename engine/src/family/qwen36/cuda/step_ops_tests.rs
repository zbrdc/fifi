//! Unit tests of the launch order of one token (test seam U13). CPU only.
use super::{
    common_tests::dims,
    step_ops::{POS_WORD_READERS, StepOp, final_resid, resid_in, resid_out, token_ops},
    step_plan::{PHASES, Phase, Slot},
};
use crate::family::AttentionKind;

fn kind_of(layer: usize) -> AttentionKind {
    dims().shape.kinds[layer]
}

fn kind_at(op: StepOp) -> AttentionKind {
    op.layer().map_or(AttentionKind::DeltaNet, kind_of)
}

#[test]
fn a_token_is_begin_embed_layers_norm_head_argmax_advance() {
    let layers = dims().shape.kinds.len();
    let ops = token_ops(layers);
    assert_eq!(ops.len(), 2 + 2 * layers + 4);
    assert_eq!(&ops[..2], &[StepOp::Begin, StepOp::Embed]);
    assert_eq!(
        &ops[ops.len() - 4..],
        &[
            StepOp::FinalNorm,
            StepOp::LmHead,
            StepOp::Argmax,
            StepOp::Advance
        ]
    );
    for layer in 0..layers {
        assert_eq!(ops[2 + 2 * layer], StepOp::Mixer(layer));
        assert_eq!(ops[3 + 2 * layer], StepOp::Moe(layer));
    }
}

#[test]
fn argmax_is_the_last_status_writer_and_advance_is_last() {
    let layers = dims().shape.kinds.len();
    let ops = token_ops(layers);
    let writers: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|&(_, &op)| op.writes_status(kind_at(op)))
        .map(|(at, _)| at)
        .collect();
    let argmax_at = ops
        .iter()
        .position(|&op| op == StepOp::Argmax)
        .expect("argmax");
    assert_eq!(writers.last(), Some(&argmax_at));
    assert_eq!(writers.first(), Some(&0), "step_begin clears status first");
    assert_eq!(ops.last(), Some(&StepOp::Advance));
    assert!(!StepOp::Advance.writes_status(AttentionKind::DeltaNet));
}

#[test]
fn begin_embed_attention_mixers_moe_blocks_and_argmax_write_status() {
    let d = dims();
    for (layer, &kind) in d.shape.kinds.iter().enumerate() {
        let attention = kind == AttentionKind::Full;
        assert_eq!(
            StepOp::Mixer(layer).writes_status(kind),
            attention,
            "layer {layer}"
        );
        assert!(
            StepOp::Moe(layer).writes_status(kind),
            "router of layer {layer}"
        );
    }
    // RD-4c ORs mask 4 into status for a token id outside the vocabulary.
    for op in [StepOp::Begin, StepOp::Embed, StepOp::Argmax] {
        assert!(op.writes_status(AttentionKind::Full), "{op:?}");
    }
    for op in [StepOp::FinalNorm, StepOp::LmHead, StepOp::Advance] {
        assert!(!op.writes_status(AttentionKind::Full), "{op:?}");
    }
}

#[test]
fn the_phases_of_every_op_ascend_and_follow_the_previous_op() {
    let layers = dims().shape.kinds.len();
    let mut last = None::<usize>;
    for op in token_ops(layers) {
        let phases: Vec<usize> = op
            .phases(kind_at(op))
            .into_iter()
            .map(|p| p as usize)
            .collect();
        assert!(
            phases.windows(2).all(|w| w[0] < w[1]),
            "{op:?} not ascending"
        );
        // Inside a layer the phases continue; a new layer starts again at the mixer prologue.
        let starts_layer = matches!(op, StepOp::Mixer(_));
        if let (Some(prev), false) = (last, starts_layer) {
            assert!(
                phases[0] > prev,
                "{op:?} starts at {} after {prev}",
                phases[0]
            );
        }
        if starts_layer {
            assert_eq!(phases[0], Phase::MixPack as usize);
        }
        last = phases.last().copied();
    }
}

#[test]
fn every_phase_of_the_plan_table_is_run_by_some_op() {
    let d = dims();
    let mut seen = Vec::new();
    for op in token_ops(d.shape.kinds.len()) {
        seen.extend(op.phases(kind_at(op)).into_iter().map(|p| p as usize));
    }
    for phase in PHASES {
        assert!(seen.contains(&(phase as usize)), "{phase:?} is never run");
    }
}

#[test]
fn the_residual_alternates_from_the_embedding_to_the_final_norm() {
    assert_eq!(resid_in(0), Slot::ResidA, "the embedding writes ResidA");
    for layer in 0..8 {
        assert_ne!(resid_in(layer), resid_out(layer));
        assert_eq!(resid_out(layer), resid_in(layer + 1));
    }
    assert_eq!(final_resid(0), Slot::ResidA);
    assert_eq!(final_resid(1), Slot::ResidB);
    let layers = dims().shape.kinds.len();
    let expected = if layers.is_multiple_of(2) {
        Slot::ResidA
    } else {
        Slot::ResidB
    };
    assert_eq!(final_resid(layers), expected);
}

fn test_source(file: &str) -> &'static str {
    match file {
        "step_params/tests.rs" => {
            include_str!("../../../kernels/cuda/resident/step_params/tests.rs")
        }
        "attn_prep/tests.rs" => include_str!("../../../kernels/cuda/resident/attn_prep/tests.rs"),
        "kv8_append/tests.rs" => {
            include_str!("../../../kernels/cuda/resident/kv8_append/tests.rs")
        }
        "attn_decode/tests.rs" => {
            include_str!("../../../kernels/cuda/resident/attn_decode/tests.rs")
        }
        "attn_decode_q8/tests.rs" => {
            include_str!("../../../kernels/cuda/resident/attn_decode_q8/tests.rs")
        }
        other => panic!("no test source is wired for {other}"),
    }
}

#[test]
fn every_reader_of_a_position_word_cites_a_merged_poisoned_word_test() {
    let kernels: Vec<&str> = POS_WORD_READERS.iter().map(|r| r.kernel).collect();
    assert_eq!(
        kernels,
        [
            "step_begin",
            "attn_prep",
            "attn_prep_rows",
            "kv8_append",
            "attn_decode",
            "attn_decode_q8",
            "step_advance"
        ],
        "a new reader of pos, kv_len or kv_slot needs a poisoned-word test first"
    );
    for reader in POS_WORD_READERS {
        let source = test_source(reader.test_file);
        let needle = format!("fn {}(", reader.test);
        assert!(
            source.contains(&needle),
            "{} cites {}, which is not in {}",
            reader.kernel,
            reader.test,
            reader.test_file
        );
    }
}

#[test]
fn the_body_op_list_is_pinned() {
    use StepOp::{Advance, Argmax, Begin, Embed, FinalNorm, LmHead, Mixer, Moe};
    assert_eq!(
        token_ops(3),
        [
            Begin,
            Embed,
            Mixer(0),
            Moe(0),
            Mixer(1),
            Moe(1),
            Mixer(2),
            Moe(2),
            FinalNorm,
            LmHead,
            Argmax,
            Advance
        ]
    );
}
