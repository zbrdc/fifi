//! Unit tests of the executor rules. CPU only.
use super::{
    dims::{KvMode, StepOptions},
    rules::{QwenRules, check_token_id, check_vocab, validate_step_inputs},
};
use crate::EngError;
use crate::family::project::cuda::WeightLayout;

/// The real token counts of Qwen3.6: the sampler draws from 248_077 ids, the head tensor is
/// padded to 248_320 rows, and the tokenizer's end-of-sequence ids are 248_046 and 248_044.
const SAMPLER_VOCAB: usize = 248_077;
const HEAD_ROWS: usize = 248_320;

fn options(forked_experts: bool) -> StepOptions {
    StepOptions {
        kv_mode: KvMode::Fp16,
        context_tokens: 1024,
        delta_slots: 1,
        forked_experts,
        tiered: None,
        layout: WeightLayout::Tiled,
    }
}

#[test]
fn check_vocab_refuses_a_head_narrower_than_the_sampler_vocabulary() {
    assert!(check_vocab(HEAD_ROWS, SAMPLER_VOCAB).is_ok());
    assert!(check_vocab(SAMPLER_VOCAB, SAMPLER_VOCAB).is_ok());
    assert!(check_vocab(SAMPLER_VOCAB - 1, SAMPLER_VOCAB).is_err());
}

#[test]
fn the_sampler_vocabulary_must_fit_the_head() {
    assert!(QwenRules::new(vec![1], 248_044, 248_320).is_ok());
    assert!(QwenRules::new(vec![1], 248_320, 248_320).is_ok());
    assert!(QwenRules::new(vec![1], 0, 248_320).is_err());
    assert!(QwenRules::new(vec![1], 248_321, 248_320).is_err());
}

#[test]
fn an_empty_eos_set_is_refused() {
    assert!(QwenRules::new(Vec::new(), SAMPLER_VOCAB, HEAD_ROWS).is_err());
}

#[test]
fn an_eos_id_outside_the_sampler_vocabulary_is_refused() {
    // Id SAMPLER_VOCAB is the first padding id: outside the sampler vocabulary.
    assert!(QwenRules::new(vec![SAMPLER_VOCAB as u32], SAMPLER_VOCAB, HEAD_ROWS).is_err());
    assert!(QwenRules::new(vec![(SAMPLER_VOCAB - 1) as u32], SAMPLER_VOCAB, HEAD_ROWS).is_ok());
}

#[test]
fn duplicate_eos_ids_are_refused() {
    assert!(QwenRules::new(vec![248_046, 248_046], SAMPLER_VOCAB, HEAD_ROWS).is_err());
    assert!(QwenRules::new(vec![248_046, 248_044], SAMPLER_VOCAB, HEAD_ROWS).is_ok());
}

#[test]
fn the_eos_ids_are_kept_in_order_and_cloning_shares_them() {
    let rules = QwenRules::new(vec![248_046, 248_044], SAMPLER_VOCAB, HEAD_ROWS).expect("rules");
    assert_eq!(&*rules.eos, [248_046, 248_044]);
    assert!(std::sync::Arc::ptr_eq(&rules.eos, &rules.clone().eos));
}

#[test]
fn validate_step_inputs_refuses_forked_experts() {
    let rules = QwenRules::new(vec![248_046], SAMPLER_VOCAB, HEAD_ROWS).expect("rules");
    match validate_step_inputs(&options(true), HEAD_ROWS, &rules) {
        Err(EngError::BadRequest(message)) => {
            assert!(message.contains("forked_experts"), "{message}");
        }
        other => panic!("forked experts must be refused before the step is built: {other:?}"),
    }
    assert!(validate_step_inputs(&options(false), HEAD_ROWS, &rules).is_ok());
}

#[test]
fn validate_step_inputs_refuses_a_head_narrower_than_the_sampler_vocabulary() {
    // A head of 248_320 padded rows cannot serve a sampler vocabulary of 248_400 ids. The
    // bound of QwenRules::new is satisfied (248_400 fits its head of 248_400); only the real
    // head width of the step is narrower, which check_vocab must refuse.
    let rules = QwenRules::new(vec![248_046], 248_400, 248_400).expect("rules");
    match validate_step_inputs(&options(false), HEAD_ROWS, &rules) {
        Err(EngError::Checkpoint(message)) => {
            assert!(message.contains("model head"), "{message}");
            assert!(message.contains("248320"), "{message}");
            assert!(message.contains("248400"), "{message}");
        }
        other => panic!("a narrow head must be refused before the step is built: {other:?}"),
    }
}

#[test]
fn validate_step_inputs_names_the_sampler_vocabulary_at_both_sites() {
    let rules = QwenRules::new(vec![248_046], SAMPLER_VOCAB, HEAD_ROWS).expect("rules");
    let widths = validate_step_inputs(&options(false), HEAD_ROWS, &rules).expect("widths");
    assert_eq!(widths.argmax_n, SAMPLER_VOCAB);
    assert_eq!(widths.verdict, SAMPLER_VOCAB);
    assert_ne!(widths.argmax_n, HEAD_ROWS);
    assert_ne!(widths.verdict, HEAD_ROWS);
    // An unpadded head: both widths are the head itself.
    let square = QwenRules::new(vec![248_046], SAMPLER_VOCAB, SAMPLER_VOCAB).expect("rules");
    let widths = validate_step_inputs(&options(false), SAMPLER_VOCAB, &square).expect("widths");
    assert_eq!(widths.argmax_n, SAMPLER_VOCAB);
    assert_eq!(widths.verdict, SAMPLER_VOCAB);
}

#[test]
fn check_token_id_accepts_every_id_inside_the_sampler_vocabulary() {
    assert!(check_token_id(0, SAMPLER_VOCAB).is_ok());
    // 248_076: the last id the sampler can draw.
    assert!(check_token_id(SAMPLER_VOCAB as u32 - 1, SAMPLER_VOCAB).is_ok());
}

#[test]
fn check_token_id_refuses_the_padding_and_head_ids_naming_the_id() {
    // 248_077 is the first padding row INSIDE the padded table (an id that would embed and run),
    // 248_319 the last padded row, 248_320 the first row above the table.
    for id in [
        SAMPLER_VOCAB as u32,
        HEAD_ROWS as u32 - 1,
        HEAD_ROWS as u32,
        u32::MAX,
    ] {
        match check_token_id(id, SAMPLER_VOCAB) {
            Err(EngError::BadRequest(message)) => {
                assert!(
                    message.contains(&format!("token id {id}")),
                    "id {id}: {message}"
                );
                assert!(message.contains("sampler vocabulary"), "id {id}: {message}");
            }
            other => panic!("id {id} must be refused before any device write: {other:?}"),
        }
    }
}
