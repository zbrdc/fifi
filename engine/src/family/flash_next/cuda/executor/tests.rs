//! The executor over the fake driver: the worker's call order, the finish rules and the limits.

use std::sync::Arc;

use super::{CudaRules, FlashCudaExecutor};
use crate::{
    EngError,
    family::{
        executor::{ExecutorOps, GenerationState},
        flash_next::cuda::driver::{
            TokenDriver,
            fakes::{FakeDevice, FakeFeeds, POSITIONS, SAMPLER_VOCAB, shape},
        },
        session::{DecodeSession, SaveKind, TokenStepper},
    },
    tokenizer::{DecodeFinish, IncrementalDecoder, Tokenizer},
};

const TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]}}"#;

type Executor = FlashCudaExecutor<TokenDriver<FakeDevice, FakeFeeds>>;

fn build(eos: &[u32]) -> (Executor, DecodeSession) {
    let driver = TokenDriver::new(FakeDevice::default(), FakeFeeds::default(), shape()).unwrap();
    FlashCudaExecutor::new(
        driver,
        CudaRules {
            eos: Arc::from(eos),
            sampler_vocab: SAMPLER_VOCAB,
        },
    )
    .unwrap()
}

fn generation(max_tokens: u32) -> GenerationState {
    let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes()).unwrap();
    GenerationState::new(IncrementalDecoder::new(&tokenizer)).with_max_tokens(max_tokens)
}

#[test]
fn a_job_prefills_all_but_the_last_token_and_decodes_greedily_in_the_vocabulary() {
    let (mut executor, mut session) = build(&[]);
    executor.reset(&mut session).unwrap();
    executor.prefill(&mut session, &[1, 2, 3]).unwrap();
    assert_eq!(
        executor.driver.fed(),
        2,
        "the last prompt token stays pending"
    );
    let mut state = generation(2);
    let mut tokens = Vec::new();
    for _ in 0..2 {
        let batch = executor.decode(&mut session, &mut state).unwrap();
        tokens.extend(batch.tokens);
    }
    // The fake head grows with the id: the draw is the last id inside the sampler vocabulary.
    let last = u32::try_from(SAMPLER_VOCAB - 1).unwrap();
    assert_eq!(tokens, [last, last]);
    assert_eq!(executor.driver.fed(), 4);
    assert_eq!(executor.history(&session), [1, 2, 3, last, last]);
}

#[test]
fn an_eos_draw_finishes_the_job() {
    let last = u32::try_from(SAMPLER_VOCAB - 1).unwrap();
    let (mut executor, mut session) = build(&[last]);
    executor.prefill(&mut session, &[1]).unwrap();
    let batch = executor.decode(&mut session, &mut generation(8)).unwrap();
    assert_eq!(batch.finish, Some(DecodeFinish::Stop));
}

#[test]
fn a_prompt_at_the_context_limit_is_refused_before_any_step() {
    let (executor, _) = build(&[]);
    assert_eq!(executor.context_limit(), POSITIONS);
    assert!(executor.check_prompt(POSITIONS - 1).is_ok());
    assert!(matches!(
        executor.check_prompt(POSITIONS),
        Err(EngError::PromptTooLong { .. })
    ));
    assert_eq!(executor.driver.fed(), 0);
}

#[test]
fn reset_starts_a_new_job_from_position_zero() {
    let (mut executor, mut session) = build(&[]);
    executor.prefill(&mut session, &[1, 2, 3]).unwrap();
    executor.reset(&mut session).unwrap();
    assert_eq!(executor.driver.fed(), 0);
    assert!(executor.history(&session).is_empty());
    assert_eq!(executor.prefix_limits().roots, 0);
}

#[test]
fn a_zero_sampler_vocabulary_is_refused() {
    let driver = TokenDriver::new(FakeDevice::default(), FakeFeeds::default(), shape()).unwrap();
    let rules = CudaRules {
        eos: Arc::from([]),
        sampler_vocab: 0,
    };
    assert!(FlashCudaExecutor::new(driver, rules).is_err());
}

#[test]
fn the_resume_switch_saves_and_restores_the_boundary() {
    crate::family::flash_next::cuda::resume::arm_for_tests();
    let (mut executor, mut session) = build(&[]);
    // Turn 1: prefill A, save the tip (with real fixed-state bytes), then decode past it.
    executor.reset(&mut session).unwrap();
    executor.prefill(&mut session, &[1, 2, 3]).unwrap();
    executor.driver.device_mut().fixed = vec![7, 7, 9];
    let saved = executor
        .save_prefix(&session, &[1, 2, 3], SaveKind::Tip)
        .unwrap();
    assert!(saved, "an armed switch saves the tip");
    for _ in 0..2 {
        executor.decode(&mut session, &mut generation(2)).unwrap();
    }
    assert_eq!(executor.driver.fed(), 4, "decode moved past the boundary");
    executor.driver.device_mut().fixed = vec![1]; // the arena's fixed image moved on
    // Turn 2: a new job whose prompt has the boundary as a prefix.
    executor.reset(&mut session).unwrap();
    let reused = executor
        .restore_prefix(&mut session, &[1, 2, 3, 4])
        .unwrap();
    assert_eq!(reused, 3, "the boundary is reused as a prefix");
    assert_eq!(
        executor.driver.device().fixed,
        vec![7, 7, 9],
        "the saved fixed-state bytes round-trip"
    );
    assert_eq!(executor.driver.fed(), 2, "the driver sits at the boundary");
    let log = executor.driver.device().log.clone();
    assert_eq!(log.last().unwrap(), "state_write");
    // The session holds the boundary's pending token, so the tail prefill continues it.
    executor.prefill(&mut session, &[4]).unwrap();
    assert_eq!(executor.driver.fed(), 3, "only the tail feeds");
    assert_eq!(executor.history(&session), [1, 2, 3, 4]);
    // A prompt without the saved prefix reuses nothing.
    executor.reset(&mut session).unwrap();
    assert_eq!(
        executor
            .restore_prefix(&mut session, &[9, 2, 3, 4])
            .unwrap(),
        0
    );
    // With the entry present, a shorter prompt that the key does not prefix reuses nothing.
    assert_eq!(executor.restore_prefix(&mut session, &[1]).unwrap(), 0);
}

#[test]
fn a_fresh_prefill_between_the_save_and_the_restore_refuses_the_entry() {
    crate::family::flash_next::cuda::resume::arm_for_tests();
    let (mut executor, mut session) = build(&[]);
    // Request A: prefill and save the boundary.
    executor.reset(&mut session).unwrap();
    executor.prefill(&mut session, &[1, 2, 3]).unwrap();
    assert!(
        executor
            .save_prefix(&session, &[1, 2, 3], SaveKind::Tip)
            .unwrap()
    );
    // Request C: an unrelated job prefills from 0 and overwrites the rows below the boundary.
    executor.reset(&mut session).unwrap();
    executor.prefill(&mut session, &[4, 5]).unwrap();
    executor.decode(&mut session, &mut generation(1)).unwrap();
    // Request D: A's prefix still matches the key, but the arena generation moved on — the
    // entry is refused before any state is written and D falls back to a full prefill.
    executor.reset(&mut session).unwrap();
    let log_len = executor.driver.device().log.len();
    assert_eq!(
        executor
            .restore_prefix(&mut session, &[1, 2, 3, 4])
            .unwrap(),
        0,
        "rows a fresh sequence overwrote are never reused"
    );
    assert_eq!(
        executor.driver.device().log.len(),
        log_len,
        "a refused restore writes no state"
    );
    executor.prefill(&mut session, &[1, 2, 3, 4]).unwrap();
    let batch = executor.decode(&mut session, &mut generation(1)).unwrap();
    // The same prompt on a fresh executor draws the same turn: D matches a fresh D.
    let (mut fresh, mut fresh_session) = build(&[]);
    fresh.reset(&mut fresh_session).unwrap();
    fresh.prefill(&mut fresh_session, &[1, 2, 3, 4]).unwrap();
    let fresh_batch = fresh
        .decode(&mut fresh_session, &mut generation(1))
        .unwrap();
    assert_eq!(batch.tokens, fresh_batch.tokens);
    assert_eq!(
        executor.history(&session),
        fresh.history(&fresh_session),
        "the refused turn equals a fresh full prefill"
    );
}

#[test]
fn two_prompts_over_the_same_boundary_both_reuse_it() {
    crate::family::flash_next::cuda::resume::arm_for_tests();
    let (mut executor, mut session) = build(&[]);
    // Turn 1: prefill A and save the boundary.
    executor.reset(&mut session).unwrap();
    executor.prefill(&mut session, &[1, 2, 3]).unwrap();
    assert!(
        executor
            .save_prefix(&session, &[1, 2, 3], SaveKind::Tip)
            .unwrap()
    );
    // Turn 2: A plus a suffix restores the boundary; the writes it makes all sit at or past
    // the boundary, so the generation does not advance.
    executor.reset(&mut session).unwrap();
    assert_eq!(
        executor
            .restore_prefix(&mut session, &[1, 2, 3, 4])
            .unwrap(),
        3
    );
    executor.prefill(&mut session, &[4]).unwrap();
    executor.decode(&mut session, &mut generation(1)).unwrap();
    // Turn 3: A plus another suffix still reuses the boundary — no fresh prefill intervened.
    executor.reset(&mut session).unwrap();
    assert_eq!(
        executor
            .restore_prefix(&mut session, &[1, 2, 3, 5])
            .unwrap(),
        3,
        "writes at or past the boundary keep the entry valid"
    );
    executor.prefill(&mut session, &[5]).unwrap();
    assert_eq!(executor.history(&session), [1, 2, 3, 5]);
    assert_eq!(executor.driver.fed(), 3, "only the tail feeds");
}

#[test]
fn a_failed_fresh_prefill_attempt_still_invalidates_the_entry() {
    crate::family::flash_next::cuda::resume::arm_for_tests();
    let (mut executor, mut session) = build(&[]);
    // Request A: prefill and save the boundary.
    executor.reset(&mut session).unwrap();
    executor.prefill(&mut session, &[1, 2, 3]).unwrap();
    assert!(
        executor
            .save_prefix(&session, &[1, 2, 3], SaveKind::Tip)
            .unwrap()
    );
    // Request C: a fresh job whose first device call fails — the attempt may still have queued
    // writes at position 0, so the generation must advance at the ATTEMPT, not the success.
    executor.reset(&mut session).unwrap();
    executor.driver.device_mut().fail = Some("begin");
    assert!(executor.prefill(&mut session, &[4, 5]).is_err());
    // Request D: the entry is refused.
    executor.reset(&mut session).unwrap();
    assert_eq!(
        executor
            .restore_prefix(&mut session, &[1, 2, 3, 4])
            .unwrap(),
        0,
        "a failed fresh attempt still invalidates the entry"
    );
}
