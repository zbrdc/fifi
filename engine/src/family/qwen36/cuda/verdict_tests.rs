//! Unit tests of the result-pair decision and the failure action (test seam U13). CPU only.
use super::{
    book::CommitRefusal,
    rules::sampler_width,
    verdict::{Decision, FailureAction, action_for, bad_token_error, decide},
};
use crate::{EngError, kernels::cuda::resident::step_params::ResultPair};

/// The Qwen3.6 head width.
const VOCAB: usize = 248_320;

fn pair(token: i32, status: i32) -> ResultPair {
    let mut bytes = [0_u8; 8];
    bytes[..4].copy_from_slice(&token.to_le_bytes());
    bytes[4..].copy_from_slice(&status.to_le_bytes());
    ResultPair::from_bytes(bytes)
}

#[test]
fn a_clean_pair_yields_a_token_inside_the_vocabulary() {
    assert_eq!(decide(pair(0, 0), VOCAB), Decision::Commit { token: 0 });
    assert_eq!(
        decide(pair(248_319, 0), VOCAB),
        Decision::Commit { token: 248_319 }
    );
    assert_eq!(decide(pair(248_320, 0), VOCAB), Decision::BadToken(248_320));
    assert_eq!(decide(pair(-1, 0), VOCAB), Decision::BadToken(-1));
}

#[test]
fn a_padding_id_is_a_bad_token_of_the_sampler_vocabulary() {
    // The head is padded to VOCAB rows while the sampler draws from 248_077 ids: token 248_100
    // sits in the padding, inside the head but outside the sampler vocabulary.
    let width = sampler_width(VOCAB, 248_077);
    assert_eq!(width, 248_077);
    assert_eq!(decide(pair(248_100, 0), width), Decision::BadToken(248_100));
}

#[test]
fn the_argmax_width_skips_the_padding_rows_of_the_head() {
    // The largest value sits at a padding index; a reference argmax over the sampler width
    // (largest value, lowest index on a tie) returns the best real index instead.
    let n = sampler_width(VOCAB, 248_077);
    let mut row = vec![-1.0_f32; VOCAB];
    row[123] = 42.0;
    row[VOCAB - 1] = 100.0;
    let mut best = 0;
    for (index, &value) in row[..n].iter().enumerate() {
        if value > row[best] {
            best = index;
        }
    }
    assert_eq!(best, 123);
    assert!(
        row[VOCAB - 1] > row[123],
        "only the width excludes the padding winner"
    );
}

#[test]
fn the_status_is_judged_before_the_token() {
    // An all-NaN row gives token -1 with mask 2; a flagged row keeps a filler token.
    for status in [1, 2, 4, 8, 1 | 2 | 4, 16, i32::MIN] {
        for token in [-1, 7, 248_320] {
            assert_eq!(
                decide(pair(token, status), VOCAB),
                Decision::Commit { token: 0 },
                "status {status:#x} token {token}"
            );
        }
    }
}

#[test]
fn a_bad_token_is_a_sample_error() {
    match bad_token_error(-1) {
        EngError::Backend { operation, message } => {
            assert_eq!(operation, "sample");
            assert!(message.contains("-1"), "{message}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn commit_step_maps_the_status_with_precedence_4_8_1_2() {
    // The rule this packet relies on (REPLY-76): the class of a refused status, 4 first.
    let order = [
        (4, "step_params"),
        (8, "prefill_chunk"),
        (1, "route_experts"),
        (2, "sample"),
    ];
    for status in 1..16 {
        let expected = order
            .iter()
            .find(|(mask, _)| status & mask != 0)
            .map(|(_, operation)| *operation);
        match EngError::from(CommitRefusal::Status(status)) {
            EngError::Backend { operation, .. } => {
                assert_eq!(Some(operation), expected, "status {status}");
            }
            other => panic!("status {status}: {other:?}"),
        }
    }
}

#[test]
fn only_a_failure_of_the_first_op_takes_the_step_back() {
    assert_eq!(action_for(0), FailureAction::Cancel);
    assert_eq!(action_for(1), FailureAction::Cancel);
    assert_eq!(action_for(2), FailureAction::Poison);
    assert_eq!(action_for(80), FailureAction::Poison);
}
