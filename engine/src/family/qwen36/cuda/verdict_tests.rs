//! Unit tests of the result-pair decision and the failure action. CPU only.
use super::{
    book::CommitRefusal,
    rules::sampler_width,
    verdict::{
        Decision, FailureAction, action_for, bad_token_error, decide, entered_after_failed_sync,
    },
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
    assert_eq!(
        decide(pair(0, 0), VOCAB),
        Decision::Commit {
            token: 0,
            status: 0
        }
    );
    assert_eq!(
        decide(pair(248_319, 0), VOCAB),
        Decision::Commit {
            token: 248_319,
            status: 0
        }
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
                Decision::Commit { token: 0, status },
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
    // The rule the step relies on: the class of a refused status, 4 first.
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

#[test]
fn a_failed_sync_after_the_first_op_poisons_the_step() {
    // With INFERENCE_CUDA_SYNC_EACH on, a synchronize error after op 1 arrives with
    // `entered == 1`, but op 1 WAS queued and the error is a device fault: the failed sync
    // counts as having entered the next op, so the failure rule poisons, not cancels.
    assert_eq!(entered_after_failed_sync(1), 2);
    assert_eq!(
        action_for(entered_after_failed_sync(1)),
        FailureAction::Poison
    );
    assert_eq!(
        action_for(entered_after_failed_sync(40)),
        FailureAction::Poison
    );
}

#[test]
fn a_flagged_pair_carries_its_exact_status_and_never_a_token() {
    // The read-back routes the pair: status 1, 2, 4 and 8 (and any mix) reaches
    // `commit_step` with that exact status, whatever token the flagged row holds (the filler
    // -1, a real id, a padding id or an id outside the head).
    for status in [1, 2, 4, 8, 1 | 2, 4 | 8, 16] {
        for token in [-1, 0, 7, 248_077, 248_100, 248_320] {
            assert_eq!(
                decide(pair(token, status), 248_077),
                Decision::Commit { token: 0, status },
                "status {status:#x} token {token}"
            );
        }
    }
}

#[test]
fn a_clean_row_at_the_sampler_bound_is_a_bad_token() {
    // The sampler width is exclusive: 248_076 is the last id it holds, while 248_077 (the
    // width itself, the first padding row of the head) and any negative token are bad.
    let width = sampler_width(VOCAB, 248_077);
    assert_eq!(
        decide(pair(248_076, 0), width),
        Decision::Commit {
            token: 248_076,
            status: 0
        }
    );
    assert_eq!(decide(pair(248_077, 0), width), Decision::BadToken(248_077));
    assert_eq!(decide(pair(-1, 0), width), Decision::BadToken(-1));
}
