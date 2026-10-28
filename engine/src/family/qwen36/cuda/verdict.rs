//! What the host decides from the result pair of a token of the resident step, and what it does
//! when a launch fails (decision 0047, packet RD-9d). No device type appears here, so the rules
//! are tested on the CPU without a `CudaContext`.
//!
//! # Status
//!
//! The status word has four masks: 1 router fault (`route_experts`), 2 non-finite logits
//! (`sample`), 4 invalid step state (`step_params`) and 8 prefill chunk (`prefill_chunk`).
//! `argmax` is the LAST kernel that writes status (`step_plan.rs` checks the phase table;
//! `step_ops.rs` checks the launch order), so the status half of the result pair is final.
//!
//! In stage 1 ANY non-zero status bit (1, 2, 4, 8 or an unknown one) poisons the state and
//! commits nothing (lane M, REPLY-76 Q2). `ResidentState::commit_step(accepted, status)` does
//! both and returns the error of the wrapper's class (`book.rs::status_error`, precedence
//! 4 > 8 > 1 > 2, `Backend`), so this file does not classify a status itself: [`decide`] only
//! keeps the order "status first, token second" and the vocabulary check of a clean token.

use crate::{EngError, kernels::cuda::resident::step_params::ResultPair};

/// What `run_token` does with a result pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Decision {
    /// Hand the pair's status to `commit_step`. A clean status commits and yields `token`; a
    /// non-zero status is refused there (the state is poisoned). `token` is 0 and unused when
    /// the status is not clean: the filler token of a flagged row never reaches the sampler.
    Commit { token: u32, status: i32 },
    /// A clean status with a token outside `0..vocab`: poison, commit nothing.
    BadToken(i32),
}

/// Maps the status before the token: a flagged row is never judged by its token.
pub(super) fn decide(pair: ResultPair, vocab: usize) -> Decision {
    if pair.status.0 != 0 {
        return Decision::Commit {
            token: 0,
            status: pair.status.0,
        };
    }
    match u32::try_from(pair.token) {
        Ok(token) if usize::try_from(token).is_ok_and(|id| id < vocab) => {
            Decision::Commit { token, status: 0 }
        }
        _ => Decision::BadToken(pair.token),
    }
}

/// The error for a clean status with a token the vocabulary does not have.
pub(super) fn bad_token_error(token: i32) -> EngError {
    EngError::Backend {
        operation: "sample",
        message: format!("the device argmax chose token {token}, outside the vocabulary"),
    }
}

/// What the caller does when a launch of the body failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FailureAction {
    /// Nothing was queued: take the admitted step back (`cancel_enqueue`).
    Cancel,
    /// Work may be queued: poison the state.
    Poison,
}

/// `entered` counts the ops of the body that were entered, the failing one included. Op 1 is
/// `step_begin`, a single wrapper that queues nothing when it refuses; any later op may have
/// queued work (a block is entered whole), so any `run_block` error poisons (RD-9b, binding).
pub(super) fn action_for(entered: usize) -> FailureAction {
    if entered <= 1 {
        FailureAction::Cancel
    } else {
        FailureAction::Poison
    }
}

/// A synchronize error after op `entered` arrived means that op was QUEUED and the device
/// faulted, but `entered` still counts only the ops that ran: with `entered == 1` the failure
/// rule of [`action_for`] would read "the first op refused to launch, nothing ran" and take the
/// step back. The failed sync counts as having entered the next op, so the rule poisons.
#[must_use]
pub(super) fn entered_after_failed_sync(entered: usize) -> usize {
    entered + 1
}
