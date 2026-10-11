//! What the executor needs to know about the model beyond the step (decision 0033, packet
//! RD-9d): the end-of-sequence ids and the width of the vocabulary the sampler draws from. They
//! become the `Limits` of the shared `DecodeSession` (`family/session.rs`). Pure host values.
//!
//! The ids are shared (`Arc<[u32]>`), so a `decode` call can hold them while it borrows the step
//! mutably, without copying the list.
//!
//! Design guidance only, no code copied.

use super::{dims::StepOptions, resources::check_stage1_options};
use crate::EngError;
use std::sync::Arc;
/// The model facts of one engine that every `decode` call applies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct QwenRules {
    pub(crate) eos: Arc<[u32]>,
    pub(crate) sampler_vocab: usize,
}

impl QwenRules {
    /// `sampler_vocab` must lie in `1..=head_vocab`, the width of the model head, and `eos`
    /// must be a non-empty list of distinct ids inside `0..sampler_vocab`: an id the sampler
    /// can never draw is not an end-of-sequence id (the RD-9d1 c6 message promises both
    /// refusals).
    pub(crate) fn new(
        eos: Vec<u32>,
        sampler_vocab: usize,
        head_vocab: usize,
    ) -> Result<Self, EngError> {
        if !(1..=head_vocab).contains(&sampler_vocab) {
            return Err(EngError::Checkpoint(format!(
                "sampler vocabulary {sampler_vocab} is outside 1..={head_vocab}, the width of the model head"
            )));
        }
        if eos.is_empty() {
            return Err(EngError::Checkpoint(
                "the end-of-sequence set is empty: a session needs at least one eos id".to_owned(),
            ));
        }
        for (index, &id) in eos.iter().enumerate() {
            if usize::try_from(id).is_ok_and(|id| id >= sampler_vocab) {
                return Err(EngError::Checkpoint(format!(
                    "end-of-sequence id {id} (index {index}) is outside the sampler vocabulary 0..{sampler_vocab}"
                )));
            }
            if eos[..index].contains(&id) {
                return Err(EngError::Checkpoint(format!(
                    "end-of-sequence id {id} appears more than once"
                )));
            }
        }
        Ok(Self {
            eos: eos.into(),
            sampler_vocab,
        })
    }
}

/// Refuses a model head narrower than the sampler vocabulary. The head tensor is padded to
/// `head_rows` rows while the sampler draws from `sampler_vocab` ids, so `head_rows` must be at
/// least `sampler_vocab`; the rows above it are padding.
pub(super) fn check_vocab(head_rows: usize, sampler_vocab: usize) -> Result<(), EngError> {
    if head_rows < sampler_vocab {
        return Err(EngError::Checkpoint(format!(
            "the model head has {head_rows} rows but the sampler vocabulary is {sampler_vocab}"
        )));
    }
    Ok(())
}

/// The width every sampler-facing call of a step takes: `sampler_vocab`, never the padded
/// `head_rows` of the model head. A padding id (`sampler_vocab..head_rows`) must never be
/// decided or sampled, so the device argmax runs over the first `sampler_vocab` values of the
/// logits row and the result pair is judged against the same width. [`check_vocab`] (run by
/// [`validate_step_inputs`] before anything is allocated) guarantees that `head_rows` is at
/// least `sampler_vocab`, so the padding range above is never empty or wrapped; the head
/// argument is kept because both call sites name the head width they were checked against.
pub(super) fn sampler_width(_head_rows: usize, sampler_vocab: usize) -> usize {
    sampler_vocab
}

/// The two sampler widths the step wires in, named per call site: the `n` of the device argmax
/// ([`StepWidths::argmax_n`]) and the width the result pair is judged against
/// ([`StepWidths::verdict`]). Both are the sampler vocabulary; they are two fields so the two
/// sites stay separately named and a mutant of one does not hide behind the other.
#[derive(Debug)]
pub(super) struct StepWidths {
    pub(super) argmax_n: usize,
    pub(super) verdict: usize,
}

/// Every refusal `QwenResidentStep::new` makes before it touches the context, in one pure seam:
/// the session options ([`check_stage1_options`]), then the head width ([`check_vocab`]), then
/// the widths the step wires in. The step itself needs a `CudaContext` and cannot run on the
/// CPU, so these refusals are tested here instead (lane C review, RD-9d c16).
pub(super) fn validate_step_inputs(
    options: &StepOptions,
    head_rows: usize,
    rules: &QwenRules,
) -> Result<StepWidths, EngError> {
    check_stage1_options(options)?;
    check_vocab(head_rows, rules.sampler_vocab)?;
    Ok(StepWidths {
        argmax_n: sampler_width(head_rows, rules.sampler_vocab),
        verdict: sampler_width(head_rows, rules.sampler_vocab),
    })
}
