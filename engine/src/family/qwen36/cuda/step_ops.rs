//! The launches of one decode token as DATA: the order in which the resident step queues its
//! kernels, written once so the CPU tests can check it against the phase table of `step_plan.rs`.
//! Pure host code.
//!
//! One token is: `step_begin`, the embedding, for every layer the mixer block and the MoE block,
//! the final norm, the `lm_head` projection, `argmax`, `step_advance`. The device runner executes
//! the list op by op; the same list is what a graph capture records, so eager and captured runs
//! share one order.
//!
//! # The residual stream
//!
//! Two buffers alternate. The embedding writes `ResidA`. Layer `l` reads the residual from
//! `resid_in(l)`; its mixer block adds its output into that buffer in place; its MoE block writes
//! the layer output into `resid_out(l)`, the other buffer, which is `resid_in(l + 1)`. The final
//! norm reads `final_resid`.
//!
//! # Poisoned step words
//!
//! The words `pos`, `kv_len` and `kv_slot` are read by a fixed set of kernels
//! ([`POS_WORD_READERS`]). Each range-checks the word before it touches memory and has a test
//! that runs the poisoned word (the poison-consumer rule). The Fp16 step reads the words through
//! `attn_prep` and `attn_decode` alone; the Int8G32 mode adds three readers: the rows entry
//! `attn_prep_rows` and `kv8_append` read `pos`, and `attn_decode_q8` reads `kv_len`. The
//! embedding reads `token_id` only and has its own bad-id path (NaN row and status mask 4).

use super::step_plan::{Phase, Slot, writes_status};
use crate::family::AttentionKind;

/// One launch group of the token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StepOp {
    /// `step_begin`: clears `status`, flags an invalid step state with mask 4.
    Begin,
    /// The embedding row of `token_id` into `ResidA`.
    Embed,
    /// `run_block` of the layer: residual to the MoE input.
    Mixer(usize),
    /// `run_moe_block` of the layer: MoE input to the layer output.
    Moe(usize),
    /// The output norm of the last residual, packed for `lm_head`.
    FinalNorm,
    /// The vendored Q6_K projection into the logits row.
    LmHead,
    /// Greedy argmax, the result pair, the LAST status writer.
    Argmax,
    /// `pos`, `kv_len` and `kv_slot` move on by one; the last kernel.
    Advance,
}

/// The ops of one token for a model of `layers` layers, in queue order.
pub(super) fn token_ops(layers: usize) -> Vec<StepOp> {
    let mut ops = vec![StepOp::Begin, StepOp::Embed];
    for layer in 0..layers {
        ops.push(StepOp::Mixer(layer));
        ops.push(StepOp::Moe(layer));
    }
    ops.extend([
        StepOp::FinalNorm,
        StepOp::LmHead,
        StepOp::Argmax,
        StepOp::Advance,
    ]);
    ops
}

impl StepOp {
    /// The phases of `step_plan.rs` the op runs through, in order. A mixer block runs the phases
    /// of its layer kind.
    pub(super) fn phases(self, kind: AttentionKind) -> Vec<Phase> {
        match self {
            Self::Begin => vec![Phase::StepBegin],
            Self::Embed => vec![Phase::Embed],
            Self::Mixer(_) if kind == AttentionKind::DeltaNet => vec![
                Phase::MixPack,
                Phase::DeltaIn,
                Phase::DeltaConv,
                Phase::DeltaRule,
                Phase::DeltaPack,
                Phase::DeltaOut,
                Phase::AddNorm,
            ],
            Self::Mixer(_) => vec![
                Phase::MixPack,
                Phase::AttnIn,
                Phase::AttnPrep,
                // The KV8 append of the Int8G32 mode: no kernel of the Fp16 path
                // enters this phase, and the Int8G32 op list runs `kv8_append` in it.
                Phase::Kv8Append,
                Phase::AttnCore,
                Phase::AttnPack,
                Phase::AttnOut,
                Phase::AddNorm,
            ],
            Self::Moe(_) => vec![
                Phase::Route,
                Phase::GateUp,
                Phase::Swiglu,
                Phase::Down,
                Phase::Combine,
            ],
            Self::FinalNorm => vec![Phase::FinalNorm],
            Self::LmHead => vec![Phase::LmHead],
            Self::Argmax => vec![Phase::Argmax],
            Self::Advance => vec![Phase::Advance],
        }
    }

    /// Whether any kernel of the op writes the status word, from `step_plan.rs`'s table.
    pub(super) fn writes_status(self, kind: AttentionKind) -> bool {
        self.phases(kind).into_iter().any(writes_status)
    }

    /// The layer of a mixer or MoE op.
    pub(super) fn layer(self) -> Option<usize> {
        match self {
            Self::Mixer(layer) | Self::Moe(layer) => Some(layer),
            _ => None,
        }
    }
}

/// The buffer layer `layer` reads its residual from (the embedding wrote layer 0's).
pub(super) fn resid_in(layer: usize) -> Slot {
    if layer.is_multiple_of(2) {
        Slot::ResidA
    } else {
        Slot::ResidB
    }
}

/// The buffer the MoE block of `layer` writes the layer output into.
pub(super) fn resid_out(layer: usize) -> Slot {
    resid_in(layer + 1)
}

/// The buffer the final norm reads after `layers` layers.
pub(super) fn final_resid(layers: usize) -> Slot {
    match layers.checked_sub(1) {
        Some(last) => resid_out(last),
        None => Slot::ResidA,
    }
}

/// One kernel that reads or writes a step word of the position family, and the merged test that
/// runs it on a poisoned word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PosReader {
    pub(super) kernel: &'static str,
    pub(super) words: &'static str,
    /// Repository path of the test file, relative to `engine/src/kernels/cuda/resident/`.
    pub(super) test_file: &'static str,
    pub(super) test: &'static str,
}

/// Every kernel of the step that touches `pos`, `kv_len` or `kv_slot`, in launch order. The
/// Int8G32 kernels sit beside the Fp16 pair they replace: `attn_prep_rows` and `kv8_append`
/// read `pos` (the append is never the first or only mask-4 writer: its guard equals the
/// rows entry's, one launch earlier), and `attn_decode_q8` reads `kv_len`.
pub(super) const POS_WORD_READERS: [PosReader; 7] = [
    PosReader {
        kernel: "step_begin",
        words: "pos kv_len kv_slot",
        test_file: "step_params/tests.rs",
        test: "step_begin_clears_the_status_and_flags_a_bad_state",
    },
    PosReader {
        kernel: "attn_prep",
        words: "pos",
        test_file: "attn_prep/tests.rs",
        test: "a_poisoned_position_word_writes_nothing_and_reports_mask_4",
    },
    PosReader {
        kernel: "attn_prep_rows",
        words: "pos",
        test_file: "attn_prep/tests.rs",
        test: "a_poisoned_word_writes_no_rows_and_reports_mask_4",
    },
    PosReader {
        kernel: "kv8_append",
        words: "pos",
        test_file: "kv8_append/tests.rs",
        test: "kv8_append_refuses_a_bad_position_word_without_writing",
    },
    PosReader {
        kernel: "attn_decode",
        words: "kv_len",
        test_file: "attn_decode/tests.rs",
        test: "attn_decode_marks_an_invalid_kv_len_with_nan",
    },
    PosReader {
        kernel: "attn_decode_q8",
        words: "kv_len",
        test_file: "attn_decode_q8/tests.rs",
        test: "attn_decode_q8_marks_an_invalid_kv_len_with_nan",
    },
    PosReader {
        kernel: "step_advance",
        words: "pos kv_len kv_slot",
        test_file: "step_params/tests.rs",
        test: "step_advance_adds_m_and_keeps_kv_len_one_ahead",
    },
];
