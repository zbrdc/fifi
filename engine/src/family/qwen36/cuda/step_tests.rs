//! Unit tests of the pure pieces `step.rs` uses (test seam U13). CPU only. The step itself needs
//! a device; its evidence is the scratch run R-3 of the packet. The wirings of the step are
//! pinned on the merged source with `include_str!` (the precedent of `step_ops_tests.rs`): the
//! step cannot run here, so a mutant of a wiring must still fail a named test.
use super::{state::StepField, state_layout::Word};
use crate::kernels::cuda::resident::step_params::{
    OFFSET_KV_LEN, OFFSET_POS, OFFSET_STATUS, OFFSET_TOKEN_ID, STEP_PARAMS_BYTES,
};

/// The text of one section of the merged `step.rs` (a function body or a match arm), from
/// `start` to the first of the `next` markers that begins the following item. The length of the
/// extracted text is printed by every wiring assert that uses it.
fn section(start: &str, next: &[&str]) -> &'static str {
    let source = include_str!("step.rs");
    let at = source
        .find(start)
        .unwrap_or_else(|| panic!("step.rs has no {start:?}"));
    let after = at + start.len();
    let end = next
        .iter()
        .filter_map(|marker| source[after..].find(marker).map(|found| after + found))
        .min()
        .unwrap_or(source.len());
    &source[at..end]
}

#[test]
fn every_step_field_is_a_distinct_four_byte_word_inside_the_block() {
    let fields = [
        (StepField::TokenId, OFFSET_TOKEN_ID),
        (StepField::Pos, OFFSET_POS),
        (StepField::KvLen, OFFSET_KV_LEN),
        (StepField::Status, OFFSET_STATUS),
    ];
    for (field, offset) in fields {
        assert_eq!(field.offset(), offset, "{field:?}");
        assert!(field.offset().is_multiple_of(4) && field.offset() + 4 <= STEP_PARAMS_BYTES);
    }
    let mut offsets = fields.map(|(field, _)| field.offset());
    offsets.sort_unstable();
    assert!(offsets.windows(2).all(|w| w[0] != w[1]), "{offsets:?}");
}

#[test]
fn the_start_block_and_the_result_pair_have_their_own_words() {
    assert_eq!(Word::StepParams.bytes(), STEP_PARAMS_BYTES);
    assert_eq!(Word::ResultPair.bytes(), 8);
    assert_ne!(Word::StepParams.offset(), Word::ResultPair.offset());
}

#[test]
fn new_validates_the_step_inputs_before_it_touches_the_context() {
    let body = section("fn new(", &["\n    pub(crate) fn ", "\n    fn "]);
    let length = body.len();
    assert!(
        body.contains("validate_step_inputs(options"),
        "the new body ({length} bytes) does not wire validate_step_inputs(options, ...)"
    );
    let call = body
        .find("validate_step_inputs(options")
        .expect("the call was just asserted");
    let stream = body
        .find("context.stream()")
        .expect("new opens the stream (context.stream())");
    assert!(
        call < stream,
        "validate_step_inputs must refuse before the first context.stream() in a body of \
         {length} bytes: call at {call}, stream at {stream}"
    );
}

#[test]
fn the_argmax_arm_passes_the_sampler_width_not_the_head_rows() {
    let arm = section("StepOp::Argmax =>", &["StepOp::"]);
    let length = arm.len();
    assert!(
        arm.contains("self.widths.argmax_n"),
        "the Argmax arm ({length} bytes) does not pass self.widths.argmax_n as n"
    );
    assert!(
        !arm.contains("shape.vocab"),
        "the Argmax arm ({length} bytes) reads the padded head width instead of the sampler \
         width"
    );
}

#[test]
fn run_token_decides_against_the_sampler_width() {
    let body = section("fn run_token", &["\n    pub(crate) fn ", "\n    fn "]);
    let length = body.len();
    assert!(
        body.contains("decide(pair, self.widths.verdict)"),
        "the run_token body ({length} bytes) does not call decide(pair, self.widths.verdict)"
    );
    assert!(
        !body.contains("shape.vocab"),
        "the run_token body ({length} bytes) reads the padded head width instead of the \
         sampler width"
    );
}
