//! Unit tests of the pure pieces `step.rs` uses (test seam U13). CPU only. The step itself needs
//! a device; its evidence is the scratch run R-3 of the packet. The wirings of the step are
//! pinned on the merged source with `include_str!` (the precedent of `step_ops_tests.rs`): the
//! step cannot run here, so a mutant of a wiring must still fail a named test. The pinned text
//! is stripped of line comments and string literals and its whitespace is collapsed first, so a
//! comment cannot satisfy a pin and a rustfmt re-wrap cannot break one.
use super::{state::StepField, state_layout::Word};
use crate::kernels::cuda::resident::step_params::{
    OFFSET_KV_LEN, OFFSET_POS, OFFSET_STATUS, OFFSET_TOKEN_ID, STEP_PARAMS_BYTES,
};

/// The text of one `fn` item of the merged `step.rs`, from the line that holds `start` to the
/// next line that starts a `fn` item at the same indentation, whatever its `pub` prefix: the end
/// of a section never depends on the visibility of the helper that follows it. The length of the
/// extracted text is printed by every wiring assert that uses it.
fn fn_section(start: &str) -> &'static str {
    let source = include_str!("step.rs");
    let at = source
        .find(start)
        .unwrap_or_else(|| panic!("step.rs has no {start:?}"));
    let line_start = source[..at].rfind('\n').map_or(0, |nl| nl + 1);
    let indent = source[line_start..]
        .find(|c: char| c != ' ')
        .unwrap_or(source[line_start..].len());
    let mut end = source.len();
    let mut line = source[at..]
        .find('\n')
        .map_or(source.len(), |nl| at + nl + 1);
    while line < source.len() {
        let next = source[line..]
            .find('\n')
            .map_or(source.len(), |nl| line + nl + 1);
        if starts_fn_item(&source[line..next], indent) {
            end = line;
            break;
        }
        line = next;
    }
    &source[line_start..end]
}

/// True when `line` sits at exactly `indent` spaces and opens a `fn` item behind any `pub`
/// prefix (`fn `, `pub fn `, `pub(crate) fn `, `pub(super) fn `).
fn starts_fn_item(line: &str, indent: usize) -> bool {
    let bytes = line.as_bytes();
    bytes.len() > indent && (0..indent).all(|at| bytes[at] == b' ') && bytes[indent] != b' ' && {
        let rest = ["pub(crate) ", "pub(super) ", "pub "]
            .iter()
            .find_map(|prefix| line[indent..].strip_prefix(prefix))
            .unwrap_or(&line[indent..]);
        rest.starts_with("fn ")
    }
}

/// The text of one match arm of the merged `step.rs`, from the line that holds `start` to the
/// next line that opens the following arm (a line whose text starts with `StepOp::`).
fn arm_section(start: &str) -> &'static str {
    let source = include_str!("step.rs");
    let at = source
        .find(start)
        .unwrap_or_else(|| panic!("step.rs has no {start:?}"));
    let line_start = source[..at].rfind('\n').map_or(0, |nl| nl + 1);
    let mut end = source.len();
    let mut line = source[at..]
        .find('\n')
        .map_or(source.len(), |nl| at + nl + 1);
    while line < source.len() {
        let next = source[line..]
            .find('\n')
            .map_or(source.len(), |nl| line + nl + 1);
        if source[line..next].trim_start().starts_with("StepOp::") {
            end = line;
            break;
        }
        line = next;
    }
    &source[line_start..end]
}

/// A copy of the section text without line comments and without string literals: every `//` to
/// the end of its line (`///` and `//!` included) and every `"..."` (with `\` escapes) becomes
/// one space. A pin must read real code only, or a comment holding the pinned text would
/// satisfy it.
fn code_only(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut plain = 0;
    let mut at = 0;
    while at < bytes.len() {
        match bytes[at] {
            b'/' if bytes.get(at + 1) == Some(&b'/') => {
                out.push_str(&text[plain..at]);
                out.push(' ');
                while at < bytes.len() && bytes[at] != b'\n' {
                    at += 1;
                }
                plain = at;
            }
            b'"' => {
                out.push_str(&text[plain..at]);
                out.push(' ');
                at += 1;
                while at < bytes.len() {
                    match bytes[at] {
                        b'\\' => at = (at + 2).min(bytes.len()),
                        b'"' => {
                            at += 1;
                            break;
                        }
                        _ => at += 1,
                    }
                }
                plain = at;
            }
            _ => at += 1,
        }
    }
    out.push_str(&text[plain..]);
    out
}

/// The pinned form of a section: code only, with every run of whitespace collapsed to one
/// space, so a rustfmt re-wrap of the source cannot break a pin.
fn pinned_text(section: &str) -> String {
    code_only(section)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
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
    let body = pinned_text(fn_section("fn new("));
    let length = body.len();
    assert!(
        body.contains("validate_step_inputs(options, res.dims.shape.vocab, &rules)?"),
        "the new body ({length} bytes) does not pin \
         validate_step_inputs(options, res.dims.shape.vocab, &rules)?"
    );
    let call = body
        .find("validate_step_inputs(")
        .expect("the call was just asserted");
    // The validation precedes every allocation of the step: the stream, the resident state,
    // the event, the pinned word, the arena and the host buffers.
    for allocation in [
        "context.stream()",
        "ResidentState::allocate",
        "context.event()",
        "context.pinned_word()",
        ".arena(",
        "vec![",
    ] {
        let mut found = body.find(allocation);
        while let Some(at) = found {
            assert!(
                call < at,
                "validate_step_inputs must refuse before {allocation} in a body of {length} \
                 bytes: call at {call}, {allocation} at {at}"
            );
            found = body[at + 1..].find(allocation).map(|more| at + 1 + more);
        }
    }
}

#[test]
fn the_argmax_arm_passes_the_sampler_width_not_the_head_rows() {
    let arm = pinned_text(arm_section("StepOp::Argmax =>"));
    let length = arm.len();
    assert!(
        arm.contains("n: self.widths.argmax_n"),
        "the Argmax arm ({length} bytes) does not pass self.widths.argmax_n as n"
    );
    for wrong in ["n_rows()", "globals.head", "shape.vocab"] {
        assert!(
            !arm.contains(wrong),
            "the Argmax arm ({length} bytes) reads {wrong} instead of the sampler width"
        );
    }
}

#[test]
fn run_token_decides_against_the_sampler_width() {
    let body = pinned_text(fn_section("fn run_token"));
    let length = body.len();
    assert!(
        body.contains("decide(pair, self.widths.verdict)"),
        "the run_token body ({length} bytes) does not call decide(pair, self.widths.verdict)"
    );
    for wrong in ["n_rows()", "globals.head", "shape.vocab"] {
        assert!(
            !body.contains(wrong),
            "the run_token body ({length} bytes) reads {wrong} instead of the sampler width"
        );
    }
}

#[test]
fn run_body_counts_a_failed_sync_as_an_entered_op() {
    // The eager debug branch synchronizes after every op; the wiring of c17b: a failed sync
    // must RAISE `entered` through `entered_after_failed_sync`, or a fault after op 1 reads
    // as a refused first launch and the step would be taken back instead of poisoned. The
    // helper is `#[must_use]`, and the pin takes the whole assignment: dropping the raise
    // with a `let _ =` keeps the call and survives a bare-name pin.
    let body = pinned_text(fn_section("fn run_body"));
    let length = body.len();
    assert!(
        body.contains("*entered = entered_after_failed_sync("),
        "the run_body body ({length} bytes) does not assign the raised count with \
         *entered = entered_after_failed_sync("
    );
}

#[test]
fn run_token_hands_the_decision_status_to_commit_step() {
    // The wiring of c18: `decide` carries the pair's status in the Commit arm and the commit
    // must take that SAME binding, not a literal clean status and not a fresh read of the
    // pair: with a literal 0 a flagged row would commit as clean and return its filler
    // token 0 (mutant m8).
    let body = pinned_text(fn_section("fn run_token"));
    let length = body.len();
    let arm = body
        .find("Decision::Commit { token, status }")
        .unwrap_or_else(|| {
            panic!(
                "the run_token body ({length} bytes) does not bind \
                 Decision::Commit {{ token, status }}"
            )
        });
    let commit = body
        .find("self.state.commit_step(STEP_ROWS, status)")
        .unwrap_or_else(|| {
            panic!(
                "the run_token body ({length} bytes) does not call \
                 self.state.commit_step(STEP_ROWS, status)"
            )
        });
    assert!(
        arm < commit,
        "the Commit arm at {arm} must bind the status before the commit at {commit} in a body \
         of {length} bytes"
    );
    for wrong in ["commit_step(STEP_ROWS, 0)", "pair.status"] {
        assert!(
            !body.contains(wrong),
            "the run_token body ({length} bytes) hands the commit {wrong}: the status must \
             come from the decision"
        );
    }
}

#[test]
fn run_token_refuses_a_bad_id_without_poisoning_or_cancelling() {
    // The refusal of c18 leaves the state idle: between the id check and the start-block
    // write there must be no poison, cancel or failure handling, or a refused host id would
    // poison or take back a session whose device saw nothing (mutant m9).
    let body = pinned_text(fn_section("fn run_token"));
    let length = body.len();
    let check = body
        .find("check_token_id(token, self.widths.verdict)")
        .unwrap_or_else(|| {
            panic!(
                "the run_token body ({length} bytes) does not call \
                 check_token_id(token, self.widths.verdict)"
            )
        });
    let write = body
        .find("write_start(")
        .unwrap_or_else(|| panic!("the run_token body ({length} bytes) does not call write_start"));
    assert!(
        check < write,
        "check_token_id at {check} must precede write_start( at {write} in a body of {length} \
         bytes"
    );
    let between = &body[check..write];
    for call in ["poison(", "cancel_enqueue(", "fail("] {
        assert!(
            !between.contains(call),
            "the run_token body ({length} bytes) calls {call} between the id check at {check} \
             and write_start( at {write}: a refusal must leave the state idle"
        );
    }
}

#[test]
fn run_token_checks_the_token_id_before_the_start_block() {
    // The wiring of c18: the host refuses a token outside the sampler vocabulary before the
    // start block, or a padding id would be written to the device before the refusal and the
    // state would be poisoned by its own write.
    let body = pinned_text(fn_section("fn run_token"));
    let length = body.len();
    let check = body
        .find("check_token_id(token, self.widths.verdict)")
        .unwrap_or_else(|| {
            panic!(
                "the run_token body ({length} bytes) does not call \
                 check_token_id(token, self.widths.verdict)"
            )
        });
    let mut found = body.find("write_start(");
    while let Some(at) = found {
        assert!(
            check < at,
            "check_token_id must refuse before write_start( in a body of {length} bytes: \
             check at {check}, write_start( at {at}"
        );
        found = body[at + 1..]
            .find("write_start(")
            .map(|more| at + 1 + more);
    }
}

#[test]
fn write_start_checks_the_token_id_before_the_device_write() {
    // The start-block path itself refuses first too: nothing reaches write_step_params (the
    // book mutation and the host-to-device copy) before the id is judged against the sampler
    // vocabulary, so a refusal leaves the state idle.
    let body = pinned_text(fn_section("fn write_start"));
    let length = body.len();
    let check = body
        .find("check_token_id(token, self.widths.verdict)")
        .unwrap_or_else(|| {
            panic!(
                "the write_start body ({length} bytes) does not call \
                 check_token_id(token, self.widths.verdict)"
            )
        });
    let mut found = body.find("write_step_params(");
    while let Some(at) = found {
        assert!(
            check < at,
            "check_token_id must refuse before write_step_params( in a body of {length} \
             bytes: check at {check}, write_step_params( at {at}"
        );
        found = body[at + 1..]
            .find("write_step_params(")
            .map(|more| at + 1 + more);
    }
}
