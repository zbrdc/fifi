//! Unit tests of the pure pieces `step.rs` uses (test seam U13). CPU only. The step itself needs
//! a device; its evidence is the scratch run R-3 of the packet. The wirings of the step are
//! pinned on the merged source with `include_str!` (the precedent of `step_ops_tests.rs`): the
//! step cannot run here, so a mutant of a wiring must still fail a named test. The pinned text
//! is stripped of line comments and string literals and its whitespace is collapsed first, so a
//! comment cannot satisfy a pin and a rustfmt re-wrap cannot break one.
use super::run_report::CaptureOutcome;
use super::state::StepField;
use super::state_layout::Word;
use super::step::{HowRan, Mode, fold_capture, next_mode, reported_outcome};
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
fn a_layout_mismatch_is_refused_before_any_context_work() {
    use crate::family::project::cuda::WeightLayout;
    // The pure refusal: the options must agree with the loaded weights.
    assert!(super::step::agree_layout(WeightLayout::Native, WeightLayout::Native).is_ok());
    assert!(super::step::agree_layout(WeightLayout::Tiled, WeightLayout::Tiled).is_ok());
    for (weights, options) in [
        (WeightLayout::Tiled, WeightLayout::Native),
        (WeightLayout::Native, WeightLayout::Tiled),
    ] {
        match super::step::agree_layout(weights, options) {
            Err(crate::EngError::Checkpoint(message)) => {
                assert!(message.contains("follows the loaded model"), "{message}")
            }
            other => panic!("a mismatched layout must be refused: {other:?}"),
        }
    }
    // And `new` runs it before anything device-side: the wiring is pinned on the merged source
    // (the c17 style), so a mutant that reads the layout from the options alone still fails.
    let body = pinned_text(fn_section("pub(crate) fn new("));
    let length = body.len();
    let agree = body
        .find("agree_layout(weights.layout()?, options.layout)?;")
        .unwrap_or_else(|| panic!("new ({length} bytes) never agrees the layouts"));
    let device = body
        .find("context.stream()")
        .unwrap_or_else(|| panic!("new ({length} bytes) never opens a stream"));
    assert!(
        agree < device,
        "the layout agreement ({agree}) runs after the device work ({device})"
    );
}

#[test]
fn the_head_resolves_its_pack_slot_inside_each_arm() {
    // The S3c scratch defect: lm_head resolved XfFinal BEFORE the layout match, so every Native
    // run refused ("the plan has no slot XfFinal") although the Native arm never reads it. The
    // pinned body must carry the Tiled pack lookup inside the tiled arm and no pre-match lookup.
    let body = pinned_text(fn_section("fn lm_head"));
    let length = body.len();
    assert!(
        !body.contains("let (xf, y) ="),
        "lm_head ({length} bytes) resolves the pack before the layout match"
    );
    assert!(
        body.contains("xf: self.view(Slot::XfFinal)?"),
        "lm_head ({length} bytes) does not read the tiled pack inside the tiled arm"
    );
    assert!(
        !body[..body.find("match self.globals.head").expect("the match")].contains("XfFinal"),
        "lm_head ({length} bytes) names the tiled pack before the match"
    );
}

/// The defect-2 WRITER pin (C4's F4): the plan-side killer catches a missing FinalNormRow
/// slot; this pins the WRITER through the same seam, so a step.rs-only regression (the native
/// final norm writing Slot::NormRow again) fails a committed CPU test too.
#[test]
fn the_native_final_norm_writes_its_own_scratch_row() {
    let body = pinned_text(fn_section("fn final_norm"));
    let length = body.len();
    assert!(
        body.contains("norm_out: self.view(Slot::FinalNormRow)?"),
        "final_norm ({length} bytes) does not write the plan's own FinalNormRow slot"
    );
    assert!(
        !body.contains("norm_out: self.view(Slot::NormRow)?"),
        "final_norm ({length} bytes) writes NormRow (the scratch-defect-2 form)"
    );
}

#[test]
fn the_head_launches_write_every_logits_row() {
    // m19 (S3c): a head launch that skips its last rows leaves logits unwritten that the
    // sampler can read (13 of the last 256 ids fall inside its 248,077-wide range), and on
    // the qualification prompts no such id reaches the top-k, so neither the tokens nor rule C
    // see it. Both head arms launch the head's full row count, and neither trims it.
    let body = pinned_text(fn_section("fn lm_head("));
    let length = body.len();
    let native = body
        .find("HeadWeight::Native(head)")
        .expect("lm_head has a native arm");
    assert_eq!(
        body.matches("rows: head.n_rows(),").count(),
        2,
        "lm_head ({length} bytes): both head arms launch head.n_rows() rows"
    );
    assert!(
        body[native..].contains("rows: head.n_rows(),"),
        "lm_head ({length} bytes): the native head launch writes every row"
    );
    assert!(
        !body.contains("head.n_rows() -") && !body.contains("head.n_rows()-"),
        "lm_head ({length} bytes): a head arm trims its row count"
    );
}

#[test]
fn new_validates_the_step_inputs_before_it_touches_the_context() {
    let body = pinned_text(fn_section("pub(crate) fn new("));
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
    let arm = pinned_text(arm_section("StepOp::Argmax => {"));
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
    let section = fn_section("fn run_token_inner");
    let body = pinned_text(section);
    let length = body.len();
    assert!(
        body.contains("decide(pair, self.widths.verdict)"),
        "the run_token body ({length} bytes) does not call decide(pair, self.widths.verdict)"
    );
    // The decision line comes from the RAW section (comments stripped,
    // newlines kept) because pinned_text normalizes every newline away;
    // the tiered readback plans legitimately size their buffers with
    // shape.vocab on other lines, but the decision line itself must still
    // read only the sampler width.
    let raw = code_only(section);
    let decide = raw
        .find("decide(pair, self.widths.verdict)")
        .expect("the call was just asserted on the normalized body");
    let line = raw[decide..]
        .split_once('\n')
        .map_or(&raw[decide..], |(l, _)| l);
    for wrong in ["n_rows()", "globals.head", "shape.vocab"] {
        assert!(
            !line.contains(wrong),
            "the decide line ({line:?}) reads {wrong} instead of the sampler width"
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
    let body = pinned_text(fn_section("fn run_token_inner"));
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
    let body = pinned_text(fn_section("fn run_token_inner"));
    let length = body.len();
    let check = body
        .find("check_token_id(token, self.widths.verdict)?")
        .unwrap_or_else(|| {
            panic!(
                "the run_token body ({length} bytes) does not call \
                 check_token_id(token, self.widths.verdict)?"
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
    let body = pinned_text(fn_section("fn run_token_inner"));
    let length = body.len();
    let check = body
        .find("check_token_id(token, self.widths.verdict)?")
        .unwrap_or_else(|| {
            panic!(
                "the run_token body ({length} bytes) does not call \
                 check_token_id(token, self.widths.verdict)?"
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
        .find("check_token_id(token, self.widths.verdict)?")
        .unwrap_or_else(|| {
            panic!(
                "the write_start body ({length} bytes) does not call \
                 check_token_id(token, self.widths.verdict)?"
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

/// The text of the `QwenResidentStep` struct item of the merged `step.rs`.
fn struct_section(name: &str) -> &'static str {
    let source = include_str!("step.rs");
    let at = source
        .find(name)
        .unwrap_or_else(|| panic!("step.rs has no {name:?}"));
    let line_start = source[..at].rfind('\n').map_or(0, |nl| nl + 1);
    let mut end = source.len();
    let mut line = source[at..]
        .find('\n')
        .map_or(source.len(), |nl| at + nl + 1);
    while line < source.len() {
        let next = source[line..]
            .find('\n')
            .map_or(source.len(), |nl| line + nl + 1);
        let text = source[line..next].trim();
        if text == "}" {
            end = line;
            break;
        }
        line = next;
    }
    &source[line_start..end]
}

#[test]
fn the_graph_field_is_declared_before_the_state() {
    // D1: the drop-order coverage of the state buffers. The `graph` field (which keeps the
    // arena holds and borrows the pin slice) must drop before the `state` field that owns the
    // `kv`, `delta` and `words` buffers whose addresses the graph bakes in.
    let body = pinned_text(struct_section("struct QwenResidentStep"));
    let length = body.len();
    let graph = body
        .find("graph: Option<GraphExec")
        .unwrap_or_else(|| panic!("the struct ({length} bytes) has no graph field"));
    let state = body
        .find("state: ResidentState")
        .unwrap_or_else(|| panic!("the struct ({length} bytes) has no state field"));
    assert!(
        graph < state,
        "the graph field must be declared before the state field (graph at {graph}, state at          {state}, {length} bytes)"
    );
}

#[test]
fn next_mode_decides_every_combination() {
    // All sixteen combinations of the pure dispatch rule: Replay iff a graph exists; Capture
    // iff none exists and the step is warmed, asked for the capture and has not already
    // refused one (the sticky guard); Eager otherwise.
    for graph_ready in [false, true] {
        for warmed in [false, true] {
            for capture_on in [false, true] {
                for refused in [false, true] {
                    let mode = next_mode(graph_ready, warmed, capture_on, refused);
                    let want = if graph_ready {
                        Mode::Replay
                    } else if warmed && capture_on && !refused {
                        Mode::Capture
                    } else {
                        Mode::Eager
                    };
                    assert_eq!(
                        std::mem::discriminant(&mode),
                        std::mem::discriminant(&want),
                        "graph {graph_ready} warmed {warmed} on {capture_on} refused {refused}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_sticky_refusal_keeps_reporting_the_fallback_outcome() {
    // All SIX (outcome, refused) combinations: an eager token after a refusal keeps reporting
    // the fallback, so the landed one-read gate FAILs the silent fallback with no new field;
    // a captured token reports Captured whatever the flag holds (a refusal never follows a
    // capture on the same step).
    for (ran, refused, want) in [
        (HowRan::Eager, false, CaptureOutcome::Eager),
        (HowRan::Eager, true, CaptureOutcome::RefusedFellBackToEager),
        (HowRan::Captured, false, CaptureOutcome::Captured),
        (HowRan::Captured, true, CaptureOutcome::Captured),
        (
            HowRan::Refused,
            false,
            CaptureOutcome::RefusedFellBackToEager,
        ),
        (
            HowRan::Refused,
            true,
            CaptureOutcome::RefusedFellBackToEager,
        ),
    ] {
        assert_eq!(
            reported_outcome(ran, refused),
            want,
            "{ran:?} with refused {refused}"
        );
    }
}

#[test]
fn a_succeeded_capture_hands_the_graph_on_and_touches_nothing() {
    let (mut entered, mut refused) = (7, false);
    let folded = fold_capture(Ok((42_u32, ())), |_| true, &mut entered, &mut refused);
    assert!(matches!(folded, Ok(Some(42))), "the graph must come back");
    assert_eq!(entered, 7, "a succeeded capture keeps the op count");
    assert!(!refused, "a succeeded capture is not a refusal");
}

#[test]
fn a_refused_capture_resets_the_count_sets_the_sticky_flag_and_asks_for_the_rerun() {
    let (mut entered, mut refused) = (5, false);
    let error = crate::EngError::BadRequest("refused".to_owned());
    let folded = fold_capture::<u32>(Err(error), |_| true, &mut entered, &mut refused);
    assert!(
        matches!(folded, Ok(None)),
        "a refusal asks for the eager rerun"
    );
    assert_eq!(entered, 0, "the failure rule must judge the rerun alone");
    assert!(refused, "the refusal is sticky");
}

#[test]
fn any_other_capture_error_passes_through_unchanged_and_touches_nothing() {
    let (mut entered, mut refused) = (5, false);
    let error = crate::EngError::BadRequest("other".to_owned());
    let folded = fold_capture::<u32>(Err(error), |_| false, &mut entered, &mut refused);
    assert!(
        matches!(&folded, Err(crate::EngError::BadRequest(text)) if text == "other"),
        "the error must come back as it was"
    );
    assert_eq!(
        entered, 5,
        "a non-refusal keeps the op count for the failure rule"
    );
    assert!(!refused, "a non-refusal never sets the sticky flag");
}

#[test]
fn capture_once_folds_the_attempt_through_the_seam_and_reruns_on_none() {
    // The wiring: the real predicate and the step's own flag go into the seam, `?` hands any
    // other error to the caller, and only the `None` arm reruns the body and reports Refused.
    let body = pinned_text(fn_section("fn capture_once"));
    let length = body.len();
    for needle in [
        "fold_capture( attempt, crate::kernels::cuda::is_capture_refusal, entered, &mut self.refused, )?",
        "Some(graph) => { self.graph = Some(graph); self.replay()?; Ok(HowRan::Captured) }",
        "None => { self.run_body(admit_pos, entered, None)?; Ok(HowRan::Refused) }",
    ] {
        assert!(
            body.contains(needle),
            "the capture_once body ({length} bytes) lacks {needle:?}"
        );
    }
}

#[test]
fn tiered_step_refuses_capture_with_a_typed_error() {
    use super::step::tiered_capture_refusal;
    assert!(
        tiered_capture_refusal(false, true).is_ok(),
        "a plain step may capture"
    );
    assert!(
        tiered_capture_refusal(true, false).is_ok(),
        "a tiered step runs eager"
    );
    match tiered_capture_refusal(true, true) {
        Err(crate::EngError::BadRequest(message)) => {
            assert!(message.contains("eager only"), "{message}");
            assert!(message.contains("amendment of 2026-10-13"), "{message}");
        }
        other => panic!("tiered with capture must be refused: {other:?}"),
    }
}
