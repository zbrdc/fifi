//! Unit tests of `StateBook` (test seam U13, approved by the user 2026-10-10). CPU only.
use super::book::{Admitted, CommitRefusal, Health, StartRefusal, StateBook, status_error};
use crate::kernels::cuda::PREFILL_STATUS_BOUND_VIOLATION;
use crate::kernels::cuda::resident::step_params::MAX_ADVANCE;
use crate::kernels::cuda::resident::step_params::StepParams;

const OWNER: u64 = 7;

fn valid_book(capacity: usize) -> StateBook {
    let mut book = StateBook::new(capacity, OWNER);
    assert_eq!(
        book.health(),
        Health::Poisoned,
        "a new book is untrusted until the first reset"
    );
    assert!(book.needs_start(), "and owes a start block");
    book.finish_overwrite(0);
    book.params_written();
    book
}

fn text(error: crate::EngError) -> String {
    format!("{error:?}")
}

#[test]
fn enqueue_mirrors_pos_and_stops_at_the_capacity() {
    let mut book = valid_book(12);
    book.finish_overwrite(10);
    book.params_written();
    assert_eq!(book.enqueue(1).expect("first"), Admitted { admit_pos: 10 });
    assert_eq!(book.enqueue(1).expect("second").admit_pos, 11);
    assert!(text(book.enqueue(1).expect_err("third overflows")).contains("rows"));
    assert_eq!(book.pending_rows(), 2);
    assert!(book.enqueue(0).is_err(), "a zero advance is refused");
    let mut roomy = valid_book(1 << 20);
    assert!(
        roomy.enqueue(MAX_ADVANCE + 1).is_err(),
        "an advance above the kernel limit is refused even when the cache has room"
    );
    roomy.enqueue(MAX_ADVANCE).expect("the limit itself");
}

#[test]
fn enqueue_waits_for_the_start_block() {
    let mut book = StateBook::new(64, OWNER);
    book.finish_overwrite(0);
    assert!(text(book.enqueue(1).expect_err("owed")).contains("write_step_params"));
    book.params_written();
    book.enqueue(1).expect("after the start block");
}

#[test]
fn a_partial_accept_poisons_because_the_step_already_ran() {
    let mut book = valid_book(64);
    book.enqueue(2).expect("window");
    let refusal = book.commit(1, 0).expect_err("partial");
    assert_eq!(
        refusal,
        CommitRefusal::Partial {
            advance: 2,
            accepted: 1
        }
    );
    assert!(text(refusal.into()).contains("RD-7c"));
    assert_eq!(book.health(), Health::Poisoned, "the step cannot be undone");
    assert_eq!(
        (book.pending_rows(), book.tokens()),
        (2, 0),
        "nothing was retired"
    );
    assert_eq!(
        book.commit(2, 0),
        Err(CommitRefusal::NotValid),
        "and the whole commit is refused afterwards"
    );
    let mut more = valid_book(64);
    more.enqueue(2).expect("window");
    assert!(
        more.commit(3, 0).is_err(),
        "more than was enqueued is refused too"
    );
    assert_eq!(more.health(), Health::Poisoned);
    let mut whole = valid_book(64);
    whole.enqueue(2).expect("window");
    whole.commit(2, 0).expect("the whole step");
    assert_eq!(
        (whole.tokens(), whole.pending_rows(), whole.health()),
        (2, 0, Health::Valid)
    );
    assert_eq!(whole.commit(1, 0), Err(CommitRefusal::NothingPending));
}

#[test]
fn a_status_report_poisons_and_retires_nothing() {
    // Any non-zero bit poisons in stage 1 (REPLY-76 Q2): the masks alone, combined, and unknown bits.
    for status in [1, 2, 4, 8, 5, 12, 16, 3, 0x7fff_ffff, -1] {
        let mut book = valid_book(64);
        book.enqueue(1).expect("step");
        assert_eq!(
            book.commit(1, status),
            Err(CommitRefusal::Status(status)),
            "status {status}"
        );
        assert_eq!(book.health(), Health::Poisoned, "status {status}");
        assert_eq!(
            (book.pending_rows(), book.tokens()),
            (1, 0),
            "status {status}"
        );
    }
}

#[test]
fn a_status_refusal_keeps_the_class_and_the_precedence_of_the_wrapper() {
    let class = |status: i32| match status_error(status) {
        crate::EngError::Backend { operation, .. } => operation,
        other => panic!("not a Backend error: {other:?}"),
    };
    assert_eq!(class(4 | 1), "step_params", "4 beats 1");
    assert_eq!(class(4 | 8), "step_params", "4 beats 8");
    assert_eq!(class(8 | 1 | 2), "prefill_chunk", "8 beats 1");
    assert_eq!(class(1 | 2), "route_experts", "1 beats 2");
    assert_eq!(class(2), "sample");
    assert_eq!(
        class(16),
        "step_status",
        "unknown bits alone are still a Backend error"
    );
    let error: crate::EngError = CommitRefusal::Status(4).into();
    assert!(matches!(
        error,
        crate::EngError::Backend {
            operation: "step_params",
            ..
        }
    ));
}

/// The kernels' prefill mask — [`PREFILL_STATUS_BOUND_VIOLATION`], the one Rust mirror of
/// `prefill/status.h` — maps through `status_error` the way mask 8 must: alone it is a
/// `prefill_chunk` fault, it beats mask 1, and mask 4 beats it (decision 0049 item 10, lane
/// C ruling 2026-10-14: "bit 8" in the plan and decision text means mask 8). Without this
/// pin a kernel writing 0x100 would read back as `step_status`, and 0x101 as `route_experts`.
#[test]
fn the_kernels_prefill_mask_maps_to_prefill_chunk() {
    let class = |status: i32| match status_error(status) {
        crate::EngError::Backend { operation, .. } => operation,
        other => panic!("not a Backend error: {other:?}"),
    };
    assert_eq!(
        class(PREFILL_STATUS_BOUND_VIOLATION),
        "prefill_chunk",
        "mask 8 (0x8) of the prefill status word is a prefill chunk fault"
    );
    assert_eq!(
        class(PREFILL_STATUS_BOUND_VIOLATION | 0x1),
        "prefill_chunk",
        "8 beats 1"
    );
    assert_eq!(
        class(PREFILL_STATUS_BOUND_VIOLATION | 0x4),
        "step_params",
        "4 beats 8"
    );
}

#[test]
fn commit_and_enqueue_refuse_a_poisoned_state_without_retiring() {
    let mut book = valid_book(64);
    book.enqueue(1).expect("step");
    book.poison();
    assert!(book.commit(1, 0).is_err());
    assert_eq!(
        book.pending_rows(),
        1,
        "the refused commit did not retire the step"
    );
    assert!(book.enqueue(1).is_err());
}

#[test]
fn a_cancelled_enqueue_restores_the_mirror() {
    let mut book = valid_book(64);
    book.enqueue(1).expect("first");
    book.enqueue(1).expect("second");
    book.cancel_last().expect("cancel");
    assert_eq!(book.enqueue(1).expect("again").admit_pos, 1);
    book.cancel_last().expect("cancel");
    book.cancel_last().expect("cancel");
    assert!(book.cancel_last().is_err(), "nothing left to take back");
}

#[test]
fn overwrite_clears_pending_sets_the_count_and_owes_a_start_block() {
    let mut book = valid_book(64);
    book.enqueue(1).expect("step");
    book.begin_reset(OWNER).expect("reset");
    assert_eq!(book.health(), Health::Poisoned);
    book.finish_overwrite(9);
    assert_eq!(
        (book.tokens(), book.pending_rows(), book.health()),
        (9, 0, Health::Valid)
    );
    assert!(book.needs_start());
}

#[test]
fn every_guard_checks_the_owning_stream() {
    let mut book = valid_book(64);
    assert!(text(book.begin_block(8).expect_err("block")).contains("owned by stream 7"));
    assert!(book.begin_export(8).is_err());
    assert!(book.begin_reset(8).is_err());
    assert!(book.begin_import(8, true).is_err());
    assert!(book.begin_poison_device(8).is_err());
    assert!(book.begin_join(8).is_err());
    book.begin_join(OWNER).expect("owner");
    assert_eq!(
        book.health(),
        Health::Valid,
        "a wrong-stream caller poisons nothing"
    );
    book.begin_block(OWNER).expect("owner");
    book.begin_export(OWNER).expect("owner");
    book.begin_poison_device(OWNER).expect("owner");
    assert_eq!(book.health(), Health::Poisoned);
}

#[test]
fn blocks_and_exports_need_a_valid_state_and_exports_an_idle_one() {
    let mut book = valid_book(64);
    book.enqueue(1).expect("step");
    book.begin_block(OWNER)
        .expect("a block runs with work in flight");
    assert!(text(book.begin_export(OWNER).expect_err("busy")).contains("export_state"));
    book.commit(1, 0).expect("commit");
    book.begin_export(OWNER).expect("idle again");
    book.poison();
    assert!(book.begin_block(OWNER).is_err());
    assert!(book.begin_export(OWNER).is_err());
}

#[test]
fn only_a_prefill_in_progress_can_be_revalidated() {
    let mut book = valid_book(64);
    assert!(
        book.revalidate(5, 0).is_err(),
        "a valid state is not revalidated"
    );
    book.begin_prefill().expect("begin");
    assert_eq!(book.health(), Health::PrefillOpen);
    assert!(book.enqueue(1).is_err(), "no decode step during a prefill");
    book.revalidate(5, 0).expect("clean chunk");
    assert_eq!((book.health(), book.tokens()), (Health::Valid, 5));
    assert!(
        book.needs_start(),
        "the first decode token needs its start block"
    );
    book.params_written();
    assert!(
        book.revalidate(5, 0).is_err(),
        "and it is not revalidated twice"
    );
    book.begin_prefill().expect("begin");
    assert!(book.revalidate(65, 0).is_err(), "above the capacity");
}

#[test]
fn a_defect_poison_survives_revalidate_until_reset() {
    let mut book = valid_book(64);
    book.begin_prefill().expect("begin");
    book.poison();
    assert!(
        book.revalidate(3, 0).is_err(),
        "REPLY-47: poisoned until reset"
    );
    assert_eq!(book.health(), Health::Poisoned);
    assert!(book.begin_prefill().is_err());
    book.begin_reset(OWNER).expect("reset");
    book.finish_overwrite(0);
    assert_eq!(book.health(), Health::Valid);
}

#[test]
fn a_prefill_status_word_poisons() {
    let mut book = valid_book(64);
    book.begin_prefill().expect("begin");
    assert!(matches!(
        book.revalidate(3, 8),
        Err(crate::EngError::Backend {
            operation: "prefill_chunk",
            ..
        })
    ));
    assert_eq!(book.health(), Health::Poisoned);
    assert!(book.begin_revalidate(8).is_err());
    assert!(
        book.begin_revalidate(OWNER).is_err(),
        "and a poisoned state is not revalidated, whoever owns it"
    );
}

#[test]
fn the_prefill_word_copies_need_an_open_chunk_on_the_owning_stream() {
    let mut book = valid_book(64);
    assert!(
        book.begin_prefill_word(OWNER, "read_prefill_status")
            .is_err(),
        "a valid state has no open chunk, so the word is not read"
    );
    book.begin_prefill().expect("open");
    // With the chunk open, the owner check is the only guard this call can fail, so the row
    // reaches it (cuda-guard-tests-must-reach-their-guard).
    assert!(
        text(
            book.begin_prefill_word(OWNER + 1, "clear_prefill_status")
                .expect_err("another stream")
        )
        .contains("owned by stream 7, not stream 8"),
        "the wrong-stream refusal names the owner, not just the caller"
    );
    book.begin_prefill_word(OWNER, "clear_prefill_status")
        .expect("the word is zeroed at chunk open");
    book.begin_prefill_word(OWNER, "read_prefill_status")
        .expect("and read at chunk close, still inside the chunk");
    book.poison();
    assert!(
        book.begin_prefill_word(OWNER, "read_prefill_status")
            .is_err(),
        "a poisoned state does not touch the word"
    );
}

#[test]
fn prefill_and_export_need_a_committed_boundary() {
    let mut book = valid_book(64);
    book.enqueue(1).expect("step");
    assert!(book.begin_prefill().is_err());
    book.commit(1, 0).expect("commit");
    book.begin_prefill().expect("idle");
}

#[test]
fn start_blocks_must_match_the_host_mirror() {
    let mut book = valid_book(64);
    book.finish_overwrite(4);
    book.check_start(&StepParams::new(11, 4).expect("block"))
        .expect("good");
    book.begin_write_params(OWNER, &StepParams::new(11, 4).expect("block"))
        .expect("owner");
    assert!(
        book.begin_write_params(8, &StepParams::new(11, 4).expect("block"))
            .is_err()
    );
    let base = StepParams::new(11, 4).expect("block");
    let cases = [
        (
            StepParams {
                pos: 5,
                kv_len: 6,
                kv_slot: 5,
                ..base
            },
            StartRefusal::PosMirror { pos: 5, tokens: 4 },
        ),
        (
            StepParams { kv_len: 4, ..base },
            StartRefusal::KvLen { kv_len: 4, pos: 4 },
        ),
        (
            StepParams { kv_slot: 3, ..base },
            StartRefusal::KvSlot { kv_slot: 3, pos: 4 },
        ),
        (StepParams { status: 2, ..base }, StartRefusal::Status(2)),
        (
            StepParams {
                token_id: -1,
                ..base
            },
            StartRefusal::Token(-1),
        ),
        (
            StepParams {
                pos: -1,
                kv_len: 0,
                kv_slot: -1,
                ..base
            },
            StartRefusal::PosRange(-1),
        ),
    ];
    for (params, expected) in cases {
        assert_eq!(book.check_start(&params), Err(expected));
    }
    book.params_written();
    book.enqueue(1).expect("step");
    assert_eq!(book.check_start(&base), Err(StartRefusal::NotIdle));
    book.poison();
    assert_eq!(book.check_start(&base), Err(StartRefusal::NotValid));
    let error: crate::EngError = StartRefusal::Status(2).into();
    assert!(text(error).contains("write_step_params"));
}

#[test]
fn a_poisoned_state_with_a_full_import_ends_valid_and_owes_a_start_block() {
    let mut book = valid_book(64);
    book.enqueue(1).expect("step");
    assert!(book.commit(1, 4).is_err());
    assert_eq!(book.health(), Health::Poisoned);
    book.begin_import(OWNER, true)
        .expect("an import that does all of reset's work");
    book.finish_overwrite(5);
    assert_eq!(
        (book.health(), book.tokens(), book.pending_rows()),
        (Health::Valid, 5, 0)
    );
    assert!(
        book.needs_start(),
        "the zeroed words are an invalid start block"
    );
    assert!(
        book.enqueue(1).is_err(),
        "so no step is admitted before write_step_params"
    );
}

#[test]
fn a_poisoned_state_is_not_imported_without_the_words_rewrite() {
    let mut book = valid_book(64);
    book.poison();
    let error = book.begin_import(OWNER, false).expect_err("refused");
    assert!(text(error).contains("reset first"));
    assert_eq!(book.health(), Health::Poisoned);
    book.begin_import(8, true)
        .expect_err("and another stream is refused as before");
    let mut valid = valid_book(64);
    valid
        .begin_import(OWNER, false)
        .expect("a valid state may be replaced either way");
}

#[test]
fn an_import_that_fails_after_the_check_leaves_the_state_poisoned() {
    let mut book = valid_book(64);
    book.enqueue(1).expect("step");
    book.begin_import(OWNER, true)
        .expect("check_import accepted the value");
    // A device failure here returns before finish_overwrite.
    assert_eq!(book.health(), Health::Poisoned);
    assert!(book.enqueue(1).is_err());
    assert_eq!(book.commit(1, 0), Err(CommitRefusal::NotValid));
    assert!(book.begin_export(OWNER).is_err());
    book.begin_import(OWNER, true).expect("a retry");
    book.finish_overwrite(3);
    assert_eq!((book.health(), book.tokens()), (Health::Valid, 3));
}

#[test]
fn revalidate_is_refused_by_the_book_before_anything_else_runs() {
    let mut book = valid_book(64);
    assert!(
        book.begin_revalidate(OWNER).is_err(),
        "a valid state has no prefill to revalidate"
    );
    assert_eq!(book.health(), Health::Valid, "the refusal changes nothing");
    book.begin_prefill().expect("open");
    assert!(book.begin_revalidate(OWNER + 1).is_err(), "another stream");
    book.begin_revalidate(OWNER)
        .expect("the owner of an open prefill");
    book.poison();
    assert!(
        book.begin_revalidate(OWNER).is_err(),
        "a poisoned state is not revalidated"
    );
}

#[test]
fn a_prefill_may_publish_the_capacity_and_not_one_more() {
    let mut book = valid_book(64);
    book.begin_prefill().expect("open");
    assert!(matches!(
        book.revalidate(65, 0),
        Err(crate::EngError::ContextLimit(_))
    ));
    assert_eq!(
        book.health(),
        Health::PrefillOpen,
        "a refused publish keeps the prefill open"
    );
    book.revalidate(64, 0).expect("the capacity itself");
    assert_eq!(book.tokens(), 64);
}

#[test]
fn the_start_position_is_bounded_by_pos_max() {
    use crate::kernels::cuda::resident::step_params::POS_MAX;
    let book = valid_book(64);
    let at = |pos: i32| StepParams {
        token_id: 0,
        pos,
        kv_len: pos.wrapping_add(1),
        kv_slot: pos,
        status: 0,
        pad: [0; 3],
    };
    assert!(matches!(
        book.check_start(&at(POS_MAX + 1)),
        Err(StartRefusal::PosRange(_))
    ));
    assert!(matches!(
        book.check_start(&at(-1)),
        Err(StartRefusal::PosRange(-1))
    ));
    // POS_MAX passes the range rule and then fails the mirror (the book holds 0 rows).
    assert!(matches!(
        book.check_start(&at(POS_MAX)),
        Err(StartRefusal::PosMirror { .. })
    ));
}
