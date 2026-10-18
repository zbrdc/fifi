//! Unit tests of the run report (test seam U13). CPU only.
use super::run_report::{CaptureOutcome, RunReport};

const ALL: [CaptureOutcome; 3] = [
    CaptureOutcome::Eager,
    CaptureOutcome::Captured,
    CaptureOutcome::RefusedFellBackToEager,
];

#[test]
fn the_default_outcome_is_eager() {
    assert_eq!(CaptureOutcome::default(), CaptureOutcome::Eager);
    let report = RunReport::default();
    assert_eq!(report.capture, CaptureOutcome::Eager);
    assert!(!report.capture.captured());
    assert_eq!((report.syncs, report.host_writes), (0, 0));
}

#[test]
fn only_a_captured_run_counts_as_captured() {
    for outcome in ALL {
        assert_eq!(
            outcome.captured(),
            outcome == CaptureOutcome::Captured,
            "{outcome:?}"
        );
    }
}

#[test]
fn the_labels_are_distinct_and_stable() {
    let labels = ALL.map(CaptureOutcome::label);
    assert_eq!(labels, ["eager", "captured", "refused_fell_back_to_eager"]);
}

fn chunk(p0: usize, rows: usize, ns: u64, status: i32) -> super::run_report::StepRecord {
    super::run_report::StepRecord {
        p0,
        rows,
        kind: super::run_report::StepKind::Chunk,
        ns,
        status,
    }
}

fn feed(p0: usize, ns: u64) -> super::run_report::StepRecord {
    super::run_report::StepRecord {
        p0,
        rows: 1,
        kind: super::run_report::StepKind::Feed,
        ns,
        status: 0,
    }
}

/// H-B10, storage core (CPF-7b-C): the records it keeps, the two row counts, and the clear.
#[test]
fn the_prefill_report_accumulates_until_a_reset() {
    let mut report = super::run_report::PrefillReport::default();
    report.set_start_rows(64);
    report.push_step(chunk(64, 127, 11, 0));
    report.push_step(feed(191, 7));
    report.push_step(chunk(192, 1, 5, 3));

    assert_eq!(
        report.steps(),
        &[chunk(64, 127, 11, 0), feed(191, 7), chunk(192, 1, 5, 3)],
        "the records keep their order and their fields"
    );
    assert_eq!(
        report.n_prefilled(),
        128,
        "chunk rows only, including a one-row chunk"
    );
    assert_eq!(report.n_fed(), 1, "feed rows are counted apart");
    assert_eq!(
        report.steps()[2].status,
        3,
        "a non-zero status word is recorded"
    );

    report.clear();

    assert!(report.steps().is_empty(), "a clear empties the steps");
    assert_eq!(report.n_prefilled(), 0);
    assert_eq!(report.n_fed(), 0);
    assert_eq!(
        report.start_rows(),
        64,
        "start_rows is kept for the driver next epoch"
    );
}

/// The fixture report of CPF-7b-C B.8, H-B12: two pieces of 128 and 1 tokens, which feed 127 and
/// 1 rows, so one chunk of 127 rows and one one-row feed.
fn fixture_report() -> super::run_report::PrefillReport {
    use super::run_report::{StepKind, StepRecord};
    use super::step::ChunkVerdict;

    let mut report = super::run_report::PrefillReport::default();
    report.refill(
        2048,
        741_200_000,
        65_536,
        800_000_000,
        super::dims::KvMode::Fp16,
        crate::family::project::cuda::WeightLayout::Native,
        false,
    );
    report.note_feed_rows(&ChunkVerdict::Ready { m_sess: 128 }, 127);
    report.note_feed_rows(&ChunkVerdict::Ready { m_sess: 128 }, 1);
    report.record_piece(128, 10);
    report.record_piece(1, 20);
    report.push_step(StepRecord {
        p0: 0,
        rows: 127,
        kind: StepKind::Chunk,
        ns: 5,
        status: 0,
    });
    report.push_step(StepRecord {
        p0: 127,
        rows: 1,
        kind: StepKind::Feed,
        ns: 7,
        status: 0,
    });
    report
}

/// H-B12, the job line of the B.8 fixture (condition 9).
#[test]
fn the_job_line_names_the_pieces_and_the_steps() {
    let line = fixture_report().job_line();
    assert_eq!(
        line,
        "q36 cuda: prefill job m_sess 128 m_sess_cap 2048 path chunk verdict Ready pieces 2 piece_tokens [128, 1] calls [127, 1] steps [Chunk(0, 127), Feed(127)] rows 127+1 piece_ns [10, 20] chunk_ns [5]"
    );
}

/// H-B10, the once-per-job guard: Some once, then None until a refill, then Some again.
#[test]
fn take_job_line_returns_once_until_a_reset() {
    let mut report = fixture_report();
    assert!(
        report.take_job_line().is_some(),
        "the first take writes the line"
    );
    assert!(
        report.take_job_line().is_none(),
        "the second take of one job writes nothing"
    );
    report.refill(
        1024,
        0,
        0,
        0,
        super::dims::KvMode::Fp16,
        crate::family::project::cuda::WeightLayout::Native,
        false,
    );
    assert!(
        report.take_job_line().is_some(),
        "a reset allows the next job its own line"
    );
    assert!(report.steps().is_empty(), "the refill emptied the steps");
}
