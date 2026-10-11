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
