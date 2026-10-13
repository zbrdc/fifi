//! Unit tests of the read-back plan and the mode switches (test seam U13). CPU only.
use super::{
    readback::{
        ReadStep, ReadbackPlan, captured_report, check_mode, eager_report, parse_sync_each,
        refused_report, with_host_syncs,
    },
    run_report::{CaptureOutcome, HostSyncs},
};

/// The Qwen3.6 head width. The test pins the byte count; the code derives it from the shape.
const VOCAB: usize = 248_320;

#[test]
fn sync_each_reads_zero_one_or_nothing() {
    for off in [None, Some(""), Some("0")] {
        assert!(!parse_sync_each(off).expect("off"), "{off:?}");
    }
    assert!(parse_sync_each(Some("1")).expect("on"));
    for bad in ["2", "yes", "true", " 1", "01"] {
        assert!(parse_sync_each(Some(bad)).is_err(), "{bad:?}");
    }
}

#[test]
fn the_debug_mode_and_a_capture_exclude_each_other() {
    assert!(check_mode(true, true).is_err());
    for (sync_each, capture) in [(false, false), (true, false), (false, true)] {
        assert!(
            check_mode(sync_each, capture).is_ok(),
            "{sync_each} {capture}"
        );
    }
}

#[test]
fn the_read_back_queues_and_records_before_any_blocking_read() {
    let with = ReadbackPlan::new(VOCAB, true).expect("plan");
    assert_eq!(
        with.steps(),
        [
            ReadStep::QueuePairCopy,
            ReadStep::RecordEvent,
            ReadStep::ReadLogits,
            ReadStep::ReadPair
        ]
    );
    let without = ReadbackPlan::new(VOCAB, false).expect("plan");
    assert_eq!(
        without.steps(),
        [
            ReadStep::QueuePairCopy,
            ReadStep::RecordEvent,
            ReadStep::ReadPair
        ]
    );
    assert_eq!((with.blocking_waits(), without.blocking_waits()), (1, 1));
}

#[test]
fn the_logits_row_is_four_bytes_a_value_and_checked() {
    let plan = ReadbackPlan::new(VOCAB, true).expect("plan");
    assert_eq!(plan.logits_bytes, Some(993_280));
    assert_eq!(
        ReadbackPlan::new(VOCAB, false).expect("plan").logits_bytes,
        None
    );
    assert!(ReadbackPlan::new(0, true).is_err());
    assert!(ReadbackPlan::new(usize::MAX / 2, true).is_err());
}

#[test]
fn an_eager_token_reports_one_wait_and_one_start_block_write() {
    let report = eager_report(&ReadbackPlan::new(VOCAB, true).expect("plan"));
    assert_eq!(report.capture, CaptureOutcome::Eager);
    assert_eq!((report.syncs, report.host_writes), (1, 1));
}

#[test]
fn captured_report_sets_the_outcome_and_the_waits() {
    for want_logits in [false, true] {
        let plan = ReadbackPlan::new(VOCAB, want_logits).expect("plan");
        let report = captured_report(&plan);
        assert_eq!(report.capture, CaptureOutcome::Captured, "{want_logits}");
        assert_eq!((report.syncs, report.host_writes), (1, 1), "{want_logits}");
    }
}

#[test]
fn refused_report_sets_the_fallback_outcome() {
    for want_logits in [false, true] {
        let plan = ReadbackPlan::new(VOCAB, want_logits).expect("plan");
        let report = refused_report(&plan);
        assert_eq!(
            report.capture,
            CaptureOutcome::RefusedFellBackToEager,
            "{want_logits}"
        );
        assert_eq!((report.syncs, report.host_writes), (1, 1), "{want_logits}");
    }
}

#[test]
fn a_tiered_report_counts_every_layer_wait_and_blocking_write_as_a_sync() {
    // 40 layers, each with the route event wait, the two plan writes and the CPU partial write,
    // and 2 bank fills of 3 writes at the token boundary, on top of the read-back wait and the
    // start-block write of the plain report.
    let plain = eager_report(&ReadbackPlan::new(VOCAB, false).expect("plan"));
    assert_eq!((plain.syncs, plain.host_writes), (1, 1));
    let mut extra = HostSyncs::default();
    for _ in 0..40 {
        extra.add(HostSyncs {
            waits: 1,
            writes: 3,
        });
    }
    extra.add(HostSyncs {
        waits: 0,
        writes: 6,
    });
    assert_eq!(
        extra,
        HostSyncs {
            waits: 40,
            writes: 126
        }
    );
    let tiered = with_host_syncs(plain, extra);
    assert_eq!(tiered.syncs, 1 + 40 + 126);
    assert_eq!(tiered.host_writes, 1 + 126);
    assert_eq!(tiered.capture, plain.capture, "the outcome is not touched");
    assert_eq!(
        with_host_syncs(plain, HostSyncs::default()),
        plain,
        "a plain step adds nothing"
    );
}
