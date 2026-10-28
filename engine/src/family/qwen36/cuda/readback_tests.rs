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
fn device_shortlist_readback_has_one_wait_and_no_vocab_copy() {
    let plan = ReadbackPlan::shortlist(VOCAB).expect("compact plan");
    assert_eq!(plan.logits_bytes, None);
    assert!(plan.shortlist);
    assert_eq!(
        plan.steps(),
        [
            ReadStep::QueueShortlist,
            ReadStep::QueuePairCopy,
            ReadStep::RecordEvent,
            ReadStep::ReadShortlist,
            ReadStep::ReadPair,
        ]
    );
    assert_eq!(plan.blocking_waits(), 1);
    let report = captured_report(&plan);
    assert_eq!((report.syncs, report.host_writes), (1, 1));
    assert!(ReadbackPlan::shortlist(0).is_err());
}

#[test]
fn device_shortlist_defaults_on_with_an_explicit_kill_switch() {
    use super::readback::device_shortlist_enabled;
    assert!(device_shortlist_enabled(None));
    assert!(device_shortlist_enabled(Some("1")));
    for value in [Some(""), Some("0"), Some("bad")] {
        assert!(!device_shortlist_enabled(value));
    }
}

#[test]
fn greedy_and_bounded_top_k_use_the_device_sampler() {
    use super::readback::device_sampling_supported;
    use crate::family::sampler::{Sampler, SamplingParams};
    assert!(device_sampling_supported(
        Sampler::greedy().params(),
        true,
        true
    ));
    for top_k in 1..=20 {
        for top_p in [0.0, 0.95, 1.0] {
            let params = SamplingParams::new(1.0, top_p, top_k).unwrap();
            assert!(device_sampling_supported(params, false, true));
        }
    }
}

#[test]
fn open_and_oversized_top_k_fall_back_to_host() {
    use super::readback::device_sampling_supported;
    use crate::family::sampler::SamplingParams;
    for top_k in [0, 21, u32::MAX] {
        let params = SamplingParams::new(1.0, 0.95, top_k).unwrap();
        assert!(!device_sampling_supported(params, false, true));
    }
}

#[test]
fn debug_sync_non_greedy_sampling_falls_back_to_host() {
    use super::readback::device_sampling_supported;
    use crate::family::sampler::SamplingParams;
    for top_k in [1, 7, 20] {
        let params = SamplingParams::new(1.0, 0.95, top_k).unwrap();
        assert!(!device_sampling_supported(params, true, true));
    }
}

#[test]
fn vocabulary_mismatch_falls_back_before_running_a_token() {
    use super::readback::device_sampling_supported;
    use crate::family::sampler::SamplingParams;
    for params in [
        SamplingParams::greedy(),
        SamplingParams::new(1.0, 0.95, 20).unwrap(),
    ] {
        assert!(!device_sampling_supported(params, false, false));
    }
}

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
