//! Unit tests of the run report. CPU only.
use super::run_report::{CaptureOutcome, PrefillReport, RunReport};
use crate::{
    EngError,
    family::session::{
        DecodeSession, PrefixSlot, SaveKind, ShadowLcp, ShadowReason, StoreLimits, TokenStepper,
    },
    sched::job::{PrefixHit, PrefixReport},
};

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

/// The storage core of the prefill report: the records it keeps, the two row counts, and the
/// clear.
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

/// The fixture report: two pieces of 128 and 1 tokens, which feed 127 and 1 rows, so one chunk of
/// 127 rows and one one-row feed.
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

/// The job line of the fixture report.
#[test]
fn the_job_line_names_the_pieces_and_the_steps() {
    let line = fixture_report().job_line();
    assert_eq!(
        line,
        "q36 cuda: prefill job m_sess 128 m_sess_cap 2048 path chunk verdict Ready pieces 2 piece_tokens [128, 1] calls [127, 1] steps [Chunk(0, 127), Feed(127)] rows 127+1 piece_ns [10, 20] chunk_ns [5]"
    );
}

/// The once-per-job guard: Some once, then None until a refill, then Some again.
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

#[test]
fn c1_reset_clears_lookup_facts_without_requiring_another_restore() {
    use crate::family::session::{ShadowLcp, ShadowReason};
    use crate::sched::job::{PrefixHit, PrefixReport};

    let mut report = super::run_report::PrefillReport::default();
    report.prefix_lookup = PrefixReport {
        hit: PrefixHit::Tip,
        uncached_shared: 23,
        shadow: Some(ShadowLcp {
            block_lcp_128: 256,
            block_lcp_256: 256,
            usable_checkpoint_tokens: 257,
            divergence_offset: 280,
            reason: ShadowReason::SameAsActual,
            token_comparisons: 280,
        }),
        ..PrefixReport::default()
    };
    report.push_step(feed(257, 7));
    assert_eq!(report.prefix_lookup.hit, PrefixHit::Tip);
    assert!(report.prefix_lookup.shadow.is_some());

    report.clear();

    assert_eq!(
        report.prefix_lookup,
        PrefixReport::default(),
        "a no-boundary or cancelled next job must not inherit lookup facts"
    );
}

#[derive(Default)]
struct PrefixHostStep {
    fed: usize,
    report: PrefillReport,
}

impl TokenStepper for PrefixHostStep {
    const OPERATION: &'static str = "c1-host-prefix";

    fn feed(&mut self, _token: u32) -> Result<(), EngError> {
        self.fed += 1;
        Ok(())
    }

    fn logits(&mut self, _token: u32) -> Result<Vec<f32>, EngError> {
        unreachable!("prefix fixture never decodes")
    }

    fn fed(&self) -> usize {
        self.fed
    }

    fn reset(&mut self) -> Result<(), EngError> {
        self.fed = 0;
        refill_prefix_report(&mut self.report);
        Ok(())
    }
}

type HostPrefixSlot = PrefixSlot<usize, u8>;

fn refill_prefix_report(report: &mut PrefillReport) {
    report.refill(
        2048,
        0,
        0,
        0,
        super::dims::KvMode::Fp16,
        crate::family::project::cuda::WeightLayout::Native,
        false,
    );
}

fn save_host_prefix(slot: &mut HostPrefixSlot, key: &[u32], kind: SaveKind) {
    let mut session = DecodeSession::default();
    let mut step = PrefixHostStep::default();
    session.prefill(&mut step, key, 2048).unwrap();
    assert!(
        session
            .save_prefix(slot, key, 0, step.fed, kind, || Ok(step.fed))
            .unwrap()
    );
}

fn restore_observed(
    report: &mut PrefillReport,
    slot: &mut HostPrefixSlot,
    prompt: &[u32],
    path: u8,
) -> usize {
    let covered = DecodeSession::default()
        .restore_prefix(slot, prompt, path, |_| {
            refill_prefix_report(report);
            Ok(())
        })
        .unwrap();
    report.observe_prefix_lookup(slot, prompt, covered);
    covered
}

#[test]
fn c1_cold_warm_save_and_replacement_keep_lookup_facts_and_shadow_flag() {
    let key: Vec<_> = (0..513).collect();
    let prompt: Vec<_> = (0..514).collect();
    for enabled in [false, true] {
        let mut slot = HostPrefixSlot::default();
        slot.set_shadow_enabled(enabled);
        let mut report = PrefillReport::default();
        assert_eq!(restore_observed(&mut report, &mut slot, &prompt, 0), 0);
        assert_eq!(
            report.prefix_lookup.shadow,
            enabled.then(ShadowLcp::default)
        );
        let cold = report.prefix_lookup;
        save_host_prefix(&mut slot, &key, SaveKind::Tip);
        assert_eq!(report.prefix_lookup, cold, "saving cannot rewrite a lookup");
        assert_eq!(restore_observed(&mut report, &mut slot, &prompt, 0), 513);
        assert_eq!(report.prefix_lookup.hit, PrefixHit::Tip);
        assert_eq!(report.prefix_lookup.uncached_shared, 0);
        if enabled {
            let shadow = report.prefix_lookup.shadow.unwrap();
            assert_eq!((shadow.block_lcp_128, shadow.block_lcp_256), (512, 512));
            assert_eq!(
                (shadow.usable_checkpoint_tokens, shadow.divergence_offset),
                (513, 513)
            );
            assert_eq!(shadow.reason, ShadowReason::SameAsActual);
        } else {
            assert_eq!(report.prefix_lookup.shadow, None);
        }
        let warm = report.prefix_lookup;
        slot = HostPrefixSlot::with_limits(slot.limits());
        slot.set_shadow_enabled(enabled);
        assert_eq!(
            report.prefix_lookup, warm,
            "replacement cannot erase job facts"
        );
        assert_eq!(restore_observed(&mut report, &mut slot, &prompt, 0), 0);
        assert_eq!(report.prefix_lookup.hit, PrefixHit::Miss);
        assert_eq!(
            report.prefix_lookup.shadow,
            enabled.then(ShadowLcp::default)
        );
    }
}

#[test]
fn c1_warm_then_divergent_miss_freezes_the_gap_across_save() {
    let key: Vec<_> = (0..513).collect();
    let mut prompt: Vec<_> = (0..514).collect();
    let mut slot = HostPrefixSlot::default();
    slot.set_shadow_enabled(true);
    save_host_prefix(&mut slot, &key, SaveKind::Tip);
    let mut report = PrefillReport::default();
    assert_eq!(restore_observed(&mut report, &mut slot, &prompt, 0), 513);
    prompt[400] = u32::MAX;
    assert_eq!(restore_observed(&mut report, &mut slot, &prompt, 0), 0);
    assert_eq!(report.prefix_lookup.hit, PrefixHit::Miss);
    assert_eq!(report.prefix_lookup.uncached_shared, 400);
    let shadow = report.prefix_lookup.shadow.unwrap();
    assert_eq!((shadow.block_lcp_128, shadow.block_lcp_256), (384, 256));
    assert_eq!(
        (shadow.usable_checkpoint_tokens, shadow.divergence_offset),
        (0, 400)
    );
    assert_eq!(shadow.reason, ShadowReason::NoUsableCheckpoint);
    let observed = report.prefix_lookup;
    save_host_prefix(&mut slot, &prompt, SaveKind::Tip);
    assert_eq!(report.prefix_lookup, observed);
    assert_eq!(
        slot.uncached_shared(&prompt, 0),
        514,
        "store now differs from the lookup"
    );
}

#[test]
fn c1_equal_length_and_wrong_path_are_observed_without_changing_reuse_rules() {
    let key: Vec<_> = (0..513).collect();
    let extension: Vec<_> = (0..514).collect();
    for (prompt, path, reason) in [
        (key.as_slice(), 0, ShadowReason::QNotBelowPrompt),
        (extension.as_slice(), 1, ShadowReason::PathMismatch),
    ] {
        let mut slot = HostPrefixSlot::default();
        slot.set_shadow_enabled(true);
        save_host_prefix(&mut slot, &key, SaveKind::Tip);
        let mut report = PrefillReport::default();
        assert_eq!(restore_observed(&mut report, &mut slot, prompt, path), 0);
        assert_eq!(report.prefix_lookup.hit, PrefixHit::Miss);
        assert_eq!(report.prefix_lookup.uncached_shared, 513);
        assert_eq!(report.prefix_lookup.shadow.unwrap().reason, reason);
    }
}

#[test]
fn c1_session_reset_and_refill_clear_facts_without_another_lookup_or_feed() {
    let key: Vec<_> = (0..513).collect();
    let prompt: Vec<_> = (0..514).collect();
    let mut slot = HostPrefixSlot::default();
    slot.set_shadow_enabled(true);
    save_host_prefix(&mut slot, &key, SaveKind::Tip);
    let mut step = PrefixHostStep::default();
    assert_eq!(
        restore_observed(&mut step.report, &mut slot, &prompt, 0),
        513
    );
    DecodeSession::default().reset(&mut step).unwrap();
    assert_eq!(step.report.prefix_lookup, PrefixReport::default());
    assert_eq!(step.fed, 0);
    assert!(step.report.steps().is_empty());
    assert_eq!(
        restore_observed(&mut step.report, &mut slot, &prompt, 0),
        513
    );
    refill_prefix_report(&mut step.report);
    assert_eq!(step.report.prefix_lookup, PrefixReport::default());
    assert!(step.report.steps().is_empty());
}

#[test]
fn c1_denied_tip_fallback_preserves_shadow_but_reports_the_actual_root() {
    let key: Vec<_> = (0..513).collect();
    let prompt: Vec<_> = (0..514).collect();
    let mut slot = HostPrefixSlot::with_limits(StoreLimits {
        roots: 1,
        entries: 2,
    });
    slot.set_shadow_enabled(true);
    save_host_prefix(&mut slot, &key[..129], SaveKind::Root);
    save_host_prefix(&mut slot, &key, SaveKind::Tip);
    assert_eq!((slot.roots_len(), slot.entries_len()), (1, 1));
    let mut report = PrefillReport::default();
    let covered = DecodeSession::default()
        .restore_prefix(&mut slot, &prompt, 0, |&fed| {
            if fed == 512 {
                return Err(EngError::MemoryPlan("denied tip".into()));
            }
            refill_prefix_report(&mut report);
            Ok(())
        })
        .unwrap();
    report.observe_prefix_lookup(&slot, &prompt, covered);
    assert_eq!(covered, 129);
    assert_eq!(report.prefix_lookup.hit, PrefixHit::Root);
    assert_eq!(report.prefix_lookup.uncached_shared, 0);
    assert_eq!(
        report
            .prefix_lookup
            .shadow
            .unwrap()
            .usable_checkpoint_tokens,
        513
    );
    assert_eq!((slot.roots_len(), slot.entries_len()), (1, 0));
}

#[test]
fn c1_all_denied_is_a_miss_and_fatal_restore_remains_an_error() {
    let key: Vec<_> = (0..513).collect();
    let prompt: Vec<_> = (0..514).collect();
    let mut slot = HostPrefixSlot::default();
    slot.set_shadow_enabled(true);
    save_host_prefix(&mut slot, &key, SaveKind::Tip);
    let mut report = PrefillReport::default();
    let covered = DecodeSession::default()
        .restore_prefix(&mut slot, &prompt, 0, |_| {
            Err(EngError::MemoryPlan("denied all".into()))
        })
        .unwrap();
    report.observe_prefix_lookup(&slot, &prompt, covered);
    assert_eq!(covered, 0);
    assert_eq!(report.prefix_lookup.hit, PrefixHit::Miss);
    assert_eq!(
        report
            .prefix_lookup
            .shadow
            .unwrap()
            .usable_checkpoint_tokens,
        513
    );
    assert_eq!(slot.entries_len(), 0);

    save_host_prefix(&mut slot, &key, SaveKind::Tip);
    refill_prefix_report(&mut report);
    let error = || EngError::Backend {
        operation: "c1-test",
        message: "restore failed".into(),
    };
    let result = DecodeSession::default().restore_prefix(&mut slot, &prompt, 0, |_| Err(error()));
    assert_eq!(result, Err(error()));
    assert_eq!(
        report.prefix_lookup,
        PrefixReport::default(),
        "failed import is not a hit"
    );
    assert_eq!(slot.entries_len(), 1);
}
