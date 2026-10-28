//! Portable worker tests. The row fake owns independent scripted state, not a model.

use super::super::{
    Engine, EngineNote, EngineState, ReadyNote, SubmitError, lock, spawn_engine, spawn_engine_noted,
};
use crate::EngError;
use crate::exec_fake::{FakeExecutor, FakeSession};
use crate::family::executor::{ExecutorOps, GenerationState};
use crate::family::sampler::{Sampler, SamplingParams};
use crate::family::session::{SaveKind, StoreLimits};
use crate::sched::job::{Job, JobEvent, JobHandle, PrefixHit, PrefixReport};
use crate::tokenizer::{DecodeBatch, DecodeFinish, IncrementalDecoder, Tokenizer};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

const JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1,"c":2},"merges":[]}}"#;
const EOS: u32 = 9;
const WAIT: Duration = Duration::from_secs(10);

type Gate = (mpsc::Sender<()>, mpsc::Receiver<()>);
type Log = Arc<Mutex<Trace>>;

#[derive(Default)]
struct Trace {
    calls: Vec<String>,
    windows: Vec<Vec<Vec<u32>>>,
    draws: Vec<(u32, u32)>,
    /// Every save point that fired: (key length, kind). The calls trace keeps its
    /// kind-less `save:N` shape; the kind assertion reads this instead (S4).
    saves: Vec<(usize, SaveKind)>,
}

struct Row {
    fake: FakeExecutor,
    state: FakeSession,
}

impl Row {
    fn new(tokens: usize) -> Self {
        let mut script = vec![vec![0]; tokens];
        script.push(vec![EOS]);
        Self {
            fake: FakeExecutor::new(script, &[EOS], 10_000),
            state: FakeSession::default(),
        }
    }
}

#[derive(Clone, Copy)]
enum Failure {
    Native,
    Local,
    RowLocal,
    RowNative,
    Panic,
    Count,
}

struct WindowFake {
    max: usize,
    script_tokens: usize,
    context_limit: Option<usize>,
    log: Log,
    hold: Option<Gate>,
    failure: Option<Failure>,
    deny_session: bool,
    /// The fake's pool-derived ceiling (fix EV-2 on the batch path): `None` means no
    /// budget to report, like the default executor.
    ceiling: Option<u64>,
    fail_after_emit: bool,
    draws: bool,
    report: PrefixReport,
    /// The number of `prefill` calls so far, for the one-shot piece denial below.
    prefill_calls: usize,
    /// When set, the `prefill` call at this index returns a memory-plan denial once
    /// (fix EV-1's restart test): a piece that fails after an earlier piece succeeded.
    fail_prefill_piece: Option<usize>,
    /// Denies every prefill piece whose tokens carry this marker token, up to the
    /// budget below (Sol round 2, SF3): a PER-JOB failure - the denied job restarts,
    /// denies again if the budget allows, and only the spent eviction budget makes it
    /// final, while other jobs' pieces prefill fine.
    deny_prefill_token: Option<u32>,
    /// The marker denial budget (usize::MAX denies every carrying piece, the exhaustion
    /// path; a small budget lets the restart succeed).
    deny_prefill_token_budget: usize,
    /// Evictions handed out so far: the fake's ceiling grows with each, modeling a pool
    /// whose honest prompt ceiling rises as entries free (round 4c, R4-MF2).
    evicted: u64,
    deny_all_prefill: bool,
    /// How many evictions the fake can still free (review round 4, MF1): `Some(0)` is
    /// honest progress - an entry went, zero bytes were charged to this fake - and the
    /// count runs out, after which `None` means the store is empty and a denial is
    /// final. The default of 8 is more than any single request should need.
    evictions: usize,
    /// When set, the `reset` call at this index returns this error once (review round 2,
    /// MF2): a native failure inside the restart of a denied slot (reset 0 belongs to the
    /// admission, reset 1 to the restart).
    fail_reset_call: Option<(usize, EngError)>,
    /// The number of `reset` calls so far, for the indexed failure below.
    reset_calls: usize,
    /// The pinned block prefix-match spacing (S4): `None` keeps every save-point
    /// walk on the no-block path regardless of the flipped environment default.
    spacing: Option<usize>,
}

impl WindowFake {
    fn new(log: &Log) -> Self {
        Self {
            max: 2,
            script_tokens: 1_100,
            context_limit: None,
            log: Arc::clone(log),
            hold: None,
            failure: None,
            deny_session: false,
            ceiling: None,
            fail_after_emit: false,
            draws: false,
            report: PrefixReport::default(),
            prefill_calls: 0,
            fail_prefill_piece: None,
            deny_all_prefill: false,
            deny_prefill_token: None,
            deny_prefill_token_budget: usize::MAX,
            evicted: 0,
            evictions: 8,
            fail_reset_call: None,
            reset_calls: 0,
            spacing: None,
        }
    }

    fn note(&self, call: &str) {
        lock(&self.log).calls.push(call.to_string());
    }

    fn row_decode(
        &mut self,
        row: &mut Row,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch, EngError> {
        if self.draws && !generation.cancel.load(Ordering::SeqCst) {
            let id = row.fake.history(&row.state)[0];
            let sampled = generation.sampler.sample(&[0.0, 0.0, 0.0])?;
            lock(&self.log).draws.push((id, sampled));
        }
        row.fake.decode(&mut row.state, generation)
    }
}

impl ExecutorOps for WindowFake {
    type Session = Row;

    fn max_seqs(&self) -> usize {
        self.max
    }

    fn new_session(&mut self) -> Result<Row, EngError> {
        self.note("new_session");
        if std::mem::take(&mut self.deny_session) {
            return Err(EngError::MemoryPlan("session admission denied".into()));
        }
        Ok(Row::new(self.script_tokens))
    }

    fn check_prompt(&self, tokens: usize) -> Result<(), EngError> {
        if let Some(limit) = self.context_limit {
            crate::family::session::check_prompt(tokens, limit)?;
        }
        Ok(())
    }

    fn evict_one_prefix_entry(&mut self, _protect: Option<u64>) -> Option<u64> {
        if self.evictions == 0 {
            return None;
        }
        self.evictions -= 1;
        self.evicted += 1;
        self.note("evict");
        Some(0)
    }

    fn pool_prompt_ceiling(&self, _row: &Row) -> Option<u64> {
        // Two tokens of honest ceiling per freed entry: a deficit of k entries needs
        // k evictions before the ceiling admits (round 4c, R4-MF2).
        self.ceiling.map(|base| base + 2 * self.evicted)
    }

    fn history<'s>(&self, row: &'s Row) -> &'s [u32] {
        row.fake.history(&row.state)
    }

    fn reset(&mut self, row: &mut Row) -> Result<(), EngError> {
        self.note("reset");
        let call = self.reset_calls;
        self.reset_calls += 1;
        if let Some((at, _)) = self.fail_reset_call.as_ref()
            && *at == call
            && let Some((_, error)) = self.fail_reset_call.take()
        {
            return Err(error);
        }
        self.report.hit = PrefixHit::Miss;
        self.report.uncached_shared = 0;
        row.fake.reset(&mut row.state)
    }

    fn boundary(&mut self, row: &mut Row) {
        self.note("boundary");
        row.fake.boundary(&mut row.state);
    }

    fn prefill(&mut self, row: &mut Row, tokens: &[u32]) -> Result<(), EngError> {
        self.note(&format!("prefill:{}", tokens.len()));
        let call = self.prefill_calls;
        self.prefill_calls += 1;
        if self
            .deny_prefill_token
            .is_some_and(|marker| tokens.contains(&marker))
            && self.deny_prefill_token_budget > 0
        {
            self.deny_prefill_token_budget -= 1;
            return Err(EngError::MemoryPlan("piece denied".into()));
        }
        if self.deny_all_prefill || self.fail_prefill_piece == Some(call) {
            self.fail_prefill_piece = None;
            return Err(EngError::MemoryPlan("piece denied".into()));
        }
        row.fake.prefill(&mut row.state, tokens)
    }

    fn decode(
        &mut self,
        row: &mut Row,
        generation: &mut GenerationState,
    ) -> Result<DecodeBatch, EngError> {
        self.note("decode");
        self.row_decode(row, generation)
    }

    fn decode_window(
        &mut self,
        rows: &mut [&mut Row],
        generations: &mut [&mut GenerationState],
    ) -> Result<Vec<Result<DecodeBatch, EngError>>, EngError> {
        self.note("window");
        lock(&self.log).windows.push(
            rows.iter()
                .map(|row| row.fake.history(&row.state).to_vec())
                .collect(),
        );
        if let Some((entered, release)) = self.hold.take() {
            entered.send(()).expect("signal window");
            release.recv_timeout(WAIT).expect("release window");
        }
        let failure = self.failure.take();
        let mut batches = Vec::new();
        for (slot, (row, generation)) in rows.iter_mut().zip(generations).enumerate() {
            if slot == 1 && matches!(failure, Some(Failure::RowLocal | Failure::RowNative)) {
                batches.push(Err(if matches!(failure, Some(Failure::RowNative)) {
                    native_error()
                } else {
                    EngError::MemoryPlan("row growth denied".into())
                }));
                continue;
            }
            batches.push(Ok(self.row_decode(row, generation)?));
            // Deliberately fail after one row consumed a token. Nothing may be published.
            match failure {
                Some(Failure::Native) => return Err(native_error()),
                Some(Failure::Local) => {
                    return Err(EngError::MemoryPlan("window growth denied".into()));
                }
                Some(Failure::Panic) => panic!("window panic"),
                Some(Failure::Count | Failure::RowLocal | Failure::RowNative) | None => {}
            }
        }
        if matches!(failure, Some(Failure::Count)) {
            batches.pop();
        }
        Ok(batches)
    }

    fn after_emit(&mut self) -> Result<(), EngError> {
        self.note("after_emit");
        if self.fail_after_emit {
            Err(native_error())
        } else {
            Ok(())
        }
    }

    fn prefix_limits(&self) -> StoreLimits {
        StoreLimits {
            roots: 1,
            entries: 3,
        }
    }

    fn prefix_block_spacing(&self) -> Option<usize> {
        // Pinned per test (S4): the fake never reads the environment, so the
        // batch save-point walk runs with a controlled spacing, not the
        // flipped default.
        self.spacing
    }

    fn restore_prefix(&mut self, _row: &mut Row, prompt: &[u32]) -> Result<usize, EngError> {
        // Deliberately executor-global restore metadata, like the real Qwen executor.
        self.report.hit = if prompt[0] == 0 {
            PrefixHit::Root
        } else {
            PrefixHit::Tip
        };
        self.report.uncached_shared = prompt[0];
        Ok(0)
    }

    fn save_prefix(&mut self, _row: &Row, key: &[u32], kind: SaveKind) -> Result<bool, EngError> {
        self.note(&format!("save:{}", key.len()));
        lock(&self.log).saves.push((key.len(), kind));
        self.report.entries += 1;
        self.report.bytes += 100;
        Ok(true)
    }

    fn prefix_report(&self) -> Option<PrefixReport> {
        Some(self.report)
    }
}

fn native_error() -> EngError {
    EngError::Backend {
        operation: "window_fake",
        message: "native failure".into(),
    }
}

fn generation(limit: u32) -> GenerationState {
    let tokenizer = Tokenizer::from_bytes(JSON.as_bytes()).expect("tokenizer");
    GenerationState::new(IncrementalDecoder::new(&tokenizer)).with_max_tokens(limit)
}

fn job(prompt: &[u32], limit: u32) -> (Job, JobHandle) {
    Job::new(prompt.to_vec(), generation(limit)).expect("job")
}

fn gate() -> (Gate, mpsc::Receiver<()>, mpsc::Sender<()>) {
    let (entered, observed) = mpsc::channel();
    let (release, held) = mpsc::channel();
    ((entered, held), observed, release)
}

/// Hold factory completion until all initial jobs have entered the bounded queue.
fn start(fake: WindowFake) -> (Engine, mpsc::Sender<()>) {
    let (release, held) = mpsc::channel();
    let engine = spawn_engine(4, move || {
        held.recv_timeout(WAIT).expect("release load");
        let session = Row::new(fake.script_tokens);
        Ok((fake, session))
    })
    .expect("spawn");
    (engine, release)
}

fn collect(mut handle: JobHandle) -> Vec<JobEvent> {
    let deadline = Instant::now() + WAIT;
    let mut events = Vec::new();
    loop {
        match handle.events.try_recv() {
            Ok(event) => events.push(event),
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => return events,
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                assert!(Instant::now() < deadline, "worker did not close the job");
                std::thread::yield_now();
            }
        }
    }
}

fn wait(condition: impl Fn() -> bool) {
    let deadline = Instant::now() + WAIT;
    while !condition() {
        assert!(
            Instant::now() < deadline,
            "worker did not reach the condition"
        );
        std::thread::yield_now();
    }
}

fn usage(events: &[JobEvent]) -> crate::sched::job::JobUsage {
    match events.last() {
        Some(JobEvent::Finished(usage)) => *usage,
        other => panic!("missing usage: {other:?}"),
    }
}

fn output(events: &[JobEvent]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            JobEvent::Batch(batch) => Some(batch.text.as_str()),
            _ => None,
        })
        .collect()
}

fn finish(events: &[JobEvent]) -> DecodeFinish {
    events
        .iter()
        .find_map(|event| match event {
            JobEvent::Batch(batch) => batch.finish,
            _ => None,
        })
        .expect("terminal batch")
}

#[test]
fn simultaneous_jobs_share_one_window_but_not_streams_limits_or_usage() {
    let log = Log::default();
    let (engine, release) = start(WindowFake::new(&log));
    let (first, first_events) = job(&[0, 0], 1);
    let (second, second_events) = job(&[1, 1, 1], 2);
    engine.handle().try_submit(first).expect("first");
    engine.handle().try_submit(second).expect("second");
    release.send(()).expect("load");
    let first = collect(first_events);
    let second = collect(second_events);
    assert_eq!(output(&first), "a");
    assert_eq!(output(&second), "aa");
    assert_eq!(
        (usage(&first).prompt_tokens, usage(&first).completion_tokens),
        (2, 1)
    );
    assert_eq!(
        (
            usage(&second).prompt_tokens,
            usage(&second).completion_tokens
        ),
        (3, 2)
    );
    assert_eq!(finish(&first), DecodeFinish::Length);
    assert_eq!(finish(&second), DecodeFinish::Length);
    let trace = lock(&log);
    assert_eq!(
        trace.windows,
        vec![vec![vec![0, 0], vec![1, 1, 1]], vec![vec![1, 1, 1, 0]]]
    );
    assert!(!trace.calls.iter().any(|call| call == "decode"));
    drop(trace);
    assert_eq!(engine.shutdown().state, EngineState::Stopped);
}

#[test]
fn a_denied_prefill_piece_restarts_the_slot_from_a_reset_session() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    // Piece 0 (4 tokens) succeeds; piece 1 is denied after it, which is the case the
    // restart exists for: re-feeding the denied piece would corrupt the partially fed
    // session, so the slot must be rebuilt from a reset session instead.
    fake.fail_prefill_piece = Some(1);
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[7, 7, 7, 7, 7], 1);
    engine.handle().try_submit(first).expect("submit");
    release.send(()).expect("load");
    let events = collect(first_events);
    assert_eq!(
        usage(&events).completion_tokens,
        1,
        "the job runs: {events:?}"
    );
    assert_eq!(usage(&events).prompt_tokens, 5, "the whole prompt re-ran");
    let trace = lock(&log);
    assert_eq!(
        trace.calls,
        vec![
            // The seed session: piece 0 succeeds, piece 1 is denied.
            "reset",
            "boundary",
            "prefill:4",
            "prefill:1",
            // The prefill denial evicts once before the restart rebuilds the slot from the
            // reset session and re-runs the whole prompt - both pieces - never the denied
            // piece alone.
            "evict",
            "reset",
            "boundary",
            "prefill:4",
            "prefill:1",
            "window",
            "after_emit",
        ],
        "the denied piece restarts from reset, boundary, and the full prompt"
    );
    engine.shutdown();
}

#[test]
fn a_batch_block_save_fires_once_and_the_restart_saves_nothing() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    // Pinned spacing (S4): the fake overrides the environment, so this batch case
    // runs the block path with a controlled spacing instead of the flipped default.
    // With spacing 1,500 and the tip at 20,000, the periodic multiple 16,384 fires
    // first and the last block boundary below the tip is 19,500 - a distinct Block
    // point, so both kinds must fire exactly once.
    fake.spacing = Some(1_500);
    // Piece 4,900 (tokens 19,600..19,604) is denied AFTER the block point at 19,500
    // saved and BEFORE the tip at 20,000: the restart must rebuild the slot with no
    // save points (retried slots never save), so the block entry is cut exactly once.
    fake.fail_prefill_piece = Some(4_900);
    let (engine, release) = start(fake);
    let prompt = vec![7u32; 20_001];
    let (first, first_events) = job(&prompt, 1);
    engine
        .handle()
        .try_submit(
            first
                .with_prefix_boundary(20_000)
                .expect("tip inside the prompt"),
        )
        .expect("submit");
    release.send(()).expect("load");
    let events = collect(first_events);
    assert_eq!(
        usage(&events).completion_tokens,
        1,
        "the job runs: {events:?}"
    );
    assert_eq!(
        usage(&events).prompt_tokens,
        20_001,
        "the whole prompt re-ran"
    );
    let trace = lock(&log);
    assert_eq!(
        trace.saves,
        vec![(16_384, SaveKind::Periodic), (19_500, SaveKind::Block)],
        "the periodic point and the pinned block point fire once; the restart saves nothing"
    );
    assert_eq!(
        trace.calls.iter().filter(|call| **call == "reset").count(),
        2,
        "the denial restarts the slot from a reset session"
    );
    assert_eq!(
        trace.calls.iter().filter(|call| **call == "evict").count(),
        1,
        "the prefill denial evicts once before the restart"
    );
    drop(trace);
    engine.shutdown();
}

#[test]
fn a_restarted_slot_never_saves_a_snapshot() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    // Piece 1 (the 1-token tail after the tip point 4) is denied. The first attempt
    // saved the tip at its point; the restart must admit the whole prompt with no
    // snapshot, because a snapshot that charges can starve the tail that follows it
    // (review round 2, MF1).
    fake.fail_prefill_piece = Some(1);
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[7, 7, 7, 7, 7], 1);
    engine
        .handle()
        .try_submit(first.with_prefix_boundary(4).expect("tip"))
        .expect("submit");
    release.send(()).expect("load");
    let events = collect(first_events);
    assert_eq!(
        usage(&events).completion_tokens,
        1,
        "the job runs: {events:?}"
    );
    let trace = lock(&log);
    let saves = trace
        .calls
        .iter()
        .filter(|call| call.starts_with("save:"))
        .count();
    assert_eq!(saves, 1, "only the first attempt saved: {:?}", trace.calls);
    let last_reset = trace
        .calls
        .iter()
        .rposition(|call| call == "reset")
        .expect("restart");
    assert!(
        !trace.calls[last_reset..]
            .iter()
            .any(|call| call.starts_with("save:")),
        "the restart saved nothing: {:?}",
        trace.calls
    );
    engine.shutdown();
}

#[test]
fn a_native_failure_during_the_restart_fans_out_as_the_backend_error() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    // Piece 0 is denied, which routes the slot into the restart; the restart's reset
    // then fails natively. The failure - not the denied prefill's memory-plan error -
    // must reach the job and the engine state (review round 2, MF2).
    fake.fail_prefill_piece = Some(0);
    fake.fail_reset_call = Some((
        1,
        EngError::Backend {
            operation: "reset",
            message: "the device is gone".into(),
        },
    ));
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[7, 7, 7, 7], 1);
    engine.handle().try_submit(first).expect("submit");
    release.send(()).expect("load");
    let events = collect(first_events);
    assert!(
        matches!(events.last(), Some(JobEvent::Error(EngError::Backend { operation, message }))
            if *operation == "reset" && message == "the device is gone"),
        "the job sees the backend failure, not the memory plan: {events:?}"
    );
    let report = engine.shutdown();
    assert_eq!(report.state, EngineState::NativeFailed);
    let failure = report.failure.clone();
    assert!(
        failure.as_deref().is_some_and(
            |cause| cause.contains("the device is gone") && cause.contains("native failure")
        ),
        "the recorded cause names the backend failure: {failure:?}"
    );
    // No executor call follows the failed reset, and the failed reset is the last call
    // of all: the engine stops exactly there, it does not decode or reset again.
    let trace = lock(&log);
    let failed_reset = trace
        .calls
        .iter()
        .rposition(|call| call == "reset")
        .expect("reset");
    assert_eq!(
        trace.calls.last(),
        Some(&"reset".to_string()),
        "nothing follows the failed reset: {:?}",
        trace.calls
    );
    assert_eq!(
        trace.calls.iter().filter(|call| *call == "reset").count(),
        2,
        "exactly the first reset and the failed one: {:?}",
        trace.calls
    );
    assert!(
        !trace.calls[failed_reset + 1..]
            .iter()
            .any(|call| call.starts_with("prefill") || call.starts_with("window")),
        "the engine made no executor call after the native failure: {:?}",
        trace.calls
    );
}

#[test]
fn single_sequence_setting_never_uses_window_or_new_session() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.max = 1;
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[0], 1);
    let (second, second_events) = job(&[1], 1);
    engine.handle().try_submit(first).expect("first");
    engine.handle().try_submit(second).expect("second");
    release.send(()).expect("load");
    assert_eq!(usage(&collect(first_events)).completion_tokens, 1);
    assert_eq!(usage(&collect(second_events)).completion_tokens, 1);
    let trace = lock(&log);
    assert!(trace.windows.is_empty());
    assert_eq!(
        trace
            .calls
            .iter()
            .filter(|call| call.as_str() == "decode")
            .count(),
        2
    );
    assert!(!trace.calls.iter().any(|call| call == "new_session"));
    drop(trace);
    engine.shutdown();
}

#[test]
fn prefill_uses_existing_pieces_and_never_adds_prompt_rows_to_a_window() {
    let log = Log::default();
    let (engine, release) = start(WindowFake::new(&log));
    let (short, short_events) = job(&[0], 3);
    let (long, long_events) = job(&[1; 10], 1);
    engine.handle().try_submit(short).expect("short");
    engine.handle().try_submit(long).expect("long");
    release.send(()).expect("load");
    collect(short_events);
    collect(long_events);
    let trace = lock(&log);
    assert_eq!(
        trace.windows.iter().map(Vec::len).collect::<Vec<_>>(),
        [1, 1, 2]
    );
    assert_eq!(trace.windows[2][1], vec![1; 10]);
    assert_eq!(
        trace
            .calls
            .iter()
            .filter(|call| call.starts_with("prefill:"))
            .cloned()
            .collect::<Vec<_>>(),
        ["prefill:1", "prefill:4", "prefill:4", "prefill:2"]
    );
    drop(trace);
    engine.shutdown();
}

#[test]
fn a_cancelled_row_finishes_and_a_queued_newcomer_joins_the_next_window() {
    let log = Log::default();
    let (hold, entered, unblock) = gate();
    let mut fake = WindowFake::new(&log);
    fake.hold = Some(hold);
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[0], 4);
    let (second, second_events) = job(&[1], 2);
    engine.handle().try_submit(first).expect("first");
    engine.handle().try_submit(second).expect("second");
    release.send(()).expect("load");
    entered.recv_timeout(WAIT).expect("entered");
    first_events.cancel();
    let (newcomer, newcomer_events) = job(&[2], 1);
    engine.handle().try_submit(newcomer).expect("newcomer");
    unblock.send(()).expect("window");
    let first = collect(first_events);
    assert_eq!(finish(&first), DecodeFinish::Cancelled);
    assert_eq!(usage(&first).completion_tokens, 0);
    assert_eq!(usage(&collect(second_events)).completion_tokens, 2);
    assert_eq!(usage(&collect(newcomer_events)).completion_tokens, 1);
    let trace = lock(&log);
    assert_eq!(trace.windows[1], vec![vec![1, 0], vec![2]]);
    drop(trace);
    engine.shutdown();
}

#[test]
fn prefix_restore_facts_are_captured_before_another_job_changes_them() {
    let log = Log::default();
    let (engine, release) = start(WindowFake::new(&log));
    let (first, first_events) = job(&[0; 9], 1);
    let (second, second_events) = job(&[1; 6], 1);
    engine
        .handle()
        .try_submit(first.with_prefix_boundary(8).expect("tip"))
        .expect("first");
    engine
        .handle()
        .try_submit(second.with_prefix_boundary(4).expect("tip"))
        .expect("second");
    release.send(()).expect("load");
    let first = usage(&collect(first_events)).prefix.expect("first report");
    let second = usage(&collect(second_events))
        .prefix
        .expect("second report");
    assert_eq!((first.hit, first.uncached_shared), (PrefixHit::Root, 0));
    assert_eq!((second.hit, second.uncached_shared), (PrefixHit::Tip, 1));
    assert!(first.entries > 0 && second.entries > 0);
    engine.shutdown();
}

#[test]
fn session_admission_denial_evicts_and_retries() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.deny_session = true;
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[0], 1);
    let (denied, denied_events) = job(&[1], 1);
    let (third, third_events) = job(&[2], 1);
    for request in [first, denied, third] {
        engine.handle().try_submit(request).expect("submit");
    }
    release.send(()).expect("load");
    assert_eq!(usage(&collect(first_events)).completion_tokens, 1);
    // Fix EV-1: the one-shot admission denial is retried after the eviction, so the job
    // runs instead of failing - a cached entry must not block session allocation.
    assert_eq!(usage(&collect(denied_events)).completion_tokens, 1);
    assert_eq!(usage(&collect(third_events)).completion_tokens, 1);
    // With max_seqs 2 the retried job takes the second slot of the first window; the third
    // job decodes in the next window.
    assert_eq!(lock(&log).windows[0], vec![vec![0], vec![1]]);
    assert_eq!(lock(&log).windows[1], vec![vec![2]]);
    engine.shutdown();
}

#[test]
fn a_retried_slots_later_denial_evicts_again_and_finishes() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    // One admission denial (spent on the retry that admits the job) and one prefill
    // denial in the retried slot: under the incremental contract (review round 4, MF1)
    // the later denial evicts one more entry and restarts the slot instead of erroring -
    // the first denial freed one entry, and the next may need only one more. The fake's
    // failure is one-shot, so the restarted prefill admits and the job finishes.
    fake.deny_session = true;
    fake.fail_prefill_piece = Some(1);
    let (engine, release) = start(fake);
    // The first job consumes the seed session, so the second job's admission is the one
    // denied and retried; prefill call 1 is that retried slot's first piece.
    let (first, first_events) = job(&[0], 1);
    let (denied, denied_events) = job(&[1], 1);
    for request in [first, denied] {
        engine.handle().try_submit(request).expect("submit");
    }
    release.send(()).expect("load");
    assert_eq!(usage(&collect(first_events)).completion_tokens, 1);
    assert_eq!(usage(&collect(denied_events)).completion_tokens, 1);
    let trace = lock(&log);
    assert_eq!(
        trace.calls.iter().filter(|call| *call == "evict").count(),
        2,
        "one eviction at the admission denial, one more at the retried slot's later denial"
    );
    drop(trace);
    engine.shutdown();
}

#[test]
fn an_admission_retry_saves_no_points() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.deny_session = true;
    let (engine, release) = start(fake);
    // The seed session goes to the plain job, so the boundary job's admission is the
    // denied-and-retried one. The retried slot must clear the points build_slot set up
    // (MF1's admission-retry half): a job whose prompt would otherwise hit its tip save
    // produces no save call at all.
    let (plain, plain_events) = job(&[0], 1);
    let (tip, tip_events) = job(&[1, 1, 1, 1, 1], 1);
    let tip = tip.with_prefix_boundary(4).expect("tip");
    for request in [plain, tip] {
        engine.handle().try_submit(request).expect("submit");
    }
    release.send(()).expect("load");
    assert_eq!(usage(&collect(plain_events)).completion_tokens, 1);
    assert_eq!(usage(&collect(tip_events)).completion_tokens, 1);
    let trace = lock(&log);
    assert!(
        !trace.calls.iter().any(|call| call.starts_with("save:")),
        "the admission-retried slot saved nothing: {:?}",
        trace.calls
    );
    drop(trace);
    engine.shutdown();
}

#[test]
fn a_final_batch_denial_reports_the_pools_honest_ceiling() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    // The admission denial spends an eviction and the retried slot's EVERY piece denies
    // (the marker token targets only that job, Sol round 2 SF3): the restart evicts one
    // more entry, the next denial finds the budget spent - `None` - and only then is the
    // refusal final, so fix EV-2 must classify it against the pool's honest ceiling
    // instead of leaking the raw memory-plan error. The other job prefills untouched.
    fake.deny_session = true;
    fake.deny_prefill_token = Some(1);
    fake.evictions = 2;
    fake.ceiling = Some(1);
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[0], 1);
    let (denied, denied_events) = job(&[1, 1, 1, 1, 1, 1], 1);
    for request in [first, denied] {
        engine.handle().try_submit(request).expect("submit");
    }
    release.send(()).expect("load");
    assert_eq!(usage(&collect(first_events)).completion_tokens, 1);
    // The pre-pass spends both entries (ceiling 1+2*2=5, still short of the 6-token
    // prompt) and hits `None`: the TERMINAL branch classifies immediately - no restart,
    // no re-feed (Sol round 3, SF2) - against the pool's honest ceiling (the base plus
    // the two freed entries) instead of leaking the raw memory-plan error.
    assert_eq!(
        collect(denied_events),
        vec![JobEvent::Error(EngError::PromptTooLong {
            prompt_tokens: 6,
            limit_tokens: 5,
        })]
    );
    let calls = lock(&log).calls.clone();
    let last_evict = calls
        .iter()
        .rposition(|call| call == "evict")
        .expect("evicted");
    assert!(
        !calls[last_evict + 1..].iter().any(|call| call == "reset"),
        "exhaustion is terminal: no dead restart after the last eviction: {calls:?}"
    );
    engine.shutdown();
}

#[test]
fn an_empty_store_denial_with_room_recovers_through_one_no_save_restart() {
    // Sol round 4, MF1: an exhausted store (no evictable entry) whose denial is
    // recoverable - a save charge the no-save retry avoids, with ceiling to spare -
    // must NOT be classified away by the terminal branch: the single no-save restart
    // finishes the prompt, exactly like the single worker's gated recovery.
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.fail_prefill_piece = Some(1);
    fake.evictions = 0;
    fake.ceiling = Some(100);
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[0], 1);
    let (denied, denied_events) = job(&[1, 1, 1, 1, 1], 1);
    for request in [first, denied] {
        engine.handle().try_submit(request).expect("submit");
    }
    release.send(()).expect("load");
    assert_eq!(usage(&collect(first_events)).completion_tokens, 1);
    assert_eq!(usage(&collect(denied_events)).prompt_tokens, 5);
    engine.shutdown();
}

#[test]
fn a_multi_entry_deficit_costs_one_batch_restart_not_k() {
    // Round 4c R4-MF2 + round 4d Sol-r3 SF1/SF2: the batch prefill restart must SIZE
    // the eviction first, like the single worker. A 3-entry deficit (the fake's ceiling
    // needs three freed entries before it admits the 6-token prompt), a ONE-piece
    // denial budget, and a SURPLUS eviction budget of five - the pre-pass stops the
    // moment the ceiling admits, so exactly three evictions happen and the two spare
    // entries stay. The note trace pins it: from the denied piece, exactly three
    // evicts, ONE reset, then the successful re-feed. A no-pre-pass mutant evicts once
    // before the reset; a drain-to-None mutant spends all five and never reaches a
    // reset; a setup-twice mutant shows a second reset before the re-feed.
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.deny_prefill_token = Some(1);
    fake.deny_prefill_token_budget = 1;
    fake.evictions = 5;
    fake.ceiling = Some(1);
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[0], 1);
    let (denied, denied_events) = job(&[1, 1, 1, 1, 1, 1], 1);
    for request in [first, denied] {
        engine.handle().try_submit(request).expect("submit");
    }
    release.send(()).expect("load");
    assert_eq!(usage(&collect(first_events)).completion_tokens, 1);
    assert_eq!(usage(&collect(denied_events)).prompt_tokens, 6);
    let calls = lock(&log).calls.clone();
    assert_eq!(
        calls.iter().filter(|call| *call == "evict").count(),
        3,
        "the sized pre-pass stops at the admitting ceiling; the surplus stays: {calls:?}"
    );
    let start_at = calls
        .iter()
        .position(|call| call == "prefill:4")
        .expect("the denied piece ran");
    // Three evicts, then the ONE restart's reset - the other slot's decode notes may
    // interleave after it, but no eviction may.
    assert_eq!(
        &calls[start_at..start_at + 5],
        ["prefill:4", "evict", "evict", "evict", "reset"],
        "three evicts size the restart, ONE restart follows: {calls:?}"
    );
    let refeed_at = calls[start_at + 5..]
        .iter()
        .position(|call| call == "prefill:4")
        .expect("the restart re-fed the piece")
        + start_at
        + 5;
    let between = &calls[start_at + 5..refeed_at];
    assert!(
        !between.iter().any(|call| call == "evict"),
        "no eviction between the restart and its re-feed: {calls:?}"
    );
    assert_eq!(
        between.iter().filter(|call| *call == "reset").count(),
        0,
        "exactly ONE restart: no second setup before the re-feed: {calls:?}"
    );
    engine.shutdown();
}

#[test]
fn a_persistent_denial_on_an_exhausted_store_takes_one_restart_then_classifies() {
    // Round 4e MF1 (Opus + Sol r5 MF1): a store already exhausted whose denial the
    // ceiling does NOT explain (the marker budget denies every carrying piece
    // forever, the ceiling 100 fits the 5-token prompt with room) must not restart
    // forever: the terminal branch takes the ONE no-save recovery, and the SECOND
    // exhausted denial classifies even though the ceiling fits - exactly one reset
    // after the denied piece, then the raw memory-plan error. The unbounded mutant
    // restarts once per tick, never closes the job, and fails collect's deadline.
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.deny_prefill_token = Some(1);
    fake.deny_prefill_token_budget = usize::MAX;
    fake.evictions = 0;
    fake.ceiling = Some(100);
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[0], 1);
    let (denied, denied_events) = job(&[1, 1, 1, 1, 1], 1);
    for request in [first, denied] {
        engine.handle().try_submit(request).expect("submit");
    }
    release.send(()).expect("load");
    assert_eq!(usage(&collect(first_events)).completion_tokens, 1);
    assert_eq!(
        collect(denied_events),
        vec![JobEvent::Error(EngError::MemoryPlan(
            "piece denied".to_string()
        ))]
    );
    let calls = lock(&log).calls.clone();
    // The two slots' admissions each reset once; the exhausted recovery adds exactly
    // ONE more. A restart-forever mutant adds one per spin and dies above at collect's
    // deadline, so this count pins the bound even under a partial fix.
    assert_eq!(
        calls.iter().filter(|call| *call == "reset").count(),
        3,
        "exactly one no-save recovery after exhaustion: {calls:?}"
    );
    assert_eq!(
        calls.iter().filter(|call| *call == "evict").count(),
        0,
        "the store was exhausted from the first denial: {calls:?}"
    );
    engine.shutdown();
}

#[test]
fn a_persistent_denial_without_a_ceiling_takes_one_restart_then_classifies() {
    // The ceiling-less twin (Sol r5 MF1 asked for both cases): `None` never proves a
    // fit impossible, so only the once-only exhaustion bound stops the spin - one
    // restart, then the raw memory-plan error, with not one eviction to spend.
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.deny_prefill_token = Some(1);
    fake.deny_prefill_token_budget = usize::MAX;
    fake.evictions = 0;
    fake.ceiling = None;
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[0], 1);
    let (denied, denied_events) = job(&[1, 1, 1, 1, 1], 1);
    for request in [first, denied] {
        engine.handle().try_submit(request).expect("submit");
    }
    release.send(()).expect("load");
    assert_eq!(usage(&collect(first_events)).completion_tokens, 1);
    assert_eq!(
        collect(denied_events),
        vec![JobEvent::Error(EngError::MemoryPlan(
            "piece denied".to_string()
        ))]
    );
    let calls = lock(&log).calls.clone();
    assert_eq!(
        calls.iter().filter(|call| *call == "reset").count(),
        3,
        "exactly one no-save recovery after exhaustion: {calls:?}"
    );
    engine.shutdown();
}

#[test]
fn a_denial_restart_restores_the_denied_jobs_own_tip_through_a_real_batch_store() {
    // Round 4e SF-1 (Opus + Sol r5 SF2) / r6 MF1: the batch pre-pass against a REAL
    // store (FakeExecutor + prefix limits + session budget with `max_seqs` 8, so
    // spawn_engine selects the BATCH worker - not the WindowFake evict counter, and
    // not the single-worker path the same fake defaults to). The denied conversation's
    // second turn restores its own tip, the growth cannot fit, the pre-pass protects
    // the restored anchor, and the rebuild re-restores it (reused > 0) - the
    // integration pin that the protect id from a real restored session survives the
    // batch restart path. A mutant that deletes the batch restart fails this fixture
    // outright (the turn errors instead of finishing).
    // HONEST SCOPE (both reviewers asked): this fixture does NOT discriminate a
    // no-protect mutant - the restore moves the denied conversation's entry to MRU,
    // so the oldest OTHER tip spends with or without protection (same for the worker
    // twin in worker.rs, `a_denial_keeps_the_denied_jobs_own_restored_anchor_and_
    // spends_a_dead_tip`). The arms that CAN be discriminated are pinned elsewhere:
    // the tier order by hand-built-store mutants in family/session.rs
    // (`an_interior_anchor_spends_before_an_older_covering_root` and the two-path
    // fixture). The SF1 knob test below exercises the STORE's own protect at the
    // engine level - the worker's protect WIRING itself is not discriminated by any
    // engine fixture (store-level tests pin the semantics); its candidate is the
    // OLDEST entry, so a drop-first or no-protect mutant spends it and reused_tokens
    // flips 100 -> 0.
    let (release, held) = mpsc::channel();
    let mut fake = FakeExecutor::new(vec![vec![EOS]; 6], &[EOS], 10_000);
    fake.with_max_seqs(8);
    fake.with_prefix_limits(2, 6);
    fake.with_session_budget(10_000_000, 1_024, 3_000_000);
    // Clone the window counter BEFORE the fake moves into the factory (round 4f MF1):
    // a positive count at the end asserts the BATCH branch ran - the single worker
    // never advances a decode window.
    let windows = fake.window_counter();
    let engine = spawn_engine(4, move || {
        held.recv_timeout(WAIT).expect("release load");
        Ok((fake, FakeSession::default()))
    })
    .expect("spawn");
    let handle = engine.handle();
    // The DENIED conversation saves its tip FIRST - the oldest tier-0 entry. Submit
    // all three before releasing the factory (collect blocks until the engine runs).
    let mut pending = Vec::new();
    for id in [4, 2, 3] {
        let prompt = vec![id; 101];
        let (job, events) = job(&prompt, 1);
        let job = job.with_prefix_boundary(100).expect("boundary");
        handle.try_submit(job).expect("submit");
        pending.push((id, events));
    }
    release.send(()).expect("load");
    drop(release);
    for (id, events) in pending {
        let events = collect(events);
        assert!(
            matches!(events.last(), Some(JobEvent::Finished(_))),
            "the conversation {id} ran: {events:?}"
        );
    }
    // The denied conversation's second turn: the admission cannot charge the 3,500
    // x 1,024 growth against the free budget, the retry loop protects the prompt's own
    // restore candidate, and after ONE eviction of another lineage's tip the rebuild
    // restores the denied conversation's own tip.
    let prompt = vec![4; 3_500];
    let (denied, denied_events) = job(&prompt, 1);
    let denied = denied.with_prefix_boundary(3_499).expect("boundary");
    handle.try_submit(denied).expect("submit");
    let events = collect(denied_events);
    assert!(
        matches!(events.last(), Some(JobEvent::Finished(usage)) if usage.reused_tokens > 0),
        "the rebuild re-restored the denied job's own tip: {events:?}"
    );
    assert!(
        windows.load(Ordering::SeqCst) > 0,
        "the batch branch ran: the engine advanced decode windows"
    );
    drop(handle);
    engine.shutdown();
}

#[test]
fn an_eviction_resets_the_exhaustion_bound_so_contention_keeps_its_restarts() {
    // Round 4f SF-3 (Opus) / r7 (Opus SF-B + Sol SF1, honest re-scope): this fixture
    // pins the flag's PLACEMENT - set only on the EXHAUSTION path, never on an
    // eviction-backed restart. One evictable entry, no ceiling, a persistent marker
    // denial: the first denial spends the entry and restart #1 is eviction-backed
    // (flag stays clear); the second denial finds the store exhausted and takes the
    // ONE no-save recovery (flag SET); the third classifies. Correct code: exactly
    // TWO restart resets. A mutant that sets the flag on EVERY restart classifies at
    // the second denial - ONE reset - and fails this count (mutant verified before
    // sealing 4f). NON-CLAIM: the eviction happens while the flag is already clear,
    // so deleting the successful-eviction RESET would also pass - pinning a reset
    // from an already-SET bound needs a fake whose store repopulates mid-flight
    // (an exhausted recovery, then another slot saves an entry, then spending it
    // refreshes eligibility); that pin is an SF1 follow-up.
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.deny_prefill_token = Some(1);
    fake.deny_prefill_token_budget = usize::MAX;
    fake.evictions = 1;
    fake.ceiling = None;
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[0], 1);
    let (denied, denied_events) = job(&[1, 1, 1, 1, 1], 1);
    for request in [first, denied] {
        engine.handle().try_submit(request).expect("submit");
    }
    release.send(()).expect("load");
    assert_eq!(usage(&collect(first_events)).completion_tokens, 1);
    assert_eq!(
        collect(denied_events),
        vec![JobEvent::Error(EngError::MemoryPlan(
            "piece denied".to_string()
        ))]
    );
    let calls = lock(&log).calls.clone();
    assert_eq!(
        calls.iter().filter(|call| *call == "evict").count(),
        1,
        "the one evictable entry went to the first denial: {calls:?}"
    );
    assert_eq!(
        calls.iter().filter(|call| *call == "reset").count(),
        4,
        "the two admissions plus TWO restart resets, then the classified refusal: {calls:?}"
    );
    engine.shutdown();
}

#[test]
fn an_adoption_denial_evicts_one_victim_and_retries_the_same_candidate_on_the_engine() {
    // SF1 on the full engine path (batch worker, real store): the fake's
    // `deny_restore_once` knob returns ONE adoption MemoryPlan from the restore
    // closure - the permit the resident executor charges for the whole restored
    // extent. The store must NOT drop the candidate: it evicts one victim with the
    // candidate itself protected and retries the SAME candidate, so the turn reuses
    // 100 tokens. The turn's growth still fits the budget (the SF1 victim's release
    // leaves ~4,000,000 free for the ~3,481,600 tail): the SECOND victim is the
    // SAVE-path snapshot denial - the boundary save at 3,499 charges its own permits
    // and evicts one more entry - and the save replaces the restored tip@100 with the
    // turn's own deeper tip@3499 (the lineage rule). A no-retry mutant fails the
    // reused assertion (the drop-and-next finds no other candidate for this prompt,
    // reused flips 100 -> 0), and a no-protect mutant IS discriminated at this tip
    // (verified by mutant rerun): the candidate is the OLDEST entry, so without
    // protection the retry's eviction spends it - the FIRST failing assertion is the
    // reused-tokens one (reused flips 100 -> 0), the keys assertion fails after. The
    // earlier not-discriminated note predates the knob fix, when this fixture was not
    // exercising SF1 at all. The knob fires ONLY when a candidate is actually
    // tried: a candidate-less first-turn restore leaves it armed. The
    // store-level twins are `an_adoption_denial_...` and
    // `a_persistent_denial_spends_one_victim_then_drops_the_candidate` in
    // family/session.rs.
    let (release, held) = mpsc::channel();
    let mut fake = FakeExecutor::new(vec![vec![EOS]; 6], &[EOS], 10_000);
    fake.with_max_seqs(8);
    fake.with_prefix_limits(2, 6);
    fake.with_session_budget(10_000_000, 1_024, 3_000_000);
    fake.deny_restore_once();
    let store = fake.prefix_store();
    let windows = fake.window_counter();
    let engine = spawn_engine(4, move || {
        held.recv_timeout(WAIT).expect("release load");
        Ok((fake, FakeSession::default()))
    })
    .expect("spawn");
    let handle = engine.handle();
    // The DENIED conversation saves its tip FIRST - the oldest entry at its denial.
    let mut pending = Vec::new();
    for id in [4, 2, 3] {
        let prompt = vec![id; 101];
        let (job, events) = job(&prompt, 1);
        let job = job.with_prefix_boundary(100).expect("boundary");
        handle.try_submit(job).expect("submit");
        pending.push((id, events));
    }
    release.send(()).expect("load");
    drop(release);
    for (id, events) in pending {
        let events = collect(events);
        assert!(
            matches!(events.last(), Some(JobEvent::Finished(_))),
            "the conversation {id} ran: {events:?}"
        );
    }
    // The denied conversation's second turn: the restore of its own tip denies ONCE
    // (the adoption permit), the store evicts the oldest OTHER tip and retries the
    // same candidate, and the turn runs on the restored 100 tokens.
    let prompt = vec![4; 3_500];
    let (denied, denied_events) = job(&prompt, 1);
    let denied = denied.with_prefix_boundary(3_499).expect("boundary");
    handle.try_submit(denied).expect("submit");
    let events = collect(denied_events);
    assert!(
        matches!(events.last(), Some(JobEvent::Finished(usage)) if usage.reused_tokens == 100),
        "the retried candidate restored its own 100 tokens: {events:?}"
    );
    // The lineage won its denial. THREE removals, ONE counted: the SF1 retry's
    // eviction (COUNTED, evict_lru_one); the boundary save at 3,499 is denied (the
    // snapshot needs ~12.48 MB against the 10 MB cap after the growth to ~9.48 MB) and
    // the SAVE's own eviction step removes a further tip - save-path removals bypass
    // evict_lru_one, so they are NOT counted; and the restored tip@100 is replaced by
    // the turn's own deeper tip@3499 through the save's lineage rule.
    let keys = store.lock().expect("store lock").keys();
    assert_eq!(
        keys,
        vec![(3_499, false)],
        "the deeper tip of the denied lineage: {keys:?}"
    );
    // Counter pins (Opus r1 MF-1): deleting either `+= 1` in session.rs must fail a
    // test. This fixture: no drop (the retry adopted), exactly one COUNTED eviction -
    // the SF1 retry's; the save-path removal and the lineage replacement do not touch
    // the counters. The drop pin lives in the store-level twins.
    let (drops, evictions) = {
        let store = store.lock().expect("store lock");
        (store.restore_drop_count(), store.eviction_count())
    };
    assert_eq!(drops, 0, "the retried candidate was not dropped");
    assert_eq!(evictions, 1, "the SF1 retry's victim, the only COUNTED one");
    assert!(windows.load(Ordering::SeqCst) > 0, "the batch branch ran");
    drop(handle);
    engine.shutdown();
}

#[test]
fn a_noted_engine_reports_its_executors_real_store_capability() {
    // The capability comes from the built executor's own prefix store, fired
    // through the one-shot ready note - no factory flag and no family name
    // takes part. The capable store is the BOUNDARY (two entries, the
    // minimum that can cut: one rolling Block entry plus the live session),
    // so a mutant that needs three or more entries dies here too.
    let (cut_tx, cut_rx) = mpsc::channel();
    let mut capable = FakeExecutor::new(vec![vec![7]], &[0], 1);
    capable.with_prefix_limits(0, 2);
    let capable_note: ReadyNote = Some(Box::new(move |note| {
        if let EngineNote::Ready(capability) = note {
            let _ = cut_tx.send(capability);
        }
    }));
    let capable_engine = spawn_engine_noted(4, capable_note, move || {
        Ok((capable, FakeSession::default()))
    })
    .expect("spawn");
    let (plain_tx, plain_rx) = mpsc::channel();
    let plain_note: ReadyNote = Some(Box::new(move |note| {
        if let EngineNote::Ready(capability) = note {
            let _ = plain_tx.send(capability);
        }
    }));
    let plain_engine = spawn_engine_noted(4, plain_note, move || {
        Ok((
            FakeExecutor::new(vec![vec![7]], &[0], 1),
            FakeSession::default(),
        ))
    })
    .expect("spawn");
    let capable_report = cut_rx
        .recv_timeout(WAIT)
        .expect("the capable engine fires its note");
    assert!(
        capable_report.cuts_block,
        "the two-entry boundary store can cut a rolling Block entry"
    );
    assert_eq!(
        capable_report.spacing,
        FakeExecutor::new(vec![vec![7]], &[0], 1).prefix_block_spacing(),
        "the note carries the built executor's own spacing"
    );
    let plain_report = plain_rx
        .recv_timeout(WAIT)
        .expect("the default engine fires its note");
    assert!(
        !plain_report.cuts_block,
        "the one-entry default store can never cut a rolling Block entry"
    );
    capable_engine.shutdown();
    plain_engine.shutdown();
}

#[test]
fn a_failed_make_fires_the_note_with_the_failure_cause() {
    // The note is one-shot for BOTH outcomes: a failed make fires Failed
    // with the same cause string record_failure stores (the stderr line the
    // server prints at spawn through block_match_note), instead of dropping
    // the note unfired - a load that fails before anyone polls /ready is no
    // longer silent. A failed engine still never claims a capability: the
    // payload is the cause, never a PrefixCapability. Documented duplicate:
    // /ready's log_new_failures prints the same line once more at the first
    // /ready call - its `reported` set cannot see the spawn-time print.
    let (fail_tx, fail_rx) = mpsc::channel();
    let fail_note: ReadyNote = Some(Box::new(move |note| {
        let _ = fail_tx.send(note);
    }));
    let failed_engine = spawn_engine_noted(4, fail_note, || {
        Err::<(FakeExecutor, FakeSession), EngError>(EngError::MemoryPlan("no executor".into()))
    })
    .expect("spawn");
    match fail_rx.recv_timeout(WAIT) {
        Ok(EngineNote::Failed(cause)) => assert!(
            cause.contains("engine load failed") && cause.contains("no executor"),
            "the note carries the recorded cause: {cause}"
        ),
        Ok(EngineNote::Ready(_)) => {
            panic!("a failed make must not claim a capability")
        }
        Err(error) => panic!("the failure note must fire, got {error:?}"),
    }
    assert!(
        matches!(
            fail_rx.recv_timeout(WAIT),
            Err(mpsc::RecvTimeoutError::Disconnected)
        ),
        "the note is one-shot: the sender disconnects after the failure"
    );
    failed_engine.shutdown();
}

#[test]
fn row_growth_denial_keeps_admitted_and_queued_jobs_running() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.failure = Some(Failure::RowLocal);
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[0], 3);
    let (denied, denied_events) = job(&[1], 3);
    let (next, next_events) = job(&[2], 2);
    for request in [first, denied, next] {
        engine.handle().try_submit(request).expect("submit");
    }
    release.send(()).expect("load");
    let first = collect(first_events);
    assert_eq!(output(&first), "aaa");
    assert_eq!(usage(&first).completion_tokens, 3);
    assert_eq!(
        collect(denied_events),
        vec![JobEvent::Error(EngError::MemoryPlan(
            "row growth denied".into()
        ))]
    );
    let next = collect(next_events);
    assert_eq!(output(&next), "aa");
    assert_eq!(usage(&next).completion_tokens, 2);
    assert_eq!(lock(&log).windows[0], vec![vec![0], vec![1]]);
    assert_eq!(lock(&log).windows[1], vec![vec![0, 0], vec![2]]);
    assert_eq!(engine.shutdown().state, EngineState::Stopped);
}

#[test]
fn row_growth_denial_preserves_a_cancelled_peers_finish() {
    let log = Log::default();
    let (hold, entered, unblock) = gate();
    let mut fake = WindowFake::new(&log);
    fake.failure = Some(Failure::RowLocal);
    fake.hold = Some(hold);
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[0], 3);
    let cancel = Arc::clone(&first.generation.cancel);
    let (denied, denied_events) = job(&[1], 3);
    engine.handle().try_submit(first).expect("first");
    engine.handle().try_submit(denied).expect("denied");
    release.send(()).expect("load");
    entered.recv_timeout(WAIT).expect("entered");
    cancel.store(true, Ordering::SeqCst);
    unblock.send(()).expect("window");
    let first = collect(first_events);
    assert_eq!(finish(&first), DecodeFinish::Cancelled);
    assert_eq!(usage(&first).completion_tokens, 0);
    assert_eq!(
        collect(denied_events),
        vec![JobEvent::Error(EngError::MemoryPlan(
            "row growth denied".into()
        ))]
    );
    assert_eq!(engine.shutdown().state, EngineState::Stopped);
}

#[test]
fn local_window_failure_discards_all_consumed_rows_without_replay() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.failure = Some(Failure::Local);
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[0], 1);
    let (second, second_events) = job(&[1], 1);
    let (next, next_events) = job(&[2], 1);
    for request in [first, second, next] {
        engine.handle().try_submit(request).expect("submit");
    }
    release.send(()).expect("load");
    for events in [first_events, second_events] {
        assert_eq!(
            collect(events),
            vec![JobEvent::Error(EngError::MemoryPlan(
                "window growth denied".into()
            ))]
        );
    }
    let next = collect(next_events);
    assert_eq!(output(&next), "a");
    assert_eq!(usage(&next).completion_tokens, 1);
    assert_eq!(lock(&log).windows[1], vec![vec![2]]);
    engine.shutdown();
}

#[test]
fn native_failure_panic_and_bad_row_count_terminate_every_job_without_partial_output() {
    for failure in [
        Failure::Native,
        Failure::RowNative,
        Failure::Panic,
        Failure::Count,
    ] {
        let log = Log::default();
        let mut fake = WindowFake::new(&log);
        fake.failure = Some(failure);
        let (engine, release) = start(fake);
        let mut handles = Vec::new();
        for id in 0..3 {
            let (request, events) = job(&[id], 2);
            engine.handle().try_submit(request).expect("submit");
            handles.push(events);
        }
        release.send(()).expect("load");
        for events in handles {
            assert!(matches!(
                collect(events).as_slice(),
                [JobEvent::Error(EngError::Backend { .. })]
            ));
        }
        let trace = lock(&log);
        assert_eq!(trace.windows.len(), 1);
        assert_eq!(trace.calls.last().map(String::as_str), Some("window"));
        drop(trace);
        let (later, _events) = job(&[0], 1);
        assert_eq!(
            engine.handle().try_submit(later),
            Err(SubmitError::NativeFailed)
        );
        let report = engine.shutdown();
        assert_eq!(report.state, EngineState::NativeFailed);
        assert!(report.failure.is_some());
    }
}

#[test]
fn deferred_native_failure_keeps_batch_then_error_order_for_every_window_row() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.fail_after_emit = true;
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[0], 1);
    let (second, second_events) = job(&[1], 1);
    engine.handle().try_submit(first).expect("first");
    engine.handle().try_submit(second).expect("second");
    release.send(()).expect("load");
    for events in [first_events, second_events] {
        assert!(matches!(
            collect(events).as_slice(),
            [
                JobEvent::Batch(_),
                JobEvent::Error(EngError::Backend { .. })
            ]
        ));
    }
    assert_eq!(engine.shutdown().state, EngineState::NativeFailed);
}

#[test]
fn shutdown_cancels_all_active_and_queued_jobs_while_handles_remain_alive() {
    let log = Log::default();
    let (hold, entered, unblock) = gate();
    let mut fake = WindowFake::new(&log);
    fake.hold = Some(hold);
    let (engine, release) = start(fake);
    let handle = engine.handle();
    let mut handles = Vec::new();
    for id in 0..4 {
        let (request, events) = job(&[id], 4);
        handle.try_submit(request).expect("submit");
        handles.push(events);
    }
    release.send(()).expect("load");
    entered.recv_timeout(WAIT).expect("entered");
    let shutdown = std::thread::spawn(move || engine.shutdown());
    wait(|| handles[0].is_cancelled() && handles[1].is_cancelled());
    unblock.send(()).expect("window");
    for events in handles {
        let events = collect(events);
        assert_eq!(finish(&events), DecodeFinish::Cancelled);
        assert_eq!(usage(&events).completion_tokens, 0);
    }
    assert_eq!(
        shutdown.join().expect("shutdown thread").state,
        EngineState::Stopped
    );
    assert_eq!(lock(&log).windows.len(), 1);
    let (later, _events) = job(&[0], 1);
    assert_eq!(handle.try_submit(later), Err(SubmitError::Unavailable));
}

#[test]
fn a_slow_receiver_cannot_block_another_job_or_shutdown() {
    let log = Log::default();
    let (engine, release) = start(WindowFake::new(&log));
    let (slow, slow_events) = job(&[0], 1_100);
    let (fast, fast_events) = job(&[1], 1);
    engine.handle().try_submit(slow).expect("slow");
    engine.handle().try_submit(fast).expect("fast");
    release.send(()).expect("load");
    assert_eq!(usage(&collect(fast_events)).completion_tokens, 1);
    wait(|| slow_events.is_cancelled());
    let (next, next_events) = job(&[2], 1);
    engine.handle().try_submit(next).expect("newcomer");
    assert_eq!(usage(&collect(next_events)).completion_tokens, 1);
    // Keep the slow receiver undrained through shutdown.
    assert_eq!(engine.shutdown().state, EngineState::Stopped);
    assert!(
        !collect(slow_events)
            .iter()
            .any(|event| matches!(event, JobEvent::Finished(_)))
    );
}

#[test]
fn eos_finishes_only_its_row_and_counts_the_committed_eos_token() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.script_tokens = 2;
    let (engine, release) = start(fake);
    let (eos, eos_events) = job(&[0], 5);
    let (limited, limited_events) = job(&[1], 1);
    engine.handle().try_submit(eos).expect("eos");
    engine.handle().try_submit(limited).expect("limited");
    release.send(()).expect("load");
    let eos = collect(eos_events);
    let limited = collect(limited_events);
    assert_eq!(output(&eos), "aa");
    assert_eq!(finish(&eos), DecodeFinish::Stop);
    assert_eq!(usage(&eos).completion_tokens, 3);
    assert_eq!(finish(&limited), DecodeFinish::Length);
    assert_eq!(usage(&limited).completion_tokens, 1);
    assert_eq!(
        lock(&log).windows.iter().map(Vec::len).collect::<Vec<_>>(),
        [2, 1, 1]
    );
    engine.shutdown();
}

#[test]
fn a_context_rejection_does_not_consume_a_session_or_change_other_rows() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.context_limit = Some(4);
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[0], 1);
    let (rejected, rejected_events) = job(&[1; 4], 1);
    let (last, last_events) = job(&[2], 1);
    for request in [first, rejected, last] {
        engine.handle().try_submit(request).expect("submit");
    }
    release.send(()).expect("load");
    assert!(matches!(
        collect(rejected_events).as_slice(),
        [JobEvent::Error(EngError::PromptTooLong { .. })]
    ));
    assert_eq!(usage(&collect(first_events)).completion_tokens, 1);
    assert_eq!(usage(&collect(last_events)).completion_tokens, 1);
    assert_eq!(
        lock(&log)
            .calls
            .iter()
            .filter(|call| call.as_str() == "new_session")
            .count(),
        1
    );
    engine.shutdown();
}

#[test]
fn seeded_samplers_are_forwarded_independently_for_each_row() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.draws = true;
    let (engine, release) = start(fake);
    let mut handles = Vec::new();
    let mut expected = Vec::new();
    for (id, seed) in [(0, 17), (1, 29)] {
        let params = SamplingParams::new(1.0, 1.0, 0).expect("params");
        let mut reference = Sampler::new(params, seed);
        expected.push((id, reference.sample(&[0.0; 3]).expect("reference draw")));
        expected.push((id, reference.sample(&[0.0; 3]).expect("reference draw")));
        let tokenizer = Tokenizer::from_bytes(JSON.as_bytes()).expect("tokenizer");
        let state = GenerationState::with_sampler(
            IncrementalDecoder::new(&tokenizer),
            Sampler::new(params, seed),
        )
        .with_max_tokens(2);
        let (request, events) = Job::new(vec![id], state).expect("job");
        engine.handle().try_submit(request).expect("submit");
        handles.push(events);
    }
    release.send(()).expect("load");
    for events in handles {
        assert_eq!(usage(&collect(events)).completion_tokens, 2);
    }
    let trace = lock(&log);
    for id in [0, 1] {
        assert_eq!(
            trace
                .draws
                .iter()
                .filter(|entry| entry.0 == id)
                .copied()
                .collect::<Vec<_>>(),
            expected
                .iter()
                .filter(|entry| entry.0 == id)
                .copied()
                .collect::<Vec<_>>()
        );
    }
    drop(trace);
    engine.shutdown();
}
