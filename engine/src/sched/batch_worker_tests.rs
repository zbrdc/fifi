//! Portable worker tests. The row fake owns independent scripted state, not a model.

use super::super::{
    Engine, EngineNote, EngineState, ReadyNote, SubmitError, lock, spawn_engine, spawn_engine_noted,
};
use crate::EngError;
use crate::exec_fake::{FakeExecutor, FakeSession};
use crate::family::executor::{ExecutorOps, GenerationState};
use crate::family::sampler::{Sampler, SamplingParams};
use crate::family::session::{SaveKind, StoreLimits};
use crate::sched::job::{Job, JobEvent, JobHandle, PrefillBankReport, PrefixHit, PrefixReport};
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
    /// kind-less `save:N` shape; the kind assertion reads this instead.
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
    hold_prefill: Option<Gate>,
    /// A gate the test arms while the engine runs: the next reset takes it, signals that it
    /// entered, and waits to be released. It holds the worker inside a call so that a job
    /// submitted meanwhile is certain to meet a live slot.
    armed_hold: Option<Arc<Mutex<Option<Gate>>>>,
    /// The same for the next window: the fake takes the gate, signals that it entered, and waits
    /// to be released.
    armed_window: Option<Arc<Mutex<Option<Gate>>>>,
    failure: Option<Failure>,
    deny_session: bool,
    /// When set, the denied `new_session` marks this flag cancelled first: the test hook
    /// for a client that gives up exactly when admission refuses.
    cancel_denied_session: Option<Arc<std::sync::atomic::AtomicBool>>,
    /// The fake's pool-derived ceiling, which classifies a final denial on the batch path:
    /// `None` means no budget to report, like the default executor.
    ceiling: Option<u64>,
    fail_after_emit: bool,
    draws: bool,
    report: PrefixReport,
    bank_report: Option<PrefillBankReport>,
    /// The number of `prefill` calls so far, for the one-shot piece denial below.
    prefill_calls: usize,
    /// When set, the `prefill` call at this index returns a memory-plan denial once
    /// (the eviction restart test): a piece that fails after an earlier piece succeeded.
    fail_prefill_piece: Option<usize>,
    /// Denies every prefill piece whose tokens carry this marker token, up to the budget
    /// below: a PER-JOB failure - the denied job restarts, denies again if the budget
    /// allows, and only the spent eviction budget makes it final, while other jobs' pieces
    /// prefill fine.
    deny_prefill_token: Option<u32>,
    /// The marker denial budget (usize::MAX denies every carrying piece, the exhaustion
    /// path; a small budget lets the restart succeed).
    deny_prefill_token_budget: usize,
    /// Evictions handed out so far: the fake's ceiling grows with each, modeling a pool
    /// whose honest prompt ceiling rises as entries free.
    evicted: u64,
    deny_all_prefill: bool,
    /// How many evictions the fake can still free: `Some(0)` is honest progress - an entry
    /// went, zero bytes were charged to this fake - and the count runs out, after which
    /// `None` means the store is empty and a denial is final. The default of 8 is more than
    /// any single request should need.
    evictions: usize,
    /// When set, the `reset` call at this index returns this error once: a native failure
    /// inside the restart of a denied slot (reset 0 belongs to the admission, reset 1 to
    /// the restart).
    fail_reset_call: Option<(usize, EngError)>,
    /// The number of `reset` calls so far, for the indexed failure below.
    reset_calls: usize,
    /// The pinned block prefix-match spacing: `None` keeps every save-point
    /// walk on the no-block path regardless of the flipped environment default.
    spacing: Option<usize>,
    /// The prompt rows per tick the fake asks the worker for; `None` keeps the worker's rule.
    prompt_share: Option<usize>,
    /// The sessions the fake can hold, the seed included: `new_session` beyond it is a memory
    /// denial, as when the device has no room for another session.
    session_cap: Option<usize>,
    /// The sessions built so far, the seed included.
    created: usize,
    /// A cap a test raises while the engine runs; it replaces `session_cap` when set.
    live_cap: Option<Arc<std::sync::atomic::AtomicUsize>>,
    /// A pause per window, so a stream stays live for a while in a test.
    window_delay: Option<Duration>,
    /// The window row that is denied its growth every time.
    deny_slot: Option<usize>,
    /// How long the fake lets the engine idle before the pool trims; `None` never trims.
    idle_trim: Option<Duration>,
    /// The cold prompts that take a whole piece in one idle tick; 1 is the default policy.
    idle_pieces: usize,
    /// Answers the trait's default policy instead of the opted-in one.
    default_policy: bool,
    /// What `can_admit_session` answers while a test holds the device full; `None` says yes.
    fits: Option<Arc<std::sync::atomic::AtomicBool>>,
    /// Whether evicting prefix entries can free a session's memory (the trait's default is yes).
    eviction_frees: bool,
    /// Background work (a disk write-behind) the fake reports as pending until this many steps
    /// ran; how many ran, how many ran while a stream was live, and whether it was drained.
    background_want: usize,
    background_steps: Arc<std::sync::atomic::AtomicUsize>,
    background_drained: Arc<std::sync::atomic::AtomicBool>,
}

impl WindowFake {
    fn new(log: &Log) -> Self {
        Self {
            max: 2,
            script_tokens: 1_100,
            context_limit: None,
            log: Arc::clone(log),
            hold: None,
            hold_prefill: None,
            armed_hold: None,
            armed_window: None,
            failure: None,
            deny_session: false,
            cancel_denied_session: None,
            ceiling: None,
            fail_after_emit: false,
            draws: false,
            report: PrefixReport::default(),
            bank_report: None,
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
            prompt_share: None,
            session_cap: None,
            created: 1,
            live_cap: None,
            window_delay: None,
            deny_slot: None,
            // The fake stands for an executor that has opted in: an hour is never reached.
            idle_trim: Some(Duration::from_secs(3600)),
            idle_pieces: 8,
            default_policy: false,
            fits: None,
            eviction_frees: true,
            background_want: 0,
            background_steps: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            background_drained: Arc::new(std::sync::atomic::AtomicBool::new(false)),
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
        // Deliberately change every counter on decode too: ready slots must keep their
        // prefill sample, not re-sample the executor at completion.
        if let Some(bank) = &mut self.bank_report {
            bank.passes += 1_000;
            bank.misses += 1_000;
            bank.upload_bytes += 1_000;
            bank.upload_ns += 1_000;
            for count in bank
                .miss_histogram
                .iter_mut()
                .chain(&mut bank.route_histogram)
            {
                *count += 1_000;
            }
        }
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

    fn prefill_piece(&self) -> Option<usize> {
        Some(4)
    }

    fn max_seqs(&self) -> usize {
        self.max
    }

    fn prompt_rows_per_tick(&self) -> Option<usize> {
        self.prompt_share
    }

    fn can_admit_session(&self) -> bool {
        self.fits
            .as_ref()
            .is_none_or(|fits| fits.load(Ordering::SeqCst))
    }

    fn eviction_frees_session_memory(&self) -> bool {
        self.eviction_frees
    }

    fn background_pending(&self) -> bool {
        self.background_steps.load(Ordering::SeqCst) < self.background_want
    }

    fn background_step(&mut self) {
        self.background_steps.fetch_add(1, Ordering::SeqCst);
        self.note("background_step");
    }

    fn background_drain(&mut self) {
        self.background_drained.store(true, Ordering::SeqCst);
    }

    fn batch_policy(&self) -> crate::family::executor::BatchPolicy {
        if self.default_policy {
            return crate::family::executor::BatchPolicy::default();
        }
        crate::family::executor::BatchPolicy {
            idle_pieces: self.idle_pieces,
            idle_trim: self.idle_trim,
        }
    }

    fn new_session(&mut self) -> Result<Row, EngError> {
        self.note("new_session");
        let cap = match &self.live_cap {
            Some(live) => Some(live.load(Ordering::SeqCst)),
            None => self.session_cap,
        };
        if cap.is_some_and(|cap| self.created >= cap) {
            self.note("new_session_refused");
            return Err(EngError::MemoryPlan(
                "the device holds no more sessions".into(),
            ));
        }
        self.created += 1;
        if std::mem::take(&mut self.deny_session) {
            if let Some(cancel) = &self.cancel_denied_session {
                cancel.store(true, Ordering::SeqCst);
            }
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
        // k evictions before the ceiling admits.
        self.ceiling.map(|base| base + 2 * self.evicted)
    }

    fn history<'s>(&self, row: &'s Row) -> &'s [u32] {
        row.fake.history(&row.state)
    }

    fn reset(&mut self, row: &mut Row) -> Result<(), EngError> {
        self.note("reset");
        let armed = self
            .armed_hold
            .as_ref()
            .and_then(|armed| lock(armed).take());
        if let Some((entered, release)) = armed {
            entered.send(()).expect("signal armed reset");
            release.recv_timeout(WAIT).expect("release armed reset");
        }
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
        if let Some((entered, release)) = self.hold_prefill.take() {
            entered.send(()).expect("signal prefill");
            release.recv_timeout(WAIT).expect("release prefill");
        }
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
        row.fake.prefill(&mut row.state, tokens)?;
        if let Some(bank) = &mut self.bank_report {
            let rows = tokens.len() as u64;
            bank.passes += 1;
            bank.misses += rows;
            bank.miss_histogram[0] += 1;
            bank.miss_histogram[1] += rows;
            bank.route_histogram[0] += rows;
            bank.route_histogram[1] += 1;
            bank.upload_bytes += rows * 10;
            bank.upload_ns += rows * 100;
        }
        Ok(())
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
        let armed = self
            .armed_window
            .as_ref()
            .and_then(|armed| lock(armed).take());
        if let Some((entered, release)) = armed {
            entered.send(()).expect("signal armed window");
            release.recv_timeout(WAIT).expect("release armed window");
        }
        if let Some(delay) = self.window_delay {
            std::thread::sleep(delay);
        }
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
            if self.deny_slot == Some(slot) {
                batches.push(Err(EngError::MemoryPlan("row growth denied".into())));
                continue;
            }
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
        // Pinned per test: the fake never reads the environment, so the
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

    fn prefill_bank_report(&self) -> Option<PrefillBankReport> {
        self.bank_report.clone()
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

/// Submits two jobs to an idle engine so that the second finds the first live: the armed gate
/// holds the worker inside the first job's admission (its reset) until the second is in the queue.
/// The worker then takes the second job through `try_recv` in the same admission pass, before the
/// tick, whatever the scheduling of the host. Returns the two handles. The engine's fake must have
/// been given `armed_hold`.
fn submit_two_live(
    engine: &Engine,
    armed: &Arc<Mutex<Option<Gate>>>,
    first: (Job, JobHandle),
    second: (Job, JobHandle),
) -> (JobHandle, JobHandle) {
    let (entered, observed) = mpsc::channel();
    let (release, held) = mpsc::channel();
    *lock(armed) = Some((entered, held));
    engine.handle().try_submit(first.0).expect("first");
    observed
        .recv_timeout(WAIT)
        .expect("the worker entered the first admission");
    engine.handle().try_submit(second.0).expect("second");
    release.send(()).expect("release");
    (first.1, second.1)
}

/// Hold factory completion until all initial jobs have entered the bounded queue.
fn start(fake: WindowFake) -> (Engine, mpsc::Sender<()>) {
    start_with_budget(fake, 8)
}

fn start_with_budget(fake: WindowFake, rows: usize) -> (Engine, mpsc::Sender<()>) {
    let (release, held) = mpsc::channel();
    let engine = spawn_engine(rows, move |_| {
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
            // Started is asserted by the worker tests; sequences here
            // compare batches and terminal events only.
            Ok(JobEvent::Started) => {}
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
        Some(JobEvent::Finished(usage)) => usage.clone(),
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
fn prefill_bank_counts_only_the_slots_own_pieces() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.bank_report = Some(PrefillBankReport {
        passes: 10,
        misses: 20,
        miss_histogram: vec![30, 40],
        route_histogram: vec![50, 60],
        upload_bytes: 70,
        upload_ns: 80,
    });
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[0; 6], 2);
    let (second, second_events) = job(&[1; 10], 3);
    engine.handle().try_submit(first).expect("first");
    engine.handle().try_submit(second).expect("second");
    release.send(()).expect("load");
    let first = usage(&collect(first_events));
    let second = usage(&collect(second_events));
    assert_eq!(
        first.prefill_bank,
        Some(PrefillBankReport {
            passes: 2,
            misses: 6,
            miss_histogram: vec![2, 6],
            route_histogram: vec![6, 2],
            upload_bytes: 60,
            upload_ns: 600,
        }),
        "only the first slot's two pieces"
    );
    assert_eq!(
        second.prefill_bank,
        Some(PrefillBankReport {
            passes: 3,
            misses: 10,
            miss_histogram: vec![3, 10],
            route_histogram: vec![10, 3],
            upload_bytes: 100,
            upload_ns: 1000,
        }),
        "only the second slot's three pieces, excluding the first slot's decode"
    );
    assert_eq!((first.completion_tokens, second.completion_tokens), (2, 3));
    assert_eq!(
        lock(&log).windows.len(),
        4,
        "decode changes the counters on every tick"
    );
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
    // The restart resets the piece count, so the line carries only the serving run's
    // two pieces (4 rows then 1), never the aborted attempt's two as well.
    assert_eq!(
        usage(&events).prefill_pieces.count,
        2,
        "only the serving run's pieces: {events:?}"
    );
    assert_eq!(
        usage(&events).prefill_pieces.max_rows,
        4,
        "the largest serving piece"
    );
    let trace = lock(&log);
    assert_eq!(
        trace.calls,
        vec![
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
    // Pinned spacing: the fake overrides the environment, so this batch case
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
    // snapshot, because a snapshot that charges can starve the tail that follows it.
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
    // must reach the job and the engine state.
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
fn a_cold_prompt_and_a_stream_share_the_iteration_budget_in_either_arrival_order() {
    for cold_first in [false, true] {
        let log = Log::default();
        let (hold, entered, unblock) = gate();
        let mut fake = WindowFake::new(&log);
        if cold_first {
            fake.hold_prefill = Some(hold);
        } else {
            fake.hold = Some(hold);
        }
        let (engine, release) = start_with_budget(fake, 4);
        let (stream, stream_events) = job(&[0], 16);
        let (cold, cold_events) = job(&[1; 24], 1);
        let (first, second) = if cold_first {
            (cold, stream)
        } else {
            (stream, cold)
        };
        engine.handle().try_submit(first).unwrap();
        release.send(()).unwrap();
        entered.recv_timeout(WAIT).unwrap();
        engine.handle().try_submit(second).unwrap();
        unblock.send(()).unwrap();
        assert_eq!(usage(&collect(stream_events)).completion_tokens, 16);
        assert_eq!(usage(&collect(cold_events)).completion_tokens, 1);
        engine.shutdown();

        let trace = lock(&log);
        let mut windows = trace.windows.iter();
        let mut rows = 0;
        let mut streaming = false;
        let mut shared_ticks = 0;
        let mut cold_decoded = false;
        for call in &trace.calls {
            if let Some(piece) = call.strip_prefix("prefill:") {
                rows += piece.parse::<usize>().unwrap();
            } else if call == "window" {
                let window = windows.next().unwrap();
                if streaming && !cold_decoded {
                    assert!(rows + window.len() <= 4, "over budget in {window:?}");
                    if window.iter().any(|history| history[0] == 1) {
                        cold_decoded = true;
                        assert_eq!(window.len(), 2, "cold client finished during streaming");
                    } else {
                        assert!(rows > 0, "cold prompt starved");
                        shared_ticks += 1;
                    }
                }
                streaming = true;
                rows = 0;
            }
        }
        assert!(shared_ticks > 0 && cold_decoded);
    }
}

#[test]
fn a_prompt_that_finishes_prefill_joins_after_one_adoption_tick() {
    // The either-order test above cannot tell WHEN the cold joins: its
    // history appearing in some window passes it however late. The join the
    // budget promises costs exactly one adoption tick after the finishing
    // piece - readiness is computed at the tick top, before prefills run,
    // and the seat needs one new_session - never a second idle tick. Every
    // prefill call here belongs to the cold (the stream's prompt is empty),
    // so the last prefill-bearing tick IS its finishing tick.
    let log = Log::default();
    let (hold, entered, unblock) = gate();
    let mut fake = WindowFake::new(&log);
    fake.hold = Some(hold);
    let (engine, release) = start_with_budget(fake, 4);
    let (stream, stream_events) = job(&[0], 16);
    let (cold, cold_events) = job(&[1; 24], 1);
    engine.handle().try_submit(stream).unwrap();
    release.send(()).unwrap();
    entered.recv_timeout(WAIT).unwrap();
    engine.handle().try_submit(cold).unwrap();
    unblock.send(()).unwrap();
    assert_eq!(usage(&collect(stream_events)).completion_tokens, 16);
    assert_eq!(usage(&collect(cold_events)).completion_tokens, 1);
    engine.shutdown();

    let trace = lock(&log);
    // Every prefill call belongs to the cold (the stream's prompt is empty),
    // so the last prefill-bearing tick IS the cold's finishing tick.
    let mut last_prefill_tick = None;
    let mut join_window = None;
    let mut tick = 0;
    for call in &trace.calls {
        if call.starts_with("prefill:") {
            last_prefill_tick = Some(tick);
        } else if call == "window" {
            let window = &trace.windows[tick];
            if join_window.is_none() && window.iter().any(|history| history[0] == 1) {
                join_window = Some(tick);
            }
            tick += 1;
        }
    }
    let last_prefill = last_prefill_tick.expect("the cold prefilled");
    let join = join_window.expect("the cold joined a decode window");
    // One adoption tick at most between the finishing piece and the window
    // seat: readiness is computed at the tick top, before prefills run, and
    // the seat needs one new_session. Gate timing can land the finishing
    // piece before or after the tick's own window call, so the join sits one
    // or two windows out - never more. A longer idle gap or a second
    // session is the defect this pins.
    assert!(
        (1..=2).contains(&(join - last_prefill)),
        "the cold idled {} windows after finishing",
        join - last_prefill
    );
    assert_eq!(trace.windows[join].len(), 2);
    assert_eq!(
        trace
            .calls
            .iter()
            .filter(|call| **call == "new_session")
            .count(),
        1,
        "adoption must spend exactly one session"
    );
}

#[test]
fn cancelling_the_last_prefill_piece_publishes_no_tokens_in_either_worker() {
    for max_seqs in [1, 2] {
        let log = Log::default();
        let (hold, entered, unblock) = gate();
        let mut fake = WindowFake::new(&log);
        fake.max = max_seqs;
        fake.hold_prefill = Some(hold);
        let (engine, release) = start_with_budget(fake, 4);
        let (request, events) = job(&[1; 3], 2);
        engine.handle().try_submit(request).unwrap();
        release.send(()).unwrap();
        entered.recv_timeout(WAIT).unwrap();
        events.cancel();
        unblock.send(()).unwrap();
        let result = collect(events);
        assert_eq!(finish(&result), DecodeFinish::Cancelled);
        assert_eq!(usage(&result).completion_tokens, 0);
        assert!(output(&result).is_empty());
        assert_eq!(engine.shutdown().state, EngineState::Stopped);
        // The single worker can call decode to finish cancellation; the batch worker
        // must not submit a cancelled row to its native window.
        assert!(lock(&log).windows.is_empty());
    }
}

#[test]
fn an_idle_batch_feeds_each_cold_prompt_a_whole_piece() {
    let log = Log::default();
    let (engine, release) = start_with_budget(WindowFake::new(&log), 1);
    let (first, first_events) = job(&[0; 3], 2);
    let (second, second_events) = job(&[1; 3], 2);
    engine.handle().try_submit(first).unwrap();
    engine.handle().try_submit(second).unwrap();
    release.send(()).unwrap();
    for events in [first_events, second_events] {
        let result = usage(&collect(events));
        // The piece is 4: with no stream decoding each cold prompt takes a whole piece.
        assert_eq!(result.prefill_pieces.count, 1);
        assert_eq!(result.prefill_pieces.max_rows, 3);
        assert_eq!(result.completion_tokens, 2);
    }
    engine.shutdown();
    assert_eq!(lock(&log).windows[0], [vec![0; 3], vec![1; 3]]);
}

#[test]
fn an_executor_that_does_not_opt_in_keeps_the_shared_idle_budget() {
    // The default policy gives two cold prompts one piece between them per tick: each takes two
    // rows at a time, where the opted-in executor above feeds each a whole piece at once.
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.idle_pieces = 1;
    let (engine, release) = start_with_budget(fake, 1);
    let (first, first_events) = job(&[0; 3], 2);
    let (second, second_events) = job(&[1; 3], 2);
    engine.handle().try_submit(first).unwrap();
    engine.handle().try_submit(second).unwrap();
    release.send(()).unwrap();
    for events in [first_events, second_events] {
        let result = usage(&collect(events));
        assert_eq!(
            result.prefill_pieces.count, 2,
            "{:?}",
            result.prefill_pieces
        );
        assert_eq!(result.prefill_pieces.max_rows, 2);
        assert_eq!(result.completion_tokens, 2);
    }
    engine.shutdown();
}

#[test]
fn a_cold_prompt_takes_the_piece_minus_the_decode_rows_while_a_stream_decodes() {
    let log = Log::default();
    let (hold, entered, unblock) = gate();
    let mut fake = WindowFake::new(&log);
    fake.hold = Some(hold);
    let (engine, release) = start_with_budget(fake, 1);
    let (stream, stream_events) = job(&[0], 16);
    let (cold, cold_events) = job(&[1; 6], 1);
    engine.handle().try_submit(stream).unwrap();
    release.send(()).unwrap();
    entered.recv_timeout(WAIT).unwrap();
    engine.handle().try_submit(cold).unwrap();
    unblock.send(()).unwrap();
    assert_eq!(usage(&collect(stream_events)).completion_tokens, 16);
    let result = usage(&collect(cold_events));
    // The piece is 4 and one row of it decodes the stream, so the prompt feeds 3 rows per tick.
    assert_eq!(result.prefill_pieces.count, 2);
    assert_eq!(result.prefill_pieces.max_rows, 3);
    assert_eq!(result.completion_tokens, 1);
    engine.shutdown();
}

#[test]
fn a_multi_piece_prompt_counts_every_piece_without_a_restart() {
    let log = Log::default();
    let (engine, release) = start(WindowFake::new(&log));
    // 10 tokens at worker piece 4 feed 4,4,2 with no denial: the count accumulates
    // across the pieces (the restart test pins the reset; this pins the steady path).
    let (long, long_events) = job(&[1; 10], 1);
    engine.handle().try_submit(long).expect("submit");
    release.send(()).expect("load");
    let events = collect(long_events);
    assert_eq!(
        usage(&events).prefill_pieces.count,
        3,
        "three pieces: {events:?}"
    );
    assert_eq!(
        usage(&events).prefill_pieces.max_rows,
        4,
        "the largest piece"
    );
    engine.shutdown();
}

#[test]
fn two_slots_count_their_own_prefill_pieces() {
    let log = Log::default();
    let (engine, release) = start(WindowFake::new(&log));
    let (short, short_events) = job(&[0], 1);
    let (long, long_events) = job(&[1; 10], 1);
    engine.handle().try_submit(short).expect("short");
    engine.handle().try_submit(long).expect("long");
    release.send(()).expect("load");
    let short_usage = usage(&collect(short_events));
    let long_usage = usage(&collect(long_events));
    assert_eq!(
        short_usage.prefill_pieces.count, 1,
        "short slot: {short_usage:?}"
    );
    assert_eq!(short_usage.prefill_pieces.max_rows, 1);
    assert_eq!(
        long_usage.prefill_pieces.count, 3,
        "long slot: {long_usage:?}"
    );
    assert_eq!(long_usage.prefill_pieces.max_rows, 4);
    engine.shutdown();
}

#[test]
fn a_batch_jobs_first_token_margin_rides_the_slot_usage() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    // The fake draws from [0,0,0]: the top two tie, so the latched margin is zero.
    fake.draws = true;
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[7, 7, 7, 7, 7], 1);
    engine.handle().try_submit(first).expect("submit");
    release.send(()).expect("load");
    let events = collect(first_events);
    assert_eq!(
        usage(&events).first_token_margin,
        Some(0.0),
        "the served margin rides the slot usage: {events:?}"
    );
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
    // The one-shot admission denial is retried after the eviction, so the job runs
    // instead of failing - a cached entry must not block session allocation.
    assert_eq!(usage(&collect(denied_events)).completion_tokens, 1);
    assert_eq!(usage(&collect(third_events)).completion_tokens, 1);
    // With max_seqs 2 the retried job takes the second slot of the first window; the third
    // job decodes in the next window.
    assert_eq!(lock(&log).windows[0], vec![vec![0], vec![1]]);
    assert_eq!(lock(&log).windows[1], vec![vec![2]]);
    engine.shutdown();
}

#[test]
fn a_cancelled_session_admission_denial_spends_no_checkpoint() {
    let log = Log::default();
    let (running, running_events) = job(&[0], 1);
    let (cancelled, cancelled_events) = job(&[1], 1);
    let mut fake = WindowFake::new(&log);
    fake.deny_session = true;
    fake.cancel_denied_session = Some(Arc::clone(&cancelled.generation.cancel));
    let (engine, release) = start(fake);
    engine.handle().try_submit(running).expect("submit running");
    engine
        .handle()
        .try_submit(cancelled)
        .expect("submit cancelled");
    release.send(()).expect("load");
    assert_eq!(finish(&collect(cancelled_events)), DecodeFinish::Cancelled);
    assert_eq!(usage(&collect(running_events)).completion_tokens, 1);
    engine.shutdown();
    // A job that was already cancelled when its admission was denied must not evict
    // another conversation's checkpoint to retry an admission that can no longer matter.
    let trace = lock(&log);
    assert_eq!(
        trace.calls.iter().filter(|call| **call == "evict").count(),
        0,
        "a cancelled denial evicted a checkpoint: {:?}",
        trace.calls
    );
    assert_eq!(
        trace
            .calls
            .iter()
            .filter(|call| **call == "new_session")
            .count(),
        1
    );
}

#[test]
fn a_retried_slots_later_denial_evicts_again_and_finishes() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    // One admission denial (spent on the retry that admits the job) and one prefill
    // denial in the retried slot: under the incremental contract the later denial
    // evicts one more entry and restarts the slot instead of erroring - the first
    // denial freed one entry, and the next may need only one more. The fake's failure
    // is one-shot, so the restarted prefill admits and the job finishes.
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
    // denied-and-retried one. The retried slot must clear the points build_slot set up: a
    // job whose prompt would otherwise hit its tip save produces no save call at all.
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
    // (the marker token targets only that job): the restart evicts one more entry, the next
    // denial finds the budget spent - `None` - and only then is the refusal final, so the
    // engine must classify it against the pool's honest ceiling instead of leaking the raw
    // memory-plan error. The other job prefills untouched.
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
    // no re-feed - against the pool's honest ceiling (the base plus the two freed
    // entries) instead of leaking the raw memory-plan error.
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
    // An exhausted store (no evictable entry) whose denial is recoverable - a save
    // charge the no-save retry avoids, with ceiling to spare - must NOT be
    // classified away by the terminal branch: the single no-save restart finishes
    // the prompt, exactly like the single worker's gated recovery.
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
    // The batch prefill restart must SIZE the eviction first, like the single worker. A
    // 3-entry deficit (the fake's ceiling needs three freed entries before it admits
    // the 6-token prompt), a ONE-piece denial budget, and a SURPLUS eviction budget of
    // five - the pre-pass stops the moment the ceiling admits, so exactly three
    // evictions happen and the two spare entries stay. The note trace pins it: from the
    // denied piece, exactly three evicts, ONE reset, then the successful re-feed. A
    // no-pre-pass mutant evicts once before the reset; a drain-to-None mutant spends
    // all five and never reaches a reset; a setup-twice mutant shows a second reset
    // before the re-feed.
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
    // A store already exhausted whose denial the ceiling does NOT explain (the
    // marker budget denies every carrying piece forever, the ceiling 100 fits the
    // 5-token prompt with room) must not restart forever: the terminal branch
    // takes the ONE no-save recovery, and the SECOND exhausted denial classifies
    // even though the ceiling fits - exactly one reset after the denied piece,
    // then the raw memory-plan error. The unbounded mutant restarts once per tick,
    // never closes the job, and fails collect's deadline.
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
    // The ceiling-less twin of the test above: `None` never proves a fit
    // impossible, so only the once-only exhaustion bound stops the spin - one
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
fn cancellation_during_session_admission_spends_no_cache_entry() {
    let log = Log::default();
    let (first, first_events) = job(&[0], 2);
    let (second, second_events) = job(&[1], 1);
    let mut fake = WindowFake::new(&log);
    fake.deny_session = true;
    fake.cancel_denied_session = Some(Arc::clone(&second.generation.cancel));
    let (engine, release) = start(fake);
    engine.handle().try_submit(first).unwrap();
    engine.handle().try_submit(second).unwrap();
    release.send(()).unwrap();
    assert_eq!(finish(&collect(second_events)), DecodeFinish::Cancelled);
    assert_eq!(usage(&collect(first_events)).completion_tokens, 2);
    engine.shutdown();
    let trace = lock(&log);
    assert!(!trace.calls.iter().any(|call| call == "evict"));
    assert_eq!(
        trace
            .calls
            .iter()
            .filter(|call| *call == "new_session")
            .count(),
        1
    );
}

#[test]
fn a_cancelled_growth_denial_returns_its_permits_and_keeps_an_independent_checkpoint() {
    for max_seqs in [1, 2] {
        let (incoming, cancelled) = job(&[1; 12], 1);
        let mut fake = FakeExecutor::new(vec![vec![0]; 3], &[EOS], 32);
        fake.with_session_budget(12, 1, 4);
        fake.with_prefix_limits(0, 2);
        fake.with_max_seqs(max_seqs);
        fake.cancel_on_prefill_denial = Some(Arc::clone(&incoming.generation.cancel));
        let engine = spawn_engine(4, move |_| Ok((fake, FakeSession::default()))).unwrap();
        let handle = engine.handle();

        let (seed, seed_events) = job(&[2; 3], 1);
        handle
            .try_submit(seed.with_prefix_boundary(2).unwrap())
            .unwrap();
        assert_eq!(usage(&collect(seed_events)).completion_tokens, 1);

        // The checkpoint holds four bytes. Two four-row grants fit for A before
        // the third is denied. Cancellation arrives inside that failed call.
        handle.try_submit(incoming).unwrap();
        let events = collect(cancelled);

        // A's eight bytes must be available again: B restores its two-token
        // prefix and uses the remaining eight bytes for seven prompt rows and decode.
        let (follow, follow_events) = job(&[2; 9], 1);
        handle
            .try_submit(follow.with_prefix_boundary(2).unwrap())
            .unwrap();
        let follow = collect(follow_events);
        assert_eq!(output(&follow), "a");
        assert_eq!(
            usage(&follow).reused_tokens,
            2,
            "lost checkpoint at max_seqs={max_seqs}"
        );
        assert_eq!(finish(&events), DecodeFinish::Cancelled);
        assert_eq!(usage(&events).completion_tokens, 0);
        assert_eq!(engine.shutdown().state, EngineState::Stopped);
    }
}

#[test]
fn a_denial_restart_restores_the_jobs_own_tip_in_both_workers() {
    for max_seqs in [1, 8] {
        assert_denial_restart_restores_tip(max_seqs);
    }
}

fn assert_denial_restart_restores_tip(max_seqs: usize) {
    // Use a real store and pool so the second turn must evict and re-restore its own
    // tip. This does not discriminate a missing protect ID: restore also makes the tip MRU.
    let (release, held) = mpsc::channel();
    let mut fake = FakeExecutor::new(vec![vec![EOS]; 6], &[EOS], 10_000);
    fake.with_max_seqs(max_seqs);
    fake.with_prefix_limits(2, 6);
    fake.with_session_budget(10_000_000, 1_024, 3_000_000);
    // Clone the window counter BEFORE the fake moves into the factory: a positive
    // count at the end asserts the BATCH branch ran - the single worker never
    // advances a decode window.
    let windows = fake.window_counter();
    let engine = spawn_engine(4, move |_| {
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
    assert_eq!(usage(&events).reused_tokens, 100, "max_seqs={max_seqs}");
    assert_eq!(finish(&events), DecodeFinish::Stop, "max_seqs={max_seqs}");
    assert_eq!(
        windows.load(Ordering::SeqCst) > 0,
        max_seqs > 1,
        "only the batch branch advances decode windows"
    );
    drop(handle);
    engine.shutdown();
}

#[test]
fn an_eviction_resets_the_exhaustion_bound_so_contention_keeps_its_restarts() {
    // This fixture pins the flag's PLACEMENT - set only on the EXHAUSTION path, never
    // on an eviction-backed restart. One evictable entry, no ceiling, a persistent
    // marker denial: the first denial spends the entry and restart #1 is
    // eviction-backed (flag stays clear); the second denial finds the store exhausted
    // and takes the ONE no-save recovery (flag SET); the third classifies. Correct
    // code: exactly TWO restart resets. A mutant that sets the flag on EVERY restart
    // classifies at the second denial - ONE reset - and fails this count (mutant
    // verified). It does not pin the reset: the eviction happens while the flag is
    // already clear, so deleting the successful-eviction RESET would also pass -
    // pinning a reset from an already-SET bound needs a fake whose store repopulates
    // mid-flight (an exhausted recovery, then another slot saves an entry, then
    // spending it refreshes eligibility); that pin is left to a later change.
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
    // The adoption-denial retry on the full engine path (batch worker, real store): the
    // fake's `deny_restore_once` knob returns ONE adoption MemoryPlan from the restore
    // closure - the permit the resident executor charges for the whole restored extent.
    // The store must NOT drop the candidate: it evicts one victim with the candidate
    // itself protected and retries the SAME candidate, so the turn reuses 100 tokens.
    // The turn's growth still fits the budget (the retry victim's release leaves
    // ~4,000,000 free for the ~3,481,600 tail): the SECOND victim is the SAVE-path
    // snapshot denial - the boundary save at 3,499 charges its own permits and evicts
    // one more entry - and the save replaces the restored tip@100 with the turn's own
    // deeper tip@3499 (the lineage rule). A no-retry mutant fails the reused assertion
    // (the drop-and-next finds no other candidate for this prompt, reused flips
    // 100 -> 0), and a no-protect mutant IS discriminated at this tip (verified by
    // mutant rerun): the candidate is the OLDEST entry, so without protection the
    // retry's eviction spends it - the FIRST failing assertion is the reused-tokens one
    // (reused flips 100 -> 0), the keys assertion fails after. The earlier
    // not-discriminated note predates the knob fix, when this fixture was not
    // exercising the adoption retry at all. The knob fires ONLY when a candidate is
    // actually tried: a candidate-less first-turn restore leaves it armed. The
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
    let engine = spawn_engine(4, move |_| {
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
    // The lineage won its denial. THREE removals, ONE counted: the adoption retry's
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
    // Counter pins: deleting either `+= 1` in session.rs must fail a test. This fixture:
    // no drop (the retry adopted), exactly one COUNTED eviction - the adoption retry's;
    // the save-path removal and the lineage replacement do not touch the counters. The
    // drop pin lives in the store-level twins.
    let (drops, evictions) = {
        let store = store.lock().expect("store lock");
        (store.restore_drop_count(), store.eviction_count())
    };
    assert_eq!(drops, 0, "the retried candidate was not dropped");
    assert_eq!(
        evictions, 1,
        "the adoption retry's victim, the only COUNTED one"
    );
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
    let capable_engine = spawn_engine_noted(4, capable_note, move |_| {
        Ok((capable, FakeSession::default()))
    })
    .expect("spawn");
    let (plain_tx, plain_rx) = mpsc::channel();
    let plain_note: ReadyNote = Some(Box::new(move |note| {
        if let EngineNote::Ready(capability) = note {
            let _ = plain_tx.send(capability);
        }
    }));
    let plain_engine = spawn_engine_noted(4, plain_note, move |_| {
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
    let failed_engine = spawn_engine_noted(4, fail_note, |_| {
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
    // The denied row waits one window and then advances: its client sees no error.
    let denied = collect(denied_events);
    assert_eq!(output(&denied), "aaa");
    assert_eq!(usage(&denied).completion_tokens, 3);
    let next = collect(next_events);
    assert_eq!(output(&next), "aa");
    assert_eq!(usage(&next).completion_tokens, 2);
    assert_eq!(lock(&log).windows[0], vec![vec![0], vec![1]]);
    assert_eq!(lock(&log).windows[1], vec![vec![0, 0], vec![1]]);
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
    // The cancelled peer's finish stands and the denied row, now alone, advances.
    let denied = collect(denied_events);
    assert_eq!(output(&denied), "aaa");
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

fn snapshot(engine: &Engine) -> super::TickSnapshot {
    engine.handle().shared.ticks.snapshot()
}

#[test]
fn ticks_count_the_rows_of_their_windows_and_prefill_pieces() {
    let log = Log::default();
    let (engine, release) = start(WindowFake::new(&log));
    let (first, first_events) = job(&[0, 0], 1);
    let (second, second_events) = job(&[1, 1, 1], 2);
    engine.handle().try_submit(first).expect("first");
    engine.handle().try_submit(second).expect("second");
    release.send(()).expect("load");
    collect(first_events);
    collect(second_events);
    let counts = snapshot(&engine);
    assert_eq!(counts.windows, 2, "{counts:?}");
    assert_eq!(counts.decode_rows, 3, "a window of two rows, then one");
    assert_eq!(counts.max_window, 2);
    assert_eq!(counts.prefill_rows, 5, "both prompts were fed");
    assert!(counts.ticks >= counts.windows);
    assert!(counts.executor_ns > 0, "{counts:?}");
    engine.shutdown();
}

#[test]
fn a_finished_session_serves_the_next_job_without_a_new_allocation() {
    let log = Log::default();
    let (engine, release) = start(WindowFake::new(&log));
    release.send(()).expect("load");
    for round in 0..3u32 {
        let (next, events) = job(&[round % 3], 2);
        engine.handle().try_submit(next).expect("submit");
        assert_eq!(usage(&collect(events)).completion_tokens, 2);
    }
    let counts = snapshot(&engine);
    assert_eq!(counts.sessions_created, 0, "{counts:?}");
    assert_eq!(
        counts.sessions_reused, 2,
        "the seed's first job is not a reuse: {counts:?}"
    );
    assert!(
        !lock(&log).calls.iter().any(|call| call == "new_session"),
        "the seed session served every job"
    );
    engine.shutdown();
}

#[test]
fn two_live_jobs_allocate_one_session_and_return_both_to_the_pool() {
    let log = Log::default();
    let armed = Arc::new(Mutex::new(None));
    let mut fake = WindowFake::new(&log);
    fake.armed_hold = Some(Arc::clone(&armed));
    let (engine, release) = start(fake);
    // Round one: both jobs are in the queue when the worker starts, so both are live together.
    let (first, first_events) = job(&[0], 3);
    let (second, second_events) = job(&[1], 3);
    engine.handle().try_submit(first).expect("first");
    engine.handle().try_submit(second).expect("second");
    release.send(()).expect("load");
    collect(first_events);
    collect(second_events);
    // Round two: the armed gate holds the worker in the first admission until the second job is
    // queued, so the pair is live together however the host schedules the threads.
    let (first_events, second_events) =
        submit_two_live(&engine, &armed, job(&[0], 3), job(&[1], 3));
    collect(first_events);
    collect(second_events);
    // The counters move at admission, before the events they belong to: nothing to wait for.
    let counts = snapshot(&engine);
    assert_eq!(counts.sessions_created, 1, "{counts:?}");
    assert_eq!(
        counts.sessions_reused + counts.sessions_created,
        3,
        "{counts:?}"
    );
    engine.shutdown();
}

/// A thousand cycles of admit, decode, finish and cancel at every stage must keep the engine
/// at its session cap: a session that leaked from the pool would make a later job allocate
/// beyond `max_seqs`.
#[test]
fn a_thousand_admit_and_cancel_cycles_never_allocate_past_the_cap() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.max = 3;
    fake.script_tokens = 9_000;
    let (engine, release) = start(fake);
    release.send(()).expect("load");
    for cycle in 0..1_000usize {
        let (first, first_events) = job(&[0, 1], 4);
        let (second, mut second_events) = job(&[1], 4);
        let (third, third_events) = job(&[2, 2, 2, 2, 2], 4);
        engine.handle().try_submit(first).expect("first");
        engine.handle().try_submit(second).expect("second");
        engine.handle().try_submit(third).expect("third");
        match cycle % 4 {
            // Cancelled before the worker may have looked at it.
            0 => second_events.cancel(),
            // Cancelled after its first batch.
            1 => loop {
                match second_events.events.try_recv() {
                    Ok(JobEvent::Batch(_)) => {
                        second_events.cancel();
                        break;
                    }
                    Ok(_) => {}
                    Err(_) => std::thread::yield_now(),
                }
            },
            // Cancelled while the queue still holds the third job.
            2 => third_events.cancel(),
            // Left to finish.
            _ => {}
        }
        let first = collect(first_events);
        collect(second_events);
        let third = collect(third_events);
        assert_eq!(usage(&first).completion_tokens, 4, "cycle {cycle}");
        assert!(
            matches!(
                finish(&third),
                DecodeFinish::Length | DecodeFinish::Cancelled
            ),
            "cycle {cycle}"
        );
    }
    let counts = snapshot(&engine);
    assert!(
        counts.sessions_created <= 2,
        "three slots hold the seed and at most two more: {counts:?}"
    );
    assert!(counts.sessions_reused >= 1_000, "{counts:?}");
    assert_eq!(engine.shutdown().state, EngineState::Stopped);
}

#[test]
fn the_sessions_of_a_failed_window_are_dropped_not_pooled() {
    let log = Log::default();
    let armed = Arc::new(Mutex::new(None));
    let mut fake = WindowFake::new(&log);
    fake.failure = Some(Failure::Local);
    fake.armed_hold = Some(Arc::clone(&armed));
    let (engine, release) = start(fake);
    // Both jobs are in the queue when the worker starts, so the first window has two rows.
    let (first, first_events) = job(&[0], 3);
    let (second, second_events) = job(&[1], 3);
    engine.handle().try_submit(first).expect("first");
    engine.handle().try_submit(second).expect("second");
    release.send(()).expect("load");
    for events in [first_events, second_events] {
        let events = collect(events);
        assert!(
            matches!(
                events.last(),
                Some(JobEvent::Error(EngError::MemoryPlan(_)))
            ),
            "{events:?}"
        );
    }
    let before = snapshot(&engine).sessions_created;
    // The next pair is live together: the armed gate holds the worker in the first admission
    // until the second job is queued.
    let (third_events, fourth_events) =
        submit_two_live(&engine, &armed, job(&[0], 2), job(&[1], 2));
    collect(third_events);
    collect(fourth_events);
    let after = snapshot(&engine).sessions_created;
    assert_eq!(after - before, 2, "both jobs built fresh sessions");
    engine.shutdown();
}

/// The longest run of prefill calls with no other executor call between them: the pieces one
/// tick fed for a cold prompt while a stream decoded.
fn longest_prefill_run(share: Option<usize>) -> usize {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.prompt_share = share;
    let (engine, release) = start(fake);
    let (stream, stream_events) = job(&[0], 40);
    let (cold, cold_events) = job(&[1; 24], 1);
    engine.handle().try_submit(stream).expect("stream");
    engine.handle().try_submit(cold).expect("cold");
    release.send(()).expect("load");
    collect(stream_events);
    collect(cold_events);
    engine.shutdown();
    let trace = lock(&log);
    let (mut longest, mut run) = (0, 0);
    for call in &trace.calls {
        if call.starts_with("prefill:") {
            run += 1;
            longest = longest.max(run);
        } else if call != "reset" && call != "boundary" {
            run = 0;
        }
    }
    longest
}

#[test]
fn a_prompt_share_feeds_several_pieces_in_one_tick_while_a_stream_decodes() {
    // The fake plans pieces of four rows. A cold prompt beside a decoding stream feeds one piece
    // per tick by the worker's rule; with a share of 24 rows it feeds the rest of its prompt in
    // one tick. The first tick has no stream decoding, so it feeds whole pieces either way.
    let default_run = longest_prefill_run(None);
    let share_run = longest_prefill_run(Some(24));
    assert!(default_run <= 2, "{default_run}");
    assert!(share_run >= 5, "{share_run}");
}

#[test]
fn a_share_below_the_planned_piece_still_feeds_one_whole_piece() {
    // A share of one row cannot slow a cold prompt below the piece the executor plans.
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.prompt_share = Some(1);
    let (engine, release) = start(fake);
    release.send(()).expect("load");
    let (stream, stream_events) = job(&[0], 30);
    let (cold, cold_events) = job(&[1; 16], 1);
    engine.handle().try_submit(stream).expect("stream");
    engine.handle().try_submit(cold).expect("cold");
    collect(stream_events);
    collect(cold_events);
    let counts = snapshot(&engine);
    engine.shutdown();
    assert_eq!(counts.prefill_rows, 17, "{counts:?}");
}

#[test]
fn a_share_smaller_than_the_cold_slots_still_gives_each_one_row() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.prompt_share = Some(1);
    fake.max = 3;
    let (engine, release) = start(fake);
    release.send(()).expect("load");
    let (first, first_events) = job(&[0, 1, 2], 1);
    let (second, second_events) = job(&[1, 1, 1], 1);
    engine.handle().try_submit(first).expect("first");
    engine.handle().try_submit(second).expect("second");
    assert_eq!(usage(&collect(first_events)).completion_tokens, 1);
    assert_eq!(usage(&collect(second_events)).completion_tokens, 1);
    engine.shutdown();
}

#[test]
fn a_job_waits_for_a_session_while_others_hold_the_capacity() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.max = 3;
    fake.session_cap = Some(2);
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[0], 4);
    let (second, second_events) = job(&[1], 4);
    let (third, third_events) = job(&[2], 3);
    for request in [first, second, third] {
        engine.handle().try_submit(request).expect("submit");
    }
    release.send(()).expect("load");
    // The third job finds no session while two are live: it queues, then takes the session a
    // finished job returns, and its client never sees a refusal.
    for (events, tokens) in [(first_events, 4), (second_events, 4), (third_events, 3)] {
        let events = collect(events);
        assert_eq!(usage(&events).completion_tokens, tokens, "{events:?}");
    }
    let new_sessions = lock(&log)
        .calls
        .iter()
        .filter(|call| *call == "new_session")
        .count();
    assert!(
        new_sessions >= 2,
        "the third job asked and was refused, then waited"
    );
    assert_eq!(snapshot(&engine).sessions_created, 1);
    engine.shutdown();
}

#[test]
fn a_lone_job_that_cannot_get_a_session_is_refused() {
    // A failed window drops its sessions, so the pool is empty and the device holds no more: the
    // next job is alone, nothing live can free memory for it, and the denial is final.
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.max = 2;
    fake.session_cap = Some(2);
    fake.evictions = 0;
    fake.failure = Some(Failure::Local);
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[0], 3);
    let (second, second_events) = job(&[1], 3);
    engine.handle().try_submit(first).expect("first");
    engine.handle().try_submit(second).expect("second");
    release.send(()).expect("load");
    for events in [first_events, second_events] {
        assert!(matches!(
            collect(events).last(),
            Some(JobEvent::Error(EngError::MemoryPlan(_)))
        ));
    }
    let (alone, alone_events) = job(&[2], 3);
    engine.handle().try_submit(alone).expect("alone");
    assert_eq!(
        collect(alone_events),
        vec![JobEvent::Error(EngError::MemoryPlan(
            "the device holds no more sessions".into()
        ))]
    );
    engine.shutdown();
}

#[test]
fn a_denied_row_waits_for_its_peer_and_a_lone_denied_row_is_refused() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.deny_slot = Some(1);
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[0], 3);
    let (denied, denied_events) = job(&[1], 3);
    engine.handle().try_submit(first).expect("first");
    engine.handle().try_submit(denied).expect("denied");
    release.send(()).expect("load");
    // The peer advances while the second row is denied every window; once the peer has
    // finished the denied row is alone in slot 0, which the fake does not deny, and finishes.
    assert_eq!(usage(&collect(first_events)).completion_tokens, 3);
    assert_eq!(usage(&collect(denied_events)).completion_tokens, 3);
    engine.shutdown();
    // Every row denied: no row advances, so the denial is final for all of them.
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.deny_slot = Some(0);
    let (engine, release) = start(fake);
    release.send(()).expect("load");
    let (only, only_events) = job(&[0], 3);
    engine.handle().try_submit(only).expect("only");
    assert_eq!(
        collect(only_events),
        vec![JobEvent::Error(EngError::MemoryPlan(
            "row growth denied".into()
        ))]
    );
    engine.shutdown();
}

#[test]
fn an_idle_engine_releases_its_pooled_sessions_but_one() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.max = 3;
    fake.idle_trim = Some(Duration::from_millis(60));
    let (engine, release) = start(fake);
    let mut events = Vec::new();
    for round in 0..3u32 {
        let (next, next_events) = job(&[round], 6);
        engine.handle().try_submit(next).expect("submit");
        events.push(next_events);
    }
    release.send(()).expect("load");
    for next_events in events {
        collect(next_events);
    }
    assert_eq!(
        snapshot(&engine).sessions_created,
        2,
        "three slots, the seed and two more"
    );
    wait(|| snapshot(&engine).sessions_trimmed == 2);
    // The engine still serves: the kept session takes the next job without an allocation.
    let (again, again_events) = job(&[0], 2);
    engine.handle().try_submit(again).expect("again");
    assert_eq!(usage(&collect(again_events)).completion_tokens, 2);
    assert_eq!(snapshot(&engine).sessions_created, 2);
    engine.shutdown();
}

#[test]
fn more_live_slots_than_the_budget_still_give_every_slot_a_row() {
    // The planned piece is four rows and six prompts are live: the budget is raised to one row
    // per live slot, so no prompt starves behind the others.
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.max = 6;
    let (engine, release) = start_with_budget(fake, 2);
    release.send(()).expect("load");
    let mut all = Vec::new();
    for index in 0..6u32 {
        let (next, events) = job(&[index % 3; 12], 2);
        engine.handle().try_submit(next).expect("submit");
        all.push(events);
    }
    for events in all {
        assert_eq!(usage(&collect(events)).completion_tokens, 2);
    }
    let counts = snapshot(&engine);
    assert_eq!(counts.prefill_rows, 72);
    engine.shutdown();
}

#[test]
fn a_cancelled_cold_prompt_is_not_fed_after_its_cancel() {
    // Two cold prompts take whole pieces in one idle tick. The first piece is held while the
    // second job is cancelled: the second prompt is never fed.
    let log = Log::default();
    let (hold, entered, unblock) = gate();
    let mut fake = WindowFake::new(&log);
    fake.hold_prefill = Some(hold);
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[0; 8], 1);
    let (second, second_events) = job(&[1; 8], 1);
    let cancel = Arc::clone(&second.generation.cancel);
    engine.handle().try_submit(first).expect("first");
    engine.handle().try_submit(second).expect("second");
    release.send(()).expect("load");
    entered.recv_timeout(WAIT).expect("entered");
    cancel.store(true, Ordering::SeqCst);
    unblock.send(()).expect("release");
    assert_eq!(usage(&collect(first_events)).completion_tokens, 1);
    assert_eq!(finish(&collect(second_events)), DecodeFinish::Cancelled);
    let fed = lock(&log)
        .calls
        .iter()
        .filter(|call| call.starts_with("prefill:"))
        .count();
    assert_eq!(fed, 2, "only the first prompt's two pieces were fed");
    engine.shutdown();
}

#[test]
fn the_pool_releases_sessions_by_their_own_idle_age() {
    let counters = super::TickCounters::default();
    let interval = Duration::from_millis(100);
    let now = Instant::now();
    let mut pool = super::Pool::new(0u32, &counters, Some(interval));
    // Three more sessions went idle at different times; the seed went idle just now.
    pool.give_at(1, now - Duration::from_millis(300));
    pool.give_at(2, now - Duration::from_millis(150));
    pool.give_at(3, now - Duration::from_millis(10));
    // With a live slot the sessions idle past the interval go, whatever the pool then holds.
    pool.trim(now, 0);
    let mut left: Vec<u32> = pool.free.iter().map(|pooled| pooled.session).collect();
    left.sort_unstable();
    assert_eq!(
        left,
        vec![0, 3],
        "the seed, which has just been built, and the session idle for 10 ms stay"
    );
    assert_eq!(counters.snapshot().sessions_trimmed, 2);
    // An idle engine keeps one session even when every one has idled past the interval.
    let mut pool = super::Pool::new(0u32, &counters, Some(interval));
    pool.give_at(1, now - Duration::from_millis(300));
    pool.give_at(2, now - Duration::from_millis(200));
    pool.trim(now + Duration::from_secs(1), 1);
    assert_eq!(pool.free.len(), 1);
    // A pool without an interval never releases.
    let mut pool = super::Pool::new(0u32, &counters, None);
    pool.give_at(1, now - Duration::from_secs(60));
    pool.trim(now, 0);
    assert_eq!(pool.free.len(), 2);
}

/// Every event of a job, `Started` included, until its channel closes.
fn raw_events(mut handle: JobHandle) -> Vec<JobEvent> {
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

/// An engine of two sessions that the device can hold one of: the first job (`long_tokens` of
/// output, a window every 4 ms) holds it, so a second job finds no session and waits. `armed` is
/// the gate the fake takes at its next window, for a test that freezes the worker there.
fn waiting_engine(
    log: &Log,
    long_tokens: u32,
    armed: &Arc<Mutex<Option<Gate>>>,
) -> (
    Engine,
    JobHandle,
    JobHandle,
    Arc<std::sync::atomic::AtomicBool>,
) {
    let mut fake = WindowFake::new(log);
    fake.max = 2;
    fake.session_cap = Some(1);
    fake.evictions = 0;
    fake.script_tokens = (long_tokens as usize).max(fake.script_tokens);
    fake.armed_window = Some(Arc::clone(armed));
    fake.window_delay = Some(Duration::from_millis(4));
    let (engine, release) = start(fake);
    let (long, long_events) = job(&[0], long_tokens);
    let (waiter, waiter_events) = job(&[1], 3);
    let cancel = Arc::clone(&waiter.generation.cancel);
    engine.handle().try_submit(long).expect("long");
    engine.handle().try_submit(waiter).expect("waiter");
    release.send(()).expect("load");
    (engine, long_events, waiter_events, cancel)
}

/// Whether a job has not yet ended, from the events it has sent so far. It reads without
/// waiting and consumes the events it reads, never the terminal one.
fn still_running(handle: &mut JobHandle) -> bool {
    loop {
        match handle.events.try_recv() {
            Ok(JobEvent::Finished(_) | JobEvent::Error(_)) => return false,
            Ok(_) => {}
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => return false,
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => return true,
        }
    }
}

/// Waits until the refused job is waiting: the fake has refused it a session and the worker has
/// gone on to a window, so the admission, including any eviction retries, is over. A caller that
/// acted on the first refusal alone would race the admission loop, which a cancel or a freed
/// session ends early.
fn wait_until_waiting(log: &Log) {
    wait(|| {
        let calls = &lock(log).calls;
        calls
            .iter()
            .position(|call| call == "new_session_refused")
            .is_some_and(|at| calls[at..].iter().any(|call| call == "window"))
    });
}

#[test]
fn a_waiting_job_counts_as_queued_and_gets_one_started_event() {
    let log = Log::default();
    let (engine, long_events, waiter_events, _cancel) = waiting_engine(&log, 600, &Arc::default());
    // Queued is also 1 while the job still sits in the channel: wait until it has been taken
    // out, refused and made to wait, and only then ask.
    wait_until_waiting(&log);
    assert_eq!(
        engine.handle().queued(),
        1,
        "the waiting job keeps its ticket"
    );
    let waiter = raw_events(waiter_events);
    let started = waiter
        .iter()
        .filter(|event| matches!(event, JobEvent::Started))
        .count();
    assert_eq!(started, 1, "{waiter:?}");
    assert_eq!(
        usage(&waiter).completion_tokens,
        3,
        "it runs once a session frees"
    );
    assert_eq!(engine.handle().queued(), 0);
    collect(long_events);
    engine.shutdown();
}

#[test]
fn a_waiting_job_that_is_cancelled_ends_while_the_stream_still_runs() {
    let log = Log::default();
    let armed = Arc::new(Mutex::new(None));
    // The stream cannot finish inside the test: it is ended by its own cancel at the end.
    let (engine, mut long_events, mut waiter_events, cancel) = waiting_engine(&log, 12_000, &armed);
    wait_until_waiting(&log);
    // Freeze the worker inside a window, so that the cancel below is set while it cannot look.
    let (entered, observed) = mpsc::channel();
    let (release, held) = mpsc::channel();
    *lock(&armed) = Some((entered, held));
    observed
        .recv_timeout(WAIT)
        .expect("the worker is inside a window");
    cancel.store(true, Ordering::SeqCst);
    // Arm the next window, then let the worker go: it finishes this tick, and the next pass of
    // its loop must end the waiter before that pass's window. A cancel seen only at the
    // one-second retry would leave the waiter alive at the next window.
    let (entered_next, observed_next) = mpsc::channel();
    let (release_next, held_next) = mpsc::channel();
    *lock(&armed) = Some((entered_next, held_next));
    release.send(()).expect("release the first window");
    observed_next
        .recv_timeout(WAIT)
        .expect("the worker reached the next window");
    // The worker is frozen again, and the event of the waiter's end was sent before it got here.
    let mut ended = Vec::new();
    while let Ok(event) = waiter_events.events.try_recv() {
        ended.push(event);
    }
    assert!(
        ended
            .iter()
            .any(|event| matches!(event, JobEvent::Finished(_) | JobEvent::Error(_))),
        "the cancel waited for the retry: {ended:?}"
    );
    assert_eq!(finish(&ended), DecodeFinish::Cancelled);
    release_next.send(()).expect("release the second window");
    // Its ticket goes when the worker drops the job it held, just after the event.
    wait(|| engine.handle().queued() == 0);
    // The stream it waited behind is still decoding.
    assert!(still_running(&mut long_events));
    long_events.cancel();
    collect(long_events);
    engine.shutdown();
}

#[test]
fn a_waiting_job_is_cancelled_when_the_engine_shuts_down() {
    let log = Log::default();
    let (engine, long_events, waiter_events, _cancel) = waiting_engine(&log, 600, &Arc::default());
    wait(|| engine.handle().queued() == 1);
    engine.shutdown();
    let waiter = raw_events(waiter_events);
    assert_eq!(finish(&waiter), DecodeFinish::Cancelled);
    drop(long_events);
}

#[test]
fn a_job_refused_for_memory_asks_again_when_memory_returns_with_every_slot_live() {
    let log = Log::default();
    let live = Arc::new(std::sync::atomic::AtomicUsize::new(1));
    let mut fake = WindowFake::new(&log);
    fake.max = 2;
    fake.live_cap = Some(Arc::clone(&live));
    fake.evictions = 0;
    fake.script_tokens = 12_000;
    fake.window_delay = Some(Duration::from_millis(5));
    let (engine, release) = start(fake);
    // The stream cannot finish inside the test: it is ended by its own cancel at the end.
    let (long, mut long_events) = job(&[0], 12_000);
    let (waiter, waiter_events) = job(&[1], 3);
    engine.handle().try_submit(long).expect("long");
    engine.handle().try_submit(waiter).expect("waiter");
    release.send(()).expect("load");
    wait_until_waiting(&log);
    // The device lets go of memory; no slot frees. The waiting job's timed retry finds it.
    live.store(2, Ordering::SeqCst);
    let waiter = collect(waiter_events);
    assert_eq!(usage(&waiter).completion_tokens, 3);
    // The stream is still running: without the timer the waiter would only have run when it ended.
    assert!(
        still_running(&mut long_events),
        "the waiter ran only after the stream ended"
    );
    long_events.cancel();
    collect(long_events);
    engine.shutdown();
}

#[test]
fn a_stop_during_a_piece_ends_the_tick_before_the_next_prompt() {
    // Two cold prompts take whole pieces in one idle tick. The first piece is held while the stop
    // flag is raised without cancelling any job (the window in which a shutdown has stored the
    // flag and not yet cancelled the jobs): the second prompt is not fed.
    let log = Log::default();
    let (hold, entered, unblock) = gate();
    let mut fake = WindowFake::new(&log);
    fake.hold_prefill = Some(hold);
    let (engine, release) = start(fake);
    let (first, first_events) = job(&[0; 8], 1);
    let (second, second_events) = job(&[1; 8], 1);
    engine.handle().try_submit(first).expect("first");
    engine.handle().try_submit(second).expect("second");
    release.send(()).expect("load");
    entered.recv_timeout(WAIT).expect("entered");
    engine.handle().shared.stop.store(true, Ordering::SeqCst);
    unblock.send(()).expect("release");
    // The worker ends on its own; no shutdown call cancels the jobs and hides the difference.
    wait(|| engine.handle().state() == EngineState::Stopped);
    let fed = lock(&log)
        .calls
        .iter()
        .filter(|call| call.starts_with("prefill:"))
        .count();
    assert_eq!(fed, 1, "only the held piece ran");
    engine.shutdown();
    drop((first_events, second_events));
}

#[test]
fn a_session_whose_setup_failed_is_not_pooled_again() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.fail_reset_call = Some((0, EngError::BadRequest("the reset was refused".into())));
    let (engine, release) = start(fake);
    release.send(()).expect("load");
    let (first, first_events) = job(&[0], 2);
    engine.handle().try_submit(first).expect("first");
    assert!(matches!(
        collect(first_events).last(),
        Some(JobEvent::Error(EngError::BadRequest(_)))
    ));
    let (second, second_events) = job(&[1], 2);
    engine.handle().try_submit(second).expect("second");
    assert_eq!(usage(&collect(second_events)).completion_tokens, 2);
    // The seed that failed its reset is gone: the second job got a session of its own.
    assert_eq!(snapshot(&engine).sessions_created, 1);
    engine.shutdown();
}

#[test]
fn an_executor_that_does_not_opt_into_the_trim_keeps_one_session_after_a_burst() {
    // Four jobs at once need four sessions (the seed and three more). Their sessions come back as
    // the jobs finish: the executor keeps one and the other three are released at once.
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.max = 4;
    fake.default_policy = true;
    let (engine, release) = start(fake);
    let mut all = Vec::new();
    for index in 0..4u32 {
        let (next, events) = job(&[index], 4);
        engine.handle().try_submit(next).expect("submit");
        all.push(events);
    }
    release.send(()).expect("load");
    for events in all {
        collect(events);
    }
    wait(|| snapshot(&engine).sessions_dropped == 3);
    let counts = snapshot(&engine);
    assert_eq!(counts.sessions_created, 3, "{counts:?}");
    let held = 1 + counts.sessions_created - counts.sessions_dropped - counts.sessions_trimmed;
    assert_eq!(held, 1, "{counts:?}");
    // The kept session serves the next job without an allocation.
    let (again, again_events) = job(&[0], 2);
    engine.handle().try_submit(again).expect("again");
    assert_eq!(usage(&collect(again_events)).completion_tokens, 2);
    assert_eq!(snapshot(&engine).sessions_created, 3);
    engine.shutdown();
}

#[test]
fn a_retry_that_cannot_fit_does_not_allocate_and_one_that_can_does() {
    // While the device is full, the one-second retries skip the allocation: only the first
    // attempt, which found the refusal, called new_session. When room returns the job is admitted
    // at the next retry with the stream still running.
    let log = Log::default();
    let live = Arc::new(std::sync::atomic::AtomicUsize::new(1));
    let fits = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut fake = WindowFake::new(&log);
    fake.max = 2;
    fake.live_cap = Some(Arc::clone(&live));
    fake.fits = Some(Arc::clone(&fits));
    // No prefix entry to evict, so the first refusal is a single allocation attempt.
    fake.evictions = 0;
    fake.script_tokens = 12_000;
    fake.window_delay = Some(Duration::from_millis(5));
    let (engine, release) = start(fake);
    let (long, mut long_events) = job(&[0], 12_000);
    let (waiter, waiter_events) = job(&[1], 3);
    engine.handle().try_submit(long).expect("long");
    engine.handle().try_submit(waiter).expect("waiter");
    release.send(()).expect("load");
    wait_until_waiting(&log);
    std::thread::sleep(Duration::from_millis(2_300));
    let asked = lock(&log)
        .calls
        .iter()
        .filter(|call| *call == "new_session")
        .count();
    assert_eq!(
        asked, 1,
        "two retry intervals passed without another allocation"
    );
    live.store(2, Ordering::SeqCst);
    fits.store(true, Ordering::SeqCst);
    assert_eq!(usage(&collect(waiter_events)).completion_tokens, 3);
    assert!(still_running(&mut long_events));
    long_events.cancel();
    collect(long_events);
    engine.shutdown();
}

#[test]
fn an_executor_whose_prefix_entries_do_not_share_session_memory_is_not_made_to_evict() {
    // A lone job after a failed window finds the pool empty and the device full. With entries that
    // share the session budget the worker evicts before it reports the denial; with host-memory
    // entries it does not, and they stay.
    let evictions = |frees: bool| {
        let log = Log::default();
        let mut fake = WindowFake::new(&log);
        fake.max = 2;
        fake.session_cap = Some(2);
        fake.failure = Some(Failure::Local);
        fake.eviction_frees = frees;
        let (engine, release) = start(fake);
        let (first, first_events) = job(&[0], 3);
        let (second, second_events) = job(&[1], 3);
        engine.handle().try_submit(first).expect("first");
        engine.handle().try_submit(second).expect("second");
        release.send(()).expect("load");
        collect(first_events);
        collect(second_events);
        let (alone, alone_events) = job(&[2], 3);
        engine.handle().try_submit(alone).expect("alone");
        assert!(matches!(
            collect(alone_events).last(),
            Some(JobEvent::Error(EngError::MemoryPlan(_)))
        ));
        let evicted = lock(&log)
            .calls
            .iter()
            .filter(|call| *call == "evict")
            .count();
        engine.shutdown();
        evicted
    };
    assert!(evictions(true) > 0, "the default evicts before it refuses");
    assert_eq!(
        evictions(false),
        0,
        "host-memory entries are not evicted for a device shortage"
    );
}

#[test]
fn the_default_batch_policy_is_the_shared_budget_and_no_trim() {
    use crate::family::executor::BatchPolicy;
    let policy = BatchPolicy::default();
    assert_eq!(
        policy.idle_pieces, 1,
        "one shared piece per tick while idle"
    );
    assert_eq!(policy.idle_trim, None, "no trim");
    // An executor that overrides nothing answers the default: the fake executor of the seam tests
    // and a bare one.
    let fake = FakeExecutor::new(vec![], &[], 1);
    assert_eq!(fake.batch_policy(), policy);
    assert!(fake.can_admit_session());
    assert!(fake.eviction_frees_session_memory());
}

#[test]
fn an_idle_batching_engine_advances_background_work_and_drains_it_at_shutdown() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.background_want = 3;
    let (steps, drained) = (
        Arc::clone(&fake.background_steps),
        Arc::clone(&fake.background_drained),
    );
    let (engine, release) = start(fake);
    release.send(()).expect("load");
    wait(|| steps.load(Ordering::SeqCst) == 3);
    assert!(!drained.load(Ordering::SeqCst), "no drain before shutdown");
    // A job still runs between steps.
    let (request, events) = job(&[0], 2);
    engine.handle().try_submit(request).expect("submit");
    assert!(matches!(
        collect(events).last(),
        Some(JobEvent::Finished(_))
    ));
    let report = engine.shutdown();
    assert_ne!(report.state, EngineState::NativeFailed);
    assert!(
        drained.load(Ordering::SeqCst),
        "a batching engine drains its background work at shutdown"
    );
}

#[test]
fn a_batching_engine_steps_background_work_between_ticks_while_streams_decode() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.background_want = usize::MAX;
    fake.script_tokens = 40;
    let (engine, release) = start(fake);
    let (request, events) = job(&[0], 30);
    engine.handle().try_submit(request).expect("submit");
    release.send(()).expect("load");
    assert!(matches!(
        collect(events).last(),
        Some(JobEvent::Finished(_))
    ));
    engine.shutdown();
    let calls = lock(&log).calls.clone();
    let first = calls
        .iter()
        .position(|call| call == "window")
        .expect("a window");
    let last = calls
        .iter()
        .rposition(|call| call == "window")
        .expect("a window");
    assert!(
        calls[first..last]
            .iter()
            .any(|call| call == "background_step"),
        "no background step ran between two windows of the stream: {calls:?}"
    );
}

#[test]
fn a_natively_failed_batching_engine_does_not_drain_background_work() {
    let log = Log::default();
    let mut fake = WindowFake::new(&log);
    fake.failure = Some(Failure::Native);
    fake.background_want = 0;
    let drained = Arc::clone(&fake.background_drained);
    let (engine, release) = start(fake);
    let (request, events) = job(&[0], 2);
    engine.handle().try_submit(request).expect("submit");
    release.send(()).expect("load");
    assert!(matches!(
        collect(events).as_slice(),
        [JobEvent::Error(EngError::Backend { .. })]
    ));
    assert_eq!(engine.shutdown().state, EngineState::NativeFailed);
    assert!(
        !drained.load(Ordering::SeqCst),
        "a failed engine's device state is not read at shutdown"
    );
}
