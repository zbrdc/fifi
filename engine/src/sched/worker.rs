//! The engine worker: one OS thread per engine, one generation at a time.
//!
//! Design: the scheduler design ("Threads and channels", "Engine lifecycle") and
//! the family descriptor design ("Executor trait", the worker call order).
//! The worker is generic over `ExecutorOps`, so Linux tests run it on the scripted fake.
//!
//! Panic safety: the whole thread body runs under one `catch_unwind`. The executor, its session,
//! and the current job live on the engine thread, so an unwind drops them there. A drop guard
//! could not send the best-effort `Backend` event to the current job, because it would need the
//! job's sender and the phase. A panic while the engine loads stores `Error`. A panic at any
//! later point stores `NativeFailed`. The worker makes no further executor call after a panic,
//! sends one `Backend` event to the current job, rejects the queued jobs, and records the cause.
//!
//! Failed engines: a failed load (`Error`), a panic, and a native failure (`NativeFailed`) all
//! give the same outcome to the server. Queued jobs get a `Backend` event, and later submits get
//! `SubmitError::LoadFailed` or `SubmitError::NativeFailed`. Both map to `Backend` (HTTP 500
//! `backend_error`, `server.md` "Native failure is process-scoped"), because `EngError` has no
//! `Overloaded` variant for an event and a failed engine does not recover by retry.
//!
//! Disconnect: `JobHandle` cancels its job when it drops. A queued job whose owner has gone
//! never reaches the executor.
//!
//! Stop: `Engine::shutdown` cancels the running job and every queued job and does not wait for
//! other `EngineHandle` clones to drop (`scheduler.md`: a stopped engine cancels its jobs).

use super::job::{
    Job, JobEvent, JobUsage, PrefillPieces, QUEUE_CAPACITY, send_cancelled, send_data,
    send_terminal,
};
use crate::EngError;
use crate::family::executor::{ExecutorOps, GenerationState, PrefillPairOutcome};
use crate::family::mlx_report::LoadedMlxExecution;
use crate::family::session::{SaveKind, StoreLimits};
use std::any::Any;
use std::ops::ControlFlow;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;
use tokio::sync::mpsc;

#[path = "batch_worker.rs"]
mod batch_worker;

/// The state of one engine. The values follow the `Engine.state` order in `server.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum EngineState {
    Loading = 0,
    Ready = 1,
    Busy = 2,
    Error = 3,
    Stopped = 4,
    NativeFailed = 5,
}

impl EngineState {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => EngineState::Loading,
            1 => EngineState::Ready,
            2 => EngineState::Busy,
            4 => EngineState::Stopped,
            5 => EngineState::NativeFailed,
            _ => EngineState::Error,
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The state that the engine thread, its handles, and `shutdown` share.
/// Whether retrying can succeed, as far as the recording site knows.
/// `AfterCorrection`: the cause sits in files, configuration, or free
/// resources, so fixing it and starting again is the retry (this build
/// has no in-place engine restart). `Unknown`: no evidence either way;
/// never presented as safely retryable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RetryClass {
    AfterCorrection,
    Unknown,
}

/// The structured form of the first recorded engine failure. `message`
/// keeps the exact historical cause text (logs and shutdown reports are
/// unchanged); `code` is the one vocabulary shared with HTTP errors
/// (`EngError::code`, plus `internal_error` for panics and broken
/// executor invariants, which no `EngError` describes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EngineFailure {
    pub(crate) code: &'static str,
    /// Where it failed: `load`, `admission`, `configuration`, `runtime`,
    /// `internal`, or a backend operation name (for example
    /// `cuda initialize`).
    pub(crate) stage: String,
    pub(crate) message: String,
    pub(crate) retry: RetryClass,
}

impl EngineFailure {
    /// Classify an engine error at its recording site. `phase` is the
    /// stage for variants that do not name their own (`load` while the
    /// executor is built, `runtime` afterwards).
    pub(crate) fn from_error(error: &EngError, phase: &'static str, message: String) -> Self {
        let stage = match error {
            EngError::Backend { operation, .. } => operation.to_string(),
            EngError::MemoryPlan(_) => "admission".to_string(),
            EngError::BadRequest(_) | EngError::ContextLimit(_) => "configuration".to_string(),
            _ => phase.to_string(),
        };
        let retry = match error {
            EngError::Backend { .. } | EngError::ModelOutput(_) => RetryClass::Unknown,
            _ => RetryClass::AfterCorrection,
        };
        Self {
            code: error.code(),
            stage,
            message,
            retry,
        }
    }

    /// A failure no `EngError` describes: a panic, or an executor
    /// invariant broken at `stage`.
    fn internal(stage: &'static str, message: String) -> Self {
        Self {
            code: "internal_error",
            stage: stage.to_string(),
            message,
            retry: RetryClass::Unknown,
        }
    }
}

struct Shared {
    state: AtomicU8,
    /// The first recorded cause of `Error` or `NativeFailed`, for `/ready` and the logs.
    failure: Mutex<Option<EngineFailure>>,
    /// The loading stage the engine thread last announced, or `None`
    /// once it has settled (ready or failed). Stages are names of the
    /// phases the family's own start code actually performs — no
    /// percentages, no invented sequence.
    load_stage: Mutex<Option<&'static str>>,
    mlx_execution: OnceLock<LoadedMlxExecution>,
    effective: OnceLock<Vec<crate::capability::effective::EffectiveSetting>>,
    /// Set by `shutdown`. The worker cancels every job that it has not finished.
    stop: AtomicBool,
    /// The cancel flag of the running job, so `shutdown` can cancel it.
    running: Mutex<Option<Arc<AtomicBool>>>,
    /// Opt-in window jobs only. The legacy single-slot path never registers here.
    active: Mutex<Vec<ActiveJob>>,
    /// Jobs being submitted or waiting to be dequeued. In-progress submissions
    /// are included conservatively, even if the channel later refuses them.
    queued: AtomicUsize,
    /// Set once the built executor reports a prefix store of no roots and no entries: a prompt
    /// boundary sent to such an engine can neither be saved nor restored.
    no_prefix_store: AtomicBool,
    /// What the batch worker's ticks and session pool did, as counters a reader snapshots.
    ticks: batch_worker::TickCounters,
}

struct ActiveJob {
    cancel: Arc<AtomicBool>,
    events: mpsc::Sender<JobEvent>,
}

impl Shared {
    fn new() -> Self {
        Shared {
            state: AtomicU8::new(EngineState::Loading as u8),
            failure: Mutex::new(None),
            load_stage: Mutex::new(None),
            mlx_execution: OnceLock::new(),
            effective: OnceLock::new(),
            stop: AtomicBool::new(false),
            running: Mutex::new(None),
            active: Mutex::new(Vec::new()),
            queued: AtomicUsize::new(0),
            no_prefix_store: AtomicBool::new(false),
            ticks: batch_worker::TickCounters::default(),
        }
    }

    fn state(&self) -> EngineState {
        EngineState::from_u8(self.state.load(Ordering::SeqCst))
    }

    fn set_state(&self, state: EngineState) {
        self.state.store(state as u8, Ordering::SeqCst);
    }

    /// Keeps the first cause: later failures are consequences of it.
    fn record_failure(&self, cause: EngineFailure) {
        let mut slot = lock(&self.failure);
        if slot.is_none() {
            *slot = Some(cause);
        }
    }

    /// The stage is part of the settled record, so it leaves with the
    /// state change: ready and failed engines report no loading stage.
    fn clear_load_stage(&self) {
        *lock(&self.load_stage) = None;
    }

    /// The cause text, exactly as the logs and shutdown reports have
    /// always carried it.
    fn failure(&self) -> Option<String> {
        lock(&self.failure).as_ref().map(|f| f.message.clone())
    }

    fn failure_info(&self) -> Option<EngineFailure> {
        lock(&self.failure).clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SubmitError {
    /// The server maps this to `Overloaded`.
    QueueFull,
    /// A native failure or an executor panic stopped the engine. The server maps this to `Backend`.
    NativeFailed,
    /// The server maps this to `Backend`, like `NativeFailed`.
    LoadFailed,
    /// The engine has stopped or is stopping. The server maps this to `Overloaded`.
    Unavailable,
}

/// One reservation exists from before publication until dequeue or drop.
/// This also accounts for refused sends and messages dropped with the receiver.
struct QueueTicket(Arc<Shared>);

impl QueueTicket {
    fn reserve(shared: &Arc<Shared>) -> Self {
        shared.queued.fetch_add(1, Ordering::SeqCst);
        Self(Arc::clone(shared))
    }
}

impl Drop for QueueTicket {
    fn drop(&mut self) {
        self.0.queued.fetch_sub(1, Ordering::SeqCst);
    }
}

// The sampler's capture handle grew Job past the large-variant floor; the
// queue stays flat rather than boxing every job.
#[allow(clippy::large_enum_variant)]
enum Message {
    Job(Job, QueueTicket),
    /// Ends the loop after the jobs ahead of it. `shutdown` sends it so that the thread stops
    /// while other handles are still alive.
    Stop,
}

fn publish_job(queue: &SyncSender<Message>, message: Message) -> Result<(), SubmitError> {
    queue.try_send(message).map_err(|error| match error {
        TrySendError::Full(_) => SubmitError::QueueFull,
        TrySendError::Disconnected(_) => SubmitError::Unavailable,
    })
}

/// The portable, shareable front of one engine thread.
#[derive(Clone)]
pub(crate) struct EngineHandle {
    queue: SyncSender<Message>,
    shared: Arc<Shared>,
}

impl EngineHandle {
    pub(crate) fn state(&self) -> EngineState {
        self.shared.state()
    }

    /// Whether the engine keeps a prefix store a prompt boundary can use. True until a built
    /// executor reports a store of no roots and no entries (a tiered Qwen3.6 start, for one):
    /// the chat handler then marks no boundary, exactly as when the config turns the flag off.
    pub(crate) fn keeps_prefix_store(&self) -> bool {
        !self.shared.no_prefix_store.load(Ordering::SeqCst)
    }

    /// The phase the engine's start code last announced, present only
    /// while the engine is still loading.
    pub(crate) fn load_stage(&self) -> Option<&'static str> {
        if self.state() != EngineState::Loading {
            return None;
        }
        *lock(&self.shared.load_stage)
    }

    pub(crate) fn failure(&self) -> Option<String> {
        self.shared.failure()
    }

    pub(crate) fn loaded_effective(
        &self,
    ) -> Option<&[crate::capability::effective::EffectiveSetting]> {
        self.shared.effective.get().map(Vec::as_slice)
    }

    pub(crate) fn mlx_execution(&self) -> Option<&LoadedMlxExecution> {
        self.shared.mlx_execution.get()
    }

    /// The structured failure, for authenticated operator surfaces.
    pub(crate) fn failure_info(&self) -> Option<EngineFailure> {
        self.shared.failure_info()
    }

    /// Queues a job without waiting. A refused job is dropped, and its channel closes.
    ///
    /// A job that a concurrent failure or stop catches in the queue can also end with a closed
    /// channel and no event. The owner then treats the closed channel as a `Backend` failure.
    pub(crate) fn try_submit(&self, job: Job) -> Result<(), SubmitError> {
        self.try_submit_with(job, publish_job)
    }

    fn try_submit_with(
        &self,
        job: Job,
        publish: impl FnOnce(&SyncSender<Message>, Message) -> Result<(), SubmitError>,
    ) -> Result<(), SubmitError> {
        if self.shared.stop.load(Ordering::SeqCst) {
            return Err(SubmitError::Unavailable);
        }
        match self.state() {
            EngineState::NativeFailed => return Err(SubmitError::NativeFailed),
            EngineState::Error => return Err(SubmitError::LoadFailed),
            EngineState::Stopped => return Err(SubmitError::Unavailable),
            EngineState::Loading | EngineState::Ready | EngineState::Busy => {}
        }
        let ticket = QueueTicket::reserve(&self.shared);
        publish(&self.queue, Message::Job(job, ticket))
    }

    /// Jobs waiting to be dequeued or currently attempting submission.
    /// This is a conservative observation, not an atomic idle fence.
    pub(crate) fn queued(&self) -> usize {
        self.shared.queued.load(Ordering::SeqCst)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ShutdownReport {
    pub(crate) state: EngineState,
    /// The cause of an `Error` or `NativeFailed` state, including every panic.
    pub(crate) failure: Option<String>,
}

pub(crate) struct Engine {
    handle: EngineHandle,
    thread: JoinHandle<()>,
}

impl Engine {
    pub(crate) fn handle(&self) -> EngineHandle {
        self.handle.clone()
    }

    /// Stops the engine. The running job and every queued job are cancelled, so each owner gets
    /// a `Cancelled` batch and a usage event. The call waits for the running pass and the
    /// thread, but not for other `EngineHandle` clones: the thread stops while they live, and
    /// they then see `Unavailable`. A panic is reported in the result, never discarded.
    pub(crate) fn shutdown(self) -> ShutdownReport {
        let Engine { handle, thread } = self;
        let shared = Arc::clone(&handle.shared);
        shared.stop.store(true, Ordering::SeqCst);
        if let Some(cancel) = lock(&shared.running).as_ref() {
            cancel.store(true, Ordering::SeqCst);
        }
        for job in lock(&shared.active).iter() {
            job.cancel.store(true, Ordering::SeqCst);
        }
        // Blocks only while the queue is full; the worker then cancels the jobs ahead of the
        // message. It fails when the thread has already ended.
        let _ = handle.queue.send(Message::Stop);
        drop(handle);
        let joined = thread.join();
        let mut failure = shared.failure();
        if let Err(payload) = joined {
            // A panic that the guard in `run` did not contain: record it, never hide it.
            let cause = format!(
                "engine thread panicked outside the guard: {}",
                panic_message(payload.as_ref())
            );
            if shared.state() != EngineState::Error {
                shared.set_state(EngineState::NativeFailed);
            }
            shared.record_failure(EngineFailure::internal("internal", cause.clone()));
            failure = Some(cause);
        }
        ShutdownReport {
            state: shared.state(),
            failure,
        }
    }
}

/// The prefix capability of a built executor, carried by the ready note:
/// whether its store can ever cut a rolling Block entry, and the spacing
/// the executor itself cuts with. Every field comes from the built executor
/// - never a factory flag, never the process-wide parse.
pub(crate) struct PrefixCapability {
    pub(crate) cuts_block: bool,
    pub(crate) spacing: Option<usize>,
}

/// The one predicate every store-capability decision shares: a store that
/// cannot hold two entries (the rolling Block entry and the live session)
/// can never cut an intermediate Block save point.
fn cuts_intermediate(limits: StoreLimits) -> bool {
    limits.entries >= 2
}

/// What the engine thread reports through its one-shot note: either the
/// built executor's prefix [`PrefixCapability`] once the engine becomes
/// ready, or the failure cause at the moment the engine gives up loading.
pub(crate) enum EngineNote {
    Ready(PrefixCapability),
    Failed(String),
}

/// A one-shot note the engine thread fires exactly once: [`EngineNote::Ready`]
/// with the built executor's prefix [`PrefixCapability`] when the engine
/// becomes ready, or [`EngineNote::Failed`] with the same cause string
/// `record_failure` stores when `make` fails. `None` fires nothing.
pub(crate) type ReadyNote = Option<Box<dyn FnOnce(EngineNote) + Send>>;

/// The handle an engine's start closure uses to announce which of its
/// own phases it is entering. Each announcement stores the stage for
/// `/v1/models` (`load_stage`, present only while loading); nothing is
/// printed here — presentation is the caller's poll (the CLI renders
/// stage lines on stderr). A family that has no phase boundaries worth
/// reporting simply never calls [`Stages::begin`].
pub(crate) struct Stages<'a> {
    shared: &'a Shared,
    pending_effective: OnceLock<Vec<crate::capability::effective::EffectiveSetting>>,
}

impl Stages<'_> {
    /// Announces the phase the start code is entering now. Cheap by
    /// construction: a mutex write per phase, a handful per engine.
    pub(crate) fn begin(&self, stage: &'static str) {
        *lock(&self.shared.load_stage) = Some(stage);
    }

    /// Stages the loaded settings for this worker. Failed startup discards them.
    #[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
    pub(crate) fn publish_effective(
        &self,
        rows: Vec<crate::capability::effective::EffectiveSetting>,
    ) {
        assert!(
            self.pending_effective.set(rows).is_ok(),
            "settings staged twice"
        );
    }
}

/// Starts the engine thread without a ready note; test callers only, so a
/// new family cannot silently skip the note - production callers use
/// [`spawn_engine_noted`]. `make` runs on that thread and builds the
/// executor and its session, so thread-confined native values never cross a
/// thread boundary. `prefill_chunk` is the largest prompt slice passed to
/// one `prefill` call when the executor does not plan its own piece. An
/// executor's initial `prefill_piece()` sets the maximum after `make`; the
/// hint is refreshed between pieces within that maximum and any request-specific limit.
#[cfg(test)]
pub(crate) fn spawn_engine<E, F>(prefill_chunk: usize, make: F) -> Result<Engine, EngError>
where
    E: ExecutorOps + 'static,
    F: FnOnce(&Stages<'_>) -> Result<(E, E::Session), EngError> + Send + 'static,
{
    spawn_engine_noted(prefill_chunk, None, make)
}

/// Like [`spawn_engine`], plus the engine thread fires `ready_note` exactly
/// once - [`EngineNote::Ready`] with the executor's own prefix capability
/// when the engine becomes ready, or [`EngineNote::Failed`] with the load
/// failure cause. The note is the ONLY capability reporting path:
/// executor-derived, no per-family flags, no model-name branches.
pub(crate) fn spawn_engine_noted<E, F>(
    prefill_chunk: usize,
    ready_note: ReadyNote,
    make: F,
) -> Result<Engine, EngError>
where
    E: ExecutorOps + 'static,
    F: FnOnce(&Stages<'_>) -> Result<(E, E::Session), EngError> + Send + 'static,
{
    if prefill_chunk == 0 {
        return Err(EngError::BadRequest(
            "prefill chunk must be at least 1".to_string(),
        ));
    }
    let (queue, jobs) = sync_channel(QUEUE_CAPACITY);
    let shared = Arc::new(Shared::new());
    let worker_shared = Arc::clone(&shared);
    let thread = std::thread::Builder::new()
        .name("engine".to_string())
        .spawn(move || run::<E, F>(ready_note, make, &jobs, &worker_shared, prefill_chunk))
        .map_err(|error| EngError::Backend {
            operation: "engine_spawn",
            message: error.to_string(),
        })?;
    Ok(Engine {
        handle: EngineHandle { queue, shared },
        thread,
    })
}

fn engine_error(message: impl Into<String>) -> EngError {
    EngError::Backend {
        operation: "engine",
        message: message.into(),
    }
}

fn reject(job: Job) {
    send_terminal(
        &job.events,
        JobEvent::Error(engine_error("engine is not available")),
    );
}

fn cancel_unstarted(job: Job) {
    job.generation.cancel.store(true, Ordering::SeqCst);
    send_cancelled(&job.events, prompt_tokens(&job.prompt));
}

fn drain(jobs: &Receiver<Message>, finish: fn(Job)) {
    while let Ok(message) = jobs.try_recv() {
        if let Message::Job(job, ticket) = message {
            drop(ticket);
            finish(job);
        }
    }
}

fn prompt_tokens(prompt: &[u32]) -> u32 {
    u32::try_from(prompt.len()).unwrap_or(u32::MAX)
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn run<E, F>(
    ready_note: ReadyNote,
    make: F,
    jobs: &Receiver<Message>,
    shared: &Shared,
    prefill_chunk: usize,
) where
    E: ExecutorOps,
    F: FnOnce(&Stages<'_>) -> Result<(E, E::Session), EngError>,
{
    let mut current: Option<mpsc::Sender<JobEvent>> = None;
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        work::<E, F>(ready_note, make, jobs, shared, prefill_chunk, &mut current);
    }));
    if let Err(payload) = outcome {
        let message = panic_message(payload.as_ref());
        let next = if shared.state() == EngineState::Loading {
            EngineState::Error
        } else {
            EngineState::NativeFailed
        };
        shared.record_failure(EngineFailure::internal(
            "internal",
            format!("engine thread panicked: {message}"),
        ));
        shared.set_state(next);
        shared.clear_load_stage();
        if let Some(events) = current.take() {
            send_terminal(
                &events,
                JobEvent::Error(engine_error(format!("engine panicked: {message}"))),
            );
        }
        *lock(&shared.running) = None;
        for job in lock(&shared.active).drain(..) {
            job.cancel.store(true, Ordering::SeqCst);
            send_terminal(
                &job.events,
                JobEvent::Error(engine_error(format!("engine panicked: {message}"))),
            );
        }
        drain(jobs, reject);
        // Last: dropping a payload can itself panic, and `shutdown` then reports that panic.
        drop(payload);
    }
}

fn work<E, F>(
    ready_note: ReadyNote,
    make: F,
    jobs: &Receiver<Message>,
    shared: &Shared,
    prefill_chunk: usize,
    current: &mut Option<mpsc::Sender<JobEvent>>,
) where
    E: ExecutorOps,
    F: FnOnce(&Stages<'_>) -> Result<(E, E::Session), EngError>,
{
    let stages = Stages {
        shared,
        pending_effective: OnceLock::new(),
    };
    let (mut executor, mut session) = match make(&stages) {
        Ok(pair) => pair,
        Err(error) => {
            let cause = format!("engine load failed: {}", error.describe());
            if let Some(note) = ready_note {
                note(EngineNote::Failed(cause.clone()));
            }
            shared.record_failure(EngineFailure::from_error(&error, "load", cause));
            shared.set_state(EngineState::Error);
            shared.clear_load_stage();
            drain(jobs, reject);
            return;
        }
    };
    let iteration_rows = prefill_chunk;
    let prefill_chunk = executor.prefill_piece().unwrap_or(prefill_chunk);
    if prefill_chunk == 0 {
        let cause = "engine load failed: the executor planned a prefill piece of 0".to_string();
        if let Some(note) = ready_note {
            note(EngineNote::Failed(cause.clone()));
        }
        shared.record_failure(EngineFailure::internal("load", cause));
        shared.set_state(EngineState::Error);
        shared.clear_load_stage();
        drain(jobs, reject);
        return;
    }
    let limits = executor.prefix_limits();
    shared
        .no_prefix_store
        .store(limits.roots == 0 && limits.entries == 0, Ordering::SeqCst);
    if let Some(plan) = executor.mlx_execution_plan() {
        let _ = shared.mlx_execution.set(LoadedMlxExecution {
            plan: plan.clone(),
            worker_piece_rows: prefill_chunk,
        });
    }
    if let Some(rows) = stages.pending_effective.into_inner() {
        let _ = shared.effective.set(rows);
    }
    shared.set_state(EngineState::Ready);
    shared.clear_load_stage();
    // The engine reports its own prefix capability to the one-shot note: the
    // built executor's store and spacing are the single source, so every
    // family reports correctly without a factory flag.
    if let Some(note) = ready_note {
        note(EngineNote::Ready(PrefixCapability {
            cuts_block: cuts_intermediate(limits),
            spacing: executor.prefix_block_spacing(),
        }));
    }
    let max_seqs = executor.max_seqs();
    if max_seqs > 1 {
        batch_worker::work(
            &mut executor,
            session,
            jobs,
            shared,
            prefill_chunk,
            iteration_rows,
            max_seqs,
        );
        return;
    }
    loop {
        // An executor with background work (a disk write-behind) wakes between jobs to advance
        // it; one without blocks on the queue as before.
        let message = if executor.background_pending() {
            match jobs.recv_timeout(BACKGROUND_POLL) {
                Ok(message) => message,
                Err(RecvTimeoutError::Timeout) => {
                    executor.background_step();
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
        } else {
            match jobs.recv() {
                Ok(message) => message,
                Err(_) => break,
            }
        };
        let Message::Job(job, ticket) = message else {
            break;
        };
        drop(ticket);
        if shared.state() == EngineState::NativeFailed {
            reject(job);
            continue;
        }
        let Job {
            prompt,
            generation,
            events,
            prefix_boundary,
            root_boundary,
        } = job;
        // Register the cancel flag first and read `stop` second. `shutdown` sets `stop` first
        // and reads the flag second, so one of the two sees the other.
        *lock(&shared.running) = Some(Arc::clone(&generation.cancel));
        if shared.stop.load(Ordering::SeqCst) {
            generation.cancel.store(true, Ordering::SeqCst);
        }
        shared.set_state(EngineState::Busy);
        send_data(&events, JobEvent::Started);
        let events = current.insert(events);
        let plan = PromptPlan {
            tokens: &prompt,
            boundary: prefix_boundary,
            root: root_boundary,
        };
        let next = run_job(
            &mut executor,
            &mut session,
            &plan,
            generation,
            events,
            prefill_chunk,
            shared,
        );
        *lock(&shared.running) = None;
        shared.set_state(next);
        *current = None;
    }
    finish_background(&mut executor, shared);
    if shared.state() != EngineState::NativeFailed {
        shared.set_state(EngineState::Stopped);
    }
    drain(jobs, cancel_unstarted);
}

/// Ends the executor's background work (a disk flush) at a stop, unless the engine failed
/// natively: its device state cannot be trusted to be read, and a flush would hang or write a
/// damaged entry.
fn finish_background<E: ExecutorOps>(executor: &mut E, shared: &Shared) {
    if shared.state() != EngineState::NativeFailed {
        executor.background_drain();
    }
}

/// How long an idle engine waits for a job before it advances its background work.
const BACKGROUND_POLL: Duration = Duration::from_millis(2);

struct PromptPlan<'a> {
    tokens: &'a [u32],
    /// The number of leading tokens that end at the last message boundary, or `None`.
    boundary: Option<usize>,
    /// The number of leading tokens that end at the first system message, or `None`. It is
    /// set only beside a boundary, and it lies inside it.
    root: Option<usize>,
}

const MIN_ROOT_TOKENS: usize = 2_048;

/// The stride of the periodic save points: one entry at every multiple of this many positions
/// counted from the start of the prompt.
const PERIODIC_TOKENS: usize = 16_384;

/// The ordered save points of a job: the root at the end of the first system message, one
/// periodic entry at every multiple of `PERIODIC_TOKENS` positions counted from the start of
/// the prompt, and the tip at the last message boundary.
/// The root runs only when the store keeps roots (`limits.roots > 0`), it reaches
/// `MIN_ROOT_TOKENS`, lies strictly inside the tip, and the restore covered none of it. Without
/// roots a root point would only split the prefill and change its chunk grouping for a save
/// the store refuses, so a job with a root boundary then runs exactly like a job without one.
/// The periodic points run only when the store keeps at least two LRU entries
/// (`limits.entries >= 2`): with one entry a periodic save would be replaced by the tip at
/// once and spend a snapshot for nothing. Of the periodic points only the deepest
/// `limits.entries - 1` are emitted, because the tip takes the last LRU place. The tip runs only
/// when the restore covered none of it.
///
/// The prefill runs in chunks of `piece` tokens that start at the current cursor: the cursor
/// starts at `restored`, moves to the root, and then to every point this function emits, so a
/// root or a restore from an arbitrary position moves it off the piece lattice. A periodic
/// target P counts only when it lies above the cursor. It moves to the end of the chunk that
/// contains it, `cursor + ceil((P - cursor) / piece) * piece` (a P that is already a chunk end
/// stays P), and the moved position must stay strictly below the tip, otherwise the target is
/// dropped. The moved position is always above the cursor, so no position is emitted twice.
/// The rolling block cut gets the same move-to-chunk-end and below-the-tip rules, except when
/// the cut is not a chunk end of the cursor (`(cut - cursor) % piece != 0`, the common case
/// after a root or a restore from an off-lattice position, and almost always when the piece
/// does not divide the block spacing): it then emits at its true boundary instead, because a
/// misaligned ride would push the cut to a chunk end at or above the tip and drop it, losing
/// the mid-turn block match. `run_job` re-anchors after every emitted point, so the split
/// costs one re-anchored chunk. The cut keeps its true position only while the ride rule kept
/// no periodic entry deeper: a rolling block is replaced by the next job, a periodic entry
/// persists, so when the old rule would have kept a deeper periodic the cut takes the old rule
/// and rides the lattice. A periodic target that the ride rule dropped (its ride reached the
/// tip) shares its position with no persistent entry, so when the true-position cut lands
/// exactly on such a dropped target the entry takes the periodic kind, the truthful one. For
/// the default spacings every periodic target is a block-spacing multiple, so the
/// deeper-periodic guard never fires on the served families and only the tip's final partial
/// chunk, at most the spacing in length, re-tiles.
/// The block spacing of one prompt: the executor's configured spacing, sized to the prompt's tip
/// when nothing sets it explicitly (see `session::block_spacing_for`).
fn block_spacing(configured: Option<usize>, tip: Option<usize>) -> Option<usize> {
    configured
        .map(|configured| crate::family::session::block_spacing_for(tip.unwrap_or(0), configured))
}

fn save_points(
    plan: &PromptPlan<'_>,
    restored: usize,
    limits: StoreLimits,
    piece: usize,
    block: Option<usize>,
) -> Vec<(usize, SaveKind)> {
    let _ = block;
    let mut points = Vec::new();
    let mut cursor = restored;
    if let Some(root) = plan
        .root
        .filter(|_| limits.roots > 0)
        .filter(|&root| root >= MIN_ROOT_TOKENS && root > restored)
        .filter(|&root| plan.boundary.is_some_and(|tip| root < tip))
    {
        points.push((root, SaveKind::Root));
        cursor = root;
    }
    if let Some(tip) = plan.boundary.filter(|_| cuts_intermediate(limits)) {
        debug_assert!(piece > 0, "the prefill piece is positive");
        // The intermediate candidates: the periodic multiples below the tip and, when the block
        // prefix-match is on, the last block boundary strictly below the tip - the ruled
        // rolling cut of the lineage. The walk visits the targets in order, a periodic first at
        // a tie, so a boundary the lattice already cuts stays one save of the periodic kind;
        // the block candidate rides the same move-to-chunk-end and below-the-tip rules, so it
        // never shifts the periodic positions - unless the cut is not a chunk end of the
        // cursor: the chunk lattice then cannot represent the boundary, so the block candidate
        // is emitted at its true boundary instead of its chunk end. The tie rule also holds
        // after the move to the chunk end: a block target whose chunk end an unprocessed
        // periodic target reaches is skipped, because every target in (block_target, moved]
        // projects onto `moved`, and the shared position must keep the periodic kind, which a
        // later rolling K=1 replacement cannot remove.
        //
        // The true position carries over only when the ride rule kept no periodic entry deeper
        // than the cut, so this pre-pass walks the same targets under the pure ride rule to
        // learn which periodic targets that rule kept and which it dropped; the entries cap
        // plays no part, it only trims the shallow end of the same walk.
        let mut targets: Vec<(usize, bool)> = (PERIODIC_TOKENS..tip)
            .step_by(PERIODIC_TOKENS)
            .map(|target| (target, false))
            .collect();
        if let Some(spacing) = block {
            targets.push(((tip - 1) / spacing * spacing, true));
        }
        targets.sort_unstable();
        let mut kept_periodic: Vec<usize> = Vec::new();
        let mut dropped_periodic: Vec<usize> = Vec::new();
        let mut ride_cursor = cursor;
        for &(target, is_block) in &targets {
            if target <= ride_cursor {
                continue;
            }
            let moved = ride_cursor + (target - ride_cursor).div_ceil(piece) * piece;
            if moved >= tip {
                if !is_block {
                    dropped_periodic.push(target);
                }
                continue;
            }
            if is_block
                && targets.iter().any(|&(candidate, candidate_is_block)| {
                    !candidate_is_block && candidate > target && candidate <= moved
                })
            {
                continue;
            }
            if !is_block {
                kept_periodic.push(target);
            }
            ride_cursor = moved;
        }
        let mut intermediate: Vec<(usize, SaveKind)> = Vec::new();
        for &(target, is_block) in &targets {
            if target <= cursor {
                continue;
            }
            if is_block
                && !(target - cursor).is_multiple_of(piece)
                && !kept_periodic.iter().any(|&kept| kept > target)
            {
                // The cut below the tip is strictly below it by construction, so the drop rule
                // cannot fire here. A periodic target that shares the boundary, or whose
                // chunk end passes it, was processed first and emitted: the periodic kind
                // keeps the shared region, the same rule as the tie below, and the entry it
                // leaves is deeper than the cut. A periodic target that the ride rule DROPPED
                // (its ride reached the tip) shares its position with no persistent entry,
                // so the cut landing exactly on it takes the periodic kind, the truthful
                // one, where the old rule kept nothing at all. A periodic target the ride
                // rule KEPT deeper than the cut holds the old rule instead: this branch
                // advances the cursor to the cut, from where that periodic's lattice ride
                // can pass the tip and drop - a rolling block is replaced by the next job,
                // a periodic entry persists, so the cut must not displace one that
                // survives.
                let kind = if dropped_periodic.contains(&target) {
                    SaveKind::Periodic
                } else {
                    SaveKind::Block
                };
                intermediate.push((target, kind));
                cursor = target;
                continue;
            }
            let moved = cursor + (target - cursor).div_ceil(piece) * piece;
            if moved >= tip {
                continue;
            }
            if is_block
                && targets.iter().any(|(candidate, candidate_is_block)| {
                    !*candidate_is_block && *candidate > target && *candidate <= moved
                })
            {
                continue;
            }
            intermediate.push((
                moved,
                if is_block {
                    SaveKind::Block
                } else {
                    SaveKind::Periodic
                },
            ));
            cursor = moved;
        }
        // Only the deepest `entries - 1` intermediate points survive in the store, whatever
        // kind: the tip takes the last LRU place, and each save beyond that evicts the oldest
        // entry, first the intermediate entries of this same job and then the entries of
        // other conversations. The final store is the same without the shallow points, which
        // only spend snapshots. A moved position lies on the chunk lattice of the cursor, so
        // skipping a shallow point does not change the position of a deeper one; the
        // true-position cut lies off that lattice, but it changes no position but its own,
        // so skipping stays safe.
        let skip = intermediate.len().saturating_sub(limits.entries - 1);
        points.extend(intermediate.into_iter().skip(skip));
    }
    if let Some(tip) = plan.boundary.filter(|&tip| tip > restored) {
        points.push((tip, SaveKind::Tip));
    }
    points
}

/// A failed allocation can return after the client has cancelled. Check the
/// request again before spending another conversation's cached state on a retry.
fn evict_for_retry<E: ExecutorOps>(
    executor: &mut E,
    session: Option<&E::Session>,
    prompt: &[u32],
    cancel: &AtomicBool,
) -> bool {
    let protect = session
        .and_then(|session| executor.restored_anchor_of(session))
        .or_else(|| executor.restore_candidate(prompt));
    !cancel.load(Ordering::SeqCst) && executor.evict_one_prefix_entry(protect).is_some()
}

/// Runs one job and returns the state of the engine after it. The first call is the prompt
/// check: a prompt that the executor refuses ends the job with the executor's error before any
/// state changes, so the session stays as the last job left it, a saved prefix survives for the
/// next job, and the engine stays `Ready`.
fn run_job<E: ExecutorOps>(
    executor: &mut E,
    session: &mut E::Session,
    plan: &PromptPlan<'_>,
    mut generation: GenerationState,
    events: &mpsc::Sender<JobEvent>,
    prefill_chunk: usize,
    shared: &Shared,
) -> EngineState {
    let prompt_tokens = prompt_tokens(plan.tokens);
    if generation.cancel.load(Ordering::SeqCst) {
        // A queued cancellation: nothing ran, so the session is untouched.
        send_cancelled(events, prompt_tokens);
        return EngineState::Ready;
    }
    // The whole job prompt, including any tokens that a restore would cover, before any state
    // change: the executor refuses a prompt that cannot fit here, so the job costs the next job
    // nothing (no reset, no restore, no prefill, and its saved prefix survives).
    if let Err(error) = executor.check_prompt(plan.tokens.len()) {
        send_terminal(events, JobEvent::Error(error));
        return EngineState::Ready;
    }
    // The admit-path bank counters before the prefill, for this request's delta (log-only;
    // `None` is the report of an executor without an expert bank, and the delta stays `None`).
    let bank_before = executor.prefill_bank_report();
    let mut pieces = PrefillPieces::default();
    let prefill = match start(
        executor,
        session,
        plan,
        prefill_chunk,
        &generation,
        &mut pieces,
        false,
    ) {
        Ok(reused) => reused,
        Err(EngError::MemoryPlan(message)) => {
            // A dead or idle conversation's cached prefixes must not pin the pool (six
            // entries of ~395 MB once blocked a fresh 17,479-token session that needed
            // 11,141,120 bytes with 7,696,432 available). Evict ONE entry at a time - a
            // dead tip first, oldest first, the denied job's own anchor last (radix tiers)
            // - and retry after each, so a denial with a one-entry deficit costs one
            // anchor, not every live lineage (a clear-all once lost all sixteen entries to
            // a ~3-6 entry deficit). Each retry admits the prompt with no snapshots at all,
            // so caching cannot re-block a prompt the pool can grow. When the last entry is
            // gone and the admission still denies, the refusal only becomes the over-length
            // form when the pool's ceiling honestly explains it - a client then compacts to
            // a size that fits instead of losing the session to repeated growth errors -
            // and the session still resets either way.
            //
            // Size the eviction before the restart: evict while the pool's honest ceiling
            // still cannot admit this prompt, so a k-entry deficit costs ONE restart, not k
            // partial prefills. The ceiling covers the free budget plus this session's own
            // permits, and every eviction keeps the denied job's own anchor for its last
            // resort. A `None` ceiling sizes nothing: stop after the first eviction and let
            // the residual one-at-a-time loop below spend more as denials recur. The loop's
            // FIRST `None` is terminal: nothing can be freed, so no restart can ever admit
            // this prompt - classify now instead of one dead restart (the batch worker's
            // admission loop does the same).
            let mut terminal = false;
            loop {
                if !evict_for_retry(executor, Some(session), plan.tokens, &generation.cancel) {
                    if generation.cancel.load(Ordering::SeqCst) {
                        send_cancelled(events, prompt_tokens);
                        return reset_state(executor, session, shared);
                    }
                    terminal = true;
                    break;
                }
                let Some(ceiling) = executor.pool_prompt_ceiling(&*session) else {
                    break;
                };
                if ceiling >= u64::from(prompt_tokens) {
                    break;
                }
            }
            if terminal {
                // Classify only when the ceiling PROVES the prompt cannot fit (the honest
                // over-length form); a save-denial recovery with room to grow still takes
                // the one no-save retry below.
                let fits_not = executor
                    .pool_prompt_ceiling(&*session)
                    .is_some_and(|ceiling| ceiling < u64::from(prompt_tokens));
                if fits_not {
                    let error = classified(
                        executor,
                        &*session,
                        prompt_tokens,
                        EngError::MemoryPlan(message),
                    );
                    send_terminal(events, JobEvent::Error(error));
                    return reset_state(executor, session, shared);
                }
            }
            let mut retried = start(
                executor,
                session,
                plan,
                prefill_chunk,
                &generation,
                &mut pieces,
                true,
            );
            while let Err(EngError::MemoryPlan(_)) = retried {
                if !evict_for_retry(executor, Some(session), plan.tokens, &generation.cancel) {
                    if generation.cancel.load(Ordering::SeqCst) {
                        send_cancelled(events, prompt_tokens);
                        return reset_state(executor, session, shared);
                    }
                    break;
                }
                retried = start(
                    executor,
                    session,
                    plan,
                    prefill_chunk,
                    &generation,
                    &mut pieces,
                    true,
                );
            }
            match retried {
                Ok(reused) => reused,
                Err(EngError::MemoryPlan(message)) => {
                    // The retry's refusal names the post-eviction pool, so it - not the first
                    // attempt's message with its pre-eviction availability - is what a client
                    // sees when the ceiling does not explain the failure.
                    let error = classified(
                        executor,
                        &*session,
                        prompt_tokens,
                        EngError::MemoryPlan(message),
                    );
                    send_terminal(events, JobEvent::Error(error));
                    return reset_state(executor, session, shared);
                }
                Err(error) => return fail(executor, session, events, error, shared),
            }
        }
        Err(error) => return fail(executor, session, events, error, shared),
    };
    let ControlFlow::Continue(reused) = prefill else {
        send_cancelled(events, prompt_tokens);
        return reset_state(executor, session, shared);
    };
    let reused_tokens = u32::try_from(reused).unwrap_or(u32::MAX);
    // The report of the job's prefix store, taken after the prefill and before the first decode:
    // the entries and bytes then include this job's save points. `None` is the report of an
    // executor that keeps none.
    let prefix = executor.prefix_report();
    // The expert-bank counters after the prefill, for the request log's `prefill_bank_*` keys
    // (log-only; `None` is the report of an executor without an expert bank).
    let prefill_bank = match (executor.prefill_bank_report(), bank_before) {
        (Some(after), Some(before)) => Some(after.saturating_sub(&before)),
        _ => None,
    };
    let mut completion_tokens: u32 = 0;
    loop {
        let batch = match executor.decode(session, &mut generation) {
            Ok(batch) => batch,
            Err(error) => return fail(executor, session, events, error, shared),
        };
        let committed = u32::try_from(batch.tokens.len()).unwrap_or(u32::MAX);
        completion_tokens = completion_tokens.saturating_add(committed);
        let finished = batch.finish.is_some();
        if !send_data(events, JobEvent::Batch(batch)) {
            // A full channel or a dropped receiver: the owner is gone or too slow.
            generation.cancel.store(true, Ordering::SeqCst);
            return reset_state(executor, session, shared);
        }
        if let Err(error) = executor.after_emit() {
            return fail(executor, session, events, error, shared);
        }
        if finished {
            send_terminal(
                events,
                JobEvent::Finished(JobUsage {
                    prompt_tokens,
                    completion_tokens,
                    reused_tokens,
                    prefix,
                    prefill_bank,
                    prefill_pieces: pieces,
                    first_token_margin: generation.sampler.first_top2_margin(),
                }),
            );
            return EngineState::Ready;
        }
    }
}

/// The final error for an admission the pool refused: the over-length form only when the pool's
/// ceiling honestly explains the refusal - a ceiling that would admit this prompt is not why it
/// failed, so the original refusal stands. A ceiling below the prompt names the largest prompt
/// a fresh attempt could charge, which never exceeds the configured limit because
/// `check_prompt` already refused prompts at or beyond it.
fn classified<E: ExecutorOps>(
    executor: &E,
    session: &E::Session,
    prompt_tokens: u32,
    original: EngError,
) -> EngError {
    match executor.pool_prompt_ceiling(session) {
        Some(limit) if limit < u64::from(prompt_tokens) => EngError::PromptTooLong {
            prompt_tokens: u64::from(prompt_tokens),
            limit_tokens: limit,
        },
        _ => original,
    }
}

/// Resets the session, restores a saved prefix when the job has a boundary, runs `boundary`,
/// and prefills the rest of the prompt in chunks. It prefills up to every save point of the job
/// in turn (the root, the periodic entries, then the tip) and saves the prefix state at each
/// point. Each segment is chunked from its own start, which is the cursor that `save_points`
/// moved to the previous point, so the chunk sequence of the prefill is the one that
/// `save_points` assumed. Returns the number of prompt tokens that the restore covered,
/// or breaks if a paired forward stopped mid-layer schedule and the session needs a reset.
///
/// `retrying` marks the eviction retry, which runs with no save points at all: a snapshot that
/// charges successfully at its own point can still starve the prefill tail that follows it, so
/// the retry admits the prompt with no snapshots and the next job rebuilds its cache from
/// scratch.
fn start<E: ExecutorOps>(
    executor: &mut E,
    session: &mut E::Session,
    plan: &PromptPlan<'_>,
    prefill_chunk: usize,
    generation: &GenerationState,
    pieces: &mut PrefillPieces,
    retrying: bool,
) -> Result<ControlFlow<(), usize>, EngError> {
    // A restart re-feeds from the cursor, so the log keeps only the run that served the
    // prompt, never the aborted attempt's pieces.
    *pieces = PrefillPieces::default();
    executor.reset_for_request(
        session,
        plan.tokens.len(),
        generation.remaining_tokens(),
        plan.boundary.is_some(),
    )?;
    let prefill_chunk = request_prefill_chunk(executor, session, prefill_chunk);
    let cancel = &generation.cancel;
    let restored = match plan.boundary {
        Some(_) => executor.restore_prefix(session, plan.tokens)?,
        None => 0,
    };
    if restored >= plan.tokens.len() {
        return Err(EngError::BadRequest(
            "restored prefix covers the whole prompt".to_string(),
        ));
    }
    executor.boundary(session);
    let mut next = restored;
    let points = if retrying {
        Vec::new()
    } else {
        save_points(
            plan,
            restored,
            executor.prefix_limits(),
            prefill_chunk,
            block_spacing(executor.prefix_block_spacing(), plan.boundary),
        )
    };
    for (point, kind) in points {
        let head = span(plan.tokens, next, point)?;
        if let ControlFlow::Break(()) = prefill_chunks(
            executor,
            session,
            head,
            prefill_chunk,
            cancel,
            pieces,
            false,
        )? {
            return Ok(ControlFlow::Break(()));
        }
        if cancel.load(Ordering::SeqCst) {
            return Ok(ControlFlow::Continue(restored));
        }
        // The head fed through `point`; the save that follows it cannot change what was fed.
        next = point;
        executor.save_prefix(session, span(plan.tokens, 0, point)?, kind)?;
    }
    let tail = span(plan.tokens, next, plan.tokens.len())?;
    Ok(prefill_chunks(
        executor,
        session,
        tail,
        prefill_chunk,
        cancel,
        pieces,
        generation.remaining_tokens() != Some(0),
    )?
    .map_continue(|()| restored))
}

fn request_prefill_chunk<E: ExecutorOps>(
    executor: &E,
    session: &E::Session,
    startup_chunk: usize,
) -> usize {
    executor
        .request_prefill_piece(session)
        .map_or(startup_chunk, |rows| rows.get().min(startup_chunk))
}

fn span(tokens: &[u32], from: usize, to: usize) -> Result<&[u32], EngError> {
    tokens
        .get(from..to)
        .ok_or_else(|| EngError::BadRequest("prompt span is outside the prompt".to_string()))
}

/// Cancellation between finished pieces leaves a session that can finish through decode.
/// A cancelled paired forward breaks with a partial session that needs a reset.
fn prefill_chunks<E: ExecutorOps>(
    executor: &mut E,
    session: &mut E::Session,
    tokens: &[u32],
    prefill_chunk: usize,
    cancel: &AtomicBool,
    pieces: &mut PrefillPieces,
    final_tail: bool,
) -> Result<ControlFlow<()>, EngError> {
    let mut remaining = tokens;
    while !remaining.is_empty() {
        if pieces.count > 0 && cancel.load(Ordering::SeqCst) {
            return Ok(ControlFlow::Continue(()));
        }
        let rows = executor
            .prefill_piece()
            .unwrap_or(prefill_chunk)
            .min(prefill_chunk);
        if rows == 0 {
            return Err(EngError::BadRequest(
                "the executor planned a prefill piece of 0".to_owned(),
            ));
        }
        let (piece, rest) = remaining.split_at(rows.min(remaining.len()));
        if pieces.count == 0
            && piece.len() >= 3
            && (2..=rows).contains(&rest.len())
            && !cancel.load(Ordering::SeqCst)
        {
            match executor.try_prefill_pair(session, remaining, piece.len(), cancel)? {
                PrefillPairOutcome::Declined => {}
                PrefillPairOutcome::Completed => {
                    pieces.count = 2;
                    pieces.max_rows = u32::try_from(piece.len()).unwrap_or(u32::MAX);
                    return Ok(ControlFlow::Continue(()));
                }
                PrefillPairOutcome::Cancelled => return Ok(ControlFlow::Break(())),
            }
        }
        // The executor's prefill error passes through unchanged: this is shared scheduler
        // code, and a piece-index prefix would reach HTTP 400 clients of every family on
        // default-off paths. The piece shape is recorded in the request log instead.
        if final_tail && rest.is_empty() && !cancel.load(Ordering::SeqCst) {
            executor.prefill_for_decode(session, piece)?;
        } else {
            executor.prefill(session, piece)?;
        }
        pieces.count = pieces.count.saturating_add(1);
        pieces.max_rows = pieces
            .max_rows
            .max(u32::try_from(piece.len()).unwrap_or(u32::MAX));
        remaining = rest;
    }
    Ok(ControlFlow::Continue(()))
}

fn reset_state<E: ExecutorOps>(
    executor: &mut E,
    session: &mut E::Session,
    shared: &Shared,
) -> EngineState {
    match executor.reset(session) {
        Ok(()) => EngineState::Ready,
        Err(error) => {
            shared.record_failure(EngineFailure::from_error(
                &error,
                "runtime",
                format!("session reset failed: {}", error.describe()),
            ));
            EngineState::NativeFailed
        }
    }
}

/// A `Backend` error is process-scoped: the worker makes no further executor call. Any other
/// error fails only this job and leaves a reset session.
fn fail<E: ExecutorOps>(
    executor: &mut E,
    session: &mut E::Session,
    events: &mpsc::Sender<JobEvent>,
    error: EngError,
    shared: &Shared,
) -> EngineState {
    let native = matches!(error, EngError::Backend { .. });
    if native {
        shared.record_failure(EngineFailure::from_error(
            &error,
            "runtime",
            format!("native failure: {}", error.describe()),
        ));
    }
    send_terminal(events, JobEvent::Error(error));
    if native {
        EngineState::NativeFailed
    } else {
        reset_state(executor, session, shared)
    }
}

/// The ruled plan for the device suites: they drive their prefill pieces off this
/// projection so the card run exercises the sequence the worker will emit.
#[cfg(test)]
pub(crate) fn planned_points_for_tests(
    boundary: Option<usize>,
    root: Option<usize>,
    restored: usize,
    piece: usize,
    limits: StoreLimits,
    block: Option<usize>,
) -> Vec<(usize, SaveKind)> {
    let tokens = vec![0; 8];
    let plan = PromptPlan {
        tokens: &tokens,
        boundary,
        root,
    };
    save_points(&plan, restored, limits, piece, block)
}

#[cfg(test)]
mod tests {
    #[test]
    fn failure_classification_names_code_stage_and_retry() {
        use super::{EngineFailure, RetryClass};
        let memory = EngineFailure::from_error(
            &EngError::MemoryPlan("needs 100 bytes".into()),
            "load",
            "engine load failed".into(),
        );
        assert_eq!(memory.code, "memory_plan");
        assert_eq!(memory.stage, "admission");
        assert_eq!(memory.retry, RetryClass::AfterCorrection);

        let checkpoint = EngineFailure::from_error(
            &EngError::Checkpoint("missing file".into()),
            "load",
            "engine load failed".into(),
        );
        assert_eq!(checkpoint.code, "checkpoint_failure");
        assert_eq!(checkpoint.stage, "load");
        assert_eq!(checkpoint.retry, RetryClass::AfterCorrection);

        let backend = EngineFailure::from_error(
            &EngError::Backend {
                operation: "cuda initialize",
                message: "no driver".into(),
            },
            "runtime",
            "native failure".into(),
        );
        assert_eq!(backend.code, "backend_error");
        assert_eq!(backend.stage, "cuda initialize");
        assert_eq!(backend.retry, RetryClass::Unknown);

        let config = EngineFailure::from_error(
            &EngError::BadRequest("kv_dtype must be f32, bf16, or q8".into()),
            "load",
            "engine load failed".into(),
        );
        assert_eq!(config.stage, "configuration");
        assert_eq!(config.retry, RetryClass::AfterCorrection);
    }

    use super::{
        EngineHandle, EngineState, PrefillPairOutcome, ShutdownReport, SubmitError, spawn_engine,
    };
    use crate::EngError;
    use crate::exec_fake::FakeExecutor;
    use crate::family::executor::{ExecutorOps, GenerationState};
    use crate::family::session::{PrefixStore, SaveKind, StoreLimits, check_prompt};
    use crate::sched::job::{
        Job, JobEvent, JobHandle, JobUsage, PrefillPieces, PrefixHit, PrefixReport,
    };
    use crate::tokenizer::{DecodeBatch, DecodeFinish, IncrementalDecoder, Tokenizer};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, mpsc};
    use std::time::{Duration, Instant};

    const TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1,"c":2},"merges":[]}}"#;

    /// The end-of-sequence id of every test script. It lies outside the three-token vocabulary
    /// because the fake finishes on it before it decodes the id.
    const EOS: u32 = 9;

    type Log = Arc<Mutex<Vec<String>>>;

    /// The real scripted fake, wrapped to record every executor call and to hold or fail the
    /// first `decode` call on request.
    struct Spy {
        inner: FakeExecutor,
        log: Log,
        hold: Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>,
        hold_after_emit: Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>,
        emit_ack: Option<mpsc::Receiver<()>>,
        fail_first_decode: Option<EngError>,
        fail_after_emit: Option<EngError>,
        cancel_after_prefill: Option<Arc<AtomicBool>>,
        /// Sets the cancel flag after the `at`-th `prefill` call (0 is the first).
        cancel_after_prefill_call: Option<(usize, Arc<AtomicBool>)>,
        prefill_calls: usize,
        /// Fails the n-th `reset` call (0 is the first) with the error.
        fail_reset_call: Option<(usize, EngError)>,
        reset_calls: usize,
        fail_prefill: Option<EngError>,
        panic_in_decode: Option<&'static str>,
        /// Makes `restore_prefix` report this many covered tokens without calling the fake.
        restore_override: Option<usize>,
        /// Fails the next `save_prefix` call with the error, before the fake sees it.
        fail_save: Option<EngError>,
        /// Fails the n-th `save_prefix` call (0 is the first) with the error, before the fake
        /// sees it.
        fail_save_call: Option<(usize, EngError)>,
        save_calls: usize,
        /// Sets the cancel flag after the first `save_prefix` call that the fake accepts.
        cancel_after_save: Option<Arc<AtomicBool>>,
        /// Fails the next `restore_prefix` call with the error, before the fake sees it.
        fail_restore: Option<EngError>,
        /// The piece that `prefill_piece` reports.
        piece: Option<usize>,
        piece_after_prefill: Option<usize>,
        request_piece: Option<std::num::NonZeroUsize>,
        prefill_pair: bool,
        log_final_prefill: bool,
        cancel_prefill_pair: bool,
        /// The context limit that `check_prompt` applies, or `None` to accept every prompt.
        context_limit: Option<usize>,
        /// The admit-path bank reports the spy answers: the `before` sample until the first
        /// prefill, the `after` sample from it on, both log-only.
        bank_reports: Option<(
            crate::sched::job::PrefillBankReport,
            crate::sched::job::PrefillBankReport,
        )>,
        /// Background work the spy reports as pending until `background_want` steps ran.
        background_steps: Arc<AtomicUsize>,
        background_want: usize,
        background_drained: Arc<AtomicBool>,
    }

    impl Spy {
        fn new(script: Vec<Vec<u32>>, log: &Log) -> Self {
            Spy {
                inner: FakeExecutor::new(script, &[EOS], 100_000),
                log: Arc::clone(log),
                hold: None,
                hold_after_emit: None,
                emit_ack: None,
                fail_first_decode: None,
                fail_after_emit: None,
                cancel_after_prefill: None,
                cancel_after_prefill_call: None,
                prefill_calls: 0,
                fail_reset_call: None,
                reset_calls: 0,
                fail_prefill: None,
                panic_in_decode: None,
                restore_override: None,
                fail_save: None,
                fail_save_call: None,
                save_calls: 0,
                cancel_after_save: None,
                fail_restore: None,
                piece: None,
                piece_after_prefill: None,
                request_piece: None,
                prefill_pair: false,
                cancel_prefill_pair: false,
                log_final_prefill: false,
                context_limit: None,
                bank_reports: None,
                background_steps: Arc::new(AtomicUsize::new(0)),
                background_want: 0,
                background_drained: Arc::new(AtomicBool::new(false)),
            }
        }

        fn note(&self, entry: String) {
            self.log.lock().expect("log lock").push(entry);
        }
    }

    impl ExecutorOps for Spy {
        type Session = <FakeExecutor as ExecutorOps>::Session;

        /// Applies the shared rule of `session::check_prompt` when the test sets a limit. The
        /// call is not logged: the log asserts what a job ran, and every job starts with this
        /// check.
        fn check_prompt(&self, prompt_tokens: usize) -> Result<(), EngError> {
            match self.context_limit {
                Some(limit) => check_prompt(prompt_tokens, limit),
                None => Ok(()),
            }
        }

        fn history<'s>(&self, st: &'s Self::Session) -> &'s [u32] {
            self.inner.history(st)
        }

        fn prefill_piece(&self) -> Option<usize> {
            if self.prefill_calls > 0 {
                self.piece_after_prefill.or(self.piece)
            } else {
                self.piece
            }
        }

        fn request_prefill_piece(&self, _st: &Self::Session) -> Option<std::num::NonZeroUsize> {
            self.request_piece
        }

        fn prefix_limits(&self) -> StoreLimits {
            self.inner.prefix_limits()
        }

        fn prefix_block_spacing(&self) -> Option<usize> {
            self.inner.prefix_block_spacing()
        }

        fn prefix_report(&self) -> Option<PrefixReport> {
            self.inner.prefix_report()
        }

        fn prefill_bank_report(&self) -> Option<crate::sched::job::PrefillBankReport> {
            let (before, after) = self.bank_reports.as_ref()?;
            if self.prefill_calls > 0 {
                Some(after.clone())
            } else {
                Some(before.clone())
            }
        }

        fn evict_one_prefix_entry(&mut self, protect: Option<u64>) -> Option<u64> {
            self.inner.evict_one_prefix_entry(protect)
        }

        fn restored_anchor_of(&self, st: &Self::Session) -> Option<u64> {
            self.inner.restored_anchor_of(st)
        }

        fn restore_candidate(&self, prompt: &[u32]) -> Option<u64> {
            self.inner.restore_candidate(prompt)
        }

        fn pool_prompt_ceiling(&self, st: &Self::Session) -> Option<u64> {
            self.inner.pool_prompt_ceiling(st)
        }

        fn reset(&mut self, st: &mut Self::Session) -> Result<(), EngError> {
            self.note("reset".to_string());
            let call = self.reset_calls;
            self.reset_calls += 1;
            if matches!(&self.fail_reset_call, Some((index, _)) if *index == call)
                && let Some((_, error)) = self.fail_reset_call.take()
            {
                return Err(error);
            }
            self.inner.reset(st)
        }

        fn prefill(&mut self, st: &mut Self::Session, toks: &[u32]) -> Result<(), EngError> {
            self.note(format!("prefill:{}", toks.len()));
            if let Some(error) = self.fail_prefill.take() {
                return Err(error);
            }
            self.inner.prefill(st, toks)?;
            let call = self.prefill_calls;
            self.prefill_calls += 1;
            if let Some(flag) = &self.cancel_after_prefill {
                flag.store(true, Ordering::SeqCst);
            }
            if let Some((at, flag)) = &self.cancel_after_prefill_call
                && *at == call
            {
                flag.store(true, Ordering::SeqCst);
            }
            Ok(())
        }

        fn prefill_for_decode(
            &mut self,
            st: &mut Self::Session,
            toks: &[u32],
        ) -> Result<(), EngError> {
            if self.log_final_prefill {
                self.note(format!("final_prefill:{}", toks.len()));
            }
            self.prefill(st, toks)
        }

        fn try_prefill_pair(
            &mut self,
            st: &mut Self::Session,
            tokens: &[u32],
            first_piece: usize,
            cancel: &AtomicBool,
        ) -> Result<PrefillPairOutcome, EngError> {
            if !self.prefill_pair {
                return Ok(PrefillPairOutcome::Declined);
            }
            self.note(format!("pair:{first_piece}+{}", tokens.len() - first_piece));
            if self.cancel_prefill_pair {
                cancel.store(true, Ordering::SeqCst);
                return Ok(PrefillPairOutcome::Cancelled);
            }
            self.inner.prefill(st, tokens)?;
            if let Some(flag) = self.cancel_after_prefill.take() {
                flag.store(true, Ordering::SeqCst);
            }
            Ok(PrefillPairOutcome::Completed)
        }

        fn decode(
            &mut self,
            st: &mut Self::Session,
            generation: &mut GenerationState,
        ) -> Result<DecodeBatch, EngError> {
            self.note("decode".to_string());
            if let Some((entered, release)) = self.hold.take() {
                let _ = entered.send(());
                let _ = release.recv();
            }
            if let Some(message) = self.panic_in_decode {
                panic!("{message}");
            }
            if let Some(error) = self.fail_first_decode.take() {
                return Err(error);
            }
            self.inner.decode(st, generation)
        }

        fn background_pending(&self) -> bool {
            self.background_steps.load(Ordering::SeqCst) < self.background_want
        }

        fn background_step(&mut self) {
            self.background_steps.fetch_add(1, Ordering::SeqCst);
        }

        fn background_drain(&mut self) {
            self.background_drained.store(true, Ordering::SeqCst);
        }

        fn after_emit(&mut self) -> Result<(), EngError> {
            if let Some((entered, release)) = self.hold_after_emit.take() {
                let _ = entered.send(());
                let _ = release.recv();
            }
            if let Some(ack) = &self.emit_ack {
                ack.recv_timeout(Duration::from_secs(10))
                    .expect("the consumer drains each batch");
            }
            if let Some(error) = self.fail_after_emit.take() {
                return Err(error);
            }
            Ok(())
        }

        fn boundary(&mut self, st: &mut Self::Session) {
            self.note("boundary".to_string());
            self.inner.boundary(st);
        }

        fn save_prefix(
            &mut self,
            st: &Self::Session,
            key: &[u32],
            kind: SaveKind,
        ) -> Result<bool, EngError> {
            self.note(format!("save_prefix:{}", key.len()));
            let call = self.save_calls;
            self.save_calls += 1;
            if let Some(error) = self.fail_save.take() {
                return Err(error);
            }
            if matches!(&self.fail_save_call, Some((index, _)) if *index == call)
                && let Some((_, error)) = self.fail_save_call.take()
            {
                return Err(error);
            }
            let saved = self.inner.save_prefix(st, key, kind)?;
            if let Some(flag) = self.cancel_after_save.take() {
                flag.store(true, Ordering::SeqCst);
            }
            Ok(saved)
        }

        fn restore_prefix(
            &mut self,
            st: &mut Self::Session,
            prompt: &[u32],
        ) -> Result<usize, EngError> {
            self.note("restore_prefix".to_string());
            if let Some(error) = self.fail_restore.take() {
                return Err(error);
            }
            if let Some(covered) = self.restore_override {
                return Ok(covered);
            }
            self.inner.restore_prefix(st, prompt)
        }
    }

    fn new_log() -> Log {
        Arc::new(Mutex::new(Vec::new()))
    }

    fn calls(log: &Log) -> Vec<String> {
        log.lock().expect("log lock").clone()
    }

    fn strings(entries: &[&str]) -> Vec<String> {
        entries.iter().map(|entry| (*entry).to_string()).collect()
    }

    fn generation() -> GenerationState {
        let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes()).expect("parse tokenizer");
        GenerationState::new(IncrementalDecoder::new(&tokenizer))
    }

    fn job(prompt: &[u32]) -> (Job, JobHandle) {
        Job::new(prompt.to_vec(), generation()).expect("job")
    }

    #[test]
    fn the_finished_usage_carries_the_prefill_bank_delta() {
        use crate::sched::job::PrefillBankReport;

        fn report(
            passes: u64,
            misses: u64,
            histogram: Vec<u64>,
            bytes: u64,
            ns: u64,
        ) -> PrefillBankReport {
            PrefillBankReport {
                passes,
                misses,
                miss_histogram: histogram.clone(),
                route_histogram: histogram,
                upload_bytes: bytes,
                upload_ns: ns,
            }
        }
        let log = Log::default();
        let mut spy = Spy::new(vec![vec![EOS]], &log);
        // Before the prefill the bank had already served one zero-miss admit; the request's
        // own prefill adds three passes, seven misses and the byte/nanosecond deltas.
        spy.bank_reports = Some((
            report(2, 0, vec![1], 10, 5),
            report(5, 7, vec![1, 0, 1], 110, 50),
        ));
        let engine = start_spy(4, spy);
        let handle = engine.handle();
        let (job, events) = job(&[1, 2, 3]);
        handle.try_submit(job).expect("submit");
        let events = collect(events);
        let report = events
            .iter()
            .find_map(|event| match event {
                JobEvent::Finished(usage) => usage.prefill_bank.as_ref(),
                _ => None,
            })
            .expect("the finished usage carries the report");
        assert_eq!(report.passes, 3);
        let pieces = events
            .iter()
            .find_map(|event| match event {
                JobEvent::Finished(usage) => Some(usage.prefill_pieces),
                _ => None,
            })
            .expect("the finished usage carries the pieces");
        // The three-token prompt runs as one piece of three rows under the four-row chunk.
        assert_eq!(
            pieces,
            PrefillPieces {
                count: 1,
                max_rows: 3
            }
        );
        assert_eq!(report.misses, 7);
        assert_eq!(report.miss_histogram, vec![0, 0, 1]);
        assert_eq!(report.route_histogram, vec![0, 0, 1]);
        assert_eq!(report.upload_bytes, 100);
        assert_eq!(report.upload_ns, 45);
        drop(handle);
        engine.shutdown();
    }

    fn start_spy(chunk: usize, spy: Spy) -> super::Engine {
        spawn_engine(chunk, move |_| {
            Ok((spy, <FakeExecutor as ExecutorOps>::Session::default()))
        })
        .expect("spawn engine")
    }

    /// Spins until the condition holds. It fails the test after ten seconds instead of hanging.
    fn wait_until(what: &str, condition: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !condition() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::yield_now();
        }
    }

    /// Stages are reported in the order the start code announces them,
    /// are visible while the engine is loading, and leave with the
    /// settle: a ready engine has no stage. Two gates make the sequence
    /// deterministic — the engine cannot leave a phase until the test
    /// releases it.
    #[test]
    fn load_stages_report_in_order_and_clear_on_ready() {
        let (release_one, hold_one) = mpsc::channel::<()>();
        let (release_two, hold_two) = mpsc::channel::<()>();
        let engine = spawn_engine::<FakeExecutor, _>(8, move |stages| {
            stages.begin("phase one");
            let _ = hold_one.recv();
            stages.begin("phase two");
            let _ = hold_two.recv();
            Ok((
                FakeExecutor::new(vec![], &[EOS], 100_000),
                <FakeExecutor as ExecutorOps>::Session::default(),
            ))
        })
        .expect("spawn engine");
        let handle = engine.handle();
        wait_until("phase one to appear", || {
            handle.load_stage() == Some("phase one")
        });
        release_one.send(()).expect("release one");
        wait_until("phase two to appear", || {
            handle.load_stage() == Some("phase two")
        });
        release_two.send(()).expect("release two");
        wait_until("the engine to become ready", || {
            handle.state() == super::EngineState::Ready
        });
        assert_eq!(handle.load_stage(), None, "a ready engine reports no stage");
    }

    /// An executor that reports a prefix store of no roots and no entries leaves the handle
    /// saying so by the time the engine is ready; every other store, the default one-entry
    /// store included, keeps the answer true.
    #[test]
    fn the_handle_reports_whether_the_engine_keeps_a_prefix_store() {
        for (roots, entries, kept) in [(0, 0, false), (0, 1, true), (2, 6, true), (1, 0, true)] {
            let engine = spawn_engine::<FakeExecutor, _>(8, move |_| {
                let mut executor = FakeExecutor::new(vec![], &[EOS], 100_000);
                executor.with_prefix_limits(roots, entries);
                Ok((executor, <FakeExecutor as ExecutorOps>::Session::default()))
            })
            .expect("spawn engine");
            let handle = engine.handle();
            wait_until("the engine to become ready", || {
                handle.state() == super::EngineState::Ready
            });
            assert_eq!(
                handle.keeps_prefix_store(),
                kept,
                "roots {roots}, entries {entries}"
            );
            drop(handle);
            engine.shutdown();
        }
    }

    /// A load that announced a stage and then failed leaves the stage
    /// behind with the state change: the error surface keeps its own
    /// failure stage (`load`), never the loading stage.
    #[test]
    fn a_failed_load_clears_its_loading_stage() {
        let engine = spawn_engine::<FakeExecutor, _>(8, |stages| {
            stages.begin("loading weights");
            Err(EngError::Checkpoint("no weights".to_string()))
        })
        .expect("spawn engine");
        let handle = engine.handle();
        wait_until("the engine to fail", || {
            handle.state() == super::EngineState::Error
        });
        assert_eq!(
            handle.load_stage(),
            None,
            "a failed engine reports no stage"
        );
        let failure = handle.failure_info().expect("the failure is recorded");
        assert_eq!(failure.stage, "load");
    }

    fn unavailable_event() -> JobEvent {
        JobEvent::Error(EngError::Backend {
            operation: "engine",
            message: "engine is not available".to_string(),
        })
    }

    fn collect(mut handle: JobHandle) -> Vec<JobEvent> {
        // Started is asserted by its own test; the sequence tests below
        // compare batches and terminal events only.
        let mut events = Vec::new();
        while let Some(event) = handle.events.blocking_recv() {
            if !matches!(event, JobEvent::Started) {
                events.push(event);
            }
        }
        events
    }

    fn batch(tokens: &[u32], text: &str, finish: Option<DecodeFinish>) -> JobEvent {
        JobEvent::Batch(DecodeBatch {
            tokens: tokens.to_vec(),
            text: text.to_string(),
            finish,
        })
    }

    fn finished(log: &Log, prompt: u32, completion: u32) -> JobEvent {
        finished_reusing(log, prompt, completion, 0)
    }

    fn finished_reusing(log: &Log, prompt: u32, completion: u32, reused: u32) -> JobEvent {
        finished_with_prefix(
            log,
            prompt,
            completion,
            reused,
            Some(PrefixReport::default()),
        )
    }

    /// The `Finished` event of a job whose executor reported `prefix`: the fake always reports
    /// its store, so the value is `Some` even without a restore. The piece shape reads the
    /// spy's own `prefill:N` notes, so the expectation is what the executor was fed — the same
    /// fact the production counter logs.
    fn finished_with_prefix(
        log: &Log,
        prompt: u32,
        completion: u32,
        reused: u32,
        prefix: Option<PrefixReport>,
    ) -> JobEvent {
        finished_pieces(prompt, completion, reused, prefix, fed_pieces_of(log))
    }

    /// The `Finished` event with an explicit piece shape, for the multi-log tests that keep
    /// the spy of only one conversation.
    fn finished_pieces(
        prompt: u32,
        completion: u32,
        reused: u32,
        prefix: Option<PrefixReport>,
        pieces: PrefillPieces,
    ) -> JobEvent {
        JobEvent::Finished(JobUsage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            reused_tokens: reused,
            prefix,
            prefill_bank: None,
            prefill_pieces: pieces,
            first_token_margin: None,
        })
    }

    /// The piece shape of a spy log: every `prefill:N` note is one piece. Each job begins
    /// with a `reset` note (`start` resets first, and so does its restart), so the pieces of
    /// the CURRENT job are the `prefill:N` notes after the last `reset` — a multi-job test's
    /// expectation reads its own job, never the earlier ones.
    fn fed_pieces_of(log: &Log) -> PrefillPieces {
        let mut pieces = PrefillPieces::default();
        for note in calls(log) {
            if note == "reset" {
                pieces = PrefillPieces::default();
            } else if let Some(rows) = note.strip_prefix("prefill:") {
                let rows: u32 = rows.parse().unwrap_or(0);
                pieces.count = pieces.count.saturating_add(1);
                pieces.max_rows = pieces.max_rows.max(rows);
            }
        }
        pieces
    }

    /// The report of the fake's store: `entries` entries at a flat 1,000 charged bytes each,
    /// the hit of this job's restore, and its uncached count.
    fn fake_report(entries: u32, hit: PrefixHit, uncached: u32) -> Option<PrefixReport> {
        Some(PrefixReport {
            entries,
            bytes: u64::from(entries) * 1_000,
            hit,
            uncached_shared: uncached,
            shadow: None,
            restore_drops: 0,
            evictions: 0,
            save_side: crate::family::session::SaveSideTotals::default(),
            tier: Default::default(),
        })
    }

    /// The report of the fake's store after one save-side action: `save` is the count the
    /// scenario's own save took (a refusal, a room removal, or an in-place replace).
    fn fake_report_save(
        entries: u32,
        hit: PrefixHit,
        uncached: u32,
        save: crate::family::session::SaveSideTotals,
    ) -> Option<PrefixReport> {
        let mut report = fake_report(entries, hit, uncached).expect("the wrapped report");
        report.save_side = save;
        Some(report)
    }

    /// One save-side refusal: the store returned without storing.
    fn refused_once() -> crate::family::session::SaveSideTotals {
        crate::family::session::SaveSideTotals {
            save_refusals: 1,
            ..Default::default()
        }
    }

    /// One in-place replace: the job's tip dropped the entry it had restored.
    fn replaced_once() -> crate::family::session::SaveSideTotals {
        crate::family::session::SaveSideTotals {
            restored_replaces: 1,
            ..Default::default()
        }
    }

    /// A job whose first `boundary` prompt tokens form the reusable prefix.
    fn job_with_boundary(prompt: &[u32], boundary: usize) -> (Job, JobHandle) {
        let (job, handle) = job(prompt);
        (
            job.with_prefix_boundary(boundary).expect("boundary"),
            handle,
        )
    }

    /// A job whose first `tip` prompt tokens form the reusable prefix with a root at `root`.
    fn job_with_points(prompt: &[u32], tip: usize, root: usize) -> (Job, JobHandle) {
        let (job, handle) = job(prompt);
        let job = job.with_prefix_boundary(tip).expect("tip boundary");
        (job.with_root_boundary(root).expect("root boundary"), handle)
    }

    /// The view of a fake's prefix store that a test reads while the fake runs on its engine
    /// thread.
    type StoreView = Arc<Mutex<PrefixStore<crate::exec_fake::FakeSnapshot, u32>>>;

    /// The stored prefix keys as `(length, is_root)` of a fake's store.
    fn stored(store: &StoreView) -> Vec<(usize, bool)> {
        store.lock().expect("store lock").keys()
    }

    /// Starts an engine on a spy whose fake holds the two-root, six-entry limits and the block
    /// prefix-match with the given spacing, and returns the store view.
    fn start_block_spy(
        chunk: usize,
        script: Vec<Vec<u32>>,
        log: &Log,
        spacing: usize,
    ) -> (super::Engine, StoreView) {
        let mut spy = Spy::new(script, log);
        spy.inner.with_prefix_limits(2, 6);
        spy.inner.with_block_match(spacing);
        let store = spy.inner.prefix_store();
        (start_spy(chunk, spy), store)
    }

    /// Starts an engine on a spy whose fake holds the prefix limits of two roots and six
    /// entries, and returns the store view for assertions.
    fn start_limited_spy(
        chunk: usize,
        script: Vec<Vec<u32>>,
        log: &Log,
    ) -> (super::Engine, StoreView) {
        let mut spy = Spy::new(script, log);
        spy.inner.with_prefix_limits(2, 6);
        let store = spy.inner.prefix_store();
        (start_spy(chunk, spy), store)
    }

    /// The save points of the worker-level root tests: a system prefix of 2,048 tokens of id
    /// 2, a first turn that ends at 2,148, and a tail of 50 tokens of id 5.
    const ROOT: usize = 2_048;
    const TIP: usize = 2_148;

    /// A prompt of 2,198 tokens whose first 2,148 tokens are id 2 and whose last 50 are id 5.
    fn root_tip_prompt() -> Vec<u32> {
        let mut prompt = vec![2; TIP];
        prompt.extend(std::iter::repeat_n(5, 50));
        prompt
    }

    fn run_one(handle: &EngineHandle, job: Job, events: JobHandle) -> Vec<JobEvent> {
        handle.try_submit(job).expect("submit");
        collect(events)
    }

    #[test]
    fn started_arrives_before_the_first_batch_and_queued_returns_to_zero() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let engine = start_spy(2, Spy::new(vec![vec![0]], &log));
        let handle = engine.handle();
        let (job, mut events) = job(&[7]);
        handle.try_submit(job).expect("submit");
        let first = events.events.blocking_recv().expect("first event");
        assert!(matches!(first, JobEvent::Started), "{first:?}");
        while events.events.blocking_recv().is_some() {}
        assert_eq!(handle.queued(), 0);
        drop(engine);
    }

    fn queue_for_test(capacity: usize) -> (EngineHandle, mpsc::Receiver<super::Message>) {
        let (queue, receiver) = mpsc::sync_channel(capacity);
        let shared = Arc::new(super::Shared::new());
        (EngineHandle { queue, shared }, receiver)
    }

    #[test]
    fn queue_ticket_survives_dequeue_before_submit_returns() {
        let (handle, receiver) = queue_for_test(1);
        let (job, _events) = job(&[1]);
        handle
            .try_submit_with(job, |queue, message| {
                assert_eq!(handle.queued(), 1, "reserve before publication");
                super::publish_job(queue, message)?;
                let super::Message::Job(_job, ticket) = receiver.try_recv().expect("job") else {
                    panic!("expected job");
                };
                drop(ticket);
                assert_eq!(handle.queued(), 0, "worker finished before publish returns");
                Ok(())
            })
            .expect("submit");
        assert_eq!(handle.queued(), 0);
    }

    #[test]
    fn queue_ticket_rolls_back_full_and_disconnected_sends() {
        let (handle, receiver) = queue_for_test(1);
        let (first, _first_events) = job(&[1]);
        handle.try_submit(first).expect("first job");
        assert_eq!(handle.queued(), 1);
        let (full, _full_events) = job(&[2]);
        assert_eq!(handle.try_submit(full), Err(SubmitError::QueueFull));
        assert_eq!(handle.queued(), 1, "failed reservation is released");
        drop(receiver);
        assert_eq!(handle.queued(), 0, "receiver drop releases queued jobs");
        let (disconnected, _disconnected_events) = job(&[3]);
        assert_eq!(
            handle.try_submit(disconnected),
            Err(SubmitError::Unavailable)
        );
        assert_eq!(handle.queued(), 0);
    }

    #[test]
    fn queue_ticket_releases_jobs_drained_after_stop() {
        let (handle, receiver) = queue_for_test(2);
        handle.queue.send(super::Message::Stop).expect("stop");
        let (late, _events) = job(&[1]);
        handle.try_submit(late).expect("job behind stop");
        assert_eq!(handle.queued(), 1);
        super::drain(&receiver, super::cancel_unstarted);
        assert_eq!(handle.queued(), 0);
    }

    #[test]
    fn queue_ticket_releases_late_job_when_receiver_drops_after_final_drain() {
        let (handle, receiver) = queue_for_test(1);
        let (late, _events) = job(&[1]);
        handle
            .try_submit_with(late, |queue, message| {
                // The submit already passed its state check. The worker can
                // fail and finish draining before this publication succeeds.
                handle.shared.set_state(EngineState::Error);
                super::drain(&receiver, super::reject);
                super::publish_job(queue, message)
            })
            .expect("late publish");
        assert_eq!(handle.queued(), 1);
        drop(receiver);
        assert_eq!(handle.queued(), 0);
    }

    /// The events of a job that stops at once, uses no extra tokens, and reports a store that
    /// holds nothing and restored nothing.
    fn stop_events(log: &Log, prompt: u32, reused: u32) -> Vec<JobEvent> {
        stop_events_with_prefix(log, prompt, reused, Some(PrefixReport::default()))
    }

    /// `stop_events` with the executor's prefix report of the job.
    fn stop_events_with_prefix(
        log: &Log,
        prompt: u32,
        reused: u32,
        prefix: Option<PrefixReport>,
    ) -> Vec<JobEvent> {
        vec![
            batch(&[EOS], "", Some(DecodeFinish::Stop)),
            finished_with_prefix(log, prompt, 1, reused, prefix),
        ]
    }

    #[test]
    fn a_batch_is_visible_before_deferred_prefix_completion() {
        let log = new_log();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let mut spy = Spy::new(vec![vec![0], vec![EOS]], &log);
        spy.hold_after_emit = Some((entered_tx, release_rx));
        let engine = start_spy(8, spy);
        let (request, mut events) = job(&[2, 2, 2]);
        engine.handle().try_submit(request).expect("submit");
        let entered = entered_rx.recv_timeout(Duration::from_secs(2)).is_ok();
        // Started precedes the first batch; the batch is what this test times.
        let first = loop {
            let event = events.events.try_recv().ok();
            if matches!(event, Some(JobEvent::Started)) {
                continue;
            }
            break event;
        };
        let _ = release_tx.send(());
        let rest = collect(events);
        engine.shutdown();
        assert!(entered, "deferred prefix completion must run");
        assert_eq!(first, Some(batch(&[0], "a", None)));
        assert_eq!(
            rest,
            vec![
                batch(&[EOS], "", Some(DecodeFinish::Stop)),
                finished(&log, 3, 2)
            ]
        );
    }

    #[test]
    fn deferred_prefix_native_failure_reports_error_after_the_batch() {
        let error = || EngError::Backend {
            operation: "test_after_emit",
            message: "prefix transfer failed".to_string(),
        };
        let log = new_log();
        let mut spy = Spy::new(vec![vec![0], vec![EOS]], &log);
        spy.fail_after_emit = Some(error());
        let engine = start_spy(8, spy);
        let (request, events) = job(&[2, 2, 2]);
        engine.handle().try_submit(request).expect("submit");
        assert_eq!(
            collect(events),
            vec![batch(&[0], "a", None), JobEvent::Error(error())]
        );
        assert_eq!(engine.shutdown().state, EngineState::NativeFailed);
    }

    #[test]
    fn a_job_streams_its_batches_then_its_usage() {
        let log = new_log();
        let engine = start_spy(8, Spy::new(vec![vec![0], vec![1], vec![EOS]], &log));
        let (job, handle) = job(&[2, 2, 2]);
        engine.handle().try_submit(job).expect("submit");
        assert_eq!(
            collect(handle),
            vec![
                batch(&[0], "a", None),
                batch(&[1], "b", None),
                batch(&[EOS], "", Some(DecodeFinish::Stop)),
                finished(&log, 3, 3),
            ]
        );
        assert_eq!(engine.shutdown().state, EngineState::Stopped);
    }

    #[test]
    fn a_job_runs_reset_boundary_chunked_prefill_then_decode() {
        let log = new_log();
        let engine = start_spy(2, Spy::new(vec![vec![EOS]; 5], &log));
        let (job, handle) = job(&[1, 1, 1, 1, 1]);
        engine.handle().try_submit(job).expect("submit");
        collect(handle);
        assert_eq!(
            calls(&log),
            strings(&[
                "reset",
                "boundary",
                "prefill:2",
                "prefill:2",
                "prefill:1",
                "decode"
            ])
        );
        engine.shutdown();
    }

    #[test]
    fn request_prefill_piece_is_used_within_the_startup_ceiling() {
        for (limit, expected) in [
            (4, &["prefill:4", "prefill:4", "prefill:2"][..]),
            (16, &["prefill:8", "prefill:2"][..]),
        ] {
            let log = new_log();
            let mut spy = Spy::new(vec![vec![EOS]], &log);
            spy.piece = Some(8);
            spy.request_piece = std::num::NonZeroUsize::new(limit);
            let engine = start_spy(1, spy);
            let (job, handle) = job(&[1; 10]);
            engine.handle().try_submit(job).expect("submit");
            collect(handle);
            assert_eq!(
                calls(&log)
                    .into_iter()
                    .filter(|call| call.starts_with("prefill:"))
                    .collect::<Vec<_>>(),
                strings(expected)
            );
            engine.shutdown();
        }
    }

    #[test]
    fn final_prefill_hook_runs_only_after_prefix_saves_and_with_generation_budget() {
        for (max_tokens, expected) in [
            (
                1,
                &[
                    "prefill:3",
                    "save_prefix:3",
                    "prefill:4",
                    "final_prefill:2",
                    "prefill:2",
                ][..],
            ),
            (
                0,
                &["prefill:3", "save_prefix:3", "prefill:4", "prefill:2"][..],
            ),
        ] {
            let log = new_log();
            let mut spy = Spy::new(vec![], &log);
            spy.log_final_prefill = true;
            let mut session = <Spy as ExecutorOps>::Session::default();
            let tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9];
            let plan = super::PromptPlan {
                tokens: &tokens,
                root: None,
                boundary: Some(3),
            };
            let mut pieces = crate::sched::job::PrefillPieces::default();
            let generation = generation().with_max_tokens(max_tokens);
            assert!(
                super::start(
                    &mut spy,
                    &mut session,
                    &plan,
                    4,
                    &generation,
                    &mut pieces,
                    false,
                )
                .unwrap()
                .is_continue()
            );
            let calls = calls(&log);
            let prefill_calls: Vec<&str> = calls
                .iter()
                .map(String::as_str)
                .filter(|call| call.contains("prefill:") || call.starts_with("save_prefix:"))
                .collect();
            assert_eq!(prefill_calls, expected);
        }
    }

    #[test]
    fn paired_prefill_reports_two_pieces_and_preserves_the_fallback() {
        use crate::sched::job::PrefillPieces;

        for (accept, calls_expected) in [
            (true, &["pair:4+2"][..]),
            (false, &["prefill:4", "prefill:2"][..]),
        ] {
            let log = new_log();
            let mut spy = Spy::new(vec![], &log);
            spy.prefill_pair = accept;
            let mut session = <Spy as ExecutorOps>::Session::default();
            let mut pieces = PrefillPieces::default();
            let tokens = [1, 2, 3, 4, 5, 6];
            assert!(
                super::prefill_chunks(
                    &mut spy,
                    &mut session,
                    &tokens,
                    4,
                    &AtomicBool::new(false),
                    &mut pieces,
                    false,
                )
                .unwrap()
                .is_continue()
            );
            assert_eq!(calls(&log), strings(calls_expected));
            assert_eq!(spy.history(&session), tokens);
            assert_eq!((pieces.count, pieces.max_rows), (2, 4));
        }
    }

    #[test]
    fn cancelling_after_pair_prefill_preserves_the_following_job() {
        for first in [512, 1024] {
            let log = new_log();
            let state = generation();
            let mut spy = Spy::new(vec![vec![0], vec![EOS]], &log);
            spy.prefill_pair = true;
            spy.cancel_after_prefill = Some(Arc::clone(&state.cancel));
            let engine = start_spy(first, spy);
            let handle = engine.handle();
            let (cancelled, events) = Job::new(vec![1; first + 2], state).expect("job");
            handle.try_submit(cancelled).expect("submit");
            assert_eq!(
                collect(events),
                vec![
                    batch(&[], "", Some(DecodeFinish::Cancelled)),
                    finished_pieces(
                        (first + 2) as u32,
                        0,
                        0,
                        Some(PrefixReport::default()),
                        PrefillPieces {
                            count: 2,
                            max_rows: first as u32
                        },
                    ),
                ],
            );
            let (follow_up, events) = job(&[2]);
            handle.try_submit(follow_up).expect("submit following job");
            assert_eq!(
                collect(events),
                vec![
                    batch(&[0], "a", None),
                    batch(&[EOS], "", Some(DecodeFinish::Stop)),
                    finished(&log, 1, 2),
                ],
            );
            assert_eq!(
                calls(&log),
                vec![
                    "reset".into(),
                    "boundary".into(),
                    format!("pair:{first}+2"),
                    "decode".into(),
                    "reset".into(),
                    "boundary".into(),
                    "prefill:1".into(),
                    "decode".into(),
                    "decode".into(),
                ],
            );
            assert_eq!(handle.state(), EngineState::Ready);
            engine.shutdown();
        }
    }

    #[test]
    fn aborting_in_flight_pair_resets_without_decoding_and_keeps_the_worker_ready() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![0], vec![EOS]], &log);
        spy.prefill_pair = true;
        spy.cancel_prefill_pair = true;
        let engine = start_spy(512, spy);
        let handle = engine.handle();
        let (cancelled, events) = job(&[1; 514]);
        handle.try_submit(cancelled).expect("submit");
        assert_eq!(
            collect(events),
            vec![
                batch(&[], "", Some(DecodeFinish::Cancelled)),
                JobEvent::Finished(JobUsage {
                    prompt_tokens: 514,
                    completion_tokens: 0,
                    reused_tokens: 0,
                    prefix: None,
                    prefill_bank: None,
                    prefill_pieces: PrefillPieces::default(),
                    first_token_margin: None,
                }),
            ]
        );
        let (next, events) = job(&[2]);
        handle.try_submit(next).expect("submit following job");
        assert_eq!(
            collect(events),
            vec![
                batch(&[0], "a", None),
                batch(&[EOS], "", Some(DecodeFinish::Stop)),
                finished(&log, 1, 2),
            ]
        );
        assert_eq!(
            calls(&log),
            strings(&[
                "reset",
                "boundary",
                "pair:512+2",
                "reset",
                "reset",
                "boundary",
                "prefill:1",
                "decode",
                "decode",
            ])
        );
        assert_eq!(handle.state(), EngineState::Ready);
        engine.shutdown();
    }

    #[test]
    fn queued_jobs_run_in_order_and_each_starts_from_a_reset_session() {
        let log = new_log();
        let script = vec![vec![0], vec![EOS], vec![1], vec![EOS]];
        let engine = start_spy(8, Spy::new(script, &log));
        let handle = engine.handle();
        let (first, first_events) = job(&[2]);
        let (second, second_events) = job(&[2]);
        handle.try_submit(first).expect("submit first");
        handle.try_submit(second).expect("submit second");
        assert_eq!(
            collect(first_events),
            vec![
                batch(&[0], "a", None),
                batch(&[EOS], "", Some(DecodeFinish::Stop)),
                // The piece shape is pinned, not read back from the live log: the worker
                // resets the next job's spy while this assert runs, and a delayed read of
                // the shared log would see the next job's reset instead of this job's
                // pieces (a delayed read of the shared log; the worker's order is innocent).
                finished_pieces(
                    1,
                    2,
                    0,
                    Some(PrefixReport::default()),
                    PrefillPieces {
                        count: 1,
                        max_rows: 1,
                    },
                ),
            ]
        );
        assert_eq!(
            collect(second_events),
            vec![
                batch(&[1], "b", None),
                batch(&[EOS], "", Some(DecodeFinish::Stop)),
                finished_pieces(
                    1,
                    2,
                    0,
                    Some(PrefixReport::default()),
                    PrefillPieces {
                        count: 1,
                        max_rows: 1,
                    },
                ),
            ]
        );
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn the_queue_holds_32_jobs_behind_the_running_one_and_the_33rd_is_refused() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 33], &log);
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        spy.hold = Some((entered_tx, release_rx));
        let engine = start_spy(8, spy);
        let handle = engine.handle();
        let mut handles = Vec::new();
        let (running, running_events) = job(&[2]);
        handle.try_submit(running).expect("submit the running job");
        handles.push(running_events);
        entered_rx.recv().expect("the worker entered decode");
        assert_eq!(handle.state(), EngineState::Busy);
        for index in 0..32 {
            let (queued, queued_events) = job(&[2]);
            handle
                .try_submit(queued)
                .unwrap_or_else(|error| panic!("queued job {index} refused: {error:?}"));
            handles.push(queued_events);
        }
        let (extra, _extra_events) = job(&[2]);
        assert_eq!(handle.try_submit(extra), Err(SubmitError::QueueFull));
        release_tx.send(()).expect("release the worker");
        for events in handles {
            assert_eq!(
                collect(events).last(),
                // Every queued job ran the same one-token prompt as one piece.
                Some(&finished_pieces(
                    1,
                    1,
                    0,
                    Some(PrefixReport::default()),
                    PrefillPieces {
                        count: 1,
                        max_rows: 1
                    }
                )),
                "a queued job did not finish"
            );
        }
        drop(handle);
        assert_eq!(engine.shutdown().state, EngineState::Stopped);
    }

    #[test]
    fn a_job_cancelled_in_the_queue_never_touches_the_executor() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        spy.hold = Some((entered_tx, release_rx));
        let engine = start_spy(8, spy);
        let handle = engine.handle();
        let (running, running_events) = job(&[2]);
        handle.try_submit(running).expect("submit running");
        entered_rx.recv().expect("the worker entered decode");
        let (queued, queued_events) = job(&[2, 2]);
        handle.try_submit(queued).expect("submit queued");
        queued_events.cancel();
        release_tx.send(()).expect("release the worker");
        collect(running_events);
        assert_eq!(
            collect(queued_events),
            vec![
                batch(&[], "", Some(DecodeFinish::Cancelled)),
                // A queued job never ran, so it carries no executor report and no pieces.
                finished_pieces(2, 0, 0, None, PrefillPieces::default())
            ]
        );
        assert_eq!(
            calls(&log),
            strings(&["reset", "boundary", "prefill:1", "decode"])
        );
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn cancellation_uses_the_updated_depth_piece_not_the_startup_maximum() {
        let log = new_log();
        let state = generation();
        let mut spy = Spy::new(vec![vec![0]], &log);
        spy.piece = Some(512);
        spy.piece_after_prefill = Some(256);
        spy.cancel_after_prefill_call = Some((1, Arc::clone(&state.cancel)));
        let engine = start_spy(512, spy);
        let (job, handle) = Job::new(vec![1; 1538], state).expect("job");
        engine.handle().try_submit(job).expect("submit");
        assert_eq!(
            collect(handle),
            vec![
                batch(&[], "", Some(DecodeFinish::Cancelled)),
                finished(&log, 1538, 0),
            ]
        );
        assert_eq!(
            calls(&log),
            strings(&["reset", "boundary", "prefill:512", "prefill:256", "decode",])
        );
        engine.shutdown();
    }

    #[test]
    fn a_cancel_between_prefill_chunks_skips_the_remaining_chunks() {
        let log = new_log();
        let state = generation();
        let flag = Arc::clone(&state.cancel);
        let mut spy = Spy::new(vec![vec![0]], &log);
        spy.cancel_after_prefill = Some(flag);
        let engine = start_spy(2, spy);
        let (job, handle) = Job::new(vec![1; 6], state).expect("job");
        engine.handle().try_submit(job).expect("submit");
        assert_eq!(
            collect(handle),
            vec![
                batch(&[], "", Some(DecodeFinish::Cancelled)),
                finished(&log, 6, 0)
            ]
        );
        assert_eq!(
            calls(&log),
            strings(&["reset", "boundary", "prefill:2", "decode"])
        );
        engine.shutdown();
    }

    #[test]
    fn a_full_output_channel_cancels_the_job_and_resets_the_session() {
        let log = new_log();
        let mut script = vec![vec![0]; 1_024];
        script.push(vec![EOS]);
        let engine = start_spy(8, Spy::new(script, &log));
        let handle = engine.handle();
        let (slow, slow_events) = job(&[2]);
        let (next, next_events) = job(&[2]);
        handle.try_submit(slow).expect("submit the slow job");
        handle.try_submit(next).expect("submit the next job");
        assert_eq!(
            collect(next_events),
            vec![
                batch(&[EOS], "", Some(DecodeFinish::Stop)),
                finished(&log, 1, 1)
            ]
        );
        assert!(slow_events.is_cancelled(), "the engine did not cancel");
        let mut expected = strings(&["reset", "boundary", "prefill:1"]);
        expected.extend(vec!["decode".to_string(); 1_024]);
        expected.extend(strings(&[
            "reset",
            "reset",
            "boundary",
            "prefill:1",
            "decode",
        ]));
        assert_eq!(calls(&log), expected);
        let delivered = collect(slow_events);
        // The last slot stays free for a terminal event, so a data event needs two free slots.
        assert_eq!(delivered.len(), 1_023);
        assert!(
            delivered
                .iter()
                .all(|event| matches!(event, JobEvent::Batch(_)))
        );
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn a_dropped_receiver_cancels_the_job_after_its_first_batch() {
        let log = new_log();
        let engine = start_spy(8, Spy::new(vec![vec![0], vec![0], vec![EOS]], &log));
        let (job, mut handle) = job(&[2]);
        let cancel = Arc::clone(&job.generation.cancel);
        // Closing the receiver keeps the handle, so the flag stays unset until the engine sets it.
        handle.events.close();
        assert!(!cancel.load(Ordering::SeqCst));
        let engine_handle = engine.handle();
        engine_handle.try_submit(job).expect("submit");
        wait_until("the worker to cancel the job", || calls(&log).len() == 5);
        wait_until("the worker to leave the job", || {
            engine_handle.state() != EngineState::Busy
        });
        assert!(cancel.load(Ordering::SeqCst), "the engine did not cancel");
        assert_eq!(
            calls(&log),
            strings(&["reset", "boundary", "prefill:1", "decode", "reset"])
        );
        drop(engine_handle);
        assert_eq!(engine.shutdown().state, EngineState::Stopped);
        drop(handle);
    }

    #[test]
    fn a_model_output_error_resets_the_job_and_the_next_job_succeeds() {
        let log = new_log();
        let error = EngError::ModelOutput("nonfinite logits".into());
        let mut spy = Spy::new(vec![vec![EOS]; 2], &log);
        spy.fail_first_decode = Some(error.clone());
        let engine = start_spy(8, spy);
        let handle = engine.handle();
        let (failed, failed_events) = job(&[2]);
        let (next, next_events) = job(&[2]);
        handle.try_submit(failed).expect("submit failed job");
        handle.try_submit(next).expect("submit next job");
        assert_eq!(collect(failed_events), vec![JobEvent::Error(error)]);
        assert_eq!(
            collect(next_events),
            vec![
                batch(&[EOS], "", Some(DecodeFinish::Stop)),
                finished(&log, 1, 1)
            ]
        );
        wait_until("the recovered worker to become ready", || {
            handle.state() == EngineState::Ready
        });
        assert!(handle.failure().is_none());
        assert_eq!(
            calls(&log),
            strings(&[
                "reset",
                "boundary",
                "prefill:1",
                "decode",
                "reset",
                "reset",
                "boundary",
                "prefill:1",
                "decode"
            ])
        );
        drop(handle);
        assert_eq!(engine.shutdown().state, EngineState::Stopped);
    }

    #[test]
    fn a_backend_error_fails_the_engine_and_the_queued_jobs() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        spy.hold = Some((entered_tx, release_rx));
        spy.fail_first_decode = Some(EngError::Backend {
            operation: "test",
            message: "boom".to_string(),
        });
        let engine = start_spy(8, spy);
        let handle = engine.handle();
        let (running, running_events) = job(&[2]);
        let (queued, queued_events) = job(&[2]);
        handle.try_submit(running).expect("submit running");
        entered_rx.recv().expect("the worker entered decode");
        handle.try_submit(queued).expect("submit queued");
        release_tx.send(()).expect("release the worker");
        assert_eq!(
            collect(running_events),
            vec![JobEvent::Error(EngError::Backend {
                operation: "test",
                message: "boom".to_string()
            })]
        );
        assert_eq!(collect(queued_events), vec![unavailable_event()]);
        assert_eq!(handle.queued(), 0, "terminal drain releases queue tickets");
        assert_eq!(handle.state(), EngineState::NativeFailed);
        let (late, _late_events) = job(&[2]);
        assert_eq!(handle.try_submit(late), Err(SubmitError::NativeFailed));
        assert_eq!(
            calls(&log),
            strings(&["reset", "boundary", "prefill:1", "decode"])
        );
        drop(handle);
        assert_eq!(engine.shutdown().state, EngineState::NativeFailed);
    }

    #[test]
    fn another_error_fails_only_its_job() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        spy.fail_first_decode = Some(EngError::BadRequest("x".to_string()));
        let engine = start_spy(8, spy);
        let handle = engine.handle();
        let (first, first_events) = job(&[2]);
        let (second, second_events) = job(&[2]);
        handle.try_submit(first).expect("submit first");
        handle.try_submit(second).expect("submit second");
        assert_eq!(
            collect(first_events),
            vec![JobEvent::Error(EngError::BadRequest("x".to_string()))]
        );
        assert_eq!(
            collect(second_events),
            vec![
                batch(&[EOS], "", Some(DecodeFinish::Stop)),
                finished(&log, 1, 1)
            ]
        );
        assert_eq!(handle.state(), EngineState::Ready);
        assert_eq!(
            calls(&log),
            strings(&[
                "reset",
                "boundary",
                "prefill:1",
                "decode",
                "reset",
                "reset",
                "boundary",
                "prefill:1",
                "decode"
            ])
        );
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn a_failed_load_keeps_its_cause_and_fails_queued_and_later_jobs_as_backend() {
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let engine = spawn_engine::<Spy, _>(8, move |_| {
            let _ = release_rx.recv();
            Err(EngError::Checkpoint("no weights".to_string()))
        })
        .expect("spawn engine");
        let handle = engine.handle();
        assert_eq!(handle.state(), EngineState::Loading);
        let (queued, queued_events) = job(&[2]);
        handle.try_submit(queued).expect("submit while loading");
        release_tx.send(()).expect("release the load");
        assert_eq!(collect(queued_events), vec![unavailable_event()]);
        assert_eq!(
            handle.queued(),
            0,
            "failed-load drain releases queue tickets"
        );
        assert_eq!(handle.state(), EngineState::Error);
        let cause = handle.failure().expect("a failed load keeps its cause");
        assert!(cause.contains("no weights"), "cause: {cause}");
        let (late, late_events) = job(&[2]);
        assert_eq!(handle.try_submit(late), Err(SubmitError::LoadFailed));
        assert!(collect(late_events).is_empty());
        drop(handle);
        let report = engine.shutdown();
        assert_eq!(report.state, EngineState::Error);
        assert!(report.failure.expect("cause").contains("no weights"));
    }

    #[test]
    fn a_panic_in_make_sets_the_error_state_and_fails_the_queued_job() {
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let engine = spawn_engine::<Spy, _>(8, move |_| {
            let _ = release_rx.recv();
            panic!("make exploded");
        })
        .expect("spawn engine");
        let handle = engine.handle();
        let (queued, queued_events) = job(&[2]);
        handle.try_submit(queued).expect("submit while loading");
        release_tx.send(()).expect("release the load");
        assert_eq!(collect(queued_events), vec![unavailable_event()]);
        assert_eq!(handle.queued(), 0, "panic drain releases queue tickets");
        wait_until("the Error state", || handle.state() != EngineState::Loading);
        assert_eq!(handle.state(), EngineState::Error);
        let cause = handle.failure().expect("a panic keeps its cause");
        assert!(cause.contains("make exploded"), "cause: {cause}");
        let (late, _late_events) = job(&[2]);
        assert_eq!(handle.try_submit(late), Err(SubmitError::LoadFailed));
        drop(handle);
        let report = engine.shutdown();
        assert_eq!(report.state, EngineState::Error);
        assert!(report.failure.expect("cause").contains("make exploded"));
    }

    #[test]
    fn a_panic_in_decode_fails_the_engine_the_running_job_and_the_queued_job() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        spy.hold = Some((entered_tx, release_rx));
        spy.panic_in_decode = Some("decode exploded");
        let engine = start_spy(8, spy);
        let handle = engine.handle();
        let (running, running_events) = job(&[2]);
        let (queued, queued_events) = job(&[2]);
        handle.try_submit(running).expect("submit running");
        entered_rx.recv().expect("the worker entered decode");
        handle.try_submit(queued).expect("submit queued");
        release_tx.send(()).expect("release the worker");
        assert_eq!(
            collect(running_events),
            vec![JobEvent::Error(EngError::Backend {
                operation: "engine",
                message: "engine panicked: decode exploded".to_string()
            })]
        );
        assert_eq!(collect(queued_events), vec![unavailable_event()]);
        assert_eq!(handle.queued(), 0, "panic drain releases queue tickets");
        assert_eq!(handle.state(), EngineState::NativeFailed);
        let (late, _late_events) = job(&[2]);
        assert_eq!(handle.try_submit(late), Err(SubmitError::NativeFailed));
        assert_eq!(
            calls(&log),
            strings(&["reset", "boundary", "prefill:1", "decode"])
        );
        drop(handle);
        let report = engine.shutdown();
        assert_eq!(report.state, EngineState::NativeFailed);
        assert!(report.failure.expect("cause").contains("decode exploded"));
    }

    #[test]
    fn shutdown_reports_a_panic_that_escapes_the_guard() {
        /// A panic payload whose drop panics, so the unwind guard itself fails.
        struct PanicOnDrop;
        impl Drop for PanicOnDrop {
            fn drop(&mut self) {
                panic!("payload dropped");
            }
        }
        let engine = spawn_engine::<Spy, _>(8, |_| std::panic::panic_any(PanicOnDrop))
            .expect("spawn engine");
        let handle = engine.handle();
        wait_until("the load to end", || handle.state() != EngineState::Loading);
        drop(handle);
        let report = engine.shutdown();
        assert_eq!(report.state, EngineState::Error);
        let cause = report.failure.expect("a panic must be reported");
        assert!(cause.contains("outside the guard"), "cause: {cause}");
    }

    #[test]
    fn a_backend_error_from_prefill_fails_the_engine_with_one_error_and_no_later_call() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        spy.fail_prefill = Some(EngError::Backend {
            operation: "test",
            message: "prefill boom".to_string(),
        });
        let engine = start_spy(8, spy);
        let handle = engine.handle();
        let (first, first_events) = job(&[2]);
        handle.try_submit(first).expect("submit");
        assert_eq!(
            collect(first_events),
            vec![JobEvent::Error(EngError::Backend {
                operation: "test",
                message: "prefill boom".to_string()
            })]
        );
        assert_eq!(handle.state(), EngineState::NativeFailed);
        let (late, _late_events) = job(&[2]);
        assert_eq!(handle.try_submit(late), Err(SubmitError::NativeFailed));
        assert_eq!(calls(&log), strings(&["reset", "boundary", "prefill:1"]));
        let cause = handle.failure().expect("a native failure keeps its cause");
        assert!(cause.contains("prefill boom"), "cause: {cause}");
        drop(handle);
        assert_eq!(engine.shutdown().state, EngineState::NativeFailed);
    }

    #[test]
    fn another_error_from_prefill_fails_only_its_job_and_the_engine_stays_ready() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        spy.fail_prefill = Some(EngError::ContextLimit("too long".to_string()));
        let engine = start_spy(8, spy);
        let handle = engine.handle();
        let (first, first_events) = job(&[2]);
        handle.try_submit(first).expect("submit first");
        assert_eq!(
            collect(first_events),
            vec![JobEvent::Error(EngError::ContextLimit(
                "too long".to_string()
            ))]
        );
        let (second, second_events) = job(&[2]);
        handle.try_submit(second).expect("submit second");
        assert_eq!(
            collect(second_events),
            vec![
                batch(&[EOS], "", Some(DecodeFinish::Stop)),
                finished(&log, 1, 1)
            ]
        );
        assert_eq!(handle.state(), EngineState::Ready);
        assert_eq!(
            calls(&log),
            strings(&[
                "reset",
                "boundary",
                "prefill:1",
                "reset",
                "reset",
                "boundary",
                "prefill:1",
                "decode"
            ])
        );
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn a_failed_reset_at_the_start_of_a_job_fails_the_engine() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        spy.fail_reset_call = Some((
            0,
            EngError::Backend {
                operation: "test",
                message: "reset boom".to_string(),
            },
        ));
        let engine = start_spy(8, spy);
        let handle = engine.handle();
        let (first, first_events) = job(&[2]);
        handle.try_submit(first).expect("submit");
        assert_eq!(
            collect(first_events),
            vec![JobEvent::Error(EngError::Backend {
                operation: "test",
                message: "reset boom".to_string()
            })]
        );
        assert_eq!(handle.state(), EngineState::NativeFailed);
        let (late, _late_events) = job(&[2]);
        assert_eq!(handle.try_submit(late), Err(SubmitError::NativeFailed));
        assert_eq!(calls(&log), strings(&["reset"]));
        drop(handle);
        assert_eq!(engine.shutdown().state, EngineState::NativeFailed);
    }

    #[test]
    fn a_failed_reset_after_a_job_error_fails_the_engine() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        spy.fail_first_decode = Some(EngError::BadRequest("x".to_string()));
        spy.fail_reset_call = Some((
            1,
            EngError::Backend {
                operation: "test",
                message: "reset boom".to_string(),
            },
        ));
        let engine = start_spy(8, spy);
        let handle = engine.handle();
        let (first, first_events) = job(&[2]);
        handle.try_submit(first).expect("submit");
        assert_eq!(
            collect(first_events),
            vec![JobEvent::Error(EngError::BadRequest("x".to_string()))]
        );
        assert_eq!(handle.state(), EngineState::NativeFailed);
        let (late, _late_events) = job(&[2]);
        assert_eq!(handle.try_submit(late), Err(SubmitError::NativeFailed));
        let cause = handle.failure().expect("a failed reset keeps its cause");
        assert!(cause.contains("session reset failed"), "cause: {cause}");
        drop(handle);
        assert_eq!(engine.shutdown().state, EngineState::NativeFailed);
    }

    #[test]
    fn a_failed_reset_after_a_slow_client_cancel_fails_the_engine() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![0]; 1_025], &log);
        spy.fail_reset_call = Some((
            1,
            EngError::Backend {
                operation: "test",
                message: "reset boom".to_string(),
            },
        ));
        let engine = start_spy(8, spy);
        let handle = engine.handle();
        let (slow, slow_events) = job(&[2]);
        handle.try_submit(slow).expect("submit");
        // The owner never reads. The call count is the setup, 1,024 decode calls, and the reset.
        wait_until("the worker to reset after the slow client", || {
            calls(&log).len() == 3 + 1_024 + 1
        });
        wait_until("the worker to leave the job", || {
            handle.state() != EngineState::Busy
        });
        assert!(slow_events.is_cancelled());
        assert_eq!(handle.state(), EngineState::NativeFailed);
        let (late, _late_events) = job(&[2]);
        assert_eq!(handle.try_submit(late), Err(SubmitError::NativeFailed));
        drop(handle);
        engine.shutdown();
    }

    /// Waits until the executor made `call_count` calls and the worker left the job.
    fn wait_job_done(log: &Log, call_count: usize, handle: &EngineHandle) {
        wait_until("the executor calls of the job", || {
            calls(log).len() == call_count
        });
        wait_until("the worker to leave the job", || {
            handle.state() != EngineState::Busy
        });
    }

    /// Fills the output channel of a job to `pending` events before the job starts.
    fn fill(job: &Job, pending: usize) {
        for index in 0..pending {
            job.events
                .try_send(finished_pieces(0, 0, 0, None, PrefillPieces::default()))
                .unwrap_or_else(|error| panic!("filler {index} did not fit: {error:?}"));
        }
    }

    #[test]
    fn an_error_still_arrives_when_1023_events_wait_in_the_channel() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        spy.fail_first_decode = Some(EngError::BadRequest("x".to_string()));
        let engine = start_spy(8, spy);
        let (job, handle) = job(&[2]);
        fill(&job, 1_023);
        let engine_handle = engine.handle();
        engine_handle.try_submit(job).expect("submit");
        // The owner reads only after the job ended, so the channel stays full while it runs.
        // The calls are reset, boundary, prefill, the failed decode, and the reset.
        wait_job_done(&log, 5, &engine_handle);
        let delivered = collect(handle);
        assert_eq!(delivered.len(), 1_024);
        assert_eq!(
            delivered.last(),
            Some(&JobEvent::Error(EngError::BadRequest("x".to_string())))
        );
        engine.shutdown();
    }

    #[test]
    fn the_usage_still_arrives_when_the_finish_batch_makes_1023_events_wait() {
        let log = new_log();
        let engine = start_spy(8, Spy::new(vec![vec![EOS]; 5], &log));
        let (job, handle) = job(&[2]);
        fill(&job, 1_022);
        let engine_handle = engine.handle();
        engine_handle.try_submit(job).expect("submit");
        // The owner reads only after the job ended. The calls are reset, boundary, prefill, decode.
        wait_job_done(&log, 4, &engine_handle);
        let delivered = collect(handle);
        assert_eq!(delivered.len(), 1_024);
        assert_eq!(
            delivered[1_022..],
            [
                batch(&[EOS], "", Some(DecodeFinish::Stop)),
                finished(&log, 1, 1)
            ]
        );
        engine.shutdown();
    }

    #[test]
    fn an_actively_drained_job_can_outlive_its_event_channel_capacity() {
        const TOKENS: usize = 2_100;
        const { assert!(TOKENS > 2 * crate::sched::job::EVENT_CAPACITY) };
        let log = new_log();
        let mut script = vec![vec![0]; TOKENS];
        script.push(vec![EOS]);
        let mut spy = Spy::new(script, &log);
        let (ack, received) = mpsc::channel();
        spy.emit_ack = Some(received);
        let engine = start_spy(8, spy);
        let engine_handle = engine.handle();
        let (job, mut handle) = job(&[2]);
        engine_handle.try_submit(job).expect("submit");
        let mut batches = 0;
        loop {
            match handle.events.blocking_recv().expect("terminal event") {
                JobEvent::Started => {}
                JobEvent::Batch(value) => {
                    let last = batches == TOKENS;
                    assert_eq!(value.tokens, if last { vec![EOS] } else { vec![0] });
                    assert_eq!(
                        value.finish,
                        if last { Some(DecodeFinish::Stop) } else { None }
                    );
                    batches += 1;
                    ack.send(())
                        .expect("allow the next batch only after this one is drained");
                }
                JobEvent::Finished(usage) => {
                    assert_eq!(usage.prompt_tokens, 1);
                    assert_eq!(usage.completion_tokens, (TOKENS + 1) as u32);
                    break;
                }
                JobEvent::Error(error) => panic!("generation failed: {error:?}"),
            }
        }
        assert_eq!(batches, TOKENS + 1);
        assert!(
            !handle.is_cancelled(),
            "a drained stream is not a slow client"
        );
        assert!(handle.events.blocking_recv().is_none());
        drop(engine_handle);
        assert_eq!(engine.shutdown().state, EngineState::Stopped);
    }

    #[test]
    fn a_batch_that_would_take_the_last_slot_cancels_the_job_as_a_slow_client() {
        let log = new_log();
        let engine = start_spy(8, Spy::new(vec![vec![0], vec![EOS]], &log));
        let (job, handle) = job(&[2]);
        fill(&job, 1_022);
        let engine_handle = engine.handle();
        engine_handle.try_submit(job).expect("submit");
        // The call count is reset, boundary, prefill, two decodes, and the reset after the cancel.
        wait_until("the worker to cancel the slow job", || {
            calls(&log).len() == 6
        });
        wait_until("the worker to leave the job", || {
            engine_handle.state() != EngineState::Busy
        });
        assert_eq!(engine_handle.state(), EngineState::Ready);
        assert!(handle.is_cancelled(), "the engine did not cancel");
        drop(engine_handle);
        assert_eq!(engine.shutdown().state, EngineState::Stopped);
        let delivered = collect(handle);
        assert_eq!(delivered.len(), 1_023);
        assert_eq!(delivered[1_022], batch(&[0], "a", None));
    }

    #[test]
    fn a_job_whose_handle_dropped_in_the_queue_never_reaches_the_executor() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        spy.hold = Some((entered_tx, release_rx));
        let engine = start_spy(8, spy);
        let handle = engine.handle();
        let (running, running_events) = job(&[2]);
        let (queued, queued_events) = job(&[2, 2]);
        handle.try_submit(running).expect("submit running");
        entered_rx.recv().expect("the worker entered decode");
        handle.try_submit(queued).expect("submit queued");
        drop(queued_events);
        release_tx.send(()).expect("release the worker");
        assert_eq!(collect(running_events).last(), Some(&finished(&log, 1, 1)));
        drop(handle);
        assert_eq!(engine.shutdown().state, EngineState::Stopped);
        assert_eq!(
            calls(&log),
            strings(&["reset", "boundary", "prefill:1", "decode"])
        );
    }

    #[test]
    fn shutdown_cancels_the_running_and_queued_jobs_while_another_handle_lives() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS], vec![EOS]], &log);
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        spy.hold = Some((entered_tx, release_rx));
        let engine = start_spy(8, spy);
        let handle = engine.handle();
        let (running, running_events) = job(&[2]);
        let (queued, queued_events) = job(&[2]);
        handle.try_submit(running).expect("submit running");
        entered_rx.recv().expect("the worker entered decode");
        handle.try_submit(queued).expect("submit queued");
        let stopper = std::thread::spawn(move || engine.shutdown());
        // `shutdown` sets the stop flag before it cancels the running job.
        wait_until("shutdown to cancel the running job", || {
            running_events.is_cancelled()
        });
        let (late, _late_events) = job(&[2]);
        assert_eq!(handle.try_submit(late), Err(SubmitError::Unavailable));
        release_tx.send(()).expect("release the worker");
        wait_until("the engine thread to stop with a handle alive", || {
            stopper.is_finished()
        });
        let report = stopper.join().expect("shutdown thread");
        assert_eq!(
            report,
            ShutdownReport {
                state: EngineState::Stopped,
                failure: None
            }
        );
        let cancelled = vec![
            batch(&[], "", Some(DecodeFinish::Cancelled)),
            finished(&log, 1, 0),
        ];
        assert_eq!(collect(running_events), cancelled);
        // The queued job never reached the executor, so its usage carries no report and no
        // pieces (the running job it was queued behind holds the log's last segment).
        assert_eq!(
            collect(queued_events),
            vec![
                batch(&[], "", Some(DecodeFinish::Cancelled)),
                finished_pieces(1, 0, 0, None, PrefillPieces::default())
            ]
        );
        assert_eq!(
            calls(&log),
            strings(&["reset", "boundary", "prefill:1", "decode"])
        );
        assert_eq!(handle.state(), EngineState::Stopped);
    }

    #[test]
    fn a_job_queued_behind_the_stop_message_is_cancelled() {
        let log = new_log();
        let spy = Spy::new(vec![vec![EOS]; 5], &log);
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let engine = spawn_engine(8, move |_| {
            let _ = release_rx.recv();
            Ok((spy, <FakeExecutor as ExecutorOps>::Session::default()))
        })
        .expect("spawn engine");
        let handle = engine.handle();
        // A submit that raced with the stop: the job lands behind the stop message.
        let (late, late_events) = job(&[2]);
        handle.queue.send(super::Message::Stop).expect("send stop");
        handle
            .queue
            .send(super::Message::Job(
                late,
                super::QueueTicket::reserve(&handle.shared),
            ))
            .unwrap_or_else(|_| panic!("the queue refused the job"));
        release_tx.send(()).expect("release the load");
        assert_eq!(
            collect(late_events),
            vec![
                batch(&[], "", Some(DecodeFinish::Cancelled)),
                // The late job never ran: no executor report.
                finished_with_prefix(&log, 1, 0, 0, None)
            ]
        );
        assert_eq!(handle.state(), EngineState::Stopped);
        assert_eq!(handle.queued(), 0, "stop drain releases queue tickets");
        assert!(calls(&log).is_empty(), "the executor ran a stopped job");
        drop(handle);
        assert_eq!(engine.shutdown().state, EngineState::Stopped);
    }

    #[test]
    fn a_prompt_that_starts_with_the_saved_key_restores_it_and_prefills_only_the_rest() {
        let log = new_log();
        let engine = start_spy(100, Spy::new(vec![vec![EOS], vec![EOS]], &log));
        let handle = engine.handle();
        // The key of the first job is its first 4 tokens. The second prompt starts with that key.
        let (first, first_events) = job_with_boundary(&[2, 2, 2, 2, 7, 7], 4);
        assert_eq!(
            run_one(&handle, first, first_events),
            stop_events_with_prefix(&log, 6, 0, fake_report(1, PrefixHit::Miss, 0))
        );
        let (second, second_events) = job_with_boundary(&[2, 2, 2, 2, 7, 7, 7, 7, 7], 6);
        assert_eq!(
            run_one(&handle, second, second_events),
            stop_events_with_prefix(
                &log,
                9,
                4,
                fake_report_save(1, PrefixHit::Tip, 0, replaced_once())
            )
        );
        assert_eq!(
            calls(&log),
            strings(&[
                "reset",
                "restore_prefix",
                "boundary",
                "prefill:4",
                "save_prefix:4",
                "prefill:2",
                "decode",
                "reset",
                "restore_prefix",
                "boundary",
                "prefill:2",
                "save_prefix:6",
                "prefill:3",
                "decode",
            ])
        );
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn a_prompt_that_cannot_fit_is_refused_before_any_executor_call() {
        // The spy applies the shared rule of `session::check_prompt` with a limit of 8: a prompt
        // of 8 tokens needs one position for its first sampled token, so only 7 fit.
        const LIMIT: usize = 8;
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS], vec![EOS]], &log);
        spy.context_limit = Some(LIMIT);
        let engine = start_spy(100, spy);
        let handle = engine.handle();
        // The accepted prompt of 7 tokens, one below the limit, saves the key of its first 4.
        let (first, first_events) = job_with_boundary(&[2, 2, 2, 2, 7, 7, 7], 4);
        assert_eq!(
            run_one(&handle, first, first_events),
            stop_events_with_prefix(&log, 7, 0, fake_report(1, PrefixHit::Miss, 0))
        );
        let saved = calls(&log);
        // Every refused prompt starts with the saved key and carries a boundary, so a check that
        // ran only after the restore would have spent the entry before refusing the prompt.
        // The over-limit refusals carry both numbers in the machine-readable fields of the
        // over-length form: the message text is the server's rendering.
        for (name, len, prompt_tokens, limit_tokens) in [
            ("the limit", LIMIT, 8, 7),
            ("one over the limit", LIMIT + 1, 9, 7),
            ("ten times the limit", 10 * LIMIT, 80, 7),
        ] {
            let mut prompt = vec![2, 2, 2, 2];
            prompt.resize(len, 7);
            let (job, events) = job_with_boundary(&prompt, 4);
            assert_eq!(
                run_one(&handle, job, events),
                vec![JobEvent::Error(EngError::PromptTooLong {
                    prompt_tokens,
                    limit_tokens
                })],
                "{name}"
            );
            assert_eq!(calls(&log), saved, "{name}: no executor call may run");
            assert_eq!(handle.state(), EngineState::Ready, "{name}");
        }
        // The next job under the limit restores the entry that the first job saved, so the
        // refused jobs cost it nothing.
        let (next, next_events) = job_with_boundary(&[2, 2, 2, 2, 7, 7, 7], 4);
        assert_eq!(
            run_one(&handle, next, next_events),
            stop_events_with_prefix(&log, 7, 4, fake_report(1, PrefixHit::Tip, 0))
        );
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn a_b_a_anonymous_prompts_start_clean_and_a_repeat_of_a_reuses_its_own_state() {
        let log = new_log();
        let engine = start_spy(100, Spy::new(vec![vec![EOS]; 4], &log));
        let handle = engine.handle();
        let (a1, a1_events) = job_with_boundary(&[2, 2, 2, 2, 2], 3);
        assert_eq!(
            run_one(&handle, a1, a1_events),
            stop_events_with_prefix(&log, 5, 0, fake_report(1, PrefixHit::Miss, 0))
        );
        // B shares no key with A. It restores nothing and replaces the saved state.
        let (b, b_events) = job_with_boundary(&[1, 1, 1, 1, 1], 3);
        assert_eq!(
            run_one(&handle, b, b_events),
            // B's save removes A's entry for room: the churn the old counters could not see.
            stop_events_with_prefix(
                &log,
                5,
                0,
                fake_report_save(
                    1,
                    PrefixHit::Miss,
                    0,
                    crate::family::session::SaveSideTotals {
                        room_removals: 1,
                        ..Default::default()
                    },
                ),
            )
        );
        // A returns after B. The saved state is now B's, so A restores nothing.
        let (a2, a2_events) = job_with_boundary(&[2, 2, 2, 2, 2, 0], 3);
        assert_eq!(
            run_one(&handle, a2, a2_events),
            // Lifetime totals: A2's save makes the second room removal (B's entry went for A2's).
            stop_events_with_prefix(
                &log,
                6,
                0,
                fake_report_save(
                    1,
                    PrefixHit::Miss,
                    0,
                    crate::family::session::SaveSideTotals {
                        room_removals: 2,
                        ..Default::default()
                    },
                ),
            )
        );
        // A third prompt of A starts with the key that A2 saved.
        let (a3, a3_events) = job_with_boundary(&[2, 2, 2, 2, 2, 0, 0], 3);
        assert_eq!(
            run_one(&handle, a3, a3_events),
            // Lifetime: the two room removals of the anonymous round ride every later report.
            stop_events_with_prefix(
                &log,
                7,
                3,
                fake_report_save(
                    1,
                    PrefixHit::Tip,
                    0,
                    crate::family::session::SaveSideTotals {
                        room_removals: 2,
                        ..Default::default()
                    },
                ),
            )
        );
        let clean_start = [
            "reset",
            "restore_prefix",
            "boundary",
            "prefill:3",
            "save_prefix:3",
        ];
        let mut want = strings(&clean_start);
        want.extend(strings(&["prefill:2", "decode"]));
        want.extend(strings(&clean_start));
        want.extend(strings(&["prefill:2", "decode"]));
        want.extend(strings(&clean_start));
        want.extend(strings(&["prefill:3", "decode"]));
        want.extend(strings(&[
            "reset",
            "restore_prefix",
            "boundary",
            "prefill:4",
            "decode",
        ]));
        assert_eq!(calls(&log), want);
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn a_job_without_a_boundary_neither_restores_nor_saves() {
        let log = new_log();
        let engine = start_spy(100, Spy::new(vec![vec![EOS]; 2], &log));
        let handle = engine.handle();
        let (first, first_events) = job_with_boundary(&[2, 2, 2, 2, 2], 3);
        assert_eq!(
            run_one(&handle, first, first_events),
            stop_events_with_prefix(&log, 5, 0, fake_report(1, PrefixHit::Miss, 0))
        );
        let (second, second_events) = job(&[2, 2, 2, 2, 2]);
        // The second job has no boundary: it reports the store the first job filled, with the
        // miss of its own restore-free run.
        assert_eq!(
            run_one(&handle, second, second_events),
            stop_events_with_prefix(&log, 5, 0, fake_report(1, PrefixHit::Miss, 0))
        );
        let calls = calls(&log);
        let tail = calls.get(7..).expect("calls of the second job");
        assert_eq!(
            tail,
            strings(&["reset", "boundary", "prefill:5", "decode"]).as_slice()
        );
        drop(handle);
        engine.shutdown();
    }

    /// The report of a job without a boundary carries the store, not the previous job's hit:
    /// the worker resets every job and skips the restore when the job has no boundary, so the
    /// reset must clear the restore facts. Job 1 stores a root and a tip, job 2 restores the
    /// root and shares 52 tokens with the stored tip beyond it, job 3 runs without a boundary
    /// and must report a miss with nothing uncached.
    ///
    /// Mutation check: not clearing `prefix_hit` and `prefix_uncached` in the fake's `reset`
    /// makes job 3 report the root hit and job 2's 52 shared tokens.
    #[test]
    fn a_job_without_a_boundary_reports_the_store_but_not_the_previous_job_hit() {
        let log = new_log();
        let (engine, store) = start_limited_spy(4_096, vec![vec![EOS]; 3], &log);
        let handle = engine.handle();
        // Job 1 stores the root and the tip of the first conversation.
        let first_prompt = root_tip_prompt();
        let (first, first_events) = job_with_points(&first_prompt, TIP, ROOT);
        assert_eq!(
            run_one(&handle, first, first_events),
            stop_events_with_prefix(&log, 2_198, 0, fake_report(2, PrefixHit::Miss, 0))
        );
        // Job 2 shares the system prefix and 52 tokens more with the stored tip, but diverges
        // before it: it restores the root, and the 52 tokens it shares with the tip beyond the
        // root are uncached (the longest common prefix).
        let mut second_prompt = vec![2; 2_100];
        second_prompt.extend(std::iter::repeat_n(7, 50));
        let (second, second_events) = job_with_points(&second_prompt, 2_100, ROOT);
        assert_eq!(
            run_one(&handle, second, second_events),
            stop_events_with_prefix(
                &log,
                2_150,
                ROOT as u32,
                fake_report(3, PrefixHit::Root, 52)
            )
        );
        assert_eq!(
            stored(&store),
            vec![(ROOT, true), (TIP, false), (2_100, false)],
            "the root, the first tip, and the tip of the second conversation"
        );
        // Job 3 has no boundary: no restore runs, so the report shows the store alone.
        let (third, third_events) = job(&first_prompt);
        assert_eq!(
            run_one(&handle, third, third_events),
            stop_events_with_prefix(&log, 2_198, 0, fake_report(3, PrefixHit::Miss, 0)),
            "the store is reported, the hit of job 2 is not"
        );
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn a_cancel_between_the_chunks_before_the_boundary_skips_the_save() {
        let log = new_log();
        let state = generation();
        let flag = Arc::clone(&state.cancel);
        let mut spy = Spy::new(vec![vec![0]], &log);
        spy.cancel_after_prefill = Some(flag);
        let engine = start_spy(2, spy);
        let (job, handle) = Job::new(vec![1; 6], state).expect("job");
        let job = job.with_prefix_boundary(4).expect("boundary");
        assert_eq!(
            run_one(&engine.handle(), job, handle),
            vec![
                batch(&[], "", Some(DecodeFinish::Cancelled)),
                finished(&log, 6, 0)
            ]
        );
        assert_eq!(
            calls(&log),
            strings(&["reset", "restore_prefix", "boundary", "prefill:2", "decode"])
        );
        engine.shutdown();
    }

    #[test]
    fn a_cancel_after_the_last_chunk_before_the_boundary_skips_the_save() {
        let log = new_log();
        let state = generation();
        let flag = Arc::clone(&state.cancel);
        let mut spy = Spy::new(vec![vec![0]], &log);
        spy.cancel_after_prefill = Some(flag);
        let engine = start_spy(4, spy);
        let (job, handle) = Job::new(vec![1; 6], state).expect("job");
        let job = job.with_prefix_boundary(4).expect("boundary");
        assert_eq!(
            run_one(&engine.handle(), job, handle),
            vec![
                batch(&[], "", Some(DecodeFinish::Cancelled)),
                finished(&log, 6, 0)
            ]
        );
        assert_eq!(
            calls(&log),
            strings(&["reset", "restore_prefix", "boundary", "prefill:4", "decode"])
        );
        engine.shutdown();
    }

    #[test]
    fn a_restore_that_covers_the_whole_prompt_fails_the_job_and_resets_the_session() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        spy.restore_override = Some(5);
        let engine = start_spy(100, spy);
        let handle = engine.handle();
        let (job, events) = job_with_boundary(&[2, 2, 2, 2, 2], 2);
        assert_eq!(
            run_one(&handle, job, events),
            vec![JobEvent::Error(EngError::BadRequest(
                "restored prefix covers the whole prompt".to_string()
            ))]
        );
        assert_eq!(handle.state(), EngineState::Ready);
        assert_eq!(calls(&log), strings(&["reset", "restore_prefix", "reset"]));
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn a_denied_save_does_not_fail_the_job_and_leaves_nothing_to_restore() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 2], &log);
        spy.inner.deny_saves();
        let store = spy.inner.prefix_store();
        let engine = start_spy(100, spy);
        let handle = engine.handle();
        let (first, first_events) = job_with_boundary(&[2, 2, 2, 2, 2], 3);
        assert_eq!(
            run_one(&handle, first, first_events),
            stop_events_with_prefix(
                &log,
                5,
                0,
                fake_report_save(0, PrefixHit::Miss, 0, refused_once())
            )
        );
        let (second, second_events) = job_with_boundary(&[2, 2, 2, 2, 2, 2], 3);
        assert_eq!(
            run_one(&handle, second, second_events),
            stop_events_with_prefix(
                &log,
                6,
                0,
                fake_report_save(
                    0,
                    PrefixHit::Miss,
                    0,
                    crate::family::session::SaveSideTotals {
                        save_refusals: 2,
                        ..Default::default()
                    }
                )
            )
        );
        let want_job = |tail: &str| {
            strings(&[
                "reset",
                "restore_prefix",
                "boundary",
                "prefill:3",
                "save_prefix:3",
                tail,
                "decode",
            ])
        };
        let mut want = want_job("prefill:2");
        want.extend(want_job("prefill:3"));
        assert_eq!(calls(&log), want);
        assert!(
            stored(&store).is_empty(),
            "the denied snapshots left no entry in the store"
        );
        drop(handle);
        engine.shutdown();
    }

    /// Starts an engine on a spy whose fake holds the two-root, six-entry limits and a session
    /// budget: every prefilled or decoded token charges `token_bytes`, every stored entry
    /// `entry_bytes`, against the cap.
    fn start_budgeted_spy(
        chunk: usize,
        script: Vec<Vec<u32>>,
        log: &Log,
        cap: u64,
        token_bytes: u64,
        entry_bytes: u64,
    ) -> (super::Engine, StoreView) {
        let mut spy = Spy::new(script, log);
        spy.inner.with_prefix_limits(2, 6);
        spy.inner.with_session_budget(cap, token_bytes, entry_bytes);
        let store = spy.inner.prefix_store();
        (start_spy(chunk, spy), store)
    }

    /// One dead conversation: a job whose prompt diverges from every other conversation's at
    /// the first token, so its tip entry is its own, and which stops at once.
    fn dead_conversation(handle: &EngineHandle, id: u32, len: usize) {
        // One token past the boundary: the entry's pending token, which the next step feeds.
        let prompt = vec![id; len + 1];
        let (job, events) = job_with_boundary(&prompt, len);
        handle.try_submit(job).expect("submit");
        let events = collect(events);
        assert!(
            matches!(events.last(), Some(JobEvent::Finished(_))),
            "the conversation {id} ran: {events:?}"
        );
    }

    /// A fresh session cannot admit while dead conversations' prefix entries hold the pool (one
    /// observed case: 11,141,120 bytes needed, 7,696,432 available). The worker sizes the
    /// eviction against the pool's ceiling first and then retries the admission ONCE (the batch
    /// worker mirrors this).
    ///
    /// Mutation check: without the eviction call the fourth conversation fails with
    /// `memory_plan` and the assertion on its finished event fails.
    #[test]
    fn an_admission_denial_evicts_prefix_entries_and_retries() {
        let log = new_log();
        // Three dead conversations hold a tip entry each (3,000,000 bytes) beside their
        // committed tokens (103,424 bytes); the fourth conversation needs 4,194,304 bytes
        // of growth against the 1,000,000 that stay free after its reset, which no chunk
        // of it can charge.
        let (engine, store) =
            start_budgeted_spy(8, vec![vec![EOS]; 5], &log, 10_000_000, 1_024, 3_000_000);
        let handle = engine.handle();
        dead_conversation(&handle, 2, 100);
        dead_conversation(&handle, 3, 100);
        dead_conversation(&handle, 4, 100);
        assert_eq!(stored(&store).len(), 3, "three dead tips pin the pool");
        // The fourth conversation: 4,097 tokens of growth against 1,000,000 free.
        let prompt = vec![5; 4_097];
        let (job, events) = job_with_boundary(&prompt, 4_096);
        handle.try_submit(job).expect("submit");
        let events = collect(events);
        assert!(
            matches!(events.last(), Some(JobEvent::Finished(usage)) if usage.prompt_tokens == 4_097),
            "the admission evicted the dead tips and retried: {events:?}"
        );
        assert_eq!(
            stored(&store).len(),
            1,
            "the two-anchor deficit cost the two oldest dead tips; the third survives (the \
             eviction runs until the prompt fits, it does not clear all)"
        );
        drop(handle);
        engine.shutdown();
    }

    /// A clear-all on a denied admission empties the WHOLE LRU - every live conversation's
    /// anchor included - and retries with no snapshots, so the conversations that look up
    /// after the denial find nothing and re-prefill in full (on a multi-agent replay, two
    /// agents restored 0 tokens and took 410 and 421 s against 235 s). The eviction must
    /// free what the denial needs: one entry at a time, so a denial with a one-entry
    /// deficit costs one anchor, not every lineage's. The admission-time protection
    /// (`restore_candidate`) is weaker than the session one. A denied restore does not drop
    /// the anchor outright - it evicts one victim and retries the same candidate - but the
    /// weak protect still loses it on a SECOND denial, or when the candidate is the only
    /// LRU entry (the tier-4 last resort spends it).
    ///
    /// Mutation check: with a clear-all the fourth conversation still finishes (the
    /// retry admits against the fully freed pool), but every anchor is gone and
    /// conversation 3's survival turn restores nothing - its assertion fails.
    #[test]
    fn a_denial_with_a_one_entry_deficit_costs_one_anchor_not_every_lineage() {
        let log = new_log();
        // Three conversations hold a tip anchor each (3,000,000 bytes beside their committed
        // tokens). The fourth conversation needs 3,500 x 1,024 = 3,584,000 bytes against the
        // 689,728..1,000,000 that stay free - a denial whose deficit (2,584,272..2,894,272)
        // ONE anchor covers with room to spare. Six scripts: one per job - three dead
        // conversations, the denied fourth, and the two second turns.
        let (engine, store) =
            start_budgeted_spy(8, vec![vec![EOS]; 6], &log, 10_000_000, 1_024, 3_000_000);
        let handle = engine.handle();
        dead_conversation(&handle, 2, 100);
        dead_conversation(&handle, 3, 100);
        dead_conversation(&handle, 4, 100);
        assert_eq!(stored(&store).len(), 3, "three anchors pin the pool");
        // The fourth conversation: a one-anchor deficit.
        let prompt = vec![5; 3_500];
        let (job, events) = job_with_boundary(&prompt, 3_499);
        handle.try_submit(job).expect("submit");
        let events = collect(events);
        assert!(
            matches!(events.last(), Some(JobEvent::Finished(usage)) if usage.prompt_tokens == 3_500),
            "the retry admits the prompt after the eviction: {events:?}"
        );
        // The first conversation's second turn extends its own tip: the denial's ONE
        // eviction spent a LEAF (every entry here is a tip of its own lineage, so the
        // oldest tip goes first), and the two newer anchors must survive (a clear-all wipes
        // all three).
        let (job, events) = job_with_boundary(&vec![2; 152], 151);
        handle.try_submit(job).expect("submit");
        let events = collect(events);
        assert!(
            matches!(events.last(), Some(JobEvent::Finished(usage)) if usage.reused_tokens == 0),
            "the oldest anchor was the one the denial spent: {events:?}"
        );
        // The newer anchors survive: conversation 3's second turn extends its own tip and must
        // restore it (a clear-all wipes every anchor and this restores nothing). Conversation 4
        // sits on the same store; one survivor discriminates the contract, and the
        // oldest-sacrifice above pins the victim order.
        let (job, events) = job_with_boundary(&vec![3; 152], 151);
        handle.try_submit(job).expect("submit");
        let events = collect(events);
        assert!(
            matches!(events.last(), Some(JobEvent::Finished(usage)) if usage.reused_tokens > 0),
            "conversation 3 restores its anchor after the denial: {events:?}"
        );
        drop(handle);
        engine.shutdown();
    }

    /// The single worker's restored-anchor protection fixture. The denied conversation's
    /// second turn restores its own tip, the growth cannot fit, the pre-pass protects the
    /// restored anchor, and the restart re-restores it (reused > 0) while the oldest OTHER
    /// dead tip spends and the newest survives.
    /// This fixture does NOT discriminate a no-protect mutant - the start RESTORES before it
    /// prefills (exec_fake's restore always succeeds), which moves the denied conversation's
    /// entry to MRU, so the oldest other tip spends with or without protection. It
    /// exercises the wiring only (protect = the session's own restored anchor,
    /// `restored_anchor_of`); the per-slot-vs-global confusion is discriminated at
    /// store level by `an_eviction_protects_the_denied_lineage_not_the_last_restore`
    /// and `a_session_without_a_restore_does_not_protect_another_sessions_root`
    /// (family/session.rs), and the adoption-denial knob test
    /// `an_adoption_denial_evicts_one_victim_and_retries_the_same_candidate_on_the_engine`
    /// (batch_worker_tests) exercises the STORE's own protect at the engine
    /// level: its candidate is the OLDEST entry, so a drop-first or no-protect mutant
    /// spends it and reused_tokens flips 100 -> 0. The worker's protect WIRING is not
    /// discriminated at engine level - this fixture covers the wiring only.
    #[test]
    fn a_denial_keeps_the_denied_jobs_own_restored_anchor_and_spends_a_dead_tip() {
        let log = new_log();
        let (engine, store) =
            start_budgeted_spy(8, vec![vec![EOS]; 6], &log, 10_000_000, 1_024, 3_000_000);
        let handle = engine.handle();
        // The DENIED conversation first: its tip becomes the oldest tier-0 entry.
        dead_conversation(&handle, 4, 100);
        dead_conversation(&handle, 2, 100);
        dead_conversation(&handle, 3, 100);
        assert_eq!(stored(&store).len(), 3, "three tips pin the pool");
        // The denied conversation's second turn: the 3,500 x 1,024 growth cannot fit
        // (a one-anchor deficit). The start RESTORED the tip first (the fake's restore
        // always succeeds), so the pre-pass protects the session's own restored
        // anchor - not a session-less candidate (see the fixture doc above).
        let (job, events) = job_with_boundary(&vec![4; 3_500], 3_499);
        handle.try_submit(job).expect("submit");
        let events = collect(events);
        assert!(
            matches!(events.last(), Some(JobEvent::Finished(usage)) if usage.reused_tokens > 0),
            "the rebuild re-restored the denied job's own anchor: {events:?}"
        );
        // The oldest OTHER tip was the victim: conversation 2's second turn restores
        // nothing, while conversation 3's newer tip survived.
        let (job, events) = job_with_boundary(&vec![2; 152], 151);
        handle.try_submit(job).expect("submit");
        let events = collect(events);
        assert!(
            matches!(events.last(), Some(JobEvent::Finished(usage)) if usage.reused_tokens == 0),
            "the oldest OTHER tip was the denial's victim: {events:?}"
        );
        let (job, events) = job_with_boundary(&vec![3; 152], 151);
        handle.try_submit(job).expect("submit");
        let events = collect(events);
        assert!(
            matches!(events.last(), Some(JobEvent::Finished(usage)) if usage.reused_tokens > 0),
            "the newest tip survived the denial: {events:?}"
        );
        drop(handle);
        engine.shutdown();
    }

    /// A request that cannot fit even in an empty pool must fail in the typed over-length
    /// form - the kind a client answers by compacting the conversation - not as a
    /// session-killer.
    ///
    /// Mutation check: without the classification the event is `MemoryPlan` - a session
    /// killer - and the assertion fails.
    #[test]
    fn an_admission_that_cannot_fit_even_after_eviction_fails_in_the_over_length_form() {
        let log = new_log();
        // One dead conversation holds a tip (3,000,000); the request needs 9,217,024 bytes
        // of growth against a cap of 8,000,000, so even the empty pool cannot admit it. The
        // failed retry leaves 7,990,272 bytes charged to the session, which its next reset
        // releases, so the honest ceiling is (8,000,000 - 1,024) / 1,024 = 7,811 tokens:
        // the largest prompt a fresh attempt could charge, minus the first sampled token.
        let (engine, store) =
            start_budgeted_spy(8, vec![vec![EOS]; 5], &log, 8_000_000, 1_024, 3_000_000);
        let handle = engine.handle();
        dead_conversation(&handle, 2, 100);
        let prompt = vec![5; 9_001];
        let (job, events) = job_with_boundary(&prompt, 9_000);
        handle.try_submit(job).expect("submit");
        let events = collect(events);
        assert!(
            matches!(events.last(), Some(JobEvent::Error(EngError::PromptTooLong { prompt_tokens, limit_tokens }))
                if *prompt_tokens == 9_001 && *limit_tokens == 7_811),
            "the request fails in the over-length form with the pool's honest ceiling: {events:?}"
        );
        assert_eq!(
            stored(&store).len(),
            0,
            "the eviction ran before the classification"
        );
        drop(handle);
        engine.shutdown();
    }

    /// A snapshot that charges successfully on the retry can starve the prefill tail that
    /// follows it, so the eviction retry admits the prompt with NO save points. The tip's
    /// entry charge (3,000,000) beside the prompt (6,145,024) exceeds the cap (8,000,000):
    /// the first attempt saves the tip and is denied mid-tail, and the retry - which with
    /// save points would save the tip again and deny at the same chunk - finishes the
    /// prompt.
    ///
    /// Mutation checks: without the no-points retry the event is
    /// `Error(PromptTooLong { 6001, 4881 })`; without the eviction at all the first
    /// denial stands as the raw memory-plan error.
    #[test]
    fn an_eviction_retry_never_saves_so_a_fitting_prompt_cannot_be_starved() {
        let log = new_log();
        let (engine, store) =
            start_budgeted_spy(8, vec![vec![EOS]; 5], &log, 8_000_000, 1_024, 3_000_000);
        let handle = engine.handle();
        let prompt = vec![5; 6_001];
        let (job, events) = job_with_boundary(&prompt, 2_000);
        handle.try_submit(job).expect("submit");
        let events = collect(events);
        assert!(
            matches!(events.last(), Some(JobEvent::Finished(usage)) if usage.prompt_tokens == 6_001),
            "the retry runs without snapshots and finishes the prompt: {events:?}"
        );
        assert_eq!(
            stored(&store).len(),
            0,
            "the eviction dropped the tip and the retry saved nothing"
        );
        drop(handle);
        engine.shutdown();
    }

    /// The eviction retry runs with no save points at all, so the block cut cannot fire in the
    /// retry window either - the cut flows through `save_points`, which the retry empties, and
    /// the next job rebuilds from scratch (the retry's no-save rule covers the new kind
    /// structurally). The spacing is 1,024 (not 4) so the cut target 1,024 is a chunk end
    /// strictly below the tip 2,000 - with spacing 4 the target 1,996 projected onto the tip
    /// itself and the block candidate never emitted, so the test proved nothing about the retry
    /// window. The first attempt reaches the Block save and stores it; the tip save's first
    /// charge fails (2,048,000 + 2 x 3,000,000 > 8,000,000), it evicts the unprotected Block
    /// and succeeds on retry (2,048,000 + 3,000,000 = 5,048,000), and the tip then starves the
    /// tail: the pool exhausts mid-tail, and the retry resets, attempts a restore on the store
    /// the eviction already emptied, re-feeds the whole prompt, and attempts no new saves.
    #[test]
    fn no_block_cuts_during_an_eviction_retry() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        spy.inner.with_prefix_limits(2, 6);
        spy.inner.with_session_budget(8_000_000, 1_024, 3_000_000);
        spy.inner.with_block_match(1_024);
        let store = spy.inner.prefix_store();
        let engine = start_spy(8, spy);
        let handle = engine.handle();
        let prompt = vec![5; 6_001];
        let (job, events) = job_with_boundary(&prompt, 2_000);
        handle.try_submit(job).expect("submit");
        let events = collect(events);
        assert!(
            matches!(events.last(), Some(JobEvent::Finished(usage)) if usage.prompt_tokens == 6_001),
            "the retry runs without snapshots and finishes the prompt: {events:?}"
        );
        assert_eq!(
            stored(&store).len(),
            0,
            "neither the tip nor the block cut fired in the retry window"
        );
        let calls = calls(&log);
        let saves = calls
            .iter()
            .filter(|call| call.starts_with("save_prefix:"))
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(
            saves.first().map(String::as_str),
            Some("save_prefix:1024"),
            "the first attempt reached the Block save at 1,024: {saves:?}"
        );
        let last_save = calls
            .iter()
            .rposition(|call| call.starts_with("save_prefix:"))
            .expect("the first attempt reached at least the Block save");
        let after_last_save = &calls[last_save + 1..];
        assert!(
            !after_last_save
                .iter()
                .any(|call| call.starts_with("save_prefix:")),
            "the retry window attempted no saves: {after_last_save:?}"
        );
        let reset = after_last_save
            .iter()
            .position(|call| call == "reset")
            .expect("the eviction retry resets the session");
        let retry = &after_last_save[reset + 1..];
        assert_eq!(
            retry.first().map(String::as_str),
            Some("restore_prefix"),
            "the retry attempts a restore on the emptied store: {retry:?}"
        );
        let fed_by_retry = retry
            .iter()
            .filter_map(|call| {
                call.strip_prefix("prefill:")
                    .and_then(|count| count.parse::<usize>().ok())
            })
            .sum::<usize>();
        assert_eq!(
            fed_by_retry, 6_001,
            "the eviction emptied the store, so the retry re-feeds the whole prompt"
        );
        drop(handle);
        engine.shutdown();
    }

    /// An executor that charges the pool for a snapshot inside `save_prefix` denies every
    /// save with a memory-plan error: the first attempt fails at its save, and the retry -
    /// which never saves at all - finishes the prompt.
    ///
    /// Mutation check: without the eviction retry the job fails with the save's
    /// memory-plan error.
    #[test]
    fn an_eviction_retry_skips_a_save_that_cannot_charge_again() {
        let log = new_log();
        // The prompt (6,001 tokens = 6,145,024 bytes) and the first sampled token fit the
        // cap of 8,000,000, but no tip entry (3,000,000 bytes) fits beside them, so every
        // snapshot charge is denied and the retry must finish the prompt without a snapshot.
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        spy.inner.with_prefix_limits(2, 6);
        spy.inner.with_session_budget(8_000_000, 1_024, 3_000_000);
        spy.inner.deny_save_charge();
        let store = spy.inner.prefix_store();
        let engine = start_spy(8, spy);
        let handle = engine.handle();
        let prompt = vec![5; 6_001];
        let (job, events) = job_with_boundary(&prompt, 2_000);
        handle.try_submit(job).expect("submit");
        let events = collect(events);
        assert!(
            matches!(events.last(), Some(JobEvent::Finished(usage)) if usage.prompt_tokens == 6_001),
            "the retry never saves and finishes the prompt: {events:?}"
        );
        assert_eq!(stored(&store).len(), 0, "no snapshot was stored");
        // The retry fed its whole prompt in one pass with no save points: the two
        // attempts together fed 2,000 (the first attempt's head, up to the denied save)
        // + 6,001 (the retry's whole prompt) tokens. A retry that re-fed the head would
        // add 8,001 instead of 6,001.
        let fed = calls(&log)
            .iter()
            .filter_map(|call| {
                call.strip_prefix("prefill:")
                    .and_then(|count| count.parse::<usize>().ok())
            })
            .sum::<usize>();
        assert_eq!(fed, 8_001, "each attempt fed its head once, the tail once");
        drop(handle);
        engine.shutdown();
    }

    /// A denied root save skips only the root and the tip still saves: roots are best effort
    /// inside the cap. The fake denies the charge of the first snapshot call only, which is
    /// the root.
    ///
    /// Mutation check: denying every charge instead makes the store and report rows fail.
    #[test]
    fn a_denied_root_save_skips_only_the_root_and_the_tip_still_saves() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        spy.inner.with_prefix_limits(2, 6);
        spy.inner.deny_first_saves(1);
        let store = spy.inner.prefix_store();
        let engine = start_spy(4_096, spy);
        let prompt = root_tip_prompt();
        let (job, events) = job_with_points(&prompt, TIP, ROOT);
        assert_eq!(
            run_one(&engine.handle(), job, events),
            stop_events_with_prefix(
                &log,
                2_198,
                0,
                fake_report_save(1, PrefixHit::Miss, 0, refused_once())
            ),
            "the job succeeds with the tip alone stored"
        );
        assert_eq!(
            stored(&store),
            vec![(TIP, false)],
            "the denied root left nothing; the tip saved"
        );
        assert_eq!(
            saves_in(&calls(&log)),
            strings(&["save_prefix:2048", "save_prefix:2148"]),
            "both save points ran; only the root charge was denied"
        );
        engine.shutdown();
    }

    #[test]
    fn a_restore_between_the_boundary_and_the_prompt_length_skips_the_save_and_prefills_the_rest() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        // The restore covers 5 of 7 tokens. The job boundary is 3, so it lies inside the restore.
        spy.restore_override = Some(5);
        let engine = start_spy(100, spy);
        let handle = engine.handle();
        let (job, events) = job_with_boundary(&[2, 2, 2, 2, 2, 2, 2], 3);
        assert_eq!(run_one(&handle, job, events), stop_events(&log, 7, 5));
        assert_eq!(
            calls(&log),
            strings(&["reset", "restore_prefix", "boundary", "prefill:2", "decode"])
        );
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn an_idle_engine_advances_background_work_and_drains_it_at_shutdown() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 2], &log);
        spy.background_want = 3;
        let (steps, drained) = (
            Arc::clone(&spy.background_steps),
            Arc::clone(&spy.background_drained),
        );
        let engine = start_spy(100, spy);
        let deadline = Instant::now() + Duration::from_secs(10);
        while steps.load(Ordering::SeqCst) < 3 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(steps.load(Ordering::SeqCst), 3, "the idle worker stepped");
        assert!(!drained.load(Ordering::SeqCst), "no drain before shutdown");
        // A job still runs between steps: the engine answers while background work is pending.
        let handle = engine.handle();
        let (job, events) = job_with_boundary(&[2, 2, 2, 2, 2], 3);
        let events = run_one(&handle, job, events);
        assert!(
            matches!(events.last(), Some(JobEvent::Finished(_))),
            "{events:?}"
        );
        drop(handle);
        engine.shutdown();
        assert!(
            drained.load(Ordering::SeqCst),
            "shutdown drains background work"
        );
    }

    #[test]
    fn a_non_native_error_from_save_prefix_fails_the_job_resets_the_session_and_keeps_the_engine() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        spy.fail_save = Some(EngError::BadRequest("save boom".to_string()));
        let engine = start_spy(100, spy);
        let handle = engine.handle();
        let (first, first_events) = job_with_boundary(&[2, 2, 2, 2, 2], 3);
        assert_eq!(
            run_one(&handle, first, first_events),
            vec![JobEvent::Error(EngError::BadRequest(
                "save boom".to_string()
            ))]
        );
        assert_eq!(handle.state(), EngineState::Ready);
        let (second, second_events) = job_with_boundary(&[2, 2, 2, 2, 2], 3);
        assert_eq!(
            run_one(&handle, second, second_events),
            stop_events_with_prefix(&log, 5, 0, fake_report(1, PrefixHit::Miss, 0))
        );
        assert_eq!(
            calls(&log),
            strings(&[
                "reset",
                "restore_prefix",
                "boundary",
                "prefill:3",
                "save_prefix:3",
                "reset",
                "reset",
                "restore_prefix",
                "boundary",
                "prefill:3",
                "save_prefix:3",
                "prefill:2",
                "decode",
            ])
        );
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn a_backend_error_from_save_prefix_fails_the_engine_with_no_later_call() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        spy.fail_save = Some(EngError::Backend {
            operation: "test",
            message: "save boom".to_string(),
        });
        let engine = start_spy(100, spy);
        let handle = engine.handle();
        let (first, first_events) = job_with_boundary(&[2, 2, 2, 2, 2], 3);
        assert_eq!(
            run_one(&handle, first, first_events),
            vec![JobEvent::Error(EngError::Backend {
                operation: "test",
                message: "save boom".to_string()
            })]
        );
        assert_eq!(handle.state(), EngineState::NativeFailed);
        assert_eq!(
            calls(&log),
            strings(&[
                "reset",
                "restore_prefix",
                "boundary",
                "prefill:3",
                "save_prefix:3"
            ])
        );
        drop(handle);
        assert_eq!(engine.shutdown().state, EngineState::NativeFailed);
    }

    #[test]
    fn a_non_native_error_from_restore_prefix_fails_the_job_resets_the_session_and_keeps_the_engine()
     {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        spy.fail_restore = Some(EngError::BadRequest("restore boom".to_string()));
        let engine = start_spy(100, spy);
        let handle = engine.handle();
        let (first, first_events) = job_with_boundary(&[2, 2, 2, 2, 2], 3);
        assert_eq!(
            run_one(&handle, first, first_events),
            vec![JobEvent::Error(EngError::BadRequest(
                "restore boom".to_string()
            ))]
        );
        assert_eq!(handle.state(), EngineState::Ready);
        let (second, second_events) = job_with_boundary(&[2, 2, 2, 2, 2], 3);
        assert_eq!(
            run_one(&handle, second, second_events),
            stop_events_with_prefix(&log, 5, 0, fake_report(1, PrefixHit::Miss, 0))
        );
        assert_eq!(
            calls(&log),
            strings(&[
                "reset",
                "restore_prefix",
                "reset",
                "reset",
                "restore_prefix",
                "boundary",
                "prefill:3",
                "save_prefix:3",
                "prefill:2",
                "decode",
            ])
        );
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn a_backend_error_from_restore_prefix_fails_the_engine_with_no_later_call() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        spy.fail_restore = Some(EngError::Backend {
            operation: "test",
            message: "restore boom".to_string(),
        });
        let engine = start_spy(100, spy);
        let handle = engine.handle();
        let (first, first_events) = job_with_boundary(&[2, 2, 2, 2, 2], 3);
        assert_eq!(
            run_one(&handle, first, first_events),
            vec![JobEvent::Error(EngError::Backend {
                operation: "test",
                message: "restore boom".to_string()
            })]
        );
        assert_eq!(handle.state(), EngineState::NativeFailed);
        assert_eq!(calls(&log), strings(&["reset", "restore_prefix"]));
        drop(handle);
        assert_eq!(engine.shutdown().state, EngineState::NativeFailed);
    }

    #[test]
    fn an_executor_piece_replaces_the_fallback_and_none_keeps_it() {
        for (piece, want) in [
            (Some(3), strings(&["prefill:3", "prefill:3", "prefill:1"])),
            (
                None,
                strings(&["prefill:2", "prefill:2", "prefill:2", "prefill:1"]),
            ),
        ] {
            let log = new_log();
            let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
            spy.piece = piece;
            let engine = start_spy(2, spy);
            let (job, handle) = job(&[1, 1, 1, 1, 1, 1, 1]);
            engine.handle().try_submit(job).expect("submit");
            collect(handle);
            let got: Vec<String> = calls(&log)
                .into_iter()
                .filter(|entry| entry.starts_with("prefill:"))
                .collect();
            assert_eq!(got, want, "piece {piece:?}");
            engine.shutdown();
        }
    }

    #[test]
    fn an_executor_piece_of_zero_fails_the_engine_load() {
        let log = new_log();
        let mut spy = Spy::new(vec![], &log);
        spy.piece = Some(0);
        // A zero piece fires the note with its failure cause, like a failed
        // make: the sender delivers Failed once and then disconnects, so an
        // engine that never became ready never claims a capability.
        let (tx, rx) = std::sync::mpsc::channel();
        let note: super::ReadyNote = Some(Box::new(move |note| {
            let _ = tx.send(note);
        }));
        let engine = super::spawn_engine_noted(2, note, move |_| {
            Ok((spy, <FakeExecutor as ExecutorOps>::Session::default()))
        })
        .expect("spawn engine");
        let handle = engine.handle();
        wait_until("the load failure", || handle.state() == EngineState::Error);
        assert!(
            matches!(
                rx.recv_timeout(Duration::from_secs(10)),
                Ok(super::EngineNote::Failed(ref cause))
                    if cause.contains("planned a prefill piece of 0")
            ),
            "a zero piece fires the note with its failure cause"
        );
        assert!(
            matches!(
                rx.recv_timeout(Duration::from_secs(10)),
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected)
            ),
            "the note is one-shot: it disconnects after the failure"
        );
        let cause = handle.failure().expect("a piece of 0 keeps its cause");
        assert!(
            cause.contains("planned a prefill piece of 0"),
            "cause: {cause}"
        );
        let (late, late_events) = job(&[2]);
        assert_eq!(handle.try_submit(late), Err(SubmitError::LoadFailed));
        assert!(collect(late_events).is_empty());
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn a_prefill_chunk_of_zero_is_refused() {
        let log = new_log();
        let spy = Spy::new(vec![], &log);
        let error = spawn_engine(0, move |_| {
            Ok((spy, <FakeExecutor as ExecutorOps>::Session::default()))
        })
        .err()
        .expect("a zero chunk must fail");
        assert!(matches!(error, EngError::BadRequest(_)), "got {error:?}");
    }

    /// The save points of a plan over eight tokens with the given tip and root, for a store
    /// with two roots and six entries.
    fn points(
        boundary: Option<usize>,
        root: Option<usize>,
        restored: usize,
    ) -> Vec<(usize, SaveKind)> {
        points_with(
            boundary,
            root,
            restored,
            StoreLimits {
                roots: 2,
                entries: 6,
            },
        )
    }

    /// The save points of a plan over eight tokens for a store with the given limits.
    fn points_with(
        boundary: Option<usize>,
        root: Option<usize>,
        restored: usize,
        limits: StoreLimits,
    ) -> Vec<(usize, SaveKind)> {
        points_full(boundary, root, restored, 256, limits)
    }

    /// The save points of a plan over eight tokens for a store with the given limits and a
    /// prefill piece of `piece` tokens.
    fn points_full(
        boundary: Option<usize>,
        root: Option<usize>,
        restored: usize,
        piece: usize,
        limits: StoreLimits,
    ) -> Vec<(usize, SaveKind)> {
        let tokens = vec![0; 8];
        let plan = super::PromptPlan {
            tokens: &tokens,
            boundary,
            root,
        };
        super::save_points(&plan, restored, limits, piece, None)
    }

    const WIDE: StoreLimits = StoreLimits {
        roots: 2,
        entries: 6,
    };

    const FLASH: StoreLimits = StoreLimits {
        roots: 0,
        entries: 2,
    };

    #[test]
    fn periodic_points_land_on_the_chunk_ends_below_the_tip() {
        // No root, restored 0, piece 256: 16,384 = 64 * 256 and 32,768 = 128 * 256 are chunk
        // ends, so both stay where they are; 49,152 lies past the tip 40,000; the tip follows.
        assert_eq!(
            points_full(Some(40_000), None, 0, 256, WIDE),
            vec![
                (16_384, SaveKind::Periodic),
                (32_768, SaveKind::Periodic),
                (40_000, SaveKind::Tip),
            ],
            "the multiples of 16,384, each on its chunk end, then the tip"
        );
    }

    #[test]
    fn a_root_moves_the_cursor_and_the_periodic_points_follow_the_chunks() {
        // Root 3,000, so the cursor starts the chunks at 3,000: 16,384 moves to
        // 3,000 + ceil(13,384 / 256) * 256 = 3,000 + 53 * 256 = 16,568; 32,768 then moves to
        // 16,568 + ceil(16,200 / 256) * 256 = 16,568 + 64 * 256 = 32,952; the tip 39,990 follows.
        assert_eq!(
            points_full(Some(39_990), Some(3_000), 0, 256, WIDE),
            vec![
                (3_000, SaveKind::Root),
                (16_568, SaveKind::Periodic),
                (32_952, SaveKind::Periodic),
                (39_990, SaveKind::Tip),
            ],
            "the root splits the chunk sequence by design and the periodic points follow"
        );
    }

    #[test]
    fn a_restore_hides_the_periodic_points_it_covers() {
        // Restored 20,000: 16,384 lies below the cursor, so it is no candidate; 32,768 moves to
        // 20,000 + ceil(12,768 / 256) * 256 = 20,000 + 50 * 256 = 32,800; then the tip.
        assert_eq!(
            points_full(Some(40_000), None, 20_000, 256, WIDE),
            vec![(32_800, SaveKind::Periodic), (40_000, SaveKind::Tip)],
            "only the multiples above the restore are candidates"
        );
    }

    #[test]
    fn a_periodic_point_that_reaches_the_tip_is_dropped() {
        // Restored 100: 16,384 moves to 100 + ceil(16,284 / 256) * 256 = 100 + 64 * 256
        // = 16,484, which equals the tip, and the moved position must stay strictly below it.
        assert_eq!(
            points_full(Some(16_484), None, 100, 256, WIDE),
            vec![(16_484, SaveKind::Tip)],
            "a target that snaps onto the tip is dropped; the tip itself runs"
        );
    }

    #[test]
    fn a_restore_that_ends_on_a_multiple_emits_no_empty_segment() {
        // Restored 16,384, no root: the target 16,384 equals the cursor, so it is no candidate
        // (a point there would be an empty segment and a second save of the restored key);
        // 32,768 is a chunk end and stays; the tip follows.
        assert_eq!(
            points_full(Some(40_000), None, 16_384, 256, WIDE),
            vec![(32_768, SaveKind::Periodic), (40_000, SaveKind::Tip)],
            "no point at the restored position"
        );
    }

    /// `points_full` with the block prefix-match spacing: `Some(spacing)` cuts one rolling
    /// intermediate entry at the last block boundary strictly below the tip.
    fn points_block(
        boundary: Option<usize>,
        root: Option<usize>,
        restored: usize,
        piece: usize,
        limits: StoreLimits,
        block: Option<usize>,
    ) -> Vec<(usize, SaveKind)> {
        let tokens = vec![0; 8];
        let plan = super::PromptPlan {
            tokens: &tokens,
            boundary,
            root,
        };
        super::save_points(&plan, restored, limits, piece, block)
    }

    #[test]
    fn a_block_cut_lands_at_the_last_boundary_below_the_tip() {
        // Spacing 4,096, tip 10,000: the last boundary strictly below the tip is
        // floor(9,999 / 4,096) * 4,096 = 8,192, itself a chunk end of the 256 piece, so it
        // stays; no periodic multiple lies below the tip.
        assert_eq!(
            points_block(Some(10_000), None, 0, 256, WIDE, Some(4_096)),
            vec![(8_192, SaveKind::Block), (10_000, SaveKind::Tip)],
            "the cut at the last block boundary, then the tip"
        );
    }

    #[test]
    fn a_root_moves_the_cursor_and_the_cut_keeps_its_boundary() {
        // Root 3,000, so the chunks start at 3,000: the cut target 8,192 is not a chunk
        // end of that cursor (5,192 is not a multiple of 256), so it emits at its true
        // boundary instead of riding to 3,000 + 21 * 256 = 8,376. A root already splits
        // the numeric path by design, and the 184-token ride would push the cut deeper,
        // losing block-match range for tips in between.
        assert_eq!(
            points_block(Some(10_000), Some(3_000), 0, 256, WIDE, Some(4_096)),
            vec![
                (3_000, SaveKind::Root),
                (8_192, SaveKind::Block),
                (10_000, SaveKind::Tip),
            ],
            "the cut at the last block boundary, then the tip"
        );
    }

    #[test]
    fn a_block_cut_at_or_below_the_cursor_is_dropped() {
        // Restored 8,192: the cut target 8,192 equals the cursor and is no candidate (a point
        // there would save the restored key a second time).
        assert_eq!(
            points_block(Some(10_000), None, 8_192, 256, WIDE, Some(4_096)),
            vec![(10_000, SaveKind::Tip)],
            "a covered boundary cuts nothing"
        );
        // Tip 4,096 with spacing 4,096: the last boundary strictly below the tip is 0, which
        // is never a candidate.
        assert_eq!(
            points_block(Some(4_096), None, 0, 256, WIDE, Some(4_096)),
            vec![(4_096, SaveKind::Tip)],
            "a tip at the first boundary cuts nothing below it"
        );
    }

    #[test]
    fn a_block_cut_never_shifts_the_periodic_lattice() {
        // Spacing 4,096, tip 40,000: the periodic multiples stay exactly where the no-cut run
        // puts them (16,384 and 32,768, both chunk ends); the cut target
        // floor(39,999 / 4,096) * 4,096 = 36,864 lands above the deepest periodic, itself a
        // chunk end of the cursor 32,768, so it stays 36,864.
        assert_eq!(
            points_block(Some(40_000), None, 0, 256, WIDE, Some(4_096)),
            vec![
                (16_384, SaveKind::Periodic),
                (32_768, SaveKind::Periodic),
                (36_864, SaveKind::Block),
                (40_000, SaveKind::Tip),
            ],
            "the periodic points keep their positions; the cut rides above the deepest"
        );
    }

    #[test]
    fn a_cut_at_the_cap_evicts_the_second_shallowest_periodic() {
        // Spacing 64, piece 256, restored 0, tip 100,000, entries 6: the six periodics
        // 16,384..98,304 are all chunk ends of the 256 piece. The old plan keeps the
        // deepest five (32,768..98,304). The cut 99,968 - not a chunk end, so the parent
        // projected it to 100,096, past the tip, and dropped it - now enters as a seventh
        // intermediate and is the deepest, so the cap of five keeps 49,152..98,304 plus
        // the Block at 99,968: the 32,768 periodic leaves through the cap, exactly the
        // deepest-intermediate-wins rule the dividing configurations already follow.
        let limits = StoreLimits {
            roots: 2,
            entries: 6,
        };
        assert_eq!(
            save_points_old(Some(100_000), None, 0, 256, limits, Some(64)),
            vec![
                (32_768, SaveKind::Periodic),
                (49_152, SaveKind::Periodic),
                (65_536, SaveKind::Periodic),
                (81_920, SaveKind::Periodic),
                (98_304, SaveKind::Periodic),
                (100_000, SaveKind::Tip),
            ],
            "the old plan keeps the deepest five periodics"
        );
        assert_eq!(
            points_block(Some(100_000), None, 0, 256, limits, Some(64)),
            vec![
                (49_152, SaveKind::Periodic),
                (65_536, SaveKind::Periodic),
                (81_920, SaveKind::Periodic),
                (98_304, SaveKind::Periodic),
                (99_968, SaveKind::Block),
                (100_000, SaveKind::Tip),
            ],
            "the cut enters as the deepest intermediate and evicts 32,768"
        );
    }

    #[test]
    fn a_block_cut_survives_a_piece_that_does_not_divide_the_spacing() {
        // Spacing 64 with pieces 256 and 1,024. The first chunk end at or above the cut falls
        // at or above the tip for most tips (10,100: cut 10,048, first chunk end 10,240 - the
        // parent rule drops the cut there), so it emits at its true boundary instead and
        // `run_job` re-anchors the split chunk.
        for piece in [256usize, 1_024] {
            let new = points_block(Some(10_100), None, 0, piece, WIDE, Some(64));
            assert_eq!(
                new.clone(),
                vec![(10_048, SaveKind::Block), (10_100, SaveKind::Tip)],
                "the rolling cut survives a wide piece"
            );
            // The parent plan drops the cut here, so the two plans differ - the test
            // discriminates the fix, not just the output shape.
            assert_ne!(
                new,
                save_points_old(Some(10_100), None, 0, piece, WIDE, Some(64)),
                "the parent rule must not produce the same plan (piece {piece})"
            );
        }
    }

    #[test]
    fn a_non_dividing_block_cut_emits_at_its_true_position() {
        // Restored 100, piece 256, spacing 64, tip 5,000: the cut 4,992 stays 4,992 instead
        // of riding the restored cursor's chunk lattice to 5,220, which the below-the-tip
        // rule would drop.
        assert_eq!(
            points_block(Some(5_000), None, 100, 256, WIDE, Some(64)),
            vec![(4_992, SaveKind::Block), (5_000, SaveKind::Tip)],
            "the cut lands on its true boundary, not the chunk lattice"
        );
        // Restored 4,992 covers the cut: a point there would save the restored key again.
        assert_eq!(
            points_block(Some(5_000), None, 4_992, 256, WIDE, Some(64)),
            vec![(5_000, SaveKind::Tip)],
            "a covered boundary cuts nothing"
        );
    }

    #[test]
    fn a_shared_boundary_keeps_the_periodic_kind_only_when_it_was_emitted() {
        // Tip 32,769, spacing 64, piece 256, restored 0: the cut 32,768 is also the second
        // periodic multiple, and 32,768 = 128 * 256 is a chunk end of the cursor, so the
        // periodic is emitted first at its own position and the cut, now at or below the
        // cursor, emits nothing - a shared position that SURVIVED its ride keeps the
        // periodic kind, the same rule as the tie below.
        assert_eq!(
            points_block(Some(32_769), None, 0, 256, WIDE, Some(64)),
            vec![
                (16_384, SaveKind::Periodic),
                (32_768, SaveKind::Periodic),
                (32_769, SaveKind::Tip),
            ],
            "an emitted periodic keeps the shared boundary"
        );
        // Restored 100: the periodic 16,384 rides to 16,484 and survives, but the periodic
        // 32,768 rides from there to 32,868, past the tip, and is DROPPED - so the cut
        // lands exactly on the dropped periodic's target and takes the periodic kind, the
        // truthful one: the position is a periodic multiple, the entry persists where the
        // old rule kept nothing at all. The parent had no entry there, so this is a gain.
        assert_eq!(
            points_block(Some(32_769), None, 100, 256, WIDE, Some(64)),
            vec![
                (16_484, SaveKind::Periodic),
                (32_768, SaveKind::Periodic),
                (32_769, SaveKind::Tip),
            ],
            "the cut on a dropped periodic's target takes the periodic kind"
        );
    }

    #[test]
    fn a_deeper_dropped_periodic_above_the_cut_keeps_the_block() {
        // A spacing that does not divide 16,384 (4,095) lets the periodic 16,384 ride the
        // cursor 100's 256-piece lattice to 16,444 - past the tip 16,400 - so it drops;
        // its target 16,384 sits ABOVE the cut 16,380, so the guard that hands the region
        // to a dropped periodic does not fire and the Block emits at its true boundary.
        assert_eq!(
            points_block(Some(16_400), None, 100, 256, WIDE, Some(4_095)),
            vec![(16_380, SaveKind::Block), (16_400, SaveKind::Tip)],
            "the block keeps the region when the dropped periodic's target is above the cut"
        );
    }

    #[test]
    fn the_dividing_plans_are_byte_for_byte_the_chunk_end_rule() {
        // The served configurations: the Flash-Next pieces 1 through 64 against the
        // 64-token spacing, and the Qwen3.6 pieces 16, 128, 256, 512, 1,024, and 2,048
        // against the 4,096-token spacing. Every one of them divides its spacing, so a
        // plan that starts cold - no restore, no root, the cursor on the 0-lattice - must
        // equal the pre-fix projection verbatim, whatever the tip and store limits,
        // including cold tips that cross the deep periodics (20,000 and 40,000); a
        // restore that covers the tip is the same cold shape; and a restored cursor or
        // root that sits ON the piece lattice takes the old path for every target
        // (`(target - cursor) % piece == 0`), so it equals the projection too. An
        // off-lattice cursor is the fix's own subject, pinned by the test below.
        for spacing in [64usize, 4_096] {
            for piece in [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1_024, 2_048] {
                if !spacing.is_multiple_of(piece) {
                    continue;
                }
                for limits in [WIDE, FLASH] {
                    for (tip, restored, root) in [
                        (10_000usize, 0usize, None),
                        (10_000, 10_000, None),
                        (10_000, 12_000, None),
                        (20_000, 0, None),
                        (40_000, 0, None),
                        (5_000, 100, None),
                        (20_000, 7_777, None),
                        (10_000, 0, Some(3_000usize)),
                        (40_000, 7_777, Some(3_000)),
                    ] {
                        let start = root.filter(|_| limits.roots > 0).unwrap_or(restored);
                        let cut = (tip - 1) / spacing * spacing;
                        let old = save_points_old(
                            Some(tip),
                            root,
                            restored,
                            piece,
                            limits,
                            Some(spacing),
                        );
                        if start > 0 && !old.contains(&(cut, SaveKind::Block)) {
                            // A restored or rooted cursor rides the periodic targets onto
                            // their own positions, which can leave the cut off the
                            // cursor's chunk lattice - the fix's own class, pinned by the
                            // test below. What stays here took the old path verbatim: the
                            // old projection's own Block sits at the cut.
                            continue;
                        }
                        assert_eq!(
                            points_block(Some(tip), root, restored, piece, limits, Some(spacing)),
                            old,
                            "dividing piece {piece} keeps the chunk-end rule \
                             (spacing {spacing}, limits {limits:?})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn an_off_lattice_cursor_cuts_at_the_true_boundary_under_a_dividing_piece() {
        // A restored cursor, or a root, moves the chunk lattice off the block boundaries
        // even when the piece divides the spacing: the ride rule would push the cut to a
        // chunk end at or above the tip and drop it - the 12 percent loss class on the
        // served default. The cut emits at its true boundary instead, and the plan keeps
        // every entry the old one kept, with the block the only mover, downward.
        for spacing in [64usize, 4_096] {
            for piece in [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1_024, 2_048] {
                if !spacing.is_multiple_of(piece) {
                    continue;
                }
                for limits in [WIDE, FLASH] {
                    for (tip, restored, root) in [
                        (5_000usize, 100usize, None),
                        (20_000, 7_777, None),
                        (10_000, 0, Some(3_000usize)),
                        (40_000, 7_777, Some(3_000)),
                    ] {
                        let old = save_points_old(
                            Some(tip),
                            root,
                            restored,
                            piece,
                            limits,
                            Some(spacing),
                        );
                        let new =
                            points_block(Some(tip), root, restored, piece, limits, Some(spacing));
                        assert_no_loss_of_old_entries(&new, &old, limits, tip, spacing, piece);
                    }
                }
            }
        }
        // The ruled hand case on the served default: piece 16 divides the 64 spacing, the
        // restored cursor 1,000 is off the 16-lattice, the cut target 4,992 rides to
        // 5,000 - the tip - and the old plan drops it; the new plan keeps it at 4,992.
        assert_eq!(
            points_block(Some(5_000), None, 1_000, 16, FLASH, Some(64)),
            vec![(4_992, SaveKind::Block), (5_000, SaveKind::Tip)],
            "the off-lattice restored cursor keeps the cut at its true boundary"
        );
        assert_eq!(
            save_points_old(Some(5_000), None, 1_000, 16, FLASH, Some(64)),
            vec![(5_000, SaveKind::Tip)],
            "the ride rule drops this cut - the loss this fix closes"
        );
    }

    /// The no-displacement rule: every entry the old plan kept survives in the new plan
    /// at the same position, except a block, which may only move DOWN to its true
    /// boundary (the periodic kind there is the truthful one for a dropped periodic), and
    /// except an eviction through the entries cap, where every kept intermediate is
    /// deeper.
    #[allow(clippy::too_many_arguments)]
    fn assert_no_loss_of_old_entries(
        new: &[(usize, SaveKind)],
        old: &[(usize, SaveKind)],
        limits: StoreLimits,
        tip: usize,
        spacing: usize,
        piece: usize,
    ) {
        let kept: Vec<usize> = new
            .iter()
            .filter(|(_, kind)| !matches!(kind, SaveKind::Tip | SaveKind::Root))
            .map(|&(position, _)| position)
            .collect();
        for &(position, kind) in old {
            match kind {
                SaveKind::Tip | SaveKind::Root => {
                    assert!(
                        new.contains(&(position, kind)),
                        "the {kind:?} at {position} survives \
                         (tip {tip}, spacing {spacing}, piece {piece})"
                    );
                }
                SaveKind::Periodic | SaveKind::Block => {
                    if new.contains(&(position, kind)) {
                        continue;
                    }
                    // The block is the one mover: it lands on the cut target, at or below
                    // the position the old ride gave it, with the truthful kind when a
                    // dropped periodic owns the target.
                    let cut = (tip - 1) / spacing * spacing;
                    let lowered = kind == SaveKind::Block && kept.contains(&cut) && cut <= position;
                    let cap = kept.len() == limits.entries - 1
                        && kept.iter().all(|&kept_position| kept_position > position);
                    assert!(
                        lowered || cap,
                        "the {kind:?} at {position} keeps its position, moves down to {cut}, \
                         or leaves through the cap (tip {tip}, spacing {spacing}, piece {piece})"
                    );
                }
            }
        }
    }

    /// The pre-fix projection, copied verbatim from `save_points` before the wide-piece fix:
    /// every target rides the move-to-chunk-end and below-the-tip rules, block or periodic.
    fn save_points_old(
        boundary: Option<usize>,
        root: Option<usize>,
        restored: usize,
        piece: usize,
        limits: StoreLimits,
        block: Option<usize>,
    ) -> Vec<(usize, SaveKind)> {
        let mut points = Vec::new();
        let mut cursor = restored;
        if let Some(root) = root
            .filter(|_| limits.roots > 0)
            .filter(|&root| root >= super::MIN_ROOT_TOKENS && root > restored)
            .filter(|&root| boundary.is_some_and(|tip| root < tip))
        {
            points.push((root, SaveKind::Root));
            cursor = root;
        }
        if let Some(tip) = boundary.filter(|_| super::cuts_intermediate(limits)) {
            debug_assert!(piece > 0, "the prefill piece is positive");
            let mut targets: Vec<(usize, bool)> = (super::PERIODIC_TOKENS..tip)
                .step_by(super::PERIODIC_TOKENS)
                .map(|target| (target, false))
                .collect();
            if let Some(spacing) = block {
                targets.push(((tip - 1) / spacing * spacing, true));
            }
            targets.sort_unstable();
            let mut intermediate: Vec<(usize, SaveKind)> = Vec::new();
            for &(target, is_block) in &targets {
                if target <= cursor {
                    continue;
                }
                let moved = cursor + (target - cursor).div_ceil(piece) * piece;
                if moved >= tip {
                    continue;
                }
                if is_block
                    && targets.iter().any(|&(candidate, candidate_is_block)| {
                        !candidate_is_block && candidate > target && candidate <= moved
                    })
                {
                    continue;
                }
                intermediate.push((
                    moved,
                    if is_block {
                        SaveKind::Block
                    } else {
                        SaveKind::Periodic
                    },
                ));
                cursor = moved;
            }
            let skip = intermediate.len().saturating_sub(limits.entries - 1);
            points.extend(intermediate.into_iter().skip(skip));
        }
        if let Some(tip) = boundary.filter(|&tip| tip > restored) {
            points.push((tip, SaveKind::Tip));
        }
        points
    }

    #[test]
    fn an_earlier_periodic_that_projects_past_the_cut_subsumes_it() {
        // Restored 32,720, piece 256, spacing 64, tip 33,000: the cut 32,960 is not a
        // periodic boundary, but the periodic 32,768 rides the restored cursor's chunk
        // lattice to 32,976, past the cut, so the periodic kind keeps the region and the
        // block emits nothing - the same periodic-kind-wins rule as the shared boundary.
        assert_eq!(
            points_block(Some(33_000), None, 32_720, 256, WIDE, Some(64)),
            vec![(32_976, SaveKind::Periodic), (33_000, SaveKind::Tip)],
            "a periodic whose chunk end passes the cut subsumes it"
        );
    }

    #[test]
    fn the_true_position_cut_never_displaces_an_old_entry() {
        // Properties, over random tips, cursors, pieces, spacings, roots, and store
        // depths: (a) whatever the spacing, every entry the old plan kept survives in the
        // new one at the same position - the block may only move DOWN to its true
        // boundary, and an entry may only leave through the entries cap; (b) a periodic
        // position appears in the new plan only where the old one kept it, or at or below
        // the cut - the cut landing on a dropped periodic's target (the truthful kind) or
        // a deeper dropped periodic riding from the advanced cursor and surviving: gains,
        // never displacements. A fresh no-root plan under a dividing piece is the exact
        // pre-fix projection - and so is any no-root cursor ON the piece lattice: under a
        // dividing piece every target of the walk is a chunk end of such a cursor, so the
        // true-position condition and the guard never fire and the plan is the old one
        // exactly. The plans also stay strictly increasing and carry at most one block.
        const DIVISORS: [&[usize]; 3] = [
            &[1, 2, 4, 8, 16, 32, 64],
            &[1, 2, 3, 4, 6, 8, 12, 16, 24, 48, 96],
            &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        ];
        let mut state = 0x5eed_1234_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..2_000 {
            let tip = 1 + next() as usize % 200_000;
            let mut restored = next() as usize % tip;
            let mut piece = 1 + next() as usize % 2_048;
            let spacing_idx = next() as usize % 3;
            let spacing = [64usize, 96, 4_096][spacing_idx];
            let mut root = if tip > 2_050 && next() % 4 == 0 {
                Some(2_048 + next() as usize % (tip - 2_049))
            } else {
                None
            };
            // A fixed share of the draws pins the equality classes by construction instead
            // of hoping `restored == 0` lands: every fifth draw is a fresh no-root plan,
            // and every fifth lands the cursor on the piece lattice (a multiple of the
            // piece, no root) - with the piece drawn from the divisors of the spacing, so
            // both classes assert the exact pre-fix projection unconditionally.
            let mut forced_equality = false;
            match next() % 5 {
                0 => {
                    piece = DIVISORS[spacing_idx][next() as usize % DIVISORS[spacing_idx].len()];
                    restored = 0;
                    root = None;
                    forced_equality = true;
                }
                1 => {
                    piece = DIVISORS[spacing_idx][next() as usize % DIVISORS[spacing_idx].len()];
                    let k_max = tip / piece;
                    restored = if k_max == 0 {
                        0
                    } else {
                        piece * (next() as usize % k_max)
                    };
                    root = None;
                    forced_equality = true;
                }
                _ => {}
            }
            let limits = StoreLimits {
                roots: 2,
                entries: 2 + next() as usize % 5,
            };
            let old = save_points_old(Some(tip), root, restored, piece, limits, Some(spacing));
            let new = points_block(Some(tip), root, restored, piece, limits, Some(spacing));
            assert!(
                new.windows(2).all(|pair| pair[0].0 < pair[1].0),
                "positions strictly increase (tip {tip}, cursor {restored})"
            );
            assert!(
                new.iter()
                    .filter(|(_, kind)| *kind == SaveKind::Block)
                    .count()
                    <= 1,
                "one block entry at most (tip {tip})"
            );
            for &(position, kind) in &old {
                if kind == SaveKind::Periodic && !new.contains(&(position, SaveKind::Periodic)) {
                    // A periodic may leave only through the LRU cap: the store keeps the
                    // deepest entries - 1 intermediates, and the cut can add one deeper
                    // entry that evicts it. Every kept intermediate must then be deeper,
                    // and the plan must be at its cap - a shallower survivor would mean the
                    // periodic never became a candidate, the displacement this test bans.
                    let kept: Vec<usize> = new
                        .iter()
                        .filter(|(_, kind)| matches!(kind, SaveKind::Periodic | SaveKind::Block))
                        .map(|&(kept_position, _)| kept_position)
                        .collect();
                    assert_eq!(
                        kept.len(),
                        limits.entries - 1,
                        "the periodic at {position} survives (tip {tip}, cursor {restored}, \
                         piece {piece}, spacing {spacing})"
                    );
                    assert!(
                        kept.iter().all(|&kept_position| kept_position > position),
                        "a periodic only leaves through the LRU cap (tip {tip}, \
                         cursor {restored}, piece {piece}, spacing {spacing})"
                    );
                }
            }
            assert_no_loss_of_old_entries(&new, &old, limits, tip, spacing, piece);
            let new_periodics: Vec<usize> = new
                .iter()
                .filter(|(_, kind)| *kind == SaveKind::Periodic)
                .map(|&(position, _)| position)
                .collect();
            let old_periodics: Vec<usize> = old
                .iter()
                .filter(|(_, kind)| *kind == SaveKind::Periodic)
                .map(|&(position, _)| position)
                .collect();
            // The true-position branch adds candidates - the cut, and any deeper dropped
            // periodic that rides from the advanced cursor and survives - but never moves
            // or invents a shallower one: whenever the branch fires no deeper periodic
            // target that the old rule kept survives it (the guard), and the shallower
            // periodics keep their identical rides; whenever the guard turns the branch
            // off, the plan is the old one exactly.
            let cut = (tip - 1) / spacing * spacing;
            assert!(
                new_periodics
                    .iter()
                    .all(|&position| old_periodics.contains(&position) || position >= cut),
                "a periodic appears only where the old plan kept it, or at or below the \
                 cut (tip {tip}, cursor {restored}, piece {piece}): \
                 old {old_periodics:?} vs new {new_periodics:?}"
            );
            if forced_equality {
                assert_eq!(
                    new, old,
                    "a fresh plan, or a cursor on the piece lattice, under a dividing piece \
                     is the exact pre-fix projection (tip {tip}, cursor {restored}, piece \
                     {piece}, spacing {spacing})"
                );
            } else if root.is_none()
                && spacing.is_multiple_of(piece)
                && restored.is_multiple_of(piece)
                && old.contains(&((tip - 1) / spacing * spacing, SaveKind::Block))
            {
                assert_eq!(
                    new, old,
                    "a fresh plan, or one whose old projection kept its Block at the cut, \
                     under a dividing piece is the exact pre-fix projection (tip {tip}, \
                     cursor {restored}, piece {piece}, spacing {spacing})"
                );
            }
        }
    }

    /// The dividing-piece fix on the full worker path: piece 16 against the 64-token
    /// spacing, a first turn that ends at the off-lattice tip 5,000 (5,000 = 312 x 16 + 8),
    /// and a restored second turn whose cursor 5,000 is off the 16-lattice. The cut
    /// target 9,984 is not a chunk end of that cursor (4,984 = 310 x 16 + 8), and the ride
    /// would land at 5,000 + 312 x 16 = 9,992 - the tip - so the old rule drops the cut
    /// and the seam falls back to the tip; the new plan saves the cut at its true boundary
    /// 9,984 and re-anchors there: 311 chunks of 16, one of 8, the save at 9,984, the
    /// 8-token tail chunk, the tip save, and the one token past the tip. The store rolls
    /// the old Block to the new one and keeps the new Tip.
    #[test]
    fn a_restored_turn_under_a_dividing_piece_keeps_the_cut_at_its_boundary() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 2], &log);
        spy.piece = Some(16);
        spy.inner.with_prefix_limits(2, 6);
        spy.inner.with_block_match(64);
        let store = spy.inner.prefix_store();
        let engine = start_spy(64, spy);
        let handle = engine.handle();
        let (first, first_events) = job_with_boundary(&[2; 5_001], 5_000);
        let first_done = run_one(&handle, first, first_events);
        assert!(
            matches!(first_done.last(), Some(JobEvent::Finished(usage)) if usage.prompt_tokens == 5_001),
            "the first turn finishes: {first_done:?}"
        );
        let mut second_prompt = vec![2; 5_000];
        second_prompt.resize(9_993, 7);
        let (second, second_events) = job_with_boundary(&second_prompt, 9_992);
        let second_done = run_one(&handle, second, second_events);
        assert!(
            matches!(second_done.last(), Some(JobEvent::Finished(usage)) if usage.prompt_tokens == 9_993),
            "the restored turn finishes: {second_done:?}"
        );
        let mut expected = strings(&["reset", "restore_prefix", "boundary"]);
        expected.extend(vec!["prefill:16".to_string(); 312]);
        expected.extend(strings(&[
            "save_prefix:4992",
            "prefill:8",
            "save_prefix:5000",
            "prefill:1",
            "decode",
        ]));
        expected.extend(strings(&["reset", "restore_prefix", "boundary"]));
        expected.extend(vec!["prefill:16".to_string(); 311]);
        expected.extend(strings(&[
            "prefill:8",
            "save_prefix:9984",
            "prefill:8",
            "save_prefix:9992",
            "prefill:1",
            "decode",
        ]));
        assert_eq!(calls(&log), expected, "the restored turn cuts at 9,984");
        assert_eq!(
            stored(&store),
            vec![(9_984, false), (9_992, false)],
            "the rolling Block and the new Tip survive"
        );
        engine.shutdown();
    }

    #[test]
    fn a_wide_piece_job_splits_the_chunk_at_the_cut() {
        // The wide-piece fix on the full worker path: piece 256 against the 64-token block
        // spacing, tip 10,100 of a 10,101-token prompt. The cut 10,048 is not a chunk end
        // of the cursor (the parent rule would ride it to 10,240, past the tip, and drop
        // it), so the job saves it at its true boundary and re-anchors there: 39 chunks of
        // 256, one of 64, the save at 10,048, then the 52-token tail chunk, the tip save,
        // and the one token past the tip.
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]], &log);
        spy.piece = Some(256);
        spy.inner.with_prefix_limits(2, 6);
        spy.inner.with_block_match(64);
        let store = spy.inner.prefix_store();
        let engine = start_spy(4_096, spy);
        let handle = engine.handle();
        let (job, events) = job_with_boundary(&[1; 10_101], 10_100);
        handle.try_submit(job).expect("submit");
        assert_eq!(
            collect(events),
            vec![
                batch(&[EOS], "", Some(DecodeFinish::Stop)),
                finished_with_prefix(&log, 10_101, 1, 0, fake_report(2, PrefixHit::Miss, 0))
            ]
        );
        let mut expected = strings(&["reset", "restore_prefix", "boundary"]);
        expected.extend(vec!["prefill:256".to_string(); 39]);
        expected.extend(strings(&[
            "prefill:64",
            "save_prefix:10048",
            "prefill:52",
            "save_prefix:10100",
            "prefill:1",
            "decode",
        ]));
        assert_eq!(calls(&log), expected, "the chunks re-anchor at the cut");
        // The store holds both saves: the rolling Block at the cut and the Tip - the
        // events pinned the calls, this pins what the store retained.
        assert_eq!(stored(&store), vec![(10_048, false), (10_100, false)]);
        engine.shutdown();
    }

    #[test]
    fn a_block_cut_needs_two_lru_places() {
        // Entries 1: the tip alone fills the LRU, so no intermediate point of any kind runs -
        // the same gate the periodic points already have.
        assert_eq!(
            points_block(
                Some(10_000),
                None,
                0,
                256,
                StoreLimits {
                    roots: 0,
                    entries: 1
                },
                Some(4_096)
            ),
            vec![(10_000, SaveKind::Tip)],
            "no cut beside a tip-only store"
        );
    }

    #[test]
    fn a_block_cut_competes_for_the_deepest_intermediate_places() {
        // Entries 2: one intermediate place beside the tip. The candidates are the periodics
        // 16,384 and 32,768 and the cut 36,864; the deepest survives, and that is the cut.
        assert_eq!(
            points_block(
                Some(40_000),
                None,
                0,
                256,
                StoreLimits {
                    roots: 0,
                    entries: 2
                },
                Some(4_096)
            ),
            vec![(36_864, SaveKind::Block), (40_000, SaveKind::Tip)],
            "the deepest intermediate wins the one place"
        );
    }

    #[test]
    fn a_block_cut_that_coincides_with_a_periodic_point_adds_nothing() {
        // Spacing 16,384 equals the periodic step: the cut target floor(39,999 / 16,384) *
        // 16,384 = 32,768 is the periodic target 32,768, so the lattice already cuts there and
        // the block candidate is dropped - the output equals the no-cut run exactly.
        assert_eq!(
            points_block(Some(40_000), None, 0, 256, WIDE, Some(16_384)),
            vec![
                (16_384, SaveKind::Periodic),
                (32_768, SaveKind::Periodic),
                (40_000, SaveKind::Tip),
            ],
            "a boundary the lattice already cuts adds no second save"
        );
    }

    #[test]
    fn a_block_below_an_unprocessed_periodic_keeps_the_chunk_end_rule() {
        // Spacing 4,095, tip 16,400, piece 256: the piece does not divide the spacing, but the
        // periodic 16,384 lies deeper than the cut 16,380 and is unprocessed, so the cut takes
        // the old rule and rides the chunk lattice to 16,384 - the shared chunk end of both
        // targets - where the tie rule drops it against the periodic. A rolling block is
        // replaced by the next job, a periodic entry persists, so the cut must not displace
        // one: the parent plan [(16,384, Periodic), (16,400, Tip)] stands unchanged.
        assert_eq!(
            points_block(
                Some(16_400),
                None,
                0,
                256,
                StoreLimits {
                    roots: 0,
                    entries: 6
                },
                Some(4_095)
            ),
            vec![(16_384, SaveKind::Periodic), (16_400, SaveKind::Tip)],
            "a deeper unprocessed periodic keeps the chunk-end rule"
        );
    }

    #[test]
    fn a_root_on_a_multiple_is_not_saved_twice() {
        // Root 16,384 moves the cursor to 16,384, so the target 16,384 equals the cursor and
        // is no candidate; 32,768 = 16,384 + 64 * 256 is a chunk end of the root lattice.
        assert_eq!(
            points_full(Some(40_000), Some(16_384), 0, 256, WIDE),
            vec![
                (16_384, SaveKind::Root),
                (32_768, SaveKind::Periodic),
                (40_000, SaveKind::Tip),
            ],
            "the root stands alone at 16,384"
        );
    }

    #[test]
    fn only_the_deepest_periodic_points_that_fit_beside_the_tip_are_emitted() {
        // Tip 262,143 in a prompt of 262,144 tokens: the multiples 16,384 * k for k = 1..15 lie
        // below the tip (16 * 16,384 = 262,144 does not). Limits (2, 6) keep 5 LRU places beside
        // the tip, so the points are the deepest five: k = 11..15.
        // Without a root every multiple is a chunk end of the 256 piece:
        // 11 * 16,384 = 180,224; 196,608; 212,992; 229,376; 245,760.
        assert_eq!(
            points_full(Some(262_143), None, 0, 256, WIDE),
            vec![
                (180_224, SaveKind::Periodic),
                (196_608, SaveKind::Periodic),
                (212_992, SaveKind::Periodic),
                (229_376, SaveKind::Periodic),
                (245_760, SaveKind::Periodic),
                (262_143, SaveKind::Tip),
            ],
            "five periodic points, the deepest ones, then the tip"
        );
        // With a root at 3,000 the chunks start at 3,000, and every step moves by 16,384:
        // the first target 16,384 moves to 3,000 + ceil(13,384 / 256) * 256 = 3,000 + 53 * 256
        // = 16,568, and each next target adds ceil(16,200 / 256) * 256 = 64 * 256 = 16,384
        // to the cursor. The deepest five are 16,568 + 16,384 * j for j = 10..14:
        // 180,408; 196,792; 213,176; 229,560; 245,944.
        assert_eq!(
            points_full(Some(262_143), Some(3_000), 0, 256, WIDE),
            vec![
                (3_000, SaveKind::Root),
                (180_408, SaveKind::Periodic),
                (196_792, SaveKind::Periodic),
                (213_176, SaveKind::Periodic),
                (229_560, SaveKind::Periodic),
                (245_944, SaveKind::Periodic),
                (262_143, SaveKind::Tip),
            ],
            "the root, the deepest five periodic points on the root lattice, then the tip"
        );
    }

    #[test]
    fn two_lru_entries_keep_one_periodic_point_beside_the_tip() {
        // Limits (2, 2): one place for the tip, one for the deepest periodic point, which is
        // 15 * 16,384 = 245,760 below the tip 262,143.
        assert_eq!(
            points_full(
                Some(262_143),
                None,
                0,
                256,
                StoreLimits {
                    roots: 2,
                    entries: 2
                }
            ),
            vec![(245_760, SaveKind::Periodic), (262_143, SaveKind::Tip)],
            "one periodic point"
        );
    }

    #[test]
    fn a_default_store_gets_the_tip_only() {
        // The default limits (0 roots, 1 entry): no root point, no periodic point.
        assert_eq!(
            points_full(Some(39_990), Some(3_000), 0, 256, StoreLimits::default()),
            vec![(39_990, SaveKind::Tip)],
            "a store with one entry and no roots gets the tip only"
        );
    }

    #[test]
    fn a_store_with_one_lru_entry_gets_no_periodic_point() {
        // Limits (2, 1): the root has a place, but a periodic entry would be replaced by the
        // tip at once. With two entries the same prompt gets one periodic point: the points
        // are 16,568 and 32,952 (see the root test), and only the deepest one fits beside the
        // tip.
        assert_eq!(
            points_full(
                Some(39_990),
                Some(3_000),
                0,
                256,
                StoreLimits {
                    roots: 2,
                    entries: 1
                }
            ),
            vec![(3_000, SaveKind::Root), (39_990, SaveKind::Tip)],
            "one LRU entry keeps the tip, so no periodic point is worth a snapshot"
        );
        assert_eq!(
            points_full(
                Some(39_990),
                Some(3_000),
                0,
                256,
                StoreLimits {
                    roots: 2,
                    entries: 2
                }
            ),
            vec![
                (3_000, SaveKind::Root),
                (32_952, SaveKind::Periodic),
                (39_990, SaveKind::Tip),
            ],
            "two LRU entries hold the tip and the deepest periodic point"
        );
    }

    #[test]
    fn a_prompt_shorter_than_the_periodic_stride_has_no_periodic_point() {
        assert_eq!(
            points_full(Some(16_000), None, 0, 256, WIDE),
            vec![(16_000, SaveKind::Tip)],
            "no multiple of 16,384 lies below the tip 16,000"
        );
    }

    #[test]
    fn a_store_without_roots_drops_the_root_point() {
        // The default limits hold zero roots: a root point would only split the prefill.
        for entries in [1, 6] {
            assert_eq!(
                points_with(
                    Some(3_000),
                    Some(2_048),
                    0,
                    StoreLimits { roots: 0, entries }
                ),
                vec![(3_000, SaveKind::Tip)],
                "entries {entries}: no root point without a root place"
            );
        }
        assert_eq!(
            points_with(
                Some(3_000),
                Some(2_048),
                0,
                StoreLimits {
                    roots: 1,
                    entries: 1
                }
            ),
            vec![(2_048, SaveKind::Root), (3_000, SaveKind::Tip)],
            "one root place is enough"
        );
    }

    #[test]
    fn a_root_below_the_minimum_token_count_is_dropped() {
        assert_eq!(
            points(Some(3_000), Some(2_047), 0),
            vec![(3_000, SaveKind::Tip)],
            "a root of 2,047 tokens stays under the 2,048 minimum"
        );
    }

    #[test]
    fn a_root_of_exactly_the_minimum_is_kept_before_the_tip() {
        assert_eq!(
            points(Some(3_000), Some(2_048), 0),
            vec![(2_048, SaveKind::Root), (3_000, SaveKind::Tip)],
            "the root of exactly 2,048 tokens runs first, then the tip"
        );
    }

    #[test]
    fn a_root_at_or_below_the_restored_tokens_is_dropped() {
        for restored in [2_048, 2_100] {
            assert_eq!(
                points(Some(3_000), Some(2_048), restored),
                vec![(3_000, SaveKind::Tip)],
                "restored {restored}: the restore already covered the root"
            );
        }
    }

    #[test]
    fn a_root_at_or_above_the_tip_is_dropped() {
        for root in [3_000, 3_100] {
            assert_eq!(
                points(Some(3_000), Some(root), 0),
                vec![(3_000, SaveKind::Tip)],
                "root {root}: only the tip lies inside the boundary"
            );
        }
    }

    #[test]
    fn a_job_without_a_boundary_has_no_save_points() {
        assert_eq!(points(None, None, 0), Vec::new(), "no boundary");
        assert_eq!(
            points(None, Some(2_048), 0),
            Vec::new(),
            "a root without a tip is impossible and yields nothing"
        );
    }

    #[test]
    fn a_first_job_saves_a_root_and_a_tip_in_the_store() {
        let log = new_log();
        let (engine, store) = start_limited_spy(4_096, vec![vec![EOS]; 5], &log);
        let prompt = root_tip_prompt();
        let (job, events) = job_with_points(&prompt, TIP, ROOT);
        assert_eq!(
            run_one(&engine.handle(), job, events),
            stop_events_with_prefix(&log, 2_198, 0, fake_report(2, PrefixHit::Miss, 0))
        );
        assert_eq!(
            stored(&store),
            vec![(ROOT, true), (TIP, false)],
            "the store holds the pinned root and the tip"
        );
        engine.shutdown();
    }

    #[test]
    fn a_second_conversation_restores_the_root_and_saves_its_own_tip() {
        let log = new_log();
        let (engine, store) = start_limited_spy(4_096, vec![vec![EOS], vec![EOS]], &log);
        let handle = engine.handle();
        let first_prompt = root_tip_prompt();
        let (first, first_events) = job_with_points(&first_prompt, TIP, ROOT);
        assert_eq!(
            run_one(&handle, first, first_events),
            stop_events_with_prefix(&log, 2_198, 0, fake_report(2, PrefixHit::Miss, 0))
        );
        // The same 2,048-token system prefix, a different first turn, and a different tail: the
        // stored tip key does not match, the root key does.
        let mut second_prompt = vec![2; ROOT];
        second_prompt.extend(std::iter::repeat_n(7, 102));
        let (second, second_events) = job_with_points(&second_prompt, 2_100, ROOT);
        assert_eq!(
            run_one(&handle, second, second_events),
            stop_events_with_prefix(&log, 2_150, ROOT as u32, fake_report(3, PrefixHit::Root, 0)),
            "the second conversation reuses the root tokens"
        );
        assert_eq!(
            stored(&store),
            vec![(ROOT, true), (TIP, false), (2_100, false)],
            "the root and the first tip stay; the second tip joins them"
        );
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn a_prompt_that_extends_the_tip_restores_the_tip_not_the_root() {
        let log = new_log();
        let (engine, store) = start_limited_spy(4_096, vec![vec![EOS], vec![EOS]], &log);
        let handle = engine.handle();
        let first_prompt = root_tip_prompt();
        let (first, first_events) = job_with_points(&first_prompt, TIP, ROOT);
        assert_eq!(
            run_one(&handle, first, first_events),
            stop_events_with_prefix(&log, 2_198, 0, fake_report(2, PrefixHit::Miss, 0))
        );
        // A strict extension of the first prompt: the longest stored key, the tip, wins, and
        // the tip that this job saves drops the restored entry (the lineage rule).
        let mut prompt = first_prompt;
        prompt.extend(std::iter::repeat_n(9, 100));
        let (job, events) = job_with_points(&prompt, 2_248, ROOT);
        assert_eq!(
            run_one(&handle, job, events),
            stop_events_with_prefix(
                &log,
                2_298,
                TIP as u32,
                fake_report_save(2, PrefixHit::Tip, 0, replaced_once())
            ),
            "the longest stored key, the tip, covers the restore"
        );
        assert_eq!(
            stored(&store),
            vec![(ROOT, true), (2_248, false)],
            "the new tip replaced the restored one; the root stays pinned"
        );
        drop(handle);
        engine.shutdown();
    }

    /// The tokens a rewritten prompt restores from a block cut count as `reused_tokens` like
    /// any restore, and the report names the class `block`. The first prompt (chunk 4, spacing
    /// 4) cuts a block entry at 2,144, the last 4-token boundary below its tip 2,148; the
    /// rewrite shares the head to 2,146 tokens, so the tip is not a prefix of it but the cut
    /// is, and the cut wins the restore. The restored cut stays stored (rule (f) drops only a
    /// restored tip), so the next rewrite finds it again.
    #[test]
    fn block_restored_tokens_count_as_reused() {
        let log = new_log();
        let (engine, store) = start_block_spy(4, vec![vec![EOS], vec![EOS]], &log, 4);
        let handle = engine.handle();
        let first_prompt = root_tip_prompt();
        let (first, first_events) = job_with_points(&first_prompt, TIP, ROOT);
        assert_eq!(
            run_one(&handle, first, first_events),
            stop_events_with_prefix(&log, 2_198, 0, fake_report(3, PrefixHit::Miss, 0))
        );
        assert_eq!(
            stored(&store),
            vec![(ROOT, true), (2_144, false), (TIP, false)],
            "the first prompt cut a block entry at the last boundary below its tip"
        );
        let mut rewritten = vec![2; 2_146];
        rewritten.extend(std::iter::repeat_n(9, 50));
        let (job, events) = job_with_points(&rewritten, 2_146, ROOT);
        assert_eq!(
            run_one(&handle, job, events),
            stop_events_with_prefix(&log, 2_196, 2_144, fake_report(4, PrefixHit::Block, 2)),
            "the rewrite restores the cut and its tokens count as reused; the two shared \
             tokens past the cut (to the old tip's divergence at 2,146) stay uncached"
        );
        assert_eq!(
            stored(&store),
            vec![(ROOT, true), (TIP, false), (2_144, false), (2_146, false)],
            "the restored cut stays and moves to the most-recently-used end (a restore is \
             a use); the new tip joins it beside the old tip"
        );
        drop(handle);
        engine.shutdown();
    }

    /// The recall-cluster shape of a logged 176K-token conversation: warm append turns
    /// restore the tip and replace it, and a recall that rewrites the last ~23 tokens of the
    /// stable conversation finds no full-prefix tip - the divergence sits above the rolled
    /// block, so the restore reaches the block instead of restoring nothing (the log's
    /// `reused_tokens: 0` cold re-prefill is exactly the no-block behavior). Spacing 1,024
    /// (chunk 4): the base turn's block boundary 3,072 rides above the root 2,048 and below
    /// the tip 3,248; the append's tip save replaces the restored tip, so only the block
    /// remains below the recall's divergence 3,275.
    #[test]
    fn a_recall_rewrite_after_a_warm_append_reaches_the_rolled_block() {
        let log = new_log();
        let (engine, store) = start_block_spy(4, vec![vec![EOS]; 3], &log, 1_024);
        let handle = engine.handle();
        // The base turn: a stable conversation of 3,248 id-2 tokens and a 50-token tail.
        let mut base = vec![2; 3_248];
        base.extend(std::iter::repeat_n(5, 50));
        let (job, events) = job_with_points(&base, 3_248, ROOT);
        assert_eq!(
            run_one(&handle, job, events),
            stop_events_with_prefix(&log, 3_298, 0, fake_report(3, PrefixHit::Miss, 0)),
            "the base turn saves the root, the rolled block, and the tip"
        );
        assert_eq!(
            stored(&store),
            vec![(ROOT, true), (3_072, false), (3_248, false)],
            "the block cut rides at 3,072, the last 1,024 boundary below the tip 3,248"
        );
        // The warm append: the base tail joins the stable conversation and a new message
        // follows, so the turn restores the tip 3,248 (the append-only reuse the log shows)
        // and its save replaces that tip.
        let mut append = base.clone();
        append.extend(std::iter::repeat_n(7, 100));
        let (job, events) = job_with_points(&append, 3_298, ROOT);
        assert_eq!(
            run_one(&handle, job, events),
            stop_events_with_prefix(
                &log,
                3_398,
                3_248,
                fake_report_save(3, PrefixHit::Tip, 0, replaced_once())
            ),
            "the append restores the tip and reuses its tokens"
        );
        assert_eq!(
            stored(&store),
            vec![(ROOT, true), (3_072, false), (3_298, false)],
            "the append's tip replaces the restored one; the block stays for the recall"
        );
        // The recall rewrite: the last 23 tokens of the stable conversation change, so the
        // tip 3,298 is not a prefix of the rewrite; the divergence 3,275 sits above the
        // rolled block 3,072, which IS a full prefix - the restore reaches the block, not 0.
        let mut recall = vec![2; 3_248];
        recall.extend(std::iter::repeat_n(5, 27));
        recall.extend(std::iter::repeat_n(9, 50));
        let (job, events) = job_with_points(&recall, 3_275, ROOT);
        assert_eq!(
            run_one(&handle, job, events),
            stop_events_with_prefix(
                &log,
                3_325,
                3_072,
                fake_report_save(4, PrefixHit::Block, 203, replaced_once())
            ),
            "the rewrite restores the rolled block and its tokens count as reused; the 203 \
             shared tokens past the block stay uncached"
        );
        assert_eq!(
            stored(&store),
            vec![(ROOT, true), (3_298, false), (3_072, false), (3_275, false)],
            "the restored block stays (a restore is a use); the recall's tip joins the store"
        );
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn a_cancel_after_the_root_prefill_and_before_the_root_save_leaves_the_store_as_it_was() {
        let log = new_log();
        let state = generation();
        let flag = Arc::clone(&state.cancel);
        let mut spy = Spy::new(vec![vec![0]], &log);
        spy.inner.with_prefix_limits(2, 6);
        let store = spy.inner.prefix_store();
        spy.cancel_after_prefill_call = Some((0, flag));
        let engine = start_spy(4_096, spy);
        let prompt = root_tip_prompt();
        let (job, handle) = Job::new(prompt, state).expect("job");
        let job = job.with_prefix_boundary(TIP).expect("tip boundary");
        let job = job.with_root_boundary(ROOT).expect("root boundary");
        assert_eq!(
            run_one(&engine.handle(), job, handle),
            vec![
                batch(&[], "", Some(DecodeFinish::Cancelled)),
                finished(&log, 2_198, 0)
            ]
        );
        assert!(stored(&store).is_empty(), "the store changed on a cancel");
        engine.shutdown();
    }

    #[test]
    fn a_cancel_between_the_root_save_and_the_tip_save_keeps_the_root() {
        let log = new_log();
        let state = generation();
        let flag = Arc::clone(&state.cancel);
        let mut spy = Spy::new(vec![vec![0]], &log);
        spy.inner.with_prefix_limits(2, 6);
        let store = spy.inner.prefix_store();
        // The chunk size 2,048 ends the first chunk at the root point, so prefill call 1 is the
        // chunk between the root save and the tip save.
        spy.cancel_after_prefill_call = Some((1, flag));
        let engine = start_spy(2_048, spy);
        let prompt = root_tip_prompt();
        let (job, handle) = Job::new(prompt, state).expect("job");
        let job = job.with_prefix_boundary(TIP).expect("tip boundary");
        let job = job.with_root_boundary(ROOT).expect("root boundary");
        assert_eq!(
            run_one(&engine.handle(), job, handle),
            vec![
                batch(&[], "", Some(DecodeFinish::Cancelled)),
                finished_with_prefix(&log, 2_198, 0, 0, fake_report(1, PrefixHit::Miss, 0))
            ]
        );
        assert_eq!(
            stored(&store),
            vec![(ROOT, true)],
            "the root stays and no tip was saved"
        );
        engine.shutdown();
    }

    #[test]
    fn a_failing_root_snapshot_fails_the_job_and_leaves_the_store_unchanged() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS], vec![EOS]], &log);
        spy.inner.with_prefix_limits(2, 6);
        let store = spy.inner.prefix_store();
        spy.inner
            .fail_snapshots(EngError::Checkpoint("snapshot boom".to_string()));
        let engine = start_spy(4_096, spy);
        let handle = engine.handle();
        let prompt = root_tip_prompt();
        let (first, first_events) = job_with_points(&prompt, TIP, ROOT);
        assert_eq!(
            run_one(&handle, first, first_events),
            vec![JobEvent::Error(EngError::Checkpoint(
                "snapshot boom".to_string()
            ))]
        );
        assert_eq!(handle.state(), EngineState::Ready);
        assert!(
            stored(&store).is_empty(),
            "the store changed on a failed save"
        );
        // The next job runs clean, exactly like a job after a failing tip save.
        let (second, second_events) = job_with_points(&prompt, TIP, ROOT);
        assert_eq!(
            run_one(&handle, second, second_events),
            stop_events_with_prefix(&log, 2_198, 0, fake_report(2, PrefixHit::Miss, 0))
        );
        drop(handle);
        engine.shutdown();
    }

    /// The executor calls of one job on a fake with the given limits (`None` keeps the default
    /// limits), with a chunk of 1,024 tokens, and the store it left. The prompt is
    /// `root_tip_prompt()` with its tip at `TIP`; the job carries a root at `ROOT` only when
    /// `with_root` is set. `entries` is the number of entries the store holds after the job,
    /// for its report.
    fn calls_of_a_root_job(
        limits: Option<(usize, usize)>,
        with_root: bool,
        entries: u32,
    ) -> (Vec<String>, Vec<(usize, bool)>) {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        if let Some((roots, entries)) = limits {
            spy.inner.with_prefix_limits(roots, entries);
        }
        let store = spy.inner.prefix_store();
        let engine = start_spy(1_024, spy);
        let prompt = root_tip_prompt();
        let (job, events) = if with_root {
            job_with_points(&prompt, TIP, ROOT)
        } else {
            job_with_boundary(&prompt, TIP)
        };
        assert_eq!(
            run_one(&engine.handle(), job, events),
            stop_events_with_prefix(&log, 2_198, 0, fake_report(entries, PrefixHit::Miss, 0))
        );
        engine.shutdown();
        (calls(&log), stored(&store))
    }

    #[test]
    fn with_the_default_limits_a_root_boundary_changes_nothing() {
        // Restored 0, root 2,048, tip 2,148, chunk 1,024. A root point would split the head
        // into 1,024 + 1,024 | 100; without it the head is 1,024 + 1,024 + 100. The default
        // store keeps no roots, so both jobs must run the same pieces, with one save at the tip.
        let without_root = calls_of_a_root_job(None, false, 1);
        assert_eq!(
            without_root.0,
            strings(&[
                "reset",
                "restore_prefix",
                "boundary",
                "prefill:1024",
                "prefill:1024",
                "prefill:100",
                "save_prefix:2148",
                "prefill:50",
                "decode",
            ]),
            "the job without a root"
        );
        assert_eq!(without_root.1, vec![(TIP, false)], "its store");
        let with_root = calls_of_a_root_job(None, true, 1);
        assert_eq!(
            with_root, without_root,
            "a root boundary changed the calls or the store under the default limits"
        );
        assert!(
            !with_root.0.contains(&"save_prefix:2048".to_string()),
            "a root save was attempted"
        );
    }

    #[test]
    fn with_roots_enabled_the_root_point_splits_the_prefill_at_the_root() {
        let (log, store) = calls_of_a_root_job(Some((2, 6)), true, 2);
        assert_eq!(
            log,
            strings(&[
                "reset",
                "restore_prefix",
                "boundary",
                "prefill:1024",
                "prefill:1024",
                "save_prefix:2048",
                "prefill:100",
                "save_prefix:2148",
                "prefill:50",
                "decode",
            ]),
            "the root point ends the second piece and saves"
        );
        assert_eq!(store, vec![(ROOT, true), (TIP, false)], "root and tip");
    }

    #[test]
    fn a_failing_tip_save_after_a_root_save_fails_the_job_and_keeps_the_root() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        spy.inner.with_prefix_limits(2, 6);
        let store = spy.inner.prefix_store();
        // Save call 0 is the root, save call 1 is the tip.
        spy.fail_save_call = Some((1, EngError::BadRequest("save boom".to_string())));
        let engine = start_spy(4_096, spy);
        let handle = engine.handle();
        let prompt = root_tip_prompt();
        let (first, first_events) = job_with_points(&prompt, TIP, ROOT);
        assert_eq!(
            run_one(&handle, first, first_events),
            vec![JobEvent::Error(EngError::BadRequest(
                "save boom".to_string()
            ))],
            "the job fails like a job with a failing tip save"
        );
        assert_eq!(handle.state(), EngineState::Ready);
        assert_eq!(
            calls(&log),
            strings(&[
                "reset",
                "restore_prefix",
                "boundary",
                "prefill:2048",
                "save_prefix:2048",
                "prefill:100",
                "save_prefix:2148",
                "reset",
            ])
        );
        assert_eq!(
            stored(&store),
            vec![(ROOT, true)],
            "the root saved before the failure stays; the tip is missing"
        );
        let (second, second_events) = job_with_points(&prompt, TIP, ROOT);
        assert_eq!(
            run_one(&handle, second, second_events),
            stop_events_with_prefix(&log, 2_198, ROOT as u32, fake_report(2, PrefixHit::Root, 0))
        );
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn a_cancel_right_after_the_root_save_runs_no_more_prefill_and_saves_no_tip() {
        let log = new_log();
        let state = generation();
        let mut spy = Spy::new(vec![vec![0]], &log);
        spy.inner.with_prefix_limits(2, 6);
        let store = spy.inner.prefix_store();
        spy.cancel_after_save = Some(Arc::clone(&state.cancel));
        let engine = start_spy(4_096, spy);
        let prompt = root_tip_prompt();
        let (job, handle) = Job::new(prompt, state).expect("job");
        let job = job.with_prefix_boundary(TIP).expect("tip boundary");
        let job = job.with_root_boundary(ROOT).expect("root boundary");
        assert_eq!(
            run_one(&engine.handle(), job, handle),
            vec![
                batch(&[], "", Some(DecodeFinish::Cancelled)),
                finished_with_prefix(&log, 2_198, 0, 0, fake_report(1, PrefixHit::Miss, 0))
            ]
        );
        assert_eq!(
            calls(&log),
            strings(&[
                "reset",
                "restore_prefix",
                "boundary",
                "prefill:2048",
                "save_prefix:2048",
                "decode",
            ]),
            "no piece of the tip head ran after the root save"
        );
        assert_eq!(stored(&store), vec![(ROOT, true)], "the root stays alone");
        engine.shutdown();
    }

    /// The periodic-point fixtures: a 40,000-token prompt of id 2 whose tip boundary lies one
    /// token inside the prompt, and the prompts that share parts of it.
    const LONG_TIP: usize = 39_999;

    fn long_prompt() -> Vec<u32> {
        vec![2; 40_000]
    }

    fn saves_in(entries: &[String]) -> Vec<String> {
        entries
            .iter()
            .filter(|entry| entry.starts_with("save_prefix:"))
            .cloned()
            .collect()
    }

    #[test]
    fn periodic_entries_restore_by_longest_prefix_across_jobs() {
        let log = new_log();
        let (engine, store) = start_limited_spy(256, vec![vec![EOS]; 3], &log);
        let handle = engine.handle();
        // The first job holds no root, so its save points are 16,384 and 32,768 (multiples of
        // the 16,384 stride, each a multiple of the 256 piece) and the tip at 39,999.
        let first_prompt = long_prompt();
        let (first, first_events) = job_with_boundary(&first_prompt, LONG_TIP);
        assert_eq!(
            run_one(&handle, first, first_events),
            stop_events_with_prefix(&log, 40_000, 0, fake_report(3, PrefixHit::Miss, 0))
        );
        assert_eq!(
            saves_in(&calls(&log)),
            strings(&[
                "save_prefix:16384",
                "save_prefix:32768",
                "save_prefix:39999"
            ]),
            "the periodic points run in position order before the tip"
        );
        assert_eq!(
            stored(&store),
            vec![(16_384, false), (32_768, false), (LONG_TIP, false)],
            "the store holds both periodic entries beside the tip"
        );
        // The second job shares the first 33,000 tokens: the longest stored key below its
        // prompt is the 32,768 periodic entry (the 39,999 tip is too long, 16,384 is shorter).
        // The restore covers 32,768, so the target 32,768 equals the cursor and no periodic
        // point remains: the job saves exactly its tip 33,999. The tip extends the restored
        // entry, but periodic entries are never dropped by the lineage rule (rule 4), so the
        // 32,768 entry stays.
        let mut second_prompt = first_prompt[..33_000].to_vec();
        second_prompt.extend(std::iter::repeat_n(7, 1_000));
        let before_second = calls(&log).len();
        let (second, second_events) = job_with_boundary(&second_prompt, 33_999);
        assert_eq!(
            run_one(&handle, second, second_events),
            stop_events_with_prefix(
                &log,
                34_000,
                32_768,
                fake_report(4, PrefixHit::Periodic, 232)
            ),
            "the second job reuses the 32,768 periodic entry"
        );
        assert_eq!(
            saves_in(&calls(&log)[before_second..]),
            strings(&["save_prefix:33999"]),
            "no save at the restored position 32,768: that would be an empty segment"
        );
        // The restore made the 32,768 entry the most recently used one; the new tip follows it.
        assert_eq!(
            stored(&store),
            vec![
                (16_384, false),
                (LONG_TIP, false),
                (32_768, false),
                (33_999, false)
            ],
            "the periodic entry stays beside the tip of the second job"
        );
        // The third job diverges at 20,000 tokens: only the 16,384 entry still matches (the
        // 32,768 and 33,999 keys are longer than the 25,000-token prompt). The target 16,384
        // equals the cursor, so the job saves its tip 24,999 only, and the periodic entry stays.
        let mut third_prompt = first_prompt[..20_000].to_vec();
        third_prompt.extend(std::iter::repeat_n(9, 5_000));
        let before_third = calls(&log).len();
        let (third, third_events) = job_with_boundary(&third_prompt, 24_999);
        assert_eq!(
            run_one(&handle, third, third_events),
            stop_events_with_prefix(
                &log,
                25_000,
                16_384,
                fake_report(5, PrefixHit::Periodic, 3_616)
            ),
            "the diverging job reuses the 16,384 periodic entry"
        );
        assert_eq!(
            saves_in(&calls(&log)[before_third..]),
            strings(&["save_prefix:24999"]),
            "only the tip"
        );
        assert_eq!(
            stored(&store),
            vec![
                (LONG_TIP, false),
                (32_768, false),
                (33_999, false),
                (16_384, false),
                (24_999, false)
            ],
            "the restore moved the 16,384 entry behind the others; the tip follows; nothing is dropped"
        );
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn a_cancel_between_two_periodic_saves_keeps_the_earlier_save() {
        let log = new_log();
        let state = generation();
        let flag = Arc::clone(&state.cancel);
        let mut spy = Spy::new(vec![vec![0]], &log);
        spy.inner.with_prefix_limits(2, 6);
        let store = spy.inner.prefix_store();
        // Prefill calls 0 to 63 cover [0, 16,384). Call 64 is the first chunk after the 16,384
        // save, so the cancel lands between the two periodic saves.
        spy.cancel_after_prefill_call = Some((64, flag));
        let engine = start_spy(256, spy);
        let (job, handle) = Job::new(long_prompt(), state).expect("job");
        let job = job.with_prefix_boundary(LONG_TIP).expect("tip boundary");
        assert_eq!(
            run_one(&engine.handle(), job, handle),
            vec![
                batch(&[], "", Some(DecodeFinish::Cancelled)),
                finished_with_prefix(&log, 40_000, 0, 0, fake_report(1, PrefixHit::Miss, 0)),
            ]
        );
        assert_eq!(
            stored(&store),
            vec![(16_384, false)],
            "the first periodic entry stays and the second was never saved"
        );
        assert_eq!(
            saves_in(&calls(&log)),
            strings(&["save_prefix:16384"]),
            "no later save ran"
        );
        engine.shutdown();
    }

    #[test]
    fn an_engine_with_one_lru_entry_saves_only_the_tip() {
        // Limits (2, 1) hold one LRU entry: a periodic save would be replaced by the tip at
        // once, so the worker emits no periodic point at all.
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]; 5], &log);
        spy.inner.with_prefix_limits(2, 1);
        let store = spy.inner.prefix_store();
        let engine = start_spy(256, spy);
        let prompt = long_prompt();
        let (job, events) = job_with_boundary(&prompt, LONG_TIP);
        assert_eq!(
            run_one(&engine.handle(), job, events),
            stop_events_with_prefix(&log, 40_000, 0, fake_report(1, PrefixHit::Miss, 0))
        );
        assert_eq!(
            saves_in(&calls(&log)),
            strings(&["save_prefix:39999"]),
            "no periodic snapshot was spent"
        );
        assert_eq!(
            stored(&store),
            vec![(LONG_TIP, false)],
            "the store holds the tip only"
        );
        engine.shutdown();
    }

    /// R10 at the fake level: the block cut sits on a chunk end, so it splits no chunk -
    /// the prefill chunk sequence with the cut equals the cold run's exactly, and the cut
    /// saves between the same chunks the periodic points already ride. (The resident-level
    /// exactness oracle against a cold session is the stage-2 follow-up.)
    #[test]
    fn the_cold_chunk_sequence_equals_the_sequence_with_the_block_cut() {
        // Spacing 4,096, tip 40,192: the cut target floor(40,191 / 4,096) * 4,096 = 36,864
        // rides above the deepest periodic (32,768) and is itself a chunk end of 256.
        let log = new_log();
        let (engine, _store) = start_block_spy(256, vec![vec![EOS], vec![EOS]], &log, 4_096);
        let handle = engine.handle();
        let prompt = vec![2; 40_292];
        let (cold, cold_events) = job(&prompt);
        assert_eq!(
            run_one(&handle, cold, cold_events),
            stop_events(&log, 40_292, 0)
        );
        let after_cold = calls(&log).len();
        let (cut, cut_events) = job_with_boundary(&prompt, 40_192);
        assert_eq!(
            run_one(&handle, cut, cut_events),
            stop_events_with_prefix(&log, 40_292, 0, fake_report(4, PrefixHit::Miss, 0))
        );
        let all = calls(&log);
        let pieces = |entries: &[String]| -> Vec<String> {
            entries
                .iter()
                .filter(|entry| entry.starts_with("prefill:"))
                .cloned()
                .collect()
        };
        assert_eq!(
            pieces(&all[..after_cold]),
            pieces(&all[after_cold..]),
            "the block cut left the chunk sequence of the prefill unchanged"
        );
        assert_eq!(
            saves_in(&all[after_cold..]),
            strings(&[
                "save_prefix:16384",
                "save_prefix:32768",
                "save_prefix:36864",
                "save_prefix:40192"
            ]),
            "the periodic points, the cut, and the tip ran between the same chunks"
        );
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn the_cold_chunk_sequence_equals_the_sequence_with_periodic_points_and_no_root() {
        // The periodic points sit on chunk ends, so they split no chunk. The tip 40,192 is a
        // multiple of the 256 piece and the prompt ends 100 tokens later, so the tail chunk of
        // the run with points has the size of the tail chunk of the cold run too.
        let log = new_log();
        let (engine, _store) = start_limited_spy(256, vec![vec![EOS], vec![EOS]], &log);
        let handle = engine.handle();
        let prompt = vec![2; 40_292];
        let (cold, cold_events) = job(&prompt);
        assert_eq!(
            run_one(&handle, cold, cold_events),
            stop_events(&log, 40_292, 0)
        );
        let after_cold = calls(&log).len();
        let (periodic, periodic_events) = job_with_boundary(&prompt, 40_192);
        assert_eq!(
            run_one(&handle, periodic, periodic_events),
            stop_events_with_prefix(&log, 40_292, 0, fake_report(3, PrefixHit::Miss, 0))
        );
        let all = calls(&log);
        let pieces = |entries: &[String]| -> Vec<String> {
            entries
                .iter()
                .filter(|entry| entry.starts_with("prefill:"))
                .cloned()
                .collect()
        };
        let cold_pieces = pieces(&all[..after_cold]);
        assert_eq!(cold_pieces.len(), 158, "157 chunks of 256 and one of 100");
        assert_eq!(
            cold_pieces,
            pieces(&all[after_cold..]),
            "the periodic saves left the chunk sequence of the prefill unchanged"
        );
        assert_eq!(
            saves_in(&all[after_cold..]),
            strings(&[
                "save_prefix:16384",
                "save_prefix:32768",
                "save_prefix:40192"
            ]),
            "the periodic points and the tip ran between the same chunks"
        );
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn the_executor_piece_decides_where_a_periodic_point_lands() {
        // The executor reports a piece of 100, which replaces the fallback 256 and does not
        // divide 16,384. The target 16,384 moves to ceil(16,384 / 100) * 100 = 164 * 100
        // = 16,400. The tip 19,900 is a multiple of 100, so the prefill runs 164 + 35 chunks of
        // 100 up to the tip and one chunk of 100 after it: 200 chunks, as in the cold run.
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS], vec![EOS]], &log);
        spy.inner.with_prefix_limits(2, 6);
        spy.piece = Some(100);
        let engine = start_spy(256, spy);
        let handle = engine.handle();
        let prompt = vec![2; 20_000];
        let (cold, cold_events) = job(&prompt);
        assert_eq!(
            run_one(&handle, cold, cold_events),
            stop_events(&log, 20_000, 0)
        );
        let after_cold = calls(&log).len();
        let (periodic, periodic_events) = job_with_boundary(&prompt, 19_900);
        assert_eq!(
            run_one(&handle, periodic, periodic_events),
            stop_events_with_prefix(&log, 20_000, 0, fake_report(2, PrefixHit::Miss, 0))
        );
        let all = calls(&log);
        let pieces = |entries: &[String]| -> Vec<String> {
            entries
                .iter()
                .filter(|entry| entry.starts_with("prefill:"))
                .cloned()
                .collect()
        };
        let cold_pieces = pieces(&all[..after_cold]);
        assert_eq!(
            cold_pieces,
            vec!["prefill:100".to_string(); 200],
            "the cold run uses the executor piece"
        );
        assert_eq!(
            pieces(&all[after_cold..]),
            cold_pieces,
            "the periodic saves left the chunk sequence unchanged"
        );
        assert_eq!(
            saves_in(&all[after_cold..]),
            strings(&["save_prefix:16400", "save_prefix:19900"]),
            "the periodic point is the end of the 164th chunk of 100"
        );
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn the_engine_handle_can_be_shared_between_threads() {
        fn assert_shareable<T: Clone + Send + Sync + 'static>() {}
        assert_shareable::<EngineHandle>();
    }
}
