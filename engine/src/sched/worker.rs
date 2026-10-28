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
    Job, JobEvent, JobUsage, QUEUE_CAPACITY, send_cancelled, send_data, send_terminal,
};
use crate::EngError;
use crate::family::executor::{ExecutorOps, GenerationState};
use crate::family::session::{SaveKind, StoreLimits};
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
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
struct Shared {
    state: AtomicU8,
    /// The first recorded cause of `Error` or `NativeFailed`, for `/ready` and the logs.
    failure: Mutex<Option<String>>,
    /// Set by `shutdown`. The worker cancels every job that it has not finished.
    stop: AtomicBool,
    /// The cancel flag of the running job, so `shutdown` can cancel it.
    running: Mutex<Option<Arc<AtomicBool>>>,
    /// Opt-in window jobs only. The legacy single-slot path never registers here.
    active: Mutex<Vec<ActiveJob>>,
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
            stop: AtomicBool::new(false),
            running: Mutex::new(None),
            active: Mutex::new(Vec::new()),
        }
    }

    fn state(&self) -> EngineState {
        EngineState::from_u8(self.state.load(Ordering::SeqCst))
    }

    fn set_state(&self, state: EngineState) {
        self.state.store(state as u8, Ordering::SeqCst);
    }

    /// Keeps the first cause: later failures are consequences of it.
    fn record_failure(&self, cause: String) {
        let mut slot = lock(&self.failure);
        if slot.is_none() {
            *slot = Some(cause);
        }
    }

    fn failure(&self) -> Option<String> {
        lock(&self.failure).clone()
    }
}

/// Why `EngineHandle::try_submit` refused a job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SubmitError {
    /// The queue holds `QUEUE_CAPACITY` jobs. The server maps this to `Overloaded`.
    QueueFull,
    /// A native failure or an executor panic stopped the engine. The server maps this to `Backend`.
    NativeFailed,
    /// The engine failed to load. The server maps this to `Backend`, like `NativeFailed`.
    LoadFailed,
    /// The engine has stopped or is stopping. The server maps this to `Overloaded`.
    Unavailable,
}

/// A message to the engine thread.
enum Message {
    Job(Job),
    /// Ends the loop after the jobs ahead of it. `shutdown` sends it so that the thread stops
    /// while other handles are still alive.
    Stop,
}

/// The portable, shareable front of one engine thread.
#[derive(Clone)]
pub(crate) struct EngineHandle {
    queue: SyncSender<Message>,
    shared: Arc<Shared>,
}

impl EngineHandle {
    /// The current state of the engine.
    pub(crate) fn state(&self) -> EngineState {
        self.shared.state()
    }

    /// The cause of the `Error` or `NativeFailed` state, for `/ready` and the logs.
    pub(crate) fn failure(&self) -> Option<String> {
        self.shared.failure()
    }

    /// Queues a job without waiting. A refused job is dropped, and its channel closes.
    ///
    /// A job that a concurrent failure or stop catches in the queue can also end with a closed
    /// channel and no event. The owner then treats the closed channel as a `Backend` failure.
    pub(crate) fn try_submit(&self, job: Job) -> Result<(), SubmitError> {
        if self.shared.stop.load(Ordering::SeqCst) {
            return Err(SubmitError::Unavailable);
        }
        match self.state() {
            EngineState::NativeFailed => return Err(SubmitError::NativeFailed),
            EngineState::Error => return Err(SubmitError::LoadFailed),
            EngineState::Stopped => return Err(SubmitError::Unavailable),
            EngineState::Loading | EngineState::Ready | EngineState::Busy => {}
        }
        self.queue
            .try_send(Message::Job(job))
            .map_err(|error| match error {
                TrySendError::Full(_) => SubmitError::QueueFull,
                TrySendError::Disconnected(_) => SubmitError::Unavailable,
            })
    }
}

/// What `Engine::shutdown` found when the thread ended.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ShutdownReport {
    /// The final state of the engine.
    pub(crate) state: EngineState,
    /// The cause of an `Error` or `NativeFailed` state, including every panic.
    pub(crate) failure: Option<String>,
}

/// One engine thread and its handle.
pub(crate) struct Engine {
    handle: EngineHandle,
    thread: JoinHandle<()>,
}

impl Engine {
    /// A new handle to the engine.
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
            shared.record_failure(cause.clone());
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

/// Starts the engine thread without a ready note; test callers only, so a
/// new family cannot silently skip the note - production callers use
/// [`spawn_engine_noted`]. `make` runs on that thread and builds the
/// executor and its session, so thread-confined native values never cross a
/// thread boundary. `prefill_chunk` is the largest prompt slice passed to
/// one `prefill` call when the executor does not plan its own piece. An
/// executor's initial `prefill_piece()` sets the maximum after `make`; the
/// hint is refreshed between pieces without exceeding that maximum.
#[cfg(test)]
pub(crate) fn spawn_engine<E, F>(prefill_chunk: usize, make: F) -> Result<Engine, EngError>
where
    E: ExecutorOps + 'static,
    F: FnOnce() -> Result<(E, E::Session), EngError> + Send + 'static,
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
    F: FnOnce() -> Result<(E, E::Session), EngError> + Send + 'static,
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

/// Cancels a job that never started: the owner gets a `Cancelled` batch and a usage event.
fn cancel_unstarted(job: Job) {
    job.generation.cancel.store(true, Ordering::SeqCst);
    send_cancelled(&job.events, prompt_tokens(&job.prompt));
}

fn drain(jobs: &Receiver<Message>, finish: fn(Job)) {
    while let Ok(message) = jobs.try_recv() {
        if let Message::Job(job) = message {
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
    F: FnOnce() -> Result<(E, E::Session), EngError>,
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
        shared.record_failure(format!("engine thread panicked: {message}"));
        shared.set_state(next);
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
    F: FnOnce() -> Result<(E, E::Session), EngError>,
{
    let (mut executor, mut session) = match make() {
        Ok(pair) => pair,
        Err(error) => {
            let cause = format!("engine load failed: {error:?}");
            if let Some(note) = ready_note {
                note(EngineNote::Failed(cause.clone()));
            }
            shared.record_failure(cause);
            shared.set_state(EngineState::Error);
            drain(jobs, reject);
            return;
        }
    };
    let prefill_chunk = executor.prefill_piece().unwrap_or(prefill_chunk);
    if prefill_chunk == 0 {
        let cause = "engine load failed: the executor planned a prefill piece of 0".to_string();
        if let Some(note) = ready_note {
            note(EngineNote::Failed(cause.clone()));
        }
        shared.record_failure(cause);
        shared.set_state(EngineState::Error);
        drain(jobs, reject);
        return;
    }
    shared.set_state(EngineState::Ready);
    // The engine reports its own prefix capability to the one-shot note: the
    // built executor's store and spacing are the single source, so every
    // family reports correctly without a factory flag.
    if let Some(note) = ready_note {
        note(EngineNote::Ready(PrefixCapability {
            cuts_block: cuts_intermediate(executor.prefix_limits()),
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
            max_seqs,
        );
        return;
    }
    while let Ok(Message::Job(job)) = jobs.recv() {
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
    if shared.state() != EngineState::NativeFailed {
        shared.set_state(EngineState::Stopped);
    }
    drain(jobs, cancel_unstarted);
}

/// The prompt of one job, the length of its reusable prefix, and the root inside it.
struct PromptPlan<'a> {
    tokens: &'a [u32],
    /// The number of leading tokens that end at the last message boundary, or `None`.
    boundary: Option<usize>,
    /// The number of leading tokens that end at the first system message, or `None`. It is
    /// set only beside a boundary, and it lies inside it.
    root: Option<usize>,
}

/// The shortest root that a save point accepts (decision 0036, addendum 2, item 2a).
const MIN_ROOT_TOKENS: usize = 2_048;

/// The stride of the periodic save points (decision 0036, addendum 2, item 2b): one entry at
/// every multiple of this many positions counted from the start of the prompt.
const PERIODIC_TOKENS: usize = 16_384;

/// The ordered save points of a job (decision 0036, addendum 2, item 2): the root at the end
/// of the first system message, one periodic entry at every multiple of `PERIODIC_TOKENS`
/// positions counted from the start of the prompt, and the tip at the last message boundary.
/// The root runs only when the store keeps roots (`limits.roots > 0`), it reaches
/// `MIN_ROOT_TOKENS`, lies strictly inside the tip, and the restore covered none of it. Without
/// roots a root point would only split the prefill and change its chunk grouping for a save
/// the store refuses, so a job with a root boundary then runs exactly like a job without one.
/// The periodic points run only when the store keeps at least two LRU entries
/// (`limits.entries >= 2`): with one entry a periodic save would be replaced by the tip at
/// once and spend a snapshot for nothing. Of the periodic points only the deepest
/// `limits.entries - 1` are emitted, because the tip takes the last LRU place. The tip runs only
/// when the restore covered none of it, today's rule.
///
/// The prefill runs in chunks of `piece` tokens that start at the current cursor: the cursor
/// starts at `restored` and moves to every point that this function emits, so a root at an
/// arbitrary position moves it. A periodic target P counts only when it lies above the cursor.
/// It moves to the end of the chunk that contains it, `cursor + ceil((P - cursor) / piece) *
/// piece` (a P that is already a chunk end stays P), and the moved position must stay strictly
/// below the tip, otherwise the target is dropped. The moved position is always above the
/// cursor, so no position is emitted twice. The points stay on chunk ends, so the periodic
/// saves never split a chunk and the numeric path of an entry equals the path of a cold run,
/// except after a root, which splits the sequence by design.
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
        // The intermediate candidates: the periodic multiples below the tip and, when the
        // block prefix-match is on (KV-v1 slice 2), the last block boundary strictly below
        // the tip - the ruled rolling cut of the lineage. The walk visits the targets in
        // order, a periodic first at a tie, so a boundary the lattice already cuts stays one
        // save of the periodic kind; the block candidate rides the same move-to-chunk-end
        // and below-the-tip rules, so it never shifts the periodic positions. The tie rule
        // also holds after the move to the chunk end: a block target whose chunk end an
        // unprocessed periodic target reaches is skipped, because every target in
        // (block_target, moved] projects onto `moved`, and the shared position must keep the
        // periodic kind, which a later rolling K=1 replacement cannot remove.
        let mut targets: Vec<(usize, bool)> = (PERIODIC_TOKENS..tip)
            .step_by(PERIODIC_TOKENS)
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
        // skipping a shallow point does not change the position of a deeper one.
        let skip = intermediate.len().saturating_sub(limits.entries - 1);
        points.extend(intermediate.into_iter().skip(skip));
    }
    if let Some(tip) = plan.boundary.filter(|&tip| tip > restored) {
        points.push((tip, SaveKind::Tip));
    }
    points
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
    let reused = match start(
        executor,
        session,
        plan,
        prefill_chunk,
        &generation.cancel,
        false,
    ) {
        Ok(reused) => reused,
        Err(EngError::MemoryPlan(message)) => {
            // Fix EV-1: a dead or idle conversation's cached prefixes must not pin the pool
            // (the Mac daily RED: six entries of ~395 MB blocked a fresh 17,479-token session
            // that needed 11,141,120 bytes with 7,696,432 available). Evict ONE entry at
            // a time - a dead tip first, oldest first, the denied job's own anchor last
            // (round 4d SF-1, radix tiers) - and retry after each, so a denial with a
            // one-entry deficit costs one anchor, not every live lineage (the replay's
            // t6 wave: the base's clear-all lost all sixteen entries to a ~3-6 entry
            // deficit; the second miss there - request 0x18 - was a restore-time drop,
            // split to the SF1 slice, not this eviction's doing). Each retry admits the
            // prompt with no snapshots at all, so caching cannot re-block a
            // prompt the pool can grow. When the last entry is gone and the admission still
            // denies, the refusal only becomes the over-length form (fix EV-2) when the
            // pool's ceiling honestly explains it - a client then compacts to a size that
            // fits instead of losing the session to the RED's requests-0018/0019 growth
            // errors - and the session still resets either way.
            //
            // Size the eviction before the restart (review round 4, SF2): evict while the
            // pool's honest ceiling (fix EV-2) still cannot admit this prompt, so a
            // k-entry deficit costs ONE restart, not k partial prefills. The ceiling
            // covers the free budget plus this session's own permits, and every eviction
            // keeps the denied job's own anchor for its last resort (review round 4, MF2).
            // A `None` ceiling sizes nothing: stop after the first eviction and let the
            // residual one-at-a-time loop below spend more as denials recur. The loop's
            // FIRST `None` is terminal (Sol round 3, MF1): nothing can be freed, so no
            // restart can ever admit this prompt - classify now instead of one dead
            // restart (the batch worker's admission loop does the same).
            let mut terminal = false;
            loop {
                let protect = match executor.restored_anchor_of(&*session) {
                    Some(id) => Some(id),
                    // No restore ran yet (an admission refusal inside `start`): the
                    // entry this restart would restore goes last (review round 4, MF2).
                    None => executor.restore_candidate(plan.tokens),
                };
                if executor.evict_one_prefix_entry(protect).is_none() {
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
                // Classify only when the ceiling PROVES the prompt cannot fit (the
                // honest over-length form, fix EV-2); a save-denial recovery with room
                // to grow still takes the one no-save retry below (Sol round 3, MF1,
                // preserves the fitting recovery of round 2).
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
                &generation.cancel,
                true,
            );
            while let Err(EngError::MemoryPlan(_)) = retried {
                let protect = match executor.restored_anchor_of(&*session) {
                    Some(id) => Some(id),
                    None => executor.restore_candidate(plan.tokens),
                };
                if executor.evict_one_prefix_entry(protect).is_none() {
                    break;
                }
                retried = start(
                    executor,
                    session,
                    plan,
                    prefill_chunk,
                    &generation.cancel,
                    true,
                );
            }
            match retried {
                Ok(reused) => reused,
                Err(EngError::MemoryPlan(message)) => {
                    // The retry's refusal names the post-eviction pool, so it - not the first
                    // attempt's message with its pre-eviction availability - is what a client
                    // sees when the ceiling does not explain the failure (review round 2).
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
    let reused_tokens = u32::try_from(reused).unwrap_or(u32::MAX);
    // The report of the job's prefix store, taken after the prefill and before the first decode
    // (packet SERVE-2-c item 3): the entries and bytes then include this job's save points.
    // `None` is the report of an executor that keeps none (fix M3).
    let prefix = executor.prefix_report();
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
                }),
            );
            return EngineState::Ready;
        }
    }
}

/// The final error for an admission the pool refused (fix EV-2): the over-length form only
/// when the pool's ceiling honestly explains the refusal - a ceiling that would admit this
/// prompt is not why it failed, so the original refusal stands. A ceiling below the prompt
/// names the largest prompt a fresh attempt could charge, which never exceeds the configured
/// limit because `check_prompt` already refused prompts at or beyond it.
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
/// and prefills the rest of the prompt in chunks. It prefills up to every save point of the
/// job in turn (decision 0036, addendum 2: the root, the periodic entries, then the tip) and
/// saves the prefix state at each point. Each segment is chunked from its own start, which is
/// the cursor that `save_points` moved to the previous point, so the chunk sequence of the
/// prefill is the one that `save_points` assumed. Returns the number of prompt tokens that the
/// restore covered.
///
/// `retrying` marks the eviction retry (fix EV-1), which runs with no save points at all: a
/// snapshot that charges successfully at its own point can still starve the prefill tail that
/// follows it (review round 2, MF1), so the retry admits the prompt with no snapshots and the
/// next job rebuilds its cache from scratch.
fn start<E: ExecutorOps>(
    executor: &mut E,
    session: &mut E::Session,
    plan: &PromptPlan<'_>,
    prefill_chunk: usize,
    cancel: &AtomicBool,
    retrying: bool,
) -> Result<usize, EngError> {
    executor.reset(session)?;
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
    let mut chunks_run = 0;
    let mut next = restored;
    let points = if retrying {
        Vec::new()
    } else {
        save_points(
            plan,
            restored,
            executor.prefix_limits(),
            prefill_chunk,
            executor.prefix_block_spacing(),
        )
    };
    for (point, kind) in points {
        let head = span(plan.tokens, next, point)?;
        if !prefill_chunks(
            executor,
            session,
            head,
            prefill_chunk,
            cancel,
            &mut chunks_run,
        )? {
            return Ok(restored);
        }
        if cancel.load(Ordering::SeqCst) {
            return Ok(restored);
        }
        // The head fed through `point`; the save that follows it cannot change what was fed.
        next = point;
        executor.save_prefix(session, span(plan.tokens, 0, point)?, kind)?;
    }
    let tail = span(plan.tokens, next, plan.tokens.len())?;
    prefill_chunks(
        executor,
        session,
        tail,
        prefill_chunk,
        cancel,
        &mut chunks_run,
    )?;
    Ok(restored)
}

/// The tokens from `from` up to `to`.
fn span(tokens: &[u32], from: usize, to: usize) -> Result<&[u32], EngError> {
    tokens
        .get(from..to)
        .ok_or_else(|| EngError::BadRequest("prompt span is outside the prompt".to_string()))
}

/// Prefills `tokens` in chunks. The cancel flag stops it between two chunks, never before the
/// first chunk of the job. Returns false when it stopped early.
fn prefill_chunks<E: ExecutorOps>(
    executor: &mut E,
    session: &mut E::Session,
    tokens: &[u32],
    prefill_chunk: usize,
    cancel: &AtomicBool,
    chunks_run: &mut usize,
) -> Result<bool, EngError> {
    let mut remaining = tokens;
    while !remaining.is_empty() {
        if *chunks_run > 0 && cancel.load(Ordering::SeqCst) {
            return Ok(false);
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
        executor.prefill(session, piece)?;
        *chunks_run += 1;
        remaining = rest;
    }
    Ok(true)
}

fn reset_state<E: ExecutorOps>(
    executor: &mut E,
    session: &mut E::Session,
    shared: &Shared,
) -> EngineState {
    match executor.reset(session) {
        Ok(()) => EngineState::Ready,
        Err(error) => {
            shared.record_failure(format!("session reset failed: {error:?}"));
            EngineState::NativeFailed
        }
    }
}

/// Reports a job error. A `Backend` error is process-scoped: the worker makes no further
/// executor call. Any other error fails only this job and leaves a reset session.
fn fail<E: ExecutorOps>(
    executor: &mut E,
    session: &mut E::Session,
    events: &mpsc::Sender<JobEvent>,
    error: EngError,
    shared: &Shared,
) -> EngineState {
    let native = matches!(error, EngError::Backend { .. });
    if native {
        shared.record_failure(format!("native failure: {error:?}"));
    }
    send_terminal(events, JobEvent::Error(error));
    if native {
        EngineState::NativeFailed
    } else {
        reset_state(executor, session, shared)
    }
}

#[cfg(test)]
mod tests {
    use super::{EngineHandle, EngineState, ShutdownReport, SubmitError, spawn_engine};
    use crate::EngError;
    use crate::exec_fake::FakeExecutor;
    use crate::family::executor::{ExecutorOps, GenerationState};
    use crate::family::session::{PrefixStore, SaveKind, StoreLimits, check_prompt};
    use crate::sched::job::{Job, JobEvent, JobHandle, JobUsage, PrefixHit, PrefixReport};
    use crate::tokenizer::{DecodeBatch, DecodeFinish, IncrementalDecoder, Tokenizer};
    use std::sync::atomic::{AtomicBool, Ordering};
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
        fail_first_decode: Option<EngError>,
        fail_after_emit: Option<EngError>,
        cancel_after_prefill: Option<Arc<AtomicBool>>,
        /// Sets the cancel flag after the `at`-th `prefill` call (0 is the first).
        cancel_after_prefill_call: Option<(usize, Arc<AtomicBool>)>,
        /// The number of `prefill` calls so far.
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
        /// The number of `save_prefix` calls so far.
        save_calls: usize,
        /// Sets the cancel flag after the first `save_prefix` call that the fake accepts.
        cancel_after_save: Option<Arc<AtomicBool>>,
        /// Fails the next `restore_prefix` call with the error, before the fake sees it.
        fail_restore: Option<EngError>,
        /// The piece that `prefill_piece` reports.
        piece: Option<usize>,
        piece_after_prefill: Option<usize>,
        /// The context limit that `check_prompt` applies, or `None` to accept every prompt.
        context_limit: Option<usize>,
    }

    impl Spy {
        fn new(script: Vec<Vec<u32>>, log: &Log) -> Self {
            Spy {
                inner: FakeExecutor::new(script, &[EOS], 100_000),
                log: Arc::clone(log),
                hold: None,
                hold_after_emit: None,
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
                context_limit: None,
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

        fn prefix_limits(&self) -> StoreLimits {
            self.inner.prefix_limits()
        }

        fn prefix_block_spacing(&self) -> Option<usize> {
            self.inner.prefix_block_spacing()
        }

        fn prefix_report(&self) -> Option<PrefixReport> {
            self.inner.prefix_report()
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

        fn after_emit(&mut self) -> Result<(), EngError> {
            if let Some((entered, release)) = self.hold_after_emit.take() {
                let _ = entered.send(());
                let _ = release.recv();
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

    fn start_spy(chunk: usize, spy: Spy) -> super::Engine {
        spawn_engine(chunk, move || {
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

    fn unavailable_event() -> JobEvent {
        JobEvent::Error(EngError::Backend {
            operation: "engine",
            message: "engine is not available".to_string(),
        })
    }

    /// Reads one job's events until the engine closes its channel.
    fn collect(mut handle: JobHandle) -> Vec<JobEvent> {
        let mut events = Vec::new();
        while let Some(event) = handle.events.blocking_recv() {
            events.push(event);
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

    fn finished(prompt: u32, completion: u32) -> JobEvent {
        finished_reusing(prompt, completion, 0)
    }

    fn finished_reusing(prompt: u32, completion: u32, reused: u32) -> JobEvent {
        finished_with_prefix(prompt, completion, reused, Some(PrefixReport::default()))
    }

    /// The `Finished` event of a job whose executor reported `prefix`: the fake always reports
    /// its store, so the value is `Some` even without a restore (fix M3).
    fn finished_with_prefix(
        prompt: u32,
        completion: u32,
        reused: u32,
        prefix: Option<PrefixReport>,
    ) -> JobEvent {
        JobEvent::Finished(JobUsage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            reused_tokens: reused,
            prefix,
        })
    }

    /// The report of the fake's store: `entries` entries at a flat 1,000 charged bytes each
    /// (fix L3), the hit of this job's restore, and its uncached count.
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

    /// Starts an engine on a spy whose fake holds the addendum 2 limits and the block
    /// prefix-match with the given spacing (KV-v1 slice 2), and returns the store view.
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

    /// Starts an engine on a spy whose fake holds the addendum 2 limits (two roots, six
    /// entries), and returns the store view for assertions.
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

    /// Submits the job, waits for its end, and returns its events.
    fn run_one(handle: &EngineHandle, job: Job, events: JobHandle) -> Vec<JobEvent> {
        handle.try_submit(job).expect("submit");
        collect(events)
    }

    /// The events of a job that stops at once, uses no extra tokens, and reports a store that
    /// holds nothing and restored nothing.
    fn stop_events(prompt: u32, reused: u32) -> Vec<JobEvent> {
        stop_events_with_prefix(prompt, reused, Some(PrefixReport::default()))
    }

    /// `stop_events` with the executor's prefix report of the job.
    fn stop_events_with_prefix(
        prompt: u32,
        reused: u32,
        prefix: Option<PrefixReport>,
    ) -> Vec<JobEvent> {
        vec![
            batch(&[EOS], "", Some(DecodeFinish::Stop)),
            finished_with_prefix(prompt, 1, reused, prefix),
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
        let first = events.events.try_recv().ok();
        let _ = release_tx.send(());
        let rest = collect(events);
        engine.shutdown();
        assert!(entered, "deferred prefix completion must run");
        assert_eq!(first, Some(batch(&[0], "a", None)));
        assert_eq!(
            rest,
            vec![batch(&[EOS], "", Some(DecodeFinish::Stop)), finished(3, 2)]
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
                finished(3, 3),
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
                finished(1, 2),
            ]
        );
        assert_eq!(
            collect(second_events),
            vec![
                batch(&[1], "b", None),
                batch(&[EOS], "", Some(DecodeFinish::Stop)),
                finished(1, 2),
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
                Some(&finished(1, 1)),
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
                // A queued job never ran, so it carries no executor report at all (fix M3).
                finished_with_prefix(2, 0, 0, None)
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
                finished(1538, 0),
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
                finished(6, 0)
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
            vec![batch(&[EOS], "", Some(DecodeFinish::Stop)), finished(1, 1)]
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
            vec![batch(&[EOS], "", Some(DecodeFinish::Stop)), finished(1, 1)]
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
            vec![batch(&[EOS], "", Some(DecodeFinish::Stop)), finished(1, 1)]
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
        let engine = spawn_engine::<Spy, _>(8, move || {
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
        let engine = spawn_engine::<Spy, _>(8, move || {
            let _ = release_rx.recv();
            panic!("make exploded");
        })
        .expect("spawn engine");
        let handle = engine.handle();
        let (queued, queued_events) = job(&[2]);
        handle.try_submit(queued).expect("submit while loading");
        release_tx.send(()).expect("release the load");
        assert_eq!(collect(queued_events), vec![unavailable_event()]);
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
        let engine =
            spawn_engine::<Spy, _>(8, || std::panic::panic_any(PanicOnDrop)).expect("spawn engine");
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
            vec![batch(&[EOS], "", Some(DecodeFinish::Stop)), finished(1, 1)]
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
                .try_send(finished(0, 0))
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
            [batch(&[EOS], "", Some(DecodeFinish::Stop)), finished(1, 1)]
        );
        engine.shutdown();
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
        assert_eq!(collect(running_events).last(), Some(&finished(1, 1)));
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
            finished(1, 0),
        ];
        assert_eq!(collect(running_events), cancelled);
        // The queued job never reached the executor, so its usage carries no report (fix M3).
        assert_eq!(
            collect(queued_events),
            vec![
                batch(&[], "", Some(DecodeFinish::Cancelled)),
                finished_with_prefix(1, 0, 0, None)
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
        let engine = spawn_engine(8, move || {
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
            .send(super::Message::Job(late))
            .unwrap_or_else(|_| panic!("the queue refused the job"));
        release_tx.send(()).expect("release the load");
        assert_eq!(
            collect(late_events),
            vec![
                batch(&[], "", Some(DecodeFinish::Cancelled)),
                // The late job never ran: no executor report (fix M3).
                finished_with_prefix(1, 0, 0, None)
            ]
        );
        assert_eq!(handle.state(), EngineState::Stopped);
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
            stop_events_with_prefix(6, 0, fake_report(1, PrefixHit::Miss, 0))
        );
        let (second, second_events) = job_with_boundary(&[2, 2, 2, 2, 7, 7, 7, 7, 7], 6);
        assert_eq!(
            run_one(&handle, second, second_events),
            stop_events_with_prefix(
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
            stop_events_with_prefix(7, 0, fake_report(1, PrefixHit::Miss, 0))
        );
        let saved = calls(&log);
        // Every refused prompt starts with the saved key and carries a boundary, so a check that
        // ran only after the restore would have spent the entry before refusing the prompt.
        // The over-limit refusals carry both numbers in the machine-readable fields of the
        // over-length form (fix EV-2): the message text is the server's rendering.
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
            stop_events_with_prefix(7, 4, fake_report(1, PrefixHit::Tip, 0))
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
            stop_events_with_prefix(5, 0, fake_report(1, PrefixHit::Miss, 0))
        );
        // B shares no key with A. It restores nothing and replaces the saved state.
        let (b, b_events) = job_with_boundary(&[1, 1, 1, 1, 1], 3);
        assert_eq!(
            run_one(&handle, b, b_events),
            // B's save removes A's entry for room: the churn the old counters could not see.
            stop_events_with_prefix(
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
            stop_events_with_prefix(5, 0, fake_report(1, PrefixHit::Miss, 0))
        );
        let (second, second_events) = job(&[2, 2, 2, 2, 2]);
        // The second job has no boundary: it reports the store the first job filled, with the
        // miss of its own restore-free run (fix M4).
        assert_eq!(
            run_one(&handle, second, second_events),
            stop_events_with_prefix(5, 0, fake_report(1, PrefixHit::Miss, 0))
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

    /// The report of a job without a boundary carries the store, not the previous job's hit
    /// (fix M4): the worker resets every job and skips the restore when the job has no
    /// boundary, so the reset must clear the restore facts. Job 1 stores a root and a tip, job
    /// 2 restores the root and shares 52 tokens with the stored tip beyond it, job 3 runs
    /// without a boundary and must report a miss with nothing uncached.
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
            stop_events_with_prefix(2_198, 0, fake_report(2, PrefixHit::Miss, 0))
        );
        // Job 2 shares the system prefix and 52 tokens more with the stored tip, but diverges
        // before it: it restores the root, and the 52 tokens it shares with the tip beyond the
        // root are uncached (fix M2's longest common prefix).
        let mut second_prompt = vec![2; 2_100];
        second_prompt.extend(std::iter::repeat_n(7, 50));
        let (second, second_events) = job_with_points(&second_prompt, 2_100, ROOT);
        assert_eq!(
            run_one(&handle, second, second_events),
            stop_events_with_prefix(2_150, ROOT as u32, fake_report(3, PrefixHit::Root, 52))
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
            stop_events_with_prefix(2_198, 0, fake_report(3, PrefixHit::Miss, 0)),
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
                finished(6, 0)
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
                finished(6, 0)
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
                5,
                0,
                fake_report_save(0, PrefixHit::Miss, 0, refused_once())
            )
        );
        let (second, second_events) = job_with_boundary(&[2, 2, 2, 2, 2, 2], 3);
        assert_eq!(
            run_one(&handle, second, second_events),
            stop_events_with_prefix(
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

    /// Starts an engine on a spy whose fake holds the addendum 2 limits and a session budget
    /// (fix EV-1): every prefilled or decoded token charges `token_bytes`, every stored entry
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

    /// Fix EV-1, RED from the Mac daily log (mac-daily-log-red-evidence.md, requests 001a and
    /// 001b): a fresh session could not admit - 11,141,120 bytes needed, 7,696,432 available -
    /// while dead conversations' prefix entries held the pool. The worker sizes the eviction
    /// against the pool's ceiling first and then retries the admission ONCE (round 4c,
    /// R4-MF2 mirrors this in the batch worker).
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
             contract change: clear-all became evict-until-fits, fix EV-1's blast radius)"
        );
        drop(handle);
        engine.shutdown();
    }

    /// Fix EV-1's blast radius (the replay's t6 cliff, run override-16): a denied admission
    /// in the BASE clears the WHOLE LRU - every live conversation's anchor included - and
    /// retries with no snapshots, so the conversations that looked up after the denial find
    /// nothing and re-prefill in full (the t6 wave: agents 0 and 2, reused_tokens 0,
    /// 410/421 s against 235 s; note the clear-all did NOT cause BOTH misses - request
    /// 0x18's loss was the restore-time adoption drop, split to the SF1 slice). The
    /// eviction must free what the denial needs: one entry at a time, so a denial with a
    /// one-entry deficit costs one anchor, not every lineage's. The admission-time
    /// protection (`restore_candidate`) is weaker than the session one. Since SF1 a
    /// denied restore no longer drops the anchor outright - it evicts one victim and
    /// retries the same candidate - but the weak protect still loses it on a SECOND
    /// denial, or when the candidate is the only LRU entry (the tier-4 last resort
    /// spends it).
    ///
    /// Mutation check: with today's clear-all the fourth conversation still finishes (the
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
        // oldest tip goes first - review round 4, MF3), and the
        // two newer anchors must survive (today the clear-all wiped all three).
        let (job, events) = job_with_boundary(&vec![2; 152], 151);
        handle.try_submit(job).expect("submit");
        let events = collect(events);
        assert!(
            matches!(events.last(), Some(JobEvent::Finished(usage)) if usage.reused_tokens == 0),
            "the oldest anchor was the one the denial spent: {events:?}"
        );
        // The newer anchors survive: conversation 3's second turn extends its own tip and
        // must restore it (today the clear-all wiped every anchor and this restores
        // nothing). Conversation 4 sits on the same store; one survivor discriminates the
        // contract, and the oldest-sacrifice above pins the victim order.
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

    /// Round 4d SF-5 / 4e SF-1 (Opus + Sol): the single worker's restored-anchor
    /// protection fixture. The denied conversation's second turn restores its own
    /// tip, the growth cannot fit, the pre-pass protects the restored anchor, and the
    /// restart re-restores it (reused > 0) while the oldest OTHER dead tip spends and
    /// the newest survives.
    /// HONEST SCOPE (Opus round 4e SF-1 / Sol r5 SF2): this fixture does NOT
    /// discriminate a no-protect mutant - the start RESTORES before it prefills
    /// (exec_fake's restore always succeeds), which moves the denied conversation's
    /// entry to MRU, so the oldest other tip spends with or without protection. It
    /// exercises the wiring only (protect = the session's own restored anchor,
    /// `restored_anchor_of`); the per-slot-vs-global confusion is discriminated at
    /// store level by `an_eviction_protects_the_denied_lineage_not_the_last_restore`
    /// and `a_session_without_a_restore_does_not_protect_another_sessions_root`
    /// (family/session.rs), and the SF1 knob test
    /// `an_adoption_denial_evicts_one_victim_and_retries_the_same_candidate_on_the_
    /// engine` (batch_worker_tests) exercises the STORE's own protect at the engine
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

    /// Fix EV-2, the second half of the Mac daily log RED (requests 0018 and 0019): a request
    /// that cannot fit even in an empty pool must fail in the typed over-length form - the
    /// kind a client answers by compacting the conversation - not as a session-killer.
    ///
    /// Mutation check: without the classification the event is `MemoryPlan` - the session
    /// killer of the RED - and the assertion fails.
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

    /// Review round 2, MF1 (Sol R2-1's exact scenario): a snapshot that charges
    /// successfully on the retry can starve the prefill tail that follows it, so the
    /// eviction retry admits the prompt with NO save points. The tip's entry charge
    /// (3,000,000) beside the prompt (6,145,024) exceeds the cap (8,000,000): the first
    /// attempt saves the tip and is denied mid-tail, and the retry - which without this
    /// fix would save the tip again and deny at the same chunk - finishes the prompt.
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

    /// R6 of the RED list: the eviction retry runs with no save points at all, so the block
    /// cut cannot fire in the retry window either - the cut flows through `save_points`,
    /// which the retry empties, and the next job rebuilds from scratch (the 1f56bf74 rule
    /// covers the new kind structurally). Sol SF2: spacing 1,024 (not 4) so the cut target
    /// 1,024 is a chunk end strictly below the tip 2,000 - with spacing 4 the target 1,996
    /// projected onto the tip itself and the block candidate never emitted, so the test
    /// proved nothing about the retry window. Now the first attempt reaches the Block save
    /// and stores it; the tip save's first charge fails (2,048,000 + 2 x 3,000,000 >
    /// 8,000,000), it evicts the unprotected Block and succeeds on retry (2,048,000 +
    /// 3,000,000 = 5,048,000), and the tip then starves the tail: the pool exhausts
    /// mid-tail, and the retry resets, attempts a restore on the store the eviction already
    /// emptied, re-feeds the whole prompt, and attempts no new saves. (Sol round-2 SF1.)
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

    /// Review round 2 (Sol P1-3): an executor that charges the pool for a snapshot inside
    /// `save_prefix` denies every save with a memory-plan error: the first attempt fails
    /// at its save, and the retry - which never saves at all (MF1) - finishes the prompt.
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
        // The retry fed its whole prompt in one pass with no save points (MF1): the two
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

    /// decision 0036 addendum 2 item 6: roots are best effort inside the cap). The fake denies
    /// the charge of the first snapshot call only, which is the root.
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
        assert_eq!(run_one(&handle, job, events), stop_events(7, 5));
        assert_eq!(
            calls(&log),
            strings(&["reset", "restore_prefix", "boundary", "prefill:2", "decode"])
        );
        drop(handle);
        engine.shutdown();
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
        // The next job starts clean and runs to its end.
        let (second, second_events) = job_with_boundary(&[2, 2, 2, 2, 2], 3);
        assert_eq!(
            run_one(&handle, second, second_events),
            stop_events_with_prefix(5, 0, fake_report(1, PrefixHit::Miss, 0))
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
            stop_events_with_prefix(5, 0, fake_report(1, PrefixHit::Miss, 0))
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
        let engine = super::spawn_engine_noted(2, note, move || {
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
        let error = spawn_engine(0, move || {
            Ok((spy, <FakeExecutor as ExecutorOps>::Session::default()))
        })
        .err()
        .expect("a zero chunk must fail");
        assert!(matches!(error, EngError::BadRequest(_)), "got {error:?}");
    }

    /// The save points of a plan over eight tokens with the given tip and root, for a store
    /// with the addendum 2 limits (two roots, six entries).
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

    /// The limits of the addendum 2 store: two roots, six entries.
    const WIDE: StoreLimits = StoreLimits {
        roots: 2,
        entries: 6,
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

    // --- KV-v1 slice 2: the block prefix-match cut in the save lattice ---

    /// `points_full` with the block prefix-match spacing (KV-v1 slice 2): `Some(spacing)` cuts
    /// one rolling intermediate entry at the last block boundary strictly below the tip.
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
    fn a_block_cut_moves_to_the_chunk_end_like_a_periodic_target() {
        // Root 3,000, so the chunks start at 3,000: the cut target 8,192 moves to
        // 3,000 + ceil(5,192 / 256) * 256 = 3,000 + 21 * 256 = 8,376, strictly below the tip.
        assert_eq!(
            points_block(Some(10_000), Some(3_000), 0, 256, WIDE, Some(4_096)),
            vec![
                (3_000, SaveKind::Root),
                (8_376, SaveKind::Block),
                (10_000, SaveKind::Tip),
            ],
            "the cut rides the chunk lattice of the cursor like a periodic target"
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
    fn a_block_target_below_a_periodic_that_reaches_its_chunk_end_yields_the_periodic() {
        // Sol MF1: spacing 4,095, tip 16,400, piece 256. The block target
        // floor(16,399 / 4,095) * 4,095 = 16,380 is below the periodic 16,384, but both reach
        // the chunk end 16,384 = 64 * 256, so the tie happens after the projection. Emitting
        // the Block there would give the shared position the replaceable kind, so the block
        // is skipped and the periodic saves 16,384: the output equals the no-cut run.
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
            "the periodic wins the chunk end both targets reach"
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
            stop_events_with_prefix(2_198, 0, fake_report(2, PrefixHit::Miss, 0))
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
            stop_events_with_prefix(2_198, 0, fake_report(2, PrefixHit::Miss, 0))
        );
        // The same 2,048-token system prefix, a different first turn, and a different tail: the
        // stored tip key does not match, the root key does.
        let mut second_prompt = vec![2; ROOT];
        second_prompt.extend(std::iter::repeat_n(7, 102));
        let (second, second_events) = job_with_points(&second_prompt, 2_100, ROOT);
        assert_eq!(
            run_one(&handle, second, second_events),
            stop_events_with_prefix(2_150, ROOT as u32, fake_report(3, PrefixHit::Root, 0)),
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
            stop_events_with_prefix(2_198, 0, fake_report(2, PrefixHit::Miss, 0))
        );
        // A strict extension of the first prompt: the longest stored key, the tip, wins, and
        // the tip that this job saves drops the restored entry (the lineage rule).
        let mut prompt = first_prompt;
        prompt.extend(std::iter::repeat_n(9, 100));
        let (job, events) = job_with_points(&prompt, 2_248, ROOT);
        assert_eq!(
            run_one(&handle, job, events),
            stop_events_with_prefix(
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

    /// R7 of the RED list: the tokens a rewritten prompt restores from a block cut count as
    /// `reused_tokens` like any restore, and the report names the class `block`. The first
    /// prompt (chunk 4, spacing 4) cuts a block entry at 2,144, the last 4-token boundary
    /// below its tip 2,148; the rewrite shares the head to 2,146 tokens, so the tip is not a
    /// prefix of it but the cut is, and the cut wins the restore. The restored cut stays
    /// stored (rule (f) drops only a restored tip), so the next rewrite finds it again.
    #[test]
    fn block_restored_tokens_count_as_reused() {
        let log = new_log();
        let (engine, store) = start_block_spy(4, vec![vec![EOS], vec![EOS]], &log, 4);
        let handle = engine.handle();
        let first_prompt = root_tip_prompt();
        let (first, first_events) = job_with_points(&first_prompt, TIP, ROOT);
        assert_eq!(
            run_one(&handle, first, first_events),
            stop_events_with_prefix(2_198, 0, fake_report(3, PrefixHit::Miss, 0))
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
            stop_events_with_prefix(2_196, 2_144, fake_report(4, PrefixHit::Block, 2)),
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

    /// The recall-cluster shape of the 176K attribution log (Sol round 2): warm append turns
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
            stop_events_with_prefix(3_298, 0, fake_report(3, PrefixHit::Miss, 0)),
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
                finished(2_198, 0)
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
                finished_with_prefix(2_198, 0, 0, fake_report(1, PrefixHit::Miss, 0))
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
            stop_events_with_prefix(2_198, 0, fake_report(2, PrefixHit::Miss, 0))
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
            stop_events_with_prefix(2_198, 0, fake_report(entries, PrefixHit::Miss, 0))
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
        // The next job restores the root and runs to its end.
        let (second, second_events) = job_with_points(&prompt, TIP, ROOT);
        assert_eq!(
            run_one(&handle, second, second_events),
            stop_events_with_prefix(2_198, ROOT as u32, fake_report(2, PrefixHit::Root, 0))
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
                finished_with_prefix(2_198, 0, 0, fake_report(1, PrefixHit::Miss, 0))
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

    /// The `save_prefix` entries of a call log.
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
            stop_events_with_prefix(40_000, 0, fake_report(3, PrefixHit::Miss, 0))
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
            stop_events_with_prefix(34_000, 32_768, fake_report(4, PrefixHit::Periodic, 232)),
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
            stop_events_with_prefix(25_000, 16_384, fake_report(5, PrefixHit::Periodic, 3_616)),
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
                finished_with_prefix(40_000, 0, 0, fake_report(1, PrefixHit::Miss, 0)),
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
            stop_events_with_prefix(40_000, 0, fake_report(1, PrefixHit::Miss, 0))
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
        assert_eq!(run_one(&handle, cold, cold_events), stop_events(40_292, 0));
        let after_cold = calls(&log).len();
        let (cut, cut_events) = job_with_boundary(&prompt, 40_192);
        assert_eq!(
            run_one(&handle, cut, cut_events),
            stop_events_with_prefix(40_292, 0, fake_report(4, PrefixHit::Miss, 0))
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
        assert_eq!(run_one(&handle, cold, cold_events), stop_events(40_292, 0));
        let after_cold = calls(&log).len();
        let (periodic, periodic_events) = job_with_boundary(&prompt, 40_192);
        assert_eq!(
            run_one(&handle, periodic, periodic_events),
            stop_events_with_prefix(40_292, 0, fake_report(3, PrefixHit::Miss, 0))
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
        assert_eq!(run_one(&handle, cold, cold_events), stop_events(20_000, 0));
        let after_cold = calls(&log).len();
        let (periodic, periodic_events) = job_with_boundary(&prompt, 19_900);
        assert_eq!(
            run_one(&handle, periodic, periodic_events),
            stop_events_with_prefix(20_000, 0, fake_report(2, PrefixHit::Miss, 0))
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
