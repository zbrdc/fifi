//! The engine worker: one OS thread per engine, one generation at a time.
//!
//! Design: `docs/design/scheduler.md` ("Threads and channels", "Engine lifecycle") and
//! `docs/design/family-descriptor.md` ("Executor trait", the worker call order).
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
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use tokio::sync::mpsc;

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
}

impl Shared {
    fn new() -> Self {
        Shared {
            state: AtomicU8::new(EngineState::Loading as u8),
            failure: Mutex::new(None),
            stop: AtomicBool::new(false),
            running: Mutex::new(None),
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

/// Starts the engine thread. `make` runs on that thread and builds the executor and its session,
/// so thread-confined native values never cross a thread boundary. `prefill_chunk` is the
/// largest prompt slice passed to one `prefill` call when the executor does not plan its own
/// piece. An executor whose `prefill_piece()` is `Some(n)` sets the piece to `n` after `make`.
pub(crate) fn spawn_engine<E, F>(prefill_chunk: usize, make: F) -> Result<Engine, EngError>
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
        .spawn(move || run::<E, F>(make, &jobs, &worker_shared, prefill_chunk))
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

fn run<E, F>(make: F, jobs: &Receiver<Message>, shared: &Shared, prefill_chunk: usize)
where
    E: ExecutorOps,
    F: FnOnce() -> Result<(E, E::Session), EngError>,
{
    let mut current: Option<mpsc::Sender<JobEvent>> = None;
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        work::<E, F>(make, jobs, shared, prefill_chunk, &mut current);
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
        drain(jobs, reject);
        // Last: dropping a payload can itself panic, and `shutdown` then reports that panic.
        drop(payload);
    }
}

fn work<E, F>(
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
            shared.record_failure(format!("engine load failed: {error:?}"));
            shared.set_state(EngineState::Error);
            drain(jobs, reject);
            return;
        }
    };
    let prefill_chunk = executor.prefill_piece().unwrap_or(prefill_chunk);
    if prefill_chunk == 0 {
        shared.record_failure(
            "engine load failed: the executor planned a prefill piece of 0".to_string(),
        );
        shared.set_state(EngineState::Error);
        drain(jobs, reject);
        return;
    }
    shared.set_state(EngineState::Ready);
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

/// The prompt of one job and the length of its reusable prefix.
struct PromptPlan<'a> {
    tokens: &'a [u32],
    /// The number of leading tokens that end at the last message boundary, or `None`.
    boundary: Option<usize>,
}

/// Runs one job and returns the state of the engine after it.
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
    let reused = match start(executor, session, plan, prefill_chunk, &generation.cancel) {
        Ok(reused) => reused,
        Err(error) => return fail(executor, session, events, error, shared),
    };
    let reused_tokens = u32::try_from(reused).unwrap_or(u32::MAX);
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
        if finished {
            send_terminal(
                events,
                JobEvent::Finished(JobUsage {
                    prompt_tokens,
                    completion_tokens,
                    reused_tokens,
                }),
            );
            return EngineState::Ready;
        }
    }
}

/// Resets the session, restores a saved prefix when the job has a boundary, runs `boundary`, and
/// prefills the rest of the prompt in chunks. When the prefill reaches the job's boundary, it
/// saves the prefix state. Returns the number of prompt tokens that the restore covered.
fn start<E: ExecutorOps>(
    executor: &mut E,
    session: &mut E::Session,
    plan: &PromptPlan<'_>,
    prefill_chunk: usize,
    cancel: &AtomicBool,
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
    if let Some(boundary) = plan.boundary.filter(|&boundary| boundary > restored) {
        let head = span(plan.tokens, restored, boundary)?;
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
        executor.save_prefix(session, span(plan.tokens, 0, boundary)?)?;
        next = boundary;
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
    for piece in tokens.chunks(prefill_chunk) {
        if *chunks_run > 0 && cancel.load(Ordering::SeqCst) {
            return Ok(false);
        }
        executor.prefill(session, piece)?;
        *chunks_run += 1;
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
    use crate::sched::job::{Job, JobEvent, JobHandle, JobUsage};
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
        fail_first_decode: Option<EngError>,
        cancel_after_prefill: Option<Arc<AtomicBool>>,
        /// Fails the n-th `reset` call (0 is the first) with the error.
        fail_reset_call: Option<(usize, EngError)>,
        reset_calls: usize,
        fail_prefill: Option<EngError>,
        panic_in_decode: Option<&'static str>,
        /// Makes `restore_prefix` report this many covered tokens without calling the fake.
        restore_override: Option<usize>,
        /// Fails the next `save_prefix` call with the error, before the fake sees it.
        fail_save: Option<EngError>,
        /// Fails the next `restore_prefix` call with the error, before the fake sees it.
        fail_restore: Option<EngError>,
        /// The piece that `prefill_piece` reports.
        piece: Option<usize>,
    }

    impl Spy {
        fn new(script: Vec<Vec<u32>>, log: &Log) -> Self {
            Spy {
                inner: FakeExecutor::new(script, &[EOS], 100_000),
                log: Arc::clone(log),
                hold: None,
                fail_first_decode: None,
                cancel_after_prefill: None,
                fail_reset_call: None,
                reset_calls: 0,
                fail_prefill: None,
                panic_in_decode: None,
                restore_override: None,
                fail_save: None,
                fail_restore: None,
                piece: None,
            }
        }

        fn note(&self, entry: String) {
            self.log.lock().expect("log lock").push(entry);
        }
    }

    impl ExecutorOps for Spy {
        type Session = <FakeExecutor as ExecutorOps>::Session;

        fn history<'s>(&self, st: &'s Self::Session) -> &'s [u32] {
            self.inner.history(st)
        }

        fn prefill_piece(&self) -> Option<usize> {
            self.piece
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
            if let Some(flag) = &self.cancel_after_prefill {
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

        fn boundary(&mut self, st: &mut Self::Session) {
            self.note("boundary".to_string());
            self.inner.boundary(st);
        }

        fn save_prefix(&mut self, st: &Self::Session, key: &[u32]) -> Result<bool, EngError> {
            self.note(format!("save_prefix:{}", key.len()));
            if let Some(error) = self.fail_save.take() {
                return Err(error);
            }
            self.inner.save_prefix(st, key)
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
        JobEvent::Finished(JobUsage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            reused_tokens: reused,
        })
    }

    /// A job whose first `boundary` prompt tokens form the reusable prefix.
    fn job_with_boundary(prompt: &[u32], boundary: usize) -> (Job, JobHandle) {
        let (job, handle) = job(prompt);
        (
            job.with_prefix_boundary(boundary).expect("boundary"),
            handle,
        )
    }

    /// Submits the job, waits for its end, and returns its events.
    fn run_one(handle: &EngineHandle, job: Job, events: JobHandle) -> Vec<JobEvent> {
        handle.try_submit(job).expect("submit");
        collect(events)
    }

    /// The events of a job that stops at once and uses no extra tokens.
    fn stop_events(prompt: u32, reused: u32) -> Vec<JobEvent> {
        vec![
            batch(&[EOS], "", Some(DecodeFinish::Stop)),
            finished_reusing(prompt, 1, reused),
        ]
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
        let engine = start_spy(2, Spy::new(vec![vec![EOS]], &log));
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
        let mut spy = Spy::new(vec![vec![EOS]], &log);
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
                finished(2, 0)
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
    fn a_backend_error_fails_the_engine_and_the_queued_jobs() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]], &log);
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
        let mut spy = Spy::new(vec![vec![EOS]], &log);
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
        let mut spy = Spy::new(vec![vec![EOS]], &log);
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
        let mut spy = Spy::new(vec![vec![EOS]], &log);
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
        let mut spy = Spy::new(vec![vec![EOS]], &log);
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
        let mut spy = Spy::new(vec![vec![EOS]], &log);
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
        let mut spy = Spy::new(vec![vec![EOS]], &log);
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
        let mut spy = Spy::new(vec![vec![EOS]], &log);
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
        let engine = start_spy(8, Spy::new(vec![vec![EOS]], &log));
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
        let mut spy = Spy::new(vec![vec![EOS]], &log);
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
        assert_eq!(collect(queued_events), cancelled);
        assert_eq!(
            calls(&log),
            strings(&["reset", "boundary", "prefill:1", "decode"])
        );
        assert_eq!(handle.state(), EngineState::Stopped);
    }

    #[test]
    fn a_job_queued_behind_the_stop_message_is_cancelled() {
        let log = new_log();
        let spy = Spy::new(vec![vec![EOS]], &log);
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
                finished(1, 0)
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
        assert_eq!(run_one(&handle, first, first_events), stop_events(6, 0));
        let (second, second_events) = job_with_boundary(&[2, 2, 2, 2, 7, 7, 7, 7, 7], 6);
        assert_eq!(run_one(&handle, second, second_events), stop_events(9, 4));
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
    fn a_b_a_anonymous_prompts_start_clean_and_a_repeat_of_a_reuses_its_own_state() {
        let log = new_log();
        let engine = start_spy(100, Spy::new(vec![vec![EOS]; 4], &log));
        let handle = engine.handle();
        let (a1, a1_events) = job_with_boundary(&[2, 2, 2, 2, 2], 3);
        assert_eq!(run_one(&handle, a1, a1_events), stop_events(5, 0));
        // B shares no key with A. It restores nothing and replaces the saved state.
        let (b, b_events) = job_with_boundary(&[1, 1, 1, 1, 1], 3);
        assert_eq!(run_one(&handle, b, b_events), stop_events(5, 0));
        // A returns after B. The saved state is now B's, so A restores nothing.
        let (a2, a2_events) = job_with_boundary(&[2, 2, 2, 2, 2, 0], 3);
        assert_eq!(run_one(&handle, a2, a2_events), stop_events(6, 0));
        // A third prompt of A starts with the key that A2 saved.
        let (a3, a3_events) = job_with_boundary(&[2, 2, 2, 2, 2, 0, 0], 3);
        assert_eq!(run_one(&handle, a3, a3_events), stop_events(7, 3));
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
        assert_eq!(run_one(&handle, first, first_events), stop_events(5, 0));
        let (second, second_events) = job(&[2, 2, 2, 2, 2]);
        assert_eq!(run_one(&handle, second, second_events), stop_events(5, 0));
        let calls = calls(&log);
        let tail = calls.get(7..).expect("calls of the second job");
        assert_eq!(
            tail,
            strings(&["reset", "boundary", "prefill:5", "decode"]).as_slice()
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
        let mut spy = Spy::new(vec![vec![EOS]], &log);
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
        let engine = start_spy(100, spy);
        let handle = engine.handle();
        let (first, first_events) = job_with_boundary(&[2, 2, 2, 2, 2], 3);
        assert_eq!(run_one(&handle, first, first_events), stop_events(5, 0));
        let (second, second_events) = job_with_boundary(&[2, 2, 2, 2, 2, 2], 3);
        assert_eq!(run_one(&handle, second, second_events), stop_events(6, 0));
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
        drop(handle);
        engine.shutdown();
    }

    #[test]
    fn a_restore_between_the_boundary_and_the_prompt_length_skips_the_save_and_prefills_the_rest() {
        let log = new_log();
        let mut spy = Spy::new(vec![vec![EOS]], &log);
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
        let mut spy = Spy::new(vec![vec![EOS]], &log);
        spy.fail_save = Some(EngError::MemoryPlan("save boom".to_string()));
        let engine = start_spy(100, spy);
        let handle = engine.handle();
        let (first, first_events) = job_with_boundary(&[2, 2, 2, 2, 2], 3);
        assert_eq!(
            run_one(&handle, first, first_events),
            vec![JobEvent::Error(EngError::MemoryPlan(
                "save boom".to_string()
            ))]
        );
        assert_eq!(handle.state(), EngineState::Ready);
        // The next job starts clean and runs to its end.
        let (second, second_events) = job_with_boundary(&[2, 2, 2, 2, 2], 3);
        assert_eq!(run_one(&handle, second, second_events), stop_events(5, 0));
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
        let mut spy = Spy::new(vec![vec![EOS]], &log);
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
        let mut spy = Spy::new(vec![vec![EOS]], &log);
        spy.fail_restore = Some(EngError::MemoryPlan("restore boom".to_string()));
        let engine = start_spy(100, spy);
        let handle = engine.handle();
        let (first, first_events) = job_with_boundary(&[2, 2, 2, 2, 2], 3);
        assert_eq!(
            run_one(&handle, first, first_events),
            vec![JobEvent::Error(EngError::MemoryPlan(
                "restore boom".to_string()
            ))]
        );
        assert_eq!(handle.state(), EngineState::Ready);
        let (second, second_events) = job_with_boundary(&[2, 2, 2, 2, 2], 3);
        assert_eq!(run_one(&handle, second, second_events), stop_events(5, 0));
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
        let mut spy = Spy::new(vec![vec![EOS]], &log);
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
            let mut spy = Spy::new(vec![vec![EOS]], &log);
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
        let engine = start_spy(2, spy);
        let handle = engine.handle();
        wait_until("the load failure", || handle.state() == EngineState::Error);
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

    #[test]
    fn the_engine_handle_can_be_shared_between_threads() {
        fn assert_shareable<T: Clone + Send + Sync + 'static>() {}
        assert_shareable::<EngineHandle>();
    }
}
