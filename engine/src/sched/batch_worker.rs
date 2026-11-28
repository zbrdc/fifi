//! The batch worker: up to `max_seqs` sessions on the engine thread, one decode window per tick.
//! No model or adapter is cloned.
//! A failed window discards its sessions, including any consumed pending tokens.
//! Output uses the job channel's reserved terminal slot and never waits for a receiver.
//!
//! Sessions live in a pool. A session that leaves its slot normally (finished, cancelled, refused
//! before its feed) goes back to the pool and is reset for the next job; `new_session` runs only
//! when the pool is empty, and the slot count bounds the pool, so an engine holds at most
//! `max_seqs` sessions. A session of a failed window is dropped, because its state cannot be
//! trusted.

use super::{
    ActiveJob, EngineState, Message, PromptPlan, QueueTicket, Shared, cancel_unstarted, classified,
    drain, engine_error, evict_for_retry, lock, prompt_tokens, reject, save_points, span,
};
use crate::EngError;
use crate::family::executor::ExecutorOps;
use crate::family::session::SaveKind;
use crate::sched::job::{
    Job, JobEvent, JobUsage, PrefillPieces, QUEUE_CAPACITY, send_cancelled, send_data,
    send_terminal,
};
use crate::tokenizer::{DecodeBatch, DecodeFinish};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

struct Slot<S> {
    session: S,
    job: Job,
    cursor: usize,
    points: VecDeque<(usize, SaveKind)>,
    usage: JobUsage,
    ready: bool,
    finished: bool,
    slow: bool,
    /// Keep attempted work across prefill retries. Publish once the slot becomes ready.
    prefill_bank: Option<crate::sched::job::PrefillBankReport>,
    /// Whether the eviction restart already ran for this slot: a restarted slot saves
    /// nothing. Later denials are NOT final by that alone: they evict one more entry and
    /// restart again - final only through `exhausted_restart` below, the classify gate, or
    /// a failing rebuild.
    retried: bool,
    /// Whether the store was already EXHAUSTED for this slot and the unclassified no-save
    /// restart was taken anyway: a persistent denial that no eviction has relieved since
    /// must not restart again - the next exhausted denial classifies regardless of the
    /// ceiling, exactly one bounded recovery. Any successful eviction resets it, so
    /// contention that frees entries keeps its sized restarts.
    exhausted_restart: bool,
}

impl<S> Slot<S> {
    fn can_start_decode(&self) -> bool {
        self.cursor == self.job.prompt.len() && !self.job.generation.cancel.load(Ordering::SeqCst)
    }
}

/// What the ticks of the batch worker did. The counters only grow; a reader takes a snapshot
/// and subtracts.
#[derive(Default)]
pub(super) struct TickCounters {
    /// Ticks that reached the decode step, whether or not a window ran.
    ticks: AtomicU64,
    /// Windows handed to the executor.
    windows: AtomicU64,
    /// Rows of those windows: one per ready slot.
    decode_rows: AtomicU64,
    /// The widest window.
    max_window: AtomicU64,
    /// Prompt rows fed by prefill pieces.
    prefill_rows: AtomicU64,
    /// Nanoseconds inside executor calls (prefill pieces, windows, `after_emit`).
    executor_ns: AtomicU64,
    /// Nanoseconds of tick time outside executor calls: planning, sampling bookkeeping,
    /// channel sends.
    host_ns: AtomicU64,
    /// Sessions the executor allocated for the pool.
    sessions_created: AtomicU64,
    /// Jobs that took a pooled session that had served a job before.
    sessions_reused: AtomicU64,
    /// Sessions released because the engine sat idle.
    sessions_trimmed: AtomicU64,
    /// Finished sessions released at once because the executor keeps only one.
    sessions_dropped: AtomicU64,
}

/// A copy of the counters at one moment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct TickSnapshot {
    pub(super) ticks: u64,
    pub(super) windows: u64,
    pub(super) decode_rows: u64,
    pub(super) max_window: u64,
    pub(super) prefill_rows: u64,
    pub(super) executor_ns: u64,
    pub(super) host_ns: u64,
    pub(super) sessions_created: u64,
    pub(super) sessions_reused: u64,
    pub(super) sessions_trimmed: u64,
    pub(super) sessions_dropped: u64,
}

impl TickCounters {
    pub(super) fn snapshot(&self) -> TickSnapshot {
        let read = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        TickSnapshot {
            ticks: read(&self.ticks),
            windows: read(&self.windows),
            decode_rows: read(&self.decode_rows),
            max_window: read(&self.max_window),
            prefill_rows: read(&self.prefill_rows),
            executor_ns: read(&self.executor_ns),
            host_ns: read(&self.host_ns),
            sessions_created: read(&self.sessions_created),
            sessions_reused: read(&self.sessions_reused),
            sessions_trimmed: read(&self.sessions_trimmed),
            sessions_dropped: read(&self.sessions_dropped),
        }
    }

    fn add(counter: &AtomicU64, amount: u64) {
        counter.fetch_add(amount, Ordering::Relaxed);
    }

    fn nanos(duration: Duration) -> u64 {
        u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
    }
}

/// Times one tick from the end of admission to the end of the loop body, however the body
/// exits, and splits the time into executor calls and the rest.
struct TickTimer<'a> {
    counters: &'a TickCounters,
    started: Instant,
    executor: Duration,
}

impl<'a> TickTimer<'a> {
    fn new(counters: &'a TickCounters) -> Self {
        Self {
            counters,
            started: Instant::now(),
            executor: Duration::ZERO,
        }
    }

    fn executor_call(&mut self, since: Instant) {
        self.executor += since.elapsed();
    }
}

impl Drop for TickTimer<'_> {
    fn drop(&mut self) {
        let host = self.started.elapsed().saturating_sub(self.executor);
        TickCounters::add(
            &self.counters.executor_ns,
            TickCounters::nanos(self.executor),
        );
        TickCounters::add(&self.counters.host_ns, TickCounters::nanos(host));
    }
}

/// The sessions that no slot holds. Each is reset by `setup_slot` when a job takes it. A session
/// that has served a job is `used`: the seed the executor built at start is not, so taking it is
/// not counted as a reuse. Each carries the time it went idle, so the pool can release the
/// sessions that have idled longest.
struct Pool<'a, S> {
    free: Vec<PooledSession<S>>,
    counters: &'a TickCounters,
    /// How long a pooled session may idle before it is released; `None` keeps every session.
    idle_trim: Option<Duration>,
}

struct PooledSession<S> {
    session: S,
    used: bool,
    idle_since: Instant,
}

impl<'a, S> Pool<'a, S> {
    fn new(first: S, counters: &'a TickCounters, idle_trim: Option<Duration>) -> Self {
        Self {
            free: vec![PooledSession {
                session: first,
                used: false,
                idle_since: Instant::now(),
            }],
            counters,
            idle_trim,
        }
    }

    fn take(&mut self) -> Option<S> {
        let pooled = self.free.pop()?;
        if pooled.used {
            TickCounters::add(&self.counters.sessions_reused, 1);
        }
        Some(pooled.session)
    }

    /// Takes a finished session back. An executor that has not opted into the trim keeps one
    /// session and releases every finished session beyond it at once, as the worker did before
    /// it pooled sessions: a burst does not leave its sessions allocated for the life of the
    /// engine.
    fn give(&mut self, session: S) {
        if self.idle_trim.is_none() && !self.free.is_empty() {
            TickCounters::add(&self.counters.sessions_dropped, 1);
            return;
        }
        self.give_at(session, Instant::now());
    }

    fn give_at(&mut self, session: S, idle_since: Instant) {
        self.free.push(PooledSession {
            session,
            used: true,
            idle_since,
        });
    }

    /// Releases the pooled sessions that have idled for the trim interval, longest idle first,
    /// until `keep` pooled sessions would remain at most: an idle engine keeps one for its next
    /// request, an engine with a live slot keeps none that has idled that long. A session that
    /// has idled less than the interval stays.
    fn trim(&mut self, now: Instant, keep: usize) {
        let Some(interval) = self.idle_trim else {
            return;
        };
        let mut released = 0u64;
        while self.free.len() > keep {
            let oldest = self
                .free
                .iter()
                .enumerate()
                .min_by_key(|(_, pooled)| pooled.idle_since)
                .map(|(index, _)| index);
            match oldest {
                Some(index)
                    if now.saturating_duration_since(self.free[index].idle_since) >= interval =>
                {
                    self.free.swap_remove(index);
                    released += 1;
                }
                _ => break,
            }
        }
        TickCounters::add(&self.counters.sessions_trimmed, released);
    }
}

/// How long a job that capacity refused waits before it asks again while every slot stays live:
/// memory can come back without a slot freeing (another process lets go), and a cancel by the
/// client is noticed within the same interval.
const WAIT_RETRY: Duration = Duration::from_secs(1);

/// A job that capacity refused while other slots were live. It keeps its queue ticket, so the
/// `queued` figure of `/v1/system` still counts it while it waits.
struct Waiting {
    job: Job,
    ticket: QueueTicket,
    /// The live slots at the refusal: the job asks again when fewer are live.
    live: usize,
    /// The time of its next timed retry.
    retry_at: Instant,
}

fn register(shared: &Shared, job: &Job) {
    lock(&shared.active).push(ActiveJob {
        cancel: Arc::clone(&job.generation.cancel),
        events: job.events.clone(),
    });
    // Register first, then read stop, matching shutdown's store-then-scan order.
    if shared.stop.load(Ordering::SeqCst) {
        job.generation.cancel.store(true, Ordering::SeqCst);
    }
}

fn unregister(shared: &Shared, job: &Job) {
    lock(&shared.active).retain(|entry| !Arc::ptr_eq(&entry.cancel, &job.generation.cancel));
}

fn copy_error(error: &EngError) -> EngError {
    match error {
        EngError::ModelOutput(message) => EngError::ModelOutput(message.clone()),
        EngError::Checkpoint(message) => EngError::Checkpoint(message.clone()),
        EngError::BadRequest(message) => EngError::BadRequest(message.clone()),
        EngError::MemoryPlan(message) => EngError::MemoryPlan(message.clone()),
        EngError::ContextLimit(message) => EngError::ContextLimit(message.clone()),
        EngError::PromptTooLong {
            prompt_tokens,
            limit_tokens,
        } => EngError::PromptTooLong {
            prompt_tokens: *prompt_tokens,
            limit_tokens: *limit_tokens,
        },
        EngError::Backend { operation, message } => EngError::Backend {
            operation,
            message: message.clone(),
        },
    }
}

/// No executor calls follow this transition, even for another active or queued job.
fn native_failed(shared: &Shared, jobs: &Receiver<Message>, error: &EngError) {
    shared.record_failure(super::EngineFailure::from_error(
        error,
        "runtime",
        format!("native failure: {}", error.describe()),
    ));
    shared.set_state(EngineState::NativeFailed);
    for job in lock(&shared.active).drain(..) {
        job.cancel.store(true, Ordering::SeqCst);
        send_terminal(&job.events, JobEvent::Error(copy_error(error)));
    }
    drain(jobs, reject);
}

fn cancelled<S>(slot: &mut Slot<S>) {
    slot.job.generation.cancel.store(true, Ordering::SeqCst);
    // Finish the job's own incremental decoder without touching the executor on shutdown.
    match slot.job.generation.decoder.finish() {
        Ok(text) => {
            if send_data(
                &slot.job.events,
                JobEvent::Batch(DecodeBatch {
                    tokens: Vec::new(),
                    text,
                    finish: Some(DecodeFinish::Cancelled),
                }),
            ) {
                send_terminal(&slot.job.events, JobEvent::Finished(slot.usage.clone()));
            }
        }
        Err(error) => send_terminal(&slot.job.events, JobEvent::Error(error)),
    }
}

fn remove_cancelled<S>(
    slots: &mut Vec<Slot<S>>,
    index: usize,
    shared: &Shared,
    pool: &mut Pool<'_, S>,
) -> bool {
    if !slots[index].job.generation.cancel.load(Ordering::SeqCst) {
        return false;
    }
    let mut slot = slots.remove(index);
    cancelled(&mut slot);
    unregister(shared, &slot.job);
    pool.give(slot.session);
    true
}

/// Rebuilds a slot's prefill state from a reset session: restore, boundary, save points,
/// cursor, and usage. A prefill piece that was denied leaves the session partially fed -
/// the pending token taken, earlier split parts published - so the eviction restart must
/// rebuild the slot the way `admit` built it, never re-feed the failed piece.
fn setup_slot<E: ExecutorOps>(
    executor: &mut E,
    slot: &mut Slot<E::Session>,
    prefill_chunk: usize,
) -> Result<(), EngError> {
    executor.reset_for_request(
        &mut slot.session,
        slot.job.prompt.len(),
        slot.job.generation.remaining_tokens(),
        slot.job.prefix_boundary.is_some(),
    )?;
    let restored = match slot.job.prefix_boundary {
        Some(tip) => executor.restore_prefix_under(&mut slot.session, &slot.job.prompt, tip)?,
        None => 0,
    };
    if restored >= slot.job.prompt.len() {
        return Err(EngError::BadRequest(
            "restored prefix covers the whole prompt".to_string(),
        ));
    }
    executor.boundary(&mut slot.session);
    // Restore reporting belongs to this job. Another slot can reset or restore next.
    let prefix = executor.prefix_report();
    let plan = PromptPlan {
        tokens: &slot.job.prompt,
        boundary: slot.job.prefix_boundary,
        root: slot.job.root_boundary,
    };
    slot.points = if slot.retried {
        // The eviction restart admits the prompt with no save points:
        // a snapshot that charges successfully at its own point can still starve the
        // prefill tail that follows it, so a restarted slot never saves.
        VecDeque::new()
    } else {
        save_points(
            &plan,
            restored,
            executor.prefix_limits(),
            super::request_prefill_chunk(executor, &slot.session, prefill_chunk),
            super::block_spacing(executor.prefix_block_spacing(), plan.boundary),
        )
        .into()
    };
    slot.usage.prompt_tokens = prompt_tokens(&slot.job.prompt);
    slot.usage.reused_tokens = u32::try_from(restored).unwrap_or(u32::MAX);
    slot.usage.prefix = prefix;
    // A restart re-feeds from the cursor, so the piece count keeps only the run that
    // served the prompt, never the aborted attempt's pieces (mirrors the single worker).
    slot.usage.prefill_pieces = PrefillPieces::default();
    slot.cursor = restored;
    slot.ready = false;
    Ok(())
}

/// Builds a slot for `job`, returning the job on every error so the caller can retry or
/// report it.
fn build_slot<E: ExecutorOps>(
    executor: &mut E,
    pool: &mut Pool<'_, E::Session>,
    job: Job,
    prefill_chunk: usize,
) -> Result<Slot<E::Session>, (EngError, Box<Job>)> {
    if let Err(error) = executor.check_prompt(job.prompt.len()) {
        return Err((error, Box::new(job)));
    }
    let session = match pool.take() {
        Some(session) => session,
        None => match executor.new_session() {
            Ok(session) => {
                TickCounters::add(&pool.counters.sessions_created, 1);
                session
            }
            Err(error) => return Err((error, Box::new(job))),
        },
    };
    let mut slot = Slot {
        session,
        job,
        cursor: 0,
        points: VecDeque::new(),
        usage: JobUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            reused_tokens: 0,
            prefix: None,
            prefill_bank: None,
            prefill_pieces: PrefillPieces::default(),
            first_token_margin: None,
        },
        ready: false,
        finished: false,
        slow: false,
        prefill_bank: None,
        retried: false,
        exhausted_restart: false,
    };
    match setup_slot(executor, &mut slot, prefill_chunk) {
        Ok(()) => Ok(slot),
        Err(error) => {
            // A session whose reset or restore failed is not trusted again: it is dropped, and
            // the pool allocates a fresh one when it is next empty.
            drop(slot.session);
            Err((error, Box::new(slot.job)))
        }
    }
}

fn admit<E: ExecutorOps>(
    executor: &mut E,
    pool: &mut Pool<'_, E::Session>,
    mut job: Job,
    prefill_chunk: usize,
) -> Result<Slot<E::Session>, (EngError, Box<Job>)> {
    match build_slot(executor, pool, job, prefill_chunk) {
        Ok(slot) => Ok(slot),
        Err((EngError::MemoryPlan(message), job_back))
            if !executor.eviction_frees_session_memory() =>
        {
            // The sessions and the prefix entries do not share memory: the denial stands.
            Err((EngError::MemoryPlan(message), job_back))
        }
        Err((EngError::MemoryPlan(message), job_back)) => {
            // The eviction restart for admission itself, incrementally: `new_session`
            // charges fixed state and the restore charges its entry before any prefill
            // runs, so cached entries can block the admission of any session at all. Evict
            // ONE entry at a time - a dead tip first, oldest first, this prompt's own
            // restore candidate last - and rebuild after each, so a denial with a one-entry
            // deficit costs one anchor, not every live lineage. This admission path is NOT
            // ceiling-sized: no session exists yet to measure a ceiling from, and
            // `new_session`'s charge arithmetic bounds each rebuild tightly, so the
            // one-entry-per-rebuild loop here is the sized form for this path. When the
            // last entry is gone and the build still denies, the refusal is not about this
            // prompt - no session survives to measure a ceiling from - so the memory-plan
            // refusal stands as-is. The denial belongs to no live session, so the eviction
            // protects the entry THIS prompt would restore next - never the store-global
            // last-restore marker, which names ANOTHER conversation in an interleaved
            // batch.
            // Terminal when nothing can be freed: an empty store cannot admit a rebuild
            // either, so the refusal stands as-is instead of one dead `build_slot`.
            if !evict_for_retry(
                executor,
                None,
                &job_back.prompt,
                &job_back.generation.cancel,
            ) {
                return Err((EngError::MemoryPlan(message), job_back));
            }
            job = *job_back;
            let mut retried = build_slot(executor, pool, job, prefill_chunk);
            loop {
                match retried {
                    Ok(slot) => {
                        // The admission retry spent this request's first eviction: mark it
                        // retried, so the slot - like every retried slot - saves nothing.
                        // It MAY still evict incrementally on a later denial: the first
                        // denial freed one entry, and the next may need only one more.
                        return Ok(Slot {
                            retried: true,
                            points: VecDeque::new(),
                            ..slot
                        });
                    }
                    Err((EngError::MemoryPlan(message), back)) => {
                        if !evict_for_retry(executor, None, &back.prompt, &back.generation.cancel) {
                            return Err((EngError::MemoryPlan(message), back));
                        }
                        retried = build_slot(executor, pool, *back, prefill_chunk);
                    }
                    Err(failed) => return Err(failed),
                }
            }
        }
        Err(failed) => Err(failed),
    }
}

fn refresh_report<E: ExecutorOps>(executor: &E, slot: &mut Slot<E::Session>) {
    // Refresh storage totals, never the restore facts that another slot may have changed.
    if let (Some(report), Some(now)) = (&mut slot.usage.prefix, executor.prefix_report()) {
        report.entries = now.entries;
        report.bytes = now.bytes;
    }
}

/// The rows of one prefill piece of a slot: the executor's planned piece, cut to the request's
/// own width and the engine's chunk, then to the next save point or the end of the prompt. It
/// depends on that prompt and its session only, never on the other slots of the tick, so a prompt
/// is fed the same pieces whatever shares its ticks.
fn piece_rows<E: ExecutorOps>(
    executor: &E,
    slot: &Slot<E::Session>,
    prefill_chunk: usize,
) -> Result<usize, EngError> {
    let prefill_chunk = super::request_prefill_chunk(executor, &slot.session, prefill_chunk);
    let rows = executor
        .prefill_piece()
        .unwrap_or(prefill_chunk)
        .min(prefill_chunk);
    if rows == 0 {
        return Err(EngError::BadRequest(
            "the executor planned a prefill piece of 0".to_string(),
        ));
    }
    let target = slot
        .points
        .front()
        .map_or(slot.job.prompt.len(), |point| point.0);
    Ok(rows.min(target.saturating_sub(slot.cursor)))
}

/// Feeds one whole piece of the slot, respecting the executor's piece limit and save points.
fn prefill<E: ExecutorOps>(
    executor: &mut E,
    slot: &mut Slot<E::Session>,
    prefill_chunk: usize,
) -> Result<(), EngError> {
    let rows = piece_rows(executor, slot, prefill_chunk)?;
    let end = slot.cursor + rows;
    let piece = span(&slot.job.prompt, slot.cursor, end)?;
    let before = executor.prefill_bank_report();
    let result = executor.prefill(&mut slot.session, piece);
    if let Some(before) = before
        && let Some(after) = executor.prefill_bank_report()
    {
        slot.prefill_bank
            .get_or_insert_default()
            .add_delta(&before, &after);
    }
    result?;
    slot.usage.prefill_pieces.count = slot.usage.prefill_pieces.count.saturating_add(1);
    slot.usage.prefill_pieces.max_rows = slot
        .usage
        .prefill_pieces
        .max_rows
        .max(u32::try_from(end - slot.cursor).unwrap_or(u32::MAX));
    slot.cursor = end;
    if slot.job.generation.cancel.load(Ordering::SeqCst) {
        refresh_report(executor, slot);
        return Ok(());
    }
    if let Some((_, kind)) = slot.points.pop_front_if(|point| point.0 == end) {
        executor.save_prefix(&slot.session, span(&slot.job.prompt, 0, end)?, kind)?;
    }
    refresh_report(executor, slot);
    if slot.cursor == slot.job.prompt.len() {
        slot.usage.prefill_bank = slot.prefill_bank.take();
    }
    Ok(())
}

/// Feeds up to `pieces` whole pieces of one slot's prompt. It stops when they are fed, the prompt
/// is whole, the slot is cancelled, or a piece feeds nothing.
fn prefill_pieces<E: ExecutorOps>(
    executor: &mut E,
    slot: &mut Slot<E::Session>,
    prefill_chunk: usize,
    pieces: usize,
) -> Result<(), EngError> {
    for _ in 0..pieces {
        let before = slot.cursor;
        prefill(executor, slot, prefill_chunk)?;
        if slot.cursor == before
            || slot.cursor == slot.job.prompt.len()
            || slot.job.generation.cancel.load(Ordering::SeqCst)
        {
            break;
        }
    }
    Ok(())
}

/// The whole pieces each cold slot feeds this tick, in slot order. `rows` is the tick's prompt row
/// budget and `piece` the engine's piece width: the budget buys `rows / piece` pieces, at least
/// one, dealt one at a time across the cold slots starting at `turn`, so slots take turns when
/// the pieces are fewer than the slots and share them evenly otherwise. Without a prompt share
/// (`several` false) a slot takes at most one piece a tick. A piece is never cut to the budget:
/// what a prompt is fed does not depend on the other prompts of the tick.
fn piece_grants(cold: usize, rows: usize, piece: usize, turn: usize, several: bool) -> Vec<usize> {
    let mut total = (rows / piece.max(1)).max(1);
    if !several {
        total = total.min(cold);
    }
    let (each, extra) = (total / cold, total % cold);
    (0..cold)
        .map(|ordinal| each + usize::from((ordinal + cold - turn % cold) % cold < extra))
        .collect()
}

/// Removes only failed window rows. Their mutated generations and sessions cannot be replayed.
fn window_failed<S>(slots: &mut Vec<Slot<S>>, shared: &Shared, error: &EngError) {
    slots.retain(|slot| {
        if slot.ready {
            if !slot.slow {
                send_terminal(&slot.job.events, JobEvent::Error(copy_error(error)));
            }
            unregister(shared, &slot.job);
            false
        } else {
            true
        }
    });
}

pub(super) fn work<E: ExecutorOps>(
    executor: &mut E,
    session: E::Session,
    jobs: &Receiver<Message>,
    shared: &Shared,
    prefill_chunk: usize,
    iteration_rows: usize,
    max_seqs: usize,
) {
    work_loop(
        executor,
        session,
        jobs,
        shared,
        prefill_chunk,
        iteration_rows,
        max_seqs,
    );
    // Every way out of the loop passes here, so the background work (a disk flush of the saved
    // prefixes) ends the same way as in the single-session worker.
    super::finish_background(executor, shared);
}

fn work_loop<E: ExecutorOps>(
    executor: &mut E,
    session: E::Session,
    jobs: &Receiver<Message>,
    shared: &Shared,
    prefill_chunk: usize,
    iteration_rows: usize,
    max_seqs: usize,
) {
    let share = executor.prompt_rows_per_tick();
    let policy = executor.batch_policy();
    let mut pool = Pool::new(session, &shared.ticks, policy.idle_trim);
    let mut slots: Vec<Slot<E::Session>> = Vec::new();
    let mut waiting: Option<Waiting> = None;
    // The cold slot that is dealt the first piece of the tick; it moves on every tick that has a
    // cold slot, so slots take turns when the budget buys fewer pieces than there are slots.
    let mut turn = 0usize;
    // `iteration_rows` is the engine's per-tick row budget (the rows the engine was spawned
    // with). A tick feeds at most one planned piece of prompt rows beside the decode rows, so
    // the budget is never below the piece: a smaller one only slows every cold prompt, while a
    // stream waits at most one piece per tick.
    let iteration_rows = iteration_rows.max(prefill_chunk);
    loop {
        if shared.stop.load(Ordering::SeqCst) {
            break;
        }
        // Pooled sessions that have idled for the trim interval go, even while other streams
        // run; an idle engine keeps one.
        pool.trim(Instant::now(), usize::from(slots.is_empty()));
        // Background work (a disk write-behind) advances by one bounded step per tick while
        // streams run; with none running the idle wait below drives it.
        if !slots.is_empty() && executor.background_pending() {
            executor.background_step();
        }
        // A waiting job whose client has gone is cancelled now, not at its next retry.
        if let Some(held) =
            waiting.take_if(|held| held.job.generation.cancel.load(Ordering::SeqCst))
        {
            unregister(shared, &held.job);
            cancel_unstarted(held.job);
        }
        // Bound queue polling as well as live sessions. A stream of cancelled submissions
        // must not starve the jobs already decoding.
        for _ in 0..QUEUE_CAPACITY {
            if slots.len() == max_seqs {
                break;
            }
            // A job that capacity refused waits at the head of the line, ahead of the queue. It
            // asks again when a slot has freed since its refusal and once a second otherwise.
            let (job, ticket, fresh) = match waiting.take() {
                // A slot has freed since the refusal: its session is in the pool, or its memory is back.
                Some(held) if slots.len() < held.live => (held.job, held.ticket, false),
                // The timed retry asks the executor first: an allocation that cannot fit would
                // stall the live streams for nothing.
                Some(held)
                    if Instant::now() >= held.retry_at
                        && (!pool.free.is_empty() || executor.can_admit_session()) =>
                {
                    (held.job, held.ticket, false)
                }
                Some(mut held) => {
                    if Instant::now() >= held.retry_at {
                        held.retry_at = Instant::now() + WAIT_RETRY;
                    }
                    waiting = Some(held);
                    break;
                }
                None => {
                    let message = if slots.is_empty() {
                        shared.set_state(EngineState::Ready);
                        // An executor with background work (a disk write-behind) wakes between
                        // jobs to advance it, as in the single-session worker.
                        let background = executor.background_pending();
                        let wait = match (pool.idle_trim, background) {
                            (Some(trim), true) => Some(trim.min(super::BACKGROUND_POLL)),
                            (Some(trim), false) => Some(trim),
                            (None, true) => Some(super::BACKGROUND_POLL),
                            (None, false) => None,
                        };
                        match wait {
                            Some(wait) => match jobs.recv_timeout(wait) {
                                Ok(message) => Some(message),
                                Err(RecvTimeoutError::Timeout) => {
                                    if background {
                                        executor.background_step();
                                    }
                                    pool.trim(Instant::now(), 1);
                                    continue;
                                }
                                Err(RecvTimeoutError::Disconnected) => None,
                            },
                            None => jobs.recv().ok(),
                        }
                    } else {
                        jobs.try_recv().ok()
                    };
                    let Some(message) = message else {
                        if slots.is_empty() {
                            shared.set_state(EngineState::Stopped);
                            return;
                        }
                        break;
                    };
                    let Message::Job(job, ticket) = message else {
                        shared.stop.store(true, Ordering::SeqCst);
                        break;
                    };
                    (job, ticket, true)
                }
            };
            if shared.stop.load(Ordering::SeqCst) || job.generation.cancel.load(Ordering::SeqCst) {
                if !fresh {
                    unregister(shared, &job);
                }
                cancel_unstarted(job);
                drop(ticket);
                continue;
            }
            if fresh {
                shared.set_state(EngineState::Busy);
                send_data(&job.events, JobEvent::Started);
                register(shared, &job);
            }
            // Keep portable handles registered even if session allocation or restore panics.
            let events = job.events.clone();
            let cancel = Arc::clone(&job.generation.cancel);
            let prompt_len = prompt_tokens(&job.prompt);
            match admit(executor, &mut pool, job, prefill_chunk) {
                Ok(slot) => {
                    drop(ticket);
                    slots.push(slot);
                }
                Err((error, job_back)) => {
                    if matches!(error, EngError::Backend { .. }) {
                        native_failed(shared, jobs, &error);
                        return;
                    }
                    if matches!(error, EngError::MemoryPlan(_))
                        && !slots.is_empty()
                        && !cancel.load(Ordering::SeqCst)
                    {
                        // The sessions the live slots hold are what the memory is spent on:
                        // the job waits for one to finish instead of failing. Only a lone job
                        // that cannot be admitted at all is refused.
                        waiting = Some(Waiting {
                            job: *job_back,
                            ticket,
                            live: slots.len(),
                            retry_at: Instant::now() + WAIT_RETRY,
                        });
                        break;
                    }
                    if matches!(error, EngError::MemoryPlan(_)) && cancel.load(Ordering::SeqCst) {
                        send_cancelled(&events, prompt_len);
                    } else {
                        send_terminal(&events, JobEvent::Error(error));
                    }
                    lock(&shared.active).retain(|entry| !Arc::ptr_eq(&entry.cancel, &cancel));
                }
            }
        }
        if shared.stop.load(Ordering::SeqCst) {
            break;
        }
        let mut tick = TickTimer::new(&shared.ticks);
        // A decode row advances one sequence. A prefill row feeds one prompt token.
        // Reserve the ready rows first, then share the remaining rows among cold slots.
        // One row per live slot is the minimum that guarantees progress in both phases.
        let mut prompt_slots = 0;
        for slot in &mut slots {
            slot.ready |= slot.can_start_decode();
            prompt_slots += usize::from(!slot.ready);
        }
        // One decode row per ready slot: the budget counts every non-prefilling
        // slot as exactly one decode row this tick, so decode_rows is the number
        // of ready slots, never a per-slot row count a family could raise.
        let decode_rows = slots.len() - prompt_slots;
        // With no stream decoding nothing waits on the tick, so an executor that opts in
        // (`BatchPolicy::idle_pieces`) has each cold prompt take a whole piece, up to that many
        // prompts per tick, which bounds the wait of a cancellation: a prompt that arrives alone
        // prefills as fast as the single-sequence worker, and several cold prompts take turns at
        // the full piece size. Without the opt-in the cold prompts share the one-piece budget.
        let tick_rows = if decode_rows == 0 {
            iteration_rows * prompt_slots.clamp(1, policy.idle_pieces.max(1))
        } else {
            iteration_rows
        };
        // While streams decode, an executor that sets a prompt share feeds that many rows per
        // tick beside the decode rows, never fewer than one planned piece, in as many pieces as
        // the rows need. Without a share the tick's budget is the iteration's rows less the
        // decode rows: the cold prompts are dealt whole pieces from it, at least one a tick and
        // at most one each, and they take turns.
        let decoding_share = share
            .filter(|_| decode_rows > 0)
            .map(|rows| rows.max(prefill_chunk));
        let mut prompt_rows = match decoding_share {
            Some(rows) => rows.max(prompt_slots),
            None => tick_rows.max(slots.len()) - decode_rows,
        };
        // Pieces are whole: the tick's budget only counts how many each cold slot is dealt, and
        // the slots that get none this tick take the next turn.
        let piece = executor
            .prefill_piece()
            .unwrap_or(prefill_chunk)
            .min(prefill_chunk);
        let grants = if prompt_slots == 0 {
            Vec::new()
        } else {
            piece_grants(
                prompt_slots,
                prompt_rows,
                piece,
                turn,
                decoding_share.is_some(),
            )
        };
        let mut cold = 0;
        if prompt_slots > 0 {
            turn = turn.wrapping_add(1);
        }
        let mut index = 0;
        while index < slots.len() {
            if !slots[index].ready {
                // A shutdown is seen between pieces; a cancelled slot is seen before its own.
                if shared.stop.load(Ordering::SeqCst) {
                    break;
                }
                let pieces = grants[cold];
                cold += 1;
                if remove_cancelled(&mut slots, index, shared, &mut pool) {
                    continue;
                }
                let cursor = slots[index].cursor;
                let called = Instant::now();
                let result = prefill_pieces(executor, &mut slots[index], prefill_chunk, pieces);
                tick.executor_call(called);
                if result.is_ok() {
                    TickCounters::add(
                        &shared.ticks.prefill_rows,
                        (slots[index].cursor - cursor) as u64,
                    );
                }
                // A denied call can already have fed rows. Charge its grant and retry
                // on a later tick. Successful short pieces return unused rows immediately.
                prompt_rows = prompt_rows.saturating_sub(if result.is_ok() {
                    slots[index].cursor - cursor
                } else {
                    pieces * piece
                });
                if let Err(error) = result {
                    if matches!(error, EngError::Backend { .. }) {
                        native_failed(shared, jobs, &error);
                        return;
                    }
                    if let EngError::MemoryPlan(_) = error {
                        // The eviction restart in the batched path: evict the LRU prefixes once
                        // and rebuild the slot from a reset session - re-feeding the denied
                        // piece would corrupt it, because the denial left it partially fed.
                        // The eviction is INCREMENTAL: a retried slot's later denial evicts
                        // one more entry and restarts again - the first denial freed one
                        // entry, the next may need only one more - until an admission fits
                        // or no entry remains; the refusal then reaches the client as the
                        // over-length form only when the pool's ceiling honestly explains
                        // it. The eviction protects THIS slot's restored anchor: the denial
                        // belongs to it, not to whichever conversation restored last. The
                        // restart saves nothing (retried). Recovery after EXHAUSTION is
                        // once-only: the second denial that no eviction relieved
                        // classifies, so a persistent denial the ceiling does not explain
                        // can never restart forever. Only this slot leaves; the batch's
                        // other slots keep decoding. A retry-time native failure fails the
                        // engine exactly like a first-attempt one.
                        slots[index].retried = true;
                        // Size the eviction before the ONE restart, mirroring the single
                        // worker's pre-pass: evict while the pool's honest ceiling still
                        // cannot admit this whole prompt, so a k-entry deficit costs ONE
                        // restart - not k partial prefills, each re-feeding every piece
                        // since the restore point (multi-agent traffic runs this batch
                        // path: at ~31K tokens of re-prefill each, k restarts cost more
                        // than one clear-all from k ~ 3). The loop's first `None` is the
                        // terminal check: no restart can ever admit, so the refusal
                        // classifies now. A `None` ceiling sizes nothing: stop after ONE
                        // eviction and let the residual one-at-a-time loop below spend more
                        // as denials recur - never a clear-all.
                        let mut terminal = false;
                        // A store already exhausted for THIS slot whose no-save restart did
                        // not admit the piece would spin forever on a persistent denial the
                        // ceiling does not explain (a save charge, an invariant, a growth
                        // overflow): the restart re-enters this branch, the empty store's
                        // eviction returns None again, and the fitting ceiling defers
                        // classification. Exhausted TWICE = final: classify now,
                        // `classified` passing the raw MemoryPlan through when the ceiling
                        // explains nothing - the single worker's fallback shape.
                        let exhausted_once = slots[index].exhausted_restart;
                        loop {
                            let slot = &slots[index];
                            if !evict_for_retry(
                                executor,
                                Some(&slot.session),
                                &slot.job.prompt,
                                &slot.job.generation.cancel,
                            ) {
                                terminal = true;
                                break;
                            }
                            // A successful eviction is progress this slot has not been
                            // forgiven for yet: the recovery bound resets, so a
                            // contention run that keeps freeing entries keeps its sized
                            // restarts, one per freed entry - never a clear-all.
                            slots[index].exhausted_restart = false;
                            let Some(ceiling) = executor.pool_prompt_ceiling(&slots[index].session)
                            else {
                                break;
                            };
                            if ceiling >= u64::from(prompt_tokens(&slots[index].job.prompt)) {
                                break;
                            }
                        }
                        if terminal {
                            if remove_cancelled(&mut slots, index, shared, &mut pool) {
                                continue;
                            }
                            // Classify only when the ceiling PROVES the piece cannot fit
                            // (the honest over-length form); an exhausted store whose
                            // denial was a save charge with room to grow still takes the
                            // ONE no-save restart below, mirroring the single worker's
                            // gated classify - but only ONCE: a second exhausted denial is
                            // final even when the ceiling fits, or a persistent denial
                            // restarts forever.
                            let fits_not = executor
                                .pool_prompt_ceiling(&slots[index].session)
                                .is_some_and(|ceiling| {
                                    ceiling < u64::from(prompt_tokens(&slots[index].job.prompt))
                                });
                            if fits_not || exhausted_once {
                                let slot = slots.remove(index);
                                let error = classified(
                                    executor,
                                    &slot.session,
                                    prompt_tokens(&slot.job.prompt),
                                    error,
                                );
                                send_terminal(&slot.job.events, JobEvent::Error(error));
                                unregister(shared, &slot.job);
                                pool.give(slot.session);
                                continue;
                            }
                            // The ONE no-save recovery after exhaustion: record it, so the
                            // next exhausted denial classifies above instead of
                            // restarting again.
                            slots[index].exhausted_restart = true;
                        }
                        let mut retried = setup_slot(executor, &mut slots[index], prefill_chunk);
                        while let Err(EngError::MemoryPlan(_)) = &retried {
                            let slot = &slots[index];
                            if !evict_for_retry(
                                executor,
                                Some(&slot.session),
                                &slot.job.prompt,
                                &slot.job.generation.cancel,
                            ) {
                                break;
                            }
                            retried = setup_slot(executor, &mut slots[index], prefill_chunk);
                        }
                        match retried {
                            Ok(()) => {}
                            Err(native @ EngError::Backend { .. }) => {
                                native_failed(shared, jobs, &native);
                                return;
                            }
                            Err(EngError::MemoryPlan(message)) => {
                                if remove_cancelled(&mut slots, index, shared, &mut pool) {
                                    continue;
                                }
                                let slot = slots.remove(index);
                                let error = classified(
                                    executor,
                                    &slot.session,
                                    prompt_tokens(&slot.job.prompt),
                                    EngError::MemoryPlan(message),
                                );
                                send_terminal(&slot.job.events, JobEvent::Error(error));
                                unregister(shared, &slot.job);
                                pool.give(slot.session);
                                continue;
                            }
                            Err(other) => {
                                let slot = slots.remove(index);
                                send_terminal(&slot.job.events, JobEvent::Error(other));
                                unregister(shared, &slot.job);
                                pool.give(slot.session);
                                continue;
                            }
                        }
                        // Restarted cleanly: the next tick runs its first prefill piece, so
                        // the loop's one-piece-per-slot-per-tick rule keeps holding.
                    } else {
                        let slot = slots.remove(index);
                        send_terminal(&slot.job.events, JobEvent::Error(error));
                        unregister(shared, &slot.job);
                        pool.give(slot.session);
                        continue;
                    }
                }
            }
            index += 1;
        }
        if shared.stop.load(Ordering::SeqCst) {
            break;
        }
        // A prompt that finished this tick can join the same window if a row remains.
        for slot in &mut slots {
            if prompt_rows == 0 {
                break;
            }
            if !slot.ready && slot.can_start_decode() {
                slot.ready = true;
                prompt_rows -= 1;
            }
        }
        let count = slots.iter().filter(|slot| slot.ready).count();
        TickCounters::add(&shared.ticks.ticks, 1);
        if count == 0 {
            continue;
        }
        TickCounters::add(&shared.ticks.windows, 1);
        TickCounters::add(&shared.ticks.decode_rows, count as u64);
        shared
            .ticks
            .max_window
            .fetch_max(count as u64, Ordering::Relaxed);
        let called = Instant::now();
        let batches = {
            let (mut sessions, mut generations): (Vec<_>, Vec<_>) = slots
                .iter_mut()
                .filter(|slot| slot.ready)
                .map(|slot| (&mut slot.session, &mut slot.job.generation))
                .unzip();
            executor.decode_window(&mut sessions, &mut generations)
        };
        tick.executor_call(called);
        let batches = match batches {
            Ok(batches) if batches.len() == count => batches,
            Ok(_) => {
                native_failed(
                    shared,
                    jobs,
                    &engine_error("decode window result count mismatch"),
                );
                return;
            }
            Err(error) => {
                if matches!(error, EngError::Backend { .. }) {
                    native_failed(shared, jobs, &error);
                    return;
                }
                window_failed(&mut slots, shared, &error);
                continue;
            }
        };
        // Even a row-scoped native failure poisons the window before any token is published.
        if let Some(error) = batches
            .iter()
            .filter_map(|row| row.as_ref().err())
            .find(|error| matches!(error, EngError::Backend { .. }))
        {
            native_failed(shared, jobs, error);
            return;
        }
        // A row whose growth was denied while another row of the window advanced waits: the
        // memory may come back when a stream finishes. A row denied with no row advancing is
        // refused, because nothing live can free memory for it.
        let any_advanced = batches.iter().any(|row| row.is_ok());
        let mut batches = batches.into_iter();
        let mut emitted = false;
        // A row the executor refused before its feed leaves with its error; its session is
        // clean and returns to the pool.
        let refused: Vec<Slot<E::Session>> = slots
            .extract_if(.., |slot| {
                if !slot.ready {
                    return false;
                }
                let batch = match batches.next().expect("row count checked") {
                    Ok(batch) => batch,
                    Err(EngError::MemoryPlan(_)) if any_advanced && !slot.slow => return false,
                    Err(error) => {
                        if !slot.slow {
                            send_terminal(&slot.job.events, JobEvent::Error(error));
                        }
                        unregister(shared, &slot.job);
                        return true;
                    }
                };
                emitted = true;
                slot.usage.completion_tokens = slot
                    .usage
                    .completion_tokens
                    .saturating_add(u32::try_from(batch.tokens.len()).unwrap_or(u32::MAX));
                // The sampler's first-draw latch is the request's first decode position; the
                // served margin rides the slot usage from the window that drew it.
                slot.usage.first_token_margin = slot.job.generation.sampler.first_top2_margin();
                slot.finished = batch.finish.is_some();
                if !send_data(&slot.job.events, JobEvent::Batch(batch)) {
                    slot.job.generation.cancel.store(true, Ordering::SeqCst);
                    slot.slow = true;
                    // A slow client gets no terminal event, including during failure cleanup.
                    unregister(shared, &slot.job);
                }
                false
            })
            .collect();
        for slot in refused {
            pool.give(slot.session);
        }
        if !emitted {
            continue;
        }
        // Complete deferred work once per emitted window, before dropping finished rows.
        let called = Instant::now();
        let emitted_result = executor.after_emit();
        tick.executor_call(called);
        if let Err(error) = emitted_result {
            if matches!(error, EngError::Backend { .. }) {
                native_failed(shared, jobs, &error);
                return;
            }
            window_failed(&mut slots, shared, &error);
            continue;
        }
        let done: Vec<Slot<E::Session>> = slots
            .extract_if(.., |slot| {
                if slot.slow || slot.finished {
                    if !slot.slow {
                        send_terminal(&slot.job.events, JobEvent::Finished(slot.usage.clone()));
                    }
                    unregister(shared, &slot.job);
                    true
                } else {
                    false
                }
            })
            .collect();
        for slot in done {
            pool.give(slot.session);
        }
    }
    // Shutdown needs no further executor calls.
    if let Some(held) = waiting {
        unregister(shared, &held.job);
        cancel_unstarted(held.job);
    }
    for mut slot in slots {
        cancelled(&mut slot);
        unregister(shared, &slot.job);
    }
    shared.set_state(EngineState::Stopped);
    drain(jobs, cancel_unstarted);
}

#[cfg(test)]
#[path = "batch_worker_tests.rs"]
mod tests;
