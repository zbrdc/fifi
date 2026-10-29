//! Opt-in bounded windows on the existing engine thread. No model or adapter is cloned.
//! A failed window discards its sessions, including any consumed pending tokens.
//! Output uses the job channel's reserved terminal slot and never waits for a receiver.

use super::{
    ActiveJob, EngineState, Message, PromptPlan, Shared, cancel_unstarted, classified, drain,
    engine_error, lock, prompt_tokens, reject, save_points, span,
};
use crate::EngError;
use crate::family::executor::ExecutorOps;
use crate::family::session::SaveKind;
use crate::sched::job::{Job, JobEvent, JobUsage, QUEUE_CAPACITY, send_data, send_terminal};
use crate::tokenizer::{DecodeBatch, DecodeFinish};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Receiver;

struct Slot<S> {
    session: S,
    job: Job,
    cursor: usize,
    points: VecDeque<(usize, SaveKind)>,
    usage: JobUsage,
    ready: bool,
    finished: bool,
    slow: bool,
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

/// Fan out the same error without changing its contract variant.
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
    shared.record_failure(format!("native failure: {error:?}"));
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
                send_terminal(&slot.job.events, JobEvent::Finished(slot.usage));
            }
        }
        Err(error) => send_terminal(&slot.job.events, JobEvent::Error(error)),
    }
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
    executor.reset(&mut slot.session)?;
    let restored = if slot.job.prefix_boundary.is_some() {
        executor.restore_prefix(&mut slot.session, &slot.job.prompt)?
    } else {
        0
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
            prefill_chunk,
            executor.prefix_block_spacing(),
        )
        .into()
    };
    slot.usage.prompt_tokens = prompt_tokens(&slot.job.prompt);
    slot.usage.reused_tokens = u32::try_from(restored).unwrap_or(u32::MAX);
    slot.usage.prefix = prefix;
    slot.cursor = restored;
    slot.ready = false;
    Ok(())
}

/// Builds a slot for `job`, returning the job on every error so the caller can retry or
/// report it.
fn build_slot<E: ExecutorOps>(
    executor: &mut E,
    seed: &mut Option<E::Session>,
    job: Job,
    prefill_chunk: usize,
) -> Result<Slot<E::Session>, (EngError, Box<Job>)> {
    if let Err(error) = executor.check_prompt(job.prompt.len()) {
        return Err((error, Box::new(job)));
    }
    let session = match seed.take() {
        Some(session) => session,
        None => match executor.new_session() {
            Ok(session) => session,
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
        },
        ready: false,
        finished: false,
        slow: false,
        retried: false,
        exhausted_restart: false,
    };
    match setup_slot(executor, &mut slot, prefill_chunk) {
        Ok(()) => Ok(slot),
        Err(error) => Err((error, Box::new(slot.job))),
    }
}

fn admit<E: ExecutorOps>(
    executor: &mut E,
    seed: &mut Option<E::Session>,
    mut job: Job,
    prefill_chunk: usize,
) -> Result<Slot<E::Session>, EngError> {
    match build_slot(executor, seed, job, prefill_chunk) {
        Ok(slot) => Ok(slot),
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
            let protect = executor.restore_candidate(&job_back.prompt);
            // Terminal when nothing can be freed: an empty store cannot admit a rebuild
            // either, so the refusal stands as-is instead of one dead `build_slot`.
            if executor.evict_one_prefix_entry(protect).is_none() {
                return Err(EngError::MemoryPlan(message));
            }
            job = *job_back;
            let mut retried = build_slot(executor, seed, job, prefill_chunk);
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
                        let protect = executor.restore_candidate(&back.prompt);
                        if executor.evict_one_prefix_entry(protect).is_none() {
                            return Err(EngError::MemoryPlan(message));
                        }
                        retried = build_slot(executor, seed, *back, prefill_chunk);
                    }
                    Err((error, _)) => return Err(error),
                }
            }
        }
        Err((error, _)) => Err(error),
    }
}

fn refresh_report<E: ExecutorOps>(executor: &E, slot: &mut Slot<E::Session>) {
    // Refresh storage totals, never the restore facts that another slot may have changed.
    if let (Some(report), Some(now)) = (&mut slot.usage.prefix, executor.prefix_report()) {
        report.entries = now.entries;
        report.bytes = now.bytes;
    }
}

/// One existing-sized prefill piece per slot per tick. Prompt rows never enter decode_window.
fn prefill<E: ExecutorOps>(
    executor: &mut E,
    slot: &mut Slot<E::Session>,
    prefill_chunk: usize,
) -> Result<(), EngError> {
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
    let remaining = span(&slot.job.prompt, slot.cursor, target)?;
    let end = slot.cursor + rows.min(remaining.len());
    executor.prefill(&mut slot.session, span(&slot.job.prompt, slot.cursor, end)?)?;
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
        slot.ready = true;
    }
    Ok(())
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
    max_seqs: usize,
) {
    let mut seed = Some(session);
    let mut slots: Vec<Slot<E::Session>> = Vec::new();
    loop {
        if shared.stop.load(Ordering::SeqCst) {
            break;
        }
        // Bound queue polling as well as live sessions. A stream of cancelled submissions
        // must not starve the jobs already decoding.
        for _ in 0..QUEUE_CAPACITY {
            if slots.len() == max_seqs {
                break;
            }
            let message = if slots.is_empty() {
                shared.set_state(EngineState::Ready);
                jobs.recv().ok()
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
            let Message::Job(job) = message else {
                shared.stop.store(true, Ordering::SeqCst);
                break;
            };
            if shared.stop.load(Ordering::SeqCst) || job.generation.cancel.load(Ordering::SeqCst) {
                cancel_unstarted(job);
                continue;
            }
            shared.set_state(EngineState::Busy);
            register(shared, &job);
            // Keep portable handles registered even if session allocation or restore panics.
            let events = job.events.clone();
            let cancel = Arc::clone(&job.generation.cancel);
            match admit(executor, &mut seed, job, prefill_chunk) {
                Ok(slot) => slots.push(slot),
                Err(error) => {
                    if matches!(error, EngError::Backend { .. }) {
                        native_failed(shared, jobs, &error);
                        return;
                    }
                    send_terminal(&events, JobEvent::Error(error));
                    lock(&shared.active).retain(|entry| !Arc::ptr_eq(&entry.cancel, &cancel));
                }
            }
        }
        if shared.stop.load(Ordering::SeqCst) {
            break;
        }
        let mut index = 0;
        while index < slots.len() {
            if !slots[index].ready {
                if slots[index].job.generation.cancel.load(Ordering::SeqCst) {
                    let mut slot = slots.remove(index);
                    cancelled(&mut slot);
                    unregister(shared, &slot.job);
                    continue;
                }
                if let Err(error) = prefill(executor, &mut slots[index], prefill_chunk) {
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
                            let protect = match executor.restored_anchor_of(&slots[index].session) {
                                Some(id) => Some(id),
                                None => executor.restore_candidate(&slots[index].job.prompt),
                            };
                            if executor.evict_one_prefix_entry(protect).is_none() {
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
                                continue;
                            }
                            // The ONE no-save recovery after exhaustion: record it, so the
                            // next exhausted denial classifies above instead of
                            // restarting again.
                            slots[index].exhausted_restart = true;
                        }
                        let mut retried = setup_slot(executor, &mut slots[index], prefill_chunk);
                        while let Err(EngError::MemoryPlan(_)) = &retried {
                            let protect = match executor.restored_anchor_of(&slots[index].session) {
                                Some(id) => Some(id),
                                None => executor.restore_candidate(&slots[index].job.prompt),
                            };
                            if executor.evict_one_prefix_entry(protect).is_none() {
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
                                let slot = slots.remove(index);
                                let error = classified(
                                    executor,
                                    &slot.session,
                                    prompt_tokens(&slot.job.prompt),
                                    EngError::MemoryPlan(message),
                                );
                                send_terminal(&slot.job.events, JobEvent::Error(error));
                                unregister(shared, &slot.job);
                                continue;
                            }
                            Err(other) => {
                                let slot = slots.remove(index);
                                send_terminal(&slot.job.events, JobEvent::Error(other));
                                unregister(shared, &slot.job);
                                continue;
                            }
                        }
                        // Restarted cleanly: the next tick runs its first prefill piece, so
                        // the loop's one-piece-per-slot-per-tick rule keeps holding.
                    } else {
                        let slot = slots.remove(index);
                        send_terminal(&slot.job.events, JobEvent::Error(error));
                        unregister(shared, &slot.job);
                        continue;
                    }
                }
            }
            index += 1;
        }
        if shared.stop.load(Ordering::SeqCst) {
            break;
        }
        let count = slots.iter().filter(|slot| slot.ready).count();
        if count == 0 {
            continue;
        }
        let batches = {
            let (mut sessions, mut generations): (Vec<_>, Vec<_>) = slots
                .iter_mut()
                .filter(|slot| slot.ready)
                .map(|slot| (&mut slot.session, &mut slot.job.generation))
                .unzip();
            executor.decode_window(&mut sessions, &mut generations)
        };
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
        let mut batches = batches.into_iter();
        let mut emitted = false;
        slots.retain_mut(|slot| {
            if !slot.ready {
                return true;
            }
            let batch = match batches.next().expect("row count checked") {
                Ok(batch) => batch,
                Err(error) => {
                    if !slot.slow {
                        send_terminal(&slot.job.events, JobEvent::Error(error));
                    }
                    unregister(shared, &slot.job);
                    return false;
                }
            };
            emitted = true;
            slot.usage.completion_tokens = slot
                .usage
                .completion_tokens
                .saturating_add(u32::try_from(batch.tokens.len()).unwrap_or(u32::MAX));
            slot.finished = batch.finish.is_some();
            if !send_data(&slot.job.events, JobEvent::Batch(batch)) {
                slot.job.generation.cancel.store(true, Ordering::SeqCst);
                slot.slow = true;
                // A slow client gets no terminal event, including during failure cleanup.
                unregister(shared, &slot.job);
            }
            true
        });
        if !emitted {
            continue;
        }
        // Complete deferred work once per emitted window, before dropping finished rows.
        if let Err(error) = executor.after_emit() {
            if matches!(error, EngError::Backend { .. }) {
                native_failed(shared, jobs, &error);
                return;
            }
            window_failed(&mut slots, shared, &error);
            continue;
        }
        slots.retain(|slot| {
            if slot.slow || slot.finished {
                if !slot.slow {
                    send_terminal(&slot.job.events, JobEvent::Finished(slot.usage));
                }
                unregister(shared, &slot.job);
                false
            } else {
                true
            }
        });
    }
    // Shutdown needs no further executor calls and does not wait for any output receiver.
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
