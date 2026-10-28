//! Non-blocking streams and timing-free events of a CUDA context (CUDA-RD
//! section 6, decision 0047).
//!
//! A [`DeviceStream`] or [`DeviceEvent`] owns one native handle and shares the
//! lease of the context that made it, like a [`super::DeviceBuffer`]. Invariants:
//!
//! - Both are `!Send` and `!Sync` (`PhantomData<*const ()>`), so the creating
//!   thread is the only one that calls them (decisions 0023 and 0024).
//! - Every method reports a terminal backend first, then checks its arguments
//!   (an owner of another adapter is a `BadRequest`), then enters the dispatch
//!   guard and makes the one native call. The raw calls stay in
//!   `native::resident`; this module has none.
//! - Drop follows decision 0022 ([`release`]): after a terminal failure it makes
//!   no native call and leaves the owner counted; a failed destroy latches the
//!   backend. The native destroy of a stream synchronizes it first. Drop never
//!   panics.
//! - Inside a capture window ([`super::graph`]) only `record`, `wait`, and the section marks'
//!   `record_timing` with its `is_capturing` query (packet B1) of this module run:
//!   creating a stream or event and both synchronizes are refused with `BadRequest`, and a drop is
//!   deferred. An event records whether its last record was made inside a window (`captured`); a
//!   host wait on such an event is refused before any native call, because CUDA answers it with
//!   an unlisted CUDA error (status 1 on the 12.2 runtime; 907 is the documented one), which would
//!   latch the backend.
//! - Each stream counts the operations enqueued through the context. An event
//!   remembers the count at its last record, which lets [`super::pinned`]
//!   refuse to read memory before the event that follows its copy.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for RD-9a (decision 0047)")
)]

use std::{
    cell::Cell,
    marker::PhantomData,
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

#[cfg(test)]
use super::DESTROY_CALLS;
use super::graph::{self, ensure_closed, run_or_defer};
use super::{CudaContext, LIVE_EVENTS, LIVE_STREAMS, STREAM_IDS, device_error, enter_dispatch};
use crate::EngError;
use crate::kernels::cuda::guard::{DispatchGuard, Lease};
use crate::kernels::cuda::native::DeviceFailure;
use crate::kernels::cuda::native::resident::{self, RawEvent, RawStream};
use crate::kernels::cuda::{TERMINAL, ensure_usable};

/// One non-blocking CUDA stream owned by a [`CudaContext`].
pub(crate) struct DeviceStream {
    raw: Option<RawStream>,
    lease: Rc<Lease>,
    id: u64,
    /// Operations enqueued on this stream through the context so far.
    enqueued: Cell<u64>,
    _thread_confined: PhantomData<*const ()>,
}

/// One CUDA event with timing disabled, owned by a [`CudaContext`].
pub(crate) struct DeviceEvent {
    raw: Option<RawEvent>,
    lease: Rc<Lease>,
    /// Stream id and enqueue count at the last record, or `None` before one.
    mark: Cell<Option<(u64, u64)>>,
    /// Whether the last record was made inside a capture window, so the event has not run.
    captured: Cell<bool>,
    _thread_confined: PhantomData<*const ()>,
}

impl CudaContext {
    /// Creates a non-blocking stream.
    pub(crate) fn stream(&self) -> Result<DeviceStream, EngError> {
        self.ensure_usable()?;
        ensure_closed("a stream creation")?;
        let _dispatch = self.enter()?;
        let raw = resident::stream_create().map_err(device_error)?;
        LIVE_STREAMS.fetch_add(1, Ordering::AcqRel);
        Ok(DeviceStream {
            raw: Some(raw),
            lease: Rc::clone(&self.lease),
            id: STREAM_IDS.fetch_add(1, Ordering::AcqRel) + 1,
            enqueued: Cell::new(0),
            _thread_confined: PhantomData,
        })
    }

    /// Creates an event with timing disabled.
    pub(crate) fn event(&self) -> Result<DeviceEvent, EngError> {
        self.ensure_usable()?;
        ensure_closed("an event creation")?;
        let _dispatch = self.enter()?;
        let raw = resident::event_create().map_err(device_error)?;
        LIVE_EVENTS.fetch_add(1, Ordering::AcqRel);
        Ok(DeviceEvent {
            raw: Some(raw),
            lease: Rc::clone(&self.lease),
            mark: Cell::new(None),
            captured: Cell::new(false),
            _thread_confined: PhantomData,
        })
    }
}

impl DeviceStream {
    /// Records `event` behind the work queued on this stream so far. A later
    /// record of the same event replaces the earlier one.
    ///
    /// Returns `BadRequest` for an event of another adapter.
    pub(crate) fn record(&self, event: &DeviceEvent) -> Result<(), EngError> {
        ensure_usable(&TERMINAL)?;
        same_adapter(&self.lease, &event.lease, "event")?;
        let (stream_raw, event_raw) = (self.raw()?, event.raw()?);
        let _dispatch = enter_dispatch()?;
        resident::event_record(event_raw, stream_raw).map_err(device_error)?;
        event.mark.set(Some((self.id, self.enqueued.get())));
        event.captured.set(graph::window_open());
        self.note_enqueue();
        Ok(())
    }

    /// Makes the work queued on this stream later wait for the last record of
    /// `event` (fork and join).
    ///
    /// Returns `BadRequest` for an event of another adapter and for an event that
    /// was never recorded.
    pub(crate) fn wait(&self, event: &DeviceEvent) -> Result<(), EngError> {
        ensure_usable(&TERMINAL)?;
        same_adapter(&self.lease, &event.lease, "event")?;
        // CUDA treats a wait on an event that was never recorded as a silent no-op,
        // so a fork and join built on it would lose its ordering without a sign.
        if event.mark().is_none() {
            return Err(EngError::BadRequest(
                "the CUDA event was never recorded, so a wait on it would order nothing".to_owned(),
            ));
        }
        // Outside a window a wait on a captured event would order behind work that never ran. Inside
        // one, CUDA decides: an event recorded outside the capture is status 905.
        if !graph::window_open() {
            event.ensure_host_waitable()?;
        }
        let (stream_raw, event_raw) = (self.raw()?, event.raw()?);
        let _dispatch = enter_dispatch()?;
        resident::stream_wait_event(stream_raw, event_raw).map_err(device_error)?;
        self.note_enqueue();
        Ok(())
    }

    /// Blocks until all work queued on this stream has finished. A fault in
    /// that work surfaces here and latches the backend.
    pub(crate) fn synchronize(&self) -> Result<(), EngError> {
        ensure_usable(&TERMINAL)?;
        ensure_closed("a stream synchronize")?;
        let raw = self.raw()?;
        let _dispatch = enter_dispatch()?;
        resident::stream_synchronize(raw).map_err(device_error)
    }

    /// Whether the stream is capturing right now. Legal inside a window.
    pub(crate) fn is_capturing(&self) -> Result<bool, EngError> {
        ensure_usable(&TERMINAL)?;
        let raw = self.raw()?;
        let _dispatch = enter_dispatch()?;
        resident::graph::stream_is_capturing(raw).map_err(device_error)
    }

    /// Operations counted on this stream so far. Tests only: the sequence number
    /// that `note_enqueue` hands out is private to the pinned word and events.
    #[cfg(test)]
    pub(crate) fn enqueued(&self) -> u64 {
        self.enqueued.get()
    }

    /// The process-unique id of the stream, from 1.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(super) fn raw(&self) -> Result<&RawStream, EngError> {
        self.raw
            .as_ref()
            .ok_or_else(|| EngError::BadRequest("the CUDA stream was already destroyed".to_owned()))
    }

    /// The lease of the adapter that made the stream.
    pub(super) fn lease(&self) -> &Rc<Lease> {
        &self.lease
    }

    /// Reject exhaustion before DMA; an old saturated event must not cover a new copy.
    pub(super) fn next_enqueue(&self) -> Result<u64, EngError> {
        self.enqueued.get().checked_add(1).ok_or_else(|| {
            EngError::BadRequest("the CUDA stream enqueue sequence is exhausted".to_owned())
        })
    }

    /// Counts one operation enqueued on this stream and returns the new count,
    /// the sequence number of that operation.
    pub(super) fn note_enqueue(&self) -> u64 {
        let next = self.enqueued.get().saturating_add(1);
        self.enqueued.set(next);
        next
    }
}

impl DeviceEvent {
    /// Blocks the host until the last record of this event has completed.
    pub(crate) fn synchronize(&self) -> Result<(), EngError> {
        ensure_usable(&TERMINAL)?;
        ensure_closed("an event synchronize")?;
        self.ensure_host_waitable()?;
        let raw = self.raw()?;
        let _dispatch = enter_dispatch()?;
        resident::event_synchronize(raw).map_err(device_error)
    }

    /// `BadRequest` when the last record was made inside a capture window: that record is a graph
    /// node that has not run, and a host wait (or a wait outside a window) on it is a CUDA error (status 1 on the
    /// 12.2 runtime) that latches the backend.
    fn ensure_host_waitable(&self) -> Result<(), EngError> {
        if self.captured.get() {
            return Err(EngError::BadRequest(
                "the CUDA event was last recorded inside a capture".to_owned(),
            ));
        }
        Ok(())
    }

    /// The stream id and the enqueue count of the last record, or `None` when
    /// the event was never recorded.
    pub(super) fn mark(&self) -> Option<(u64, u64)> {
        self.mark.get()
    }

    /// The lease of the adapter that made the event.
    pub(super) fn lease(&self) -> &Rc<Lease> {
        &self.lease
    }

    fn raw(&self) -> Result<&RawEvent, EngError> {
        self.raw
            .as_ref()
            .ok_or_else(|| EngError::BadRequest("the CUDA event was already destroyed".to_owned()))
    }
}

/// One CUDA timing event of a captured graph's section marks (packet B1). Timing is enabled,
/// so an elapsed query works; there is no ordering bookkeeping, because nothing waits on it.
/// Created before the capture that records it, so its address is stable across replays.
pub(crate) struct TimingEvent {
    raw: Option<RawEvent>,
    lease: Rc<Lease>,
    _thread_confined: PhantomData<*const ()>,
}

impl CudaContext {
    /// Creates a timing-enabled event (packet B1). Refused inside a capture window, like an
    /// ordering event: the timer's events exist before the capture that records them.
    pub(crate) fn event_timing(&self) -> Result<TimingEvent, EngError> {
        self.ensure_usable()?;
        ensure_closed("a timing event creation")?;
        let _dispatch = self.enter()?;
        let raw = resident::event_create_timing().map_err(device_error)?;
        LIVE_EVENTS.fetch_add(1, Ordering::AcqRel);
        Ok(TimingEvent {
            raw: Some(raw),
            lease: Rc::clone(&self.lease),
            _thread_confined: PhantomData,
        })
    }
}

impl DeviceStream {
    /// Records the timing event after the work queued so far, as a mark of the graph timer
    /// (packet B1). Inside a capture window the record carries the external-record flag and
    /// becomes an event-record node of the captured graph, so every replay re-records it;
    /// outside a window (the warm-up token, the eager fallback of a refused capture) a plain
    /// record runs, because `cudaEventRecordExternal` is valid only while capturing (CUDA 401,
    /// found by the device row).
    pub(crate) fn record_timing(&self, event: &TimingEvent) -> Result<(), EngError> {
        ensure_usable(&TERMINAL)?;
        same_adapter(&self.lease, &event.lease, "event")?;
        let (stream_raw, event_raw) = (self.raw()?, event.raw()?);
        let _dispatch = enter_dispatch()?;
        let status = if self.is_capturing()? {
            resident::event_record_external(event_raw, stream_raw)
        } else {
            resident::event_record(event_raw, stream_raw)
        };
        status.map_err(device_error)?;
        self.note_enqueue();
        Ok(())
    }
}

impl TimingEvent {
    /// The milliseconds from `self` (earlier) to `later`, after both records completed. A
    /// query on unfinished or never-recorded events fails (600 not-ready, 400 invalid
    /// handle), and every failure here propagates through `native_error` and LATCHES the
    /// backend terminal — so the step reads its sections only after the token's read-back,
    /// gated by its host-side readiness flag.
    pub(crate) fn elapsed_ms_since(&self, later: &TimingEvent) -> Result<f32, EngError> {
        ensure_usable(&TERMINAL)?;
        same_adapter(&self.lease, &later.lease, "event")?;
        let (start, end) = (self.raw()?, later.raw()?);
        let _dispatch = enter_dispatch()?;
        resident::event_elapsed_ms(start, end).map_err(device_error)
    }

    fn raw(&self) -> Result<&RawEvent, EngError> {
        self.raw.as_ref().ok_or_else(|| {
            EngError::BadRequest("the CUDA timing event was already destroyed".to_owned())
        })
    }
}

impl Drop for TimingEvent {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            run_or_defer(move || release(raw, resident::event_destroy, &LIVE_EVENTS, 1));
        }
    }
}

/// `BadRequest` unless both owners hold the lease of one adapter.
pub(super) fn same_adapter(
    ours: &Rc<Lease>,
    theirs: &Rc<Lease>,
    what: &str,
) -> Result<(), EngError> {
    if Rc::ptr_eq(ours, theirs) {
        return Ok(());
    }
    Err(EngError::BadRequest(format!(
        "the CUDA {what} belongs to another adapter"
    )))
}

/// Destroys one native owner under decision 0022, for `Drop`.
///
/// After a terminal failure nothing is destroyed and the owner stays counted in
/// `live`, because the memory or the handle is left to the process exit. A
/// failed destroy latches the backend and also leaves the count. `count` is
/// the owner's share of `live`.
pub(super) fn release<R>(
    raw: R,
    destroy: fn(R) -> Result<(), DeviceFailure>,
    live: &AtomicUsize,
    count: usize,
) {
    let _dispatch = DispatchGuard::enter(&TERMINAL);
    if TERMINAL.load(Ordering::Acquire) {
        return;
    }
    #[cfg(test)]
    DESTROY_CALLS.fetch_add(1, Ordering::AcqRel);
    match destroy(raw) {
        Ok(()) => {
            live.fetch_sub(count, Ordering::AcqRel);
        }
        Err(_) => TERMINAL.store(true, Ordering::Release),
    }
}

impl Drop for DeviceStream {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            run_or_defer(move || release(raw, resident::stream_destroy, &LIVE_STREAMS, 1));
        }
    }
}

impl Drop for DeviceEvent {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            run_or_defer(move || release(raw, resident::event_destroy, &LIVE_EVENTS, 1));
        }
    }
}

#[cfg(test)]
mod tests;
