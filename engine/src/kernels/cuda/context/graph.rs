//! The capture window and executable graphs of a CUDA context (CUDA-RD rev 2.3 sections 3, 6.5, 6.6
//! and 7, decision 0047).
//!
//! [`CudaContext::capture`] opens a window on one owned stream, runs a body that queues work through
//! the ordinary launch wrappers, closes the window and turns what was recorded into a [`GraphExec`]
//! that replays it. Invariants:
//!
//! - The wrappers are the only launch API. While a window is open they behave as before and only
//!   enqueue through `launch_on`; every call that is not launch-only (synchronous copies,
//!   allocations, frees, synchronizes, the workspace grow, a nested capture, a graph launch) is
//!   refused with a `BadRequest` before any native call ([`ensure_closed`]).
//! - The window lives on one thread (a `thread_local!` flag) and runs under the dispatch guard for
//!   its whole length, in `cudaStreamCaptureModeThreadLocal`. It is not reentrant.
//! - A drop inside a window is deferred ([`run_or_defer`]): the free of a buffer, stream, event,
//!   pinned word or graph synchronizes, which a capture forbids. The owner's own free path runs
//!   when the window closes, in order, and applies decision 0022 itself (return if terminal, latch
//!   on a failed free). After a terminal failure inside the body the deferred list is dropped
//!   without running (the handles are left to the process exit, the counters stay).
//!   Cached buffers retain their backing owner and lease: healthy close frees without
//!   caching; abandonment runs its terminal Drop to record quarantine, with no native call.
//! - The closed list of capture statuses (`mod.rs`, `is_capture_status`) is non-terminal; any other
//!   status latches the backend. A capture that ends with an error leaves the stream out of capture
//!   mode (the C entry checks it and reports an adapter failure otherwise).
//! - A [`GraphExec`] borrows the buffers it was told to pin (`CapturePlan::pins`, a lifetime, not a
//!   check against the body) and owns the [`ArenaHold`]s it was given, so a pinned buffer cannot be
//!   dropped and a held workspace cannot be replaced while the graph lives.
//! - Drop order of a resident step: graphs, then the arena and step state, then the pinned word,
//!   then the events, then the stream; the caller then drops the weights and the context.
//! - Graph update (`cudaGraphExecUpdate`, 12.x form `(exec, graph, &resultInfo)`) is not wrapped: a
//!   changed address means a new arena and new graphs (CUDA-RD 2.6). It would need a second
//!   capture and its own closed-list ruling.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for RD-9a (decision 0047)")
)]

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::Ordering;

use super::stream::{DeviceStream, release, same_adapter};
use super::{ArenaHold, CudaContext, DeviceBuffer, LIVE_GRAPHS, device_error, enter_dispatch};
use crate::EngError;
use crate::kernels::cuda::guard::Lease;
use crate::kernels::cuda::native::resident::RawStream;
use crate::kernels::cuda::native::resident::graph::{self as raw, RawGraphExec};
use crate::kernels::cuda::native::{DeviceFailure, STAGE_NONE, STATUS_INVALID_ARGUMENT};
use crate::kernels::cuda::{
    CAPTURE_OPERATION, CAPTURE_STATUSES, TERMINAL, ensure_usable, stage_operation,
};

type Deferred = Box<dyn FnOnce()>;

/// Whether this thread has a capture window open, and the frees it keeps until the window closes.
struct Window {
    open: bool,
    deferred: Vec<Deferred>,
}

impl Window {
    const fn new() -> Self {
        Self {
            open: false,
            deferred: Vec::new(),
        }
    }
}

thread_local! {
    static WINDOW: RefCell<Window> = const { RefCell::new(Window::new()) };
}

/// Whether a capture window is open on this thread.
pub(super) fn window_open() -> bool {
    WINDOW.with(|window| window.borrow().open)
}

/// Runs `free` now when no window is open; keeps it until the window closes otherwise.
pub(super) fn run_or_defer(free: impl FnOnce() + 'static) {
    let kept = WINDOW.with(|window| {
        let mut window = window.borrow_mut();
        if window.open {
            window.deferred.push(Box::new(free));
            None
        } else {
            Some(free)
        }
    });
    if let Some(free) = kept {
        free();
    }
}

/// `BadRequest` while a window is open: `what` would block or allocate, which a capture forbids.
pub(super) fn ensure_closed(what: &str) -> Result<(), EngError> {
    if window_open() {
        return Err(EngError::BadRequest(format!(
            "cuda capture: {what} is not allowed inside a capture window"
        )));
    }
    Ok(())
}

/// `CAPTURE_OPERATION` for a status 900..=906 inside a window or at the capture stage, else the
/// operation of the failed stage.
pub(super) fn operation_of(failure: &DeviceFailure) -> &'static str {
    if window_open() && CAPTURE_STATUSES.contains(&failure.status) {
        return CAPTURE_OPERATION;
    }
    stage_operation(failure.stage)
}

/// Whether `error` is a capture refusal on the closed list: the non-terminal operation `cuda
/// capture`. A caller that falls back to the eager path after a refused capture records
/// `captured = false` on this answer, without matching the operation string. A `cuda capture`
/// error that latched the backend (an unlisted status at the end of the capture) is not a refusal.
pub(crate) fn is_capture_refusal(error: &EngError) -> bool {
    matches!(
        error,
        EngError::Backend {
            operation: CAPTURE_OPERATION,
            ..
        }
    ) && ensure_usable(&TERMINAL).is_ok()
}

/// Frees kept by the open window, for the tests.
#[cfg(test)]
fn deferred_len() -> usize {
    WINDOW.with(|window| window.borrow().deferred.len())
}

/// Closes the window of this thread on every exit path, including an unwind.
struct WindowScope;

impl WindowScope {
    fn open() -> Self {
        WINDOW.with(|window| window.borrow_mut().open = true);
        Self
    }

    /// Marks the window closed and hands out the frees it kept.
    fn close(&self) -> Vec<Deferred> {
        WINDOW.with(|window| {
            let mut window = window.borrow_mut();
            window.open = false;
            std::mem::take(&mut window.deferred)
        })
    }
}

impl Drop for WindowScope {
    fn drop(&mut self) {
        // Reached with the window still open only by an unwind: a capture of unknown state may
        // still be open, so no native call is made and the backend latches.
        drop(self.close());
        if std::thread::panicking() {
            TERMINAL.store(true, Ordering::Release);
        }
    }
}

/// What a captured graph needs to stay valid.
pub(crate) struct CapturePlan<'p> {
    /// The weights and every buffer whose address the graph bakes in. The graph borrows them for
    /// its whole life. They are not compared with the buffers the body uses; the caller lists all.
    pub(crate) pins: &'p [&'p DeviceBuffer],
    /// Holds on the arenas the graph uses (`workspace_hold`, `DeviceArena::hold`); the graph owns
    /// them until it is dropped.
    pub(crate) holds: Vec<ArenaHold>,
}

/// One executable graph: a captured sequence of launches that replays on a stream.
pub(crate) struct GraphExec<'p> {
    raw: Option<RawGraphExec>,
    lease: Rc<Lease>,
    nodes: usize,
    /// Kept for their `Drop`: while a hold lives the arena it came from is not replaced.
    _holds: Vec<ArenaHold>,
    _pins: PhantomData<&'p [&'p DeviceBuffer]>,
    _thread_confined: PhantomData<*const ()>,
}

/// An open window on one stream, with what closing it needs.
struct OpenWindow<'s> {
    stream: &'s RawStream,
    lease: &'s Rc<Lease>,
    scope: WindowScope,
}

impl CudaContext {
    /// Refuses operations whose captured allocation ownership is not yet enforced.
    pub(crate) fn ensure_uncaptured(&self, what: &str) -> Result<(), EngError> {
        ensure_closed(what)
    }

    /// Runs `body` with a capture window open on `stream` and returns the graph that replays what
    /// the body queued on it (and on the streams it forked), with the body's result.
    ///
    /// The body queues work only through the launch wrappers, `record` and `wait`; any other call
    /// is a `BadRequest` (see the module doc) and a native refusal from the capture API is the
    /// non-terminal operation `cuda capture` when its status is on the closed list. A body that
    /// returns an error ends the capture without a graph and the error is returned unchanged,
    /// unless the end itself latched the backend. A window that cannot be opened is the same
    /// error without a body run.
    ///
    /// A refused capture ([`is_capture_refusal`]) or a failing body leaves the work the body queued
    /// unrun and these side effects: an event recorded in the body stays marked as recorded inside
    /// a capture, so a host wait or a wait outside a window on it is a `BadRequest` until the
    /// caller records it eagerly (an eager record clears the mark); and the enqueue counts of the
    /// streams it used are inflated by the launches that never ran. The mark is not reset here:
    /// the event's last native record is a capture-time record, and a host wait on it is an
    /// unlisted CUDA error (status 1 on the 12.2 runtime) until it is recorded again. A fallback that reruns the body eagerly re-records its
    /// events, which clears both. The caller warms every kernel it captures first: a first launch
    /// loads its module, which a capture forbids.
    pub(crate) fn capture<'p, R>(
        &self,
        stream: &DeviceStream,
        plan: CapturePlan<'p>,
        body: impl FnOnce() -> Result<R, EngError>,
    ) -> Result<(GraphExec<'p>, R), EngError> {
        self.ensure_usable()?;
        same_adapter(&self.lease, stream.lease(), "stream")?;
        ensure_closed("a nested capture")?;
        for pin in plan.pins {
            self.owned(pin)?;
        }
        let stream_raw = stream.raw()?;
        let _dispatch = enter_dispatch()?;
        let open = begin_window(stream_raw, &self.lease)?;
        let outcome = body();
        finish_window(open, plan, outcome)
    }
}

/// Begins the native capture, then marks the window open.
fn begin_window<'s>(
    stream: &'s RawStream,
    lease: &'s Rc<Lease>,
) -> Result<OpenWindow<'s>, EngError> {
    raw::capture_begin(stream).map_err(device_error)?;
    Ok(OpenWindow {
        stream,
        lease,
        scope: WindowScope::open(),
    })
}

/// Ends the capture and closes the window (decision 7 of the packet): an `Ok` body gets a graph; an
/// `Err` body gets its own error back; a latch inside the body makes no native call at all.
fn finish_window<'p, R>(
    open: OpenWindow<'_>,
    plan: CapturePlan<'p>,
    body: Result<R, EngError>,
) -> Result<(GraphExec<'p>, R), EngError> {
    if let Err(terminal) = ensure_usable(&TERMINAL) {
        // Decision 0022: nothing more is touched; the stream is left to the process exit and the
        // deferred frees are dropped unrun.
        drop(open.scope.close());
        return Err(body.err().unwrap_or(terminal));
    }
    let end = raw::capture_end(open.stream, body.is_ok());
    let deferred = open.scope.close();
    let result = match (body, end) {
        (Ok(value), Ok(Some((exec, nodes)))) => Ok((exec, nodes, value)),
        (Ok(_), Ok(None)) => Err(device_error(DeviceFailure {
            status: STATUS_INVALID_ARGUMENT,
            stage: STAGE_NONE,
        })),
        (Err(error), Ok(_)) => Err(error),
        (Ok(_), Err(failure)) => Err(device_error(failure)),
        (Err(error), Err(failure)) => {
            let refusal = device_error(failure);
            Err(if ensure_usable(&TERMINAL).is_err() {
                refusal
            } else {
                error
            })
        }
    };
    let built = result.map(|(exec, nodes, value)| {
        LIVE_GRAPHS.fetch_add(1, Ordering::AcqRel);
        let graph = GraphExec {
            raw: Some(exec),
            lease: Rc::clone(open.lease),
            nodes,
            _holds: plan.holds,
            _pins: PhantomData,
            _thread_confined: PhantomData,
        };
        (graph, value)
    });
    for free in deferred {
        free();
    }
    // A deferred free that failed latched the backend; a graph must not be returned then. No
    // test reaches this: it needs a failed native destroy of a deferred owner.
    built.and_then(|made| ensure_usable(&TERMINAL).map(|()| made))
}

impl GraphExec<'_> {
    /// Queues one replay on `stream`. The replay has not finished when the call returns; a fault
    /// in it surfaces at the next synchronize of the stream and latches the backend there.
    ///
    /// Returns `BadRequest` inside a capture window and for a stream of another adapter.
    pub(crate) fn launch(&self, stream: &DeviceStream) -> Result<(), EngError> {
        ensure_usable(&TERMINAL)?;
        ensure_closed("a graph launch")?;
        same_adapter(&self.lease, stream.lease(), "stream")?;
        let (exec, stream_raw) = (self.raw()?, stream.raw()?);
        let _dispatch = enter_dispatch()?;
        raw::graph_launch(exec, stream_raw).map_err(device_error)?;
        stream.note_enqueue();
        Ok(())
    }

    /// Nodes of the captured graph. Event records and waits are edges, not nodes.
    pub(crate) fn nodes(&self) -> usize {
        self.nodes
    }

    fn raw(&self) -> Result<&RawGraphExec, EngError> {
        self.raw
            .as_ref()
            .ok_or_else(|| EngError::BadRequest("the CUDA graph was already destroyed".to_owned()))
    }
}

impl Drop for GraphExec<'_> {
    fn drop(&mut self) {
        if let Some(exec) = self.raw.take() {
            run_or_defer(move || release(exec, raw::exec_destroy, &LIVE_GRAPHS, 1));
        }
    }
}

#[cfg(test)]
mod tests;
