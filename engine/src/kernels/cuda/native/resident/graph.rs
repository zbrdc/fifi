//! Raw entries of `native/resident/graph.cu`: stream capture, instantiate and launch of executable
//! graphs (RD-2d, decision 0047, child modules of `native.rs`). Design guidance only, no code copied.
//!
//! Safety invariants this file relies on:
//! - A graph handle (`RawGraphExec`) is opaque. It is created once, by `capture_end`, and consumed
//!   once, by `exec_destroy`. It has no `Drop`: the owner in `context::graph` calls the destroy so
//!   that a failed destroy is reported and latches the backend (decision 0022).
//! - A capture is begun and ended on one thread, on a stream that this thread owns, under the
//!   process dispatch guard (decisions 0023 and 0024); `capture_begin` uses thread-local capture
//!   mode, so the other threads of the process are unaffected. The guard of the caller keeps a
//!   window from interleaving with a native call of another thread.
//! - `capture_end` leaves no graph behind except the returned executable one: the captured graph is
//!   destroyed in C on every path, and the entry refuses (an adapter failure) when the stream is
//!   still capturing afterwards.
//! - `graph_launch` only queues the replay and returns; a fault in the replay surfaces at the next
//!   synchronize. The executable graph and everything its nodes address stay alive until the
//!   replays finish because `exec_destroy` and `inference_cuda_free` synchronize the device first.
//! - No raw pointer and no unsafe function leaves this module; handles are opaque.
//!
//! `cudaGraphExecUpdate` (12.x form `(exec, graph, &resultInfo)`) is a possible future option and is
//! not declared here: what changes between replays is buffer content, never an address.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for RD-9a (decision 0047)")
)]

use std::ffi::{c_int, c_void};
use std::ptr::NonNull;

use super::{DeviceFailure, RawStream, STAGE_NONE, checked, invalid_argument};

/// Matches `INFERENCE_CUDA_STAGE_CAPTURE` in `graph.h`: the stage of a failure in begin or end of a
/// capture, or in instantiating it.
pub(in crate::kernels::cuda) const STAGE_CAPTURE: c_int = 5;

/// An executable graph. Consumed by [`exec_destroy`].
#[derive(Debug)]
#[must_use = "an executable graph must be passed to exec_destroy"]
pub(in crate::kernels::cuda) struct RawGraphExec(NonNull<c_void>);

unsafe extern "C" {
    #[link_name = "inference_cuda_capture_begin"]
    fn capture_begin_raw(stream: *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_capture_end"]
    fn capture_end_raw(
        stream: *mut c_void,
        instantiate: c_int,
        out_exec: *mut *mut c_void,
        out_nodes: *mut usize,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_stream_is_capturing"]
    fn stream_is_capturing_raw(
        stream: *mut c_void,
        out_capturing: *mut c_int,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_graph_launch"]
    fn graph_launch_raw(exec: *mut c_void, stream: *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_graph_exec_destroy"]
    fn graph_exec_destroy_raw(exec: *mut c_void, out_stage: *mut c_int) -> c_int;
}

#[cfg(all(test, feature = "cuda-fault-qualification"))]
unsafe extern "C" {
    #[link_name = "inference_cuda_fault_null_write_launch"]
    fn fault_null_write_launch_raw(stream: *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_fault_null_write_warm"]
    fn fault_null_write_warm_raw(stream: *mut c_void, out_stage: *mut c_int) -> c_int;
}

/// Begins a thread-local capture on `stream`. Work queued on the stream is recorded, not run, until
/// [`capture_end`].
pub(in crate::kernels::cuda) fn capture_begin(stream: &RawStream) -> Result<(), DeviceFailure> {
    let mut stage = STAGE_NONE;
    // SAFETY: `stream` is a live handle for the whole call and `stage` is a live, writable c_int.
    let status = unsafe { capture_begin_raw(stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Ends the capture of `stream`. With `instantiate` the captured graph becomes an executable graph
/// and its node count is returned; without it nothing is kept and the result is `Ok(None)`.
/// After an `Ok` or an `Err` that is not an adapter failure the stream is no longer capturing.
pub(in crate::kernels::cuda) fn capture_end(
    stream: &RawStream,
    instantiate: bool,
) -> Result<Option<(RawGraphExec, usize)>, DeviceFailure> {
    let mut exec: *mut c_void = std::ptr::null_mut();
    let mut nodes = 0usize;
    let mut stage = STAGE_NONE;
    // SAFETY: `stream` is a live handle for the whole call; `exec`, `nodes` and `stage` are live,
    // writable values. The adapter writes `exec` only on success and destroys the captured graph
    // itself, so no other handle is left behind.
    let status = unsafe {
        capture_end_raw(
            stream.0.as_ptr(),
            c_int::from(instantiate),
            &mut exec,
            &mut nodes,
            &mut stage,
        )
    };
    checked(status, stage)?;
    if !instantiate {
        return Ok(None);
    }
    let exec = NonNull::new(exec).ok_or_else(invalid_argument)?;
    Ok(Some((RawGraphExec(exec), nodes)))
}

/// Whether `stream` is capturing right now.
pub(in crate::kernels::cuda) fn stream_is_capturing(
    stream: &RawStream,
) -> Result<bool, DeviceFailure> {
    let mut capturing: c_int = 0;
    let mut stage = STAGE_NONE;
    // SAFETY: `stream` is a live handle for the whole call; `capturing` and `stage` are live,
    // writable c_ints.
    let status = unsafe { stream_is_capturing_raw(stream.0.as_ptr(), &mut capturing, &mut stage) };
    checked(status, stage)?;
    Ok(capturing != 0)
}

/// Queues one replay of `exec` on `stream`. The replay has not finished when the call returns.
pub(in crate::kernels::cuda) fn graph_launch(
    exec: &RawGraphExec,
    stream: &RawStream,
) -> Result<(), DeviceFailure> {
    let mut stage = STAGE_NONE;
    // SAFETY: both handles are live for the whole call and `stage` is a live, writable c_int. The
    // call only queues the replay.
    let status = unsafe { graph_launch_raw(exec.0.as_ptr(), stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Waits for the device, then destroys `exec`. When the wait fails nothing is destroyed and the
/// failure is returned.
pub(in crate::kernels::cuda) fn exec_destroy(exec: RawGraphExec) -> Result<(), DeviceFailure> {
    let mut stage = STAGE_NONE;
    // SAFETY: `exec` came from `capture_end` and is consumed here, so it is destroyed at most once
    // and no other handle refers to it afterwards. The adapter synchronizes the device first, so
    // no replay is still running.
    let status = unsafe { graph_exec_destroy_raw(exec.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Queues a kernel that writes through a null pointer: the fault surfaces at the next synchronize.
#[cfg(all(test, feature = "cuda-fault-qualification"))]
pub(in crate::kernels::cuda) fn fault_null_write_launch(
    stream: &RawStream,
) -> Result<(), DeviceFailure> {
    let mut stage = STAGE_NONE;
    // SAFETY: `stream` is a live handle and `stage` a live, writable c_int. The fault happens on
    // the device; the host side touches no memory through the null pointer.
    let status = unsafe { fault_null_write_launch_raw(stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Queues the same kernel with the write disarmed, so its module is loaded before a capture.
#[cfg(all(test, feature = "cuda-fault-qualification"))]
pub(in crate::kernels::cuda) fn fault_null_write_warm(
    stream: &RawStream,
) -> Result<(), DeviceFailure> {
    let mut stage = STAGE_NONE;
    // SAFETY: `stream` is a live handle and `stage` a live, writable c_int; nothing is written.
    let status = unsafe { fault_null_write_warm_raw(stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
