//! One token's command structure on Metal: a frame.
//!
//! A [`Frame`] encodes each dispatch into one open command buffer, commits that buffer when it
//! holds `DISPATCHES_PER_BUFFER` dispatches, keeps one committed command buffer ahead of the
//! GPU, and waits once, in [`Frame::drain`]. The walk of a token encodes its device calls into
//! a frame and drains it at the end.
//!
//! Host access. The host may touch a buffer only while no command buffer that binds it is in
//! flight. A launch waits, so it keeps the rule. A frame does not wait, so the context keeps
//! the buffers each committed command buffer binds, and a host read or write waits for them
//! first. A kernel with host-side scratch between two dispatches, as the non-finite flag of
//! the dense matvec, needs a scratch a command buffer before it moves into a frame.

use std::{ffi::c_void, rc::Rc};

use crate::EngError;

use super::context::{Grid, Shared, launch_failure};
use super::guard::DispatchGuard;
use super::native::{self, Binding, RawCommand, RawKernel};
use super::{TERMINAL, ensure_usable};

/// The dispatches one command buffer of a frame holds before it is committed.
pub(super) const DISPATCHES_PER_BUFFER: usize = 64;

/// The command buffer being filled, with the buffers bound so far, each once.
struct Open {
    command: RawCommand,
    dispatches: usize,
    bound: Vec<*mut c_void>,
}

pub(super) struct Frame {
    shared: Rc<Shared>,
    /// The dispatches one command buffer holds before its commit.
    per_buffer: usize,
    open: Option<Open>,
}

impl Frame {
    pub(super) fn new(shared: Rc<Shared>, per_buffer: usize) -> Self {
        debug_assert!(
            per_buffer >= 1,
            "a command buffer holds at least one dispatch"
        );
        Self {
            shared,
            per_buffer,
            open: None,
        }
    }

    /// Encodes `repeats` dispatches of `kernel` back to back into the open command buffer,
    /// and commits that buffer when it is full. `bytes` are the kernel's parameters, bound
    /// after `bindings`.
    pub(super) fn encode(
        &mut self,
        kernel: &RawKernel,
        bindings: &[Binding],
        bytes: &[u8],
        grid: Grid,
        per_threadgroup: Grid,
        repeats: usize,
    ) -> Result<(), EngError> {
        ensure_usable(&TERMINAL)?;
        let _guard = DispatchGuard::enter(&TERMINAL);
        if self.open.is_none() {
            let command = native::command_new(self.shared.runtime()).map_err(launch_failure)?;
            self.open = Some(Open {
                command,
                dispatches: 0,
                bound: Vec::new(),
            });
        }
        let open = self.open.as_mut().expect("opened above");
        native::dispatch(
            &open.command,
            kernel,
            bindings,
            bytes,
            grid,
            per_threadgroup,
            repeats,
        )
        .map_err(launch_failure)?;
        for pointer in bindings.iter().map(|binding| binding.pointer) {
            if !open.bound.contains(&pointer) {
                open.bound.push(pointer);
            }
        }
        open.dispatches += repeats;
        if open.dispatches >= self.per_buffer {
            let full = self.open.take().expect("open");
            self.shared.commit_ahead(full.command, full.bound)?;
        }
        Ok(())
    }

    /// Commits the open command buffer, then waits for every command buffer of the frame: their
    /// GPU time in nanoseconds, 0 when the device reports none or the frame encoded nothing.
    pub(super) fn drain(&mut self) -> Result<u64, EngError> {
        if let Some(open) = self.open.take() {
            self.shared.commit_ahead(open.command, open.bound)?;
        }
        self.shared.wait_all()?;
        Ok(self.shared.take_settled())
    }

    /// The command buffers committed and not waited for.
    #[cfg(test)]
    pub(super) fn in_flight(&self) -> usize {
        self.shared.in_flight_count()
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        // Unwinding or a terminal backend: no native call, and the process exit reclaims the
        // handles.
        if std::thread::panicking() || TERMINAL.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        // The work a frame encoded is not lost with it.
        let _ = self.drain();
    }
}

#[cfg(test)]
mod tests;
