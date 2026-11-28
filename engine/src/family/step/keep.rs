//! Keeping the running state of a step.
//!
//! A cache plane is addressed by position: a sequence that returns to an earlier position
//! writes over the later rows, so the plane needs no copy. The rings and the records of the
//! delta-rule visits hold the whole sequence up to the last token and cannot be cut back. A
//! copy of them at a position, with the cache rows below that position left where they are, is
//! the state of the step at that position.
//!
//! The copies are of whole buffers, in the order of [`State::running`], and run behind the work
//! queued on the step's stream.

use super::state::State;
use crate::EngError;
use crate::family::session::Kept;
use crate::kernels::cuda::DeviceStream;

/// The bytes of one kept state.
pub(crate) fn bytes(state: &State) -> u64 {
    state.running().map(|buffer| buffer.bytes() as u64).sum()
}

/// The running state after `rows` positions, read back once the work queued on `stream` ran.
pub(crate) fn keep(state: &State, stream: &DeviceStream, rows: usize) -> Result<Kept, EngError> {
    let mut parts = Vec::new();
    for buffer in state.running() {
        let mut part = Vec::new();
        part.try_reserve_exact(buffer.bytes()).map_err(|_| {
            EngError::MemoryPlan(format!(
                "the host cannot hold a kept step state of {} bytes",
                buffer.bytes()
            ))
        })?;
        part.resize(buffer.bytes(), 0);
        buffer.read_at(stream, 0, &mut part)?;
        parts.push(part);
    }
    Ok(Kept { rows, parts })
}

/// Writes a kept state back. One of another shape or depth is refused before the first write.
/// Copies enqueue directly from the host's pageable `Vec<u8>` (consumed before return, no host
/// staging allocation); if a write fails mid-way, the running state is zeroed again so a fallback
/// cold start never sees partial bytes.
pub(crate) fn put_back(state: &State, stream: &DeviceStream, kept: &Kept) -> Result<(), EngError> {
    if kept.rows > state.rows() {
        return Err(EngError::BadRequest(format!(
            "a kept step state at {} rows exceeds the step capacity of {}",
            kept.rows,
            state.rows()
        )));
    }
    let sizes = state.running().map(|buffer| buffer.bytes());
    if !sizes.eq(kept.parts.iter().map(Vec::len)) {
        return Err(EngError::BadRequest(
            "a kept step state has another shape than the step".to_owned(),
        ));
    }
    let result = state
        .running()
        .zip(&kept.parts)
        .try_for_each(|(buffer, part)| buffer.write_at_enqueue(stream, 0, part));
    if result.is_err() {
        let _ = reset_running(state, stream);
    }
    result
}

/// Queues the zeros of a new sequence for the running state alone. The cache planes keep their
/// rows: a step reads no row at or above its position.
pub(crate) fn reset_running(state: &State, stream: &DeviceStream) -> Result<(), EngError> {
    state
        .running()
        .try_for_each(|buffer| buffer.zero_at(stream, 0, buffer.bytes()))
}

/// Whether the host has room for one more kept step state of `bytes` now.
pub(crate) fn admit(bytes: usize) -> Result<(), EngError> {
    const WHAT: &str = "a kept step state";
    crate::family::host_store::admit_entry(
        WHAT,
        bytes,
        crate::family::mem::HostFacts::read(WHAT, WHAT),
    )
}
