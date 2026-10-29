//! A generic device arena that hands out only checked views, and the hold that
//! keeps an arena from being replaced.
//!
//! A [`DeviceArena`] is ONE device allocation, planned once from a list of
//! [`ArenaTensor`]s by [`plan_arena`]. Invariants:
//!
//! - It never grows. A call that needs more builds a new arena (and, for a
//!   captured graph, new graphs); nothing here moves or resizes an allocation.
//! - Every view is a checked [`DeviceRange`]: inside the allocation and, for
//!   [`DeviceArena::view`], a multiple of `ARENA_ALIGN` bytes from its start, so
//!   it is aligned for every kernel (the allocation base is 256-byte aligned).
//!   [`DeviceArena::tensor`] returns the planned range of one tensor.
//! - An [`ArenaHold`] is a runtime token, not a borrow: the holder (a graph that
//!   lives as long as the resident step) cannot keep a Rust borrow of the owner,
//!   which sits behind a `RefCell`. While a hold lives, `reserve_workspace`
//!   refuses to replace the allocation (see `workspace`). The counter is `Rc`
//!   based, so holds are `!Send` like the context.
//! - Dropping the arena drops its `DeviceBuffer`, which follows the terminal-release
//!   rule (after a terminal failure nothing is freed). The arena has no `Drop` of its
//!   own and does not look at the hold count: the holder keeps the arena alive by
//!   borrowing it, so the step cannot drop an arena that a graph still uses. The
//!   counter protects the one path that replaces an allocation behind a borrow.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the resident decode step")
)]

use std::{cell::Cell, rc::Rc};

use super::{CudaContext, DeviceBuffer, DeviceRange};
use crate::EngError;
use crate::kernels::cuda::arena::{ARENA_ALIGN, ArenaPlan, ArenaTensor, plan_arena};

/// Counts the live holds of one arena or workspace.
#[derive(Default)]
pub(super) struct HoldCount(Rc<Cell<usize>>);

impl HoldCount {
    /// Takes one hold; it counts until the returned token is dropped.
    pub(super) fn take(&self) -> ArenaHold {
        self.0.set(self.0.get().saturating_add(1));
        ArenaHold {
            count: Rc::clone(&self.0),
        }
    }

    /// Holds taken and not yet dropped.
    pub(super) fn active(&self) -> usize {
        self.0.get()
    }
}

/// While a hold lives, the arena it came from is not replaced, grown or freed by
/// the context ("retire, never free"). It is not `Clone`: every hold is taken
/// from the owner and counted.
pub(crate) struct ArenaHold {
    count: Rc<Cell<usize>>,
}

impl Drop for ArenaHold {
    fn drop(&mut self) {
        self.count.set(self.count.get().saturating_sub(1));
    }
}

/// One device allocation that holds a planned set of tensors.
pub(crate) struct DeviceArena {
    buffer: DeviceBuffer,
    plan: ArenaPlan,
    tensors: Vec<ArenaTensor>,
    holds: HoldCount,
}

impl CudaContext {
    /// Plans `tensors` with [`plan_arena`] and allocates the arena that holds them.
    ///
    /// Returns `BadRequest` for no tensors, a tensor of 0 bytes or one that ends
    /// before it starts, and `MemoryPlan` when the planned bytes overflow or the
    /// device cannot hold them (nothing is kept then).
    pub(crate) fn arena(&self, tensors: &[ArenaTensor]) -> Result<DeviceArena, EngError> {
        self.ensure_usable()?;
        if tensors.is_empty() {
            return Err(EngError::BadRequest("cuda arena of no tensors".to_owned()));
        }
        let plan = plan_arena(tensors)?;
        let buffer = self.alloc(plan.total)?;
        Ok(DeviceArena {
            buffer,
            plan,
            tensors: tensors.to_vec(),
            holds: HoldCount::default(),
        })
    }
}

impl DeviceArena {
    /// Bytes of the allocation.
    pub(crate) fn bytes(&self) -> usize {
        self.buffer.bytes()
    }

    /// Tensors the arena was planned for.
    pub(crate) fn len(&self) -> usize {
        self.tensors.len()
    }

    /// The planned range of tensor `index`, in the order given to [`CudaContext::arena`].
    ///
    /// Returns `BadRequest` for an index past the last tensor.
    pub(crate) fn tensor(&self, index: usize) -> Result<DeviceRange<'_>, EngError> {
        let (Some(&offset), Some(tensor)) = (self.plan.offsets.get(index), self.tensors.get(index))
        else {
            return Err(EngError::BadRequest(format!(
                "the CUDA arena has {} tensors, not a tensor {index}",
                self.len()
            )));
        };
        self.buffer.range(offset, tensor.bytes)
    }

    /// The `bytes` bytes of the arena from `offset` on.
    ///
    /// Returns `BadRequest` unless `offset` is a multiple of `ARENA_ALIGN`,
    /// `bytes` is not 0 and the range ends inside the arena (an offset sum that
    /// overflows is outside).
    pub(crate) fn view(&self, offset: usize, bytes: usize) -> Result<DeviceRange<'_>, EngError> {
        if !offset.is_multiple_of(ARENA_ALIGN) {
            return Err(EngError::BadRequest(format!(
                "the CUDA arena view starts at offset {offset}, which is not a multiple of {ARENA_ALIGN}"
            )));
        }
        let inside = offset
            .checked_add(bytes)
            .is_some_and(|end| end <= self.plan.total);
        if bytes == 0 || !inside {
            return Err(EngError::BadRequest(format!(
                "the CUDA arena view of {bytes} bytes at offset {offset} does not fit an arena of {} bytes",
                self.plan.total
            )));
        }
        self.buffer.range(offset, bytes)
    }

    /// Takes a hold on this arena.
    pub(crate) fn hold(&self) -> ArenaHold {
        self.holds.take()
    }

    /// Holds of this arena that are still alive.
    pub(crate) fn holds(&self) -> usize {
        self.holds.active()
    }

    /// The allocation, for `read_at` and `write_at`.
    pub(crate) fn buffer(&self) -> &DeviceBuffer {
        &self.buffer
    }
}

#[cfg(test)]
mod tests;
