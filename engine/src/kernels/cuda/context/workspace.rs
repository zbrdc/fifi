//! The activation workspace of a CUDA context (CUDA-RD section 2, item 7).
//!
//! The tiled matmul reads the fp16 buffer `xf`, which it packs from the f32
//! input `x`, and writes the f32 output `y`. A [`Workspace`] keeps these three
//! device regions in ONE allocation at the offsets of [`place`], so a linear
//! call allocates and frees nothing (P6-M1: the three `cudaMalloc`/`cudaFree`
//! pairs were 58 percent of a token). It is the seed of the resident decode's
//! static arena; RD-2 adds streams to this owner.
//!
//! Size. A context allocates its workspace when it is created, with the sizes
//! of [`default_caps`], so its live-byte count is fixed from its first line (the
//! projector's leak tests compare that count before and after a load). A call
//! that needs more in any region grows the workspace ([`CudaContext::
//! reserve_workspace`]): each region becomes the larger of its old and new size
//! and one new allocation replaces the old one. The workspace never shrinks. A
//! failed grow is `MemoryPlan` and leaves the old workspace in place.
//!
//! Rule for RD-11, "retire, never free": a buffer whose address a captured
//! graph holds is never freed, moved or resized while that graph lives. There is
//! no graph yet, so a grow frees the old allocation at once (it holds old plus
//! new bytes for a moment). When graphs exist, `reserve_workspace` must move the
//! old allocation to a retired list that is dropped with the last graph that
//! refers to it, and a grow must invalidate those graphs.
//!
//! Serialization. A context is `!Send` and `!Sync`, so one thread owns its
//! workspace. `linear_rows` enters the process dispatch guard first (decision
//! 0023), then borrows the workspace for the length of one call, so the native
//! call, a grow and the free of the old allocation are serialized with the
//! native calls of every other context. Nothing between the borrow and its
//! release calls back into the context; a nested borrow is still a typed error.

use std::rc::Rc;

use super::{CudaContext, DeviceBuffer, Lease};
use crate::EngError;
use crate::kernels::cuda::arena::{ArenaTensor, plan_arena};
use crate::kernels::cuda::native::{self, Activations, RawDevice, Region, Work};
use crate::kernels::cuda::{ADAPTER_OPERATION, TERMINAL, native_error};

/// Tokens of the largest call that the default workspace holds without growing:
/// the smallest row-major call (one more than the 16 tokens of the fragment-order
/// buffer), used by the parity and headroom tests.
const SERVED_TOKENS: usize = 17;
/// Longest input row of the default workspace: the surveyed Q8_0 shapes reach 5120.
const SERVED_K: usize = 5120;
/// Most output rows of the default workspace: `output.weight` of
/// Qwen3.6-35B-A3B has 248,320.
const SERVED_ROWS: usize = 248_320;

/// Byte ranges of the three regions inside the workspace allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Placement {
    pub(super) x: Region,
    pub(super) xf: Region,
    pub(super) y: Region,
    pub(super) total: usize,
}

/// One device allocation that holds the x, xf and y regions of a linear call.
pub(super) struct Workspace {
    pub(super) buffer: DeviceBuffer,
    pub(super) caps: Activations,
    placement: Placement,
}

/// The sizes of the default workspace: the larger of a one-token call and a
/// [`SERVED_TOKENS`]-token call at the widest served shape.
pub(super) fn default_caps() -> Result<Activations, EngError> {
    let size = |tokens: usize| {
        native::activation_bytes(SERVED_ROWS, SERVED_K, tokens)
            .map_err(|status| native_error(&TERMINAL, ADAPTER_OPERATION, status))
    };
    Ok(size(1)?.union(size(SERVED_TOKENS)?))
}

/// Places x, xf and y with the arena planner. All three are live for the whole
/// call (x is read by the pack, xf by the matmul, y is written by the matmul and
/// copied back after), so they get one lifetime and never share an address.
pub(super) fn place(caps: Activations) -> Result<Placement, EngError> {
    let whole_call = |bytes: usize| ArenaTensor {
        bytes,
        first: 0,
        last: 0,
    };
    let plan = plan_arena(&[whole_call(caps.x), whole_call(caps.xf), whole_call(caps.y)])?;
    let region = |index: usize, bytes: usize| Region {
        offset: plan.offsets[index],
        bytes,
    };
    Ok(Placement {
        x: region(0, caps.x),
        xf: region(1, caps.xf),
        y: region(2, caps.y),
        total: plan.total,
    })
}

impl Workspace {
    /// Allocates one workspace that holds `caps`, on the adapter of `lease`.
    pub(super) fn allocate(lease: &Rc<Lease>, caps: Activations) -> Result<Self, EngError> {
        let placement = place(caps)?;
        let buffer = DeviceBuffer::allocate(lease, placement.total)?;
        Ok(Self {
            buffer,
            caps,
            placement,
        })
    }

    /// The regions as the native call takes them, inside the allocation `device`.
    pub(super) fn work<'a>(&self, device: &'a RawDevice) -> Work<'a> {
        Work {
            device,
            x: self.placement.x,
            xf: self.placement.xf,
            y: self.placement.y,
        }
    }
}

impl CudaContext {
    /// Grows `workspace` until it holds `need`; a no-op when it already does.
    ///
    /// The new allocation is made first, so a failure (`MemoryPlan`) leaves the
    /// old workspace and its sizes untouched. Callers hold the dispatch guard.
    pub(super) fn reserve_workspace(
        &self,
        workspace: &mut Workspace,
        need: Activations,
    ) -> Result<(), EngError> {
        if workspace.caps.covers(need) {
            return Ok(());
        }
        let next = Workspace::allocate(&self.lease, workspace.caps.union(need))?;
        #[cfg(test)]
        self.workspace_allocations
            .set(self.workspace_allocations.get() + 1);
        // No graph holds a pointer into the old allocation yet, so it is freed
        // here. RD-11: retire it instead (see the module doc).
        *workspace = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
