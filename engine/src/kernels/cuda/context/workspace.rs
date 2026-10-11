//! The activation workspace of a CUDA context (CUDA-RD section 2, item 7).
//!
//! The tiled matmul reads the fp16 buffer `xf`, which it packs from the f32
//! input `x`, and writes the f32 output `y`. A [`Workspace`] keeps these three
//! device regions in ONE allocation at the offsets of [`place`], so a linear
//! call allocates and frees nothing (P6-M1: the three `cudaMalloc`/`cudaFree`
//! pairs were 58 percent of a token). It is the seed of the resident decode's
//! static arena; RD-2 adds streams, holds and admission sizing to this owner.
//!
//! Size. A context allocates its workspace when it is created, with the sizes
//! of its [`Admission`] (the shapes the caller admits; the model's numbers live
//! with the caller, not here), so its live-byte count is fixed from its first
//! line (the projector's leak tests compare that count before and after a load).
//! What a larger call does depends on the admission's [`Growth`]. With
//! `Unbounded`, a call that needs more in any region grows the workspace
//! ([`CudaContext::reserve_workspace`]): each region becomes the larger of its old
//! and new size and one new allocation replaces the old one; the workspace never
//! shrinks, and a failed grow is `MemoryPlan` that leaves the old workspace in
//! place. With `Fixed`, a larger call is `MemoryPlan` and nothing changes: a
//! prefill-sized call is served by a `DeviceArena` made for it, which is freed when
//! it is dropped, so a 2048-token projection never keeps a gigabyte `y` for good.
//!
//! "Retire, never free": a buffer whose address a captured graph holds is never
//! freed, moved or resized while that graph lives. A graph takes an [`ArenaHold`]
//! ([`CudaContext::workspace_hold`]); while any hold lives a grow is refused with
//! `BadRequest` before it allocates, so the old allocation is never freed under a
//! graph. The caller builds a new arena (and new graphs) instead. A grow that no
//! hold prevents frees the old allocation at once (it holds old plus new bytes for
//! a moment); a retired list for growing under graphs is not built (CUDA-RD 2.6).
//!
//! Serialization. A context is `!Send` and `!Sync`, so one thread owns its
//! workspace. `linear_rows` enters the process dispatch guard first (decision
//! 0023), then borrows the workspace for the length of one call, so the native
//! call, a grow and the free of the old allocation are serialized with the
//! native calls of every other context. Nothing between the borrow and its
//! release calls back into the context; a nested borrow is still a typed error.

use std::rc::Rc;

use super::device_arena::{ArenaHold, HoldCount};
use super::graph::ensure_closed;
use super::{CudaContext, DeviceBuffer, Lease, workspace_in_use};
use crate::EngError;
use crate::kernels::cuda::arena::{ArenaTensor, plan_arena};
use crate::kernels::cuda::native::{
    self, Activations, MAX_DIM, RawDevice, Region, TILE_COLS, TILE_ROWS, Work,
};
use crate::kernels::cuda::{ADAPTER_OPERATION, TERMINAL, native_error};

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
    /// Holds on this allocation; a grown workspace starts with none.
    holds: HoldCount,
}

/// How a workspace may change size after it is allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Growth {
    /// A larger call replaces the workspace with a bigger one.
    Unbounded,
    /// A larger call is refused; the workspace keeps its allocation.
    Fixed,
}

/// One projection shape the workspace must hold: `tokens` rows of `k` values
/// against `n_rows` output rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorkspaceShape {
    pub(crate) n_rows: usize,
    pub(crate) k: usize,
    pub(crate) tokens: usize,
}

/// The sizes a context allocates its workspace for, and whether a larger call may
/// grow it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Admission {
    pub(crate) shapes: Vec<WorkspaceShape>,
    pub(crate) growth: Growth,
}

impl WorkspaceShape {
    /// `BadRequest` unless the adapter accepts the shape. The checks repeat the
    /// adapter's, so a caller mistake is refused here and never reaches a native
    /// call whose rejection would latch the backend.
    fn validate(&self) -> Result<(), EngError> {
        let tile_ok = self.n_rows > 0 && self.n_rows.is_multiple_of(TILE_ROWS);
        let k_ok = (1..=MAX_DIM).contains(&self.k) && self.k.is_multiple_of(TILE_COLS);
        let tokens_ok = (1..=MAX_DIM).contains(&self.tokens);
        if tile_ok && k_ok && tokens_ok {
            return Ok(());
        }
        Err(EngError::BadRequest(format!(
            "cuda workspace shape of {} rows, k {} and {} tokens: rows must be a positive multiple of {TILE_ROWS}, k a multiple of {TILE_COLS} up to {MAX_DIM}, tokens 1 to {MAX_DIM}",
            self.n_rows, self.k, self.tokens
        )))
    }
}

impl Admission {
    /// The workspace sizes that hold every shape: the largest of each region.
    ///
    /// Returns `BadRequest` for no shapes or a shape the adapter would reject.
    pub(in crate::kernels::cuda) fn caps(&self) -> Result<Activations, EngError> {
        let mut caps: Option<Activations> = None;
        for shape in &self.shapes {
            shape.validate()?;
            let need = native::activation_bytes(shape.n_rows, shape.k, shape.tokens)
                .map_err(|status| native_error(&TERMINAL, ADAPTER_OPERATION, status))?;
            caps = Some(caps.map_or(need, |sofar| sofar.union(need)));
        }
        caps.ok_or_else(|| EngError::BadRequest("cuda admission lists no shapes".to_owned()))
    }
}

/// The sizes of the served model's workspace; the tests that pin the default
/// numbers read them here.
#[cfg(test)]
pub(super) fn default_caps() -> Result<Activations, EngError> {
    crate::family::project::cuda::served_admission().caps()
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
            holds: HoldCount::default(),
        })
    }

    /// Takes a hold on the current allocation.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "staged for RD-9a (decision 0047)")
    )]
    pub(super) fn hold(&self) -> ArenaHold {
        self.holds.take()
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
    /// Refusals come before any allocation and leave the workspace, the live
    /// bytes and the terminal flag untouched: a `Fixed` workspace answers
    /// `MemoryPlan` (the call does not fit), then a workspace under a hold
    /// answers `BadRequest` (the caller must build a new arena instead). The new
    /// allocation of a grow is made first, so a failure (`MemoryPlan`) leaves the
    /// old workspace and its sizes untouched. Callers hold the dispatch guard.
    pub(super) fn reserve_workspace(
        &self,
        workspace: &mut Workspace,
        need: Activations,
    ) -> Result<(), EngError> {
        if workspace.caps.covers(need) {
            return Ok(());
        }
        // A grow allocates and frees, which a capture window forbids; refuse before any change.
        ensure_closed("a workspace grow")?;
        if self.growth == Growth::Fixed {
            return Err(EngError::MemoryPlan(format!(
                "cuda workspace: a call needs {need:?} but this workspace is fixed at {:?}; serve it from a separate arena",
                workspace.caps
            )));
        }
        if workspace.holds.active() > 0 {
            return Err(EngError::BadRequest(
                "cuda workspace: a captured graph holds the current allocation (retire, never free); build a new arena instead of growing it"
                    .to_owned(),
            ));
        }
        let next = Workspace::allocate(&self.lease, workspace.caps.union(need))?;
        #[cfg(test)]
        self.workspace_allocations
            .set(self.workspace_allocations.get() + 1);
        // No hold refers to the old allocation, so it is freed here.
        *workspace = next;
        Ok(())
    }

    /// Takes a hold on the current workspace allocation, so a grow is refused
    /// until the hold is dropped. The capture of a graph (RD-2d) is the first
    /// caller.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "staged for RD-9a (decision 0047)")
    )]
    pub(crate) fn workspace_hold(&self) -> Result<ArenaHold, EngError> {
        self.ensure_usable()?;
        let workspace = self
            .workspace
            .try_borrow()
            .map_err(|_| workspace_in_use())?;
        Ok(workspace.hold())
    }
}

#[cfg(test)]
mod tests;
