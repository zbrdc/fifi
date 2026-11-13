//! The launch scaffolding of the EXL3 MoE block: the dims record, the routing
//! buffer views, and the context+stream pair every launcher takes. The Qwen3.6
//! constructors (from `StepDims`) stay in `qwen36::cuda`; the paths re-export
//! from there.
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::family) struct MoeDims {
    /// Token rows; stage 1 runs this block at one row (verify windows need their own sizing).
    pub(in crate::family) rows: usize,
    /// Hidden size: the router and shared-projection width, the residual width.
    pub(in crate::family) hidden: usize,
    pub(in crate::family) n_expert: usize,
    /// Experts picked per row.
    pub(in crate::family) top_k: usize,
    /// Rows of one expert's gate and up projections.
    pub(in crate::family) inter: usize,
    /// Rows of the shared expert's gate and up projections.
    pub(in crate::family) shared_inter: usize,
}

/// The routing and residual ranges of the block: views of the step arena and the status word.
pub(in crate::family) struct RouteBufs<'b> {
    /// f32 `[hidden]`: the residual after the mixer add, the block input.
    pub(in crate::family) resid_in: DeviceRange<'b>,
    /// f32 `[hidden]`: the other residual buffer, the block output.
    pub(in crate::family) resid_out: DeviceRange<'b>,
    /// f32 `[hidden]`: the normalized row, the router input.
    pub(in crate::family) norm_row: DeviceRange<'b>,
    /// The router's logits scratch (`RouterLogits`).
    pub(in crate::family) logits: DeviceRange<'b>,
    /// i32 `[top_k]`: the pick ids, read on the device only.
    pub(in crate::family) ids: DeviceRange<'b>,
    /// f32 `[top_k]`: the renormalized pick weights.
    pub(in crate::family) weights: DeviceRange<'b>,
    /// f32: the shared gate, sigmoid-ed by the router.
    pub(in crate::family) shared_gate: DeviceRange<'b>,
    /// The status word the router ORs its fault bit into.
    pub(in crate::family) status: DeviceRange<'b>,
}

/// The context and the one owned stream the block queues on. Grouping them keeps every function
/// of this file at four arguments or fewer (the project rules).
pub(in crate::family) struct Launch<'a> {
    /// The adapter that owns every buffer of the block.
    pub(in crate::family) ctx: &'a CudaContext,
    /// The one owned stream: every op queues on it, and none of them makes the host wait.
    pub(in crate::family) stream: &'a DeviceStream,
}
