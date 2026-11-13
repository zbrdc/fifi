//! The launch shape helpers of the EXL3 MoE prefill: the routed and shared SwiGLU row
//! shapes and the chunk-combine shape, pure functions of [`MoeDims`].

use crate::family::exl3::launch::MoeDims;
use crate::kernels::cuda::prefill::swiglu_rows::SwigluRowsShape;
use crate::kernels::cuda::resident::moe_combine::CombineShape;

/// The SwiGLU shape of the routed stacks (stage 14): `m * top_k` rows, the up half at column
/// `inter` of a `2 * inter` row.
pub(in crate::family) fn routed_swiglu(m: usize, dims: &MoeDims) -> SwigluRowsShape {
    SwigluRowsShape {
        rows: m * dims.top_k,
        n_ff: dims.inter,
        up_off: dims.inter,
        ld_in: 2 * dims.inter,
        ld_out: dims.inter,
    }
}

/// The combine shape (stage 17).
pub(in crate::family) fn chunk_combine_shape(dims: &MoeDims, m: usize) -> CombineShape {
    CombineShape {
        m,
        n: dims.hidden,
        top_k: dims.top_k,
        ld_down: dims.hidden,
    }
}

/// The shared expert's SwiGLU shape (stage 10d).
pub(in crate::family) fn shared_swiglu(m: usize, dims: &MoeDims) -> SwigluRowsShape {
    SwigluRowsShape {
        rows: m,
        n_ff: dims.shared_inter,
        up_off: dims.shared_inter,
        ld_in: 2 * dims.shared_inter,
        ld_out: dims.shared_inter,
    }
}
