//! Raw FFI to the prefill kernel entries of `native/prefill/<n>.cu`: a child module of `native.rs`,
//! one child per kernel, declared by one line at the end of this file. The recipe of a kernel is in
//! the doc of [`crate::kernels::cuda::resident`]; the prefill tree follows it.
//!
//! Safety invariants every child of this module relies on (the resident list, restated):
//! - Every launch entry is called only through `CudaContext::launch_on`, on the thread that
//!   owns the context and under the process dispatch guard. It only enqueues and never
//!   synchronizes.
//! - Every range is a `RawRange` proved inside its live allocation; each child proves sizes,
//!   alignment and disjointness again before the call (the one overlap rule,
//!   `super::resident::Span`), and the C entry repeats its checks on the addresses.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes
//!   the device before `cudaFree`.
//! - No raw pointer and no unsafe function leaves this tree; the `extern` declarations, the
//!   `repr(C)` mirrors and every raw pointer stay private to the child that owns them.
//!
//! This parent holds one shared item, the prefill status mask the children's kernels OR into the
//! word; the rest is the imports its children share. It still carries no module staging attribute
//! (unlike its children, which each carry their own): the mask's own `dead_code` expectation is the
//! only one here, and the non-test builds fulfil it while the tree is staged.

use std::ffi::c_int;

use super::resident::{RawRange, RawStream, Span};
use super::{DeviceFailure, STAGE_NONE, checked, invalid_argument};

/// The kernels' mirror of `INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION` in `prefill/status.h`:
/// mask 8 (0x8) of the prefill status word, a prefill chunk fault — the mask
/// `family/qwen36/cuda/book.rs` `status_error` maps to Backend `"prefill_chunk"`. One const here;
/// every user imports this one.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the tests read it today; the prefill driver will too"
    )
)]
pub(crate) const PREFILL_STATUS_BOUND_VIOLATION: c_int = 0x8;

// One `pub(in crate::kernels::cuda) mod <n>;` line per kernel child module, alphabetical.
pub(in crate::kernels::cuda) mod attn_prefill;
pub(in crate::kernels::cuda) mod attn_prep_chunk;
pub(in crate::kernels::cuda) mod flint_qk;
pub(in crate::kernels::cuda) mod gdn_scan_seq;
pub(in crate::kernels::cuda) mod kv8_dequant;
pub(in crate::kernels::cuda) mod mmq_dense;
pub(in crate::kernels::cuda) mod mmq_facts;
pub(in crate::kernels::cuda) mod mmq_mmid;
pub(in crate::kernels::cuda) mod mmq_quant;
pub(in crate::kernels::cuda) mod row_ops;
pub(in crate::kernels::cuda) mod swiglu_rows;
