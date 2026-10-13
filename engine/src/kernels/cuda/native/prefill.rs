//! Raw FFI to the prefill kernel entries of `native/prefill/<n>.cu` (decision 0049 item 11):
//! a child module of `native.rs` under the conditions of 0047 item 4, one child per kernel,
//! declared by one line at the end of this file. The recipe of a kernel is in the doc of
//! [`crate::kernels::cuda::resident`]; the prefill tree follows it.
//!
//! Safety invariants every child of this module relies on (the resident list, restated):
//! - Every launch entry is called only through `CudaContext::launch_on`, on the thread that
//!   owns the context and under the process dispatch guard (decisions 0023 and 0024). It only
//!   enqueues and never synchronizes.
//! - Every range is a `RawRange` proved inside its live allocation; each child proves sizes,
//!   alignment and disjointness again before the call (the one overlap rule,
//!   `super::resident::Span`), and the C entry repeats its checks on the addresses.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes
//!   the device before `cudaFree`.
//! - No raw pointer and no unsafe function leaves this tree; the `extern` declarations, the
//!   `repr(C)` mirrors and every raw pointer stay private to the child that owns them.
//!
//! This parent carries no staging attribute (unlike its children): it holds no item of its
//! own, only the imports its children share, so a `dead_code` expectation here would be
//! unfulfilled while the tree is staged. Each child carries its own.

use super::resident::{RawRange, RawStream, Span};
use super::{DeviceFailure, STAGE_NONE, checked, invalid_argument};

// One `pub(in crate::kernels::cuda) mod <n>;` line per kernel child module, alphabetical.
pub(in crate::kernels::cuda) mod attn_prefill;
pub(in crate::kernels::cuda) mod attn_prep_chunk;
