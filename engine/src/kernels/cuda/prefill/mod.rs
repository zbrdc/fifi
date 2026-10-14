//! Safe launch wrappers and pure plan/sizing functions of the chunked-prefill kernels
//! (decision 0049). CPF-ATTN-A: `attn_prep_chunk`, the chunk KV write for both cache formats,
//! and `attn_prefill`, the flash-style prefill attention with its KV-split plan. CPF-DN-A:
//! `gdn_scan_seq`, the P-a value-sliced sequential gated-delta scan over a chunk, with the
//! gated output RMSNorm, and the reused decode conv (RD-7a's `gdn_conv`, called unedited with
//! `m = M`) that feeds it. CPF-7a: `swiglu_rows`, the M-row SwiGLU of the MoE path.
//!
//! This tree follows the recipe of `resident/` (its doc says how to add a kernel): one file
//! per kernel, `prefill/<n>.rs` with `prefill/<n>/tests.rs`, declared by one line at the end
//! of this file; the raw FFI lives in the child modules of `native/prefill.rs`. A wrapper
//! checks every argument, then queues its kernels on a [`DeviceStream`] and returns; it
//! allocates nothing, copies nothing and never synchronizes, and every range is converted
//! with `raw_range` and queued through `launch_on` with `Claim::of` and `require_apart`
//! (decision 0049 item 11).
//!
//! Prefill is eager (decision 0049 item 1) and no launch reads or writes a `StepParams`
//! field: `p0` and `m` are launch arguments, checked by the host before any launch and again
//! on the device (item 2), and a bound violation ORs mask 8 of the optional prefill status
//! word (item 10). `kernels::cuda` never depends on `family`. Every geometry value stays a
//! run-time argument of the entries, the plan and the sizing (lane C condition 8): head_dim
//! 256 with 8 query heads per kv head is the compiled attention case, head_dim 128 is the
//! compiled DeltaNet case, and any other value is a typed refusal before any launch.
//!
//! This parent carries no staging attribute (like `resident/mod.rs`): it holds no code, and
//! each child file carries its own.

// One `pub(crate) mod <n>;` line per kernel, alphabetical, below this line.
pub(crate) mod attn_prefill;
pub(crate) mod attn_prep_chunk;
pub(crate) mod gdn_scan_seq;
pub(crate) mod mmq_size;
pub(crate) mod swiglu_rows;
