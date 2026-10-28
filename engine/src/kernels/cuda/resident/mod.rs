//! Safe launch wrappers of the resident-decode kernels (decision 0047, stage 1).
//!
//! One file per kernel, `resident/<n>.rs`, declared at the end of this file by one line. A wrapper
//! checks every argument, then queues its kernel on a `DeviceStream` and returns; it allocates
//! nothing, copies nothing and never synchronizes. This tree holds no `unsafe` and no `extern "C"`:
//! both stay in the `native` tree (decision 0047, child modules of `native.rs`).
//!
//! # How to add a kernel `<n>`
//!
//! `<n>` is a snake_case name such as `attn_decode`. A kernel packet adds new files and one line
//! to each of four lists, and edits no other shared line. Paths are under `engine/src/kernels/cuda/`
//! unless they start with `engine/`.
//!
//! 1. CUDA source and header: new files `native/resident/<n>.cu` and `native/resident/<n>.h`.
//!    The header holds the `extern "C"` prototype of the launch entry and its contract comment,
//!    in the style of `native/abi_resident.h`, with the guard `INFERENCE_CUDA_RESIDENT_<N>_H`.
//!    `abi_resident.h` stays closed: it neither gains the prototype nor includes this header.
//!    The `.cu` file includes `"abi.h"`, `"device.h"` and `"resident/<n>.h"` (the include root is
//!    `native/`). The entry takes the stream explicitly and never falls back to the default
//!    stream, returns `INFERENCE_CUDA_INVALID_ARGUMENT` for a bad argument before any CUDA call,
//!    and does not synchronize.
//! 2. Build: `engine/build.rs` gets one line in `CUDA_SOURCES` (the `.cu`, appended at the END:
//!    the order is the member order of the archive) and one in `CUDA_HEADERS` (the `.h`, in
//!    alphabetical place). There is no directory scan.
//! 3. Native child module: new file `native/resident/<n>.rs`, declared by one line,
//!    `pub(in crate::kernels::cuda) mod <n>;`, in `native/resident.rs`. It starts with a module
//!    doc that lists the safety invariants it relies on, then (mandatory, in every native child
//!    and every wrapper file, each with its own copy; a parent and a child both carrying it
//!    builds clean)
//!    `#![cfg_attr(not(test), expect(dead_code, reason = "staged for RD-9a (decision 0047)"))]`
//!    until the resident step calls it. It holds the `unsafe extern "C"` declaration and one
//!    `pub(in crate::kernels::cuda)` safe function that takes `&RawStream` and a struct of
//!    `RawRange`s, proves sizes, alignment and overlap again, calls the entry under a
//!    `// SAFETY:` comment and returns `checked(status, stage)`. Import from `super::` what
//!    `native/resident.rs` already imports (`RawStream`, `RawRange`, `DeviceFailure`,
//!    `STAGE_NONE`, `checked`, `invalid_argument`). The overlap rule of a native child is
//!    `super::Span` (`Span::of(..).apart(..)`, private in `native/resident.rs`, which a child
//!    may use). No raw pointer and no unsafe function leaves the module.
//! 4. Safe wrapper: new file `resident/<n>.rs`, declared at the end of this file by
//!    `pub(crate) mod <n>;`. It adds a `pub(crate)` method `<n>` on `CudaContext` that takes the
//!    `DeviceStream` and a `<N>Call<'_>` struct of `DeviceRange`s. The method calls
//!    `ensure_usable`, rejects every bad argument as `BadRequest` before any native call, turns
//!    each `DeviceRange` into a native range with `raw_range`, and queues with
//!    `self.launch_on(stream, |raw| <native child function>(raw, &native_call))`. `launch_on`
//!    (in `context/launch.rs`, RD-2a-2) checks the terminal flag and that the stream belongs to
//!    this context, enters the dispatch guard, runs the closure with the raw stream, and counts
//!    one enqueue on the stream when the closure succeeds. The closure must only queue work, and
//!    the raw stream cannot leave it. The file starts with the same staging attribute as step 3.
//!    Call types are named by path (`kernels::cuda::resident::<n>::<N>Call`); nothing is
//!    re-exported from `kernels::cuda`.
//!
//!    Shared helpers (defined in `context/launch.rs`, re-exported at `kernels::cuda::context` with
//!    visibility `pub(in crate::kernels::cuda)`): a wrapper proves that the range a launch writes
//!    shares no byte with a range it reads by
//!    `require_apart(&Claim::of("name", &range, bytes), &Claim::of(..))`, and proves size and
//!    alignment of each range by `require_fit(name, &range, need_bytes, align)`. Never write a
//!    local overlap or fit helper. The first wrapper that uses them deletes the
//!    `#[expect(unused_imports)]` staging line on that re-export in `context.rs` once clippy
//!    reports it as unfulfilled, because an `expect` that nothing fulfils fails
//!    `clippy -D warnings`.
//! 5. Tests, seam 1 only (decision 0047, REPLY-29): `resident/<n>/tests.rs`, declared by
//!    `#[cfg(test)] mod tests;` at the end of `resident/<n>.rs`. The tree is compiled only with
//!    `native-cuda`. Start each device test with `testkit::device_present("<test name>")` (a host
//!    with no device skips; `INFERENCE_CUDA_REQUIRE_DEVICE=1` turns that into a failure). Keep
//!    helpers in the tests file, not in `testkit.rs`. Random test data comes from the shared
//!    `testkit::Rng` (xorshift64*, `Rng::new(seed)`, `next_u64`, `below`), never from a local
//!    copy of a generator. The first user deletes each `#[expect(dead_code)]` staging line on
//!    `Rng` and its `impl` in `testkit.rs` that clippy then reports as unfulfilled (the `impl`
//!    one stays fulfilled while any method, such as `below`, is unused). Compare against a host reference,
//!    with tolerances frozen by lane M, and check each rejection of the wrapper by its error
//!    text. One mutation per test. Device runs go through `gpu_run.sh` only.
//!
//! A missing `mod <n>;` line here does not fail the build: the file is not compiled and its tests
//! vanish. A kernel packet's gate therefore lists its test names with
//! `cargo test --features native-cuda --lib -- --list`. A missing line in `native/resident.rs`
//! or `build.rs` does fail the build (unresolved module, undefined symbol).

// One `pub(crate) mod <n>;` line per kernel, alphabetical, below this line.
pub(crate) mod add_norm_pack;
pub(crate) mod add_rmsnorm_q8;
pub(crate) mod attn_decode;
pub(crate) mod attn_decode_gemma;
pub(crate) mod attn_decode_q8;
pub(crate) mod attn_prep;
pub(crate) mod embed;
pub(crate) mod embed_q8;
pub(crate) mod exl3;
pub(crate) mod expert_activation;
pub(crate) mod gdn_conv;
pub(crate) mod gdn_delta;
pub(crate) mod gelu_ffn;
pub(crate) mod gemma_matvec;
pub(crate) mod kv8_append;
pub(crate) mod matmul_expert;
pub(crate) mod matvec_expert;
pub(crate) mod matvec_q;
pub(crate) mod moe_combine;
pub(crate) mod paged_position;
pub(crate) mod quantize_q8_1;
pub(crate) mod rope_gemma;
pub(crate) mod router;
pub(crate) mod row_add;
pub(crate) mod row_scale;
pub(crate) mod sample_gemma;
pub(crate) mod shortlist;
pub(crate) mod softcap;
pub(crate) mod step_params;
pub(crate) mod swiglu_pack;
pub(crate) mod swiglu_q8;
pub(crate) mod tiered_classify;
pub(crate) mod tiered_wait_add;

#[cfg(test)]
mod gemma4_reference_tests;

#[cfg(test)]
mod top_logits;
