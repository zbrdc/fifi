//! Safe launch wrappers of the resident-decode kernels.
//!
//! One file per kernel, `resident/<n>.rs`, declared at the end of this file by one line. A wrapper
//! checks every argument, then queues its kernel on a `DeviceStream` and returns; it allocates
//! nothing, copies nothing and never synchronizes. This tree holds no `unsafe` and no `extern "C"`:
//! both stay in the `native` tree.
//!
//! # How to add a kernel `<n>`
//!
//! `<n>` is a snake_case name such as `attn_decode`. A new kernel adds new files and one line to
//! each of four lists, and edits no other shared line. Paths are under `engine/src/kernels/cuda/`
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
//!    `#![cfg_attr(not(test), expect(dead_code, reason = "staged for the resident decode step"))]`
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
//!    `self.launch_on(stream, |raw| <native child function>(raw, &native_call))`. `launch_on` (in
//!    `context/launch.rs`) checks the terminal flag and that the stream belongs to this context,
//!    enters the dispatch guard, runs the closure with the raw stream, and counts one enqueue on
//!    the stream when the closure succeeds. The closure must only queue work, and the raw stream
//!    cannot leave it. The file starts with the same staging attribute as step 3. Call types are
//!    named by path (`kernels::cuda::resident::<n>::<N>Call`); nothing is re-exported from
//!    `kernels::cuda`.
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
//! 5. Tests, at the wrapper boundary only: `resident/<n>/tests.rs`, declared by
//!    `#[cfg(test)] mod tests;` at the end of `resident/<n>.rs`. The tree is compiled only with
//!    `native-cuda`. Start each device test with `testkit::device_present("<test name>")` (a host
//!    with no device skips; `INFERENCE_CUDA_REQUIRE_DEVICE=1` turns that into a failure). Keep
//!    helpers in the tests file, not in `testkit.rs`. Random test data comes from the shared
//!    `testkit::Rng` (xorshift64*, `Rng::new(seed)`, `next_u64`, `below`), never from a local copy
//!    of a generator. The first user deletes each `#[expect(dead_code)]` staging line on `Rng` and
//!    its `impl` in `testkit.rs` that clippy then reports as unfulfilled (the `impl` one stays
//!    fulfilled while any method, such as `below`, is unused). Compare against a host reference,
//!    with frozen tolerances, and check each rejection of the wrapper by its error text. One
//!    mutation per test. Device runs go through `gpu_run.sh` only.
//!
//! A missing `mod <n>;` line here does not fail the build: the file is not compiled and its tests
//! vanish. So list the test names of a new kernel with
//! `cargo test --features native-cuda --lib -- --list`. A missing line in `native/resident.rs` or
//! `build.rs` does fail the build (unresolved module, undefined symbol).

// One `pub(crate) mod <n>;` line per kernel, alphabetical, below this line.
pub(crate) mod add_norm_pack;
pub(crate) mod add_rmsnorm_q8;
pub(crate) mod argmax_rows;
pub(crate) mod attn_decode;
pub(crate) mod attn_decode_gemma;
pub(crate) mod attn_decode_loop;
pub(crate) mod attn_decode_q8;
pub(crate) mod attn_prep;
pub(crate) mod block_ops;
pub(crate) mod correct_draft_logits_bf16;
pub(crate) mod dense_gemv;
pub(crate) mod draft_attention_bf16;
pub(crate) mod embed;
pub(crate) mod exl3;
pub(crate) mod expert_activation;
pub(crate) mod gdn_conv;
pub(crate) mod gdn_delta;
pub(crate) mod gelu_ffn;
pub(crate) mod head_tile;
pub(crate) mod hyper;
pub(crate) mod inkling_controller;
pub(crate) mod inkling_conv;
pub(crate) mod inkling_decode_controller;
pub(crate) mod inkling_local_prep;
pub(crate) mod inkling_stream;
pub(crate) mod kv8_append;
pub(crate) mod kv_f16_append;
pub(crate) mod matmul_expert;
pub(crate) mod matvec_dq;
pub(crate) mod matvec_expert;
#[cfg(test)]
mod matvec_gguf;
pub(crate) mod matvec_gguf_f32;
pub(crate) mod matvec_native;
pub(crate) mod matvec_q;
pub(crate) mod mhc_route;
pub(crate) mod moe_combine;
pub(crate) mod multi_row;
pub(crate) mod pack_rows;
pub(crate) mod paged_position;
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the first caller is the device PLE step of the Flash-Next serve path"
    )
)]
pub(crate) mod ple_block;
#[cfg(test)]
pub(crate) mod q6_f32;
pub(crate) mod qsa_attn;
pub(crate) mod qsa_index;
pub(crate) mod quantize_q8_1;
pub(crate) mod relative_attention;
pub(crate) mod rope_gemma;
pub(crate) mod router;
pub(crate) mod router_rows;
pub(crate) mod row_add;
pub(crate) mod row_scale;
pub(crate) mod sample;
pub(crate) mod shortlist;
pub(crate) mod softcap;
pub(crate) mod sparse_decode;
pub(crate) mod sparse_score;
pub(crate) mod sparse_sketch;
pub(crate) mod sparse_topk;
pub(crate) mod step_params;
pub(crate) mod swiglu_pack;
pub(crate) mod swiglu_q8;
pub(crate) mod tiered_classify;
pub(crate) mod tiered_wait_add;
pub(crate) mod word_rows;

#[cfg(test)]
mod gemma4_reference_tests;

#[cfg(all(test, feature = "native-cuda"))]
mod krow_attention_tests;

#[cfg(test)]
mod top_logits;

#[cfg(all(test, feature = "native-cuda"))]
mod wrapper_completeness {
    //! Guards the one silent list of the recipe above: a missing `pub(crate) mod <n>;` line here
    //! compiles the wrapper file nowhere, so its tests vanish without a failure (a missing line
    //! in `native/resident.rs` or `build.rs` fails the build instead).

    use std::{fs, path::PathBuf};

    /// Resident `.rs` files that are not modules of this directory (none today). Name a file
    /// here when a new kind of non-module source joins the directory.
    const NON_MODULE_FILES: &[&str] = &[];

    #[test]
    fn every_resident_source_file_has_an_active_module_declaration() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/kernels/cuda/resident");
        let source = fs::read_to_string(dir.join("mod.rs")).expect("read resident/mod.rs");
        let declared = active_module_names(&source);
        let mut missing: Vec<String> = Vec::new();
        for entry in fs::read_dir(&dir).expect("read the resident directory") {
            let entry = entry.expect("resident directory entry");
            if !entry.file_type().expect("resident entry type").is_file() {
                continue; // wrapper test directories such as resident/<n>/
            }
            let name = entry.file_name().into_string().expect("UTF-8 file name");
            let Some(stem) = name.strip_suffix(".rs") else {
                continue;
            };
            if stem != "mod"
                && !NON_MODULE_FILES.contains(&name.as_str())
                && !declared.iter().any(|declared| declared == stem)
            {
                missing.push(name);
            }
        }
        assert!(
            missing.is_empty(),
            "resident files without an active `mod` declaration in resident/mod.rs, so their \
             tests silently vanish: {}",
            missing.join(", ")
        );
    }

    /// Names of the active `mod <n>;` and `pub(crate) mod <n>;` declarations of `source`, in
    /// order. Commented-out declarations never count: the block-comment depth carries across
    /// lines, because a block comment may open mid-line and nest, and a line counts only when
    /// it starts outside a block comment with the declaration keyword. Odd spellings such as
    /// `mod /* c */ n;` are rejected on purpose: the guard fails closed.
    fn active_module_names(source: &str) -> Vec<String> {
        let mut names = Vec::new();
        let mut block_depth = 0u32;
        for line in source.lines() {
            let bytes = line.as_bytes();
            let start_depth = block_depth;
            let mut at = 0;
            while at < bytes.len() {
                if block_depth == 0 && bytes[at] == b'/' && bytes.get(at + 1) == Some(&b'/') {
                    break; // a line comment hides the rest of the line
                }
                if bytes[at] == b'/' && bytes.get(at + 1) == Some(&b'*') {
                    block_depth += 1;
                    at += 2;
                    continue;
                }
                if block_depth > 0 && bytes[at] == b'*' && bytes.get(at + 1) == Some(&b'/') {
                    block_depth -= 1;
                    at += 2;
                    continue;
                }
                at += 1;
            }
            if start_depth != 0 {
                continue;
            }
            let trimmed = line.trim_start();
            let name = trimmed
                .strip_prefix("pub(crate) mod ")
                .or_else(|| trimmed.strip_prefix("mod "))
                .and_then(|rest| rest.split(';').next())
                .filter(|name| {
                    !name.is_empty()
                        && name
                            .chars()
                            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
                });
            if let Some(name) = name {
                names.push(name.to_string());
            }
        }
        names
    }
}
