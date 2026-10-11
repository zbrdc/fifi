//! Tests of the workspace: its placement (pure) and its use by a context (3090).

use std::process::Command;

use super::*;
use crate::kernels::cuda::arena::ARENA_ALIGN;
use crate::kernels::cuda::{CHUNK_LIMIT, EXECUTE_OPERATION, QuantTensor, TiledWeight};
use crate::weights::GgufType;

const K: usize = 256;
/// The longest row the matmul takes.
const WIDE_K: usize = 32_768;

/// Mirrors `fault.rs`: a host without a device skips the test, unless
/// `INFERENCE_CUDA_REQUIRE_DEVICE` turns the missing device into a failure.
fn device_present(test: &str) -> bool {
    let count = crate::kernels::cuda::device_count().expect("CUDA device count query");
    if count > 0 {
        return true;
    }
    let required = std::env::var("INFERENCE_CUDA_REQUIRE_DEVICE")
        .is_ok_and(|value| !value.is_empty() && value != "0");
    assert!(
        !required,
        "INFERENCE_CUDA_REQUIRE_DEVICE is set but no CUDA device is visible"
    );
    println!("SKIPPED {test}: no CUDA device");
    false
}

/// Row `row`, column `col` of the test weight: small integers, scale 1.0.
fn quant(row: usize, col: usize) -> i8 {
    i8::try_from((3 * row + col) % 7).expect("below 7") - 3
}

/// Q8_0 blocks of `n_rows` rows of `k` values.
fn q8_source(n_rows: usize, k: usize) -> Vec<u8> {
    let mut source = Vec::new();
    for row in 0..n_rows {
        for block in 0..k / 32 {
            source.extend_from_slice(&0x3c00u16.to_le_bytes());
            source.extend((0..32).map(|i| quant(row, block * 32 + i).cast_unsigned()));
        }
    }
    source
}

fn upload_q8(ctx: &CudaContext, n_rows: usize, k: usize) -> TiledWeight {
    let source = q8_source(n_rows, k);
    let tensor = QuantTensor {
        name: "w",
        ty: GgufType::Q80,
        n_rows,
        k,
        source_len: source.len(),
    };
    ctx.upload_tiled(tensor, CHUNK_LIMIT, &mut |offset, buffer| {
        buffer.copy_from_slice(&source[offset..offset + buffer.len()]);
        Ok(())
    })
    .expect("upload the Q8_0 weight")
}

/// `tokens` rows of `k` small integers, exact in fp16.
fn input(tokens: usize, k: usize) -> Vec<f32> {
    (0..tokens * k)
        .map(|i| f32::from(u8::try_from((i * 5 + i / k) % 9).expect("below 9")) - 4.0)
        .collect()
}

/// The exact result for rows `first..first + rows`: every product and partial sum
/// is an integer far below 2^24, so the device sum equals this one bit for bit.
fn expected(first: usize, rows: usize, x: &[f32], k: usize) -> Vec<f32> {
    x.chunks_exact(k)
        .flat_map(|token| {
            (first..first + rows).map(move |row| {
                token
                    .iter()
                    .enumerate()
                    .map(|(col, &value)| value * f32::from(quant(row, col)))
                    .sum::<f32>()
            })
        })
        .collect()
}

fn assert_linear(ctx: &CudaContext, weight: &TiledWeight, n_rows: usize, tokens: usize) {
    let k = weight.k();
    let x = input(tokens, k);
    let y = ctx.linear(weight, &x).expect("linear");
    assert_eq!(
        y,
        expected(0, n_rows, &x, k),
        "{n_rows} rows, {tokens} tokens"
    );
}

/// The workspace of one token at 16 rows: x 1024, xf 1_048_592, y 64 bytes.
fn small_caps() -> Activations {
    native::activation_bytes(16, K, 1).expect("sizes of a 16-row call")
}

/// Sizes whose x region alone is a gigabyte larger than the whole device.
fn huge_caps(ctx: &CudaContext) -> Activations {
    let total = ctx.memory_info().expect("memory info").total;
    Activations {
        x: total + (1 << 30),
        xf: 1_048_592,
        y: 64,
    }
}

#[test]
fn the_default_workspace_places_three_disjoint_aligned_regions() {
    let caps = default_caps().expect("default sizes");
    assert_eq!(
        caps,
        Activations {
            x: 348_160,
            xf: 1_048_592,
            y: 16_885_760
        }
    );
    let placement = place(caps).expect("place the default workspace");
    assert_eq!(
        placement.y,
        Region {
            offset: 0,
            bytes: 16_885_760
        }
    );
    assert_eq!(
        placement.xf,
        Region {
            offset: 16_885_760,
            bytes: 1_048_592
        }
    );
    assert_eq!(
        placement.x,
        Region {
            offset: 17_934_592,
            bytes: 348_160
        }
    );
    assert_eq!(placement.total, 18_282_752);
    let regions = [placement.x, placement.xf, placement.y];
    for (i, a) in regions.iter().enumerate() {
        assert_eq!(a.offset % ARENA_ALIGN, 0, "region {i}");
        for b in &regions[i + 1..] {
            assert!(
                a.offset + a.bytes <= b.offset || b.offset + b.bytes <= a.offset,
                "{a:?} overlaps {b:?}"
            );
        }
    }
}

#[test]
fn one_workspace_serves_every_token_count_without_allocating() {
    if !device_present("one_workspace_serves_every_token_count_without_allocating") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let weight = upload_q8(&ctx, 16, K);
    assert_linear(&ctx, &weight, 16, 1);
    let live = ctx.live_bytes();
    assert_eq!(ctx.workspace_allocations(), 1);
    // Only a build that qualifies fault handling counts the adapter's cudaMalloc
    // and cudaFree calls (see `native/projector.cu`); the other builds check the
    // workspace and the live bytes, which a transient allocation does not move.
    #[cfg(feature = "cuda-fault-qualification")]
    let native_calls = native::allocation_calls();
    for tokens in [1, 17, 4, 16, 17, 1, 32, 3] {
        assert_linear(&ctx, &weight, 16, tokens);
        let x = input(tokens, K);
        let half = ctx.linear_rows(&weight, 8, 8, &x).expect("linear rows");
        assert_eq!(half, expected(8, 8, &x, K), "rows 8..16, {tokens} tokens");
        assert_eq!(ctx.workspace_allocations(), 1, "{tokens} tokens");
        assert_eq!(ctx.live_bytes(), live, "{tokens} tokens");
        #[cfg(feature = "cuda-fault-qualification")]
        assert_eq!(native::allocation_calls(), native_calls, "{tokens} tokens");
    }
}

#[test]
fn a_larger_call_grows_the_workspace_once_and_never_shrinks_it() {
    if !device_present("a_larger_call_grows_the_workspace_once_and_never_shrinks_it") {
        return;
    }
    let ctx = CudaContext::with_workspace(small_caps()).expect("context with a small workspace");
    let narrow = upload_q8(&ctx, 16, K);
    let wide = upload_q8(&ctx, 1024, K);
    let first = ctx.workspace_bytes();
    // The wide call needs 4096 bytes of y, so y alone grows.
    assert_linear(&ctx, &wide, 1024, 1);
    assert_eq!(ctx.workspace_allocations(), 2);
    // 17 tokens need 17,408 bytes of x; y (1088) and the row-major xf already fit.
    assert_linear(&ctx, &narrow, 16, 17);
    assert_eq!(ctx.workspace_allocations(), 3);
    let grown = ctx.workspace_bytes();
    assert!(grown > first, "{grown} <= {first}");
    let live = ctx.live_bytes();
    for _ in 0..2 {
        assert_linear(&ctx, &narrow, 16, 1);
        assert_linear(&ctx, &wide, 1024, 1);
        assert_linear(&ctx, &narrow, 16, 17);
    }
    assert_eq!(ctx.workspace_allocations(), 3);
    assert_eq!(ctx.workspace_bytes(), grown);
    assert_eq!(ctx.live_bytes(), live);
}

#[test]
fn a_row_major_call_above_one_mebibyte_of_xf_grows_only_xf() {
    if !device_present("a_row_major_call_above_one_mebibyte_of_xf_grows_only_xf") {
        return;
    }
    // 17 tokens at k = 32,768 need an xf of 17 * 32,768 * 2 = 1,114,112 bytes,
    // above the 1,048,592 of the fixed fragment-order buffer that every call of
    // 16 tokens or fewer uses. x and y already hold the call, so xf alone grows.
    let need = native::activation_bytes(8, WIDE_K, 17).expect("sizes of the wide call");
    assert_eq!(need.xf, 1_114_112);
    let caps = Activations {
        xf: 1_048_592,
        ..need
    };
    let ctx = CudaContext::with_workspace(caps).expect("context with a narrow xf");
    let wide = upload_q8(&ctx, 8, WIDE_K);
    let narrow = upload_q8(&ctx, 16, K);
    let first = ctx.workspace_bytes();
    #[cfg(feature = "cuda-fault-qualification")]
    let before = native::allocation_calls();
    assert_linear(&ctx, &wide, 8, 17);
    // A grow is one allocation and one free; this also shows the counter counts.
    #[cfg(feature = "cuda-fault-qualification")]
    assert_eq!(
        native::allocation_calls(),
        (before.0 + 1, before.1 + 1),
        "one malloc and one free"
    );
    assert_eq!(ctx.workspace_allocations(), 2);
    assert_eq!(ctx.workspace.borrow().caps, need);
    let grown = ctx.workspace_bytes();
    assert!(grown > first, "{grown} <= {first}");
    // A decode call and the wide call again need no further growth.
    assert_linear(&ctx, &narrow, 16, 1);
    assert_linear(&ctx, &wide, 8, 17);
    assert_eq!(ctx.workspace_allocations(), 2);
    assert_eq!(ctx.workspace_bytes(), grown);
}

#[test]
fn a_nested_borrow_of_the_workspace_is_a_plain_error_and_does_not_latch() {
    if !device_present("a_nested_borrow_of_the_workspace_is_a_plain_error_and_does_not_latch") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let weight = upload_q8(&ctx, 16, K);
    let x = input(1, K);
    {
        // Safe code reaches the error only from inside the module tree, by
        // holding the borrow across the call.
        let _held = ctx.workspace.borrow_mut();
        let error = ctx
            .linear_rows(&weight, 0, 16, &x)
            .expect_err("the workspace is borrowed");
        assert_eq!(
            error,
            EngError::Backend {
                operation: ADAPTER_OPERATION,
                message: "the CUDA workspace is already in use".to_owned(),
            }
        );
    }
    assert_linear(&ctx, &weight, 16, 1);
}

const CHILD_ENV: &str = "INFERENCE_CUDA_WORKSPACE_CHILD";
const CHILD_TEST: &str = "kernels::cuda::context::workspace::tests::a_failed_free_in_a_grow_child";

/// A failed free latches the backend for the whole process, so the case runs in
/// a child process: the parent re-runs this binary on the child test only.
#[test]
fn a_failed_free_inside_a_grow_stops_the_call_in_flight() {
    if !device_present("a_failed_free_inside_a_grow_stops_the_call_in_flight") {
        return;
    }
    let exe = std::env::current_exe().expect("test binary path");
    let output = Command::new(exe)
        .args(["--exact", CHILD_TEST, "--test-threads=1", "--nocapture"])
        .env(CHILD_ENV, "1")
        .output()
        .expect("spawn the workspace child");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "workspace child exited with {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    assert!(
        stdout.contains("workspace child: done"),
        "workspace child did not finish\nstdout:\n{stdout}"
    );
}

#[test]
fn a_failed_free_in_a_grow_child() {
    if !std::env::var(CHILD_ENV).is_ok_and(|value| value == "1") {
        println!("SKIPPED a_failed_free_in_a_grow_child: not a workspace child");
        return;
    }
    let ctx = CudaContext::with_workspace(small_caps()).expect("context with a small workspace");
    let weight = upload_q8(&ctx, 16, K);
    // Swap in a buffer that no allocation backs. The grow below drops it, and
    // `cudaFree` rejects it without a sticky device error: the device stays
    // healthy, so the matmul would run if the call went on after the latch.
    let unbacked = ctx.wrap(RawDevice::unbacked(1));
    let real = std::mem::replace(&mut ctx.workspace.borrow_mut().buffer, unbacked);
    drop(real);
    assert!(!TERMINAL.load(std::sync::atomic::Ordering::Acquire));
    // 17 tokens need more x than the small workspace holds, so this call grows it.
    let x = input(17, K);
    let result = ctx.linear_rows(&weight, 0, 16, &x);
    assert!(
        TERMINAL.load(std::sync::atomic::Ordering::Acquire),
        "the failed free must latch the backend"
    );
    let Err(EngError::Backend { operation, .. }) = result else {
        panic!("the call went on after the latch: {result:?}");
    };
    assert_eq!(operation, EXECUTE_OPERATION);
    println!("workspace child: done");
}

#[test]
fn a_workspace_that_cannot_be_allocated_fails_the_context_and_frees_its_lease() {
    if !device_present("a_workspace_that_cannot_be_allocated_fails_the_context_and_frees_its_lease")
    {
        return;
    }
    let first = CudaContext::new().expect("context");
    let huge = huge_caps(&first);
    let Err(error) = CudaContext::with_workspace(huge) else {
        panic!("a workspace larger than the device was allocated");
    };
    let EngError::MemoryPlan(message) = &error else {
        panic!("expected MemoryPlan, got {error:?}");
    };
    assert!(message.starts_with("cuda allocate: "), "{message}");
    // Two leases are the process limit, so a leaked lease would fail here.
    let second = CudaContext::new().expect("a second lease after the failed attempt");
    drop(second);
    drop(first);
}

#[test]
fn a_failed_grow_leaves_the_workspace_and_its_caps_untouched() {
    if !device_present("a_failed_grow_leaves_the_workspace_and_its_caps_untouched") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let weight = upload_q8(&ctx, 16, K);
    let huge = huge_caps(&ctx);
    let recorded = (
        ctx.workspace_allocations(),
        ctx.workspace_bytes(),
        ctx.live_bytes(),
    );
    for _ in 0..2 {
        let result = ctx.reserve_workspace(&mut ctx.workspace.borrow_mut(), huge);
        assert!(matches!(result, Err(EngError::MemoryPlan(_))), "{result:?}");
        let now = (
            ctx.workspace_allocations(),
            ctx.workspace_bytes(),
            ctx.live_bytes(),
        );
        assert_eq!(now, recorded);
    }
    assert_linear(&ctx, &weight, 16, 17);
}

#[test]
fn dropping_a_context_returns_its_workspace_bytes_also_after_growth() {
    if !device_present("dropping_a_context_returns_its_workspace_bytes_also_after_growth") {
        return;
    }
    let anchor = CudaContext::new().expect("anchor context");
    let before = anchor.live_bytes();
    {
        let ctx = CudaContext::new().expect("second context");
        assert_eq!(ctx.workspace_bytes(), 18_282_752);
        assert_eq!(ctx.live_bytes() - before, 18_282_752);
    }
    assert_eq!(anchor.live_bytes(), before);
    {
        let ctx = CudaContext::with_workspace(small_caps()).expect("small context");
        let weight = upload_q8(&ctx, 16, K);
        assert_linear(&ctx, &weight, 16, 17);
        assert_eq!(ctx.workspace_allocations(), 2);
        assert_eq!(
            ctx.live_bytes() - before,
            ctx.workspace_bytes() + weight.buffer().bytes()
        );
    }
    assert_eq!(anchor.live_bytes(), before);
}

/// One shape of 16 rows of `K` values.
fn shape(n_rows: usize, k: usize, tokens: usize) -> WorkspaceShape {
    WorkspaceShape { n_rows, k, tokens }
}

fn assert_bad_shape(shapes: Vec<WorkspaceShape>, case: &str) {
    let admission = Admission {
        shapes,
        growth: Growth::Unbounded,
    };
    assert!(
        matches!(admission.caps(), Err(EngError::BadRequest(_))),
        "{case}: expected a BadRequest"
    );
    assert!(
        !TERMINAL.load(std::sync::atomic::Ordering::Acquire),
        "{case}: a refused shape must not latch the backend"
    );
}

#[test]
fn admission_caps_are_the_union_of_the_shapes_and_bad_shapes_are_refused() {
    // One shape is wide with one token (large y), the other narrow with many tokens
    // (large x): the union takes the larger of each region.
    let wide = native::activation_bytes(1024, K, 1).expect("wide sizes");
    let long = native::activation_bytes(16, K, 40).expect("long sizes");
    assert!(
        wide.y > long.y && long.x > wide.x,
        "the shapes must each win a region"
    );
    let admission = Admission {
        shapes: vec![shape(1024, K, 1), shape(16, K, 40)],
        growth: Growth::Fixed,
    };
    let caps = admission.caps().expect("caps of two shapes");
    assert_eq!(caps.x, wide.x.max(long.x));
    assert_eq!(caps.xf, wide.xf.max(long.xf));
    assert_eq!(caps.y, wide.y.max(long.y));

    let served = crate::family::project::cuda::served_admission();
    assert_eq!(
        served.caps().expect("served sizes"),
        Activations {
            x: 348_160,
            xf: 1_048_592,
            y: 16_885_760
        }
    );

    assert_bad_shape(Vec::new(), "no shapes");
    assert_bad_shape(vec![shape(16, 100, 1)], "k = 100");
    assert_bad_shape(vec![shape(12, K, 1)], "n_rows = 12");
    assert_bad_shape(vec![shape(16, K, 0)], "tokens = 0");
    assert_bad_shape(
        vec![shape(16, K, native::MAX_DIM + 1)],
        "tokens past MAX_DIM",
    );
    assert_bad_shape(vec![shape(16, native::MAX_DIM + 256, 1)], "k past MAX_DIM");
    assert_bad_shape(vec![shape(0, K, 1)], "n_rows = 0");
    assert_bad_shape(
        vec![shape(16, K, 1), shape(16, 100, 1)],
        "a bad shape after a good one",
    );
}

#[test]
fn a_fixed_workspace_refuses_a_larger_call_and_stays_unchanged() {
    if !device_present("a_fixed_workspace_refuses_a_larger_call_and_stays_unchanged") {
        return;
    }
    let admission = Admission {
        shapes: vec![shape(16, K, 1)],
        growth: Growth::Fixed,
    };
    let ctx = CudaContext::with_admission(&admission).expect("fixed context");
    let weight = upload_q8(&ctx, 16, K);
    assert_linear(&ctx, &weight, 16, 1);
    let (allocations, bytes, live) = (
        ctx.workspace_allocations(),
        ctx.workspace_bytes(),
        ctx.live_bytes(),
    );

    // 17 tokens need 17,408 bytes of x, and the workspace holds 1,024.
    let x = input(17, K);
    let Err(EngError::MemoryPlan(message)) = ctx.linear(&weight, &x) else {
        panic!("a larger call on a fixed workspace must be MemoryPlan");
    };
    assert!(
        message.contains("fixed"),
        "the message names the policy: {message}"
    );
    assert_eq!(ctx.workspace_allocations(), allocations);
    assert_eq!(ctx.workspace_bytes(), bytes);
    assert_eq!(ctx.live_bytes(), live);
    ctx.ensure_usable()
        .expect("the refusal does not latch the backend");
    assert_linear(&ctx, &weight, 16, 1);
    assert_eq!(ctx.workspace_allocations(), allocations);
}

#[test]
fn a_held_workspace_refuses_to_grow_until_the_hold_is_released() {
    if !device_present("a_held_workspace_refuses_to_grow_until_the_hold_is_released") {
        return;
    }
    let ctx = CudaContext::with_workspace(small_caps()).expect("context with a small workspace");
    let narrow = upload_q8(&ctx, 16, K);
    let wide = upload_q8(&ctx, 1024, K);
    let hold = ctx.workspace_hold().expect("hold");
    assert_linear(&ctx, &narrow, 16, 1);
    let (allocations, bytes) = (ctx.workspace_allocations(), ctx.workspace_bytes());

    let Err(EngError::BadRequest(message)) = ctx.linear(&wide, &input(1, K)) else {
        panic!("a grow under a hold must be BadRequest");
    };
    assert!(message.contains("retire, never free"), "{message}");
    assert_eq!(ctx.workspace_allocations(), allocations);
    assert_eq!(ctx.workspace_bytes(), bytes);
    ctx.ensure_usable()
        .expect("the refusal does not latch the backend");
    assert_linear(&ctx, &narrow, 16, 1);

    drop(hold);
    assert_linear(&ctx, &wide, 1024, 1);
    assert_eq!(ctx.workspace_allocations(), allocations + 1);
}
