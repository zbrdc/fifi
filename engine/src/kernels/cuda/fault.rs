//! Child-process fault qualification for the CUDA backend (decision 0040 item 2
//! and REPLY-16 item 1).
//!
//! A sticky CUDA error poisons the whole process, so the fault runs in a child
//! process. The parent re-runs this test binary with `--exact` on the child
//! test and `INFERENCE_CUDA_FAULT_CHILD=1`, then checks the child's exit and
//! output. The parent never touches the device after the child runs.
//!
//! The child orders its steps so each one is reachable. The illegal-address
//! kernel runs first because it creates the sticky error. The vendored launch
//! functions only return their error status after that: on a healthy context
//! the null-pointer launch would start a kernel that faults asynchronously and
//! is never reported at the launch stage.
//!
//! The vendored launch reaches the GEMM launch path in `gemm.cuh`, the site
//! whose failure REPLY-16 patched: before the patch a failed launch or
//! shared-memory opt-in ended the whole process from inside the vendored code
//! (REPLY-16 lists `gemm.cuh:350`), so the child would never print
//! `fault child: done`.

use std::ffi::c_int;
use std::process::{Command, Output};
use std::sync::atomic::Ordering;

use super::guard::DispatchGuard;
use super::native::{self, DeviceFailure};
use super::{
    CHUNK_LIMIT, CudaContext, DeviceBuffer, QuantTensor, TERMINAL, native_error, stage_operation,
};
use crate::EngError;
use crate::family::project::Projector;
use crate::family::project::cuda::{CudaProjector, CudaWeight};
use crate::weights::GgufType;

const CHILD_ENV: &str = "INFERENCE_CUDA_FAULT_CHILD";
const CHILD_TEST: &str = "kernels::cuda::fault::fault_child_latches_terminal";
const TERMINAL_MESSAGE: &str = "CUDA backend is terminal after an earlier native failure";
/// `cudaErrorIllegalAddress`, the status a null-pointer write reports at synchronize.
const CUDA_ERROR_ILLEGAL_ADDRESS: c_int = 700;
const ILLEGAL_ADDRESS_MESSAGE: &str = "an illegal memory access was encountered (status 700)";

/// Mirrors `tests/cuda_smoke.rs`: a host with no device skips the test, unless
/// `INFERENCE_CUDA_REQUIRE_DEVICE` turns the missing device into a failure.
fn device_present(test: &str) -> bool {
    let count = super::device_count().expect("CUDA device count query");
    println!("CUDA devices visible: {count}");
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

/// Maps a device failure exactly as the production calls do, so the child test
/// checks the same operation names and terminal latch.
fn backend_error(failure: DeviceFailure) -> EngError {
    native_error(&TERMINAL, stage_operation(failure.stage), failure.status)
}

/// Runs the child test in a fresh process. The device flag passes through the
/// inherited environment.
fn run_child() -> Output {
    let exe = std::env::current_exe().expect("test binary path");
    Command::new(exe)
        .args(["--exact", CHILD_TEST, "--test-threads=1", "--nocapture"])
        .env(CHILD_ENV, "1")
        .output()
        .expect("spawn fault child")
}

#[test]
fn fault_latches_terminal_in_child_process() {
    if !device_present("fault_latches_terminal_in_child_process") {
        return;
    }
    let output = run_child();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "fault child exited with {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    assert!(
        stdout.contains("fault child: done"),
        "fault child did not finish\nstdout:\n{stdout}"
    );
}

#[test]
fn fault_child_latches_terminal() {
    if !std::env::var(CHILD_ENV).is_ok_and(|value| value == "1") {
        println!("SKIPPED fault_child_latches_terminal: not a fault child");
        return;
    }
    // Create the context and a buffer while the device is healthy, so the
    // later drop of the buffer has something to skip.
    let ctx = CudaContext::new().expect("context before the fault");
    let buffer = ctx.upload(&[0u8; 256]).expect("buffer before the fault");
    // Two experts of 8 rows, so `expert_linear` has a valid stack to refuse.
    let stack = CudaWeight::Tiled {
        name: "w".to_owned(),
        weight: ctx
            .upload_tiled(
                QuantTensor {
                    name: "w",
                    ty: GgufType::Q4K,
                    n_rows: 16,
                    k: 256,
                    source_len: 16 * 144,
                },
                CHUNK_LIMIT,
                &mut |_, source| {
                    source.fill(0);
                    Ok(())
                },
            )
            .expect("tiled weight before the fault"),
        experts: Some(2),
    };
    // The second of the two process leases, taken before the latch refuses it.
    let mut projector = CudaProjector::new().expect("projector before the fault");

    trigger_illegal_address();
    trigger_vendor_launch();
    refuse_later_calls(&ctx, &mut projector, &stack);
    drop_after_terminal(buffer, stack, ctx);
    drop(projector);
    println!("fault child: done");
}

/// The kernel writes through a null pointer and waits, so the fault surfaces at
/// synchronization and latches the process terminal.
fn trigger_illegal_address() {
    let _dispatch = DispatchGuard::enter(&TERMINAL);
    let failure = native::fault_illegal_address().expect_err("illegal address must fault");
    assert_eq!(failure.stage, native::STAGE_SYNCHRONIZE);
    assert_eq!(failure.status, CUDA_ERROR_ILLEGAL_ADDRESS);
    // The latch is set by the native_error mapping, so it is checked after it.
    let error = backend_error(failure);
    assert!(
        TERMINAL.load(Ordering::Acquire),
        "fault must latch terminal"
    );
    assert_eq!(
        error,
        EngError::Backend {
            operation: "cuda synchronize",
            message: ILLEGAL_ADDRESS_MESSAGE.to_owned(),
        }
    );
}

/// After the sticky fault the vendored launches return their error status
/// instead of ending the process, which is the patched return path.
fn trigger_vendor_launch() {
    let _dispatch = DispatchGuard::enter(&TERMINAL);
    let failure = native::fault_vendor_launch().expect_err("vendored launch after sticky fault");
    assert_eq!(failure.stage, native::STAGE_LAUNCH);
    let error = backend_error(failure);
    assert_eq!(
        error,
        EngError::Backend {
            operation: "cuda launch",
            message: ILLEGAL_ADDRESS_MESSAGE.to_owned(),
        }
    );
}

/// The error every call returns once the backend is terminal.
fn terminal_error() -> EngError {
    EngError::Backend {
        operation: "execute",
        message: TERMINAL_MESSAGE.to_owned(),
    }
}

/// Every later call returns `execute` without reaching the native layer. The
/// requests are valid, so only the latch can reject them.
fn refuse_later_calls(ctx: &CudaContext, projector: &mut CudaProjector, stack: &CudaWeight) {
    let CudaWeight::Tiled { weight, .. } = stack else {
        panic!("the fault child builds a tiled stack");
    };
    let error = ctx
        .memory_info()
        .expect_err("terminal context refuses memory_info");
    assert_eq!(error, terminal_error());
    let Err(error) = ctx.upload(&[0u8; 256]) else {
        panic!("terminal context must refuse upload");
    };
    assert_eq!(error, terminal_error());
    let error = ctx
        .linear(weight, &[0.0f32; 256])
        .expect_err("terminal context refuses linear");
    assert_eq!(error, terminal_error());
    let error = ctx
        .linear_rows(weight, 8, 8, &[0.0f32; 256])
        .expect_err("terminal context refuses linear_rows");
    assert_eq!(error, terminal_error());
    // Expert 99 is outside the stack, so only the latch can explain `execute`
    // instead of `BadRequest`: the projector checks it before anything else.
    let error = projector
        .expert_linear(stack, 99, &[0.0f32; 256])
        .expect_err("terminal projector refuses expert_linear");
    assert_eq!(error, terminal_error());
    let Err(error) = CudaContext::new() else {
        panic!("terminal backend must refuse a new context");
    };
    assert_eq!(error, terminal_error());
}

/// Dropping a buffer after the latch must neither panic nor free device memory.
fn drop_after_terminal(buffer: DeviceBuffer, stack: CudaWeight, ctx: CudaContext) {
    drop(buffer);
    drop(stack);
    assert!(
        TERMINAL.load(Ordering::Acquire),
        "drop must not clear terminal"
    );
    drop(ctx);
}
