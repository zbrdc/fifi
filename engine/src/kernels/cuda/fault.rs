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
use super::native::resident::graph::{fault_null_write_launch, fault_null_write_warm};
use super::native::{self, DeviceFailure};
use super::{
    CHUNK_LIMIT, CapturePlan, CudaContext, DeviceBuffer, DeviceEvent, DeviceStream, MatmulIo,
    PackCall, PinnedWord, Projection, QuantTensor, TERMINAL, device_present, native_error,
    stage_operation,
};
use crate::EngError;
use crate::family::project::Projector;
use crate::family::project::cuda::{CudaProjector, CudaWeight};
use crate::weights::GgufType;

const CHILD_ENV: &str = "INFERENCE_CUDA_FAULT_CHILD";
const CHILD_TEST: &str = "kernels::cuda::fault::fault_child_latches_terminal";
const DESTROY_CHILD_TEST: &str = "kernels::cuda::fault::failed_destroy_child_latches_terminal";
const REPLAY_CHILD_TEST: &str = "kernels::cuda::fault::fault_replay_child_latches_terminal";
const TERMINAL_MESSAGE: &str = "CUDA backend is terminal after an earlier native failure";
/// `cudaErrorIllegalAddress`, the status a null-pointer write reports at synchronize.
const CUDA_ERROR_ILLEGAL_ADDRESS: c_int = 700;
const ILLEGAL_ADDRESS_MESSAGE: &str = "an illegal memory access was encountered (status 700)";

/// The resident owners made while the device is healthy: later calls on them must
/// be refused, and their drops must make no native call.
struct Owners {
    stream: DeviceStream,
    event: DeviceEvent,
    word: PinnedWord,
    launch: LaunchBuffers,
}

/// The device buffers of one valid token-sized pack and matmul, made while the device is healthy.
struct LaunchBuffers {
    x: DeviceBuffer,
    xf: DeviceBuffer,
    y: DeviceBuffer,
}

impl Owners {
    fn new(ctx: &CudaContext) -> Self {
        Self {
            stream: ctx.stream().expect("stream before the fault"),
            event: ctx.event().expect("event before the fault"),
            word: ctx.pinned_word().expect("pinned word before the fault"),
            launch: LaunchBuffers {
                x: ctx.upload(&[0u8; 1024]).expect("x before the fault"),
                xf: ctx
                    .upload(&vec![0u8; 1_048_592])
                    .expect("xf before the fault"),
                y: ctx.upload(&[0u8; 64]).expect("y before the fault"),
            },
        }
    }
}

/// Maps a device failure exactly as the production calls do, so the child test
/// checks the same operation names and terminal latch.
fn backend_error(failure: DeviceFailure) -> EngError {
    native_error(&TERMINAL, stage_operation(failure.stage), failure.status)
}

/// Runs the child test `test` in a fresh process. The device flag passes through
/// the inherited environment.
fn run_child(test: &str) -> Output {
    let exe = std::env::current_exe().expect("test binary path");
    Command::new(exe)
        .args(["--exact", test, "--test-threads=1", "--nocapture"])
        .env(CHILD_ENV, "1")
        .output()
        .expect("spawn fault child")
}

#[test]
fn fault_latches_terminal_in_child_process() {
    if !device_present("fault_latches_terminal_in_child_process") {
        return;
    }
    assert_child_finishes(CHILD_TEST);
}

#[test]
fn a_failed_destroy_latches_terminal_in_child_process() {
    if !device_present("a_failed_destroy_latches_terminal_in_child_process") {
        return;
    }
    assert_child_finishes(DESTROY_CHILD_TEST);
}

#[test]
fn fault_replay_latches_terminal_in_child_process() {
    if !device_present("fault_replay_latches_terminal_in_child_process") {
        return;
    }
    assert_child_finishes(REPLAY_CHILD_TEST);
}

/// Runs the child `test` and checks that it exited cleanly and printed its marker.
fn assert_child_finishes(test: &str) {
    let output = run_child(test);
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

/// The failed-destroy half of decision 0022. The fault is not mapped through
/// `backend_error`, so the latch is still clear when the owners drop: the first
/// destroy runs, fails at its synchronize and latches, and the rest are skipped.
/// The stream is the first field of `Owners`, so it is the one dropped first.
/// That the C destroy then frees nothing is not observable here; the test pins
/// the Rust half (one attempt, the latch, the counts left as they were).
#[test]
fn failed_destroy_child_latches_terminal() {
    if !std::env::var(CHILD_ENV).is_ok_and(|value| value == "1") {
        println!("SKIPPED failed_destroy_child_latches_terminal: not a fault child");
        return;
    }
    let ctx = CudaContext::new().expect("context before the fault");
    let owners = Owners::new(&ctx);
    let live = ctx.live_owners();
    let attempts = ctx.destroy_calls();
    {
        let _dispatch = DispatchGuard::enter(&TERMINAL);
        let failure = native::fault_illegal_address().expect_err("illegal address must fault");
        assert_eq!(failure.status, CUDA_ERROR_ILLEGAL_ADDRESS);
    }
    assert!(
        !TERMINAL.load(Ordering::Acquire),
        "the fault was not mapped, so the latch is still clear"
    );
    drop(owners);
    assert!(
        TERMINAL.load(Ordering::Acquire),
        "a failed destroy latches terminal"
    );
    assert_eq!(
        ctx.destroy_calls(),
        attempts + 1,
        "one failed destroy, then every other owner is skipped"
    );
    assert_eq!(ctx.live_owners(), live, "no owner released its count");
    println!("fault child: done");
}

/// A fault inside a replayed graph surfaces at the next synchronize, latches the backend, and the
/// graph is then skipped by its drop: no native destroy and the live count stays (decision 0022).
#[test]
fn fault_replay_child_latches_terminal() {
    if !std::env::var(CHILD_ENV).is_ok_and(|value| value == "1") {
        println!("SKIPPED fault_replay_child_latches_terminal: not a fault child");
        return;
    }
    let ctx = CudaContext::new().expect("context before the fault");
    let owners = Owners::new(&ctx);
    let stream = &owners.stream;
    // The kernel's module loads at its first launch, which a capture forbids: load it disarmed.
    ctx.launch_on(stream, fault_null_write_warm)
        .expect("disarmed warm-up launch");
    stream.synchronize().expect("drain the warm-up");
    let plan = CapturePlan {
        pins: &[],
        holds: Vec::new(),
    };
    let (graph, ()) = ctx
        .capture(stream, plan, || {
            ctx.launch_on(stream, fault_null_write_launch)
        })
        .expect("capture of the faulting kernel");
    graph.launch(stream).expect("the replay is only queued");
    let error = stream
        .synchronize()
        .expect_err("the replay faults at the synchronize");
    assert_eq!(
        error,
        EngError::Backend {
            operation: "cuda synchronize",
            message: ILLEGAL_ADDRESS_MESSAGE.to_owned(),
        }
    );
    assert!(TERMINAL.load(Ordering::Acquire), "a replay fault latches");
    refuse_after_replay_fault(&ctx, &owners);
    let (attempts, live) = (ctx.destroy_calls(), ctx.live_owners());
    assert_eq!(live.graphs, 1, "the graph is counted");
    drop(graph);
    drop(owners);
    assert_eq!(
        ctx.destroy_calls(),
        attempts,
        "no native destroy after the latch"
    );
    assert_eq!(ctx.live_owners(), live, "skipped owners stay counted");
    drop(ctx);
    println!("fault child: done");
}

/// A pack and a capture with valid arguments are refused by the latch alone.
fn refuse_after_replay_fault(ctx: &CudaContext, owners: &Owners) {
    let buffers = &owners.launch;
    let pack = PackCall {
        x: buffers.x.range(0, 1024).expect("x range"),
        xf: buffers.xf.range(0, 1_048_592).expect("xf range"),
        m: 1,
        k: 256,
    };
    let error = ctx
        .pack(&owners.stream, pack)
        .expect_err("terminal context refuses pack");
    assert_eq!(error, terminal_error());
    let plan = CapturePlan {
        pins: &[],
        holds: Vec::new(),
    };
    let Err(error) = ctx.capture(&owners.stream, plan, || Ok(())) else {
        panic!("terminal context must refuse a capture");
    };
    assert_eq!(error, terminal_error());
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
    let owners = Owners::new(&ctx);
    // The second of the two process leases, taken before the latch refuses it.
    let mut projector = CudaProjector::new().expect("projector before the fault");

    trigger_illegal_address();
    trigger_vendor_launch();
    refuse_later_calls(&ctx, &mut projector, &stack);
    refuse_resident_calls(&ctx, &owners);
    refuse_launches(&ctx, &owners, &stack);
    drop_after_terminal(buffer, stack, owners, ctx);
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

/// The streams, events and pinned word made before the fault refuse every call.
fn refuse_resident_calls(ctx: &CudaContext, owners: &Owners) {
    let error = owners
        .stream
        .synchronize()
        .expect_err("terminal stream refuses synchronize");
    assert_eq!(error, terminal_error());
    let error = owners
        .stream
        .record(&owners.event)
        .expect_err("terminal stream refuses record");
    assert_eq!(error, terminal_error());
    let error = owners
        .event
        .synchronize()
        .expect_err("terminal event refuses synchronize");
    assert_eq!(error, terminal_error());
    let error = owners
        .word
        .read_after(&owners.event)
        .expect_err("terminal word refuses read_after");
    assert_eq!(error, terminal_error());
    let Err(error) = ctx.stream() else {
        panic!("terminal context must refuse a new stream");
    };
    assert_eq!(error, terminal_error());
}

/// A pack and a matmul with valid ranges and a valid stream are refused by the
/// latch alone, before any argument check could reach the native layer.
fn refuse_launches(ctx: &CudaContext, owners: &Owners, stack: &CudaWeight) {
    let CudaWeight::Tiled { weight, .. } = stack else {
        panic!("the fault child builds a tiled stack");
    };
    let buffers = &owners.launch;
    let xf = buffers.xf.range(0, 1_048_592).expect("xf range");
    let pack = PackCall {
        x: buffers.x.range(0, 1024).expect("x range"),
        xf,
        m: 1,
        k: 256,
    };
    let error = ctx
        .pack(&owners.stream, pack)
        .expect_err("terminal context refuses pack");
    assert_eq!(error, terminal_error());
    let projection = Projection {
        weight,
        first_row: 0,
        rows: 8,
    };
    let io = MatmulIo {
        xf,
        y: buffers.y.range(0, 32).expect("y range"),
        m: 1,
    };
    let error = ctx
        .matmul(&owners.stream, projection, io)
        .expect_err("terminal context refuses matmul");
    assert_eq!(error, terminal_error());
}

/// Dropping owners after the latch must neither panic nor make a native
/// destroy or free call, and each skipped owner stays counted.
fn drop_after_terminal(buffer: DeviceBuffer, stack: CudaWeight, owners: Owners, ctx: CudaContext) {
    let attempts = ctx.destroy_calls();
    let live = ctx.live_owners();
    drop(buffer);
    drop(stack);
    drop(owners);
    assert_eq!(
        ctx.destroy_calls(),
        attempts,
        "no native destroy after the latch"
    );
    assert_eq!(ctx.live_owners(), live, "skipped owners stay counted");
    assert!(
        TERMINAL.load(Ordering::Acquire),
        "drop must not clear terminal"
    );
    drop(ctx);
}
