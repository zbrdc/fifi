//! Safe CUDA kernels for Linux hosts with an NVIDIA GPU.
//!
//! The parent module declares this module under
//! `#[cfg(all(feature = "native-cuda", target_os = "linux"))]`, the same gate
//! that `build.rs` and `tests/cuda_smoke.rs` use. The smoke operations are the
//! device count and an f32 vector add. Raw native calls live in `native` and its
//! child modules, the only Rust files here that contain `unsafe` or `extern "C"`.
//!
//! Failure rule: `cudaErrorMemoryAllocation` at the allocate stage is a
//! `MemoryPlan` error when cleanup succeeds. A failed smoke free is terminal,
//! even after an allocation OOM. A host with no device is a
//! non-terminal `Backend` error from kernel calls. Every other native failure,
//! including the allocation status at a later stage, is `Backend` and terminal
//! for the process, because a CUDA context may be poisoned and the failed
//! call's completion is unknown. Later calls then return `Backend` with the
//! operation `execute` and make no native call.
//!
//! Operation names: `cuda initialize` for the runtime and device query,
//! `cuda allocate`, `cuda copy`, `cuda launch` and `cuda synchronize` for the
//! stage of a device call that failed first, and `cuda adapter` for a failure
//! inside the adapter itself (rejected argument, caught C++ exception).
//!
//! `context` holds resident device memory and the tiled matmul.
//! Each `CudaContext` holds one of two process leases and stays on its
//! creating thread; every native call runs inside the process dispatch guard
//! of `guard`.
//!
//! Each `CudaContext` owns one activation workspace planned by
//! `arena::plan_arena`, so a linear call allocates no device memory.
//!
//! `context` also has non-blocking streams, timing-free events and a pinned
//! result word, and the launch-only `pack` and `matmul` entries on a stream;
//! `native/resident.rs` holds their raw calls. Nothing outside the tests calls
//! them until the resident decode step, which also takes the arena planner
//! (`plan_arena`) and the `xf` size rule (`xf_bytes`) re-exported here.
//!
//! `context` also has `allocate` and the stream-ordered ranged reads and writes of
//! a buffer, a generic `DeviceArena` with checked views and the `ArenaHold` that
//! keeps the workspace from being replaced under a graph. The caller sizes the
//! workspace with an `Admission`; the model's numbers are not here.
//!
//! `context` also has the capture window, executable graphs (`GraphExec`), the refusal of every
//! non-capturable call inside a window and deferred drops; `native/resident/graph.rs` holds the
//! raw calls. The operation name `cuda capture` names a capture refusal on the closed list (the
//! seven `cudaErrorStreamCapture*` statuses 900 to 906, and an allocation failure when the
//! capture is instantiated); such a refusal leaves the stream clean and the backend usable.
//! Every other status stays terminal.
//!
//! `resident` holds the safe wrappers of the resident-decode kernels, one file per kernel. Its
//! doc says how to add one.

mod arena;
mod context;
#[cfg(all(test, feature = "cuda-fault-qualification"))]
mod fault;
#[cfg(all(test, feature = "cuda-fault-qualification"))]
pub(crate) use fault::{
    assert_child_finishes as assert_cuda_fault_child_finishes, test_inject_unmapped_fault,
};
mod guard;
mod native;
pub(crate) mod plan;
pub(crate) mod prefill;
pub(crate) mod resident;
#[cfg(test)]
mod testkit;

pub(crate) use arena::ARENA_ALIGN;
pub(crate) use context::{
    Admission, CHUNK_LIMIT, CudaContext, DeviceBuffer, Growth, MemoryInfo, NativeWeight,
    Nvfp4WeightRange, PINNED_STAGE_LIMIT_BYTES, PinnedBuf, QuantTensor, TiledWeight,
    WorkspaceShape, check_q5_0_scales, check_q6k_scales, check_q8_0_scales, drop_page_cache,
};
#[cfg_attr(
    not(test),
    expect(
        unused_imports,
        reason = "staged for the resident decode step, its first user"
    )
)]
pub(crate) use context::{
    ArenaHold, CapturePlan, DeviceArena, DeviceEvent, DeviceRange, DeviceStream, ExpertBank,
    ExpertBankGeometry, GraphExec, MatmulIo, PackCall, PinnedWord, Projection, TimingEvent,
    WordValue, is_capture_refusal,
};
#[cfg(test)]
pub(crate) use context::{Nvfp4GroupedPlan, Nvfp4LaunchStatus};
pub(crate) use {
    arena::{ArenaPlan, ArenaTensor, plan_arena},
    native::xf_bytes,
};

/// The kernels' prefill status mask, re-exported for the family book:
/// `status_error` maps it to `prefill_chunk`, and its tests pin that mapping against this one
/// const. The native tree stays private; the mask is its public contract.
#[cfg_attr(
    not(test),
    expect(
        unused_imports,
        reason = "the book's tests read it today; the prefill driver will too"
    )
)]
pub(crate) use native::prefill::PREFILL_STATUS_BOUND_VIOLATION;

// The mamba step launch wrappers and the error mapper were removed with
// the decommissioned device-chain driver; the CUDA kernels themselves
// remain in native/mamba_step.cu.

/// The device gate every CUDA device test shares, which also skips on a host without a
/// driver: the family test rigs reach `testkit::device_present` through this path, the
/// `testkit` module being private to `kernels::cuda`.
#[cfg(test)]
pub(crate) use testkit::device_present;

use std::ffi::c_int;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::EngError;

const INITIALIZE_OPERATION: &str = "cuda initialize";
const MEMORY_INFO_OPERATION: &str = "cuda memory info";
const ALLOCATE_OPERATION: &str = "cuda allocate";
const COPY_OPERATION: &str = "cuda copy";
const LAUNCH_OPERATION: &str = "cuda launch";
const SYNCHRONIZE_OPERATION: &str = "cuda synchronize";
const ADAPTER_OPERATION: &str = "cuda adapter";
const CAPTURE_OPERATION: &str = "cuda capture";
const EXECUTE_OPERATION: &str = "execute";

/// `cudaErrorMemoryAllocation` from `cuda_runtime_api.h`.
const CUDA_ERROR_MEMORY_ALLOCATION: c_int = 2;

/// Set after the first terminal native failure in this process.
static TERMINAL: AtomicBool = AtomicBool::new(false);

/// Returns the number of CUDA devices the runtime can use.
///
/// A host without a GPU returns `Ok(0)`. A host whose driver is missing, too
/// old for the runtime, or failing returns `EngError::Backend`.
pub fn device_count() -> Result<usize, EngError> {
    guarded_device_count(
        &TERMINAL,
        || guard::DispatchGuard::enter(&TERMINAL),
        native::device_count,
    )
}

fn guarded_device_count<'a>(
    terminal: &'a AtomicBool,
    enter: impl FnOnce() -> guard::DispatchGuard<'a>,
    query: impl FnOnce() -> Result<c_int, c_int>,
) -> Result<usize, EngError> {
    let _dispatch = enter();
    ensure_usable(terminal)?;
    let count =
        query().map_err(|status| native_error(terminal, count_operation(status), status))?;
    usable_device_count(count)
}

/// Whether the first device shares memory with the host (a Jetson), from
/// `cudaDevAttrIntegrated`. Callers hold a context, so a device exists; a
/// terminal backend refuses the query like every other call.
pub(crate) fn device_is_integrated() -> Result<bool, EngError> {
    ensure_usable(&TERMINAL)?;
    let _dispatch = guard::DispatchGuard::enter(&TERMINAL);
    ensure_usable(&TERMINAL)?;
    native::device_info::device_integrated()
        .map_err(|status| native_error(&TERMINAL, count_operation(status), status))
}

/// Adds two `f32` slices element by element on the first CUDA device.
///
/// Returns `EngError::BadRequest` when the lengths differ. Empty inputs
/// return an empty vector and skip the device call. Returns
/// `EngError::MemoryPlan` when the host or device cannot hold the buffers and
/// `EngError::Backend` when no device is present or the device fails.
pub fn vector_add_f32(lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, EngError> {
    if lhs.len() != rhs.len() {
        return Err(EngError::BadRequest(format!(
            "vector add lengths differ: {} and {}",
            lhs.len(),
            rhs.len()
        )));
    }
    if lhs.is_empty() {
        return Ok(Vec::new());
    }
    require_device(device_count()?)?;
    let mut out = Vec::new();
    out.try_reserve_exact(lhs.len()).map_err(|error| {
        EngError::MemoryPlan(format!(
            "{ALLOCATE_OPERATION}: host output of {} floats: {error}",
            lhs.len()
        ))
    })?;
    // The zero fill is host work a later change can avoid with an
    // uninitialized buffer. It keeps this smoke path free of extra unsafe.
    out.resize(lhs.len(), 0.0);
    let _dispatch = guard::DispatchGuard::enter(&TERMINAL);
    ensure_usable(&TERMINAL)?;
    native::vector_add_f32(lhs, rhs, &mut out)
        .map_err(|failure| smoke_error(&TERMINAL, failure))?;
    Ok(out)
}

/// Keeps cleanup failure terminal even when the first allocation OOM is recoverable.
fn smoke_error(terminal: &AtomicBool, error: native::SmokeFailure) -> EngError {
    let failure = error.failure;
    let operation = stage_operation(failure.stage);
    if error.cleanup_status == native::STATUS_OK {
        return native_error(terminal, operation, failure.status);
    }
    terminal.store(true, Ordering::Release);
    EngError::Backend {
        operation,
        message: format!(
            "{} (status {}). Cleanup free failed: {} (status {})",
            native::status_message(failure.status),
            failure.status,
            native::status_message(error.cleanup_status),
            error.cleanup_status
        ),
    }
}

/// Names the failed stage of a device call. Adapter-level failures report no
/// stage.
fn stage_operation(stage: c_int) -> &'static str {
    match stage {
        native::STAGE_ALLOCATE => ALLOCATE_OPERATION,
        native::STAGE_COPY => COPY_OPERATION,
        native::STAGE_LAUNCH => LAUNCH_OPERATION,
        native::STAGE_SYNCHRONIZE => SYNCHRONIZE_OPERATION,
        native::resident::graph::STAGE_CAPTURE => CAPTURE_OPERATION,
        _ => ADAPTER_OPERATION,
    }
}

/// Names a failed device query. Negative statuses come from the adapter, the
/// others from the runtime.
fn count_operation(status: c_int) -> &'static str {
    if status < 0 {
        ADAPTER_OPERATION
    } else {
        INITIALIZE_OPERATION
    }
}

/// Names a failed free-memory query: negative statuses come from the adapter,
/// the others from `cudaMemGetInfo`.
fn memory_info_operation(status: c_int) -> &'static str {
    if status < 0 {
        ADAPTER_OPERATION
    } else {
        MEMORY_INFO_OPERATION
    }
}

/// Converts the native count. The adapter never returns a negative count, so a
/// negative value means the runtime misbehaved and must not reach `usize`.
fn usable_device_count(count: c_int) -> Result<usize, EngError> {
    usize::try_from(count).map_err(|_| EngError::Backend {
        operation: INITIALIZE_OPERATION,
        message: format!("runtime reported negative device count {count}"),
    })
}

/// Stops before any kernel launch when no device is present. The error is not
/// terminal, because the runtime itself reported no failure.
fn require_device(count: usize) -> Result<(), EngError> {
    if count == 0 {
        return Err(EngError::Backend {
            operation: INITIALIZE_OPERATION,
            message: "no CUDA device".to_owned(),
        });
    }
    Ok(())
}

/// Refuses every call after a terminal failure.
fn ensure_usable(terminal: &AtomicBool) -> Result<(), EngError> {
    if terminal.load(Ordering::Acquire) {
        return Err(EngError::Backend {
            operation: EXECUTE_OPERATION,
            message: "CUDA backend is terminal after an earlier native failure".to_owned(),
        });
    }
    Ok(())
}

/// Maps a native status to its error variant and latches terminal failures.
fn native_error(terminal: &AtomicBool, operation: &'static str, status: c_int) -> EngError {
    let message = format!("{} (status {status})", native::status_message(status));
    if operation == ALLOCATE_OPERATION && status == CUDA_ERROR_MEMORY_ALLOCATION {
        return EngError::MemoryPlan(format!("{operation}: {message}"));
    }
    if operation == CAPTURE_OPERATION && is_capture_status(status) {
        return EngError::Backend { operation, message };
    }
    terminal.store(true, Ordering::Release);
    EngError::Backend { operation, message }
}

/// The seven `cudaErrorStreamCapture*` statuses. The one definition
/// of the closed list: [`is_capture_status`] and the capture window both use it.
const CAPTURE_STATUSES: std::ops::RangeInclusive<c_int> = 900..=906;

/// The closed list of capture statuses that do not latch the backend: [`CAPTURE_STATUSES`], and an
/// allocation failure at the capture stage (instantiate).
fn is_capture_status(status: c_int) -> bool {
    CAPTURE_STATUSES.contains(&status) || status == CUDA_ERROR_MEMORY_ALLOCATION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_device_query_does_not_call_native() {
        let terminal = AtomicBool::new(true);
        let queried = std::cell::Cell::new(false);
        let error = guarded_device_count(
            &terminal,
            || guard::DispatchGuard::enter(&terminal),
            || {
                queried.set(true);
                Ok(1)
            },
        )
        .expect_err("a terminal query must be refused");
        assert!(matches!(
            error,
            EngError::Backend {
                operation: EXECUTE_OPERATION,
                ..
            }
        ));
        assert!(!queried.get());
    }

    #[test]
    fn device_query_rechecks_terminal_after_waiting_for_dispatch() {
        let terminal = AtomicBool::new(false);
        let queried = std::sync::atomic::AtomicUsize::new(0);
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let dispatch = guard::DispatchGuard::enter(&terminal);
        let (entered, result) = std::thread::scope(|scope| {
            let terminal_ref = &terminal;
            let queried_ref = &queried;
            let worker = scope.spawn(move || {
                guarded_device_count(
                    terminal_ref,
                    move || {
                        entered_tx.send(()).expect("notify dispatch attempt");
                        guard::DispatchGuard::enter(terminal_ref)
                    },
                    || {
                        queried_ref.fetch_add(1, Ordering::Relaxed);
                        Ok(1)
                    },
                )
            });
            let entered = entered_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .is_ok();
            terminal.store(true, Ordering::Release);
            drop(dispatch);
            (entered, worker.join())
        });
        assert!(entered, "the query must attempt dispatch acquisition");
        let error = result
            .expect("query worker must not panic")
            .expect_err("terminal state must be checked after dispatch acquisition");
        assert!(matches!(
            error,
            EngError::Backend {
                operation: EXECUTE_OPERATION,
                ..
            }
        ));
        assert_eq!(queried.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn device_query_failure_latches_local_terminal_state() {
        let terminal = AtomicBool::new(false);
        let error = guarded_device_count(
            &terminal,
            || guard::DispatchGuard::enter(&terminal),
            || Err(999),
        )
        .expect_err("a failed query must return its error");
        assert!(matches!(
            error,
            EngError::Backend {
                operation: INITIALIZE_OPERATION,
                ..
            }
        ));
        assert!(terminal.load(Ordering::Acquire));
    }

    #[test]
    fn non_negative_device_count_passes_through() {
        assert_eq!(usable_device_count(2).expect("two devices"), 2);
    }

    #[test]
    fn negative_device_count_is_a_backend_error() {
        let error = usable_device_count(-1).expect_err("negative count must fail");
        assert_eq!(
            error,
            EngError::Backend {
                operation: INITIALIZE_OPERATION,
                message: "runtime reported negative device count -1".to_owned(),
            }
        );
    }

    #[test]
    fn zero_devices_fail_before_any_kernel_launch() {
        let error = require_device(0).expect_err("zero devices must fail");
        assert_eq!(
            error,
            EngError::Backend {
                operation: INITIALIZE_OPERATION,
                message: "no CUDA device".to_owned(),
            }
        );
        require_device(1).expect("one device is enough");
    }

    #[test]
    fn stages_map_to_operation_names() {
        assert_eq!(stage_operation(native::STAGE_ALLOCATE), "cuda allocate");
        assert_eq!(stage_operation(native::STAGE_COPY), "cuda copy");
        assert_eq!(stage_operation(native::STAGE_LAUNCH), "cuda launch");
        assert_eq!(
            stage_operation(native::STAGE_SYNCHRONIZE),
            "cuda synchronize"
        );
        assert_eq!(stage_operation(native::STAGE_NONE), "cuda adapter");
        assert_eq!(stage_operation(99), "cuda adapter");
    }

    #[test]
    fn device_query_failures_name_their_source() {
        assert_eq!(
            count_operation(native::STATUS_INVALID_ARGUMENT),
            "cuda adapter"
        );
        assert_eq!(count_operation(-2), "cuda adapter");
        assert_eq!(count_operation(35), "cuda initialize");
    }

    #[test]
    fn memory_info_failures_have_their_own_operation() {
        assert_eq!(
            memory_info_operation(native::STATUS_INVALID_ARGUMENT),
            "cuda adapter"
        );
        assert_eq!(memory_info_operation(2), "cuda memory info");
        assert_eq!(memory_info_operation(35), "cuda memory info");
    }

    #[test]
    fn native_failure_is_terminal_backend_error() {
        let terminal = AtomicBool::new(false);
        let error = native_error(&terminal, LAUNCH_OPERATION, native::STATUS_INVALID_ARGUMENT);
        assert_eq!(
            error,
            EngError::Backend {
                operation: LAUNCH_OPERATION,
                message: "invalid argument (status -1)".to_owned(),
            }
        );
        assert!(terminal.load(Ordering::Acquire));
        assert_eq!(
            ensure_usable(&terminal).expect_err("terminal backend refuses calls"),
            EngError::Backend {
                operation: EXECUTE_OPERATION,
                message: "CUDA backend is terminal after an earlier native failure".to_owned(),
            }
        );
    }

    #[test]
    fn device_allocation_failure_is_a_memory_plan_error() {
        let terminal = AtomicBool::new(false);
        let error = native_error(&terminal, ALLOCATE_OPERATION, CUDA_ERROR_MEMORY_ALLOCATION);
        let EngError::MemoryPlan(message) = error else {
            panic!("expected MemoryPlan, got {error:?}");
        };
        assert!(message.starts_with("cuda allocate: "), "{message}");
        assert!(message.ends_with("(status 2)"), "{message}");
        assert!(!terminal.load(Ordering::Acquire));
        ensure_usable(&terminal).expect("allocation failure is not terminal");
    }

    #[test]
    fn smoke_allocation_oom_with_successful_cleanup_remains_recoverable() {
        let terminal = AtomicBool::new(false);
        let error = smoke_error(
            &terminal,
            native::SmokeFailure {
                failure: native::DeviceFailure {
                    status: CUDA_ERROR_MEMORY_ALLOCATION,
                    stage: native::STAGE_ALLOCATE,
                },
                cleanup_status: native::STATUS_OK,
            },
        );
        assert!(matches!(error, EngError::MemoryPlan(_)), "{error:?}");
        assert!(!terminal.load(Ordering::Acquire));
        ensure_usable(&terminal).expect("successful cleanup permits allocation recovery");
    }

    #[test]
    fn smoke_allocation_oom_with_failed_cleanup_is_terminal() {
        for cleanup_status in [CUDA_ERROR_MEMORY_ALLOCATION, 999] {
            let terminal = AtomicBool::new(false);
            let error = smoke_error(
                &terminal,
                native::SmokeFailure {
                    failure: native::DeviceFailure {
                        status: CUDA_ERROR_MEMORY_ALLOCATION,
                        stage: native::STAGE_ALLOCATE,
                    },
                    cleanup_status,
                },
            );
            let EngError::Backend { operation, message } = error else {
                panic!("cleanup failure must be Backend, got {error:?}");
            };
            assert_eq!(operation, ALLOCATE_OPERATION);
            assert!(
                message.contains("(status 2). Cleanup free failed: "),
                "{message}"
            );
            assert!(
                message.ends_with(&format!("(status {cleanup_status})")),
                "{message}"
            );
            assert!(terminal.load(Ordering::Acquire));
            assert!(ensure_usable(&terminal).is_err());
        }
    }

    #[test]
    fn smoke_cleanup_failure_preserves_the_first_copy_cause() {
        let terminal = AtomicBool::new(false);
        let error = smoke_error(
            &terminal,
            native::SmokeFailure {
                failure: native::DeviceFailure {
                    status: 1,
                    stage: native::STAGE_COPY,
                },
                cleanup_status: 999,
            },
        );
        let EngError::Backend { operation, message } = error else {
            panic!("cleanup failure must be Backend, got {error:?}");
        };
        assert_eq!(operation, COPY_OPERATION);
        assert!(
            message.contains("(status 1). Cleanup free failed: "),
            "{message}"
        );
        assert!(message.ends_with("(status 999)"), "{message}");
        assert!(terminal.load(Ordering::Acquire));
    }

    #[test]
    fn allocation_status_after_the_allocate_stage_is_terminal() {
        let terminal = AtomicBool::new(false);
        let error = native_error(&terminal, COPY_OPERATION, CUDA_ERROR_MEMORY_ALLOCATION);
        assert!(
            matches!(
                error,
                EngError::Backend {
                    operation: COPY_OPERATION,
                    ..
                }
            ),
            "{error:?}"
        );
        assert!(terminal.load(Ordering::Acquire));
    }
}
