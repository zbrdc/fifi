//! Safe CUDA kernels for Linux hosts with an NVIDIA GPU.
//!
//! The parent module declares this module under
//! `#[cfg(all(feature = "native-cuda", target_os = "linux"))]`, the same gate
//! that `build.rs` and `tests/cuda_smoke.rs` use. Packet 1 covers smoke
//! operations only: device count and f32 vector add. Raw native calls live in
//! `native` and its child modules, the only Rust files here that contain `unsafe`
//! or `extern "C"` (decision 0047).
//!
//! Failure rule: `cudaErrorMemoryAllocation` at the allocate stage is a
//! `MemoryPlan` error and leaves the module usable. A host with no device is a
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
//! Packet 3 adds resident device memory and the tiled matmul in `context`.
//! Each `CudaContext` holds one of two process leases and stays on its
//! creating thread; every native call runs inside the process dispatch guard
//! of `guard` (decisions 0023, 0024 and 0040 item 4).
//!
//! P6-workbuf: each `CudaContext` owns one activation workspace planned by
//! `arena::plan_arena`, so a linear call allocates no device memory.
//!
//! RD-2a (decision 0047): `context` adds non-blocking streams, timing-free events
//! and a pinned result word, and the launch-only `pack` and `matmul` entries on a
//! stream; `native/resident.rs` holds their raw calls. Nothing outside the tests
//! calls them until the resident step (RD-9a), which also takes the arena planner
//! (`plan_arena`) and the `xf` size rule (`xf_bytes`) re-exported here.
//!
//! RD-2c (decision 0047): `context` adds `allocate` and the stream-ordered ranged
//! reads and writes of a buffer, a generic `DeviceArena` with checked views and the
//! `ArenaHold` that keeps the workspace from being replaced under a graph. The
//! caller sizes the workspace with an `Admission`; the model's numbers are not here.
//!
//! RD-2d (decision 0047): `context` adds the capture window, executable graphs (`GraphExec`), the
//! refusal of every non-capturable call inside a window and deferred drops; `native/resident/graph.rs`
//! holds the raw calls. The operation name `cuda capture` names a capture refusal on the closed list
//! (the seven `cudaErrorStreamCapture*` statuses 900 to 906, and an allocation failure when the
//! capture is instantiated); such a refusal leaves the stream clean and the backend usable. Every
//! other status stays terminal.
//!
//! `resident` holds the safe wrappers of the stage 1 kernels, one file per kernel (RD-2p). Its doc
//! says how to add one.

mod arena;
mod context;
#[cfg(all(test, feature = "cuda-fault-qualification"))]
mod fault;
mod guard;
mod native;
pub(crate) mod plan;
pub(crate) mod resident;
#[cfg(test)]
mod testkit;

pub(crate) use context::{
    Admission, CHUNK_LIMIT, CudaContext, DeviceBuffer, Growth, MemoryInfo, NativeWeight, PinnedBuf,
    QuantTensor, TiledWeight, WorkspaceShape, check_q5_0_scales, check_q6k_scales,
    check_q8_0_scales, drop_page_cache,
};
#[cfg_attr(
    not(test),
    expect(
        unused_imports,
        reason = "staged for RD-9a (decision 0047): the resident step is the first user"
    )
)]
pub(crate) use context::{
    ArenaHold, CapturePlan, DeviceArena, DeviceEvent, DeviceRange, DeviceStream, ExpertBank,
    ExpertBankGeometry, GraphExec, MatmulIo, PackCall, PinnedWord, Projection, WordValue,
    is_capture_refusal,
};
pub(crate) use {
    arena::{ArenaPlan, ArenaTensor, plan_arena},
    native::xf_bytes,
};

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
    ensure_usable(&TERMINAL)?;
    let count = native::device_count()
        .map_err(|status| native_error(&TERMINAL, count_operation(status), status))?;
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
    // The zero fill is host work a later packet can avoid with an
    // uninitialized buffer. It keeps this smoke path free of extra unsafe.
    out.resize(lhs.len(), 0.0);
    native::vector_add_f32(lhs, rhs, &mut out).map_err(|failure| {
        native_error(&TERMINAL, stage_operation(failure.stage), failure.status)
    })?;
    Ok(out)
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

/// The seven `cudaErrorStreamCapture*` statuses (REPLY-30 item 1, decision 0047). The one definition
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
