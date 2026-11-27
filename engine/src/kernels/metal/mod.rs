//! Safe Metal kernels for Apple silicon Macs.
//!
//! The parent module declares this module under
//! `#[cfg(all(feature = "native-metal", target_os = "macos", target_arch = "aarch64"))`, the
//! gate `build.rs` uses for the Objective-C++ bridge. The feature pulls in no crate: this module
//! builds and runs without MLX. Raw native calls live in `native` and its child modules, the
//! only Rust files here that contain `unsafe` or `extern "C"`.
//!
//! Failure rule, the rule of the CUDA adapter: a host with no supported device is a
//! non-terminal `Backend` error from kernel calls. Every other native failure is `Backend`
//! and terminal for the process, because the failed call's completion is unknown. Later
//! calls then return `Backend` with the operation `execute` and make no native call.
//!
//! Operation names: `metal initialize` for the device and its queue, `metal allocate` for a
//! buffer, `metal compile` for a shader, `metal launch` for an encoded dispatch, `metal
//! synchronize` for a command buffer that failed, and `metal adapter` for a failure inside the
//! adapter itself (rejected argument, caught exception).
//!
//! Memory is unified: a buffer's storage is the same memory to the host and the GPU, so there
//! is no copy stage. The host may touch a buffer only while no command buffer that uses it is
//! in flight. A launch waits for its command buffer. A frame (`frame`) keeps one command buffer
//! ahead, and the context waits for it before a host read or write of a buffer it binds.
// The module is staged: no family calls it until the Inkling 2 Mini decode step moves here.
#![allow(
    dead_code,
    unused_imports,
    reason = "staged until a family's decode step runs on this runtime"
)]

mod context;
mod frame;
mod guard;
mod native;
pub(crate) mod resident;
#[cfg(test)]
mod testkit;

pub(crate) use context::{DeviceBuffer, DeviceRange, MetalContext};
pub(crate) use resident::dense_gemv::{
    DenseDtype, DenseGemvBytes, DenseGemvCall, DenseGemvShape, dense_gemv_bytes,
};
pub(crate) use resident::rms_norm::RmsNormCall;
pub(crate) use resident::row_add::RowAddCall;
pub(crate) use resident::softcap::SoftcapCall;
pub(crate) use resident::swiglu_rows::{SwigluRowsCall, SwigluRowsShape};

use std::sync::atomic::{AtomicBool, Ordering};

use crate::EngError;

const INITIALIZE_OPERATION: &str = "metal initialize";
const ALLOCATE_OPERATION: &str = "metal allocate";
const COMPILE_OPERATION: &str = "metal compile";
const LAUNCH_OPERATION: &str = "metal launch";
const SYNCHRONIZE_OPERATION: &str = "metal synchronize";
const ADAPTER_OPERATION: &str = "metal adapter";
/// The operation of every refusal after a native failure.
const EXECUTE_OPERATION: &str = "execute";

static TERMINAL: AtomicBool = AtomicBool::new(false);

fn ensure_usable(terminal: &AtomicBool) -> Result<(), EngError> {
    if terminal.load(Ordering::Acquire) {
        return Err(EngError::Backend {
            operation: EXECUTE_OPERATION,
            message: "Metal backend is terminal after an earlier native failure".to_owned(),
        });
    }
    Ok(())
}

/// Latches the backend terminal and names the operation of the failed native call.
fn terminal_failure(operation: &'static str) -> EngError {
    TERMINAL.store(true, Ordering::Release);
    let detail = native::last_error();
    EngError::Backend {
        operation,
        message: if detail.is_empty() {
            "the Metal bridge failed without a message".to_owned()
        } else {
            detail
        },
    }
}
