//! First device-buffer drop after an unmapped real CUDA fault: the drop latches the backend
//! terminal and keeps the allocation.

use super::{
    CHILD_ENV, CUDA_ERROR_ILLEGAL_ADDRESS, CudaContext, DispatchGuard, EngError, Ordering,
    TERMINAL, assert_child_finishes, device_present, native,
};

const CHILD: &str = "kernels::cuda::fault::device_drop::device_buffer_drop_fault_child";

struct DropFault {
    fault: Result<(), std::ffi::c_int>,
    terminal_before: bool,
    terminal_after: bool,
    native_before: (u64, u64),
    native_after: (u64, u64),
    bytes_before: usize,
    bytes_after: usize,
}

fn device_drop_observations() -> Result<DropFault, EngError> {
    let ctx = CudaContext::new()?;
    let buffer = ctx.allocate(256)?;
    let bytes_before = ctx.live_bytes();
    let fault = {
        let _dispatch = DispatchGuard::enter(&TERMINAL);
        native::fault_illegal_address().map_err(|failure| failure.status)
    };
    let terminal_before = TERMINAL.load(Ordering::Acquire);
    let native_before = native::allocation_calls();
    drop(buffer);
    let native_after = native::allocation_calls();
    let terminal_after = TERMINAL.load(Ordering::Acquire);
    let bytes_after = ctx.live_bytes();
    Ok(DropFault {
        fault,
        terminal_before,
        terminal_after,
        native_before,
        native_after,
        bytes_before,
        bytes_after,
    })
}

#[test]
fn first_device_buffer_drop_retains_native_allocation_in_child() {
    if !device_present("first_device_buffer_drop_retains_native_allocation_in_child") {
        return;
    }
    assert_child_finishes(CHILD);
}

#[test]
fn device_buffer_drop_fault_child() {
    if !std::env::var(CHILD_ENV).is_ok_and(|value| value == "1") {
        println!("SKIPPED device_buffer_drop_fault_child: not a fault child");
        return;
    }
    let observed = device_drop_observations().expect("device drop fault observations");
    assert_eq!(observed.fault, Err(CUDA_ERROR_ILLEGAL_ADDRESS));
    assert!(
        !observed.terminal_before,
        "the fault was deliberately not mapped"
    );
    assert!(
        observed.terminal_after,
        "the first failed device drop latches terminal"
    );
    assert_eq!(
        observed.native_after, observed.native_before,
        "a failed completion proof must not attempt cudaFree"
    );
    assert_eq!(
        observed.bytes_after, observed.bytes_before,
        "unreleased device storage remains counted"
    );
    println!("fault child: done");
}
