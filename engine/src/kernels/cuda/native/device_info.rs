//! Raw FFI to the device-property query of `native/device_info.cu` (declared in
//! `native/abi_device.h`). Child module of `native`, so it inherits the
//! `allow(unsafe_code)` of `native.rs` (decision 0047, child modules of `native.rs`).
//!
//! Safety invariants every wrapper here relies on:
//! - The one extern is host only: it reads a device attribute and allocates, copies and
//!   launches nothing.
//! - The out pointer is a live, writable `c_int` for the whole call and the adapter keeps no
//!   pointer after it returns.
//! - No raw pointer and no unsafe function leaves this module.
//! - The call runs under the process dispatch guard on the thread that owns the context
//!   (decisions 0023 and 0024); the caller in `mod.rs` enters the guard.

use std::ffi::c_int;

use super::STATUS_OK;

unsafe extern "C" {
    #[link_name = "inference_cuda_device_integrated"]
    fn device_integrated_raw(out_integrated: *mut c_int) -> c_int;
}

/// Whether the current device shares memory with the host, or the native status.
pub(in crate::kernels::cuda) fn device_integrated() -> Result<bool, c_int> {
    let mut value: c_int = 0;
    // SAFETY: `value` is a live, writable c_int for the whole call; the adapter
    // keeps no pointer after it returns and touches no other memory.
    let status = unsafe { device_integrated_raw(&mut value) };
    if status == STATUS_OK {
        Ok(value != 0)
    } else {
        Err(status)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::super::STATUS_INVALID_ARGUMENT;
    use super::*;
    use crate::kernels::cuda::testkit::device_present;

    #[test]
    fn the_integrated_attribute_matches_the_tegra_release_file() {
        if !device_present("the_integrated_attribute_matches_the_tegra_release_file") {
            return;
        }
        // The oracle is the L4T release file, independent of the attribute: it
        // exists on a Jetson and not on a discrete-GPU host or in the container.
        assert_eq!(
            device_integrated().expect("integrated query"),
            Path::new("/etc/nv_tegra_release").exists()
        );
    }

    #[test]
    fn a_null_out_pointer_is_an_invalid_argument() {
        // SAFETY: the adapter checks the out pointer before any CUDA call and
        // writes nothing through a null pointer.
        let status = unsafe { device_integrated_raw(std::ptr::null_mut()) };
        assert_eq!(status, STATUS_INVALID_ARGUMENT);
    }
}
