//! Raw FFI to the Metal bridge in `bridge.mm`.
//!
//! `unsafe` and `extern "C"` live only in this file and its child modules. Functions here are
//! private to the Metal module. Each one checks every length and count the native side relies
//! on for memory safety before the native call. Callers hold the dispatch guard
//! (`super::guard`) around every call and have checked the terminal flag.
//!
//! A handle is an object the bridge retains for Rust under automatic reference counting, and
//! each is released once by [`release`] unless the backend went terminal first. Safety
//! invariants: the pointer of a [`RawBuffer`] is a live buffer of `bytes` bytes whose storage
//! stays valid and host-visible until it is released; the pointer of a [`RawKernel`] is a live
//! compute pipeline state of `max_threads` threads a threadgroup at most; the pointer of a
//! [`RawRuntime`] carries the device and the queue every launch is encoded on.
#![allow(unsafe_code)]

use std::ffi::{CStr, c_char, c_int, c_void};

/// The bridge found an argument it refuses.
pub(super) const STATUS_INVALID: c_int = 1;
/// The device is missing. Not terminal: the runtime itself reported no failure.
pub(super) const STATUS_NO_DEVICE: c_int = 4;
/// The command buffer did not complete.
pub(super) const STATUS_COMMAND_BUFFER: c_int = 3;
/// A guarded body raised an exception.
pub(super) const STATUS_EXCEPTION: c_int = 100;

/// The most buffers one dispatch binds: below Metal's argument limit, the slot after them holds
/// the parameters.
pub(super) const MAX_BUFFERS: usize = 30;

/// The most parameter bytes one dispatch passes outside a buffer.
pub(super) const MAX_PARAMETER_BYTES: usize = 4096;

pub(super) struct RawRuntime {
    pointer: *mut c_void,
}

pub(super) struct RawBuffer {
    pointer: *mut c_void,
    bytes: usize,
}

pub(super) struct RawKernel {
    pointer: *mut c_void,
    max_threads: usize,
}

/// One buffer a dispatch binds, at `offset` bytes into it.
#[derive(Clone, Copy)]
pub(super) struct Binding {
    pub(super) pointer: *mut c_void,
    pub(super) offset: usize,
}

impl RawRuntime {
    pub(super) fn pointer(&self) -> *mut c_void {
        self.pointer
    }
}

impl RawBuffer {
    pub(super) fn pointer(&self) -> *mut c_void {
        self.pointer
    }

    pub(super) fn bytes(&self) -> usize {
        self.bytes
    }
}

impl RawKernel {
    pub(super) fn max_threads(&self) -> usize {
        self.max_threads
    }
}

unsafe extern "C" {
    #[link_name = "inference_metal_last_error"]
    fn bridge_last_error(buffer: *mut c_char, capacity: usize) -> c_int;
    #[link_name = "inference_metal_runtime_new"]
    fn bridge_runtime_new(out: *mut *mut c_void) -> c_int;
    #[link_name = "inference_metal_object_free"]
    fn bridge_object_free(object: *mut c_void);
    #[link_name = "inference_metal_buffer_new"]
    fn bridge_buffer_new(runtime: *mut c_void, bytes: usize, out: *mut *mut c_void) -> c_int;
    #[link_name = "inference_metal_buffer_contents"]
    fn bridge_buffer_contents(buffer: *mut c_void) -> *mut u8;
    #[link_name = "inference_metal_kernel_new"]
    fn bridge_kernel_new(
        runtime: *mut c_void,
        source: *const c_char,
        name: *const c_char,
        out: *mut *mut c_void,
        max_threads: *mut usize,
    ) -> c_int;
    #[link_name = "inference_metal_run"]
    fn bridge_run(
        runtime: *mut c_void,
        kernel: *mut c_void,
        buffers: *const *mut c_void,
        offsets: *const usize,
        count: usize,
        bytes: *const c_void,
        bytes_len: usize,
        grid: *const usize,
        per_threadgroup: *const usize,
        repeats: usize,
        gpu_ns: *mut u64,
    ) -> c_int;
}

/// The text of the failure of the last native call on this thread.
pub(super) fn last_error() -> String {
    let mut buffer = [0 as c_char; 512];
    unsafe { bridge_last_error(buffer.as_mut_ptr(), buffer.len()) };
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

pub(super) fn runtime_new() -> Result<RawRuntime, c_int> {
    let mut out: *mut c_void = std::ptr::null_mut();
    let status = unsafe { bridge_runtime_new(&mut out) };
    if status != 0 {
        return Err(status);
    }
    if out.is_null() {
        return Err(STATUS_INVALID);
    }
    Ok(RawRuntime { pointer: out })
}

/// Releases a handle the bridge retained. A null handle is nothing.
pub(super) fn release(handle: *mut c_void) {
    if !handle.is_null() {
        unsafe { bridge_object_free(handle) };
    }
}

pub(super) fn buffer_new(runtime: &RawRuntime, bytes: usize) -> Result<RawBuffer, c_int> {
    if bytes == 0 {
        return Err(STATUS_INVALID);
    }
    let mut out: *mut c_void = std::ptr::null_mut();
    let status = unsafe { bridge_buffer_new(runtime.pointer, bytes, &mut out) };
    if status != 0 {
        return Err(status);
    }
    if out.is_null() {
        return Err(STATUS_INVALID);
    }
    Ok(RawBuffer {
        pointer: out,
        bytes,
    })
}

fn inside(buffer: &RawBuffer, offset: usize, len: usize) -> bool {
    offset
        .checked_add(len)
        .is_some_and(|end| end <= buffer.bytes)
}

/// The host address of a buffer's storage, the same for the buffer's life.
pub(super) fn contents(buffer: &RawBuffer) -> *mut u8 {
    // SAFETY: the pointer is a live buffer, the invariant of this module, and `contents`
    // raises no exception.
    unsafe { bridge_buffer_contents(buffer.pointer) }
}

/// Copies `data` into the buffer at `offset`.
pub(super) fn buffer_write(buffer: &RawBuffer, offset: usize, data: &[u8]) -> Result<(), c_int> {
    if !inside(buffer, offset, data.len()) {
        return Err(STATUS_INVALID);
    }
    if !data.is_empty() {
        // SAFETY: `inside` proved `data.len()` bytes of the buffer start at `offset`, and no
        // command buffer is in flight: every launch of this module waits.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                bridge_buffer_contents(buffer.pointer).add(offset),
                data.len(),
            );
        }
    }
    Ok(())
}

/// Copies `bytes` bytes of the buffer from `offset` on.
pub(super) fn buffer_read(
    buffer: &RawBuffer,
    offset: usize,
    bytes: usize,
) -> Result<Vec<u8>, c_int> {
    if !inside(buffer, offset, bytes) {
        return Err(STATUS_INVALID);
    }
    let mut out = vec![0u8; bytes];
    if bytes != 0 {
        // SAFETY: as `buffer_write`; the source is `bytes` long and distinct from `out`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bridge_buffer_contents(buffer.pointer).add(offset),
                out.as_mut_ptr(),
                bytes,
            );
        }
    }
    Ok(out)
}

pub(super) fn kernel_new(
    runtime: &RawRuntime,
    source: &CStr,
    name: &CStr,
) -> Result<RawKernel, c_int> {
    let mut out: *mut c_void = std::ptr::null_mut();
    let mut max_threads = 0usize;
    let status = unsafe {
        bridge_kernel_new(
            runtime.pointer,
            source.as_ptr(),
            name.as_ptr(),
            &mut out,
            &mut max_threads,
        )
    };
    if status != 0 {
        return Err(status);
    }
    if out.is_null() || max_threads == 0 {
        release(out);
        return Err(STATUS_INVALID);
    }
    Ok(RawKernel {
        pointer: out,
        max_threads,
    })
}

/// Encodes `repeats` dispatches of `kernel` back to back, binds `bindings` and then `bytes`
/// as the parameters, commits the command buffer and waits for it. Returns the GPU time of the
/// command buffer in nanoseconds, 0 when the device reports none.
pub(super) fn run(
    runtime: &RawRuntime,
    kernel: &RawKernel,
    bindings: &[Binding],
    bytes: &[u8],
    grid: [usize; 3],
    per_threadgroup: [usize; 3],
    repeats: usize,
) -> Result<u64, c_int> {
    let threads = product(grid);
    let group = product(per_threadgroup);
    let invalid = bindings.is_empty()
        || bindings.len() > MAX_BUFFERS
        || bytes.is_empty()
        || bytes.len() > MAX_PARAMETER_BYTES
        || repeats == 0
        || threads.is_none_or(|n| n == 0 || n > u32::MAX as usize)
        || group.is_none_or(|n| n == 0 || n > kernel.max_threads)
        || bindings
            .iter()
            .any(|b| b.pointer.is_null() || b.offset % 4 != 0);
    if invalid {
        return Err(STATUS_INVALID);
    }
    let pointers: Vec<*mut c_void> = bindings.iter().map(|b| b.pointer).collect();
    let offsets: Vec<usize> = bindings.iter().map(|b| b.offset).collect();
    let mut gpu_ns = 0u64;
    let status = unsafe {
        bridge_run(
            runtime.pointer,
            kernel.pointer,
            pointers.as_ptr(),
            offsets.as_ptr(),
            pointers.len(),
            bytes.as_ptr().cast(),
            bytes.len(),
            grid.as_ptr(),
            per_threadgroup.as_ptr(),
            repeats,
            &mut gpu_ns,
        )
    };
    if status != 0 {
        return Err(status);
    }
    Ok(gpu_ns)
}

fn product(axes: [usize; 3]) -> Option<usize> {
    axes[0].checked_mul(axes[1])?.checked_mul(axes[2])
}
