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
//! [`RawRuntime`] carries the device and the queue every launch is encoded on; the pointer of a
//! [`RawCommand`] is a live command buffer of that queue. The host may touch a buffer only
//! while no command buffer that binds it is in flight, so [`run`] waits for its command buffer
//! before it returns.
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

/// A command buffer, open until [`commit`] takes it.
pub(super) struct RawCommand {
    pointer: *mut c_void,
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

impl RawCommand {
    pub(super) fn pointer(&self) -> *mut c_void {
        self.pointer
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
    #[link_name = "inference_metal_command_new"]
    fn bridge_command_new(runtime: *mut c_void, out: *mut *mut c_void) -> c_int;
    #[link_name = "inference_metal_command_dispatch"]
    fn bridge_command_dispatch(
        command: *mut c_void,
        kernel: *mut c_void,
        buffers: *const *mut c_void,
        offsets: *const usize,
        count: usize,
        bytes: *const c_void,
        bytes_len: usize,
        grid: *const usize,
        per_threadgroup: *const usize,
        repeats: usize,
    ) -> c_int;
    #[link_name = "inference_metal_command_commit"]
    fn bridge_command_commit(command: *mut c_void) -> c_int;
    #[link_name = "inference_metal_command_wait"]
    fn bridge_command_wait(command: *mut c_void, gpu_ns: *mut u64) -> c_int;
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
        // command buffer in flight binds the buffer: a launch waits for its command buffer,
        // and a host touch through the context waits for a frame's first.
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

/// Opens a command buffer on the queue of the runtime. Rust holds it until [`release`].
pub(super) fn command_new(runtime: &RawRuntime) -> Result<RawCommand, c_int> {
    let mut out: *mut c_void = std::ptr::null_mut();
    let status = unsafe { bridge_command_new(runtime.pointer, &mut out) };
    if status != 0 {
        return Err(status);
    }
    if out.is_null() {
        return Err(STATUS_INVALID);
    }
    Ok(RawCommand { pointer: out })
}

/// Encodes `repeats` dispatches of `kernel` back to back into the open command buffer, binds
/// `bindings` and then `bytes` as the parameters, and ends the encoder.
pub(super) fn dispatch(
    command: &RawCommand,
    kernel: &RawKernel,
    bindings: &[Binding],
    bytes: &[u8],
    grid: [usize; 3],
    per_threadgroup: [usize; 3],
    repeats: usize,
) -> Result<(), c_int> {
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
    let status = unsafe {
        bridge_command_dispatch(
            command.pointer,
            kernel.pointer,
            pointers.as_ptr(),
            offsets.as_ptr(),
            pointers.len(),
            bytes.as_ptr().cast(),
            bytes.len(),
            grid.as_ptr(),
            per_threadgroup.as_ptr(),
            repeats,
        )
    };
    if status != 0 {
        return Err(status);
    }
    Ok(())
}

/// Commits the command buffer. A committed one takes no further dispatch.
pub(super) fn commit(command: &RawCommand) -> Result<(), c_int> {
    let status = unsafe { bridge_command_commit(command.pointer) };
    if status != 0 {
        return Err(status);
    }
    Ok(())
}

/// Waits for the command buffer. Returns its GPU time in nanoseconds, 0 when the device
/// reports none.
pub(super) fn wait(command: &RawCommand) -> Result<u64, c_int> {
    let mut gpu_ns = 0u64;
    let status = unsafe { bridge_command_wait(command.pointer, &mut gpu_ns) };
    if status != 0 {
        return Err(status);
    }
    Ok(gpu_ns)
}

/// Opens a command buffer, encodes `repeats` dispatches of `kernel` back to back, binds
/// `bindings` and then `bytes` as the parameters, commits the command buffer and waits for it.
/// Returns the GPU time of the command buffer in nanoseconds, 0 when the device reports none.
/// The handle is released on every path.
pub(super) fn run(
    runtime: &RawRuntime,
    kernel: &RawKernel,
    bindings: &[Binding],
    bytes: &[u8],
    grid: [usize; 3],
    per_threadgroup: [usize; 3],
    repeats: usize,
) -> Result<u64, c_int> {
    let command = command_new(runtime)?;
    let outcome = dispatch(
        &command,
        kernel,
        bindings,
        bytes,
        grid,
        per_threadgroup,
        repeats,
    )
    .and_then(|()| commit(&command))
    .and_then(|()| wait(&command));
    release(command.pointer());
    outcome
}

fn product(axes: [usize; 3]) -> Option<usize> {
    axes[0].checked_mul(axes[1])?.checked_mul(axes[2])
}

#[cfg(test)]
mod tests {
    //! The command buffer of the bridge at this level: opened, filled by [`dispatch`] calls,
    //! committed and waited for, under a lease and the dispatch guard as the context holds
    //! them. The shader is `x[i] = x[i] * 2 + by`, whose result tells two orders of two
    //! dispatches apart. Device tests (Mac); the suite runs with `--test-threads=1`.

    use std::rc::Rc;

    use super::*;
    use crate::kernels::metal::TERMINAL;
    use crate::kernels::metal::guard::{DispatchGuard, Lease, SLOTS};

    const SOURCE: &CStr = c"#include <metal_stdlib>\nusing namespace metal;\nkernel void affine_step(device float *x [[buffer(0)]], constant float &by [[buffer(1)]],\n                                uint i [[thread_position_in_grid]]) {\n  x[i] = x[i] * 2.0f + by;\n}\n";
    const NAME: &CStr = c"affine_step";
    /// The values of `x`. The grid has one thread for each.
    const VALUES: usize = 64;
    const GRID: [usize; 3] = [VALUES, 1, 1];
    const GROUP: [usize; 3] = [32, 1, 1];

    struct Rig {
        runtime: RawRuntime,
        data: RawBuffer,
        kernel: RawKernel,
        // Declared last, so it drops after the handles.
        lease: Rc<Lease>,
    }

    impl Rig {
        fn new() -> Self {
            let lease = Lease::acquire(&SLOTS, &TERMINAL).expect("a lease");
            let _dispatch = DispatchGuard::enter(&TERMINAL);
            let runtime = runtime_new().expect("the runtime");
            let data = buffer_new(&runtime, 4 * VALUES).expect("a buffer");
            let kernel = kernel_new(&runtime, SOURCE, NAME).expect("the shader");
            Self {
                runtime,
                data,
                kernel,
                lease,
            }
        }

        fn binding(&self) -> Binding {
            Binding {
                pointer: self.data.pointer(),
                offset: 0,
            }
        }

        fn encode(&self, command: &RawCommand, by: f32, repeats: usize) -> Result<(), c_int> {
            dispatch(
                command,
                &self.kernel,
                &[self.binding()],
                &by.to_le_bytes(),
                GRID,
                GROUP,
                repeats,
            )
        }

        /// One waited launch through [`run`].
        fn launch(&self, by: f32, repeats: usize) {
            run(
                &self.runtime,
                &self.kernel,
                &[self.binding()],
                &by.to_le_bytes(),
                GRID,
                GROUP,
                repeats,
            )
            .expect("a waited launch");
        }

        fn zero(&self) {
            buffer_write(&self.data, 0, &vec![0u8; 4 * VALUES]).expect("the fill");
        }

        fn values(&self) -> Vec<f32> {
            let bytes = buffer_read(&self.data, 0, 4 * VALUES).expect("the read");
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        }

        /// Releases the handles under the dispatch guard, then the lease.
        fn close(self) {
            let Self {
                lease,
                runtime,
                data,
                kernel,
            } = self;
            {
                let _dispatch = DispatchGuard::enter(&TERMINAL);
                release(kernel.pointer);
                release(data.pointer());
                release(runtime.pointer());
            }
            drop(lease);
        }
    }

    #[test]
    fn dispatches_of_one_command_buffer_equal_the_same_waited_launches() {
        let rig = Rig::new();
        let _dispatch = DispatchGuard::enter(&TERMINAL);
        let command = command_new(&rig.runtime).expect("a command buffer");
        rig.encode(&command, 0.75, 3).expect("three dispatches");
        commit(&command).expect("the commit");
        let gpu_ns = wait(&command).expect("the wait");
        release(command.pointer());
        assert!(gpu_ns > 0, "the device reports the GPU time");
        // 0.75, then 2.25, then 5.25: each is exact in f32.
        assert_eq!(rig.values(), vec![5.25f32; VALUES]);

        rig.zero();
        for _ in 0..3 {
            rig.launch(0.75, 1);
        }
        assert_eq!(rig.values(), vec![5.25f32; VALUES]);
        rig.close();
    }

    #[test]
    fn two_dispatch_calls_run_in_their_order() {
        let rig = Rig::new();
        let _dispatch = DispatchGuard::enter(&TERMINAL);
        let command = command_new(&rig.runtime).expect("a command buffer");
        rig.encode(&command, 1.0, 1).expect("the first dispatch");
        rig.encode(&command, 0.5, 1).expect("the second dispatch");
        commit(&command).expect("the commit");
        wait(&command).expect("the wait");
        release(command.pointer());
        // 0 * 2 + 1, then 1 * 2 + 0.5. The other order gives 2.
        assert_eq!(rig.values(), vec![2.5f32; VALUES]);
        rig.close();
    }

    #[test]
    fn a_committed_command_buffer_takes_no_dispatch_and_no_second_commit() {
        let rig = Rig::new();
        let _dispatch = DispatchGuard::enter(&TERMINAL);
        let command = command_new(&rig.runtime).expect("a command buffer");
        rig.encode(&command, 1.0, 1).expect("the dispatch");
        commit(&command).expect("the commit");
        assert_eq!(rig.encode(&command, 1.0, 1), Err(STATUS_INVALID));
        assert_eq!(commit(&command), Err(STATUS_INVALID));
        wait(&command).expect("the wait");
        release(command.pointer());
        assert_eq!(rig.values(), vec![1.0f32; VALUES]);
        rig.close();
    }

    #[test]
    fn an_open_command_buffer_is_not_waited_for_and_takes_no_empty_dispatch() {
        let rig = Rig::new();
        let _dispatch = DispatchGuard::enter(&TERMINAL);
        let command = command_new(&rig.runtime).expect("a command buffer");
        assert_eq!(wait(&command), Err(STATUS_INVALID));
        assert_eq!(rig.encode(&command, 1.0, 0), Err(STATUS_INVALID));
        release(command.pointer());
        rig.close();
    }
}
