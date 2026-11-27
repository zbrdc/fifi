//! Device memory behind one Metal adapter lease.
//!
//! A [`MetalContext`] holds one of the two process leases. Every buffer it creates shares that
//! lease, so the slot stays reserved until the context and all its buffers are gone. Context and
//! buffers stay on the creating thread: none of them is `Send`. Each native call runs inside the
//! process dispatch guard.
//!
//! Memory is unified, so a buffer is host memory the GPU reads: [`MetalContext::upload`] fills
//! a new buffer with no copy stage, and a [`DeviceBuffer`] reads and writes through the same
//! storage. The host may touch a buffer only while no command buffer that uses it is in
//! flight. A launch waits for its command buffer; a frame keeps one command buffer ahead of the
//! one the GPU runs, and the context waits for the command buffers in flight before a host read
//! or write of a buffer one of them binds.
//!
//! A failed free latches the backend terminal. Drop never panics. After a terminal failure no
//! native call runs, so the remaining buffers are left to the process exit.
//!
//! Every public operation first reports a terminal failure as `Backend` `execute`, then
//! validates its input, then enters the dispatch guard.

use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    ffi::{CString, c_int, c_void},
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::EngError;

use super::frame::{BUFFERS_IN_FLIGHT, DISPATCHES_PER_BUFFER, Frame};
use super::guard::{DispatchGuard, Lease, SLOTS};
use super::native::{self, Binding, RawBuffer, RawCommand, RawKernel, RawRuntime};
use super::{
    ADAPTER_OPERATION, ALLOCATE_OPERATION, COMPILE_OPERATION, INITIALIZE_OPERATION,
    LAUNCH_OPERATION, SYNCHRONIZE_OPERATION, TERMINAL, ensure_usable, terminal_failure,
};

/// The threads of each axis of a dispatch.
pub(crate) type Grid = [usize; 3];

/// The non-finite weight flag of the dense matvec, one `u32`.
const FLAG_BYTES: usize = 4;
/// The flag cleared: no weight marked.
const FLAG_ZERO: [u8; 4] = [0; 4];
/// The shader of the first read of the pages of a mapped file.
const PAGE_TOUCH_SOURCE: &str = include_str!("native/page_touch.metal");
const PAGE_TOUCH_KERNEL: &str = "page_touch";
/// The stride of that read: below any page size, so every page is read.
const PAGE_READ_STRIDE: usize = 4096;
const PAGE_TOUCH_THREADGROUP: usize = 256;

pub(crate) struct MetalContext {
    pub(super) shared: Rc<Shared>,
    lease: Rc<Lease>,
}

/// What a context and its buffers keep alive together: the runtime, the accounting and the
/// compiled shaders.
pub(super) struct Shared {
    runtime: RawRuntime,
    live_bytes: AtomicUsize,
    live_buffers: AtomicUsize,
    /// The compiled shaders, by name and source.
    kernels: RefCell<HashMap<(&'static str, &'static str), Rc<RawKernel>>>,
    /// The non-finite weight flag of the dense matvec, made at first use.
    flag: RefCell<Option<RawBuffer>>,
    /// The frame that carries the flag, while one does.
    flag_frame: Cell<Option<u64>>,
    /// The id of the next frame.
    next_frame_id: Cell<u64>,
    /// The command buffers a frame committed and did not wait for, oldest first.
    in_flight: RefCell<Vec<InFlight>>,
    /// The GPU time of the command buffers waited before a frame's drain.
    settled_gpu_ns: Cell<u64>,
}

/// A command buffer a frame committed and did not wait for, with the handles of the buffers
/// it binds, each once.
struct InFlight {
    command: RawCommand,
    bound: Vec<*mut c_void>,
}

pub(crate) struct DeviceBuffer {
    raw: RawBuffer,
    shared: Rc<Shared>,
    _lease: Rc<Lease>,
    /// A buffer over the pages of a mapped file: the host cannot write it and a frame cannot
    /// fill it.
    read_only: bool,
}

/// The `bytes` bytes of a [`DeviceBuffer`] from `offset` on.
#[derive(Clone, Copy)]
pub(crate) struct DeviceRange<'b> {
    buffer: &'b DeviceBuffer,
    offset: usize,
    bytes: usize,
}

impl std::fmt::Debug for MetalContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalContext")
            .field("live_bytes", &self.live_bytes())
            .field("live_buffers", &self.live_buffers())
            .finish()
    }
}

impl MetalContext {
    /// Opens the default device and its queue and takes a process lease for them. A host with
    /// no device is a `Backend` error that does not make the backend terminal.
    pub(crate) fn new() -> Result<Self, EngError> {
        let lease = Lease::acquire(&SLOTS, &TERMINAL)?;
        let runtime = {
            let _guard = DispatchGuard::enter(&TERMINAL);
            native::runtime_new().map_err(|status| {
                if status == native::STATUS_NO_DEVICE {
                    EngError::Backend {
                        operation: INITIALIZE_OPERATION,
                        message: "no supported Metal device".to_owned(),
                    }
                } else {
                    terminal_failure(INITIALIZE_OPERATION)
                }
            })?
        };
        let shared = Rc::new(Shared {
            runtime,
            live_bytes: AtomicUsize::new(0),
            live_buffers: AtomicUsize::new(0),
            kernels: RefCell::new(HashMap::new()),
            flag: RefCell::new(None),
            flag_frame: Cell::new(None),
            next_frame_id: Cell::new(0),
            in_flight: RefCell::new(Vec::new()),
            settled_gpu_ns: Cell::new(0),
        });
        Ok(Self { shared, lease })
    }

    /// The identity of this context, for the check that a range belongs to it.
    pub(super) fn id(&self) -> usize {
        self.shared.id()
    }

    pub(crate) fn live_bytes(&self) -> usize {
        self.shared.live_bytes.load(Ordering::Acquire)
    }

    pub(crate) fn live_buffers(&self) -> usize {
        self.shared.live_buffers.load(Ordering::Acquire)
    }

    /// A new buffer that holds a copy of `bytes`.
    pub(crate) fn upload(&self, bytes: &[u8]) -> Result<DeviceBuffer, EngError> {
        if bytes.is_empty() {
            return Err(EngError::BadRequest(
                "a Metal buffer needs at least one byte".to_owned(),
            ));
        }
        ensure_usable(&TERMINAL)?;
        let raw = {
            let _guard = DispatchGuard::enter(&TERMINAL);
            native::buffer_new(&self.shared.runtime, bytes.len())
                .and_then(|raw| native::buffer_write(&raw, 0, bytes).map(|()| raw))
                .map_err(|_| terminal_failure(ALLOCATE_OPERATION))?
        };
        self.shared.note(&raw);
        Ok(DeviceBuffer {
            raw,
            shared: Rc::clone(&self.shared),
            _lease: Rc::clone(&self.lease),
            read_only: false,
        })
    }

    /// The longest buffer the device gives, in bytes.
    pub(super) fn max_buffer_length(&self) -> Result<usize, EngError> {
        ensure_usable(&TERMINAL)?;
        let _guard = DispatchGuard::enter(&TERMINAL);
        native::max_buffer_length(&self.shared.runtime)
            .map_err(|_| terminal_failure(ALLOCATE_OPERATION))
    }

    /// The first read of every page of `buffer`, the whole pages of a mapped file: one word
    /// every 4 KiB, one thread each, a waited launch. A dispatch that comes after it finds the
    /// pages mapped for the GPU. The GPU time of the pass in nanoseconds; `BadRequest` for a
    /// buffer the host can write or an empty one, before any native call.
    pub(super) fn read_pages(&self, buffer: &DeviceBuffer) -> Result<u64, EngError> {
        ensure_usable(&TERMINAL)?;
        if !buffer.read_only {
            return Err(EngError::BadRequest(
                "the buffer is not over the pages of a mapped file".to_owned(),
            ));
        }
        let bytes = buffer.raw.bytes();
        if bytes == 0 {
            return Err(EngError::BadRequest(
                "an empty buffer has no page to read".to_owned(),
            ));
        }
        let threads = bytes.div_ceil(PAGE_READ_STRIDE);
        let kernel = self.kernel(PAGE_TOUCH_KERNEL, PAGE_TOUCH_SOURCE)?;
        let touched = self.upload(&vec![0u8; 4 * threads])?;
        self.launch_repeated(
            &kernel,
            &[
                buffer.range(0, bytes)?.binding(),
                touched.range(0, 4 * threads)?.binding(),
            ],
            &(bytes as u64).to_le_bytes(),
            [threads, 1, 1],
            [PAGE_TOUCH_THREADGROUP, 1, 1],
            1,
        )
    }

    /// A buffer with no copy over the whole pages of `storage`, which the caller keeps mapped
    /// and read-only until the buffer is released: the pages of a file `family::MappedFile`
    /// maps. `BadRequest` for a storage without a whole page, which refused nothing: the
    /// backend stays usable.
    pub(super) fn wrap_mapped(&self, storage: &[u8]) -> Result<DeviceBuffer, EngError> {
        ensure_usable(&TERMINAL)?;
        let raw = {
            let _guard = DispatchGuard::enter(&TERMINAL);
            native::buffer_wrap(&self.shared.runtime, storage).map_err(|status| {
                if status == native::STATUS_INVALID {
                    EngError::BadRequest(
                        "a Metal buffer over mapped pages needs at least one whole page".to_owned(),
                    )
                } else {
                    terminal_failure(ALLOCATE_OPERATION)
                }
            })?
        };
        self.shared.note(&raw);
        Ok(DeviceBuffer {
            raw,
            shared: Rc::clone(&self.shared),
            _lease: Rc::clone(&self.lease),
            read_only: true,
        })
    }

    /// The compiled shader `name` of `source`, compiled once per context.
    pub(super) fn kernel(
        &self,
        name: &'static str,
        source: &'static str,
    ) -> Result<Rc<RawKernel>, EngError> {
        self.shared.kernel(name, source)
    }

    /// Encodes `repeats` dispatches of `kernel` back to back and waits for the command buffer:
    /// its GPU time in nanoseconds, 0 when the device reports none. `bytes` are the kernel's
    /// parameters, bound after `bindings`.
    pub(super) fn launch_repeated(
        &self,
        kernel: &RawKernel,
        bindings: &[Binding],
        bytes: &[u8],
        grid: Grid,
        per_threadgroup: Grid,
        repeats: usize,
    ) -> Result<u64, EngError> {
        ensure_usable(&TERMINAL)?;
        let _guard = DispatchGuard::enter(&TERMINAL);
        native::run(
            &self.shared.runtime,
            kernel,
            bindings,
            bytes,
            grid,
            per_threadgroup,
            repeats,
        )
        .map_err(launch_failure)
    }

    /// A frame on this context: `DISPATCHES_PER_BUFFER` dispatches a command buffer, one
    /// command buffer ahead, one wait in [`Frame::drain`].
    pub(super) fn frame(&self) -> Frame {
        Frame::new(Rc::clone(&self.shared), DISPATCHES_PER_BUFFER)
    }

    /// A frame of `per_buffer` dispatches a command buffer, for the tests.
    #[cfg(test)]
    pub(super) fn frame_of(&self, per_buffer: usize) -> Frame {
        Frame::new(Rc::clone(&self.shared), per_buffer)
    }
}

impl Shared {
    pub(super) fn runtime(&self) -> &RawRuntime {
        &self.runtime
    }

    /// The identity of this context, for the check that a range belongs to it.
    pub(super) fn id(&self) -> usize {
        self as *const Self as usize
    }

    /// The compiled shader `name` of `source`, compiled once per context.
    pub(super) fn kernel(
        &self,
        name: &'static str,
        source: &'static str,
    ) -> Result<Rc<RawKernel>, EngError> {
        ensure_usable(&TERMINAL)?;
        let key = (name, source);
        if let Some(kernel) = self.kernels.borrow().get(&key) {
            return Ok(Rc::clone(kernel));
        }
        let name_c = CString::new(name)
            .map_err(|_| EngError::BadRequest("a shader name holds a null byte".to_owned()))?;
        let source_c = CString::new(source)
            .map_err(|_| EngError::BadRequest("a shader source holds a null byte".to_owned()))?;
        let kernel = {
            let _guard = DispatchGuard::enter(&TERMINAL);
            // Another thread may have compiled it while this one waited for the guard.
            if let Some(kernel) = self.kernels.borrow().get(&key) {
                return Ok(Rc::clone(kernel));
            }
            Rc::new(
                native::kernel_new(&self.runtime, &source_c, &name_c)
                    .map_err(|_| terminal_failure(COMPILE_OPERATION))?,
            )
        };
        self.kernels.borrow_mut().insert(key, Rc::clone(&kernel));
        Ok(kernel)
    }

    /// Waits for the oldest command buffer in flight and releases it. Its GPU time in
    /// nanoseconds is kept for the next [`Self::take_settled`]. Nothing in flight is nothing.
    pub(super) fn wait_oldest(&self) -> Result<(), EngError> {
        if self.in_flight.borrow().is_empty() {
            return Ok(());
        }
        let _guard = DispatchGuard::enter(&TERMINAL);
        let oldest = self.in_flight.borrow_mut().remove(0);
        let outcome = native::wait(&oldest.command).map_err(launch_failure);
        native::release(oldest.command.pointer());
        if let Ok(gpu_ns) = outcome {
            self.settled_gpu_ns.set(self.settled_gpu_ns.get() + gpu_ns);
        }
        outcome.map(|_| ())
    }

    /// The GPU time of the command buffers waited since the last drain, in nanoseconds.
    pub(super) fn take_settled(&self) -> u64 {
        self.settled_gpu_ns.replace(0)
    }

    /// Commits `command`, which binds `bound`. At most `BUFFERS_IN_FLIGHT` command buffers are
    /// in flight: the one the GPU runs and one ahead of it. A further commit waits for the
    /// oldest first, so the GPU takes the next buffer from the queue, without a stop.
    pub(super) fn commit_ahead(
        &self,
        command: RawCommand,
        bound: Vec<*mut c_void>,
    ) -> Result<(), EngError> {
        ensure_usable(&TERMINAL)?;
        let _guard = DispatchGuard::enter(&TERMINAL);
        while self.in_flight.borrow().len() >= BUFFERS_IN_FLIGHT {
            self.wait_oldest()?;
        }
        native::commit(&command).map_err(launch_failure)?;
        self.in_flight
            .borrow_mut()
            .push(InFlight { command, bound });
        Ok(())
    }

    /// Waits for the command buffers in flight that bind `buffer`, and for those ahead of
    /// them. The host touches a buffer only after this.
    pub(super) fn settle(&self, buffer: *mut c_void) -> Result<(), EngError> {
        let binds = |entry: &InFlight| entry.bound.contains(&buffer);
        if TERMINAL.load(Ordering::Acquire) || !self.in_flight.borrow().iter().any(binds) {
            return Ok(());
        }
        let _guard = DispatchGuard::enter(&TERMINAL);
        while self.in_flight.borrow().iter().any(binds) {
            self.wait_oldest()?;
        }
        Ok(())
    }

    /// Waits for every command buffer in flight.
    pub(super) fn wait_all(&self) -> Result<(), EngError> {
        if TERMINAL.load(Ordering::Acquire) {
            return Ok(());
        }
        let _guard = DispatchGuard::enter(&TERMINAL);
        while !self.in_flight.borrow().is_empty() {
            self.wait_oldest()?;
        }
        Ok(())
    }

    /// The command buffers in flight, for the tests.
    #[cfg(test)]
    pub(super) fn in_flight_count(&self) -> usize {
        self.in_flight.borrow().len()
    }

    fn note(&self, raw: &RawBuffer) {
        self.live_bytes.fetch_add(raw.bytes(), Ordering::AcqRel);
        self.live_buffers.fetch_add(1, Ordering::AcqRel);
    }

    /// The non-finite weight flag, made at first use, for `body`.
    fn with_flag<T>(
        &self,
        body: impl FnOnce(&RawBuffer) -> Result<T, EngError>,
    ) -> Result<T, EngError> {
        if self.flag.borrow().is_none() {
            let raw = {
                let _guard = DispatchGuard::enter(&TERMINAL);
                native::buffer_new(&self.runtime, FLAG_BYTES)
                    .map_err(|_| terminal_failure(ALLOCATE_OPERATION))?
            };
            self.note(&raw);
            *self.flag.borrow_mut() = Some(raw);
        }
        let held = self.flag.borrow();
        let flag = held
            .as_ref()
            .expect("the flag exists after the check above");
        body(flag)
    }

    /// The id of a new frame of this context.
    pub(super) fn next_frame_id(&self) -> u64 {
        let id = self.next_frame_id.get();
        self.next_frame_id.set(id + 1);
        id
    }

    /// Gives the flag to frame `id`. A context's flag is one a frame: a second frame is refused,
    /// because its clear would erase the marks of the first.
    pub(super) fn frame_takes_flag(&self, id: u64) -> Result<(), EngError> {
        if self.flag_frame.get().is_some_and(|holder| holder != id) {
            return Err(adapter_bug(
                "two frames of one Metal context carry the non-finite weight flag",
            ));
        }
        self.flag_frame.set(Some(id));
        Ok(())
    }

    /// The flag is no longer frame `id`'s.
    pub(super) fn release_flag(&self, id: u64) {
        if self.flag_frame.get() == Some(id) {
            self.flag_frame.set(None);
        }
    }

    /// Clears the flag. A host write: the command buffers that bind the flag are waited for
    /// first, and under the rule above none is in flight.
    pub(super) fn clear_flag(&self) -> Result<(), EngError> {
        self.with_flag(|flag| {
            self.settle(flag.pointer())?;
            native::buffer_write(flag, 0, &FLAG_ZERO)
                .map_err(|_| adapter_bug("the write that clears the non-finite weight flag"))
        })
    }

    /// The flag's address, the last binding of a dispatch of the dense matvec.
    pub(super) fn flag_pointer(&self) -> Result<*mut c_void, EngError> {
        self.with_flag(|flag| Ok(flag.pointer()))
    }

    /// Reads the flag after the wait of a frame's drain: a mark is a weight that is not finite.
    pub(super) fn check_flag(&self) -> Result<(), EngError> {
        let set = self.with_flag(|flag| {
            self.settle(flag.pointer())?;
            native::buffer_read(flag, 0, FLAG_ZERO.len())
                .map_err(|_| adapter_bug("the read of the non-finite weight flag"))
        })?;
        if set != FLAG_ZERO {
            return Err(EngError::BadRequest(
                "the Metal dense matvec read a weight that is not finite".to_owned(),
            ));
        }
        Ok(())
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // After a terminal failure no native call runs; the process exit reclaims the objects.
        if TERMINAL.load(Ordering::Acquire) {
            return;
        }
        let _guard = DispatchGuard::enter(&TERMINAL);
        if let Some(flag) = self.flag.get_mut().take() {
            native::release(flag.pointer());
        }
        native::release(self.runtime.pointer());
    }
}

/// A refusal from inside this module, where the caller's input passed the checks: a bug of the
/// adapter, reported without making the backend terminal.
pub(super) fn adapter_bug(what: &str) -> EngError {
    EngError::Backend {
        operation: ADAPTER_OPERATION,
        message: format!("{what}: refused inside the Metal module, an error of the adapter"),
    }
}

/// The failure of a launch's native call: terminal, named by the call's status.
pub(super) fn launch_failure(status: c_int) -> EngError {
    terminal_failure(match status {
        native::STATUS_INVALID | native::STATUS_EXCEPTION => ADAPTER_OPERATION,
        native::STATUS_COMMAND_BUFFER => SYNCHRONIZE_OPERATION,
        _ => LAUNCH_OPERATION,
    })
}

impl DeviceBuffer {
    pub(crate) fn bytes(&self) -> usize {
        self.raw.bytes()
    }

    /// The buffer's raw handle, for the fill a frame encodes.
    pub(super) fn raw(&self) -> &RawBuffer {
        &self.raw
    }

    /// A buffer over a file's pages takes no host write and no fill.
    fn check_writable(&self) -> Result<(), EngError> {
        if self.read_only {
            return Err(EngError::BadRequest(
                "a Metal buffer over the pages of a file is read-only".to_owned(),
            ));
        }
        Ok(())
    }

    /// The `bytes` bytes of the buffer from `offset` on.
    pub(crate) fn range(&self, offset: usize, bytes: usize) -> Result<DeviceRange<'_>, EngError> {
        let inside = offset
            .checked_add(bytes)
            .is_some_and(|end| end <= self.bytes());
        if bytes == 0 || !inside {
            return Err(EngError::BadRequest(format!(
                "the Metal range of {bytes} bytes at offset {offset} does not fit a buffer of {} bytes",
                self.bytes()
            )));
        }
        Ok(DeviceRange {
            buffer: self,
            offset,
            bytes,
        })
    }

    /// Copies `bytes` bytes of the buffer from `offset` on. A host copy: no native call can
    /// fail.
    pub(crate) fn read(&self, offset: usize, bytes: usize) -> Result<Vec<u8>, EngError> {
        let _ = self.range(offset, bytes)?;
        self.shared.settle(self.raw.pointer())?;
        native::buffer_read(&self.raw, offset, bytes)
            .map_err(|_| adapter_bug("the host read of a Metal buffer"))
    }

    /// Copies `data` into the buffer at `offset`.
    pub(crate) fn write(&self, offset: usize, data: &[u8]) -> Result<(), EngError> {
        self.check_writable()?;
        let _ = self.range(offset, data.len())?;
        self.shared.settle(self.raw.pointer())?;
        native::buffer_write(&self.raw, offset, data)
            .map_err(|_| adapter_bug("the host write into a Metal buffer"))
    }

    /// Encodes a fill of the `bytes` bytes at `offset` with zero into `frame`, behind the work
    /// the frame encoded before: the next work of the frame reads the zeros. The device fills;
    /// the host neither waits nor copies. `BadRequest` for an empty range and for one past
    /// the buffer; a refused call changes nothing.
    pub(super) fn zero_at(
        &self,
        frame: &mut Frame,
        offset: usize,
        bytes: usize,
    ) -> Result<(), EngError> {
        self.check_writable()?;
        frame.zero(&self.range(offset, bytes)?)
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // After a terminal failure no native call runs; the process exit reclaims the buffer.
        if TERMINAL.load(Ordering::Acquire) {
            return;
        }
        // A command buffer of a frame may still bind it; the handle is released after that wait.
        let _ = self.shared.settle(self.raw.pointer());
        let _guard = DispatchGuard::enter(&TERMINAL);
        native::release(self.raw.pointer());
        self.shared
            .live_bytes
            .fetch_sub(self.raw.bytes(), Ordering::AcqRel);
        self.shared.live_buffers.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Identity and text for the tests and the logs: a range is its context, its offset and its
/// length.
impl PartialEq for DeviceRange<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.owner() == other.owner() && self.offset == other.offset && self.bytes == other.bytes
    }
}

impl Eq for DeviceRange<'_> {}

impl std::fmt::Debug for DeviceRange<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceRange")
            .field("offset", &self.offset)
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

impl<'b> DeviceRange<'b> {
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    pub(super) fn offset(&self) -> usize {
        self.offset
    }

    /// The buffer of the range.
    pub(super) fn buffer(&self) -> &DeviceBuffer {
        self.buffer
    }

    /// The identity of the context that owns the buffer.
    pub(super) fn owner(&self) -> usize {
        Rc::as_ptr(&self.buffer.shared) as usize
    }

    /// The first address the range covers, in the buffer's storage.
    pub(super) fn address(&self) -> usize {
        native::contents(&self.buffer.raw) as usize + self.offset
    }

    pub(super) fn binding(&self) -> Binding {
        Binding {
            pointer: self.buffer.raw.pointer(),
            offset: self.offset,
        }
    }

    pub(crate) fn read(&self) -> Result<Vec<u8>, EngError> {
        self.buffer.read(self.offset, self.bytes)
    }

    /// Writes the whole range; `data` must have the range's length.
    pub(crate) fn write(&self, data: &[u8]) -> Result<(), EngError> {
        if data.len() != self.bytes {
            return Err(EngError::BadRequest(format!(
                "the Metal range holds {} bytes, the write carries {}",
                self.bytes,
                data.len()
            )));
        }
        self.buffer.write(self.offset, data)
    }
}
