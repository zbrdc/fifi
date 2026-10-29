//! Owning wrappers for native handles, and the resident tensor storage built from them.
//!
//! Every wrapper releases its handle under the dispatch guard. After the owner fails, a wrapper
//! drops without releasing, so the native object stays allocated until process exit.

use std::cell::{Cell, RefCell};
#[cfg(feature = "native-fault-qualification")]
use std::sync::atomic::AtomicU64;
use std::{
    ffi::c_void,
    ptr,
    rc::Rc,
    sync::{
        OnceLock,
        atomic::{AtomicPtr, Ordering},
    },
};

use super::{
    ffi,
    guard::{DispatchGuard, Lease, check, failure, has_failed, mark_failed},
};
#[cfg(feature = "native-fault-qualification")]
use crate::kernels::QualificationFlushStats;
use crate::{
    EngError,
    tensor::{Dtype, OwnerKey, Tensor},
};

static PROCESS_STREAM: OnceLock<AtomicPtr<c_void>> = OnceLock::new();

/// The one stream that every adapter copies. It lives until process exit.
///
/// Only call this after `Stream::for_device` has run, which `Context::try_new` guarantees.
pub(super) fn process_stream() -> ffi::mlx_stream {
    ffi::mlx_stream {
        ctx: PROCESS_STREAM.get().unwrap().load(Ordering::Relaxed),
    }
}

pub(super) struct Device(ffi::mlx_device);

impl Device {
    pub(super) fn try_new() -> Result<Self, EngError> {
        let device = Self(unsafe { ffi::mlx_device_new_type(ffi::mlx_device_type__MLX_GPU, 0) });
        if device.0.ctx.is_null() {
            return Err(failure("device"));
        }
        let mut available = false;
        check(
            unsafe { ffi::mlx_device_is_available(&mut available, device.0) },
            "device availability",
        )?;
        if !available {
            return Err(EngError::Backend {
                operation: "initialize",
                message: "the MLX GPU device is unavailable".into(),
            });
        }
        Ok(device)
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        let _dispatch = DispatchGuard::enter();
        if !has_failed() && unsafe { ffi::mlx_device_free(self.0) } != 0 {
            mark_failed();
        }
    }
}

pub(super) struct Stream(pub(super) ffi::mlx_stream);

impl Stream {
    pub(super) fn for_device(device: &Device) -> Result<Self, EngError> {
        // Retain one cross-thread stream. Cache failure because creation can register before failing.
        let process_stream = PROCESS_STREAM
            .get_or_init(|| {
                AtomicPtr::new(unsafe { ffi::mlx_stream_new_thread_unsafe(device.0) }.ctx)
            })
            .load(Ordering::Relaxed);
        if process_stream.is_null() {
            return Err(failure("process stream initialization (restart required)"));
        }
        let mut stream = Self(ffi::mlx_stream {
            ctx: ptr::null_mut(),
        });
        check(
            unsafe {
                ffi::mlx_stream_set(
                    &mut stream.0,
                    ffi::mlx_stream {
                        ctx: process_stream,
                    },
                )
            },
            "stream copy",
        )?;
        if stream.0.ctx.is_null() {
            return Err(failure("stream copy"));
        }
        // MLX evaluation reads the thread default before selecting the output's explicit stream.
        check(
            unsafe { ffi::mlx_set_default_stream(stream.0) },
            "default stream",
        )?;
        Ok(stream)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let _dispatch = DispatchGuard::enter();
        if !has_failed() && unsafe { ffi::mlx_stream_free(self.0) } != 0 {
            mark_failed();
        }
    }
}

pub(super) struct Array(pub(super) ffi::mlx_array);

impl Array {
    /// An empty slot that a native call fills. Dropping it releases nothing.
    pub(super) fn null() -> Self {
        Self(ffi::NULL_ARRAY)
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        let _dispatch = DispatchGuard::enter();
        if !has_failed() && unsafe { ffi::mlx_array_free(self.0) } != 0 {
            mark_failed();
        }
    }
}

/// Default for the largest retired byte total before a retire forces a flush
/// synchronization. `RetireCore::set_cap` changes it for one adapter.
pub(super) const RETIRE_CAP_BYTES: u64 = 64 << 20;

/// Forced retire-cap flush attempts in this process, counted before the synchronization
/// (qualification builds only).
#[cfg(feature = "native-fault-qualification")]
static FORCED_FLUSHES: AtomicU64 = AtomicU64::new(0);
/// Largest retired byte total of any single adapter since the last reset (qualification builds
/// only).
#[cfg(feature = "native-fault-qualification")]
static PEAK_RETIRED: AtomicU64 = AtomicU64::new(0);

/// Forced retire-cap flushes since the last `qualification_reset_flush_stats`, with their wall
/// time and the retired bytes at each flush (qualification builds only). The counters are
/// process-wide. The `FORCED_FLUSHES` count never resets, so a window uses `FLUSH_COUNT`.
#[cfg(feature = "native-fault-qualification")]
static FLUSH_COUNT: AtomicU64 = AtomicU64::new(0);
/// Nanoseconds inside the forced `mlx_synchronize` calls.
#[cfg(feature = "native-fault-qualification")]
static FLUSH_SYNC_NS: AtomicU64 = AtomicU64::new(0);
/// Nanoseconds that `flush_after_sync` took to free the retired arrays after a forced flush.
#[cfg(feature = "native-fault-qualification")]
static FLUSH_FREE_NS: AtomicU64 = AtomicU64::new(0);
/// Sum of the retired byte total at each forced flush.
#[cfg(feature = "native-fault-qualification")]
static FLUSH_BYTES_SUM: AtomicU64 = AtomicU64::new(0);
/// Minimum of the retired byte total at each forced flush. It starts at `u64::MAX`, so a fetch-min
/// works; the accessor reports zero when no flush ran.
#[cfg(feature = "native-fault-qualification")]
static FLUSH_BYTES_MIN: AtomicU64 = AtomicU64::new(u64::MAX);
/// Maximum of the retired byte total at each forced flush.
#[cfg(feature = "native-fault-qualification")]
static FLUSH_BYTES_MAX: AtomicU64 = AtomicU64::new(0);

/// Returns the forced flush count and the peak retired bytes (qualification builds only).
#[cfg(feature = "native-fault-qualification")]
pub(super) fn qualification_retire_stats() -> (u64, u64) {
    (
        FORCED_FLUSHES.load(Ordering::Relaxed),
        PEAK_RETIRED.load(Ordering::Relaxed),
    )
}

/// Resets the peak retired bytes to zero (qualification builds only).
#[cfg(feature = "native-fault-qualification")]
pub(super) fn qualification_reset_retire_peak() {
    PEAK_RETIRED.store(0, Ordering::Relaxed);
}

/// Reads the forced flush window (qualification builds only).
#[cfg(feature = "native-fault-qualification")]
pub(super) fn qualification_flush_stats() -> QualificationFlushStats {
    let flushes = FLUSH_COUNT.load(Ordering::Relaxed);
    QualificationFlushStats {
        flushes,
        sync_ns: FLUSH_SYNC_NS.load(Ordering::Relaxed),
        free_ns: FLUSH_FREE_NS.load(Ordering::Relaxed),
        bytes_sum: FLUSH_BYTES_SUM.load(Ordering::Relaxed),
        bytes_min: if flushes == 0 {
            0
        } else {
            FLUSH_BYTES_MIN.load(Ordering::Relaxed)
        },
        bytes_max: FLUSH_BYTES_MAX.load(Ordering::Relaxed),
    }
}

/// Sets the forced flush window to zero (qualification builds only).
#[cfg(feature = "native-fault-qualification")]
pub(super) fn qualification_reset_flush_stats() {
    for value in [
        &FLUSH_COUNT,
        &FLUSH_SYNC_NS,
        &FLUSH_FREE_NS,
        &FLUSH_BYTES_SUM,
        &FLUSH_BYTES_MAX,
    ] {
        value.store(0, Ordering::Relaxed);
    }
    FLUSH_BYTES_MIN.store(u64::MAX, Ordering::Relaxed);
}

/// Adds one forced flush attempt: the retired `total` at the flush and the wall time of its
/// synchronization (qualification builds only).
#[cfg(feature = "native-fault-qualification")]
fn record_flush_sync(total: u64, elapsed: std::time::Duration) {
    FLUSH_COUNT.fetch_add(1, Ordering::Relaxed);
    FLUSH_SYNC_NS.fetch_add(
        u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
    FLUSH_BYTES_SUM.fetch_add(total, Ordering::Relaxed);
    FLUSH_BYTES_MIN.fetch_min(total, Ordering::Relaxed);
    FLUSH_BYTES_MAX.fetch_max(total, Ordering::Relaxed);
}

/// Adds the wall time that a forced flush spent freeing its arrays (qualification builds only).
#[cfg(feature = "native-fault-qualification")]
fn record_flush_free(elapsed: std::time::Duration) {
    FLUSH_FREE_NS.fetch_add(
        u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
}

/// Arrays that residents released, waiting for a completed synchronization.
///
/// One core exists per adapter generation. `State` and every `Resident` hold a share. The core
/// holds the lease, so the adapter slot stays held while any retired array exists.
pub(crate) struct RetireCore {
    arrays: RefCell<Vec<Array>>,
    bytes: Cell<u64>,
    cap: Cell<u64>,
    lease: Option<Rc<Lease>>,
}

impl RetireCore {
    pub(super) fn new(lease: &Rc<Lease>) -> Rc<Self> {
        Rc::new(Self {
            arrays: RefCell::new(Vec::new()),
            bytes: Cell::new(0),
            cap: Cell::new(RETIRE_CAP_BYTES),
            lease: Some(Rc::clone(lease)),
        })
    }

    /// Sets the largest retired byte total before a retire forces a flush and returns the
    /// previous cap. A lower cap takes effect at the next retire; it does not flush at once.
    pub(super) fn set_cap(&self, bytes: u64) -> u64 {
        self.cap.replace(bytes)
    }

    /// Frees every retired array. The caller holds the dispatch guard and has just completed a
    /// successful synchronization, so every retired array's work is complete.
    pub(super) fn flush_after_sync(&self) {
        let arrays = std::mem::take(&mut *self.arrays.borrow_mut());
        self.bytes.set(0);
        drop(arrays);
    }

    /// Takes a released array. The caller holds the dispatch guard.
    fn retire(&self, array: Array, byte_len: u64) {
        self.arrays.borrow_mut().push(array);
        let total = self.bytes.get().saturating_add(byte_len);
        self.bytes.set(total);
        #[cfg(feature = "native-fault-qualification")]
        PEAK_RETIRED.fetch_max(total, Ordering::Relaxed);
        if total > self.cap.get() {
            #[cfg(feature = "native-fault-qualification")]
            FORCED_FLUSHES.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "native-fault-qualification")]
            let started = std::time::Instant::now();
            let status = unsafe { ffi::mlx_synchronize(process_stream()) };
            #[cfg(feature = "native-fault-qualification")]
            record_flush_sync(total, started.elapsed());
            if status != 0 {
                mark_failed();
                std::mem::forget(std::mem::take(&mut *self.arrays.borrow_mut()));
                return;
            }
            #[cfg(feature = "native-fault-qualification")]
            let started = std::time::Instant::now();
            self.flush_after_sync();
            #[cfg(feature = "native-fault-qualification")]
            record_flush_free(started.elapsed());
        }
    }
}

impl Drop for RetireCore {
    fn drop(&mut self) {
        let _dispatch = DispatchGuard::enter();
        let arrays = std::mem::take(self.arrays.get_mut());
        if has_failed()
            || (!arrays.is_empty() && unsafe { ffi::mlx_synchronize(process_stream()) } != 0)
        {
            mark_failed();
            std::mem::forget(arrays);
            if let Some(lease) = self.lease.take() {
                std::mem::forget(lease);
            }
            return;
        }
        drop(arrays);
        drop(self.lease.take());
    }
}

/// Native storage behind one `ExecutionTensor`. On drop it moves its array to the retire list.
pub(crate) struct Resident {
    array: Option<Array>,
    dtype: Dtype,
    shape: Vec<u32>,
    byte_len: u64,
    owner: OwnerKey,
    retire: Option<Rc<RetireCore>>,
}

impl Resident {
    /// Wraps an output, lazy or evaluated. The caller has checked that `array` matches the metadata.
    pub(super) fn new(
        array: Array,
        dtype: Dtype,
        shape: Vec<u32>,
        byte_len: u64,
        owner: OwnerKey,
        retire: &Rc<RetireCore>,
    ) -> Self {
        Self {
            array: Some(array),
            dtype,
            shape,
            byte_len,
            owner,
            retire: Some(Rc::clone(retire)),
        }
    }

    pub(super) fn native_array(&self) -> ffi::mlx_array {
        self.array.as_ref().unwrap().0
    }

    /// Takes the native handle out of this resident without freeing or retiring it (for
    /// `kv_append`). The caller owns the taken handle and releases it once, right after the
    /// consuming graph node is built: the node holds the array alive until the GPU completes,
    /// so the retire list never needs it. A later drop of this resident retires nothing.
    pub(super) fn take_array(&mut self) -> Option<Array> {
        self.array.take()
    }

    pub(crate) fn dtype(&self) -> Dtype {
        self.dtype
    }
    pub(crate) fn shape(&self) -> &[u32] {
        &self.shape
    }
    pub(crate) fn byte_len(&self) -> u64 {
        self.byte_len
    }
    pub(crate) fn owner_key(&self) -> OwnerKey {
        self.owner
    }

    /// Evaluates the lazy array under the backend label `evaluate`. `readback` calls this first.
    /// The caller synchronizes the stream afterwards.
    pub(super) fn evaluate(&self) -> Result<(), EngError> {
        check(
            unsafe { ffi::mlx_array_eval(self.array.as_ref().unwrap().0) },
            "evaluate",
        )
    }

    /// Copies the array into portable host bytes. The caller has evaluated the array
    /// and synchronized the stream, so an array that an earlier lazy call built is complete here.
    ///
    /// An array that is not row-contiguous — the strided view a `slice`, `transpose`, `repeat`,
    /// or `broadcast` can return — is made contiguous HERE, at the read: one `mlx_contiguous`
    /// node, evaluated and synchronized on `stream`, and its bytes are what the read returns.
    /// `readback` therefore no longer fails on a strided array; a failure of the copy, its
    /// evaluation, or its synchronization follows the terminal readback rule.
    pub(super) fn readback(&self, stream: ffi::mlx_stream) -> Result<Tensor, EngError> {
        let array = self.array.as_ref().unwrap().0;
        let mut row_contiguous = false;
        check(
            unsafe { ffi::_mlx_array_is_row_contiguous(&mut row_contiguous, array) },
            "readback layout",
        )?;
        if row_contiguous {
            return self.read_array(array);
        }
        let mut contiguous = Array::null();
        check(
            unsafe { ffi::mlx_contiguous(&mut contiguous.0, array, false, stream) },
            "readback layout",
        )?;
        if contiguous.0.ctx.is_null() {
            return Err(failure("readback layout"));
        }
        check(unsafe { ffi::mlx_array_eval(contiguous.0) }, "evaluate")?;
        check(unsafe { ffi::mlx_synchronize(stream) }, "synchronize")?;
        // The copy is evaluated and synchronized, so its drop below frees a completed buffer and
        // never a lazy temporary that a failure would have to retain.
        self.read_array(contiguous.0)
    }

    /// Reads the bytes of one evaluated, row-contiguous `array` after the metadata checks.
    fn read_array(&self, array: ffi::mlx_array) -> Result<Tensor, EngError> {
        if unsafe { ffi::mlx_array_dtype(array) } != native_dtype(self.dtype) {
            return Err(failure("readback layout"));
        }
        if unsafe { ffi::mlx_array_ndim(array) } != self.shape.len() {
            return Err(failure("readback rank"));
        }
        let expected = usize::try_from(self.byte_len).map_err(|_| failure("readback size"))?;
        if native_byte_len(array, self.dtype) != Some(self.byte_len) {
            return Err(failure("readback size"));
        }
        let shape = unsafe { ffi::mlx_array_shape(array) };
        if !self.shape.is_empty() {
            if shape.is_null() || !shape.is_aligned() {
                return Err(failure("readback shape"));
            }
            let actual = unsafe { std::slice::from_raw_parts(shape, self.shape.len()) };
            if actual
                .iter()
                .zip(&self.shape)
                .any(|(&a, &b)| a < 0 || a as u32 != b)
            {
                return Err(failure("readback shape"));
            }
        }
        // A zero-size array has no data pointer to read.
        let bytes = if expected == 0 {
            Vec::new()
        } else {
            let data = unsafe { native_data(array, self.dtype) };
            if data.is_null() {
                return Err(failure("readback data"));
            }
            unsafe { std::slice::from_raw_parts(data, expected) }.to_vec()
        };
        Tensor::try_from_host(self.dtype, self.shape.clone(), bytes)
            .map_err(|_| failure("readback storage"))
    }
}

impl Drop for Resident {
    fn drop(&mut self) {
        let _dispatch = DispatchGuard::enter();
        if has_failed() {
            if let Some(array) = self.array.take() {
                std::mem::forget(array);
            }
            if let Some(retire) = self.retire.take() {
                std::mem::forget(retire);
            }
            return;
        }
        if let (Some(array), Some(retire)) = (self.array.take(), self.retire.as_ref()) {
            retire.retire(array, self.byte_len);
        }
        drop(self.retire.take());
    }
}

/// The byte length that the native array's element count implies for `dtype`.
pub(super) fn native_byte_len(array: ffi::mlx_array, dtype: Dtype) -> Option<u64> {
    u64::try_from(unsafe { ffi::mlx_array_size(array) })
        .ok()?
        .checked_mul(dtype.byte_width())
}

pub(super) fn native_dtype(dtype: Dtype) -> ffi::mlx_dtype {
    match dtype {
        Dtype::U8 => ffi::mlx_dtype__MLX_UINT8,
        Dtype::F16 => ffi::mlx_dtype__MLX_FLOAT16,
        Dtype::Bf16 => ffi::mlx_dtype__MLX_BFLOAT16,
        Dtype::F32 => ffi::mlx_dtype__MLX_FLOAT32,
        Dtype::U32 => ffi::mlx_dtype__MLX_UINT32,
        Dtype::I64 => ffi::mlx_dtype__MLX_INT64,
        Dtype::I16 => ffi::mlx_dtype__MLX_INT16,
        Dtype::I32 => ffi::mlx_dtype__MLX_INT32,
    }
}

unsafe fn native_data(array: ffi::mlx_array, dtype: Dtype) -> *const u8 {
    match dtype {
        Dtype::U8 => unsafe { ffi::mlx_array_data_uint8(array) },
        Dtype::F16 | Dtype::Bf16 => unsafe { ffi::mlx_array_data_uint16(array) }.cast(),
        Dtype::F32 => unsafe { ffi::mlx_array_data_float32(array) }.cast(),
        Dtype::U32 => unsafe { ffi::mlx_array_data_uint32(array) }.cast(),
        Dtype::I64 => unsafe { ffi::mlx_array_data_int64(array) }.cast(),
        Dtype::I16 => unsafe { ffi::mlx_array_data_int16(array) }.cast(),
        Dtype::I32 => unsafe { ffi::mlx_array_data_int32(array) }.cast(),
    }
}
