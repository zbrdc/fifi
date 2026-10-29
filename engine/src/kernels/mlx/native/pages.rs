//! Caller-owned host pages that MLX reads in place.
//!
//! Ownership and lifetime:
//!
//! - `PageStorage` owns one anonymous mapping. `HostPages` holds one `Arc` share of it. Every
//!   imported array holds one more share, passed to MLX as the deleter payload (`Arc::into_raw`).
//!   The deleter `release_share` rebuilds that share with `Arc::from_raw` and drops it. It takes
//!   no lock and calls no MLX function. A panic inside it (for example a failed `munmap` check in
//!   `memmap2`) aborts the process, which is not unsound. MLX may call it on a Metal completion
//!   thread, so `PageStorage` is `Send + Sync` (it holds only a `memmap2::MmapRaw`).
//! - MLX calls the deleter when the last reference to the array's data drops: Rust handles after
//!   the retire-list flush, graph inputs after evaluation, and the completion handler's buffer set.
//!   That is after the GPU work that read the pages completes.
//! - `HostPages` also pins every resident that it lent: it keeps one more handle to each imported
//!   array until `release` or its own drop. While a pin exists, the array is never the only
//!   reference that a pending graph holds, so MLX cannot donate the lent buffer to an output
//!   (Z0 case S9 showed that MLX donates a managed input that only a graph holds). The pins are
//!   ordinary resident shares, so they retire and free through the adapter's retire list.
//! - Only `release` and the drop of `HostPages` remove pins. Both are allowed at any time for
//!   memory safety. If a consuming graph is still unevaluated, MLX may then write into the pages,
//!   so the caller forfeits their contents.
//! - `try_write` never removes pins. It returns `None` while a pin exists, and otherwise uses
//!   `Arc::get_mut`. It succeeds only when no MLX share exists, which is after every reference and
//!   every completion handler is gone. So no host write overlaps a GPU read, and no page is
//!   unmapped while MLX uses it.
//! - After a terminal failure, residents and pins never release their arrays, so the shares
//!   never drop and `try_write` returns `None` until process exit. A failed import whose
//!   deleter state is unknown keeps its share.
//! - `HostPages` holds resident shares, so it is neither `Send` nor `Sync`.

#[cfg(feature = "native-fault-qualification")]
use std::sync::atomic::{AtomicU64, Ordering};
use std::{cell::RefCell, ffi::c_void, sync::Arc};

use memmap2::{Mmap, MmapOptions, MmapRaw};

use super::{
    Context, State, ffi,
    guard::{DispatchGuard, clear_message, failure, mark_failed},
    handles::{Array, native_byte_len, native_dtype},
};
use crate::{
    EngError,
    kernels::{
        PackedWeight,
        mlx::{PackedPages, PageRange},
    },
    tensor::{Backing, Dtype, ExecutionTensor, Tensor},
};

/// Page imports that MLX served with its copy path (qualification builds only).
#[cfg(feature = "native-fault-qualification")]
static PAGE_COPIES: AtomicU64 = AtomicU64::new(0);

/// One anonymous mapping. It unmaps when the last share drops, on any thread.
struct PageStorage {
    map: MmapRaw,
}

/// Page-aligned host bytes that MLX can use in place. The caller owns them.
///
/// MLX holds one share per imported array until it no longer needs the bytes. The pages also pin
/// each imported resident until `release` or drop, so MLX never writes into lent pages.
///
/// The pin list grows with every import and is unbounded until `release`. Each pin is a resident
/// share, so it holds its adapter's retire list and lease: pages with pins that outlive their
/// `MlxKernels` keep that adapter slot, and a third adapter cannot start.
///
/// Dropping `HostPages` drops its pins, like `release`. That is memory-safe at any time, but if a
/// consuming graph is still unevaluated, MLX may write into the pages and their contents are lost.
pub struct HostPages {
    storage: Arc<PageStorage>,
    /// A never-touched read-only twin of the same length. Packed-layout checks build metadata-only
    /// tensors over it, so they reuse `Tensor` and `PackedWeight` validation without a copy.
    shadow: Arc<Mmap>,
    len: usize,
    pins: RefCell<Vec<ExecutionTensor>>,
}

/// A page range after the checks that precede any native call.
struct CheckedRange {
    offset: usize,
    end: usize,
    dtype: Dtype,
    shape: Vec<u32>,
    native_shape: Vec<i32>,
    rank: i32,
    byte_len: u64,
}

impl HostPages {
    /// The OS page size (16,384 B on the target Mac), or 0 when the query fails.
    pub fn page_size() -> usize {
        // SAFETY: the shim takes no arguments, only queries `sysconf`, and cannot throw.
        unsafe { ffi::inference_page_size() }
    }

    /// Maps `len` bytes rounded up to the page size.
    ///
    /// Returns `BadRequest` for a zero or overflowing length and `MemoryPlan` when the mapping fails.
    pub fn try_new(len: usize) -> Result<Self, EngError> {
        let page = Self::page_size();
        if page == 0 {
            return Err(EngError::Backend {
                operation: "page size",
                message: "the OS page size query failed".into(),
            });
        }
        if len == 0 {
            return Err(EngError::BadRequest(
                "host pages need at least one byte".into(),
            ));
        }
        let rounded = len
            .checked_next_multiple_of(page)
            .ok_or_else(|| EngError::BadRequest("host page length overflows usize".into()))?;
        let map_failed =
            |error: std::io::Error| EngError::MemoryPlan(format!("host page mapping: {error}"));
        let map = MmapOptions::new()
            .len(rounded)
            .map_anon()
            .map_err(map_failed)?;
        let shadow = MmapOptions::new()
            .len(rounded)
            .map_anon()
            .and_then(|twin| twin.make_read_only())
            .map_err(map_failed)?;
        Ok(Self {
            storage: Arc::new(PageStorage {
                map: MmapRaw::from(map),
            }),
            shadow: Arc::new(shadow),
            len: rounded,
            pins: RefCell::new(Vec::new()),
        })
    }

    /// The mapped length: the requested length rounded up to the page size.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Always false: `try_new` rejects a zero length.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// True while an imported resident is pinned or any MLX array still holds the pages.
    pub fn in_use(&self) -> bool {
        !self.pins.borrow().is_empty() || Arc::strong_count(&self.storage) > 1
    }

    /// Drops the pins of every imported resident.
    ///
    /// Contract: call it only after every graph that consumes the lent residents has been
    /// evaluated by a `readback` or `evaluate` that covers it. Calling it earlier lets MLX donate
    /// the pages to an output of a pending graph, which overwrites their contents. The pinned
    /// arrays free at the importing adapter's next successful synchronization.
    pub fn release(&mut self) {
        self.pins.get_mut().clear();
    }

    /// Mutable bytes, only while no MLX array uses the pages.
    ///
    /// It never removes pins. It returns `None` while a pin exists (see `release`) and while MLX
    /// still holds a share: retired handles before the next synchronization, pending graphs, and
    /// unfinished GPU work. After a terminal native failure it returns `None` for good.
    pub fn try_write(&mut self) -> Option<&mut [u8]> {
        if !self.pins.get_mut().is_empty() {
            return None;
        }
        let storage = Arc::get_mut(&mut self.storage)?;
        let base = storage.map.as_mut_ptr();
        let len = storage.map.len();
        // SAFETY: `get_mut` proves that no MLX array holds a share, so nothing else reads or writes
        // the mapping. The slice borrows `self` mutably, so no import can start while it lives.
        Some(unsafe { std::slice::from_raw_parts_mut(base, len) })
    }

    /// Applies every check that precedes a native call. Returns `BadRequest` on failure.
    fn check(&self, range: &PageRange) -> Result<CheckedRange, EngError> {
        let page = Self::page_size();
        if page == 0 || !range.offset.is_multiple_of(page) {
            return Err(EngError::BadRequest(
                "page range offset must be a multiple of the page size".into(),
            ));
        }
        let native_shape = range
            .shape
            .iter()
            .map(|&dimension| {
                i32::try_from(dimension).map_err(|_| {
                    EngError::BadRequest("page range dimension exceeds native int".into())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let rank = i32::try_from(range.shape.len())
            .map_err(|_| EngError::BadRequest("page range rank exceeds native int".into()))?;
        let byte_len = range
            .shape
            .iter()
            .try_fold(range.dtype.byte_width(), |count, &dimension| {
                count.checked_mul(u64::from(dimension))
            })
            .ok_or_else(|| EngError::BadRequest("page range byte length overflows".into()))?;
        if byte_len == 0 {
            return Err(EngError::BadRequest(
                "page range must hold at least one element".into(),
            ));
        }
        let end = usize::try_from(byte_len)
            .ok()
            .and_then(|length| range.offset.checked_add(length))
            .filter(|&end| end <= self.len)
            .ok_or_else(|| EngError::BadRequest("page range passes the end of the pages".into()))?;
        Ok(CheckedRange {
            offset: range.offset,
            end,
            dtype: range.dtype,
            shape: range.shape.clone(),
            native_shape,
            rank,
            byte_len,
        })
    }

    /// A metadata-only tensor of `range` over the untouched shadow mapping.
    fn shadow_tensor(&self, range: &PageRange) -> Result<Tensor, EngError> {
        let checked = self.check(range)?;
        Tensor::from_shared(
            checked.dtype,
            checked.shape,
            Backing::Mapped(Arc::clone(&self.shadow)),
            checked.offset..checked.end,
        )
        .map_err(as_bad_request)
    }

    /// Checks the ranges of `weight` and its layout with the `PackedWeight` rules.
    pub(in crate::kernels::mlx) fn check_packed(
        &self,
        weight: &PackedPages,
    ) -> Result<(), EngError> {
        let biases = weight
            .biases
            .as_ref()
            .map(|range| self.shadow_tensor(range))
            .transpose()?;
        PackedWeight::try_new(
            self.shadow_tensor(&weight.payload)?,
            self.shadow_tensor(&weight.scales)?,
            biases,
            weight.logical_shape.clone(),
            weight.quantization,
        )
        .map(drop)
        .map_err(as_bad_request)
    }
}

/// Layout checks report `Checkpoint` for stored tensors. A page range is a request argument.
fn as_bad_request(error: EngError) -> EngError {
    match error {
        EngError::Checkpoint(message) => EngError::BadRequest(message),
        other => other,
    }
}

/// The MLX deleter of a page import: drops the share that the import passed as `payload`.
///
/// It may run on any thread. It takes no lock and calls no MLX function. Dropping the last share
/// unmaps the pages. `memmap2` ignores the `munmap` status; any panic here would abort the process
/// at the `extern "C"` boundary, which is not unsound.
unsafe extern "C" fn release_share(payload: *mut c_void) {
    // SAFETY: `payload` came from `Arc::into_raw` in `State::import_range`, and MLX calls the
    // deleter of an array exactly once.
    drop(unsafe { Arc::from_raw(payload.cast_const().cast::<PageStorage>()) });
}

/// Whether the native array has the checked dtype, rank, shape, and byte length.
fn native_layout_matches(array: ffi::mlx_array, range: &CheckedRange) -> bool {
    // SAFETY (each block in this function): `array` is a live, non-null handle that the caller
    // owns under the dispatch guard. The accessors only read its metadata and cannot throw.
    if unsafe { ffi::mlx_array_dtype(array) } != native_dtype(range.dtype)
        // SAFETY: as above.
        || unsafe { ffi::mlx_array_ndim(array) } != range.native_shape.len()
        || native_byte_len(array, range.dtype) != Some(range.byte_len)
    {
        return false;
    }
    if range.native_shape.is_empty() {
        return true;
    }
    // SAFETY: as above; the accessor returns a pointer into the array's shape storage.
    let shape = unsafe { ffi::mlx_array_shape(array) };
    if shape.is_null() || !shape.is_aligned() {
        return false;
    }
    // SAFETY: MLX returns a non-null, aligned pointer to `ndim` ints, and `ndim` equals this length.
    let actual = unsafe { std::slice::from_raw_parts(shape, range.native_shape.len()) };
    actual == range.native_shape.as_slice()
}

/// The data address of a page import, read through the accessor of its dtype.
#[cfg(feature = "native-fault-qualification")]
fn data_address(array: ffi::mlx_array, dtype: Dtype) -> usize {
    // SAFETY: `array` is a live, non-null handle owned under the dispatch guard. The accessor only
    // reads the data pointer of an available array and cannot throw. Nothing dereferences it.
    let pointer: *const u8 = unsafe {
        match dtype {
            Dtype::U8 => ffi::mlx_array_data_uint8(array),
            Dtype::F16 | Dtype::Bf16 => ffi::mlx_array_data_uint16(array).cast(),
            Dtype::F32 => ffi::mlx_array_data_float32(array).cast(),
            Dtype::U32 => ffi::mlx_array_data_uint32(array).cast(),
            Dtype::I64 => ffi::mlx_array_data_int64(array).cast(),
            Dtype::I16 => ffi::mlx_array_data_int16(array).cast(),
            Dtype::I32 => ffi::mlx_array_data_int32(array).cast(),
        }
    };
    pointer.addr()
}

impl State {
    /// Wraps one checked range without a copy and without a synchronization, and pins the result.
    ///
    /// Every failure marks the adapter terminal before any temporary array drops.
    fn import_range(
        &self,
        pages: &HostPages,
        range: &CheckedRange,
    ) -> Result<ExecutionTensor, EngError> {
        let source = pages.storage.map.as_mut_ptr().wrapping_add(range.offset);
        let share = Arc::into_raw(Arc::clone(&pages.storage))
            .cast_mut()
            .cast::<c_void>();
        let deleter: unsafe extern "C" fn(*mut c_void) = release_share;
        let mut array = Array::null();
        // SAFETY: the dispatch guard is held. `source` points into the mapping at a checked range
        // that lies inside it, and the mapping lives at least as long as `share`. `native_shape`
        // holds `rank` ints. `share` is one owned `Arc` share that MLX passes to `release_share`
        // once. The stream is this adapter's live stream. The shim catches C++ exceptions.
        let status = unsafe {
            ffi::mlx_array_new_data_managed_payload(
                &mut array.0,
                source.cast(),
                range.native_shape.as_ptr(),
                range.rank,
                native_dtype(range.dtype),
                share,
                Some(deleter),
                self.stream.0,
            )
        };
        if status != 0 || array.0.ctx.is_null() {
            // The deleter may or may not have run, so the share stays with MLX or leaks.
            mark_failed();
            return Err(failure("page import"));
        }
        if !native_layout_matches(array.0, range) {
            mark_failed();
            return Err(EngError::Backend {
                operation: "page import",
                message: "native page import output mismatch".into(),
            });
        }
        #[cfg(feature = "native-fault-qualification")]
        if data_address(array.0, range.dtype) != source.addr() {
            PAGE_COPIES.fetch_add(1, Ordering::Relaxed);
        }
        let tensor = self.resident(array, range.dtype, range.shape.clone(), range.byte_len);
        pages.pins.borrow_mut().push(tensor.share());
        Ok(tensor)
    }
}

impl Context {
    /// Wraps `ranges` of `pages` as resident tensors without copying and without synchronizing.
    pub(in crate::kernels::mlx) fn import_pages(
        &mut self,
        pages: &HostPages,
        ranges: &[PageRange],
    ) -> Result<Vec<ExecutionTensor>, EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        if ranges.is_empty() {
            return Err(EngError::BadRequest(
                "import_pages needs at least one range".into(),
            ));
        }
        let checked = ranges
            .iter()
            .map(|range| pages.check(range))
            .collect::<Result<Vec<_>, _>>()?;
        let state = self.state.as_ref().unwrap();
        clear_message();
        checked
            .iter()
            .map(|range| state.import_range(pages, range))
            .collect()
    }

    /// Page imports that MLX served with its copy path since process start.
    #[cfg(feature = "native-fault-qualification")]
    pub(in crate::kernels::mlx) fn qualification_page_copies(&self) -> u64 {
        PAGE_COPIES.load(Ordering::Relaxed)
    }
}
