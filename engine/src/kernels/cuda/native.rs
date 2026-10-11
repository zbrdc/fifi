//! Raw FFI to the CUDA adapter in `native/smoke.cu` and `native/projector.cu`.
//!
//! `unsafe` and `extern "C"` live only in this file and its child modules
//! (`native/resident.rs`, `native/device_info.rs`), per decision 0047. Each
//! child module starts with a doc that lists its safety invariants. Functions here are private to the CUDA
//! module. Each one checks every length and shape the native side relies on for
//! memory safety before the native call. Callers hold the dispatch guard
//! (`super::guard`) around every call.

#![allow(unsafe_code)]

use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr::NonNull;

pub(super) mod device_info;
pub(super) mod resident;

/// Matches `INFERENCE_CUDA_OK` in `native/abi.h`.
pub(super) const STATUS_OK: c_int = 0;
/// Matches `INFERENCE_CUDA_INVALID_ARGUMENT` in `native/abi.h`.
pub(super) const STATUS_INVALID_ARGUMENT: c_int = -1;

/// Match the `INFERENCE_CUDA_STAGE_*` values in `native/abi.h`.
pub(super) const STAGE_NONE: c_int = 0;
pub(super) const STAGE_ALLOCATE: c_int = 1;
pub(super) const STAGE_COPY: c_int = 2;
pub(super) const STAGE_LAUNCH: c_int = 3;
pub(super) const STAGE_SYNCHRONIZE: c_int = 4;

/// Rows and columns of one tile of the vendored `mma884` layout.
pub(super) const TILE_ROWS: usize = 8;
pub(super) const TILE_COLS: usize = 256;
/// `kern::kMaxK`: the longest row and the largest batch the matmul accepts.
pub(super) const MAX_DIM: usize = 32768;

/// A failed device call: the native status and the stage that failed first.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct DeviceFailure {
    pub(super) status: c_int,
    pub(super) stage: c_int,
}

/// A failed `posix_fadvise` call: the adapter status and the error number it
/// reports, which is 0 when the adapter rejected an argument before the call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PageCacheFailure {
    pub(super) status: c_int,
    pub(super) errno: c_int,
}

/// A weight type the tiled matmul executes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TiledType {
    Q4K,
    Q6K,
    Q80,
    /// Built on the host by `context::tile_q5_0`; the matmul is the own kernel of
    /// `native/resident/matmul_q5_0.cu` (MEL-C1b), not a vendored one.
    Q50,
}

impl TiledType {
    /// `INFERENCE_CUDA_TYPE_*` in `native/abi.h` (the ggml type id).
    fn ggml(self) -> c_int {
        match self {
            Self::Q4K => 12,
            Self::Q6K => 14,
            Self::Q80 => 8,
            // Passed only to `inference_cuda_linear_ws`; the vendored repack and matmul entries
            // refuse it, and `repack` below refuses it before the call.
            Self::Q50 => 6,
        }
    }

    /// Values per row of one tile: 256 for the vendored tiles, 128 for Q5_0
    /// (its K = 896 rows are not a multiple of 256).
    pub(super) fn tile_cols(self) -> usize {
        match self {
            Self::Q50 => 128,
            Self::Q4K | Self::Q6K | Self::Q80 => TILE_COLS,
        }
    }

    /// Source bytes of one row inside one tile: the ggml blocks of
    /// [`Self::tile_cols`] values.
    pub(super) fn row_tile_bytes(self) -> usize {
        match self {
            Self::Q4K => 144,
            Self::Q6K => 210,
            Self::Q80 => 8 * 34,
            Self::Q50 => 4 * 22,
        }
    }

    /// Source bytes of 256 values of one row: one Q4_K or Q6_K block, or
    /// eight Q8_0 blocks of 32.
    pub(super) fn bytes_per_256(self) -> usize {
        match self {
            Self::Q4K => 144,
            Self::Q6K => 210,
            Self::Q80 => 8 * 34,
            Self::Q50 => 8 * 22,
        }
    }

    /// 16-byte vectors per tile: `mma884::{Q4_K,Q6_K,Q8_0}::kTileVecs` for the
    /// 8 x 256 tiles, 44 (704 bytes) for the 8 x 128 Q5_0 tile.
    pub(super) fn tile_vecs(self) -> usize {
        match self {
            Self::Q4K => 72,
            Self::Q6K => 112,
            Self::Q80 => 136,
            Self::Q50 => 44,
        }
    }
}

/// The shape of a tiled weight: `n_rows` rows of `k` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Tiled {
    pub(super) ty: TiledType,
    pub(super) n_rows: usize,
    pub(super) k: usize,
}

impl Tiled {
    /// True when the vendored tiles can hold this shape.
    pub(super) fn fits(&self) -> bool {
        self.n_rows > 0
            && self.n_rows.is_multiple_of(TILE_ROWS)
            && c_int::try_from(self.n_rows).is_ok()
            && self.k > 0
            && self.k.is_multiple_of(self.ty.tile_cols())
            && self.k <= MAX_DIM
    }

    /// Bytes of the ggml blocks the shape holds, or `None` on overflow.
    pub(super) fn source_bytes(&self) -> Option<usize> {
        self.n_rows
            .checked_mul(self.k / self.ty.tile_cols())?
            .checked_mul(self.ty.row_tile_bytes())
    }

    /// Tile vectors of the repacked shape, or `None` on overflow.
    pub(super) fn tile_vecs(&self) -> Option<usize> {
        (self.n_rows / TILE_ROWS)
            .checked_mul(self.k / self.ty.tile_cols())?
            .checked_mul(self.ty.tile_vecs())
    }

    /// Bytes of one 8-row group of tiles, or `None` on overflow. Upload and
    /// row-range reads must agree on it: group `g` starts at `g * group_bytes`.
    pub(super) fn group_bytes(&self) -> Option<usize> {
        self.with_groups(1)
            .tile_vecs()?
            .checked_mul(size_of::<TileVec>())
    }

    /// Number of 8-row groups. The tiles are group-major (group outer, 256-wide
    /// tile inner), so consecutive groups are consecutive rows of the source
    /// and consecutive tile bytes, which lets a load repack them piecewise.
    pub(super) fn groups(&self) -> usize {
        self.n_rows / TILE_ROWS
    }

    /// The shape of `groups` whole groups of this tensor.
    pub(super) fn with_groups(&self, groups: usize) -> Self {
        Self {
            n_rows: groups.saturating_mul(TILE_ROWS),
            ..*self
        }
    }

    fn native_dims(&self) -> Option<(c_int, c_int)> {
        if !self.fits() {
            return None;
        }
        Some((
            c_int::try_from(self.n_rows).ok()?,
            c_int::try_from(self.k).ok()?,
        ))
    }
}

/// One 16-byte vector of a repacked weight (`uint4` on the device side).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C, align(16))]
pub(super) struct TileVec(pub(super) [u8; 16]);

impl TileVec {
    /// Views tile vectors as their bytes.
    pub(super) fn as_bytes(vecs: &[Self]) -> &[u8] {
        // SAFETY: `TileVec` is `repr(C)` over `[u8; 16]` with no padding, so
        // the slice covers `16 * len` initialized bytes for its lifetime.
        unsafe {
            std::slice::from_raw_parts(vecs.as_ptr().cast::<u8>(), std::mem::size_of_val(vecs))
        }
    }

    /// Views tile vectors as their bytes, writable.
    pub(super) fn as_bytes_mut(vecs: &mut [Self]) -> &mut [u8] {
        // SAFETY: as in `as_bytes`; the exclusive borrow of `vecs` passes to
        // the returned slice, and every byte pattern is a valid `[u8; 16]`.
        unsafe {
            std::slice::from_raw_parts_mut(
                vecs.as_mut_ptr().cast::<u8>(),
                std::mem::size_of_val(vecs),
            )
        }
    }
}

/// A device allocation made by [`upload`].
///
/// It has no `Drop`: the owner in `super` passes it to [`free`], so that a
/// failed free is reported and latches the backend instead of being lost.
#[derive(Debug)]
#[must_use = "a device allocation must be passed to free"]
pub(super) struct RawDevice {
    ptr: NonNull<c_void>,
    bytes: usize,
}

impl RawDevice {
    pub(super) fn bytes(&self) -> usize {
        self.bytes
    }

    /// The part of the allocation that starts `offset` bytes in. The offset is
    /// checked by the call that uses the view.
    pub(super) fn at(&self, offset: usize) -> DeviceView<'_> {
        DeviceView {
            device: self,
            offset,
        }
    }
}

#[cfg(test)]
impl RawDevice {
    /// A handle that no allocation backs. Freeing it makes `cudaFree` fail with
    /// an invalid value, which is not a sticky device error, so a test can fail a
    /// free and keep the device healthy.
    pub(super) fn unbacked(bytes: usize) -> Self {
        Self {
            ptr: NonNull::dangling(),
            bytes,
        }
    }
}

/// A byte offset into a live device allocation, for a call that reads only
/// the tail of a buffer, such as one expert of a 3-D stack.
#[derive(Debug, Clone, Copy)]
pub(super) struct DeviceView<'a> {
    device: &'a RawDevice,
    offset: usize,
}

/// Bytes of the x, xf and y regions of one linear call, or the sizes of a workspace.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Activations {
    pub(super) x: usize,
    pub(super) xf: usize,
    pub(super) y: usize,
}

impl Activations {
    /// True when every region of `self` holds at least the bytes of `need`.
    pub(super) fn covers(self, need: Self) -> bool {
        self.x >= need.x && self.xf >= need.xf && self.y >= need.y
    }

    /// The smallest sizes that hold both `self` and `other`.
    pub(super) fn union(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            xf: self.xf.max(other.xf),
            y: self.y.max(other.y),
        }
    }
}

/// A byte range inside the workspace allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Region {
    pub(super) offset: usize,
    pub(super) bytes: usize,
}

/// The three activation regions of one linear call, inside one live allocation.
#[derive(Debug, Clone, Copy)]
pub(super) struct Work<'a> {
    pub(super) device: &'a RawDevice,
    pub(super) x: Region,
    pub(super) xf: Region,
    pub(super) y: Region,
}

impl Work<'_> {
    /// True when the regions lie inside the allocation, hold `need`, are aligned
    /// for the kernels (x and y 4 bytes, xf 16) and do not overlap in the bytes
    /// the call uses. The adapter repeats these checks on the addresses.
    fn fits(&self, need: Activations) -> bool {
        let used = [
            (self.x, need.x, 4),
            (self.xf, need.xf, 16),
            (self.y, need.y, 4),
        ];
        let inside = used.iter().all(|&(region, bytes, align)| {
            region.bytes >= bytes
                && region.offset.is_multiple_of(align)
                && region
                    .offset
                    .checked_add(region.bytes)
                    .is_some_and(|end| end <= self.device.bytes)
        });
        // `inside` bounds every offset, so these sums cannot overflow.
        let apart = |a: usize, b: usize| {
            used[a].0.offset + used[a].1 <= used[b].0.offset
                || used[b].0.offset + used[b].1 <= used[a].0.offset
        };
        inside && apart(0, 1) && apart(0, 2) && apart(1, 2)
    }
}

/// `inference_cuda_workspace` in `native/abi.h`.
#[derive(Clone, Copy)]
#[repr(C)]
struct WorkspaceRaw {
    x: *mut c_void,
    x_bytes: usize,
    xf: *mut c_void,
    xf_bytes: usize,
    y: *mut c_void,
    y_bytes: usize,
}

/// The C view of `work`, or `None` when its regions cannot hold `need`.
fn workspace_raw(work: &Work<'_>, need: Activations) -> Option<WorkspaceRaw> {
    if !work.fits(need) {
        return None;
    }
    let base = work.device.ptr.as_ptr().cast::<u8>();
    let at = |region: Region| {
        // SAFETY: `fits` proved that the region lies inside the live allocation
        // that starts at `base`, so the offset pointer is in bounds.
        let pointer = unsafe { base.add(region.offset) };
        pointer.cast::<c_void>()
    };
    Some(WorkspaceRaw {
        x: at(work.x),
        x_bytes: work.x.bytes,
        xf: at(work.xf),
        xf_bytes: work.xf.bytes,
        y: at(work.y),
        y_bytes: work.y.bytes,
    })
}

unsafe extern "C" {
    #[link_name = "inference_cuda_device_count"]
    fn device_count_raw(out_count: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_vector_add_f32"]
    fn vector_add_f32_raw(
        lhs: *const f32,
        rhs: *const f32,
        out: *mut f32,
        len: usize,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_status_message"]
    fn status_message_raw(status: c_int) -> *const c_char;
    #[link_name = "inference_cuda_drop_page_cache"]
    fn drop_page_cache_raw(
        fd: c_int,
        offset: std::ffi::c_longlong,
        len: std::ffi::c_longlong,
        out_errno: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_memory_info"]
    fn memory_info_raw(out_free: *mut usize, out_total: *mut usize) -> c_int;
    #[link_name = "inference_cuda_upload"]
    fn upload_raw(
        host: *const c_void,
        bytes: usize,
        out_device: *mut *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_alloc"]
    fn alloc_raw(bytes: usize, out_device: *mut *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_upload_at"]
    fn upload_at_raw(
        device: *mut c_void,
        device_bytes: usize,
        offset: usize,
        host: *const c_void,
        bytes: usize,
        out_stage: *mut c_int,
    ) -> c_int;
    #[cfg(test)]
    #[link_name = "inference_cuda_download"]
    fn download_raw(
        device: *const c_void,
        host: *mut c_void,
        bytes: usize,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_free"]
    fn free_raw(device: *mut c_void, out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_repack"]
    fn repack_raw(
        ggml_type: c_int,
        src: *const u8,
        n_rows: c_int,
        k: c_int,
        dst: *mut c_void,
        dst_bytes: usize,
    ) -> c_int;
    #[link_name = "inference_cuda_linear_workspace_bytes"]
    fn workspace_bytes_raw(
        m: c_int,
        k: c_int,
        n_rows: c_int,
        out_x: *mut usize,
        out_xf: *mut usize,
        out_y: *mut usize,
    ) -> c_int;
    #[link_name = "inference_cuda_linear_ws"]
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the C declaration in abi.h"
    )]
    fn linear_ws_raw(
        ggml_type: c_int,
        weight: *const c_void,
        n_rows: c_int,
        k: c_int,
        x: *const f32,
        m: c_int,
        y: *mut f32,
        workspace: *const WorkspaceRaw,
        out_stage: *mut c_int,
    ) -> c_int;
}

#[cfg(all(test, feature = "cuda-fault-qualification"))]
unsafe extern "C" {
    #[link_name = "inference_cuda_fault_illegal_address"]
    fn fault_illegal_address_raw(out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_fault_vendor_launch"]
    fn fault_vendor_launch_raw(out_stage: *mut c_int) -> c_int;
    #[link_name = "inference_cuda_allocation_calls"]
    fn allocation_calls_raw(out_mallocs: *mut u64, out_frees: *mut u64) -> c_int;
}

fn invalid_argument() -> DeviceFailure {
    DeviceFailure {
        status: STATUS_INVALID_ARGUMENT,
        stage: STAGE_NONE,
    }
}

fn checked(status: c_int, stage: c_int) -> Result<(), DeviceFailure> {
    if status == STATUS_OK {
        Ok(())
    } else {
        Err(DeviceFailure { status, stage })
    }
}

/// Returns the device count, or the native status when the query fails.
pub(super) fn device_count() -> Result<c_int, c_int> {
    let mut count: c_int = 0;
    // SAFETY: `count` is a live, writable c_int for the whole call. The adapter
    // writes it only on success.
    let status = unsafe { device_count_raw(&mut count) };
    if status == STATUS_OK {
        Ok(count)
    } else {
        Err(status)
    }
}

/// Computes `out[i] = lhs[i] + rhs[i]` on the first CUDA device.
///
/// The caller must pass slices of equal length. The adapter rejects a zero length.
pub(super) fn vector_add_f32(
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
) -> Result<(), DeviceFailure> {
    if lhs.len() != rhs.len() || lhs.len() != out.len() || lhs.is_empty() {
        return Err(DeviceFailure {
            status: STATUS_INVALID_ARGUMENT,
            stage: STAGE_NONE,
        });
    }
    let mut stage = STAGE_NONE;
    // SAFETY: each slice pointer addresses `len` initialized or writable f32
    // values that outlive the call. `out` is exclusively borrowed. `stage` is a
    // live, writable c_int. The adapter keeps no pointer after it returns.
    let status = unsafe {
        vector_add_f32_raw(
            lhs.as_ptr(),
            rhs.as_ptr(),
            out.as_mut_ptr(),
            lhs.len(),
            &mut stage,
        )
    };
    if status == STATUS_OK {
        Ok(())
    } else {
        Err(DeviceFailure { status, stage })
    }
}

/// Drops the page cache of `len` bytes of `file` from `offset`. Returns the
/// adapter status and error number of a failed call. A range above `i64::MAX`
/// is rejected here with `STATUS_INVALID_ARGUMENT` and error number 0.
pub(super) fn drop_page_cache(
    file: &std::fs::File,
    offset: u64,
    len: u64,
) -> Result<(), PageCacheFailure> {
    use std::os::fd::AsRawFd;
    let (Ok(offset), Ok(len)) = (i64::try_from(offset), i64::try_from(len)) else {
        return Err(PageCacheFailure {
            status: STATUS_INVALID_ARGUMENT,
            errno: 0,
        });
    };
    let mut errno = 0;
    // SAFETY: `errno` is a live, writable c_int. The descriptor stays open
    // for the call because `file` is borrowed; the call touches no memory.
    let status = unsafe { drop_page_cache_raw(file.as_raw_fd(), offset, len, &mut errno) };
    if status == STATUS_OK {
        Ok(())
    } else {
        Err(PageCacheFailure { status, errno })
    }
}

/// `kern::kMaxTokens`: up to this many tokens the matmul reads a fragment-order `xf`.
const FRAGMENT_TOKENS: usize = 16;
/// `kern::xf_vecs() * sizeof(uint4)`: the fixed size of a fragment-order `xf`.
const XF_FRAGMENT_BYTES: usize = 65_537 * 16;

/// Bytes of the fp16 input buffer `xf` for `tokens` tokens of `k` values: the
/// fixed fragment-order buffer up to 16 tokens, else row-major halves. This is
/// the one owner of the rule on the Rust side; the adapter repeats it in
/// `linear_needs` (`native/projector.cu`) and a test pins that they agree.
/// Returns `None` when the byte count overflows.
pub(crate) fn xf_bytes(tokens: usize, k: usize) -> Option<usize> {
    if tokens <= FRAGMENT_TOKENS {
        return Some(XF_FRAGMENT_BYTES);
    }
    tokens.checked_mul(k)?.checked_mul(size_of::<u16>())
}

/// Bytes of the f32 input `x` and of `xf` for a pack of `tokens` tokens of `k`
/// values, or `None` on overflow.
pub(super) fn pack_bytes(tokens: usize, k: usize) -> Option<(usize, usize)> {
    let x = tokens.checked_mul(k)?.checked_mul(size_of::<f32>())?;
    Some((x, xf_bytes(tokens, k)?))
}

/// Values per token row that the activation regions of a call hold. A Q5_0 weight (K = 896 is no
/// multiple of 256) reads the vendored fragment-order `xf` of a row rounded up to the 256 grid,
/// whose tail is zero (MEL-C1b, kernel contract); every other type uses its own `k`. Pass this
/// width, not `tiled.k`, to [`activation_bytes`] and to the pack of a Q5_0 call.
pub(super) fn activation_k(ty: TiledType, k: usize) -> usize {
    match ty {
        TiledType::Q50 => k.checked_next_multiple_of(TILE_COLS).unwrap_or(k),
        TiledType::Q4K | TiledType::Q6K | TiledType::Q80 => k,
    }
}

/// The bytes the x, xf and y regions of a call with `tokens` tokens need.
/// Host only. The adapter checks the shape and sizes x and y; `xf` comes from
/// [`xf_bytes`]. Returns the native status for a shape the matmul cannot run.
pub(super) fn activation_bytes(
    n_rows: usize,
    k: usize,
    tokens: usize,
) -> Result<Activations, c_int> {
    let xf = xf_bytes(tokens, k).ok_or(STATUS_INVALID_ARGUMENT)?;
    let (Ok(n_rows), Ok(k), Ok(m)) = (
        c_int::try_from(n_rows),
        c_int::try_from(k),
        c_int::try_from(tokens),
    ) else {
        return Err(STATUS_INVALID_ARGUMENT);
    };
    let mut bytes = Activations::default();
    // SAFETY: the three out pointers address live, writable usize values for
    // the call; the call touches no device.
    let status =
        unsafe { workspace_bytes_raw(m, k, n_rows, &mut bytes.x, &mut bytes.xf, &mut bytes.y) };
    if status == STATUS_OK {
        // The adapter repeats the rule `xf_bytes` owns; a drift shows in debug builds.
        debug_assert_eq!(bytes.xf, xf, "the adapter and xf_bytes disagree on xf");
        Ok(Activations { xf, ..bytes })
    } else {
        Err(status)
    }
}

/// Returns `(free, total)` device memory in bytes, or the native status.
pub(super) fn memory_info() -> Result<(usize, usize), c_int> {
    let mut free = 0usize;
    let mut total = 0usize;
    // SAFETY: both pointers address live, writable usize values for the call.
    let status = unsafe { memory_info_raw(&mut free, &mut total) };
    if status == STATUS_OK {
        Ok((free, total))
    } else {
        Err(status)
    }
}

/// Copies `host` into a new device allocation and waits for the copy.
///
/// The adapter rejects an empty slice.
pub(super) fn upload(host: &[u8]) -> Result<RawDevice, DeviceFailure> {
    if host.is_empty() {
        return Err(invalid_argument());
    }
    let mut device: *mut c_void = std::ptr::null_mut();
    let mut stage = STAGE_NONE;
    // SAFETY: `host` addresses `len` initialized bytes for the whole call;
    // `device` and `stage` are live, writable values. The adapter keeps no
    // host pointer after it returns.
    let status = unsafe { upload_raw(host.as_ptr().cast(), host.len(), &mut device, &mut stage) };
    checked(status, stage)?;
    let ptr = NonNull::new(device).ok_or_else(invalid_argument)?;
    Ok(RawDevice {
        ptr,
        bytes: host.len(),
    })
}

/// Allocates `bytes` of uninitialized device memory.
///
/// The adapter rejects 0 bytes. Fill the allocation with [`upload_at`].
pub(super) fn alloc(bytes: usize) -> Result<RawDevice, DeviceFailure> {
    if bytes == 0 {
        return Err(invalid_argument());
    }
    let mut device: *mut c_void = std::ptr::null_mut();
    let mut stage = STAGE_NONE;
    // SAFETY: `device` and `stage` are live, writable values for the call.
    let status = unsafe { alloc_raw(bytes, &mut device, &mut stage) };
    checked(status, stage)?;
    let ptr = NonNull::new(device).ok_or_else(invalid_argument)?;
    Ok(RawDevice { ptr, bytes })
}

/// Copies `host` into `device` at byte `offset` and waits for the copy.
///
/// The range must lie inside the allocation. Unlike [`upload`], a failure
/// leaves the allocation with the caller, who still frees it.
pub(super) fn upload_at(
    device: &RawDevice,
    offset: usize,
    host: &[u8],
) -> Result<(), DeviceFailure> {
    let in_range = offset
        .checked_add(host.len())
        .is_some_and(|end| end <= device.bytes);
    if host.is_empty() || !in_range {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: `device` is a live allocation of `device.bytes` bytes and the
    // check above keeps `offset + host.len()` inside it. `host` addresses
    // `host.len()` initialized bytes for the whole call, and the adapter keeps
    // no pointer after it returns.
    let status = unsafe {
        upload_at_raw(
            device.ptr.as_ptr(),
            device.bytes,
            offset,
            host.as_ptr().cast(),
            host.len(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// Copies the first `host.len()` bytes of `device` into `host`.
#[cfg(test)]
pub(super) fn download(device: &RawDevice, host: &mut [u8]) -> Result<(), DeviceFailure> {
    if host.is_empty() || host.len() > device.bytes {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: `device` is a live allocation of `device.bytes >= host.len()`
    // bytes, and `host` is an exclusive, writable slice of `host.len()` bytes.
    let status = unsafe {
        download_raw(
            device.ptr.as_ptr(),
            host.as_mut_ptr().cast(),
            host.len(),
            &mut stage,
        )
    };
    checked(status, stage)
}

/// Waits for the device and frees `device`.
pub(super) fn free(device: RawDevice) -> Result<(), DeviceFailure> {
    let mut stage = STAGE_NONE;
    // SAFETY: `device` came from `upload` and is consumed here, so it is freed
    // at most once and no other handle refers to it afterwards.
    let status = unsafe { free_raw(device.ptr.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Repacks the ggml blocks `src` of shape `tiled` into `dst` on the host.
///
/// `src` must hold exactly the blocks of the shape and `dst` exactly its tile
/// vectors. Returns the native status on failure; no CUDA call is made.
pub(super) fn repack(tiled: Tiled, src: &[u8], dst: &mut [TileVec]) -> Result<(), c_int> {
    // The vendored repack has no Q5_0 entry; `context::tile_q5_0` repacks it on the host.
    if tiled.ty == TiledType::Q50 {
        return Err(STATUS_INVALID_ARGUMENT);
    }
    let (n_rows, k) = tiled.native_dims().ok_or(STATUS_INVALID_ARGUMENT)?;
    if tiled.source_bytes() != Some(src.len()) || tiled.tile_vecs() != Some(dst.len()) {
        return Err(STATUS_INVALID_ARGUMENT);
    }
    // SAFETY: the shape checks above make the adapter read exactly
    // `src.len()` bytes and write exactly `size_of_val(dst)` bytes; `dst` is
    // exclusively borrowed and 16-byte aligned by `TileVec`'s layout.
    let status = unsafe {
        repack_raw(
            tiled.ty.ggml(),
            src.as_ptr(),
            n_rows,
            k,
            dst.as_mut_ptr().cast(),
            std::mem::size_of_val(dst),
        )
    };
    if status == STATUS_OK {
        Ok(())
    } else {
        Err(status)
    }
}

/// Computes `y = x · Wᵀ` for the repacked weight of shape `tiled` that starts
/// at `weight`, using the device regions of `work` for the fp16 input and the
/// output before it is copied back. The call allocates nothing.
///
/// `x` holds `m` rows of `tiled.k` values and `y` must hold `m` rows of
/// `tiled.n_rows` values, with `m` in `1..=MAX_DIM`. The view's offset must be
/// a multiple of 16 bytes, because the matmul reads 16-byte vectors, and the
/// tiles of `tiled` must end inside the allocation. The regions of `work` must
/// hold the call (see [`Work`]); otherwise the call fails before the adapter.
///
/// A Q5_0 weight takes at most `MAX_DENSE_TOKENS` (16) tokens: the adapter refuses more, and a
/// native refusal latches the backend terminal, so the caller (`context::token_count`) must
/// have refused it as a `BadRequest` first. Debug builds assert the precondition.
pub(super) fn linear(
    tiled: Tiled,
    weight: DeviceView<'_>,
    work: Work<'_>,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), DeviceFailure> {
    let (n_rows, k) = tiled.native_dims().ok_or_else(invalid_argument)?;
    let tokens = x.len() / tiled.k;
    let m = c_int::try_from(tokens).map_err(|_| invalid_argument())?;
    debug_assert!(
        tiled.ty != TiledType::Q50 || tokens <= resident::matmul_q5_0::MAX_DENSE_TOKENS,
        "a Q5_0 linear takes at most 16 tokens; context::token_count refuses more first"
    );
    let tile_bytes = tiled.tile_vecs().and_then(|vecs| vecs.checked_mul(16));
    let in_range = tile_bytes
        .and_then(|bytes| bytes.checked_add(weight.offset))
        .is_some_and(|end| end <= weight.device.bytes);
    let shapes_match = (1..=MAX_DIM).contains(&tokens)
        && x.len() == tokens * tiled.k
        && tokens.checked_mul(tiled.n_rows) == Some(y.len())
        && weight.offset.is_multiple_of(16)
        && in_range;
    if !shapes_match {
        return Err(invalid_argument());
    }
    let need = activation_bytes(tiled.n_rows, activation_k(tiled.ty, tiled.k), tokens)
        .map_err(|_| invalid_argument())?;
    let workspace = workspace_raw(&work, need).ok_or_else(invalid_argument)?;
    let mut stage = STAGE_NONE;
    // SAFETY: the checks above keep the `weight.offset` bytes before the tiles
    // and the tiles themselves inside the live allocation, and `workspace_raw`
    // returned only after `fits` proved every region of the workspace
    // allocation in bounds, aligned, large enough and pairwise disjoint, so the
    // adapter reads and writes only bytes of those allocations. It reads
    // `x.len()` host floats and writes exactly `y.len()` host floats into the
    // exclusively borrowed `y`. `workspace` lives for the call and no pointer
    // outlives it.
    let status = unsafe {
        linear_ws_raw(
            tiled.ty.ggml(),
            weight
                .device
                .ptr
                .as_ptr()
                .cast::<u8>()
                .add(weight.offset)
                .cast(),
            n_rows,
            k,
            x.as_ptr(),
            m,
            y.as_mut_ptr(),
            &workspace,
            &mut stage,
        )
    };
    checked(status, stage)
}

/// The `cudaMalloc` and `cudaFree` calls the adapter has made so far, as
/// `(mallocs, frees)`. Host only. The counters exist only in builds that
/// qualify fault handling, because only those define the adapter's counting
/// hook (see `native/projector.cu`).
#[cfg(all(test, feature = "cuda-fault-qualification"))]
pub(super) fn allocation_calls() -> (u64, u64) {
    let (mut mallocs, mut frees) = (0u64, 0u64);
    // SAFETY: both pointers address live, writable u64 values for the call; the
    // call touches no device. `unsigned long long` is 64 bits on every target
    // this module builds for.
    let status = unsafe { allocation_calls_raw(&mut mallocs, &mut frees) };
    assert_eq!(status, STATUS_OK, "allocation counters: {status}");
    (mallocs, frees)
}

/// Writes through a null device pointer and waits: a sticky illegal address.
#[cfg(all(test, feature = "cuda-fault-qualification"))]
pub(super) fn fault_illegal_address() -> Result<(), DeviceFailure> {
    let mut stage = STAGE_NONE;
    // SAFETY: `stage` is a live, writable c_int. The fault happens on the
    // device; the host side touches no memory through the null pointer.
    let status = unsafe { fault_illegal_address_raw(&mut stage) };
    checked(status, stage)
}

/// Launches the vendored kernels after a sticky fault, reaching their
/// patched error returns.
#[cfg(all(test, feature = "cuda-fault-qualification"))]
pub(super) fn fault_vendor_launch() -> Result<(), DeviceFailure> {
    let mut stage = STAGE_NONE;
    // SAFETY: `stage` is a live, writable c_int. After a sticky fault the
    // launches fail before any kernel dereferences its null arguments.
    let status = unsafe { fault_vendor_launch_raw(&mut stage) };
    checked(status, stage)
}

/// Returns the adapter's description of a status code.
pub(super) fn status_message(status: c_int) -> String {
    // SAFETY: the adapter returns a pointer to a static, NUL-terminated string
    // for every status.
    let text = unsafe { status_message_raw(status) };
    if text.is_null() {
        return "unknown status".to_owned();
    }
    // SAFETY: `text` is non-null and points to a static, NUL-terminated string
    // as described above.
    unsafe { CStr::from_ptr(text) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_add_rejects_mismatched_lengths_without_native_call() {
        let lhs = [1.0_f32, 2.0];
        let rhs = [1.0_f32];
        let mut out = [0.0_f32; 2];
        assert_eq!(
            vector_add_f32(&lhs, &rhs, &mut out),
            Err(DeviceFailure {
                status: STATUS_INVALID_ARGUMENT,
                stage: STAGE_NONE,
            })
        );
    }

    #[test]
    fn vector_add_rejects_empty_input_without_native_call() {
        let mut out: [f32; 0] = [];
        assert_eq!(
            vector_add_f32(&[], &[], &mut out),
            Err(DeviceFailure {
                status: STATUS_INVALID_ARGUMENT,
                stage: STAGE_NONE,
            })
        );
    }

    const Q4K_SHAPE: Tiled = Tiled {
        ty: TiledType::Q4K,
        n_rows: 8,
        k: 256,
    };

    // The x, xf and y bytes of `Q4K_SHAPE` for one token are 1024, 1_048_592 and 32,
    // laid out with gaps so each rejection case below has a single cause.
    const UNIT_X: Region = Region {
        offset: 0,
        bytes: 1024,
    };
    const UNIT_XF: Region = Region {
        offset: 2048,
        bytes: 1_048_592,
    };
    const UNIT_Y: Region = Region {
        offset: 1_051_648,
        bytes: 32,
    };
    const UNIT_TOTAL: usize = 1_052_672;
    const UNIT_NEED: Activations = Activations {
        x: 1024,
        xf: 1_048_592,
        y: 32,
    };

    fn unit_work(device: &RawDevice) -> Work<'_> {
        Work {
            device,
            x: UNIT_X,
            xf: UNIT_XF,
            y: UNIT_Y,
        }
    }

    #[test]
    fn tiled_shapes_follow_the_vendored_tiles() {
        assert!(Q4K_SHAPE.fits());
        assert_eq!(Q4K_SHAPE.source_bytes(), Some(8 * 144));
        assert_eq!(Q4K_SHAPE.tile_vecs(), Some(72));
        let q6k = Tiled {
            ty: TiledType::Q6K,
            n_rows: 16,
            k: 512,
        };
        assert_eq!(q6k.source_bytes(), Some(16 * 2 * 210));
        assert_eq!(q6k.tile_vecs(), Some(2 * 2 * 112));
        let q80 = Tiled {
            ty: TiledType::Q80,
            n_rows: 8,
            k: 512,
        };
        assert_eq!(q80.source_bytes(), Some(8 * 16 * 34));
        assert_eq!(q80.tile_vecs(), Some(2 * 136));
        for (n_rows, k) in [(0, 256), (12, 256), (8, 0), (8, 300), (8, MAX_DIM + 256)] {
            let shape = Tiled {
                n_rows,
                k,
                ..Q4K_SHAPE
            };
            assert!(!shape.fits(), "{n_rows} x {k}");
        }
    }

    #[test]
    fn repack_rejects_wrong_lengths_without_native_call() {
        let src = vec![0u8; 8 * 144];
        let mut dst = vec![TileVec::default(); 72];
        assert_eq!(
            repack(Q4K_SHAPE, &src[1..], &mut dst),
            Err(STATUS_INVALID_ARGUMENT)
        );
        assert_eq!(
            repack(Q4K_SHAPE, &src, &mut dst[1..]),
            Err(STATUS_INVALID_ARGUMENT)
        );
        let bad = Tiled {
            n_rows: 4,
            ..Q4K_SHAPE
        };
        assert_eq!(repack(bad, &src, &mut dst), Err(STATUS_INVALID_ARGUMENT));
    }

    #[test]
    fn linear_rejects_bad_views_without_native_call() {
        // One 8-row, 256-value Q4_K tile is 72 vectors; the live handle is not
        // needed because every check runs before the call.
        let tile_bytes = 72 * 16;
        let device = RawDevice {
            ptr: NonNull::dangling(),
            bytes: 3 * tile_bytes,
        };
        let scratch = RawDevice {
            ptr: NonNull::dangling(),
            bytes: UNIT_TOTAL,
        };
        let x = [0.0_f32; 256];
        let mut y = [0.0_f32; 8];
        let run = |offset: usize, x: &[f32], y: &mut [f32]| {
            linear(Q4K_SHAPE, device.at(offset), unit_work(&scratch), x, y)
        };
        // Misaligned by one byte, past the end by one vector, offset overflow.
        assert_eq!(run(1, &x, &mut y), Err(invalid_argument()));
        assert_eq!(
            run(2 * tile_bytes + 16, &x, &mut y),
            Err(invalid_argument())
        );
        assert_eq!(run(usize::MAX - 15, &x, &mut y), Err(invalid_argument()));
        // Wrong input and output lengths.
        assert_eq!(run(0, &x[1..], &mut y), Err(invalid_argument()));
        assert_eq!(run(0, &x, &mut y[1..]), Err(invalid_argument()));
        // A scratch region one float too short for the input.
        let short = Work {
            x: Region {
                offset: 0,
                bytes: 1020,
            },
            ..unit_work(&scratch)
        };
        assert_eq!(
            linear(Q4K_SHAPE, device.at(0), short, &x, &mut y),
            Err(invalid_argument())
        );
    }

    #[test]
    fn the_adapter_rejects_a_misaligned_weight_pointer_before_any_cuda_call() {
        // The Rust check of `linear` rejects a misaligned offset first, so this
        // calls the FFI directly to reach the guard in `projector.cu`. The guard
        // returns before the first CUDA call, so a 16-byte-aligned host buffer
        // stands in for the allocation and no device is needed.
        let tiles = [TileVec::default(); 72];
        let x = [0.0_f32; 256];
        let mut y = [0.0_f32; 8];
        let mut backing = vec![TileVec::default(); UNIT_TOTAL / 16];
        let good = host_workspace(&mut backing);
        let mut stage = STAGE_LAUNCH;
        // SAFETY: `x` and `y` are live host arrays of 256 and 8 floats for the
        // call, and `stage` is a writable c_int. The weight pointer lies inside
        // `tiles`, the workspace regions inside `backing`, and the guard under
        // test returns before anything reads or writes them.
        let status = unsafe {
            linear_ws_raw(
                TiledType::Q4K.ggml(),
                tiles.as_ptr().cast::<u8>().add(1).cast(),
                8,
                256,
                x.as_ptr(),
                1,
                y.as_mut_ptr(),
                &good,
                &mut stage,
            )
        };
        assert_eq!(status, STATUS_INVALID_ARGUMENT);
        assert_eq!(stage, STAGE_NONE);
    }

    #[test]
    fn activation_bytes_follow_the_vendored_xf_layout() {
        assert_eq!(
            activation_bytes(8, 256, 1),
            Ok(Activations {
                x: 1024,
                xf: 1_048_592,
                y: 32
            })
        );
        // Sixteen tokens still use the fixed fragment-order buffer ...
        assert_eq!(
            activation_bytes(8, 256, 16),
            Ok(Activations {
                x: 16_384,
                xf: 1_048_592,
                y: 512
            })
        );
        // ... and the seventeenth switches to row-major halves.
        assert_eq!(
            activation_bytes(8, 256, 17),
            Ok(Activations {
                x: 17_408,
                xf: 8_704,
                y: 544
            })
        );
        assert_eq!(
            activation_bytes(8, 32768, 17),
            Ok(Activations {
                x: 2_228_224,
                xf: 1_114_112,
                y: 544
            })
        );
        for (n_rows, k, tokens) in [
            (8, 256, 0),
            (8, 256, 32769),
            (8, 300, 1),
            (12, 256, 1),
            (0, 256, 1),
        ] {
            assert_eq!(
                activation_bytes(n_rows, k, tokens),
                Err(STATUS_INVALID_ARGUMENT),
                "{n_rows} x {k} x {tokens}"
            );
        }
    }

    #[test]
    fn xf_bytes_is_one_fixed_buffer_up_to_sixteen_tokens_then_row_major_halves() {
        assert_eq!(xf_bytes(1, 256), Some(1_048_592));
        assert_eq!(xf_bytes(16, 32768), Some(1_048_592), "sixteen: still fixed");
        assert_eq!(xf_bytes(17, 256), Some(8_704), "seventeen: 17 * 256 halves");
        assert_eq!(xf_bytes(17, 32768), Some(1_114_112));
        assert_eq!(xf_bytes(usize::MAX, 2), None, "the byte count overflows");
        assert_eq!(pack_bytes(17, 256), Some((17_408, 8_704)));
        assert_eq!(pack_bytes(usize::MAX, 2), None);
    }

    /// Host only: a Q5_0 call sizes its activation regions on the 256 grid (the xf of K = 896 is
    /// 1024 wide with a zero tail), and no other type changes width.
    #[test]
    fn q5_0_activation_width_rounds_to_the_256_grid() {
        assert_eq!(activation_k(TiledType::Q50, 896), 1024);
        assert_eq!(activation_k(TiledType::Q50, 128), 256);
        assert_eq!(activation_k(TiledType::Q50, 1024), 1024);
        for ty in [TiledType::Q4K, TiledType::Q6K, TiledType::Q80] {
            assert_eq!(activation_k(ty, 896), 896, "{ty:?}");
            assert_eq!(activation_k(ty, 256), 256, "{ty:?}");
        }
    }

    /// Host only: the adapter's query touches no device. The Rust rule and the C
    /// rule it repeats must give the same bytes, because `pack` zeroes the size
    /// the C rule gives while the callers check a range against `xf_bytes`.
    #[test]
    fn the_rust_xf_rule_agrees_with_the_adapter() {
        for tokens in [1usize, 2, 15, 16, 17, 18, 64, 32768] {
            for k in [256usize, 4096, 32768] {
                let (m, c_k) = (
                    c_int::try_from(tokens).expect("small"),
                    c_int::try_from(k).expect("small"),
                );
                let mut bytes = Activations::default();
                // SAFETY: the three out pointers address live, writable usize
                // values for the call; the call touches no device.
                let status = unsafe {
                    workspace_bytes_raw(m, c_k, 8, &mut bytes.x, &mut bytes.xf, &mut bytes.y)
                };
                assert_eq!(status, STATUS_OK, "{tokens} tokens, k {k}");
                assert_eq!(Some(bytes.xf), xf_bytes(tokens, k), "{tokens} x {k}");
                assert_eq!(Some((bytes.x, bytes.xf)), pack_bytes(tokens, k));
            }
        }
    }

    const HELD: Activations = Activations {
        x: 10,
        xf: 20,
        y: 30,
    };

    #[test]
    fn a_workspace_covers_a_need_only_when_every_region_holds_it() {
        assert!(HELD.covers(HELD));
        assert!(HELD.covers(Activations::default()));
        let one_more = [
            ("x", Activations { x: 11, ..HELD }),
            ("xf", Activations { xf: 21, ..HELD }),
            ("y", Activations { y: 31, ..HELD }),
        ];
        for (label, need) in one_more {
            assert!(!HELD.covers(need), "{label} is one byte short");
            // Covering is not symmetric: the larger side holds the smaller.
            assert!(need.covers(HELD), "{label}");
        }
        // A need that is smaller in one region and larger in another is not covered.
        let mixed = Activations { x: 1, xf: 99, y: 1 };
        assert!(!HELD.covers(mixed));
    }

    #[test]
    fn the_union_of_two_sizes_takes_the_larger_of_each_region_in_either_order() {
        // Each pair makes every region the larger one in one operand and the
        // smaller one in the other.
        let pairs = [
            (
                Activations { x: 10, xf: 2, y: 3 },
                Activations {
                    x: 1,
                    xf: 20,
                    y: 30,
                },
            ),
            (
                Activations { x: 1, xf: 20, y: 3 },
                Activations {
                    x: 10,
                    xf: 2,
                    y: 30,
                },
            ),
            (
                Activations { x: 1, xf: 2, y: 30 },
                Activations {
                    x: 10,
                    xf: 20,
                    y: 3,
                },
            ),
        ];
        for (a, b) in pairs {
            assert_eq!(a.union(b), HELD, "{a:?} {b:?}");
            assert_eq!(b.union(a), HELD, "{b:?} {a:?}");
        }
        assert_eq!(HELD.union(HELD), HELD);
    }

    #[test]
    fn work_regions_that_cannot_hold_the_call_are_rejected() {
        let device = RawDevice {
            ptr: NonNull::dangling(),
            bytes: UNIT_TOTAL,
        };
        assert!(unit_work(&device).fits(UNIT_NEED));
        let at = |offset, bytes| Region { offset, bytes };
        let cases = [
            (
                "x too short",
                Work {
                    x: at(0, 1020),
                    ..unit_work(&device)
                },
            ),
            (
                "xf too short",
                Work {
                    xf: at(2048, 1_048_591),
                    ..unit_work(&device)
                },
            ),
            (
                "y too short",
                Work {
                    y: at(1_051_648, 28),
                    ..unit_work(&device)
                },
            ),
            (
                "y past the end",
                Work {
                    y: at(1_052_656, 32),
                    ..unit_work(&device)
                },
            ),
            (
                "offset overflow",
                Work {
                    y: at(usize::MAX - 3, 32),
                    ..unit_work(&device)
                },
            ),
            (
                "x misaligned",
                Work {
                    x: at(2, 1024),
                    ..unit_work(&device)
                },
            ),
            (
                "xf misaligned",
                Work {
                    xf: at(2056, 1_048_592),
                    ..unit_work(&device)
                },
            ),
            (
                "y misaligned",
                Work {
                    y: at(1_051_650, 32),
                    ..unit_work(&device)
                },
            ),
            (
                "x over xf",
                Work {
                    x: at(1536, 1024),
                    ..unit_work(&device)
                },
            ),
            (
                "xf over y",
                Work {
                    y: at(1_050_000, 32),
                    ..unit_work(&device)
                },
            ),
            (
                "y over x",
                Work {
                    y: at(512, 32),
                    ..unit_work(&device)
                },
            ),
        ];
        for (label, work) in cases {
            assert!(!work.fits(UNIT_NEED), "{label}");
        }
    }

    /// The unit layout inside one honest, 16-byte-aligned host allocation of
    /// `UNIT_TOTAL` bytes. The adapter guards return before any CUDA call touches it.
    fn host_workspace(backing: &mut [TileVec]) -> WorkspaceRaw {
        assert!(size_of_val(backing) >= UNIT_TOTAL);
        let base = backing.as_mut_ptr().cast::<u8>();
        // SAFETY: every offset lies inside `backing`.
        unsafe {
            WorkspaceRaw {
                x: base.add(UNIT_X.offset).cast(),
                x_bytes: UNIT_X.bytes,
                xf: base.add(UNIT_XF.offset).cast(),
                xf_bytes: UNIT_XF.bytes,
                y: base.add(UNIT_Y.offset).cast(),
                y_bytes: UNIT_Y.bytes,
            }
        }
    }

    #[test]
    fn the_adapter_rejects_unusable_workspaces_before_any_cuda_call() {
        let tiles = [TileVec::default(); 72];
        let x = [0.0_f32; 256];
        let mut y = [0.0_f32; 8];
        let mut backing = vec![TileVec::default(); UNIT_TOTAL / 16];
        let good = host_workspace(&mut backing);
        let mut call = |workspace: *const WorkspaceRaw| {
            let mut stage = STAGE_LAUNCH;
            // SAFETY: `x` and `y` are live host arrays of 256 and 8 floats and `stage`
            // a writable c_int. The weight pointer is 16-byte aligned and inside
            // `tiles`. Every case below fails a guard that returns before anything is
            // read or written, so no region is touched. The null case is the
            // guard against a null workspace pointer.
            let status = unsafe {
                linear_ws_raw(
                    TiledType::Q4K.ggml(),
                    tiles.as_ptr().cast(),
                    8,
                    256,
                    x.as_ptr(),
                    1,
                    y.as_mut_ptr(),
                    workspace,
                    &mut stage,
                )
            };
            (status, stage)
        };
        // Each case breaks one rule of `workspace_usable` and keeps the others, so a
        // guard that is missing lets exactly its case through. Only addresses move;
        // none is dereferenced.
        let broken = |change: fn(&mut WorkspaceRaw)| {
            let mut workspace = good;
            change(&mut workspace);
            workspace
        };
        let cases = [
            ("null x", broken(|w| w.x = std::ptr::null_mut())),
            ("null xf", broken(|w| w.xf = std::ptr::null_mut())),
            ("null y", broken(|w| w.y = std::ptr::null_mut())),
            ("x short", broken(|w| w.x_bytes = 1020)),
            ("xf short", broken(|w| w.xf_bytes = 1_048_591)),
            ("y short", broken(|w| w.y_bytes = 28)),
            ("x misaligned", broken(|w| w.x = w.x.wrapping_byte_add(2))),
            (
                "xf misaligned",
                broken(|w| w.xf = w.xf.wrapping_byte_add(8)),
            ),
            ("y misaligned", broken(|w| w.y = w.y.wrapping_byte_add(2))),
            ("x over xf", broken(|w| w.x = w.xf)),
            ("x over y", broken(|w| w.x = w.y)),
            ("xf over y", broken(|w| w.xf = w.xf.wrapping_byte_add(1024))),
        ];
        for (label, workspace) in cases {
            assert_eq!(
                call(&workspace),
                (STATUS_INVALID_ARGUMENT, STAGE_NONE),
                "{label}"
            );
        }
        assert_eq!(
            call(std::ptr::null()),
            (STATUS_INVALID_ARGUMENT, STAGE_NONE),
            "null workspace"
        );
    }

    #[test]
    fn group_bytes_is_the_tile_bytes_of_one_group() {
        for (ty, k, bytes) in [
            (TiledType::Q4K, 256, 1152),
            (TiledType::Q4K, 512, 2304),
            (TiledType::Q6K, 256, 1792),
            (TiledType::Q6K, 512, 3584),
            (TiledType::Q80, 256, 2176),
            (TiledType::Q80, 512, 4352),
        ] {
            let shape = Tiled { ty, n_rows: 24, k };
            assert_eq!(shape.group_bytes(), Some(bytes), "{ty:?} k={k}");
        }
    }

    #[test]
    fn upload_rejects_empty_input_without_native_call() {
        assert_eq!(upload(&[]).map(|_| ()), Err(invalid_argument()));
    }

    #[test]
    fn device_writes_reject_bad_ranges_without_native_call() {
        assert_eq!(alloc(0).map(|_| ()), Err(invalid_argument()));
        // A live handle is not needed: the range checks run before the call.
        let device = RawDevice {
            ptr: NonNull::dangling(),
            bytes: 16,
        };
        assert_eq!(upload_at(&device, 0, &[]), Err(invalid_argument()));
        assert_eq!(upload_at(&device, 9, &[0; 8]), Err(invalid_argument()));
        assert_eq!(
            upload_at(&device, usize::MAX, &[0; 8]),
            Err(invalid_argument())
        );
    }

    #[test]
    fn tiles_split_into_whole_groups() {
        let shape = Tiled {
            n_rows: 24,
            k: 512,
            ..Q4K_SHAPE
        };
        assert_eq!(shape.groups(), 3);
        let one = shape.with_groups(1);
        assert_eq!(one.n_rows, 8);
        assert_eq!(one.source_bytes(), Some(8 * 2 * 144));
        assert_eq!(one.tile_vecs().map(|vecs| vecs * 3), shape.tile_vecs());
    }

    #[test]
    fn tile_vectors_view_as_their_bytes() {
        let vecs = [TileVec([1; 16]), TileVec([2; 16])];
        let bytes = TileVec::as_bytes(&vecs);
        assert_eq!(bytes.len(), 32);
        assert_eq!(&bytes[..16], &[1; 16]);
        assert_eq!(&bytes[16..], &[2; 16]);
    }

    #[test]
    fn status_messages_describe_adapter_statuses() {
        assert_eq!(status_message(STATUS_OK), "success");
        assert_eq!(status_message(STATUS_INVALID_ARGUMENT), "invalid argument");
        assert_eq!(status_message(-2), "native exception");
        assert_eq!(status_message(-3), "unknown status");
    }

    #[test]
    fn cuda_statuses_use_the_runtime_description() {
        // cudaErrorMemoryAllocation. The runtime describes it without a device.
        let message = status_message(2);
        assert!(!message.is_empty());
        assert_ne!(message, "unknown status");
    }
}
