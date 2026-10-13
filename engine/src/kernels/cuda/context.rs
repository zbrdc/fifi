//! Device memory and the tiled matmul behind one CUDA adapter lease.
//!
//! A [`CudaContext`] holds one of the two process leases (decision 0024).
//! Every buffer it creates shares that lease, so the slot stays reserved until
//! the context and all its buffers are gone. Context and buffers stay on the
//! creating thread. Each native call runs inside the process dispatch guard.
//!
//! A buffer frees its device memory in `Drop`, after the adapter synchronizes
//! the device. A failed free latches the backend terminal. Drop never panics.
//! After a terminal failure no native call runs, so the remaining buffers are
//! left to the process exit instead of a poisoned context.
//!
//! Every public operation first reports a terminal failure as `Backend`
//! `execute`, then validates its input, then enters the dispatch guard.
//!
//! A capture window (`graph`) lets only launch-only calls through: while one is open on the
//! thread, every call above that copies synchronously, allocates or synchronizes is refused
//! with `BadRequest` before any native call, and a drop is deferred until the window closes.
//!
//! A tiled upload stages the host in whole 8-row groups: one chunk of source
//! blocks read from the caller plus one chunk of repacked tiles, each of at
//! most the chunk limit ([`CHUNK_LIMIT`] in production) however large the
//! tensor is. The device buffer is allocated once and each chunk lands at the
//! byte offset of its first group. Host memory stays near twice the chunk
//! limit, far inside the 1 GiB host bound of decision 0040.
//!
//! Every context owns one activation workspace (see `workspace`): the device
//! regions that a linear call reads its input from and writes its output to.
//! It is allocated with the context and grown, never shrunk, by a larger call,
//! so a call allocates and frees no device memory. The caller sizes it with an
//! `Admission` and says whether a larger call may grow it; a `DeviceArena`
//! serves what the workspace refuses, and an `ArenaHold` keeps the workspace from
//! being replaced while a graph uses it.

use std::{
    cell::RefCell,
    ffi::c_int,
    marker::PhantomData,
    rc::Rc,
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
};

use super::{
    ADAPTER_OPERATION, ALLOCATE_OPERATION, TERMINAL,
    guard::{DispatchGuard, Lease},
    memory_info_operation,
    native::{self, Activations, DeviceFailure, RawDevice, TILE_ROWS, TileVec, Tiled, TiledType},
    native_error, require_device,
};
use crate::EngError;
use crate::kernels::cuda::native::resident::matmul_q5_0::MAX_DENSE_TOKENS;
use crate::weights::GgufType;
use workspace::Workspace;

/// Adapter leases held in this process.
static SLOTS: AtomicUsize = AtomicUsize::new(0);

/// Device bytes held by live buffers of this process. On an integrated GPU
/// (Jetson) device memory is system RAM, so `cudaMemGetInfo` free memory moves
/// with the host heap and page cache and cannot show a leak. Tests read this
/// count instead. A buffer skipped after a terminal failure stays counted.
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Live streams of this process. Like [`LIVE_BYTES`], an owner skipped after a
/// terminal failure stays counted.
static LIVE_STREAMS: AtomicUsize = AtomicUsize::new(0);

/// Live events of this process. An owner skipped after a terminal failure stays
/// counted.
static LIVE_EVENTS: AtomicUsize = AtomicUsize::new(0);

/// Bytes of pinned host memory held by live owners of this process. An owner
/// skipped after a terminal failure stays counted.
static LIVE_PINNED: AtomicUsize = AtomicUsize::new(0);

/// The last stream id handed out. Ids start at 1, so 0 names no stream.
static STREAM_IDS: AtomicU64 = AtomicU64::new(0);

/// Live executable graphs of this process. An owner skipped after a terminal
/// failure stays counted.
static LIVE_GRAPHS: AtomicUsize = AtomicUsize::new(0);

/// Native destroy calls made by streams, events, pinned words and graphs. Incremented
/// right before the call, so a test proves that a terminal `Drop` makes none.
#[cfg(test)]
static DESTROY_CALLS: AtomicUsize = AtomicUsize::new(0);

/// The live resident owners of the process, for tests.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OwnerCounts {
    pub(crate) streams: usize,
    pub(crate) events: usize,
    pub(crate) pinned: usize,
    pub(crate) graphs: usize,
}

/// The largest buffer of repacked tiles held on the host at once, and so also
/// the largest chunk of source blocks (a repacked group is never smaller than
/// its source). Source blocks are read, repacked and uploaded in chunks of
/// whole 8-row groups, so the host holds at most two such buffers, 128 MiB,
/// however large the tensor is. A single group is the floor: one group of the
/// widest row (`k` = 32768) is under 300 KB.
pub(crate) const CHUNK_LIMIT: usize = 64 << 20;

/// The largest finite fp16 value. Repacked Q6_K stores `d * scale` as fp16.
const F16_MAX: f32 = 65504.0;

/// Free and total device memory in bytes, from `cudaMemGetInfo`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MemoryInfo {
    pub(crate) free: usize,
    pub(crate) total: usize,
}

/// One quantized checkpoint tensor with `n_rows` rows of `k` values, whose
/// ggml blocks are `source_len` bytes read through a [`BlockReader`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct QuantTensor<'a> {
    pub(crate) name: &'a str,
    pub(crate) ty: GgufType,
    pub(crate) n_rows: usize,
    pub(crate) k: usize,
    pub(crate) source_len: usize,
}

/// Fills a buffer with the tensor's source bytes that start at the given
/// byte offset. The upload asks for whole 8-row groups, in order.
pub(crate) type BlockReader<'a> = &'a mut dyn FnMut(usize, &mut [u8]) -> Result<(), EngError>;

/// One CUDA adapter lease on the first device.
pub(crate) struct CudaContext {
    lease: Rc<Lease>,
    workspace: RefCell<Workspace>,
    /// Whether a call larger than the workspace may grow it.
    growth: Growth,
    /// Workspace allocations made, the first one included; tests compare it.
    #[cfg(test)]
    workspace_allocations: std::cell::Cell<usize>,
    _thread_confined: PhantomData<*const ()>,
}

/// One device allocation owned by a [`CudaContext`].
pub(crate) struct DeviceBuffer {
    raw: Option<RawDevice>,
    lease: Rc<Lease>,
    _thread_confined: PhantomData<*const ()>,
}

/// A weight repacked into the vendored tile layout and resident on the device.
pub(crate) struct TiledWeight {
    buffer: DeviceBuffer,
    tiled: Tiled,
}

impl CudaContext {
    /// Takes one adapter lease and allocates the workspace that `admission` sizes.
    /// Returns `BadRequest` for an admission the adapter would reject, `Backend`
    /// when no device is present, when two leases are already held, or after a
    /// terminal failure, and `MemoryPlan` when the device cannot hold the
    /// workspace (the lease is released again).
    pub(crate) fn with_admission(admission: &Admission) -> Result<Self, EngError> {
        Self::build(admission.caps()?, admission.growth)
    }

    /// A context sized for the served model (`served_admission`), for the tests.
    #[cfg(test)]
    pub(crate) fn new() -> Result<Self, EngError> {
        Self::with_admission(&crate::family::project::cuda::served_admission())
    }

    /// Like [`Self::new`], with a growable workspace of the given sizes.
    #[cfg(test)]
    fn with_workspace(caps: Activations) -> Result<Self, EngError> {
        Self::build(caps, Growth::Unbounded)
    }

    fn build(caps: Activations, growth: Growth) -> Result<Self, EngError> {
        require_device(super::device_count()?)?;
        let lease = Lease::acquire(&SLOTS, &TERMINAL)?;
        let workspace = Workspace::allocate(&lease, caps)?;
        Ok(Self {
            lease,
            workspace: RefCell::new(workspace),
            growth,
            #[cfg(test)]
            workspace_allocations: std::cell::Cell::new(1),
            _thread_confined: PhantomData,
        })
    }

    /// Fails with `Backend` `execute` after a terminal failure.
    pub(crate) fn ensure_usable(&self) -> Result<(), EngError> {
        super::ensure_usable(&TERMINAL)
    }

    /// Reports free and total device memory.
    pub(crate) fn memory_info(&self) -> Result<MemoryInfo, EngError> {
        graph::ensure_closed("a memory query")?;
        let _dispatch = self.enter()?;
        let (free, total) = native::memory_info()
            .map_err(|status| native_error(&TERMINAL, memory_info_operation(status), status))?;
        Ok(MemoryInfo { free, total })
    }

    /// Device bytes held by the live buffers of every context in the process.
    /// The count is process-global, so tests compare it before and after.
    #[cfg(test)]
    pub(crate) fn live_bytes(&self) -> usize {
        LIVE_BYTES.load(Ordering::Acquire)
    }

    /// Live streams, events and pinned bytes of every context in the process.
    /// The counts are process-global, so tests compare them before and after.
    #[cfg(test)]
    pub(crate) fn live_owners(&self) -> OwnerCounts {
        OwnerCounts {
            streams: LIVE_STREAMS.load(Ordering::Acquire),
            events: LIVE_EVENTS.load(Ordering::Acquire),
            pinned: LIVE_PINNED.load(Ordering::Acquire),
            graphs: LIVE_GRAPHS.load(Ordering::Acquire),
        }
    }

    /// Native destroy calls made so far by streams, events, pinned words and graphs.
    #[cfg(test)]
    pub(crate) fn destroy_calls(&self) -> usize {
        DESTROY_CALLS.load(Ordering::Acquire)
    }

    /// Workspace allocations made so far, the first one included.
    #[cfg(test)]
    pub(crate) fn workspace_allocations(&self) -> usize {
        self.workspace_allocations.get()
    }

    /// Bytes of the current workspace allocation.
    #[cfg(test)]
    pub(crate) fn workspace_bytes(&self) -> usize {
        self.workspace.borrow().buffer.bytes()
    }

    /// Copies `bytes` into a new device allocation. Empty input is a
    /// `BadRequest`, because the device holds no zero-length allocation.
    pub(crate) fn upload(&self, bytes: &[u8]) -> Result<DeviceBuffer, EngError> {
        self.ensure_usable()?;
        graph::ensure_closed("an upload")?;
        if bytes.is_empty() {
            return Err(EngError::BadRequest("cuda upload of zero bytes".to_owned()));
        }
        let _dispatch = self.enter()?;
        let raw = native::upload(bytes).map_err(device_error)?;
        Ok(self.wrap(raw))
    }

    /// Copies a device buffer back to the host. `out` must not be longer than
    /// the buffer. Only the parity tests read device memory back.
    #[cfg(test)]
    pub(crate) fn download(&self, buffer: &DeviceBuffer, out: &mut [u8]) -> Result<(), EngError> {
        self.ensure_usable()?;
        graph::ensure_closed("a download")?;
        let raw = self.owned(buffer)?;
        if out.is_empty() || out.len() > raw.bytes() {
            return Err(EngError::BadRequest(format!(
                "cuda download of {} bytes from a buffer of {} bytes",
                out.len(),
                raw.bytes()
            )));
        }
        let _dispatch = self.enter()?;
        native::download(raw, out).map_err(device_error)
    }

    /// Reads a Q4_K, Q6_K, Q8_0 or Q5_0 tensor through `read` in chunks of whole
    /// 8-row groups, repacks each chunk on the host and uploads its tiles.
    ///
    /// A chunk holds as many groups as fit in `chunk_limit` tile bytes, and
    /// at least one. The device buffer is allocated once up front.
    ///
    /// Returns `Checkpoint` naming the tensor for another type, a shape the
    /// tiles cannot hold, or a wrong source length. A Q6_K or Q8_0 scale
    /// outside the fp16 range is the caller's to reject with
    /// [`check_q6k_scales`] or [`check_q8_0_scales`], which the load does
    /// during planning. Returns `MemoryPlan` when the device cannot hold the
    /// tiles or the host cannot allocate the chunks. An error from `read` is
    /// returned unchanged. A repack failure is an adapter failure and
    /// terminal. A failed read or upload frees the device tiles again.
    pub(crate) fn upload_tiled(
        &self,
        tensor: QuantTensor<'_>,
        chunk_limit: usize,
        read: BlockReader<'_>,
    ) -> Result<TiledWeight, EngError> {
        self.ensure_usable()?;
        graph::ensure_closed("a tiled upload")?;
        let tiled = checked_tiled(tensor)?;
        let tile_bytes = tiled
            .tile_vecs()
            .and_then(|vecs| vecs.checked_mul(size_of::<TileVec>()))
            .ok_or_else(|| {
                EngError::MemoryPlan(format!(
                    "{ALLOCATE_OPERATION}: tiles of tensor {} overflow the address space",
                    tensor.name
                ))
            })?;
        let buffer = self.alloc(tile_bytes)?;
        let plan = ChunkPlan::new(tensor, tiled, tile_bytes, chunk_limit);
        self.upload_groups(plan, &buffer, read)?;
        Ok(TiledWeight { buffer, tiled })
    }

    /// Returns `x · Wᵀ` for `x.len() / k` tokens of `k` values each.
    ///
    /// Returns `BadRequest` when `x` is not a whole number of tokens, holds
    /// none, or holds more than the matmul accepts, or when the weight belongs
    /// to another context.
    pub(crate) fn linear(&self, weight: &TiledWeight, x: &[f32]) -> Result<Vec<f32>, EngError> {
        self.linear_rows(weight, 0, weight.tiled.n_rows, x)
    }

    /// Like [`Self::linear`], for the `rows` rows of the weight from row
    /// `first` on, without copying them. The tiles are group-major, so a run of
    /// whole 8-row groups is contiguous in the device buffer and the matmul
    /// reads it at a byte offset. One expert of a stacked 3-D tensor is such a
    /// run. The result holds `rows` values per token.
    ///
    /// Returns `BadRequest` when `first` or `rows` is not a multiple of 8, when
    /// `rows` is 0 or the range ends past the weight, or for the input
    /// conditions of [`Self::linear`].
    ///
    /// Uses the context's workspace; allocates only when the call is larger than
    /// every earlier call (see `workspace`).
    pub(crate) fn linear_rows(
        &self,
        weight: &TiledWeight,
        first: usize,
        rows: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        self.ensure_usable()?;
        graph::ensure_closed("a blocking linear call")?;
        let raw = self.owned(&weight.buffer)?;
        let (view, offset) = row_range(weight.tiled, first, rows)?;
        let tokens = token_count(view, x.len())?;
        let mut y = host_floats(tokens * view.n_rows)?;
        let k_x = native::activation_k(view.ty, view.k);
        let need = native::activation_bytes(view.n_rows, k_x, tokens)
            .map_err(|status| native_error(&TERMINAL, ADAPTER_OPERATION, status))?;
        let _dispatch = self.enter()?;
        let mut workspace = self
            .workspace
            .try_borrow_mut()
            .map_err(|_| workspace_in_use())?;
        self.reserve_workspace(&mut workspace, need)?;
        // A grow frees the old allocation, and a failed free latches the
        // backend. `enter` checked the latch before that, so check it again.
        self.ensure_usable()?;
        let device = self.owned(&workspace.buffer)?;
        native::linear(view, raw.at(offset), workspace.work(device), x, &mut y)
            .map_err(device_error)?;
        Ok(y)
    }

    /// Allocates `bytes` of uninitialized device memory.
    fn alloc(&self, bytes: usize) -> Result<DeviceBuffer, EngError> {
        DeviceBuffer::allocate(&self.lease, bytes)
    }

    /// Reads, repacks and uploads the groups of `plan` one chunk at a time
    /// into `buffer`. The tiles are group-major, so each chunk lands at the
    /// byte offset of its first group.
    fn upload_groups(
        &self,
        plan: ChunkPlan<'_>,
        buffer: &DeviceBuffer,
        read: BlockReader<'_>,
    ) -> Result<(), EngError> {
        let raw = self.owned(buffer)?;
        let name = plan.tensor.name;
        let mut source = host_source(name, plan.per_chunk * plan.group_src)?;
        let mut staging = host_tiles(name, plan.per_chunk * plan.group_vecs())?;
        for first in (0..plan.groups).step_by(plan.per_chunk) {
            let part = plan
                .tiled
                .with_groups(plan.per_chunk.min(plan.groups - first));
            let src = &mut source[..part.groups() * plan.group_src];
            read(first * plan.group_src, src)?;
            let vecs = &mut staging[..part.groups() * plan.group_vecs()];
            repack_chunk(name, part, src, vecs)?;
            let _dispatch = self.enter()?;
            native::upload_at(raw, first * plan.group_bytes, TileVec::as_bytes(vecs))
                .map_err(device_error)?;
        }
        Ok(())
    }

    fn wrap(&self, raw: RawDevice) -> DeviceBuffer {
        DeviceBuffer::from_raw(&self.lease, raw)
    }

    fn enter(&self) -> Result<DispatchGuard<'static>, EngError> {
        enter_dispatch()
    }

    /// Returns the allocation of a buffer that this context created.
    fn owned<'b>(&self, buffer: &'b DeviceBuffer) -> Result<&'b RawDevice, EngError> {
        if !Rc::ptr_eq(&self.lease, &buffer.lease) {
            return Err(EngError::BadRequest(
                "the CUDA buffer belongs to another adapter".to_owned(),
            ));
        }
        buffer
            .raw
            .as_ref()
            .ok_or_else(|| EngError::BadRequest("the CUDA buffer was already freed".to_owned()))
    }
}

/// Enters the dispatch guard, or fails with `execute` after a terminal failure.
fn enter_dispatch() -> Result<DispatchGuard<'static>, EngError> {
    let dispatch = DispatchGuard::enter(&TERMINAL);
    super::ensure_usable(&TERMINAL)?;
    Ok(dispatch)
}

impl DeviceBuffer {
    /// Bytes of the device allocation.
    pub(crate) fn bytes(&self) -> usize {
        self.raw.as_ref().map_or(0, RawDevice::bytes)
    }

    /// Allocates `bytes` of uninitialized device memory on the adapter of `lease`.
    fn allocate(lease: &Rc<Lease>, bytes: usize) -> Result<Self, EngError> {
        graph::ensure_closed("an allocation")?;
        let _dispatch = enter_dispatch()?;
        let raw = native::alloc(bytes).map_err(device_error)?;
        Ok(Self::from_raw(lease, raw))
    }

    /// Takes ownership of a device allocation and counts its bytes as live.
    fn from_raw(lease: &Rc<Lease>, raw: RawDevice) -> Self {
        LIVE_BYTES.fetch_add(raw.bytes(), Ordering::AcqRel);
        Self {
            raw: Some(raw),
            lease: Rc::clone(lease),
            _thread_confined: PhantomData,
        }
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            // The free synchronizes the device, which a capture window forbids.
            graph::run_or_defer(move || release_buffer(raw));
        }
    }
}

/// Frees one allocation under decision 0022, for `Drop` and for the frees a capture window kept.
fn release_buffer(raw: RawDevice) {
    let _dispatch = DispatchGuard::enter(&TERMINAL);
    // The context may be poisoned after a terminal failure. `RawDevice` has
    // no `Drop`, so skipping `free` leaves the memory to the process exit.
    if TERMINAL.load(Ordering::Acquire) {
        return;
    }
    let bytes = raw.bytes();
    match native::free(raw) {
        Ok(()) => {
            LIVE_BYTES.fetch_sub(bytes, Ordering::AcqRel);
        }
        Err(_) => TERMINAL.store(true, Ordering::Release),
    }
}

impl TiledWeight {
    /// Rows (output values per token) of the weight.
    pub(crate) fn n_rows(&self) -> usize {
        self.tiled.n_rows
    }

    /// Values per row (input values per token) of the weight.
    pub(crate) fn k(&self) -> usize {
        self.tiled.k
    }

    /// The device tiles, for parity checks of the repacked layout.
    pub(crate) fn buffer(&self) -> &DeviceBuffer {
        &self.buffer
    }
}

/// Drops the page cache of `len` bytes of `file` from `offset`.
///
/// The call is host only, so its failure leaves the backend usable. After a
/// terminal failure it still reports `Backend` `execute`, like every call.
pub(crate) fn drop_page_cache(file: &std::fs::File, offset: u64, len: u64) -> Result<(), EngError> {
    super::ensure_usable(&TERMINAL)?;
    native::drop_page_cache(file, offset, len).map_err(|failure| {
        // Errno 0 means the adapter rejected the call itself, and
        // `io::Error` would render it as "Success".
        let cause = match failure.errno {
            0 => String::new(),
            errno => format!(": {}", std::io::Error::from_raw_os_error(errno)),
        };
        EngError::Backend {
            operation: ADAPTER_OPERATION,
            message: format!(
                "posix_fadvise(DONTNEED) of {len} bytes at {offset} failed with status {}, errno {}{cause}",
                failure.status, failure.errno
            ),
        }
    })
}

/// Maps a failed device call to the error of its stage.
fn device_error(failure: DeviceFailure) -> EngError {
    native_error(&TERMINAL, graph::operation_of(&failure), failure.status)
}

/// A nested borrow of the workspace; no code path takes one today.
fn workspace_in_use() -> EngError {
    EngError::Backend {
        operation: ADAPTER_OPERATION,
        message: "the CUDA workspace is already in use".to_owned(),
    }
}

/// How a tensor splits into chunks of whole 8-row groups.
#[derive(Clone, Copy)]
struct ChunkPlan<'a> {
    tensor: QuantTensor<'a>,
    tiled: Tiled,
    groups: usize,
    /// Groups per chunk: at least one, and as many as fit in the chunk limit
    /// of tile bytes. A group's source is never larger than its tiles, so the
    /// source chunk obeys the same limit.
    per_chunk: usize,
    /// Source bytes of one group.
    group_src: usize,
    /// Device bytes of one group's tiles.
    group_bytes: usize,
}

impl<'a> ChunkPlan<'a> {
    /// `tile_bytes` is the device size of the whole tensor, which the caller
    /// computed from `tiled`; `tensor.source_len` was checked against it.
    fn new(tensor: QuantTensor<'a>, tiled: Tiled, tile_bytes: usize, chunk_limit: usize) -> Self {
        let groups = tiled.groups();
        let group_bytes = tile_bytes / groups;
        // `linear_rows` finds a row range at `Tiled::group_bytes` per group and
        // the upload writes group `g` at `g * group_bytes`; they must agree.
        debug_assert_eq!(
            tiled.group_bytes(),
            Some(group_bytes),
            "upload and row-range reads disagree on the bytes of a group"
        );
        Self {
            tensor,
            tiled,
            groups,
            per_chunk: (chunk_limit / group_bytes).clamp(1, groups),
            group_src: tensor.source_len / groups,
            group_bytes,
        }
    }

    fn group_vecs(&self) -> usize {
        self.group_bytes / size_of::<TileVec>()
    }
}

fn checked_tiled(tensor: QuantTensor<'_>) -> Result<Tiled, EngError> {
    let ty = match tensor.ty {
        GgufType::Q4K => TiledType::Q4K,
        GgufType::Q6K => TiledType::Q6K,
        GgufType::Q80 => TiledType::Q80,
        GgufType::Q50 => TiledType::Q50,
        other => {
            return Err(EngError::Checkpoint(format!(
                "tensor {} has type {other:?}, which the CUDA matmul does not execute",
                tensor.name
            )));
        }
    };
    let tiled = Tiled {
        ty,
        n_rows: tensor.n_rows,
        k: tensor.k,
    };
    if !tiled.fits() {
        return Err(EngError::Checkpoint(format!(
            "tensor {} has {} rows of {} values, which the CUDA tiles cannot hold",
            tensor.name, tensor.n_rows, tensor.k
        )));
    }
    if tiled.source_bytes() != Some(tensor.source_len) {
        return Err(EngError::Checkpoint(format!(
            "tensor {} has {} bytes of blocks for {} rows of {} values",
            tensor.name, tensor.source_len, tensor.n_rows, tensor.k
        )));
    }
    Ok(tiled)
}

/// Rejects a Q6_K tensor whose `d * scale` would overflow the fp16 scale of
/// the repacked tile (REPLY-17 item 1). The vendored repack would store
/// infinity there. `CudaProjector::load` runs it for every Q6_K tensor
/// before the first upload, so `upload_tiled` does not repeat it.
///
/// `blocks` may be one chunk of the tensor that starts at block `first_block`;
/// the error names the block by its index from the start of the tensor.
pub(crate) fn check_q6k_scales(
    name: &str,
    first_block: usize,
    blocks: &[u8],
) -> Result<(), EngError> {
    for (offset, block) in blocks
        .chunks_exact(TiledType::Q6K.bytes_per_256())
        .enumerate()
    {
        let index = first_block + offset;
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        for &scale in &block[192..208] {
            // NaN (from a NaN or infinite `d`) is rejected with the overflow.
            let product = (d * f32::from(scale.cast_signed())).abs();
            if product.is_nan() || product > F16_MAX {
                return Err(EngError::Checkpoint(format!(
                    "tensor {name} block {index}: Q6_K scale product {product} exceeds the fp16 range"
                )));
            }
        }
    }
    Ok(())
}

/// Bytes of one Q8_0 block: an fp16 `d` and 32 signed quants.
const Q8_0_BLOCK_BYTES: usize = 34;

/// Rejects a Q8_0 tensor whose `d * q` would leave the fp16 range when the
/// device dequantizes it (REPLY-20 item 3): `|d| * max|q| > 65504` over the
/// block's quants, with `q = -128` counted as 128, or a NaN or infinite `d`.
/// `CudaProjector::load` runs it for every Q8_0 tensor before the first upload.
///
/// `blocks` may be one chunk of the tensor that starts at block `first_block`;
/// the error names the block by its index from the start of the tensor.
pub(crate) fn check_q8_0_scales(
    name: &str,
    first_block: usize,
    blocks: &[u8],
) -> Result<(), EngError> {
    for (offset, block) in blocks.as_chunks::<Q8_0_BLOCK_BYTES>().0.iter().enumerate() {
        let index = first_block + offset;
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let max_q = block[2..]
            .iter()
            .map(|&q| q.cast_signed().unsigned_abs())
            .max()
            .unwrap_or(0);
        // A NaN or infinite `d` fails even when every quant is zero.
        let product = d.abs() * f32::from(max_q);
        if !d.is_finite() || product > F16_MAX {
            return Err(EngError::Checkpoint(format!(
                "tensor {name} block {index}: Q8_0 scale {d} times quant {max_q} exceeds the fp16 range"
            )));
        }
    }
    Ok(())
}

/// Bytes of one Q5_0 block: an fp16 `d`, four `qh` bytes and 16 `qs` bytes.
const Q5_0_BLOCK_BYTES: usize = 22;
/// Largest `|q5 - 16|` of a Q5_0 quant (`q5` is 0..=31).
const Q5_0_MAX_QUANT: f32 = 16.0;

/// Rejects a Q5_0 tensor whose `(q5 - 16) * d` would leave the fp16 range when the device
/// dequantizes it (MEL-C1b, B-L4): `16 |d| > 65504`, or a NaN or infinite `d`. The quants are
/// unconstrained, so the bound is on `d` alone (`|d| <= 4094`). `CudaProjector::load` runs it
/// for every Q5_0 tensor before the first upload.
///
/// `blocks` may be one chunk of the tensor that starts at block `first_block`; the error names
/// the block by its index from the start of the tensor.
pub(crate) fn check_q5_0_scales(
    name: &str,
    first_block: usize,
    blocks: &[u8],
) -> Result<(), EngError> {
    for (offset, block) in blocks.as_chunks::<Q5_0_BLOCK_BYTES>().0.iter().enumerate() {
        let index = first_block + offset;
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        if !d.is_finite() || d.abs() * Q5_0_MAX_QUANT > F16_MAX {
            return Err(EngError::Checkpoint(format!(
                "tensor {name} block {index}: Q5_0 scale {d} times quant {Q5_0_MAX_QUANT} exceeds the fp16 range"
            )));
        }
    }
    Ok(())
}

/// Decodes an IEEE binary16 value. `gguf_dequant` keeps its own private copy;
/// this one feeds the range check above and the tile checks of the tests.
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    let magnitude = match exponent {
        0 => mantissa * 2.0_f32.powi(-24),
        31 if mantissa == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + mantissa / 1024.0) * 2.0_f32.powi(exponent - 15),
    };
    sign * magnitude
}

/// Repacks one chunk of whole groups into `staging`. Q5_0 is a host byte
/// permutation; the other types go through the vendored repack. A future
/// repack-time conversion (MEL-C2) is a new arm here, never a decode-path branch.
fn repack_chunk(
    name: &str,
    part: Tiled,
    src: &[u8],
    staging: &mut [TileVec],
) -> Result<(), EngError> {
    if part.ty == TiledType::Q50 {
        return tile_q5_0::pack_q5_0(src, part.k, TileVec::as_bytes_mut(staging));
    }
    native::repack(part, src, staging).map_err(|status| repack_error(name, status))
}

/// A repack failure is an adapter rejection or exception, which is terminal
/// (decision 0040 item 9) even though no CUDA call ran.
fn repack_error(name: &str, status: c_int) -> EngError {
    match native_error(&TERMINAL, ADAPTER_OPERATION, status) {
        EngError::Backend { operation, message } => EngError::Backend {
            operation,
            message: format!("repack of tensor {name}: {message}"),
        },
        other => other,
    }
}

fn host_source(name: &str, len: usize) -> Result<Vec<u8>, EngError> {
    let mut source = Vec::new();
    source.try_reserve_exact(len).map_err(|error| {
        EngError::MemoryPlan(format!(
            "{ALLOCATE_OPERATION}: host source chunk of tensor {name}: {error}"
        ))
    })?;
    source.resize(len, 0);
    Ok(source)
}

fn host_tiles(name: &str, vecs: usize) -> Result<Vec<TileVec>, EngError> {
    let mut tiles = Vec::new();
    tiles.try_reserve_exact(vecs).map_err(|error| {
        EngError::MemoryPlan(format!(
            "{ALLOCATE_OPERATION}: host tiles of tensor {name}: {error}"
        ))
    })?;
    tiles.resize(vecs, TileVec::default());
    Ok(tiles)
}

/// The shape of rows `first..first + rows` of `tiled` and the byte offset of
/// their tiles. Both bounds are whole 8-row groups, so the offset is a multiple
/// of the 16-byte tile vector.
fn row_range(tiled: Tiled, first: usize, rows: usize) -> Result<(Tiled, usize), EngError> {
    let end = first.checked_add(rows).filter(|&end| end <= tiled.n_rows);
    let whole_groups = first.is_multiple_of(TILE_ROWS) && rows.is_multiple_of(TILE_ROWS);
    if rows == 0 || end.is_none() || !whole_groups {
        return Err(EngError::BadRequest(format!(
            "cuda linear rows: {rows} rows from row {first} are not whole 8-row groups of a weight with {} rows",
            tiled.n_rows
        )));
    }
    let group_bytes = tiled
        .group_bytes()
        .ok_or_else(|| EngError::BadRequest("cuda linear group size overflows".to_owned()))?;
    let offset = (first / TILE_ROWS)
        .checked_mul(group_bytes)
        .ok_or_else(|| {
            EngError::BadRequest(format!("cuda linear offset of row {first} overflows"))
        })?;
    Ok((
        Tiled {
            n_rows: rows,
            ..tiled
        },
        offset,
    ))
}

/// `BadRequest` unless a Q5_0 weight is asked for 1 to 16 tokens, the most its decode kernel takes.
/// Raised in Rust before any native call, because a native rejection latches the backend
/// terminal. Other types are unaffected. Prefill on a Q5_0 weight waits for the MMQ read patch (V-5b).
fn check_q5_0_tokens(ty: TiledType, tokens: usize) -> Result<(), EngError> {
    if ty == TiledType::Q50 && !(1..=MAX_DENSE_TOKENS).contains(&tokens) {
        return Err(EngError::BadRequest(format!(
            "the CUDA Q5_0 decode matmul takes 1 to {MAX_DENSE_TOKENS} tokens, got {tokens}; \
             prefill is V-5b"
        )));
    }
    Ok(())
}

fn token_count(tiled: Tiled, values: usize) -> Result<usize, EngError> {
    let tokens = values / tiled.k;
    check_q5_0_tokens(tiled.ty, tokens)?;
    if tokens == 0 || tokens * tiled.k != values || tokens > native::MAX_DIM {
        return Err(EngError::BadRequest(format!(
            "cuda linear input has {values} values; the weight takes 1 to {} tokens of {}",
            native::MAX_DIM,
            tiled.k
        )));
    }
    Ok(tokens)
}

fn host_floats(len: usize) -> Result<Vec<f32>, EngError> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|error| {
        EngError::MemoryPlan(format!(
            "{ALLOCATE_OPERATION}: host output of {len} floats: {error}"
        ))
    })?;
    out.resize(len, 0.0);
    Ok(out)
}

mod buffer_io;
mod device_arena;
mod expert_bank;
mod graph;
mod launch;
mod native_weight;
mod pinned;
mod pinned_buf;
mod range;
mod stream;
mod tile_q5_0;
mod workspace;

pub(crate) use device_arena::{ArenaHold, DeviceArena};
pub(crate) use expert_bank::{ExpertBank, ExpertBankGeometry};
pub(crate) use graph::{CapturePlan, GraphExec, is_capture_refusal};
pub(in crate::kernels::cuda) use launch::{Claim, require_apart, require_fit};
pub(crate) use launch::{MatmulIo, PackCall, Projection};
pub(crate) use native_weight::NativeWeight;
pub(crate) use pinned::{PinnedWord, WordValue};
pub(crate) use pinned_buf::PinnedBuf;
pub(crate) use range::DeviceRange;
pub(crate) use stream::{DeviceEvent, DeviceStream};
pub(crate) use workspace::{Admission, Growth, WorkspaceShape};

#[cfg(test)]
mod tests;

/// The upload and the row-range reads must agree on the bytes of a group.
#[cfg(test)]
mod group_stride_tests {
    use super::*;

    /// A tensor of `n_rows` rows of `k` values and its plan, with no chunk limit.
    fn plan_of(ty: GgufType, n_rows: usize, k: usize) -> (ChunkPlan<'static>, Tiled, usize) {
        let block_bytes = match ty {
            GgufType::Q4K => TiledType::Q4K.bytes_per_256(),
            GgufType::Q6K => TiledType::Q6K.bytes_per_256(),
            _ => TiledType::Q80.bytes_per_256(),
        };
        let tensor = QuantTensor {
            name: "w",
            ty,
            n_rows,
            k,
            source_len: n_rows * (k / 256) * block_bytes,
        };
        let tiled = checked_tiled(tensor).expect("a valid shape");
        let tile_bytes = tiled.tile_vecs().expect("tile vectors") * size_of::<TileVec>();
        (
            ChunkPlan::new(tensor, tiled, tile_bytes, usize::MAX),
            tiled,
            tile_bytes,
        )
    }

    #[test]
    fn the_upload_stride_is_the_row_range_stride_for_every_tiled_type() {
        for (ty, k, group_bytes) in [
            (GgufType::Q4K, 256, 1152),
            (GgufType::Q4K, 512, 2304),
            (GgufType::Q6K, 256, 1792),
            (GgufType::Q6K, 512, 3584),
            (GgufType::Q80, 256, 2176),
            (GgufType::Q80, 512, 4352),
        ] {
            let (plan, tiled, _) = plan_of(ty, 24, k);
            assert_eq!(plan.group_bytes, group_bytes, "{ty:?} k={k}");
            assert_eq!(tiled.group_bytes(), Some(group_bytes), "{ty:?} k={k}");
            for group in 0..3 {
                let (_, offset) = row_range(tiled, group * TILE_ROWS, TILE_ROWS).expect("a group");
                assert_eq!(
                    offset,
                    group * plan.group_bytes,
                    "{ty:?} k={k} group {group}"
                );
            }
        }
    }

    #[test]
    fn each_expert_of_a_stacked_tensor_starts_where_the_upload_wrote_it() {
        // Three experts of 16 rows stacked into one 3-D tensor of 48 rows.
        let (plan, tiled, tile_bytes) = plan_of(GgufType::Q6K, 48, 512);
        for expert in 0..3 {
            let (view, offset) = row_range(tiled, expert * 16, 16).expect("one expert");
            assert_eq!(view.n_rows, 16);
            assert_eq!(offset, expert * tile_bytes / 3, "expert {expert}");
            assert_eq!(offset, expert * 2 * plan.group_bytes, "expert {expert}");
        }
    }
}
