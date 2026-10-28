//! Device memory and projections behind one CUDA adapter lease.
//!
//! A [`CudaContext`] holds one of the two process leases (decision 0024).
//! Every buffer it creates shares that lease, so the slot stays reserved until
//! the context and all its buffers are gone. Context and buffers stay on the
//! creating thread. Each native call runs inside the process dispatch guard.
//!
//! Uncached buffers attempt a synchronized free in `Drop`. Opted-in buffers synchronize
//! before exact-size idle reuse, or stay charged in quarantine after a terminal failure.
//! A failed free latches the backend terminal. Drop never panics.
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
use crate::kernels::cuda::prefill::mmq_facts::MmqFactsCache;
use crate::weights::GgufType;
use workspace::Workspace;

mod alloc_cache;

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
    pinned_stage: RefCell<Option<pinned_stage::Stage>>,
    /// Whether a call larger than the workspace may grow it.
    growth: Growth,
    /// The MMQ device facts of this context's device, read at most once.
    pub(in crate::kernels::cuda) mmq_facts: MmqFactsCache,
    // Device region of one argmax reduction, allocated on first use.
    argmax_scratch: RefCell<Option<DeviceBuffer>>,
    /// Workspace allocations made, the first one included; tests compare it.
    #[cfg(test)]
    workspace_allocations: std::cell::Cell<usize>,
    #[cfg(all(test, feature = "native-cuda"))]
    successful_allocations: std::cell::Cell<usize>,
    _thread_confined: PhantomData<*const ()>,
    alloc_cache: Option<Rc<alloc_cache::AllocCache>>,
    /// Keep this field last. It returns credit only after native owners finish cleanup.
    pinned_stage_admission: Option<pinned_stage::StageAdmission>,
}

/// One device allocation owned by a [`CudaContext`].
pub(crate) struct DeviceBuffer {
    raw: Option<RawDevice>,
    lease: Rc<Lease>,
    _thread_confined: PhantomData<*const ()>,
    alloc_cache: Option<Rc<alloc_cache::AllocCache>>,
}

/// A weight repacked into the vendored tile layout and resident on the device.
pub(crate) struct TiledWeight {
    buffer: DeviceBuffer,
    tiled: Tiled,
}

impl CudaContext {
    /// Plans one fixed single-row workspace and its lazy head scratch.
    ///
    /// This calls host-only sizing and does not acquire a device lease.
    pub(crate) fn fixed_projection_plan(
        [rows, columns]: [usize; 2],
    ) -> Result<(Admission, u64), EngError> {
        // Invalid axes are a client error regardless of the backend state:
        // the pure validation runs before the usability gate, so a dead
        // backend still answers BadRequest for bad shapes (the contract
        // test rejects invalid axes before any device work).
        let invalid = || {
            EngError::BadRequest(
                "cuda fixed projection dimensions exceed backend bounds".to_owned(),
            )
        };
        if rows == 0 || columns == 0 || columns > native::MAX_DIM {
            return Err(invalid());
        }
        let rows = rows
            .checked_next_multiple_of(TILE_ROWS)
            .ok_or_else(invalid)?;
        let columns = columns
            .checked_next_multiple_of(native::TILE_COLS)
            .ok_or_else(invalid)?;
        if rows > c_int::MAX as usize || columns > native::MAX_DIM {
            return Err(invalid());
        }
        super::ensure_usable(&TERMINAL)?;
        let admission = Admission {
            shapes: vec![WorkspaceShape {
                n_rows: rows,
                k: columns,
                tokens: 1,
            }],
            growth: Growth::Fixed,
        };
        let _dispatch = enter_dispatch()?;
        let workspace = workspace::place(admission.caps()?)?;
        let bytes = workspace
            .total
            .checked_add(native::ARGMAX_SCRATCH_BYTES)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| {
                EngError::MemoryPlan("cuda fixed projection byte count overflows".to_owned())
            })?;
        Ok((admission, bytes))
    }

    /// Takes one adapter lease and allocates the workspace that `admission` sizes.
    /// Returns `BadRequest` for an admission the adapter would reject, `Backend`
    /// when no device is present, when two leases are already held, or after a
    /// terminal failure, and `MemoryPlan` when the device cannot hold the
    /// workspace (the lease is released again).
    pub(crate) fn with_admission(admission: &Admission) -> Result<Self, EngError> {
        Self::build(admission.caps()?, admission.growth)
    }

    /// Opts this lease into bounded backing reuse; other constructors stay uncached.
    pub(crate) fn with_admission_cache(
        admission: &Admission,
        enabled: bool,
    ) -> Result<Self, EngError> {
        Self::with_admission(admission)?.attach_alloc_cache(enabled)
    }

    pub(crate) fn alloc_cache_reserve(enabled: bool) -> u64 {
        if enabled {
            alloc_cache::MAX_IDLE_BYTES as u64
        } else {
            0
        }
    }

    #[cfg(test)]
    pub(crate) fn alloc_cache_enabled(&self) -> bool {
        self.alloc_cache.is_some()
    }

    pub(crate) fn drain_alloc_cache(&self) -> Result<(), EngError> {
        match &self.alloc_cache {
            Some(cache) => cache.drain(),
            None => Ok(()),
        }
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

    #[cfg(test)]
    fn with_workspace_cache(caps: Activations, enabled: bool) -> Result<Self, EngError> {
        Self::with_workspace(caps)?.attach_alloc_cache(enabled)
    }

    fn attach_alloc_cache(mut self, enabled: bool) -> Result<Self, EngError> {
        if enabled {
            self.alloc_cache = Some(Rc::new(alloc_cache::AllocCache::new(Rc::clone(
                &self.lease,
            ))?));
        }
        Ok(self)
    }

    fn build(caps: Activations, growth: Growth) -> Result<Self, EngError> {
        require_device(super::device_count()?)?;
        let lease = Lease::acquire(&SLOTS, &TERMINAL)?;
        let workspace = Workspace::allocate(&lease, caps)?;
        Ok(Self::from_workspace(lease, workspace, growth))
    }

    fn from_workspace(lease: Rc<Lease>, workspace: Workspace, growth: Growth) -> Self {
        Self {
            lease,
            workspace: RefCell::new(workspace),
            growth,
            mmq_facts: MmqFactsCache::default(),
            argmax_scratch: RefCell::new(None),
            pinned_stage: RefCell::new(None),
            #[cfg(test)]
            workspace_allocations: std::cell::Cell::new(1),
            #[cfg(all(test, feature = "native-cuda"))]
            successful_allocations: std::cell::Cell::new(0),
            _thread_confined: PhantomData,
            alloc_cache: None,
            pinned_stage_admission: None,
        }
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

    /// The driver's module loading mode (KV8-B B4'): 1 is `CU_MODULE_EAGER_LOADING` and 2 is
    /// `CU_MODULE_LAZY_LOADING` (cuda.h of the build image), through the adapter's own query
    /// entry. The device-side observation point of the adapter's eager-loading constructor.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the query entry of the KV8-B B4' pins, called by their tests only"
        )
    )]
    pub(crate) fn module_loading_mode(&self) -> Result<i32, EngError> {
        self.ensure_usable()?;
        let _dispatch = self.enter()?;
        native::module_loading_mode()
            .map_err(|status| native_error(&TERMINAL, "inference_cuda_module_loading_mode", status))
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

    /// Successful `ctx.alloc` requests; native upload/workspace allocations are separate.
    #[cfg(all(test, feature = "native-cuda"))]
    pub(crate) fn successful_allocations(&self) -> usize {
        self.successful_allocations.get()
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
        if self.alloc_cache.is_some() {
            let buffer = self.alloc(bytes.len())?;
            let raw = buffer.raw.as_ref().expect("newly owned buffer");
            if let Some(cache) = &buffer.alloc_cache {
                cache.copy(raw, bytes)?;
            } else {
                native::upload_at(raw, 0, bytes).map_err(device_error)?;
            }
            return Ok(buffer);
        }
        let raw = native::upload(bytes).map_err(device_error)?;
        Ok(self.wrap(raw))
    }

    /// Copies a device buffer back to the host. `out` must not be longer than
    /// the buffer. The device chain reads kernel outputs and the small
    /// values a caller must see; the same-stream cudaMemcpy provides
    /// ordering for launch-only kernels.
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
    /// terminal. A failed read or upload drops the device owner. Cleanup runs
    /// only while the backend remains usable.
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

    /// Returns `W · x` for a contiguous row-major F32 weight of `[rows, columns]`.
    /// Rejects invalid shapes, extents, inputs and foreign buffers as `BadRequest`.
    /// This staged binding has no production family caller yet.
    #[allow(
        dead_code,
        reason = "staged Nemotron F32 binding; family caller follows separately"
    )]
    pub(crate) fn linear_f32(
        &self,
        weight: &DeviceBuffer,
        shape: [usize; 2],
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        self.ensure_usable()?;
        let raw = self.owned(weight)?;
        let (weight_bytes, need) = native::f32_projection_bytes(shape).ok_or_else(|| {
            EngError::BadRequest("cuda F32 projection shape is empty or overflows".to_owned())
        })?;
        if raw.bytes() < weight_bytes || x.len() != shape[1] || !x.iter().all(|v| v.is_finite()) {
            return Err(EngError::BadRequest(
                "cuda F32 projection requires sufficient weight bytes and one finite input row"
                    .to_owned(),
            ));
        }
        let mut y = host_floats(shape[0])?;
        let _dispatch = self.enter()?;
        let mut workspace = self
            .workspace
            .try_borrow_mut()
            .map_err(|_| workspace_in_use())?;
        self.reserve_workspace(&mut workspace, need)?;
        // Growth frees the old allocation, which can latch a terminal failure.
        self.ensure_usable()?;
        let device = self.owned(&workspace.buffer)?;
        native::linear_f32(raw, shape, workspace.work(device), x, &mut y).map_err(device_error)?;
        Ok(y)
    }

    /// Computes `W_nvfp4 * x` for one finite host row against packed NVFP4
    /// device weights, without expanding the matrix to F32 on the host.
    /// `packed` holds one nibble per value, low nibble first, `scale` one
    /// E4M3FN byte per group of sixteen values, both row-major, and
    /// `weight_scale_2` is the F32 per-tensor multiplier. Returns
    /// `BadRequest` for a shape, extent or input the host reference rejects,
    /// and for a group scale or decoded weight the device flags as not
    /// finite.
    pub(crate) fn nvfp4_matvec(
        &self,
        packed: &DeviceBuffer,
        scale: &DeviceBuffer,
        weight_scale_2: f32,
        shape: [usize; 2],
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        self.ensure_usable()?;
        let packed_raw = self.owned(packed)?;
        let scale_raw = self.owned(scale)?;
        let (packed_bytes, scale_bytes, need) =
            native::nvfp4_matvec_bytes(shape).ok_or_else(|| {
                EngError::BadRequest(
                    "cuda NVFP4 matvec: matrix dimensions must be positive and group aligned"
                        .to_owned(),
                )
            })?;
        if packed.bytes() != packed_bytes {
            return Err(EngError::BadRequest(
                "cuda NVFP4 matvec: stored packed bytes differ from the matrix shape".to_owned(),
            ));
        }
        if scale.bytes() != scale_bytes {
            return Err(EngError::BadRequest(
                "cuda NVFP4 matvec: stored scale bytes differ from the matrix shape".to_owned(),
            ));
        }
        if x.len() != shape[1] || !x.iter().all(|v| v.is_finite()) {
            return Err(EngError::BadRequest(
                "cuda NVFP4 matvec requires one finite activation row of the weight width"
                    .to_owned(),
            ));
        }
        if !weight_scale_2.is_finite() || weight_scale_2 <= 0.0 {
            return Err(EngError::BadRequest(
                "cuda NVFP4 matvec: weight_scale_2 must be finite and positive".to_owned(),
            ));
        }
        let mut y = host_floats(shape[0])?;
        let _dispatch = self.enter()?;
        let mut workspace = self
            .workspace
            .try_borrow_mut()
            .map_err(|_| workspace_in_use())?;
        self.reserve_workspace(&mut workspace, need)?;
        // Growth frees the old allocation, which can latch a terminal failure.
        self.ensure_usable()?;
        let device = self.owned(&workspace.buffer)?;
        native::nvfp4_gemv(
            packed_raw,
            scale_raw,
            weight_scale_2,
            shape,
            workspace.work(device),
            x,
            &mut y,
        )
        .map_err(|failure| {
            if failure.status == native::STATUS_INVALID_ARGUMENT {
                EngError::BadRequest(
                    "cuda NVFP4 matvec: a group scale or decoded weight was not finite".to_owned(),
                )
            } else {
                device_error(failure)
            }
        })?;
        Ok(y)
    }

    /// Synchronously computes one BF16 matvec from a resident code buffer.
    /// The device widens the stored 16-bit codes, so the host never
    /// materializes the F32 matrix (M2 rule).
    pub(crate) fn bf16_matvec(
        &self,
        codes: &DeviceBuffer,
        shape: [usize; 2],
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        self.ensure_usable()?;
        let codes_raw = self.owned(codes)?;
        let (codes_bytes, need) = native::bf16_matvec_bytes(shape).ok_or_else(|| {
            EngError::BadRequest("cuda BF16 matvec: matrix dimensions must be positive".to_owned())
        })?;
        if codes.bytes() != codes_bytes {
            return Err(EngError::BadRequest(
                "cuda BF16 matvec: stored code bytes differ from the matrix shape".to_owned(),
            ));
        }
        if x.len() != shape[1] || !x.iter().all(|v| v.is_finite()) {
            return Err(EngError::BadRequest(
                "cuda BF16 matvec requires one finite activation row of the weight width"
                    .to_owned(),
            ));
        }
        let mut y = host_floats(shape[0])?;
        let _dispatch = self.enter()?;
        let mut workspace = self
            .workspace
            .try_borrow_mut()
            .map_err(|_| workspace_in_use())?;
        self.reserve_workspace(&mut workspace, need)?;
        // Growth frees the old allocation, which can latch a terminal failure.
        self.ensure_usable()?;
        let device = self.owned(&workspace.buffer)?;
        native::bf16_gemv(codes_raw, shape, workspace.work(device), x, &mut y).map_err(
            |failure| {
                if failure.status == native::STATUS_INVALID_ARGUMENT {
                    EngError::BadRequest(
                        "cuda BF16 matvec: a widened weight was not finite".to_owned(),
                    )
                } else {
                    device_error(failure)
                }
            },
        )?;
        Ok(y)
    }

    /// Little-endian host bytes of one descriptor table, for small uploads
    /// that die with the call that consumes them.
    fn table_bytes(values: &[u64]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    /// Little-endian host bytes of one f32 row (per-expert scale factors).
    fn scale_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    /// Reference batch using the qualified single-row GPU GEMV for each pair.
    /// The old native batch returns zero, so it is not used for flag-off runs.
    #[allow(clippy::too_many_arguments, reason = "mirrors the native batch ABI")]
    pub(crate) fn nvfp4_matvec_batch(
        &self,
        packed: &[&DeviceBuffer],
        scale: &[&DeviceBuffer],
        weight_scales: &[f32],
        shape: [usize; 2],
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        self.nvfp4_batch_impl(packed, scale, weight_scales, shape, x, false)
    }

    /// NVFP4 grouped GEMM: FP16 operands, FP32 tensor-core accumulation.
    pub(crate) fn nvfp4_grouped_gemm(
        &self,
        packed: &[&DeviceBuffer],
        scale: &[&DeviceBuffer],
        weight_scales: &[f32],
        shape: [usize; 2],
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        self.nvfp4_batch_impl(packed, scale, weight_scales, shape, x, true)
    }

    fn nvfp4_batch_impl(
        &self,
        packed: &[&DeviceBuffer],
        scale: &[&DeviceBuffer],
        weight_scales: &[f32],
        shape: [usize; 2],
        x: &[f32],
        grouped: bool,
    ) -> Result<Vec<f32>, EngError> {
        self.ensure_usable()?;
        let n = packed.len();
        if n == 0 || scale.len() != n || weight_scales.len() != n {
            return Err(EngError::BadRequest(
                "cuda NVFP4 batch: experts, scales and tables must align".to_owned(),
            ));
        }
        if weight_scales
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(EngError::BadRequest(
                "cuda NVFP4 batch: weight_scale_2 values must be finite and positive".to_owned(),
            ));
        }
        let (packed_bytes, scale_bytes, _) =
            native::nvfp4_matvec_bytes(shape).ok_or_else(|| {
                EngError::BadRequest(
                    "cuda NVFP4 batch: dimensions must be positive and group aligned".to_owned(),
                )
            })?;
        for expert in 0..n {
            if packed[expert].bytes() != packed_bytes || scale[expert].bytes() != scale_bytes {
                return Err(EngError::BadRequest(
                    "cuda NVFP4 batch: a payload differs from the batch shape".to_owned(),
                ));
            }
        }
        let (_, need) = native::nvfp4_batch_bytes(n, shape)
            .ok_or_else(|| EngError::BadRequest("cuda NVFP4 batch: extents overflow".to_owned()))?;
        if x.len() != need.x / size_of::<f32>() || !x.iter().all(|v| v.is_finite()) {
            return Err(EngError::BadRequest(
                "cuda NVFP4 batch needs one finite input row per expert of the width".to_owned(),
            ));
        }
        let packed_raws: Vec<_> = packed
            .iter()
            .map(|buffer| self.owned(buffer))
            .collect::<Result<Vec<_>, _>>()?;
        let scale_raws: Vec<_> = scale
            .iter()
            .map(|buffer| self.owned(buffer))
            .collect::<Result<Vec<_>, _>>()?;
        if !grouped {
            // The legacy native batch returns zeros; keep flag-off on the qualified GPU reference.
            let mut output = host_floats(need.y / size_of::<f32>())?;
            for pair in 0..n {
                let values = self.nvfp4_matvec(
                    packed[pair],
                    scale[pair],
                    weight_scales[pair],
                    shape,
                    &x[pair * shape[1]..(pair + 1) * shape[1]],
                )?;
                output[pair * shape[0]..(pair + 1) * shape[0]].copy_from_slice(&values);
            }
            return Ok(output);
        }
        let packed_table: Vec<u64> = packed_raws.iter().map(|raw| raw.address()).collect();
        let scale_table: Vec<u64> = scale_raws.iter().map(|raw| raw.address()).collect();
        let mut y = host_floats(n * shape[0])?;
        let _dispatch = self.enter()?;
        let packed_table_device = self.upload(&Self::table_bytes(&packed_table))?;
        let scale_table_device = self.upload(&Self::table_bytes(&scale_table))?;
        let scales_device = self.upload(&Self::scale_bytes(weight_scales))?;
        // shortcut: the shared workspace cannot grow to batch size without
        // freeing the old allocation (which latches terminal); a dedicated
        // buffer serves the batch. Layout: x, then xf at a 16-byte boundary,
        // then y at a 4-byte boundary; x and y are float arrays so already 4.
        let xf_offset = need.x.checked_next_multiple_of(16).ok_or_else(|| {
            EngError::BadRequest("cuda NVFP4 batch: workspace extents overflow".to_owned())
        })?;
        let y_offset = xf_offset
            .checked_add(need.xf)
            .and_then(|end| end.checked_next_multiple_of(4))
            .ok_or_else(|| {
                EngError::BadRequest("cuda NVFP4 batch: workspace extents overflow".to_owned())
            })?;
        let total = y_offset.checked_add(need.y).ok_or_else(|| {
            EngError::BadRequest("cuda NVFP4 batch: workspace extents overflow".to_owned())
        })?;
        let batch_scratch = self.upload(&vec![0u8; total])?;
        let scratch_raw = self.owned(&batch_scratch)?;
        let batch_work = native::Work {
            device: scratch_raw,
            x: native::Region {
                offset: 0,
                bytes: need.x,
            },
            xf: native::Region {
                offset: xf_offset,
                bytes: need.xf,
            },
            y: native::Region {
                offset: y_offset,
                bytes: need.y,
            },
        };
        native::nvfp4_gemv_batch(
            self.owned(&packed_table_device)?,
            self.owned(&scale_table_device)?,
            self.owned(&scales_device)?,
            n,
            shape,
            batch_work,
            x,
            &mut y,
            grouped,
        )
        .map_err(|failure| {
            if failure.status == native::STATUS_INVALID_ARGUMENT {
                EngError::BadRequest(
                    "cuda NVFP4 batch: invalid weight or FP16 operand overflow".to_owned(),
                )
            } else {
                device_error(failure)
            }
        })?;
        Ok(y)
    }

    /// Synchronously computes one F8_E4M3 matvec from a resident code buffer.
    /// The device decodes the stored codes, so the host never materializes the
    /// F32 matrix (M2 rule: no F32 weight materialization on host or device).
    pub(crate) fn f8_matvec(
        &self,
        codes: &DeviceBuffer,
        weight_scale: f32,
        shape: [usize; 2],
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        self.ensure_usable()?;
        let codes_raw = self.owned(codes)?;
        let (codes_bytes, need) = native::f8_matvec_bytes(shape).ok_or_else(|| {
            EngError::BadRequest("cuda F8 matvec: matrix dimensions must be positive".to_owned())
        })?;
        if codes.bytes() != codes_bytes {
            return Err(EngError::BadRequest(
                "cuda F8 matvec: stored code bytes differ from the matrix shape".to_owned(),
            ));
        }
        if x.len() != shape[1] || !x.iter().all(|v| v.is_finite()) {
            return Err(EngError::BadRequest(
                "cuda F8 matvec requires one finite activation row of the weight width".to_owned(),
            ));
        }
        if !weight_scale.is_finite() || weight_scale <= 0.0 {
            return Err(EngError::BadRequest(
                "cuda F8 matvec: weight_scale must be finite and positive".to_owned(),
            ));
        }
        let mut y = host_floats(shape[0])?;
        let _dispatch = self.enter()?;
        let mut workspace = self
            .workspace
            .try_borrow_mut()
            .map_err(|_| workspace_in_use())?;
        self.reserve_workspace(&mut workspace, need)?;
        // Growth frees the old allocation, which can latch a terminal failure.
        self.ensure_usable()?;
        let device = self.owned(&workspace.buffer)?;
        native::f8_gemv(
            codes_raw,
            weight_scale,
            shape,
            workspace.work(device),
            x,
            &mut y,
        )
        .map_err(|failure| {
            if failure.status == native::STATUS_INVALID_ARGUMENT {
                EngError::BadRequest("cuda F8 matvec: a decoded weight was not finite".to_owned())
            } else {
                device_error(failure)
            }
        })?;
        Ok(y)
    }

    /// Projects one finite host row and returns the lowest index holding the
    /// largest logit, with its value. The device reduces the row, so the host
    /// receives 8 bytes instead of the whole vocabulary.
    pub(crate) fn linear_argmax_f32(
        &self,
        weight: &DeviceBuffer,
        shape: [usize; 2],
        x: &[f32],
    ) -> Result<(u32, f32), EngError> {
        self.ensure_usable()?;
        let raw = self.owned(weight)?;
        let (weight_bytes, need) = native::f32_projection_bytes(shape).ok_or_else(|| {
            EngError::BadRequest(
                "cuda argmax head: projection needs positive dimensions".to_owned(),
            )
        })?;
        if raw.bytes() < weight_bytes || x.len() != shape[1] || !x.iter().all(|v| v.is_finite()) {
            return Err(EngError::BadRequest(
                "cuda argmax head: projection needs sufficient weight bytes and one finite input row"
                    .to_owned(),
            ));
        }
        let mut scratch_slot = self
            .argmax_scratch
            .try_borrow_mut()
            .map_err(|_| workspace_in_use())?;
        if scratch_slot
            .as_ref()
            .is_none_or(|buffer| buffer.bytes() < native::ARGMAX_SCRATCH_BYTES)
        {
            *scratch_slot = Some(DeviceBuffer::allocate(
                &self.lease,
                native::ARGMAX_SCRATCH_BYTES,
            )?);
        }
        let scratch = self.owned(scratch_slot.as_ref().expect("scratch was just ensured"))?;
        let _dispatch = self.enter()?;
        let mut workspace = self
            .workspace
            .try_borrow_mut()
            .map_err(|_| workspace_in_use())?;
        self.reserve_workspace(&mut workspace, need)?;
        self.ensure_usable()?;
        let device = self.owned(&workspace.buffer)?;
        native::linear_argmax_f32(raw, shape, workspace.work(device), scratch, x).map_err(
            |failure| {
                if failure.status == native::STATUS_INVALID_ARGUMENT {
                    EngError::BadRequest("cuda argmax head: a logit was not finite".to_owned())
                } else {
                    device_error(failure)
                }
            },
        )
    }

    /// Allocates `bytes` of uninitialized device memory.
    fn alloc(&self, bytes: usize) -> Result<DeviceBuffer, EngError> {
        let buffer = match &self.alloc_cache {
            Some(cache) if bytes <= alloc_cache::MAX_BLOCK_BYTES => cache.allocate(bytes)?,
            _ => DeviceBuffer::allocate(&self.lease, bytes)?,
        };
        #[cfg(all(test, feature = "native-cuda"))]
        self.successful_allocations
            .set(self.successful_allocations.get().saturating_add(1));
        Ok(buffer)
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
    pub(crate) fn owned<'b>(&self, buffer: &'b DeviceBuffer) -> Result<&'b RawDevice, EngError> {
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

    fn free_after_capture(mut self) {
        let raw = self.raw.take().expect("deferred buffer owns backing");
        let cache = self
            .alloc_cache
            .as_ref()
            .expect("deferred buffer owns cache");
        let _ = cache.free(raw);
    }

    /// Takes ownership of a device allocation and counts its bytes as live.
    fn from_raw(lease: &Rc<Lease>, raw: RawDevice) -> Self {
        LIVE_BYTES.fetch_add(raw.bytes(), Ordering::AcqRel);
        Self {
            raw: Some(raw),
            lease: Rc::clone(lease),
            _thread_confined: PhantomData,
            alloc_cache: None,
        }
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            if TERMINAL.load(Ordering::Acquire) {
                if let Some(cache) = &self.alloc_cache {
                    cache.quarantine(raw);
                }
                return;
            }
            if let Some(cache) = self.alloc_cache.take() {
                if graph::window_open() {
                    // Abandoned closures must drop an owner that records quarantine.
                    let deferred = Self {
                        raw: Some(raw),
                        lease: Rc::clone(&self.lease),
                        _thread_confined: PhantomData,
                        alloc_cache: Some(cache),
                    };
                    graph::run_or_defer(move || deferred.free_after_capture());
                } else {
                    cache.return_block(raw);
                }
                return;
            }
            // Default-OFF backing also owns its lease until the deferred free completes.
            let lease = Rc::clone(&self.lease);
            graph::run_or_defer(move || {
                let _lease = lease;
                release_buffer(raw);
            });
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
mod mailbox;
mod native_weight;
mod nvfp4_launch;
mod pinned;
mod pinned_buf;
mod pinned_stage;
mod range;
mod stream;
mod tile_q5_0;
mod workspace;

pub(crate) use device_arena::{ArenaHold, DeviceArena};
pub(crate) use expert_bank::{ExpertBank, ExpertBankGeometry};
pub(crate) use graph::{CapturePlan, GraphExec, is_capture_refusal};
pub(in crate::kernels::cuda) use launch::{Claim, require_apart, require_fit};
pub(crate) use launch::{MatmulIo, PackCall, Projection};
pub(crate) use mailbox::Mailbox;
pub(crate) use native_weight::NativeWeight;
pub(crate) use nvfp4_launch::Nvfp4WeightRange;
#[cfg(test)]
pub(crate) use nvfp4_launch::{Nvfp4GroupedPlan, Nvfp4LaunchStatus};
pub(crate) use pinned::{PinnedWord, WordValue};
pub(crate) use pinned_buf::PinnedBuf;
pub(crate) use pinned_stage::PINNED_STAGE_LIMIT_BYTES;
pub(crate) use range::DeviceRange;
pub(crate) use stream::{DeviceEvent, DeviceStream, TimingEvent};
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

#[cfg(test)]
mod f32_projection_tests {
    use super::*;

    // Use the existing projection test's device gate; no device is needed to type-check.
    fn device_present() -> bool {
        if super::super::device_count().expect("CUDA device count") > 0 {
            return true;
        }
        let required = std::env::var("INFERENCE_CUDA_REQUIRE_DEVICE")
            .is_ok_and(|value| !value.is_empty() && value != "0");
        assert!(
            !required,
            "INFERENCE_CUDA_REQUIRE_DEVICE is set but no CUDA device is visible"
        );
        println!("SKIPPED F32 projection: no CUDA device");
        false
    }

    fn upload_floats(ctx: &CudaContext, values: &[f32]) -> DeviceBuffer {
        let bytes: Vec<_> = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        ctx.upload(&bytes).expect("upload F32 weight")
    }

    #[test]
    fn linear_f32_rejects_caller_errors_without_poisoning_or_growing() {
        if !device_present() {
            return;
        }
        let caps = Activations { x: 4, xf: 4, y: 4 };
        let ctx = CudaContext::with_workspace(caps).expect("context");
        let other = CudaContext::with_workspace(caps).expect("another context");
        let weight = upload_floats(&ctx, &[1.0, -2.0, 3.0, -4.0, 5.0, 6.0]);
        let foreign = upload_floats(&other, &[1.0, -2.0, 3.0, -4.0, 5.0, 6.0]);
        let short = upload_floats(&ctx, &[1.0]);
        for (buffer, shape, x) in [
            (&weight, [0, 3], vec![2.0, -1.0, 0.5]),
            (&weight, [2, 0], vec![]),
            (&weight, [usize::MAX, 3], vec![2.0, -1.0, 0.5]),
            (&weight, [2, usize::MAX], vec![]),
            (&weight, [usize::MAX / 4, 3], vec![2.0, -1.0, 0.5]),
            (&weight, [2, 3], vec![]),
            (&weight, [2, 3], vec![2.0, -1.0]),
            (&weight, [2, 3], vec![2.0, -1.0, 0.5, 4.0]),
            (&weight, [2, 3], vec![2.0, f32::NAN, 0.5]),
            (&weight, [2, 3], vec![f32::INFINITY, -1.0, 0.5]),
            (&weight, [2, 3], vec![2.0, -1.0, f32::NEG_INFINITY]),
            (&short, [2, 3], vec![2.0, -1.0, 0.5]),
            (&foreign, [2, 3], vec![2.0, -1.0, 0.5]),
        ] {
            assert!(matches!(
                ctx.linear_f32(buffer, shape, &x),
                Err(EngError::BadRequest(_))
            ));
            ctx.ensure_usable()
                .expect("caller errors leave the backend usable");
            assert_eq!(ctx.workspace_allocations(), 1);
        }
        assert_eq!(
            ctx.linear_f32(&weight, [2, 3], &[2.0, -1.0, 0.5]).unwrap(),
            [5.5, -10.0]
        );
    }

    #[test]
    fn linear_f32_returns_nonzero_rows_and_reuses_a_grown_workspace() {
        if !device_present() {
            return;
        }
        let ctx = CudaContext::with_workspace(Activations { x: 4, xf: 4, y: 4 })
            .expect("small workspace");
        let weight = upload_floats(&ctx, &[1.0, -2.0, 3.0, -4.0, 5.0, 6.0]);
        // [1, -2, 3] dot [2, -1, 0.5] = 5.5; [-4, 5, 6] dot it = -10.
        assert_eq!(
            ctx.linear_f32(&weight, [2, 3], &[2.0, -1.0, 0.5]).unwrap(),
            [5.5, -10.0]
        );
        assert_eq!(ctx.workspace_allocations(), 2);
        let bytes = ctx.workspace_bytes();
        for _ in 0..3 {
            assert_eq!(
                ctx.linear_f32(&weight, [2, 3], &[-1.0, 2.0, -0.5]).unwrap(),
                [-6.5, 11.0]
            );
            assert_eq!(ctx.workspace_allocations(), 2);
            assert_eq!(ctx.workspace_bytes(), bytes);
        }
        assert_eq!(ctx.linear_f32(&weight, [1, 1], &[3.0]).unwrap(), [3.0]);
        assert_eq!(ctx.workspace_allocations(), 2);
        assert_eq!(ctx.workspace_bytes(), bytes);
    }
}

#[cfg(test)]
mod fixed_projection_contract_tests {
    use super::*;

    #[test]
    fn fixed_projection_plan_contract_accepts_vocab_above_input_width_limit() {
        let (plan, charged) = CudaContext::fixed_projection_plan([131_071, 8191]).unwrap();
        assert_eq!(plan.growth, Growth::Fixed);
        assert_eq!(plan.shapes.len(), 1);
        assert_eq!(plan.shapes[0].n_rows, 131_072);
        assert_eq!(plan.shapes[0].k, 8192);
        assert_eq!(plan.shapes[0].tokens, 1);
        assert!(plan.shapes[0].n_rows > native::MAX_DIM);
        let workspace = workspace::place(plan.caps().unwrap()).unwrap().total;
        assert_eq!(
            charged,
            u64::try_from(workspace.checked_add(native::ARGMAX_SCRATCH_BYTES).unwrap()).unwrap()
        );
    }

    #[test]
    fn fixed_projection_plan_contract_rejects_invalid_axes_before_native_sizing() {
        for shape in [
            [0, 16],
            [128, 0],
            [128, native::MAX_DIM + 1],
            [usize::MAX, 16],
            [c_int::MAX as usize, 16],
        ] {
            assert!(matches!(
                CudaContext::fixed_projection_plan(shape),
                Err(EngError::BadRequest(_))
            ));
        }
    }
}
