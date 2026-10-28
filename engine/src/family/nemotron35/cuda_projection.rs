//! Staged CUDA projections for host-orchestrated NemotronH forwards.

#![allow(
    dead_code,
    reason = "staged adapter; stack wiring follows qualification"
)]

use super::{
    admission::cuda::{CudaAdmission, CudaTransferLimits},
    diagnostics::HostSpan,
    transport_accounting::{self as accounting, Category},
};

use std::{
    cell::Cell,
    collections::{HashMap, VecDeque},
    rc::Rc,
    time::{Duration, Instant},
};

use crate::{
    EngError,
    family::project::Projector,
    kernels::cuda::CudaContext,
    kernels::cuda::DeviceBuffer,
    kernels::cuda::PINNED_STAGE_LIMIT_BYTES,
    tensor::{Dtype, Tensor},
};

/// One admitted device-resident weight. Retaining the host tensor pins the
/// keyed allocation, so a cache key can never be reused by a different live
/// tensor while the entry exists.
struct CachedWeight {
    tensor: Tensor,
    buffer: DeviceBuffer,
    bytes: u64,
}

/// One admitted device-resident packed pair. The key is a caller-assigned
/// checkpoint identity (layer, expert, projection), not a pointer: the loader
/// re-reads packed bytes on every use, so the retained vectors both pin the
/// host bytes and supply the upload payload.
struct CachedPacked {
    packed: Vec<u8>,
    scale: Vec<u8>,
    weight_scale_2: f32,
    device_packed: DeviceBuffer,
    device_scale: DeviceBuffer,
    bytes: u64,
}

/// One packed expert pair that stays on the device for the whole run (a GPU-resident MoE layer
/// of the hybrid). It keeps no host copy and is never evicted.
struct ResidentPacked {
    device_packed: DeviceBuffer,
    device_scale: DeviceBuffer,
    weight_scale_2: f32,
    bytes: u64,
}

/// Cumulative wall time of the device entry points, for the per-token split of the hybrid.
/// The cells are shared with the timers so a timer needs no borrow of the projector.
#[derive(Default)]
struct CallClock {
    calls: Cell<u64>,
    call_nanos: Cell<u64>,
    upload_nanos: Cell<u64>,
    upload_bytes: Cell<u64>,
}

/// A snapshot of the clock. `call` includes the uploads made inside the calls.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DeviceStats {
    pub(crate) calls: u64,
    pub(crate) call: Duration,
    pub(crate) upload: Duration,
    pub(crate) upload_bytes: u64,
}

/// Adds the lifetime of one device entry point to the clock when it drops.
struct CallTimer {
    clock: Rc<CallClock>,
    started: Instant,
}

impl CallTimer {
    fn start(clock: &Rc<CallClock>) -> Self {
        Self {
            clock: Rc::clone(clock),
            started: Instant::now(),
        }
    }
}

impl Drop for CallTimer {
    fn drop(&mut self) {
        let nanos = u64::try_from(self.started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.clock.calls.set(self.clock.calls.get() + 1);
        self.clock
            .call_nanos
            .set(self.clock.call_nanos.get().saturating_add(nanos));
    }
}

/// One MoE layer's shared expert, device-resident in stored NVFP4 form.
struct ResidentShared {
    packed_up: DeviceBuffer,
    scale_up: DeviceBuffer,
    ws_up: f32,
    shape_up: [u32; 2],
    packed_down: DeviceBuffer,
    scale_down: DeviceBuffer,
    ws_down: f32,
    shape_down: [u32; 2],
}

/// One dense weight, device-resident in stored code form (FP8 or BF16).
struct ResidentF8 {
    codes: DeviceBuffer,
    scale: f32,
    shape: [usize; 2],
    /// True when the codes are BF16 (bf16_matvec computes; scale unused).
    is_bf16: bool,
}

/// One LRU identity across both cache maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheKey {
    F32(usize),
    Packed(u64),
}

/// Owns one CUDA context and reuses its dense-weight staging allocation.
/// `weight_cache` keeps admitted weights resident on the device (LRU by byte
/// budget) so repeated projections skip the per-call host transfer.
pub(crate) struct CudaProjector {
    context: CudaContext,
    limits: CudaTransferLimits,
    weight_cache: HashMap<usize, CachedWeight>,
    packed_cache: HashMap<u64, CachedPacked>,
    resident_shared: HashMap<u32, ResidentShared>,
    resident_f8: HashMap<String, ResidentF8>,
    /// Maps a decoded F32 tensor's data pointer to its FP8-resident name,
    /// so `linear` can route to `f8_matvec` without a per-call upload.
    f8_pointer_map: HashMap<usize, String>,
    /// Device-resident mamba conv/ssm state (NEM35_DEVICE_CHAIN).
    mamba_conv_state: HashMap<u32, DeviceBuffer>,
    mamba_ssm_state: HashMap<u32, DeviceBuffer>,
    /// Small mamba weights (conv/bias/dt/A/D/norm) uploaded once.
    mamba_small_weights: HashMap<String, DeviceBuffer>,
    /// Per-token device path counters (coordinator guard).
    pub(crate) device_mamba_steps: std::cell::Cell<u64>,
    pub(crate) grouped_launches: std::cell::Cell<u64>,
    lru: VecDeque<CacheKey>,
    cached_bytes: u64,
    cache_budget: u64,
    stream_plan: Option<super::admission::ChunkStreamPlan>,
    resident_packed: HashMap<u64, ResidentPacked>,
    resident_bytes: u64,
    /// Hybrid decode: routed experts are resident or CPU-served, so a miss is a bug, not an upload.
    resident_experts_only: bool,
    /// Device bytes the pinned staging slot of the admitted maximum payload can take.
    stage_bytes: u64,
    clock: Rc<CallClock>,
}

/// Geometry for one mamba layer's device-chain execution.
#[allow(clippy::too_many_arguments, reason = "mirrors MambaConfig")]
/// A borrowed view of one device-resident packed expert (the #5 streaming
/// tier seam: grouped callback reads these without owning them).
pub(crate) struct PackedExpertView<'a> {
    pub packed: &'a DeviceBuffer,
    pub scale: &'a DeviceBuffer,
    pub weight_scale_2: f32,
    pub bytes: u64,
}

pub(crate) struct MambaDeviceChain {
    pub num_heads: u32,
    pub head_dim: u32,
    pub state_size: u32,
    pub n_groups: u32,
    pub conv_kernel: u32,
    pub conv_dim: u32,
    pub intermediate: u32,
    pub time_step_min: f32,
    pub eps: f32,
}

/// Admission and timing guard for one projection launched through the
/// streaming tier's shared callback instead of this file's wrappers.
/// Dropping it records the call, matching the wrapper contract. The
/// caller records the per-key `transport_accounting` counters after its
/// real kernel call — this seam deliberately does not.
pub(crate) struct ProjectionGuard {
    _timer: CallTimer,
}

impl CudaProjector {
    /// Ensures one packed expert pair is device-resident (upload on first
    /// use), without computing. The key must be a checkpoint identity.
    pub(crate) fn ensure_packed_resident(
        &mut self,
        key: u64,
        packed: &[u8],
        scale: &[u8],
        weight_scale_2: f32,
        shape: [u32; 2],
    ) -> Result<(), EngError> {
        let _timer = CallTimer::start(&self.clock);
        if self.resident_packed.contains_key(&key) || self.packed_cache.contains_key(&key) {
            return Ok(());
        }
        self.refuse_unresident_expert()?;
        self.context.ensure_usable()?;
        let rows =
            usize::try_from(shape[0]).map_err(|_| invalid("row count exceeds host limits"))?;
        let columns =
            usize::try_from(shape[1]).map_err(|_| invalid("column count exceeds host limits"))?;
        self.limits.check_shape([rows, columns])?;
        self.limits.check_packed(packed.len(), scale.len())?;
        if self.cache_budget == 0 {
            return Err(invalid(
                "stored packed residency requires an admitted device weight cache",
            ));
        }
        if !weight_scale_2.is_finite() || weight_scale_2 <= 0.0 {
            return Err(invalid("weight_scale_2 must be finite and positive"));
        }
        let pair_bytes = u64::try_from(packed.len().saturating_add(scale.len()))
            .map_err(|_| invalid("packed pair exceeds u64"))?;
        if pair_bytes > self.cache_budget {
            return Err(invalid("packed pair exceeds the admitted cache budget"));
        }
        accounting::record_upload(Category::Routed, packed.len() + scale.len());
        let device_packed = self.timed_upload(packed)?;
        let device_scale = self.timed_upload(scale)?;
        self.evict_for(pair_bytes);
        self.cached_bytes += pair_bytes;
        self.lru.push_back(CacheKey::Packed(key));
        self.packed_cache.insert(
            key,
            CachedPacked {
                packed: packed.to_vec(),
                scale: scale.to_vec(),
                weight_scale_2,
                device_packed,
                device_scale,
                bytes: pair_bytes,
            },
        );
        Ok(())
    }

    /// Loads one MoE layer's shared expert as device-resident NVFP4.
    /// Uploads once; subsequent calls skip all checkpoint reads and uploads.
    #[allow(clippy::too_many_arguments, reason = "up/down pair per layer")]
    pub(crate) fn ensure_shared_resident(
        &mut self,
        layer: u32,
        packed_up: &[u8],
        scale_up: &[u8],
        ws_up: f32,
        shape_up: [u32; 2],
        packed_down: &[u8],
        scale_down: &[u8],
        ws_down: f32,
        shape_down: [u32; 2],
    ) -> Result<(), EngError> {
        if self.resident_shared.contains_key(&layer) {
            return Ok(());
        }
        self.context.ensure_usable()?;
        let up_rows = usize::try_from(shape_up[0]).map_err(|_| invalid("row count"))?;
        let up_cols = usize::try_from(shape_up[1]).map_err(|_| invalid("col count"))?;
        self.limits.check_shape([up_rows, up_cols])?;
        self.limits.check_packed(packed_up.len(), scale_up.len())?;
        let d_pu = self.context.upload(packed_up)?;
        let d_su = self.context.upload(scale_up)?;
        let d_pd = self.context.upload(packed_down)?;
        let d_sd = self.context.upload(scale_down)?;
        self.resident_shared.insert(
            layer,
            ResidentShared {
                packed_up: d_pu,
                scale_up: d_su,
                ws_up,
                shape_up,
                packed_down: d_pd,
                scale_down: d_sd,
                ws_down,
                shape_down,
            },
        );
        Ok(())
    }

    /// Computes shared expert up → relu2 → down on device-resident NVFP4.
    /// No checkpoint read, no per-call upload, no F32 materialization.
    pub(crate) fn shared_mlp(&self, layer: u32, input: &[f32]) -> Result<Vec<f32>, EngError> {
        let entry = self
            .resident_shared
            .get(&layer)
            .ok_or_else(|| invalid("shared expert not resident"))?;
        let up_rows = usize::try_from(entry.shape_up[0]).map_err(|_| invalid("rows"))?;
        let up_cols = usize::try_from(entry.shape_up[1]).map_err(|_| invalid("cols"))?;
        let down_rows = usize::try_from(entry.shape_down[0]).map_err(|_| invalid("rows"))?;
        let down_cols = usize::try_from(entry.shape_down[1]).map_err(|_| invalid("cols"))?;
        if input.len() != up_cols {
            return Err(invalid("shared expert input width mismatch"));
        }
        let up = self.context.nvfp4_matvec(
            &entry.packed_up,
            &entry.scale_up,
            entry.ws_up,
            [up_rows, up_cols],
            input,
        )?;
        // relu2: clamp to zero then square, matching the reference mlp.
        let activated: Vec<f32> = up.iter().map(|v| v.max(0.0).powi(2)).collect();
        self.context.nvfp4_matvec(
            &entry.packed_down,
            &entry.scale_down,
            entry.ws_down,
            [down_rows, down_cols],
            &activated,
        )
    }

    /// Links a decoded F32 tensor pointer to its FP8-resident entry so
    /// `linear` can route the compute through `f8_matvec` (no upload).
    pub(crate) fn link_f8_pointer(&mut self, pointer: usize, name: &str) {
        self.f8_pointer_map.insert(pointer, name.to_owned());
    }

    /// Returns the FP8-resident name linked to this tensor, if any.
    pub(crate) fn f8_name_for(&self, weight: &Tensor) -> Option<&str> {
        let pointer = weight
            .host_bytes()
            .map(|b| b.as_ptr() as usize)
            .unwrap_or(0);
        self.f8_pointer_map.get(&pointer).map(String::as_str)
    }

    /// Startup-guard accessors: expose resident/cache sizes without
    /// leaking the private fields.
    pub(crate) fn resident_f8_count(&self) -> usize {
        self.resident_f8.len()
    }
    pub(crate) fn resident_f8_bytes(&self) -> u64 {
        self.resident_f8
            .values()
            .map(|e| e.codes.bytes() as u64)
            .sum()
    }
    pub(crate) fn resident_shared_count(&self) -> usize {
        self.resident_shared.len()
    }
    pub(crate) fn f32_cache_count(&self) -> usize {
        self.weight_cache.len()
    }
    pub(crate) fn f32_cache_bytes(&self) -> u64 {
        self.cached_bytes
    }

    /// Name-based linear compute on a device-resident weight (the qwen36
    /// CudaWeights pattern: lookup by name, no host tensor needed).
    /// Returns None when the name is not resident (the caller falls back
    /// to the old pointer-based path).
    pub(crate) fn linear_by_name(
        &self,
        name: &str,
        input: &[f32],
    ) -> Option<Result<Vec<f32>, EngError>> {
        let entry = self.resident_f8.get(name)?;
        Some(if entry.is_bf16 {
            self.context.bf16_matvec(&entry.codes, entry.shape, input)
        } else {
            self.context
                .f8_matvec(&entry.codes, entry.scale, entry.shape, input)
        })
    }

    /// Whether a dense weight name is device-resident.
    pub(crate) fn is_dense_resident(&self, name: &str) -> bool {
        self.resident_f8.contains_key(name)
    }

    /// The stored shape of a device-resident weight by name.
    pub(crate) fn resident_shape_of(&self, name: &str) -> Option<[usize; 2]> {
        self.resident_f8.get(name).map(|e| e.shape)
    }

    /// Counter accessors for the coordinator's gate guard.
    pub(crate) fn device_mamba_count(&self) -> u64 {
        self.device_mamba_steps.get()
    }
    pub(crate) fn grouped_launch_count(&self) -> u64 {
        self.grouped_launches.get()
    }

    /// Increments the device mamba step counter (coordinator guard:
    /// device_mamba_steps must be > 0 when NEM35_DEVICE_CHAIN is on).
    pub(crate) fn count_mamba_step(&self) {
        self.device_mamba_steps
            .set(self.device_mamba_steps.get() + 1);
    }

    /// A borrowed view of one resident packed expert (the seam #5 uses
    /// for the streaming tier's grouped callback).
    pub(crate) fn packed_expert_view(&self, key: u64) -> Option<PackedExpertView<'_>> {
        // Permanent residents take precedence, matching resident_entry.
        if let Some(entry) = self.resident_packed.get(&key) {
            return Some(PackedExpertView {
                packed: &entry.device_packed,
                scale: &entry.device_scale,
                weight_scale_2: entry.weight_scale_2,
                bytes: entry.bytes,
            });
        }
        let entry = self.packed_cache.get(&key)?;
        Some(PackedExpertView {
            packed: &entry.device_packed,
            scale: &entry.device_scale,
            weight_scale_2: entry.weight_scale_2,
            bytes: entry.bytes,
        })
    }

    pub(crate) fn begin_projection(&self, shape: [u32; 2]) -> Result<ProjectionGuard, EngError> {
        let _timer = CallTimer::start(&self.clock);
        self.context.ensure_usable()?;
        let rows =
            usize::try_from(shape[0]).map_err(|_| invalid("row count exceeds host limits"))?;
        let columns =
            usize::try_from(shape[1]).map_err(|_| invalid("column count exceeds host limits"))?;
        self.limits.check_shape([rows, columns])?;
        Ok(ProjectionGuard { _timer })
    }

    /// Loads one dense weight as device-resident codes (FP8 or BF16).
    /// Uploads once; `is_bf16` dispatches the compute kernel.
    #[allow(clippy::too_many_arguments, reason = "mirrors the weight shape")]
    pub(crate) fn ensure_f8_resident(
        &mut self,
        name: &str,
        codes: &[u8],
        scale: f32,
        shape: [usize; 2],
        is_bf16: bool,
    ) -> Result<(), EngError> {
        if self.resident_f8.contains_key(name) {
            return Ok(());
        }
        self.context.ensure_usable()?;
        let buffer = self.context.upload(codes)?;
        self.resident_f8.insert(
            name.to_owned(),
            ResidentF8 {
                codes: buffer,
                scale,
                shape,
                is_bf16,
            },
        );
        Ok(())
    }

    /// Computes one dense matvec on device-resident codes (FP8 or BF16).
    pub(crate) fn f8_linear(&self, name: &str, input: &[f32]) -> Result<Vec<f32>, EngError> {
        let entry = self
            .resident_f8
            .get(name)
            .ok_or_else(|| invalid("dense weight not resident"))?;
        if entry.is_bf16 {
            self.context.bf16_matvec(&entry.codes, entry.shape, input)
        } else {
            self.context
                .f8_matvec(&entry.codes, entry.scale, entry.shape, input)
        }
    }

    /// Every key must already be resident (ensure_packed_resident); repeated
    /// keys are legal (the same expert serving several tokens).
    pub(crate) fn nvfp4_matvec_batch_keyed(
        &mut self,
        keys: &[u64],
        shape: [u32; 2],
        inputs: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let _timer = CallTimer::start(&self.clock);
        self.context.ensure_usable()?;
        let rows =
            usize::try_from(shape[0]).map_err(|_| invalid("row count exceeds host limits"))?;
        let columns =
            usize::try_from(shape[1]).map_err(|_| invalid("column count exceeds host limits"))?;
        self.limits.check_shape([rows, columns])?;
        let n = keys.len();
        // Borrow the resident buffers immutably; the context call is also an
        // immutable borrow, so the cache stays alive through the call.
        let mut packed_refs: Vec<&DeviceBuffer> = Vec::with_capacity(n);
        let mut scale_refs: Vec<&DeviceBuffer> = Vec::with_capacity(n);
        let mut weight_scales: Vec<f32> = Vec::with_capacity(n);
        for &key in keys {
            let (packed, scale, weight_scale_2) = self
                .resident_entry(key)
                .ok_or_else(|| invalid("batch key is not resident"))?;
            packed_refs.push(packed);
            scale_refs.push(scale);
            weight_scales.push(weight_scale_2);
        }
        let result = if std::env::var_os("NEM35_CUDA_GROUPED_EXPERTS").is_some_and(|v| v == "1") {
            self.grouped_launches.set(self.grouped_launches.get() + 1);
            self.context.nvfp4_grouped_gemm(
                &packed_refs,
                &scale_refs,
                &weight_scales,
                [rows, columns],
                inputs,
            )?
        } else {
            self.context.nvfp4_matvec_batch(
                &packed_refs,
                &scale_refs,
                &weight_scales,
                [rows, columns],
                inputs,
            )?
        };
        for _ in 0..n {
            accounting::record_resident_call(Category::Routed);
        }
        Ok(result)
    }

    /// The device buffers and per-tensor scale of one resident or cached packed projection.
    fn resident_entry(&self, key: u64) -> Option<(&DeviceBuffer, &DeviceBuffer, f32)> {
        if let Some(entry) = self.resident_packed.get(&key) {
            return Some((
                &entry.device_packed,
                &entry.device_scale,
                entry.weight_scale_2,
            ));
        }
        self.packed_cache.get(&key).map(|entry| {
            (
                &entry.device_packed,
                &entry.device_scale,
                entry.weight_scale_2,
            )
        })
    }

    fn refuse_unresident_expert(&self) -> Result<(), EngError> {
        if self.resident_experts_only {
            return Err(invalid(
                "a routed expert is neither device-resident nor CPU-served; hybrid decode uploads none per token",
            ));
        }
        Ok(())
    }

    /// Uploads through the context and charges the wall time and bytes to the clock.
    fn timed_upload(&self, bytes: &[u8]) -> Result<DeviceBuffer, EngError> {
        let started = Instant::now();
        let buffer = self.context.upload(bytes);
        let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.clock
            .upload_nanos
            .set(self.clock.upload_nanos.get().saturating_add(nanos));
        self.clock.upload_bytes.set(
            self.clock
                .upload_bytes
                .get()
                .saturating_add(bytes.len() as u64),
        );
        buffer
    }

    /// Hybrid decode: routed experts are device-resident or CPU-served from here on, so the
    /// per-token packed upload paths and the packed LRU are closed.
    pub(crate) fn close_packed_uploads(&mut self) {
        self.resident_experts_only = true;
        self.packed_cache.clear();
        self.lru.retain(|key| !matches!(key, CacheKey::Packed(_)));
        self.cached_bytes = self.weight_cache.values().map(|entry| entry.bytes).sum();
    }

    /// Uploads one routed expert projection for the whole run. Keeps no host copy.
    pub(crate) fn load_resident_expert(
        &mut self,
        key: u64,
        packed: &[u8],
        scale: &[u8],
        weight_scale_2: f32,
        shape: [u32; 2],
    ) -> Result<(), EngError> {
        let _timer = CallTimer::start(&self.clock);
        self.context.ensure_usable()?;
        let rows =
            usize::try_from(shape[0]).map_err(|_| invalid("row count exceeds host limits"))?;
        let columns =
            usize::try_from(shape[1]).map_err(|_| invalid("column count exceeds host limits"))?;
        self.limits.check_shape([rows, columns])?;
        self.limits.check_packed(packed.len(), scale.len())?;
        if !weight_scale_2.is_finite() || weight_scale_2 <= 0.0 {
            return Err(invalid("weight_scale_2 must be finite and positive"));
        }
        if self.resident_packed.contains_key(&key) {
            return Err(invalid("resident expert key loaded twice"));
        }
        let device_packed = self.timed_upload(packed)?;
        let device_scale = self.timed_upload(scale)?;
        self.resident_bytes += (packed.len() + scale.len()) as u64;
        self.resident_packed.insert(
            key,
            ResidentPacked {
                device_packed,
                device_scale,
                weight_scale_2,
                bytes: u64::try_from(packed.len().saturating_add(scale.len())).unwrap_or(u64::MAX),
            },
        );
        Ok(())
    }

    /// Whether the key is a device-resident expert projection (never evicted, no host copy).
    pub(crate) fn is_resident(&self, key: u64) -> bool {
        self.resident_packed.contains_key(&key)
    }

    /// One projection of a resident expert straight from its device buffers, or `None` when the
    /// key is not resident.
    pub(crate) fn resident_nvfp4_matvec(
        &mut self,
        key: u64,
        shape: [u32; 2],
        input: &[f32],
    ) -> Option<Result<Vec<f32>, EngError>> {
        let entry = self.resident_packed.get(&key)?;
        let _timer = CallTimer::start(&self.clock);
        let result = (|| {
            let rows =
                usize::try_from(shape[0]).map_err(|_| invalid("row count exceeds host limits"))?;
            let columns = usize::try_from(shape[1])
                .map_err(|_| invalid("column count exceeds host limits"))?;
            self.context.ensure_usable()?;
            self.limits.check_shape([rows, columns])?;
            accounting::record_resident_call(Category::Routed);
            self.context.nvfp4_matvec(
                &entry.device_packed,
                &entry.device_scale,
                entry.weight_scale_2,
                [rows, columns],
                input,
            )
        })();
        Some(result)
    }

    pub(crate) fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    /// Free device memory as the driver reports it.
    pub(crate) fn free_device_bytes(&self) -> Result<u64, EngError> {
        let info = self.context.memory_info()?;
        u64::try_from(info.free).map_err(|_| invalid("free device memory exceeds u64"))
    }

    /// Device bytes the pinned staging slot of the admitted maximum payload can take.
    pub(crate) fn stage_bytes(&self) -> u64 {
        self.stage_bytes
    }

    pub(crate) fn weight_cache_budget(&self) -> u64 {
        self.cache_budget
    }

    /// Device memory the dense weight cache and the staging slot may still take: the part of the
    /// free VRAM a hybrid must not give to resident experts.
    pub(crate) fn dense_reserve_bytes(&self) -> u64 {
        self.cache_budget.saturating_add(self.stage_bytes)
    }

    /// Lowers the dense weight cache ceiling; the cache evicts down to it at the next insert.
    pub(crate) fn clamp_weight_cache_budget(&mut self, bytes: u64) {
        self.cache_budget = self.cache_budget.min(bytes);
    }

    pub(crate) fn stats(&self) -> DeviceStats {
        DeviceStats {
            calls: self.clock.calls.get(),
            call: Duration::from_nanos(self.clock.call_nanos.get()),
            upload: Duration::from_nanos(self.clock.upload_nanos.get()),
            upload_bytes: self.clock.upload_bytes.get(),
        }
    }

    pub(crate) fn stream_plan(&self) -> Option<super::admission::ChunkStreamPlan> {
        self.stream_plan
    }

    pub(crate) fn context(&self) -> &CudaContext {
        &self.context
    }

    fn touch(lru: &mut VecDeque<CacheKey>, key: CacheKey) {
        if let Some(position) = lru.iter().position(|&existing| existing == key) {
            lru.remove(position);
        }
        lru.push_back(key);
    }

    /// Evicts least-recently-used entries until `incoming` bytes fit.
    fn evict_for(&mut self, incoming: u64) {
        while self.cached_bytes.saturating_add(incoming) > self.cache_budget {
            let Some(evicted) = self.lru.pop_front() else {
                return;
            };
            let freed = match evicted {
                CacheKey::F32(pointer) => self.weight_cache.remove(&pointer).map(|old| old.bytes),
                CacheKey::Packed(key) => self.packed_cache.remove(&key).map(|old| old.bytes),
            };
            if let Some(bytes) = freed {
                self.cached_bytes = self.cached_bytes.saturating_sub(bytes);
            }
        }
    }

    #[cfg(test)]
    fn cached_weights(&self) -> usize {
        self.weight_cache.len()
    }

    pub(crate) fn new(admission: CudaAdmission) -> Result<Self, EngError> {
        let limits = admission.transfer_limits();
        let max_payload = admission.max_payload();
        let workspace = admission.workspace().clone();
        let cache_budget = admission.device_weight_cache_bytes();
        let stream_plan = admission.stream_plan();
        let context =
            CudaContext::with_pinned_stage(&workspace, max_payload, move |memory, integrated| {
                admission.into_cleanup(memory, integrated)
            })?;
        Ok(Self {
            context,
            limits,
            weight_cache: HashMap::new(),
            packed_cache: HashMap::new(),
            resident_shared: HashMap::new(),
            resident_f8: HashMap::new(),
            f8_pointer_map: HashMap::new(),
            mamba_conv_state: HashMap::new(),
            mamba_ssm_state: HashMap::new(),
            mamba_small_weights: HashMap::new(),
            device_mamba_steps: std::cell::Cell::new(0),
            grouped_launches: std::cell::Cell::new(0),
            lru: VecDeque::new(),
            cached_bytes: 0,
            cache_budget,
            stream_plan,
            resident_packed: HashMap::new(),
            resident_bytes: 0,
            resident_experts_only: false,
            stage_bytes: u64::try_from(max_payload).unwrap_or(u64::MAX),
            clock: Rc::new(CallClock::default()),
        })
    }

    /// Projects one row and returns the winning vocabulary index without
    /// copying the row back to the host.
    pub(crate) fn head_argmax(
        &mut self,
        weight: &Tensor,
        input: &[f32],
    ) -> Result<(u32, f32), EngError> {
        let _timer = CallTimer::start(&self.clock);
        let (shape, bytes) = checked_matrix(weight, input)?;
        let _host_span = HostSpan::new("head_projection");
        self.context.ensure_usable()?;
        self.limits.check_shape(shape)?;
        if bytes.len() <= PINNED_STAGE_LIMIT_BYTES {
            self.context.linear_argmax_pinned_f32(bytes, shape, input)
        } else {
            self.limits.check_pageable(bytes.len())?;
            let device_weight = self.timed_upload(bytes)?;
            self.context.linear_argmax_f32(&device_weight, shape, input)
        }
    }
}

impl Projector for CudaProjector {
    type Weight = Tensor;

    fn linear(&mut self, weight: &Tensor, input: &[f32]) -> Result<Vec<f32>, EngError> {
        let _timer = CallTimer::start(&self.clock);
        // FP8-resident weights skip the F32 upload entirely (the pointer
        // map routes to f8_matvec on device-resident codes).
        if let Some(name) = self.f8_name_for(weight) {
            return self.f8_linear(name, input);
        }
        let (shape, bytes) = checked_matrix(weight, input)?;
        let _host_span = HostSpan::new("f32_projection");
        self.context.ensure_usable()?;
        self.limits.check_shape(shape)?;
        // NEM35_NO_F32_CACHE=1 falls back to the per-call pinned/pageable
        // upload path for isolating F32-cache issues at production scale.
        if std::env::var_os("NEM35_NO_F32_CACHE").is_some_and(|v| v == "1") {
            if bytes.len() <= PINNED_STAGE_LIMIT_BYTES {
                return self.context.linear_pinned_f32(bytes, shape, input);
            }
            self.limits.check_pageable(bytes.len())?;
            let device_weight = self.timed_upload(bytes)?;
            return self.context.linear_f32(&device_weight, shape, input);
        }
        let key = bytes.as_ptr() as usize;
        if let Some(entry) = self.weight_cache.get(&key) {
            Self::touch(&mut self.lru, CacheKey::F32(key));
            accounting::record_resident_call(Category::Dense);
            return self.context.linear_f32(&entry.buffer, shape, input);
        }
        if self.cache_budget == 0 {
            if bytes.len() <= PINNED_STAGE_LIMIT_BYTES {
                return self.context.linear_pinned_f32(bytes, shape, input);
            }
            self.limits.check_pageable(bytes.len())?;
            let device_weight = self.timed_upload(bytes)?;
            return self.context.linear_f32(&device_weight, shape, input);
        }
        // Cache mode: one pageable upload per weight, then device residency.
        self.limits.check_pageable(bytes.len())?;
        let byte_len = u64::try_from(bytes.len()).map_err(|_| invalid("weight exceeds u64"))?;
        accounting::record_upload(Category::Dense, bytes.len());
        let buffer = self.timed_upload(bytes)?;
        let result = self.context.linear_f32(&buffer, shape, input);
        if result.is_ok() && byte_len <= self.cache_budget {
            self.evict_for(byte_len);
            self.cached_bytes += byte_len;
            self.lru.push_back(CacheKey::F32(key));
            self.weight_cache.insert(
                key,
                CachedWeight {
                    tensor: weight.clone(),
                    buffer,
                    bytes: byte_len,
                },
            );
        }
        result
    }

    fn nvfp4_matvec_keyed(
        &mut self,
        key: u64,
        packed: &[u8],
        scale: &[u8],
        weight_scale_2: f32,
        shape: [u32; 2],
        input: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let _timer = CallTimer::start(&self.clock);
        let _host_span = HostSpan::new("nvfp4_projection");
        self.context.ensure_usable()?;
        let rows =
            usize::try_from(shape[0]).map_err(|_| invalid("row count exceeds host limits"))?;
        let columns =
            usize::try_from(shape[1]).map_err(|_| invalid("column count exceeds host limits"))?;
        self.limits.check_shape([rows, columns])?;
        self.limits.check_packed(packed.len(), scale.len())?;
        if let Some(entry) = self.resident_packed.get(&key) {
            accounting::record_resident_call(Category::Routed);
            return self.context.nvfp4_matvec(
                &entry.device_packed,
                &entry.device_scale,
                entry.weight_scale_2,
                [rows, columns],
                input,
            );
        }
        self.refuse_unresident_expert()?;
        if let Some(entry) = self.packed_cache.get(&key) {
            Self::touch(&mut self.lru, CacheKey::Packed(key));
            accounting::record_resident_call(Category::Routed);
            return self.context.nvfp4_matvec(
                &entry.device_packed,
                &entry.device_scale,
                weight_scale_2,
                [rows, columns],
                input,
            );
        }
        accounting::record_upload(Category::Routed, packed.len() + scale.len());
        let device_packed = self.timed_upload(packed)?;
        let device_scale = self.timed_upload(scale)?;
        let result = self.context.nvfp4_matvec(
            &device_packed,
            &device_scale,
            weight_scale_2,
            [rows, columns],
            input,
        );
        let pair_bytes = u64::try_from(packed.len().saturating_add(scale.len()))
            .map_err(|_| invalid("packed pair exceeds u64"))?;
        // NEM35_NO_PACKED_CACHE=1 skips device residency (upload per call).
        let cache_enabled = self.cache_budget > 0
            && !std::env::var_os("NEM35_NO_PACKED_CACHE").is_some_and(|v| v == "1");
        if result.is_ok() && cache_enabled && pair_bytes <= self.cache_budget {
            self.evict_for(pair_bytes);
            self.cached_bytes += pair_bytes;
            self.lru.push_back(CacheKey::Packed(key));
            self.packed_cache.insert(
                key,
                CachedPacked {
                    packed: packed.to_vec(),
                    scale: scale.to_vec(),
                    weight_scale_2,
                    device_packed,
                    device_scale,
                    bytes: pair_bytes,
                },
            );
        }
        result
    }

    fn bf16_matvec(
        &mut self,
        codes: &[u8],
        shape: [u32; 2],
        input: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let _host_span = HostSpan::new("bf16_projection");
        self.context.ensure_usable()?;
        let rows =
            usize::try_from(shape[0]).map_err(|_| invalid("row count exceeds host limits"))?;
        let columns =
            usize::try_from(shape[1]).map_err(|_| invalid("column count exceeds host limits"))?;
        self.limits.check_shape([rows, columns])?;
        if !codes.len().is_multiple_of(2) {
            return Err(invalid("BF16 codes must hold whole 16-bit values"));
        }
        if self.cache_budget == 0 {
            return Err(invalid(
                "stored BF16 projection requires an admitted device weight cache",
            ));
        }
        self.limits.check_pageable(codes.len())?;
        // No residency yet: caller slices lack stable identity (loader wiring
        // supplies checkpoint keys, as with the F8 path).
        let buffer = self.context.upload(codes)?;
        self.context.bf16_matvec(&buffer, [rows, columns], input)
    }

    fn f8_matvec(
        &mut self,
        codes: &[u8],
        weight_scale: f32,
        shape: [u32; 2],
        input: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let _host_span = HostSpan::new("f8_projection");
        self.context.ensure_usable()?;
        let rows =
            usize::try_from(shape[0]).map_err(|_| invalid("row count exceeds host limits"))?;
        let columns =
            usize::try_from(shape[1]).map_err(|_| invalid("column count exceeds host limits"))?;
        self.limits.check_shape([rows, columns])?;
        if self.cache_budget == 0 {
            return Err(invalid(
                "stored F8 projection requires an admitted device weight cache",
            ));
        }
        self.limits.check_pageable(codes.len())?;
        // No residency yet: the caller's slice has no stable identity here, so
        // a cached entry could be confused with a different later allocation
        // at the same address. Device residency for F8 arrives with the loader
        // wiring, which supplies checkpoint-identity keys like the packed path.
        let buffer = self.context.upload(codes)?;
        self.context
            .f8_matvec(&buffer, weight_scale, [rows, columns], input)
    }

    fn expert_linear(&mut self, _: &Tensor, _: u32, _: &[f32]) -> Result<Vec<f32>, EngError> {
        self.context.ensure_usable()?;
        Err(invalid(
            "CUDA Mamba projector does not support expert_linear",
        ))
    }

    fn rows(&mut self, _: &Tensor, _: &[u32]) -> Result<Vec<f32>, EngError> {
        self.context.ensure_usable()?;
        Err(invalid("CUDA Mamba projector does not support rows"))
    }
}

/// Checks one matrix weight and one input row, returning the shape and the
/// weight bytes the device upload needs.
fn checked_matrix<'a>(
    weight: &'a Tensor,
    input: &[f32],
) -> Result<([usize; 2], &'a [u8]), EngError> {
    let _host_span = HostSpan::new("matrix_validation");
    let &[rows, columns] = weight.shape() else {
        return Err(invalid("linear weight must be a matrix"));
    };
    if weight.dtype() != Dtype::F32 {
        return Err(invalid("linear weight must be F32"));
    }
    let rows = usize::try_from(rows).map_err(|_| invalid("row count exceeds host limits"))?;
    let columns =
        usize::try_from(columns).map_err(|_| invalid("column count exceeds host limits"))?;
    if rows == 0 || columns == 0 {
        return Err(invalid("linear dimensions must be positive"));
    }
    if input.len() != columns || input.iter().any(|value| !value.is_finite()) {
        return Err(invalid(
            "linear input must be one finite row of the weight width",
        ));
    }
    let expected_bytes = rows
        .checked_mul(columns)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| invalid("linear weight byte length exceeds host limits"))?;
    let bytes = weight
        .host_bytes()
        .ok_or_else(|| invalid("linear weight is not host-backed"))?;
    if bytes.len() != expected_bytes {
        return Err(invalid("linear weight byte length differs from its shape"));
    }
    if bytes
        .as_chunks::<4>()
        .0
        .iter()
        .any(|bytes| !f32::from_le_bytes(*bytes).is_finite())
    {
        return Err(invalid("linear weight must be finite"));
    }
    Ok(([rows, columns], bytes))
}

fn invalid(message: &str) -> EngError {
    EngError::BadRequest(format!("NemotronH CUDA projector: {message}"))
}

#[cfg(test)]
mod batch_tests {
    use super::CudaProjector;
    use crate::{
        EngError,
        family::nemotron35::admission::{
            self, ACTIVATION_WORKSPACE_BYTES,
            cuda::{CudaPlanInput, admit_cuda_plan},
        },
        kernels::cuda::CudaContext,
        memory::MemoryPolicy,
    };

    fn projector() -> CudaProjector {
        let largest = 2048_u64;
        let host_base = largest
            .checked_mul(8)
            .and_then(|copies| copies.checked_add(ACTIVATION_WORKSPACE_BYTES))
            .and_then(|bytes| bytes.checked_add(1 << 20))
            .unwrap();
        let admission = admit_cuda_plan(
            CudaPlanInput {
                host_base_bytes: host_base,
                largest_tensor_bytes: largest,
                packed_pair_bytes: largest,
                // The batch multiplies rows by the expert count, so the fixed
                // workspace must cover n * rows of output (3 * 16 here).
                max_shape: [48, 32],
                device_weight_cache_bytes: 1 << 20,
            },
            MemoryPolicy::default_share(),
            admission::host_available_bytes().expect("measured host memory"),
        )
        .unwrap();
        CudaProjector::new(admission).unwrap()
    }

    fn context(projector: &CudaProjector) -> &CudaContext {
        projector.context()
    }

    /// One synthetic expert: distinct packed nibbles and scales so any cross
    /// expert mixing in the batch kernel would change the result.
    fn expert(seed: u8) -> (Vec<u8>, Vec<u8>, f32) {
        let mut packed = vec![0u8; 16 * 32 / 2];
        let mut scale = vec![0u8; 16 * 32 / 16];
        for (index, byte) in packed.iter_mut().enumerate() {
            *byte = seed.wrapping_add(index as u8);
        }
        for (index, byte) in scale.iter_mut().enumerate() {
            *byte = 0x30u8.wrapping_add(seed).wrapping_add(index as u8);
        }
        let weight_scale_2 = 0.5 + f32::from(seed) * 0.01;
        (packed, scale, weight_scale_2)
    }

    #[test]
    fn batched_matvec_is_bit_identical_to_single_calls() -> Result<(), EngError> {
        let projector = projector();
        let context = context(&projector);
        const N: usize = 3;
        const ROWS: usize = 16;
        const COLS: usize = 32;
        let experts: Vec<_> = (0..N).map(|seed| expert(seed as u8)).collect();
        let inputs: Vec<Vec<f32>> = (0..N)
            .map(|e| (0..COLS).map(|c| (e * 7 + c) as f32 * 0.25 - 8.0).collect())
            .collect();
        // Single-call truth.
        let mut singles = Vec::new();
        for ((packed, scale, weight_scale_2), input) in experts.iter().zip(&inputs) {
            let device_packed = context.upload(packed)?;
            let device_scale = context.upload(scale)?;
            singles.push(context.nvfp4_matvec(
                &device_packed,
                &device_scale,
                *weight_scale_2,
                [ROWS, COLS],
                input,
            )?);
        }
        // Batch: one launch, one sync.
        let packed_buffers: Vec<_> = experts
            .iter()
            .map(|(packed, _, _)| context.upload(packed))
            .collect::<Result<Vec<_>, _>>()?;
        let scale_buffers: Vec<_> = experts
            .iter()
            .map(|(_, scale, _)| context.upload(scale))
            .collect::<Result<Vec<_>, _>>()?;
        let packed_refs: Vec<&_> = packed_buffers.iter().collect();
        let scale_refs: Vec<&_> = scale_buffers.iter().collect();
        let scales: Vec<f32> = experts.iter().map(|(_, _, s)| *s).collect();
        let mut flat_inputs = Vec::with_capacity(N * COLS);
        for input in &inputs {
            flat_inputs.extend_from_slice(input);
        }
        let batched = context.nvfp4_matvec_batch(
            &packed_refs,
            &scale_refs,
            &scales,
            [ROWS, COLS],
            &flat_inputs,
        )?;
        for expert in 0..N {
            let single = &singles[expert];
            let batch_slice = &batched[expert * ROWS..(expert + 1) * ROWS];
            assert_eq!(single, batch_slice, "expert {expert} differs");
        }
        Ok(())
    }
}

#[cfg(test)]
mod bf16_f8_tests {
    use super::CudaProjector;
    use crate::family::project::Projector;

    fn projector(cache_bytes: u64) -> CudaProjector {
        use crate::{
            family::nemotron35::admission::{
                self, ACTIVATION_WORKSPACE_BYTES,
                cuda::{CudaPlanInput, admit_cuda_plan},
            },
            memory::MemoryPolicy,
        };
        let largest = 32_u64;
        let host_base_bytes = largest
            .checked_mul(4)
            .and_then(|copies| copies.checked_add(ACTIVATION_WORKSPACE_BYTES))
            .and_then(|bytes| bytes.checked_add(cache_bytes))
            .unwrap();
        let admission = admit_cuda_plan(
            CudaPlanInput {
                host_base_bytes,
                largest_tensor_bytes: largest,
                packed_pair_bytes: 0,
                max_shape: [4, 4],
                device_weight_cache_bytes: cache_bytes,
            },
            MemoryPolicy::default_share(),
            admission::host_available_bytes().expect("measured host memory"),
        )
        .unwrap();
        CudaProjector::new(admission).unwrap()
    }

    #[test]
    fn stored_bf16_projection_matches_the_widened_reference_exactly() {
        // 1.0, -2.0, 0.5, 4.0 as BF16 bit patterns: exactly representable, so
        // widening is exact and the reference dot products are the F32 truth.
        let values = [1.0f32, -2.0, 0.5, 4.0, 0.25, -0.5, 2.0, 8.0];
        let codes: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input = [1.0, -2.0, 0.5, 4.0];
        let dot = |row: usize| {
            values[row * 4..row * 4 + 4]
                .iter()
                .zip(&input)
                .map(|(w, x)| w * x)
                .sum::<f32>()
        };
        let mut cuda = projector(64);
        let actual = cuda.bf16_matvec(&codes, [2, 4], &input).unwrap();
        assert_eq!(actual.len(), 2);
        assert_eq!(actual[0], dot(0));
        assert_eq!(actual[1], dot(1));
    }
}

#[cfg(test)]
mod f8_tests {
    use super::CudaProjector;
    use crate::{family::project::Projector, kernels::f8};

    fn projector() -> CudaProjector {
        use crate::{
            family::nemotron35::admission::{
                self, ACTIVATION_WORKSPACE_BYTES,
                cuda::{CudaPlanInput, admit_cuda_plan},
            },
            memory::MemoryPolicy,
        };
        let largest = 16_u64;
        let host_base_bytes = largest
            .checked_mul(4)
            .and_then(|copies| copies.checked_add(ACTIVATION_WORKSPACE_BYTES))
            .unwrap();
        let admission = admit_cuda_plan(
            CudaPlanInput {
                host_base_bytes: host_base_bytes
                    .checked_add(16)
                    .expect("fixture budget fits"),
                largest_tensor_bytes: largest,
                packed_pair_bytes: 0,
                max_shape: [4, 4],
                device_weight_cache_bytes: 16,
            },
            MemoryPolicy::default_share(),
            admission::host_available_bytes().expect("measured host memory"),
        )
        .unwrap();
        CudaProjector::new(admission).unwrap()
    }

    #[test]
    fn stored_f8_projection_matches_the_decoded_reference_exactly() {
        // Codes from the existing fixture family: finite E4M3FN values with a
        // positive per-tensor scale, so the decoded reference is finite.
        let codes = [0x38, 0x40, 0xb8, 0x30, 0x38, 0x40, 0xb8, 0x30];
        let weight_scale = 2.0f32;
        let (reference, _) = f8::WeightBlock {
            codes: &codes,
            weight_scale: &weight_scale.to_le_bytes(),
            input_scale: None,
        }
        .decode(&[2, 4])
        .unwrap();
        let expected: Vec<f32> = reference
            .host_bytes()
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let input = [1.0, -2.0, 0.5, 4.0];
        let dot = |row: usize| {
            expected[row * 4..row * 4 + 4]
                .iter()
                .zip(&input)
                .map(|(w, x)| w * x)
                .sum::<f32>()
        };
        let mut cuda = projector();
        let actual = cuda
            .f8_matvec(&codes, weight_scale, [2, 4], &input)
            .unwrap();
        assert_eq!(actual.len(), 2);
        assert_eq!(actual[0], dot(0));
        assert_eq!(actual[1], dot(1));
    }
}

#[cfg(test)]
mod tests {
    // Each test owns a live CUDA adapter lease and the process-wide admission
    // ledger retains host credit on ordinary drops, so this group is designed
    // for serialized execution: run it with --test-threads=1 (the process cap
    // is two concurrent adapter leases; parallel construction fails fast).

    use super::CudaProjector;
    use crate::{
        EngError,
        family::project::Projector,
        kernels::{f8, nvfp4},
        tensor::{Dtype, Tensor},
    };

    fn tensor(shape: &[u32], values: &[f32]) -> Tensor {
        Tensor::try_from_host(
            Dtype::F32,
            shape.to_vec(),
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        )
        .unwrap()
    }

    fn admitted_projector(max_shape: [u64; 2]) -> CudaProjector {
        use crate::{
            family::nemotron35::admission::{
                self, ACTIVATION_WORKSPACE_BYTES,
                cuda::{CudaPlanInput, admit_cuda_plan},
            },
            memory::MemoryPolicy,
        };
        let largest = max_shape[0]
            .checked_mul(max_shape[1])
            .and_then(|elements| elements.checked_mul(4))
            .unwrap();
        let host_base_bytes = largest
            .checked_mul(4)
            .and_then(|copies| copies.checked_add(ACTIVATION_WORKSPACE_BYTES))
            .unwrap();
        let admission = admit_cuda_plan(
            CudaPlanInput {
                host_base_bytes,
                largest_tensor_bytes: largest,
                packed_pair_bytes: 0,
                max_shape,
                device_weight_cache_bytes: 0,
            },
            MemoryPolicy::default_share(),
            admission::host_available_bytes().expect("measured host memory"),
        )
        .unwrap();
        CudaProjector::new(admission).unwrap()
    }

    fn cached_projector(max_shape: [u64; 2], cache_bytes: u64) -> CudaProjector {
        use crate::{
            family::nemotron35::admission::{
                self, ACTIVATION_WORKSPACE_BYTES,
                cuda::{CudaPlanInput, admit_cuda_plan},
            },
            memory::MemoryPolicy,
        };
        let largest = max_shape[0]
            .checked_mul(max_shape[1])
            .and_then(|elements| elements.checked_mul(4))
            .unwrap();
        let host_base_bytes = largest
            .checked_mul(4)
            .and_then(|copies| copies.checked_add(ACTIVATION_WORKSPACE_BYTES))
            .unwrap()
            .saturating_add(cache_bytes);
        let admission = admit_cuda_plan(
            CudaPlanInput {
                host_base_bytes,
                largest_tensor_bytes: largest,
                packed_pair_bytes: 0,
                max_shape,
                device_weight_cache_bytes: cache_bytes,
            },
            MemoryPolicy::default_share(),
            admission::host_available_bytes().expect("measured host memory"),
        )
        .unwrap();
        CudaProjector::new(admission).unwrap()
    }

    #[test]
    fn cuda_projection_matches_exact_decoded_fp8_and_nvfp4_fixtures() {
        let mut projector = admitted_projector([1, 16]);
        // Same codes and scales as tests/nemotron35_loader.rs; input scales are not weights.
        let (fp8, scale) = f8::WeightBlock {
            codes: &[0x38, 0x40, 0xb8, 0x30],
            weight_scale: &2.0f32.to_le_bytes(),
            input_scale: Some(&0.25f32.to_le_bytes()),
        }
        .decode(&[1, 4])
        .unwrap();
        assert_eq!(scale, Some(0.25));
        let nvfp4 = nvfp4::WeightBlock {
            packed: &[0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe],
            weight_scale: &[0x40],
            weight_scale_2: 0.5,
            input_scale: Some(0.125),
        }
        .decode(&[1, 16], Dtype::F32)
        .unwrap();
        let fp8_bytes = fp8.host_bytes().unwrap().to_vec();
        let nvfp4_bytes = nvfp4.host_bytes().unwrap().to_vec();
        assert_eq!(
            projector.linear(&fp8, &[1.0, -2.0, 0.5, 4.0]).unwrap(),
            [-3.0]
        );
        let mut selector = [0.0; 16];
        selector[7] = 2.0;
        selector[15] = -1.0;
        assert_eq!(projector.linear(&nvfp4, &selector).unwrap(), [18.0]);
        assert_eq!(
            projector.linear(&fp8, &[-1.0, 2.0, -0.5, -4.0]).unwrap(),
            [3.0]
        );
        assert_eq!(fp8.host_bytes().unwrap(), fp8_bytes);
        assert_eq!(nvfp4.host_bytes().unwrap(), nvfp4_bytes);
        assert_eq!(selector[7], 2.0);
        assert_eq!(selector[15], -1.0);
    }

    #[test]
    fn cuda_projection_rejects_invalid_weights_inputs_and_unsupported_operations() {
        let mut projector = admitted_projector([1, 2]);
        let weight = tensor(&[1, 2], &[1.0, 2.0]);
        for invalid in [
            tensor(&[2], &[1.0, 2.0]),
            tensor(&[0, 2], &[]),
            tensor(&[1, 0], &[]),
            tensor(&[1, 2], &[f32::NAN, 1.0]),
            tensor(&[1, 2], &[1.0, f32::INFINITY]),
            tensor(&[1, 2], &[f32::NEG_INFINITY, 1.0]),
            Tensor::try_from_host(Dtype::U8, vec![1, 2], vec![1, 2]).unwrap(),
        ] {
            assert!(matches!(
                projector.linear(&invalid, &[1.0, 1.0]),
                Err(EngError::BadRequest(_))
            ));
        }
        for input in [
            &[][..],
            &[1.0],
            &[1.0, 2.0, 3.0],
            &[f32::NAN, 1.0],
            &[1.0, f32::INFINITY],
            &[f32::NEG_INFINITY, 1.0],
        ] {
            assert!(matches!(
                projector.linear(&weight, input),
                Err(EngError::BadRequest(_))
            ));
        }
        assert!(matches!(
            projector.expert_linear(&weight, 0, &[1.0, 1.0]),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            projector.rows(&weight, &[0]),
            Err(EngError::BadRequest(_))
        ));
        assert_eq!(projector.linear(&weight, &[2.0, 3.0]).unwrap(), [8.0]);
    }

    #[test]
    fn cached_linear_returns_identical_results_and_resuses_resident_weights() {
        let mut cached = cached_projector([2, 2], 1 << 20);
        let mut streaming = admitted_projector([2, 2]);
        let a = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = tensor(&[2, 2], &[0.5, -1.0, 2.5, 0.0]);
        for weight in [&a, &b] {
            let expected = streaming.linear(weight, &[2.0, 1.0]).unwrap();
            assert_eq!(cached.linear(weight, &[2.0, 1.0]).unwrap(), expected);
            // The repeat comes from the resident device copy, not a new upload.
            assert_eq!(cached.linear(weight, &[2.0, 1.0]).unwrap(), expected);
        }
        assert_eq!(cached.cached_weights(), 2);
    }

    #[test]
    fn cache_evicts_lru_under_budget_and_keeps_exact_results() {
        // The budget holds exactly one [2, 2] tensor (16 bytes).
        let mut cached = cached_projector([2, 2], 16);
        let mut streaming = admitted_projector([2, 2]);
        let a = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = tensor(&[2, 2], &[0.5, -1.0, 2.5, 0.0]);
        for round in 0..3 {
            for weight in [&a, &b] {
                let expected = streaming.linear(weight, &[2.0, 1.0]).unwrap();
                assert_eq!(
                    cached.linear(weight, &[2.0, 1.0]).unwrap(),
                    expected,
                    "round {round}"
                );
            }
        }
        assert_eq!(
            cached.cached_weights(),
            1,
            "budget admits exactly one entry"
        );
    }

    #[test]
    fn cuda_head_argmax_returns_the_lowest_index_of_the_largest_logit() {
        let mut projector = admitted_projector([3, 2]);
        // Rows 1 and 2 tie at the maximum.
        let weight = tensor(&[3, 2], &[1.0, 0.0, 3.0, 0.0, 3.0, 0.0]);
        let (token, value) = projector.head_argmax(&weight, &[2.0, 5.0]).unwrap();
        assert_eq!(token, 1);
        assert_eq!(value, 6.0);
        let row = projector.linear(&weight, &[2.0, 5.0]).unwrap();
        assert_eq!(row, vec![2.0, 6.0, 6.0]);
        assert_eq!(row[token as usize], value);
        let single = tensor(&[1, 2], &[2.0, -1.0]);
        assert_eq!(projector.head_argmax(&single, &[1.0, 1.0]).unwrap().0, 0);
    }
}
