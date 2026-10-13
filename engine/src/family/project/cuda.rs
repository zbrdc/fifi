//! CUDA adapter of the projection seam (decision 0040 packet 3).
//!
//! [`CudaProjector`] loads Q4_K, Q6_K and Q8_0 tensors of a GGUF file into the
//! vendored tile layout and runs `linear` through the tiled matmul — or, when
//! `LoadOptions::layout` says `Native` (MATVEC-PAR S2), uploads the file's own
//! blocks for the S1 MMVQ kernels: one layout per load, never both, and the
//! default stays `Tiled`, so no existing caller changes behaviour. F32 and
//! F16 tensors load as raw device copies for later norms and tables, and
//! `linear` rejects them. A rank-3 tensor is a stack of experts: it loads as
//! one tiled weight of `out * experts` rows, `expert_linear` runs the matmul
//! on the tiles of one expert in place, and `linear` rejects the stack. Row gathers follow in a later
//! packet, so `rows` returns `BadRequest`.
//!
//! A load checks every tensor and admits the whole device demand before the first
//! upload. A tiled load also scans the fp16 scale range of every Q6_K, Q8_0 and Q5_0
//! tensor first; a native load scans nothing, because a native block keeps its exact
//! scales (R-S2-4). A quantized tensor is read, repacked and uploaded in chunks of whole 8-row
//! groups, and the scale scan reads in chunks of the same size, so host memory
//! is one chunk of source blocks plus one chunk of repacked tiles (each at
//! most [`CHUNK_LIMIT`], 64 MiB) however large the tensor is. The tensor size
//! itself is bounded only by the cap of `GgufWeights::open` (1 GiB, decision
//! 0040). An F32 or F16 tensor is staged whole, which that cap also bounds.
//! Uploaded weights are freed in reverse load order, also when a load fails
//! midway.
//!
//! [`CudaProjector::load_model`] loads a whole [`GgufModel`]: every tensor of
//! its plan goes to the device, the tiled ones as tiles and the F32 ones as
//! raw copies (decision 0047 A1).
//!
//! On an integrated GPU (Jetson) the device memory is system RAM, so a load
//! also evicts the model file before admission and after the scale scan, drops
//! each tensor's pages after its upload even when the caller passed no file,
//! and retries one tensor's upload at most twice after a device out-of-memory.
//! A discrete GPU keeps the caller's options and never retries. See
//! `load_policy` for the rules and the ownership argument.
//!
//! Every operation first reports a terminal backend failure as `Backend`
//! `execute`, then validates its input as `BadRequest`.

use std::{
    cell::Cell,
    collections::HashMap,
    fs::File,
    time::{Duration, Instant},
};

use self::load_policy::{LoadPolicy, evict_model, recover, retry_on_oom};
use super::Projector;
use crate::{
    EngError,
    family::gguf_model::GgufModel,
    kernels::cuda::{
        self, Admission, CHUNK_LIMIT, CudaContext, DeviceBuffer, Growth, MemoryInfo, NativeWeight,
        QuantTensor, TiledWeight, WorkspaceShape, plan,
    },
    weights::{GgufTensor, GgufType, GgufWeights},
};

/// Device memory that a load keeps free beside its weights, in bytes.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Reserve {
    pub(crate) kv_cache: u64,
    pub(crate) work: u64,
    pub(crate) headroom: u64,
}

/// The device layout of the quantized tensors of one load (MATVEC-PAR S2): the vendored tiles
/// (the default; every merged reader) or the file's own native blocks, which the S1 MMVQ
/// kernels and the native embedding gather read. One layout per load, never both; a load in
/// one layout serves only readers of that layout (R-S2-2). The default keeps every existing
/// caller and test on the tiled path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum WeightLayout {
    #[default]
    Tiled,
    /// Read by the resident step of MATVEC-PAR S3 through [`CudaWeights::layout`].
    Native,
}

/// Options of one [`CudaProjector::load`].
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct LoadOptions<'a> {
    pub(crate) reserve: Reserve,
    /// The checkpoint file whose page cache is dropped for each uploaded range
    /// (`posix_fadvise(DONTNEED)`), or `None` to keep the page cache. Decision
    /// 0040 item 5 keeps the call only if the Xavier measurement shows a gain.
    pub(crate) drop_page_cache: Option<&'a File>,
    /// The device layout of the quantized tensors of this load (MATVEC-PAR S2). `Tiled` by
    /// default; `Native` applies to resident-step loads only — the host `Projector` oracle
    /// reads tiles and refuses a native weight with a typed error (R-S2-2).
    pub(crate) layout: WeightLayout,
}

/// One checkpoint tensor resident on the device.
pub(crate) enum CudaWeight {
    /// A Q4_K, Q6_K or Q8_0 matrix in the vendored tile layout. `experts` is
    /// the stack depth of a rank-3 tensor, whose rows are the `experts`
    /// consecutive matrices of `weight.n_rows() / experts` rows each.
    Tiled {
        name: String,
        weight: TiledWeight,
        experts: Option<usize>,
    },
    /// A quantized matrix in the file's native block layout (MATVEC-PAR S2): the S1 MMVQ
    /// matvec kernels and the native embedding gather read it. The stack depth of a rank-3
    /// tensor rides `weight.experts()`.
    Native { name: String, weight: NativeWeight },
    /// An F32 or F16 tensor copied as read.
    Raw {
        name: String,
        ty: GgufType,
        buffer: DeviceBuffer,
    },
}

impl CudaWeight {
    pub(crate) fn name(&self) -> &str {
        match self {
            Self::Tiled { name, .. } | Self::Native { name, .. } | Self::Raw { name, .. } => name,
        }
    }

    /// Device bytes held by the tensor.
    pub(crate) fn device_bytes(&self) -> usize {
        match self {
            Self::Tiled { weight, .. } => weight.buffer().bytes(),
            Self::Native { weight, .. } => weight.buffer().bytes(),
            Self::Raw { buffer, .. } => buffer.bytes(),
        }
    }
}

/// Where the time of one [`CudaProjector::load`] went.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct LoadTimes {
    /// The pre-upload scale scan of the Q6_K and Q8_0 tensors.
    pub(crate) scan: Duration,
    /// The reads, repacks, uploads and page-cache drops of all tensors.
    pub(crate) upload: Duration,
}

/// The tensors of one load, freed in reverse load order.
pub(crate) struct CudaWeights {
    weights: Vec<CudaWeight>,
    index: HashMap<String, usize>,
    times: LoadTimes,
}

impl CudaWeights {
    pub(crate) fn get(&self, name: &str) -> Option<&CudaWeight> {
        self.index.get(name).map(|&slot| &self.weights[slot])
    }

    pub(crate) fn len(&self) -> usize {
        self.weights.len()
    }

    /// Device bytes held by all tensors of the load.
    pub(crate) fn device_bytes(&self) -> usize {
        self.weights.iter().map(CudaWeight::device_bytes).sum()
    }

    /// The device buffer of every tensor of the load, in load order: the leaked pin slice of a
    /// resident step reads them (RD-11a), beside the arena slot.
    pub(crate) fn device_buffers(&self) -> impl Iterator<Item = &DeviceBuffer> {
        self.weights.iter().map(|weight| match weight {
            CudaWeight::Tiled { weight, .. } => weight.buffer(),
            CudaWeight::Native { weight, .. } => weight.buffer(),
            CudaWeight::Raw { buffer, .. } => buffer,
        })
    }

    pub(crate) fn times(&self) -> LoadTimes {
        self.times
    }

    /// The one device layout of this load's quantized tensors (MATVEC-PAR S3b): the resident
    /// step derives its mode from it, and a mixed load is a typed refusal naming the tensor.
    pub(crate) fn layout(&self) -> Result<WeightLayout, EngError> {
        load_layout(
            self.weights
                .iter()
                .map(|weight| (weight.name(), weight.kind())),
        )
    }
}

impl Drop for CudaWeights {
    fn drop(&mut self) {
        while self.weights.pop().is_some() {}
    }
}

/// The layout kind of one checkpoint tensor as a layout input (MATVEC-PAR S3b): the vendored
/// tiles, the file's own native blocks, or a raw F32/F16 copy, which no layout rule reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TensorKind {
    Tiled,
    Native,
    Raw,
}

impl CudaWeight {
    /// The layout kind of the tensor for [`load_layout`].
    pub(crate) fn kind(&self) -> TensorKind {
        match self {
            Self::Tiled { .. } => TensorKind::Tiled,
            Self::Native { .. } => TensorKind::Native,
            Self::Raw { .. } => TensorKind::Raw,
        }
    }
}

/// The one device layout of a load's quantized tensors (MATVEC-PAR S3b): every quantized tensor
/// of a load is `Tiled` or every one is `Native` — one layout per load, never both (R-S2-2). A
/// mix is a `Checkpoint` naming the tensor that broke it; a load of raw tensors only has no
/// quantized reader and answers `Tiled`. Pure: no context, no device.
pub(crate) fn load_layout<'a>(
    tensors: impl Iterator<Item = (&'a str, TensorKind)>,
) -> Result<WeightLayout, EngError> {
    let mut layout: Option<WeightLayout> = None;
    for (name, kind) in tensors {
        let this = match kind {
            TensorKind::Tiled => WeightLayout::Tiled,
            TensorKind::Native => WeightLayout::Native,
            TensorKind::Raw => continue,
        };
        let earlier = layout.replace(this);
        if earlier.is_some_and(|earlier| earlier != this) {
            return Err(EngError::Checkpoint(format!(
                "tensor {name} is loaded in the {this:?} layout beside {earlier:?} tensors: \
                 one load serves one layout"
            )));
        }
    }
    Ok(layout.unwrap_or(WeightLayout::Tiled))
}

/// A whole model after [`CudaProjector::load_model`].
pub(crate) struct LoadedModel {
    /// All tensors of the plan, on the device.
    pub(crate) device: CudaWeights,
    /// The layout the quantized tensors were loaded in (MATVEC-PAR S2), so the resident step
    /// can assert it: one layout per load, never both. Staged (KUBDAN-MSG-68 M-1): no
    /// production reader yet; the S3 resident step asserts it.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the S3 resident step asserts the layout of its weights"
        )
    )]
    pub(crate) layout: WeightLayout,
}

/// What [`CudaProjector::transfer`] works from, after the checks and the
/// admission of a load.
struct LoadPlan<'a> {
    tensors: Vec<(&'a GgufTensor, u64)>,
    index: HashMap<String, usize>,
    policy: LoadPolicy,
    drop_file: Option<&'a File>,
    /// The device layout of the quantized tensors (MATVEC-PAR S2).
    layout: WeightLayout,
}

/// Tokens of the largest call the default workspace holds without growing: the
/// smallest row-major call (one more than the 16 tokens of the fragment-order
/// buffer).
const SERVED_TOKENS: usize = 17;
/// Longest input row of the surveyed Qwen3.6-35B-A3B Q8_0 shapes.
const SERVED_K: usize = 5120;
/// Output rows of `output.weight` of Qwen3.6-35B-A3B.
const SERVED_ROWS: usize = 248_320;

/// The workspace sizes of this model, from P6-workbuf: the larger of a one-token
/// and a `SERVED_TOKENS`-token call at the widest shape. The kernel wrappers take
/// an [`Admission`] and know no model.
pub(crate) fn served_admission() -> Admission {
    let widest = |tokens| WorkspaceShape {
        n_rows: SERVED_ROWS,
        k: SERVED_K,
        tokens,
    };
    Admission {
        shapes: vec![widest(1), widest(SERVED_TOKENS)],
        growth: Growth::Unbounded,
    }
}

/// Runs the projections of one forward step on the first CUDA device.
pub(crate) struct CudaProjector {
    context: CudaContext,
    /// Tile bytes per upload chunk, and the size of a scan chunk.
    chunk_limit: usize,
    /// Whether the device shares memory with the host (a Jetson); it selects
    /// the [`LoadPolicy`] of every load.
    integrated: bool,
    /// Device out-of-memory retries over all loads of this projector.
    oom_retries: Cell<u32>,
}

impl CudaProjector {
    /// Takes one of the two process adapter leases.
    pub(crate) fn new() -> Result<Self, EngError> {
        Ok(Self {
            context: CudaContext::with_admission(&served_admission())?,
            chunk_limit: CHUNK_LIMIT,
            integrated: cuda::device_is_integrated()?,
            oom_retries: Cell::new(0),
        })
    }

    /// Shrinks the upload and scan chunks so a small test tensor spans several.
    #[cfg(test)]
    pub(crate) fn with_chunk_limit(self, chunk_limit: usize) -> Self {
        Self {
            chunk_limit,
            ..self
        }
    }

    /// Forces the integrated or discrete load policy, so the policy logic runs
    /// on either kind of device.
    #[cfg(test)]
    pub(crate) fn with_integrated(self, integrated: bool) -> Self {
        Self { integrated, ..self }
    }

    /// Whether loads use the integrated-GPU policy.
    #[cfg(test)]
    pub(crate) fn integrated(&self) -> bool {
        self.integrated
    }

    /// Device out-of-memory retries made by all loads of this projector.
    pub(crate) fn oom_retries(&self) -> u32 {
        self.oom_retries.get()
    }

    pub(crate) fn memory_info(&self) -> Result<MemoryInfo, EngError> {
        self.context.memory_info()
    }

    /// The CUDA context that this projector's weights live on. The resident step (decision
    /// 0047, RD-9a) is built from it and a `&CudaWeights`, so both share one adapter lease.
    pub(crate) fn context(&self) -> &CudaContext {
        &self.context
    }

    /// Device bytes held by live buffers in the process; see
    /// [`CudaContext::live_bytes`].
    #[cfg(test)]
    pub(crate) fn live_bytes(&self) -> usize {
        self.context.live_bytes()
    }

    /// Uploads the tensors `names` of `weights`.
    ///
    /// Every tensor is checked and the whole demand admitted against free
    /// device memory before the first upload. A missing tensor, a repeated
    /// name, an unsupported type or shape, or - on the tiled layout only - a
    /// Q6_K, Q8_0 or Q5_0 scale outside the fp16 range fails without any
    /// upload, as does a demand above free memory (`MemoryPlan`). The scale
    /// error names the tensor and the block index from the start of the
    /// tensor, wherever its chunk lies.
    ///
    /// On an integrated GPU the load also evicts the model file before
    /// admission, forces the per-tensor page-cache drop and retries a tensor's
    /// upload after a device out-of-memory; see `load_policy`.
    pub(crate) fn load(
        &self,
        weights: &GgufWeights,
        names: &[&str],
        options: LoadOptions<'_>,
    ) -> Result<CudaWeights, EngError> {
        self.context.ensure_usable()?;
        let index = load_index(names)?;
        let layout = options.layout;
        let tensors = names
            .iter()
            .map(|name| planned(weights, name, layout))
            .collect::<Result<Vec<_>, _>>()?;
        let total = tensors
            .iter()
            .try_fold(0u64, |sum, &(_, bytes)| sum.checked_add(bytes))
            .ok_or_else(|| EngError::MemoryPlan("cuda admission: tensors overflow u64".into()))?;
        let policy = LoadPolicy::for_device(self.integrated);
        // Before admission: the free figure of an integrated GPU counts cache as
        // free, so evicting first lets the allocator find that memory.
        if policy.evict_file {
            evict_model(weights.file())?;
        }
        self.admit(total, options.reserve)?;
        let drop_file = policy.drop_file(weights.file(), options.drop_page_cache);
        self.transfer(
            weights,
            LoadPlan {
                tensors,
                index,
                policy,
                drop_file,
                layout,
            },
        )
    }

    /// Scans, then uploads the planned tensors. Tests call it to reach the upload
    /// loop with a tensor that admission would refuse.
    ///
    /// Ownership: a tensor joins `loaded` only after its upload returned `Ok`.
    /// Any `?` drops `loaded`, which frees its tensors in reverse load order
    /// (decision 0022). A failed attempt owns nothing and leaves the live byte
    /// count unchanged, so a retry starts from the same state. No retry runs
    /// after a terminal latch, because the attempt then returns `Backend`
    /// `execute`, which is never retried.
    fn transfer(&self, weights: &GgufWeights, plan: LoadPlan<'_>) -> Result<CudaWeights, EngError> {
        let LoadPlan {
            tensors,
            index,
            policy,
            drop_file,
            layout,
        } = plan;
        let scan_began = Instant::now();
        // The fp16 scale scans guard the tile dequantisation (fp16(d * scale)); a native load
        // keeps the file's exact scales and scans nothing (R-S2-4).
        if layout == WeightLayout::Tiled {
            check_scale_ranges(weights, &tensors, self.chunk_limit)?;
        }
        let scan = scan_began.elapsed();
        // A tiled load's scan read every Q6_K, Q8_0 and Q5_0 tensor once and refilled the
        // cache; a native load ran no scan (R-S2-4).
        if policy.evict_file {
            evict_model(weights.file())?;
        }
        let mut loaded = CudaWeights {
            weights: Vec::with_capacity(tensors.len()),
            index,
            times: LoadTimes::default(),
        };
        let upload_began = Instant::now();
        for (tensor, _) in tensors {
            let weight = retry_on_oom(
                &policy,
                &self.oom_retries,
                || self.upload(weights, tensor, layout),
                || recover(weights.file(), policy.retry_pause),
            )?;
            loaded.weights.push(weight);
            if let Some(file) = drop_file {
                cuda::drop_page_cache(file, tensor.abs_offset(), tensor.byte_len())?;
            }
        }
        loaded.times = LoadTimes {
            scan,
            upload: upload_began.elapsed(),
        };
        Ok(loaded)
    }

    /// Loads every tensor of `model` onto the device: the tiled tensors as
    /// tiles, the F32 tensors as raw copies.
    ///
    /// This is one [`Self::load`] of the plan's names in inventory order, so
    /// the caller's `options` carry the reserve and the file whose page cache
    /// is dropped, admission counts the F32 bytes with the tiles, and a
    /// failure after any upload frees what was uploaded.
    pub(crate) fn load_model(
        &self,
        model: &GgufModel,
        options: LoadOptions<'_>,
    ) -> Result<LoadedModel, EngError> {
        let names = model.plan().device_names();
        let layout = options.layout;
        let device = self.load(&model.weights, &names, options)?;
        Ok(LoadedModel { device, layout })
    }

    fn admit(&self, tensors: u64, reserve: Reserve) -> Result<(), EngError> {
        let demand = plan::Demand {
            tensors,
            kv_cache: reserve.kv_cache,
            work: reserve.work,
            headroom: reserve.headroom,
        };
        let free = self.context.memory_info()?.free;
        plan::admit(&demand, u64::try_from(free).unwrap_or(u64::MAX)).map(drop)
    }

    fn upload(
        &self,
        weights: &GgufWeights,
        tensor: &GgufTensor,
        layout: WeightLayout,
    ) -> Result<CudaWeight, EngError> {
        let name = tensor.name();
        match (tensor.ggml_type(), layout) {
            (ty @ (GgufType::Q4K | GgufType::Q6K | GgufType::Q80 | GgufType::Q50), layout) => {
                self.upload_quant(weights, tensor, name, ty, layout)
            }
            (ty @ (GgufType::F32 | GgufType::F16), _) => {
                let (_, bytes) = weights.read_tensor(name)?;
                Ok(CudaWeight::Raw {
                    name: name.to_owned(),
                    ty,
                    buffer: self.context.upload(&bytes)?,
                })
            }
            (other, WeightLayout::Native) => Err(EngError::Checkpoint(format!(
                "tensor {name} has type {other:?}, which the CUDA native weights do not take"
            ))),
            (other, _) => Err(EngError::Checkpoint(format!(
                "tensor {name} has type {other:?}, which the CUDA backend does not load"
            ))),
        }
    }

    /// Uploads one quantized tensor in `layout`: the repacked tiles through `upload_tiled`
    /// (whose dequantisation the fp16 scale scans guard) or the file's own blocks through
    /// `upload_native` (no repack, no scan, R-S2-4), with the stack depth of a rank-3
    /// tensor riding the weight.
    fn upload_quant(
        &self,
        weights: &GgufWeights,
        tensor: &GgufTensor,
        name: &str,
        ty: GgufType,
        layout: WeightLayout,
    ) -> Result<CudaWeight, EngError> {
        let (n_rows, k) = matrix_shape(tensor)?;
        let experts = stack_depth(tensor)?;
        let source_len = usize::try_from(tensor.byte_len()).map_err(|_| {
            EngError::MemoryPlan(format!("cuda allocate: tensor {name} does not fit"))
        })?;
        let quant = QuantTensor {
            name,
            ty,
            n_rows,
            k,
            source_len,
        };
        match layout {
            WeightLayout::Tiled => {
                let weight =
                    self.context
                        .upload_tiled(quant, self.chunk_limit, &mut |at, buf| {
                            weights.read_range(name, at as u64, buf)
                        })?;
                Ok(CudaWeight::Tiled {
                    name: name.to_owned(),
                    weight,
                    experts,
                })
            }
            WeightLayout::Native => {
                // The file's own blocks, no repack and no scale scan (R-S2-4).
                let weight = self
                    .context
                    .upload_native(quant, self.chunk_limit, &mut |at, buf| {
                        weights.read_range(name, at as u64, buf)
                    })?
                    .with_experts(experts);
                Ok(CudaWeight::Native {
                    name: name.to_owned(),
                    weight,
                })
            }
        }
    }
}

impl Projector for CudaProjector {
    type Weight = CudaWeight;

    fn linear(&mut self, weight: &CudaWeight, x: &[f32]) -> Result<Vec<f32>, EngError> {
        self.context.ensure_usable()?;
        let tiled = tiled(weight, "linear")?;
        if let CudaWeight::Tiled {
            name,
            experts: Some(_),
            ..
        } = weight
        {
            return Err(EngError::BadRequest(format!(
                "tensor {name} is a 3-D expert stack; use expert_linear"
            )));
        }
        if x.len() != tiled.k() {
            return Err(EngError::BadRequest(format!(
                "cuda linear input has {} values, expected {}",
                x.len(),
                tiled.k()
            )));
        }
        self.context.linear(tiled, x)
    }

    fn expert_linear(
        &mut self,
        stack: &CudaWeight,
        expert: u32,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        self.context.ensure_usable()?;
        let (name, tiled, experts) = expert_stack(stack)?;
        let index = usize::try_from(expert)
            .ok()
            .filter(|&index| index < experts)
            .ok_or_else(|| {
                EngError::BadRequest(format!(
                    "expert {expert} is outside the {experts} experts of tensor {name}"
                ))
            })?;
        if x.len() != tiled.k() {
            return Err(EngError::BadRequest(format!(
                "cuda expert_linear input has {} values, expected {}",
                x.len(),
                tiled.k()
            )));
        }
        // `experts` divides the rows of the weight, see `stack_depth`, and it is
        // not 0 because a weight of 0 rows fails to upload.
        let out = tiled.n_rows() / experts;
        self.context.linear_rows(tiled, index * out, out, x)
    }

    fn rows(&mut self, _table: &CudaWeight, _rows: &[u32]) -> Result<Vec<f32>, EngError> {
        self.context.ensure_usable()?;
        Err(EngError::BadRequest(
            "cuda rows is not implemented in packet 3".into(),
        ))
    }
}

/// Returns the tiles of a weight that the projection `operation` executes. A raw tensor
/// fails before any native call with a `Checkpoint` naming the operation, and the backend
/// stays usable (REPLY-17 item 2). A native weight fails the same way: the host projection
/// reads tiles and never reinterprets the other layout (R-S2-2).
fn tiled<'w>(weight: &'w CudaWeight, operation: &str) -> Result<&'w TiledWeight, EngError> {
    match weight {
        CudaWeight::Tiled { weight, .. } => Ok(weight),
        CudaWeight::Native { name, .. } => Err(EngError::Checkpoint(format!(
            "tensor {name} is loaded in the native layout; cuda {operation} reads tiles"
        ))),
        CudaWeight::Raw { name, ty, .. } => Err(EngError::Checkpoint(format!(
            "tensor {name} has type {ty:?}; cuda {operation} executes Q4_K, Q6_K and Q8_0 only"
        ))),
    }
}

/// Returns the name, tiles and depth of a stack that `expert_linear` executes.
/// A raw tensor fails like in [`tiled`], and a tiled tensor that is not a
/// rank-3 stack is a `BadRequest` naming it.
fn expert_stack(stack: &CudaWeight) -> Result<(&str, &TiledWeight, usize), EngError> {
    let weight = tiled(stack, "expert_linear")?;
    match stack {
        CudaWeight::Tiled {
            name,
            experts: Some(experts),
            ..
        } => Ok((name, weight, *experts)),
        _ => Err(EngError::BadRequest(format!(
            "tensor {} is not a 3-D expert stack",
            stack.name()
        ))),
    }
}

fn load_index(names: &[&str]) -> Result<HashMap<String, usize>, EngError> {
    let mut index = HashMap::with_capacity(names.len());
    for (slot, &name) in names.iter().enumerate() {
        if index.insert(name.to_owned(), slot).is_some() {
            return Err(EngError::BadRequest(format!(
                "cuda load names tensor {name} twice"
            )));
        }
    }
    Ok(index)
}

/// Looks up a tensor and returns its device bytes after upload, counted for `layout`
/// (MATVEC-PAR S2: the tiles and the native blocks differ only for Q6_K). A tiled rank-3
/// stack whose experts do not start on 8-row tile groups fails here, before any upload; a
/// native stack has no such rule (the S1 expert kernel reads the device-side id).
fn planned<'w>(
    weights: &'w GgufWeights,
    name: &str,
    layout: WeightLayout,
) -> Result<(&'w GgufTensor, u64), EngError> {
    let tensor = weights
        .header()
        .tensor(name)
        .ok_or_else(|| EngError::Checkpoint(format!("tensor {name} is not in the file")))?;
    let bytes = match layout {
        WeightLayout::Tiled => plan::device_bytes(name, tensor.ggml_type(), tensor.dims())?,
        WeightLayout::Native => plan::native_device_bytes(name, tensor.ggml_type(), tensor.dims())?,
    };
    if layout == WeightLayout::Tiled
        && let &[_, out, _] = tensor.dims()
        && matches!(
            tensor.ggml_type(),
            GgufType::Q4K | GgufType::Q6K | GgufType::Q80 | GgufType::Q50
        )
        && !out.is_multiple_of(plan::TILE_ROWS)
    {
        return Err(EngError::Checkpoint(format!(
            "tensor {name} stacks experts of {out} rows, which do not start on {}-row tile groups",
            plan::TILE_ROWS
        )));
    }
    Ok((tensor, bytes))
}

/// Rejects a Q6_K, Q8_0 or Q5_0 tensor whose scales overflow fp16 before the first
/// upload. This reads each such tensor once more than the upload does, one
/// tensor and one chunk of whole blocks at a time, so host memory stays at
/// one chunk of at most `chunk_limit` bytes.
fn check_scale_ranges(
    weights: &GgufWeights,
    tensors: &[(&GgufTensor, u64)],
    chunk_limit: usize,
) -> Result<(), EngError> {
    for &(tensor, _) in tensors {
        let check = match tensor.ggml_type() {
            GgufType::Q6K => cuda::check_q6k_scales,
            GgufType::Q80 => cuda::check_q8_0_scales,
            GgufType::Q50 => cuda::check_q5_0_scales,
            _ => continue,
        };
        scan_blocks(weights, tensor, chunk_limit, check)?;
    }
    Ok(())
}

/// Reads `tensor` in chunks of whole blocks, each at most `chunk_limit`
/// bytes (one block at least), and passes every chunk to `check` with the
/// index of its first block.
fn scan_blocks(
    weights: &GgufWeights,
    tensor: &GgufTensor,
    chunk_limit: usize,
    check: fn(&str, usize, &[u8]) -> Result<(), EngError>,
) -> Result<(), EngError> {
    let name = tensor.name();
    let does_not_fit =
        || EngError::MemoryPlan(format!("cuda allocate: tensor {name} does not fit"));
    let block = usize::try_from(tensor.ggml_type().block_bytes()).map_err(|_| does_not_fit())?;
    let len = usize::try_from(tensor.byte_len()).map_err(|_| does_not_fit())?;
    let total = len / block;
    let per_chunk = (chunk_limit / block).clamp(1, total.max(1));
    let mut chunk = Vec::new();
    chunk
        .try_reserve_exact(per_chunk * block)
        .map_err(|error| {
            EngError::MemoryPlan(format!(
                "cuda allocate: scan chunk of tensor {name}: {error}"
            ))
        })?;
    chunk.resize(per_chunk * block, 0);
    for first in (0..total).step_by(per_chunk) {
        let blocks = per_chunk.min(total - first);
        let part = &mut chunk[..blocks * block];
        weights.read_range(name, (first * block) as u64, part)?;
        check(name, first, part)?;
    }
    Ok(())
}

/// Returns the stack depth of a rank-3 tensor `[k, out, experts]`, or `None`
/// for any other rank. Expert `e` owns rows `e * out..(e + 1) * out`, which
/// start on a tile group only when `out` is a multiple of 8; `planned` rejects
/// other stacks for a TILED load before any upload (a native stack has no tile
/// groups and loads at any expert height).
fn stack_depth(tensor: &GgufTensor) -> Result<Option<usize>, EngError> {
    let &[_, _, experts] = tensor.dims() else {
        return Ok(None);
    };
    usize::try_from(experts).map(Some).map_err(|_| {
        EngError::Checkpoint(format!("tensor {} has an unusable shape", tensor.name()))
    })
}

/// Returns `(n_rows, k)` of a tensor whose GGUF `dims[0]` is the row length.
fn matrix_shape(tensor: &GgufTensor) -> Result<(usize, usize), EngError> {
    let shape_error =
        || EngError::Checkpoint(format!("tensor {} has an unusable shape", tensor.name()));
    let (&k, rest) = tensor.dims().split_first().ok_or_else(shape_error)?;
    let n_rows = rest
        .iter()
        .try_fold(1u64, |rows, &dim| rows.checked_mul(dim))
        .ok_or_else(shape_error)?;
    let n_rows = usize::try_from(n_rows).map_err(|_| shape_error())?;
    let k = usize::try_from(k).map_err(|_| shape_error())?;
    Ok((n_rows, k))
}

mod load_policy;
#[cfg(test)]
mod model_tests;
#[cfg(test)]
mod tests;
