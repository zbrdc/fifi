//! CUDA adapter of the projection seam (decision 0040 packet 3).
//!
//! [`CudaProjector`] loads Q4_K, Q6_K and Q8_0 tensors of a GGUF file into the
//! vendored tile layout and runs `linear` through the tiled matmul. F32 and
//! F16 tensors load as raw device copies for later norms and tables, and
//! `linear` rejects them. A rank-3 tensor is a stack of experts: it loads as
//! one tiled weight of `out * experts` rows, `expert_linear` runs the matmul
//! on the tiles of one expert in place, and `linear` rejects the stack. Row gathers follow in a later
//! packet, so `rows` returns `BadRequest`.
//!
//! A load checks every tensor, including the fp16 range of Q6_K and Q8_0
//! scales, and admits the whole device demand before the first upload. A
//! quantized tensor is read, repacked and uploaded in chunks of whole 8-row
//! groups, and the scale scan reads in chunks of the same size, so host memory
//! is one chunk of source blocks plus one chunk of repacked tiles (each at
//! most [`CHUNK_LIMIT`], 64 MiB) however large the tensor is. The tensor size
//! itself is bounded only by the cap of `GgufWeights::open` (1 GiB, decision
//! 0040). An F32 or F16 tensor is staged whole, which that cap also bounds.
//! Uploaded weights are freed in reverse load order, also when a load fails
//! midway.
//!
//! [`CudaProjector::load_model`] loads a whole [`GgufModel`]: the tiled
//! tensors of its plan go to the device and its F32 tensors stay on the host.
//!
//! Every operation first reports a terminal backend failure as `Backend`
//! `execute`, then validates its input as `BadRequest`.

use std::{
    collections::HashMap,
    fs::File,
    time::{Duration, Instant},
};

use super::Projector;
use crate::{
    EngError,
    family::gguf_model::GgufModel,
    kernels::cuda::{
        self, CHUNK_LIMIT, CudaContext, DeviceBuffer, MemoryInfo, QuantTensor, TiledWeight, plan,
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

/// Options of one [`CudaProjector::load`].
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct LoadOptions<'a> {
    pub(crate) reserve: Reserve,
    /// The checkpoint file whose page cache is dropped for each uploaded range
    /// (`posix_fadvise(DONTNEED)`), or `None` to keep the page cache. Decision
    /// 0040 item 5 keeps the call only if the Xavier measurement shows a gain.
    pub(crate) drop_page_cache: Option<&'a File>,
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
            Self::Tiled { name, .. } | Self::Raw { name, .. } => name,
        }
    }

    /// Device bytes held by the tensor.
    pub(crate) fn device_bytes(&self) -> usize {
        match self {
            Self::Tiled { weight, .. } => weight.buffer().bytes(),
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

    pub(crate) fn times(&self) -> LoadTimes {
        self.times
    }
}

impl Drop for CudaWeights {
    fn drop(&mut self) {
        while self.weights.pop().is_some() {}
    }
}

/// A whole model after [`CudaProjector::load_model`].
pub(crate) struct LoadedModel {
    /// The tiled tensors of the plan, on the device.
    pub(crate) device: CudaWeights,
    /// The F32 tensors of the plan, by name.
    pub(crate) host: HashMap<String, Vec<f32>>,
}

/// Runs the projections of one forward step on the first CUDA device.
pub(crate) struct CudaProjector {
    context: CudaContext,
    /// Tile bytes per upload chunk, and the size of a scan chunk.
    chunk_limit: usize,
}

impl CudaProjector {
    /// Takes one of the two process adapter leases.
    pub(crate) fn new() -> Result<Self, EngError> {
        Ok(Self {
            context: CudaContext::new()?,
            chunk_limit: CHUNK_LIMIT,
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

    pub(crate) fn memory_info(&self) -> Result<MemoryInfo, EngError> {
        self.context.memory_info()
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
    /// name, an unsupported type or shape, or a Q6_K or Q8_0 scale outside the fp16
    /// range fails without any upload, as does a demand above free memory
    /// (`MemoryPlan`). The scale error names the tensor and the block index
    /// from the start of the tensor, wherever its chunk lies.
    pub(crate) fn load(
        &self,
        weights: &GgufWeights,
        names: &[&str],
        options: LoadOptions<'_>,
    ) -> Result<CudaWeights, EngError> {
        self.context.ensure_usable()?;
        let index = load_index(names)?;
        let tensors = names
            .iter()
            .map(|name| planned(weights, name))
            .collect::<Result<Vec<_>, _>>()?;
        let total = tensors
            .iter()
            .try_fold(0u64, |sum, &(_, bytes)| sum.checked_add(bytes))
            .ok_or_else(|| EngError::MemoryPlan("cuda admission: tensors overflow u64".into()))?;
        self.admit(total, options.reserve)?;
        let scan_began = Instant::now();
        check_scale_ranges(weights, &tensors, self.chunk_limit)?;
        let scan = scan_began.elapsed();
        let mut loaded = CudaWeights {
            weights: Vec::with_capacity(tensors.len()),
            index,
            times: LoadTimes::default(),
        };
        let upload_began = Instant::now();
        for (tensor, _) in tensors {
            loaded.weights.push(self.upload(weights, tensor)?);
            if let Some(file) = options.drop_page_cache {
                cuda::drop_page_cache(file, tensor.abs_offset(), tensor.byte_len())?;
            }
        }
        loaded.times = LoadTimes {
            scan,
            upload: upload_began.elapsed(),
        };
        Ok(loaded)
    }

    /// Loads the tiled tensors of `model` onto the device and reads its F32
    /// tensors to the host.
    ///
    /// The host tensors are read first: a bad one fails the call after about
    /// 89 MB of reads and before the first device byte is admitted or
    /// uploaded. The device load then follows [`Self::load`], so the caller's
    /// `options` carry the reserve and the file whose page cache is dropped.
    pub(crate) fn load_model(
        &self,
        model: &GgufModel,
        options: LoadOptions<'_>,
    ) -> Result<LoadedModel, EngError> {
        let host = model.read_host_f32()?;
        let names = model.plan().tiled_names();
        let device = self.load(&model.weights, &names, options)?;
        Ok(LoadedModel { device, host })
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

    fn upload(&self, weights: &GgufWeights, tensor: &GgufTensor) -> Result<CudaWeight, EngError> {
        let name = tensor.name();
        match tensor.ggml_type() {
            ty @ (GgufType::Q4K | GgufType::Q6K | GgufType::Q80) => {
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
            ty @ (GgufType::F32 | GgufType::F16) => {
                let (_, bytes) = weights.read_tensor(name)?;
                Ok(CudaWeight::Raw {
                    name: name.to_owned(),
                    ty,
                    buffer: self.context.upload(&bytes)?,
                })
            }
            other => Err(EngError::Checkpoint(format!(
                "tensor {name} has type {other:?}, which the CUDA backend does not load"
            ))),
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

/// Returns the tiles of a weight that the projection `operation` executes. A
/// raw tensor fails before any native call with a `Checkpoint` naming the
/// operation, and the backend stays usable (REPLY-17 item 2).
fn tiled<'w>(weight: &'w CudaWeight, operation: &str) -> Result<&'w TiledWeight, EngError> {
    match weight {
        CudaWeight::Tiled { weight, .. } => Ok(weight),
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

/// Looks up a tensor and returns its device bytes after upload. A tiled
/// rank-3 stack whose experts do not start on 8-row tile groups fails here,
/// before any upload.
fn planned<'w>(weights: &'w GgufWeights, name: &str) -> Result<(&'w GgufTensor, u64), EngError> {
    let tensor = weights
        .header()
        .tensor(name)
        .ok_or_else(|| EngError::Checkpoint(format!("tensor {name} is not in the file")))?;
    let bytes = plan::device_bytes(name, tensor.ggml_type(), tensor.dims())?;
    if let &[_, out, _] = tensor.dims()
        && matches!(
            tensor.ggml_type(),
            GgufType::Q4K | GgufType::Q6K | GgufType::Q80
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

/// Rejects a Q6_K or Q8_0 tensor whose scales overflow fp16 before the first
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
/// start on a tile group only when `out` is a multiple of 8, so `planned`
/// rejects other stacks before any upload.
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

#[cfg(test)]
mod model_tests;
#[cfg(test)]
mod tests;
