//! CUDA adapter of the projection seam.
//!
//! [`CudaProjector`] loads Q4_K, Q6_K and Q8_0 tensors of a GGUF file into the vendored tile
//! layout and runs `linear` through the tiled matmul — or, when `LoadOptions::layout` says
//! `Native`, uploads the file's own blocks for the MMVQ kernels: one layout per load, never
//! both, and the default stays `Tiled`, so no existing caller changes behaviour. F32 and F16
//! tensors load as raw device copies for later norms and tables, and `linear` rejects them.
//! A rank-3 tensor is a stack of experts: it loads as one tiled weight of `out * experts`
//! rows, `expert_linear` runs the matmul on the tiles of one expert in place, and `linear`
//! rejects the stack. Row gathers are not implemented, so `rows` returns `BadRequest`.
//!
//! A load checks every tensor and admits the whole device demand before the first upload. A
//! tiled load also scans the fp16 scale range of every Q6_K, Q8_0 and Q5_0 tensor first; a
//! native load scans nothing, because a native block keeps its exact scales. A quantized
//! tensor is read, repacked and uploaded in chunks of whole 8-row groups, and the scale scan
//! reads in chunks of the same size, so host memory is one chunk of source blocks plus one
//! chunk of repacked tiles (each at most [`CHUNK_LIMIT`], 64 MiB) however large the tensor
//! is. The tensor size itself is bounded only by the cap of `GgufWeights::open` (1 GiB). An
//! F32 or F16 tensor is staged whole, which that cap also bounds. Uploaded weights are freed
//! in reverse load order, also when a load fails midway.
//!
//! [`CudaProjector::load_model`] loads a whole [`GgufModel`]: every tensor of its plan goes
//! to the device, the tiled ones as tiles and the F32 ones as raw copies.
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
    collections::{HashMap, HashSet},
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

/// The device layout of the quantized tensors of one load: the vendored tiles (the default;
/// every merged reader) or the file's own native blocks, which the MMVQ kernels and the
/// native embedding gather read. One layout per load, never both; a load in one layout
/// serves only readers of that layout. The default keeps every existing caller and test on
/// the tiled path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum WeightLayout {
    #[default]
    Tiled,
    /// Read by the resident step through [`CudaWeights::layout`].
    Native,
    Exl3,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct LoadOptions<'a> {
    pub(crate) reserve: Reserve,
    /// The checkpoint file whose page cache is dropped for each uploaded range
    /// (`posix_fadvise(DONTNEED)`), or `None` to keep the page cache. The call stays only if
    /// a measurement on the Xavier shows a gain.
    pub(crate) drop_page_cache: Option<&'a File>,
    /// The device layout of the quantized tensors of this load. `Tiled` by default; `Native`
    /// applies to resident-step loads only — the host `Projector` oracle reads tiles and
    /// refuses a native weight with a typed error.
    pub(crate) layout: WeightLayout,
}

/// One tensor's slice of a sliced load: the column range and row ranges
/// (in values, GGUF order) this device uploads. Ranges must land on the
/// tensor type's native block boundaries; `plan::plan_slice_rows` checks
/// them. Several row ranges concatenate in order (a fused tensor's
/// logical row groups).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SliceSpec {
    pub(crate) cols: std::ops::Range<u64>,
    pub(crate) rows: Vec<std::ops::Range<u64>>,
}

/// Reads `buf.len()` source bytes of one tensor at byte `offset`, the shape
/// of `GgufWeights::read_range` a sliced upload reads through.
type RangeReader<'a> = dyn FnMut(u64, &mut [u8]) -> Result<(), EngError> + 'a;

/// Serves an assembled slice byte stream from its source segments: given an
/// offset into the assembled stream, fill the buffer by reading the
/// corresponding file ranges. This is the reader a sliced native upload
/// hands to `upload_native`, whose chunk offsets address the assembled
/// (sliced) tensor exactly as they address a whole one today.
struct SliceReader<'s> {
    segments: &'s [(u64, u64)],
    /// Assembled-stream offset at which each segment starts.
    starts: Vec<u64>,
}

impl<'s> SliceReader<'s> {
    fn new(segments: &'s [(u64, u64)]) -> Self {
        let mut starts = Vec::with_capacity(segments.len());
        let mut at = 0u64;
        for &(_, len) in segments {
            starts.push(at);
            at += len;
        }
        Self { segments, starts }
    }

    fn fill(
        &self,
        at: u64,
        out: &mut [u8],
        read_range: &mut RangeReader<'_>,
    ) -> Result<(), EngError> {
        let mut done = 0usize;
        while done < out.len() {
            let pos = at + done as u64;
            let index = match self.starts.binary_search(&pos) {
                Ok(index) => index,
                Err(next) => next.checked_sub(1).ok_or_else(|| {
                    EngError::Checkpoint(
                        "cuda sliced upload: a read starts before the slice".into(),
                    )
                })?,
            };
            let (source, len) = self.segments[index];
            let within = pos - self.starts[index];
            if within >= len {
                return Err(EngError::Checkpoint(
                    "cuda sliced upload: a read runs past the slice".into(),
                ));
            }
            let take = usize::try_from(len - within)
                .map_err(|_| EngError::MemoryPlan("cuda sliced upload: size overflows".into()))?
                .min(out.len() - done);
            read_range(source + within, &mut out[done..done + take])?;
            done += take;
        }
        Ok(())
    }
}

/// One planned tensor of a load: its file record, its admitted device
/// bytes, and the slice it uploads under, if this is a sliced load.
struct PlannedTensor<'w> {
    tensor: &'w GgufTensor,
    bytes: u64,
    slice: Option<plan::TensorSlice>,
    mapped: bool,
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
    /// A quantized matrix in the file's native block layout: the MMVQ matvec kernels and the
    /// native embedding gather read it. The stack depth of a rank-3 tensor rides
    /// `weight.experts()`.
    Native { name: String, weight: NativeWeight },
    Exl3 {
        name: String,
        weight: exl3::Exl3Weight,
    },
    /// The routed experts of one projection of one layer, in one bank.
    Exl3Bank { name: String, bank: exl3::Exl3Bank },
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
            Self::Exl3 { name, .. } | Self::Exl3Bank { name, .. } => name,
        }
    }

    fn buffers(&self) -> Vec<&DeviceBuffer> {
        match self {
            Self::Tiled { weight, .. } => vec![weight.buffer()],
            Self::Native { weight, .. } => vec![weight.buffer()],
            Self::Raw { buffer, .. } => vec![buffer],
            Self::Exl3 { weight, .. } => weight.device_buffers().collect(),
            Self::Exl3Bank { bank, .. } => bank.device_buffers().collect(),
        }
    }

    pub(crate) fn device_bytes(&self) -> usize {
        match self {
            Self::Tiled { weight, .. } => weight.buffer().bytes(),
            Self::Native { weight, .. } => weight.buffer().bytes(),
            Self::Raw { buffer, .. } => buffer.bytes(),
            Self::Exl3 { weight, .. } => weight.device_bytes(),
            Self::Exl3Bank { bank, .. } => bank.device_bytes(),
        }
    }
}

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
    /// Own an explicit EXL3 load after its source adapter checked the stored metadata.
    pub(crate) fn from_exl3(weights: Vec<CudaWeight>) -> Result<Self, EngError> {
        let mut loaded = Self {
            weights,
            index: HashMap::new(),
            times: LoadTimes::default(),
        };
        let names: Vec<_> = loaded.weights.iter().map(CudaWeight::name).collect();
        loaded.index = load_index(&names)?;
        if loaded.layout()? != WeightLayout::Exl3 {
            return Err(EngError::Checkpoint(
                "EXL3 load has no EXL3 matrices".into(),
            ));
        }
        Ok(loaded)
    }

    pub(crate) fn get(&self, name: &str) -> Option<&CudaWeight> {
        self.index.get(name).map(|&slot| &self.weights[slot])
    }

    pub(crate) fn len(&self) -> usize {
        self.weights.len()
    }

    pub(crate) fn device_bytes(&self) -> usize {
        self.weights.iter().map(CudaWeight::device_bytes).sum()
    }

    /// Every device buffer of the load, in load order: the leaked pin slice of a resident
    /// step reads them, beside the arena slot.
    pub(crate) fn device_buffers(&self) -> impl Iterator<Item = &DeviceBuffer> {
        self.weights.iter().flat_map(CudaWeight::buffers)
    }

    pub(crate) fn times(&self) -> LoadTimes {
        self.times
    }

    /// The one device layout of this load's quantized tensors: the resident step derives its
    /// mode from it, and a mixed load is a typed refusal naming the tensor.
    pub(crate) fn layout(&self) -> Result<WeightLayout, EngError> {
        // The tiled-head hybrid's overridden names do not make the load mixed: they are exempt
        // from the one-layout rule exactly like the fused in-projection's hidden siblings.
        let overrides = tiled_head_overrides();
        load_layout(
            self.weights
                .iter()
                .map(|weight| (weight.name(), weight.kind()))
                .filter(|(name, _)| !overrides.iter().any(|o| o == name)),
        )
    }
}

impl Drop for CudaWeights {
    fn drop(&mut self) {
        while self.weights.pop().is_some() {}
    }
}

/// The layout kind of one checkpoint tensor as a layout input: the vendored tiles, the
/// file's own native blocks, or a raw F32/F16 copy, which no layout rule reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TensorKind {
    Tiled,
    Native,
    Raw,
    Exl3,
}

impl CudaWeight {
    pub(crate) fn kind(&self) -> TensorKind {
        match self {
            Self::Tiled { .. } => TensorKind::Tiled,
            Self::Native { .. } => TensorKind::Native,
            Self::Raw { .. } => TensorKind::Raw,
            Self::Exl3 { .. } | Self::Exl3Bank { .. } => TensorKind::Exl3,
        }
    }
}

/// The one device layout of a load's quantized tensors: every quantized tensor of a load is
/// `Tiled` or every one is `Native` — one layout per load, never both. A mix is a
/// `Checkpoint` naming the tensor that broke it; a load of raw tensors only has no quantized
/// reader and answers `Tiled`. Pure: no context, no device.
pub(crate) fn load_layout<'a>(
    tensors: impl Iterator<Item = (&'a str, TensorKind)>,
) -> Result<WeightLayout, EngError> {
    let mut layout: Option<WeightLayout> = None;
    for (name, kind) in tensors {
        let this = match kind {
            TensorKind::Tiled => WeightLayout::Tiled,
            TensorKind::Native => WeightLayout::Native,
            TensorKind::Raw => continue,
            TensorKind::Exl3 => WeightLayout::Exl3,
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

pub(crate) struct LoadedModel {
    pub(crate) device: CudaWeights,
    /// The layout the quantized tensors were loaded in, so the resident step can assert it: one
    /// layout per load, never both. Staged: no production reader yet; the resident step asserts
    /// it.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the resident step asserts the layout of its weights"
        )
    )]
    pub(crate) layout: WeightLayout,
}

struct LoadPlan<'a> {
    tensors: Vec<PlannedTensor<'a>>,
    index: HashMap<String, usize>,
    policy: LoadPolicy,
    drop_file: Option<&'a File>,
    /// The load layout - the `tiled_overrides` names of a native load upload as tiles beside
    /// it and do not change this field.
    layout: WeightLayout,
    /// The names this load uploads in the `Tiled` layout beside the `Native` load
    /// (`INFERENCE_CUDA_TILED_HEAD`); empty unless the hybrid is on.
    tiled_overrides: Vec<String>,
    /// The fused GDN input-projection groups of this load, as `(qkv position, [sibling
    /// positions in column order])`; empty unless the fusion flag is set. The sibling
    /// positions still hold their admission bytes but never upload.
    fused: Vec<(usize, [usize; 3])>,
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

pub(crate) struct CudaProjector {
    context: CudaContext,
    /// Tile bytes per upload chunk, and the size of a scan chunk.
    chunk_limit: usize,
    /// Whether the device shares memory with the host (a Jetson); it selects
    /// the [`LoadPolicy`] of every load.
    integrated: bool,
    oom_retries: Cell<u32>,
}

/// What the weight load path reads from its owner: the context that
/// allocates, the upload chunk limit, the device's load policy flag, and
/// the out-of-memory retry budget. [`CudaProjector`] supplies its own; a
/// [`BorrowedLoader`] runs the same functions over a caller's context.
pub(crate) struct LoadKnobs<'a> {
    context: &'a CudaContext,
    chunk_limit: usize,
    integrated: bool,
    oom_retries: &'a Cell<u32>,
}

/// One sliced load over a borrowed context: the per-card loads of the
/// two-device start, which run on the session's own card contexts and take
/// no projector adapter lease of their own (the process admits two).
pub(crate) struct BorrowedLoader<'a> {
    context: &'a CudaContext,
    chunk_limit: usize,
    integrated: bool,
    oom_retries: Cell<u32>,
}

impl<'a> BorrowedLoader<'a> {
    pub(crate) fn new(context: &'a CudaContext) -> Result<Self, EngError> {
        Ok(Self {
            context,
            chunk_limit: CHUNK_LIMIT,
            integrated: cuda::device_is_integrated()?,
            oom_retries: Cell::new(0),
        })
    }

    fn knobs(&self) -> LoadKnobs<'_> {
        LoadKnobs {
            context: self.context,
            chunk_limit: self.chunk_limit,
            integrated: self.integrated,
            oom_retries: &self.oom_retries,
        }
    }

    /// The sliced load of [`CudaProjector::load_sliced`], on the caller's
    /// context.
    pub(crate) fn load_sliced(
        &self,
        weights: &GgufWeights,
        names: &[&str],
        options: LoadOptions<'_>,
        slices: &[(&str, SliceSpec)],
        host_expert_bytes: u64,
    ) -> Result<CudaWeights, EngError> {
        load_inner(
            &self.knobs(),
            weights,
            names,
            options,
            slices,
            host_expert_bytes,
        )
    }
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

    /// Like [`Self::new`], on CUDA device `device`: one projector per device
    /// of a multi-device load, each admitting against its own device's free
    /// memory. The integrated-GPU load policy is a one-device (Jetson)
    /// concern and is read for device 0 as today.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "staged for the two-device start, which builds one projector per device"
        )
    )]
    pub(crate) fn on_device(device: usize) -> Result<Self, EngError> {
        Ok(Self {
            context: CudaContext::with_admission_on(device, &served_admission())?,
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

    #[cfg(test)]
    pub(crate) fn integrated(&self) -> bool {
        self.integrated
    }

    pub(crate) fn oom_retries(&self) -> u32 {
        self.oom_retries.get()
    }

    pub(crate) fn memory_info(&self) -> Result<MemoryInfo, EngError> {
        self.context.memory_info()
    }

    /// The CUDA context that this projector's weights live on. The resident step is built from
    /// it and a `&CudaWeights`, so both share one adapter lease.
    pub(crate) fn context(&self) -> &CudaContext {
        &self.context
    }

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
    fn knobs(&self) -> LoadKnobs<'_> {
        LoadKnobs {
            context: &self.context,
            chunk_limit: self.chunk_limit,
            integrated: self.integrated,
            oom_retries: &self.oom_retries,
        }
    }

    /// Admission of one load request against this projector's device.
    pub(crate) fn admit(&self, tensors: u64, reserve: Reserve) -> Result<(), EngError> {
        admit(&self.knobs(), tensors, reserve)
    }

    /// The transfer of one planned load, for the load-policy tests.
    #[cfg(test)]
    fn transfer(&self, weights: &GgufWeights, plan: LoadPlan<'_>) -> Result<CudaWeights, EngError> {
        transfer(&self.knobs(), weights, plan)
    }

    pub(crate) fn load(
        &self,
        weights: &GgufWeights,
        names: &[&str],
        options: LoadOptions<'_>,
    ) -> Result<CudaWeights, EngError> {
        load_inner(&self.knobs(), weights, names, options, &[], 0)
    }

    /// Like [`Self::load`], uploading only the named tensors' slices: each
    /// tensor of `slices` lands under its own name with the sliced shape
    /// (its `NativeWeight` rows and `k` are the slice's), every other named
    /// tensor loads whole. Admission counts slice bytes, so per-device
    /// demand is this device's share alone. Sliced loads are Native-layout
    /// only, refuse a slice for a tensor the load did not name or a name
    /// sliced twice, and never fuse the GDN input projections. An empty
    /// `slices` is exactly [`Self::load`].
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "staged for the two-device start, which loads one slice set per device"
        )
    )]
    pub(crate) fn load_sliced(
        &self,
        weights: &GgufWeights,
        names: &[&str],
        options: LoadOptions<'_>,
        slices: &[(&str, SliceSpec)],
    ) -> Result<CudaWeights, EngError> {
        load_inner(&self.knobs(), weights, names, options, slices, 0)
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

    /// Uploads the fused GDN input projection: the `qkv` tensor's bytes followed by its three
    /// siblings', as ONE native matrix of `n_rows = sum` under the `qkv` name. Refuses unless
    /// all four parts share the quant type and input width, and is only called on the Native
    /// layout for a full four-name group (`gdn_inproj_groups`). The device holds exactly the
    /// same bytes the four separate uploads would hold - VRAM delta zero - and every output row
    /// keeps its dot, so the fused launch is bitwise per segment (pinned by an A/B run).
    fn upload_fused_inproj(
        &self,
        weights: &GgufWeights,
        qkv: &GgufTensor,
        sibs: [&GgufTensor; 3],
        layout: WeightLayout,
    ) -> Result<CudaWeight, EngError> {
        if layout != WeightLayout::Native {
            return Err(EngError::Checkpoint(format!(
                "tensor {}: the GDN input-projection fusion is native-only",
                qkv.name()
            )));
        }
        let parts = [qkv, sibs[0], sibs[1], sibs[2]];
        let (mut rows, k) = matrix_shape(qkv)?;
        for part in &parts[1..] {
            let (part_rows, part_k) = matrix_shape(part)?;
            if part_k != k || part.ggml_type() != qkv.ggml_type() {
                return Err(EngError::Checkpoint(format!(
                    "tensor {}: the GDN input-projection fusion needs {} siblings of the same \
                     type and input width, found {} of {:?}",
                    qkv.name(),
                    k,
                    part.name(),
                    part.ggml_type()
                )));
            }
            rows += part_rows;
        }
        let sources: Vec<(&str, u64)> = parts
            .iter()
            .map(|part| (part.name(), part.byte_len()))
            .collect();
        let total: u64 = sources.iter().map(|&(_, len)| len).sum();
        let quant = QuantTensor {
            name: qkv.name(),
            ty: qkv.ggml_type(),
            n_rows: rows,
            k,
            source_len: usize::try_from(total).map_err(|_| {
                EngError::MemoryPlan(format!(
                    "tensor {}: the fused input projection does not fit",
                    qkv.name()
                ))
            })?,
        };
        let weight = self
            .context
            .upload_native(quant, self.chunk_limit, &mut |at, buf| {
                // One chunk can span several sources: walk to the source holding `at`, read
                // its tail, then every later source from its start (offset 0).
                let at64 = at as u64;
                let mut filled = 0;
                let mut base = 0u64;
                let mut started = false;
                for &(name, len) in &sources {
                    let offset = if started {
                        0
                    } else {
                        if at64 >= base + len {
                            base += len;
                            continue;
                        }
                        at64 - base
                    };
                    // take is bounded by the REMAINING destination, not buf.len(): a chunk
                    // that crosses into a later source must not overrun the slice (F2).
                    let take = (buf.len() - filled).min((len - offset) as usize);
                    weights.read_range(name, offset, &mut buf[filled..filled + take])?;
                    filled += take;
                    if filled == buf.len() {
                        return Ok(());
                    }
                    started = true;
                    base += len;
                }
                Err(EngError::BadRequest(format!(
                    "tensor {}: the fused input-projection read ran past its sources",
                    qkv.name()
                )))
            })?;
        Ok(CudaWeight::Native {
            name: qkv.name().to_owned(),
            weight,
        })
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
    /// `upload_native` (no repack, no scan), with the stack depth of a rank-3 tensor riding
    /// the weight.
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
                // The file's own blocks, no repack and no scale scan.
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
            WeightLayout::Exl3 => Err(EngError::Checkpoint(format!(
                "tensor {name}: GGUF blocks cannot be uploaded as EXL3"
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
        Err(EngError::BadRequest("cuda rows is not implemented".into()))
    }
}

/// Returns the tiles of a weight that the projection `operation` executes. A raw tensor
/// fails before any native call with a `Checkpoint` naming the operation, and the backend
/// stays usable. A native weight fails the same way: the host projection reads tiles and
/// never reinterprets the other layout.
fn tiled<'w>(weight: &'w CudaWeight, operation: &str) -> Result<&'w TiledWeight, EngError> {
    match weight {
        CudaWeight::Tiled { weight, .. } => Ok(weight),
        CudaWeight::Exl3 { name, .. } | CudaWeight::Exl3Bank { name, .. } => {
            Err(EngError::Checkpoint(format!(
                "tensor {name} is EXL3; cuda {operation} reads tiles"
            )))
        }
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

/// The opt-in GDN input-projection fusion (default OFF). The four Q8_0 siblings that read one
/// q8_1 row (`attn_qkv` | `attn_gate` | `ssm_beta` | `ssm_alpha`, in DeltaProj column order)
/// upload as ONE row-concatenated matrix and launch once; an exclusive A/B run
/// (`matvec_q_gdn_sibling_fusion_ab_bench`) showed the fused launch bitwise-identical per
/// segment and ~20% faster on the sibling set. Read once per load and per step construction,
/// like the `INFERENCE_CUDA_ATTN_Q8_MMA` precedent.
pub(crate) fn gdn_fuse_inproj() -> bool {
    std::env::var("INFERENCE_CUDA_GDN_FUSE_INPROJ")
        .ok()
        .as_deref()
        == Some("1")
}

/// The opt-in fused MoE gate+up+SwiGLU matvec (default OFF). When set, a Native qwen36 load's
/// expert gate and up matvecs launch as ONE `matvec_expert` call whose kernel applies
/// up*silu(gate) per element (`glu_gate`), and the expert swiglu then only quantizes that
/// result (`glu_done`). Read per step construction, like the `INFERENCE_CUDA_GDN_FUSE_INPROJ`
/// precedent.
pub(crate) fn moe_glu_fuse() -> bool {
    std::env::var("INFERENCE_CUDA_MOE_GLU_FUSE").ok().as_deref() == Some("1")
}

/// The decode-only tiled-head hybrid: tensor names (comma-separated) that a `Native` load
/// uploads in the `Tiled` layout instead, so the qwen36 step runs the vendored tiled head
/// matmul beside its native ops. Read per load and per `CudaWeights::layout()`, like the
/// `INFERENCE_CUDA_GDN_FUSE_INPROJ` precedent; empty or unset = no override (default).
pub(crate) fn tiled_head_overrides() -> Vec<String> {
    std::env::var("INFERENCE_CUDA_TILED_HEAD")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect()
}

/// The tiled-head override admission: every override name must be part of this load (a typo
/// must not silently load the head native), inert on the other layouts. Pure so the CPU tests
/// pin the refusal without a device load.
pub(crate) fn tiled_override_admission(
    layout: WeightLayout,
    names: &[&str],
    overrides: &[String],
) -> Result<Vec<String>, EngError> {
    if layout != WeightLayout::Native {
        return Ok(Vec::new());
    }
    for name in overrides {
        if !names.contains(&name.as_str()) {
            return Err(EngError::Checkpoint(format!(
                "tensor {name}: the INFERENCE_CUDA_TILED_HEAD override is not part of this load"
            )));
        }
    }
    Ok(overrides.to_vec())
}

/// A name that is both sliced and overridden to tiles refuses: a sliced tensor plans rows from
/// the weight's own shape, and only the native readers take those shapes. Pure so the CPU tests
/// pin the refusal.
pub(crate) fn check_sliced_override(
    name: &str,
    tensor_layout: WeightLayout,
    sliced: bool,
) -> Result<(), EngError> {
    if sliced && tensor_layout == WeightLayout::Tiled {
        return Err(EngError::Checkpoint(format!(
            "tensor {name}: a sliced load cannot override a name to tiles"
        )));
    }
    Ok(())
}

/// The sibling suffixes of an `attn_qkv` tensor, in DeltaProj column order.
const GDN_INPROJ_SIBLINGS: [&str; 3] =
    [".attn_gate.weight", ".ssm_beta.weight", ".ssm_alpha.weight"];

/// The fused groups of a native load: for every `*.attn_qkv.weight` name whose three siblings
/// are also loaded, `(qkv position, [sibling positions in column order])`. A `qkv` without all
/// three siblings is left alone (a partial group never fuses).
fn gdn_inproj_groups(names: &[&str]) -> Vec<(usize, [usize; 3])> {
    let mut groups = Vec::new();
    for (qkv, name) in names.iter().enumerate() {
        let Some(prefix) = name.strip_suffix(".attn_qkv.weight") else {
            continue;
        };
        let sibs = GDN_INPROJ_SIBLINGS.map(|suffix| {
            names.iter().position(|other| {
                other
                    .strip_prefix(prefix)
                    .is_some_and(|rest| rest == suffix)
            })
        });
        if let [Some(a), Some(b), Some(c)] = sibs {
            groups.push((qkv, [a, b, c]));
        }
    }
    groups
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

/// Looks up a tensor and returns its device bytes after upload, counted for `layout` (the
/// tiles and the native blocks differ only for Q6_K). A tiled rank-3 stack whose experts do
/// not start on 8-row tile groups fails here, before any upload; a native stack has no such
/// rule (the MMVQ expert kernel reads the device-side id).
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
        WeightLayout::Exl3 => {
            return Err(EngError::Checkpoint(format!(
                "tensor {name}: EXL3 admission requires its stored components"
            )));
        }
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
    tensors: &[&PlannedTensor<'_>],
    chunk_limit: usize,
) -> Result<(), EngError> {
    for planned_tensor in tensors {
        let tensor = planned_tensor.tensor;
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

pub(crate) mod exl3;
pub(crate) use exl3::{Exl3JobsIo, Exl3MatrixView};
pub(crate) mod exl3_jobs;
mod load_policy;
#[cfg(test)]
mod model_tests;

/// Select whole eligible slices from the end of the load until the requested
/// minimum is met. The only possible overshoot is one complete expert slice.
fn select_host_experts(
    candidates: &[(bool, u64)],
    minimum: u64,
) -> Result<(Vec<bool>, u64), EngError> {
    let mut selected = vec![false; candidates.len()];
    let mut bytes = 0u64;
    for (index, &(eligible, size)) in candidates.iter().enumerate().rev() {
        if bytes >= minimum {
            break;
        }
        if eligible && size != 0 {
            bytes = bytes.checked_add(size).ok_or_else(|| {
                EngError::MemoryPlan("cuda mapped experts: selected bytes overflow".into())
            })?;
            selected[index] = true;
        }
    }
    if bytes < minimum {
        return Err(EngError::MemoryPlan(format!(
            "cuda mapped experts: requested {minimum} bytes, but only {bytes} eligible bytes exist"
        )));
    }
    Ok((selected, bytes))
}

fn load_inner(
    knobs: &LoadKnobs<'_>,
    weights: &GgufWeights,
    names: &[&str],
    options: LoadOptions<'_>,
    slices: &[(&str, SliceSpec)],
    host_expert_bytes: u64,
) -> Result<CudaWeights, EngError> {
    knobs.context.ensure_usable()?;
    if options.layout == WeightLayout::Exl3 {
        return Err(EngError::Checkpoint(
            "EXL3 weights require selected EXL3 components, not GGUF blocks".into(),
        ));
    }
    load_index(names)?; // the duplicate-name refusal still covers the whole request
    let layout = options.layout;
    // The tiled-head hybrid (`INFERENCE_CUDA_TILED_HEAD`): tensor names that upload in the
    // Tiled layout beside this native load. On any other layout the override is inert (the
    // tensors already upload as tiles under Tiled, and EXL3 never reaches here); an override
    // that names no tensor of this load refuses - a typo must not silently load the head
    // native.
    let tiled_overrides: Vec<String> =
        tiled_override_admission(layout, names, &tiled_head_overrides())?;
    // The slice set of a sliced load, by tensor name. A slice for a tensor this load
    // did not name, a name sliced twice, or any slice on a non-native layout refuses
    // before planning: a sliced tensor has a different shape than its file record,
    // and only the native readers take shapes from the weight itself.
    let mut slice_specs: HashMap<&str, &SliceSpec> = HashMap::with_capacity(slices.len());
    for (name, spec) in slices {
        if slice_specs.insert(name, spec).is_some() {
            return Err(EngError::Checkpoint(format!(
                "tensor {name} is sliced twice in one load"
            )));
        }
        if !names.contains(name) {
            return Err(EngError::Checkpoint(format!(
                "tensor {name} is sliced but not part of this load"
            )));
        }
    }
    if !slice_specs.is_empty() && layout != WeightLayout::Native {
        return Err(EngError::Checkpoint(
            "a sliced load needs the Native layout".into(),
        ));
    }
    // The fused GDN input projections (default OFF): the four siblings keep their admission
    // bytes (the fused matrix is the same bytes) but upload as ONE tensor under the qkv
    // name, and the sibling index entries vanish. The skipped siblings also COMPACT the
    // uploaded vector, so the surviving entries renumber to their upload positions - a
    // later tensor must not keep its pre-skip slot (review F1). A sliced load never
    // fuses: the siblings slice by different head ranges than the fused matrix has.
    let fused = if layout == WeightLayout::Native && slice_specs.is_empty() && gdn_fuse_inproj() {
        gdn_inproj_groups(names)
    } else {
        Vec::new()
    };
    let skip: HashSet<usize> = fused.iter().flat_map(|(_, s)| s.iter().copied()).collect();
    let mut index = HashMap::with_capacity(names.len() - skip.len());
    for (slot, &name) in names.iter().enumerate() {
        if !skip.contains(&slot) {
            index.insert(name.to_owned(), index.len());
        }
    }
    let mut tensors = names
        .iter()
        .map(|name| {
            // A hybrid-overridden name plans (admission bytes, tile-row stacks) as tiles.
            let tensor_layout = if tiled_overrides.iter().any(|o| o == name) {
                WeightLayout::Tiled
            } else {
                layout
            };
            let (tensor, whole_bytes) = planned(weights, name, tensor_layout)?;
            check_sliced_override(name, tensor_layout, slice_specs.contains_key(name))?;
            match slice_specs.get(name) {
                None => Ok(PlannedTensor {
                    tensor,
                    bytes: whole_bytes,
                    slice: None,
                    mapped: false,
                }),
                Some(spec) => {
                    let slice = plan::plan_slice_rows(
                        name,
                        tensor.ggml_type(),
                        tensor.dims(),
                        spec.cols.clone(),
                        &spec.rows,
                    )?;
                    let bytes = slice.bytes;
                    Ok(PlannedTensor {
                        tensor,
                        bytes,
                        slice: Some(slice),
                        mapped: false,
                    })
                }
            }
        })
        .collect::<Result<Vec<PlannedTensor<'_>>, EngError>>()?;
    let candidates: Vec<_> = tensors
        .iter()
        .map(|t| {
            let name = t.tensor.name();
            let eligible = layout == WeightLayout::Native
                && t.slice.is_some()
                && t.tensor.dims().len() == 3
                && [
                    "ffn_gate_exps.weight",
                    "ffn_up_exps.weight",
                    "ffn_down_exps.weight",
                ]
                .iter()
                .any(|suffix| name.ends_with(suffix))
                && matches!(
                    t.tensor.ggml_type(),
                    GgufType::Q4K | GgufType::Q6K | GgufType::Q80 | GgufType::Q50
                );
            (eligible, t.bytes)
        })
        .collect();
    let (mapped, host_bytes) = select_host_experts(&candidates, host_expert_bytes)?;
    for (tensor, mapped) in tensors.iter_mut().zip(mapped) {
        tensor.mapped = mapped;
    }
    if host_bytes != 0 {
        let available = crate::capability::probe::probe_host()
            .mem_available_bytes
            .as_ref()
            .copied()
            .ok_or_else(|| {
                EngError::MemoryPlan(
                    "cuda mapped experts: host memory availability is unknown".into(),
                )
            })?;
        let needed = host_bytes.checked_add(1 << 30).ok_or_else(|| {
            EngError::MemoryPlan("cuda mapped experts: host memory demand overflows".into())
        })?;
        if needed > available {
            return Err(EngError::MemoryPlan(format!(
                "cuda mapped experts: need {needed} host bytes including 1 GiB free, have {available}"
            )));
        }
        eprintln!(
            "cuda mapped experts: placing {host_bytes} bytes in host memory (host available {available} bytes)"
        );
    }
    let total = tensors
        .iter()
        .try_fold(0u64, |sum, planned_tensor| {
            sum.checked_add(if planned_tensor.mapped {
                0
            } else {
                planned_tensor.bytes
            })
        })
        .ok_or_else(|| EngError::MemoryPlan("cuda admission: tensors overflow u64".into()))?;
    let policy = LoadPolicy::for_device(knobs.integrated);
    // Before admission: the free figure of an integrated GPU counts cache as
    // free, so evicting first lets the allocator find that memory.
    if policy.evict_file {
        evict_model(weights.file())?;
    }
    admit(knobs, total, options.reserve)?;
    let drop_file = policy.drop_file(weights.file(), options.drop_page_cache);
    transfer(
        knobs,
        weights,
        LoadPlan {
            tensors,
            index,
            policy,
            drop_file,
            layout,
            tiled_overrides,
            fused,
        },
    )
}

/// Scans, then uploads the planned tensors. Tests call it to reach the upload
/// loop with a tensor that admission would refuse.
///
/// Ownership: a tensor joins `loaded` only after its upload returned `Ok`. Any `?` drops
/// `loaded`, which frees its tensors in reverse load order. A failed attempt owns
/// nothing and leaves the live byte count unchanged, so a retry starts from the same
/// state. No retry runs after a terminal latch, because the attempt then returns
/// `Backend` `execute`, which is never retried.
fn transfer(
    knobs: &LoadKnobs<'_>,
    weights: &GgufWeights,
    plan: LoadPlan<'_>,
) -> Result<CudaWeights, EngError> {
    let LoadPlan {
        tensors,
        index,
        policy,
        drop_file,
        layout,
        tiled_overrides,
        fused,
    } = plan;
    let scan_began = Instant::now();
    // The fp16 scale scans guard the tile dequantisation (fp16(d * scale)); a native load
    // keeps the file's exact scales and scans nothing - except the hybrid's overridden
    // names, which dequantise through the same fp16 tile path.
    if layout == WeightLayout::Tiled {
        let scanned: Vec<&PlannedTensor> = tensors.iter().collect();
        check_scale_ranges(weights, &scanned, knobs.chunk_limit)?;
    } else if !tiled_overrides.is_empty() {
        let overridden: Vec<&PlannedTensor> = tensors
            .iter()
            .filter(|t| tiled_overrides.iter().any(|o| o == t.tensor.name()))
            .collect();
        check_scale_ranges(weights, &overridden, knobs.chunk_limit)?;
    }
    let scan = scan_began.elapsed();
    // A tiled load's scan read every Q6_K, Q8_0 and Q5_0 tensor once and refilled the
    // cache; a native load ran no scan.
    if policy.evict_file {
        evict_model(weights.file())?;
    }
    let mut loaded = CudaWeights {
        weights: Vec::with_capacity(tensors.len()),
        index,
        times: LoadTimes::default(),
    };
    let upload_began = Instant::now();
    let mut fused_at: HashMap<usize, [usize; 3]> = fused.into_iter().collect();
    let mut skip: HashSet<usize> = fused_at.values().flatten().copied().collect();
    for (position, planned_tensor) in tensors.iter().enumerate() {
        let tensor = planned_tensor.tensor;
        if skip.remove(&position) {
            continue;
        }
        let weight = if let Some(sibs) = fused_at.remove(&position) {
            let sib_positions = sibs;
            let sibs = sib_positions.map(|slot| tensors[slot].tensor);
            let weight = retry_on_oom(
                &policy,
                knobs.oom_retries,
                || upload_fused_inproj(knobs, weights, tensor, sibs, layout),
                || recover(weights.file(), policy.retry_pause),
            )?;
            // The fused upload read every sibling's range, so every sibling's pages are
            // evictable exactly like the pushed tensor's own range (review F5).
            if let Some(file) = drop_file {
                for &slot in &sib_positions {
                    let sib = tensors[slot].tensor;
                    cuda::drop_page_cache(file, sib.abs_offset(), sib.byte_len())?;
                }
            }
            weight
        } else {
            retry_on_oom(
                &policy,
                knobs.oom_retries,
                || {
                    upload(
                        knobs,
                        weights,
                        tensor,
                        // A hybrid-overridden name uploads as tiles beside the native load.
                        if tiled_overrides.iter().any(|o| o == tensor.name()) {
                            WeightLayout::Tiled
                        } else {
                            layout
                        },
                        planned_tensor.slice.as_ref(),
                        planned_tensor.mapped,
                    )
                },
                || recover(weights.file(), policy.retry_pause),
            )?
        };
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

pub(crate) fn admit(knobs: &LoadKnobs<'_>, tensors: u64, reserve: Reserve) -> Result<(), EngError> {
    let demand = plan::Demand {
        tensors,
        kv_cache: reserve.kv_cache,
        work: reserve.work,
        headroom: reserve.headroom,
    };
    let free = knobs.context.memory_info()?.free;
    plan::admit(&demand, u64::try_from(free).unwrap_or(u64::MAX)).map(drop)
}

/// Uploads the fused GDN input projection: the `qkv` tensor's bytes followed by its three
/// siblings', as ONE native matrix of `n_rows = sum` under the `qkv` name. Refuses unless
/// all four parts share the quant type and input width, and is only called on the Native
/// layout for a full four-name group (`gdn_inproj_groups`). The device holds exactly the
/// same bytes the four separate uploads would hold - VRAM delta zero - and every output row
/// keeps its dot, so the fused launch is bitwise per segment (pinned by an A/B run).
fn upload_fused_inproj(
    knobs: &LoadKnobs<'_>,
    weights: &GgufWeights,
    qkv: &GgufTensor,
    sibs: [&GgufTensor; 3],
    layout: WeightLayout,
) -> Result<CudaWeight, EngError> {
    if layout != WeightLayout::Native {
        return Err(EngError::Checkpoint(format!(
            "tensor {}: the GDN input-projection fusion is native-only",
            qkv.name()
        )));
    }
    let parts = [qkv, sibs[0], sibs[1], sibs[2]];
    let (mut rows, k) = matrix_shape(qkv)?;
    for part in &parts[1..] {
        let (part_rows, part_k) = matrix_shape(part)?;
        if part_k != k || part.ggml_type() != qkv.ggml_type() {
            return Err(EngError::Checkpoint(format!(
                "tensor {}: the GDN input-projection fusion needs {} siblings of the same \
                 type and input width, found {} of {:?}",
                qkv.name(),
                k,
                part.name(),
                part.ggml_type()
            )));
        }
        rows += part_rows;
    }
    let sources: Vec<(&str, u64)> = parts
        .iter()
        .map(|part| (part.name(), part.byte_len()))
        .collect();
    let total: u64 = sources.iter().map(|&(_, len)| len).sum();
    let quant = QuantTensor {
        name: qkv.name(),
        ty: qkv.ggml_type(),
        n_rows: rows,
        k,
        source_len: usize::try_from(total).map_err(|_| {
            EngError::MemoryPlan(format!(
                "tensor {}: the fused input projection does not fit",
                qkv.name()
            ))
        })?,
    };
    let weight = knobs
        .context
        .upload_native(quant, knobs.chunk_limit, &mut |at, buf| {
            // One chunk can span several sources: walk to the source holding `at`, read
            // its tail, then every later source from its start (offset 0).
            let at64 = at as u64;
            let mut filled = 0;
            let mut base = 0u64;
            let mut started = false;
            for &(name, len) in &sources {
                let offset = if started {
                    0
                } else {
                    if at64 >= base + len {
                        base += len;
                        continue;
                    }
                    at64 - base
                };
                // take is bounded by the REMAINING destination, not buf.len(): a chunk
                // that crosses into a later source must not overrun the slice (F2).
                let take = (buf.len() - filled).min((len - offset) as usize);
                weights.read_range(name, offset, &mut buf[filled..filled + take])?;
                filled += take;
                if filled == buf.len() {
                    return Ok(());
                }
                started = true;
                base += len;
            }
            Err(EngError::BadRequest(format!(
                "tensor {}: the fused input-projection read ran past its sources",
                qkv.name()
            )))
        })?;
    Ok(CudaWeight::Native {
        name: qkv.name().to_owned(),
        weight,
    })
}

fn upload(
    knobs: &LoadKnobs<'_>,
    weights: &GgufWeights,
    tensor: &GgufTensor,
    layout: WeightLayout,
    slice: Option<&plan::TensorSlice>,
    mapped: bool,
) -> Result<CudaWeight, EngError> {
    let name = tensor.name();
    match (tensor.ggml_type(), layout) {
        (ty @ (GgufType::Q4K | GgufType::Q6K | GgufType::Q80 | GgufType::Q50), layout) => {
            upload_quant(knobs, weights, tensor, ty, layout, slice, mapped)
        }
        (ty @ (GgufType::F32 | GgufType::F16), _) => {
            let (_, bytes) = weights.read_tensor(name)?;
            let bytes = match slice {
                Some(slice) => plan::assemble_slice(&bytes, slice)?,
                None => bytes,
            };
            Ok(CudaWeight::Raw {
                name: name.to_owned(),
                ty,
                buffer: knobs.context.upload(&bytes)?,
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
/// `upload_native` (no repack, no scan), with the stack depth of a rank-3 tensor riding
/// the weight.
fn upload_quant(
    knobs: &LoadKnobs<'_>,
    weights: &GgufWeights,
    tensor: &GgufTensor,
    ty: GgufType,
    layout: WeightLayout,
    slice: Option<&plan::TensorSlice>,
    mapped: bool,
) -> Result<CudaWeight, EngError> {
    let name = tensor.name();
    if let Some(slice) = slice {
        return upload_quant_sliced(knobs, weights, tensor, name, ty, slice, mapped);
    }
    let (n_rows, k) = matrix_shape(tensor)?;
    let experts = stack_depth(tensor)?;
    let source_len = usize::try_from(tensor.byte_len())
        .map_err(|_| EngError::MemoryPlan(format!("cuda allocate: tensor {name} does not fit")))?;
    let quant = QuantTensor {
        name,
        ty,
        n_rows,
        k,
        source_len,
    };
    match layout {
        WeightLayout::Tiled => {
            let weight = knobs
                .context
                .upload_tiled(quant, knobs.chunk_limit, &mut |at, buf| {
                    weights.read_range(name, at as u64, buf)
                })?;
            Ok(CudaWeight::Tiled {
                name: name.to_owned(),
                weight,
                experts,
            })
        }
        WeightLayout::Native => {
            // The file's own blocks, no repack and no scale scan.
            let weight = knobs
                .context
                .upload_native(quant, knobs.chunk_limit, &mut |at, buf| {
                    weights.read_range(name, at as u64, buf)
                })?
                .with_experts(experts);
            Ok(CudaWeight::Native {
                name: name.to_owned(),
                weight,
            })
        }
        WeightLayout::Exl3 => Err(EngError::Checkpoint(format!(
            "tensor {name}: GGUF blocks cannot be uploaded as EXL3"
        ))),
    }
}

/// Uploads one slice of a quantized tensor in the native layout: the
/// sliced shape goes through the same `upload_native` chunking as a
/// whole tensor, with a reader that maps assembled offsets to the
/// slice's file segments. The device then holds exactly the bytes a
/// file containing only the sliced tensor would upload.
fn upload_quant_sliced(
    knobs: &LoadKnobs<'_>,
    weights: &GgufWeights,
    tensor: &GgufTensor,
    name: &str,
    ty: GgufType,
    slice: &plan::TensorSlice,
    mapped: bool,
) -> Result<CudaWeight, EngError> {
    let shape_error =
        || EngError::Checkpoint(format!("tensor {name} has an unusable sliced shape"));
    if slice.dims.len() != tensor.dims().len() {
        return Err(shape_error());
    }
    let k = usize::try_from(slice.dims[0]).map_err(|_| shape_error())?;
    let n_rows = slice.dims[1..]
        .iter()
        .try_fold(1u64, |rows, &dim| rows.checked_mul(dim))
        .and_then(|rows| usize::try_from(rows).ok())
        .ok_or_else(shape_error)?;
    let experts = match slice.dims.get(2) {
        Some(&depth) => Some(usize::try_from(depth).map_err(|_| shape_error())?),
        None => None,
    };
    let source_len = usize::try_from(slice.bytes).map_err(|_| {
        EngError::MemoryPlan(format!("cuda allocate: tensor {name} slice does not fit"))
    })?;
    let quant = QuantTensor {
        name,
        ty,
        n_rows,
        k,
        source_len,
    };
    let reader = SliceReader::new(&slice.segments);
    let mut fill = |at, buf: &mut [u8]| {
        reader.fill(at as u64, buf, &mut |offset, part| {
            weights.read_range(name, offset, part)
        })
    };
    let weight = if mapped {
        knobs
            .context
            .upload_native_mapped(quant, knobs.chunk_limit, &mut fill)?
    } else {
        knobs
            .context
            .upload_native(quant, knobs.chunk_limit, &mut fill)?
    }
    .with_experts(experts);
    Ok(CudaWeight::Native {
        name: name.to_owned(),
        weight,
    })
}

#[cfg(test)]
mod tests;
