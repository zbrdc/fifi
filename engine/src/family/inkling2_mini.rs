//! Checkpoint geometry, text memory planning, and packed expert reads for Inkling 2 Mini.

use std::{collections::BTreeMap, fs::File, ops::Range, path::Path};

use safetensors::{Dtype, tensor::TensorInfo};
use serde::Deserialize;

use super::{
    kv_format::{KvFormat, KvRowShape},
    plan_decision::{PlanDecision, PlanFixity},
};
use crate::{EngError, checkpoint_json::UniqueValue, weights::read_stored_header};

pub mod gguf_metadata;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub mod mlx;
mod routes;
pub use routes::InklingRoutes;

pub(crate) const FAMILY: &str = "inkling2_mini";
pub(crate) const MODEL_TYPE: &str = "inkling2_loop";
const SHARD: &str = "model-00001-of-00001.safetensors";
const LAYERS: usize = 24;
const VISITS: usize = 48;
const CORE_START: usize = 6;
const CORE_END: usize = 18;
const PASSES: usize = 3;
const EXPERTS: usize = 128;
const TOP_K: usize = 8;
const ROUTE_SCALE: f32 = 8.0;
const GATE_CAP: f32 = 4.0;
const UP_CAP: f32 = 25.0;
const HIDDEN: usize = 2304;
const INTERMEDIATE: usize = 1152;
const QUERY_HEADS: usize = 24;
const KV_HEADS: u64 = 8;
const HEAD_DIM: u64 = 128;
const WINDOW: u32 = 1024;
const MAX_CONTEXT: u32 = 1_048_576;
const CONV_TAPS: u64 = 4;
const HEADER_LIMIT: u64 = 64 << 20;

#[derive(Deserialize)]
struct ModelConfig {
    model_type: String,
    architectures: Vec<String>,
    text_config: TextConfig,
    quantization_config: QuantizationConfig,
}

#[derive(Deserialize)]
struct TextConfig {
    model_type: String,
    num_hidden_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: u64,
    head_dim: u64,
    swa_num_attention_heads: usize,
    swa_num_key_value_heads: u64,
    swa_head_dim: u64,
    vocab_size: usize,
    unpadded_vocab_size: usize,
    model_max_length: u32,
    local_layer_ids: Vec<usize>,
    sliding_window_size: u32,
    sconv_kernel_size: u64,
    n_layers_per_attn_res_block: usize,
    n_routed_experts: usize,
    num_experts_per_tok: usize,
    n_shared_experts: usize,
    mlp_gate_softcap: f32,
    mlp_up_softcap: f32,
    route_scale: f32,
    use_gate_bias: bool,
    loop_config: LoopConfig,
}

#[derive(Deserialize)]
struct LoopConfig {
    core_start: usize,
    core_end: usize,
    num_passes: usize,
    gate_rank: usize,
    is_varloop: bool,
}

#[derive(Deserialize)]
struct QuantizationConfig {
    quant_method: String,
    quant_algo: String,
}

fn checkpoint_error(detail: impl std::fmt::Display) -> EngError {
    EngError::Checkpoint(format!("Inkling 2 Mini: {detail}"))
}

fn parse_json<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, EngError> {
    let value = serde_json::from_slice::<UniqueValue>(bytes)
        .map_err(checkpoint_error)?
        .0;
    serde_json::from_value(value).map_err(checkpoint_error)
}

fn validate_config(bytes: &[u8]) -> Result<(), EngError> {
    let config: ModelConfig = parse_json(bytes)?;
    let text = config.text_config;
    let loop_config = text.loop_config;
    if config.model_type != MODEL_TYPE
        || config.architectures != ["Inkling2LoopForConditionalGeneration"]
        || text.model_type != "inkling2_loop_text"
        || (
            loop_config.core_start,
            loop_config.core_end,
            loop_config.num_passes,
        ) != (CORE_START, CORE_END, PASSES)
        || loop_config.gate_rank != 64
        || loop_config.is_varloop
    {
        return Err(checkpoint_error("expected the fixed three-pass Mini loop"));
    }
    let local = (0..LAYERS).filter(|layer| is_local(*layer));
    if text.num_hidden_layers != LAYERS
        || text.hidden_size != HIDDEN
        || text.intermediate_size != INTERMEDIATE
        || text.num_attention_heads != QUERY_HEADS
        || text.num_key_value_heads != KV_HEADS
        || text.head_dim != HEAD_DIM
        || text.swa_num_attention_heads != QUERY_HEADS
        || text.swa_num_key_value_heads != KV_HEADS
        || text.swa_head_dim != HEAD_DIM
        || text.vocab_size != 201_024
        || text.unpadded_vocab_size != 200_058
        || text.model_max_length != MAX_CONTEXT
        || !text.local_layer_ids.into_iter().eq(local)
        || text.sliding_window_size != WINDOW
        || text.sconv_kernel_size != CONV_TAPS
        || text.n_layers_per_attn_res_block != 6
        || text.n_routed_experts != EXPERTS
        || text.num_experts_per_tok != TOP_K
        || text.n_shared_experts != 0
        || text.mlp_gate_softcap != GATE_CAP
        || text.mlp_up_softcap != UP_CAP
        || text.route_scale != ROUTE_SCALE
        || !text.use_gate_bias
        || config.quantization_config.quant_method != "modelopt"
        || config.quantization_config.quant_algo != "NVFP4"
    {
        return Err(checkpoint_error(
            "checkpoint geometry differs from the released Mini",
        ));
    }
    Ok(())
}

fn is_local(layer: usize) -> bool {
    layer % 6 != 5
}

/// The weight index can repeat. KV and short-convolution buffers use the unique visit index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LayerVisit {
    weight_layer: usize,
    state_slot: usize,
    pass: Option<usize>,
}

fn visits() -> [LayerVisit; VISITS] {
    let width = CORE_END - CORE_START;
    std::array::from_fn(|state_slot| {
        let (weight_layer, pass) = if state_slot < CORE_START {
            (state_slot, None)
        } else if state_slot < CORE_START + width * PASSES {
            let core_visit = state_slot - CORE_START;
            (CORE_START + core_visit % width, Some(core_visit / width))
        } else {
            (state_slot - width * (PASSES - 1), None)
        };
        LayerVisit {
            weight_layer,
            state_slot,
            pass,
        }
    })
}

#[derive(Deserialize)]
struct WeightIndex {
    metadata: IndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct IndexMetadata {
    total_size: u64,
}

#[derive(Default)]
struct WeightPlan {
    text_resident: u64,
    vision: u64,
    expert_storage: [u64; LAYERS],
}

fn tensor_bytes(tensor: &TensorInfo) -> u64 {
    (tensor.data_offsets.1 - tensor.data_offsets.0) as u64
}

enum ExpertRegion {
    Slices { start: u64, stride: u64 },
    Shared(Range<u64>),
}

struct ProjectionRegions {
    rows: usize,
    cols: usize,
    weight: ExpertRegion,
    scale: ExpertRegion,
    amax: Option<ExpertRegion>,
    denominator: Option<ExpertRegion>,
}

struct RouterRegions {
    weight_offset: u64,
    bias_offset: u64,
}

/// One stored layer's router parameters.
pub struct InklingRouterWeights {
    /// Row-major little-endian BF16 matrix in expert, hidden column order.
    pub weight: Vec<u8>,
    /// Finite F32 routing bias, in expert order.
    pub bias: [f32; EXPERTS],
}

/// A validated Inkling 2 Mini checkpoint and its router and packed expert tensor locations.
pub struct InklingExpertSource {
    shard: File,
    len: u64,
    layers: Vec<[ProjectionRegions; 2]>,
    routers: Vec<RouterRegions>,
    weights: WeightPlan,
}

/// Byte ranges in caller staging for one packed projection.
#[derive(Debug, PartialEq, Eq)]
pub struct InklingExpertProjection {
    /// Logical dimensions in compact expert, output row, input column order.
    pub shape: [usize; 3],
    /// Raw NVFP4 bytes or MXFP8 E4M3 bytes.
    pub weight: Range<usize>,
    /// Raw E4M3 scales for NVFP4 or E8M0 scales for MXFP8.
    pub scale: Range<usize>,
    /// Per-expert little-endian F32 values for NVFP4.
    pub amax: Option<Range<usize>>,
    /// One shared little-endian F32 value for NVFP4.
    pub denominator: Option<Range<usize>>,
}

struct ExpertRead {
    offset: u64,
    destination: Range<usize>,
}

/// A compact expert selection that can be read into caller-owned staging.
pub struct InklingExpertPlan<'a> {
    source: &'a InklingExpertSource,
    compact_slots: Vec<u32>,
    gate_up: InklingExpertProjection,
    down: InklingExpertProjection,
    reads: Vec<ExpertRead>,
    read_bytes: usize,
    staging_bytes: usize,
    alignment: usize,
}

impl InklingExpertSource {
    /// Opens and validates the config, index and shard header without reading tensor payloads.
    pub fn open(dir: &Path) -> Result<Self, EngError> {
        let guard = crate::storage_guard::LocalGuard::new()?;
        let config_path = dir.join("config.json");
        let index_path = dir.join("model.safetensors.index.json");
        let shard_path = dir.join(SHARD);
        for path in [&config_path, &index_path, &shard_path] {
            guard.check(path, "model weights")?;
        }
        validate_config(&std::fs::read(config_path).map_err(checkpoint_error)?)?;
        let index = parse_json(&std::fs::read(index_path).map_err(checkpoint_error)?)?;
        let mut shard = File::open(shard_path).map_err(checkpoint_error)?;
        let len = shard.metadata().map_err(checkpoint_error)?.len();
        let (data_start, entries) = read_stored_header(&mut shard, len, HEADER_LIMIT)?;
        let entries = entries.into_iter().collect();
        let weights = weight_plan(&index, &entries)?;
        let layers = (0..LAYERS)
            .map(|layer| expert_regions(&entries, data_start, layer))
            .collect::<Result<_, _>>()?;
        let routers = (0..LAYERS)
            .map(|layer| router_regions(&entries, data_start, layer))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            shard,
            len,
            layers,
            routers,
            weights,
        })
    }

    /// Reads the BF16 router matrix and F32 bias for a stored weight layer.
    pub fn read_router(&self, layer: usize) -> Result<InklingRouterWeights, EngError> {
        let router = self.routers.get(layer).ok_or_else(|| {
            EngError::BadRequest(format!("Inkling router layer {layer} is out of range"))
        })?;
        self.check_shard_len()?;
        let mut weight = vec![0; EXPERTS * HIDDEN * size_of::<u16>()];
        let mut bias_bytes = [[0; size_of::<f32>()]; EXPERTS];
        for (name, bytes, offset) in [
            ("weight", weight.as_mut_slice(), router.weight_offset),
            ("bias", bias_bytes.as_flattened_mut(), router.bias_offset),
        ] {
            read_exact_at(&self.shard, bytes, offset).map_err(|error| {
                checkpoint_error(format!(
                    "layer {layer} router {name} read at {offset}: {error}"
                ))
            })?;
        }
        self.check_shard_len()?;
        let bias = bias_bytes.map(f32::from_le_bytes);
        if let Some(expert) = bias.iter().position(|value| !value.is_finite()) {
            return Err(checkpoint_error(format!(
                "layer {layer} router bias for expert {expert} is not finite"
            )));
        }
        Ok(InklingRouterWeights { weight, bias })
    }

    fn check_shard_len(&self) -> Result<(), EngError> {
        let observed = self.shard.metadata().map_err(checkpoint_error)?.len();
        if observed != self.len {
            return Err(checkpoint_error(format!(
                "shard length changed from {} to {observed}",
                self.len
            )));
        }
        Ok(())
    }

    /// Plans selected expert reads for a stored layer. Each tensor begins at a
    /// multiple of the power-of-two alignment in the caller's staging slice.
    pub fn plan(
        &self,
        layer: usize,
        experts: &[usize],
        alignment: usize,
    ) -> Result<InklingExpertPlan<'_>, EngError> {
        let [gate_up, down] = self.layers.get(layer).ok_or_else(|| {
            EngError::BadRequest(format!("Inkling expert layer {layer} is out of range"))
        })?;
        if !alignment.is_power_of_two() {
            return Err(EngError::BadRequest(
                "Inkling expert staging alignment must be a power of two".into(),
            ));
        }
        if experts.is_empty() || experts.iter().any(|&expert| expert >= EXPERTS) {
            return Err(EngError::BadRequest(
                "Inkling expert selection is empty or out of range".into(),
            ));
        }
        let mut selected = [false; EXPERTS];
        for &expert in experts {
            selected[expert] = true;
        }
        let mut slots = [0; EXPERTS];
        let mut sorted = Vec::new();
        for (expert, selected) in selected.into_iter().enumerate() {
            if selected {
                slots[expert] = sorted.len() as u32;
                sorted.push(expert);
            }
        }
        let compact_slots = experts.iter().map(|&expert| slots[expert]).collect();
        let mut reads = Vec::new();
        let mut staging_bytes = 0usize;
        let mut plan_region = |region: &ExpertRegion| -> Result<Range<usize>, EngError> {
            let start = staging_bytes
                .checked_next_multiple_of(alignment)
                .ok_or_else(staging_overflow)?;
            staging_bytes = start;
            let mut push = |source: Range<u64>| -> Result<(), EngError> {
                let bytes =
                    usize::try_from(source.end - source.start).map_err(|_| staging_overflow())?;
                let end = staging_bytes
                    .checked_add(bytes)
                    .ok_or_else(staging_overflow)?;
                reads.push(ExpertRead {
                    offset: source.start,
                    destination: staging_bytes..end,
                });
                staging_bytes = end;
                Ok(())
            };
            match region {
                ExpertRegion::Slices { start, stride } => {
                    for run in sorted.chunk_by(|&left, &right| left + 1 == right) {
                        let first = run[0] as u64;
                        let end = first + run.len() as u64;
                        push((start + first * stride)..(start + end * stride))?;
                    }
                }
                ExpertRegion::Shared(range) => push(range.clone())?,
            }
            Ok(start..staging_bytes)
        };
        let mut projection = |regions: &ProjectionRegions| -> Result<_, EngError> {
            Ok(InklingExpertProjection {
                shape: [sorted.len(), regions.rows, regions.cols],
                weight: plan_region(&regions.weight)?,
                scale: plan_region(&regions.scale)?,
                amax: regions.amax.as_ref().map(&mut plan_region).transpose()?,
                denominator: regions
                    .denominator
                    .as_ref()
                    .map(&mut plan_region)
                    .transpose()?,
            })
        };
        let gate_up = projection(gate_up)?;
        let down = projection(down)?;
        let read_bytes = reads.iter().map(|read| read.destination.len()).sum();
        Ok(InklingExpertPlan {
            source: self,
            compact_slots,
            gate_up,
            down,
            reads,
            read_bytes,
            staging_bytes,
            alignment,
        })
    }
}

fn staging_overflow() -> EngError {
    EngError::MemoryPlan("Inkling expert staging size exceeds host limits".into())
}

impl InklingExpertPlan<'_> {
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Minimum staging length, including alignment gaps.
    pub fn staging_bytes(&self) -> usize {
        self.staging_bytes
    }

    /// Total payload bytes transferred from the shard.
    pub fn read_bytes(&self) -> usize {
        self.read_bytes
    }

    pub fn read_count(&self) -> usize {
        self.reads.len()
    }

    /// One compact slot per requested expert, in request order including duplicates.
    pub fn compact_slots(&self) -> &[u32] {
        &self.compact_slots
    }

    /// Packed gate and up projection (w13).
    pub fn gate_up(&self) -> &InklingExpertProjection {
        &self.gate_up
    }

    /// Packed down projection (w2).
    pub fn down(&self) -> &InklingExpertProjection {
        &self.down
    }

    /// Reads packed bytes directly into staging, leaving alignment gaps untouched.
    /// An undersized slice is rejected before I/O. A failed read may partially
    /// overwrite staging, which the caller must discard.
    pub fn read_into(&self, scratch: &mut [u8]) -> Result<(), EngError> {
        if scratch.len() < self.staging_bytes {
            return Err(EngError::BadRequest(format!(
                "Inkling expert staging needs {} bytes, received {}",
                self.staging_bytes,
                scratch.len()
            )));
        }
        self.source.check_shard_len()?;
        for read in &self.reads {
            read_exact_at(
                &self.source.shard,
                &mut scratch[read.destination.clone()],
                read.offset,
            )
            .map_err(|error| {
                checkpoint_error(format!("expert read at {}: {error}", read.offset))
            })?;
        }
        self.source.check_shard_len()
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, dst: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(dst, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut dst: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::{
        io::{Error, ErrorKind},
        os::windows::fs::FileExt,
    };

    while !dst.is_empty() {
        match file.seek_read(dst, offset) {
            Ok(0) => return Err(Error::from(ErrorKind::UnexpectedEof)),
            Ok(bytes) => {
                offset += bytes as u64;
                dst = &mut dst[bytes..];
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn router_regions(
    entries: &BTreeMap<String, TensorInfo>,
    data_start: u64,
    layer: usize,
) -> Result<RouterRegions, EngError> {
    let stem = format!("model.llm.layers.{layer}.mlp.gate");
    let offset = |name: &str, dtype: Dtype, shape: &[usize]| {
        let key = format!("{stem}.{name}");
        let info = entries
            .get(&key)
            .ok_or_else(|| checkpoint_error(format!("missing {key}")))?;
        if info.dtype != dtype || info.shape != shape {
            return Err(checkpoint_error(format!(
                "unexpected storage layout for {key}"
            )));
        }
        Ok(data_start + info.data_offsets.0 as u64)
    };
    Ok(RouterRegions {
        weight_offset: offset("weight", Dtype::BF16, &[EXPERTS, HIDDEN])?,
        bias_offset: offset("bias", Dtype::F32, &[EXPERTS])?,
    })
}

fn expert_regions(
    entries: &BTreeMap<String, TensorInfo>,
    data_start: u64,
    layer: usize,
) -> Result<[ProjectionRegions; 2], EngError> {
    let nvfp4 = layer < 20;
    let group = if nvfp4 { 16 } else { 32 };
    let projection = |name, rows, cols| -> Result<ProjectionRegions, EngError> {
        let stem = format!("model.llm.layers.{layer}.mlp.experts.{name}_weight");
        let get = |suffix: &str,
                   dtype: Dtype,
                   shape: &[usize],
                   per_expert: bool|
         -> Result<_, EngError> {
            let key = format!("{stem}{suffix}");
            let info = entries
                .get(&key)
                .ok_or_else(|| checkpoint_error(format!("missing {key}")))?;
            if info.dtype != dtype || info.shape != shape {
                return Err(checkpoint_error(format!(
                    "unexpected storage layout for {key}"
                )));
            }
            let start = data_start + info.data_offsets.0 as u64;
            let stored = tensor_bytes(info);
            Ok(if per_expert {
                ExpertRegion::Slices {
                    start,
                    stride: stored / EXPERTS as u64,
                }
            } else {
                ExpertRegion::Shared(start..start + stored)
            })
        };
        let (weight_dtype, packed_cols, scale_dtype) = if nvfp4 {
            (Dtype::U8, cols / 2, Dtype::F8_E4M3)
        } else {
            (Dtype::F8_E4M3, cols, Dtype::U8)
        };
        Ok(ProjectionRegions {
            rows,
            cols,
            weight: get("", weight_dtype, &[EXPERTS, rows, packed_cols], true)?,
            scale: get("_scale", scale_dtype, &[EXPERTS, rows, cols / group], true)?,
            amax: nvfp4
                .then(|| get("_amax", Dtype::F32, &[EXPERTS], true))
                .transpose()?,
            denominator: nvfp4
                .then(|| get("_outer_scale_denominator", Dtype::F32, &[1], false))
                .transpose()?,
        })
    };
    Ok([
        projection("w13", INTERMEDIATE * 2, HIDDEN)?,
        projection("w2", HIDDEN, INTERMEDIATE)?,
    ])
}

fn weight_plan(
    index: &WeightIndex,
    entries: &BTreeMap<String, TensorInfo>,
) -> Result<WeightPlan, EngError> {
    if index.weight_map.len() != entries.len()
        || index
            .weight_map
            .iter()
            .any(|(key, file)| file != SHARD || !entries.contains_key(key))
        || index.metadata.total_size != entries.values().map(tensor_bytes).sum::<u64>()
    {
        return Err(checkpoint_error("index and shard inventory disagree"));
    }
    let mut plan = WeightPlan::default();
    let mut expert_tensors = 0;
    for (key, tensor) in entries {
        let bytes = tensor_bytes(tensor);
        if key.starts_with("model.visual.") {
            plan.vision += bytes;
        } else if key.starts_with("model.llm.") {
            if key.contains(".mlp.experts.") {
                let layer = key
                    .strip_prefix("model.llm.layers.")
                    .and_then(|name| name.split_once(".mlp.experts."))
                    .and_then(|(layer, _)| layer.parse::<usize>().ok())
                    .filter(|&layer| layer < LAYERS)
                    .ok_or_else(|| checkpoint_error(format!("unknown expert layer in {key}")))?;
                plan.expert_storage[layer] += bytes;
                expert_tensors += 1;
            } else {
                plan.text_resident += bytes;
            }
        } else {
            return Err(checkpoint_error(format!(
                "unknown tensor namespace in {key}"
            )));
        }
    }
    if expert_tensors != 20 * 8 + 4 * 4 {
        return Err(checkpoint_error("unexpected number of expert tensors"));
    }
    Ok(plan)
}

#[derive(Debug, PartialEq, Eq)]
struct StatePlan {
    kv: u64,
    convolution: u64,
    local: usize,
    full: usize,
}

fn state_plan(context: u32) -> Result<StatePlan, EngError> {
    if !(2..=MAX_CONTEXT).contains(&context) {
        return Err(EngError::BadRequest(format!(
            "Inkling 2 Mini context must be between 2 and {MAX_CONTEXT} tokens"
        )));
    }
    let local = visits()
        .iter()
        .filter(|visit| is_local(visit.weight_layer))
        .count();
    let full = VISITS - local;
    let row = KvRowShape {
        heads_kv: KV_HEADS,
        head_dim: HEAD_DIM,
    };
    let layer_bytes = KvFormat::Bf16.layer_position_bytes(row)?;
    let kv = layer_bytes
        * (local as u64 * u64::from(context.min(WINDOW)) + full as u64 * u64::from(context));
    // The K/V convolutions each keep a KV plane's channels. The attention and MLP
    // output convolutions each keep hidden channels. All four retain taps - 1 rows.
    let channels = 2 * (KV_HEADS * HEAD_DIM + HIDDEN as u64);
    let convolution = VISITS as u64 * channels * (CONV_TAPS - 1) * 2;
    Ok(StatePlan {
        kv,
        convolution,
        local,
        full,
    })
}

pub(crate) fn inspect(dir: &Path, context: Option<u32>) -> Result<Vec<PlanDecision>, EngError> {
    let source = InklingExpertSource::open(dir)?;
    let mut routed_per_layer = [0; LAYERS];
    let mut coalesced_reads = [0; LAYERS];
    let mut scattered_reads = [0; LAYERS];
    let adjacent: [usize; TOP_K] = std::array::from_fn(|i| i);
    let spread: [usize; TOP_K] = std::array::from_fn(|i| 2 * i);
    for layer in 0..LAYERS {
        let grouped = source.plan(layer, &adjacent, 1)?;
        routed_per_layer[layer] = grouped.read_bytes() as u64;
        coalesced_reads[layer] = grouped.read_count();
        scattered_reads[layer] = source.plan(layer, &spread, 1)?.read_count();
    }
    let weights = &source.weights;
    let context = context.unwrap_or(MAX_CONTEXT);
    let state = state_plan(context)?;
    let routed_peak = routed_per_layer.into_iter().max().expect("24 layers");
    let schedule = visits();
    let state_slots = schedule[VISITS - 1].state_slot + 1;
    let core_visits = schedule.iter().filter(|visit| visit.pass.is_some()).count();
    let routed_uncached: u64 = schedule
        .iter()
        .map(|visit| routed_per_layer[visit.weight_layer])
        .sum();
    let core_resident: u64 = weights.expert_storage[CORE_START..CORE_END].iter().sum();
    let outside_core: u64 = schedule
        .iter()
        .filter(|visit| visit.pass.is_none())
        .map(|visit| routed_per_layer[visit.weight_layer])
        .sum();
    let expert_storage: u64 = weights.expert_storage.iter().sum();
    let row = |name: &str, chosen: String, fixed_at| PlanDecision {
        name: name.into(),
        chosen,
        alternatives: Vec::new(),
        fixed_at,
    };
    Ok(vec![
        row(
            "decoder loop",
            format!(
                "{LAYERS} stored layers, {VISITS} visits ({core_visits} in the core), {state_slots} state slots"
            ),
            PlanFixity::Load,
        ),
        row(
            "text weight inventory",
            format!(
                "{} B non-expert text, {} B routed experts (20 NVFP4 layers, 4 MXFP8)",
                weights.text_resident, expert_storage
            ),
            PlanFixity::Load,
        ),
        row(
            "vision weights",
            format!("{} B outside the text memory plan", weights.vision),
            PlanFixity::Load,
        ),
        row(
            "text state",
            format!(
                "{} B BF16 KV at {context} tokens ({} full, {} local visits), {} B BF16 convolution history",
                state.kv, state.full, state.local, state.convolution
            ),
            PlanFixity::Start,
        ),
        row(
            "exact expert read ranges",
            format!(
                "adjacent versus scattered top-8 IDs need {} vs {} ranges for NVFP4 and {} vs {} for MXFP8",
                coalesced_reads[0], scattered_reads[0], coalesced_reads[20], scattered_reads[20],
            ),
            PlanFixity::Load,
        ),
        row(
            "loop-core residency",
            format!(
                "{core_resident} B of stored core experts once, {outside_core} B raw expert demand per token at uncached outer visits"
            ),
            PlanFixity::Load,
        ),
        row(
            "routed expert reads",
            format!(
                "{routed_peak} B peak raw top-8 layer stage, {routed_uncached} B per token if every visit misses"
            ),
            PlanFixity::Load,
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_core_pass_has_private_state_and_shared_weight_indices() {
        let schedule = visits();
        let expected = (0..6).chain((6..18).cycle().take(36)).chain(18..24);
        assert!(schedule.iter().map(|visit| visit.weight_layer).eq(expected));
        for (i, visit) in schedule.iter().enumerate() {
            assert_eq!(visit.state_slot, i);
            assert_eq!(visit.pass, (6..42).contains(&i).then(|| (i - 6) / 12));
        }
    }

    #[test]
    fn cache_charges_repeat_state_but_bound_local_context() {
        let short = state_plan(512).unwrap();
        assert_eq!(short.kv, 48 * 512 * 4096);
        let long = state_plan(4096).unwrap();
        assert_eq!(long.kv, 40 * 1024 * 4096 + 8 * 4096 * 4096);
        assert_eq!(long.convolution, 1_916_928);
        assert_eq!((long.local, long.full), (40, 8));
        assert!(state_plan(MAX_CONTEXT + 1).is_err());
    }

    #[test]
    fn expert_stage_prices_nvfp4_and_mxfp8_without_repeating_weight_storage() {
        let mut nv = BTreeMap::new();
        let mut mx = BTreeMap::new();
        for (projection, cols) in [("w13", HIDDEN), ("w2", INTERMEDIATE)] {
            let rows = HIDDEN;
            let mut add = |layer, suffix, dtype, shape: Vec<usize>, bytes_per_element| {
                let bytes = shape.iter().product::<usize>() * bytes_per_element;
                let table = if layer == 0 { &mut nv } else { &mut mx };
                table.insert(
                    format!("model.llm.layers.{layer}.mlp.experts.{projection}_weight{suffix}"),
                    TensorInfo {
                        dtype,
                        shape,
                        data_offsets: (0, bytes),
                    },
                );
            };
            add(0, "", Dtype::U8, vec![EXPERTS, rows, cols / 2], 1);
            add(
                0,
                "_scale",
                Dtype::F8_E4M3,
                vec![EXPERTS, rows, cols / 16],
                1,
            );
            add(0, "_amax", Dtype::F32, vec![EXPERTS], 4);
            add(0, "_outer_scale_denominator", Dtype::F32, vec![1], 4);
            add(20, "", Dtype::F8_E4M3, vec![EXPERTS, rows, cols], 1);
            add(20, "_scale", Dtype::U8, vec![EXPERTS, rows, cols / 32], 1);
        }
        let adjacent = [7, 6, 5, 4, 3, 2, 1, 0, 0];
        let scattered = [0, 2, 4, 6, 8, 10, 12, 127];
        for (table, layer, bytes, grouped, individual) in
            [(&nv, 0, 35_831_880, 8, 50), (&mx, 20, 65_691_648, 4, 32)]
        {
            let source = InklingExpertSource {
                shard: tempfile::tempfile().unwrap(),
                len: 0,
                layers: vec![expert_regions(table, 1024, layer).unwrap()],
                routers: Vec::new(),
                weights: WeightPlan::default(),
            };
            for (ids, count) in [
                (adjacent.as_slice(), grouped),
                (scattered.as_slice(), individual),
            ] {
                let plan = source.plan(0, ids, 1).unwrap();
                assert_eq!(plan.read_count(), count);
                assert_eq!(plan.read_bytes(), bytes);
                assert_eq!(plan.staging_bytes(), bytes);
                assert!(plan.reads.iter().all(|run| run.offset >= 1024));
                assert_eq!(plan.gate_up().shape, [8, INTERMEDIATE * 2, HIDDEN]);
                assert_eq!(plan.down().shape, [8, HIDDEN, INTERMEDIATE]);
                assert_eq!(plan.gate_up().amax.is_some(), layer == 0);
                assert_eq!(plan.down().denominator.is_some(), layer == 0);
            }
            assert!(source.plan(0, &[128], 1).is_err());
        }
        assert!(expert_regions(&mx, 0, 0).is_err());
        nv.remove("model.llm.layers.0.mlp.experts.w2_weight_amax");
        assert!(expert_regions(&nv, 0, 0).is_err());
    }

    #[test]
    fn router_layout_requires_official_shapes_and_dtypes() {
        const WEIGHT: &str = "model.llm.layers.3.mlp.gate.weight";
        const BIAS: &str = "model.llm.layers.3.mlp.gate.bias";
        let mut entries = BTreeMap::from([
            (
                WEIGHT.to_string(),
                TensorInfo {
                    dtype: Dtype::BF16,
                    shape: vec![EXPERTS, HIDDEN],
                    data_offsets: (0, EXPERTS * HIDDEN * size_of::<u16>()),
                },
            ),
            (
                BIAS.to_string(),
                TensorInfo {
                    dtype: Dtype::F32,
                    shape: vec![EXPERTS],
                    data_offsets: (0, EXPERTS * size_of::<f32>()),
                },
            ),
        ]);
        for (key, dtype, shape) in [
            (WEIGHT, Dtype::F16, vec![EXPERTS, HIDDEN]),
            (WEIGHT, Dtype::BF16, vec![HIDDEN, EXPERTS]),
            (BIAS, Dtype::BF16, vec![EXPERTS]),
            (BIAS, Dtype::F32, vec![EXPERTS, 1]),
        ] {
            let original = entries[key].clone();
            let changed = entries.get_mut(key).unwrap();
            changed.dtype = dtype;
            changed.shape = shape;
            assert!(matches!(
                router_regions(&entries, 1024, 3),
                Err(EngError::Checkpoint(message)) if message.contains(key)
            ));
            entries.insert(key.to_string(), original);
        }
        entries.remove(BIAS);
        assert!(matches!(
            router_regions(&entries, 1024, 3),
            Err(EngError::Checkpoint(message)) if message.contains(BIAS)
        ));
    }

    #[test]
    fn router_reads_exact_payloads_from_the_validated_shard() {
        use std::io::{Seek, SeekFrom, Write};

        let bias = std::array::from_fn::<_, EXPERTS, _>(|i| i as f32 - 64.0);
        let weight: Vec<u8> = (0..EXPERTS * HIDDEN)
            .flat_map(|i| (0x3f00u16 | (i % 128) as u16).to_le_bytes())
            .collect();
        let bias_len = EXPERTS * size_of::<f32>();
        let header = serde_json::json!({
            "model.llm.layers.0.mlp.gate.bias": {
                "dtype": "F32",
                "shape": [EXPERTS],
                "data_offsets": [0, bias_len],
            },
            "model.llm.layers.0.mlp.gate.weight": {
                "dtype": "BF16",
                "shape": [EXPERTS, HIDDEN],
                "data_offsets": [bias_len, bias_len + weight.len()],
            },
        })
        .to_string();
        let mut shard = tempfile::tempfile().unwrap();
        shard
            .write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        shard.write_all(header.as_bytes()).unwrap();
        for value in bias {
            shard.write_all(&value.to_le_bytes()).unwrap();
        }
        shard.write_all(&weight).unwrap();
        let len = shard.metadata().unwrap().len();
        shard.rewind().unwrap();
        let (data_start, entries) = read_stored_header(&mut shard, len, HEADER_LIMIT).unwrap();
        let routers = vec![router_regions(&entries.into_iter().collect(), data_start, 0).unwrap()];
        let mut source = InklingExpertSource {
            shard,
            len,
            layers: Vec::new(),
            routers,
            weights: WeightPlan::default(),
        };
        let actual = source.read_router(0).unwrap();
        assert_eq!(actual.weight, weight);
        assert_eq!(actual.bias, bias);
        assert!(matches!(
            source.read_router(1),
            Err(EngError::BadRequest(_))
        ));

        let bias_offset = data_start + 5 * size_of::<f32>() as u64;
        source.shard.seek(SeekFrom::Start(bias_offset)).unwrap();
        source.shard.write_all(&f32::NAN.to_le_bytes()).unwrap();
        assert!(matches!(
            source.read_router(0),
            Err(EngError::Checkpoint(message)) if message.contains("bias for expert 5")
        ));
        source.shard.set_len(len - 1).unwrap();
        assert!(matches!(
            source.read_router(0),
            Err(EngError::Checkpoint(message)) if message.contains("shard length changed")
        ));
    }

    #[test]
    fn selected_experts_read_packed_regions_into_aligned_compact_staging() {
        use std::io::Write;

        let mut shard = tempfile::tempfile().unwrap();
        let mut contents = vec![255; 32];
        let mut slices = |stride: usize, tag: u8| {
            let start = contents.len() as u64;
            for expert in 0..EXPERTS {
                contents.extend(std::iter::repeat_n(tag + expert as u8, stride));
            }
            ExpertRegion::Slices {
                start,
                stride: stride as u64,
            }
        };
        let mut gate_up = ProjectionRegions {
            rows: 4,
            cols: 4,
            weight: slices(3, 1),
            scale: slices(2, 10),
            amax: Some(slices(4, 20)),
            denominator: None,
        };
        let mut down = ProjectionRegions {
            rows: 4,
            cols: 2,
            weight: slices(2, 30),
            scale: slices(1, 40),
            amax: Some(slices(4, 50)),
            denominator: None,
        };
        let mut shared = |tag| {
            let start = contents.len() as u64;
            contents.extend([tag; 4]);
            Some(ExpertRegion::Shared(start..start + 4))
        };
        gate_up.denominator = shared(60);
        down.denominator = shared(70);
        shard.write_all(&contents).unwrap();
        let source = InklingExpertSource {
            shard,
            len: contents.len() as u64,
            layers: vec![[gate_up, down]],
            routers: Vec::new(),
            weights: WeightPlan::default(),
        };

        let alignment = 16_384;
        let plan = source.plan(0, &[7, 2, 3, 7], alignment).unwrap();
        assert_eq!(plan.alignment(), alignment);
        assert_eq!(plan.compact_slots(), &[2, 0, 1, 2]);
        assert_eq!(plan.read_count(), 14);
        assert_eq!(plan.read_bytes(), 56);
        assert_eq!(plan.gate_up().shape, [3, 4, 4]);
        assert_eq!(plan.down().shape, [3, 4, 2]);
        assert_eq!(plan.staging_bytes(), 7 * alignment + 4);
        let mut staging = vec![255; plan.staging_bytes() + 1];
        plan.read_into(&mut staging).unwrap();

        for (ordinal, (range, expected)) in [
            (&plan.gate_up().weight, vec![3, 3, 3, 4, 4, 4, 8, 8, 8]),
            (&plan.gate_up().scale, vec![12, 12, 13, 13, 17, 17]),
            (
                plan.gate_up().amax.as_ref().unwrap(),
                [[22; 4], [23; 4], [27; 4]].concat(),
            ),
            (plan.gate_up().denominator.as_ref().unwrap(), vec![60; 4]),
            (&plan.down().weight, vec![32, 32, 33, 33, 37, 37]),
            (&plan.down().scale, vec![42, 43, 47]),
            (
                plan.down().amax.as_ref().unwrap(),
                [[52; 4], [53; 4], [57; 4]].concat(),
            ),
            (plan.down().denominator.as_ref().unwrap(), vec![70; 4]),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(range.start, ordinal * alignment);
            assert_eq!(staging[range.clone()], expected);
            assert_eq!(staging[range.end], 255);
        }

        source.shard.set_len(source.len - 1).unwrap();
        let mut too_small = vec![255; plan.staging_bytes() - 1];
        assert!(matches!(
            plan.read_into(&mut too_small),
            Err(EngError::BadRequest(_))
        ));
        assert!(too_small.iter().all(|&byte| byte == 255));
        assert!(matches!(
            plan.read_into(&mut staging),
            Err(EngError::Checkpoint(_))
        ));
        assert!(source.plan(0, &[], alignment).is_err());
        assert!(source.plan(0, &[1], 3).is_err());
        assert!(source.plan(1, &[1], alignment).is_err());
        assert!(source.plan(0, &[1], 1usize << (usize::BITS - 1)).is_err());
    }
}
