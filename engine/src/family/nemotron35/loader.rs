//! Calling-thread checkpoint decoding with lazy, retained shard handles.

use std::{
    collections::{BTreeMap, VecDeque, btree_map::Entry},
    fs,
    ops::Range,
    path::{Path, PathBuf},
};

use super::{NemotronTextSpec, exec::StoredWeight};
use crate::{
    EngError,
    kernels::{f8, nvfp4},
    tensor::{Dtype, Tensor},
    weights::{
        nemotron_index::NemotronIndex,
        nemotron_shards::{NemotronShardReader, NemotronShardTensor},
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertIndex {
    Routed(u32),
    Shared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertProjection {
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertInputScales {
    pub up_input_scale: f32,
    pub down_input_scale: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpertScaleFacts {
    pub routed: Vec<ExpertInputScales>,
    pub shared: ExpertInputScales,
}

/// Byte ranges of one NVFP4 projection inside its shard file, for a caller that maps the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionRanges {
    pub packed: Range<usize>,
    pub scales: Range<usize>,
    /// Four bytes of little-endian F32; the mapping caller validates the value.
    pub weight_scale_2: Range<usize>,
    pub rows: usize,
    pub columns: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertRanges {
    pub up: ProjectionRanges,
    pub down: ProjectionRanges,
}

/// Where every routed expert of one MoE layer sits inside the layer's single shard file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertLayerMapping {
    pub path: PathBuf,
    pub file_len: u64,
    /// Indexed by routed expert id.
    pub experts: Vec<ExpertRanges>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExpertKey {
    layer: usize,
    expert: ExpertIndex,
    which: ExpertProjection,
}

/// The cache bounds decoded F32 tensor entries, not caller-retained clones or shard metadata.
/// Each entry retains D = rows * columns * 4 payload bytes. Clones share that storage.
/// A miss evicts before decoding. Its temporary payload peak is at most 2*D + packed + group scales,
/// including the codec's Vec-to-Arc copy. Add other cached entries and caller-retained tensors.
/// Scalar buffers, allocator overhead, index, and lazily retained shard headers are additional.
/// Shapes are checked structurally; callers must check full architecture geometry through the descriptor.
pub struct NemotronLoader {
    directory: PathBuf,
    index: NemotronIndex,
    spec: NemotronTextSpec,
    shards: BTreeMap<String, NemotronShardReader>,
    cache: VecDeque<(ExpertKey, Tensor)>,
    cache_entries: usize,
    evictions: u64,
}

impl NemotronLoader {
    /// Reads config and index only. A nonzero cache entry capacity is required.
    pub fn open(directory: impl AsRef<Path>, cache_entries: usize) -> Result<Self, EngError> {
        let cache_bytes = cache_entries
            .checked_mul(std::mem::size_of::<(ExpertKey, Tensor)>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| bad("cache entry capacity exceeds host limits"))?;
        if cache_bytes == 0 {
            return Err(bad("cache entry capacity must exceed zero"));
        }
        let mut cache = VecDeque::new();
        cache
            .try_reserve_exact(cache_entries)
            .map_err(|error| bad(format!("allocate cache entries: {error}")))?;
        let directory = directory.as_ref().to_path_buf();
        let read = |name| {
            fs::read(directory.join(name)).map_err(|error| bad(format!("read {name}: {error}")))
        };
        let spec = NemotronTextSpec::from_config(&read("config.json")?)?;
        let index = NemotronIndex::from_json(&read("model.safetensors.index.json")?)?;
        Ok(Self {
            directory,
            index,
            spec,
            shards: BTreeMap::new(),
            cache,
            cache_entries,
            evictions: 0,
        })
    }

    pub(crate) fn moe_geometry(&self) -> super::NemotronMoeGeometry {
        self.spec.moe_geometry()
    }

    pub fn index(&self) -> &NemotronIndex {
        &self.index
    }

    pub fn eviction_count(&self) -> u64 {
        self.evictions
    }

    /// Decodes BF16, F32, or per-tensor FP8 weights. Activation scales are never multiplied into weights.
    pub fn dense_weight(&mut self, name: &str) -> Result<(Tensor, Option<f32>), EngError> {
        self.spec.key_plan(name)?;
        let metadata = self.metadata(name)?;
        let shape = tensor_shape(&metadata)?;
        match metadata.dtype() {
            "BF16" => {
                let packed = self.shard(name)?.read_tensor(name)?;
                let output_len = packed
                    .len()
                    .checked_mul(2)
                    .ok_or_else(|| bad("BF16 decoded byte length overflows"))?;
                let mut bytes = Vec::new();
                bytes
                    .try_reserve_exact(output_len)
                    .map_err(|error| bad(format!("allocate BF16 output: {error}")))?;
                for pair in packed.as_chunks::<2>().0 {
                    let value = f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16);
                    if !value.is_finite() {
                        return Err(bad(format!("{name}: non-finite BF16 weight")));
                    }
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                Ok((Tensor::try_from_host(Dtype::F32, shape, bytes)?, None))
            }
            "F32" => {
                let bytes = self.shard(name)?.read_tensor(name)?;
                for element in bytes.as_chunks::<4>().0 {
                    if !f32::from_le_bytes(*element).is_finite() {
                        return Err(bad(format!("{name}: non-finite F32 weight")));
                    }
                }
                Ok((Tensor::try_from_host(Dtype::F32, shape, bytes)?, None))
            }
            "F8_E4M3" => {
                let base = name
                    .strip_suffix(".weight")
                    .ok_or_else(|| bad("FP8 projection name must end with .weight"))?;
                let weight_scale = self.scalar(&format!("{base}.weight_scale"))?.to_le_bytes();
                let input_scale = self.scalar(&format!("{base}.input_scale"))?.to_le_bytes();
                let codes = self.shard(name)?.read_tensor(name)?;
                f8::WeightBlock {
                    codes: &codes,
                    weight_scale: &weight_scale,
                    input_scale: Some(&input_scale),
                }
                .decode(&shape)
                .map_err(checkpoint_decode_error)
            }
            dtype => Err(bad(format!("{name}: unsupported dense dtype {dtype}"))),
        }
    }

    /// Reads one dense weight in stored form: BF16 and F8_E4M3 stay packed
    /// codes; F32 stays F32. Shapes are checked structurally against the
    /// header exactly as `dense_weight` checks them.
    pub fn dense_stored(&mut self, name: &str) -> Result<StoredWeight, EngError> {
        self.spec.key_plan(name)?;
        let metadata = self.metadata(name)?;
        let shape = tensor_shape(&metadata)?;
        match metadata.dtype() {
            "BF16" => {
                let codes = self.shard(name)?.read_tensor(name)?;
                let elements = shape.iter().try_fold(1usize, |count, &dimension| {
                    usize::try_from(dimension)
                        .map_err(|_| bad("shape dimension exceeds host limits"))?
                        .checked_mul(count)
                        .ok_or_else(|| bad("shape product overflows host limits"))
                })?;
                if codes.len()
                    != elements
                        .checked_mul(2)
                        .ok_or_else(|| bad("stored BF16 byte length overflows host limits"))?
                {
                    return Err(bad(format!(
                        "{name}: stored BF16 length differs from shape"
                    )));
                }
                Ok(StoredWeight::Bf16 { codes, shape })
            }
            "F8_E4M3" => {
                let base = name
                    .strip_suffix(".weight")
                    .ok_or_else(|| bad("FP8 projection name must end with .weight"))?;
                let weight_scale =
                    f32::from_le_bytes(self.scalar(&format!("{base}.weight_scale"))?.to_le_bytes());
                if !weight_scale.is_finite() || weight_scale <= 0.0 {
                    return Err(bad(format!(
                        "{name}: weight_scale must be finite and positive"
                    )));
                }
                let codes = self.shard(name)?.read_tensor(name)?;
                let elements = shape.iter().try_fold(1usize, |count, &dimension| {
                    usize::try_from(dimension)
                        .map_err(|_| bad("shape dimension exceeds host limits"))?
                        .checked_mul(count)
                        .ok_or_else(|| bad("shape product overflows host limits"))
                })?;
                if codes.len() != elements {
                    return Err(bad(format!("{name}: stored F8 length differs from shape")));
                }
                Ok(StoredWeight::F8 {
                    codes,
                    weight_scale,
                    shape,
                })
            }
            "F32" => {
                let (tensor, _) = self.dense_weight(name)?;
                Ok(StoredWeight::F32(tensor))
            }
            dtype => Err(bad(format!("{name}: unsupported dense dtype {dtype}"))),
        }
    }

    pub fn expert_block(
        &mut self,
        layer: usize,
        expert: ExpertIndex,
        which: ExpertProjection,
    ) -> Result<Tensor, EngError> {
        let key = ExpertKey {
            layer,
            expert,
            which,
        };
        let base = self.expert_base(key)?;
        if let Some(position) = self.cache.iter().position(|(cached, _)| *cached == key) {
            let entry = self
                .cache
                .remove(position)
                .ok_or_else(|| bad("cache entry missing"))?;
            let tensor = entry.1.clone();
            self.cache.push_back(entry);
            return Ok(tensor);
        }
        if self.cache.len() == self.cache_entries {
            self.evictions = self
                .evictions
                .checked_add(1)
                .ok_or_else(|| bad("cache eviction counter overflows"))?;
            self.cache.pop_front();
        }
        let weight_name = format!("{base}.weight");
        let metadata = self.metadata(&weight_name)?;
        let stored_shape = tensor_shape(&metadata)?;
        let [rows, packed_columns] = stored_shape.as_slice() else {
            return Err(bad(format!("{weight_name}: expected a rank-two U8 tensor")));
        };
        if metadata.dtype() != "U8" || *rows == 0 || *packed_columns == 0 {
            return Err(bad(format!("{weight_name}: expected nonempty U8 matrix")));
        }
        let columns = packed_columns
            .checked_mul(2)
            .ok_or_else(|| bad("expert column count overflows"))?;
        if !columns.is_multiple_of(16) {
            return Err(bad("expert columns must be a multiple of 16"));
        }
        let group_name = format!("{base}.weight_scale");
        let group_metadata = self.metadata(&group_name)?;
        if group_metadata.dtype() != "F8_E4M3"
            || group_metadata.shape() != [u64::from(*rows), u64::from(columns / 16)]
        {
            return Err(bad(format!(
                "{group_name}: expected matching E4M3 group-scale matrix"
            )));
        }
        let weight_scale_2 = self.scalar(&format!("{base}.weight_scale_2"))?;
        let input_scale = self.scalar(&format!("{base}.input_scale"))?;
        let packed = self.shard(&weight_name)?.read_tensor(&weight_name)?;
        let group_scales = self.shard(&group_name)?.read_tensor(&group_name)?;
        let tensor = nvfp4::WeightBlock {
            packed: &packed,
            weight_scale: &group_scales,
            weight_scale_2,
            input_scale: Some(input_scale),
        }
        .decode(&[*rows, columns], Dtype::F32)
        .map_err(checkpoint_decode_error)?;
        self.cache.push_back((key, tensor.clone()));
        Ok(tensor)
    }

    /// Reads one expert projection exactly as stored, with no host expansion.
    pub fn expert_packed(
        &mut self,
        layer: usize,
        expert: ExpertIndex,
        which: ExpertProjection,
    ) -> Result<crate::family::nemotron35::exec::PackedExpert, EngError> {
        let base = self.expert_base(ExpertKey {
            layer,
            expert,
            which,
        })?;
        let weight_name = format!("{base}.weight");
        let metadata = self.metadata(&weight_name)?;
        let stored_shape = tensor_shape(&metadata)?;
        let [rows, packed_columns] = stored_shape.as_slice() else {
            return Err(bad(format!("{weight_name}: expected a rank-two U8 tensor")));
        };
        if metadata.dtype() != "U8" || *rows == 0 || *packed_columns == 0 {
            return Err(bad(format!("{weight_name}: expected nonempty U8 matrix")));
        }
        let columns = packed_columns
            .checked_mul(2)
            .ok_or_else(|| bad("expert column count overflows"))?;
        if !columns.is_multiple_of(16) {
            return Err(bad("expert columns must be a multiple of 16"));
        }
        let group_name = format!("{base}.weight_scale");
        let group_metadata = self.metadata(&group_name)?;
        if group_metadata.dtype() != "F8_E4M3"
            || group_metadata.shape() != [u64::from(*rows), u64::from(columns / 16)]
        {
            return Err(bad(format!(
                "{group_name}: expected matching E4M3 group-scale matrix"
            )));
        }
        let weight_scale_2 = self.scalar(&format!("{base}.weight_scale_2"))?;
        let input_scale = self.scalar(&format!("{base}.input_scale"))?;
        let packed = self.shard(&weight_name)?.read_tensor(&weight_name)?;
        let scale = self.shard(&group_name)?.read_tensor(&group_name)?;
        Ok(crate::family::nemotron35::exec::PackedExpert {
            packed,
            scale,
            weight_scale_2,
            input_scale,
            shape: [*rows, columns],
        })
    }

    /// Locates the packed bytes, group scales and per-tensor scale of every routed expert of
    /// `layer` inside the shard file that holds them, without reading any payload. The layer's
    /// experts must share one shard; the up and down shapes must match the MoE geometry.
    pub fn expert_layer_mapping(&mut self, layer: usize) -> Result<ExpertLayerMapping, EngError> {
        let moe = self.spec.moe_geometry();
        let intermediate = to_usize(moe.moe_intermediate_size)?;
        let latent = to_usize(moe.latent_size)?;
        let count = to_usize(moe.routed_experts)?;
        let mut path: Option<PathBuf> = None;
        let mut file_len = 0u64;
        let mut experts = Vec::new();
        experts
            .try_reserve_exact(count)
            .map_err(|error| bad(format!("allocate expert mapping: {error}")))?;
        for expert in 0..moe.routed_experts {
            let mut ranges = Vec::with_capacity(2);
            for (which, rows, columns) in [
                (ExpertProjection::Up, intermediate, latent),
                (ExpertProjection::Down, latent, intermediate),
            ] {
                let base = self.expert_base(ExpertKey {
                    layer,
                    expert: ExpertIndex::Routed(expert),
                    which,
                })?;
                let shard_name = self
                    .index
                    .shard_for_tensor(&format!("{base}.weight"))
                    .ok_or_else(|| bad(format!("index has no tensor {base}.weight")))?
                    .to_owned();
                let shard_path = self.directory.join(&shard_name);
                if *path.get_or_insert_with(|| shard_path.clone()) != shard_path {
                    return Err(bad(format!(
                        "layer {layer} routed experts span more than one shard"
                    )));
                }
                let reader = self.shard(&format!("{base}.weight"))?;
                file_len = reader.file_len();
                ranges.push(projection_ranges(reader, &base, rows, columns)?);
            }
            let down = ranges.pop().expect("two projections were pushed");
            let up = ranges.pop().expect("two projections were pushed");
            experts.push(ExpertRanges { up, down });
        }
        Ok(ExpertLayerMapping {
            path: path.ok_or_else(|| bad("a layer needs at least one routed expert"))?,
            file_len,
            experts,
        })
    }

    /// Evicts the page-cache pages of every routed-expert tensor of `layer` from its shard file,
    /// for a layer whose experts now live on the device. Pages that are mapped (a CPU-served
    /// layer) are never asked for: the caller only releases layers it uploaded.
    pub fn release_expert_layer(&mut self, layer: usize) -> Result<(), EngError> {
        let experts: Vec<u32> = (0..self.spec.moe_geometry().routed_experts).collect();
        self.release_routed_experts(layer, &experts)
    }

    /// Drops only the selected experts' packed and group-scale pages after their copy finishes.
    pub fn release_routed_experts(
        &mut self,
        layer: usize,
        experts: &[u32],
    ) -> Result<(), EngError> {
        if experts.is_empty() {
            return Ok(());
        }
        let mapping = self.expert_layer_mapping(layer)?;
        let mut ranges = Vec::new();
        for &id in experts {
            let expert = mapping
                .experts
                .get(to_usize(id)?)
                .ok_or_else(|| bad(format!("unknown routed expert {id}")))?;
            for part in [&expert.up, &expert.down] {
                ranges.extend([part.packed.clone(), part.scales.clone()]);
            }
        }
        ranges.sort_by_key(|range| range.start);
        let merged = merge_adjacent(ranges);
        let first =
            format!("language_model.backbone.layers.{layer}.mixer.experts.0.up_proj.weight");
        self.shard(&first)?.drop_cache(&merged)
    }

    /// Reads only activation scalars for every routed expert and the shared expert.
    pub fn expert_scale_facts(&mut self, layer: usize) -> Result<ExpertScaleFacts, EngError> {
        let mut routed = Vec::new();
        let count = usize::try_from(self.spec.moe_geometry().routed_experts)
            .map_err(|_| bad("expert count exceeds host limits"))?;
        routed
            .try_reserve_exact(count)
            .map_err(|error| bad(format!("allocate expert scale facts: {error}")))?;
        for expert in 0..self.spec.moe_geometry().routed_experts {
            routed.push(self.expert_input_scales(layer, ExpertIndex::Routed(expert))?);
        }
        let shared = self.expert_input_scales(layer, ExpertIndex::Shared)?;
        Ok(ExpertScaleFacts { routed, shared })
    }

    fn expert_input_scales(
        &mut self,
        layer: usize,
        expert: ExpertIndex,
    ) -> Result<ExpertInputScales, EngError> {
        let read = |this: &mut Self, which| {
            let base = this.expert_base(ExpertKey {
                layer,
                expert,
                which,
            })?;
            this.scalar(&format!("{base}.input_scale"))
        };
        Ok(ExpertInputScales {
            up_input_scale: read(self, ExpertProjection::Up)?,
            down_input_scale: read(self, ExpertProjection::Down)?,
        })
    }

    fn expert_base(&self, key: ExpertKey) -> Result<String, EngError> {
        let expert = match key.expert {
            ExpertIndex::Routed(index) => format!("experts.{index}"),
            ExpertIndex::Shared => "shared_experts".into(),
        };
        let projection = match key.which {
            ExpertProjection::Up => "up_proj",
            ExpertProjection::Down => "down_proj",
        };
        let base = format!(
            "language_model.backbone.layers.{}.mixer.{expert}.{projection}",
            key.layer
        );
        self.spec.key_plan(&format!("{base}.weight"))?;
        Ok(base)
    }

    fn scalar(&mut self, name: &str) -> Result<f32, EngError> {
        let metadata = self.metadata(name)?;
        if metadata.dtype() != "F32" || !matches!(metadata.shape(), [] | [1]) {
            return Err(bad(format!("{name}: expected F32 scalar or [1]")));
        }
        let bytes = self.shard(name)?.read_tensor(name)?;
        let bytes: [u8; 4] = bytes
            .try_into()
            .map_err(|_| bad(format!("{name}: expected four bytes")))?;
        let value = f32::from_le_bytes(bytes);
        if !value.is_finite() || value <= 0.0 {
            return Err(bad(format!("{name}: scale must be finite and positive")));
        }
        Ok(value)
    }

    fn metadata(&mut self, name: &str) -> Result<NemotronShardTensor, EngError> {
        self.shard(name)?
            .tensor(name)
            .cloned()
            .ok_or_else(|| bad(format!("shard does not contain indexed tensor {name:?}")))
    }

    fn shard(&mut self, name: &str) -> Result<&mut NemotronShardReader, EngError> {
        let shard_name = self
            .index
            .shard_for_tensor(name)
            .ok_or_else(|| bad(format!("index has no tensor {name:?}")))?;
        match self.shards.entry(shard_name.to_owned()) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let shard = NemotronShardReader::open(self.directory.join(shard_name))?;
                Ok(entry.insert(shard))
            }
        }
    }
}

/// Joins sorted ranges that touch or overlap, so a layer of 2048 tensors becomes a few calls.
fn merge_adjacent(sorted: Vec<Range<usize>>) -> Vec<Range<usize>> {
    let mut merged: Vec<Range<usize>> = Vec::new();
    for range in sorted {
        match merged.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => merged.push(range),
        }
    }
    merged
}

fn to_usize(value: u32) -> Result<usize, EngError> {
    usize::try_from(value).map_err(|_| bad("dimension exceeds host limits"))
}

/// Checks one NVFP4 projection against its expected `[rows, columns]` and returns its ranges.
fn projection_ranges(
    reader: &NemotronShardReader,
    base: &str,
    rows: usize,
    columns: usize,
) -> Result<ProjectionRanges, EngError> {
    let part = |suffix: &str, dtype: &str, shape: &[usize]| {
        let name = format!("{base}.{suffix}");
        let tensor = reader
            .tensor(&name)
            .ok_or_else(|| bad(format!("shard does not contain indexed tensor {name:?}")))?;
        let actual: Vec<usize> = tensor.shape().iter().map(|&dim| dim as usize).collect();
        if tensor.dtype() != dtype || actual != shape {
            return Err(bad(format!(
                "{name}: expected {dtype} {shape:?}, found {} {actual:?}",
                tensor.dtype()
            )));
        }
        reader
            .absolute_range(&name)
            .ok_or_else(|| bad(format!("{name}: payload range does not fit host limits")))
    };
    Ok(ProjectionRanges {
        packed: part("weight", "U8", &[rows, columns / 2])?,
        scales: part("weight_scale", "F8_E4M3", &[rows, columns / 16])?,
        weight_scale_2: part("weight_scale_2", "F32", &[])
            .or_else(|_| part("weight_scale_2", "F32", &[1]))?,
        rows,
        columns,
    })
}

fn tensor_shape(metadata: &NemotronShardTensor) -> Result<Vec<u32>, EngError> {
    metadata
        .shape()
        .iter()
        .map(|&dim| u32::try_from(dim).map_err(|_| bad("tensor dimension exceeds u32")))
        .collect()
}

fn checkpoint_decode_error(error: EngError) -> EngError {
    match error {
        EngError::BadRequest(message) => bad(message),
        other => other,
    }
}

fn bad(message: impl Into<String>) -> EngError {
    EngError::Checkpoint(format!("Nemotron loader: {}", message.into()))
}

#[cfg(test)]
mod merge_tests {
    use super::merge_adjacent;

    #[test]
    fn touching_and_overlapping_ranges_join_and_gaps_stay() {
        assert_eq!(
            merge_adjacent(vec![0..4, 4..8, 6..10, 12..14, 14..15, 20..21]),
            vec![0..10, 12..15, 20..21]
        );
        assert_eq!(
            merge_adjacent(Vec::new()),
            Vec::<std::ops::Range<usize>>::new()
        );
        assert_eq!(merge_adjacent(vec![3..9, 4..5]), vec![3..9]);
    }
}
