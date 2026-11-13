//! Source adaptation of a stock EXL3 pack for the resident step: matrices go to the device in
//! their stored components, norms and small plain tensors as F32 vectors, all under the GGUF
//! names the step looks up (`LayerTable`, `GlobalNames`).
//!
//! Stock `architecture/qwen3_5.py:324-328,355-392` defines these source names and
//! zero-centered norms. Stock `exllamav3_ext/norm.cu:229-244` converts weights to
//! float4 BEFORE adding constant_bias. Supply that effective F32 weight once:
//! our add_norm_pack and attn_prep kernels multiply their supplied weights directly.
//! Query/gate rows remain in the source's per-head [query | gate] order.
//!
//! A DeltaNet layer additionally moves its value heads from the pack's grouped order into the
//! tiled order of the kernels (see `exl3_tensors`): whole heads of every matrix, vector and conv
//! row that carries a value-head axis.

use crate::family::project::cuda::WeightLayout;
use crate::family::qwen36::exl3_pack::{Root, layer_name};
use crate::family::qwen36::exl3_tensors::{
    decay_from_log, effective_norm_bytes, f32_le_bytes, norm_layout, permute_blocks,
    tiled_head_order, widen_f32,
};

use super::{
    layer_table::LayerTable,
    mixer_geometry::{DeltaGeometry, DenseSpec, MixerGeometry, RawSpec, check_dense},
};
use crate::{
    EngError,
    family::{
        AttentionKind,
        gguf_model::Role,
        project::cuda::{CudaProjector, CudaWeight, Reserve},
    },
    tensor::Tensor,
    weights::{
        GgufType, WeightSource,
        exl3::{Axis, COMPONENT_SUFFIXES, Checkpoint, GROUP, Matrix},
    },
};

/// Largest stored matrix read at once: the head's trellis is the biggest, at 318 MB.
const MATRIX_READ_LIMIT: u64 = 1 << 30;

/// A group order that moves each of the `order` heads as `groups_per_head` whole groups.
fn head_groups(order: &[usize], groups_per_head: usize) -> Vec<usize> {
    order
        .iter()
        .flat_map(|&head| (0..groups_per_head).map(move |g| head * groups_per_head + g))
        .collect()
}

pub(in crate::family) struct PackReader<'a> {
    projector: &'a CudaProjector,
    pub(in crate::family) checkpoint: &'a Checkpoint,
    reserve: Reserve,
    root: Root,
}

impl<'a> PackReader<'a> {
    pub(in crate::family) fn new(
        projector: &'a CudaProjector,
        checkpoint: &'a Checkpoint,
        reserve: Reserve,
    ) -> Self {
        Self {
            projector,
            checkpoint,
            reserve,
            root: Root::Layers,
        }
    }

    /// Reads the layers from `root` instead of the model's own: the MTP head's block.
    pub(in crate::family) fn under(mut self, root: Root) -> Self {
        self.root = root;
        self
    }

    /// Where the layers of this pack are named.
    pub(in crate::family) fn root(&self) -> Root {
        self.root
    }

    /// Admits `bytes` of weights against the free device memory and the reserve of the load.
    pub(in crate::family) fn admit(&self, bytes: u64) -> Result<(), EngError> {
        self.projector.admit(bytes, self.reserve)
    }

    /// The stored components of the matrix under `prefix`. Unsupported old representations are
    /// selected too, so `Matrix::load` can refuse them.
    pub(in crate::family) fn matrix(&self, prefix: &str) -> Result<Matrix, EngError> {
        let names = COMPONENT_SUFFIXES.map(|suffix| format!("{prefix}.{suffix}"));
        let keys = names.each_ref().map(String::as_str);
        let selected = self
            .checkpoint
            .source
            .read_selected(&keys, MATRIX_READ_LIMIT)?;
        Matrix::load(&selected, prefix, &self.checkpoint.config)
    }

    fn plain(&self, name: &str, shape: &[u32]) -> Result<Tensor, EngError> {
        let missing = || EngError::Checkpoint(format!("EXL3 source: missing tensor {name}"));
        let (dtype, stored) = self.checkpoint.source.stored(name).ok_or_else(missing)?;
        if stored != shape {
            return Err(EngError::Checkpoint(format!(
                "EXL3 source: {name} is stored as {stored:?}, expected {shape:?}"
            )));
        }
        let values = shape.iter().map(|&dim| u64::from(dim)).product::<u64>();
        let budget = values.saturating_mul(dtype.byte_width());
        let selected = self.checkpoint.source.read_selected(&[name], budget)?;
        selected.tensor(name).ok_or_else(missing)
    }

    pub(in crate::family) fn exl3(
        &self,
        name: String,
        prefix: &str,
        want: &DenseSpec,
        reorder: Option<(Axis, &[usize])>,
    ) -> Result<CudaWeight, EngError> {
        let matrix = self.matrix(prefix)?;
        let [k, n] = matrix.shape();
        check_dense(prefix, (n, k, None), want)?;
        let matrix = match reorder {
            Some((axis, order)) => matrix.reorder_groups(axis, order)?,
            None => matrix,
        };
        self.upload(name, &matrix)
    }

    /// Uploads `matrix` as it is stored (no geometry check, no reorder): the caller has checked
    /// its shape, as the Flash-Next trunk does through the pack's header walk.
    pub(in crate::family) fn upload(
        &self,
        name: String,
        matrix: &Matrix,
    ) -> Result<CudaWeight, EngError> {
        let weight = self.projector.upload_exl3(matrix, self.reserve)?;
        Ok(CudaWeight::Exl3 { name, weight })
    }

    pub(in crate::family) fn bank(
        &self,
        name: String,
        count: usize,
        member: impl FnMut(usize) -> Result<Matrix, EngError>,
    ) -> Result<CudaWeight, EngError> {
        let bank = self
            .projector
            .upload_exl3_bank(count, self.reserve, member)?;
        Ok(CudaWeight::Exl3Bank { name, bank })
    }

    fn raw_bytes(&self, name: String, bytes: &[u8]) -> Result<CudaWeight, EngError> {
        let budget = u64::try_from(bytes.len())
            .map_err(|_| EngError::MemoryPlan("EXL3 raw upload byte count overflow".into()))?;
        self.projector.admit(budget, self.reserve)?;
        let buffer = self.projector.context().upload(bytes)?;
        Ok(CudaWeight::Raw {
            name,
            ty: GgufType::F32,
            buffer,
        })
    }

    /// Uploads the zero-centered norm `source` as `name` with one added in F32.
    pub(in crate::family) fn norm(
        &self,
        name: String,
        source: &str,
        values: usize,
    ) -> Result<CudaWeight, EngError> {
        let (dtype, shape) = self.checkpoint.source.stored(source).ok_or_else(|| {
            EngError::Checkpoint(format!("EXL3 attention source: missing tensor {source}"))
        })?;
        // Validate the exact source shape and type before reading even a tiny norm.
        let output_len = norm_layout(dtype, &shape, values)?;
        let budget = u64::try_from(output_len)
            .map_err(|_| EngError::MemoryPlan("EXL3 norm read budget overflow".into()))?;
        let selected = self.checkpoint.source.read_selected(&[source], budget)?;
        let tensor = selected.tensor(source).ok_or_else(|| {
            EngError::Checkpoint(format!("EXL3 attention source: missing tensor {source}"))
        })?;
        self.raw_bytes(name, &effective_norm_bytes(&tensor, values)?)
    }

    pub(in crate::family) fn vector(
        &self,
        name: String,
        source: &str,
        shape: &[u32],
        transform: impl FnOnce(Vec<f32>) -> Result<Vec<f32>, EngError>,
    ) -> Result<CudaWeight, EngError> {
        let tensor = self.plain(source, shape)?;
        let values = shape.iter().map(|&dim| dim as usize).product();
        let values = transform(widen_f32(&tensor, values, source)?)?;
        self.raw_bytes(name, &f32_le_bytes(&values))
    }

    pub(in crate::family) fn table(
        &self,
        name: String,
        source: &str,
        shape: &[u32],
    ) -> Result<CudaWeight, EngError> {
        let tensor = self.plain(source, shape)?;
        let ty = match tensor.dtype() {
            crate::tensor::Dtype::Bf16 => GgufType::Bf16,
            crate::tensor::Dtype::F16 => GgufType::F16,
            other => {
                return Err(EngError::Checkpoint(format!(
                    "EXL3 source: table {source} has dtype {other:?}"
                )));
            }
        };
        let bytes = tensor
            .host_bytes()
            .ok_or_else(|| EngError::Checkpoint(format!("EXL3 table {source} needs host bytes")))?;
        let budget = u64::try_from(bytes.len())
            .map_err(|_| EngError::MemoryPlan("EXL3 table byte count overflow".into()))?;
        self.projector.admit(budget, self.reserve)?;
        let buffer = self.projector.context().upload(bytes)?;
        Ok(CudaWeight::Raw { name, ty, buffer })
    }
}

fn layer_kind(table: &LayerTable, layer: usize, want: AttentionKind) -> Result<(), EngError> {
    if table.slot(layer)?.kind != want {
        return Err(EngError::Checkpoint(format!(
            "EXL3 source: layer {layer} is not {want:?}"
        )));
    }
    Ok(())
}

/// Loads only the four projections and four norms of a full-attention mixer, as a self-contained
/// set. The whole-model loader takes [`attention_weights`] instead.
#[cfg(test)]
pub(super) fn load_attention(
    projector: &CudaProjector,
    checkpoint: &Checkpoint,
    table: &LayerTable,
    layer: usize,
    geometry: &MixerGeometry,
    reserve: Reserve,
) -> Result<crate::family::project::cuda::CudaWeights, EngError> {
    let reader = PackReader::new(projector, checkpoint, reserve);
    crate::family::project::cuda::CudaWeights::from_exl3(attention_weights(
        &reader, table, layer, geometry,
    )?)
}

/// The four projections and four norms of a full-attention mixer.
/// `table` owns logical names; checkpoint headers and `Matrix` own stored metadata.
/// Each matrix is checked against geometry before its first device allocation.
pub(super) fn attention_weights(
    reader: &PackReader<'_>,
    table: &LayerTable,
    layer: usize,
    geometry: &MixerGeometry,
) -> Result<Vec<CudaWeight>, EngError> {
    // Refuse DeltaNet (including its incompatible head mapping) before any staging.
    layer_kind(table, layer, AttentionKind::Full)?;
    reader.projector.context().ensure_usable()?;
    // The MTP block is a full-attention layer by construction: only a model layer is looked up
    // in the config's `layer_types`.
    let typed_layer = (reader.root() == Root::Layers).then_some(layer);
    check_source_geometry(&reader.checkpoint.model_config, typed_layer, geometry)?;
    let dense = geometry.dense_specs(WeightLayout::Exl3, AttentionKind::Full)?;
    let raw = geometry.raw_specs(AttentionKind::Full)?;
    let mut weights = Vec::new();
    weights
        .try_reserve_exact(dense.len() + raw.len())
        .map_err(|error| {
            EngError::MemoryPlan(format!("EXL3 attention weight allocation: {error}"))
        })?;
    for spec in dense {
        let name = table.name(layer, spec.role)?.to_owned();
        let prefix = reader.root().name(layer, projection_suffix(spec.role)?);
        weights.push(reader.exl3(name, &prefix, &spec, None)?);
    }
    for spec in raw {
        let name = table.name(layer, spec.role)?.to_owned();
        let source = reader.root().name(layer, norm_suffix(spec.role)?);
        weights.push(reader.norm(name, &source, spec.values)?);
    }
    Ok(weights)
}

/// The projections, norms and small plain tensors of a DeltaNet mixer, with the value heads in
/// the order of the kernels.
pub(super) fn delta_weights(
    reader: &PackReader<'_>,
    table: &LayerTable,
    layer: usize,
    geometry: &MixerGeometry,
) -> Result<Vec<CudaWeight>, EngError> {
    layer_kind(table, layer, AttentionKind::DeltaNet)?;
    reader.projector.context().ensure_usable()?;
    check_delta_source(&reader.checkpoint.model_config, layer, geometry)?;
    let source = DeltaSource::new(reader, layer, &geometry.delta)?;
    let mut weights = Vec::new();
    for spec in geometry.dense_specs(WeightLayout::Exl3, AttentionKind::DeltaNet)? {
        let name = table.name(layer, spec.role)?.to_owned();
        weights.push(source.matrix(name, &spec)?);
    }
    for spec in geometry.raw_specs(AttentionKind::DeltaNet)? {
        let name = table.name(layer, spec.role)?.to_owned();
        weights.push(source.raw(name, &spec)?);
    }
    for spec in geometry.plain_specs(WeightLayout::Exl3, AttentionKind::DeltaNet) {
        let name = table.name(layer, spec.role)?.to_owned();
        weights.push(source.plain(name, &spec)?);
    }
    Ok(weights)
}

/// The moves of the value heads of a DeltaNet layer, as orders over heads and over 128-wide
/// groups.
#[derive(Debug, PartialEq, Eq)]
struct DeltaOrders {
    /// For each tiled head, the grouped head it takes.
    heads: Vec<usize>,
    /// Output groups of the fused q, k, v projection: q and k stay, v heads move.
    qkv: Vec<usize>,
    /// Groups of one value-head axis of width `heads * value_dim` (z output, out input).
    head_groups: Vec<usize>,
}

impl DeltaOrders {
    fn new(d: &DeltaGeometry) -> Result<Self, EngError> {
        if !d.value_dim.is_multiple_of(GROUP) || !d.channels.is_multiple_of(GROUP) {
            return Err(EngError::Checkpoint(format!(
                "EXL3 DeltaNet: value heads of {} channels do not span whole groups of {GROUP}",
                d.value_dim
            )));
        }
        let heads = tiled_head_order(d.key_heads, d.value_heads)?;
        let value_groups = head_groups(&heads, d.value_dim / GROUP);
        let lead = (d.channels - d.value_heads * d.value_dim) / GROUP;
        let qkv = (0..lead)
            .chain(value_groups.iter().map(|group| lead + group))
            .collect();
        Ok(Self {
            heads,
            qkv,
            head_groups: value_groups,
        })
    }
}

struct DeltaSource<'a, 'r> {
    reader: &'r PackReader<'a>,
    layer: usize,
    geometry: &'r DeltaGeometry,
    orders: DeltaOrders,
}

impl<'a, 'r> DeltaSource<'a, 'r> {
    fn new(
        reader: &'r PackReader<'a>,
        layer: usize,
        d: &'r DeltaGeometry,
    ) -> Result<Self, EngError> {
        Ok(Self {
            reader,
            layer,
            geometry: d,
            orders: DeltaOrders::new(d)?,
        })
    }

    fn name(&self, suffix: &str) -> String {
        layer_name(self.layer, suffix)
    }

    fn matrix(&self, name: String, spec: &DenseSpec) -> Result<CudaWeight, EngError> {
        let (suffix, reorder) = match spec.role {
            Role::AttnQkv => ("linear_attn.in_proj_qkv", (Axis::Output, &self.orders.qkv)),
            Role::AttnGate => (
                "linear_attn.in_proj_z",
                (Axis::Output, &self.orders.head_groups),
            ),
            Role::SsmOut => (
                "linear_attn.out_proj",
                (Axis::Input, &self.orders.head_groups),
            ),
            other => return Err(unsupported("projection", other)),
        };
        let reorder = Some((reorder.0, reorder.1.as_slice()));
        self.reader.exl3(name, &self.name(suffix), spec, reorder)
    }

    /// A plain `[heads, k]` projection with its rows in the order of the kernels.
    fn plain(&self, name: String, spec: &DenseSpec) -> Result<CudaWeight, EngError> {
        let suffix = match spec.role {
            Role::SsmBeta => "linear_attn.in_proj_b.weight",
            Role::SsmAlpha => "linear_attn.in_proj_a.weight",
            other => return Err(unsupported("plain projection", other)),
        };
        let shape = [spec.rows as u32, spec.k as u32];
        let rows = |values: Vec<f32>| permute_blocks(&values, spec.k, &self.orders.heads);
        self.reader.vector(name, &self.name(suffix), &shape, rows)
    }

    fn raw(&self, name: String, spec: &RawSpec) -> Result<CudaWeight, EngError> {
        let (reader, heads, d) = (self.reader, &self.orders.heads, self.geometry);
        let count = [spec.values as u32];
        let by_head = |values: Vec<f32>| permute_blocks(&values, 1, heads);
        match spec.role {
            Role::AttnNorm => reader.norm(name, &self.name("input_layernorm.weight"), spec.values),
            Role::PostAttentionNorm => reader.norm(
                name,
                &self.name("post_attention_layernorm.weight"),
                spec.values,
            ),
            Role::SsmNorm => reader.vector(name, &self.name("linear_attn.norm.weight"), &count, Ok),
            Role::SsmDtBias => {
                reader.vector(name, &self.name("linear_attn.dt_bias"), &count, by_head)
            }
            // The kernels read `-exp(A_log)` as a GGUF file stores it.
            Role::SsmA => reader.vector(name, &self.name("linear_attn.A_log"), &count, |log| {
                decay_from_log(&by_head(log)?)
            }),
            Role::SsmConv1d => {
                let taps = d.conv_kernel;
                let shape = [d.channels as u32, 1, taps as u32];
                let lead = (d.channels - d.value_heads * d.value_dim) * taps;
                let source = self.name("linear_attn.conv1d.weight");
                reader.vector(name, &source, &shape, |taps_by_channel| {
                    let moved =
                        permute_blocks(&taps_by_channel[lead..], d.value_dim * taps, heads)?;
                    Ok(taps_by_channel[..lead]
                        .iter()
                        .copied()
                        .chain(moved)
                        .collect())
                })
            }
            other => Err(unsupported("raw vector", other)),
        }
    }
}

fn unsupported(what: &str, role: Role) -> EngError {
    EngError::Checkpoint(format!(
        "EXL3 DeltaNet source: unsupported {what} role {role:?}"
    ))
}

fn check_delta_source(bytes: &[u8], layer: usize, g: &MixerGeometry) -> Result<(), EngError> {
    if bytes.len() as u64 > crate::weights::exl3::CONFIG_LIMIT {
        return Err(EngError::MemoryPlan(
            "EXL3 model config exceeds 1 MiB".into(),
        ));
    }
    let crate::checkpoint_json::UniqueValue(root) = serde_json::from_slice(bytes)
        .map_err(|error| EngError::Checkpoint(format!("EXL3 model config: {error}")))?;
    let text = root
        .get("text_config")
        .ok_or_else(|| EngError::Checkpoint("EXL3 DeltaNet source: missing text_config".into()))?;
    let d = &g.delta;
    let mismatch = || {
        EngError::Checkpoint(
            "EXL3 DeltaNet source: config does not match mixer geometry or linear-attention layer"
                .into(),
        )
    };
    let count = |key: &str| {
        text.get(key)
            .and_then(|v| v.as_u64())
            .and_then(|v| usize::try_from(v).ok())
    };
    let key_dim = count("linear_key_head_dim").ok_or_else(mismatch)?;
    let wanted = [
        ("hidden_size", g.hidden),
        ("linear_num_key_heads", d.key_heads),
        ("linear_num_value_heads", d.value_heads),
        ("linear_value_head_dim", d.value_dim),
        ("linear_conv_kernel_dim", d.conv_kernel),
    ];
    let layer_kind = text
        .get("layer_types")
        .and_then(|v| v.as_array())
        .and_then(|v| v.get(layer))
        .and_then(|v| v.as_str());
    let channels = 2 * d.key_heads * key_dim + d.value_heads * d.value_dim;
    if wanted.iter().any(|&(key, want)| count(key) != Some(want))
        || channels != d.channels
        || layer_kind != Some("linear_attention")
    {
        return Err(mismatch());
    }
    Ok(())
}

/// Refuse interpreting source attention with the caller's different geometry or epsilon.
fn check_source_geometry(
    bytes: &[u8],
    layer: Option<usize>,
    g: &MixerGeometry,
) -> Result<(), EngError> {
    if bytes.len() as u64 > crate::weights::exl3::CONFIG_LIMIT {
        return Err(EngError::MemoryPlan(
            "EXL3 model config exceeds 1 MiB".into(),
        ));
    }
    let crate::checkpoint_json::UniqueValue(root) = serde_json::from_slice(bytes)
        .map_err(|error| EngError::Checkpoint(format!("EXL3 model config: {error}")))?;
    let text = root
        .get("text_config")
        .ok_or_else(|| EngError::Checkpoint("EXL3 attention source: missing text_config".into()))?;
    let mismatch = || {
        EngError::Checkpoint(
            "EXL3 attention source: config does not match mixer geometry or full-attention layer"
                .into(),
        )
    };
    for (key, expected) in [
        ("hidden_size", g.hidden),
        ("head_dim", g.attn.head_dim),
        ("num_attention_heads", g.attn.heads_q),
        ("num_key_value_heads", g.attn.heads_kv),
    ] {
        if text
            .get(key)
            .and_then(|v| v.as_u64())
            .and_then(|v| usize::try_from(v).ok())
            != Some(expected)
        {
            return Err(mismatch());
        }
    }
    let epsilon = text
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .ok_or_else(mismatch)?;
    if !epsilon.is_finite()
        || epsilon <= 0.0
        || epsilon as f32 != g.eps
        || text.get("attn_output_gate").and_then(|v| v.as_bool()) != Some(true)
        || layer.is_some_and(|layer| {
            text.get("layer_types")
                .and_then(|v| v.as_array())
                .and_then(|v| v.get(layer))
                .and_then(|v| v.as_str())
                != Some("full_attention")
        })
    {
        return Err(mismatch());
    }
    let rope = text.get("rope_parameters").ok_or_else(mismatch)?;
    let theta = rope
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .ok_or_else(mismatch)?;
    let fraction = rope
        .get("partial_rotary_factor")
        .and_then(|v| v.as_f64())
        .ok_or_else(mismatch)?;
    let rotary = fraction * g.attn.head_dim as f64;
    if !theta.is_finite()
        || theta <= 0.0
        || theta as f32 != g.attn.rope_base
        || !fraction.is_finite()
        || fraction <= 0.0
        || fraction > 1.0
        || rotary != g.attn.n_rot as f64
        || rope.get("rope_type").and_then(|v| v.as_str()) != Some("default")
    {
        return Err(mismatch());
    }
    Ok(())
}

fn projection_suffix(role: Role) -> Result<&'static str, EngError> {
    match role {
        Role::AttnQ => Ok("self_attn.q_proj"),
        Role::AttnK => Ok("self_attn.k_proj"),
        Role::AttnV => Ok("self_attn.v_proj"),
        Role::AttnOutput => Ok("self_attn.o_proj"),
        _ => Err(EngError::Checkpoint(format!(
            "EXL3 attention source: unsupported projection role {role:?}"
        ))),
    }
}

fn norm_suffix(role: Role) -> Result<&'static str, EngError> {
    match role {
        Role::AttnNorm => Ok("input_layernorm.weight"),
        Role::PostAttentionNorm => Ok("post_attention_layernorm.weight"),
        Role::AttnQNorm => Ok("self_attn.q_norm.weight"),
        Role::AttnKNorm => Ok("self_attn.k_norm.weight"),
        _ => Err(EngError::Checkpoint(format!(
            "EXL3 attention source: unsupported norm role {role:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_config() -> (MixerGeometry, serde_json::Value) {
        let geometry = MixerGeometry::new(&super::super::common_tests::dims()).unwrap();
        let a = geometry.attn;
        let config = serde_json::json!({"text_config": {
            "hidden_size": geometry.hidden, "head_dim": a.head_dim,
            "num_attention_heads": a.heads_q, "num_key_value_heads": a.heads_kv,
            "rms_norm_eps": geometry.eps, "attn_output_gate": true,
            "layer_types": ["full_attention"],
            "rope_parameters": {"rope_theta": a.rope_base, "rope_type": "default",
                "partial_rotary_factor": a.n_rot as f64 / a.head_dim as f64}
        }});
        (geometry, config)
    }

    #[test]
    fn source_config_must_match_geometry_epsilon_and_layer_kind() {
        let (geometry, config) = source_config();
        let bytes = serde_json::to_vec(&config).unwrap();
        check_source_geometry(&bytes, Some(0), &geometry).unwrap();
        assert!(matches!(
            check_source_geometry(&bytes, Some(1), &geometry),
            Err(EngError::Checkpoint(_))
        ));
        for (key, wrong) in [
            ("hidden_size", serde_json::json!(0)),
            ("num_key_value_heads", serde_json::json!(0)),
            ("head_dim", serde_json::json!(128)),
            ("num_attention_heads", serde_json::json!(32)),
            ("rms_norm_eps", serde_json::json!(0.001)),
            ("attn_output_gate", serde_json::json!(false)),
            ("layer_types", serde_json::json!(["linear_attention"])),
        ] {
            let mut bad = config.clone();
            bad["text_config"][key] = wrong;
            assert!(matches!(
                check_source_geometry(&serde_json::to_vec(&bad).unwrap(), Some(0), &geometry),
                Err(EngError::Checkpoint(_))
            ));
        }
    }

    #[test]
    fn source_config_refuses_oversize_duplicates_and_missing_fields() {
        let (geometry, config) = source_config();
        let mut oversized = serde_json::to_vec(&config).unwrap();
        oversized.resize(1024 * 1024 + 1, b' ');
        assert!(matches!(
            check_source_geometry(&oversized, Some(0), &geometry),
            Err(EngError::MemoryPlan(_))
        ));
        let text = serde_json::to_string(&config["text_config"]).unwrap();
        let duplicate = format!(r#"{{"text_config":{text},"text_config":{text}}}"#);
        for bytes in [b"{".as_slice(), b"{}".as_slice(), duplicate.as_bytes()] {
            assert!(matches!(
                check_source_geometry(bytes, Some(0), &geometry),
                Err(EngError::Checkpoint(_))
            ));
        }
        for key in [
            "hidden_size",
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "rms_norm_eps",
            "attn_output_gate",
            "layer_types",
            "rope_parameters",
        ] {
            let mut bad = config.clone();
            bad["text_config"].as_object_mut().unwrap().remove(key);
            assert!(
                matches!(
                    check_source_geometry(&serde_json::to_vec(&bad).unwrap(), Some(0), &geometry),
                    Err(EngError::Checkpoint(_))
                ),
                "missing {key}"
            );
        }
    }

    #[test]
    fn source_config_refuses_missing_or_incompatible_rotary_parameters() {
        let (geometry, config) = source_config();
        let fraction = config["text_config"]["rope_parameters"]["partial_rotary_factor"]
            .as_f64()
            .unwrap();
        for (key, wrong) in [
            ("rope_theta", serde_json::json!(0)),
            (
                "rope_theta",
                serde_json::json!(geometry.attn.rope_base as f64 * 2.0),
            ),
            ("partial_rotary_factor", serde_json::json!(0)),
            ("partial_rotary_factor", serde_json::json!(2)),
            ("partial_rotary_factor", serde_json::json!(fraction / 2.0)),
            ("rope_type", serde_json::json!("dynamic")),
        ] {
            let mut bad = config.clone();
            bad["text_config"]["rope_parameters"][key] = wrong;
            assert!(
                matches!(
                    check_source_geometry(&serde_json::to_vec(&bad).unwrap(), Some(0), &geometry),
                    Err(EngError::Checkpoint(_))
                ),
                "wrong {key}"
            );
        }
        for key in ["rope_theta", "partial_rotary_factor", "rope_type"] {
            let mut bad = config.clone();
            bad["text_config"]["rope_parameters"]
                .as_object_mut()
                .unwrap()
                .remove(key);
            assert!(
                matches!(
                    check_source_geometry(&serde_json::to_vec(&bad).unwrap(), Some(0), &geometry),
                    Err(EngError::Checkpoint(_))
                ),
                "missing {key}"
            );
        }
    }

    #[test]
    fn maps_only_full_attention_source_roles() {
        for (role, suffix) in [
            (Role::AttnQ, "self_attn.q_proj"),
            (Role::AttnK, "self_attn.k_proj"),
            (Role::AttnV, "self_attn.v_proj"),
            (Role::AttnOutput, "self_attn.o_proj"),
        ] {
            assert_eq!(projection_suffix(role).unwrap(), suffix);
        }
        for (role, suffix) in [
            (Role::AttnNorm, "input_layernorm.weight"),
            (Role::PostAttentionNorm, "post_attention_layernorm.weight"),
            (Role::AttnQNorm, "self_attn.q_norm.weight"),
            (Role::AttnKNorm, "self_attn.k_norm.weight"),
        ] {
            assert_eq!(norm_suffix(role).unwrap(), suffix);
        }
        assert!(matches!(
            projection_suffix(Role::AttnQkv),
            Err(EngError::Checkpoint(_))
        ));
        assert!(matches!(
            norm_suffix(Role::SsmNorm),
            Err(EngError::Checkpoint(_))
        ));
    }

    fn delta_geometry() -> MixerGeometry {
        MixerGeometry::new(&super::super::common_tests::dims()).unwrap()
    }

    #[test]
    fn delta_orders_move_only_the_value_heads_of_the_qkv_columns() {
        let geometry = delta_geometry();
        let orders = DeltaOrders::new(&geometry.delta).unwrap();
        let d = geometry.delta;
        assert_eq!(orders.heads.len(), d.value_heads);
        // q and k are 2 * 16 heads of 128, one group each, and stay where they are.
        let lead = 2 * d.key_heads;
        assert_eq!(orders.qkv[..lead], (0..lead).collect::<Vec<_>>());
        assert_eq!(orders.qkv.len(), d.channels / GROUP);
        let moved: Vec<_> = orders.qkv[lead..].iter().map(|g| g - lead).collect();
        assert_eq!(moved, orders.heads);
        assert_eq!(orders.head_groups, orders.heads);
        let mut sorted = orders.qkv.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..d.channels / GROUP).collect::<Vec<_>>());
    }

    #[test]
    fn delta_orders_expand_a_wide_head_into_whole_groups() {
        let mut d = delta_geometry().delta;
        d.value_dim = 2 * GROUP;
        d.channels = 2 * d.key_heads * GROUP + d.value_heads * d.value_dim;
        let orders = DeltaOrders::new(&d).unwrap();
        assert_eq!(orders.head_groups.len(), 2 * d.value_heads);
        assert_eq!(orders.head_groups[..4], [0, 1, 4, 5]);
        d.value_dim = 96;
        assert!(matches!(DeltaOrders::new(&d), Err(EngError::Checkpoint(_))));
    }

    fn delta_config() -> (MixerGeometry, serde_json::Value) {
        let geometry = delta_geometry();
        let d = geometry.delta;
        let config = serde_json::json!({"text_config": {
            "hidden_size": geometry.hidden,
            "linear_num_key_heads": d.key_heads,
            "linear_num_value_heads": d.value_heads,
            "linear_key_head_dim": (d.channels - d.value_heads * d.value_dim) / (2 * d.key_heads),
            "linear_value_head_dim": d.value_dim,
            "linear_conv_kernel_dim": d.conv_kernel,
            "layer_types": ["linear_attention", "full_attention"],
        }});
        (geometry, config)
    }

    #[test]
    fn delta_source_config_must_match_geometry_and_layer_kind() {
        let (geometry, config) = delta_config();
        let check = |value: &serde_json::Value, layer| {
            check_delta_source(&serde_json::to_vec(value).unwrap(), layer, &geometry)
        };
        check(&config, 0).unwrap();
        assert!(check(&config, 1).is_err(), "an attention layer");
        assert!(check(&config, 2).is_err(), "a layer past the list");
        for key in [
            "hidden_size",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
        ] {
            let mut wrong = config.clone();
            wrong["text_config"][key] = serde_json::json!(3);
            assert!(check(&wrong, 0).is_err(), "wrong {key}");
            wrong["text_config"].as_object_mut().unwrap().remove(key);
            assert!(check(&wrong, 0).is_err(), "missing {key}");
        }
        assert!(matches!(
            check_delta_source(&vec![b' '; 1024 * 1024 + 1], 0, &geometry),
            Err(EngError::MemoryPlan(_))
        ));
    }

    #[test]
    fn plain_gates_leave_the_dense_specs_of_an_exl3_delta_layer() {
        let geometry = delta_geometry();
        let roles = |specs: Vec<DenseSpec>| specs.iter().map(|s| s.role).collect::<Vec<_>>();
        let exl3 = geometry
            .dense_specs(WeightLayout::Exl3, AttentionKind::DeltaNet)
            .unwrap();
        assert_eq!(
            roles(exl3),
            [Role::AttnQkv, Role::AttnGate, Role::SsmOut],
            "the 32-row gates are not EXL3 matrices"
        );
        let plain = geometry.plain_specs(WeightLayout::Exl3, AttentionKind::DeltaNet);
        assert_eq!(roles(plain.clone()), [Role::SsmBeta, Role::SsmAlpha]);
        assert!(plain.iter().all(|s| s.rows == 32 && s.k == geometry.hidden));
        // Other layouts and kinds keep every projection quantized.
        assert!(
            geometry
                .plain_specs(WeightLayout::Native, AttentionKind::DeltaNet)
                .is_empty()
        );
        assert!(
            geometry
                .plain_specs(WeightLayout::Exl3, AttentionKind::Full)
                .is_empty()
        );
        let native = geometry
            .dense_specs(WeightLayout::Native, AttentionKind::DeltaNet)
            .unwrap();
        assert_eq!(native.len(), 5);
    }
}
