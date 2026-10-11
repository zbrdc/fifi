use super::quantization::ModuleQuantization;
use super::spec::{AttentionKind, FamilySpec};
use super::weight_plan::WeightPlan;
use crate::EngError;
use crate::kernels::{PackedWeight, QuantizationMode, QuantizationSpec};
use crate::tensor::{Dtype, Tensor};
use crate::weights::WeightSource;

/// One checked packed projection or expert stack.
///
/// The constructor validates payload, scale, and bias storage against the logical shape and the
/// quantization the weight plan resolved. The checked `PackedWeight` is the only stored state.
#[derive(Debug)]
pub struct ProjectionWeights {
    packed: PackedWeight,
}

impl ProjectionWeights {
    pub fn packed(&self) -> &PackedWeight {
        &self.packed
    }
}

pub struct ExpertStack {
    pub gate: ProjectionWeights,
    pub up: ProjectionWeights,
    pub down: ProjectionWeights,
}

pub struct SharedExpertWeights {
    pub shared_gate: ProjectionWeights,
    pub gate: ProjectionWeights,
    pub up: ProjectionWeights,
    pub down: ProjectionWeights,
}

pub struct AttentionWeights {
    pub q: ProjectionWeights,
    pub k: ProjectionWeights,
    pub v: ProjectionWeights,
    pub o: ProjectionWeights,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
}

pub struct DeltaWeights {
    pub in_proj_qkv: ProjectionWeights,
    pub in_proj_z: ProjectionWeights,
    pub in_proj_a: ProjectionWeights,
    pub in_proj_b: ProjectionWeights,
    pub out_proj: ProjectionWeights,
    pub conv1d: Tensor,
    pub a_log: Tensor,
    pub dt_bias: Tensor,
    pub norm: Tensor,
}

pub enum MixerWeights {
    Attention(Box<AttentionWeights>),
    DeltaNet(Box<DeltaWeights>),
}

pub struct LayerWeights {
    pub input_norm: Tensor,
    pub post_attention_norm: Tensor,
    pub router: ProjectionWeights,
    pub experts: ExpertStack,
    pub shared: SharedExpertWeights,
    pub mixer: MixerWeights,
}

impl LayerWeights {
    pub fn resolve<S: WeightSource + ?Sized>(
        source: &S,
        spec: &FamilySpec,
        layer: u32,
        plan: &WeightPlan,
    ) -> Result<Self, EngError> {
        let kind = spec
            .layers_attention
            .get(layer as usize)
            .copied()
            .ok_or_else(|| EngError::Checkpoint(format!("layer {layer} is out of range")))?;
        let loader = Loader::new(source, spec, layer, plan);
        let load_mixer: fn(&Loader<'_, S>) -> Result<MixerWeights, EngError> = match kind {
            AttentionKind::Full => |l| Ok(MixerWeights::Attention(Box::new(l.attention()?))),
            AttentionKind::DeltaNet => |l| Ok(MixerWeights::DeltaNet(Box::new(l.delta()?))),
            AttentionKind::Qsa => {
                return Err(EngError::Checkpoint(format!(
                    "layer {layer} uses unsupported qsa attention"
                )));
            }
        };
        let (router, experts, shared) = loader.moe()?;
        let mixer = load_mixer(&loader)?;
        Ok(Self {
            input_norm: loader.vector("input_layernorm.weight", spec.hidden)?,
            post_attention_norm: loader.vector("post_attention_layernorm.weight", spec.hidden)?,
            router,
            experts,
            shared,
            mixer,
        })
    }
}

struct Loader<'a, S: WeightSource + ?Sized> {
    source: &'a S,
    spec: &'a FamilySpec,
    plan: &'a WeightPlan,
    stem: String,
}

impl<'a, S: WeightSource + ?Sized> Loader<'a, S> {
    fn new(source: &'a S, spec: &'a FamilySpec, layer: u32, plan: &'a WeightPlan) -> Self {
        Self {
            source,
            spec,
            plan,
            stem: format!("language_model.model.layers.{layer}."),
        }
    }

    fn tensor(&self, name: &str, shape: &[u32]) -> Result<Tensor, EngError> {
        let key = format!("{}{name}", self.stem);
        let tensor = self
            .source
            .tensor(&key)
            .ok_or_else(|| EngError::Checkpoint(format!("missing tensor {key}")))?;
        if tensor.shape() != shape {
            return Err(EngError::Checkpoint(format!(
                "tensor {key} has shape {:?}, expected {shape:?}",
                tensor.shape()
            )));
        }
        Ok(tensor)
    }

    fn dense_tensor(&self, name: &str, shape: &[u32]) -> Result<Tensor, EngError> {
        let tensor = self.tensor(name, shape)?;
        check_floating(&format!("{}{name}", self.stem), &tensor)?;
        Ok(tensor)
    }

    fn vector(&self, name: &str, len: u32) -> Result<Tensor, EngError> {
        self.dense_tensor(name, &[len])
    }

    fn projection_quantization(&self, name: &str) -> Result<QuantizationSpec, EngError> {
        let key = format!("{}{name}.weight", self.stem);
        let ModuleQuantization::Quantized(quantization) = self
            .plan
            .quantization(&key)
            .ok_or_else(|| EngError::Checkpoint(format!("weight plan lacks {key}")))?
        else {
            return Err(EngError::Checkpoint(format!(
                "weight key {key} is dense; a packed projection requires quantization"
            )));
        };
        if quantization.mode != QuantizationMode::Affine {
            return Err(EngError::Checkpoint(format!(
                "weight key {key} uses unsupported {:?} quantization; affine weights are required",
                quantization.mode
            )));
        }
        Ok(quantization)
    }

    fn projection(
        &self,
        name: &str,
        rows: u32,
        in_dim: u32,
    ) -> Result<ProjectionWeights, EngError> {
        let quantization = self.projection_quantization(name)?;
        let weight_cols = packed_width(in_dim, quantization.bits)?;
        let scale_cols = scale_width(in_dim, quantization.group_size)?;
        let weight = self.tensor(&format!("{name}.weight"), &[rows, weight_cols])?;
        let scales = self.tensor(&format!("{name}.scales"), &[rows, scale_cols])?;
        let biases = self.tensor(&format!("{name}.biases"), &[rows, scale_cols])?;
        checked_packed(
            &format!("{}{name}.weight", self.stem),
            weight,
            scales,
            biases,
            vec![rows, in_dim],
            quantization,
        )
    }

    fn expert_stack(
        &self,
        name: &str,
        rows: u32,
        in_dim: u32,
    ) -> Result<ProjectionWeights, EngError> {
        let experts = self.spec.experts;
        let quantization = self.projection_quantization(name)?;
        let weight_cols = packed_width(in_dim, quantization.bits)?;
        let scale_cols = scale_width(in_dim, quantization.group_size)?;
        let weight = self.tensor(&format!("{name}.weight"), &[experts, rows, weight_cols])?;
        let scales = self.tensor(&format!("{name}.scales"), &[experts, rows, scale_cols])?;
        let biases = self.tensor(&format!("{name}.biases"), &[experts, rows, scale_cols])?;
        checked_packed(
            &format!("{}{name}.weight", self.stem),
            weight,
            scales,
            biases,
            vec![experts, rows, in_dim],
            quantization,
        )
    }

    fn moe(&self) -> Result<(ProjectionWeights, ExpertStack, SharedExpertWeights), EngError> {
        let spec = self.spec;
        let router = self.projection("mlp.gate", spec.experts, spec.hidden)?;
        let experts = ExpertStack {
            gate: self.expert_stack("mlp.switch_mlp.gate_proj", spec.moe_inter, spec.hidden)?,
            up: self.expert_stack("mlp.switch_mlp.up_proj", spec.moe_inter, spec.hidden)?,
            down: self.expert_stack("mlp.switch_mlp.down_proj", spec.hidden, spec.moe_inter)?,
        };
        let shared = SharedExpertWeights {
            shared_gate: self.projection("mlp.shared_expert_gate", 1, spec.hidden)?,
            gate: self.projection(
                "mlp.shared_expert.gate_proj",
                spec.shared_inter,
                spec.hidden,
            )?,
            up: self.projection("mlp.shared_expert.up_proj", spec.shared_inter, spec.hidden)?,
            down: self.projection(
                "mlp.shared_expert.down_proj",
                spec.hidden,
                spec.shared_inter,
            )?,
        };
        Ok((router, experts, shared))
    }

    fn attention(&self) -> Result<AttentionWeights, EngError> {
        let spec = self.spec;
        let kv_rows = mul_dims(spec.heads_kv, spec.head_dim)?;
        // The query projection also carries the output gate rows.
        let q_rows = mul_dims(mul_dims(spec.heads_q, spec.head_dim)?, 2)?;
        let o_in = mul_dims(spec.heads_q, spec.head_dim)?;
        Ok(AttentionWeights {
            q: self.projection("self_attn.q_proj", q_rows, spec.hidden)?,
            k: self.projection("self_attn.k_proj", kv_rows, spec.hidden)?,
            v: self.projection("self_attn.v_proj", kv_rows, spec.hidden)?,
            o: self.projection("self_attn.o_proj", spec.hidden, o_in)?,
            q_norm: self.vector("self_attn.q_norm.weight", spec.head_dim)?,
            k_norm: self.vector("self_attn.k_norm.weight", spec.head_dim)?,
        })
    }

    fn delta(&self) -> Result<DeltaWeights, EngError> {
        let spec = self.spec;
        let hidden = spec.hidden;
        let linear = spec.linear.ok_or_else(|| {
            EngError::Checkpoint(format!(
                "family {} declares DeltaNet layers without linear attention dimensions",
                spec.name
            ))
        })?;
        linear.validate()?;
        let heads = linear.value_heads;
        let value_dim = linear.value_dim()?;
        let conv_dim = linear.conv_dim()?;
        Ok(DeltaWeights {
            in_proj_qkv: self.projection("linear_attn.in_proj_qkv", conv_dim, hidden)?,
            in_proj_z: self.projection("linear_attn.in_proj_z", value_dim, hidden)?,
            in_proj_a: self.projection("linear_attn.in_proj_a", heads, hidden)?,
            in_proj_b: self.projection("linear_attn.in_proj_b", heads, hidden)?,
            out_proj: self.projection("linear_attn.out_proj", hidden, value_dim)?,
            conv1d: self.dense_tensor(
                "linear_attn.conv1d.weight",
                &[conv_dim, linear.conv_kernel, 1],
            )?,
            a_log: self.vector("linear_attn.A_log", heads)?,
            dt_bias: self.vector("linear_attn.dt_bias", heads)?,
            norm: self.vector("linear_attn.norm.weight", linear.value_head_dim)?,
        })
    }
}

fn checked_packed(
    key: &str,
    weight: Tensor,
    scales: Tensor,
    biases: Tensor,
    logical_shape: Vec<u32>,
    quantization: QuantizationSpec,
) -> Result<ProjectionWeights, EngError> {
    let packed = PackedWeight::try_new(weight, scales, Some(biases), logical_shape, quantization)
        .map_err(|err| match err {
        EngError::Checkpoint(message) => EngError::Checkpoint(format!("{key}: {message}")),
        err => err,
    })?;
    Ok(ProjectionWeights { packed })
}

fn check_floating(key: &str, tensor: &Tensor) -> Result<(), EngError> {
    if !matches!(tensor.dtype(), Dtype::F16 | Dtype::Bf16 | Dtype::F32) {
        return Err(EngError::Checkpoint(format!(
            "tensor {key} has dtype {:?}, expected a floating dtype",
            tensor.dtype()
        )));
    }
    Ok(())
}

fn mul_dims(a: u32, b: u32) -> Result<u32, EngError> {
    a.checked_mul(b)
        .ok_or_else(|| EngError::Checkpoint("tensor dimensions overflow".to_string()))
}

fn packed_width(in_dim: u32, bits: u32) -> Result<u32, EngError> {
    let total = mul_dims(in_dim, bits)?;
    if !total.is_multiple_of(32) {
        return Err(EngError::Checkpoint(format!(
            "{in_dim} inputs at {bits} bits do not pack into u32 words"
        )));
    }
    Ok(total / 32)
}

fn scale_width(in_dim: u32, group: u32) -> Result<u32, EngError> {
    if group == 0 || !in_dim.is_multiple_of(group) {
        return Err(EngError::Checkpoint(format!(
            "{in_dim} inputs do not divide into groups of {group}"
        )));
    }
    Ok(in_dim / group)
}
