//! The GGUF tensor inventory of one family: name, role, allowed types, dims, load path.
//!
//! Pure data derived from a `FamilySpec`. This module never opens or reads a file.

use std::{collections::HashMap, sync::LazyLock};

use crate::{
    EngError,
    family::{AttentionKind, FamilySpec, LinearAttention},
    weights::GgufType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorPath {
    /// Quantized matrix that the CUDA projector tiles on the device.
    DeviceTiled,
    /// Small F32 tensor that the CUDA loader uploads as a raw device copy.
    DeviceF32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    TokenEmbd,
    OutputNorm,
    Output,
    AttnNorm,
    PostAttentionNorm,
    FfnGateInp,
    FfnGateInpShexp,
    FfnGateExps,
    FfnUpExps,
    FfnDownExps,
    FfnGateShexp,
    FfnUpShexp,
    FfnDownShexp,
    AttnQ,
    AttnK,
    AttnV,
    AttnOutput,
    AttnQNorm,
    AttnKNorm,
    AttnQkv,
    AttnGate,
    SsmOut,
    SsmAlpha,
    SsmBeta,
    SsmA,
    SsmConv1d,
    SsmDtBias,
    SsmNorm,
}

impl Role {
    /// Short projection tag matched by `FIFI_CUDA_DECODE_DQ_ROLES`.
    #[cfg(feature = "native-cuda")]
    pub(crate) fn dq_tag(self) -> &'static str {
        match self {
            Self::AttnQ => "attn_q",
            Self::AttnK => "attn_k",
            Self::AttnV => "attn_v",
            Self::AttnOutput => "attn_output",
            Self::AttnQkv => "attn_qkv",
            Self::AttnGate => "attn_gate",
            Self::SsmAlpha => "ssm_alpha",
            Self::SsmBeta => "ssm_beta",
            Self::SsmOut => "ssm_out",
            Self::Output => "output",
            _ => "",
        }
    }
}

const COMMON_ROLES: [Role; 10] = [
    Role::AttnNorm,
    Role::PostAttentionNorm,
    Role::FfnGateInp,
    Role::FfnGateInpShexp,
    Role::FfnGateExps,
    Role::FfnUpExps,
    Role::FfnDownExps,
    Role::FfnGateShexp,
    Role::FfnUpShexp,
    Role::FfnDownShexp,
];
const FULL_ROLES: [Role; 6] = [
    Role::AttnQ,
    Role::AttnK,
    Role::AttnV,
    Role::AttnOutput,
    Role::AttnQNorm,
    Role::AttnKNorm,
];
const DELTA_ROLES: [Role; 9] = [
    Role::AttnQkv,
    Role::AttnGate,
    Role::SsmOut,
    Role::SsmAlpha,
    Role::SsmBeta,
    Role::SsmA,
    Role::SsmConv1d,
    Role::SsmDtBias,
    Role::SsmNorm,
];
// Every quantized role (dense projections, routed experts, the embedding) takes every type the
// native kernels read (`GgufType::has_native_kernel`): the decode matvec, the expert matvec,
// the prefill GEMMs and the embedding gather all dispatch on the type, so no role has a list of
// its own. A dense role also takes the plain float types, which the load encodes to Q8_0 blocks
// (`weights::encode_plain_float_rows`); the routed expert stacks do not, their tiers read the
// file's own records. The order is the order the refusal text names them in.
static NATIVE_TYPES: LazyLock<Vec<GgufType>> = LazyLock::new(GgufType::native_types);
static DENSE_TYPES: LazyLock<Vec<GgufType>> = LazyLock::new(|| {
    let mut types = GgufType::native_types();
    types.extend([GgufType::F32, GgufType::F16, GgufType::Bf16]);
    types
});
const RAW_TYPES: [GgufType; 1] = [GgufType::F32];
impl Role {
    /// The GGUF name of the tensor of this role: `blk.N.` and the suffix for a layer's role, the
    /// whole name for a global one.
    pub(crate) fn tensor_name(self, layer: Option<usize>) -> String {
        match layer {
            Some(layer) => format!("blk.{layer}.{}", self.suffix()),
            None => self.suffix().to_owned(),
        }
    }

    /// Whole tensor name for the three global roles, else the part after `blk.N.`.
    fn suffix(self) -> &'static str {
        match self {
            Role::TokenEmbd => "token_embd.weight",
            Role::OutputNorm => "output_norm.weight",
            Role::Output => "output.weight",
            Role::AttnNorm => "attn_norm.weight",
            Role::PostAttentionNorm => "post_attention_norm.weight",
            Role::FfnGateInp => "ffn_gate_inp.weight",
            Role::FfnGateInpShexp => "ffn_gate_inp_shexp.weight",
            Role::FfnGateExps => "ffn_gate_exps.weight",
            Role::FfnUpExps => "ffn_up_exps.weight",
            Role::FfnDownExps => "ffn_down_exps.weight",
            Role::FfnGateShexp => "ffn_gate_shexp.weight",
            Role::FfnUpShexp => "ffn_up_shexp.weight",
            Role::FfnDownShexp => "ffn_down_shexp.weight",
            Role::AttnQ => "attn_q.weight",
            Role::AttnK => "attn_k.weight",
            Role::AttnV => "attn_v.weight",
            Role::AttnOutput => "attn_output.weight",
            Role::AttnQNorm => "attn_q_norm.weight",
            Role::AttnKNorm => "attn_k_norm.weight",
            Role::AttnQkv => "attn_qkv.weight",
            Role::AttnGate => "attn_gate.weight",
            Role::SsmOut => "ssm_out.weight",
            Role::SsmAlpha => "ssm_alpha.weight",
            Role::SsmBeta => "ssm_beta.weight",
            Role::SsmA => "ssm_a",
            Role::SsmConv1d => "ssm_conv1d.weight",
            Role::SsmDtBias => "ssm_dt.bias",
            Role::SsmNorm => "ssm_norm.weight",
        }
    }

    /// The role decides the path, not the file type, so a Q8_0 file and a Q4_K file of the same
    /// family yield the same inventory; the file type is checked against `allowed_types`.
    fn path(self) -> TensorPath {
        if matches!(
            self,
            Role::OutputNorm
                | Role::AttnNorm
                | Role::PostAttentionNorm
                | Role::FfnGateInp
                | Role::FfnGateInpShexp
                | Role::AttnQNorm
                | Role::AttnKNorm
                | Role::SsmA
                | Role::SsmConv1d
                | Role::SsmDtBias
                | Role::SsmNorm
        ) {
            TensorPath::DeviceF32
        } else {
            TensorPath::DeviceTiled
        }
    }

    fn is_expert_stack(self) -> bool {
        matches!(
            self,
            Role::FfnGateExps | Role::FfnUpExps | Role::FfnDownExps
        )
    }

    /// A dense projection: a quantized matrix that is not a routed expert stack.
    fn is_dense_matrix(self) -> bool {
        self.path() == TensorPath::DeviceTiled && !self.is_expert_stack()
    }

    fn allowed_types(self) -> &'static [GgufType] {
        match self.path() {
            TensorPath::DeviceTiled if self.is_expert_stack() => NATIVE_TYPES.as_slice(),
            TensorPath::DeviceTiled => DENSE_TYPES.as_slice(),
            TensorPath::DeviceF32 => &RAW_TYPES,
        }
    }

    /// Dims in GGUF order: `dims[0]` is the row length; the other dims count the rows.
    fn dims(self, spec: &FamilySpec) -> Result<Vec<u64>, EngError> {
        let hidden = u64::from(spec.hidden);
        let vocab = u64::from(spec.vocab);
        let experts = u64::from(spec.experts);
        let moe_inter = u64::from(spec.moe_inter);
        let shared_inter = u64::from(spec.shared_inter);
        let head_dim = u64::from(spec.head_dim);
        let q_width = u64::from(spec.heads_q) * u64::from(spec.head_dim);
        let kv_width = u64::from(spec.heads_kv) * u64::from(spec.head_dim);
        Ok(match self {
            Role::TokenEmbd | Role::Output => vec![hidden, vocab],
            Role::OutputNorm | Role::AttnNorm | Role::PostAttentionNorm | Role::FfnGateInpShexp => {
                vec![hidden]
            }
            Role::FfnGateInp => vec![hidden, experts],
            Role::FfnGateExps | Role::FfnUpExps => vec![hidden, moe_inter, experts],
            Role::FfnDownExps => vec![moe_inter, hidden, experts],
            Role::FfnGateShexp | Role::FfnUpShexp => vec![hidden, shared_inter],
            Role::FfnDownShexp => vec![shared_inter, hidden],
            // The query projection also carries the output gate; saturating so a hostile
            // spec cannot overflow and never equals a real dim.
            Role::AttnQ => vec![hidden, q_width.saturating_mul(2)],
            Role::AttnK | Role::AttnV => vec![hidden, kv_width],
            Role::AttnOutput => vec![q_width, hidden],
            Role::AttnQNorm | Role::AttnKNorm => vec![head_dim],
            Role::AttnQkv => vec![hidden, u64::from(linear(spec)?.conv_dim()?)],
            Role::AttnGate => vec![hidden, u64::from(linear(spec)?.value_dim()?)],
            Role::SsmOut => vec![u64::from(linear(spec)?.value_dim()?), hidden],
            Role::SsmAlpha | Role::SsmBeta => vec![hidden, u64::from(linear(spec)?.value_heads)],
            Role::SsmA | Role::SsmDtBias => vec![u64::from(linear(spec)?.value_heads)],
            Role::SsmConv1d => vec![
                u64::from(linear(spec)?.conv_kernel),
                u64::from(linear(spec)?.conv_dim()?),
            ],
            Role::SsmNorm => vec![u64::from(linear(spec)?.value_head_dim)],
        })
    }
}

fn linear(spec: &FamilySpec) -> Result<&LinearAttention, EngError> {
    spec.linear.as_ref().ok_or_else(|| {
        EngError::Checkpoint(format!(
            "family {} has DeltaNet layers but no linear attention block",
            spec.name
        ))
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExpectedTensor {
    /// GGUF tensor name, for example `blk.3.attn_q.weight`.
    pub(crate) name: String,
    pub(crate) role: Role,
    pub(crate) dims: Vec<u64>,
}

impl ExpectedTensor {
    pub(crate) fn path(&self) -> TensorPath {
        self.role.path()
    }

    pub(crate) fn is_dense_matrix(&self) -> bool {
        self.role.is_dense_matrix()
    }

    pub(crate) fn allowed_types(&self) -> &'static [GgufType] {
        self.role.allowed_types()
    }

    pub(crate) fn byte_len(&self, ty: GgufType) -> Result<u64, EngError> {
        let block = ty.block_elems();
        let row = self.dims[0];
        if !row.is_multiple_of(block) {
            return Err(EngError::Checkpoint(format!(
                "tensor {}: row length {row} is not a multiple of the {} block size {block}",
                self.name,
                ty.name()
            )));
        }
        self.dims
            .iter()
            .try_fold(1u64, |total, &dim| total.checked_mul(dim))
            .map(|elements| elements / block)
            .and_then(|blocks| blocks.checked_mul(ty.block_bytes()))
            .ok_or_else(|| {
                EngError::Checkpoint(format!("tensor {}: size overflows u64", self.name))
            })
    }
}

/// The tensors a GGUF file of one family must hold, in file-independent layer order.
#[derive(Debug)]
pub(crate) struct Inventory {
    tensors: Vec<ExpectedTensor>,
    index: HashMap<String, usize>,
}

impl Inventory {
    pub(crate) fn for_spec(spec: &FamilySpec) -> Result<Self, EngError> {
        if spec.layers_attention.len() != spec.layers as usize {
            return Err(EngError::Checkpoint(format!(
                "family {} lists {} attention kinds for {} layers",
                spec.name,
                spec.layers_attention.len(),
                spec.layers
            )));
        }
        let mut tensors = Vec::new();
        tensors.push(expected(spec, Role::TokenEmbd, None)?);
        for (layer, kind) in spec.layers_attention.iter().enumerate() {
            let mixer = match kind {
                AttentionKind::Full => FULL_ROLES.as_slice(),
                AttentionKind::Sliding => {
                    return Err(EngError::Checkpoint(format!(
                        "family {} layer {layer}: attention kind Sliding has no GGUF tensors",
                        spec.name
                    )));
                }
                AttentionKind::DeltaNet => DELTA_ROLES.as_slice(),
                AttentionKind::Qsa => {
                    return Err(EngError::Checkpoint(format!(
                        "family {} layer {layer}: attention kind Qsa has no GGUF tensors",
                        spec.name
                    )));
                }
            };
            for &role in COMMON_ROLES.iter().chain(mixer) {
                tensors.push(expected(spec, role, Some(layer))?);
            }
        }
        tensors.push(expected(spec, Role::OutputNorm, None)?);
        tensors.push(expected(spec, Role::Output, None)?);
        let index: HashMap<String, usize> = tensors
            .iter()
            .enumerate()
            .map(|(position, tensor)| (tensor.name.clone(), position))
            .collect();
        Ok(Inventory { tensors, index })
    }

    pub(crate) fn tensors(&self) -> &[ExpectedTensor] {
        &self.tensors
    }

    pub(crate) fn get(&self, name: &str) -> Option<&ExpectedTensor> {
        self.index
            .get(name)
            .map(|&position| &self.tensors[position])
    }
}

fn expected(
    spec: &FamilySpec,
    role: Role,
    layer: Option<usize>,
) -> Result<ExpectedTensor, EngError> {
    let name = role.tensor_name(layer);
    let dims = role.dims(spec)?;
    Ok(ExpectedTensor { name, role, dims })
}

#[cfg(test)]
pub(super) mod tests;
