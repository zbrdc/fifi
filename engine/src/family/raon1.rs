//! GGUF model opener for Raon-1-mini-preview (`general.architecture` = `raon1`).
//!
//! The open is strict, like the Qwen3.6 one: a tensor of any type the role does not
//! allow, an extra or missing tensor, or a wrong shape fails and names the tensor or
//! key. There is no skip mode. This is the CPU host path: it builds the full
//! 731-tensor inventory of the published Q4_K_M file and reads tensors back as host
//! F32. The host reference and the serving path are not written yet; only tests call
//! the opener, and no CUDA load path exists, so no tile rules apply.

use std::path::{Path, PathBuf};

use crate::{
    EngError,
    weights::{GgufHeader, GgufLimits, GgufType, GgufWeights, MetaValue, dequantize},
};

/// The Q4_K_M conversion's largest tensor is the untied head: Q6_K, 2560 by 259,840,
/// 545,664,000 bytes of source data. One gibibyte bounds each raw tensor read; the
/// dequantized f32 output can be larger (the embedding's read_f32 allocates about
/// 2.66 GB).
const LIMITS: GgufLimits = GgufLimits {
    max_header_bytes: 100_000_000,
    max_tensor_bytes: 1 << 30,
};

/// Which attention a layer uses. MLA layers sit at indexes 3, 7, 11, 15, 19, 23, 24
/// (the stack ends with two of them); the other 18 are Raon Delta Attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaonKind {
    Rda,
    Mla,
}

/// Geometry of one raon1 checkpoint. All dims in GGUF order: `dims[0]` is the row
/// length. `RAON_1_MINI_PREVIEW` holds the published model; tests build tiny members.
#[derive(Debug, Clone, Copy)]
pub struct Raon1Spec {
    pub layers: usize,
    /// Dense feed-forward layers at the start of the stack (leading dense block).
    pub leading_dense: usize,
    pub hidden: u64,
    pub vocab: u64,
    /// Attention head count: the MLA query heads and the RDA heads share it; the
    /// per-head width is `head_dim`.
    pub heads: u64,
    /// RDA q/k/v and gate width per head; also the MLA gate and output width.
    pub head_dim: u64,
    /// MLA query LoRA rank (attn_q_a / attn_q_b boundary).
    pub q_lora: u64,
    /// MLA per-head key width that `attn_q_b` produces (`attention.key_length_mla`).
    pub key_mla: u64,
    /// MLA compressed key/value latent width (attn_kv_a_mqa row length).
    pub kv_latent: u64,
    /// The `attn_kv_a_mqa` output width: the key/value latent plus the rope dims
    /// (512 + 64 on the model).
    pub kv_a_width: u64,
    /// MLA per-head key width after `attn_k_b`; on the model this is
    /// `key_length_mla` minus the rope part (192 - 64).
    pub k_head: u64,
    /// MLA per-head value width after `attn_v_b`.
    pub v_head: u64,
    pub time_step_rank: u64,
    pub conv_kernel: u64,
    pub experts: u64,
    pub moe_inter: u64,
    pub dense_inter: u64,
    /// Raon-mHC residual streams (`hyper_connection.count`).
    pub streams: u64,
    pub mla_layers: &'static [usize],
    /// The MoE layers whose down experts are Q5_0 in the published Q4_K_M file;
    /// every other MoE layer's down experts are Q8_0.
    pub down_q5_0_layers: &'static [usize],
}

pub const RAON_1_MINI_PREVIEW: Raon1Spec = Raon1Spec {
    layers: 25,
    leading_dense: 1,
    hidden: 2560,
    vocab: 259_840,
    heads: 40,
    head_dim: 128,
    q_lora: 576,
    key_mla: 192,
    kv_latent: 512,
    kv_a_width: 576,
    k_head: 128,
    v_head: 128,
    time_step_rank: 128,
    conv_kernel: 4,
    experts: 128,
    moe_inter: 1088,
    dense_inter: 12_032,
    streams: 4,
    mla_layers: &[3, 7, 11, 15, 19, 23, 24],
    down_q5_0_layers: &[3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19],
};

impl Raon1Spec {
    fn kind(self, layer: usize) -> RaonKind {
        if self.mla_layers.contains(&layer) {
            RaonKind::Mla
        } else {
            RaonKind::Rda
        }
    }

    fn moe(self, layer: usize) -> bool {
        layer >= self.leading_dense
    }

    /// The shared RDA/MLA projection width: `heads * head_dim` (5,120 on the model).
    fn proj_width(self) -> u64 {
        self.heads * self.head_dim
    }

    /// The mHC mix matrix row count: `streams * streams + 1` (17 on the model).
    fn mix_count(self) -> u64 {
        self.streams * self.streams + 1
    }

    /// The expected tensors of one checkpoint, in layer order. 731 entries for
    /// `RAON_1_MINI_PREVIEW` on the published Q4_K_M file.
    pub fn inventory(self) -> Vec<Expected> {
        let mut out = vec![self.expected(
            "token_embd.weight",
            GgufType::Q4K,
            &[self.hidden, self.vocab],
        )];
        for layer in 0..self.layers {
            for suffix in self.layer_suffixes(layer) {
                out.push(self.layer_tensor(layer, &suffix));
            }
        }
        out.push(self.expected("output_norm.weight", GgufType::F32, &[self.hidden]));
        out.push(self.expected("output.weight", GgufType::Q6K, &[self.hidden, self.vocab]));
        out
    }

    /// Every tensor of one layer, in file order: the norm and mHC group, the mixer,
    /// then the feed-forward group.
    fn layer_suffixes(self, layer: usize) -> Vec<Suffix> {
        let mut out = vec![
            Suffix::AttnNorm,
            Suffix::FfnNorm,
            Suffix::HcAttnBase,
            Suffix::HcAttnFn,
            Suffix::HcAttnScale,
            Suffix::HcFfnBase,
            Suffix::HcFfnFn,
            Suffix::HcFfnScale,
        ];
        match self.kind(layer) {
            RaonKind::Rda => out.extend([
                Suffix::AttnQ,
                Suffix::AttnK,
                Suffix::AttnV,
                Suffix::AttnOutput,
                Suffix::SsmA,
                Suffix::SsmBeta,
                Suffix::SsmRda,
                Suffix::SsmConv1dK,
                Suffix::SsmConv1dQ,
                Suffix::SsmConv1dV,
                Suffix::SsmDtBias,
                Suffix::SsmFA,
                Suffix::SsmFB,
                Suffix::SsmG,
                Suffix::SsmNorm,
            ]),
            RaonKind::Mla => out.extend([
                Suffix::AttnGate,
                Suffix::AttnQA,
                Suffix::AttnQANorm,
                Suffix::AttnQB,
                Suffix::AttnKvAMqa,
                Suffix::AttnKvANorm,
                Suffix::AttnKB,
                Suffix::AttnVB,
                Suffix::AttnOutput,
            ]),
        }
        if self.moe(layer) {
            out.extend([
                Suffix::FfnGateInp,
                Suffix::FfnGateExps,
                Suffix::FfnUpExps,
                Suffix::FfnDownExps,
                Suffix::FfnGateShexp,
                Suffix::FfnUpShexp,
                Suffix::FfnDownShexp,
                Suffix::ExpProbsB,
            ]);
        } else {
            out.extend([Suffix::FfnGate, Suffix::FfnUp, Suffix::FfnDown]);
        }
        out
    }

    fn layer_tensor(self, layer: usize, suffix: &Suffix) -> Expected {
        let name = format!("blk.{layer}.{}", suffix.name());
        let (ty, dims) = self.shape(layer, *suffix);
        Expected {
            name,
            ty,
            dims: dims.to_vec(),
        }
    }

    fn expected(self, name: &str, ty: GgufType, dims: &[u64]) -> Expected {
        Expected {
            name: name.to_owned(),
            ty,
            dims: dims.to_vec(),
        }
    }

    /// The published down-expert type of one MoE layer.
    fn down_type(self, layer: usize) -> GgufType {
        if self.down_q5_0_layers.contains(&layer) {
            GgufType::Q50
        } else {
            GgufType::Q80
        }
    }

    /// Type and dims of one suffix. Every role pins the exact Q4_K_M type it was
    /// converted to, the down experts included, per layer.
    #[allow(clippy::too_many_lines)]
    fn shape(self, layer: usize, suffix: Suffix) -> (GgufType, Vec<u64>) {
        use GgufType::{F16, F32, Q4K, Q6K, Q80};
        let h = self.hidden;
        let w = self.proj_width();
        let mix = self.mix_count();
        match suffix {
            Suffix::AttnNorm | Suffix::FfnNorm => (F32, vec![h]),
            Suffix::HcAttnBase | Suffix::HcFfnBase => (F32, vec![mix]),
            Suffix::HcAttnFn | Suffix::HcFfnFn => (F16, vec![self.streams * h, mix]),
            Suffix::HcAttnScale | Suffix::HcFfnScale => (F32, vec![self.streams]),
            Suffix::AttnQ | Suffix::AttnK | Suffix::AttnV | Suffix::SsmG => (Q6K, vec![h, w]),
            Suffix::AttnOutput => (Q80, vec![w, h]),
            Suffix::AttnGate => (Q80, vec![h, w]),
            Suffix::SsmA => (F32, vec![self.heads]),
            Suffix::SsmBeta | Suffix::SsmRda => (F16, vec![h, self.heads]),
            Suffix::SsmConv1dK | Suffix::SsmConv1dQ | Suffix::SsmConv1dV => {
                (F32, vec![self.conv_kernel, 1, w])
            }
            Suffix::SsmDtBias => (F32, vec![w]),
            Suffix::SsmFA => (F16, vec![h, self.time_step_rank]),
            Suffix::SsmFB => (F16, vec![self.time_step_rank, w]),
            Suffix::SsmNorm => (F32, vec![self.head_dim]),
            Suffix::AttnQA => (Q6K, vec![h, self.q_lora]),
            Suffix::AttnQANorm => (F32, vec![self.q_lora]),
            Suffix::AttnQB => (Q80, vec![self.q_lora, self.heads * self.key_mla]),
            Suffix::AttnKvAMqa => (Q6K, vec![h, self.kv_a_width]),
            Suffix::AttnKvANorm => (F32, vec![self.kv_latent]),
            Suffix::AttnKB => (Q80, vec![self.k_head, self.kv_latent, self.heads]),
            Suffix::AttnVB => (Q6K, vec![self.kv_latent, self.v_head, self.heads]),
            Suffix::FfnGateInp => (F32, vec![h, self.experts]),
            Suffix::FfnGateExps | Suffix::FfnUpExps => (Q4K, vec![h, self.moe_inter, self.experts]),
            Suffix::FfnDownExps => (self.down_type(layer), vec![self.moe_inter, h, self.experts]),
            Suffix::FfnGateShexp | Suffix::FfnUpShexp => (Q6K, vec![h, self.moe_inter]),
            Suffix::FfnDownShexp => (Q80, vec![self.moe_inter, h]),
            Suffix::ExpProbsB => (F32, vec![self.experts]),
            Suffix::FfnGate | Suffix::FfnUp => (Q4K, vec![h, self.dense_inter]),
            Suffix::FfnDown => (Q6K, vec![self.dense_inter, h]),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Suffix {
    AttnNorm,
    FfnNorm,
    HcAttnBase,
    HcAttnFn,
    HcAttnScale,
    HcFfnBase,
    HcFfnFn,
    HcFfnScale,
    AttnQ,
    AttnK,
    AttnV,
    AttnOutput,
    AttnGate,
    SsmA,
    SsmBeta,
    SsmRda,
    SsmConv1dK,
    SsmConv1dQ,
    SsmConv1dV,
    SsmDtBias,
    SsmFA,
    SsmFB,
    SsmG,
    SsmNorm,
    AttnQA,
    AttnQANorm,
    AttnQB,
    AttnKvAMqa,
    AttnKvANorm,
    AttnKB,
    AttnVB,
    FfnGateInp,
    FfnGateExps,
    FfnUpExps,
    FfnDownExps,
    FfnGateShexp,
    FfnUpShexp,
    FfnDownShexp,
    ExpProbsB,
    FfnGate,
    FfnUp,
    FfnDown,
}

impl Suffix {
    fn name(self) -> &'static str {
        match self {
            Suffix::AttnNorm => "attn_norm.weight",
            Suffix::FfnNorm => "ffn_norm.weight",
            Suffix::HcAttnBase => "hc_attn_base.weight",
            Suffix::HcAttnFn => "hc_attn_fn.weight",
            Suffix::HcAttnScale => "hc_attn_scale.weight",
            Suffix::HcFfnBase => "hc_ffn_base.weight",
            Suffix::HcFfnFn => "hc_ffn_fn.weight",
            Suffix::HcFfnScale => "hc_ffn_scale.weight",
            Suffix::AttnQ => "attn_q.weight",
            Suffix::AttnK => "attn_k.weight",
            Suffix::AttnV => "attn_v.weight",
            Suffix::AttnOutput => "attn_output.weight",
            Suffix::AttnGate => "attn_gate.weight",
            Suffix::SsmA => "ssm_a",
            Suffix::SsmBeta => "ssm_beta.weight",
            Suffix::SsmRda => "ssm_rda.weight",
            Suffix::SsmConv1dK => "ssm_conv1d_k.weight",
            Suffix::SsmConv1dQ => "ssm_conv1d_q.weight",
            Suffix::SsmConv1dV => "ssm_conv1d_v.weight",
            Suffix::SsmDtBias => "ssm_dt.bias",
            Suffix::SsmFA => "ssm_f_a.weight",
            Suffix::SsmFB => "ssm_f_b.weight",
            Suffix::SsmG => "ssm_g.weight",
            Suffix::SsmNorm => "ssm_norm.weight",
            Suffix::AttnQA => "attn_q_a.weight",
            Suffix::AttnQANorm => "attn_q_a_norm.weight",
            Suffix::AttnQB => "attn_q_b.weight",
            Suffix::AttnKvAMqa => "attn_kv_a_mqa.weight",
            Suffix::AttnKvANorm => "attn_kv_a_norm.weight",
            Suffix::AttnKB => "attn_k_b.weight",
            Suffix::AttnVB => "attn_v_b.weight",
            Suffix::FfnGateInp => "ffn_gate_inp.weight",
            Suffix::FfnGateExps => "ffn_gate_exps.weight",
            Suffix::FfnUpExps => "ffn_up_exps.weight",
            Suffix::FfnDownExps => "ffn_down_exps.weight",
            Suffix::FfnGateShexp => "ffn_gate_shexp.weight",
            Suffix::FfnUpShexp => "ffn_up_shexp.weight",
            Suffix::FfnDownShexp => "ffn_down_shexp.weight",
            Suffix::ExpProbsB => "exp_probs_b.bias",
            Suffix::FfnGate => "ffn_gate.weight",
            Suffix::FfnUp => "ffn_up.weight",
            Suffix::FfnDown => "ffn_down.weight",
        }
    }
}

/// One expected tensor: the name, the exact GGUF type, and the exact dims.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expected {
    pub name: String,
    pub ty: GgufType,
    pub dims: Vec<u64>,
}

impl Expected {
    /// Source bytes of this tensor in the file's own type, or the error a wrong
    /// row length or an overflowing product produces.
    #[cfg(test)]
    fn byte_len(&self, ty: GgufType) -> Result<u64, EngError> {
        let block = ty.block_elems();
        if !self.dims[0].is_multiple_of(block) {
            return Err(reject(format!(
                "tensor {} has row length {} not a multiple of the {} block {}",
                self.name,
                self.dims[0],
                ty.name(),
                block
            )));
        }
        let elements = self
            .dims
            .iter()
            .try_fold(1u64, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| reject(format!("tensor {} dims product overflows u64", self.name)))?;
        elements
            .checked_div(block)
            .and_then(|blocks| blocks.checked_mul(ty.block_bytes()))
            .ok_or_else(|| reject(format!("tensor {} byte length overflows u64", self.name)))
    }
}

/// An opened Raon-1-mini-preview GGUF file.
#[derive(Debug)]
pub struct Raon1Model {
    weights: GgufWeights,
    spec: Raon1Spec,
    path: PathBuf,
}

impl Raon1Model {
    pub fn spec(&self) -> Raon1Spec {
        self.spec
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn header(&self) -> &GgufHeader {
        self.weights.header()
    }

    /// One tensor dequantized to host F32, read by name. The host reference reads
    /// layer by layer, so this is per name, not a whole-model copy. The open already
    /// pinned every tensor's type and dims, and `read_tensor` fails on an unknown
    /// name, so nothing is re-checked here.
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>, EngError> {
        let (ty, bytes) = self.weights.read_tensor(name)?;
        dequantize(ty, &bytes)
    }
}

/// Opens the published Raon-1-mini-preview Q4_K_M GGUF strictly: architecture,
/// pinned geometry keys, then the full tensor set, types and dims.
pub fn open_raon_1_mini_preview(
    path: &Path,
    header_sha256: Option<&str>,
) -> Result<Raon1Model, EngError> {
    open_with_spec(path, header_sha256, RAON_1_MINI_PREVIEW)
}

fn open_with_spec(
    path: &Path,
    header_sha256: Option<&str>,
    spec: Raon1Spec,
) -> Result<Raon1Model, EngError> {
    let weights = GgufWeights::open(path, LIMITS, header_sha256)?;
    check_metadata(weights.header(), spec)?;
    check_tensors(weights.header(), spec)?;
    Ok(Raon1Model {
        weights,
        spec,
        path: path.to_path_buf(),
    })
}

fn reject(message: impl Into<String>) -> EngError {
    EngError::Checkpoint(format!("raon1 model: {}", message.into()))
}

/// The geometry keys the inventory is generated from. Hyperparameters (situ betas,
/// epsilons, gating function) are read by the host reference when it is written,
/// not pinned here.
fn check_metadata(header: &GgufHeader, spec: Raon1Spec) -> Result<(), EngError> {
    match header.meta("general.architecture") {
        Some(MetaValue::String(found)) if found == "raon1" => {}
        Some(MetaValue::String(found)) => {
            return Err(reject(format!(
                "metadata key general.architecture is {found}, expected raon1"
            )));
        }
        _ => {
            return Err(reject(
                "metadata key general.architecture is missing or not a string",
            ));
        }
    }
    let pinned: [(&str, u64); 20] = [
        ("raon1.block_count", spec.layers as u64),
        ("raon1.embedding_length", spec.hidden),
        ("raon1.vocab_size", spec.vocab),
        ("raon1.attention.head_count", spec.heads),
        ("raon1.attention.q_lora_rank", spec.q_lora),
        ("raon1.attention.kv_lora_rank", spec.kv_latent),
        ("raon1.attention.key_length", spec.kv_a_width),
        ("raon1.attention.key_length_mla", spec.key_mla),
        ("raon1.attention.value_length", spec.kv_latent),
        ("raon1.attention.value_length_mla", spec.v_head),
        ("raon1.kda.head_dim", spec.head_dim),
        ("raon1.ssm.conv_kernel", spec.conv_kernel),
        ("raon1.ssm.time_step_rank", spec.time_step_rank),
        ("raon1.expert_count", spec.experts),
        ("raon1.expert_used_count", 6),
        ("raon1.expert_shared_count", 1),
        ("raon1.expert_feed_forward_length", spec.moe_inter),
        ("raon1.feed_forward_length", spec.dense_inter),
        ("raon1.leading_dense_block_count", spec.leading_dense as u64),
        ("raon1.hyper_connection.count", spec.streams),
    ];
    for (key, want) in pinned {
        match header.meta(key) {
            Some(MetaValue::U32(found)) if u64::from(*found) == want => {}
            Some(MetaValue::U64(found)) if *found == want => {}
            Some(MetaValue::U32(found)) => {
                return Err(reject(format!(
                    "metadata key {key} is {found}, expected {want}"
                )));
            }
            Some(MetaValue::U64(found)) => {
                return Err(reject(format!(
                    "metadata key {key} is {found}, expected {want}"
                )));
            }
            Some(_) => {
                return Err(reject(format!(
                    "metadata key {key} is not a u32 or u64 integer"
                )));
            }
            None => return Err(reject(format!("metadata key {key} is missing"))),
        }
    }
    Ok(())
}

fn check_tensors(header: &GgufHeader, spec: Raon1Spec) -> Result<(), EngError> {
    let inventory = spec.inventory();
    for tensor in header.tensors() {
        if !inventory
            .iter()
            .any(|expected| expected.name == tensor.name())
        {
            return Err(reject(format!(
                "tensor {} is not in the raon1 inventory",
                tensor.name()
            )));
        }
    }
    for expected in &inventory {
        let tensor = header
            .tensor(&expected.name)
            .ok_or_else(|| reject(format!("tensor {} is missing", expected.name)))?;
        let ty = tensor.ggml_type();
        if ty != expected.ty {
            return Err(reject(format!(
                "tensor {} has type {}, expected {}",
                expected.name,
                ty.name(),
                expected.ty.name()
            )));
        }
        if tensor.dims() != expected.dims.as_slice() {
            return Err(reject(format!(
                "tensor {} has dims {:?}, expected {:?}",
                expected.name,
                tensor.dims(),
                expected.dims
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
