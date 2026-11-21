//! The fixed-loop Inkling 2 Mini text checkpoint and its independent per-visit state.

mod checkpoint;
#[cfg(target_pointer_width = "64")]
mod cpu_attention;
#[cfg(target_pointer_width = "64")]
mod cpu_forward;
#[cfg(target_pointer_width = "64")]
mod cpu_weights;
#[cfg(all(
    feature = "native-cuda",
    target_os = "linux",
    target_pointer_width = "64"
))]
mod cuda_projection;
#[cfg(target_pointer_width = "64")]
mod gguf_weights;
#[cfg(target_pointer_width = "64")]
mod serve;

#[cfg(target_pointer_width = "64")]
pub(crate) use cpu_forward::{inspect_cpu_config, inspect_gguf_config};
#[cfg(target_pointer_width = "64")]
pub(crate) use gguf_weights::{GgufPlan, gguf_path};
#[cfg(target_pointer_width = "64")]
pub(crate) use serve::{TextSettings, start_engine};

pub(crate) use checkpoint::inspect_checkpoint;
use serde::Deserialize;

use crate::{EngError, checkpoint_json::UniqueValue};

pub(crate) const FAMILY: &str = "inkling2_loop";
const CPU_REFERENCE_ENV: &str = "FIFI_INKLING_CPU_REFERENCE";
const CUDA_REFERENCE_ENV: &str = "FIFI_INKLING_CUDA_REFERENCE";
pub(crate) const DEFAULT_TEXT_CONTEXT: u32 = 128;
#[cfg(target_pointer_width = "64")]
const MAX_PREFILL_ROWS: usize = 16;

pub(crate) fn text_context() -> Result<u32, EngError> {
    const NAME: &str = "FIFI_INKLING_CONTEXT_TOKENS";
    let Some(raw) = std::env::var_os(NAME) else {
        return Ok(DEFAULT_TEXT_CONTEXT);
    };
    raw.to_str()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| bad(format!("{NAME} must be a positive token count")))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TextBackend {
    Cpu,
    Cuda,
}

impl TextBackend {
    pub(crate) fn from_env() -> Result<Option<Self>, EngError> {
        let enabled = |flag| std::env::var_os(flag).as_deref() == Some(std::ffi::OsStr::new("1"));
        Self::select(enabled(CPU_REFERENCE_ENV), enabled(CUDA_REFERENCE_ENV))
    }

    fn select(cpu: bool, cuda: bool) -> Result<Option<Self>, EngError> {
        match (cpu, cuda) {
            (false, false) => Ok(None),
            (true, true) => Err(EngError::BadRequest(format!(
                "select one Inkling backend with {CPU_REFERENCE_ENV}=1 or {CUDA_REFERENCE_ENV}=1"
            ))),
            (true, false) if cfg!(target_pointer_width = "64") => Ok(Some(Self::Cpu)),
            (false, true)
                if cfg!(all(
                    feature = "native-cuda",
                    target_os = "linux",
                    target_pointer_width = "64"
                )) =>
            {
                Ok(Some(Self::Cuda))
            }
            (true, false) => Err(EngError::BadRequest(
                "Inkling CPU serving requires a 64-bit build".into(),
            )),
            (false, true) => Err(EngError::BadRequest(
                "Inkling CUDA serving requires a 64-bit Linux build with native-cuda".into(),
            )),
        }
    }

    #[cfg(target_pointer_width = "64")]
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA",
        }
    }
}

pub(crate) fn enable_text_backend() -> String {
    format!(
        "Set {CPU_REFERENCE_ENV}=1 or {CUDA_REFERENCE_ENV}=1 to enable experimental Inkling text serving."
    )
}

#[derive(Deserialize)]
struct Config {
    model_type: String,
    architectures: Vec<String>,
    eos_token_id: u32,
    text_config: TextSpec,
    quantization_config: Quantization,
}

#[derive(Deserialize)]
struct Quantization {
    quant_method: String,
    quant_algo: String,
}

#[derive(Deserialize)]
pub(crate) struct TextSpec {
    model_type: String,
    model_max_length: u32,
    hidden_size: u64,
    num_hidden_layers: usize,
    vocab_size: u64,
    unpadded_vocab_size: u64,
    num_attention_heads: u64,
    num_key_value_heads: u64,
    head_dim: u64,
    swa_num_attention_heads: u64,
    swa_num_key_value_heads: u64,
    swa_head_dim: u64,
    sliding_window_size: u32,
    local_layer_ids: Vec<usize>,
    n_layers_per_attn_res_block: usize,
    n_routed_experts: u64,
    num_experts_per_tok: u64,
    n_shared_experts: u64,
    intermediate_size: u64,
    #[serde(rename = "loop_config")]
    core: LoopConfig,
}

#[derive(Deserialize)]
struct LoopConfig {
    core_start: usize,
    core_end: usize,
    num_passes: usize,
    gate_rank: u64,
    is_varloop: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Attention {
    Local,
    Global,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LayerVisit {
    /// Repeated core visits look up the same stored weights.
    pub(crate) weight_layer: usize,
    /// Each visit owns independent attention and short-convolution state.
    pub(crate) state_slot: usize,
    pub(crate) attention: Attention,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct KvPlan {
    pub(crate) context: u32,
    pub(crate) local_visits: usize,
    pub(crate) global_visits: usize,
    pub(crate) windowed_bytes: u64,
    pub(crate) all_full_bytes: u64,
}

impl TextSpec {
    pub(crate) fn from_config(bytes: &[u8]) -> Result<Self, EngError> {
        let raw = serde_json::from_slice::<UniqueValue>(bytes)
            .map_err(|error| bad(format!("config JSON: {error}")))?;
        let config: Config = serde_json::from_value(raw.0)
            .map_err(|error| bad(format!("config fields: {error}")))?;
        if config.model_type != FAMILY
            || config.architectures != ["Inkling2LoopForConditionalGeneration"]
            || config.text_config.model_type != "inkling2_loop_text"
            || config.quantization_config.quant_method != "modelopt"
            || config.quantization_config.quant_algo != "NVFP4"
        {
            return Err(bad("expected the Inkling 2 Mini ModelOpt checkpoint"));
        }
        let spec = config.text_config;
        spec.validate()?;
        if u64::from(config.eos_token_id) >= spec.unpadded_vocab_size {
            return Err(bad("EOS token is outside the unpadded vocabulary"));
        }
        Ok(spec)
    }

    fn validate(&self) -> Result<(), EngError> {
        let core = &self.core;
        let block = self.n_layers_per_attn_res_block;
        if core.is_varloop
            || !(1..=256).contains(&self.num_hidden_layers)
            || block == 0
            || core.core_start == 0
            || core.core_start >= core.core_end
            || core.core_end >= self.num_hidden_layers
            || !(2..=8).contains(&core.num_passes)
            || ![core.core_start, core.core_end, self.num_hidden_layers]
                .iter()
                .all(|n| n.is_multiple_of(block))
            || core.gate_rank == 0
        {
            return Err(bad(
                "invalid fixed loop or attention residual block geometry",
            ));
        }

        let heads_valid = |query: u64, kv: u64, width: u64| {
            kv > 0 && query >= kv && query.is_multiple_of(kv) && width > 0
        };
        if !heads_valid(
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        ) || !heads_valid(
            self.swa_num_attention_heads,
            self.swa_num_key_value_heads,
            self.swa_head_dim,
        ) || self.sliding_window_size == 0
            || self.sliding_window_size > self.model_max_length
            || self
                .local_layer_ids
                .iter()
                .any(|&i| i >= self.num_hidden_layers)
            || self
                .local_layer_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(bad("invalid local/global attention geometry"));
        }

        if self.hidden_size == 0
            || self.intermediate_size == 0
            || !self.hidden_size.is_multiple_of(32)
            || !self.intermediate_size.is_multiple_of(32)
            || self.unpadded_vocab_size == 0
            || self.unpadded_vocab_size > self.vocab_size
            || self.num_experts_per_tok == 0
            || self.num_experts_per_tok > self.n_routed_experts
            || self.n_shared_experts != 0
        {
            return Err(bad("invalid vocabulary or expert geometry"));
        }
        Ok(())
    }

    pub(crate) fn stored_layers(&self) -> usize {
        self.num_hidden_layers
    }

    pub(crate) fn visits(&self) -> Vec<LayerVisit> {
        let core = &self.core;
        (0..core.core_start)
            .chain((0..core.num_passes).flat_map(|_| core.core_start..core.core_end))
            .chain(core.core_end..self.num_hidden_layers)
            .enumerate()
            .map(|(state_slot, weight_layer)| LayerVisit {
                weight_layer,
                state_slot,
                attention: if self.local_layer_ids.binary_search(&weight_layer).is_ok() {
                    Attention::Local
                } else {
                    Attention::Global
                },
            })
            .collect()
    }

    /// BF16 storage for two KV planes, with each local cache capped to the sliding window.
    /// This excludes convolution state, allocations and prefill workspace.
    pub(crate) fn kv_plan(&self, context: u32) -> Result<KvPlan, EngError> {
        if context == 0 || context > self.model_max_length {
            return Err(bad(format!(
                "context must be in 1..={}",
                self.model_max_length
            )));
        }
        let mut plan = KvPlan {
            context,
            local_visits: 0,
            global_visits: 0,
            windowed_bytes: 0,
            all_full_bytes: 0,
        };
        for visit in self.visits() {
            let (heads, width, rows) = match visit.attention {
                Attention::Local => {
                    plan.local_visits += 1;
                    (
                        self.swa_num_key_value_heads,
                        self.swa_head_dim,
                        context.min(self.sliding_window_size),
                    )
                }
                Attention::Global => {
                    plan.global_visits += 1;
                    (self.num_key_value_heads, self.head_dim, context)
                }
            };
            // Two planes of BF16 values.
            let row_bytes = checked_product([4, heads, width])?;
            plan.windowed_bytes = add_bytes(plan.windowed_bytes, row_bytes, rows)?;
            plan.all_full_bytes = add_bytes(plan.all_full_bytes, row_bytes, context)?;
        }
        Ok(plan)
    }
}

fn checked_product<const N: usize>(dimensions: [u64; N]) -> Result<u64, EngError> {
    dimensions
        .into_iter()
        .try_fold(1u64, |product, dimension| product.checked_mul(dimension))
        .ok_or_else(|| bad("tensor size overflows"))
}

fn add_bytes(total: u64, row_bytes: u64, rows: u32) -> Result<u64, EngError> {
    total
        .checked_add(checked_product([row_bytes, u64::from(rows)])?)
        .ok_or_else(|| bad("KV cache size overflows"))
}

fn bad(message: impl std::fmt::Display) -> EngError {
    EngError::Checkpoint(format!("Inkling 2 Mini: {message}"))
}

#[cfg(test)]
mod tests;
