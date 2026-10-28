use crate::EngError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    Silu,
    GeluApprox,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Router {
    SoftmaxTopK {
        renorm: bool,
    },
    /// Ungrouped sigmoid scores, normalized over the selected experts.
    SigmoidTopK,
    SigmoidGroup {
        groups: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKind {
    Full,
    DeltaNet,
    Qsa,
}

/// Gated delta recurrence dimensions read from the checkpoint configuration.
///
/// The fields mirror `linear_num_key_heads`, `linear_key_head_dim`,
/// `linear_num_value_heads`, `linear_value_head_dim`, and `linear_conv_kernel_dim`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearAttention {
    pub key_heads: u32,
    pub key_head_dim: u32,
    pub value_heads: u32,
    pub value_head_dim: u32,
    pub conv_kernel: u32,
}

impl LinearAttention {
    /// Query or key width across all key heads.
    pub fn key_dim(&self) -> Result<u32, EngError> {
        checked_product(self.key_heads, self.key_head_dim, "key dimension")
    }

    /// Value width across all value heads.
    pub fn value_dim(&self) -> Result<u32, EngError> {
        checked_product(self.value_heads, self.value_head_dim, "value dimension")
    }

    /// Channel count of the fused query, key, and value projection and its convolution.
    pub fn conv_dim(&self) -> Result<u32, EngError> {
        let keys = checked_product(self.key_dim()?, 2, "convolution dimension")?;
        keys.checked_add(self.value_dim()?)
            .ok_or_else(|| overflow("convolution dimension"))
    }

    pub(crate) fn validate(&self) -> Result<(), EngError> {
        for (field, value) in [
            ("key heads", self.key_heads),
            ("key head dimension", self.key_head_dim),
            ("value heads", self.value_heads),
            ("value head dimension", self.value_head_dim),
        ] {
            if value == 0 {
                return Err(EngError::Checkpoint(format!(
                    "linear attention {field} must exceed zero"
                )));
            }
        }
        // The convolution state keeps kernel - 1 positions. A kernel of 1 would make it empty.
        if self.conv_kernel < 2 {
            return Err(EngError::Checkpoint(format!(
                "linear attention convolution kernel {} must be at least 2",
                self.conv_kernel
            )));
        }
        if !self.value_heads.is_multiple_of(self.key_heads) {
            return Err(EngError::Checkpoint(format!(
                "linear attention value heads {} must divide by key heads {}",
                self.value_heads, self.key_heads
            )));
        }
        self.conv_dim().map(|_| ())
    }
}

fn checked_product(a: u32, b: u32, what: &str) -> Result<u32, EngError> {
    a.checked_mul(b).ok_or_else(|| overflow(what))
}

fn overflow(what: &str) -> EngError {
    EngError::Checkpoint(format!("linear attention {what} overflows u32"))
}

/// Hyper-connection widening of the residual stream.
///
/// The fields mirror `hc_count` and `hc_lowrank` in the checkpoint configuration.
/// The widened residual holds `branches` streams of `hidden` values each, so its
/// width is `branches * hidden`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyperConnection {
    pub branches: u32,
    pub rank: u32,
}

impl HyperConnection {
    /// Total width of the widened residual stream.
    pub fn width(&self, hidden: u32) -> Result<u32, EngError> {
        self.branches
            .checked_mul(hidden)
            .ok_or_else(|| EngError::Checkpoint("hyper-connection width overflows u32".into()))
    }
}

/// Dense feed-forward dimensions and the checkpoint's maximum context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseMlp {
    pub intermediate: u32,
    pub context_limit: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FamilySpec {
    pub name: &'static str,
    pub layers: u32,
    pub hidden: u32,
    pub heads_q: u32,
    pub heads_kv: u32,
    pub head_dim: u32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// Accepted `model_type` values of the configuration, at the root and in `text_config`.
    pub model_types: &'static [&'static str],
    pub vocab: u32,
    /// `Some` selects dense feed-forward layers and requires all four MoE dimensions to be zero.
    /// `None` preserves the MoE layout and its existing dimensions.
    pub feed_forward: Option<DenseMlp>,
    pub experts: u32,
    pub top_k: u32,
    pub moe_inter: u32,
    pub shared_inter: u32,
    pub quant_bits: u32,
    pub quant_group: u32,
    pub act: Act,
    pub router: Router,
    pub mtp: bool,
    pub layers_attention: &'static [AttentionKind],
    /// Required when any layer uses `AttentionKind::DeltaNet`.
    pub linear: Option<LinearAttention>,
    /// Hyper-connection widening of the residual stream, when the family uses it.
    pub hyper: Option<HyperConnection>,
}

const QWEN3_6_35B_A3B_LAYERS: [AttentionKind; 40] = [
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Full,
];

pub const QWEN3_6_35B_A3B: FamilySpec = FamilySpec {
    name: "qwen3_6_35b_a3b",
    layers: 40,
    hidden: 2048,
    heads_q: 16,
    heads_kv: 2,
    head_dim: 256,
    rope_theta: 10_000_000.0,
    rms_norm_eps: 1e-6,
    model_types: &["qwen3_5_moe", "qwen3_5_moe_text"],
    vocab: 248_320,
    feed_forward: None,
    experts: 256,
    top_k: 8,
    moe_inter: 512,
    shared_inter: 512,
    quant_bits: 4,
    quant_group: 64,
    act: Act::Silu,
    router: Router::SoftmaxTopK { renorm: true },
    mtp: false,
    layers_attention: &QWEN3_6_35B_A3B_LAYERS,
    linear: Some(LinearAttention {
        key_heads: 16,
        key_head_dim: 128,
        value_heads: 32,
        value_head_dim: 128,
        conv_kernel: 4,
    }),
    hyper: None,
};

const fn dense_27b_attention() -> [AttentionKind; 64] {
    let mut kinds = [AttentionKind::DeltaNet; 64];
    let mut index = 3;
    while index < kinds.len() {
        kinds[index] = AttentionKind::Full;
        index += 4;
    }
    kinds
}

const QWEN3_8_27B_DENSE_LAYERS: [AttentionKind; 64] = dense_27b_attention();

/// Portable descriptor for the affine-6 group-64 Qwen3.8-27B checkpoint.
///
/// Attention uses output gating and untied embeddings, as checked by `check_config`.
/// The checkpoint also declares RoPE partial factor 0.25, `mrope_interleaved: true`,
/// and `mrope_section: [11, 11, 10]`. This descriptor does not implement multimodal RoPE.
pub const QWEN3_8_27B_DENSE: FamilySpec = FamilySpec {
    name: "qwen3_8_27b_dense",
    layers: 64,
    hidden: 5120,
    heads_q: 24,
    heads_kv: 4,
    head_dim: 256,
    rope_theta: 10_000_000.0,
    rms_norm_eps: 1e-6,
    model_types: &["qwen3_5", "qwen3_5_text"],
    vocab: 248_320,
    feed_forward: Some(DenseMlp {
        intermediate: 17_408,
        context_limit: 262_144,
    }),
    experts: 0,
    top_k: 0,
    moe_inter: 0,
    shared_inter: 0,
    quant_bits: 6,
    quant_group: 64,
    act: Act::Silu,
    // Dense layers do not use a router.
    router: Router::SoftmaxTopK { renorm: true },
    mtp: false,
    layers_attention: &QWEN3_8_27B_DENSE_LAYERS,
    linear: Some(LinearAttention {
        key_heads: 16,
        key_head_dim: 128,
        value_heads: 48,
        value_head_dim: 128,
        conv_kernel: 4,
    }),
    hyper: None,
};

/// The twelve QSA (sparse full-attention) layers sit at indices 3, 7, ..., 47.
const QWEN3_8_FLASH_NEXT_LAYERS: [AttentionKind; 48] = [
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::DeltaNet,
    AttentionKind::Qsa,
];

/// The Flash-Next family: `Qwen/Qwen3.8-Flash-Next`, model type `qwen4_exp`.
///
/// The values come from the pinned artifact `config.json` (see
/// the Flash-Next executor design and the environment record).
/// The configuration names the root model type `qwen4_exp` and the `text_config`
/// model type `qwen4_exp_text`, so both belong to the family.
pub const QWEN3_8_FLASH_NEXT: FamilySpec = FamilySpec {
    name: "qwen3_8_flash_next",
    layers: 48,
    hidden: 2560,
    heads_q: 24,
    heads_kv: 2,
    head_dim: 256,
    rope_theta: 10_000_000.0,
    rms_norm_eps: 1e-6,
    model_types: &["qwen4_exp", "qwen4_exp_text"],
    vocab: 248_320,
    feed_forward: None,
    experts: 512,
    top_k: 10,
    moe_inter: 640,
    shared_inter: 640,
    quant_bits: 4,
    quant_group: 64,
    act: Act::Silu,
    router: Router::SoftmaxTopK { renorm: true },
    mtp: true,
    layers_attention: &QWEN3_8_FLASH_NEXT_LAYERS,
    linear: Some(LinearAttention {
        key_heads: 16,
        key_head_dim: 128,
        value_heads: 48,
        value_head_dim: 128,
        conv_kernel: 4,
    }),
    hyper: Some(HyperConnection {
        branches: 4,
        rank: 320,
    }),
};

/// The Flash-Next family specification.
pub fn qwen3_8_flash_next() -> FamilySpec {
    QWEN3_8_FLASH_NEXT
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::keys::TierAssignment;

    #[test]
    fn dense_27b_descriptor_matches_checkpoint_dimensions() {
        let spec = QWEN3_8_27B_DENSE;
        TierAssignment::from_keys(&[], &spec).expect("valid dense family");
        assert_eq!((spec.layers, spec.hidden, spec.vocab), (64, 5120, 248_320));
        assert_eq!((spec.heads_q, spec.heads_kv, spec.head_dim), (24, 4, 256));
        assert_eq!(
            spec.feed_forward,
            Some(DenseMlp {
                intermediate: 17_408,
                context_limit: 262_144,
            })
        );
        assert_eq!((spec.quant_bits, spec.quant_group), (6, 64));
        let full: Vec<_> = spec
            .layers_attention
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == AttentionKind::Full)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(full, (3..64).step_by(4).collect::<Vec<_>>());
        assert_eq!(spec.linear, QWEN3_8_FLASH_NEXT.linear);
        assert!(!spec.mtp);
        assert_eq!(spec.hyper, None);
    }

    #[test]
    fn q36_keeps_its_moe_descriptor() {
        let spec = QWEN3_6_35B_A3B;
        TierAssignment::from_keys(&[], &spec).expect("valid MoE family");
        assert_eq!(spec.feed_forward, None);
        assert_eq!((spec.experts, spec.top_k), (256, 8));
        assert_eq!((spec.moe_inter, spec.shared_inter), (512, 512));
    }

    #[test]
    fn flash_next_layer_attention_has_48_entries() {
        let spec = qwen3_8_flash_next();
        assert_eq!(spec.name, "qwen3_8_flash_next");
        assert_eq!(spec.layers, 48);
        assert_eq!(spec.layers_attention.len(), 48);
    }

    #[test]
    fn flash_next_places_twelve_qsa_layers_at_the_documented_indices() {
        let spec = qwen3_8_flash_next();
        let qsa: Vec<usize> = spec
            .layers_attention
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == AttentionKind::Qsa)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(qsa.len(), 12);
        assert_eq!(qsa, [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]);
    }

    #[test]
    fn flash_next_has_thirty_six_delta_net_layers() {
        let spec = qwen3_8_flash_next();
        let delta = spec
            .layers_attention
            .iter()
            .filter(|kind| **kind == AttentionKind::DeltaNet)
            .count();
        assert_eq!(delta, 36);
    }

    #[test]
    fn flash_next_linear_attention_geometry_matches_the_pinned_dimensions() {
        let linear = qwen3_8_flash_next()
            .linear
            .expect("flash-next delta layers have linear attention dimensions");
        assert_eq!((linear.key_heads, linear.key_head_dim), (16, 128));
        assert_eq!((linear.value_heads, linear.value_head_dim), (48, 128));
        assert_eq!(linear.conv_kernel, 4);
        assert_eq!(linear.key_dim().expect("key width"), 2048);
        assert_eq!(linear.value_dim().expect("value width"), 6144);
        assert_eq!(linear.conv_dim().expect("fused width"), 10_240);
    }

    #[test]
    fn flash_next_expert_count_and_top_k_match() {
        let spec = qwen3_8_flash_next();
        assert_eq!(spec.experts, 512);
        assert_eq!(spec.top_k, 10);
        assert_eq!(spec.moe_inter, 640);
        assert_eq!(spec.shared_inter, 640);
    }

    #[test]
    fn flash_next_hyper_connection_width_is_four_times_hidden() {
        let spec = qwen3_8_flash_next();
        let hyper = spec.hyper.expect("flash-next widens the residual stream");
        assert_eq!(hyper.branches, 4);
        assert_eq!(hyper.rank, 320);
        let width = hyper.width(spec.hidden).expect("hyper width");
        assert_eq!(width, 4 * 2560);
        assert_eq!(width, 10_240);
    }

    #[test]
    fn flash_next_passes_family_validation() {
        TierAssignment::from_keys(&[], &qwen3_8_flash_next()).expect("valid family");
    }
}
