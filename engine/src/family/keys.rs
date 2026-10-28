use std::collections::BTreeMap;

use crate::EngError;

use super::spec::{AttentionKind, FamilySpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyTier {
    Resident,
    ExpertStacked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Projection {
    Gate,
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KeyClass {
    ExpertStack { layer: u32, projection: Projection },
    SharedExpert { layer: u32, projection: Projection },
    DenseMlp { layer: u32, projection: Projection },
    Router { layer: u32 },
    SharedExpertGate { layer: u32 },
    LinearAttn { layer: u32 },
    SelfAttn { layer: u32 },
    EmbedTokens,
    LmHead,
    LayerNorm { layer: u32 },
    FinalNorm,
}

/// Whether a key belongs to a packed projection or to a dense tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KeyRole {
    Packed,
    Dense,
}

impl KeyClass {
    pub(crate) fn tier(self) -> KeyTier {
        match self {
            KeyClass::ExpertStack { .. } => KeyTier::ExpertStacked,
            _ => KeyTier::Resident,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Files {
    Triplet,
    WeightOnly,
    AuxOnly,
}

impl Files {
    fn role(self) -> KeyRole {
        match self {
            Files::Triplet => KeyRole::Packed,
            Files::WeightOnly | Files::AuxOnly => KeyRole::Dense,
        }
    }

    pub(crate) fn permits(self, file: FileKind) -> bool {
        matches!(
            (self, file),
            (Files::Triplet, FileKind::Weight)
                | (Files::Triplet, FileKind::Scales)
                | (Files::Triplet, FileKind::Biases)
                | (Files::WeightOnly, FileKind::Weight)
                | (Files::AuxOnly, FileKind::ALog)
                | (Files::AuxOnly, FileKind::DtBias)
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FileKind {
    Weight,
    Scales,
    Biases,
    ALog,
    DtBias,
}

fn file_kind(file: &str) -> Option<FileKind> {
    match file {
        "weight" => Some(FileKind::Weight),
        "scales" => Some(FileKind::Scales),
        "biases" => Some(FileKind::Biases),
        "A_log" => Some(FileKind::ALog),
        "dt_bias" => Some(FileKind::DtBias),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Assignment {
    tier: KeyTier,
    role: KeyRole,
}

#[derive(Debug, PartialEq, Eq)]
pub struct TierAssignment {
    assignments: BTreeMap<String, Assignment>,
}

impl TierAssignment {
    pub fn from_keys(keys: &[String], spec: &FamilySpec) -> Result<Self, EngError> {
        validate_family(spec)?;
        let mut assignments = BTreeMap::new();
        for key in keys {
            let (class, role) = classify(key, spec)?;
            let assignment = Assignment {
                tier: class.tier(),
                role,
            };
            if assignments.insert(key.clone(), assignment).is_some() {
                return Err(EngError::Checkpoint(format!(
                    "duplicate weight key {key:?}"
                )));
            }
        }
        Ok(TierAssignment { assignments })
    }

    pub fn tier(&self, key: &str) -> Option<KeyTier> {
        self.assignments.get(key).map(|assignment| assignment.tier)
    }

    pub(crate) fn role(&self, key: &str) -> Option<KeyRole> {
        self.assignments.get(key).map(|assignment| assignment.role)
    }

    pub fn len(&self) -> usize {
        self.assignments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, KeyTier)> {
        self.assignments
            .iter()
            .map(|(key, assignment)| (key.as_str(), assignment.tier))
    }
}

pub(crate) fn validate_family(spec: &FamilySpec) -> Result<(), EngError> {
    let invalid_feed_forward = match spec.feed_forward {
        Some(dense) => {
            dense.intermediate == 0
                || dense.context_limit == 0
                || [spec.experts, spec.top_k, spec.moe_inter, spec.shared_inter]
                    .iter()
                    .any(|dim| *dim != 0)
        }
        None => spec.experts == 0 || spec.moe_inter == 0,
    };
    if spec.layers == 0 || spec.hidden == 0 || spec.vocab == 0 || invalid_feed_forward {
        return Err(EngError::Checkpoint(format!(
            "family {} has non-positive dimensions",
            spec.name
        )));
    }
    let declared = spec.layers_attention.len();
    if declared != spec.layers as usize {
        return Err(EngError::Checkpoint(format!(
            "family {} declares {} layers but {} attention kinds",
            spec.name, spec.layers, declared
        )));
    }
    let needs_linear = spec.layers_attention.contains(&AttentionKind::DeltaNet);
    match spec.linear {
        None if needs_linear => Err(EngError::Checkpoint(format!(
            "family {} declares DeltaNet layers without linear attention dimensions",
            spec.name
        ))),
        None => Ok(()),
        Some(linear) => linear.validate().map_err(|error| match error {
            EngError::Checkpoint(message) => {
                EngError::Checkpoint(format!("family {}: {message}", spec.name))
            }
            other => other,
        }),
    }
}

pub(crate) fn classify(key: &str, spec: &FamilySpec) -> Result<(KeyClass, KeyRole), EngError> {
    let unknown = || EngError::Checkpoint(format!("unknown weight key shape {key:?}"));
    let Some((module, file)) = key.rsplit_once('.') else {
        return Err(unknown());
    };
    let Some(file) = file_kind(file) else {
        return Err(unknown());
    };
    let Some((class, files)) = classify_module(module, spec)? else {
        return Err(unknown());
    };
    if !files.permits(file) {
        return Err(unknown());
    }
    Ok((class, files.role()))
}

fn classify_module(module: &str, spec: &FamilySpec) -> Result<Option<(KeyClass, Files)>, EngError> {
    if let Some(pair) = classify_root(module) {
        return Ok(Some(pair));
    }
    let Some(rest) = module.strip_prefix("language_model.model.layers.") else {
        return Ok(None);
    };
    let Some((layer_str, inner)) = rest.split_once('.') else {
        return Ok(None);
    };
    let Some(layer) = canonical_index(layer_str) else {
        return Ok(None);
    };
    let Some(pair) = classify_layer(inner, layer) else {
        return Ok(None);
    };
    check_module(module, pair.0, spec)?;
    Ok(Some(pair))
}

/// Parses a layer index written in canonical decimal: digits only, with no sign or leading zero.
fn canonical_index(text: &str) -> Option<u32> {
    let digits = !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit());
    if !digits || (text.len() > 1 && text.starts_with('0')) {
        return None;
    }
    text.parse().ok()
}

fn classify_root(module: &str) -> Option<(KeyClass, Files)> {
    match module {
        "language_model.model.embed_tokens" => Some((KeyClass::EmbedTokens, Files::Triplet)),
        "language_model.lm_head" => Some((KeyClass::LmHead, Files::Triplet)),
        "language_model.model.norm" => Some((KeyClass::FinalNorm, Files::WeightOnly)),
        _ => None,
    }
}

fn classify_layer(inner: &str, layer: u32) -> Option<(KeyClass, Files)> {
    match inner {
        "input_layernorm" | "post_attention_layernorm" => {
            Some((KeyClass::LayerNorm { layer }, Files::WeightOnly))
        }
        "linear_attn" => Some((KeyClass::LinearAttn { layer }, Files::AuxOnly)),
        "linear_attn.norm" | "linear_attn.conv1d" => {
            Some((KeyClass::LinearAttn { layer }, Files::WeightOnly))
        }
        "linear_attn.in_proj_a"
        | "linear_attn.in_proj_b"
        | "linear_attn.in_proj_qkv"
        | "linear_attn.in_proj_z"
        | "linear_attn.out_proj" => Some((KeyClass::LinearAttn { layer }, Files::Triplet)),
        "self_attn.q_proj" | "self_attn.k_proj" | "self_attn.v_proj" | "self_attn.o_proj" => {
            Some((KeyClass::SelfAttn { layer }, Files::Triplet))
        }
        "self_attn.q_norm" | "self_attn.k_norm" => {
            Some((KeyClass::SelfAttn { layer }, Files::WeightOnly))
        }
        "mlp.gate" => Some((KeyClass::Router { layer }, Files::Triplet)),
        "mlp.shared_expert_gate" => Some((KeyClass::SharedExpertGate { layer }, Files::Triplet)),
        _ => classify_mlp(inner, layer),
    }
}

fn classify_mlp(inner: &str, layer: u32) -> Option<(KeyClass, Files)> {
    if let Some(projection) = inner.strip_prefix("mlp.").and_then(projection) {
        return Some((KeyClass::DenseMlp { layer, projection }, Files::Triplet));
    }
    if let Some(name) = inner.strip_prefix("mlp.shared_expert.") {
        let projection = projection(name)?;
        return Some((KeyClass::SharedExpert { layer, projection }, Files::Triplet));
    }
    let name = inner.strip_prefix("mlp.switch_mlp.")?;
    let projection = projection(name)?;
    Some((KeyClass::ExpertStack { layer, projection }, Files::Triplet))
}

fn projection(name: &str) -> Option<Projection> {
    match name {
        "gate_proj" => Some(Projection::Gate),
        "up_proj" => Some(Projection::Up),
        "down_proj" => Some(Projection::Down),
        _ => None,
    }
}

fn check_module(module: &str, class: KeyClass, spec: &FamilySpec) -> Result<(), EngError> {
    let wrong_feed_forward = match class {
        KeyClass::DenseMlp { .. } => spec.feed_forward.is_none(),
        KeyClass::ExpertStack { .. }
        | KeyClass::SharedExpert { .. }
        | KeyClass::Router { .. }
        | KeyClass::SharedExpertGate { .. } => spec.feed_forward.is_some(),
        _ => false,
    };
    if wrong_feed_forward {
        return Err(EngError::Checkpoint(format!(
            "weight module {module:?} does not match the family feed-forward kind"
        )));
    }
    let (layer, required) = match class {
        KeyClass::LinearAttn { layer } => (Some(layer), Some(AttentionKind::DeltaNet)),
        KeyClass::SelfAttn { layer } => (Some(layer), Some(AttentionKind::Full)),
        KeyClass::ExpertStack { layer, .. }
        | KeyClass::SharedExpert { layer, .. }
        | KeyClass::DenseMlp { layer, .. }
        | KeyClass::Router { layer }
        | KeyClass::SharedExpertGate { layer }
        | KeyClass::LayerNorm { layer } => (Some(layer), None),
        KeyClass::EmbedTokens | KeyClass::LmHead | KeyClass::FinalNorm => (None, None),
    };
    if let Some(layer) = layer {
        check_layer_module(module, layer, required, spec)?;
    }
    Ok(())
}

fn check_layer_module(
    module: &str,
    layer: u32,
    required: Option<AttentionKind>,
    spec: &FamilySpec,
) -> Result<(), EngError> {
    if layer >= spec.layers {
        return Err(EngError::Checkpoint(format!(
            "weight module {module:?} exceeds the family layer count"
        )));
    }
    if let Some(required) = required {
        let declared = spec.layers_attention[usize::try_from(layer).expect("checked above")];
        if declared != required {
            return Err(EngError::Checkpoint(format!(
                "weight module {module:?} does not match the family attention kind"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::spec::{Act, AttentionKind, LinearAttention, Router};

    static SPEC: FamilySpec = FamilySpec {
        name: "test",
        layers: 2,
        hidden: 64,
        heads_q: 8,
        heads_kv: 2,
        head_dim: 32,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-6,
        model_types: &["test_model"],
        vocab: 96,
        feed_forward: None,
        experts: 4,
        top_k: 4,
        moe_inter: 16,
        shared_inter: 16,
        quant_bits: 4,
        quant_group: 64,
        act: Act::Silu,
        router: Router::SoftmaxTopK { renorm: true },
        mtp: false,
        layers_attention: &[AttentionKind::DeltaNet, AttentionKind::Full],
        linear: Some(LinearAttention {
            key_heads: 2,
            key_head_dim: 16,
            value_heads: 4,
            value_head_dim: 16,
            conv_kernel: 4,
        }),
        hyper: None,
    };

    fn assignment(keys: &[&str]) -> Result<TierAssignment, EngError> {
        let keys: Vec<String> = keys.iter().map(|key| key.to_string()).collect();
        TierAssignment::from_keys(&keys, &SPEC)
    }

    #[test]
    fn dense_mlp_projections_are_resident_packed_triplets() {
        let spec = crate::family::spec::QWEN3_8_27B_DENSE;
        for (name, projection) in [
            ("gate_proj", Projection::Gate),
            ("up_proj", Projection::Up),
            ("down_proj", Projection::Down),
        ] {
            for file in ["weight", "scales", "biases"] {
                let key = format!("language_model.model.layers.0.mlp.{name}.{file}");
                let tiers = TierAssignment::from_keys(std::slice::from_ref(&key), &spec)
                    .expect("dense projection");
                assert_eq!(tiers.tier(&key), Some(KeyTier::Resident));
                assert_eq!(tiers.role(&key), Some(KeyRole::Packed));
                assert_eq!(
                    classify(&key, &spec).unwrap(),
                    (
                        KeyClass::DenseMlp {
                            layer: 0,
                            projection,
                        },
                        KeyRole::Packed,
                    )
                );
                assert!(classify(&key, &SPEC).is_err());
            }
        }
        assert!(
            classify(
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
                &spec
            )
            .is_err()
        );
    }

    #[test]
    fn dense_dimensions_exclude_moe_dimensions() {
        use crate::family::spec::{DenseMlp, QWEN3_8_27B_DENSE};

        for spec in [
            FamilySpec {
                feed_forward: Some(DenseMlp {
                    intermediate: 0,
                    context_limit: 262_144,
                }),
                ..QWEN3_8_27B_DENSE
            },
            FamilySpec {
                feed_forward: Some(DenseMlp {
                    intermediate: 17_408,
                    context_limit: 0,
                }),
                ..QWEN3_8_27B_DENSE
            },
            FamilySpec {
                experts: 1,
                ..QWEN3_8_27B_DENSE
            },
            FamilySpec {
                top_k: 1,
                ..QWEN3_8_27B_DENSE
            },
            FamilySpec {
                moe_inter: 1,
                ..QWEN3_8_27B_DENSE
            },
            FamilySpec {
                shared_inter: 1,
                ..QWEN3_8_27B_DENSE
            },
        ] {
            assert!(TierAssignment::from_keys(&[], &spec).is_err());
        }
    }

    #[test]
    fn assigns_resident_and_expert_stacked_tiers() {
        let tiers = assignment(&[
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
            "language_model.model.layers.1.mlp.switch_mlp.down_proj.biases",
            "language_model.model.layers.0.mlp.gate.weight",
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
            "language_model.model.layers.0.linear_attn.dt_bias",
            "language_model.model.layers.0.linear_attn.conv1d.weight",
            "language_model.model.layers.1.self_attn.q_proj.scales",
            "language_model.model.layers.1.self_attn.k_norm.weight",
            "language_model.model.layers.1.mlp.shared_expert.up_proj.biases",
            "language_model.model.layers.0.mlp.shared_expert_gate.scales",
            "language_model.model.embed_tokens.weight",
            "language_model.lm_head.biases",
            "language_model.model.norm.weight",
            "language_model.model.layers.1.post_attention_layernorm.weight",
        ])
        .unwrap();
        assert_eq!(tiers.len(), 14);
        assert!(!tiers.is_empty());
        for key in [
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
            "language_model.model.layers.1.mlp.switch_mlp.down_proj.biases",
        ] {
            assert_eq!(tiers.tier(key), Some(KeyTier::ExpertStacked), "{key}");
        }
        for key in [
            "language_model.model.layers.0.mlp.gate.weight",
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
            "language_model.model.layers.0.linear_attn.dt_bias",
            "language_model.model.layers.0.linear_attn.conv1d.weight",
            "language_model.model.layers.1.self_attn.q_proj.scales",
            "language_model.model.layers.1.self_attn.k_norm.weight",
            "language_model.model.layers.1.mlp.shared_expert.up_proj.biases",
            "language_model.model.layers.0.mlp.shared_expert_gate.scales",
            "language_model.model.embed_tokens.weight",
            "language_model.lm_head.biases",
            "language_model.model.norm.weight",
            "language_model.model.layers.1.post_attention_layernorm.weight",
        ] {
            assert_eq!(tiers.tier(key), Some(KeyTier::Resident), "{key}");
        }
        assert_eq!(
            tiers.tier("language_model.model.layers.0.mlp.gate.scales"),
            None
        );
    }

    #[test]
    fn rejects_zero_dimensions_before_reading_keys() {
        let keys = ["language_model.model.norm.weight".to_string()];
        for spec in [
            FamilySpec { layers: 0, ..SPEC },
            FamilySpec { hidden: 0, ..SPEC },
            FamilySpec { vocab: 0, ..SPEC },
            FamilySpec { experts: 0, ..SPEC },
            FamilySpec {
                moe_inter: 0,
                ..SPEC
            },
        ] {
            match TierAssignment::from_keys(&keys, &spec) {
                Err(EngError::Checkpoint(message)) => {
                    assert_eq!(message, "family test has non-positive dimensions");
                }
                other => panic!("expected a checkpoint error, got {other:?}"),
            }
        }
    }

    #[test]
    fn accepts_an_empty_key_set() {
        let tiers = assignment(&[]).unwrap();
        assert!(tiers.is_empty());
        assert_eq!(tiers.len(), 0);
    }

    #[test]
    fn rejects_unknown_key_shapes() {
        for key in [
            "language_model.model.layers.0.mlp.other.weight",
            "language_model.model.layers.0.linear_attn.in_proj_x.weight",
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.qweight",
            "language_model.model.layers.0.input_layernorm.scales",
            "language_model.model.layers.0.mlp.gate.A_log",
            "language_model.model.layers.0.linear_attn.norm.scales",
            "language_model.model.layers.0.linear_attn.conv",
            "language_model.model.layers.0.linear_attn.conv1d.scales",
            "language_model.model.layers.0.linear_attn.convNd.weight",
            "language_model.model.layers.0.self_attn.proj.weight",
            "language_model.other.weight",
            "language_model.model.layers.0.mlp.gate",
            "language_model.model.layers.2x.mlp.gate.weight",
            "language_model.model.layers.+1.mlp.gate.weight",
            "language_model.model.layers.01.mlp.gate.weight",
            "language_model.model.layers.00.mlp.gate.weight",
            "language_model.model.layers..mlp.gate.weight",
            "language_model.model.layers.-0.mlp.gate.weight",
            "language_model.model.layers.\u{663}.mlp.gate.weight",
            "language_model.model.layers.4294967296.mlp.gate.weight",
        ] {
            let error = assignment(&[key]).unwrap_err();
            let EngError::Checkpoint(message) = error else {
                panic!("{key}: {error:?}");
            };
            assert_eq!(
                message,
                format!("unknown weight key shape {key:?}"),
                "{key}"
            );
        }
    }

    #[test]
    fn accepts_the_canonical_zero_index() {
        let tiers = assignment(&["language_model.model.layers.0.mlp.gate.weight"]).unwrap();
        assert_eq!(
            tiers.tier("language_model.model.layers.0.mlp.gate.weight"),
            Some(KeyTier::Resident)
        );
    }

    #[test]
    fn rejects_a_key_listed_twice() {
        let key = "language_model.model.layers.0.mlp.gate.weight";
        let error = assignment(&[key, "language_model.model.norm.weight", key]).unwrap_err();
        let EngError::Checkpoint(message) = error else {
            panic!("{error:?}");
        };
        assert_eq!(message, format!("duplicate weight key {key:?}"));
    }

    #[test]
    fn rejects_layer_outside_the_family() {
        let error = assignment(&["language_model.model.layers.2.mlp.gate.weight"]).unwrap_err();
        let EngError::Checkpoint(message) = error else {
            panic!("{error:?}");
        };
        assert_eq!(
            message,
            "weight module \"language_model.model.layers.2.mlp.gate\" exceeds the family layer count"
        );
    }

    #[test]
    fn rejects_attention_kind_mismatches() {
        for module in [
            "language_model.model.layers.1.linear_attn.in_proj_qkv",
            "language_model.model.layers.0.self_attn.q_proj",
        ] {
            let error = assignment(&[&format!("{module}.weight")]).unwrap_err();
            let EngError::Checkpoint(message) = error else {
                panic!("{module}: {error:?}");
            };
            assert_eq!(
                message,
                format!("weight module {module:?} does not match the family attention kind"),
                "{module}"
            );
        }
    }

    #[test]
    fn rejects_inconsistent_layer_attention_coverage() {
        static BROKEN: FamilySpec = FamilySpec {
            name: "broken",
            layers: 2,
            hidden: 64,
            heads_q: 8,
            heads_kv: 2,
            head_dim: 32,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            model_types: &["test_model"],
            vocab: 96,
            feed_forward: None,
            experts: 4,
            top_k: 4,
            moe_inter: 16,
            shared_inter: 16,
            quant_bits: 4,
            quant_group: 64,
            act: Act::Silu,
            router: Router::SoftmaxTopK { renorm: true },
            mtp: false,
            layers_attention: &[AttentionKind::Full],
            linear: None,
            hyper: None,
        };
        let error = TierAssignment::from_keys(&[], &BROKEN).unwrap_err();
        let EngError::Checkpoint(message) = error else {
            panic!("{error:?}");
        };
        assert_eq!(
            message,
            "family broken declares 2 layers but 1 attention kinds"
        );
    }
}
