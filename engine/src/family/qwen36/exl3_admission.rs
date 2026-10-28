//! CPU attention admission seam, test-only until the model loader consumes it.

use half::{bf16, f16};

use super::{host::HostShape, host_load::rotary_dim};
use crate::{
    EngError,
    family::{AttentionKind, FamilySpec},
    tensor::{Dtype, Tensor},
    weights::{
        WeightSource,
        exl3::{COMPONENT_SUFFIXES, CONFIG_LIMIT, Checkpoint, Matrix},
    },
};

struct AttentionAdmission {
    /// Q, K, V, O in source order, without reconstructing W-prime.
    projections: [Matrix; 4],
    /// Input, post-attention, Q, K raw zero-centered norms; consumers add one in F32.
    norms: Vec<Tensor>,
}

fn admit_attention(
    checkpoint: &Checkpoint,
    spec: &FamilySpec,
    layer: usize,
    byte_budget: u64,
) -> Result<AttentionAdmission, EngError> {
    if checkpoint.model_config.len() as u64 > CONFIG_LIMIT {
        return Err(EngError::MemoryPlan(
            "EXL3 model config exceeds 1 MiB".into(),
        ));
    }
    spec.check_architecture_config(&checkpoint.model_config)?;
    let shape = HostShape::from_spec(spec, rotary_dim(&checkpoint.model_config, spec.head_dim)?)?;
    let kind = shape.kinds.get(layer).ok_or_else(|| {
        EngError::Checkpoint(format!(
            "EXL3 attention admission: layer {layer} is out of range"
        ))
    })?;
    if *kind != AttentionKind::Full {
        return Err(EngError::Checkpoint(format!(
            "EXL3 attention admission: layer {layer} is not full attention"
        )));
    }
    let width = |heads: usize, dim: usize| {
        heads
            .checked_mul(dim)
            .ok_or_else(|| EngError::MemoryPlan("EXL3 attention width overflow".into()))
    };
    let q = width(shape.attn.heads_q, shape.attn.head_dim)?;
    let kv = width(shape.attn.heads_kv, shape.attn.head_dim)?;
    let expected = [
        [shape.hidden, width(q, 2)?],
        [shape.hidden, kv],
        [shape.hidden, kv],
        [q, shape.hidden],
    ];
    let prefixes = ["q_proj", "k_proj", "v_proj", "o_proj"]
        .map(|role| format!("model.language_model.layers.{layer}.self_attn.{role}"));
    let norms = [
        ("input_layernorm.weight", shape.hidden),
        ("post_attention_layernorm.weight", shape.hidden),
        ("self_attn.q_norm.weight", shape.attn.head_dim),
        ("self_attn.k_norm.weight", shape.attn.head_dim),
    ]
    .map(|(role, values)| {
        (
            format!("model.language_model.layers.{layer}.{role}"),
            values,
        )
    });
    let mut names: Vec<String> = prefixes
        .iter()
        .flat_map(|prefix| COMPONENT_SUFFIXES.map(|suffix| format!("{prefix}.{suffix}")))
        .collect();
    for (name, values) in &norms {
        let (dtype, dims) = checkpoint.source.stored(name).ok_or_else(|| {
            EngError::Checkpoint(format!("EXL3 attention admission: missing tensor {name}"))
        })?;
        norm_layout(dtype, &dims, *values)?;
        names.push(name.clone());
    }
    let keys: Vec<_> = names.iter().map(String::as_str).collect();
    let selected = checkpoint.source.read_selected(&keys, byte_budget)?;
    let load = |index: usize| {
        let prefix = &prefixes[index];
        let matrix = Matrix::load(&selected, prefix, &checkpoint.config)?;
        if matrix.shape() != expected[index] {
            return Err(EngError::Checkpoint(format!(
                "EXL3 attention admission: {prefix} shape {:?}, expected {:?}",
                matrix.shape(),
                expected[index]
            )));
        }
        Ok(matrix)
    };
    let projections = [load(0)?, load(1)?, load(2)?, load(3)?];
    let norms = norms
        .iter()
        .map(|(name, values)| {
            let tensor = selected.tensor(name).ok_or_else(|| {
                EngError::Checkpoint(format!("EXL3 attention admission: missing tensor {name}"))
            })?;
            // Use the uploader's exact F32 validation; preserve the original host tensor.
            effective_norm_bytes(&tensor, *values)?;
            Ok(tensor)
        })
        .collect::<Result<Vec<_>, EngError>>()?;
    Ok(AttentionAdmission { projections, norms })
}

/// Exact source vectors, not reshapes of another tensor with the same byte count.
pub(super) fn norm_layout(dtype: Dtype, shape: &[u32], values: usize) -> Result<usize, EngError> {
    if values == 0 || shape.len() != 1 || shape[0] as usize != values {
        return Err(EngError::Checkpoint(format!(
            "EXL3 attention norm: shape {shape:?}, expected [{values}]"
        )));
    }
    if !matches!(dtype, Dtype::F16 | Dtype::Bf16 | Dtype::F32) {
        return Err(EngError::Checkpoint(format!(
            "EXL3 attention norm: unsupported dtype {dtype:?}"
        )));
    }
    values
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| EngError::MemoryPlan("EXL3 norm staging byte count overflow".into()))
}

/// Convert actual storage to F32, then add one in F32, never in F16 or BF16.
pub(super) fn effective_norm_bytes(tensor: &Tensor, values: usize) -> Result<Vec<u8>, EngError> {
    let output_len = norm_layout(tensor.dtype(), tensor.shape(), values)?;
    let bytes = tensor
        .host_bytes()
        .ok_or_else(|| EngError::Checkpoint("EXL3 attention norm requires host storage".into()))?;
    let width = if tensor.dtype() == Dtype::F32 { 4 } else { 2 };
    if bytes.len() != output_len / (4 / width) {
        return Err(EngError::Checkpoint(
            "EXL3 attention norm has invalid storage length".into(),
        ));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|error| EngError::MemoryPlan(format!("EXL3 norm staging allocation: {error}")))?;
    for word in bytes.chunks_exact(width) {
        let value = match tensor.dtype() {
            Dtype::F16 => f16::from_bits(u16::from_le_bytes([word[0], word[1]])).to_f32(),
            Dtype::Bf16 => bf16::from_bits(u16::from_le_bytes([word[0], word[1]])).to_f32(),
            Dtype::F32 => f32::from_le_bytes([word[0], word[1], word[2], word[3]]),
            _ => unreachable!("norm_layout checked the dtype"),
        };
        let effective = value + 1.0f32;
        if !value.is_finite() || !effective.is_finite() {
            return Err(EngError::Checkpoint(
                "EXL3 attention norm contains nonfinite values".into(),
            ));
        }
        output.extend_from_slice(&effective.to_le_bytes());
    }
    Ok(output)
}

mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::{
        family::{
            qwen36::fixtures::{self, Entry},
            spec::QWEN3_6_35B_A3B,
        },
        weights::exl3::Config,
    };

    const LAYER: usize = 3;
    const BUDGET: u64 = 32 * 1024 * 1024;

    #[test]
    fn complete_attention_is_admitted_without_mlx_or_cuda() {
        let checkpoint = complete_checkpoint();
        let spec = &QWEN3_6_35B_A3B;
        let admitted = admit_attention(&checkpoint, spec, LAYER, BUDGET)
            .expect("complete eight-role EXL3 attention must pass CPU admission");
        let hidden = spec.hidden as usize;
        let q = (spec.heads_q * spec.head_dim) as usize;
        let kv = (spec.heads_kv * spec.head_dim) as usize;
        drop(checkpoint);
        for matrix in &admitted.projections {
            for tensor in [matrix.trellis(), matrix.suh(), matrix.svh()] {
                assert!(tensor.host_bytes().is_some_and(|bytes| !bytes.is_empty()));
            }
        }
        let shapes: Vec<_> = admitted.projections.iter().map(Matrix::shape).collect();
        assert_eq!(
            shapes,
            [[hidden, 2 * q], [hidden, kv], [hidden, kv], [q, hidden]]
        );
        assert!(
            admitted
                .projections
                .iter()
                .all(|matrix| matrix.bitrate_twice() == 6)
        );
        assert_eq!(admitted.norms.len(), 4);
        for (norm, length) in
            admitted
                .norms
                .iter()
                .zip([spec.hidden, spec.hidden, spec.head_dim, spec.head_dim])
        {
            assert_eq!(norm.dtype(), Dtype::Bf16);
            assert_eq!(norm.shape(), [length]);
            assert!(norm.host_bytes().unwrap().iter().all(|byte| *byte == 0));
        }
    }

    #[test]
    fn late_missing_norm_is_a_typed_refusal() {
        let checkpoint = checkpoint_with(|entries, _| {
            let name = format!("model.language_model.layers.{LAYER}.self_attn.k_norm.weight");
            let index = entries.iter().position(|entry| entry.0 == name).unwrap();
            entries.remove(index);
        });
        checkpoint_refusal(checkpoint, "missing tensor", 1);
    }

    #[test]
    fn norm_shape_and_type_are_checked() {
        for member in ["shape", "dtype"] {
            let checkpoint = checkpoint_with(|entries, _| {
                let name = format!("model.language_model.layers.{LAYER}.self_attn.k_norm.weight");
                let entry = entries.iter_mut().find(|entry| entry.0 == name).unwrap();
                if member == "shape" {
                    entry.2 = vec![2, QWEN3_6_35B_A3B.head_dim / 2];
                } else {
                    entry.1 = "I16";
                }
            });
            checkpoint_refusal(
                checkpoint,
                if member == "shape" {
                    "EXL3 attention norm: shape"
                } else {
                    "EXL3 attention norm: unsupported dtype I16"
                },
                1,
            );
        }
    }

    #[test]
    fn otherwise_valid_projection_with_wrong_role_shape_is_refused() {
        let checkpoint = checkpoint_with(|entries, _| {
            for suffix in ["trellis", "suh", "svh", "mul1"] {
                let q = format!("model.language_model.layers.{LAYER}.self_attn.q_proj.{suffix}");
                let k = format!("model.language_model.layers.{LAYER}.self_attn.k_proj.{suffix}");
                let qi = entries.iter().position(|entry| entry.0 == q).unwrap();
                let ki = entries.iter().position(|entry| entry.0 == k).unwrap();
                entries[qi].0 = k;
                entries[ki].0 = q;
            }
        });
        checkpoint_refusal(checkpoint, "q_proj shape", BUDGET);
    }

    #[test]
    fn norm_nonfinite_values_are_refused() {
        let checkpoint = checkpoint_with(|entries, _| {
            let name = format!("model.language_model.layers.{LAYER}.self_attn.k_norm.weight");
            let entry = entries.iter_mut().find(|entry| entry.0 == name).unwrap();
            entry.3[..2].copy_from_slice(&0x7f80u16.to_le_bytes());
        });
        checkpoint_refusal(checkpoint, "nonfinite", BUDGET);
    }

    #[test]
    fn selected_payload_obeys_one_aggregate_budget() {
        let mut total = 0;
        let checkpoint = checkpoint_with(|entries, _| {
            total = entries
                .iter()
                .map(|entry| entry.3.len() as u64)
                .sum::<u64>();
            for role in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                let prefix = format!("model.language_model.layers.{LAYER}.self_attn.{role}.");
                let bundle = entries
                    .iter()
                    .filter(|entry| entry.0.starts_with(&prefix))
                    .map(|entry| entry.3.len() as u64)
                    .sum::<u64>();
                assert!(
                    bundle < total - 1,
                    "each complete bundle fits the refused aggregate budget"
                );
            }
        });
        assert!(admit_attention(&checkpoint, &QWEN3_6_35B_A3B, LAYER, total).is_ok());
        assert!(matches!(
            admit_attention(&checkpoint, &QWEN3_6_35B_A3B, LAYER, total - 1),
            Err(EngError::MemoryPlan(_))
        ));
    }

    #[test]
    fn non_attention_and_out_of_range_layers_are_refused() {
        let checkpoint = complete_checkpoint();
        for (layer, expected) in [
            (0, "is not full attention"),
            (QWEN3_6_35B_A3B.layers as usize, "is out of range"),
        ] {
            let Err(EngError::Checkpoint(message)) =
                admit_attention(&checkpoint, &QWEN3_6_35B_A3B, layer, 1)
            else {
                panic!("expected a typed layer refusal");
            };
            assert!(message.contains(expected), "{message}");
        }
    }

    #[test]
    fn architecture_and_rotary_validation_are_not_bypassed() {
        checkpoint_refusal(
            checkpoint_with(|_, model| model["text_config"]["hidden_size"] = json!(2049)),
            "hidden_size",
            1,
        );
        checkpoint_refusal(
            checkpoint_with(|_, model| {
                model["text_config"]["rope_parameters"]["partial_rotary_factor"] = json!(0.1)
            }),
            "rope configuration",
            1,
        );
        checkpoint_refusal(
            checkpoint_with(|_, model| {
                model["text_config"]["rope_parameters"]["mrope_section"] = json!([u64::MAX, 33]);
            }),
            "mrope_section must sum to half the rotary dimension",
            1,
        );
    }

    #[test]
    fn model_metadata_is_bounded_before_json_parsing() {
        let mut checkpoint = complete_checkpoint();
        checkpoint
            .model_config
            .resize(CONFIG_LIMIT as usize + 1, b' ');
        assert!(matches!(
            admit_attention(&checkpoint, &QWEN3_6_35B_A3B, LAYER, BUDGET),
            Err(EngError::MemoryPlan(_))
        ));
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn f16_norm_adds_one_after_conversion_without_half_rounding() {
        // 2^-11 is representable in F16; 1 + 2^-11 would round to 1 in F16.
        let tensor = Tensor::try_from_host(
            Dtype::F16,
            vec![4],
            [0x1000u16, 0x0000, 0xbc00, 0x3c00]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        )
        .unwrap();
        assert_eq!(
            effective_norm_bytes(&tensor, 4).unwrap(),
            f32_bytes(&[f32::from_bits(0x3f80_1000), 1.0, 0.0, 2.0])
        );
    }

    #[test]
    fn bf16_norm_adds_one_after_conversion_without_bfloat_rounding() {
        // 2^-8 is representable in BF16; 1 + 2^-8 would round to 1 in BF16.
        let tensor = Tensor::try_from_host(
            Dtype::Bf16,
            vec![4],
            [0x3b80u16, 0x8000, 0xbf80, 0x3f80]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        )
        .unwrap();
        assert_eq!(
            effective_norm_bytes(&tensor, 4).unwrap(),
            f32_bytes(&[f32::from_bits(0x3f80_8000), 1.0, 0.0, 2.0])
        );
    }

    #[test]
    fn f32_norm_applies_bias_once() {
        let tensor =
            Tensor::try_from_host(Dtype::F32, vec![3], f32_bytes(&[-1.0, 0.25, 2.0])).unwrap();
        assert_eq!(
            effective_norm_bytes(&tensor, 3).unwrap(),
            f32_bytes(&[0.0, 1.25, 3.0])
        );
    }

    #[test]
    fn refuses_wrong_shape_and_non_float_dtype() {
        for shape in [vec![1, 2], vec![2, 1], vec![]] {
            let count = if shape.is_empty() { 1 } else { 2 };
            let tensor = Tensor::try_from_host(Dtype::F32, shape, vec![0; count * 4]).unwrap();
            assert!(matches!(
                effective_norm_bytes(&tensor, 2),
                Err(EngError::Checkpoint(_))
            ));
        }
        let tensor = Tensor::try_from_host(Dtype::I16, vec![2], vec![0; 4]).unwrap();
        assert!(matches!(
            effective_norm_bytes(&tensor, 2),
            Err(EngError::Checkpoint(_))
        ));
        assert!(matches!(
            norm_layout(Dtype::F32, &[2], 3),
            Err(EngError::Checkpoint(_))
        ));
        assert!(matches!(
            norm_layout(Dtype::F32, &[0], 0),
            Err(EngError::Checkpoint(_))
        ));
    }

    #[test]
    fn refuses_nonfinite_values_in_all_supported_types() {
        for (dtype, bytes) in [
            (Dtype::F16, 0x7c00u16.to_le_bytes().to_vec()),
            (Dtype::F16, 0x7e00u16.to_le_bytes().to_vec()),
            (Dtype::Bf16, 0x7f80u16.to_le_bytes().to_vec()),
            (Dtype::Bf16, 0x7fc0u16.to_le_bytes().to_vec()),
            (Dtype::F32, f32::INFINITY.to_le_bytes().to_vec()),
            (Dtype::F32, f32::NAN.to_le_bytes().to_vec()),
        ] {
            let tensor = Tensor::try_from_host(dtype, vec![1], bytes).unwrap();
            assert!(matches!(
                effective_norm_bytes(&tensor, 1),
                Err(EngError::Checkpoint(_))
            ));
        }
    }

    fn checkpoint_refusal(checkpoint: Checkpoint, expected: &str, budget: u64) {
        let Err(EngError::Checkpoint(message)) =
            admit_attention(&checkpoint, &QWEN3_6_35B_A3B, LAYER, budget)
        else {
            panic!("expected a typed checkpoint refusal");
        };
        assert!(message.contains(expected), "{message}");
    }

    fn complete_checkpoint() -> Checkpoint {
        checkpoint_with(|_, _| {})
    }

    fn checkpoint_with(mutate: impl FnOnce(&mut Vec<Entry>, &mut Value)) -> Checkpoint {
        let spec = &QWEN3_6_35B_A3B;
        let mut model: Value = serde_json::from_slice(include_bytes!(
            "../../../tests/fixtures/qwen3_6_35b_a3b_config.json"
        ))
        .unwrap();
        model.as_object_mut().unwrap().remove("quantization");
        model["quantization_config"] = json!({"quant_method": "exl3", "version": "1.4.9"});

        let hidden = spec.hidden as usize;
        let q = (spec.heads_q * spec.head_dim) as usize;
        let kv = (spec.heads_kv * spec.head_dim) as usize;
        let mut entries = Vec::new();
        for (role, k, n) in [
            ("q_proj", hidden, 2 * q),
            ("k_proj", hidden, kv),
            ("v_proj", hidden, kv),
            ("o_proj", q, hidden),
        ] {
            let prefix = format!("model.language_model.layers.{LAYER}.self_attn.{role}");
            entries.push((
                format!("{prefix}.trellis"),
                "I16",
                vec![(k / 16) as u32, (n / 16) as u32, 48],
                vec![0; (k / 16) * (n / 16) * 48 * 2],
            ));
            for (suffix, values) in [("suh", k), ("svh", n)] {
                entries.push((
                    format!("{prefix}.{suffix}"),
                    "F16",
                    vec![values as u32],
                    vec![0; values * 2],
                ));
            }
            entries.push((
                format!("{prefix}.mul1"),
                "I32",
                vec![],
                0x83dc_d12du32.to_le_bytes().to_vec(),
            ));
        }
        for (name, values) in [
            ("input_layernorm.weight", hidden),
            ("post_attention_layernorm.weight", hidden),
            ("self_attn.q_norm.weight", spec.head_dim as usize),
            ("self_attn.k_norm.weight", spec.head_dim as usize),
        ] {
            fixtures::dense(
                &mut entries,
                &format!("model.language_model.layers.{LAYER}.{name}"),
                vec![values as u32],
                &vec![0.0; values],
            );
        }
        mutate(&mut entries, &mut model);
        let model_config = serde_json::to_vec(&model).unwrap();
        Checkpoint {
            source: fixtures::checkpoint(entries),
            config: Config::from_json(&model_config).unwrap(),
            model_config,
        }
    }
}
