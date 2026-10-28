//! The pack audit on a tiny Flint-shaped pack, one violated fact per control.

use serde_json::{Value, json};

use super::{
    Geometry, Limits, audit,
    audit::{Admitted, payload_matches},
    fixture::{
        MARKER, PINNED_CONFIG, PackBuilder, SHARDS, expect_refusal, refusal, refuses, run,
        tiny_config,
    },
};
use crate::{
    EngError,
    weights::exl3::{Codebook, source::Checkpoint},
};

const UP_SUH: &str = "model.layers.1.mlp.experts.2.up_proj.suh";

fn nested(pack: &mut PackBuilder) -> &mut serde_json::Map<String, Value> {
    pack.config["quantization_config"].as_object_mut().unwrap()
}

#[test]
fn the_tiny_pack_passes_with_exact_counts_and_extents() {
    let dir = PackBuilder::tiny().write();
    let audit = run(dir.path()).unwrap();
    // 46 matrices (12 attention, 3 dense, 2 x (3 shared + 12 expert), 1 head) x 4 tensors,
    // 3 x 5 layer tensors, 2 routers, the embedding and the final norm.
    assert_eq!(audit.tensors, 46 * 4 + 15 + 2 + 2);
    let roles: Vec<_> = audit.matrices.iter().map(|(r, n)| (*r, *n)).collect();
    assert_eq!(
        roles,
        [
            ("experts.down_proj", 8),
            ("experts.gate_proj", 8),
            ("experts.up_proj", 8),
            ("lm_head", 1),
            ("mlp.down_proj", 1),
            ("mlp.gate_proj", 1),
            ("mlp.up_proj", 1),
            ("self_attn.k_proj", 3),
            ("self_attn.o_proj", 3),
            ("self_attn.q_proj", 3),
            ("self_attn.v_proj", 3),
            ("shared_expert.down_proj", 2),
            ("shared_expert.gate_proj", 2),
            ("shared_expert.up_proj", 2),
        ]
    );
    // Physical rate is twice the bits: 4 bits -> 8, the 5-bit head -> 10.
    let rates: Vec<_> = audit.bits_twice.iter().map(|(b, n)| (*b, *n)).collect();
    assert_eq!(rates, [(8, 45), (10, 1)]);
    assert_eq!(audit.shards.len(), 2);
    assert_eq!(
        audit.shards.iter().map(|s| s.tensors).sum::<usize>(),
        audit.tensors
    );
    for shard in &audit.shards {
        let file = std::fs::metadata(dir.path().join(&shard.name))
            .unwrap()
            .len();
        assert_eq!(shard.file_len, file);
        assert_eq!(8 + shard.header_len + shard.payload, file);
    }
    assert!(audit.dead_vectors.is_empty());
    // What `Matrix::load` reported for the sample, with the axes in [input, output] order.
    for (prefix, shape) in [
        ("model.layers.1.mlp.experts.0.gate_proj", [128, 128]),
        ("model.layers.2.mlp.experts.3.down_proj", [128, 128]),
        ("model.layers.1.mlp.shared_expert.up_proj", [128, 128]),
        ("model.layers.0.mlp.down_proj", [256, 128]),
        ("model.layers.0.self_attn.q_proj", [128, 256]),
        ("model.layers.0.self_attn.o_proj", [256, 128]),
    ] {
        let expected = Admitted {
            prefix: prefix.to_string(),
            shape,
            bits_twice: 8,
            codebook: Codebook::Mul1,
        };
        assert!(
            audit.admitted.contains(&expected),
            "{prefix}: {:?}",
            audit.admitted
        );
    }
    assert_eq!(audit.admitted.len(), 4 + 1 + 3 + 6 + 3);
}

#[test]
fn a_missing_per_expert_tensor_is_named() {
    let message = refusal(|p| p.remove(UP_SUH));
    assert!(message.contains("missing from the index"), "{message}");
    assert!(message.contains(UP_SUH), "{message}");
}

#[test]
fn unexpected_tensors_are_named() {
    let extra_expert = "model.layers.1.mlp.experts.4.up_proj.suh";
    let message = refusal(|p| p.plain(extra_expert, "F16", &[128], 0x3c00));
    assert!(
        message.contains("not part of the Flint schema"),
        "{message}"
    );
    assert!(message.contains(extra_expert), "{message}");
    let bias = "model.layers.1.mlp.gate.e_score_correction_bias";
    refuses(|p| p.plain(bias, "F16", &[4], 0), bias);
    let old_format = "model.layers.0.self_attn.q_proj.su";
    refuses(|p| p.put(old_format, "I32", &[1], vec![0; 4]), old_format);
}

#[test]
fn dtype_and_shape_drift_is_refused_for_every_role() {
    let cases: [(&str, &'static str, &[u32], usize); 16] = [
        (UP_SUH, "F32", &[128], 4),
        (
            "model.layers.1.mlp.experts.0.down_proj.svh",
            "F16",
            &[256],
            2,
        ),
        (
            "model.layers.1.mlp.experts.0.gate_proj.trellis",
            "I32",
            &[8, 8, 32],
            4,
        ),
        ("model.layers.1.mlp.experts.0.gate_proj.mul1", "U32", &[], 4),
        (
            "model.layers.1.mlp.experts.0.gate_proj.mul1",
            "I32",
            &[1],
            4,
        ),
        ("model.layers.1.mlp.experts.0.gate_proj.mul1", "I16", &[], 2),
        ("model.embed_tokens.weight", "F16", &[256, 128], 2),
        ("model.embed_tokens.weight", "BF16", &[128, 256], 2),
        ("model.norm.weight", "F16", &[128], 2),
        ("model.layers.1.input_layernorm.weight", "F32", &[128], 4),
        ("model.layers.1.self_attn.q_norm.weight", "BF16", &[64], 2),
        ("model.layers.1.self_attn.softmax_offset", "BF16", &[2], 2),
        ("model.layers.1.self_attn.softmax_offset", "F16", &[1], 2),
        ("model.layers.1.mlp.gate.weight", "BF16", &[4, 128], 2),
        ("model.layers.1.mlp.gate.weight", "F16", &[128, 4], 2),
        (
            "model.layers.1.mlp.shared_expert.up_proj.suh",
            "F16",
            &[256],
            2,
        ),
    ];
    for (name, dtype, shape, width) in cases {
        let elements: u32 = shape.iter().product();
        let bytes = vec![0u8; elements as usize * width];
        let message = refusal(|p| p.put(name, dtype, shape, bytes));
        assert!(message.contains(name), "{name}: {message}");
        assert!(message.contains("schema requires"), "{name}: {message}");
    }
}

#[test]
fn trellis_extent_must_follow_the_matrix_axes_and_the_declared_bits() {
    // A 3-bit expert, swapped axes on a nonsquare matrix, a 4-bit head, a wrong output tile count.
    for (name, shape) in [
        (
            "model.layers.1.mlp.experts.0.up_proj.trellis",
            [8u32, 8, 48],
        ),
        ("model.layers.0.self_attn.q_proj.trellis", [16, 8, 64]),
        ("lm_head.trellis", [8, 16, 64]),
        ("model.layers.0.self_attn.o_proj.trellis", [8, 16, 64]),
    ] {
        let bytes = vec![0u8; shape.iter().product::<u32>() as usize * 2];
        refuses(|p| p.put(name, "I16", &shape, bytes), name);
    }
}

#[test]
fn mul1_is_the_signed_i32_holding_the_u32_marker() {
    assert_eq!(MARKER, 0x83dc_d12d);
    assert_eq!(MARKER as i32, -2_082_680_531);
    let dir = PackBuilder::tiny().write();
    assert!(run(dir.path()).is_ok());
    let mul1 = "model.layers.1.mlp.experts.0.gate_proj.mul1";
    // The MCG marker, off by one, zero, the sign-flipped word and its complement.
    for word in [0xcbac_1fedu32, MARKER + 1, 0, MARKER ^ 0x8000_0000, !MARKER] {
        let message = refusal(|p| p.entry(mul1).bytes = word.to_le_bytes().to_vec());
        assert!(message.contains("mul1 marker"), "{word:#x}: {message}");
        assert!(message.contains(mul1), "{word:#x}: {message}");
    }
}

#[test]
fn nonfinite_scales_and_16_bit_roles_are_refused() {
    let repeat = |word: u16, count: usize| word.to_le_bytes().repeat(count);
    for (name, bytes) in [
        (UP_SUH, repeat(0x7c00, 128)),
        (UP_SUH, repeat(0xfc00, 128)),
        (
            "model.layers.1.mlp.experts.0.down_proj.svh",
            repeat(0x7e00, 128),
        ),
        ("model.norm.weight", repeat(0x7f80, 128)),
        ("model.layers.1.input_layernorm.weight", repeat(0x7fc0, 128)),
        (
            "model.layers.1.self_attn.q_norm.weight",
            repeat(0xff80, 128),
        ),
        ("model.layers.1.self_attn.softmax_offset", repeat(0x7e00, 2)),
        ("model.layers.1.mlp.gate.weight", repeat(0x7c00, 512)),
    ] {
        let message = refusal(|p| p.entry(name).bytes = bytes);
        assert!(message.contains("nonfinite"), "{name}: {message}");
        assert!(message.contains(name), "{name}: {message}");
    }
}

#[test]
fn zero_and_subnormal_scales_are_reported_not_refused() {
    let mut pack = PackBuilder::tiny();
    let subnormal: Vec<u8> = (1..=128u16).flat_map(u16::to_le_bytes).collect();
    pack.entry("model.layers.1.mlp.experts.1.up_proj.suh").bytes = subnormal;
    let dead = "model.layers.2.mlp.experts.3.down_proj.svh";
    pack.entry(dead).bytes = vec![0; 256];
    pack.entry("model.layers.1.mlp.experts.0.gate_proj.svh")
        .bytes[..2]
        .fill(0);
    let dir = pack.write();
    let audit = run(dir.path()).unwrap();
    let up = &audit.scales["experts.up_proj.suh"];
    assert_eq!(
        (
            up.vectors,
            up.elements,
            up.subnormals,
            up.zeros,
            up.subnormal_vectors
        ),
        (8, 8 * 128, 128, 0, 1)
    );
    let down = &audit.scales["experts.down_proj.svh"];
    assert_eq!((down.zero_vectors, down.zeros), (1, 128));
    let gate = &audit.scales["experts.gate_proj.svh"];
    assert_eq!((gate.zeros, gate.zero_vectors), (1, 0));
    assert_eq!(audit.dead_vectors, [dead]);
}

#[test]
fn shard_extents_are_checked_against_file_sizes() {
    for grow in [false, true] {
        let dir = PackBuilder::tiny().write();
        let path = dir.path().join(SHARDS[1]);
        let mut bytes = std::fs::read(&path).unwrap();
        if grow {
            bytes.push(0);
        } else {
            bytes.pop();
        }
        std::fs::write(&path, bytes).unwrap();
        expect_refusal(run(dir.path()), "does not match the data section end");
    }
}

#[test]
fn the_payload_extent_arithmetic_refuses_every_mismatch() {
    assert_eq!(payload_matches("s", 100, 20, 72).unwrap(), 72);
    for (file, header, tensors) in [
        (100, 20, 71),
        (100, 20, 73),
        (27, 20, 0),
        (7, 0, 0),
        (u64::MAX, u64::MAX, 0),
        (u64::MAX, 0, u64::MAX),
    ] {
        let result = payload_matches("s", file, header, tensors);
        assert!(
            matches!(&result, Err(EngError::Checkpoint(m)) if m.contains("payload")),
            "{file} {header} {tensors}: {result:?}"
        );
    }
}

#[test]
fn the_factor_is_part_of_the_audit() {
    refuses(
        |p| {
            nested(p).remove("expert_interm_div");
        },
        "expert_interm_div is missing",
    );
    refuses(
        |p| {
            nested(p).remove("expert_interm_div");
            p.config["expert_interm_div"] = json!(128);
        },
        "must be nested in quantization_config",
    );
    refuses(
        |p| {
            nested(p).insert("expert_interm_div".into(), json!(1));
        },
        "expert_interm_div must be 128",
    );
    let dir = PackBuilder::tiny().write();
    let path = dir.path().join("config.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let member = "\"expert_interm_div\":128";
    assert_eq!(text.matches(member).count(), 1);
    let duplicated = text.replace(member, &format!("{member},\"expert_interm_div\":1"));
    std::fs::write(&path, duplicated).unwrap();
    expect_refusal(run(dir.path()), "duplicate JSON member");
}

#[test]
fn the_codebook_must_be_declared_as_mul1() {
    refuses(
        |p| {
            nested(p).remove("codebook");
        },
        "codebook must be declared as mul1",
    );
    for other in ["mcg", "3inst"] {
        refuses(
            |p| {
                nested(p).insert("codebook".into(), json!(other));
            },
            "codebook must be declared as mul1",
        );
    }
}

#[test]
fn bits_come_from_the_config_as_integers() {
    for (key, value) in [
        ("bits", json!(3.08)),
        ("bits", json!(0)),
        ("bits", json!(9)),
        ("bits", json!("4")),
        ("head_bits", json!(4.5)),
        ("head_bits", json!(0)),
    ] {
        let needle = format!("quantization_config.{key} must be an integer");
        refuses(
            |p| {
                nested(p).insert(key.into(), value);
            },
            &needle,
        );
    }
    refuses(
        |p| {
            nested(p).remove("head_bits");
        },
        "quantization_config.head_bits must be an integer",
    );
}

#[test]
fn geometry_fields_are_validated_before_any_allocation() {
    for (field, value) in [
        ("hidden_size", json!(129)),
        ("hidden_size", json!(0)),
        ("num_experts", json!(100_000)),
        ("num_experts", json!(0)),
        ("num_hidden_layers", json!(100_000)),
        ("head_dim", json!(0)),
        ("head_dim", json!(96)),
        ("vocab_size", json!(u64::MAX)),
        ("mlp_only_layers", json!([7])),
        ("mlp_only_layers", json!([0, 0])),
        ("mlp_only_layers", json!("0")),
        ("decoder_sparse_step", json!(2)),
        ("num_key_value_heads", json!(3)),
        ("moe_intermediate_size", json!(100)),
    ] {
        let mut config = tiny_config();
        config[field] = value.clone();
        let result = Geometry::from_config(&serde_json::to_vec(&config).unwrap());
        assert!(
            matches!(&result, Err(EngError::Checkpoint(m)) if m.contains(field)),
            "{field}={value}: {result:?}"
        );
    }
    let mut missing = tiny_config();
    missing.as_object_mut().unwrap().remove("num_experts");
    expect_refusal(
        Geometry::from_config(&serde_json::to_vec(&missing).unwrap()),
        "num_experts",
    );
}

#[test]
fn the_published_config_derives_the_pack_geometry() {
    let g = Geometry::published(PINNED_CONFIG).unwrap();
    assert_eq!(
        (
            g.hidden,
            g.dense_ff,
            g.moe_ff,
            g.shared_ff,
            g.experts,
            g.layers
        ),
        (2048, 6144, 512, 1024, 256, 49)
    );
    assert_eq!(
        (
            g.heads,
            g.kv_heads,
            g.head_dim,
            g.vocab,
            g.body_bits,
            g.head_bits
        ),
        (32, 4, 128, 150_144, 4, 5)
    );
    assert_eq!(g.dense_layers, [0, 1]);
}

#[test]
fn the_published_schema_matches_the_scanned_pack() {
    // Independent of the code under test: a header scan of revision e7ab11242b0b counted
    // 146,054 index names, 36,440 EXL3 modules and 20,617,542,816 tensor bytes.
    let plan = Geometry::published(PINNED_CONFIG).unwrap().plan();
    assert_eq!(plan.modules.len(), 36_440);
    assert_eq!(plan.plain.len(), 294);
    assert_eq!(plan.tensor_count(), 146_054);
    assert_eq!(plan.payload_bytes(), 20_617_542_816);
    let experts = plan.modules.iter().filter(|m| m.role == "experts.up_proj");
    assert_eq!(experts.count(), 47 * 256);
}

#[test]
fn the_published_geometry_refuses_a_changed_pinned_dimension() {
    for (field, value) in [
        ("num_experts", json!(255)),
        ("hidden_size", json!(2176)),
        ("mlp_only_layers", json!([0])),
    ] {
        let mut config: Value = serde_json::from_slice(PINNED_CONFIG).unwrap();
        config[field] = value;
        let bytes = serde_json::to_vec(&config).unwrap();
        let result = Geometry::published(&bytes);
        assert!(
            matches!(&result, Err(EngError::Checkpoint(m))
                if m.contains(field) && m.contains("differs from the pinned profile")),
            "{field}: {result:?}"
        );
    }
}

#[test]
fn payload_reads_are_budgeted() {
    let dir = PackBuilder::tiny().write();
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let geometry = Geometry::from_config(&checkpoint.model_config).unwrap();
    let default = Limits::default();
    for limits in [
        Limits {
            batch_bytes: 64,
            ..default
        },
        Limits {
            matrix_bytes: 100,
            ..default
        },
    ] {
        expect_refusal(
            audit(dir.path(), &checkpoint, &geometry, &limits),
            "selected payload exceeds",
        );
    }
}

#[test]
fn the_rate_comes_from_the_config_bits_not_from_four() {
    // A 6-bit body (the 6.0 pack's shape) with the 5-bit head: rate 12 and 10.
    let dir = PackBuilder::tiny_with_bits(6).write();
    let audit = run(dir.path()).unwrap();
    let rates: Vec<_> = audit.bits_twice.iter().map(|(b, n)| (*b, *n)).collect();
    assert_eq!(rates, [(10, 1), (12, 45)]);
    assert!(audit.admitted.iter().all(|a| a.bits_twice == 12));
    // The same tensors under a config that still says 4 bits are refused.
    let mut pack = PackBuilder::tiny_with_bits(6);
    pack.config["quantization_config"]["bits"] = json!(4.0);
    let dir = pack.write();
    expect_refusal(run(dir.path()), "schema requires");
}
