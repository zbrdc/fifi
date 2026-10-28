use super::*;

use crate::EngError;
use crate::family::{AttentionKind, FamilySpec, LinearAttention, QWEN3_6_35B_A3B};
use crate::weights::GgufType;

/// Two layers (DeltaNet, then Full), all dims valid for the CUDA tile rules.
///
/// Some dims coincide: `moe_inter` and the kv width (`heads_kv * head_dim`) are both 256,
/// `shared_inter` and the query width (`heads_q * head_dim`) are both 768, `key_dim` and
/// `value_head_dim` are both 160, and `key_heads` and `heads_kv` are both 2. A test that must
/// tell two of these apart needs its own spec.
pub(in crate::family::gguf_model) const TWO_LAYER_SPEC: FamilySpec = FamilySpec {
    name: "two-layer-test",
    layers: 2,
    hidden: 512,
    heads_q: 6,
    heads_kv: 2,
    head_dim: 128,
    vocab: 24,
    experts: 16,
    top_k: 2,
    moe_inter: 256,
    shared_inter: 768,
    layers_attention: &[AttentionKind::DeltaNet, AttentionKind::Full],
    linear: Some(LinearAttention {
        key_heads: 2,
        key_head_dim: 80,
        value_heads: 8,
        value_head_dim: 160,
        conv_kernel: 4,
    }),
    ..QWEN3_6_35B_A3B
};

// Derived from the header of Qwen3.6-35B-A3B-Q4_K_M.gguf (20,419,565,568 B, 733 tensors), file SHA-256 671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7.
#[derive(Clone, Copy, PartialEq)]
enum Scope {
    Global,
    Every,
    Delta,
    Full,
}
struct Row {
    scope: Scope,
    suffix: &'static str,
    ty: GgufType,
    dims: &'static [u64],
}

const REAL_ROWS: [Row; 28] = [
    Row {
        scope: Scope::Global,
        suffix: "token_embd.weight",
        ty: GgufType::Q4K,
        dims: &[2048, 248_320],
    },
    Row {
        scope: Scope::Global,
        suffix: "output_norm.weight",
        ty: GgufType::F32,
        dims: &[2048],
    },
    Row {
        scope: Scope::Global,
        suffix: "output.weight",
        ty: GgufType::Q6K,
        dims: &[2048, 248_320],
    },
    Row {
        scope: Scope::Every,
        suffix: "attn_norm.weight",
        ty: GgufType::F32,
        dims: &[2048],
    },
    Row {
        scope: Scope::Every,
        suffix: "post_attention_norm.weight",
        ty: GgufType::F32,
        dims: &[2048],
    },
    Row {
        scope: Scope::Every,
        suffix: "ffn_gate_inp.weight",
        ty: GgufType::F32,
        dims: &[2048, 256],
    },
    Row {
        scope: Scope::Every,
        suffix: "ffn_gate_inp_shexp.weight",
        ty: GgufType::F32,
        dims: &[2048],
    },
    Row {
        scope: Scope::Every,
        suffix: "ffn_gate_exps.weight",
        ty: GgufType::Q4K,
        dims: &[2048, 512, 256],
    },
    Row {
        scope: Scope::Every,
        suffix: "ffn_up_exps.weight",
        ty: GgufType::Q4K,
        dims: &[2048, 512, 256],
    },
    Row {
        scope: Scope::Every,
        suffix: "ffn_down_exps.weight",
        ty: GgufType::Q4K,
        dims: &[512, 2048, 256],
    },
    Row {
        scope: Scope::Every,
        suffix: "ffn_gate_shexp.weight",
        ty: GgufType::Q80,
        dims: &[2048, 512],
    },
    Row {
        scope: Scope::Every,
        suffix: "ffn_up_shexp.weight",
        ty: GgufType::Q80,
        dims: &[2048, 512],
    },
    Row {
        scope: Scope::Every,
        suffix: "ffn_down_shexp.weight",
        ty: GgufType::Q80,
        dims: &[512, 2048],
    },
    Row {
        scope: Scope::Delta,
        suffix: "attn_qkv.weight",
        ty: GgufType::Q80,
        dims: &[2048, 8192],
    },
    Row {
        scope: Scope::Delta,
        suffix: "attn_gate.weight",
        ty: GgufType::Q80,
        dims: &[2048, 4096],
    },
    Row {
        scope: Scope::Delta,
        suffix: "ssm_out.weight",
        ty: GgufType::Q80,
        dims: &[4096, 2048],
    },
    Row {
        scope: Scope::Delta,
        suffix: "ssm_alpha.weight",
        ty: GgufType::Q80,
        dims: &[2048, 32],
    },
    Row {
        scope: Scope::Delta,
        suffix: "ssm_beta.weight",
        ty: GgufType::Q80,
        dims: &[2048, 32],
    },
    Row {
        scope: Scope::Delta,
        suffix: "ssm_a",
        ty: GgufType::F32,
        dims: &[32],
    },
    Row {
        scope: Scope::Delta,
        suffix: "ssm_conv1d.weight",
        ty: GgufType::F32,
        dims: &[4, 8192],
    },
    Row {
        scope: Scope::Delta,
        suffix: "ssm_dt.bias",
        ty: GgufType::F32,
        dims: &[32],
    },
    Row {
        scope: Scope::Delta,
        suffix: "ssm_norm.weight",
        ty: GgufType::F32,
        dims: &[128],
    },
    Row {
        scope: Scope::Full,
        suffix: "attn_q.weight",
        ty: GgufType::Q80,
        dims: &[2048, 8192],
    },
    Row {
        scope: Scope::Full,
        suffix: "attn_k.weight",
        ty: GgufType::Q80,
        dims: &[2048, 512],
    },
    Row {
        scope: Scope::Full,
        suffix: "attn_v.weight",
        ty: GgufType::Q80,
        dims: &[2048, 512],
    },
    Row {
        scope: Scope::Full,
        suffix: "attn_output.weight",
        ty: GgufType::Q80,
        dims: &[4096, 2048],
    },
    Row {
        scope: Scope::Full,
        suffix: "attn_q_norm.weight",
        ty: GgufType::F32,
        dims: &[256],
    },
    Row {
        scope: Scope::Full,
        suffix: "attn_k_norm.weight",
        ty: GgufType::F32,
        dims: &[256],
    },
];

fn real_entries() -> Vec<(String, GgufType, Vec<u64>)> {
    let mut entries = Vec::new();
    for row in &REAL_ROWS {
        match row.scope {
            Scope::Global => entries.push((row.suffix.to_owned(), row.ty, row.dims.to_vec())),
            Scope::Every | Scope::Delta | Scope::Full => {
                for (layer, kind) in QWEN3_6_35B_A3B.layers_attention.iter().enumerate() {
                    let wanted = matches!(
                        (row.scope, *kind),
                        (Scope::Every, _)
                            | (Scope::Delta, AttentionKind::DeltaNet)
                            | (Scope::Full, AttentionKind::Full)
                    );
                    if wanted {
                        entries.push((
                            format!("blk.{layer}.{}", row.suffix),
                            row.ty,
                            row.dims.to_vec(),
                        ));
                    }
                }
            }
        }
    }
    entries
}

fn qwen_inventory() -> Inventory {
    Inventory::for_spec(&QWEN3_6_35B_A3B).expect("the Qwen3.6 spec has an inventory")
}

#[test]
fn the_qwen3_6_inventory_matches_the_28_role_table_of_the_real_header() {
    let entries = real_entries();
    assert_eq!(entries.len(), 733, "the table expands to 733 real tensors");
    let inventory = qwen_inventory();
    assert_eq!(inventory.tensors().len(), 733, "inventory tensor count");
    for (name, ty, dims) in entries {
        let tensor = inventory
            .get(&name)
            .unwrap_or_else(|| panic!("tensor {name} is missing from the inventory"));
        assert_eq!(tensor.dims, dims, "tensor {name} dims");
        assert!(
            tensor.allowed_types().contains(&ty),
            "tensor {name} must allow {}",
            ty.name()
        );
        assert_eq!(
            tensor.path() == TensorPath::DeviceF32,
            ty == GgufType::F32,
            "tensor {name} path"
        );
    }
}

#[test]
fn the_qwen3_6_inventory_has_432_tiled_and_301_f32_device_tensors_with_the_real_byte_totals() {
    // The tiled device total 20,347,535,360 B (Q6_K tiles are larger than their source) needs
    // `kernels::cuda::plan::device_bytes`, which is CUDA-only, so the real-file test of
    // packet 5 asserts it. An F32 tensor uploads as read, so its device bytes are the
    // source bytes below and the device total is 20,436,388,352 B (decision 0047 A1).
    let inventory = qwen_inventory();
    let mut tiled = 0u32;
    let mut f32_count = 0u32;
    let mut tiled_by_type: HashMap<GgufType, u32> = HashMap::new();
    let mut tiled_bytes = 0u64;
    let mut f32_bytes = 0u64;
    for (name, ty, _) in real_entries() {
        let tensor = inventory
            .get(&name)
            .unwrap_or_else(|| panic!("tensor {name} is missing from the inventory"));
        let bytes = tensor
            .byte_len(ty)
            .unwrap_or_else(|error| panic!("tensor {name} byte length: {error:?}"));
        if tensor.path() == TensorPath::DeviceTiled {
            tiled += 1;
            tiled_bytes += bytes;
            *tiled_by_type.entry(ty).or_default() += 1;
        } else {
            f32_count += 1;
            f32_bytes += bytes;
        }
    }
    assert_eq!(tiled, 432, "tiled tensors");
    assert_eq!(f32_count, 301, "F32 device tensors");
    assert_eq!(
        tiled_by_type.get(&GgufType::Q80).copied().unwrap_or(0),
        310,
        "Q8_0 tiled tensors"
    );
    assert_eq!(
        tiled_by_type.get(&GgufType::Q4K).copied().unwrap_or(0),
        121,
        "Q4_K tiled tensors"
    );
    assert_eq!(
        tiled_by_type.get(&GgufType::Q6K).copied().unwrap_or(0),
        1,
        "Q6_K tiled tensors"
    );
    assert_eq!(tiled_bytes, 20_319_723_520, "tiled source bytes");
    assert_eq!(f32_bytes, 88_852_992, "F32 device bytes");
}

#[test]
fn a_two_layer_spec_lists_its_38_tensors_in_layer_order_with_dims_from_the_spec_fields() {
    let inventory = Inventory::for_spec(&TWO_LAYER_SPEC)
        .unwrap_or_else(|error| panic!("two-layer inventory: {error:?}"));
    let listed: Vec<(String, Vec<u64>)> = inventory
        .tensors()
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor.dims.clone()))
        .collect();
    let expected: Vec<(String, Vec<u64>)> = vec![
        ("token_embd.weight".to_owned(), vec![512, 24]),
        ("blk.0.attn_norm.weight".to_owned(), vec![512]),
        ("blk.0.post_attention_norm.weight".to_owned(), vec![512]),
        ("blk.0.ffn_gate_inp.weight".to_owned(), vec![512, 16]),
        ("blk.0.ffn_gate_inp_shexp.weight".to_owned(), vec![512]),
        ("blk.0.ffn_gate_exps.weight".to_owned(), vec![512, 256, 16]),
        ("blk.0.ffn_up_exps.weight".to_owned(), vec![512, 256, 16]),
        ("blk.0.ffn_down_exps.weight".to_owned(), vec![256, 512, 16]),
        ("blk.0.ffn_gate_shexp.weight".to_owned(), vec![512, 768]),
        ("blk.0.ffn_up_shexp.weight".to_owned(), vec![512, 768]),
        ("blk.0.ffn_down_shexp.weight".to_owned(), vec![768, 512]),
        ("blk.0.attn_qkv.weight".to_owned(), vec![512, 1600]),
        ("blk.0.attn_gate.weight".to_owned(), vec![512, 1280]),
        ("blk.0.ssm_out.weight".to_owned(), vec![1280, 512]),
        ("blk.0.ssm_alpha.weight".to_owned(), vec![512, 8]),
        ("blk.0.ssm_beta.weight".to_owned(), vec![512, 8]),
        ("blk.0.ssm_a".to_owned(), vec![8]),
        ("blk.0.ssm_conv1d.weight".to_owned(), vec![4, 1600]),
        ("blk.0.ssm_dt.bias".to_owned(), vec![8]),
        ("blk.0.ssm_norm.weight".to_owned(), vec![160]),
        ("blk.1.attn_norm.weight".to_owned(), vec![512]),
        ("blk.1.post_attention_norm.weight".to_owned(), vec![512]),
        ("blk.1.ffn_gate_inp.weight".to_owned(), vec![512, 16]),
        ("blk.1.ffn_gate_inp_shexp.weight".to_owned(), vec![512]),
        ("blk.1.ffn_gate_exps.weight".to_owned(), vec![512, 256, 16]),
        ("blk.1.ffn_up_exps.weight".to_owned(), vec![512, 256, 16]),
        ("blk.1.ffn_down_exps.weight".to_owned(), vec![256, 512, 16]),
        ("blk.1.ffn_gate_shexp.weight".to_owned(), vec![512, 768]),
        ("blk.1.ffn_up_shexp.weight".to_owned(), vec![512, 768]),
        ("blk.1.ffn_down_shexp.weight".to_owned(), vec![768, 512]),
        ("blk.1.attn_q.weight".to_owned(), vec![512, 1536]),
        ("blk.1.attn_k.weight".to_owned(), vec![512, 256]),
        ("blk.1.attn_v.weight".to_owned(), vec![512, 256]),
        ("blk.1.attn_output.weight".to_owned(), vec![768, 512]),
        ("blk.1.attn_q_norm.weight".to_owned(), vec![128]),
        ("blk.1.attn_k_norm.weight".to_owned(), vec![128]),
        ("output_norm.weight".to_owned(), vec![512]),
        ("output.weight".to_owned(), vec![512, 24]),
    ];
    assert_eq!(listed, expected);
    let tiled = inventory
        .tensors()
        .iter()
        .filter(|tensor| tensor.path() == TensorPath::DeviceTiled)
        .count();
    let f32_count = inventory
        .tensors()
        .iter()
        .filter(|tensor| tensor.path() == TensorPath::DeviceF32)
        .count();
    assert_eq!(tiled, 23, "tiled tensors");
    assert_eq!(f32_count, 15, "F32 device tensors");
}

#[test]
fn a_wider_kv_group_changes_only_the_kv_projection_dims() {
    // heads_kv 3 makes the kv width 384, equal to no other quantity of the spec.
    let spec = FamilySpec {
        heads_kv: 3,
        ..TWO_LAYER_SPEC
    };
    let inventory =
        Inventory::for_spec(&spec).unwrap_or_else(|error| panic!("wide-kv inventory: {error:?}"));
    let dims_of = |name: &str| -> Vec<u64> {
        inventory
            .get(name)
            .unwrap_or_else(|| panic!("tensor {name} is missing"))
            .dims
            .clone()
    };
    assert_eq!(dims_of("blk.1.attn_k.weight"), vec![512, 384]);
    assert_eq!(dims_of("blk.1.attn_v.weight"), vec![512, 384]);
    for layer in 0..2 {
        assert_eq!(
            dims_of(&format!("blk.{layer}.ffn_gate_exps.weight")),
            vec![512, 256, 16]
        );
        assert_eq!(
            dims_of(&format!("blk.{layer}.ffn_up_exps.weight")),
            vec![512, 256, 16]
        );
        assert_eq!(
            dims_of(&format!("blk.{layer}.ffn_down_exps.weight")),
            vec![256, 512, 16]
        );
    }
}

#[test]
fn tiled_roles_allow_only_q4_k_q6_k_and_q8_0_and_f32_roles_only_f32() {
    let inventory = Inventory::for_spec(&TWO_LAYER_SPEC)
        .unwrap_or_else(|error| panic!("two-layer inventory: {error:?}"));
    for tensor in inventory.tensors() {
        match tensor.path() {
            TensorPath::DeviceTiled => assert_eq!(
                tensor.allowed_types(),
                &[GgufType::Q4K, GgufType::Q6K, GgufType::Q80],
                "tensor {}",
                tensor.name
            ),
            TensorPath::DeviceF32 => assert_eq!(
                tensor.allowed_types(),
                &[GgufType::F32],
                "tensor {}",
                tensor.name
            ),
        }
        assert!(
            !tensor.allowed_types().contains(&GgufType::F16),
            "tensor {} must not allow F16",
            tensor.name
        );
        assert!(
            !tensor.allowed_types().contains(&GgufType::Q5K),
            "tensor {} must not allow Q5_K",
            tensor.name
        );
    }
}

#[test]
fn get_finds_a_tensor_by_name_and_misses_names_outside_the_spec() {
    let inventory = Inventory::for_spec(&TWO_LAYER_SPEC)
        .unwrap_or_else(|error| panic!("two-layer inventory: {error:?}"));
    let tensor = inventory
        .get("blk.1.attn_q.weight")
        .unwrap_or_else(|| panic!("tensor blk.1.attn_q.weight is missing"));
    assert_eq!(tensor.role, Role::AttnQ, "tensor blk.1.attn_q.weight role");
    assert_eq!(tensor.name, "blk.1.attn_q.weight", "tensor name");
    assert!(
        inventory.get("blk.0.attn_q.weight").is_none(),
        "layer 0 is DeltaNet and must not hold attn_q"
    );
    assert!(
        inventory.get("blk.2.attn_norm.weight").is_none(),
        "the spec has only two layers"
    );
    assert!(
        inventory.get("token_embd").is_none(),
        "token_embd is not a tensor name"
    );
}

#[test]
fn byte_len_counts_whole_blocks_of_the_given_type() {
    let inventory = qwen_inventory();
    for (name, ty, want) in [
        ("blk.3.attn_q.weight", GgufType::Q80, 17_825_792u64),
        ("blk.0.ffn_gate_exps.weight", GgufType::Q4K, 150_994_944),
        ("output.weight", GgufType::Q6K, 417_177_600),
        ("blk.0.ssm_a", GgufType::F32, 128),
    ] {
        let tensor = inventory
            .get(name)
            .unwrap_or_else(|| panic!("tensor {name} is missing from the inventory"));
        let bytes = tensor
            .byte_len(ty)
            .unwrap_or_else(|error| panic!("tensor {name} byte length: {error:?}"));
        assert_eq!(bytes, want, "tensor {name} as {} bytes", ty.name());
    }
}

#[test]
fn byte_len_rejects_a_ragged_row_and_a_size_that_overflows_u64() {
    let inventory = qwen_inventory();
    let ragged = inventory
        .get("blk.0.ssm_norm.weight")
        .unwrap_or_else(|| panic!("tensor blk.0.ssm_norm.weight is missing"));
    let error = ragged.byte_len(GgufType::Q4K).unwrap_err();
    assert_eq!(
        error,
        EngError::Checkpoint(
            "tensor blk.0.ssm_norm.weight: row length 128 is not a multiple of the Q4_K block size 256"
                .to_owned()
        )
    );
    let big = ExpectedTensor {
        name: "big".to_owned(),
        role: Role::AttnQ,
        dims: vec![1 << 40, 1 << 40],
    };
    let error = big.byte_len(GgufType::F32).unwrap_err();
    assert_eq!(
        error,
        EngError::Checkpoint("tensor big: size overflows u64".to_owned())
    );
}

#[test]
fn a_spec_with_the_wrong_layer_count_is_rejected() {
    let spec = FamilySpec {
        layers: 3,
        ..TWO_LAYER_SPEC
    };
    let result = Inventory::for_spec(&spec);
    assert_eq!(
        result.unwrap_err(),
        EngError::Checkpoint(
            "family two-layer-test lists 2 attention kinds for 3 layers".to_owned()
        )
    );
}

#[test]
fn a_qsa_layer_is_rejected() {
    let spec = FamilySpec {
        layers_attention: &[AttentionKind::Qsa, AttentionKind::Full],
        ..TWO_LAYER_SPEC
    };
    let result = Inventory::for_spec(&spec);
    assert_eq!(
        result.unwrap_err(),
        EngError::Checkpoint(
            "family two-layer-test layer 0: attention kind Qsa has no GGUF tensors".to_owned()
        )
    );
}

#[test]
fn a_delta_net_layer_without_a_linear_block_is_rejected() {
    let spec = FamilySpec {
        linear: None,
        ..TWO_LAYER_SPEC
    };
    let result = Inventory::for_spec(&spec);
    assert_eq!(
        result.unwrap_err(),
        EngError::Checkpoint(
            "family two-layer-test has DeltaNet layers but no linear attention block".to_owned()
        )
    );
}
