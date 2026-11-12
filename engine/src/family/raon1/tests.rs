//! Tests for the raon1 opener. A tiny spec with a distinct value in every field
//! exercises every rejection path on synthetic files; an always-run golden table
//! pins the real spec to the published Q4_K_M file role by role; the env-gated
//! test opens the real artifact.

use std::path::PathBuf;

use super::*;

const ALIGN: usize = 32;

/// All 18 scalar fields mutually distinct (streams and the derived mix count of 17
/// and projection width of 480 included), so a generator that reads the wrong field
/// fails some assertion. Block rules kept: 32 for Q8_0 and Q5_0 rows (the projection
/// width 480, q_lora 288, k_head 64, moe_inter 352), 256 for Q4_K and Q6_K rows
/// (hidden 512, kv_latent 768, dense_inter 1024).
fn tiny_spec() -> Raon1Spec {
    Raon1Spec {
        layers: 3,
        leading_dense: 1,
        hidden: 512,
        vocab: 640,
        heads: 5,
        head_dim: 96,
        q_lora: 288,
        key_mla: 160,
        kv_latent: 768,
        kv_a_width: 832,
        k_head: 64,
        v_head: 80,
        time_step_rank: 208,
        conv_kernel: 9,
        experts: 6,
        moe_inter: 352,
        dense_inter: 1024,
        streams: 4,
        mla_layers: &[2],
        down_q5_0_layers: &[1],
    }
}

/// A unique temporary file path per call, so parallel tests never share one.
fn temp_path(tag: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "raon1-{tag}-{:x}-{}.gguf",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ))
}

fn string(text: &str) -> Vec<u8> {
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(text.as_bytes());
    out
}

fn kv_string(key: &str, value: &str) -> Vec<u8> {
    let mut out = string(key);
    out.extend_from_slice(&8u32.to_le_bytes()); // string
    out.extend_from_slice(&string(value));
    out
}

fn kv_u32(key: &str, value: u32) -> Vec<u8> {
    let mut out = string(key);
    out.extend_from_slice(&4u32.to_le_bytes()); // u32
    out.extend_from_slice(&value.to_le_bytes());
    out
}

/// The pinned metadata of one spec, as the converter writes it (u32 keys).
fn metadata(spec: Raon1Spec, architecture: &str) -> Vec<Vec<u8>> {
    let mut kvs = vec![kv_string("general.architecture", architecture)];
    let pairs: [(&str, u64); 20] = [
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
    kvs.extend(pairs.iter().map(|(key, value)| kv_u32(key, *value as u32)));
    kvs
}

/// The 20 pinned keys, for loops that corrupt one at a time.
const PINNED_KEYS: [&str; 20] = [
    "raon1.block_count",
    "raon1.embedding_length",
    "raon1.vocab_size",
    "raon1.attention.head_count",
    "raon1.attention.q_lora_rank",
    "raon1.attention.kv_lora_rank",
    "raon1.attention.key_length",
    "raon1.attention.key_length_mla",
    "raon1.attention.value_length",
    "raon1.attention.value_length_mla",
    "raon1.kda.head_dim",
    "raon1.ssm.conv_kernel",
    "raon1.ssm.time_step_rank",
    "raon1.expert_count",
    "raon1.expert_used_count",
    "raon1.expert_shared_count",
    "raon1.expert_feed_forward_length",
    "raon1.feed_forward_length",
    "raon1.leading_dense_block_count",
    "raon1.hyper_connection.count",
];

#[derive(Clone)]
struct FileTensor {
    name: String,
    ty: GgufType,
    dims: Vec<u64>,
}

impl FileTensor {
    fn byte_len(&self) -> usize {
        let elements = self.dims.iter().product::<u64>();
        let blocks = elements / self.ty.block_elems();
        (blocks * self.ty.block_bytes()) as usize
    }
}

/// The file tensors of a spec, in inventory order, down-expert types read from
/// the spec's own layer table.
fn file_tensors(spec: Raon1Spec) -> Vec<FileTensor> {
    spec.inventory()
        .into_iter()
        .map(|expected| FileTensor {
            name: expected.name,
            ty: expected.ty,
            dims: expected.dims,
        })
        .collect()
}

/// One deterministic byte per position; the content is dequantizable junk, only the
/// block structure must be valid.
fn gguf_file(kvs: &[Vec<u8>], tensors: &[FileTensor]) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for kv in kvs {
        out.extend_from_slice(kv);
    }
    let mut offsets = Vec::new();
    let mut offset = 0u64;
    for tensor in tensors {
        offsets.push(offset);
        offset += tensor.byte_len() as u64;
        offset = offset.div_ceil(ALIGN as u64) * ALIGN as u64;
    }
    for (tensor, offset) in tensors.iter().zip(&offsets) {
        out.extend_from_slice(&string(&tensor.name));
        out.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        for dim in &tensor.dims {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&tensor.ty.id().to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
    }
    while !out.len().is_multiple_of(ALIGN) {
        out.push(0);
    }
    for tensor in tensors {
        let start = out.len();
        let len = tensor.byte_len();
        out.resize(start + len, 0);
        for (index, byte) in out[start..].iter_mut().enumerate() {
            // bytes stay <= 0x3E so no F16/F32 halfword lands in the NaN/Inf
            // exponent range while still varying across every quant block
            *byte = (index % 63) as u8;
        }
        while !out.len().is_multiple_of(ALIGN) {
            out.push(0);
        }
    }
    out
}

fn open(kvs: &[Vec<u8>], tensors: &[FileTensor], spec: Raon1Spec) -> Result<(), EngError> {
    let path = temp_path("open");
    std::fs::write(&path, gguf_file(kvs, tensors)).unwrap();
    let result = open_with_spec(&path, None, spec).map(|model| {
        let _ = model;
    });
    let _ = std::fs::remove_file(&path);
    result
}

/// The error text of a rejection, or the panic that there was none.
fn rejection(result: Result<(), EngError>) -> String {
    match result {
        Ok(()) => panic!("the open must fail"),
        Err(EngError::Checkpoint(text)) => text,
        Err(other) => panic!("expected a Checkpoint rejection, got {other:?}"),
    }
}

fn replace(tensors: &mut [FileTensor], name: &str, ty: GgufType, dims: Vec<u64>) {
    let tensor = tensors
        .iter_mut()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("no tensor {name} in the file"));
    tensor.ty = ty;
    tensor.dims = dims;
}

#[test]
fn the_tiny_spec_opens() {
    let spec = tiny_spec();
    open(&metadata(spec, "raon1"), &file_tensors(spec), spec).unwrap();
}

#[test]
fn a_truncated_file_fails() {
    let spec = tiny_spec();
    let file = gguf_file(&metadata(spec, "raon1"), &file_tensors(spec));
    let path = temp_path("truncated");
    std::fs::write(&path, &file[..file.len() / 2]).unwrap();
    let error = match open_with_spec(&path, None, spec) {
        Ok(_) => panic!("a truncated file must fail"),
        Err(EngError::Checkpoint(text)) => text,
        Err(other) => panic!("expected a Checkpoint rejection, got {other:?}"),
    };
    let _ = std::fs::remove_file(&path);
    assert!(
        error.contains("beyond the end of the file") || error.contains("truncated"),
        "{error}"
    );
}

#[test]
fn a_duplicate_tensor_name_fails_naming_it() {
    let spec = tiny_spec();
    let mut tensors = file_tensors(spec);
    let first = tensors[1].clone();
    tensors.insert(1, first);
    let error = rejection(open(&metadata(spec, "raon1"), &tensors, spec));
    assert!(error.contains("duplicate tensor name"), "{error}");
    assert!(error.contains(&tensors[1].name), "{error}");
}

#[test]
fn a_missing_architecture_key_fails() {
    let spec = tiny_spec();
    let mut kvs = metadata(spec, "raon1");
    kvs.remove(0);
    let error = rejection(open(&kvs, &file_tensors(spec), spec));
    assert!(error.contains("architecture"), "{error}");
}

#[test]
fn a_wrong_architecture_fails_naming_both() {
    let spec = tiny_spec();
    let error = rejection(open(&metadata(spec, "llama"), &file_tensors(spec), spec));
    assert!(error.contains("llama"), "{error}");
    assert!(error.contains("raon1"), "{error}");
}

#[test]
fn every_pinned_key_is_required_with_its_value() {
    let spec = tiny_spec();
    let tensors = file_tensors(spec);
    for key in PINNED_KEYS {
        let mut kvs = metadata(spec, "raon1");
        let pos = kvs
            .iter()
            .position(|kv| kv.starts_with(&string(key)))
            .unwrap_or_else(|| panic!("key {key} not written by the builder"));
        let old = kvs.remove(pos);
        // missing: named as missing
        let error = rejection(open(&kvs, &tensors, spec));
        assert!(
            error.contains(key) && error.contains("missing"),
            "{key}: {error}"
        );
        // wrong value: named with both values
        kvs.insert(pos, corrupt_u32(&old));
        let error = rejection(open(&kvs, &tensors, spec));
        assert!(error.contains(key), "{key}: {error}");
        assert!(error.contains("expected"), "{key}: {error}");
        kvs.remove(pos);
        kvs.insert(pos, old);
    }
}

/// One u32 metadata pair with its value incremented, so the open names the key.
fn corrupt_u32(kv: &[u8]) -> Vec<u8> {
    let mut out = kv.to_vec();
    let last = out.len();
    let value = u32::from_le_bytes(out[last - 4..].try_into().unwrap());
    out[last - 4..].copy_from_slice(&(value + 1).to_le_bytes());
    out
}

#[test]
fn a_non_integer_pinned_key_fails() {
    let spec = tiny_spec();
    let mut kvs = metadata(spec, "raon1");
    let pos = kvs
        .iter()
        .position(|kv| kv.starts_with(&string("raon1.block_count")))
        .unwrap();
    let old = kvs.remove(pos);
    let value = "three";
    let mut out = string("raon1.block_count");
    out.extend_from_slice(&8u32.to_le_bytes()); // string
    out.extend_from_slice(&string(value));
    kvs.insert(pos, out);
    let error = rejection(open(&kvs, &file_tensors(spec), spec));
    assert!(
        error.contains("raon1.block_count") && error.contains("not a u32 or u64 integer"),
        "{error}"
    );
    let _ = old;
}

#[test]
fn a_missing_tensor_fails_naming_it() {
    let spec = tiny_spec();
    let mut tensors = file_tensors(spec);
    let name = "blk.2.attn_v_b.weight";
    tensors.retain(|tensor| tensor.name != name);
    let error = rejection(open(&metadata(spec, "raon1"), &tensors, spec));
    assert!(error.contains(name) && error.contains("missing"), "{error}");
}

#[test]
fn an_extra_tensor_fails_naming_it() {
    let spec = tiny_spec();
    let mut tensors = file_tensors(spec);
    tensors.push(FileTensor {
        name: "blk.0.attn_k_b.weight".to_owned(), // MLA tensor on an RDA layer
        ty: GgufType::Q80,
        dims: vec![spec.k_head, spec.kv_latent, spec.heads],
    });
    let error = rejection(open(&metadata(spec, "raon1"), &tensors, spec));
    assert!(
        error.contains("blk.0.attn_k_b.weight") && error.contains("not in the raon1 inventory"),
        "{error}"
    );
}

#[test]
fn a_wrong_type_fails_naming_the_tensor_and_both_types() {
    let spec = tiny_spec();
    let mut tensors = file_tensors(spec);
    replace(
        &mut tensors,
        "token_embd.weight",
        GgufType::F16,
        vec![spec.hidden, spec.vocab],
    );
    let error = rejection(open(&metadata(spec, "raon1"), &tensors, spec));
    assert!(error.contains("token_embd.weight"), "{error}");
    assert!(error.contains("F16"), "{error}");
    assert!(error.contains("Q4_K"), "{error}");
}

#[test]
fn a_swapped_dim_pair_fails_naming_both_shapes() {
    let spec = tiny_spec();
    let mut tensors = file_tensors(spec);
    // attn_q_b is not square on the tiny spec (288 x 384), so the swap changes dims
    replace(
        &mut tensors,
        "blk.2.attn_q_b.weight",
        GgufType::Q80,
        vec![spec.heads * spec.key_mla, spec.q_lora],
    );
    let error = rejection(open(&metadata(spec, "raon1"), &tensors, spec));
    assert!(error.contains("blk.2.attn_q_b.weight"), "{error}");
    assert!(error.contains("has dims"), "{error}");
    assert!(error.contains("expected"), "{error}");
}

#[test]
fn a_down_expert_type_swap_fails_naming_the_layer_tensor() {
    let spec = tiny_spec();
    // layer 1 is Q5_0 in the spec's table; Q8_0 there must fail.
    let mut tensors = file_tensors(spec);
    replace(
        &mut tensors,
        "blk.1.ffn_down_exps.weight",
        GgufType::Q80,
        vec![spec.moe_inter, spec.hidden, spec.experts],
    );
    let error = rejection(open(&metadata(spec, "raon1"), &tensors, spec));
    assert!(
        error.contains("blk.1.ffn_down_exps.weight") && error.contains("expected Q5_0"),
        "{error}"
    );
    // layer 2 is Q8_0 in the spec's table; Q5_0 there must fail.
    let mut tensors = file_tensors(spec);
    replace(
        &mut tensors,
        "blk.2.ffn_down_exps.weight",
        GgufType::Q50,
        vec![spec.moe_inter, spec.hidden, spec.experts],
    );
    let error = rejection(open(&metadata(spec, "raon1"), &tensors, spec));
    assert!(
        error.contains("blk.2.ffn_down_exps.weight") && error.contains("expected Q8_0"),
        "{error}"
    );
}

#[test]
fn read_f32_of_an_unknown_name_fails() {
    let spec = tiny_spec();
    let path = temp_path("read-unknown");
    std::fs::write(
        &path,
        gguf_file(&metadata(spec, "raon1"), &file_tensors(spec)),
    )
    .unwrap();
    let model = open_with_spec(&path, None, spec).unwrap();
    let _ = std::fs::remove_file(&path);
    let error = match model.read_f32("blk.9.attn_norm.weight") {
        Ok(values) => panic!(
            "an unknown tensor must not read, got {} values",
            values.len()
        ),
        Err(EngError::Checkpoint(text)) => text,
        Err(other) => panic!("expected a Checkpoint rejection, got {other:?}"),
    };
    assert!(error.contains("blk.9.attn_norm.weight"), "{error}");
}

#[test]
fn read_f32_dequantizes_every_quantized_type_of_the_file() {
    let spec = tiny_spec();
    let path = temp_path("read-types");
    std::fs::write(
        &path,
        gguf_file(&metadata(spec, "raon1"), &file_tensors(spec)),
    )
    .unwrap();
    let model = open_with_spec(&path, None, spec).unwrap();
    let _ = std::fs::remove_file(&path);
    let cases: [(&str, GgufType, u64); 6] = [
        ("blk.0.attn_norm.weight", GgufType::F32, spec.hidden),
        (
            "blk.1.hc_attn_fn.weight",
            GgufType::F16,
            spec.streams * spec.hidden * spec.mix_count(),
        ),
        ("token_embd.weight", GgufType::Q4K, spec.hidden * spec.vocab),
        (
            "blk.1.attn_q.weight",
            GgufType::Q6K,
            spec.hidden * spec.proj_width(),
        ),
        (
            "blk.1.ffn_down_exps.weight",
            GgufType::Q50,
            spec.moe_inter * spec.hidden * spec.experts,
        ),
        (
            "blk.2.ffn_down_exps.weight",
            GgufType::Q80,
            spec.moe_inter * spec.hidden * spec.experts,
        ),
    ];
    for (name, want_ty, elements) in cases {
        let (found_ty, _) = model.weights.read_tensor(name).unwrap();
        assert_eq!(found_ty, want_ty, "{name}");
        let values = model.read_f32(name).unwrap();
        assert_eq!(values.len() as u64, elements, "{name}");
        assert!(
            values.iter().all(|value| value.is_finite()),
            "{name} dequantized to a non-finite value"
        );
    }
}

/// Dims of the roles whose fields alias on the real model, written as literals
/// from the field table — NOT from spec.inventory(). These are the roles at
/// stake: a generator that reads the wrong field (ssm_norm on time_step_rank
/// instead of head_dim) still opens its own generated file, and only these
/// hand-written rows catch it. RDA roles on layer 1, MLA roles on layer 2.
#[test]
fn the_tiny_spec_pins_hand_written_dims_for_the_aliasing_roles() {
    let spec = tiny_spec();
    let tensors = file_tensors(spec);
    let oracle: &[(&str, GgufType, &[u64])] = &[
        ("blk.1.ssm_norm.weight", GgufType::F32, &[96]),
        ("blk.1.ssm_f_a.weight", GgufType::F16, &[512, 208]),
        ("blk.1.ssm_f_b.weight", GgufType::F16, &[208, 480]),
        ("blk.1.ssm_a", GgufType::F32, &[5]),
        ("blk.1.hc_attn_fn.weight", GgufType::F16, &[2048, 17]),
        ("blk.1.hc_ffn_fn.weight", GgufType::F16, &[2048, 17]),
        ("blk.1.ffn_gate_inp.weight", GgufType::F32, &[512, 6]),
        ("blk.2.attn_q_b.weight", GgufType::Q80, &[288, 800]),
        ("blk.2.attn_k_b.weight", GgufType::Q80, &[64, 768, 5]),
        ("blk.2.attn_v_b.weight", GgufType::Q6K, &[768, 80, 5]),
    ];
    for (name, want_ty, want_dims) in oracle {
        let tensor = tensors
            .iter()
            .find(|tensor| &tensor.name == name)
            .unwrap_or_else(|| panic!("{name} missing from the inventory"));
        assert_eq!(&tensor.ty, want_ty, "{name}");
        assert_eq!(tensor.dims, *want_dims, "{name}");
    }
}

/// The tiny file's unpadded source-byte total, as a literal: any drift in a
/// role's type or dims moves it.
#[test]
fn the_tiny_spec_source_bytes_total_12_073_968() {
    let spec = tiny_spec();
    let total: u64 = file_tensors(spec)
        .iter()
        .map(|tensor| tensor.byte_len() as u64)
        .sum();
    assert_eq!(total, 12_073_968);
}

/// The layer index of a "blk.N." name, for scoped role counting.
fn layer_of(name: &str) -> usize {
    name.trim_start_matches("blk.")
        .split('.')
        .next()
        .and_then(|index| index.parse().ok())
        .unwrap_or(0)
}

/// The real spec, pinned role by role to the published Q4_K_M file. This runs
/// without the 19.7 GB artifact: a generator that reads the wrong field fails
/// here, not only in the env-gated test.
#[test]
fn the_real_spec_matches_the_published_q4_k_m_role_table() {
    let spec = RAON_1_MINI_PREVIEW;
    let inventory = spec.inventory();
    assert_eq!(inventory.len(), 731);

    let h = 2560u64;
    let w = 5120u64;
    let mix = 17u64;
    let vocab = 259_840u64;
    let rows: &[(&str, GgufType, &[u64], usize, &str)] = &[
        // (name suffix, type, dims, expected count, layer-kind scope)
        // global
        ("token_embd.weight", GgufType::Q4K, &[h, vocab], 1, "token"),
        ("output_norm.weight", GgufType::F32, &[h], 1, "token"),
        ("output.weight", GgufType::Q6K, &[h, vocab], 1, "token"),
        // every layer
        ("attn_norm.weight", GgufType::F32, &[h], 25, "every"),
        ("ffn_norm.weight", GgufType::F32, &[h], 25, "every"),
        ("hc_attn_base.weight", GgufType::F32, &[mix], 25, "every"),
        ("hc_ffn_base.weight", GgufType::F32, &[mix], 25, "every"),
        (
            "hc_attn_fn.weight",
            GgufType::F16,
            &[10240, mix],
            25,
            "every",
        ),
        (
            "hc_ffn_fn.weight",
            GgufType::F16,
            &[10240, mix],
            25,
            "every",
        ),
        ("hc_attn_scale.weight", GgufType::F32, &[4], 25, "every"),
        ("hc_ffn_scale.weight", GgufType::F32, &[4], 25, "every"),
        // RDA layers (18)
        ("attn_q.weight", GgufType::Q6K, &[h, w], 18, "rda"),
        ("attn_k.weight", GgufType::Q6K, &[h, w], 18, "rda"),
        ("attn_v.weight", GgufType::Q6K, &[h, w], 18, "rda"),
        ("attn_output.weight", GgufType::Q80, &[w, h], 18, "rda"),
        ("ssm_a", GgufType::F32, &[40], 18, "rda"),
        ("ssm_beta.weight", GgufType::F16, &[h, 40], 18, "rda"),
        ("ssm_rda.weight", GgufType::F16, &[h, 40], 18, "rda"),
        ("ssm_conv1d_k.weight", GgufType::F32, &[4, 1, w], 18, "rda"),
        ("ssm_conv1d_q.weight", GgufType::F32, &[4, 1, w], 18, "rda"),
        ("ssm_conv1d_v.weight", GgufType::F32, &[4, 1, w], 18, "rda"),
        ("ssm_dt.bias", GgufType::F32, &[w], 18, "rda"),
        ("ssm_f_a.weight", GgufType::F16, &[h, 128], 18, "rda"),
        ("ssm_f_b.weight", GgufType::F16, &[128, w], 18, "rda"),
        ("ssm_g.weight", GgufType::Q6K, &[h, w], 18, "rda"),
        ("ssm_norm.weight", GgufType::F32, &[128], 18, "rda"),
        // MLA layers (7)
        ("attn_gate.weight", GgufType::Q80, &[h, w], 7, "mla"),
        ("attn_q_a.weight", GgufType::Q6K, &[h, 576], 7, "mla"),
        ("attn_q_a_norm.weight", GgufType::F32, &[576], 7, "mla"),
        ("attn_q_b.weight", GgufType::Q80, &[576, 7680], 7, "mla"),
        ("attn_kv_a_mqa.weight", GgufType::Q6K, &[h, 576], 7, "mla"),
        ("attn_kv_a_norm.weight", GgufType::F32, &[512], 7, "mla"),
        ("attn_k_b.weight", GgufType::Q80, &[128, 512, 40], 7, "mla"),
        ("attn_v_b.weight", GgufType::Q6K, &[512, 128, 40], 7, "mla"),
        ("attn_output.weight", GgufType::Q80, &[w, h], 7, "mla"),
        // MoE layers (24)
        ("ffn_gate_inp.weight", GgufType::F32, &[h, 128], 24, "moe"),
        (
            "ffn_gate_exps.weight",
            GgufType::Q4K,
            &[h, 1088, 128],
            24,
            "moe",
        ),
        (
            "ffn_up_exps.weight",
            GgufType::Q4K,
            &[h, 1088, 128],
            24,
            "moe",
        ),
        (
            "ffn_down_exps.weight",
            GgufType::Q50,
            &[1088, h, 128],
            12,
            "moe",
        ),
        (
            "ffn_down_exps.weight",
            GgufType::Q80,
            &[1088, h, 128],
            12,
            "moe",
        ),
        (
            "ffn_gate_shexp.weight",
            GgufType::Q6K,
            &[h, 1088],
            24,
            "moe",
        ),
        ("ffn_up_shexp.weight", GgufType::Q6K, &[h, 1088], 24, "moe"),
        (
            "ffn_down_shexp.weight",
            GgufType::Q80,
            &[1088, h],
            24,
            "moe",
        ),
        ("exp_probs_b.bias", GgufType::F32, &[128], 24, "moe"),
        // leading dense layer (1)
        ("ffn_gate.weight", GgufType::Q4K, &[h, 12_032], 1, "dense"),
        ("ffn_up.weight", GgufType::Q4K, &[h, 12_032], 1, "dense"),
        ("ffn_down.weight", GgufType::Q6K, &[12_032, h], 1, "dense"),
    ];

    let mut covered = 0usize;
    for (suffix, ty, dims, count, scope) in rows {
        let found = inventory
            .iter()
            .filter(|tensor| {
                if !tensor.name.ends_with(suffix)
                    || tensor.ty != *ty
                    || tensor.dims.as_slice() != *dims
                {
                    return false;
                }
                // scopes pick the layer kind, so roles shared by RDA and MLA (attn_output)
                // count separately instead of matching each other's layers
                match *scope {
                    "token" => !tensor.name.starts_with("blk."),
                    "every" => tensor.name.starts_with("blk."),
                    "rda" => {
                        tensor.name.starts_with("blk.")
                            && spec.kind(layer_of(&tensor.name)) == RaonKind::Rda
                    }
                    "mla" => {
                        tensor.name.starts_with("blk.")
                            && spec.kind(layer_of(&tensor.name)) == RaonKind::Mla
                    }
                    "moe" => {
                        tensor.name.starts_with("blk.")
                            && layer_of(&tensor.name) >= spec.leading_dense
                            && layer_of(&tensor.name) < spec.layers
                    }
                    "dense" => {
                        tensor.name.starts_with("blk.")
                            && layer_of(&tensor.name) < spec.leading_dense
                    }
                    other => unreachable!("unknown scope {other}"),
                }
            })
            .count();
        assert_eq!(
            found, *count,
            "role {suffix} as {ty:?} {dims:?} in scope {scope}"
        );
        covered += found;
    }

    assert_eq!(covered, inventory.len(), "every tensor is covered by a row");

    // the two down-expert layer lists, exactly
    let q5_0: Vec<usize> = (1..spec.layers)
        .filter(|&layer| spec.down_type(layer) == GgufType::Q50)
        .collect();
    assert_eq!(q5_0, vec![3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19]);
    let mla: Vec<usize> = (0..spec.layers)
        .filter(|&layer| spec.kind(layer) == RaonKind::Mla)
        .collect();
    assert_eq!(mla, vec![3, 7, 11, 15, 19, 23, 24]);

    // the byte total of the published conversion, from the table alone
    let total: u64 = inventory
        .iter()
        .map(|tensor| tensor.byte_len(tensor.ty).unwrap())
        .sum();
    assert_eq!(total, 19_674_383_528);
}

/// The published Q4_K_M file when RAON_Q4_KM points at it: the strict open of
/// the real 19,686,434,240-byte artifact. Skipped when the variable is unset.
#[test]
fn the_published_q4_k_m_file_opens_strictly() {
    let Ok(path) = std::env::var("RAON_Q4_KM") else {
        eprintln!("skipped: RAON_Q4_KM is unset");
        return;
    };
    let path = PathBuf::from(path);
    // SHA-256 over the first header_len bytes of the published conversion, checked
    // by the engine's hasher
    const HEADER_SHA: &str = "b8e287e3eae14db453bf8f13addace471cc2472324021369cd809dcac6cf9f17";
    let model = open_raon_1_mini_preview(&path, Some(HEADER_SHA))
        .unwrap_or_else(|error| panic!("the published file must open: {error:?}"));
    assert_eq!(model.weights.header().tensors().len(), 731);

    // one read per type present in the file, with length and finite checks
    let cases: [(&str, GgufType, usize); 6] = [
        ("blk.0.attn_norm.weight", GgufType::F32, 2560),
        ("blk.3.hc_attn_fn.weight", GgufType::F16, 10240 * 17),
        ("blk.0.ffn_gate.weight", GgufType::Q4K, 2560 * 12_032),
        ("blk.3.attn_q_a.weight", GgufType::Q6K, 2560 * 576),
        (
            "blk.3.ffn_down_exps.weight",
            GgufType::Q50,
            1088 * 2560 * 128,
        ),
        ("blk.3.attn_q_b.weight", GgufType::Q80, 576 * 7680),
    ];
    for (name, want_ty, elements) in cases {
        let (found_ty, _) = model.weights.read_tensor(name).unwrap();
        assert_eq!(found_ty, want_ty, "{name}");
        let values = model.read_f32(name).unwrap();
        assert_eq!(values.len(), elements, "{name}");
        assert!(
            values.iter().all(|value| value.is_finite()),
            "{name} dequantized to a non-finite value"
        );
    }
}

#[test]
fn a_wrong_header_hash_is_refused() {
    let Ok(path) = std::env::var("RAON_Q4_KM") else {
        eprintln!("skipped: RAON_Q4_KM is unset");
        return;
    };
    let error = match open_raon_1_mini_preview(Path::new(&path), Some(&"0".repeat(64))) {
        Ok(_) => panic!("a wrong header hash must be refused"),
        Err(EngError::Checkpoint(text)) => text,
        Err(other) => panic!("expected a Checkpoint rejection, got {other:?}"),
    };
    assert!(error.to_lowercase().contains("sha256"), "{error}");
}
