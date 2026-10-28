//! Checked opener for the Humanizer Gemma 4 Q8_0 GGUF checkpoint.
//!
//! open_humanizer_q8 opens one single-file GGUF through the shared reader,
//! checks the gemma4 architecture, the scalar metadata, the per-layer
//! attention arrays, and the exact tensor inventory. It never maps the file
//! and claims no execution.

use std::path::Path;

use crate::{
    EngError,
    weights::{GgufLimits, GgufType, GgufWeights, MetaValue},
};

/// general.architecture of the llama.cpp conversion of Gemma 4.
const ARCHITECTURE: &str = "gemma4";
/// Reader limits of the item 4 rules: a 100-MiB header cap and a 1-GiB tensor cap.
const LIMITS: GgufLimits = GgufLimits {
    max_header_bytes: 100_000_000,
    max_tensor_bytes: 1 << 30,
};
const LAYERS: usize = 48;
const HIDDEN: u64 = 3840;
const FEED_FORWARD: u64 = 15360;
const QUERY_HEADS: u64 = 16;
const VOCAB: u64 = 262_144;
const LOCAL_KV_HEADS: u64 = 8;
const LOCAL_HEAD: u64 = 256;
const GLOBAL_KV_HEADS: u64 = 1;
const GLOBAL_HEAD: u64 = 512;
const GGUF_I32: u32 = 5;
const GGUF_BOOL: u32 = 7;
const SUPPRESSED_TOKENS: [i32; 2] = [258_883, 258_882];

/// Scalar integer metadata that must hold exactly the listed value.
const INT_SCALARS: [(&str, u64); 16] = [
    ("gemma4.block_count", LAYERS as u64),
    ("gemma4.context_length", 262_144),
    ("gemma4.embedding_length", HIDDEN),
    ("gemma4.feed_forward_length", FEED_FORWARD),
    ("gemma4.attention.head_count", QUERY_HEADS),
    ("gemma4.attention.key_length", GLOBAL_HEAD),
    ("gemma4.attention.key_length_swa", LOCAL_HEAD),
    ("gemma4.attention.value_length", GLOBAL_HEAD),
    ("gemma4.attention.value_length_swa", LOCAL_HEAD),
    ("gemma4.attention.sliding_window", 1024),
    ("gemma4.attention.shared_kv_layers", 0),
    ("gemma4.rope.dimension_count", 512),
    ("gemma4.rope.dimension_count_swa", 256),
    ("tokenizer.ggml.bos_token_id", 2),
    ("tokenizer.ggml.eos_token_id", 1),
    ("tokenizer.ggml.unknown_token_id", 3),
];

/// Scalar float metadata that must hold exactly the listed value.
const FLOAT_SCALARS: [(&str, f64); 4] = [
    ("gemma4.rope.freq_base", 1_000_000.0),
    ("gemma4.rope.freq_base_swa", 10_000.0),
    ("gemma4.final_logit_softcapping", 30.0),
    ("gemma4.attention.layer_norm_rms_epsilon", 1.0e-6),
];

/// Opens the pinned Humanizer Q8_0 GGUF and checks its metadata and tensors.
///
/// Opener rejections are EngError::Checkpoint with the prefix gguf model: and
/// name the offending metadata key or tensor. Failures of GgufWeights::open
/// pass through unchanged, so a parse or header-hash failure keeps its own
/// gguf: prefix and a tensor over the limit stays EngError::MemoryPlan.
pub fn open_humanizer_q8(path: &Path, header_sha256: Option<&str>) -> Result<(), EngError> {
    open_humanizer_weights(path, header_sha256).map(|_| ())
}

/// Opens the pinned checkpoint and returns the checked reader, for the resident
/// loader. Applies exactly the checks of open_humanizer_q8.
pub(crate) fn open_humanizer_weights(
    path: &Path,
    header_sha256: Option<&str>,
) -> Result<GgufWeights, EngError> {
    let weights = GgufWeights::open(path, LIMITS, header_sha256)?;
    let header = weights.header();
    check_architecture(header)?;
    check_int_scalars(header)?;
    check_float_scalars(header)?;
    check_bool_scalar(header, "tokenizer.ggml.add_bos_token", true)?;
    let kv = read_i32_array(
        &weights,
        header,
        "gemma4.attention.head_count_kv",
        LAYERS,
        GGUF_I32,
    )?;
    let pattern = read_bool_array(
        &weights,
        header,
        "gemma4.attention.sliding_window_pattern",
        LAYERS,
        GGUF_BOOL,
    )?;
    let suppressed = read_i32_array(
        &weights,
        header,
        "tokenizer.ggml.suppress_tokens",
        SUPPRESSED_TOKENS.len(),
        GGUF_I32,
    )?;
    check_layer_pattern(&kv, &pattern)?;
    if suppressed != SUPPRESSED_TOKENS {
        return Err(reject(format!(
            "metadata key tokenizer.ggml.suppress_tokens is {suppressed:?}, expected {SUPPRESSED_TOKENS:?}"
        )));
    }
    check_inventory(header, &kv)?;
    Ok(weights)
}

/// One expected tensor: storage type and dimensions in file order.
struct Expected {
    name: String,
    ty: GgufType,
    dims: Vec<u64>,
}

/// Builds the exact 667-tensor inventory driven by the checked KV-head array.
fn expected_inventory(kv: &[i32]) -> Vec<Expected> {
    let mut expected = Vec::with_capacity(667);
    expected.push(Expected {
        name: "token_embd.weight".to_owned(),
        ty: GgufType::Q80,
        dims: vec![HIDDEN, VOCAB],
    });
    expected.push(Expected {
        name: "output_norm.weight".to_owned(),
        ty: GgufType::F32,
        dims: vec![HIDDEN],
    });
    expected.push(Expected {
        name: "rope_freqs.weight".to_owned(),
        ty: GgufType::F32,
        dims: vec![256],
    });
    for (layer, &heads) in kv.iter().enumerate() {
        let local = heads == LOCAL_KV_HEADS as i32;
        let head = if local { LOCAL_HEAD } else { GLOBAL_HEAD };
        let kv_heads = if local {
            LOCAL_KV_HEADS
        } else {
            GLOBAL_KV_HEADS
        };
        let base = format!("blk.{layer}.");
        let norm = |suffix: &str, width: u64| Expected {
            name: format!("{base}{suffix}"),
            ty: GgufType::F32,
            dims: vec![width],
        };
        let matrix = |suffix: &str, rows: u64, cols: u64| Expected {
            name: format!("{base}{suffix}"),
            ty: GgufType::Q80,
            dims: vec![rows, cols],
        };
        expected.push(norm("attn_norm.weight", HIDDEN));
        expected.push(matrix("attn_q.weight", HIDDEN, QUERY_HEADS * head));
        expected.push(norm("attn_q_norm.weight", head));
        expected.push(matrix("attn_k.weight", HIDDEN, kv_heads * head));
        expected.push(norm("attn_k_norm.weight", head));
        if local {
            expected.push(matrix("attn_v.weight", HIDDEN, kv_heads * head));
        }
        expected.push(matrix("attn_output.weight", QUERY_HEADS * head, HIDDEN));
        expected.push(norm("post_attention_norm.weight", HIDDEN));
        expected.push(norm("ffn_norm.weight", HIDDEN));
        expected.push(matrix("ffn_gate.weight", HIDDEN, FEED_FORWARD));
        expected.push(matrix("ffn_up.weight", HIDDEN, FEED_FORWARD));
        expected.push(matrix("ffn_down.weight", FEED_FORWARD, HIDDEN));
        expected.push(norm("post_ffw_norm.weight", HIDDEN));
        expected.push(norm("layer_output_scale.weight", 1));
    }
    expected
}

/// Checks the exact tensor set, then every tensor type and dimensions.
fn check_inventory(header: &crate::weights::GgufHeader, kv: &[i32]) -> Result<(), EngError> {
    let mut expected = expected_inventory(kv);
    debug_assert_eq!(expected.len(), 667);
    expected.sort_by(|left, right| left.name.cmp(&right.name));
    let table: std::collections::HashMap<&str, (&GgufType, &[u64])> = expected
        .iter()
        .map(|item| (item.name.as_str(), (&item.ty, item.dims.as_slice())))
        .collect();
    for tensor in header.tensors() {
        let Some((ty, dims)) = table.get(tensor.name()) else {
            return Err(reject(format!("unexpected tensor {}", tensor.name())));
        };
        if tensor.ggml_type() != **ty {
            return Err(reject(format!(
                "tensor {} has type {}, expected {}",
                tensor.name(),
                tensor.ggml_type().name(),
                ty.name()
            )));
        }
        if tensor.dims() != *dims {
            return Err(reject(format!(
                "tensor {} has dims {:?}, expected {:?}",
                tensor.name(),
                tensor.dims(),
                dims
            )));
        }
    }
    for item in &expected {
        if header.tensor(&item.name).is_none() {
            return Err(reject(format!("missing tensor {}", item.name)));
        }
    }
    Ok(())
}

/// Checks the five-local, one-full pattern against both per-layer arrays.
fn check_layer_pattern(kv: &[i32], pattern: &[u8]) -> Result<(), EngError> {
    for (layer, (kv_heads, sliding)) in kv.iter().zip(pattern).enumerate() {
        let local = layer % 6 != 5;
        let want_kv = if local {
            LOCAL_KV_HEADS as i32
        } else {
            GLOBAL_KV_HEADS as i32
        };
        let want_pattern = u8::from(local);
        if *kv_heads != want_kv {
            return Err(reject(format!(
                "metadata key gemma4.attention.head_count_kv is {kv_heads} at layer {layer}, expected {want_kv}"
            )));
        }
        if *sliding != want_pattern {
            return Err(reject(format!(
                "metadata key gemma4.attention.sliding_window_pattern is {sliding} at layer {layer}, expected {want_pattern}"
            )));
        }
    }
    Ok(())
}

fn check_architecture(header: &crate::weights::GgufHeader) -> Result<(), EngError> {
    match header.meta("general.architecture") {
        Some(MetaValue::String(found)) if found == ARCHITECTURE => Ok(()),
        Some(MetaValue::String(found)) => Err(reject(format!(
            "metadata key general.architecture is {found:?}, expected {ARCHITECTURE:?}"
        ))),
        Some(_) => Err(reject("metadata key general.architecture is not a string")),
        None => Err(reject("metadata key general.architecture is missing")),
    }
}

/// Checks every integer scalar; only exact matches pass.
fn check_int_scalars(header: &crate::weights::GgufHeader) -> Result<(), EngError> {
    for (key, want) in INT_SCALARS {
        let found = match header.meta(key) {
            Some(MetaValue::U32(value)) => u64::from(*value),
            Some(MetaValue::U64(value)) => *value,
            Some(_) => {
                return Err(reject(format!(
                    "metadata key {key} is not an unsigned integer"
                )));
            }
            None => return Err(reject(format!("metadata key {key} is missing"))),
        };
        if found != want {
            return Err(reject(format!(
                "metadata key {key} is {found}, expected {want}"
            )));
        }
    }
    Ok(())
}

/// Checks each float scalar against the expected value in its stored precision.
fn check_float_scalars(header: &crate::weights::GgufHeader) -> Result<(), EngError> {
    for (key, want) in FLOAT_SCALARS {
        let (found, want) = match header.meta(key) {
            Some(MetaValue::F32(value)) => (f64::from(*value), f64::from(want as f32)),
            Some(MetaValue::F64(value)) => (*value, want),
            Some(_) => return Err(reject(format!("metadata key {key} is not a float"))),
            None => return Err(reject(format!("metadata key {key} is missing"))),
        };
        if found != want {
            return Err(reject(format!(
                "metadata key {key} is {found}, expected {want}"
            )));
        }
    }
    Ok(())
}

fn check_bool_scalar(
    header: &crate::weights::GgufHeader,
    key: &str,
    want: bool,
) -> Result<(), EngError> {
    match header.meta(key) {
        Some(MetaValue::Bool(found)) if *found == want => Ok(()),
        Some(MetaValue::Bool(found)) => Err(reject(format!(
            "metadata key {key} is {found}, expected {want}"
        ))),
        Some(_) => Err(reject(format!("metadata key {key} is not a bool"))),
        None => Err(reject(format!("metadata key {key} is missing"))),
    }
}

/// Reads one i32 metadata array with a checked element type and length.
fn read_i32_array(
    weights: &GgufWeights,
    header: &crate::weights::GgufHeader,
    key: &str,
    len: usize,
    elem_type: u32,
) -> Result<Vec<i32>, EngError> {
    let bytes = read_array_bytes(weights, header, key, len, elem_type, len * 4)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| i32::from_le_bytes(*chunk))
        .collect())
}

/// Reads one bool metadata array with a checked element type and length.
fn read_bool_array(
    weights: &GgufWeights,
    header: &crate::weights::GgufHeader,
    key: &str,
    len: usize,
    elem_type: u32,
) -> Result<Vec<u8>, EngError> {
    read_array_bytes(weights, header, key, len, elem_type, len)
        .map(|bytes| bytes.iter().map(|byte| u8::from(*byte != 0)).collect())
}

/// Reads the element bytes of one metadata array through the checked seam.
fn read_array_bytes(
    weights: &GgufWeights,
    header: &crate::weights::GgufHeader,
    key: &str,
    len: usize,
    elem_type: u32,
    byte_len: usize,
) -> Result<Vec<u8>, EngError> {
    let meta = match header.meta(key) {
        Some(MetaValue::Array(meta)) => meta,
        Some(_) => return Err(reject(format!("metadata key {key} is not an array"))),
        None => return Err(reject(format!("metadata key {key} is missing"))),
    };
    if meta.elem_type != elem_type {
        return Err(reject(format!(
            "metadata key {key} has element type {}, expected {elem_type}",
            meta.elem_type
        )));
    }
    if meta.len != len as u64 {
        return Err(reject(format!(
            "metadata key {key} has {} elements, expected {len}",
            meta.len
        )));
    }
    let mut buf = vec![0; byte_len];
    weights.read_header_range(meta.byte_range.start, &mut buf)?;
    Ok(buf)
}

fn reject(message: impl Into<String>) -> EngError {
    EngError::Checkpoint(format!("gguf model: {}", message.into()))
}
