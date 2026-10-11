use std::path::PathBuf;

use super::inventory::{Inventory, tests::TWO_LAYER_SPEC};
use super::*;
use crate::EngError;
use crate::family::FamilySpec;
use crate::weights::{GgufHeader, GgufType, sha256_hex};

/// Data-section alignment of the fixture files; the header parser's default.
const ALIGN: usize = 32;
/// The four count keys the opener checks, in check order.
const COUNT_KEYS: [&str; 4] = [
    "qwen35moe.block_count",
    "qwen35moe.embedding_length",
    "qwen35moe.expert_count",
    "qwen35moe.expert_used_count",
];

fn string(text: &str) -> Vec<u8> {
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(text.as_bytes());
    out
}

fn kv_string(key: &str, value: &str) -> Vec<u8> {
    let mut out = string(key);
    out.extend_from_slice(&8u32.to_le_bytes());
    out.extend_from_slice(&string(value));
    out
}

fn kv_u32(key: &str, value: u32) -> Vec<u8> {
    let mut out = string(key);
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&value.to_le_bytes());
    out
}

fn kv_u64(key: &str, value: u64) -> Vec<u8> {
    let mut out = string(key);
    out.extend_from_slice(&10u32.to_le_bytes());
    out.extend_from_slice(&value.to_le_bytes());
    out
}

fn spec_counts(spec: &FamilySpec) -> [u32; 4] {
    [spec.layers, spec.hidden, spec.experts, spec.top_k]
}

/// `general.architecture` plus the four count keys of `spec`.
fn metadata(architecture: &str, counts: [u32; 4]) -> Vec<Vec<u8>> {
    let mut kvs = vec![kv_string("general.architecture", architecture)];
    for (key, value) in COUNT_KEYS.iter().zip(counts) {
        kvs.push(kv_u32(key, value));
    }
    kvs
}

/// Distinct exact f32 values for one tensor, so a read-back mix-up cannot pass.
fn pattern(index: usize, count: usize) -> Vec<f32> {
    (0..count)
        .map(|j| (index * 1000 + j) as f32 / 8.0)
        .collect()
}

/// One tensor of the fixture file.
struct FileTensor {
    name: String,
    ty: GgufType,
    dims: Vec<u64>,
    payload: Vec<u8>,
}

impl FileTensor {
    /// Zero payload sized for `dims` stored as `ty`; the division floors, so a
    /// ragged row sizes short instead of panicking, like the header parser.
    fn new(name: &str, ty: GgufType, dims: Vec<u64>) -> Self {
        let elements = dims.iter().product::<u64>();
        let blocks = elements / ty.block_elems() * ty.block_bytes();
        Self {
            name: name.to_owned(),
            ty,
            dims,
            payload: vec![0u8; blocks as usize],
        }
    }
}

/// One valid file of `spec`: F32 for the host tensors, Q8_0 for the tiled ones.
fn file_tensors(spec: &FamilySpec) -> Vec<FileTensor> {
    let inventory =
        Inventory::for_spec(spec).unwrap_or_else(|error| panic!("fixture inventory: {error:?}"));
    inventory
        .tensors()
        .iter()
        .map(|tensor| {
            let ty = match tensor.path() {
                TensorPath::HostF32 => GgufType::F32,
                TensorPath::DeviceTiled => GgufType::Q80,
            };
            FileTensor::new(&tensor.name, ty, tensor.dims.clone())
        })
        .collect()
}

/// Swaps one tensor of the fixture for a different type and shape.
fn replace(tensors: &mut [FileTensor], name: &str, ty: GgufType, dims: Vec<u64>) {
    let tensor = tensors
        .iter_mut()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("tensor {name} is not in the fixture"));
    *tensor = FileTensor::new(name, ty, dims);
}

/// A GGUF v3 file: the kv blobs, then the tensor table, then the payloads at
/// `ALIGN`-aligned offsets of the data section. All integers are little-endian.
fn gguf_file(kvs: &[Vec<u8>], tensors: &[FileTensor]) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for kv in kvs {
        out.extend_from_slice(kv);
    }
    let mut data: Vec<u8> = Vec::new();
    for tensor in tensors {
        out.extend_from_slice(&string(&tensor.name));
        out.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        for &dim in &tensor.dims {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&tensor.ty.id().to_le_bytes());
        data.resize(data.len().next_multiple_of(ALIGN), 0);
        out.extend_from_slice(&(data.len() as u64).to_le_bytes());
        data.extend_from_slice(&tensor.payload);
    }
    out.resize(out.len().next_multiple_of(ALIGN), 0);
    out.extend_from_slice(&data);
    out
}

/// Writes the fixture to a temp file and opens it. The open file descriptor keeps
/// reading after the directory is removed, so the temp dir can drop here.
fn open(kvs: &[Vec<u8>], tensors: &[FileTensor], spec: &FamilySpec) -> Result<GgufModel, EngError> {
    let dir = tempfile::tempdir().expect("create the fixture directory");
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, gguf_file(kvs, tensors)).expect("write the fixture file");
    open_with_spec(&path, None, spec)
}

/// The `Checkpoint` message of a rejection, or a panic naming what came back.
fn rejection(result: Result<GgufModel, EngError>) -> String {
    match result {
        Err(EngError::Checkpoint(message)) => message,
        other => panic!("expected a Checkpoint error, got {other:?}"),
    }
}

#[test]
fn a_two_layer_file_opens_and_reads_its_host_tensors_back_bit_exact() {
    let spec = TWO_LAYER_SPEC;
    let kvs = metadata("qwen35moe", spec_counts(&spec));
    let mut tensors = file_tensors(&spec);
    let mut expected: HashMap<String, Vec<f32>> = HashMap::new();
    for (index, tensor) in tensors.iter_mut().enumerate() {
        if tensor.ty != GgufType::F32 {
            continue;
        }
        let count = tensor.dims.iter().product::<u64>() as usize;
        let values = pattern(index, count);
        tensor.payload = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        expected.insert(tensor.name.clone(), values);
    }
    replace(
        &mut tensors,
        "token_embd.weight",
        GgufType::Q4K,
        vec![512, 24],
    );
    replace(&mut tensors, "output.weight", GgufType::Q6K, vec![512, 24]);
    let bytes = gguf_file(&kvs, &tensors);
    let header =
        GgufHeader::from_bytes(&bytes, bytes.len() as u64, 100_000_000).expect("parse the header");
    let hash = sha256_hex(&header.header_sha256());
    let dir = tempfile::tempdir().expect("create the fixture directory");
    let path: PathBuf = dir.path().join("model.gguf");
    std::fs::write(&path, &bytes).expect("write the fixture file");
    let model = open_with_spec(&path, Some(&hash), &spec)
        .unwrap_or_else(|error| panic!("the two-layer fixture must open: {error:?}"));
    assert_eq!(model.path(), path.as_path());
    assert_eq!(model.plan().count(TensorPath::DeviceTiled), 23);
    assert_eq!(model.plan().count(TensorPath::HostF32), 15);
    assert_eq!(model.plan().tensors().count(), 38);
    let mut entries = model.plan().tensors();
    assert_eq!(
        entries.next(),
        Some(("token_embd.weight", TensorPath::DeviceTiled))
    );
    assert_eq!(
        entries.last(),
        Some(("output.weight", TensorPath::DeviceTiled))
    );
    let tiled = model.plan().tiled_names();
    assert_eq!(tiled.len(), 23);
    assert!(tiled.contains(&"blk.1.attn_q.weight"));
    assert!(!tiled.contains(&"blk.0.attn_norm.weight"));
    let values: u64 = expected.values().map(Vec::len).sum::<usize>() as u64;
    assert_eq!(model.plan().source_bytes(TensorPath::HostF32), 4 * values);
    let host = model
        .read_host_f32()
        .unwrap_or_else(|error| panic!("read the host tensors: {error:?}"));
    assert_eq!(host.len(), expected.len(), "exactly the host tensors");
    for (name, want) in &expected {
        let got = host
            .get(name)
            .unwrap_or_else(|| panic!("tensor {name} is not a host tensor"));
        let got_bits: Vec<u32> = got.iter().map(|value| value.to_bits()).collect();
        let want_bits: Vec<u32> = want.iter().map(|value| value.to_bits()).collect();
        assert_eq!(got_bits, want_bits, "tensor {name} values differ");
    }
}

#[test]
fn a_missing_tensor_fails_the_open_naming_it() {
    let spec = TWO_LAYER_SPEC;
    let kvs = metadata("qwen35moe", spec_counts(&spec));
    let mut tensors = file_tensors(&spec);
    tensors.retain(|tensor| tensor.name != "blk.1.attn_k.weight");
    let message = rejection(open(&kvs, &tensors, &spec));
    assert_eq!(message, "gguf model: tensor blk.1.attn_k.weight is missing");
}

#[test]
fn a_tensor_outside_the_inventory_fails_the_open_naming_it() {
    let spec = TWO_LAYER_SPEC;
    let kvs = metadata("qwen35moe", spec_counts(&spec));
    let mut tensors = file_tensors(&spec);
    tensors.push(FileTensor::new(
        "blk.1.extra.weight",
        GgufType::F32,
        vec![8],
    ));
    let message = rejection(open(&kvs, &tensors, &spec));
    assert_eq!(
        message,
        "gguf model: tensor blk.1.extra.weight is not in the two-layer-test inventory"
    );
}

#[test]
fn a_wrong_dim_fails_the_open_naming_the_tensor_and_both_shapes() {
    let spec = TWO_LAYER_SPEC;
    let kvs = metadata("qwen35moe", spec_counts(&spec));
    let mut tensors = file_tensors(&spec);
    replace(
        &mut tensors,
        "blk.0.ffn_gate_shexp.weight",
        GgufType::Q80,
        vec![512, 760],
    );
    let message = rejection(open(&kvs, &tensors, &spec));
    assert_eq!(
        message,
        "gguf model: tensor blk.0.ffn_gate_shexp.weight has dims [512, 760], expected [512, 768]"
    );
}

#[test]
fn an_f16_tensor_fails_the_open_naming_the_tensor_and_type() {
    let spec = TWO_LAYER_SPEC;
    let kvs = metadata("qwen35moe", spec_counts(&spec));
    let mut tensors = file_tensors(&spec);
    replace(
        &mut tensors,
        "blk.1.attn_k.weight",
        GgufType::F16,
        vec![512, 256],
    );
    let message = rejection(open(&kvs, &tensors, &spec));
    assert_eq!(
        message,
        "gguf model: tensor blk.1.attn_k.weight has type F16, expected one of Q4_K, Q6_K, Q8_0"
    );
}

#[test]
fn a_q5_k_tensor_fails_the_open_naming_the_tensor_and_type() {
    let spec = TWO_LAYER_SPEC;
    let kvs = metadata("qwen35moe", spec_counts(&spec));
    let mut tensors = file_tensors(&spec);
    replace(
        &mut tensors,
        "blk.1.attn_k.weight",
        GgufType::Q5K,
        vec![512, 256],
    );
    let message = rejection(open(&kvs, &tensors, &spec));
    assert_eq!(
        message,
        "gguf: tensor blk.1.attn_k.weight has unsupported type Q5_K"
    );
}

#[test]
fn a_rank_3_stack_with_out_not_a_multiple_of_8_fails_the_open_naming_the_tensor() {
    let spec = FamilySpec {
        moe_inter: 252,
        ..TWO_LAYER_SPEC
    };
    let kvs = metadata("qwen35moe", spec_counts(&spec));
    let mut tensors = file_tensors(&spec);
    // Row length 252 is not a multiple of the Q8_0 block of 32, so those two
    // stacks must be stored as F32 for the file to parse at all.
    replace(
        &mut tensors,
        "blk.0.ffn_down_exps.weight",
        GgufType::F32,
        vec![252, 512, 16],
    );
    replace(
        &mut tensors,
        "blk.1.ffn_down_exps.weight",
        GgufType::F32,
        vec![252, 512, 16],
    );
    let message = rejection(open(&kvs, &tensors, &spec));
    assert!(
        message.starts_with(
            "gguf model: tensor blk.0.ffn_gate_exps.weight has dims [512, 252, 16] that the CUDA tiles cannot hold"
        ),
        "got {message:?}"
    );
}

#[test]
fn a_matrix_that_misses_the_tile_rules_fails_the_open_naming_the_tensor() {
    let cases = [
        (
            FamilySpec {
                vocab: 20,
                ..TWO_LAYER_SPEC
            },
            "gguf model: tensor token_embd.weight has dims [512, 20] that the CUDA tiles cannot hold",
        ),
        (
            FamilySpec {
                hidden: 544,
                ..TWO_LAYER_SPEC
            },
            "gguf model: tensor token_embd.weight has dims [544, 24] that the CUDA tiles cannot hold",
        ),
    ];
    for (index, (spec, want)) in cases.into_iter().enumerate() {
        let kvs = metadata("qwen35moe", spec_counts(&spec));
        let tensors = file_tensors(&spec);
        let message = rejection(open(&kvs, &tensors, &spec));
        assert!(
            message.starts_with(want),
            "case {index}: want {want:?}, got {message:?}"
        );
    }
}

#[test]
fn a_wrong_architecture_fails_the_open_naming_the_key() {
    let spec = TWO_LAYER_SPEC;
    let kvs = metadata("llama", spec_counts(&spec));
    let tensors = file_tensors(&spec);
    let message = rejection(open(&kvs, &tensors, &spec));
    assert_eq!(
        message,
        "gguf model: metadata key general.architecture is \"llama\", expected \"qwen35moe\""
    );
}

#[test]
fn a_missing_architecture_key_fails_the_open_naming_the_key() {
    let spec = TWO_LAYER_SPEC;
    let all = metadata("qwen35moe", spec_counts(&spec));
    let kvs = &all[1..];
    let tensors = file_tensors(&spec);
    let message = rejection(open(kvs, &tensors, &spec));
    assert_eq!(
        message,
        "gguf model: metadata key general.architecture is missing"
    );
}

#[test]
fn a_wrong_count_fails_the_open_naming_the_key() {
    let spec = TWO_LAYER_SPEC;
    let right = spec_counts(&spec);
    for (index, key) in COUNT_KEYS.iter().enumerate() {
        let mut counts = right;
        counts[index] += 1;
        let kvs = metadata("qwen35moe", counts);
        let tensors = file_tensors(&spec);
        let message = rejection(open(&kvs, &tensors, &spec));
        assert_eq!(
            message,
            format!(
                "gguf model: metadata key {key} is {}, expected {}",
                counts[index], right[index]
            )
        );
    }
}

#[test]
fn a_count_that_is_not_an_integer_fails_the_open_naming_the_key() {
    let spec = TWO_LAYER_SPEC;
    let mut kvs = metadata("qwen35moe", spec_counts(&spec));
    kvs[1] = kv_string("qwen35moe.block_count", "2");
    let tensors = file_tensors(&spec);
    let message = rejection(open(&kvs, &tensors, &spec));
    assert_eq!(
        message,
        "gguf model: metadata key qwen35moe.block_count is not a u32 or u64 integer"
    );
}

#[test]
fn a_wrong_header_hash_fails_the_open_naming_the_hash() {
    let spec = TWO_LAYER_SPEC;
    let kvs = metadata("qwen35moe", spec_counts(&spec));
    let tensors = file_tensors(&spec);
    let dir = tempfile::tempdir().expect("create the fixture directory");
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, gguf_file(&kvs, &tensors)).expect("write the fixture file");
    let recorded = "0".repeat(64);
    let message = rejection(open_with_spec(&path, Some(&recorded), &spec));
    assert!(
        message.starts_with("gguf: header sha256 ")
            && message.ends_with(&format!("differs from the recorded {recorded}")),
        "got {message:?}"
    );
}

#[test]
fn a_non_string_architecture_fails_the_open_naming_the_key() {
    let spec = TWO_LAYER_SPEC;
    let mut kvs = metadata("qwen35moe", spec_counts(&spec));
    kvs[0] = kv_u32("general.architecture", 7);
    let tensors = file_tensors(&spec);
    let message = rejection(open(&kvs, &tensors, &spec));
    assert_eq!(
        message,
        "gguf model: metadata key general.architecture is not a string"
    );
}

#[test]
fn a_count_stored_as_u64_is_accepted_and_a_wrong_one_is_rejected() {
    let spec = TWO_LAYER_SPEC;
    let tensors = file_tensors(&spec);
    let mut kvs = metadata("qwen35moe", spec_counts(&spec));
    kvs[3] = kv_u64("qwen35moe.expert_count", u64::from(spec.experts));
    open(&kvs, &tensors, &spec).expect("a u64 count with the right value must open");
    kvs[3] = kv_u64("qwen35moe.expert_count", u64::from(spec.experts) + 1);
    let message = rejection(open(&kvs, &tensors, &spec));
    assert_eq!(
        message,
        format!(
            "gguf model: metadata key qwen35moe.expert_count is {}, expected {}",
            spec.experts + 1,
            spec.experts
        )
    );
}

#[test]
fn a_missing_count_key_fails_the_open_naming_the_key() {
    let spec = TWO_LAYER_SPEC;
    let mut kvs = metadata("qwen35moe", spec_counts(&spec));
    kvs.remove(3);
    let tensors = file_tensors(&spec);
    let message = rejection(open(&kvs, &tensors, &spec));
    assert_eq!(
        message,
        "gguf model: metadata key qwen35moe.expert_count is missing"
    );
}

#[test]
fn a_host_tensor_with_wrong_dims_fails_the_open_naming_the_tensor() {
    let spec = TWO_LAYER_SPEC;
    let kvs = metadata("qwen35moe", spec_counts(&spec));
    let mut tensors = file_tensors(&spec);
    replace(
        &mut tensors,
        "blk.0.ffn_gate_inp.weight",
        GgufType::F32,
        vec![512, 15],
    );
    let message = rejection(open(&kvs, &tensors, &spec));
    assert_eq!(
        message,
        "gguf model: tensor blk.0.ffn_gate_inp.weight has dims [512, 15], expected [512, 16]"
    );
}

#[test]
fn the_tile_rules_accept_the_edges_and_reject_the_next_step() {
    for dims in [
        vec![256, 8],
        vec![32768, 8],
        vec![256, 8, 16],
        vec![512, 8, 16],
    ] {
        check_tiles("t", &dims).unwrap_or_else(|error| panic!("{dims:?} must pass: {error:?}"));
    }
    // Rank 3: 12 * 16 rows is a multiple of 8, but the output count 12 is not.
    let rejected = [
        vec![33024, 8],
        vec![128, 8],
        vec![256, 4],
        vec![256, 12, 16],
    ];
    for dims in rejected {
        let message = match check_tiles("t", &dims) {
            Err(EngError::Checkpoint(message)) => message,
            other => panic!("{dims:?} must be rejected, got {other:?}"),
        };
        let want = format!(
            "gguf model: tensor t has dims {dims:?} that the CUDA tiles cannot hold: rows must be a multiple of 8, the row length a multiple of 256 up to 32768, and the output count of a rank-3 stack a multiple of 8"
        );
        assert_eq!(message, want);
    }
}

#[test]
fn the_limits_are_the_decision_0040_values() {
    assert_eq!(LIMITS.max_header_bytes, 100_000_000);
    assert_eq!(LIMITS.max_tensor_bytes, 1 << 30);
}

/// Opens a sparse file whose only tensor is `blk.0.attn_norm.weight` with `elements`
/// F32 values. The file is `set_len` to the payload end, so almost no disk is written.
fn open_sparse_f32(elements: u64) -> Result<GgufModel, EngError> {
    let spec = TWO_LAYER_SPEC;
    let kvs = metadata("qwen35moe", spec_counts(&spec));
    let tensor = FileTensor {
        name: "blk.0.attn_norm.weight".to_owned(),
        ty: GgufType::F32,
        dims: vec![elements],
        payload: Vec::new(),
    };
    let header = gguf_file(&kvs, &[tensor]);
    let dir = tempfile::tempdir().expect("create the fixture directory");
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, &header).expect("write the fixture header");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("reopen the fixture file");
    file.set_len(header.len() as u64 + elements * 4)
        .expect("extend the fixture file");
    open_with_spec(&path, None, &spec)
}

#[test]
fn a_tensor_over_the_one_gib_limit_fails_the_open_with_memory_plan() {
    match open_sparse_f32((1 << 28) + 8) {
        Err(EngError::MemoryPlan(message)) => assert!(
            message.contains("above the limit of 1073741824"),
            "got {message:?}"
        ),
        other => panic!("expected MemoryPlan, got {other:?}"),
    }
    // Exactly 1 GiB passes the size check; the file then lacks the other tensors.
    let message = rejection(open_sparse_f32(1 << 28));
    assert_eq!(message, "gguf model: tensor token_embd.weight is missing");
}
