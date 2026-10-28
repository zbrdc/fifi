use std::{
    collections::BTreeMap,
    io::{Cursor, Read, Seek, SeekFrom},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use inference_engine::{
    EngError,
    family::{Act, AttentionKind, FamilySpec, Router, WeightPlan, quantization::QuantizationMap},
    tensor::Dtype,
    weights::{LoadLimits, ShardHeader, ShardIo, Tier, TieredWeights, WeightSource},
};

static SPEC: FamilySpec = FamilySpec {
    name: "tiered-test",
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
    top_k: 2,
    moe_inter: 16,
    shared_inter: 16,
    quant_bits: 4,
    quant_group: 64,
    act: Act::Silu,
    router: Router::SoftmaxTopK { renorm: true },
    mtp: false,
    layers_attention: &[AttentionKind::Full, AttentionKind::Full],
    linear: None,
    hyper: None,
};

const CONFIG: &str = r#"{"quantization":{"mode":"affine","bits":4,"group_size":64}}"#;
const SHARD_A: &str = "model-00001-of-00002.safetensors";
const SHARD_B: &str = "model-00002-of-00002.safetensors";
const EMBED: &str = "language_model.model.embed_tokens";
const HEAD: &str = "language_model.lm_head";
const NORM: &str = "language_model.model.norm.weight";
const EXPERT: &str = "language_model.model.layers.0.mlp.switch_mlp.gate_proj";

struct Spec {
    key: String,
    dtype: Dtype,
    shape: Vec<u32>,
    bytes: Vec<u8>,
}

fn dtype_name(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::U32 => "U32",
        Dtype::Bf16 => "BF16",
        Dtype::U8 => "U8",
        Dtype::F32 => "F32",
        other => panic!("unused test dtype {other:?}"),
    }
}

fn width(dtype: Dtype) -> usize {
    match dtype {
        Dtype::U8 => 1,
        Dtype::Bf16 => 2,
        _ => 4,
    }
}

fn spec(key: &str, dtype: Dtype, shape: &[u32]) -> Spec {
    let count: u32 = shape.iter().product();
    let len = count as usize * width(dtype);
    let seed = key.bytes().map(usize::from).sum::<usize>();
    let bytes = (0..len).map(|i| ((i * 7 + seed) % 251) as u8).collect();
    Spec {
        key: key.to_string(),
        dtype,
        shape: shape.to_vec(),
        bytes,
    }
}

fn triplet(module: &str, rows: u32, words: u32) -> Vec<Spec> {
    vec![
        spec(&format!("{module}.weight"), Dtype::U32, &[rows, words]),
        spec(&format!("{module}.scales"), Dtype::Bf16, &[rows, 1]),
        spec(&format!("{module}.biases"), Dtype::Bf16, &[rows, 1]),
    ]
}

fn shard_a() -> Vec<Spec> {
    let mut specs = triplet(EMBED, 2, 8);
    specs.extend(triplet(EXPERT, 2, 2));
    specs
}

fn shard_b() -> Vec<Spec> {
    let mut specs = triplet(HEAD, 2, 8);
    specs.push(spec(NORM, Dtype::Bf16, &[4]));
    specs
}

fn build(specs: &[Spec]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for item in specs {
        let start = data.len();
        data.extend_from_slice(&item.bytes);
        header.insert(
            item.key.clone(),
            serde_json::json!({
                "dtype": dtype_name(item.dtype),
                "shape": item.shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let json = serde_json::to_vec(&header).unwrap();
    let mut file = (json.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&json);
    file.extend_from_slice(&data);
    file
}

fn index_for(shards: &[(&str, &[Spec])]) -> Vec<u8> {
    let mut map = serde_json::Map::new();
    for (name, specs) in shards {
        for item in *specs {
            map.insert(item.key.clone(), serde_json::json!(name));
        }
    }
    serde_json::to_vec(&serde_json::json!({ "weight_map": map })).unwrap()
}

fn plan_of(files: &[&[u8]]) -> WeightPlan {
    let headers: Vec<ShardHeader> = files
        .iter()
        .map(|file| ShardHeader::read(&mut Cursor::new(*file), file.len() as u64, 1 << 20).unwrap())
        .collect();
    let quant = QuantizationMap::from_json(CONFIG.as_bytes()).unwrap();
    WeightPlan::build(&SPEC, &quant, &headers).unwrap()
}

fn limits() -> LoadLimits {
    LoadLimits {
        resident_cap: 1 << 20,
        max_input_bytes: 1 << 20,
        max_header_bytes: 1 << 20,
    }
}

struct Counting {
    inner: Cursor<Vec<u8>>,
    count: Arc<AtomicU64>,
}

impl Read for Counting {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.count.fetch_add(n as u64, Ordering::SeqCst);
        Ok(n)
    }
}

impl Seek for Counting {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

/// Reports the full length on `SeekFrom::End` but holds fewer bytes.
struct Lying {
    inner: Cursor<Vec<u8>>,
    claimed: u64,
}

impl Read for Lying {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buf)
    }
}

impl Seek for Lying {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match pos {
            SeekFrom::End(delta) => Ok(self.claimed.checked_add_signed(delta).unwrap()),
            other => self.inner.seek(other),
        }
    }
}

type Readers = BTreeMap<String, Box<dyn ShardIo>>;

struct Fixture {
    a: Vec<u8>,
    b: Vec<u8>,
    index: Vec<u8>,
    plan: WeightPlan,
}

fn fixture() -> Fixture {
    let (sa, sb) = (shard_a(), shard_b());
    let (a, b) = (build(&sa), build(&sb));
    let index = index_for(&[(SHARD_A, &sa), (SHARD_B, &sb)]);
    let plan = plan_of(&[&a, &b]);
    Fixture { a, b, index, plan }
}

fn plain(fixture: &Fixture) -> Readers {
    BTreeMap::from([
        (
            SHARD_A.to_string(),
            Box::new(Cursor::new(fixture.a.clone())) as Box<dyn ShardIo>,
        ),
        (
            SHARD_B.to_string(),
            Box::new(Cursor::new(fixture.b.clone())) as Box<dyn ShardIo>,
        ),
    ])
}

fn resident_total() -> u64 {
    let all: u64 = shard_a()
        .iter()
        .chain(&shard_b())
        .map(|s| s.bytes.len() as u64)
        .sum();
    let expert: u64 = triplet(EXPERT, 2, 2)
        .iter()
        .map(|s| s.bytes.len() as u64)
        .sum();
    all - expert
}

#[test]
fn two_shards_load_with_resident_bytes_identical_to_the_source() {
    let f = fixture();
    let weights = TieredWeights::from_readers(&f.index, &f.plan, plain(&f), limits()).unwrap();
    for item in shard_a().iter().chain(&shard_b()) {
        if item.key.starts_with(EXPERT) {
            continue;
        }
        let tensor = weights
            .tensor(&item.key)
            .unwrap_or_else(|| panic!("{}", item.key));
        assert_eq!(tensor.dtype(), item.dtype, "{}", item.key);
        assert_eq!(tensor.shape(), item.shape.as_slice(), "{}", item.key);
        assert_eq!(
            tensor.host_bytes(),
            Some(item.bytes.as_slice()),
            "{}",
            item.key
        );
    }
    assert_eq!(weights.resident_bytes(), resident_total());
    assert_eq!(weights.tier(), Tier::Resident);
}

#[test]
fn expert_keys_are_absent_and_their_payload_is_never_read() {
    let f = fixture();
    let count = Arc::new(AtomicU64::new(0));
    let mut readers = plain(&f);
    readers.insert(
        SHARD_A.to_string(),
        Box::new(Counting {
            inner: Cursor::new(f.a.clone()),
            count: count.clone(),
        }),
    );
    let weights = TieredWeights::from_readers(&f.index, &f.plan, readers, limits()).unwrap();
    for suffix in ["weight", "scales", "biases"] {
        assert!(weights.tensor(&format!("{EXPERT}.{suffix}")).is_none());
    }
    let header_len = u64::from_le_bytes(f.a[..8].try_into().unwrap()) + 8;
    let resident_a: u64 = triplet(EMBED, 2, 8)
        .iter()
        .map(|s| s.bytes.len() as u64)
        .sum();
    assert_eq!(count.load(Ordering::SeqCst), header_len + resident_a);
    assert_eq!(weights.resident_bytes(), resident_total());
}

#[test]
fn a_short_read_names_the_shard_and_key() {
    let f = fixture();
    let mut readers = plain(&f);
    let cut = f.b.len() - 2;
    readers.insert(
        SHARD_B.to_string(),
        Box::new(Lying {
            inner: Cursor::new(f.b[..cut].to_vec()),
            claimed: f.b.len() as u64,
        }),
    );
    let error = TieredWeights::from_readers(&f.index, &f.plan, readers, limits()).unwrap_err();
    let EngError::Checkpoint(message) = error else {
        panic!("{error:?}")
    };
    assert!(
        message.contains(SHARD_B) && message.contains(NORM),
        "{message}"
    );
}

#[test]
fn resident_bytes_above_either_cap_fail_with_memory_plan() {
    let f = fixture();
    let total = resident_total();
    let low_resident = LoadLimits {
        resident_cap: total - 1,
        ..limits()
    };
    let low_input = LoadLimits {
        max_input_bytes: total - 1,
        ..limits()
    };
    for bad in [low_resident, low_input] {
        let error = TieredWeights::from_readers(&f.index, &f.plan, plain(&f), bad).unwrap_err();
        assert!(matches!(error, EngError::MemoryPlan(_)), "{error:?}");
    }
    let exact = LoadLimits {
        resident_cap: total,
        max_input_bytes: total,
        ..limits()
    };
    assert!(TieredWeights::from_readers(&f.index, &f.plan, plain(&f), exact).is_ok());
}

#[test]
fn an_indexed_key_missing_from_every_shard_header_fails() {
    let f = fixture();
    let mut map: serde_json::Value = serde_json::from_slice(&f.index).unwrap();
    map["weight_map"]["language_model.model.layers.1.mlp.gate.weight"] = SHARD_A.into();
    let index = serde_json::to_vec(&map).unwrap();
    let error = TieredWeights::from_readers(&index, &f.plan, plain(&f), limits()).unwrap_err();
    let EngError::Checkpoint(message) = error else {
        panic!("{error:?}")
    };
    assert!(
        message.contains("language_model.model.layers.1.mlp.gate.weight"),
        "{message}"
    );
}

#[test]
fn a_header_key_missing_from_the_index_fails() {
    let f = fixture();
    let mut map: serde_json::Value = serde_json::from_slice(&f.index).unwrap();
    map["weight_map"].as_object_mut().unwrap().remove(NORM);
    let index = serde_json::to_vec(&map).unwrap();
    let error = TieredWeights::from_readers(&index, &f.plan, plain(&f), limits()).unwrap_err();
    assert!(matches!(error, EngError::Checkpoint(_)), "{error:?}");
}

#[test]
fn a_key_indexed_under_the_wrong_shard_fails() {
    let f = fixture();
    let mut map: serde_json::Value = serde_json::from_slice(&f.index).unwrap();
    map["weight_map"][NORM] = SHARD_A.into();
    let index = serde_json::to_vec(&map).unwrap();
    let error = TieredWeights::from_readers(&index, &f.plan, plain(&f), limits()).unwrap_err();
    let EngError::Checkpoint(message) = error else {
        panic!("{error:?}")
    };
    assert!(
        message.contains(&format!("missing from shard {SHARD_A:?}")),
        "{message}"
    );
    assert!(
        message.contains(SHARD_A) && message.contains(NORM),
        "{message}"
    );
}

#[test]
fn a_header_key_that_the_index_assigns_to_another_shard_fails() {
    let f = fixture();
    let key = format!("{EMBED}.weight");
    let mut map: serde_json::Value = serde_json::from_slice(&f.index).unwrap();
    map["weight_map"][&key] = SHARD_B.into();
    let index = serde_json::to_vec(&map).unwrap();
    let error = TieredWeights::from_readers(&index, &f.plan, plain(&f), limits()).unwrap_err();
    let EngError::Checkpoint(message) = error else {
        panic!("{error:?}")
    };
    assert!(message.contains("index names shard"), "{message}");
    assert!(
        message.contains(&key) && message.contains(SHARD_A) && message.contains(SHARD_B),
        "{message}"
    );
}

#[test]
fn a_shard_with_only_expert_tensors_loads_no_resident_bytes() {
    let (only_experts, rest) = (triplet(EXPERT, 2, 2), {
        let mut specs = triplet(EMBED, 2, 8);
        specs.extend(triplet(HEAD, 2, 8));
        specs.push(spec(NORM, Dtype::Bf16, &[4]));
        specs
    });
    let (a, b) = (build(&only_experts), build(&rest));
    let index = index_for(&[(SHARD_A, &only_experts), (SHARD_B, &rest)]);
    let plan = plan_of(&[&a, &b]);
    let readers: Readers = BTreeMap::from([
        (
            SHARD_A.to_string(),
            Box::new(Cursor::new(a)) as Box<dyn ShardIo>,
        ),
        (
            SHARD_B.to_string(),
            Box::new(Cursor::new(b)) as Box<dyn ShardIo>,
        ),
    ]);
    let weights = TieredWeights::from_readers(&index, &plan, readers, limits()).unwrap();
    let expected: u64 = rest.iter().map(|s| s.bytes.len() as u64).sum();
    assert_eq!(weights.resident_bytes(), expected);
    assert!(weights.tensor(&format!("{EXPERT}.weight")).is_none());
    assert!(weights.tensor(NORM).is_some());
}

#[test]
fn the_reader_set_must_equal_the_index_shard_set() {
    let f = fixture();
    let mut readers = plain(&f);
    readers.remove(SHARD_B);
    let error = TieredWeights::from_readers(&f.index, &f.plan, readers, limits()).unwrap_err();
    assert!(matches!(error, EngError::Checkpoint(_)), "{error:?}");
}

#[test]
fn a_stored_dtype_that_disagrees_with_the_plan_fails() {
    let f = fixture();
    let mut specs = shard_a();
    specs[0] = spec(&format!("{EMBED}.weight"), Dtype::F32, &[2, 8]);
    let altered = build(&specs);
    let mut readers = plain(&f);
    readers.insert(SHARD_A.to_string(), Box::new(Cursor::new(altered)));
    let error = TieredWeights::from_readers(&f.index, &f.plan, readers, limits()).unwrap_err();
    let EngError::Checkpoint(message) = error else {
        panic!("{error:?}")
    };
    assert!(message.contains(&format!("{EMBED}.weight")), "{message}");
}

#[test]
fn a_header_larger_than_the_header_budget_fails_with_memory_plan() {
    let f = fixture();
    let tight = LoadLimits {
        max_header_bytes: 8,
        ..limits()
    };
    let error = TieredWeights::from_readers(&f.index, &f.plan, plain(&f), tight).unwrap_err();
    assert!(matches!(error, EngError::MemoryPlan(_)), "{error:?}");
}

// Packed weight assembly (seam S5).

const PACKED_SHARD: &str = "model-00001-of-00001.safetensors";
const GATE0: &str = "language_model.model.layers.0.mlp.gate";
const GATE1: &str = "language_model.model.layers.1.mlp.gate";
const SHARED1: &str = "language_model.model.layers.1.mlp.shared_expert_gate";
const PACKED_CONFIG: &str = r#"{"quantization":{"mode":"affine","bits":4,"group_size":64,
 "language_model.model.layers.0.mlp.gate":{"bits":8,"group_size":64},
 "language_model.model.layers.1.mlp.gate":{"bits":8,"group_size":64},
 "language_model.model.layers.1.mlp.shared_expert_gate":{"bits":8,"group_size":64}}}"#;

fn packed_specs() -> Vec<Spec> {
    let mut specs = triplet(EMBED, 2, 8);
    specs.extend(triplet(GATE0, 2, 16));
    specs.extend(triplet(GATE1, 2, 16).into_iter().take(2));
    specs.push(spec(&format!("{SHARED1}.weight"), Dtype::U32, &[2, 16]));
    specs.push(spec(&format!("{SHARED1}.scales"), Dtype::Bf16, &[2, 2]));
    specs.push(spec(&format!("{SHARED1}.biases"), Dtype::Bf16, &[2, 2]));
    specs.push(spec(NORM, Dtype::Bf16, &[4]));
    specs
}

fn packed_weights() -> (TieredWeights, WeightPlan) {
    let specs = packed_specs();
    let file = build(&specs);
    let index = index_for(&[(PACKED_SHARD, &specs)]);
    let headers = [ShardHeader::read(
        &mut Cursor::new(file.as_slice()),
        file.len() as u64,
        1 << 20,
    )
    .unwrap()];
    let quant = QuantizationMap::from_json(PACKED_CONFIG.as_bytes()).unwrap();
    let plan = WeightPlan::build(&SPEC, &quant, &headers).unwrap();
    let readers: Readers = BTreeMap::from([(
        PACKED_SHARD.to_string(),
        Box::new(Cursor::new(file)) as Box<dyn ShardIo>,
    )]);
    (
        TieredWeights::from_readers(&index, &plan, readers, limits()).unwrap(),
        plan,
    )
}

#[test]
fn an_int4_module_assembles_a_packed_weight() {
    let (weights, plan) = packed_weights();
    let packed = weights.packed(&plan, &format!("{EMBED}.weight")).unwrap();
    assert_eq!(packed.quantization().bits, 4);
    assert_eq!(packed.quantization().group_size, 64);
    assert_eq!(packed.logical_shape(), &[2, 64]);
    assert_eq!(packed.payload().dtype(), Dtype::U32);
    assert_eq!(packed.payload().shape(), &[2, 8]);
    assert_eq!(packed.scales().shape(), &[2, 1]);
    assert_eq!(packed.scales().dtype(), Dtype::Bf16);
    assert_eq!(packed.biases().unwrap().shape(), &[2, 1]);
}

#[test]
fn an_int8_module_takes_the_same_path() {
    let (weights, plan) = packed_weights();
    let packed = weights.packed(&plan, &format!("{GATE0}.weight")).unwrap();
    assert_eq!(packed.quantization().bits, 8);
    assert_eq!(packed.logical_shape(), &[2, 64]);
    assert_eq!(packed.payload().shape(), &[2, 16]);
    assert_eq!(packed.scales().shape(), &[2, 1]);
    assert_eq!(packed.biases().unwrap().shape(), &[2, 1]);
}

#[test]
fn a_missing_biases_tensor_fails_with_checkpoint() {
    let (weights, plan) = packed_weights();
    let error = weights
        .packed(&plan, &format!("{GATE1}.weight"))
        .unwrap_err();
    assert!(matches!(error, EngError::Checkpoint(_)), "{error:?}");
}

#[test]
fn an_unquantized_key_requested_as_packed_fails_with_checkpoint() {
    let (weights, plan) = packed_weights();
    let error = weights.packed(&plan, NORM).unwrap_err();
    let EngError::Checkpoint(message) = error else {
        panic!("{error:?}")
    };
    assert!(message.contains(NORM), "{message}");
}

#[test]
fn a_sidecar_whose_group_count_disagrees_with_the_payload_fails() {
    let (weights, plan) = packed_weights();
    let error = weights
        .packed(&plan, &format!("{SHARED1}.weight"))
        .unwrap_err();
    assert!(matches!(error, EngError::Checkpoint(_)), "{error:?}");
}

#[test]
fn unknown_and_non_weight_keys_fail_with_checkpoint() {
    let (weights, plan) = packed_weights();
    for key in [
        "language_model.model.nothing.weight",
        &format!("{EMBED}.scales"),
    ] {
        let error = weights.packed(&plan, key).unwrap_err();
        assert!(matches!(error, EngError::Checkpoint(_)), "{key}: {error:?}");
    }
}

// MXFP modules store U8 scales and no biases.

const MXFP_CONFIG: &str = r#"{"quantization":{"mode":"mxfp4","bits":4,"group_size":32}}"#;

fn mxfp_specs() -> Vec<Spec> {
    vec![
        spec(&format!("{EMBED}.weight"), Dtype::U32, &[2, 8]),
        spec(&format!("{EMBED}.scales"), Dtype::U8, &[2, 2]),
        spec(NORM, Dtype::Bf16, &[4]),
    ]
}

fn mxfp_load(config: &str) -> Result<(TieredWeights, WeightPlan), EngError> {
    mxfp_load_specs(config, mxfp_specs())
}

fn mxfp_load_specs(
    config: &str,
    specs: Vec<Spec>,
) -> Result<(TieredWeights, WeightPlan), EngError> {
    let file = build(&specs);
    let index = index_for(&[(PACKED_SHARD, &specs)]);
    let headers = [ShardHeader::read(
        &mut Cursor::new(file.as_slice()),
        file.len() as u64,
        1 << 20,
    )
    .unwrap()];
    let quant = QuantizationMap::from_json(config.as_bytes()).unwrap();
    let plan = WeightPlan::build(&SPEC, &quant, &headers).unwrap();
    let readers: Readers = BTreeMap::from([(
        PACKED_SHARD.to_string(),
        Box::new(Cursor::new(file)) as Box<dyn ShardIo>,
    )]);
    TieredWeights::from_readers(&index, &plan, readers, limits()).map(|w| (w, plan))
}

#[test]
fn an_mxfp4_module_with_u8_scales_loads() {
    let (weights, _) = mxfp_load(MXFP_CONFIG).unwrap();
    let scales = weights.tensor(&format!("{EMBED}.scales")).unwrap();
    assert_eq!(scales.dtype(), Dtype::U8);
}

#[test]
fn u8_scales_in_an_affine_module_fail_with_checkpoint() {
    let error = mxfp_load(CONFIG).unwrap_err();
    let EngError::Checkpoint(message) = error else {
        panic!("{error:?}")
    };
    assert!(message.contains(&format!("{EMBED}.scales")), "{message}");
}

#[test]
fn an_mxfp4_module_with_a_biases_tensor_fails_and_says_biases_are_forbidden() {
    let mut specs = mxfp_specs();
    specs.push(spec(&format!("{EMBED}.biases"), Dtype::U8, &[2, 2]));
    let error = mxfp_load_specs(MXFP_CONFIG, specs).unwrap_err();
    let EngError::Checkpoint(message) = error else {
        panic!("{error:?}")
    };
    assert!(message.contains(&format!("{EMBED}.biases")), "{message}");
    assert!(message.contains("no biases"), "{message}");
}

#[test]
fn an_mxfp4_module_assembles_a_packed_weight() {
    let (weights, plan) = mxfp_load(MXFP_CONFIG).unwrap();
    let packed = weights.packed(&plan, &format!("{EMBED}.weight")).unwrap();
    assert_eq!(packed.quantization().bits, 4);
    assert_eq!(packed.quantization().group_size, 32);
    assert_eq!(packed.logical_shape(), &[2, 64]);
    assert_eq!(packed.scales().dtype(), Dtype::U8);
    assert!(packed.biases().is_none());
}

// Limit validation, header error context, and non-power-of-two widths.

#[test]
fn zero_limits_fail_with_bad_request() {
    let f = fixture();
    for zeroed in [
        LoadLimits {
            resident_cap: 0,
            ..limits()
        },
        LoadLimits {
            max_input_bytes: 0,
            ..limits()
        },
        LoadLimits {
            max_header_bytes: 0,
            ..limits()
        },
    ] {
        let error = TieredWeights::from_readers(&f.index, &f.plan, plain(&f), zeroed).unwrap_err();
        assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
    }
}

#[test]
fn a_header_budget_failure_names_the_shard() {
    let f = fixture();
    let tight = LoadLimits {
        max_header_bytes: 8,
        ..limits()
    };
    let error = TieredWeights::from_readers(&f.index, &f.plan, plain(&f), tight).unwrap_err();
    let EngError::MemoryPlan(message) = error else {
        panic!("{error:?}")
    };
    assert!(message.contains(SHARD_A), "{message}");
}

#[test]
fn an_int3_module_derives_k_from_the_payload_width() {
    const CONFIG3: &str = r#"{"quantization":{"mode":"affine","bits":3,"group_size":64}}"#;
    let specs = vec![
        spec(&format!("{EMBED}.weight"), Dtype::U32, &[2, 6]),
        spec(&format!("{EMBED}.scales"), Dtype::Bf16, &[2, 1]),
        spec(&format!("{EMBED}.biases"), Dtype::Bf16, &[2, 1]),
        spec(NORM, Dtype::Bf16, &[4]),
    ];
    let file = build(&specs);
    let index = index_for(&[(PACKED_SHARD, &specs)]);
    let headers = [ShardHeader::read(
        &mut Cursor::new(file.as_slice()),
        file.len() as u64,
        1 << 20,
    )
    .unwrap()];
    let quant = QuantizationMap::from_json(CONFIG3.as_bytes()).unwrap();
    let plan = WeightPlan::build(&SPEC, &quant, &headers).unwrap();
    let readers: Readers = BTreeMap::from([(
        PACKED_SHARD.to_string(),
        Box::new(Cursor::new(file)) as Box<dyn ShardIo>,
    )]);
    let weights = TieredWeights::from_readers(&index, &plan, readers, limits()).unwrap();
    let packed = weights.packed(&plan, &format!("{EMBED}.weight")).unwrap();
    assert_eq!(packed.quantization().bits, 3);
    assert_eq!(packed.logical_shape(), &[2, 64]);
}

#[test]
fn a_header_key_that_the_plan_omits_fails_with_the_shard_and_key() {
    let key = "language_model.model.unplanned.weight";
    let (sa, mut sb) = (shard_a(), shard_b());
    let (a, plain_b) = (build(&sa), build(&sb));
    let plan = plan_of(&[&a, &plain_b]);
    sb.push(spec(key, Dtype::Bf16, &[4]));
    let b = build(&sb);
    let index = index_for(&[(SHARD_A, &sa), (SHARD_B, &sb)]);
    let f = Fixture { a, b, index, plan };
    let error = TieredWeights::from_readers(&f.index, &f.plan, plain(&f), limits()).unwrap_err();
    let EngError::Checkpoint(message) = error else {
        panic!("{error:?}")
    };
    assert_eq!(
        message,
        format!("shard {SHARD_B:?}: tensor {key:?} is not in the plan")
    );
}

/// Fails every seek, as a broken device would.
struct SeekFails;

impl Read for SeekFails {
    fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
        Ok(0)
    }
}

impl Seek for SeekFails {
    fn seek(&mut self, _pos: SeekFrom) -> std::io::Result<u64> {
        Err(std::io::Error::other("device gone"))
    }
}

#[test]
fn a_shard_whose_seek_fails_is_rejected_with_the_shard_name() {
    let f = fixture();
    let mut readers = plain(&f);
    readers.insert(SHARD_B.to_string(), Box::new(SeekFails) as Box<dyn ShardIo>);
    let error = TieredWeights::from_readers(&f.index, &f.plan, readers, limits()).unwrap_err();
    let EngError::Checkpoint(message) = error else {
        panic!("{error:?}")
    };
    assert_eq!(
        message,
        format!("shard {SHARD_B:?}: seek failed: device gone")
    );
}

/// Seeks to the end, then fails the seek back to the start.
struct SeekToStartFails;

impl Read for SeekToStartFails {
    fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
        Ok(0)
    }
}

impl Seek for SeekToStartFails {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match pos {
            SeekFrom::Start(_) => Err(std::io::Error::other("device gone")),
            _ => Ok(64),
        }
    }
}

#[test]
fn a_shard_whose_seek_to_the_start_fails_is_rejected_with_the_shard_name() {
    let f = fixture();
    let mut readers = plain(&f);
    readers.insert(
        SHARD_B.to_string(),
        Box::new(SeekToStartFails) as Box<dyn ShardIo>,
    );
    let error = TieredWeights::from_readers(&f.index, &f.plan, readers, limits()).unwrap_err();
    let EngError::Checkpoint(message) = error else {
        panic!("{error:?}")
    };
    assert_eq!(
        message,
        format!("shard {SHARD_B:?}: seek failed: device gone")
    );
}

/// Fails only the seek to the end, which reads the shard length.
struct SeekToEndFails;

impl Read for SeekToEndFails {
    fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
        Ok(0)
    }
}

impl Seek for SeekToEndFails {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match pos {
            SeekFrom::End(_) => Err(std::io::Error::other("length unavailable")),
            _ => Ok(0),
        }
    }
}

#[test]
fn a_shard_whose_seek_to_the_end_fails_is_rejected_with_the_shard_name() {
    let f = fixture();
    let mut readers = plain(&f);
    readers.insert(
        SHARD_B.to_string(),
        Box::new(SeekToEndFails) as Box<dyn ShardIo>,
    );
    let error = TieredWeights::from_readers(&f.index, &f.plan, readers, limits()).unwrap_err();
    let EngError::Checkpoint(message) = error else {
        panic!("{error:?}")
    };
    assert_eq!(
        message,
        format!("shard {SHARD_B:?}: seek failed: length unavailable")
    );
}

#[test]
fn a_shard_with_a_malformed_header_is_rejected_with_the_shard_name() {
    let f = fixture();
    let mut readers = plain(&f);
    let mut broken = f.b.clone();
    // Byte index 8 is the opening `{` of the JSON header, after the 8-byte length prefix.
    broken[8] = b'!';
    readers.insert(
        SHARD_B.to_string(),
        Box::new(Cursor::new(broken)) as Box<dyn ShardIo>,
    );
    let error = TieredWeights::from_readers(&f.index, &f.plan, readers, limits()).unwrap_err();
    let EngError::Checkpoint(message) = error else {
        panic!("{error:?}")
    };
    let prefix = format!("shard {SHARD_B:?}: shard header is not valid JSON: ");
    assert!(message.starts_with(&prefix), "{message}");
    assert!(message.len() > prefix.len(), "{message}");
}
