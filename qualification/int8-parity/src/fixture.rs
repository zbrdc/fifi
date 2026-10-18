//! Portable fixture loading and integrity checks (case 3 of decision 0031).
//! Nothing in this module needs the Mac GPU.

use std::{collections::BTreeMap, io::Cursor, path::Path};

use inference_engine::{
    family::{QWEN3_6_35B_A3B, WeightPlan, quantization::QuantizationMap},
    kernels::PackedWeight,
    weights::{LoadLimits, ShardHeader, ShardIo, TieredWeights},
};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

pub const SCHEMA: &str = "int8-reference/1";
pub const HIDDEN: u32 = 2048;
const MAX_ASSET: u64 = 64 << 20;
const REFERENCE_PIN: &[u8] = include_bytes!("../reference-pin.json");
const RECORD: &[u8] = include_bytes!("../../../engine/testdata/mac-phase0a.json");
const KINDS: [&str; 2] = ["mlp.gate", "mlp.shared_expert_gate"];
const SHARD: &str = "int8-fixture-00001-of-00001.safetensors";

#[derive(Debug, Clone, Deserialize)]
pub struct Asset {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Assets {
    pub config: Asset,
    pub index: Asset,
    pub shard: Asset,
    pub cases: Asset,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct SourceFile {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct CaseRecord {
    pub id: String,
    pub module: String,
    pub m: u32,
    pub seed: u64,
    pub signed_zero: bool,
    pub expected_kernel: String,
    pub bits: u32,
    pub group_size: u32,
    pub mode: String,
    pub activation: String,
    pub reference: String,
    pub float64_max_abs_diff: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // recorded metadata that the gate verifies only by digest
pub struct Manifest {
    pub schema: String,
    pub synthetic: bool,
    pub layer_count: u32,
    pub reference_pin: String,
    pub versions: Value,
    pub recorded_binaries: Value,
    pub build_flags: Value,
    pub env: Value,
    pub device: Value,
    pub source_files: Vec<SourceFile>,
    pub sources: Value,
    pub assets: Assets,
    pub shard_tensors: Vec<TensorEntry>,
    pub case_tensors: Vec<TensorEntry>,
    pub cases: Vec<CaseRecord>,
    pub float64_worst_relative_diff: f64,
}

pub fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn module_name(layer: u32, kind: &str) -> String {
    format!("language_model.model.layers.{layer}.{kind}")
}

fn out_rows(module: &str) -> u64 {
    if module.ends_with("mlp.gate") { 256 } else { 1 }
}

fn kernel(module: &str, m: u32) -> &'static str {
    match (module.ends_with("mlp.gate"), m) {
        (true, 1) => "affine_qmv_fast",
        (false, 1) => "affine_qmv",
        (_, 5) => "affine_qmv_wide",
        _ => "affine_qmm_splitk",
    }
}

/// The case list of decision 0031 for a given layer count.
pub fn expected_plan(layers: u32) -> Vec<CaseRecord> {
    let mut plan: Vec<(String, u32, bool)> = Vec::new();
    for layer in 0..layers {
        for kind in KINDS {
            plan.push((module_name(layer, kind), 1, false));
        }
    }
    for m in [5, 16] {
        for layer in [0, layers - 1] {
            for kind in KINDS {
                plan.push((module_name(layer, kind), m, false));
            }
        }
    }
    for kind in KINDS {
        plan.push((module_name(0, kind), 1, true));
    }
    plan.into_iter()
        .enumerate()
        .map(|(i, (module, m, signed_zero))| CaseRecord {
            id: format!("c{i:03}"),
            seed: 31000 + i as u64,
            expected_kernel: kernel(&module, m).into(),
            module,
            m,
            signed_zero,
            bits: 8,
            group_size: 64,
            mode: "affine".into(),
            activation: format!("act.c{i:03}"),
            reference: format!("ref.c{i:03}"),
            float64_max_abs_diff: 0.0,
        })
        .collect()
}

pub fn modules(layers: u32) -> Vec<String> {
    (0..layers)
        .flat_map(|l| KINDS.map(|k| module_name(l, k)))
        .collect()
}

struct Header {
    tensors: BTreeMap<String, (String, Vec<u64>, usize, usize)>,
}

fn parse_header(bytes: &[u8]) -> Result<Header, String> {
    let prefix: [u8; 8] = bytes
        .get(..8)
        .ok_or("short safetensors")?
        .try_into()
        .unwrap();
    let length = usize::try_from(u64::from_le_bytes(prefix)).map_err(|e| e.to_string())?;
    let end = 8usize.checked_add(length).ok_or("header overflow")?;
    let json: Value = serde_json::from_slice(bytes.get(8..end).ok_or("short header")?)
        .map_err(|e| e.to_string())?;
    let mut tensors = BTreeMap::new();
    for (name, entry) in json.as_object().ok_or("header is not an object")? {
        if name == "__metadata__" {
            continue;
        }
        let dtype = entry["dtype"].as_str().ok_or("dtype")?.to_owned();
        let shape = entry["shape"]
            .as_array()
            .ok_or("shape")?
            .iter()
            .map(|v| v.as_u64().ok_or("dim"))
            .collect::<Result<Vec<_>, _>>()?;
        let offsets = entry["data_offsets"].as_array().ok_or("offsets")?;
        let a = offsets.first().and_then(Value::as_u64).ok_or("offset")? as usize;
        let b = offsets.get(1).and_then(Value::as_u64).ok_or("offset")? as usize;
        let (start, stop) = (end.checked_add(a), end.checked_add(b));
        let (Some(start), Some(stop)) = (start, stop) else {
            return Err(format!("tensor {name} offset overflows"));
        };
        if a > b || stop > bytes.len() {
            return Err(format!("tensor {name} is out of range"));
        }
        tensors.insert(name.clone(), (dtype, shape, start, stop));
    }
    Ok(Header { tensors })
}

fn check_inventory(label: &str, bytes: &[u8], inventory: &[TensorEntry]) -> Result<(), String> {
    let header = parse_header(bytes)?;
    if header.tensors.len() != inventory.len() {
        return Err(format!("{label} tensor count differs from manifest"));
    }
    for entry in inventory {
        let (dtype, shape, a, b) = header
            .tensors
            .get(&entry.name)
            .ok_or_else(|| format!("{label} lacks {}", entry.name))?;
        if *dtype != entry.dtype || *shape != entry.shape || hex(&bytes[*a..*b]) != entry.sha256 {
            return Err(format!(
                "{label} tensor {} differs from manifest",
                entry.name
            ));
        }
    }
    Ok(())
}

pub fn tensor_bytes<'a>(bytes: &'a [u8], name: &str) -> Result<&'a [u8], String> {
    let header = parse_header(bytes)?;
    let (_, _, a, b) = header
        .tensors
        .get(name)
        .ok_or_else(|| format!("no tensor {name}"))?;
    Ok(&bytes[*a..*b])
}

pub struct Suite {
    pub manifest: Manifest,
    pub config: Vec<u8>,
    pub index: Vec<u8>,
    pub shard: Vec<u8>,
    pub cases: Vec<u8>,
}

fn read_asset(dir: &Path, asset: &Asset) -> Result<Vec<u8>, String> {
    if asset.path.contains(['/', '\\']) || asset.path.starts_with('.') {
        return Err("invalid asset name".into());
    }
    let path = dir.join(&asset.path);
    let meta = std::fs::symlink_metadata(&path).map_err(|e| format!("{}: {e}", asset.path))?;
    if !meta.is_file() || meta.len() != asset.bytes || meta.len() > MAX_ASSET {
        return Err(format!("asset {} has an invalid length", asset.path));
    }
    let data = std::fs::read(&path).map_err(|e| e.to_string())?;
    if hex(&data) != asset.sha256 {
        return Err(format!("asset {} digest mismatch", asset.path));
    }
    Ok(data)
}

fn pinned_sources() -> Result<Vec<SourceFile>, String> {
    let record: Value = serde_json::from_slice(RECORD).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for item in record["checkpoint"]["files"]
        .as_array()
        .ok_or("record files")?
    {
        let path = item["path"].as_str().unwrap_or_default();
        if item["role"] == "base_weight"
            || path == "config.json"
            || path == "model.safetensors.index.json"
        {
            out.push(SourceFile {
                path: path.into(),
                bytes: item["bytes"].as_u64().ok_or("bytes")?,
                sha256: item["sha256"].as_str().ok_or("sha")?.into(),
            });
        }
    }
    Ok(out)
}

impl Suite {
    /// Case 3: verifies every digest, inventory, and the case list.
    pub fn load(dir: &Path, manifest_sha256: &str) -> Result<Self, String> {
        let valid = manifest_sha256.len() == 64
            && manifest_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        if !valid {
            return Err("manifest digest must be 64 lowercase hex characters".into());
        }
        let path = dir.join("manifest.json");
        let meta = std::fs::symlink_metadata(&path).map_err(|e| format!("manifest.json: {e}"))?;
        if !meta.is_file() || meta.len() > 4 << 20 {
            return Err("manifest.json is not a regular file of at most 4 MiB".into());
        }
        let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
        if hex(&bytes) != manifest_sha256 {
            return Err("manifest digest mismatch".into());
        }
        let manifest: Manifest =
            serde_json::from_slice(&bytes).map_err(|e| format!("manifest: {e}"))?;
        if manifest.schema != SCHEMA {
            return Err("unsupported manifest schema".into());
        }
        if manifest.layer_count < 2 || manifest.layer_count > 40 {
            return Err("invalid layer count".into());
        }
        if !manifest.synthetic {
            if manifest.layer_count != 40 {
                return Err("a real fixture needs 40 layers".into());
            }
            if manifest.source_files != pinned_sources()? {
                return Err("source file pins differ from the Mac environment record".into());
            }
            if manifest.reference_pin != hex(REFERENCE_PIN) {
                return Err("reference pin differs from reference-pin.json".into());
            }
        }
        let a = &manifest.assets;
        let expected_names = [
            (&a.config.path, "config.json"),
            (&a.index.path, "model.safetensors.index.json"),
            (&a.shard.path, SHARD),
            (&a.cases.path, "cases.safetensors"),
        ];
        if expected_names.iter().any(|(got, want)| got != want) {
            return Err("unexpected asset names".into());
        }
        let config = read_asset(dir, &a.config)?;
        let index = read_asset(dir, &a.index)?;
        let shard = read_asset(dir, &a.shard)?;
        let cases = read_asset(dir, &a.cases)?;
        check_inventory("shard", &shard, &manifest.shard_tensors)?;
        check_inventory("cases", &cases, &manifest.case_tensors)?;
        let suite = Self {
            manifest,
            config,
            index,
            shard,
            cases,
        };
        suite.check_structure()?;
        Ok(suite)
    }

    fn check_structure(&self) -> Result<(), String> {
        let m = &self.manifest;
        let layers = m.layer_count;
        let names: Vec<&str> = m.shard_tensors.iter().map(|t| t.name.as_str()).collect();
        let mut want = Vec::new();
        for module in modules(layers) {
            let router = module.ends_with("mlp.gate");
            let (rows, dtype_shapes) = if router {
                (256u64, ["weight", "scales", "biases"])
            } else {
                (1, ["weight", "scales", "biases"])
            };
            for role in dtype_shapes {
                let entry = m
                    .shard_tensors
                    .iter()
                    .find(|t| t.name == format!("{module}.{role}"))
                    .ok_or_else(|| format!("missing {module}.{role}"))?;
                let (dtype, width) = if role == "weight" {
                    ("U32", HIDDEN as u64 / 4)
                } else {
                    ("BF16", HIDDEN as u64 / 64)
                };
                if entry.dtype != dtype || entry.shape != [rows, width] {
                    return Err(format!("unexpected layout for {}", entry.name));
                }
                want.push(entry.name.as_str());
            }
        }
        if names.len() != want.len() {
            return Err("shard holds unexpected tensors".into());
        }
        let mut plan = expected_plan(layers);
        if m.cases.len() != plan.len() {
            return Err("case list length differs from decision 0031".into());
        }
        for (case, expected) in m.cases.iter().zip(plan.iter_mut()) {
            expected.float64_max_abs_diff = case.float64_max_abs_diff;
            if case != expected {
                return Err(format!("case {} differs from decision 0031", case.id));
            }
            let rows = out_rows(&case.module);
            for (name, shape) in [
                (&case.activation, [case.m as u64, HIDDEN as u64]),
                (&case.reference, [case.m as u64, rows]),
            ] {
                let entry = m
                    .case_tensors
                    .iter()
                    .find(|t| &t.name == name)
                    .ok_or("missing case tensor")?;
                if entry.dtype != "F32" || entry.shape != shape {
                    return Err(format!("unexpected layout for {name}"));
                }
            }
            if case.signed_zero {
                let act = tensor_bytes(&self.cases, &case.activation)?;
                let word = |i: usize| u32::from_le_bytes(act[i * 4..i * 4 + 4].try_into().unwrap());
                for j in 0..(HIDDEN as usize / 4) {
                    if word(4 * j) != 0 || word(4 * j + 1) != 0x8000_0000 {
                        return Err(format!("signed zero pattern missing in {}", case.id));
                    }
                }
            }
        }
        if m.case_tensors.len() != 2 * m.cases.len() {
            return Err("case tensor count differs".into());
        }
        Ok(())
    }

    /// Case 1 loading path: real engine loader with the derived config.
    pub fn packed_weights(
        &self,
        shard_override: Option<&[u8]>,
    ) -> Result<BTreeMap<String, PackedWeight>, String> {
        let shard = shard_override.unwrap_or(&self.shard);
        let quant = QuantizationMap::from_json(&self.config).map_err(|e| format!("{e:?}"))?;
        let header = ShardHeader::read(&mut Cursor::new(shard), shard.len() as u64, 1 << 20)
            .map_err(|e| format!("{e:?}"))?;
        let plan =
            WeightPlan::build(&QWEN3_6_35B_A3B, &quant, &[header]).map_err(|e| format!("{e:?}"))?;
        let mut readers: BTreeMap<String, Box<dyn ShardIo>> = BTreeMap::new();
        readers.insert(SHARD.into(), Box::new(Cursor::new(shard.to_vec())));
        let cap = 4 * shard.len() as u64 + (1 << 20);
        let limits = LoadLimits {
            resident_cap: cap,
            max_input_bytes: cap,
            max_header_bytes: 1 << 20,
        };
        let weights = TieredWeights::from_readers(&self.index, &plan, readers, limits)
            .map_err(|e| format!("{e:?}"))?;
        let mut out = BTreeMap::new();
        for module in modules(self.manifest.layer_count) {
            let packed = weights
                .packed(&plan, &format!("{module}.weight"))
                .map_err(|e| format!("{module}: {e:?}"))?;
            out.insert(module, packed);
        }
        Ok(out)
    }
}
