use std::{fs::File, io::Read, path::Path};

use sha2::{Digest, Sha256};

pub(super) fn read_manifest(root: &Path, expected: &str) -> Result<Vec<u8>, String> {
    if expected.len() != 64
        || !expected
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err("manifest-sha256 must contain 64 lowercase hexadecimal digits".into());
    }
    let path = root.join("manifest.json");
    require_regular(&path)?;
    let mut bytes = Vec::new();
    File::open(&path)
        .map_err(|e| format!("manifest: {e}"))?
        .take(4 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("manifest: {e}"))?;
    if bytes.len() > 4 * 1024 * 1024 {
        return Err("manifest exceeds 4 MiB".into());
    }
    if format!("{:x}", Sha256::digest(&bytes)) != expected {
        return Err("manifest digest mismatch".into());
    }
    Ok(bytes)
}

use inference_engine::{
    family::quantization::{ModuleQuantization, QuantizationMap},
    kernels::{PackedWeight, QuantizationMode, QuantizationSpec},
    tensor::{Dtype, Tensor},
    weights::{ResidentWeights, WeightSource},
};
use serde::Deserialize;
use std::{collections::BTreeMap, sync::Arc};

#[derive(Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Asset {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Group {
    pub module: String,
    pub coordinates: Vec<u64>,
    pub payload_shape: Vec<u64>,
    pub weight: Asset,
    pub reference: Asset,
    pub shards: Vec<String>,
    pub negative_bias: bool,
}

#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Manifest {
    pub version: u32,
    pub fixture: bool,
    pub group_count: usize,
    pub negative_biases: usize,
    pub eligible_modules: Vec<String>,
    pub excluded_int8: Vec<String>,
    pub reference_pin: String,
    pub source_files: Vec<Asset>,
    pub config: Asset,
    pub activations: Asset,
    pub groups: Vec<Group>,
    pub versions: BTreeMap<String, String>,
    pub hardware: String,
}

#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]
pub(super) struct Case {
    pub module: String,
    pub coordinates: Vec<u64>,
    pub weight: PackedWeight,
    pub expected: [Tensor; 2],
}

#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]
pub(super) struct Suite {
    pub manifest: Manifest,
    pub activation: [Tensor; 2],
    pub cases: Vec<Case>,
}

fn valid_name(name: &str) -> bool {
    Path::new(name).file_name().and_then(|n| n.to_str()) == Some(name)
        && !matches!(name, "." | ".." | "")
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn read_asset(root: &Path, asset: &Asset) -> Result<Vec<u8>, String> {
    if Path::new(&asset.path).file_name().and_then(|n| n.to_str()) != Some(&asset.path)
        || matches!(asset.path.as_str(), "." | ".." | "")
    {
        return Err("invalid artifact path".into());
    }
    let path = root.join(&asset.path);
    require_regular(&path)?;
    let mut bytes = Vec::new();
    File::open(&path)
        .map_err(|e| format!("artifact {}: {e}", asset.path))?
        .take(64 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > 64 * 1024 || bytes.len() as u64 != asset.bytes {
        return Err("artifact length exceeds limit or differs from manifest".into());
    }
    if format!("{:x}", Sha256::digest(&bytes)) != asset.sha256 {
        return Err("artifact digest mismatch".into());
    }
    Ok(bytes)
}

fn source(root: &Path, asset: &Asset) -> Result<ResidentWeights, String> {
    ResidentWeights::from_bytes(Arc::from(read_asset(root, asset)?))
        .map_err(|e| format!("prepared safetensors: {e:?}"))
}

fn tensor(source: &ResidentWeights, name: &str) -> Result<Tensor, String> {
    source
        .tensor(name)
        .ok_or_else(|| format!("missing tensor {name}"))
}

pub(super) fn values(tensor: &Tensor, shape: &[u32]) -> Result<Vec<f32>, String> {
    if tensor.dtype() != Dtype::F32 || tensor.shape() != shape {
        return Err("F32 tensor dtype or shape mismatch".into());
    }
    let bytes = tensor.host_bytes().ok_or("tensor has no host bytes")?;
    let values: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("nonfinite tensor value".into());
    }
    Ok(values)
}

impl Suite {
    pub fn load(root: &Path, digest: &str, fixture: bool) -> Result<Self, String> {
        let manifest: Manifest = serde_json::from_slice(&read_manifest(root, digest)?)
            .map_err(|e| format!("manifest JSON: {e}"))?;
        if fixture != manifest.fixture {
            return Err("fixture mode does not match manifest".into());
        }
        manifest.check_provenance()?;
        if manifest.group_count == 0
            || manifest.group_count > 1000
            || manifest.group_count != manifest.groups.len()
        {
            return Err("invalid or incomplete group count".into());
        }
        let mut identities = std::collections::HashSet::new();
        for group in &manifest.groups {
            if !identities.insert((&group.module, &group.coordinates)) {
                return Err("duplicate group identity".into());
            }
        }
        manifest.check_selection()?;
        let mut paths = std::collections::HashSet::new();
        let mut total = 0_u64;
        for asset in [&manifest.config, &manifest.activations].into_iter().chain(
            manifest
                .groups
                .iter()
                .flat_map(|g| [&g.weight, &g.reference]),
        ) {
            if !valid_name(&asset.path)
                || !valid_digest(&asset.sha256)
                || asset.bytes > 64 * 1024
                || !paths.insert(&asset.path)
            {
                return Err("invalid artifact inventory".into());
            }
            total = total
                .checked_add(asset.bytes)
                .ok_or("artifact inventory size overflow")?;
        }
        if total > 32 * 1024 * 1024 {
            return Err("artifact inventory exceeds 32 MiB".into());
        }
        let quantization = QuantizationMap::from_json(&read_asset(root, &manifest.config)?)
            .map_err(|e| format!("quantization metadata: {e:?}"))?;
        let activation_source = source(root, &manifest.activations)?;
        let activation = [
            tensor(&activation_source, "basis")?,
            tensor(&activation_source, "dense")?,
        ];
        let basis = values(&activation[0], &[64, 64])?;
        let dense = values(&activation[1], &[1, 64])?;
        if basis
            .iter()
            .enumerate()
            .any(|(i, &v)| v != f32::from(i / 64 == i % 64))
            || dense
                .iter()
                .zip([1., -1., 0.5, -0.25, 2., -2., 0.25, 0.125].iter().cycle())
                .any(|(a, b)| a != b)
        {
            return Err("activation probe values differ from contract".into());
        }
        let mut cases = Vec::new();
        let mut negative_biases = 0;
        for group in &manifest.groups {
            let ModuleQuantization::Quantized(spec) = quantization
                .resolve(&group.module)
                .map_err(|e| format!("module: {e:?}"))?
            else {
                return Err("selected module is unquantized".into());
            };
            if spec.mode != QuantizationMode::Affine || spec.bits != 4 || spec.group_size != 64 {
                return Err("selected module is not affine int4 group 64".into());
            }
            let weights = source(root, &group.weight)?;
            let reference = source(root, &group.reference)?;
            let expected = [tensor(&reference, "basis")?, tensor(&reference, "dense")?];
            values(&expected[0], &[64, 1])?;
            values(&expected[1], &[1, 1])?;
            let weight = PackedWeight::try_new(
                tensor(&weights, "weight")?,
                tensor(&weights, "scales")?,
                Some(tensor(&weights, "biases")?),
                vec![1, 64],
                QuantizationSpec {
                    mode: QuantizationMode::Affine,
                    bits: 4,
                    group_size: 64,
                },
            )
            .map_err(|e| format!("packed group: {e:?}"))?;
            finite_bf16(weight.scales())?;
            let biases = weight
                .biases()
                .ok_or_else(|| format!("{}: affine group has no biases", group.module))?;
            let negative = finite_bf16(biases)? < 0.0;
            if negative != group.negative_bias {
                return Err("negative bias metadata differs from storage".into());
            }
            negative_biases += usize::from(negative);
            cases.push(Case {
                module: group.module.clone(),
                coordinates: group.coordinates.clone(),
                weight,
                expected,
            });
        }
        if negative_biases == 0 || negative_biases != manifest.negative_biases {
            return Err("negative bias coverage is missing or inconsistent".into());
        }
        Ok(Self {
            manifest,
            activation,
            cases,
        })
    }
}

impl Manifest {
    fn check_provenance(&self) -> Result<(), String> {
        let invalid = || "invalid qualification provenance".to_string();
        let reference = include_bytes!("../reference-pin.json");
        let pin: serde_json::Value = serde_json::from_slice(reference).map_err(|_| invalid())?;
        let versions = BTreeMap::from([
            (
                "mlx".into(),
                pin["mlx_version"].as_str().ok_or_else(invalid)?.into(),
            ),
            ("numpy".into(), "2.5.3".into()),
            ("safetensors".into(), "0.8.0".into()),
            ("ml_dtypes".into(), "0.5.3".into()),
        ]);
        if self.version != 1
            || self.reference_pin != format!("{:x}", Sha256::digest(reference))
            || self.versions != versions
            || self.hardware.is_empty()
        {
            return Err(invalid());
        }
        let mut sources = BTreeMap::new();
        for asset in &self.source_files {
            if !valid_name(&asset.path)
                || asset.bytes == 0
                || asset.bytes > 6 * 1024 * 1024 * 1024
                || !valid_digest(&asset.sha256)
                || sources.insert(&asset.path, asset).is_some()
            {
                return Err(invalid());
            }
        }
        if sources.get(&"config.json".to_string()).copied() != Some(&self.config)
            || !sources.contains_key(&"model.safetensors.index.json".to_string())
        {
            return Err(invalid());
        }
        let mut excluded = std::collections::BTreeSet::new();
        for module in &self.excluded_int8 {
            if !excluded.insert(module) || self.eligible_modules.contains(module) {
                return Err(invalid());
            }
        }
        for group in &self.groups {
            if group.shards.len() != 3
                || group
                    .shards
                    .iter()
                    .any(|name| !name.ends_with(".safetensors") || !sources.contains_key(name))
            {
                return Err(invalid());
            }
        }
        if !self.fixture {
            let environment: serde_json::Value =
                serde_json::from_str(include_str!("../../../engine/testdata/mac-phase0a.json"))
                    .map_err(|_| invalid())?;
            let files = environment["checkpoint"]["files"]
                .as_array()
                .ok_or_else(invalid)?;
            let mut expected = BTreeMap::new();
            for item in files.iter().filter(|item| {
                item["role"] == "base_weight"
                    || item["path"] == "config.json"
                    || item["path"] == "model.safetensors.index.json"
            }) {
                let asset = Asset {
                    path: item["path"].as_str().ok_or_else(invalid)?.into(),
                    bytes: item["bytes"].as_u64().ok_or_else(invalid)?,
                    sha256: item["sha256"].as_str().ok_or_else(invalid)?.into(),
                };
                expected.insert(asset.path.clone(), asset);
            }
            if sources.len() != expected.len()
                || expected
                    .iter()
                    .any(|(name, asset)| sources.get(name).copied() != Some(asset))
                || self.group_count != 1000
                || self.eligible_modules.len() != 432
                || self.excluded_int8.len() != 80
            {
                return Err(invalid());
            }
            let expert_modules: std::collections::BTreeSet<_> = self
                .groups
                .iter()
                .filter(|g| g.payload_shape.len() == 3)
                .map(|g| &g.module)
                .collect();
            if expert_modules.len() != 120 {
                return Err(invalid());
            }
        }
        Ok(())
    }

    fn check_selection(&self) -> Result<(), String> {
        use std::collections::{BTreeSet, HashSet};
        let eligible: BTreeSet<_> = self.eligible_modules.iter().collect();
        if eligible.is_empty() || eligible.iter().copied().ne(self.eligible_modules.iter()) {
            return Err("eligible modules must be unique and sorted".into());
        }
        let mut shapes = BTreeMap::new();
        for group in &self.groups {
            let shape = &group.payload_shape;
            if !(2..=3).contains(&shape.len())
                || shape.contains(&0)
                || shape.last().unwrap() % 8 != 0
            {
                return Err("invalid original payload shape".into());
            }
            if !eligible.contains(&group.module) {
                return Err("group module is absent from eligible inventory".into());
            }
            if let Some(previous) = shapes.insert(&group.module, shape.clone())
                && previous != *shape
            {
                return Err("inconsistent original payload shapes".into());
            }
        }
        if shapes.len() != eligible.len() {
            return Err("selected groups do not cover eligible inventory".into());
        }
        let mut position = 0;
        let mut ordinals = BTreeMap::new();
        let mut selected = BTreeMap::new();
        while position < self.groups.len() {
            let before = position;
            for (name, shape) in &shapes {
                let mut dimensions = shape.clone();
                *dimensions.last_mut().unwrap() /= 8;
                let count = dimensions
                    .iter()
                    .try_fold(1_u64, |n, d| n.checked_mul(*d))
                    .ok_or("group count overflow")?;
                let used = selected.entry(*name).or_insert_with(HashSet::new);
                if used.len() as u64 == count {
                    continue;
                }
                let ordinal = ordinals.entry(*name).or_insert(0_u64);
                let mut chosen = None;
                for _ in 0..1024 {
                    let digest = Sha256::digest(format!("{name}\0{ordinal}"));
                    *ordinal += 1;
                    let candidate = u64::from_le_bytes(digest[..8].try_into().unwrap()) % count;
                    if used.insert(candidate) {
                        chosen = Some(candidate);
                        break;
                    }
                }
                let mut index = chosen.ok_or("selection candidate limit exceeded")?;
                let mut coordinates = vec![0; dimensions.len()];
                for (coordinate, dimension) in coordinates.iter_mut().zip(dimensions).rev() {
                    *coordinate = index % dimension;
                    index /= dimension;
                }
                let group = &self.groups[position];
                if group.module != **name || group.coordinates != coordinates {
                    return Err("group differs from deterministic selection".into());
                }
                position += 1;
                if position == self.groups.len() {
                    break;
                }
            }
            if position == before {
                return Err("insufficient distinct groups".into());
            }
        }
        Ok(())
    }
}

fn finite_bf16(tensor: &Tensor) -> Result<f32, String> {
    if tensor.dtype() != Dtype::Bf16 || tensor.shape() != [1, 1] {
        return Err("selected sidecar must be BF16 [1, 1]".into());
    }
    let bytes = tensor.host_bytes().ok_or("sidecar has no host bytes")?;
    let value = half::bf16::from_bits(u16::from_le_bytes(
        bytes.try_into().map_err(|_| "invalid BF16 byte count")?,
    ))
    .to_f32();
    if !value.is_finite() {
        return Err("nonfinite selected sidecar".into());
    }
    Ok(value)
}

fn require_regular(path: &Path) -> Result<(), String> {
    let metadata = path
        .symlink_metadata()
        .map_err(|e| format!("artifact metadata: {e}"))?;
    if !metadata.is_file() {
        return Err("artifact must be a regular file, without a symlink".into());
    }
    Ok(())
}
