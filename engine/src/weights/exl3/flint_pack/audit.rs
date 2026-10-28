//! The pack audit: schema names, dtypes and shapes, shard extents, payload scans and samples.
//!
//! The pack is opened with the house `Checkpoint::open`, which already bounds the config, the
//! index and every shard header and compares each header's data section with the file length.
//! The audit adds the Flint schema on top and never reads trellis payloads except for samples.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::Path,
};

use super::{
    Geometry, RoutedScale,
    scan::{admit_samples, scan_small},
    schema::{Kind, Plan, Spec},
};
use crate::{
    EngError,
    weights::{
        StreamedCheckpoint,
        exl3::{Codebook, Config, source::Checkpoint},
        indexed::parse_weight_map,
    },
};

/// Same bound as `Checkpoint::open` applies to the index it reads first.
const INDEX_LIMIT: u64 = 16 * 1024 * 1024;

/// Byte budgets of the payload passes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Limits {
    /// Largest batch of small tensors read at once.
    pub(crate) batch_bytes: u64,
    /// Largest payload of one sample matrix.
    pub(crate) matrix_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            batch_bytes: 64 << 20,
            matrix_bytes: 8 << 20,
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub(super) struct ScaleCounts {
    pub(super) vectors: usize,
    pub(super) elements: u64,
    pub(super) zeros: u64,
    pub(super) subnormals: u64,
    pub(super) zero_vectors: usize,
    pub(super) subnormal_vectors: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ShardExtent {
    pub(super) name: String,
    pub(super) file_len: u64,
    pub(super) header_len: u64,
    pub(super) tensors: usize,
    pub(super) payload: u64,
}

/// A sample matrix as the real loader admitted it.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Admitted {
    pub(super) prefix: String,
    /// `[input, output]` widths.
    pub(super) shape: [usize; 2],
    pub(super) bits_twice: u8,
    pub(super) codebook: Codebook,
}

#[derive(Debug, Default)]
pub(crate) struct Audit {
    pub(super) tensors: usize,
    /// Matrices per role, such as `experts.up_proj`.
    pub(super) matrices: BTreeMap<&'static str, usize>,
    /// Observed trellis rate (twice the bits per weight) and how many matrices have it.
    pub(super) bits_twice: BTreeMap<u8, usize>,
    pub(super) shards: Vec<ShardExtent>,
    /// Zero and subnormal counts per scale role (`experts.up_proj.suh`) and plain role.
    pub(super) scales: BTreeMap<String, ScaleCounts>,
    /// Scale vectors that are entirely zero: every output (`svh`) or input (`suh`) channel
    /// they scale is exactly zero.
    pub(super) dead_vectors: Vec<String>,
    /// Sample matrices admitted through `Matrix::load`.
    pub(super) admitted: Vec<Admitted>,
}

/// Audits `dir` against the pinned published Flint geometry.
pub(super) fn audit_published(dir: &Path) -> Result<Audit, EngError> {
    let checkpoint = Checkpoint::open(dir)?;
    let geometry = Geometry::published(&checkpoint.model_config)?;
    audit(dir, &checkpoint, &geometry, &Limits::default())
}

pub(crate) fn audit(
    dir: &Path,
    checkpoint: &Checkpoint,
    geometry: &Geometry,
    limits: &Limits,
) -> Result<Audit, EngError> {
    RoutedScale::from_model_config(&checkpoint.model_config)?;
    require_mul1(&checkpoint.config)?;
    let plan = geometry.plan();
    let specs = plan.specs();
    let shard_of = read_index(dir)?;
    check_names(&specs, &shard_of)?;
    let bits_twice = check_headers(&checkpoint.source, &specs)?;
    let shards = check_extents(dir, &specs, &shard_of)?;
    let scan = scan_small(&checkpoint.source, &specs, limits)?;
    let admitted = admit_samples(
        &checkpoint.source,
        &checkpoint.config,
        &plan,
        geometry,
        limits,
    )?;
    Ok(Audit {
        tensors: specs.len(),
        matrices: matrices_per_role(&plan),
        bits_twice,
        shards,
        scales: scan.scales,
        dead_vectors: scan.dead_vectors,
        admitted,
    })
}

fn require_mul1(config: &Config) -> Result<(), EngError> {
    if config.codebook != Some(Codebook::Mul1) {
        return Err(invalid("config codebook must be declared as mul1"));
    }
    Ok(())
}

fn matrices_per_role(plan: &Plan) -> BTreeMap<&'static str, usize> {
    let mut roles = BTreeMap::new();
    for module in &plan.modules {
        *roles.entry(module.role).or_default() += 1;
    }
    roles
}

/// The index's tensor-to-shard map, read through the reader `Checkpoint::open` uses.
fn read_index(dir: &Path) -> Result<BTreeMap<String, String>, EngError> {
    let path = dir.join("model.safetensors.index.json");
    let mut bytes = Vec::new();
    File::open(&path)
        .and_then(|file| file.take(INDEX_LIMIT + 1).read_to_end(&mut bytes))
        .map_err(|error| invalid(&format!("{}: {error}", path.display())))?;
    if bytes.len() as u64 > INDEX_LIMIT {
        return Err(EngError::MemoryPlan(format!(
            "EXL3 Flint pack: index exceeds {INDEX_LIMIT} bytes"
        )));
    }
    parse_weight_map(&bytes)
}

/// Every schema name is in the index and every index name is in the schema.
fn check_names(specs: &[Spec], shard_of: &BTreeMap<String, String>) -> Result<(), EngError> {
    let expected: BTreeSet<&str> = specs.iter().map(|spec| spec.name.as_str()).collect();
    let missing: Vec<&str> = expected
        .iter()
        .filter(|name| !shard_of.contains_key(**name))
        .copied()
        .collect();
    let extra: Vec<&str> = shard_of
        .keys()
        .map(String::as_str)
        .filter(|name| !expected.contains(name))
        .collect();
    if !missing.is_empty() {
        return Err(invalid(&format!(
            "{} schema tensor(s) missing from the index, first {:?}",
            missing.len(),
            &missing[..missing.len().min(3)]
        )));
    }
    if !extra.is_empty() {
        return Err(invalid(&format!(
            "{} index tensor(s) not part of the Flint schema, first {:?}",
            extra.len(),
            &extra[..extra.len().min(3)]
        )));
    }
    Ok(())
}

/// Stored dtype and shape of every tensor, from headers only; returns the trellis rates.
fn check_headers(
    source: &StreamedCheckpoint,
    specs: &[Spec],
) -> Result<BTreeMap<u8, usize>, EngError> {
    let mut rates = BTreeMap::new();
    for spec in specs {
        let stored = source.stored(&spec.name);
        if stored.as_ref() != Some(&(spec.dtype, spec.shape.clone())) {
            return Err(invalid(&format!(
                "{}: stored {stored:?}, schema requires {:?}",
                spec.name,
                (spec.dtype, &spec.shape)
            )));
        }
        if spec.kind == Kind::Trellis {
            let words = spec.shape[2] as usize;
            *rates.entry((words / 8) as u8).or_default() += 1;
        }
    }
    Ok(rates)
}

/// Each shard's file length must equal its prefix, header and the schema's tensor bytes.
fn check_extents(
    dir: &Path,
    specs: &[Spec],
    shard_of: &BTreeMap<String, String>,
) -> Result<Vec<ShardExtent>, EngError> {
    let mut per_shard: BTreeMap<&str, (usize, u64)> = BTreeMap::new();
    for spec in specs {
        let shard = shard_of
            .get(&spec.name)
            .ok_or_else(|| invalid(&format!("{}: not in the index", spec.name)))?;
        let entry = per_shard.entry(shard).or_default();
        entry.0 += 1;
        entry.1 = entry
            .1
            .checked_add(spec.bytes())
            .ok_or_else(|| invalid(&format!("shard {shard:?}: payload bytes overflow")))?;
    }
    let mut extents = Vec::new();
    for (name, (tensors, bytes)) in per_shard {
        let path = dir.join(name);
        let mut file = File::open(&path).map_err(|e| invalid(&format!("{name}: {e}")))?;
        let file_len = file
            .metadata()
            .map_err(|e| invalid(&format!("{name}: {e}")))?
            .len();
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)
            .map_err(|e| invalid(&format!("{name}: header length: {e}")))?;
        let header_len = u64::from_le_bytes(prefix);
        let payload = payload_matches(name, file_len, header_len, bytes)?;
        extents.push(ShardExtent {
            name: name.to_string(),
            file_len,
            header_len,
            tensors,
            payload,
        });
    }
    Ok(extents)
}

/// Returns the payload length when it equals the bytes the schema's tensors occupy.
pub(super) fn payload_matches(
    shard: &str,
    file_len: u64,
    header_len: u64,
    tensor_bytes: u64,
) -> Result<u64, EngError> {
    let payload = file_len
        .checked_sub(8)
        .and_then(|rest| rest.checked_sub(header_len))
        .ok_or_else(|| {
            invalid(&format!(
                "shard {shard:?}: header length {header_len} leaves no payload in {file_len} bytes"
            ))
        })?;
    if payload != tensor_bytes {
        return Err(invalid(&format!(
            "shard {shard:?}: tensors occupy {tensor_bytes} bytes but the file holds {payload} payload bytes"
        )));
    }
    Ok(payload)
}

fn invalid(message: &str) -> EngError {
    EngError::Checkpoint(format!("EXL3 Flint pack: {message}"))
}
