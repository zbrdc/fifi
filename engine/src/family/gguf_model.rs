//! GGUF model opener for Qwen3.6-35B-A3B (decision 0040, packet 5).
//!
//! `open_qwen3_6_35b_a3b` opens one single-file GGUF under the item 4 limits, checks
//! `general.architecture` and the block, embedding, expert and used-expert counts,
//! matches the tensor set against the `inventory` exactly, checks every tensor's type
//! and dims, and checks the CUDA tile rules of the `DeviceTiled` tensors. The open is
//! strict: a tensor of any type the role does not allow, an extra or missing tensor,
//! or a wrong count fails and names the tensor or key. There is no skip mode. The
//! public surface is `open_qwen3_6_35b_a3b`, `GgufModel`, `ModelPlan` and `TensorPath`.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use self::inventory::ExpectedTensor;
pub(crate) use self::inventory::Inventory;
use super::{FamilySpec, QWEN3_6_35B_A3B};
use crate::{
    EngError,
    weights::{GgufHeader, GgufLimits, GgufTensor, GgufWeights, MetaValue, dequantize},
};

mod inventory;
#[cfg(test)]
mod tests;

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) use inventory::Role;
pub use inventory::TensorPath;

/// `general.architecture` of the llama.cpp conversion of Qwen3.6-35B-A3B.
const ARCHITECTURE: &str = "qwen35moe";
/// Decision 0040 item 4 limits.
const LIMITS: GgufLimits = GgufLimits {
    max_header_bytes: 100_000_000,
    max_tensor_bytes: 1 << 30,
};
// The CUDA tile rules of `kernels::cuda::plan` (Linux with native-cuda only). Keep both in step.
const TILE_ROWS: u64 = 8;
const TILE_COLS: u64 = 256;
const MAX_TILE_COLS: u64 = 32768;

/// Opens a Qwen3.6-35B-A3B GGUF and plans its tensors.
///
/// The checks of this module (metadata, tensor set, type, dims, tile rules) fail as
/// `EngError::Checkpoint` with the prefix `gguf model: `, and the message names the
/// tensor or metadata key. Failures of `GgufWeights::open` pass through unchanged: a
/// parse, IO, split, unsupported-type or header-hash failure is `Checkpoint` with the
/// prefix `gguf: `, and a tensor over the 1 GiB limit is `EngError::MemoryPlan`.
pub fn open_qwen3_6_35b_a3b(
    path: &Path,
    header_sha256: Option<&str>,
) -> Result<GgufModel, EngError> {
    open_with_spec(path, header_sha256, &QWEN3_6_35B_A3B)
}

/// One opened model: the checked weights, the plan of its tensors, and the path.
#[derive(Debug)]
pub struct GgufModel {
    /// The CUDA projector reads the planned tensors from this reader.
    pub(crate) weights: GgufWeights,
    plan: ModelPlan,
    path: PathBuf,
}

impl GgufModel {
    pub fn plan(&self) -> &ModelPlan {
        &self.plan
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Reads every `DeviceF32` tensor of the plan to the host; F32 is copied as is. This is the
    /// oracle copy of the tensors that also sit on the device; the CUDA load does not call it.
    pub fn read_host_f32(&self) -> Result<HashMap<String, Vec<f32>>, EngError> {
        let mut host = HashMap::new();
        for (name, path) in self.plan.tensors() {
            if path != TensorPath::DeviceF32 {
                continue;
            }
            let (ty, bytes) = self.weights.read_tensor(name)?;
            host.insert(name.to_owned(), dequantize(ty, &bytes)?);
        }
        Ok(host)
    }
}

/// The classified tensors of one model, in inventory order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPlan {
    tensors: Vec<PlannedTensor>,
}

impl ModelPlan {
    /// Name and path of every tensor, in inventory order.
    pub fn tensors(&self) -> impl Iterator<Item = (&str, TensorPath)> {
        self.tensors
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor.path))
    }

    pub fn count(&self, path: TensorPath) -> usize {
        self.tensors
            .iter()
            .filter(|tensor| tensor.path == path)
            .count()
    }

    /// Source bytes in the file of all tensors with `path`.
    pub fn source_bytes(&self, path: TensorPath) -> u64 {
        self.tensors
            .iter()
            .filter(|tensor| tensor.path == path)
            .map(|tensor| tensor.source_bytes)
            .sum()
    }

    /// Names of every tensor of the plan, in inventory order: each one goes to the device.
    pub fn device_names(&self) -> Vec<&str> {
        self.tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect()
    }
}

/// One classified tensor: name, load path, and source bytes in the file.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannedTensor {
    name: String,
    path: TensorPath,
    source_bytes: u64,
}

/// Opens `path` against `spec`: the inventory, the header and metadata checks, the
/// exact tensor set, then the classification of every tensor in inventory order.
fn open_with_spec(
    path: &Path,
    header_sha256: Option<&str>,
    spec: &FamilySpec,
) -> Result<GgufModel, EngError> {
    let inventory = Inventory::for_spec(spec)?;
    let weights = GgufWeights::open(path, LIMITS, header_sha256)?;
    check_metadata(weights.header(), spec)?;
    let pairs = match_tensors(weights.header(), &inventory, spec.name)?;
    let mut tensors = Vec::with_capacity(pairs.len());
    for (expected, tensor) in pairs {
        tensors.push(classify(expected, tensor)?);
    }
    Ok(GgufModel {
        weights,
        plan: ModelPlan { tensors },
        path: path.to_path_buf(),
    })
}

/// Every opener rejection shares this prefix, so callers can tell it from the header
/// reader's own `gguf: ` errors.
fn reject(message: impl Into<String>) -> EngError {
    EngError::Checkpoint(format!("gguf model: {}", message.into()))
}

/// Checks `general.architecture` and the four counts that fix the tensor shapes.
fn check_metadata(header: &GgufHeader, spec: &FamilySpec) -> Result<(), EngError> {
    match header.meta("general.architecture") {
        Some(MetaValue::String(found)) if found == ARCHITECTURE => {}
        Some(MetaValue::String(found)) => {
            return Err(reject(format!(
                "metadata key general.architecture is {found:?}, expected {ARCHITECTURE:?}"
            )));
        }
        Some(_) => return Err(reject("metadata key general.architecture is not a string")),
        None => return Err(reject("metadata key general.architecture is missing")),
    }
    let counts = [
        ("block_count", spec.layers),
        ("embedding_length", spec.hidden),
        ("expert_count", spec.experts),
        ("expert_used_count", spec.top_k),
    ];
    for (suffix, expected) in counts {
        let key = format!("{ARCHITECTURE}.{suffix}");
        let found = meta_count(header, &key)?;
        if found != u64::from(expected) {
            return Err(reject(format!(
                "metadata key {key} is {found}, expected {expected}"
            )));
        }
    }
    Ok(())
}

/// Reads one count key. Only u32 and u64 are accepted; the conversion writes u32.
fn meta_count(header: &GgufHeader, key: &str) -> Result<u64, EngError> {
    match header.meta(key) {
        Some(MetaValue::U32(value)) => Ok(u64::from(*value)),
        Some(MetaValue::U64(value)) => Ok(*value),
        Some(_) => Err(reject(format!(
            "metadata key {key} is not a u32 or u64 integer"
        ))),
        None => Err(reject(format!("metadata key {key} is missing"))),
    }
}

/// Pairs every inventory tensor with its file entry. Extra file tensors fail first
/// in file order, then missing inventory tensors in inventory order.
fn match_tensors<'a>(
    header: &'a GgufHeader,
    inventory: &'a Inventory,
    family: &str,
) -> Result<Vec<(&'a ExpectedTensor, &'a GgufTensor)>, EngError> {
    for tensor in header.tensors() {
        if inventory.get(tensor.name()).is_none() {
            return Err(reject(format!(
                "tensor {} is not in the {family} inventory",
                tensor.name()
            )));
        }
    }
    let mut pairs = Vec::with_capacity(inventory.tensors().len());
    for expected in inventory.tensors() {
        let tensor = header
            .tensor(&expected.name)
            .ok_or_else(|| reject(format!("tensor {} is missing", expected.name)))?;
        pairs.push((expected, tensor));
    }
    Ok(pairs)
}

/// Classifies one tensor: the type within the role's allowed types, the exact dims,
/// then the tile rules for a `DeviceTiled` tensor.
fn classify(expected: &ExpectedTensor, tensor: &GgufTensor) -> Result<PlannedTensor, EngError> {
    let ty = tensor.ggml_type();
    let allowed = expected.allowed_types();
    if !allowed.contains(&ty) {
        let names = allowed
            .iter()
            .map(|allowed| allowed.name())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(reject(format!(
            "tensor {} has type {}, expected one of {}",
            expected.name,
            ty.name(),
            names
        )));
    }
    if tensor.dims() != expected.dims.as_slice() {
        return Err(reject(format!(
            "tensor {} has dims {:?}, expected {:?}",
            expected.name,
            tensor.dims(),
            expected.dims
        )));
    }
    let path = expected.path();
    if path == TensorPath::DeviceTiled {
        check_tiles(&expected.name, tensor.dims())?;
    }
    Ok(PlannedTensor {
        name: expected.name.clone(),
        path,
        source_bytes: expected.byte_len(ty)?,
    })
}

/// The CUDA tile rules: the row count a multiple of 8 and the row length a multiple
/// of 256 up to 32768. A rank-3 expert stack also needs its output count a multiple
/// of 8, so every expert slice starts on a tile row group. `dims` must be non-empty:
/// `classify` calls this only after the dims equal the inventory's, which never has none.
fn check_tiles(name: &str, dims: &[u64]) -> Result<(), EngError> {
    let cols = dims[0];
    let rows = dims[1..].iter().product::<u64>();
    let stack_ok = dims.len() != 3 || dims[1].is_multiple_of(TILE_ROWS);
    if rows.is_multiple_of(TILE_ROWS)
        && cols.is_multiple_of(TILE_COLS)
        && cols <= MAX_TILE_COLS
        && stack_ok
    {
        return Ok(());
    }
    Err(reject(format!(
        "tensor {name} has dims {dims:?} that the CUDA tiles cannot hold: rows must be a multiple of {TILE_ROWS}, the row length a multiple of {TILE_COLS} up to {MAX_TILE_COLS}, and the output count of a rank-3 stack a multiple of {TILE_ROWS}"
    )))
}
