//! Bounded header-driven importer for the published Apple affine checkpoint.
//!
//! Opening reads headers only. Each projection read is bounded by its checked
//! stored shape; no full checkpoint copy or expert-handle expansion is retained.

use std::{collections::BTreeMap, fs::File, io::Read, path::Path};

use crate::{
    EngError,
    kernels::{PackedWeight, QuantizationMode, QuantizationSpec},
    tensor::{Dtype, Tensor},
    weights::{
        AcquireRequest, CancelHandle, Completion, Priority, Selection, ShardHeader, ShardIo,
        StreamedCheckpoint,
    },
};

#[path = "weights_profile.rs"]
mod profile;
use profile::Profile;

const HEADER_LIMIT: u64 = 16 * 1024 * 1024;
const EMBEDDING: &str = "model.embed_tokens";

pub(super) struct FlintCheckpoint {
    weights: StreamedCheckpoint,
    profile: Profile,
}

impl FlintCheckpoint {
    pub(super) fn open(root: &Path) -> Result<Self, EngError> {
        check_apple_config(&read_config(root)?)?;
        let mut paths = std::fs::read_dir(root)
            .map_err(checkpoint_io)?
            .map(|entry| entry.map(|entry| entry.path()).map_err(checkpoint_io))
            .collect::<Result<Vec<_>, _>>()?;
        paths.retain(|path| path.extension().is_some_and(|ext| ext == "safetensors"));
        paths.sort();
        let wanted: Vec<_> = (0..5)
            .map(|index| root.join(format!("model-{index:05}.safetensors")))
            .collect();
        if paths != wanted {
            return Err(invalid(
                "requires exactly the five published safetensors shards",
            ));
        }
        let mut shards: Vec<(String, Box<dyn ShardIo + Send>, u64)> = Vec::new();
        let mut routes = BTreeMap::new();
        for path in paths {
            let mut file = File::open(&path).map_err(checkpoint_io)?;
            let len = file.metadata().map_err(checkpoint_io)?.len();
            let header = ShardHeader::read(&mut file, len, HEADER_LIMIT)?;
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| invalid("invalid shard filename"))?
                .to_owned();
            for (key, _) in header.iter() {
                if routes.insert(key.to_owned(), name.clone()).is_some() {
                    return Err(invalid(&format!("duplicate tensor {key}")));
                }
            }
            shards.push((name, Box::new(file), len));
        }
        // The publisher has no index. Derive only routing, then let the common
        // indexed reader check the same file handles and all header membership.
        let index = serde_json::to_vec(&serde_json::json!({"weight_map": routes}))
            .map_err(|error| invalid(&format!("derived index: {error}")))?;
        let weights = StreamedCheckpoint::open(&index, shards, HEADER_LIMIT)?;
        let profile = Profile::pinned();
        if routes.len() != profile.tensors.len() {
            return Err(invalid(
                "tensor inventory differs from the published profile",
            ));
        }
        for (name, expected) in &profile.tensors {
            if weights.stored(name).as_ref() != Some(expected) {
                return Err(invalid(&format!(
                    "tensor {name} is missing or has an incorrect shape/dtype"
                )));
            }
        }
        Ok(Self { weights, profile })
    }

    pub(super) fn tensor(&mut self, name: &str) -> Result<Tensor, EngError> {
        if let Some((dtype, _)) = self.weights.stored(name)
            && !matches!(dtype, Dtype::Bf16 | Dtype::F32)
        {
            return Err(invalid("tensor() requires an unpacked BF16/F32 tensor"));
        }
        self.read(name)
    }

    pub(super) fn packed(&mut self, stem: &str) -> Result<PackedWeight, EngError> {
        let (logical, bits) = self
            .profile
            .packed
            .get(stem)
            .cloned()
            .ok_or_else(|| invalid(&format!("unknown packed stem {stem}")))?;
        let payload = word_view(self.read(&format!("{stem}.weight"))?)?;
        let scales = self.read(&format!("{stem}.scales"))?;
        let biases = self.read(&format!("{stem}.biases"))?;
        PackedWeight::try_new(payload, scales, Some(biases), logical, affine(bits))
    }

    pub(super) fn embedding_rows(&mut self, ids: &[u32]) -> Result<PackedWeight, EngError> {
        let (shape, bits) = self
            .profile
            .packed
            .get(EMBEDDING)
            .ok_or_else(|| invalid("missing embedding profile"))?;
        if *bits != 8 || shape.len() != 2 || shape[1] != 2048 {
            return Err(invalid("embedding requires the published INT8 row profile"));
        }
        if ids.is_empty() || ids.iter().any(|&id| id >= shape[0]) {
            return Err(EngError::BadRequest(
                "Flint embedding IDs must be nonempty and within the vocabulary".into(),
            ));
        }
        let rows =
            u32::try_from(ids.len()).map_err(|_| invalid("embedding row count exceeds u32"))?;
        let payload = self.gather(&format!("{EMBEDDING}.weight"), ids, 2048, Dtype::U8)?;
        let scales = self.gather(&format!("{EMBEDDING}.scales"), ids, 32, Dtype::Bf16)?;
        let biases = self.gather(&format!("{EMBEDDING}.biases"), ids, 32, Dtype::Bf16)?;
        PackedWeight::try_new(
            word_view(payload)?,
            scales,
            Some(biases),
            vec![rows, 2048],
            affine(8),
        )
    }

    fn read(&self, name: &str) -> Result<Tensor, EngError> {
        let request = AcquireRequest::new(
            name.into(),
            Selection::Whole,
            Priority::Demand,
            CancelHandle::new(),
        );
        match self.weights.wait(self.weights.acquire(request)?)? {
            Completion::Ready(tensor) => Ok(tensor),
            Completion::Missing => Err(invalid(&format!("missing tensor {name}"))),
        }
    }

    fn gather(
        &self,
        name: &str,
        ids: &[u32],
        width: u32,
        dtype: Dtype,
    ) -> Result<Tensor, EngError> {
        let row_bytes = usize::try_from(u64::from(width) * dtype.byte_width())
            .map_err(|_| invalid("embedding row bytes exceed usize"))?;
        let len = ids
            .len()
            .checked_mul(row_bytes)
            .ok_or_else(|| invalid("embedding byte count overflows"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .map_err(|_| EngError::MemoryPlan("Flint embedding gather allocation failed".into()))?;
        bytes.resize(len, 0);
        for (&id, row) in ids.iter().zip(bytes.chunks_exact_mut(row_bytes)) {
            self.weights.read_slice_into(name, id, row)?;
        }
        let rows =
            u32::try_from(ids.len()).map_err(|_| invalid("embedding row count exceeds u32"))?;
        Tensor::try_from_host(dtype, vec![rows, width], bytes)
    }
}

fn affine(bits: u8) -> QuantizationSpec {
    QuantizationSpec {
        mode: QuantizationMode::Affine,
        bits: u32::from(bits),
        group_size: 64,
    }
}

fn word_view(payload: Tensor) -> Result<Tensor, EngError> {
    if payload.dtype() != Dtype::U8 || !matches!(payload.shape().len(), 2 | 3) {
        return Err(invalid("packed payload must be rank-two or rank-three U8"));
    }
    let mut shape = payload.shape().to_vec();
    let last = shape.len() - 1;
    if shape[last] == 0 || !shape[last].is_multiple_of(4) {
        return Err(invalid(
            "packed rows must contain complete little-endian U32 words",
        ));
    }
    shape[last] /= 4;
    // Tensor stores canonical bytes. This changes dtype/shape, not numerical
    // values or byte order. The bounded source buffer drops after this copy.
    let bytes = payload
        .host_bytes()
        .ok_or_else(|| invalid("packed payload is not host-readable"))?
        .to_vec();
    Tensor::try_from_host(Dtype::U32, shape, bytes)
}

fn read_config(root: &Path) -> Result<Vec<u8>, EngError> {
    let file = File::open(root.join("config.json")).map_err(checkpoint_io)?;
    let mut bytes = Vec::new();
    file.take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(checkpoint_io)?;
    if bytes.len() > 1024 * 1024 {
        return Err(invalid("configuration exceeds one MiB"));
    }
    Ok(bytes)
}

fn check_apple_config(bytes: &[u8]) -> Result<(), EngError> {
    super::check_config(bytes)?;
    let root: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| invalid(&error.to_string()))?;
    let quant = root
        .get("quantization")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| invalid("requires the qualified Apple affine quantization"))?;
    if quant
        .keys()
        .any(|key| !matches!(key.as_str(), "bits" | "group_size" | "mode"))
    {
        return Err(invalid("quantization overrides are not qualified"));
    }
    if root
        .get("mlx_export")
        .and_then(|export| export.get("scale_rule"))
        .and_then(serde_json::Value::as_str)
        != Some("mse")
    {
        return Err(invalid("requires the qualified Apple MSE affine export"));
    }
    Ok(())
}

fn checkpoint_io(error: std::io::Error) -> EngError {
    invalid(&error.to_string())
}

fn invalid(message: &str) -> EngError {
    EngError::Checkpoint(format!("Flint checkpoint: {message}"))
}

#[cfg(test)]
#[path = "weights_tests.rs"]
mod tests;
