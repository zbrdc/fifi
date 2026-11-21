//! Validate the published text tensor geometry from the safetensors header.

use std::{
    fs,
    path::{Path, PathBuf},
};

use super::{TextSpec, bad, checked_product};
use crate::{
    EngError, checkpoint_json::UniqueValue,
    weights::nemotron_shards::NemotronShardReader as ShardReader,
};

pub(crate) struct CheckpointFacts {
    pub(crate) tensors: u64,
    pub(crate) payload_bytes: u64,
    pub(crate) nvfp4_layers: usize,
    pub(crate) mxfp8_layers: usize,
}

pub(crate) fn inspect_checkpoint(
    directory: &Path,
    spec: &TextSpec,
) -> Result<CheckpointFacts, EngError> {
    let (_, file) = open_shard(directory)?;
    validate_text(&file, spec, file.tensor_count() as u64)
}

#[cfg(target_pointer_width = "64")]
pub(super) fn open_checkpoint(
    directory: &Path,
    spec: &TextSpec,
) -> Result<(PathBuf, ShardReader), EngError> {
    let (path, file) = open_shard(directory)?;
    validate_text(&file, spec, file.tensor_count() as u64)?;
    Ok((path, file))
}

fn open_shard(directory: &Path) -> Result<(PathBuf, ShardReader), EngError> {
    let bytes = fs::read(directory.join("model.safetensors.index.json"))
        .map_err(|error| bad(format!("read safetensors index: {error}")))?;
    let index = serde_json::from_slice::<UniqueValue>(&bytes)
        .map_err(|error| bad(format!("index JSON: {error}")))?
        .0;
    let map = index["weight_map"]
        .as_object()
        .ok_or_else(|| bad("the index requires a weight_map object"))?;
    let name = map
        .values()
        .next()
        .and_then(|name| name.as_str())
        .filter(|name| !name.contains(['/', '\\']) && name.ends_with(".safetensors"))
        .ok_or_else(|| bad("the index requires a safetensors shard filename"))?;
    if map.values().any(|value| value.as_str() != Some(name)) {
        return Err(bad(
            "expected the single safetensors shard of the Mini release",
        ));
    }
    let path = directory.join(name);
    let file = ShardReader::open(&path)?;
    if file.tensor_count() != map.len()
        || index["metadata"]["total_size"].as_u64() != Some(file.payload_len())
        || map.keys().any(|key| file.tensor(key).is_none())
    {
        return Err(bad(
            "the tensor names or byte count disagree with the shard index",
        ));
    }

    Ok((path, file))
}

fn validate_text(
    file: &ShardReader,
    spec: &TextSpec,
    tensors: u64,
) -> Result<CheckpointFacts, EngError> {
    for name in ["embed", "unembed"] {
        expect(
            file,
            &format!("model.llm.{name}.weight"),
            "BF16",
            &[spec.vocab_size, spec.hidden_size],
        )?;
    }
    for name in ["embed_norm", "norm"] {
        expect(
            file,
            &format!("model.llm.{name}.weight"),
            "BF16",
            &[spec.hidden_size],
        )?;
    }
    let queries = [
        checked_product([2, (spec.core.core_end - spec.core.core_start) as u64])?,
        spec.hidden_size,
    ];
    for name in [
        "middle_cycle_cross_pass_queries",
        "middle_cycle_cross_pass_kv_queries",
    ] {
        expect(file, &format!("model.llm.{name}"), "F32", &queries)?;
    }

    let mut nvfp4_layers = 0;
    let mut mxfp8_layers = 0;
    for layer in 0..spec.num_hidden_layers {
        let prefix = format!("model.llm.layers.{layer}.mlp");
        expect(
            file,
            &format!("{prefix}.gate.weight"),
            "BF16",
            &[spec.n_routed_experts, spec.hidden_size],
        )?;
        expect(
            file,
            &format!("{prefix}.gate.bias"),
            "F32",
            &[spec.n_routed_experts],
        )?;
        let prefix = format!("{prefix}.experts");
        let name = format!("{prefix}.w13_weight");
        let nvfp4 = match file.tensor(&name).map(|tensor| tensor.dtype()) {
            Some("U8") => true,
            Some("F8_E4M3") => false,
            _ => return Err(bad(format!("unsupported expert payload for {name}"))),
        };
        expert_matrix(file, spec, &prefix, "w13", true, nvfp4)?;
        expert_matrix(file, spec, &prefix, "w2", false, nvfp4)?;
        nvfp4_layers += usize::from(nvfp4);
        mxfp8_layers += usize::from(!nvfp4);
    }
    Ok(CheckpointFacts {
        tensors,
        payload_bytes: file.payload_len(),
        nvfp4_layers,
        mxfp8_layers,
    })
}

fn expert_matrix(
    file: &ShardReader,
    spec: &TextSpec,
    prefix: &str,
    matrix: &str,
    fused_gate_up: bool,
    nvfp4: bool,
) -> Result<(), EngError> {
    let (rows, columns) = if fused_gate_up {
        (
            checked_product([2, spec.intermediate_size])?,
            spec.hidden_size,
        )
    } else {
        (spec.hidden_size, spec.intermediate_size)
    };
    let experts = spec.n_routed_experts;
    let name = format!("{prefix}.{matrix}_weight");
    let (dtype, stored_columns, scale_dtype, group) = if nvfp4 {
        ("U8", columns / 2, "F8_E4M3", 16)
    } else {
        ("F8_E4M3", columns, "U8", 32)
    };
    expect(file, &name, dtype, &[experts, rows, stored_columns])?;
    expect(
        file,
        &format!("{name}_scale"),
        scale_dtype,
        &[experts, rows, columns / group],
    )?;
    if nvfp4 {
        expect(file, &format!("{name}_amax"), "F32", &[experts])?;
        expect(
            file,
            &format!("{name}_outer_scale_denominator"),
            "F32",
            &[1],
        )?;
    }
    Ok(())
}

fn expect(file: &ShardReader, name: &str, dtype: &str, shape: &[u64]) -> Result<(), EngError> {
    let tensor = file
        .tensor(name)
        .ok_or_else(|| bad(format!("missing {name}")))?;
    if tensor.dtype() == dtype && tensor.shape() == shape {
        return Ok(());
    }
    Err(bad(format!(
        "{name} is {} {:?}, expected {dtype} {shape:?}",
        tensor.dtype(),
        tensor.shape()
    )))
}
