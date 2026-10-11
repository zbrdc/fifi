use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use super::{ResidentWeights, Tier, WeightSource};
use crate::{EngError, checkpoint_json::UniqueValue, tensor::Tensor};

#[derive(Debug)]
pub struct IndexedWeights {
    resident: ResidentWeights,
}

impl IndexedWeights {
    pub fn from_shards(
        index: &[u8],
        shards: BTreeMap<String, Arc<[u8]>>,
    ) -> Result<Self, EngError> {
        let weight_map = parse_weight_map(index)?;
        let supplied: BTreeSet<&str> = shards.keys().map(String::as_str).collect();
        check_shard_set(&weight_map, &supplied)?;
        let mut resident = ResidentWeights {
            tensors: Default::default(),
            resident_bytes: 0,
        };
        let mut assignments = BTreeMap::new();
        for (name, bytes) in shards {
            let shard = ResidentWeights::from_bytes(bytes).map_err(|error| match error {
                EngError::Checkpoint(message) => {
                    EngError::Checkpoint(format!("shard {name:?}: {message}"))
                }
                other => other,
            })?;
            resident.resident_bytes = resident
                .resident_bytes
                .checked_add(shard.resident_bytes)
                .ok_or_else(|| {
                    EngError::Checkpoint("indexed payload byte count overflows u64".into())
                })?;
            for (key, tensor) in shard.tensors {
                assignments.insert(key.clone(), name.clone());
                if resident.tensors.insert(key.clone(), tensor).is_some() {
                    return Err(EngError::Checkpoint(format!(
                        "shard {name:?}: duplicate tensor {key:?}"
                    )));
                }
            }
        }
        for (key, actual) in &assignments {
            let expected = weight_map
                .get(key.as_str())
                .map(String::as_str)
                .ok_or_else(|| {
                    EngError::Checkpoint(format!("shard {actual:?}: unindexed tensor {key:?}"))
                })?;
            if expected != actual {
                return Err(EngError::Checkpoint(format!(
                    "tensor {key:?}: index names shard {expected:?}, found in {actual:?}"
                )));
            }
        }
        for (key, expected) in &weight_map {
            if !resident.tensors.contains_key(key.as_str()) {
                return Err(EngError::Checkpoint(format!(
                    "index tensor {key:?} is missing from shard {expected}"
                )));
            }
        }
        Ok(Self { resident })
    }
}

impl WeightSource for IndexedWeights {
    fn tensor(&self, key: &str) -> Option<Tensor> {
        self.resident.tensor(key)
    }

    fn resident_bytes(&self) -> u64 {
        self.resident.resident_bytes()
    }

    fn tier(&self) -> Tier {
        Tier::Resident
    }
}

/// Parses an index and maps each tensor key to its shard basename.
pub(super) fn parse_weight_map(index: &[u8]) -> Result<BTreeMap<String, String>, EngError> {
    let index = serde_json::from_slice::<UniqueValue>(index)
        .map_err(|error| EngError::Checkpoint(format!("index JSON: {error}")))?
        .0;
    let raw_map = index
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .filter(|map| !map.is_empty())
        .ok_or_else(|| {
            EngError::Checkpoint("index requires a nonempty weight_map object".into())
        })?;
    let mut weight_map = BTreeMap::new();
    for (key, shard) in raw_map {
        if key.is_empty() {
            return Err(EngError::Checkpoint(
                "index tensor name must not be empty".into(),
            ));
        }
        let name = shard.as_str().ok_or_else(|| {
            EngError::Checkpoint(format!("index tensor {key:?} requires a string shard"))
        })?;
        if !valid_shard_name(name) {
            return Err(EngError::Checkpoint(format!(
                "index tensor {key:?}: invalid shard basename {name:?}"
            )));
        }
        weight_map.insert(key.clone(), name.to_string());
    }
    Ok(weight_map)
}

fn valid_shard_name(name: &str) -> bool {
    name.is_ascii()
        && name.ends_with(".safetensors")
        && !name
            .bytes()
            .any(|b| b.is_ascii_control() || b == b'/' || b == b'\\')
}

/// Requires the supplied shard names to equal the shard names the index references.
pub(super) fn check_shard_set(
    weight_map: &BTreeMap<String, String>,
    supplied: &BTreeSet<&str>,
) -> Result<(), EngError> {
    let referenced: BTreeSet<&str> = weight_map.values().map(String::as_str).collect();
    if &referenced != supplied {
        return Err(EngError::Checkpoint(format!(
            "index shard set differs: missing {:?}, extra {:?}",
            referenced.difference(supplied).collect::<Vec<_>>(),
            supplied.difference(&referenced).collect::<Vec<_>>()
        )));
    }
    Ok(())
}
