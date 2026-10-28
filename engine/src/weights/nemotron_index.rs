use std::{collections::BTreeMap, path::Path};

use serde::Serialize;
use serde_json::Value;

use crate::{EngError, checkpoint_json::UniqueValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NemotronComponent {
    Text,
    Mtp,
    Vision,
    Connector,
    Ambiguous,
}

impl NemotronComponent {
    pub fn classify_tensor_name(name: &str) -> Self {
        if name.starts_with("language_model.backbone.") {
            Self::Text
        } else if name.starts_with("language_model.mtp.") {
            Self::Mtp
        } else if name.starts_with("vision_model.") {
            Self::Vision
        } else if name.starts_with("vision_projector.") {
            Self::Connector
        } else {
            Self::Ambiguous
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct NemotronComponentCounts {
    pub text: u64,
    pub mtp: u64,
    pub vision: u64,
    pub connector: u64,
    pub ambiguous: u64,
}

impl NemotronComponentCounts {
    fn increment(&mut self, component: NemotronComponent) -> Result<(), EngError> {
        let count = match component {
            NemotronComponent::Text => &mut self.text,
            NemotronComponent::Mtp => &mut self.mtp,
            NemotronComponent::Vision => &mut self.vision,
            NemotronComponent::Connector => &mut self.connector,
            NemotronComponent::Ambiguous => &mut self.ambiguous,
        };
        *count = count
            .checked_add(1)
            .ok_or_else(|| index_error("component tensor count overflows u64"))?;
        Ok(())
    }
}

#[derive(Debug)]
struct TensorLocation {
    shard: String,
    component: NemotronComponent,
}

#[derive(Debug)]
pub struct NemotronIndex {
    tensors: BTreeMap<String, TensorLocation>,
    shard_key_counts: BTreeMap<String, u64>,
    component_counts: NemotronComponentCounts,
    tensor_count: u64,
}

impl NemotronIndex {
    pub fn from_json(bytes: &[u8]) -> Result<Self, EngError> {
        let root = serde_json::from_slice::<UniqueValue>(bytes)
            .map_err(|error| index_error(format!("index JSON: {error}")))?
            .0;
        if !root.is_object() {
            return Err(index_error("index root must be an object"));
        }
        let weight_map = root
            .get("weight_map")
            .and_then(Value::as_object)
            .filter(|map| !map.is_empty())
            .ok_or_else(|| index_error("index requires a nonempty weight_map object"))?;

        let mut tensors = BTreeMap::new();
        let mut shard_key_counts = BTreeMap::new();
        let mut component_counts = NemotronComponentCounts::default();
        let mut tensor_count = 0u64;

        for (tensor_name, shard_value) in weight_map {
            if tensor_name.is_empty() {
                return Err(index_error("tensor name must not be empty"));
            }
            let shard = shard_value.as_str().ok_or_else(|| {
                index_error(format!(
                    "index tensor {tensor_name:?} requires a string shard"
                ))
            })?;
            if !valid_shard_name(shard) {
                return Err(index_error(format!(
                    "index tensor {tensor_name:?}: invalid shard name {shard:?}"
                )));
            }

            let component = NemotronComponent::classify_tensor_name(tensor_name);
            let count = shard_key_counts.entry(shard.to_string()).or_insert(0u64);
            *count = count
                .checked_add(1)
                .ok_or_else(|| index_error("shard key count overflows u64"))?;
            component_counts.increment(component)?;
            tensor_count = tensor_count
                .checked_add(1)
                .ok_or_else(|| index_error("tensor count overflows u64"))?;
            tensors.insert(
                tensor_name.clone(),
                TensorLocation {
                    shard: shard.to_string(),
                    component,
                },
            );
        }

        Ok(Self {
            tensors,
            shard_key_counts,
            component_counts,
            tensor_count,
        })
    }

    pub fn shard_for_tensor(&self, name: &str) -> Option<&str> {
        self.tensors
            .get(name)
            .map(|location| location.shard.as_str())
    }

    pub fn tensor_count(&self) -> u64 {
        self.tensor_count
    }

    pub fn shard_key_counts(&self) -> &BTreeMap<String, u64> {
        &self.shard_key_counts
    }

    pub fn component_counts(&self) -> NemotronComponentCounts {
        self.component_counts
    }

    pub fn write_inventory(&self, path: impl AsRef<Path>) -> Result<(), EngError> {
        let inventory = self
            .tensors
            .iter()
            .map(|(tensor_name, location)| InventoryEntry {
                tensor_name,
                shard: &location.shard,
                component: location.component,
            })
            .collect::<Vec<_>>();
        let bytes = serde_json::to_vec(&inventory)
            .map_err(|error| index_error(format!("serialize index inventory: {error}")))?;
        std::fs::write(path, bytes)
            .map_err(|error| index_error(format!("write index inventory: {error}")))
    }
}

#[derive(Serialize)]
struct InventoryEntry<'a> {
    tensor_name: &'a str,
    shard: &'a str,
    component: NemotronComponent,
}

fn valid_shard_name(name: &str) -> bool {
    !name.is_empty() && !name.contains(['/', '\\']) && name.ends_with(".safetensors")
}

fn index_error(message: impl Into<String>) -> EngError {
    EngError::Checkpoint(format!("Nemotron index: {}", message.into()))
}
