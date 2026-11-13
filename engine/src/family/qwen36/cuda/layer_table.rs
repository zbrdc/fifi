//! The Qwen3.6 half of the layer table: the constructor from [`StepDims`] and the
//! inventory. The table itself moved to [`crate::family::exl3::layer_table`]; its paths
//! re-export from here.

use super::dims::StepDims;
use crate::{
    EngError,
    family::{AttentionKind, gguf_model::Inventory},
};

use crate::family::exl3::layer_table::LayerEntry;
pub(in crate::family) use crate::family::exl3::layer_table::{LayerTable, layer_of};

impl LayerTable {
    pub(super) fn new(dims: &StepDims, inventory: &Inventory) -> Result<Self, EngError> {
        let mut counts = (0usize, 0usize);
        let mut layers = Vec::with_capacity(dims.shape.kinds.len());
        for &kind in &dims.shape.kinds {
            let counter = match kind {
                AttentionKind::DeltaNet => &mut counts.0,
                AttentionKind::Full => &mut counts.1,
                other => {
                    return Err(EngError::Checkpoint(format!(
                        "layer table: attention kind {other:?} has no resident mixer"
                    )));
                }
            };
            layers.push(LayerEntry {
                kind,
                ordinal: *counter,
                names: Vec::new(),
            });
            *counter += 1;
        }
        for tensor in inventory.tensors() {
            let Some(layer) = layer_of(&tensor.name) else {
                continue;
            };
            let entry = layers.get_mut(layer).ok_or_else(|| {
                EngError::Checkpoint(format!(
                    "layer table: tensor {} names layer {layer}, the shape has {}",
                    tensor.name,
                    dims.shape.kinds.len()
                ))
            })?;
            if entry.names.iter().any(|(role, _)| *role == tensor.role) {
                return Err(EngError::Checkpoint(format!(
                    "layer table: layer {layer} holds role {:?} twice",
                    tensor.role
                )));
            }
            entry.names.push((tensor.role, tensor.name.clone()));
        }
        Ok(Self { layers })
    }
}
