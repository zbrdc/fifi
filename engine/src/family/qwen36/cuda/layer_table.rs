//! The `(layer, role) -> tensor name` lookup and the per-kind layer ordinals of the resident step
//! (REPLY-43 item 7, packet RD-9b).
//!
//! Pure host code. The inventory (lane C) lists every expected tensor with its role and its GGUF
//! name `blk.<layer>.<suffix>`; this table groups them by layer once, so a launch asks for
//! `(layer, Role)` and gets the name to look up in `CudaWeights::get`, and for the ordinal of the
//! layer among the layers of its kind, which is the index of its DeltaNet state or KV planes.
//! The suffix text stays private to the inventory: the layer number is read from the name prefix.

use super::dims::StepDims;
use crate::{
    EngError,
    family::{
        AttentionKind,
        gguf_model::{Inventory, Role},
    },
};

/// One layer: its kind, its ordinal among the layers of that kind and its tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LayerEntry {
    kind: AttentionKind,
    ordinal: usize,
    names: Vec<(Role, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LayerTable {
    layers: Vec<LayerEntry>,
}

/// Where a layer's mixer state lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LayerSlot {
    pub(super) kind: AttentionKind,
    pub(super) ordinal: usize,
}

/// The layer number of `blk.<layer>.<suffix>`, or `None` for a global tensor.
pub(super) fn layer_of(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("blk.")?;
    let (number, _) = rest.split_once('.')?;
    number.parse().ok()
}

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

    pub(super) fn len(&self) -> usize {
        self.layers.len()
    }

    pub(super) fn slot(&self, layer: usize) -> Result<LayerSlot, EngError> {
        let entry = self.entry(layer)?;
        Ok(LayerSlot {
            kind: entry.kind,
            ordinal: entry.ordinal,
        })
    }

    /// The GGUF name of `role` in `layer`; `Checkpoint` when the layer does not hold the role.
    pub(super) fn name(&self, layer: usize, role: Role) -> Result<&str, EngError> {
        self.entry(layer)?
            .names
            .iter()
            .find(|(candidate, _)| *candidate == role)
            .map(|(_, name)| name.as_str())
            .ok_or_else(|| {
                EngError::Checkpoint(format!("layer table: layer {layer} has no role {role:?}"))
            })
    }

    fn entry(&self, layer: usize) -> Result<&LayerEntry, EngError> {
        self.layers.get(layer).ok_or_else(|| {
            EngError::BadRequest(format!(
                "layer table: layer {layer} is outside {} layers",
                self.layers.len()
            ))
        })
    }
}
