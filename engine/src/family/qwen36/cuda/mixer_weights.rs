//! The weights of the mixer blocks, resolved by name once (packet RD-9b3a; REPLY-43 item 7).
//!
//! `MixerWeights::new` walks the dense and raw specs of `MixerGeometry` for each wanted layer,
//! finds the tensor by the name `LayerTable` gives, and refuses a tensor of another shape, type or
//! stack depth with a `Checkpoint` error at session build, never at launch. A block then asks a
//! layer for a weight by role. There is no pure part to test on the CPU beyond the specs and
//! `check_dense` (RD-9b2); the resolution against a real `CudaWeights` is covered by the scratch
//! device run of RD-9b3b.
//!
//! Design guidance only, no code copied.

use super::{
    layer_table::LayerTable,
    mixer_geometry::{MixerGeometry, check_dense},
};
use crate::{
    EngError,
    family::{
        gguf_model::Role,
        project::cuda::{CudaWeight, CudaWeights},
    },
    kernels::cuda::{DeviceRange, TiledWeight},
    weights::GgufType,
};

const F32: usize = size_of::<f32>();

/// The weights of one layer: dense tiles and raw f32 vectors by role.
pub(super) struct LayerWeights<'w> {
    dense: Vec<(Role, &'w TiledWeight)>,
    raw: Vec<(Role, DeviceRange<'w>)>,
}

impl<'w> LayerWeights<'w> {
    pub(super) fn dense(&self, role: Role) -> Result<&'w TiledWeight, EngError> {
        self.dense
            .iter()
            .find(|(known, _)| *known == role)
            .map(|(_, weight)| *weight)
            .ok_or_else(|| EngError::BadRequest(format!("mixer: no dense weight for {role:?}")))
    }

    pub(super) fn raw(&self, role: Role) -> Result<DeviceRange<'w>, EngError> {
        self.raw
            .iter()
            .find(|(known, _)| *known == role)
            .map(|(_, range)| *range)
            .ok_or_else(|| EngError::BadRequest(format!("mixer: no raw weight for {role:?}")))
    }
}

/// The weights of every wanted layer.
pub(super) struct MixerWeights<'w> {
    /// `None` for a layer that was not asked for (a device test loads two or three layers).
    layers: Vec<Option<LayerWeights<'w>>>,
}

fn raw_f32<'w>(
    weights: &'w CudaWeights,
    name: &str,
    want_values: usize,
) -> Result<DeviceRange<'w>, EngError> {
    let Some(CudaWeight::Raw { ty, buffer, .. }) = weights.get(name) else {
        return Err(EngError::Checkpoint(format!(
            "mixer weights: {name} is not a raw tensor"
        )));
    };
    if *ty != GgufType::F32 || buffer.bytes() != want_values * F32 {
        return Err(EngError::Checkpoint(format!(
            "mixer weights: {name} is {ty:?} of {} bytes, the step needs {want_values} f32",
            buffer.bytes()
        )));
    }
    buffer.range(0, buffer.bytes())
}

impl<'w> MixerWeights<'w> {
    /// Resolves the layers in `wanted` (every layer when the session serves tokens).
    pub(super) fn new(
        geometry: &MixerGeometry,
        table: &LayerTable,
        weights: &'w CudaWeights,
        wanted: &[usize],
    ) -> Result<Self, EngError> {
        let layers = (0..table.len())
            .map(|layer| {
                if wanted.contains(&layer) {
                    Self::layer(geometry, table, weights, layer).map(Some)
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { layers })
    }

    /// The weights of `layer`, or `BadRequest` when it was not resolved.
    pub(super) fn layer_weights(&self, layer: usize) -> Result<&LayerWeights<'w>, EngError> {
        self.layers
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                EngError::BadRequest(format!("mixer: layer {layer} has no resolved weights"))
            })
    }

    fn layer(
        g: &MixerGeometry,
        table: &LayerTable,
        weights: &'w CudaWeights,
        layer: usize,
    ) -> Result<LayerWeights<'w>, EngError> {
        let kind = table.slot(layer)?.kind;
        let mut dense = Vec::new();
        for spec in g.dense_specs(kind)? {
            let name = table.name(layer, spec.role)?;
            let Some(CudaWeight::Tiled {
                weight, experts, ..
            }) = weights.get(name)
            else {
                return Err(EngError::Checkpoint(format!(
                    "mixer weights: {name} is not a tiled tensor"
                )));
            };
            check_dense(name, (weight.n_rows(), weight.k(), *experts), &spec)?;
            dense.push((spec.role, weight));
        }
        let mut raw = Vec::new();
        for spec in g.raw_specs(kind)? {
            raw.push((
                spec.role,
                raw_f32(weights, table.name(layer, spec.role)?, spec.values)?,
            ));
        }
        Ok(LayerWeights { dense, raw })
    }
}
