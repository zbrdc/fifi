//! The weights of the mixer blocks, resolved by name once (packet RD-9b3a; REPLY-43 item 7).
//!
//! `MixerWeights::new` walks the dense and raw specs of `MixerGeometry` for each wanted layer,
//! finds the tensor by the name `LayerTable` gives, and refuses a tensor of another shape, type or
//! stack depth with a `Checkpoint` error at session build, never at launch. A block then asks a
//! layer for a weight by role. There is no pure part to test on the CPU beyond the specs and
//! `check_dense` (RD-9b2); the resolution against a real `CudaWeights` is covered by the scratch
//! device run of RD-9b3b.
//!
//! MATVEC-PAR S3b: a dense entry is [`DenseWeight`], the tiles of the Tiled load or the file's
//! own blocks of the Native load; `new` takes the session's layout and refuses a tensor of the
//! other layout by name, so a mixed load cannot reach a launch (beside
//! `CudaWeights::layout`'s own mixed refusal). The Native rows%4 rule is
//! `mixer_geometry::check_native_dense`.

use super::{
    layer_table::LayerTable,
    mixer_geometry::{DenseSpec, MixerGeometry, check_dense, check_native_dense},
};
use crate::{
    EngError,
    family::{
        gguf_model::Role,
        project::cuda::{CudaWeight, CudaWeights, WeightLayout, exl3::Exl3Weight},
    },
    kernels::cuda::{DeviceRange, NativeWeight, TiledWeight},
    weights::GgufType,
};

const F32: usize = size_of::<f32>();

/// One dense projection weight of a layer (MATVEC-PAR S3b): the tiles of the Tiled load that
/// `matmul` reads, or the file's own blocks of the Native load that `matvec_q` reads.
#[derive(Clone, Copy)]
pub(super) enum DenseWeight<'w> {
    Tiled(&'w TiledWeight),
    Native(&'w NativeWeight),
    Exl3(&'w Exl3Weight),
}

/// The weights of one layer: dense projections and raw f32 vectors by role.
pub(super) struct LayerWeights<'w> {
    dense: Vec<(Role, DenseWeight<'w>)>,
    raw: Vec<(Role, DeviceRange<'w>)>,
}

impl<'w> LayerWeights<'w> {
    /// The tiles of `role` for a `matmul` launch (the Tiled blocks; a Native entry is a typed
    /// refusal naming the role, so the two layouts cannot cross inside a launch).
    pub(super) fn dense(&self, role: Role) -> Result<&'w TiledWeight, EngError> {
        match self.entry(role)? {
            DenseWeight::Tiled(weight) => Ok(weight),
            DenseWeight::Exl3(_) => Err(EngError::BadRequest(format!(
                "mixer: {role:?} is EXL3; the tiled matmul reads tiles"
            ))),
            DenseWeight::Native(_) => Err(EngError::BadRequest(format!(
                "mixer: {role:?} is loaded in the native layout; the tiled matmul reads tiles"
            ))),
        }
    }

    /// The native blocks of `role` for a `matvec_q` launch (the Tiled entry refuses the same way).
    pub(super) fn native(&self, role: Role) -> Result<&'w NativeWeight, EngError> {
        match self.entry(role)? {
            DenseWeight::Native(weight) => Ok(weight),
            DenseWeight::Exl3(_) => Err(EngError::BadRequest(format!(
                "mixer: {role:?} is EXL3; the native matvec reads blocks"
            ))),
            DenseWeight::Tiled(_) => Err(EngError::BadRequest(format!(
                "mixer: {role:?} is loaded in the tiled layout; the native matvec reads blocks"
            ))),
        }
    }

    pub(super) fn exl3(&self, role: Role) -> Result<&'w Exl3Weight, EngError> {
        match self.entry(role)? {
            DenseWeight::Exl3(weight) => Ok(weight),
            _ => Err(EngError::BadRequest(format!(
                "mixer: {role:?} is not an EXL3 matrix"
            ))),
        }
    }

    fn entry(&self, role: Role) -> Result<&DenseWeight<'w>, EngError> {
        self.dense
            .iter()
            .find(|(known, _)| *known == role)
            .map(|(_, weight)| weight)
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
    /// Resolves the layers in `wanted` (every layer when the session serves tokens) in `layout`.
    pub(super) fn new(
        geometry: &MixerGeometry,
        table: &LayerTable,
        weights: &'w CudaWeights,
        wanted: &[usize],
        layout: WeightLayout,
    ) -> Result<Self, EngError> {
        let layers = (0..table.len())
            .map(|layer| {
                if wanted.contains(&layer) {
                    Self::layer(geometry, table, weights, layer, layout).map(Some)
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { layers })
    }

    /// The ready weight sets of the device rigs (the resident block tests of MATVEC-PAR S3b):
    /// one entry per layer in order, `None` for an unresolved layer, exactly what `new` builds.
    #[cfg(test)]
    pub(super) fn from_layers(layers: Vec<Option<LayerWeights<'w>>>) -> Self {
        Self { layers }
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
        layout: WeightLayout,
    ) -> Result<LayerWeights<'w>, EngError> {
        let kind = table.slot(layer)?.kind;
        let mut dense = Vec::new();
        for spec in g.dense_specs(layout, kind)? {
            let name = table.name(layer, spec.role)?;
            let weight = dense_weight(weights, name, &spec, layout)?;
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

/// Resolves one dense spec of `layer` in `layout`: the tiles (shape-checked by `check_dense`) or
/// the native blocks (shape-checked plus the rows%4 rule of `check_native_dense`), and a tensor
/// of the other layout is a `Checkpoint` naming it, so a mixed load stops at the lookup.
pub(super) fn dense_weight<'w>(
    weights: &'w CudaWeights,
    name: &str,
    spec: &DenseSpec,
    layout: WeightLayout,
) -> Result<DenseWeight<'w>, EngError> {
    match (weights.get(name), layout) {
        (
            Some(CudaWeight::Tiled {
                weight, experts, ..
            }),
            WeightLayout::Tiled,
        ) => {
            check_dense(name, (weight.n_rows(), weight.k(), *experts), spec)?;
            Ok(DenseWeight::Tiled(weight))
        }
        (Some(CudaWeight::Native { weight, .. }), WeightLayout::Native) => {
            check_native_dense(name, (weight.n_rows(), weight.k()), spec)?;
            Ok(DenseWeight::Native(weight))
        }
        (Some(CudaWeight::Exl3 { weight, .. }), WeightLayout::Exl3) => {
            check_dense(name, (weight.shape[1], weight.shape[0], None), spec)?;
            Ok(DenseWeight::Exl3(weight))
        }
        (Some(CudaWeight::Exl3 { .. }), _) | (_, WeightLayout::Exl3) => Err(EngError::Checkpoint(
            format!("mixer weights: {name} does not match the requested EXL3 layout"),
        )),
        (Some(CudaWeight::Tiled { .. }), WeightLayout::Native) => Err(EngError::Checkpoint(
            format!("mixer weights: {name} is loaded in the tiled layout, the step runs native"),
        )),
        (Some(CudaWeight::Native { .. }), WeightLayout::Tiled) => Err(EngError::Checkpoint(
            format!("mixer weights: {name} is loaded in the native layout, the step runs tiled"),
        )),
        (Some(CudaWeight::Raw { .. }), _) | (None, _) => Err(EngError::Checkpoint(format!(
            "mixer weights: {name} is not a quantized tensor"
        ))),
    }
}

#[cfg(test)]
impl<'w> LayerWeights<'w> {
    /// The device-rig seam (MATVEC-PAR S3b): a layer's weights from ready entries, exactly the
    /// shape `Self::layer` builds (the private fields stay private to production).
    pub(super) fn for_tests(
        dense: Vec<(Role, DenseWeight<'w>)>,
        raw: Vec<(Role, DeviceRange<'w>)>,
    ) -> Self {
        Self { dense, raw }
    }
}
