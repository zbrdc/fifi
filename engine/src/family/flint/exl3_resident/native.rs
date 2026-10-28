//! Borrowed native views. Only the loader may construct the fully initialized owner.

mod host;
mod load;
mod packing;

#[cfg(test)]
pub(crate) use load::load_flint_fixture;
pub(crate) use load::load_flint_resident;

use super::{FfnRoles, ResidentPlan};
use crate::{
    EngError,
    family::project::cuda::Exl3MatrixView,
    kernels::cuda::{DeviceBuffer, DeviceRange},
    weights::exl3::Codebook,
};

pub(crate) struct FlintExl3Resident {
    buffer: DeviceBuffer,
    host: Option<host::HostOwner>,
    matrices: Vec<PackedMatrix>,
    plain: Vec<PackedRange>,
    roles: ResidentPlan,
    model_config: Vec<u8>,
}

#[derive(Clone, Copy)]
struct PackedRange {
    offset: usize,
    bytes: usize,
}

#[derive(Clone, Copy)]
struct PackedMatrix {
    trellis: PackedRange,
    suh: PackedRange,
    svh: PackedRange,
    bias: Option<PackedRange>,
    shape: [usize; 2],
    bits_twice: u8,
    book: Codebook,
}

pub(crate) struct FlintExpertBank<'a> {
    owner: &'a FlintExl3Resident,
    indices: &'a [[usize; 3]],
    layer: usize,
    prepared: Option<&'a [Option<usize>; 256]>,
}

#[derive(Clone, Copy)]
pub(crate) struct FlintExpertView<'a> {
    pub(crate) gate: Exl3MatrixView<'a>,
    pub(crate) up: Exl3MatrixView<'a>,
    pub(crate) down: Exl3MatrixView<'a>,
}

pub(crate) enum FlintFfnView<'a> {
    Dense(FlintExpertView<'a>),
    Moe {
        router: DeviceRange<'a>,
        experts: FlintExpertBank<'a>,
        shared: FlintExpertView<'a>,
        routing_weight_multiplier: f32,
    },
}

pub(crate) struct FlintLayerView<'a> {
    pub(crate) q: Exl3MatrixView<'a>,
    pub(crate) k: Exl3MatrixView<'a>,
    pub(crate) v: Exl3MatrixView<'a>,
    pub(crate) o: Exl3MatrixView<'a>,
    pub(crate) input_norm: DeviceRange<'a>,
    pub(crate) post_attention_norm: DeviceRange<'a>,
    pub(crate) q_norm: DeviceRange<'a>,
    pub(crate) k_norm: DeviceRange<'a>,
    pub(crate) sinks: DeviceRange<'a>,
    pub(crate) ffn: FlintFfnView<'a>,
}

impl FlintExl3Resident {
    pub(crate) fn model_config(&self) -> &[u8] {
        &self.model_config
    }

    pub(crate) fn layer_count(&self) -> usize {
        self.roles.layers.len()
    }

    pub(crate) fn device_weight_bytes(&self) -> u64 {
        self.buffer.bytes() as u64
    }

    pub(crate) fn embedding(&self) -> Result<DeviceRange<'_>, EngError> {
        self.plain(self.roles.embedding)
    }

    pub(crate) fn final_norm(&self) -> Result<DeviceRange<'_>, EngError> {
        self.plain(self.roles.final_norm)
    }

    pub(crate) fn head(&self) -> Result<Exl3MatrixView<'_>, EngError> {
        self.matrix(self.roles.head)
    }

    pub(crate) fn layer(&self, layer: usize) -> Result<FlintLayerView<'_>, EngError> {
        let roles = self.roles.layers.get(layer).ok_or_else(|| {
            EngError::BadRequest(format!("Flint EXL3 layer {layer} is out of range"))
        })?;
        let ffn = match &roles.ffn {
            FfnRoles::Dense(indices) => FlintFfnView::Dense(self.expert(*indices)?),
            FfnRoles::Moe {
                router,
                experts,
                shared,
                routing_weight_multiplier,
            } => FlintFfnView::Moe {
                router: self.plain(*router)?,
                experts: FlintExpertBank {
                    owner: self,
                    indices: experts,
                    layer,
                    prepared: None,
                },
                shared: self.expert(*shared)?,
                routing_weight_multiplier: *routing_weight_multiplier,
            },
        };
        Ok(FlintLayerView {
            q: self.matrix(roles.qkvo[0])?,
            k: self.matrix(roles.qkvo[1])?,
            v: self.matrix(roles.qkvo[2])?,
            o: self.matrix(roles.qkvo[3])?,
            input_norm: self.plain(roles.norms[0])?,
            post_attention_norm: self.plain(roles.norms[1])?,
            q_norm: self.plain(roles.norms[2])?,
            k_norm: self.plain(roles.norms[3])?,
            sinks: self.plain(roles.sinks)?,
            ffn,
        })
    }

    fn range(&self, range: PackedRange) -> Result<DeviceRange<'_>, EngError> {
        self.buffer.range(range.offset, range.bytes)
    }

    fn plain(&self, index: usize) -> Result<DeviceRange<'_>, EngError> {
        let range = self.plain.get(index).ok_or_else(|| {
            EngError::BadRequest(format!(
                "Flint EXL3 plain descriptor {index} is out of range"
            ))
        })?;
        self.range(*range)
    }

    fn matrix(&self, index: usize) -> Result<Exl3MatrixView<'_>, EngError> {
        let matrix = self.matrices.get(index).ok_or_else(|| {
            EngError::BadRequest(format!(
                "Flint EXL3 matrix descriptor {index} is out of range"
            ))
        })?;
        Ok(Exl3MatrixView {
            trellis: self.range(matrix.trellis)?,
            suh: self.range(matrix.suh)?,
            svh: self.range(matrix.svh)?,
            bias: matrix.bias.map(|range| self.range(range)).transpose()?,
            shape: matrix.shape,
            bits_twice: matrix.bits_twice,
            book: matrix.book,
        })
    }

    fn expert(&self, indices: [usize; 3]) -> Result<FlintExpertView<'_>, EngError> {
        Ok(FlintExpertView {
            gate: self.matrix(indices[0])?,
            up: self.matrix(indices[1])?,
            down: self.matrix(indices[2])?,
        })
    }
}

impl FlintExpertBank<'_> {
    pub(crate) fn len(&self) -> usize {
        self.indices.len()
    }

    pub(crate) fn shape(&self, expert: usize) -> Result<(usize, usize), EngError> {
        let indices = self.indices.get(expert).ok_or_else(|| {
            EngError::BadRequest(format!("Flint EXL3 expert {expert} is out of range"))
        })?;
        let shape = |index: usize| {
            self.owner
                .matrices
                .get(index)
                .map(|matrix| matrix.shape)
                .ok_or_else(|| {
                    EngError::BadRequest(
                        "Flint CUDA FFN: expert matrix index is out of range".into(),
                    )
                })
        };
        let gate = shape(indices[0])?;
        let up = shape(indices[1])?;
        let down = shape(indices[2])?;
        let [hidden, intermediate] = gate;
        if up != gate || down != [intermediate, hidden] {
            return Err(EngError::BadRequest(
                "Flint CUDA FFN: EXL3 gate/up/down matrix shapes disagree".into(),
            ));
        }
        Ok((hidden, intermediate))
    }

    pub(crate) fn get(&self, expert: usize) -> Result<FlintExpertView<'_>, EngError> {
        let indices = self.indices.get(expert).ok_or_else(|| {
            EngError::BadRequest(format!("Flint EXL3 expert {expert} is out of range"))
        })?;
        if self.owner.host.is_some() {
            let base = self
                .prepared
                .as_ref()
                .and_then(|map| map.get(expert))
                .copied()
                .flatten()
                .ok_or_else(|| {
                    EngError::BadRequest("Flint host expert payload is not prepared".into())
                })?;
            return self.owner.prepared_expert(*indices, base);
        }
        self.owner.expert(*indices)
    }
}
