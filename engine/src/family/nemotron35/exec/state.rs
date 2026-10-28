use super::{NemotronBlockKind, NemotronExecSpec, byte_len, invalid};
use crate::{
    EngError,
    tensor::{Dtype, Tensor},
};

#[derive(Clone)]
pub(super) enum BlockState {
    Mamba {
        conv: Tensor,
        ssm: Tensor,
    },
    /// The projected K/V caches live on the stack (mutable rows, one per
    /// token); the state only tracks the geometry via the block kind.
    Attention,
    Moe,
}

/// One session's zero-initialized, independently owned recurrent and attention storage.
#[derive(Clone)]
pub struct NemotronState {
    pub(super) spec: NemotronExecSpec,
    pub(super) blocks: Vec<BlockState>,
    pub(super) position: u32,
    pub(super) capacity: u32,
    bytes: u64,
}

impl NemotronState {
    pub fn new(spec: &NemotronExecSpec, context_capacity: u32) -> Result<Self, EngError> {
        let bytes = Self::planned_bytes(spec, context_capacity)?;
        let m = spec.mamba;
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(spec.blocks.len())
            .map_err(|_| invalid("cannot allocate block states"))?;
        for kind in &spec.blocks {
            blocks.push(match kind {
                NemotronBlockKind::Mamba => BlockState::Mamba {
                    conv: zeros(&[spec.conv_dim()?, m.conv_kernel])?,
                    ssm: zeros(&[m.heads, m.head_dim, m.state_size])?,
                },
                NemotronBlockKind::Attention => BlockState::Attention,
                NemotronBlockKind::Moe => BlockState::Moe,
            });
        }
        Ok(Self {
            spec: spec.clone(),
            blocks,
            position: 0,
            capacity: context_capacity,
            bytes,
        })
    }

    /// Estimates the tensor storage allocated by `new` without allocating state tensors.
    pub fn planned_bytes(spec: &NemotronExecSpec, context_capacity: u32) -> Result<u64, EngError> {
        if context_capacity == 0 || context_capacity > spec.geometry.max_position_embeddings {
            return Err(invalid(
                "context capacity must be positive and within model context",
            ));
        }
        let m = spec.mamba;
        let g = spec.geometry;
        let mut bytes = 0u64;
        // Check the entire session extent before allocating any of its tensors.
        for kind in &spec.blocks {
            let shapes = match kind {
                // Mamba stores the full convolution window, unlike DeltaNet's kernel-1 ring.
                NemotronBlockKind::Mamba => vec![
                    vec![spec.conv_dim()?, m.conv_kernel],
                    vec![m.heads, m.head_dim, m.state_size],
                ],
                // Incremental KV: the projected key/value caches
                // (2 x [capacity, kv_width]) are allocated by the stack, not
                // the state; the session budget still accounts for them.
                NemotronBlockKind::Attention => vec![
                    vec![context_capacity, g.num_key_value_heads * g.head_dim],
                    vec![context_capacity, g.num_key_value_heads * g.head_dim],
                ],
                NemotronBlockKind::Moe => vec![],
            };
            for shape in shapes {
                bytes = bytes
                    .checked_add(byte_len(&shape)? as u64)
                    .ok_or_else(|| invalid("state bytes overflow"))?;
            }
        }
        if bytes > isize::MAX as u64 {
            return Err(invalid("session allocation exceeds host limits"));
        }
        Ok(bytes)
    }

    pub fn position(&self) -> u32 {
        self.position
    }

    /// Allocated tensor bytes: the recurrent state plus the stack-owned
    /// incremental K/V caches (accounted here although they are not state
    /// tensors). Excludes container overhead, cached weights, logits, and
    /// transient forward scratch.
    pub fn total_state_bytes(&self) -> u64 {
        self.bytes
    }

    pub fn mamba_state(&self, block: usize) -> Option<(&Tensor, &Tensor)> {
        match self.blocks.get(block)? {
            BlockState::Mamba { conv, ssm } => Some((conv, ssm)),
            _ => None,
        }
    }
}

fn zeros(shape: &[u32]) -> Result<Tensor, EngError> {
    let len = byte_len(shape)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| invalid("cannot allocate session tensor"))?;
    bytes.resize(len, 0);
    Tensor::try_from_host(Dtype::F32, shape.to_vec(), bytes)
}
