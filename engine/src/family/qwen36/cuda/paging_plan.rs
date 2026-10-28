//! Checked host-paging sizes and device-first residency planning.

use std::mem::size_of;

use super::{dims::StepDims, state_layout::KvPool};
use crate::{
    EngError,
    family::AttentionKind,
    kernels::cuda::resident::{attn_decode::AttnSplit, attn_decode_q8::MAX_MIN_KEYS},
};

const PAGE_POSITIONS: usize = 256;

#[derive(Clone)]
pub(super) struct PagingPlan {
    pub(super) capacity: usize,
    pub(super) full_layers: usize,
    pub(super) kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) group_cells: usize,
    pub(super) groups: usize,
    pub(super) group: KvPool,
    pub(super) slot_bytes: usize,
    pub(super) backing_bytes: usize,
    pub(super) device_bytes: usize,
    pub(super) pinned_bytes: usize,
}

/// Complete KV groups that fit after fixed resources and VRAM headroom.
pub(super) struct DeviceResidencyPlan {
    pub(super) device_groups: usize,
    pub(super) device_rows: usize,
    pub(super) device_bytes: usize,
    pub(super) host_payload_bytes: usize,
}

pub(super) fn sized(bytes: Option<usize>) -> Result<usize, EngError> {
    bytes.ok_or_else(|| EngError::MemoryPlan("paged KV allocation size overflows".to_owned()))
}

impl PagingPlan {
    /// The cache-only budget excludes fixed resources and the existing VRAM headroom.
    pub(super) fn device_residency(
        &self,
        available_bytes: usize,
    ) -> Result<DeviceResidencyPlan, EngError> {
        let group_bytes = sized(self.group.total.checked_mul(self.full_layers))?;
        let device_groups = sized(available_bytes.checked_div(group_bytes))?.min(self.groups);
        let device_bytes = sized(device_groups.checked_mul(group_bytes))?;
        Ok(DeviceResidencyPlan {
            device_groups,
            device_rows: sized(device_groups.checked_mul(self.group_cells))?,
            device_bytes,
            host_payload_bytes: sized(self.backing_bytes.checked_sub(device_bytes))?,
        })
    }

    pub(super) fn new(dims: &StepDims, capacity: usize) -> Result<Self, EngError> {
        if capacity == 0 {
            return Err(EngError::BadRequest(
                "paged KV needs a positive context capacity".to_owned(),
            ));
        }
        let layers = dims.layers_of(AttentionKind::Full);
        if layers == 0 {
            return Err(EngError::Checkpoint(
                "paged KV needs a full-attention layer".to_owned(),
            ));
        }
        let group_cells = sized(
            capacity
                .div_ceil(AttnSplit::Q8.max_splits)
                .checked_next_multiple_of(PAGE_POSITIONS),
        )?;
        if group_cells > MAX_MIN_KEYS {
            return Err(EngError::BadRequest(
                "paged KV transfer group exceeds the CUDA attention bound".to_owned(),
            ));
        }
        let group = KvPool::q8_staging_layer(dims, group_cells)?;
        let groups = capacity.div_ceil(group_cells);
        let slot_bytes = sized(group.total.checked_add(size_of::<i32>()))?;
        let backing_bytes = sized(
            group
                .total
                .checked_mul(groups)
                .and_then(|n| n.checked_mul(layers)),
        )?;
        let device_bytes = sized(slot_bytes.checked_mul(2))?;
        Ok(Self {
            capacity,
            full_layers: layers,
            kv_heads: dims.shape.attn.heads_kv,
            head_dim: dims.shape.attn.head_dim,
            group_cells,
            groups,
            group,
            slot_bytes,
            backing_bytes,
            device_bytes,
            pinned_bytes: device_bytes,
        })
    }
}
