//! Per-layer packed expert banks for one-row disk streaming.

use super::{
    config::LAYERS,
    mlx_forward::grouped_experts,
    mlx_model::SwiGlu,
    weights::{ExpertPageLayout, FlintCheckpoint},
};
use crate::{
    EngError,
    kernels::{HostPages, MlxKernels, PageRange},
    tensor::{Dtype, ExecutionTensor},
};

const CACHE_MIB: &str = "FIFI_FLINT_MLX_EXPERT_CACHE_MIB";
const ROUTED_LAYERS: usize = LAYERS - 2;
const ROUTES: usize = 6;
const EXPERTS: usize = 256;
const INDEX_BYTES: usize = ROUTES * size_of::<u32>();

type Result<T> = std::result::Result<T, EngError>;

#[derive(Clone, Copy)]
struct Entry {
    expert: u32,
    used: u64,
}

pub(super) struct LayerExpertCache {
    layout: ExpertPageLayout,
    pages: HostPages,
    slots: Vec<Option<Entry>>,
    tick: u64,
}

pub(super) fn plan(
    source: &FlintCheckpoint,
    available: u64,
) -> Result<(Vec<Option<LayerExpertCache>>, u64)> {
    let empty = || (0..LAYERS).map(|_| None).collect();
    let mib = match std::env::var(CACHE_MIB) {
        Ok(value) => value.parse::<u64>().ok(),
        Err(std::env::VarError::NotPresent) => Some(0),
        Err(std::env::VarError::NotUnicode(_)) => None,
    }
    .ok_or_else(|| EngError::MemoryPlan(format!("{CACHE_MIB} must be an integer MiB")))?;
    if mib == 0 {
        return Ok((empty(), 0));
    }
    let cap = mib
        .checked_mul(1 << 20)
        .ok_or_else(|| EngError::MemoryPlan(format!("{CACHE_MIB} overflows bytes")))?;
    let per_expert = source.bytes_per_routed_expert();
    // Each Flint expert role is page aligned on Apple Silicon. Reserve an extra
    // page per layer for its local route IDs, then check the resulting layouts.
    let index_reserve = (ROUTED_LAYERS * HostPages::page_size()) as u64;
    let slots = (cap.min(available).saturating_sub(index_reserve) / per_expert)
        .min((ROUTED_LAYERS * EXPERTS) as u64) as usize;
    if slots < ROUTED_LAYERS * ROUTES {
        return Err(EngError::MemoryPlan(format!(
            "Flint expert cache needs {} bytes for {ROUTES} slots per routed layer, with {} bytes allowed",
            per_expert * (ROUTED_LAYERS * ROUTES) as u64 + index_reserve,
            cap.min(available),
        )));
    }

    let mut layers = empty();
    let mut bytes = 0;
    for (index, layer) in layers.iter_mut().enumerate().skip(2) {
        let count = slots / ROUTED_LAYERS + usize::from(index - 2 < slots % ROUTED_LAYERS);
        let cache = LayerExpertCache::new(
            source,
            &format!("model.layers.{index}.mlp.switch_mlp"),
            count,
        )?;
        bytes += cache.pages.len() as u64;
        *layer = Some(cache);
    }
    if bytes > cap.min(available) {
        return Err(EngError::MemoryPlan(format!(
            "Flint expert cache needs {bytes} bytes with {} bytes allowed",
            cap.min(available)
        )));
    }
    eprintln!("flint: MLX expert cache bytes={bytes} slots={slots}");
    Ok((layers, bytes))
}

fn route_slots(
    slots: &mut [Option<Entry>],
    experts: &[u32; ROUTES],
    tick: u64,
) -> Result<([u32; ROUTES], Vec<(u32, usize)>)> {
    let mut local = [0; ROUTES];
    let mut writes = Vec::with_capacity(ROUTES);
    for (rank, &expert) in experts.iter().enumerate() {
        let slot = match slots
            .iter()
            .position(|entry| entry.is_some_and(|e| e.expert == expert))
        {
            Some(slot) => slot,
            None => {
                // Protect the entire route, including hits later in rank order.
                let (slot, _) = slots
                    .iter()
                    .enumerate()
                    .filter(|(_, entry)| entry.is_none_or(|e| !experts.contains(&e.expert)))
                    .min_by_key(|(_, entry)| entry.map_or(0, |e| e.used))
                    .ok_or_else(|| {
                        EngError::BadRequest("Flint expert cache route has no free slot".into())
                    })?;
                writes.push((expert, slot));
                slot
            }
        };
        slots[slot] = Some(Entry { expert, used: tick });
        local[rank] = slot as u32;
    }
    Ok((local, writes))
}

impl LayerExpertCache {
    pub(super) fn new(source: &FlintCheckpoint, stem: &str, count: usize) -> Result<Self> {
        let layout = source.expert_page_layout(stem, count as u32)?;
        let pages = HostPages::try_new(layout.index_offset + INDEX_BYTES)?;
        Ok(Self {
            layout,
            pages,
            slots: vec![None; count],
            tick: 0,
        })
    }

    pub(super) fn evaluate(
        &mut self,
        k: &mut MlxKernels,
        source: &FlintCheckpoint,
        x: &ExecutionTensor,
        experts: &[u32; ROUTES],
    ) -> Result<ExecutionTensor> {
        let dst = self.pages.try_write().ok_or_else(|| EngError::Backend {
            operation: "Flint expert cache",
            message: "an earlier stage still holds expert cache pages".into(),
        })?;
        self.tick += 1;
        let mut next = self.slots.clone();
        let (local, writes) = route_slots(&mut next, experts, self.tick)?;
        // A failed read can overwrite part of a victim. Publish replacements
        // only after all three projections have been read.
        for &(_, slot) in &writes {
            self.slots[slot] = None;
        }
        source.read_expert_slots(&self.layout, &writes, dst)?;
        self.slots = next;

        let offset = self.layout.index_offset;
        for (dst, id) in dst[offset..offset + INDEX_BYTES]
            .chunks_exact_mut(4)
            .zip(local)
        {
            dst.copy_from_slice(&id.to_le_bytes());
        }
        let indices = PageRange {
            offset,
            dtype: Dtype::U32,
            shape: vec![ROUTES as u32],
        };
        let result = (|| {
            let (weights, ids) =
                SwiGlu::import_expert_pages(k, &self.pages, &self.layout.packed, indices)?;
            let local = k.bound_indices(&ids, self.slots.len() as u32)?;
            let rows = grouped_experts(k, &weights, x, &local)?;
            k.evaluate(&[&rows])?;
            Ok(rows)
        })();
        // The graph has completed, and only the evaluated output can escape.
        self.pages.release();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_protects_later_hits_and_evicts_oldest_unselected() {
        let mut slots = vec![None; 7];
        let (_, writes) = route_slots(&mut slots, &[10, 11, 12, 13, 14, 15], 1).unwrap();
        assert_eq!(writes.len(), 6);
        let (local, writes) = route_slots(&mut slots, &[16, 11, 12, 13, 14, 15], 2).unwrap();
        assert_eq!(local, [6, 1, 2, 3, 4, 5]);
        assert_eq!(writes, [(16, 6)]);
        let (local, writes) = route_slots(&mut slots, &[17, 11, 12, 13, 14, 15], 3).unwrap();
        assert_eq!(local, [0, 1, 2, 3, 4, 5]);
        assert_eq!(writes, [(17, 0)]);
    }
}
