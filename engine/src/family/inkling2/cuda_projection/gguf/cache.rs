//! Retain frequently selected expert pairs in the admitted VRAM left after whole layers.
use super::{GgufCuda, bad, streaming::Matrix};
use crate::{
    EngError,
    expert_tier::{CudaBankLease, ExpertId, VRAM_HEADROOM_BYTES},
    family::{
        inkling2::TextSpec,
        qwen36::tiered::{PolicyMode, TieredConfig, policy::ExpertPolicy},
    },
    kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange},
};

type Part = ([usize; 2], f32);
struct Slot {
    expert: Option<u32>,
    ready: Option<[Part; 2]>,
}

pub(super) struct LayerCache {
    policy: ExpertPolicy,
    slots: Vec<Slot>,
    arena: DeviceBuffer,
    bytes: [usize; 2],
    hits: u64,
    demand: u64,
    visits: u64,
    profile: bool,
}

impl LayerCache {
    fn new(
        ctx: &CudaContext,
        spec: &TextSpec,
        slots: usize,
        bytes: [usize; 2],
        profile: bool,
    ) -> Result<Self, EngError> {
        let cfg = TieredConfig {
            layers: 1,
            experts: spec.n_routed_experts as usize,
            top_k: spec.num_experts_per_tok as usize,
            bank_slots_per_layer: slots,
            mode: PolicyMode::Adaptive,
            swaps_per_token: 1.,
            decay: 0.97,
        };
        let policy =
            ExpertPolicy::new(&cfg, None).map_err(|e| bad(format!("GGUF expert cache: {e}")))?;
        Ok(Self {
            policy,
            slots: (0..slots)
                .map(|_| Slot {
                    expert: None,
                    ready: None,
                })
                .collect(),
            arena: ctx.allocate(slots * (bytes[0] + bytes[1]))?,
            bytes,
            hits: 0,
            demand: 0,
            visits: 0,
            profile,
        })
    }

    pub(super) fn observe(&mut self, route: &[ExpertId]) -> Result<(), EngError> {
        let ids = route.iter().map(|id| id.expert as u16).collect();
        self.policy.observe(&[ids]);
        for swap in self.policy.boundary() {
            let victim = swap.evict.map(u32::from);
            let slot = self
                .slots
                .iter_mut()
                .find(|slot| slot.expert == victim)
                .ok_or_else(|| bad("adaptive GGUF cache lost its victim"))?;
            slot.expert = Some(u32::from(swap.promote));
            slot.ready = None;
        }
        Ok(())
    }

    fn assigned(&self, expert: u32) -> Option<usize> {
        self.slots.iter().position(|s| s.expert == Some(expert))
    }

    pub(super) fn hit(&self, expert: u32) -> Option<usize> {
        self.assigned(expert)
            .filter(|&slot| self.slots[slot].ready.is_some())
    }

    fn range(&self, slot: usize, part: usize) -> Result<DeviceRange<'_>, EngError> {
        let start =
            slot * (self.bytes[0] + self.bytes[1]) + if part == 0 { 0 } else { self.bytes[0] };
        self.arena.range(start, self.bytes[part])
    }

    pub(super) fn part(&self, slot: usize, part: usize) -> Result<Matrix<'_>, EngError> {
        let (shape, scale) = self.slots[slot]
            .ready
            .ok_or_else(|| bad("unpublished cached GGUF expert"))?[part];
        Ok(Matrix {
            data: self.range(slot, part)?,
            shape,
            scale,
        })
    }

    /// Publish only after the stream has completed both device-to-device copies.
    /// A failed or partial promotion therefore remains a demand miss on retry.
    pub(super) fn finish(
        &mut self,
        ctx: &CudaContext,
        lease: CudaBankLease<'_>,
        id: ExpertId,
    ) -> Result<(), EngError> {
        let Some(slot) = self.assigned(id.expert) else {
            return lease.finish();
        };
        let copy = |index| -> Result<Part, EngError> {
            let part = lease.part(id, index)?;
            ctx.copy_device(lease.stream(), part.data, self.range(slot, index)?)?;
            Ok((part.shape.map(|d| d as usize), part.tensor_scale))
        };
        let parts = [copy(0)?, copy(1)?];
        lease.finish()?;
        self.slots[slot].ready = Some(parts);
        Ok(())
    }

    pub(super) fn record(&mut self, layer: u32, demand: usize, hits: usize) {
        self.visits += 1;
        self.demand += demand as u64;
        self.hits += hits as u64;
        if self.profile && self.visits.is_multiple_of(16) {
            eprintln!(
                "Inkling GGUF expert cache traffic {}",
                serde_json::json!({
                    "layer":layer, "visits":self.visits, "demand":self.demand, "hits":self.hits,
                    "avoided_h2d_bytes":self.hits * (self.bytes[0] + self.bytes[1]) as u64
                })
            );
        }
    }
}

fn slot_counts(layout: &[(u32, [usize; 2])], mut budget: usize, experts: usize) -> Vec<usize> {
    let mut counts = vec![0; layout.len()];
    // Distribute one slot per eligible layer per round, in repeat-core priority order.
    for _ in 0..experts {
        let before = budget;
        for (count, (_, bytes)) in counts.iter_mut().zip(layout) {
            let size = bytes[0] + bytes[1];
            if size <= budget {
                budget -= size;
                *count += 1;
            }
        }
        if before == budget {
            break;
        }
    }
    counts
}

impl GgufCuda {
    pub(in crate::family::inkling2) fn enable_cache(
        &mut self,
        layout: Vec<(u32, [usize; 2])>,
        spec: &TextSpec,
        expert_budget: u64,
    ) -> Result<(), EngError> {
        let spare = (self.context.memory_info()?.free as u64).saturating_sub(VRAM_HEADROOM_BYTES);
        let cap = spare.min(expert_budget.saturating_sub(self.resident.bytes()));
        let slots = slot_counts(&layout, cap as usize, spec.n_routed_experts as usize);
        let profile = std::env::var("FIFI_INKLING_CUDA_PROFILE").as_deref() == Ok("1");
        let cache = self.cache.get_mut();
        for ((layer, parts), n) in layout.into_iter().zip(slots).filter(|(_, n)| *n > 0) {
            let free = self.context.memory_info()?.free as u64;
            let fit = free.saturating_sub(VRAM_HEADROOM_BYTES) / (parts[0] + parts[1]) as u64;
            // The driver may round an allocation above the GGUF payload size.
            // Keep only banks whose measured allocation respects the shared floor.
            for slots in (1..=n.min(fit as usize)).rev() {
                let bank = LayerCache::new(&self.context, spec, slots, parts, profile)?;
                if self.context.memory_info()?.free as u64 >= VRAM_HEADROOM_BYTES {
                    cache.insert(layer, bank);
                    break;
                }
            }
        }
        let bytes: usize = cache.values().map(|c| c.arena.bytes()).sum();
        let count: usize = cache.values().map(|c| c.slots.len()).sum();
        eprintln!(
            "Inkling GGUF expert cache {}",
            serde_json::json!({
                "bytes":bytes,"slots":count,"layers":cache.len(),"budget_after_placement":cap,
                "free_device_bytes": self.context.memory_info()?.free
            })
        );
        Ok(())
    }

    pub(in crate::family::inkling2) fn cached(&self, layer: usize, expert: usize) -> bool {
        self.cache
            .borrow()
            .get(&(layer as u32))
            .is_some_and(|bank| bank.hit(expert as u32).is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_slots_respect_bytes_layer_priority_and_expert_count() {
        let layout = [(6, [2, 1]), (0, [4, 2])];
        assert_eq!(slot_counts(&layout, 0, 3), [0, 0]);
        assert_eq!(slot_counts(&layout, 12, 3), [2, 1]);
        assert_eq!(slot_counts(&layout, 99, 3), [3, 3]);
    }
}
