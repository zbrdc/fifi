//! Which routed experts are device-resident, and how a layer's picks split.

use std::collections::BTreeMap;

use super::ExpertId;
use crate::EngError;

/// The persistent home of an expert, not a temporary streaming bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Device-resident weights.
    Gpu,
    /// Host-resident weights computed by the CPU tier.
    Cpu,
    /// The read-only mapped tail, fetched on demand.
    Nvme,
}

/// How the picks of one layer split between the device and the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coverage {
    /// Every pick is device-resident (also the answer for an empty selection).
    Device,
    /// No pick is device-resident: the whole layer call belongs to the CPU kernel.
    Host,
    /// Some picks are resident and some are cold. A per-expert hot set produces this.
    Mixed,
}

/// The single persistent placement map, keyed by layer and expert.
///
/// Legacy dispatch uses [`Self::coverage`] for GPU versus host; streaming uses
/// [`Self::placement`] to leave CPU-owned experts in place. Temporary GPU banks do
/// not change an expert's persistent assignment.
#[derive(Debug, Default)]
pub struct Residency {
    layers: BTreeMap<u32, Vec<Placement>>,
}

impl Residency {
    /// Classifies a source-validated id; unassigned experts belong to the mapped tail.
    pub fn placement(&self, id: ExpertId) -> Placement {
        self.layers
            .get(&id.layer)
            .and_then(|placements| placements.get(id.expert as usize))
            .copied()
            .unwrap_or(Placement::Nvme)
    }

    /// Records placement after the caller has moved the weights and released the old tier.
    pub fn set_placement(
        &mut self,
        id: ExpertId,
        experts: u32,
        placement: Placement,
    ) -> Result<(), EngError> {
        if experts == 0 || id.expert >= experts {
            return Err(EngError::BadRequest(format!(
                "expert {} is outside layer {} count {experts}",
                id.expert, id.layer
            )));
        }
        let placements = self
            .layers
            .entry(id.layer)
            .or_insert_with(|| vec![Placement::Nvme; experts as usize]);
        if placements.len() != experts as usize {
            return Err(EngError::BadRequest(format!(
                "layer {} has {} experts, not {experts}",
                id.layer,
                placements.len()
            )));
        }
        placements[id.expert as usize] = placement;
        Ok(())
    }

    pub fn mark_expert(&mut self, layer: u32, expert: u32, experts: u32) {
        let flags = self
            .layers
            .entry(layer)
            .or_insert_with(|| vec![Placement::Nvme; experts as usize]);
        if let Some(flag) = flags.get_mut(expert as usize) {
            *flag = Placement::Gpu;
        }
    }

    /// Splits a layer's picks (expert ids) into resident and cold ones.
    pub fn coverage(&self, layer: u32, picks: &[u32]) -> Coverage {
        let resident = |expert: u32| self.placement(ExpertId { layer, expert }) == Placement::Gpu;
        let hot = picks.iter().filter(|&&expert| resident(expert)).count();
        match hot {
            _ if hot == picks.len() => Coverage::Device,
            0 => Coverage::Host,
            _ => Coverage::Mixed,
        }
    }

    /// Whether any expert of the layer is cold, so the CPU kernel needs the layer mapped.
    pub fn has_cold(&self, layer: u32) -> bool {
        self.layers
            .get(&layer)
            .is_none_or(|flags| flags.iter().any(|&placement| placement != Placement::Gpu))
    }

    /// The layers whose every expert is resident.
    pub fn full_layers(&self) -> Vec<u32> {
        self.layers
            .iter()
            .filter(|(_, flags)| flags.iter().all(|&placement| placement == Placement::Gpu))
            .map(|(layer, _)| *layer)
            .collect()
    }

    pub fn resident_experts(&self) -> usize {
        self.layers
            .values()
            .map(|flags| {
                flags
                    .iter()
                    .filter(|&&placement| placement == Placement::Gpu)
                    .count()
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EngError;
    use crate::expert_tier::{ExpertId, Placement};

    #[test]
    fn placement_distinguishes_cpu_tail_and_gpu() {
        let mut residency = Residency::default();
        let id = ExpertId {
            layer: 7,
            expert: 1,
        };
        assert_eq!(residency.placement(id), Placement::Nvme);
        residency.set_placement(id, 4, Placement::Cpu).unwrap();
        assert_eq!(residency.placement(id), Placement::Cpu);
        assert_eq!(residency.coverage(7, &[1]), Coverage::Host);
        residency.mark_expert(7, 2, 4);
        assert_eq!(residency.coverage(7, &[1, 2]), Coverage::Mixed);
        residency.set_placement(id, 4, Placement::Gpu).unwrap();
        assert_eq!(residency.resident_experts(), 2);
        assert_eq!(residency.coverage(7, &[1, 2]), Coverage::Device);
        residency.set_placement(id, 4, Placement::Nvme).unwrap();
        assert_eq!(residency.placement(id), Placement::Nvme);
        assert_eq!(residency.resident_experts(), 1);
    }

    #[test]
    fn invalid_placement_preserves_the_existing_assignment() {
        let mut residency = Residency::default();
        let id = ExpertId {
            layer: 7,
            expert: 1,
        };
        residency.set_placement(id, 4, Placement::Cpu).unwrap();
        for (candidate, count) in [
            (
                ExpertId {
                    layer: 7,
                    expert: 4,
                },
                4,
            ),
            (id, 0),
            (id, 3),
        ] {
            assert!(matches!(
                residency.set_placement(candidate, count, Placement::Gpu),
                Err(EngError::BadRequest(_))
            ));
            assert_eq!(residency.placement(id), Placement::Cpu);
            assert_eq!(residency.resident_experts(), 0);
        }
    }

    #[test]
    fn residency_answers_per_expert_and_per_layer() {
        let mut residency = Residency::default();
        // Layer 4 whole, layer 7 only experts 1 and 2 (a hot set), layer 9 nothing.
        for expert in 0..4 {
            residency.mark_expert(4, expert, 4);
        }
        residency.mark_expert(7, 1, 4);
        residency.mark_expert(7, 2, 4);
        assert_eq!(residency.coverage(4, &[0, 3]), Coverage::Device);
        assert_eq!(residency.coverage(7, &[1, 2]), Coverage::Device);
        assert_eq!(residency.coverage(7, &[0, 3]), Coverage::Host);
        assert_eq!(residency.coverage(7, &[1, 3]), Coverage::Mixed);
        assert_eq!(residency.coverage(9, &[0]), Coverage::Host);
        assert_eq!(residency.coverage(9, &[]), Coverage::Device);
        assert!(!residency.has_cold(4));
        assert!(residency.has_cold(7) && residency.has_cold(9));
        assert_eq!(residency.full_layers(), vec![4]);
        assert_eq!(residency.resident_experts(), 6);
    }
}
