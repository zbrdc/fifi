//! Decode-only residency for exact packed expert records.
//!
//! Cache bytes live in one page-rounded host mapping. A hit copies its packed slices into the
//! ordinary rank-ordered staging set. Only the staging set is imported by MLX.

use std::{mem::size_of, ops::Range};

use memmap2::{MmapMut, MmapOptions};

use super::{
    slots::SetLayout,
    weights::{RMoe, ResidentWeights},
};
use crate::{
    EngError,
    family::stored::{StackLayout, read_stack_slots_into},
    kernels::HostPages,
    weights::StreamedCheckpoint,
};

pub(super) const ENV: &str = "FIFI_FLASH_NEXT_MLX_DECODE_CACHE_SLOTS";
const MAX_SLOTS: usize = 4;

pub(super) fn configured_slots(value: Option<&std::ffi::OsStr>) -> Result<usize, EngError> {
    value.map_or(Ok(0), |value| {
        value
            .to_str()
            .and_then(|text| text.parse().ok())
            .filter(|&slots| slots <= MAX_SLOTS)
            .ok_or_else(|| EngError::BadRequest(format!("{ENV} must be between 0 and {MAX_SLOTS}")))
    })
}

struct LayerBank {
    offset: usize,
    record_bytes: usize,
    resident: [Option<u32>; MAX_SLOTS],
    previous: Vec<u32>,
}

pub(super) struct DecodeCache {
    banks: Vec<LayerBank>,
    misses: Vec<(u32, usize)>,
    slots: usize,
    mapping_bytes: usize,
    charge_bytes: u64,
    mapping: Option<MmapMut>,
    #[cfg(feature = "native-fault-qualification")]
    pub(super) stats: CopyStats,
}

#[cfg(feature = "native-fault-qualification")]
#[derive(Clone, Copy, Default)]
pub(super) struct CopyStats {
    pub(super) hit_bytes: u64,
    pub(super) placement_bytes: u64,
    pub(super) copy_ns: u64,
}

impl DecodeCache {
    pub(super) fn plan(
        ckpt: &StreamedCheckpoint,
        weights: &ResidentWeights,
        slots: usize,
    ) -> Result<Self, EngError> {
        let top_k = weights.shape.moe.top_k();
        if slots == 0 || slots > MAX_SLOTS || slots > top_k {
            return Err(EngError::BadRequest(format!(
                "{ENV} needs between 1 and {} slots for this model",
                top_k.min(MAX_SLOTS),
            )));
        }
        let overflow = || EngError::MemoryPlan("Flash-Next decode cache size overflows".into());
        let mut banks = Vec::with_capacity(weights.layers.len());
        let mut bytes = 0usize;
        for layer in &weights.layers {
            // Each record packs the exact role slices without per-role page padding.
            let record_bytes = SetLayout::new(ckpt, &layer.moe, 1, 1)?.gates.offset;
            banks.push(LayerBank {
                offset: bytes,
                record_bytes,
                resident: [None; MAX_SLOTS],
                previous: Vec::with_capacity(top_k),
            });
            bytes = bytes
                .checked_add(record_bytes.checked_mul(slots).ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
        }
        let mapping_bytes = bytes
            .checked_next_multiple_of(HostPages::page_size())
            .ok_or_else(overflow)?;
        let misses = Vec::with_capacity(top_k);
        let metadata = banks
            .capacity()
            .checked_mul(size_of::<LayerBank>())
            .and_then(|bytes| {
                banks.iter().try_fold(bytes, |sum, bank| {
                    sum.checked_add(bank.previous.capacity().checked_mul(size_of::<u32>())?)
                })
            })
            .and_then(|bytes| {
                bytes.checked_add(misses.capacity().checked_mul(size_of::<(u32, usize)>())?)
            })
            .ok_or_else(overflow)?;
        let charge_bytes = mapping_bytes
            .checked_add(metadata)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(overflow)?;
        Ok(Self {
            banks,
            misses,
            slots,
            mapping_bytes,
            charge_bytes,
            mapping: None,
            #[cfg(feature = "native-fault-qualification")]
            stats: CopyStats::default(),
        })
    }

    pub(super) fn charge_bytes(&self) -> u64 {
        self.charge_bytes
    }

    pub(super) fn mapping_bytes(&self) -> u64 {
        self.mapping_bytes as u64
    }

    pub(super) fn slots(&self) -> usize {
        self.slots
    }

    pub(super) fn active(&self) -> bool {
        self.mapping.is_some()
    }

    pub(super) fn start(&mut self) -> Result<(), EngError> {
        if !self.active() {
            self.mapping = Some(
                MmapOptions::new()
                    .len(self.mapping_bytes)
                    .map_anon()
                    .map_err(|err| {
                        EngError::MemoryPlan(format!("map Flash-Next decode cache: {err}"))
                    })?,
            );
        }
        Ok(())
    }

    pub(super) fn release(&mut self) {
        if self.mapping.take().is_some() {
            for bank in &mut self.banks {
                bank.resident.fill(None);
                bank.previous.clear();
            }
        }
    }

    pub(super) fn stage(
        &mut self,
        layer: usize,
        ckpt: &StreamedCheckpoint,
        weight: &RMoe,
        ids: &[u32],
        stage: &mut [u8],
        layout: &SetLayout,
    ) -> Result<(), EngError> {
        let bank = &mut self.banks[layer];
        debug_assert!(ids.len() <= bank.previous.capacity());
        let resident = &mut bank.resident[..self.slots];
        let mapping = self.mapping.as_mut().expect("the decode cache is active");
        self.misses.clear();
        self.misses.extend(
            ids.iter()
                .copied()
                .enumerate()
                .filter_map(|(rank, id)| (!resident.contains(&Some(id))).then_some((id, rank))),
        );
        // Complete every demand read before copying or publishing any replacement.
        for (weight, stack) in [&weight.gate, &weight.up, &weight.down]
            .into_iter()
            .zip(&layout.stacks)
        {
            read_stack_slots_into(ckpt, weight, &self.misses, stage, stack)?;
        }

        for (rank, &id) in ids.iter().enumerate() {
            if let Some(slot) = resident
                .iter()
                .position(|id_in_slot| *id_in_slot == Some(id))
            {
                let start = bank.offset + slot * bank.record_bytes;
                #[cfg(feature = "native-fault-qualification")]
                let started = std::time::Instant::now();
                record_ranges(&layout.stacks, rank, start, |staged, cached| {
                    stage[staged].copy_from_slice(&mapping[cached]);
                });
                #[cfg(feature = "native-fault-qualification")]
                {
                    self.stats.hit_bytes += bank.record_bytes as u64;
                    self.stats.copy_ns += started.elapsed().as_nanos() as u64;
                }
            }
        }
        for &(id, rank) in &self.misses {
            if !bank.previous.contains(&id) {
                continue;
            }
            // Keep experts used in this forward. Empty slots win, then the highest ID of the
            // unused residents, giving replacements a stable order.
            let victim = resident
                .iter()
                .enumerate()
                .filter(|(_, expert)| expert.is_none_or(|e| !ids.contains(&e)))
                .max_by_key(|(_, expert)| (expert.is_none(), **expert));
            let Some((slot, _)) = victim else { break };
            let start = bank.offset + slot * bank.record_bytes;
            #[cfg(feature = "native-fault-qualification")]
            let started = std::time::Instant::now();
            record_ranges(&layout.stacks, rank, start, |staged, cached| {
                mapping[cached].copy_from_slice(&stage[staged]);
            });
            resident[slot] = Some(id);
            #[cfg(feature = "native-fault-qualification")]
            {
                self.stats.placement_bytes += bank.record_bytes as u64;
                self.stats.copy_ns += started.elapsed().as_nanos() as u64;
            }
        }
        bank.previous.clear();
        bank.previous.extend_from_slice(ids);
        Ok(())
    }
}

fn record_ranges(
    stacks: &[StackLayout; 3],
    rank: usize,
    mut offset: usize,
    mut copy: impl FnMut(Range<usize>, Range<usize>),
) {
    for stack in stacks {
        for (_, role) in stack.roles() {
            let start = role.offset + rank * role.slice_len;
            copy(
                start..start + role.slice_len,
                offset..offset + role.slice_len,
            );
            offset += role.slice_len;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::load::FlashWeights;
    use super::super::{
        fixtures::config,
        moe::tests::{checkpoint_from, routed_entries},
    };
    use super::*;
    use crate::{family::stored::read_stack_into, kernels::MlxKernels};

    #[test]
    fn repeated_experts_copy_exact_slices_at_the_selected_rank() {
        let _serial = crate::native_test_lock();
        let ckpt = checkpoint_from(routed_entries());
        let flash = FlashWeights::load(&ckpt, &config()).unwrap();
        let mut kernels = MlxKernels::try_new().unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let moe = &weights.layers[0].moe;
        let layout = SetLayout::new(&ckpt, moe, 2, HostPages::page_size()).unwrap();
        let mut cache = DecodeCache::plan(&ckpt, &weights, 1).unwrap();
        assert!(
            cache
                .mapping_bytes()
                .is_multiple_of(HostPages::page_size() as u64)
        );
        let mut stage = vec![0; layout.end];
        let mut control = vec![0; layout.end];

        cache.start().unwrap();
        for (ids, want) in [
            ([0, 1], None),
            ([2, 1], Some(1)),
            ([1, 2], Some(1)),
            ([0, 2], Some(2)),
            ([2, 3], Some(2)),
        ] {
            for (weight, stack) in [&moe.gate, &moe.up, &moe.down]
                .into_iter()
                .zip(&layout.stacks)
            {
                read_stack_into(&ckpt, weight, &ids, &mut control, stack).unwrap();
            }
            cache
                .stage(0, &ckpt, moe, &ids, &mut stage, &layout)
                .unwrap();
            assert_eq!(stage, control, "slices for {ids:?}");
            assert_eq!(cache.banks[0].resident[0], want);
        }
        #[cfg(feature = "native-fault-qualification")]
        {
            assert_eq!(
                cache.stats.hit_bytes,
                2 * cache.banks[0].record_bytes as u64
            );
            assert_eq!(
                cache.stats.placement_bytes,
                2 * cache.banks[0].record_bytes as u64
            );
        }
        cache.release();
        assert!(!cache.active());
        assert!(cache.banks[0].previous.is_empty());
        assert!(cache.banks[0].resident.iter().all(Option::is_none));
    }
}
