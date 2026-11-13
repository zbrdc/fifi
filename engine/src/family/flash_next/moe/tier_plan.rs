//! Placement of the routed experts of Qwen3.8 Flash-Next on one card: how many experts of every
//! layer stay in VRAM at a given context, which ones, and the slot table the device classifier
//! reads.
//!
//! WHAT: [`TierPlan::new`] settles the split before the first token. K/V and the indexer keys of
//! the context take VRAM first, the expert slots (the same count in every layer) take what is left,
//! and the experts outside the slots must fit the pinned host budget. [`TierPlan::hot_set`] then
//! picks the slot holders of every layer from a popularity profile and returns the slot table
//! ([`SlotTable`], one row of `-1` or a slot per layer) and the placement map
//! ([`expert_tier::Residency`](crate::expert_tier::Residency)). WHY: 24 GiB cannot hold 43.07 GiB of
//! EXL3 experts next to a long context, and the slot count shrinks as the context grows
//! (`docs/plans/flash-next-cuda.md`, memory budget).
//!
//! Reused unchanged from the Qwen3.6 tiered offload, which takes every number as an input:
//! `budget::plan_vram` (K/V first, bank the remainder, uniform slots), `budget::plan_host`,
//! `StaticProfile` (hottest experts per layer, ties to the lowest id) and `classify_reference`
//! (the oracle of `tiered_classify`). New here: the Flash-Next sizes, the cap of the slot count at the
//! expert count, the cold-expert host check, the placement map and the slot table. The expert
//! bank is a different object from Qwen3.6's Q4_K `ExpertBank`: a Flash-Next slot holds 1,881,600
//! bytes of EXL3 trellis and scales, and the bank itself (with the filler-expert rule for a missed
//! pick) is the CUDA step's job.
//!
//! The host check compares the cold experts with the page-cache share of the host budget
//! (`plan_host`); the page cache of the n-gram table's random reads competes for the same RAM and
//! is not budgeted yet.
//!
//! Pure host arithmetic: no device, no clock, no I/O.

// Staged: the Flash-Next CUDA step is the first caller.
#![cfg_attr(not(test), allow(dead_code))]

use crate::{
    EngError,
    expert_tier::{ExpertId, Placement, Residency},
    family::qwen36::tiered::{
        PolicyMode, TieredConfig,
        budget::{
            BudgetError, ExpertSize, HostBudget, HostPlan, KvPlan, VramBudget, VramPlan, plan_host,
            plan_vram,
        },
        classify::{Classified, LengthMismatch, classify_reference},
        policy::StaticProfile,
    },
};

/// The sizes of the routed experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family::flash_next) struct TierShape {
    /// Layers with routed experts (48).
    pub(in crate::family::flash_next) layers: usize,
    /// Routed experts per layer (512).
    pub(in crate::family::flash_next) experts: usize,
    /// Picks per token and layer (10).
    pub(in crate::family::flash_next) top_k: usize,
    /// Device bytes of one routed expert, identical for all of them (1,881,600 in the 3.05 pack).
    pub(in crate::family::flash_next) expert_bytes: u64,
}

impl TierShape {
    /// The shape from the header walk: every routed expert has the same size because every bank
    /// has one rate and the shapes are fixed (`PackReport::expert_bytes` is their sum).
    pub(in crate::family::flash_next) fn from_pack(
        layers: usize,
        experts: usize,
        top_k: usize,
        total_expert_bytes: u64,
    ) -> Result<Self, EngError> {
        let count = (layers as u64)
            .checked_mul(experts as u64)
            .filter(|&count| count > 0)
            .ok_or_else(|| refuse("the layer and expert counts must be nonzero".into()))?;
        if !total_expert_bytes.is_multiple_of(count) {
            return Err(refuse(format!(
                "{total_expert_bytes} expert bytes do not divide over {layers} x {experts} experts"
            )));
        }
        Ok(Self {
            layers,
            experts,
            top_k,
            expert_bytes: total_expert_bytes / count,
        })
    }

    /// Device bytes of one more slot in every layer.
    pub(in crate::family::flash_next) fn slot_cost(&self) -> u64 {
        self.layers as u64 * self.expert_bytes
    }
}

/// The indexer key bytes one position carries. Refused unless the key of `idx_dim` f32
/// values divides evenly over `ratio` positions.
pub(in crate::family::flash_next) fn indexer_key_bytes(
    idx_dim: u64,
    ratio: u64,
) -> Result<u64, EngError> {
    if ratio == 0 || !(idx_dim * 4).is_multiple_of(ratio) {
        return Err(refuse(format!(
            "the indexer key of {idx_dim} f32 values does not divide over {ratio} positions"
        )));
    }
    Ok(idx_dim * 4 / ratio)
}

/// K/V and indexer-key bytes of one context position: per QSA layer a K row and a V row of
/// `heads_kv * head_dim` values of `kv_elem_bytes`, and one f32 indexer key of `idx_dim` per `ratio`
/// positions. Int8 K/V (`kv_elem_bytes` 1) counts no scale bytes here; `KvGeometry::position_bytes`
/// (`context_budget.rs`) adds the f32 row scales and is what the step's budget uses.
/// Bytes one context position costs in the raw element size `kv_elem_bytes`. The per-layer
/// K/V payload is the shared arithmetic, mapped by element size; the indexer key and the
/// layer multiply are this family's geometry.
pub(in crate::family::flash_next) fn position_bytes(
    qsa_layers: u64,
    heads_kv: u64,
    head_dim: u64,
    kv_elem_bytes: u64,
    (idx_dim, ratio): (u64, u64),
) -> Result<u64, EngError> {
    let format = match kv_elem_bytes {
        1 => crate::family::kv_format::KvFormat::Int8 {
            group: 1,
            scale_bytes: 0,
        },
        2 => crate::family::kv_format::KvFormat::F16,
        4 => crate::family::kv_format::KvFormat::F32,
        other => {
            return Err(refuse(format!(
                "the K/V element size {other} bytes is neither 1, 2 nor 4"
            )));
        }
    };
    let kv =
        format.layer_position_bytes(crate::family::kv_format::KvRowShape { heads_kv, head_dim })?;
    indexer_key_bytes(idx_dim, ratio)?
        .checked_add(kv)
        .and_then(|per_position| per_position.checked_mul(qsa_layers))
        .ok_or_else(|| refuse("the bytes of one position overflow".into()))
}

/// What the plan reads. The VRAM and host fields are the caller's measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family::flash_next) struct TierInputs {
    pub(in crate::family::flash_next) shape: TierShape,
    pub(in crate::family::flash_next) vram: VramBudget,
    /// Context positions and the K/V plus indexer bytes of one.
    pub(in crate::family::flash_next) kv: KvPlan,
    pub(in crate::family::flash_next) host: HostBudget,
}

impl TierInputs {
    /// The inputs of a load, from what the machine reports: `free_vram` is the device's free memory
    /// measured after the context exists and before any weight is resident (the desktop and other
    /// processes are already excluded), `(mem_total, mem_available)` is `/proc/meminfo`
    /// (`capability::probe`). The device reserves are the fixed admission ledger of Qwen3.6's
    /// `enginesrv check` (`qwen36::admission_sketch`): the measured runtime overhead floor for the
    /// kernel arena and the CUDA context, and the admission margin, the minimum spare and the
    /// uncertainty band as the margin (`desktop_reserve` of `VramBudget`, whose error text still says
    /// "desktop reserve"), plus the ledger's 64 MiB chunked-attention part charge, which stands in for
    /// Flash-Next's own engine state until the step's arena plan supplies the real term. The host keeps the larger of
    /// the memory in use now (`total - available`) and Flint's 8 GiB OS floor, 2 GiB of engine
    /// heap, 1 GiB of staging and a 1 GiB margin. Prefill transients (the hyper stream, QSA scratch
    /// and expert streaming planes) are not charged yet; the step's arena plan supplies them.
    pub(in crate::family::flash_next) fn measured(
        shape: TierShape,
        (trunk_bytes, state_bytes): (u64, u64),
        kv: KvPlan,
        free_vram: u64,
        (mem_total, mem_available): (u64, u64),
    ) -> Self {
        use crate::family::qwen36::admission_sketch::{
            ADMISSION_UNCERTAINTY_BAND_BYTES, FIXED_ADMISSION_MARGIN_BYTES,
            PREFILL_PART_CHARGE_BYTES, PREFILL_SPARE_MIN_BYTES, RUNTIME_OVERHEAD_FLOOR_BYTES,
        };
        const GIB: u64 = 1 << 30;
        Self {
            shape,
            vram: VramBudget {
                usable: free_vram,
                non_expert: trunk_bytes.saturating_add(state_bytes),
                state: PREFILL_PART_CHARGE_BYTES as u64,
                arena: RUNTIME_OVERHEAD_FLOOR_BYTES,
                desktop_reserve: FIXED_ADMISSION_MARGIN_BYTES as u64
                    + PREFILL_SPARE_MIN_BYTES as u64
                    + ADMISSION_UNCERTAINTY_BAND_BYTES,
            },
            kv,
            host: HostBudget {
                total: mem_total,
                non_reclaimable: mem_total.saturating_sub(mem_available).max(8 * GIB),
                engine_heap: 2 * GIB,
                staging: GIB,
                margin: GIB,
            },
        }
    }

    /// The cost model's card: an RTX 3090 (24 GiB) with a 62 GiB host and fixed reserves of 64 MiB of
    /// engine state, 512 MiB of kernel scratch and 1400 MiB for the desktop and the allocator,
    /// chosen so the slot counts match the residency cost model (142, 180 and 208 slots at 262,144,
    /// 131,072 and 32,768 positions). A test fixture: the load uses [`Self::measured`].
    #[cfg(test)]
    pub(in crate::family::flash_next) fn cost_model_3090(
        shape: TierShape,
        (trunk_bytes, state_bytes): (u64, u64),
        kv: KvPlan,
    ) -> Self {
        const GIB: u64 = 1 << 30;
        const MIB: u64 = 1 << 20;
        Self {
            shape,
            vram: VramBudget {
                usable: 24 * GIB,
                non_expert: trunk_bytes + state_bytes,
                state: 64 * MIB,
                arena: 512 * MIB,
                desktop_reserve: 1400 * MIB,
            },
            kv,
            host: HostBudget {
                total: 62 * GIB,
                non_reclaimable: 8 * GIB,
                engine_heap: 2 * GIB,
                staging: GIB,
                margin: GIB,
            },
        }
    }
}

/// The settled split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family::flash_next) struct TierPlan {
    shape: TierShape,
    slots_per_layer: usize,
    vram: VramPlan,
    host: HostPlan,
    cold_bytes: u64,
}

fn refuse(detail: String) -> EngError {
    EngError::MemoryPlan(format!("Flash-Next expert tier: {detail}"))
}

fn budget_error(error: BudgetError) -> EngError {
    refuse(format!("{error:?}"))
}

impl TierPlan {
    /// Refuses, with the budget that lost, a context whose K/V leaves fewer than `top_k` slots per
    /// layer, a K/V that does not fit the card (it never spills to the host), and a split whose
    /// cold experts exceed the host's page-locked budget.
    pub(in crate::family::flash_next) fn new(inputs: &TierInputs) -> Result<Self, EngError> {
        let shape = inputs.shape;
        if shape.top_k == 0 || shape.top_k > shape.experts {
            return Err(refuse(format!(
                "top_k {} is outside 1..={}",
                shape.top_k, shape.experts
            )));
        }
        let size = ExpertSize {
            bytes: shape.expert_bytes,
            layers: shape.layers,
        };
        let vram =
            plan_vram(&inputs.vram, &inputs.kv, &size, shape.top_k, 0).map_err(budget_error)?;
        let slots_per_layer = vram.bank_slots_per_layer.min(shape.experts);
        let cold_bytes = (shape.experts - slots_per_layer) as u64 * shape.slot_cost();
        let host = plan_host(&inputs.host, 0, shape.expert_bytes).map_err(budget_error)?;
        if cold_bytes > host.page_cache {
            return Err(refuse(format!(
                "{} experts outside the {slots_per_layer} slots of each layer need {cold_bytes} \
                 pinned host bytes, the host budget allows {}",
                shape.experts - slots_per_layer,
                host.page_cache
            )));
        }
        Ok(Self {
            shape,
            slots_per_layer,
            vram,
            host,
            cold_bytes,
        })
    }

    /// Resident experts in every layer (at least `top_k`, at most the expert count).
    pub(in crate::family::flash_next) fn slots_per_layer(&self) -> usize {
        self.slots_per_layer
    }

    /// Device bytes of the slots of all layers.
    pub(in crate::family::flash_next) fn bank_bytes(&self) -> u64 {
        self.slots_per_layer as u64 * self.shape.slot_cost()
    }

    /// Pinned host bytes of the experts outside the slots.
    pub(in crate::family::flash_next) fn cold_bytes(&self) -> u64 {
        self.cold_bytes
    }

    /// Host bytes still free for the page-locked experts after the plan.
    pub(in crate::family::flash_next) fn host_spare_bytes(&self) -> u64 {
        self.host.page_cache - self.cold_bytes
    }

    /// Device bytes the card has left after K/V, the trunk and the slots (less than one more slot).
    pub(in crate::family::flash_next) fn vram_left(&self) -> u64 {
        self.vram.vram_left
    }

    pub(in crate::family::flash_next) fn shape(&self) -> TierShape {
        self.shape
    }

    /// The slot holders of every layer: the `slots_per_layer` experts with the highest counts
    /// (`counts[layer][expert]`), ties to the lowest id. Holders get slots `0..slots` in id order.
    pub(in crate::family::flash_next) fn hot_set(
        &self,
        counts: &[Vec<u64>],
    ) -> Result<HotSet, EngError> {
        let shape = self.shape;
        let config = TieredConfig {
            layers: shape.layers,
            experts: shape.experts,
            top_k: shape.top_k,
            bank_slots_per_layer: self.slots_per_layer,
            mode: PolicyMode::Static,
            swaps_per_token: 0.0,
            decay: 1.0,
        };
        let profile = StaticProfile::from_counts(&config, counts).map_err(refuse)?;
        let experts = u32::try_from(shape.experts)
            .map_err(|_| refuse("the expert count does not fit u32".into()))?;
        let mut residency = Residency::default();
        let mut rows = Vec::with_capacity(shape.layers);
        for layer in 0..shape.layers {
            let held = profile.held(layer);
            let mut row = vec![-1i32; shape.experts];
            for (slot, &expert) in held.iter().enumerate() {
                row[usize::from(expert)] = slot as i32;
            }
            for (expert, &slot) in row.iter().enumerate() {
                let id = ExpertId {
                    layer: layer as u32,
                    expert: expert as u32,
                };
                let placement = if slot >= 0 {
                    Placement::Gpu
                } else {
                    Placement::Cpu
                };
                residency.set_placement(id, experts, placement)?;
            }
            rows.push(row);
        }
        Ok(HotSet {
            table: SlotTable {
                slots: self.slots_per_layer,
                rows,
            },
            residency,
        })
    }
}

/// Per layer, the slot of every expert or -1: the `slot_row` the device classifier reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family::flash_next) struct SlotTable {
    slots: usize,
    rows: Vec<Vec<i32>>,
}

impl SlotTable {
    pub(in crate::family::flash_next) fn slots(&self) -> usize {
        self.slots
    }

    /// The row of `layer`, `experts` entries.
    pub(in crate::family::flash_next) fn row(&self, layer: usize) -> &[i32] {
        self.rows.get(layer).map_or(&[], Vec::as_slice)
    }

    pub(in crate::family::flash_next) fn slot_of(
        &self,
        layer: usize,
        expert: usize,
    ) -> Option<usize> {
        let slot = *self.rows.get(layer)?.get(expert)?;
        usize::try_from(slot).ok()
    }

    /// What `tiered_classify` computes for `layer` from the router's picks: the host oracle.
    pub(in crate::family::flash_next) fn classify(
        &self,
        layer: usize,
        ids: &[i32],
        weights: &[f32],
    ) -> Result<Classified, LengthMismatch> {
        classify_reference(self.row(layer), self.slots, ids, weights)
    }
}

/// The slot holders of every layer as a slot table and as a placement map: GPU for a holder, CPU
/// (page-locked host) for every other expert.
#[derive(Debug)]
pub(in crate::family::flash_next) struct HotSet {
    pub(in crate::family::flash_next) table: SlotTable,
    pub(in crate::family::flash_next) residency: Residency,
}

/// Router pick counts per layer and expert: the popularity profile of [`TierPlan::hot_set`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family::flash_next) struct RouteCounts {
    counts: Vec<Vec<u64>>,
}

impl RouteCounts {
    pub(in crate::family::flash_next) fn new(layers: usize, experts: usize) -> Self {
        Self {
            counts: vec![vec![0; experts]; layers],
        }
    }

    /// Counts the picks of one token in `layer`; refuses an id outside the layer.
    pub(in crate::family::flash_next) fn record(
        &mut self,
        layer: usize,
        picks: &[u32],
    ) -> Result<(), EngError> {
        let row = self
            .counts
            .get_mut(layer)
            .ok_or_else(|| refuse(format!("layer {layer} is outside the profile")))?;
        for &expert in picks {
            *row.get_mut(expert as usize)
                .ok_or_else(|| refuse(format!("expert {expert} is outside layer {layer}")))? += 1;
        }
        Ok(())
    }

    pub(in crate::family::flash_next) fn counts(&self) -> &[Vec<u64>] {
        &self.counts
    }
}

#[cfg(test)]
mod tests;
