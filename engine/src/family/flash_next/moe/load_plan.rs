//! The memory plan of a full Flash-Next load: what the pack's own header walk and config say about
//! the sizes, joined to the context-aware expert budget.
//!
//! WHAT: [`FlashLoadPlan::new`] takes the header-walk report (`PackReport`), the model config
//! and the n-gram table length, and derives every number the load needs without reading a payload:
//! the routed-expert shape, the trunk bytes on the device (everything but the experts and the
//! embedding, which the host gathers), the DeltaNet state, the K/V geometry of the attention layers
//! and the host-resident parts (embedding, n-gram table on local NVMe). [`FlashLoadPlan::at_context`]
//! then settles a served context with [`ContextBudget`]: the expert slots per layer that fit next to
//! its K/V, the experts that stay in pinned host RAM, and the line for the load log.
//! WHY: the same plan runs on the sparse pack the tests use (its headers are complete) and on the
//! full 79 GiB pack, so the step's load reads numbers from the pack instead of constants.
//!
//! Nothing here opens a file, allocates device memory or changes how any model loads.
//! Pure host arithmetic.

// Staged: the Flash-Next CUDA loader is the first caller.
#![cfg_attr(not(test), allow(dead_code))]

use std::fmt;

use super::{
    context_budget::{ContextBudget, ContextPlan, KvFormat, KvGeometry},
    tier_plan::{TierInputs, TierShape},
};
use crate::{
    EngError,
    family::exl3_pack::PackReport,
    family::flash_next::{deltanet::DeltaState, load::model_shape},
    family::qwen36::tiered::budget::KvPlan,
};

const GIB: f64 = (1u64 << 30) as f64;

fn refuse(detail: String) -> EngError {
    EngError::MemoryPlan(format!("Flash-Next load plan: {detail}"))
}

/// The sizes of one pack and the context-independent part of its memory plan.
#[derive(Debug, Clone, Copy)]
pub(in crate::family::flash_next) struct FlashLoadPlan {
    pub(in crate::family::flash_next) shape: TierShape,
    /// Device bytes of the trunk: attention, DeltaNet, router, shared experts, hyper-connections
    /// and the head (the header walk's total less the routed experts and the embedding).
    pub(in crate::family::flash_next) trunk_bytes: u64,
    /// The token embedding, gathered on the host.
    pub(super) embedding_bytes: u64,
    /// The recurrent state and conv ring of every DeltaNet layer.
    pub(in crate::family::flash_next) delta_state_bytes: u64,
    /// The n-gram table, read from local NVMe.
    pub(super) ngram_bytes: u64,
    budget: ContextBudget,
}

impl FlashLoadPlan {
    /// `card` carries the VRAM and host numbers (`TierInputs::measured` from the machine; the tests use
    /// the cost-model fixture); its `shape`,
    /// `vram.non_expert` and `kv` fields are replaced by what the pack says.
    pub(in crate::family::flash_next) fn new(
        report: &PackReport,
        model_config: &[u8],
        ngram_bytes: u64,
        card: TierInputs,
    ) -> Result<Self, EngError> {
        let (model, kinds) = model_shape(model_config)?;
        let moe = model.moe;
        let shape = TierShape::from_pack(kinds.len(), moe.experts, moe.top_k, report.expert_bytes)?;
        // Two bytes an element: the real pack stores the embedding in BF16. The walk accepts any half
        // dtype or F32; an F32 table would make this figure, and so the trunk's, too small by half.
        let embedding_bytes = (model.vocab as u64)
            .checked_mul(moe.hidden as u64 * 2)
            .ok_or_else(|| refuse("the embedding size overflows".into()))?;
        let trunk_bytes = report
            .device_bytes
            .checked_sub(report.expert_bytes)
            .and_then(|rest| rest.checked_sub(embedding_bytes))
            .ok_or_else(|| {
                refuse("the experts and the embedding exceed the pack's device bytes".into())
            })?;
        let cached_layers = kinds.iter().filter(|&&attention| attention).count() as u64;
        let delta_layers = kinds.len() as u64 - cached_layers;
        let delta_state_bytes = delta_layers
            .checked_mul(DeltaState::new(&model.delta).byte_len() as u64)
            .ok_or_else(|| refuse("the DeltaNet state bytes overflow".into()))?;
        let dims = model.qsa.dims();
        let geometry = KvGeometry {
            cached_layers,
            heads_kv: u64::from(dims.heads_kv),
            head_dim: u64::from(dims.head_dim),
            idx_dim: u64::from(dims.idx_dim),
            ratio: u64::from(dims.ratio),
        };
        let mut base = card;
        base.shape = shape;
        base.vram.non_expert = trunk_bytes
            .checked_add(delta_state_bytes)
            .ok_or_else(|| refuse("the trunk and state bytes overflow".into()))?;
        base.kv = KvPlan {
            positions: 1,
            bytes_per_position: 1,
        };
        Ok(Self {
            shape,
            trunk_bytes,
            embedding_bytes,
            delta_state_bytes,
            ngram_bytes,
            budget: ContextBudget::new(base, geometry),
        })
    }

    pub(super) fn budget(&self) -> &ContextBudget {
        &self.budget
    }

    /// The plan for `requested` positions, or the refusal that names the longest context with
    /// `min_slots`.
    pub(in crate::family::flash_next) fn at_context(
        &self,
        requested: u64,
        format: KvFormat,
        min_slots: usize,
    ) -> Result<LoadPlan, EngError> {
        let context = self.budget.resolve(requested, format, min_slots)?;
        Ok(LoadPlan {
            context,
            trunk_bytes: self.trunk_bytes,
            embedding_bytes: self.embedding_bytes,
            ngram_bytes: self.ngram_bytes,
            kv_bytes: requested
                .checked_mul(self.budget.geometry().position_bytes(format)?)
                .ok_or_else(|| refuse("the K/V bytes overflow".into()))?,
        })
    }
}

/// A settled load: the context, its expert placement and the bytes of every part.
#[derive(Debug, Clone, Copy)]
pub(in crate::family::flash_next) struct LoadPlan {
    pub(in crate::family::flash_next) context: ContextPlan,
    pub(in crate::family::flash_next) trunk_bytes: u64,
    pub(in crate::family::flash_next) embedding_bytes: u64,
    pub(in crate::family::flash_next) ngram_bytes: u64,
    pub(in crate::family::flash_next) kv_bytes: u64,
}

impl fmt::Display for LoadPlan {
    /// The line for the load log.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let plan = &self.context.plan;
        write!(
            f,
            "Flash-Next load: context {} positions, K/V {:?} {:.2} GiB; trunk {:.2} GiB on the device; \
             {} of {} experts per layer in VRAM ({:.2} GiB), {:.2} GiB pinned in host RAM; \
             embedding {:.2} GiB host; n-gram table {:.2} GiB local NVMe",
            self.context.positions,
            self.context.format,
            self.kv_bytes as f64 / GIB,
            self.trunk_bytes as f64 / GIB,
            plan.slots_per_layer(),
            plan.shape().experts,
            plan.bank_bytes() as f64 / GIB,
            plan.cold_bytes() as f64 / GIB,
            self.embedding_bytes as f64 / GIB,
            self.ngram_bytes as f64 / GIB,
        )
    }
}

#[cfg(test)]
mod tests;
