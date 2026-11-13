//! The expert budget as a function of the served context: what the Flash-Next step asks at load
//! ("serve `N` positions with this K/V format") and what it gets back (how many experts of every
//! layer stay on the card, or a refusal that names the longest context that would work).
//!
//! WHAT: [`ContextBudget`] holds everything of the plan that does not depend on the context (the
//! model's expert sizes, the card, the host, the trunk) and the K/V geometry of the attention
//! layers. [`ContextBudget::resolve`] turns a requested context into a [`TierPlan`]; the plan is the
//! existing admission budget (`plan_vram`: K/V first, the expert bank takes the remainder), so the
//! hot set shrinks as the context grows and nothing here adds a second budget.
//! [`ContextBudget::largest_context`] searches the longest context that keeps a slot floor, and
//! [`ContextBudget::curve`] lists the plan at several contexts. WHY: the memory budget in
//! `docs/plans/flash-next-cuda.md` puts the VRAM rows at 262,144 tokens at 23.1 to 25.1 GiB with a
//! 13 to 15 GiB hot set, so a fixed hot set does not fit a 128K to 256K context on 24 GiB.
//!
//! Onboarding another model with a hybrid attention stack and routed experts means filling
//! [`KvGeometry`] and a `TierShape` from its config and header walk. Nothing in this module names a
//! Flash-Next dimension.
//!
//! K/V formats are parameters of the plan, not decisions: [`KvFormat::F16`] is what the QSA
//! kernels read today, [`KvFormat::Int8`] is planning arithmetic that [`ContextBudget::resolve`]
//! refuses until a kernel reads it.
//!
//! Pure host arithmetic: no device, no clock, no I/O.

// Staged: the Flash-Next CUDA step is the first caller.
#![cfg_attr(not(test), allow(dead_code))]

use super::tier_plan::{TierInputs, TierPlan, indexer_key_bytes};
use crate::{
    EngError,
    family::qwen36::{
        admission_sketch::{CONTEXT_SEARCH_STEP, largest_fitting_context},
        tiered::budget::KvPlan,
    },
};

/// Context positions are searched and reported in multiples of this, the step of the context
/// proposal of `enginesrv check`.
pub(super) const CONTEXT_GRANULE: u64 = CONTEXT_SEARCH_STEP as u64;

fn refuse(detail: String) -> EngError {
    EngError::MemoryPlan(format!("Flash-Next context budget: {detail}"))
}

/// How the K/V of the attention layers is stored on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family::flash_next) enum KvFormat {
    /// Two-byte elements: the format of the QSA kernels (`qsa_attn` reads an fp16 cache).
    F16,
    /// One-byte elements and one f32 scale per cached K or V row of a key-value head.
    Int8,
}

impl KvFormat {
    pub(in crate::family::flash_next) fn element_bytes(self) -> u64 {
        match self {
            Self::F16 => 2,
            Self::Int8 => 1,
        }
    }

    /// Scale bytes of one cached row (a K row or a V row of one key-value head).
    fn row_scale_bytes(self) -> u64 {
        match self {
            Self::F16 => 0,
            Self::Int8 => 4,
        }
    }

    /// True once a QSA kernel reads this format.
    pub(in crate::family::flash_next) fn has_kernels(self) -> bool {
        matches!(self, Self::F16)
    }
}

/// The attention layers' share of the per-position memory: K and V rows of the key-value heads, and
/// the indexer's pooled keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family::flash_next) struct KvGeometry {
    /// Layers that cache K/V and indexer keys (12 of 48 in Flash-Next).
    pub(in crate::family::flash_next) cached_layers: u64,
    pub(in crate::family::flash_next) heads_kv: u64,
    pub(in crate::family::flash_next) head_dim: u64,
    /// Width of one f32 pooled indexer key.
    pub(in crate::family::flash_next) idx_dim: u64,
    /// Positions pooled into one indexer key.
    pub(in crate::family::flash_next) ratio: u64,
}

impl KvGeometry {
    /// Bytes one context position costs in `format`. The per-layer K/V term (payload plus
    /// scales) is the shared arithmetic; the indexer key and the layer multiply are this
    /// family's geometry.
    pub(in crate::family::flash_next) fn position_bytes(
        &self,
        format: KvFormat,
    ) -> Result<u64, EngError> {
        let kv = shared_kv_format(format, self.head_dim)?.layer_position_bytes(
            crate::family::kv_format::KvRowShape {
                heads_kv: self.heads_kv,
                head_dim: self.head_dim,
            },
        )?;
        self.cached_layers
            .checked_mul(
                kv.checked_add(indexer_key_bytes(self.idx_dim, self.ratio)?)
                    .ok_or_else(|| refuse("the bytes of one position overflow".into()))?,
            )
            .ok_or_else(|| refuse("the bytes of one position overflow".into()))
    }
}

/// This family's [`KvFormat`] in the shared type: F16 direct; Int8 keeps its one f32 scale
/// per head per plane, which the shared row states as one group the width of the head
/// dimension.
fn shared_kv_format(
    format: KvFormat,
    head_dim: u64,
) -> Result<crate::family::kv_format::KvFormat, EngError> {
    let group = u32::try_from(head_dim)
        .map_err(|_| refuse("the K/V head dimension does not fit the shared format".into()))?;
    Ok(match format {
        KvFormat::F16 => crate::family::kv_format::KvFormat::F16,
        KvFormat::Int8 => crate::family::kv_format::KvFormat::Int8 {
            group,
            scale_bytes: 4,
        },
    })
}

/// A context the card can serve, with the expert placement it leaves room for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family::flash_next) struct ContextPlan {
    pub(in crate::family::flash_next) positions: u64,
    pub(in crate::family::flash_next) format: KvFormat,
    pub(in crate::family::flash_next) plan: TierPlan,
}

/// The plan's inputs with the context left open.
#[derive(Debug, Clone, Copy)]
pub(super) struct ContextBudget {
    base: TierInputs,
    geometry: KvGeometry,
}

impl ContextBudget {
    /// `base.kv` is ignored: the context decides it.
    pub(super) fn new(base: TierInputs, geometry: KvGeometry) -> Self {
        Self { base, geometry }
    }

    pub(super) fn geometry(&self) -> KvGeometry {
        self.geometry
    }

    /// The plan at `positions`, whatever the slot count: refuses only what [`TierPlan::new`]
    /// refuses (K/V that does not fit, fewer than `top_k` slots, cold experts beyond the host).
    pub(super) fn plan_at(&self, positions: u64, format: KvFormat) -> Result<TierPlan, EngError> {
        if positions == 0 {
            return Err(refuse("a context of zero positions".into()));
        }
        let mut inputs = self.base;
        inputs.kv = KvPlan {
            positions,
            bytes_per_position: self.geometry.position_bytes(format)?,
        };
        TierPlan::new(&inputs)
    }

    fn meets(&self, positions: u64, format: KvFormat, min_slots: usize) -> bool {
        self.plan_at(positions, format)
            .is_ok_and(|plan| plan.slots_per_layer() >= min_slots)
    }

    /// The longest context, a multiple of [`CONTEXT_GRANULE`] and at most `limit`, whose plan has at
    /// least `min_slots` experts per layer. `None` when not even one granule does. A longer context
    /// never has more slots, so the search is the bisection `enginesrv check` uses for its context
    /// proposal ([`largest_fitting_context`]); `limit` itself is tried first, because that search
    /// never asks for its upper end.
    pub(super) fn largest_context(
        &self,
        format: KvFormat,
        min_slots: usize,
        limit: u64,
    ) -> Option<u64> {
        let top = u32::try_from(limit / CONTEXT_GRANULE * CONTEXT_GRANULE).ok()?;
        if top < CONTEXT_SEARCH_STEP {
            return None;
        }
        let fits = |positions: u32| self.meets(u64::from(positions), format, min_slots);
        if fits(top) {
            return Some(u64::from(top));
        }
        largest_fitting_context(top, &fits).map(u64::from)
    }

    /// What the step calls at load: the plan for `requested` positions, or a refusal that states
    /// the slots the request leaves and the longest context that keeps `min_slots`.
    pub(super) fn resolve(
        &self,
        requested: u64,
        format: KvFormat,
        min_slots: usize,
    ) -> Result<ContextPlan, EngError> {
        if !format.has_kernels() {
            return Err(refuse(format!(
                "{format:?} K/V has no attention kernel yet; plan it with plan_at"
            )));
        }
        let shortfall = |detail: String| {
            let longest = match self.largest_context(format, min_slots, requested) {
                Some(positions) => {
                    format!("the longest context with {min_slots} slots is {positions}")
                }
                None => format!("no context keeps {min_slots} slots"),
            };
            refuse(format!("{requested} positions {detail}; {longest}"))
        };
        let plan = self
            .plan_at(requested, format)
            .map_err(|error| shortfall(format!("are refused ({})", error.describe())))?;
        if plan.slots_per_layer() < min_slots {
            return Err(shortfall(format!(
                "leave {} slots per layer",
                plan.slots_per_layer()
            )));
        }
        Ok(ContextPlan {
            positions: requested,
            format,
            plan,
        })
    }

    /// The plan at each of `contexts`, in order.
    pub(super) fn curve(
        &self,
        format: KvFormat,
        contexts: &[u64],
    ) -> Vec<(u64, Result<TierPlan, EngError>)> {
        contexts
            .iter()
            .map(|&positions| (positions, self.plan_at(positions, format)))
            .collect()
    }
}

#[cfg(test)]
mod kv_bytes_golden_tests;
#[cfg(test)]
mod tests;
