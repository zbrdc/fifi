//! The tiered runtime of the resident step: what the step needs, beyond the plain weights, to run
//! the tiered MoE block of `moe/tiered.rs` in every layer.
//!
//! WHAT: [`TieredRuntime`] owns the per-layer expert banks, the scratch buffers and the host
//! (`TieredHost`: policy, residency ledger, CPU pool), and borrows each layer's router and shared
//! expert weights from the loaded model ([`LayerParts`]); the routed expert stacks are NOT loaded
//! in this mode (the bank holds the hot experts). [`TieredRuntime::run_layer`] runs one layer's
//! block, [`TieredRuntime::end_token`] runs the token boundary: policy P decides the swaps and the
//! bank slots are filled with blocking writes.

use std::ops::Range;
use std::sync::Arc;

use super::mixer_weights::DenseWeight;
use super::moe::tiered::{
    LayerPlan, RouteRead, TieredBlock, TieredLayerHost, TieredScratch, run_moe_block_tiered,
};
use super::moe::{Launch, MoeBuffers, MoeDims, MoeLayerWeights, StackWeights};
use super::run_report::HostSyncs;
use crate::EngError;
use crate::family::qwen36::tiered::cpu_pool::MappedRegion;
use crate::family::qwen36::tiered::host::{BankDevice, HostError, TieredHost};
use crate::family::qwen36::tiered::locator::ExpertRanges;
use crate::family::qwen36::tiered::report::TokenReport;
use crate::kernels::cuda::{DeviceBuffer, DeviceStream, ExpertBank};

/// The weights of one layer's MoE block that stay in the loaded model: the router, the shared
/// gate vector and the shared expert's three projections.
pub(super) struct LayerParts<'w> {
    pub(super) router: &'w DeviceBuffer,
    pub(super) router_shared: &'w DeviceBuffer,
    pub(super) shared_gate: DenseWeight<'w>,
    pub(super) shared_up: DenseWeight<'w>,
    pub(super) shared_down: DenseWeight<'w>,
}

fn host_error(error: HostError) -> EngError {
    EngError::Backend {
        operation: "tiered_host",
        message: error.to_string(),
    }
}

/// The block-side view of the host: converts the plain route into the host's slices and its
/// plan and errors back, and issues the router-guess prefetch fills of the advice that
/// arrived for this layer (advice exists to be early: it publishes at the token boundary
/// and serves a later token, so issuing it here cannot change this token's result).
struct HostAdapter<'h, 'd> {
    host: &'h mut TieredHost,
    fills: &'d mut dyn BankDevice,
}

impl TieredLayerHost for HostAdapter<'_, '_> {
    fn begin_layer(&mut self, layer: usize, route: &RouteRead) -> Result<LayerPlan, EngError> {
        let plan = self
            .host
            .begin_layer(
                layer,
                &route.ids,
                &route.weights,
                &route.hidden,
                &route.resid,
            )
            .map_err(host_error)?;
        if let Some((target, advice)) = self.host.take_arrived_advice() {
            self.host
                .prefetch(self.fills, target, &advice)
                .map_err(host_error)?;
        }
        Ok(LayerPlan {
            slot_ids: plan.slot_ids,
            device_weights: plan.device_weights,
        })
    }

    fn finish_layer(&mut self, layer: usize) -> Result<Option<Vec<f32>>, EngError> {
        self.host.finish_layer(layer).map_err(host_error)
    }
}

/// The device half of a bank change: blocking writes of one expert's three slices.
pub(super) struct BankFill<'d> {
    pub(super) banks: &'d [ExpertBank],
    pub(super) stream: &'d DeviceStream,
}

impl BankDevice for BankFill<'_> {
    fn fill_slot(
        &mut self,
        layer: usize,
        slot: usize,
        ranges: &ExpertRanges,
        region: &Arc<dyn MappedRegion>,
    ) -> Result<(), HostError> {
        let device = |what: String| HostError::Device(what);
        let bank = self
            .banks
            .get(layer)
            .ok_or_else(|| device(format!("no bank for layer {layer}")))?;
        let bytes = region.bytes();
        let cut = |range: &Range<usize>| {
            bytes
                .get(range.clone())
                .ok_or_else(|| device(format!("expert range {range:?} is outside the region")))
        };
        bank.fill(
            self.stream,
            slot,
            cut(&ranges.gate)?,
            cut(&ranges.up)?,
            cut(&ranges.down)?,
        )
        .map_err(|error| device(format!("{error:?}")))
    }
}

pub(super) struct TieredRuntime<'w> {
    parts: Vec<LayerParts<'w>>,
    banks: Vec<ExpertBank>,
    scratch: TieredScratch,
    host: TieredHost,
    /// The bank depth (slots per layer) every bank was built with.
    slot_count: usize,
    /// The host waits of the token so far: the layers' and, after `end_token`, the bank fills'.
    syncs: HostSyncs,
}

impl<'w> TieredRuntime<'w> {
    pub(super) fn new(
        parts: Vec<LayerParts<'w>>,
        banks: Vec<ExpertBank>,
        scratch: TieredScratch,
        host: TieredHost,
    ) -> Result<Self, EngError> {
        if parts.len() != banks.len() {
            return Err(EngError::BadRequest(format!(
                "tiered runtime: {} layers of weights but {} banks",
                parts.len(),
                banks.len()
            )));
        }
        let slot_count = banks.first().map(ExpertBank::slots_of).unwrap_or(0);
        Ok(Self {
            parts,
            banks,
            scratch,
            host,
            slot_count,
            syncs: HostSyncs::default(),
        })
    }

    /// The weights of `layer` for any MoE launch — the per-token block today, the chunk
    /// arm of batched prefill (MR #20 slice 4) tomorrow: the router and shared parts from
    /// the loaded model, the three BANK stacks in place of the full-depth stacks.
    pub(super) fn layer_weights(&self, layer: usize) -> Result<MoeLayerWeights<'_>, EngError> {
        let missing = || EngError::BadRequest(format!("tiered runtime: layer {layer} has no bank"));
        let (parts, bank) = (
            self.parts.get(layer).ok_or_else(missing)?,
            self.banks.get(layer).ok_or_else(missing)?,
        );
        Ok(MoeLayerWeights {
            router: parts.router,
            router_shared: parts.router_shared,
            gate: StackWeights::Native(bank.gate()),
            up: StackWeights::Native(bank.up()),
            down: StackWeights::Native(bank.down()),
            shared_gate: parts.shared_gate,
            shared_up: parts.shared_up,
            shared_down: parts.shared_down,
        })
    }

    /// The bank depth every layer was built with (the tiered plan's slots per layer).
    pub(super) fn slots(&self) -> usize {
        self.banks.first().map(|_| self.slot_count).unwrap_or(0)
    }

    /// Runs the MoE block of `layer` with the host decision between the router and the experts.
    /// The weights construction stays inline: block borrow of `scratch`/`host` beside an
    /// immutable `self` is the disjoint-field shape the accessor cannot express.
    pub(super) fn run_layer(
        &mut self,
        launch: &Launch<'_>,
        layer: usize,
        bufs: &MoeBuffers<'_>,
        dims: &MoeDims,
    ) -> Result<(), EngError> {
        let missing = || EngError::BadRequest(format!("tiered runtime: layer {layer} has no bank"));
        let (parts, bank) = (
            self.parts.get(layer).ok_or_else(missing)?,
            self.banks.get(layer).ok_or_else(missing)?,
        );
        let weights = MoeLayerWeights {
            router: parts.router,
            router_shared: parts.router_shared,
            gate: StackWeights::Native(bank.gate()),
            up: StackWeights::Native(bank.up()),
            down: StackWeights::Native(bank.down()),
            shared_gate: parts.shared_gate,
            shared_up: parts.shared_up,
            shared_down: parts.shared_down,
        };
        let mut block = TieredBlock {
            weights: &weights,
            bufs,
            scratch: &mut self.scratch,
            dims,
        };
        let mut fills = BankFill {
            banks: &self.banks,
            stream: launch.stream,
        };
        let made = run_moe_block_tiered(
            launch,
            &mut HostAdapter {
                host: &mut self.host,
                fills: &mut fills,
            },
            layer,
            &mut block,
        )?;
        self.syncs.add(made);
        Ok(())
    }

    pub(super) fn preload(&mut self, stream: &DeviceStream) -> Result<usize, EngError> {
        let mut fills = BankFill {
            banks: &self.banks,
            stream,
        };
        self.host.preload_all(&mut fills).map_err(host_error)
    }

    pub(super) fn abort_token(&mut self) {
        self.host.abort_token();
        self.syncs = HostSyncs::default();
    }

    /// The host waits of the token that just ended (layers and fills), and zero for the next one.
    pub(super) fn take_syncs(&mut self) -> HostSyncs {
        std::mem::take(&mut self.syncs)
    }

    pub(super) fn end_token(&mut self, stream: &DeviceStream) -> Result<TokenReport, EngError> {
        let mut fills = BankFill {
            banks: &self.banks,
            stream,
        };
        let report = self.host.end_token(&mut fills).map_err(host_error)?;
        self.syncs.writes += report.swaps * ExpertBank::FILL_WRITES;
        Ok(report)
    }
}
