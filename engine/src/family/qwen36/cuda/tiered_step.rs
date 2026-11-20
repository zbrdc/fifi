//! The tiered runtime of the resident step: what the step needs, beyond the plain weights, to run
//! the tiered MoE block of `moe/tiered.rs` in every layer.
//!
//! WHAT: [`TieredRuntime`] owns the per-layer expert banks, the scratch buffers and the host
//! (`TieredHost`: policy, residency ledger, CPU pool), and borrows each layer's router and shared
//! expert weights from the loaded model ([`LayerParts`]); the routed expert stacks are NOT loaded
//! in this mode (the bank holds the hot experts). [`TieredRuntime::run_layer`] runs one layer's
//! block, [`TieredRuntime::end_token`] runs the token boundary: policy P decides the swaps and the
//! bank slots are filled with blocking writes.

use crate::family::exl3::launch::{Launch, MoeDims};
use std::ops::Range;
use std::sync::Arc;

use super::mixer_weights::DenseWeight;
use super::moe::tiered::{
    LayerPlan, RouteRead, TieredBlock, TieredLayerHost, TieredScratch, run_moe_block_tiered,
};
use super::moe::{MoeBuffers, MoeLayerWeights, StackWeights};
use super::run_report::HostSyncs;
use crate::EngError;
use crate::family::qwen36::tiered::cpu_pool::MappedRegion;
use crate::family::qwen36::tiered::host::{BankDevice, ChunkColdStage, HostError, TieredHost};
use crate::family::qwen36::tiered::locator::ExpertRanges;
use crate::family::qwen36::tiered::report::TokenReport;
use crate::kernels::cuda::{
    CudaContext, DeviceBuffer, DeviceEvent, DeviceRange, DeviceStream, ExpertBank,
    ExpertBankGeometry, NativeWeight, PinnedBuf,
};

/// The cold staging bank of the chunked tiered prefill: ONE zero-initialised
/// [`ExpertBank`] of `cap` slots reused by every layer — the misses of the layer's
/// hottest `cap` distinct cold experts are gathered from the checkpoint mapping into a
/// pinned ring and DMA'd in stack-major, then served by the same grouped MMQ launches
/// as the banks (numerically identical to a hit; strictly closer than the CPU partial).
/// Zero-filled at attach, so an unstaged slot a bug could still address yields a finite
/// zero row (the `0 * Inf` argument of the dummy slot).
///
/// The chunk path reassigns the whole staging every layer (the channels travel in the
/// plan's `cold_ids`); a decode-side reuse would add an expert→slot directory on top.
pub(super) struct ColdStaging {
    bank: ExpertBank,
    cap: usize,
    ring: Vec<RingSlot>,
}

/// One slot of the pinned staging ring: a slice-sized page-locked buffer, the event
/// recorded behind its last DMA and whether a transfer is still outstanding. The host
/// gather (mmap → pinned) runs ahead of the device inside the ring, so the ~9 ms/layer
/// of memcpy hides under the ~12 ms/layer of DMA.
struct RingSlot {
    buf: PinnedBuf,
    event: DeviceEvent,
    pending: bool,
}

impl RingSlot {
    fn wait_ready(&mut self, stream: &DeviceStream) -> Result<(), EngError> {
        if self.pending {
            self.buf.wait(&self.event)?;
            self.pending = false;
        } else if self.buf.pending_stream().is_some() {
            // An enqueue can succeed before a later enqueue or its event record fails.
            self.buf.synchronize(stream)?;
        }
        Ok(())
    }
}

/// Ring depth: eight batch-sized slots (~18.9 MiB at the Qwen slice of 589,824 B and
/// batches of four) — enough ahead-runs to keep the DMA fed while one slot waits its
/// event.
const COLD_RING_DEPTH: usize = 8;

impl ColdStaging {
    /// Builds the zero-filled staging bank of `cap` slots plus its ring. Refuses before
    /// any allocation for a cap the bank geometry refuses (below `top_k`).
    pub(super) fn new(
        ctx: &CudaContext,
        geometry: ExpertBankGeometry,
        cap: usize,
    ) -> Result<Self, EngError> {
        let bank = ExpertBank::new(ctx, geometry, cap)?;
        let (gate_up, down) = bank.slice_bytes()?;
        let slot_bytes = gate_up.max(down) * Self::BATCH;
        let mut ring = Vec::with_capacity(COLD_RING_DEPTH);
        for _ in 0..COLD_RING_DEPTH {
            ring.push(RingSlot {
                buf: ctx.pinned_buf(slot_bytes)?,
                event: ctx.event()?,
                pending: false,
            });
        }
        Ok(Self { bank, cap, ring })
    }

    pub(super) fn cap(&self) -> usize {
        self.cap
    }

    /// The staged gate/up/down stacks for the cold grouped launches.
    pub(super) fn weights(&self) -> (&NativeWeight, &NativeWeight, &NativeWeight) {
        (self.bank.gate(), self.bank.up(), self.bank.down())
    }

    /// Slices per ring batch: consecutive cold channels are contiguous in the staging
    /// stacks (the list is channel-ordered), so one pinned slot carries a batch as one
    /// wide DMA — a 590 KiB copy runs the link at a third of a 2.4 MiB one (measured
    /// 3.4 GB/s per-slice early layers).
    const BATCH: usize = 4;

    /// Stages one chunk layer's cold experts: for each stack (gate, up, down —
    /// stack-major), batches of [`Self::BATCH`] consecutive channels' slices are
    /// gathered mapping → ring slot on the host and queued ring slot → staging as one
    /// DMA on `stream` (enqueue-only; the waits that recycle a ring slot synchronize
    /// only that slot's event). The launches queued behind this on the same stream read
    /// the staged bytes in order.
    pub(super) fn stage(
        &mut self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        cold: &[ChunkColdStage],
        region: &Arc<dyn MappedRegion>,
    ) -> Result<(), EngError> {
        if cold.is_empty() {
            return Ok(());
        }
        let bytes = region.bytes();
        let ColdStaging { bank, cap, ring } = self;
        let (gate_up, down) = bank.slice_bytes()?;
        type ColdStack<'a> = (
            &'a DeviceBuffer,
            usize,
            fn(&ChunkColdStage) -> &Range<usize>,
        );
        let stacks: [ColdStack<'_>; 3] = [
            (bank.gate().buffer(), gate_up, |c| &c.gate),
            (bank.up().buffer(), gate_up, |c| &c.up),
            (bank.down().buffer(), down, |c| &c.down),
        ];
        let mut at = 0usize;
        for (stack, slice_bytes, range_of) in stacks {
            for batch in cold.chunks(Self::BATCH) {
                let first = usize::from(batch[0].channel);
                if first + batch.len() > *cap {
                    return Err(EngError::BadRequest(format!(
                        "tiered cold staging: channels {first}..{} of {cap}",
                        first + batch.len()
                    )));
                }
                // Consecutive channels, or the batched DMA would scramble slots.
                for (step, entry) in batch.iter().enumerate() {
                    if usize::from(entry.channel) != first + step {
                        return Err(EngError::BadRequest(format!(
                            "tiered cold staging: channel {} follows {} — the staging list is \
                             not channel-ordered",
                            entry.channel, first
                        )));
                    }
                }
                let at_slot = at % ring.len();
                at += 1;
                let slot = &mut ring[at_slot];
                slot.wait_ready(stream)?;
                for (step, entry) in batch.iter().enumerate() {
                    let range = range_of(entry);
                    let slice = bytes.get(range.clone()).ok_or_else(|| {
                        EngError::Checkpoint(format!(
                            "tiered cold staging: expert {} range {range:?} is outside the mapping",
                            entry.expert
                        ))
                    })?;
                    if slice.len() != slice_bytes {
                        return Err(EngError::Checkpoint(format!(
                            "tiered cold staging: expert {} slice holds {} bytes, a slot holds {slice_bytes}",
                            entry.expert,
                            slice.len()
                        )));
                    }
                    slot.buf.write(step * slice_bytes, slice)?;
                }
                let dst = stack.range(first * slice_bytes, batch.len() * slice_bytes)?;
                ctx.copy_from_pinned(stream, &mut slot.buf, 0, dst)?;
                stream.record(&slot.event)?;
                slot.pending = true;
            }
        }
        Ok(())
    }
}

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

impl<'d> BankFill<'d> {
    fn expert<'r>(
        &self,
        layer: usize,
        ranges: &ExpertRanges,
        region: &'r dyn MappedRegion,
    ) -> Result<(&'d ExpertBank, [&'r [u8]; 3]), HostError> {
        let bank = self
            .banks
            .get(layer)
            .ok_or_else(|| HostError::Device(format!("no bank for layer {layer}")))?;
        let bytes = region.bytes();
        let cut = |range: &Range<usize>| {
            bytes.get(range.clone()).ok_or_else(|| {
                HostError::Device(format!("expert range {range:?} is outside the region"))
            })
        };
        Ok((
            bank,
            [cut(&ranges.gate)?, cut(&ranges.up)?, cut(&ranges.down)?],
        ))
    }
}

impl BankDevice for BankFill<'_> {
    fn fill_slot(
        &mut self,
        layer: usize,
        slot: usize,
        ranges: &ExpertRanges,
        region: &Arc<dyn MappedRegion>,
    ) -> Result<(), HostError> {
        let (bank, [gate, up, down]) = self.expert(layer, ranges, region.as_ref())?;
        bank.fill(self.stream, slot, gate, up, down)
            .map_err(fill_error)
    }
}

fn fill_error(error: EngError) -> HostError {
    HostError::Device(format!("{error:?}"))
}

/// A warm-up fill uses a slot of the cold staging ring for all three projections.
/// The ledger's wait still precedes publication of any destination slot.
struct PinnedWarmFill<'d, 'r> {
    targets: BankFill<'d>,
    context: &'d CudaContext,
    ring: &'r mut [RingSlot],
    next: usize,
}

impl BankDevice for PinnedWarmFill<'_, '_> {
    fn fill_slot(
        &mut self,
        layer: usize,
        slot: usize,
        ranges: &ExpertRanges,
        region: &Arc<dyn MappedRegion>,
    ) -> Result<(), HostError> {
        let (bank, slices) = self.targets.expert(layer, ranges, region.as_ref())?;
        let lengths = slices.map(|slice| slice.len());
        let [gate, up, down] = bank.fill_ranges(slot, lengths).map_err(fill_error)?;
        let offsets = [0, lengths[0], lengths[0] + lengths[1]];
        let index = self.next;
        self.next = (index + 1) % self.ring.len();
        let staging = &mut self.ring[index];
        let stream = self.targets.stream;
        staging.wait_ready(stream).map_err(fill_error)?;
        // Fill the whole host slot before enqueuing any copy from it.
        for (offset, slice) in offsets.into_iter().zip(slices) {
            staging.buf.write(offset, slice).map_err(fill_error)?;
        }
        for (offset, dst) in offsets.into_iter().zip([gate, up, down]) {
            self.context
                .copy_from_pinned(stream, &mut staging.buf, offset, dst)
                .map_err(fill_error)?;
        }
        stream.record(&staging.event).map_err(fill_error)?;
        staging.pending = true;
        Ok(())
    }

    fn wait_fills(&mut self) -> Result<(), HostError> {
        for staging in &mut *self.ring {
            staging
                .wait_ready(self.targets.stream)
                .map_err(fill_error)?;
        }
        Ok(())
    }
}

pub(super) struct TieredRuntime<'w> {
    parts: Vec<LayerParts<'w>>,
    banks: Vec<ExpertBank>,
    scratch: TieredScratch,
    host: TieredHost,
    /// The cold staging bank of the chunked prefill: `None` when admission left no
    /// spare VRAM for it (the chunk path then keeps the CPU pool for every miss).
    cold: Option<ColdStaging>,
    /// The bank depth (slots per layer) every bank was built with.
    slot_count: usize,
    warm_prefill: bool,
    /// The host waits of the token so far: the layers' and, after `end_token`, the bank fills'.
    syncs: HostSyncs,
    /// The TCAP sync-vs-drain probe's state (Q36_TIERED_PROBE=1 only): the previous
    /// token's device-span event pair, the summed route waits, and a token counter.
    probe_start: Option<crate::kernels::cuda::TimingEvent>,
    probe_end: Option<crate::kernels::cuda::TimingEvent>,
    probe_waits_ns: u128,
    /// Probe: the per-layer host-phase sums (parse, begin, plan, launch, finish) —
    /// the host-paced phases where the 52%-util decode idles the device.
    probe_phases_ns: [u128; 5],
    probe_tokens: u64,
    /// Probe: one timing event per layer start, layer 1 on (layer 0's start IS
    /// `probe_start`), for the span-interior split of the device timeline.
    probe_layers: Vec<crate::kernels::cuda::TimingEvent>,
    /// Probe: the token-report sum since the last print — bank hits and CPU serves
    /// per token are the bytes/token crossing PCIe.
    probe_report: TokenReport,
    /// Probe: `probe_tokens` at the last counter print, the window's denominator.
    probe_report_base: u64,
    /// Reusable buffers of the chunk route readback and the pool-partial add. A fresh
    /// 15.85 MB pinned allocation (cudaHostAlloc + full memset) whose drop runs a
    /// device-wide synchronize, per layer, measured 5-27 ms/layer at m up to 2,048
    /// (the round-2 panel); these grow only when the piece's m grows.
    route: RouteScratch,
}

/// The layout of one layer's route readback inside [`RouteScratch::pinned`].
#[derive(Clone, Copy)]
pub(super) struct RouteLayout {
    pub(super) ids_bytes: usize,
    pub(super) weights_bytes: usize,
    pub(super) total: usize,
}

/// Persistent pinned/device buffers of the chunk route readback and the overflow
/// partial upload (grow-only in `m`; a few tens of MiB, below any admission ledger).
#[derive(Default)]
pub(super) struct RouteScratch {
    pinned: Option<PinnedBuf>,
    event: Option<DeviceEvent>,
    cap: usize,
    partial: Option<PinnedBuf>,
    partial_event: Option<DeviceEvent>,
    partial_cap: usize,
    partial_bytes: Vec<u8>,
    partial_dst: Option<DeviceBuffer>,
    /// The hidden block's f32 staging, grown with m and refilled per layer.
    hidden_f32: Vec<f32>,
}

impl<'w> TieredRuntime<'w> {
    pub(super) fn new(
        parts: Vec<LayerParts<'w>>,
        banks: Vec<ExpertBank>,
        scratch: TieredScratch,
        host: TieredHost,
        cold: Option<ColdStaging>,
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
            cold,
            slot_count,
            warm_prefill: std::env::var_os("Q36_TIERED_PREFILL_WARM")
                .is_some_and(|value| value == "1"),
            syncs: HostSyncs::default(),
            probe_start: None,
            probe_end: None,
            probe_waits_ns: 0,
            probe_phases_ns: [0; 5],
            probe_tokens: 0,
            probe_layers: Vec::new(),
            probe_report: TokenReport::default(),
            probe_report_base: 0,
            route: RouteScratch::default(),
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

    /// The router buffers of `layer` (the raw F32 matrix and the shared gate),
    /// at the parts' 'w lifetime — the chunk's router stage reads them.
    pub(super) fn router_of(
        &self,
        layer: usize,
    ) -> Result<
        (
            &'w crate::kernels::cuda::DeviceBuffer,
            &'w crate::kernels::cuda::DeviceBuffer,
        ),
        EngError,
    > {
        let parts = self.parts.get(layer).ok_or_else(|| {
            EngError::BadRequest(format!("tiered runtime: layer {layer} has no parts"))
        })?;
        Ok((parts.router, parts.router_shared))
    }

    /// Starts the grouped pool serving of a chunk layer's misses (the delegate to
    /// the host's `start_chunk_pool`): one GroupedExpertJob per DISTINCT missed expert,
    /// started BEFORE the GPU stages so the CPU work overlaps them.
    pub(super) fn start_chunk_pool(
        &mut self,
        layer: usize,
        misses: &[crate::family::qwen36::tiered::host::ChunkMissGroup],
        block: &[f32],
        m: usize,
    ) -> Result<
        Option<crate::family::qwen36::tiered::cpu_pool::PendingChunk>,
        crate::family::qwen36::tiered::host::HostError,
    > {
        self.host.start_chunk_pool(layer, misses, block, m)
    }

    /// Enqueues the batched route readback (ids, weights, normed rows) into the
    /// REUSABLE pinned buffer, growing it only when the piece's `m` grows.
    #[allow(clippy::too_many_arguments)] // The buffer ranges and their shape form one transfer.
    pub(super) fn route_enqueue(
        &mut self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        ids: DeviceRange<'_>,
        weights: DeviceRange<'_>,
        x2: DeviceRange<'_>,
        m: usize,
        top_k: usize,
        hidden: usize,
    ) -> Result<RouteLayout, EngError> {
        let layout = RouteLayout {
            ids_bytes: m * top_k * 4,
            weights_bytes: m * top_k * 4,
            total: m * (2 * top_k + hidden) * 4,
        };
        if self.route.cap < layout.total {
            self.route.pinned = Some(ctx.pinned_buf(layout.total)?);
            self.route.cap = layout.total;
        }
        let pinned = self
            .route
            .pinned
            .as_mut()
            .expect("the route buffer was just grown");
        ctx.copy_to_pinned(stream, ids, pinned, 0)?;
        ctx.copy_to_pinned(stream, weights, pinned, layout.ids_bytes)?;
        // The normed row the CPU experts read: the router's x2 (the post-norm hidden).
        ctx.copy_to_pinned(stream, x2, pinned, layout.ids_bytes + layout.weights_bytes)?;
        let event = match &self.route.event {
            Some(event) => event,
            None => {
                self.route.event = Some(ctx.event()?);
                self.route.event.as_ref().expect("just created")
            }
        };
        stream.record(event)?;
        Ok(layout)
    }

    /// Waits the readback and parses ONLY the ids and weights (the hidden block stays
    /// in pinned memory — the pool reads it through [`Self::pool_dispatch`]).
    pub(super) fn route_finish(
        &mut self,
        layout: RouteLayout,
    ) -> Result<(Vec<i32>, Vec<f32>), EngError> {
        let route = &mut self.route;
        let (pinned, event) = (
            route.pinned.as_mut().expect("route_enqueue ran first"),
            route.event.as_ref().expect("route_enqueue ran first"),
        );
        pinned.wait(event)?;
        let mut raw = vec![0u8; layout.ids_bytes + layout.weights_bytes];
        pinned.read(0, &mut raw)?;
        let floats = |bytes: &[u8]| -> Vec<f32> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes(*w))
                .collect()
        };
        let ints = |bytes: &[u8]| -> Vec<i32> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| i32::from_le_bytes(*w))
                .collect()
        };
        Ok((
            ints(&raw[..layout.ids_bytes]),
            floats(&raw[layout.ids_bytes..]),
        ))
    }

    /// Dispatches the overflow pool jobs with the hidden block read out of the pinned
    /// route buffer through the word-wide `read_f32` (the per-byte read plus per-value
    /// f32 conversion of 3.9 M values measured 4-15 ms/layer at m 1,922 — the round-2
    /// panel's MoeStart decomposition). Skips the read entirely when there is no
    /// overflow (the pool is hidden's only consumer).
    pub(super) fn pool_dispatch(
        &mut self,
        layer: usize,
        misses: &[crate::family::qwen36::tiered::host::ChunkMissGroup],
        layout: RouteLayout,
        m: usize,
    ) -> Result<Option<crate::family::qwen36::tiered::cpu_pool::PendingChunk>, EngError> {
        let values = (layout.total - layout.ids_bytes - layout.weights_bytes) / 4;
        let block: &[f32] = if misses.is_empty() {
            &[]
        } else {
            let route = &mut self.route;
            route.hidden_f32.resize(values, 0.0);
            let pinned = route.pinned.as_mut().expect("route_enqueue ran first");
            // `route_finish` already waited the readback's event, which releases the
            // buffer's pending copy — `read_f32` refuses only while one is outstanding
            // (and `wait` refuses when none is).
            pinned.read_f32(
                layout.ids_bytes + layout.weights_bytes,
                &mut route.hidden_f32,
            )?;
            &route.hidden_f32
        };
        self.host
            .start_chunk_pool(layer, misses, block, m)
            .map_err(|e| EngError::Backend {
                operation: "tiered_chunk_pool",
                message: e.to_string(),
            })
    }

    /// Uploads the pool partial through the REUSED pinned staging and device buffer
    /// and adds it in place (1.0 * partial, top_k 1) — the pageable `ctx.upload` per
    /// layer (fresh device alloc + 15.7 MB pageable copy + device free) was the
    /// MoeCombine interval's growth to 14.8 ms/layer at m 1,922.
    pub(super) fn add_pool_partial(
        &mut self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        res: DeviceRange<'_>,
        partial: &[f32],
        m: usize,
        hidden: usize,
    ) -> Result<(), EngError> {
        let bytes = m * hidden * 4;
        self.route.partial_bytes.clear();
        self.route.partial_bytes.reserve(bytes);
        for value in partial {
            self.route
                .partial_bytes
                .extend_from_slice(&value.to_le_bytes());
        }
        if self.route.partial_cap < bytes {
            self.route.partial = Some(ctx.pinned_buf(bytes)?);
            self.route.partial_cap = bytes;
            self.route.partial_dst = None;
            // A fresh buffer holds no outstanding copy; `wait` refuses without one.
            self.route.partial_event = None;
        }
        {
            let route = &mut self.route;
            let event = route.partial_event.as_ref();
            let pinned = route.partial.as_mut().expect("just grown");
            // Steady state: the previous layer's copy_from_pinned is the only
            // outstanding transfer and this wait releases it (a no-op for the
            // device — long complete — but it clears the write refusal).
            if let Some(event) = event {
                pinned.wait(event)?;
            }
            pinned.write(0, &route.partial_bytes)?;
        }
        // The destination grows only with m; the first copy allocates it.
        if self
            .route
            .partial_dst
            .as_ref()
            .is_none_or(|dst| dst.bytes() < bytes)
        {
            self.route.partial_dst = Some(ctx.upload(&self.route.partial_bytes)?);
        } else {
            let dst_range = self
                .route
                .partial_dst
                .as_ref()
                .expect("checked above")
                .range(0, bytes)?;
            let pinned = self.route.partial.as_mut().expect("grown above");
            ctx.copy_from_pinned(stream, pinned, 0, dst_range)?;
            let event = match &self.route.partial_event {
                Some(event) => event,
                None => {
                    self.route.partial_event = Some(ctx.event()?);
                    self.route.partial_event.as_ref().expect("just created")
                }
            };
            stream.record(event)?;
        }
        let ones: Vec<u8> = [0x00, 0x00, 0x80, 0x3f].repeat(m); // 1.0f32 LE
        let ones_buf = ctx.upload(&ones)?;
        let down = self
            .route
            .partial_dst
            .as_ref()
            .expect("grown above")
            .range(0, bytes)?;
        ctx.moe_combine(
            stream,
            crate::kernels::cuda::resident::moe_combine::CombineCall {
                shape: crate::kernels::cuda::resident::moe_combine::CombineShape {
                    m,
                    n: hidden,
                    top_k: 1,
                    ld_down: hidden,
                },
                res,
                down,
                weights: ones_buf.range(0, ones.len())?,
                shared: None,
                out: crate::kernels::cuda::resident::moe_combine::CombineOut::InPlace,
            },
        )?;
        Ok(())
    }

    /// The M-row classification delegate: the chunk arm's route readback flows through
    /// the host's `begin_chunk_layer`, with the cold staging's capacity as the cut.
    pub(super) fn begin_chunk(
        &mut self,
        layer: usize,
        rows: &[crate::family::qwen36::tiered::host::ChunkRouteRow],
    ) -> Result<
        (
            crate::family::qwen36::tiered::host::ChunkLayerPlanHost,
            Vec<crate::family::qwen36::tiered::host::ChunkMissGroup>,
            Vec<crate::family::qwen36::tiered::host::ChunkColdStage>,
        ),
        crate::family::qwen36::tiered::host::HostError,
    > {
        let allocated = self.cold.as_ref().map_or(0, ColdStaging::cap);
        // Experiment: trade fewer PCIe expert transfers for more CPU overflow work.
        // The allocation, admission charge and launch bank depth remain unchanged.
        static LIMIT: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        let limit = LIMIT.get_or_init(|| {
            std::env::var("Q36_TIERED_COLD_LIMIT")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
        });
        // Opt-in short-chunk cap: leave the ordinary limit as a hard ceiling,
        // but avoid staging as many experts for chunks with less reuse. This
        // changes only classification; the existing per-expert minimum-use
        // filter, allocated bank and admission charge are unchanged.
        static SHORT_LIMIT: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        let short_limit = SHORT_LIMIT.get_or_init(|| {
            std::env::var("Q36_TIERED_COLD_SHORT_LIMIT")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
        });
        let cap = crate::family::qwen36::tiered::host::chunk_cold_cap(
            allocated,
            *limit,
            *short_limit,
            rows.len(),
        );
        if short_limit.is_some() && layer == 0 {
            eprintln!("tiered-cold-short-limit: rows {} using {cap}", rows.len());
        }
        if limit.is_some() && layer == 0 {
            eprintln!("tiered-cold-limit: allocated {allocated} using {cap}");
        }
        self.host.begin_chunk_layer(layer, rows, cap)
    }

    /// The cold staging depth (0 when admission built none).
    pub(super) fn cold_cap(&self) -> usize {
        self.cold.as_ref().map_or(0, ColdStaging::cap)
    }

    /// The checkpoint mapping the cold staging fill gathers its slices from.
    pub(super) fn cold_region(&self) -> Arc<dyn MappedRegion> {
        self.host.cold_region()
    }

    /// Queues one layer's cold staging fills (the delegate over [`ColdStaging::stage`]);
    /// a no-op without a staging bank or cold experts.
    pub(super) fn stage_cold(
        &mut self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        cold: &[crate::family::qwen36::tiered::host::ChunkColdStage],
        region: &Arc<dyn MappedRegion>,
    ) -> Result<(), EngError> {
        if let Some(staging) = self.cold.as_mut() {
            staging.stage(ctx, stream, cold, region)?;
        }
        Ok(())
    }

    /// The cold staging stacks for the cold grouped launches, borrowed beside the banks'
    /// own in `layer_weights`.
    pub(super) fn cold_weights(
        &self,
    ) -> Result<(&NativeWeight, &NativeWeight, &NativeWeight), EngError> {
        let staging = self.cold.as_ref().ok_or_else(|| {
            EngError::BadRequest(String::from("tiered runtime: no cold staging bank"))
        })?;
        Ok(staging.weights())
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
        if layer == 0 && super::moe::tiered::tiered_probe() {
            // Close the previous token: device span vs the summed route waits. The
            // waits minus the span (clamped at zero) is the SYNC share TCAP could
            // remove; anything the span covers is drain that returns at the boundary.
            if let (Some(start), Some(end)) = (self.probe_start.take(), self.probe_end.take()) {
                // The end event completed long ago (the whole boundary ran since);
                // elapsed on a completed event needs no host sync.
                let span_ms = start.elapsed_ms_since(&end)?;
                let waits_ms = self.probe_waits_ns as f32 / 1e6;
                let sync_ms = (waits_ms - span_ms).max(0.0);
                // The layer-duration stats: consecutive deltas over the layer start
                // events (layer 1 on) plus the token-end event cover every layer.
                let mut count = 0usize;
                let (mut sum, mut min, mut max) = (0.0f32, f32::INFINITY, 0.0f32);
                let mut prev = &start;
                for event in self.probe_layers.iter().chain(std::iter::once(&end)) {
                    let ms = prev.elapsed_ms_since(event)?;
                    count += 1;
                    sum += ms;
                    min = min.min(ms);
                    max = max.max(ms);
                    prev = event;
                }
                self.probe_layers.clear();
                self.probe_tokens += 1;
                if self.probe_tokens % 50 == 1 {
                    let window = (self.probe_tokens - self.probe_report_base).max(1);
                    let per_token = |n: u32| n as f32 / window as f32;
                    let report = std::mem::take(&mut self.probe_report);
                    let (avg_ms, min_ms, max_ms) = if count > 0 {
                        (sum / count as f32, min, max)
                    } else {
                        (0.0, 0.0, 0.0)
                    };
                    let [parse, begin, plan, launch, finish] = self.probe_phases_ns;
                    let phase_ms = |ns: u128| ns as f32 / 1e6 / window as f32;
                    eprintln!(
                        "tcap-probe: token {} device_span_ms {:.2} waits_ms {:.2} sync_share_ms {:.2} layer_ms avg {:.3} min {:.3} max {:.3} hits/tok {:.2} cpu_served/tok {:.2} swaps/tok {:.2} host_ms/tok parse {:.2} begin {:.2} plan {:.2} launch {:.2} finish {:.2}",
                        self.probe_tokens,
                        span_ms,
                        waits_ms,
                        sync_ms,
                        avg_ms,
                        min_ms,
                        max_ms,
                        per_token(report.bank_hits),
                        per_token(report.cpu_served),
                        per_token(report.swaps),
                        phase_ms(parse),
                        phase_ms(begin),
                        phase_ms(plan),
                        phase_ms(launch),
                        phase_ms(finish),
                    );
                    self.probe_phases_ns = [0; 5];
                    self.probe_report_base = self.probe_tokens;
                }
                self.probe_waits_ns = 0;
            }
            let start = launch.ctx.event_timing()?;
            launch.stream.record_timing(&start)?;
            self.probe_start = Some(start);
        }
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
        if super::moe::tiered::tiered_probe() {
            // The per-layer route wait sample landed in the scratch's cell.
            self.probe_waits_ns += self.scratch.probe_route_ns.get() as u128;
            self.scratch.probe_route_ns.set(0);
            let scratch = &self.scratch;
            for (sum, cell) in self.probe_phases_ns.iter_mut().zip([
                scratch.probe_parse_ns.get(),
                scratch.probe_begin_ns.get(),
                scratch.probe_plan_ns.get(),
                scratch.probe_launch_ns.get(),
                scratch.probe_finish_ns.get(),
            ]) {
                *sum += cell as u128;
            }
            // The next layer's start event: layer 0's start IS probe_start.
            if layer > 0 {
                let event = launch.ctx.event_timing()?;
                launch.stream.record_timing(&event)?;
                self.probe_layers.push(event);
            }
            let layers = self.parts.len();
            if layer + 1 == layers {
                let end = launch.ctx.event_timing()?;
                launch.stream.record_timing(&end)?;
                self.probe_end = Some(end);
            }
        }
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

    pub(super) fn warm_after_prefill(
        &mut self,
        context: &CudaContext,
        stream: &DeviceStream,
        rows: usize,
    ) -> Result<(), EngError> {
        // Small chunks seldom contain enough routing history to guide a warm-up.
        if !self.warm_prefill || rows < 64 {
            return Ok(());
        }
        self.warm_prefill = false;
        let start = std::time::Instant::now();
        let targets = BankFill {
            banks: &self.banks,
            stream,
        };
        let use_pinned =
            std::env::var_os("Q36_TIERED_PREFILL_PINNED").is_some_and(|value| value == "1");
        let (swaps, transfer) = if let Some(staging) = self.cold.as_mut().filter(|_| use_pinned) {
            let mut fills = PinnedWarmFill {
                targets,
                context,
                ring: &mut staging.ring,
                next: 0,
            };
            (self.host.warm_prefill(&mut fills), "pinned")
        } else {
            let mut fills = targets;
            (self.host.warm_prefill(&mut fills), "blocking")
        };
        let swaps = swaps.map_err(host_error)?;
        eprintln!(
            "tiered prefill warm-up: {swaps} swaps in {:.1} ms ({transfer})",
            start.elapsed().as_secs_f64() * 1e3
        );
        Ok(())
    }

    pub(super) fn end_token(&mut self, stream: &DeviceStream) -> Result<TokenReport, EngError> {
        let mut fills = BankFill {
            banks: &self.banks,
            stream,
        };
        let report = self.host.end_token(&mut fills).map_err(host_error)?;
        self.syncs.writes += report.swaps * ExpertBank::FILL_WRITES;
        if super::moe::tiered::tiered_probe() {
            self.probe_report.merge(&report);
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::cuda::device_present;

    #[test]
    fn pinned_warm_fill_round_trips_after_ring_reuse() {
        if !device_present("pinned_warm_fill_round_trips_after_ring_reuse") {
            return;
        }
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let geometry = ExpertBankGeometry {
            hidden: 512,
            inter: 256,
            top_k: 2,
        };
        let mut staging = ColdStaging::new(&ctx, geometry, 2).unwrap();
        let fills = COLD_RING_DEPTH + 3;
        let bank = ExpertBank::new(&ctx, geometry, fills + 1).unwrap();
        let (gate, down) = bank.slice_bytes().unwrap();
        let lengths = [gate, gate, down];
        let stride = lengths.iter().sum::<usize>();
        let mut bytes = Vec::new();
        for slot in 0..fills {
            for (projection, length) in lengths.into_iter().enumerate() {
                bytes.extend(std::iter::repeat_n(
                    (slot * 3 + projection + 1) as u8,
                    length,
                ));
            }
        }
        let region: Arc<dyn MappedRegion> = Arc::new(bytes);
        let mut transfer = PinnedWarmFill {
            targets: BankFill {
                banks: std::slice::from_ref(&bank),
                stream: &stream,
            },
            context: &ctx,
            ring: &mut staging.ring,
            next: 0,
        };
        for slot in 0..fills {
            let offset = slot * stride;
            let ranges = ExpertRanges {
                gate: offset..offset + gate,
                up: offset + gate..offset + 2 * gate,
                down: offset + 2 * gate..offset + stride,
            };
            transfer.fill_slot(0, slot, &ranges, &region).unwrap();
        }
        transfer.wait_fills().unwrap();
        for (projection, (stack, width)) in [bank.gate(), bank.up(), bank.down()]
            .into_iter()
            .zip(lengths)
            .enumerate()
        {
            let mut bytes = vec![0; (fills + 1) * width];
            stack.buffer().read_at(&stream, 0, &mut bytes).unwrap();
            for (slot, row) in bytes.chunks_exact(width).enumerate() {
                let expected = if slot < fills {
                    (3 * slot + projection + 1) as u8
                } else {
                    0
                };
                assert!(
                    row.iter().all(|&byte| byte == expected),
                    "projection {projection} slot {slot}"
                );
            }
        }
    }
}
