//! Two bounded transport slots for the canonical NVMe tail, never a residency policy.
//! All GPU reads of a lease's parts must use its stream. Finish before consuming
//! results; kernel-specific numerical status checks remain the format adapter's job.
//! A dropped lease also waits, so the slot cannot overwrite an outstanding consumer.

use std::ops::Range;

use super::{
    ExpertId, ExpertPayloadSource, ExpertPayloadView, Placement, Residency, StreamBankPlan,
};
use crate::EngError;
use crate::kernels::cuda::{
    CudaContext, DeviceBuffer, DeviceEvent, DeviceRange, DeviceStream, PinnedBuf,
};

// Preserve CUDA allocation alignment for every packed/scales subrange.
const ALIGN: usize = 256;

pub(crate) struct CudaStreamBanks {
    slots: [CudaStreamBank; 2],
}

pub(crate) struct CudaStreamBank {
    host: PinnedBuf,
    device: DeviceBuffer,
    transfer: DeviceStream,
    compute: DeviceStream,
    ready: DeviceEvent,
    done: DeviceEvent,
    parts: Vec<Part>,
    used: usize,
    scratch_limit: usize,
    poisoned: bool,
    #[cfg(test)]
    timing: Option<super::stream_cuda_profile::Pair>,
}

struct Part {
    key: (ExpertId, usize),
    data: Range<usize>,
    scales: Option<Range<usize>>,
    tensor_scale: f32,
    shape: [u32; 2],
}

pub(crate) struct StreamedPart<'a> {
    pub(crate) data: DeviceRange<'a>,
    pub(crate) scales: Option<DeviceRange<'a>>,
    pub(crate) tensor_scale: f32,
    pub(crate) shape: [u32; 2],
}

#[must_use = "finish the bank lease and check kernel status before consuming output"]
pub(crate) struct CudaBankLease<'a> {
    bank: &'a mut CudaStreamBank,
    finished: bool,
}

fn bad(message: &str) -> EngError {
    EngError::BadRequest(format!("expert streaming: {message}"))
}

fn capacity() -> EngError {
    EngError::MemoryPlan("expert streaming exceeds its admitted bank or host scratch".into())
}

impl CudaStreamBanks {
    /// Use the plan's reservation for hot placement first; allocate after releasing that wrapper.
    /// Host scratch covers demand ids, part metadata and one temporary payload per slot.
    pub(crate) fn new(ctx: &CudaContext, plan: StreamBankPlan) -> Result<Self, EngError> {
        let bytes = usize::try_from(plan.bank_bytes()).map_err(|_| capacity())?;
        let scratch = plan.host_bytes() - 2 * plan.bank_bytes();
        let limit = usize::try_from(scratch / 2).map_err(|_| capacity())?;
        Ok(Self {
            slots: [
                CudaStreamBank::new(ctx, bytes, limit)?,
                CudaStreamBank::new(ctx, bytes, limit)?,
            ],
        })
    }

    pub(crate) fn slots_mut(&mut self) -> (&mut CudaStreamBank, &mut CudaStreamBank) {
        let [a, b] = &mut self.slots;
        (a, b)
    }
}

impl CudaStreamBank {
    fn new(ctx: &CudaContext, bytes: usize, scratch_limit: usize) -> Result<Self, EngError> {
        Ok(Self {
            host: ctx.pinned_buf(bytes)?,
            device: ctx.allocate(bytes)?,
            transfer: ctx.stream()?,
            compute: ctx.stream()?,
            ready: ctx.event()?,
            done: ctx.event()?,
            parts: Vec::new(),
            used: 0,
            scratch_limit,
            poisoned: false,
            #[cfg(test)]
            timing: None,
        })
    }

    /// Stage only the NVMe subset. GPU/CPU assignments stay in their canonical tiers.
    /// The allocating context must be supplied; its typed copy API checks ownership.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn stage<'a, S: ExpertPayloadSource>(
        &'a mut self,
        ctx: &CudaContext,
        source: &mut S,
        residency: &Residency,
        demand: &[ExpertId],
    ) -> Result<Option<CudaBankLease<'a>>, EngError> {
        let ids = tail_ids(residency, demand, self.scratch_limit)?;
        self.stage_ids(ctx, source, ids)
    }

    /// Stage every demanded expert (the prefill stream): placement tiers are
    /// untouched — this is a per-chunk transport, not a residency decision.
    pub(crate) fn stage_demand<'a, S: ExpertPayloadSource>(
        &'a mut self,
        ctx: &CudaContext,
        source: &mut S,
        demand: &[ExpertId],
    ) -> Result<Option<CudaBankLease<'a>>, EngError> {
        let ids = demand_ids(demand, self.scratch_limit)?;
        self.stage_ids(ctx, source, ids)
    }

    fn stage_ids<'a, S: ExpertPayloadSource>(
        &'a mut self,
        ctx: &CudaContext,
        source: &mut S,
        ids: Vec<ExpertId>,
    ) -> Result<Option<CudaBankLease<'a>>, EngError> {
        ctx.ensure_usable()?;
        if self.poisoned {
            return Err(bad("a previous bank completion failed"));
        }
        // Free previous metadata before allocating this refill's admitted scratch.
        self.parts = Vec::new();
        self.used = 0;
        if ids.is_empty() {
            return Ok(None);
        }
        let ids_bytes = ids
            .capacity()
            .checked_mul(size_of::<ExpertId>())
            .ok_or_else(capacity)?;
        let scratch = self
            .scratch_limit
            .checked_sub(ids_bytes)
            .ok_or_else(capacity)?;
        let (parts, used) = prepare(&mut self.host, source, &ids, scratch)?;
        // The source is no longer borrowed: DMA reads only our owned pinned bytes.
        source.release_experts(&ids)?;
        self.copy(ctx, used)?;
        self.parts = parts;
        self.used = used;
        Ok(Some(CudaBankLease {
            bank: self,
            finished: false,
        }))
    }

    fn copy(&mut self, ctx: &CudaContext, used: usize) -> Result<(), EngError> {
        #[cfg(test)]
        let mut timing = super::stream_cuda_profile::begin(ctx, &self.transfer, used)?;
        ctx.copy_from_pinned(
            &self.transfer,
            &mut self.host,
            0,
            self.device.range(0, used)?,
        )?;
        #[cfg(test)]
        {
            super::stream_cuda_profile::end(&self.transfer, &mut timing)?;
            self.timing = timing;
        }
        let ordered = self
            .transfer
            .record(&self.ready)
            .and_then(|_| self.compute.wait(&self.ready));
        if ordered.is_err() {
            self.poisoned = true;
        }
        ordered
    }

    fn complete(&mut self) -> Result<(), EngError> {
        let outcome = (|| {
            self.compute.record(&self.done)?;
            self.done.synchronize()?;
            self.host.wait(&self.ready)
        })();
        #[cfg(test)]
        {
            let timing = self.timing.take();
            if outcome.is_ok() {
                super::stream_cuda_profile::complete(timing);
            }
        }
        if outcome.is_err() {
            self.poisoned = true;
        }
        outcome
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn tail_ids(
    residency: &Residency,
    demand: &[ExpertId],
    limit: usize,
) -> Result<Vec<ExpertId>, EngError> {
    let tail = |id: &&ExpertId| residency.placement(**id) == Placement::Nvme;
    let count = demand.iter().filter(tail).count();
    let bytes = count
        .checked_mul(size_of::<ExpertId>())
        .ok_or_else(capacity)?;
    if bytes > limit {
        return Err(capacity());
    }
    let mut ids = Vec::new();
    ids.try_reserve_exact(count).map_err(|_| capacity())?;
    ids.extend(demand.iter().filter(tail).copied());
    ids.sort_unstable();
    ids.dedup();
    Ok(ids)
}

/// The prefill stream's demand set: every demanded expert regardless of
/// placement — streaming is a per-chunk transport, never a residency tier.
fn demand_ids(demand: &[ExpertId], limit: usize) -> Result<Vec<ExpertId>, EngError> {
    let mut ids = Vec::new();
    ids.try_reserve_exact(demand.len())
        .map_err(|_| capacity())?;
    ids.extend(demand.iter().copied());
    ids.sort_unstable();
    ids.dedup();
    if ids
        .capacity()
        .checked_mul(size_of::<ExpertId>())
        .is_some_and(|bytes| bytes > limit)
    {
        return Err(capacity());
    }
    Ok(ids)
}

fn prepare<S: ExpertPayloadSource>(
    host: &mut PinnedBuf,
    source: &mut S,
    ids: &[ExpertId],
    scratch: usize,
) -> Result<(Vec<Part>, usize), EngError> {
    let per_expert = source.parts();
    if per_expert == 0 {
        return Err(bad("an expert must have at least one part"));
    }
    let count = ids.len().checked_mul(per_expert).ok_or_else(capacity)?;
    let mut parts = Vec::new();
    let metadata = count.checked_mul(size_of::<Part>()).ok_or_else(capacity)?;
    if metadata > scratch {
        return Err(capacity());
    }
    parts.try_reserve_exact(count).map_err(|_| capacity())?;
    let metadata = parts
        .capacity()
        .checked_mul(size_of::<Part>())
        .ok_or_else(capacity)?;
    let payload_budget = scratch.checked_sub(metadata).ok_or_else(capacity)?;
    source.prefetch_experts(ids)?;
    let mut used = 0;
    for &id in ids {
        for index in 0..per_expert {
            let (part, end) = source.with_payload(id, index, |view| {
                let bytes = view
                    .data
                    .len()
                    .checked_add(view.scales.len())
                    .ok_or_else(capacity)?;
                if bytes > payload_budget {
                    return Err(capacity());
                }
                pack_part(host, used, (id, index), view)
            })?;
            parts.push(part);
            used = end;
        }
    }
    Ok((parts, used))
}

fn pack_part(
    host: &mut PinnedBuf,
    used: usize,
    key: (ExpertId, usize),
    view: ExpertPayloadView<'_>,
) -> Result<(Part, usize), EngError> {
    if view.data.is_empty() || view.shape.contains(&0) {
        return Err(bad("expert parts need data and positive matrix dimensions"));
    }
    let start = used.checked_next_multiple_of(ALIGN).ok_or_else(capacity)?;
    let end = start.checked_add(view.data.len()).ok_or_else(capacity)?;
    let (scales, used) = if view.scales.is_empty() {
        (None, end)
    } else {
        let start = end.checked_next_multiple_of(ALIGN).ok_or_else(capacity)?;
        let end = start.checked_add(view.scales.len()).ok_or_else(capacity)?;
        (Some(start..end), end)
    };
    if used > host.len() {
        return Err(capacity());
    }
    host.write(start, view.data)?;
    if let Some(range) = &scales {
        host.write(range.start, view.scales)?;
    }
    Ok((
        Part {
            key,
            data: start..end,
            scales,
            tensor_scale: view.tensor_scale,
            shape: view.shape,
        },
        used,
    ))
}

impl CudaBankLease<'_> {
    /// Queue every consumer of the bank on this stream; capture/replay is unsupported.
    pub(crate) fn stream(&self) -> &DeviceStream {
        &self.bank.compute
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn bytes(&self) -> usize {
        self.bank.used
    }

    pub(crate) fn part(&self, id: ExpertId, index: usize) -> Result<StreamedPart<'_>, EngError> {
        let position = self
            .bank
            .parts
            .binary_search_by_key(&(id, index), |part| part.key)
            .map_err(|_| bad("the requested expert part is not in this bank lease"))?;
        let part = &self.bank.parts[position];
        Ok(StreamedPart {
            data: self
                .bank
                .device
                .range(part.data.start, part.data.end - part.data.start)?,
            scales: part
                .scales
                .as_ref()
                .map(|range| self.bank.device.range(range.start, range.end - range.start))
                .transpose()?,
            tensor_scale: part.tensor_scale,
            shape: part.shape,
        })
    }

    pub(crate) fn finish(mut self) -> Result<(), EngError> {
        let outcome = self.bank.complete();
        self.finished = true;
        outcome
    }
}

impl Drop for CudaBankLease<'_> {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.bank.complete();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CudaBankLease;
    use crate::EngError;
    use crate::expert_tier::{
        CudaStreamBanks, ExpertId, ExpertPayload, ExpertPayloadSource, ExpertPayloadView,
        Placement, Residency, StreamBankPlan,
    };
    use crate::kernels::cuda::{CudaContext, DeviceEvent, PinnedBuf};

    struct Source {
        data: Vec<Vec<u8>>,
        reads: Vec<ExpertId>,
        released: Vec<ExpertId>,
    }

    impl ExpertPayloadSource for Source {
        fn parts(&self) -> usize {
            1
        }
        fn payload(&mut self, _: ExpertId, _: usize) -> Result<ExpertPayload, EngError> {
            Err(EngError::Checkpoint("owned payload was requested".into()))
        }
        fn release_layer(&mut self, _: u32) -> Result<(), EngError> {
            Err(EngError::BadRequest(
                "whole-layer release was requested".into(),
            ))
        }
        fn with_payload<R>(
            &mut self,
            id: ExpertId,
            _: usize,
            consume: impl FnOnce(ExpertPayloadView<'_>) -> Result<R, EngError>,
        ) -> Result<R, EngError> {
            self.reads.push(id);
            let data = self
                .data
                .get(id.expert as usize)
                .ok_or_else(|| EngError::Checkpoint("unknown expert".into()))?;
            consume(ExpertPayloadView {
                data,
                scales: &[17],
                tensor_scale: 1.0,
                shape: [1, data.len() as u32],
            })
        }
        fn release_experts(&mut self, ids: &[ExpertId]) -> Result<(), EngError> {
            self.released.extend_from_slice(ids);
            Ok(())
        }
    }

    fn queue_read(
        ctx: &CudaContext,
        lease: &CudaBankLease<'_>,
        id: ExpertId,
        out: &mut PinnedBuf,
    ) -> Result<DeviceEvent, EngError> {
        let done = ctx.event()?;
        let part = lease.part(id, 0)?;
        ctx.copy_to_pinned(lease.stream(), part.data, out, 0)?;
        lease.stream().record(&done)?;
        Ok(done)
    }

    fn read(out: &mut PinnedBuf, done: &DeviceEvent) -> Result<Vec<u8>, EngError> {
        out.wait(done)?;
        let mut bytes = vec![0; out.len()];
        out.read(0, &mut bytes)?;
        Ok(bytes)
    }

    fn missing() -> EngError {
        EngError::BadRequest("expected an NVMe bank".into())
    }

    #[test]
    fn two_banks_copy_in_order_and_filter_persistent_tiers() {
        let observed = (|| -> Result<_, EngError> {
            let capture = super::super::stream_cuda_profile::Capture::start();
            let ctx = CudaContext::new()?;
            let mut banks = CudaStreamBanks::new(&ctx, StreamBankPlan::new(1024, 0, 4096)?)?;
            let mut out_a = ctx.pinned_buf(4)?;
            let mut out_b = ctx.pinned_buf(4)?;
            let mut scale = ctx.pinned_buf(1)?;
            let ids = [0, 1, 2, 3].map(|expert| ExpertId { layer: 7, expert });
            let mut placement = Residency::default();
            placement.mark_expert(7, 0, 4);
            placement.set_placement(ids[3], 4, Placement::Cpu)?;
            let mut source = Source {
                data: vec![vec![0; 4], vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9; 4]],
                reads: Vec::new(),
                released: Vec::new(),
            };
            let (a, b) = banks.slots_mut();
            let a_lease = a
                .stage(
                    &ctx,
                    &mut source,
                    &placement,
                    &[ids[0], ids[1], ids[3], ids[1]],
                )?
                .ok_or_else(missing)?;
            let bytes = a_lease.bytes();
            let metadata = (
                a_lease.part(ids[1], 0)?.shape,
                a_lease.part(ids[1], 0)?.tensor_scale,
            );
            let a_done = queue_read(&ctx, &a_lease, ids[1], &mut out_a)?;
            let a_scales = a_lease.part(ids[1], 0)?.scales.ok_or_else(missing)?;
            ctx.copy_to_pinned(a_lease.stream(), a_scales, &mut scale, 0)?;
            a_lease.stream().record(&a_done)?;
            let mut early = [0; 4];
            let premature = out_a.read(0, &mut early);
            // The first consumer is still outstanding while the second bank stages.
            let b_lease = b
                .stage(&ctx, &mut source, &placement, &[ids[2]])?
                .ok_or_else(missing)?;
            let b_done = queue_read(&ctx, &b_lease, ids[2], &mut out_b)?;
            let second = read(&mut out_b, &b_done)?;
            let first = read(&mut out_a, &a_done)?;
            let scales = read(&mut scale, &a_done)?;
            drop(a_lease);
            b_lease.finish()?;
            let again = a
                .stage(&ctx, &mut source, &placement, &[ids[2]])?
                .ok_or_else(missing)?;
            let stale = again.part(ids[1], 0).map(|_| ());
            let again_done = queue_read(&ctx, &again, ids[2], &mut out_a)?;
            let third = read(&mut out_a, &again_done)?;
            again.finish()?;
            ctx.ensure_usable()?;
            Ok((
                first,
                second,
                third,
                scales,
                premature,
                stale,
                source.reads,
                source.released,
                bytes,
                metadata,
                capture.finish()?,
            ))
        })();
        let (
            first,
            second,
            third,
            scales,
            premature,
            stale,
            reads,
            released,
            bytes,
            metadata,
            bank,
        ) = observed.expect("two owned streaming banks");
        assert_eq!(first, [1, 2, 3, 4]);
        assert_eq!(second, [5, 6, 7, 8]);
        assert_eq!(third, second);
        assert_eq!(scales, [17]);
        assert_eq!(metadata, ([1, 4], 1.0));
        assert!(matches!(premature, Err(EngError::BadRequest(_))));
        assert!(matches!(stale, Err(EngError::BadRequest(_))));
        let expected = [1, 2, 2].map(|expert| ExpertId { layer: 7, expert });
        assert_eq!(reads, expected);
        assert_eq!(released, expected);
        assert_eq!(bytes, 257);
        assert_eq!(bank.calls, 3);
        assert_eq!(bank.dma_bytes, 3 * bytes as u64);
        assert!(bank.gpu_ms.is_finite() && bank.gpu_ms >= 0.0);
        assert!(bank.event_api_host_ns > 0);
    }
}
