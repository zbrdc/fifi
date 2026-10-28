//! Budgeted native KV caching and immutable host spill for eager sessions.

use super::super::{
    dims::{KvMode, StepDims},
    paging_plan::{PagingPlan, sized},
    state_layout::{KvPool, Plane},
};
use super::{PageAttention, PageId, PageToken, TransferSlot, backing::SpillBacking};
use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceStream,
        resident::{
            attn_decode::{AttnSpan, AttnSplit},
            attn_decode_q8::{
                AttnDecodeQ8Call,
                paged::{PagedQ8Input, PagedQ8Page, PagedQ8Scratch},
            },
            kv8_append::{Kv8CodeScale, Kv8Data, Kv8Shape},
            paged_position::PagedPositionCall,
        },
    },
};
use std::{cell::RefCell, mem::size_of};

#[derive(Clone)]
pub(in super::super) struct DeviceCachePlan {
    paging: PagingPlan,
    pool: KvPool,
    cached_groups: usize,
    host_limit: usize,
}

impl DeviceCachePlan {
    /// The device budget excludes fixed state, staging, attention scratch and headroom.
    pub(in super::super) fn new(
        dims: &StepDims,
        capacity: usize,
        device_budget: usize,
        host_limit: usize,
    ) -> Result<Self, EngError> {
        let paging = PagingPlan::new(dims, capacity)?;
        let residency = paging.device_residency(device_budget)?;
        if residency.device_groups == 0 {
            return Err(EngError::MemoryPlan(
                "device cache needs at least one whole page group".into(),
            ));
        }
        let pool = KvPool::new(dims, KvMode::Int8G32, residency.device_rows)?;
        if pool.total != residency.device_bytes {
            return Err(EngError::MemoryPlan(
                "device cache padding exceeds the page budget".into(),
            ));
        }
        let plan = Self {
            paging,
            pool,
            cached_groups: residency.device_groups,
            host_limit,
        };
        if plan.required_host_bytes()? > host_limit {
            return Err(EngError::MemoryPlan(
                "device cache overflow exceeds the host quota".into(),
            ));
        }
        plan.device_bytes()?;
        Ok(plan)
    }

    pub(in super::super) fn rows(&self) -> usize {
        self.pool.capacity
    }
    pub(in super::super) fn host_limit(&self) -> usize {
        self.host_limit
    }

    pub(in super::super) fn check_geometry(
        &self,
        dims: &StepDims,
        capacity: usize,
    ) -> Result<(), EngError> {
        if capacity != self.paging.capacity
            || self.pool != KvPool::new(dims, KvMode::Int8G32, self.pool.capacity)?
        {
            return Err(EngError::BadRequest(
                "device cache plan geometry differs from state".into(),
            ));
        }
        Ok(())
    }

    fn count(&self) -> Result<usize, EngError> {
        sized(self.paging.groups.checked_mul(self.paging.full_layers))
    }

    fn fixed(&self) -> Result<usize, EngError> {
        let positions = sized(self.paging.full_layers.checked_mul(size_of::<usize>()))?;
        sized(
            self.paging
                .pinned_bytes
                .checked_add(size_of::<RefCell<DeviceFirstKv>>())
                .and_then(|bytes| bytes.checked_add(positions)),
        )
    }

    fn required_host_bytes(&self) -> Result<usize, EngError> {
        let metadata = SpillBacking::minimum_bytes(self.count()?, self.fixed()?)?;
        let spilled =
            sized((self.paging.groups - self.cached_groups).checked_mul(self.paging.full_layers))?;
        sized(metadata.checked_add(SpillBacking::payload_bytes(
            spilled,
            self.paging.group.total,
        )?))
    }

    pub(in super::super) fn device_bytes(&self) -> Result<usize, EngError> {
        sized(self.pool.total.checked_add(self.paging.device_bytes))
    }
}

pub(in super::super) struct CachedAttention<'a> {
    pub(in super::super) page: PageAttention<'a>,
    pub(in super::super) kv_len: crate::kernels::cuda::DeviceRange<'a>,
}

/// Budgeted ring cache with immutable host spill and one ordered stream.
pub(in super::super) struct DeviceFirstKv {
    cache: DeviceBuffer,
    slots: [TransferSlot; 2],
    backing: SpillBacking,
    plan: DeviceCachePlan,
    positions: Vec<usize>,
    stream_id: u64,
}

impl DeviceFirstKv {
    pub(in super::super) fn allocate(
        context: &CudaContext,
        stream: &DeviceStream,
        plan: &DeviceCachePlan,
    ) -> Result<Self, EngError> {
        context.ensure_uncaptured("device cache allocation")?;
        context.check_launch_stream(stream)?;
        let mut backing = SpillBacking::new(
            plan.count()?,
            plan.paging.group.total,
            plan.fixed()?,
            plan.host_limit,
        )?;
        let mut positions = Vec::new();
        positions
            .try_reserve_exact(plan.paging.full_layers)
            .map_err(|error| EngError::MemoryPlan(format!("device cache positions: {error}")))?;
        let extra = sized(
            (positions.capacity() - plan.paging.full_layers).checked_mul(size_of::<usize>()),
        )?;
        if sized(plan.required_host_bytes()?.checked_add(extra))? > plan.host_limit {
            return Err(EngError::MemoryPlan(
                "device cache position capacity exceeds host quota".into(),
            ));
        }
        backing.grow_fixed(extra)?;
        positions.resize(plan.paging.full_layers, 0);
        let cache = context.allocate(plan.pool.total)?;
        let slots = [
            TransferSlot::allocate(context, plan.paging.slot_bytes)?,
            TransferSlot::allocate(context, plan.paging.slot_bytes)?,
        ];
        let mut owner = Self {
            cache,
            slots,
            backing,
            plan: plan.clone(),
            positions,
            stream_id: stream.id(),
        };
        owner.zero_cache(context, stream)?;
        Ok(owner)
    }

    fn zero_cache(&mut self, context: &CudaContext, stream: &DeviceStream) -> Result<(), EngError> {
        let zeros = [0u8; 4096];
        let slot = &mut self.slots[0];
        let chunk = self.plan.paging.slot_bytes;
        for at in (0..chunk).step_by(zeros.len()) {
            slot.pinned
                .write(at, &zeros[..(chunk - at).min(zeros.len())])?;
        }
        for at in (0..self.cache.bytes()).step_by(chunk) {
            let bytes = (self.cache.bytes() - at).min(chunk);
            context.copy_from_pinned(stream, &mut slot.pinned, 0, self.cache.range(at, bytes)?)?;
            stream.record(&slot.event)?;
            slot.pinned.wait(&slot.event)?;
        }
        Ok(())
    }

    fn ready(&self, context: &CudaContext, stream: &DeviceStream) -> Result<(), EngError> {
        context.ensure_uncaptured("device cache")?;
        context.check_launch_stream(stream)?;
        if stream.id() != self.stream_id {
            return Err(EngError::BadRequest(
                "device cache belongs to another stream".into(),
            ));
        }
        Ok(())
    }

    fn gather(
        &mut self,
        context: &CudaContext,
        stream: &DeviceStream,
        slot: usize,
        page: PageId,
    ) -> Result<(), EngError> {
        self.slots
            .get_mut(slot)
            .ok_or_else(|| EngError::BadRequest("device cache slot is invalid".into()))?
            .pinned
            .synchronize(stream)?;
        let p = &self.plan.paging;
        let row = (page.group % self.plan.cached_groups) * p.group_cells;
        for &plane in p.group.planes() {
            let width = match plane {
                Plane::K | Plane::V => p.head_dim,
                Plane::KScales | Plane::VScales => p.head_dim / 32 * 2,
            };
            let (source, _) = self.plan.pool.plane_extent(page.full_layer, plane)?;
            let (destination, _) = p.group.plane_extent(0, plane)?;
            for head in 0..p.kv_heads {
                let at = source + (head * self.plan.pool.capacity + row) * width;
                let to = destination + head * p.group_cells * width;
                let bytes = p.group_cells * width;
                context.copy_device(
                    stream,
                    self.cache.range(at, bytes)?,
                    self.slots[slot].device.range(to, bytes)?,
                )?;
            }
        }
        self.slots[slot].page = Some(page);
        Ok(())
    }

    fn evict(
        &mut self,
        context: &CudaContext,
        stream: &DeviceStream,
        page: PageId,
    ) -> Result<(), EngError> {
        let index = page.full_layer * self.plan.paging.groups + page.group;
        // A refused producer may be retried after a successful physical migration.
        if self.backing.bytes(index).is_ok() {
            return Ok(());
        }
        let mut replacement = self.backing.prepare(index)?;
        self.gather(context, stream, 0, page)?;
        let slot = &mut self.slots[0];
        context.copy_to_pinned(
            stream,
            slot.device.range(0, self.plan.paging.group.total)?,
            &mut slot.pinned,
            0,
        )?;
        stream.record(&slot.event)?;
        slot.pinned.wait(&slot.event)?;
        slot.pinned.read(0, replacement.bytes_mut())?;
        self.backing.publish(index, replacement)
    }

    pub(in super::super) fn append(
        &mut self,
        context: &CudaContext,
        stream: &DeviceStream,
        call: PageToken<'_>,
    ) -> Result<(), EngError> {
        self.ready(context, stream)?;
        let layer = call.page.full_layer;
        let p = &self.plan.paging;
        if self.positions.get(layer).copied() != Some(call.position)
            || call.position >= p.capacity
            || call.page.group != call.position / p.group_cells
        {
            return Err(EngError::BadRequest(
                "device cache append is not the next logical row".into(),
            ));
        }
        if call.position >= self.plan.pool.capacity && call.position.is_multiple_of(p.group_cells) {
            self.evict(
                context,
                stream,
                PageId {
                    full_layer: layer,
                    group: call.page.group - self.plan.cached_groups,
                },
            )?;
        }
        let p = &self.plan.paging;
        let view = |plane| {
            let (offset, bytes) = self.plan.pool.plane_extent(layer, plane)?;
            self.cache.range(offset, bytes)
        };
        context.kv8_paged_append(
            stream,
            Kv8Shape {
                rows: 1,
                n_kv_head: p.kv_heads,
                head_dim: p.head_dim,
                cells: self.plan.pool.capacity,
                code_scale: Kv8CodeScale::StoredHalf,
            },
            Kv8Data {
                k: call.k,
                v: call.v,
                k_codes: view(Plane::K)?,
                k_scales: view(Plane::KScales)?,
                v_codes: view(Plane::V)?,
                v_scales: view(Plane::VScales)?,
            },
            PagedPositionCall {
                params: call.params,
                local: self.slots[0].device.range(p.group.total, 4)?,
                expected: call.position,
                base: call.position / self.plan.pool.capacity * self.plan.pool.capacity,
                capacity: p.capacity,
            },
        )?;
        self.positions[layer] = call.position + 1;
        Ok(())
    }

    fn page_view(&self, slot: usize) -> Result<PagedQ8Page<'_>, EngError> {
        let slot = self
            .slots
            .get(slot)
            .filter(|slot| slot.page.is_some())
            .ok_or_else(|| EngError::BadRequest("cache page is not staged".into()))?;
        let view = |plane| {
            let (offset, bytes) = self.plan.paging.group.plane_extent(0, plane)?;
            slot.device.range(offset, bytes)
        };
        Ok(PagedQ8Page {
            group: slot
                .page
                .ok_or_else(|| EngError::BadRequest("missing cache page".into()))?
                .group,
            k_codes: view(Plane::K)?,
            k_scales: view(Plane::KScales)?,
            v_codes: view(Plane::V)?,
            v_scales: view(Plane::VScales)?,
        })
    }

    pub(in super::super) fn attend(
        &mut self,
        context: &CudaContext,
        stream: &DeviceStream,
        scratch: &mut PagedQ8Scratch,
        call: CachedAttention<'_>,
    ) -> Result<(), EngError> {
        self.ready(context, stream)?;
        let page = call.page;
        let p = &self.plan.paging;
        if page.visible_keys == 0
            || self.positions.get(page.full_layer).copied() != Some(page.visible_keys)
        {
            return Err(EngError::BadRequest(
                "cache attention is not at the appended logical row".into(),
            ));
        }
        scratch.check_geometry(p.kv_heads, p.head_dim, p.capacity, p.group_cells)?;
        if page.visible_keys <= self.plan.pool.capacity {
            let (mut shape, part) = scratch.resident_workspace(context, stream)?;
            shape.cells = self.plan.pool.capacity;
            let view = |plane| {
                let (offset, bytes) = self.plan.pool.plane_extent(page.full_layer, plane)?;
                self.cache.range(offset, bytes)
            };
            return context.attn_decode_q8(
                stream,
                AttnDecodeQ8Call {
                    shape,
                    span: AttnSpan::FULL,
                    split: AttnSplit::Q8,
                    scale: page.scale,
                    q: page.q,
                    proj: page.proj,
                    k_codes: view(Plane::K)?,
                    k_scales: view(Plane::KScales)?,
                    v_codes: view(Plane::V)?,
                    v_scales: view(Plane::VScales)?,
                    kv_len: call.kv_len,
                    part,
                    out: page.out,
                },
            );
        }
        let cells = p.group_cells;
        let mut pass = scratch.begin(
            context,
            stream,
            PagedQ8Input {
                q: page.q,
                visible_keys: page.visible_keys,
                scale: page.scale,
            },
        )?;
        for group in 0..page.visible_keys.div_ceil(cells) {
            let slot = group % self.slots.len();
            self.load_page(
                context,
                stream,
                slot,
                PageId {
                    full_layer: page.full_layer,
                    group,
                },
            )?;
            pass.push(self.page_view(slot)?)?;
        }
        pass.finish(page.proj, page.out)
    }

    fn load_page(
        &mut self,
        context: &CudaContext,
        stream: &DeviceStream,
        slot: usize,
        page: PageId,
    ) -> Result<(), EngError> {
        self.ready(context, stream)?;
        let written = self.positions.get(page.full_layer).copied().unwrap_or(0);
        let p = &self.plan.paging;
        if written == 0 || page.group >= written.div_ceil(p.group_cells) {
            return Err(EngError::BadRequest(
                "device cache page is not visible".into(),
            ));
        }
        let first = ((written - 1) / p.group_cells + 1).saturating_sub(self.plan.cached_groups);
        if page.group >= first {
            return self.gather(context, stream, slot, page);
        }
        let index = page.full_layer * p.groups + page.group;
        let slot = self
            .slots
            .get_mut(slot)
            .ok_or_else(|| EngError::BadRequest("device cache slot is invalid".into()))?;
        slot.pinned.synchronize(stream)?;
        slot.pinned.write(0, self.backing.bytes(index)?)?;
        context.copy_from_pinned(
            stream,
            &mut slot.pinned,
            0,
            slot.device.range(0, p.group.total)?,
        )?;
        stream.record(&slot.event)?;
        slot.pinned.wait(&slot.event)?;
        slot.page = Some(page);
        Ok(())
    }

    pub(in super::super) fn reset(
        &mut self,
        context: &CudaContext,
        stream: &DeviceStream,
    ) -> Result<(), EngError> {
        self.ready(context, stream)?;
        self.invalidate(stream)
    }

    pub(in super::super) fn invalidate(&mut self, stream: &DeviceStream) -> Result<(), EngError> {
        if stream.id() != self.stream_id {
            return Err(EngError::BadRequest(
                "device cache belongs to another stream".into(),
            ));
        }
        for slot in &mut self.slots {
            slot.pinned.synchronize(stream)?;
            slot.page = None;
        }
        self.backing.clear();
        self.positions.fill(0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::qwen36::cuda::{
        common_tests::dims,
        native_device_rig::device_present,
        paging_pipeline_tests::{floats, inputs, plane, read, upload},
        paging_plan::PagingPlan,
        paging_store::{PageId, PageToken, PagedKv},
        state_layout::{KvPool, Plane},
    };
    use crate::{
        EngError,
        kernels::cuda::{
            CudaContext,
            resident::{
                kv8_append::{Kv8CodeScale, Kv8Data, Kv8Shape},
                step_params::{STEP_PARAMS_BYTES, StepParams},
            },
        },
    };

    const ROWS: usize = 1280;

    #[test]
    fn cache_plan_bounds_device_groups_and_includes_spill_ownership() {
        let dimensions = dims();
        let paging = PagingPlan::new(&dimensions, ROWS).unwrap();
        let group = paging.group.total * paging.full_layers;
        let plan = DeviceCachePlan::new(&dimensions, ROWS, 2 * group, 32 << 20).unwrap();
        assert_eq!(plan.cached_groups, 2);
        assert_eq!(plan.pool.capacity, 2 * paging.group_cells);
        assert_eq!(plan.pool.total, 2 * group);
        assert_eq!(
            plan.device_bytes().unwrap(),
            2 * group + paging.device_bytes
        );
        let required = plan.required_host_bytes().unwrap();
        assert!(required < PagedKv::minimum_host_bytes(&paging).unwrap());
        assert!(DeviceCachePlan::new(&dimensions, ROWS, 2 * group, required).is_ok());
        assert!(DeviceCachePlan::new(&dimensions, ROWS, 2 * group, required - 1).is_err());
        assert!(DeviceCachePlan::new(&dimensions, ROWS, group - 1, 32 << 20).is_err());
    }

    struct Observation {
        equal: bool,
        no_host_before_spill: bool,
        spilled: bool,
        reset: bool,
        unreadable: bool,
    }

    fn route_ring() -> Result<Observation, EngError> {
        let dimensions = dims();
        let paging = PagingPlan::new(&dimensions, ROWS)?;
        let budget = 2 * paging.group.total * paging.full_layers;
        let plan = DeviceCachePlan::new(&dimensions, ROWS, budget, 32 << 20)?;
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let mut cache = DeviceFirstKv::allocate(&ctx, &stream, &plan)?;
        let baseline = cache.backing.owned_bytes();
        let unreadable = cache
            .load_page(
                &ctx,
                &stream,
                0,
                PageId {
                    full_layer: 9,
                    group: 0,
                },
            )
            .is_err();
        let (keys, values) = inputs(ROWS);
        let keys = upload(&ctx, &stream, &floats(&keys))?;
        let values = upload(&ctx, &stream, &floats(&values))?;
        let reference_pool = KvPool::q8_staging_layer(&dimensions, ROWS)?;
        let references = [
            upload(&ctx, &stream, &vec![0; reference_pool.total])?,
            upload(&ctx, &stream, &vec![0; reference_pool.total])?,
        ];
        for (key, value, reference) in [
            (&keys, &values, &references[0]),
            (&values, &keys, &references[1]),
        ] {
            ctx.kv8_chunk(
                &stream,
                Kv8Shape {
                    rows: ROWS,
                    n_kv_head: 2,
                    head_dim: 256,
                    cells: ROWS,
                    code_scale: Kv8CodeScale::StoredHalf,
                },
                Kv8Data {
                    k: key.range(0, key.bytes())?,
                    v: value.range(0, value.bytes())?,
                    k_codes: plane(reference, &reference_pool, Plane::K)?,
                    k_scales: plane(reference, &reference_pool, Plane::KScales)?,
                    v_codes: plane(reference, &reference_pool, Plane::V)?,
                    v_scales: plane(reference, &reference_pool, Plane::VScales)?,
                },
                0,
            )?;
        }
        let params = upload(&ctx, &stream, &StepParams::new(1, 0)?.to_bytes())?;
        let row_bytes = 2 * 256 * 4;
        let mut no_host_before_spill = true;
        for position in 0..ROWS {
            params.write_at(&stream, 0, &StepParams::new(1, position as i32)?.to_bytes())?;
            for (full_layer, key, value) in [(0, &keys, &values), (9, &values, &keys)] {
                if position == plan.pool.capacity && full_layer == 0 {
                    let old = PageId {
                        full_layer,
                        group: 0,
                    };
                    cache.load_page(&ctx, &stream, 0, old)?;
                    let before = read(
                        &ctx,
                        &stream,
                        cache.slots[0].device.range(0, paging.group.total)?,
                    )?;
                    let refused = cache.append(
                        &ctx,
                        &stream,
                        PageToken {
                            page: PageId {
                                full_layer,
                                group: position / paging.group_cells,
                            },
                            position,
                            params: params.range(0, STEP_PARAMS_BYTES)?,
                            k: key.range(position * row_bytes, row_bytes - 4)?,
                            v: value.range(position * row_bytes, row_bytes)?,
                        },
                    );
                    assert!(matches!(refused, Err(EngError::BadRequest(_))));
                    assert_eq!(cache.positions[full_layer], position);
                    cache.load_page(&ctx, &stream, 0, old)?;
                    assert_eq!(
                        read(
                            &ctx,
                            &stream,
                            cache.slots[0].device.range(0, paging.group.total)?
                        )?,
                        before
                    );
                    eprintln!(
                        "refused overflow preserves the old page at {position}; retry follows"
                    );
                }
                cache.append(
                    &ctx,
                    &stream,
                    PageToken {
                        page: PageId {
                            full_layer,
                            group: position / paging.group_cells,
                        },
                        position,
                        params: params.range(0, STEP_PARAMS_BYTES)?,
                        k: key.range(position * row_bytes, row_bytes)?,
                        v: value.range(position * row_bytes, row_bytes)?,
                    },
                )?;
            }
            if position < plan.pool.capacity {
                no_host_before_spill &= cache.backing.owned_bytes() == baseline;
            }
        }
        let spilled = cache.backing.owned_bytes() > baseline;
        let references = [
            read(
                &ctx,
                &stream,
                references[0].range(0, references[0].bytes())?,
            )?,
            read(
                &ctx,
                &stream,
                references[1].range(0, references[1].bytes())?,
            )?,
        ];
        let mut equal = true;
        for (full_layer, reference) in [(0, &references[0]), (9, &references[1])] {
            for group in 0..paging.groups {
                cache.load_page(&ctx, &stream, 0, PageId { full_layer, group })?;
                let actual = read(
                    &ctx,
                    &stream,
                    cache.slots[0].device.range(0, paging.group.total)?,
                )?;
                for &plane in paging.group.planes() {
                    let width = match plane {
                        Plane::K | Plane::V => 256,
                        Plane::KScales | Plane::VScales => 256 / 32 * 2,
                    };
                    let (local, _) = paging.group.plane_extent(0, plane)?;
                    let (global, _) = reference_pool.plane_extent(0, plane)?;
                    for head in 0..2 {
                        let a = local + head * paging.group_cells * width;
                        let b = global + (head * ROWS + group * paging.group_cells) * width;
                        let n = paging.group_cells * width;
                        equal &= actual[a..a + n] == reference[b..b + n];
                    }
                }
            }
        }
        cache.reset(&ctx, &stream)?;
        let reset = cache.backing.owned_bytes() == baseline
            && cache
                .load_page(
                    &ctx,
                    &stream,
                    0,
                    PageId {
                        full_layer: 9,
                        group: 0,
                    },
                )
                .is_err();
        Ok(Observation {
            equal,
            no_host_before_spill,
            spilled,
            reset,
            unreadable,
        })
    }

    #[test]
    #[ignore = "requires the isolated standing GPU lease"]
    fn ring_cache_spills_oldest_and_matches_native_for_1280_positions() {
        if !device_present("device-first ring cache routing") {
            return;
        }
        let observed = route_ring().expect("native device-cache routing");
        assert!(
            observed.equal,
            "all four planes must match the unchanged native producer"
        );
        assert!(
            observed.no_host_before_spill,
            "resident rows must not allocate host payloads"
        );
        assert!(observed.spilled && observed.reset && observed.unreadable);
        eprintln!(
            "DEVICE_CACHE_RING positions=1280 layers=2 pages=5 cached=2 byte_exact=true reset=true"
        );
    }
}
