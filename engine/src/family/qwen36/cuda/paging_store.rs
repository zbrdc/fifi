//! Native paged-KV ownership and bounded allocation.

use std::{cell::Cell, mem::size_of};

mod backing;
pub(super) mod device_first;
mod state_copy;
use backing::Backing;
pub(super) use state_copy::HostCharge;

use super::{dims::StepDims, paging_plan::PagingPlan, state_layout::Plane};
use crate::{
    EngError,
    kernels::cuda::resident::{
        attn_decode_q8::paged::{PagedQ8Input, PagedQ8Page, PagedQ8Scratch},
        kv8_append::{Kv8CodeScale, Kv8Data, Kv8Shape},
        paged_position::PagedPositionCall,
    },
    kernels::cuda::{CudaContext, DeviceBuffer, DeviceEvent, DeviceRange, DeviceStream, PinnedBuf},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PageId {
    /// Full-attention ordinal, not the original transformer block index.
    pub(super) full_layer: usize,
    pub(super) group: usize,
}

struct TransferSlot {
    page: Option<PageId>,
    dirty: Cell<bool>,
    device: DeviceBuffer,
    pinned: PinnedBuf,
    event: DeviceEvent,
}

impl TransferSlot {
    fn allocate(context: &CudaContext, bytes: usize) -> Result<Self, EngError> {
        Ok(Self {
            page: None,
            dirty: Cell::new(false),
            device: context.allocate(bytes)?,
            pinned: context.pinned_buf(bytes)?,
            event: context.event()?,
        })
    }
}

/// Two separately allocated transfer slots and the complete host backing.
/// A slot's byte count is not an aligned stride for packing the slots together.
pub(super) struct PagedKv {
    slots: [TransferSlot; 2],
    backing: Backing,
    plan: PagingPlan,
    logical_capacity: usize,
    full_layers: usize,
    kv_heads: usize,
    head_dim: usize,
    stream_id: Option<u64>,
}

impl PagedKv {
    pub(super) fn allocate(
        context: &CudaContext,
        dims: &StepDims,
        capacity: usize,
        host_limit_bytes: usize,
    ) -> Result<Self, EngError> {
        context.ensure_usable()?;
        context.ensure_uncaptured("paged KV allocation")?;
        let plan = PagingPlan::new(dims, capacity)?;
        Self::allocate_planned(context, &plan, host_limit_bytes)
    }

    fn backing_shape(plan: &PagingPlan) -> Result<(usize, usize), EngError> {
        let count = plan
            .groups
            .checked_mul(plan.full_layers)
            .ok_or_else(|| EngError::MemoryPlan("paged KV page count overflows".to_owned()))?;
        // Production owns a boxed RefCell, including its borrow flag.
        let fixed = plan
            .pinned_bytes
            .checked_add(size_of::<std::cell::RefCell<Self>>())
            .ok_or_else(|| EngError::MemoryPlan("paged KV fixed host size overflows".to_owned()))?;
        Ok((count, fixed))
    }

    pub(super) fn minimum_host_bytes(plan: &PagingPlan) -> Result<usize, EngError> {
        let (count, fixed) = Self::backing_shape(plan)?;
        Backing::minimum_bytes(count, plan.group.total, fixed)
    }

    pub(super) fn allocate_planned(
        context: &CudaContext,
        plan: &PagingPlan,
        host_limit_bytes: usize,
    ) -> Result<Self, EngError> {
        context.ensure_usable()?;
        context.ensure_uncaptured("paged KV allocation")?;
        let (count, fixed) = Self::backing_shape(plan)?;
        let backing = Backing::new(count, plan.group.total, fixed, host_limit_bytes)?;
        let slots = [
            TransferSlot::allocate(context, plan.slot_bytes)?,
            TransferSlot::allocate(context, plan.slot_bytes)?,
        ];
        Ok(Self {
            slots,
            backing,
            full_layers: plan.full_layers,
            kv_heads: plan.kv_heads,
            head_dim: plan.head_dim,
            plan: plan.clone(),
            logical_capacity: plan.capacity,
            stream_id: None,
        })
    }

    /// Loads a vacant slot. Eviction needs a separate consumer-completion/writeback proof.
    pub(super) fn load(
        &mut self,
        context: &CudaContext,
        stream: &DeviceStream,
        slot: usize,
        page: PageId,
    ) -> Result<(), EngError> {
        self.ensure_stream(stream)?;
        if page.full_layer >= self.full_layers || page.group >= self.plan.groups {
            return Err(EngError::BadRequest(
                "paged KV page is outside the backing".to_owned(),
            ));
        }
        if self.slots.iter().any(|slot| slot.page == Some(page)) {
            return Err(EngError::BadRequest(
                "paged KV page is already resident".to_owned(),
            ));
        }
        let slot = self.slots.get_mut(slot).ok_or_else(|| {
            EngError::BadRequest("paged KV slot is outside the two-slot store".to_owned())
        })?;
        if slot.page.is_some() {
            return Err(EngError::BadRequest(
                "paged KV slot needs writeback before replacement".to_owned(),
            ));
        }
        // PagingPlan checked the complete layer/group product before allocating backing.
        let index = page.full_layer * self.plan.groups + page.group;
        slot.pinned.write(0, self.backing.bytes(index))?;
        slot.pinned
            .write(self.plan.group.total, &0_i32.to_le_bytes())?;
        context.copy_from_pinned(
            stream,
            &mut slot.pinned,
            0,
            slot.device.range(0, self.plan.slot_bytes)?,
        )?;
        stream.record(&slot.event)?;
        slot.pinned.wait(&slot.event)?;
        slot.page = Some(page);
        slot.dirty.set(false);
        self.stream_id = Some(stream.id());
        Ok(())
    }

    fn ensure_stream(&self, stream: &DeviceStream) -> Result<(), EngError> {
        if self.stream_id.is_some_and(|id| id != stream.id()) {
            return Err(EngError::BadRequest(
                "paged KV work belongs to another stream".to_owned(),
            ));
        }
        Ok(())
    }

    /// Consumers must run on the owning stream, or be joined into it before this call.
    pub(super) fn writeback(
        &mut self,
        context: &CudaContext,
        stream: &DeviceStream,
        slot: usize,
    ) -> Result<(), EngError> {
        self.ensure_stream(stream)?;
        let slot = self.slots.get_mut(slot).ok_or_else(|| {
            EngError::BadRequest("paged KV slot is outside the two-slot store".to_owned())
        })?;
        let page = slot
            .page
            .ok_or_else(|| EngError::BadRequest("paged KV slot is not loaded".to_owned()))?;
        context.ensure_uncaptured("paged KV writeback")?;
        context.check_launch_stream(stream)?;
        if !slot.dirty.get() {
            slot.pinned.synchronize(stream)?;
            slot.page = None;
            return Ok(());
        }
        let index = page.full_layer * self.plan.groups + page.group;
        let mut replacement = self.backing.detach(index)?;
        context.copy_to_pinned(
            stream,
            slot.device.range(0, self.plan.group.total)?,
            &mut slot.pinned,
            0,
        )?;
        // This new mark covers preceding kernel consumers AND the ordered D2H copy.
        stream.record(&slot.event)?;
        slot.pinned.wait(&slot.event)?;
        if let Some(page) = &mut replacement {
            slot.pinned.read(0, page.bytes_mut())?;
        } else {
            slot.pinned.read(0, self.backing.unique_bytes(index)?)?;
        }
        self.backing.publish(index, replacement);
        slot.page = None;
        slot.dirty.set(false);
        Ok(())
    }

    /// Exposes a potentially writable view; internal attention uses a clean private view.
    pub(super) fn resident(&self, slot: usize) -> Result<DeviceRange<'_>, EngError> {
        let slot = self
            .slots
            .get(slot)
            .filter(|slot| slot.page.is_some())
            .ok_or_else(|| EngError::BadRequest("paged KV slot is not loaded".to_owned()))?;
        let range = slot.device.range(0, slot.device.bytes())?;
        slot.dirty.set(true);
        Ok(range)
    }

    /// Requested physical KV device bytes, frozen for this owner's lifetime.
    pub(super) fn device_bytes(&self) -> usize {
        self.plan.device_bytes
    }

    /// Drains the owning stream and retires pending pin use before forgetting cached pages.
    /// Reset keeps stale backing; committed visibility restarts at zero in StateBook.
    pub(super) fn invalidate(&mut self, stream: &DeviceStream) -> Result<(), EngError> {
        self.ensure_stream(stream)?;
        for slot in &mut self.slots {
            slot.pinned.synchronize(stream)?;
            slot.page = None;
            slot.dirty.set(false);
        }
        self.stream_id = Some(stream.id());
        Ok(())
    }

    /// Host backing actually held by this owner, excluding the two pinned slots.
    pub(super) fn backing_bytes(&self) -> usize {
        self.plan.backing_bytes
    }
}

/// A decode row with an expected host position and authoritative device StepParams.
pub(super) struct PageToken<'a> {
    pub(super) page: PageId,
    pub(super) position: usize,
    pub(super) params: DeviceRange<'a>,
    pub(super) k: DeviceRange<'a>,
    pub(super) v: DeviceRange<'a>,
}

impl PagedKv {
    pub(super) fn append_decode(
        &mut self,
        context: &CudaContext,
        stream: &DeviceStream,
        slot: usize,
        call: PageToken<'_>,
    ) -> Result<(), EngError> {
        context.ensure_uncaptured("paged KV decode append")?;
        self.ensure_stream(stream)?;
        let loaded = self.slots.get(slot).and_then(|s| s.page);
        if loaded != Some(call.page)
            || call.position >= self.logical_capacity
            || call.position / self.plan.group_cells != call.page.group
        {
            return Err(EngError::BadRequest(
                "paged decode is outside its loaded page or logical capacity".to_owned(),
            ));
        }
        let page = self.page(slot)?;
        let was_dirty = self.slots[slot].dirty.replace(true);
        let result = context.kv8_paged_append(
            stream,
            Kv8Shape {
                rows: 1,
                n_kv_head: self.kv_heads,
                head_dim: self.head_dim,
                cells: self.plan.group_cells,
                code_scale: Kv8CodeScale::StoredHalf,
            },
            Kv8Data {
                k: call.k,
                v: call.v,
                k_codes: page.k_codes,
                k_scales: page.k_scales,
                v_codes: page.v_codes,
                v_scales: page.v_scales,
            },
            PagedPositionCall {
                params: call.params,
                local: self.slots[slot].device.range(self.plan.group.total, 4)?,
                expected: call.position,
                base: call.page.group * self.plan.group_cells,
                capacity: self.logical_capacity,
            },
        );
        if matches!(&result, Err(EngError::BadRequest(_))) {
            self.slots[slot].dirty.set(was_dirty);
        }
        result
    }
}

/// A real prefill chunk whose global rows stay within one loaded page.
/// Decode's device control-word entry is a separate integration seam.
pub(super) struct PageChunk<'a> {
    pub(super) page: PageId,
    pub(super) position: usize,
    pub(super) rows: usize,
    pub(super) k: DeviceRange<'a>,
    pub(super) v: DeviceRange<'a>,
}

impl PagedKv {
    /// Quantizes post-norm/post-RoPE rows through the existing native chunk producer.
    pub(super) fn append_chunk(
        &mut self,
        context: &CudaContext,
        stream: &DeviceStream,
        slot: usize,
        call: PageChunk<'_>,
    ) -> Result<(), EngError> {
        context.ensure_uncaptured("paged KV chunk append")?;
        self.ensure_stream(stream)?;
        let loaded = self.slots.get(slot).and_then(|s| s.page);
        let local = call.position % self.plan.group_cells;
        if loaded != Some(call.page)
            || call.rows == 0
            || call
                .position
                .checked_add(call.rows)
                .is_none_or(|end| end > self.logical_capacity)
            || call.position / self.plan.group_cells != call.page.group
            || call.rows > self.plan.group_cells - local
        {
            return Err(EngError::BadRequest(
                "paged KV chunk is outside its loaded page or logical capacity".to_owned(),
            ));
        }
        let page = self.page(slot)?;
        let was_dirty = self.slots[slot].dirty.replace(true);
        let result = context.kv8_chunk(
            stream,
            Kv8Shape {
                rows: call.rows,
                n_kv_head: self.kv_heads,
                head_dim: self.head_dim,
                cells: self.plan.group_cells,
                code_scale: Kv8CodeScale::StoredHalf,
            },
            Kv8Data {
                k: call.k,
                v: call.v,
                k_codes: page.k_codes,
                k_scales: page.k_scales,
                v_codes: page.v_codes,
                v_scales: page.v_scales,
            },
            local,
        );
        // A refused launch did not mutate the page; uncertain native failure remains dirty.
        if matches!(&result, Err(EngError::BadRequest(_))) {
            self.slots[slot].dirty.set(was_dirty);
        }
        result
    }

    /// The loaded page's authoritative group and four physical plane views.
    fn page(&self, slot: usize) -> Result<PagedQ8Page<'_>, EngError> {
        let page = self
            .slots
            .get(slot)
            .and_then(|s| s.page)
            .ok_or_else(|| EngError::BadRequest("paged KV slot is not loaded".to_owned()))?;
        let allocation = self.slots[slot].device.range(0, self.plan.slot_bytes)?;
        let view = |plane| {
            let (offset, bytes) = self.plan.group.plane_extent(0, plane)?;
            allocation.buffer().range(offset, bytes)
        };
        Ok(PagedQ8Page {
            group: page.group,
            k_codes: view(Plane::K)?,
            k_scales: view(Plane::KScales)?,
            v_codes: view(Plane::V)?,
            v_scales: view(Plane::VScales)?,
        })
    }
}

/// One full-attention layer's immutable request. visible_keys is a positive global count.
pub(super) struct PageAttention<'a> {
    pub(super) full_layer: usize,
    pub(super) visible_keys: usize,
    pub(super) scale: f32,
    pub(super) q: DeviceRange<'a>,
    pub(super) proj: Option<DeviceRange<'a>>,
    pub(super) out: DeviceRange<'a>,
}

impl PagedKv {
    /// Owns the whole store-bound page iteration; no caller-supplied page can enter.
    pub(super) fn attend(
        &mut self,
        context: &CudaContext,
        stream: &DeviceStream,
        scratch: &mut PagedQ8Scratch,
        call: PageAttention<'_>,
    ) -> Result<(), EngError> {
        context.ensure_uncaptured("store-bound paged attention")?;
        self.ensure_stream(stream)?;
        if call.full_layer >= self.full_layers
            || call.visible_keys == 0
            || call.visible_keys > self.logical_capacity
        {
            return Err(EngError::BadRequest(
                "paged attention is outside its store layer or logical capacity".to_owned(),
            ));
        }
        scratch.check_geometry(
            self.kv_heads,
            self.head_dim,
            self.logical_capacity,
            self.plan.group_cells,
        )?;
        let mut pass = scratch.begin(
            context,
            stream,
            PagedQ8Input {
                q: call.q,
                visible_keys: call.visible_keys,
                scale: call.scale,
            },
        )?;
        for slot in 0..self.slots.len() {
            if self.slots[slot].page.is_some() {
                self.writeback(context, stream, slot)?;
            }
        }
        for group in 0..call.visible_keys.div_ceil(self.plan.group_cells) {
            let slot = group % self.slots.len();
            self.load(
                context,
                stream,
                slot,
                PageId {
                    full_layer: call.full_layer,
                    group,
                },
            )?;
            pass.push(self.page(slot)?)?;
            self.writeback(context, stream, slot)?;
        }
        pass.finish(call.proj, call.out)
    }
}

/// Immutable committed page range; clones share its charged metadata and allocation holds.
#[derive(Clone)]
pub(super) struct PageSnapshot {
    pages: backing::Snapshot,
    geometry: [usize; 5],
    visible_positions: usize,
}

impl PagedKv {
    fn snapshot_geometry(&self) -> [usize; 5] {
        [
            self.logical_capacity,
            self.plan.group_cells,
            self.full_layers,
            self.kv_heads,
            self.head_dim,
        ]
    }

    pub(super) fn snapshot(
        &mut self,
        context: &CudaContext,
        stream: &DeviceStream,
        visible_positions: usize,
    ) -> Result<PageSnapshot, EngError> {
        context.ensure_uncaptured("paged KV snapshot")?;
        context.check_launch_stream(stream)?;
        self.ensure_stream(stream)?;
        if visible_positions > self.logical_capacity {
            return Err(EngError::BadRequest(
                "paged snapshot exceeds logical capacity".to_owned(),
            ));
        }
        for slot in 0..self.slots.len() {
            if self.slots[slot].page.is_some() {
                self.writeback(context, stream, slot)?;
            }
        }
        self.slots[0].pinned.synchronize(stream)?;
        let pages = self.backing.snapshot(
            self.full_layers,
            self.plan.groups,
            visible_positions.div_ceil(self.plan.group_cells),
        )?;
        self.stream_id = Some(stream.id());
        Ok(PageSnapshot {
            pages,
            geometry: self.snapshot_geometry(),
            visible_positions,
        })
    }

    pub(super) fn check_restore(
        &self,
        stream: &DeviceStream,
        snapshot: &PageSnapshot,
    ) -> Result<(), EngError> {
        self.ensure_stream(stream)?;
        if snapshot.geometry != self.snapshot_geometry()
            || snapshot.visible_positions > self.logical_capacity
            || !self.backing.same_domain(&snapshot.pages)
        {
            return Err(EngError::BadRequest(
                "paged snapshot geometry or ownership differs".to_owned(),
            ));
        }
        Ok(())
    }

    pub(super) fn restore(
        &mut self,
        stream: &DeviceStream,
        snapshot: &PageSnapshot,
    ) -> Result<(), EngError> {
        self.check_restore(stream, snapshot)?;
        self.invalidate(stream)?;
        self.backing
            .restore(&snapshot.pages, self.full_layers, self.plan.groups);
        Ok(())
    }

    pub(super) fn owned_host_bytes(&self) -> usize {
        self.backing.owned_bytes()
    }
}

#[cfg(test)]
mod owner_layout_tests {
    use super::{Backing, PagedKv};
    use crate::family::{
        project::cuda::WeightLayout,
        qwen36::cuda::{
            common_tests::dims,
            dims::{KvMode, StepOptions},
            paging_plan::PagingPlan,
            state::PagedStatePlan,
        },
    };
    use std::{cell::RefCell, mem::size_of};

    #[test]
    fn host_quota_includes_the_boxed_refcell_owner() {
        let dims = dims();
        let options = StepOptions {
            kv_mode: KvMode::Int8G32,
            context_tokens: 320,
            delta_slots: 1,
            forked_experts: false,
            tiered: None,
            layout: WeightLayout::Native,
        };
        let p = PagingPlan::new(&dims, options.context_tokens).unwrap();
        let count = p.groups * p.full_layers;
        let owner = size_of::<RefCell<PagedKv>>();
        assert!(owner > size_of::<PagedKv>());
        let fixed = p.pinned_bytes + owner;
        let expected = Backing::minimum_bytes(count, p.group.total, fixed).unwrap();
        assert_eq!(PagedKv::minimum_host_bytes(&p).unwrap(), expected);
        assert!(PagedStatePlan::new(&dims, &options, expected - 1).is_err());
        assert!(PagedStatePlan::new(&dims, &options, expected).is_ok());
        assert!(Backing::new(count, p.group.total, fixed, expected - 1).is_err());
        let backing = Backing::new(count, p.group.total, fixed, expected).unwrap();
        assert_eq!(backing.owned_bytes(), expected);
    }
}
