//! The fixed pool of expert page sets of the resident decoder (S2 plan sections 4.3 and 5).
//!
//! One set holds a layer's gate, up, and down expert stacks and its gate weights. The pool is
//! allocated once with the decoder and never grows. A set is lent when its stacks are imported,
//! and released after the evaluation that consumes them (`release_lent`):
//!
//! - Layer `L`'s routed graph is an ancestor of layer `L + 1`'s router, so `moe::step_paged` releases
//!   every lent set right after its router readback.
//! - The step releases the rest when it returns. `Generate` and `Record` have evaluated the last
//!   layer's graph by then. In `Prompt`, and after an error, the remaining graphs are dropped
//!   unevaluated, so no evaluation can read or write the pages any more.
//!
//! A released set becomes writable again only after the next successful synchronization on the
//! engine thread frees its residents (decision 0041). `acquire` therefore does not wait: nothing
//! else can free a set while it runs. With the release points above, the set of layer `L - 2` is
//! always free at layer `L`, so two sets suffice and the third is slack.

use super::super::stored::{RoleRange, StackLayout};
use super::weights::RMoe;
use crate::{
    EngError,
    kernels::{HostPages, PackedPages, PageRange},
    tensor::Dtype,
    weights::StreamedCheckpoint,
};

/// Sets in the pool (coordinator answer Q4).
pub(super) const POOL_SETS: usize = 3;

/// The page layout of one layer set: three expert stacks and the F32 gates.
pub(super) struct SetLayout {
    pub(super) stacks: [StackLayout; 3],
    pub(super) gates: PageRange,
    /// Bytes from the set start to the end of the gates page.
    pub(super) end: usize,
}

impl SetLayout {
    /// Places the gate, up, and down stacks of `w` with `count` experts, then the gates.
    pub(super) fn new(
        ckpt: &StreamedCheckpoint,
        w: &RMoe,
        count: u32,
        page: usize,
    ) -> Result<Self, EngError> {
        let gate = StackLayout::new(ckpt, &w.gate, count, 0, page)?;
        let up = StackLayout::new(ckpt, &w.up, count, gate.end, page)?;
        let down = StackLayout::new(ckpt, &w.down, count, up.end, page)?;
        let gates_offset = down.end;
        let overflow = || EngError::BadRequest("expert set layout overflows".into());
        let gates_len = usize::try_from(count)
            .ok()
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(overflow)?;
        let end = gates_offset
            .checked_add(gates_len)
            .and_then(|end| end.checked_next_multiple_of(page))
            .ok_or_else(overflow)?;
        Ok(Self {
            stacks: [gate, up, down],
            gates: PageRange {
                offset: gates_offset,
                dtype: Dtype::F32,
                shape: vec![count],
            },
            end,
        })
    }

    /// The page ranges of the three stacks for `import_packed_pages`.
    pub(super) fn packed(&self) -> Vec<PackedPages> {
        self.stacks
            .iter()
            .map(|stack| PackedPages {
                payload: page_range(&stack.payload),
                scales: page_range(&stack.scales),
                biases: stack.biases.as_ref().map(page_range),
                logical_shape: stack.logical_shape.clone(),
                quantization: stack.spec,
            })
            .collect()
    }
}

fn page_range(range: &RoleRange) -> PageRange {
    PageRange {
        offset: range.offset,
        dtype: range.dtype,
        shape: range.shape.clone(),
    }
}

/// A fixed pool of page sets. Index `i` names set `i`.
pub(super) struct ExpertSlots {
    sets: Vec<HostPages>,
    lent: Vec<usize>,
}

impl ExpertSlots {
    /// Maps `count` sets of `set_bytes` each (rounded up to the page size).
    pub(super) fn new(count: usize, set_bytes: usize) -> Result<Self, EngError> {
        if count == 0 {
            return Err(EngError::BadRequest(
                "the expert pool needs at least one set".into(),
            ));
        }
        let sets = (0..count)
            .map(|_| HostPages::try_new(set_bytes))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            sets,
            lent: Vec::new(),
        })
    }

    /// Mapped bytes of the whole pool. They do not change after `new`.
    pub(super) fn byte_len(&self) -> u64 {
        self.sets
            .iter()
            .map(|set| u64::try_from(set.len()).unwrap_or(u64::MAX))
            .fold(0, u64::saturating_add)
    }

    /// Returns the first set whose pages are writable, without waiting.
    ///
    /// Only a synchronization on this thread frees released sets, so a wait cannot help. When no
    /// set is free it returns `Backend` at once: a release was missed, or the pool is too small.
    pub(super) fn acquire(&mut self) -> Result<usize, EngError> {
        self.sets
            .iter_mut()
            .position(|set| set.try_write().is_some())
            .ok_or_else(|| EngError::Backend {
                operation: "expert_slots",
                message: format!(
                    "no expert page set is free among {}: a release was missed or the pool is \
                     too small",
                    self.sets.len()
                ),
            })
    }

    /// The writable bytes of an acquired set.
    pub(super) fn write(&mut self, index: usize) -> Result<&mut [u8], EngError> {
        self.sets
            .get_mut(index)
            .and_then(HostPages::try_write)
            .ok_or_else(|| EngError::Backend {
                operation: "expert_slots",
                message: format!("expert page set {index} is not writable"),
            })
    }

    pub(super) fn pages(&self, index: usize) -> Result<&HostPages, EngError> {
        self.sets
            .get(index)
            .ok_or_else(|| EngError::BadRequest(format!("expert page set {index} is absent")))
    }

    /// Records that residents over set `index` now feed a graph that is not yet evaluated.
    pub(super) fn lend(&mut self, index: usize) {
        if !self.lent.contains(&index) {
            self.lent.push(index);
        }
    }

    /// Releases every lent set. Call it only after an evaluation that covers every graph that
    /// reads them, or when those graphs are dropped and will never be evaluated.
    pub(super) fn release_lent(&mut self) {
        for index in self.lent.drain(..) {
            if let Some(set) = self.sets.get_mut(index) {
                set.release();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::MlxKernels;
    use crate::tensor::Tensor;

    fn anchor(kernels: &mut MlxKernels) -> crate::tensor::ExecutionTensor {
        let zero = Tensor::try_from_host(Dtype::F32, vec![1], vec![0; 4]).unwrap();
        kernels.import_tensor(&zero).unwrap()
    }

    /// Writes one byte into set `index` and imports it as a lent resident.
    fn lend_one(
        slots: &mut ExpertSlots,
        kernels: &mut MlxKernels,
        index: usize,
    ) -> crate::tensor::ExecutionTensor {
        slots.write(index).unwrap()[0] = 7;
        let range = PageRange {
            offset: 0,
            dtype: Dtype::U8,
            shape: vec![1],
        };
        let resident = kernels
            .import_pages(slots.pages(index).unwrap(), &[range])
            .unwrap()
            .pop()
            .unwrap();
        slots.lend(index);
        resident
    }

    #[test]
    fn acquire_skips_lent_sets_and_reuses_released_ones() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let anchor = anchor(&mut kernels);
        let page = HostPages::page_size();
        let mut slots = ExpertSlots::new(2, page).unwrap();
        assert_eq!(
            slots.byte_len(),
            2 * u64::try_from(page).unwrap(),
            "pool bytes"
        );

        let first = slots.acquire().unwrap();
        let held = lend_one(&mut slots, &mut kernels, first);
        let second = slots.acquire().unwrap();
        assert_ne!(second, first, "a lent set is not acquired again");
        let other = lend_one(&mut slots, &mut kernels, second);

        let exhausted = slots.acquire();
        assert!(
            matches!(
                exhausted,
                Err(EngError::Backend {
                    operation: "expert_slots",
                    ..
                })
            ),
            "{exhausted:?}"
        );

        drop(held);
        drop(other);
        slots.release_lent();
        kernels.evaluate(&[&anchor]).unwrap();
        assert_eq!(
            slots.acquire().unwrap(),
            first,
            "the first set returns after release and flush"
        );
    }
}
