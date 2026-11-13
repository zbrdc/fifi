//! The chunk-facing contract over the bank: what a prefill chunk does to the expert bank,
//! formalized as one type so the chunked prefill paths of both families code against the same
//! shape. The served path of Flash-Next (`moe_block::run_rows`) still carries its own inline
//! copy of this sequence; the two can drift until the shared prefill pass routes `run_rows`
//! through the adapter. The adapter changes nothing that runs today.
//!
//! The sequence of one chunk, per `run_rows`:
//!
//! 1. The router runs over every row and its picks come back to the host once (the readback
//!    the chunk path already pays for its own classify work).
//! 2. [`AdmitPlan::of`] decides the chunk admit: the union of the chunk's picks, whether the
//!    union fits the bank's slots, and how many of the union are not resident.
//! 3. If the union fits and anything is missing, the caller drains its own stream, then
//!    [`PrefillAdapter::admit_union`] admits the union in one pass. An admit that does not
//!    run returns `None`; the per-row step tail is the caller's fallback, also for an
//!    over-capacity union (the shared pass keeps that rule until it needs the policy's
//!    keep-last-slots admit, which `DeviceBank::admit` already accepts).
//! 4. Whenever the returned copies have ops, the caller uploads the bank's slot row again
//!    before the next classify.
//!
//! The ring path streams cold experts without admitting: `place_of`, `stream_place` and
//! `synchronize_store` are forwarded with their contracts, so a ring window never has to reach
//! around the adapter.

use super::moe_block::pick_union;

/// The chunk-admit decision of one layer: the union of the chunk's pick rows in first-seen
/// order, whether that union fits the bank's slots, and how many of the union are not
/// resident. An admit runs only when `fits && missing > 0`.
///
/// Ids must be inside `0..experts`: `pick_union` indexes its seen table by the id and panics
/// otherwise, and `slot_row` is indexed the same way. The served caller refuses faults (-1)
/// and out-of-range ids before this (`route_rows`), and [`PrefillAdapter::admit_plan`] refuses
/// them itself.
///
/// Pure host data: it is computed from a slot row, so the host tests pin it without a card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::family::flash_next) struct AdmitPlan {
    /// The chunk's distinct picks, in first-seen order across the rows.
    pub(in crate::family::flash_next) union: Vec<u16>,
    /// Whether `union` fits the bank's slots; an over-capacity union is never admitted whole.
    pub(in crate::family::flash_next) fits: bool,
    /// How many experts of `union` the slot row does not hold.
    pub(in crate::family::flash_next) missing: usize,
}

impl AdmitPlan {
    /// The decision over one layer's slot row. `slot_row` is the bank's row of `layer`
    /// (`DeviceBank::slot_row`), `experts` and `slots` its geometry.
    pub(in crate::family::flash_next) fn of(
        picks_rows: &[&[i32]],
        experts: usize,
        slots: usize,
        slot_row: &[i32],
    ) -> Self {
        let union = pick_union(picks_rows, experts);
        let missing = union
            .iter()
            .filter(|&&id| slot_row[usize::from(id)] < 0)
            .count();
        Self {
            fits: union.len() <= slots,
            missing,
            union,
        }
    }

    /// Whether the chunk admit of this plan runs.
    pub(in crate::family::flash_next) fn admits(&self) -> bool {
        self.fits && self.missing > 0
    }
}

/// The device half of the contract: [`DeviceBank`](super::bank::device::DeviceBank) behind the
/// chunk-facing methods. Borrowing, additive: the served path keeps constructing the bank it
/// constructs today; a caller that wants the contract constructs this over it.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(in crate::family::flash_next) mod device {
    use super::super::bank::{BankGeometry, BankStats, PassCopies, device::DeviceBank};
    use super::AdmitPlan;
    use crate::{
        EngError,
        kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream},
    };

    pub(in crate::family::flash_next) struct PrefillAdapter<'a> {
        bank: &'a mut DeviceBank,
    }

    impl<'a> PrefillAdapter<'a> {
        pub(in crate::family::flash_next) fn new(bank: &'a mut DeviceBank) -> Self {
            Self { bank }
        }

        /// The chunk-admit decision of `layer` for the chunk's pick rows. Faults (-1) and ids
        /// outside the layer are refused here, so a caller that cannot guarantee its router
        /// output still cannot panic the union.
        pub(in crate::family::flash_next) fn admit_plan(
            &self,
            layer: usize,
            picks_rows: &[&[i32]],
        ) -> Result<AdmitPlan, EngError> {
            let geometry = self.bank.geometry();
            for row in picks_rows {
                if let Some(&id) = row
                    .iter()
                    .find(|&&id| id < 0 || id >= geometry.experts as i32)
                {
                    return Err(EngError::BadRequest(format!(
                        "the chunk admit takes picks inside 0..{}, got {id}",
                        geometry.experts
                    )));
                }
            }
            Ok(AdmitPlan::of(
                picks_rows,
                geometry.experts,
                geometry.slots,
                self.bank.slot_row(layer)?,
            ))
        }

        /// Runs a plan `admit_plan` already computed, so a caller that drains its stream only
        /// when an admit will run does not pay the union twice. `None` when the plan does not
        /// admit. Same contract as [`Self::admit_union`]: copies on the bank's own stream, the
        /// caller drained first, re-upload [`Self::slot_row`] when ops are returned.
        pub(in crate::family::flash_next) fn admit_planned(
            &mut self,
            ctx: &CudaContext,
            layer: usize,
            plan: &AdmitPlan,
        ) -> Result<Option<PassCopies>, EngError> {
            if !plan.admits() {
                return Ok(None);
            }
            Ok(Some(self.bank.admit(ctx, layer, &plan.union)?))
        }

        /// Admits the chunk's union in one pass when the plan admits; `None` when it does not
        /// (the union does not fit, or everything is already resident) — the caller's fallback
        /// is the per-row [`Self::step`] tail. A caller that wants to drain its own stream only
        /// when an admit will run computes [`Self::admit_plan`] first and uses
        /// [`Self::admit_planned`] instead, so the union is not computed twice. The copies run
        /// on the bank's own stream and the call returns when they have finished; the caller
        /// must drain its own stream first, exactly as `DeviceBank::step` requires. When the
        /// returned copies have ops, upload [`Self::slot_row`] again before the next classify.
        pub(in crate::family::flash_next) fn admit_union(
            &mut self,
            ctx: &CudaContext,
            layer: usize,
            picks_rows: &[&[i32]],
        ) -> Result<Option<PassCopies>, EngError> {
            let plan = self.admit_plan(layer, picks_rows)?;
            self.admit_planned(ctx, layer, &plan)
        }

        /// The per-row fallback: one decode-shaped pass for the rows the union admit did not
        /// cover. Same contract as `DeviceBank::step` — copies on the bank's stream, the
        /// caller drains its own stream first.
        pub(in crate::family::flash_next) fn step(
            &mut self,
            ctx: &CudaContext,
            layer: usize,
            picks: &[i32],
        ) -> Result<PassCopies, EngError> {
            self.bank.step(ctx, layer, picks)
        }

        /// The classifier's slot row of `layer`; upload it after every pass that moved a slot.
        pub(in crate::family::flash_next) fn slot_row(
            &self,
            layer: usize,
        ) -> Result<&[i32], EngError> {
            self.bank.slot_row(layer)
        }

        /// The bank buffer of `layer`: slot `s` starts at `s * record_bytes`.
        pub(in crate::family::flash_next) fn slots(
            &self,
            layer: usize,
        ) -> Result<&DeviceBuffer, EngError> {
            self.bank.slots(layer)
        }

        /// The pinned host place of an expert's record, for the ring's stream plan (a
        /// non-resident expert without a place cannot stream).
        pub(in crate::family::flash_next) fn place_of(
            &self,
            layer: usize,
            expert: u16,
        ) -> Option<u32> {
            self.bank.place_of(layer, expert)
        }

        /// Queues one expert place's bytes from the pinned store into `dst` on the ring's copy
        /// stream. The host model is not changed: streaming is not an admit.
        pub(in crate::family::flash_next) fn stream_place(
            &mut self,
            ctx: &CudaContext,
            stream: &DeviceStream,
            layer: usize,
            place: u32,
            dst: DeviceRange<'_>,
        ) -> Result<(), EngError> {
            self.bank.stream_place(ctx, stream, layer, place, dst)
        }

        /// Waits the ring's copies still outstanding on `layer`'s pinned store and clears its
        /// pending record: the ring window hands the store back this way.
        pub(in crate::family::flash_next) fn synchronize_store(
            &mut self,
            layer: usize,
            stream: &DeviceStream,
        ) -> Result<(), EngError> {
            self.bank.synchronize_store(layer, stream)
        }

        /// The bank's own copy stream (the one `step` and `admit` run on).
        pub(in crate::family::flash_next) fn stream(&self) -> &DeviceStream {
            self.bank.stream()
        }

        /// The counters of the passes so far (hits, misses, write-backs, miss histogram).
        pub(in crate::family::flash_next) fn stats(&self) -> &BankStats {
            self.bank.stats()
        }

        /// The sizes of the bank: layers, experts, top-k, slots per layer and the record size.
        pub(in crate::family::flash_next) fn geometry(&self) -> &BankGeometry {
            self.bank.geometry()
        }

        /// Whether a failed copy left the device out of step with the host model. While this
        /// holds, `admit_plan`, `admit_union`, `admit_planned`, `step`, `slot_row`, `slots` and
        /// `stream_place` are refused; `place_of`, `synchronize_store`, `stream`, `stats`,
        /// `geometry` and this method still answer (the ring must be able to hand its stores
        /// back and the counters to be read out while a rebuild is decided).
        pub(in crate::family::flash_next) fn is_poisoned(&self) -> bool {
            self.bank.is_poisoned()
        }
    }
}
