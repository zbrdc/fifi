//! The banked EXL3 expert load's Qwen3.6 half: the admission arithmetic that decides
//! banking and the slot count (pure, mirrored on `project::cuda`'s demand rule) and the
//! seam's table half ([`LayerTables`]). The store, the slot table and the resolver moved
//! to [`crate::family::exl3::bank`]; their paths re-export from here.

use crate::{
    EngError,
    family::{
        project::cuda::Reserve,
        qwen36::settings::{BankMode, SlotRequest},
    },
};

/// The refusal helper of this module: every admission error names its numbers.
fn refuse(detail: String) -> EngError {
    EngError::MemoryPlan(format!("EXL3 expert bank: {detail}"))
}

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1 << 30) as f64)
}

/// The resolved banked load: the slot count and the figures the start line reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BankAdmission {
    pub(super) slots: usize,
    /// Dense bytes plus the slot banks' bytes: what the device holds after the load.
    pub(super) device_bytes: u64,
    /// The host store's bytes: every routed expert of every layer, in the bank layout.
    pub(super) host_bytes: u64,
}

/// Decides the load shape from the pack figures and the free memory. `Ok(None)` means stay
/// resident: the whole pack fits and the mode is `auto`. `dense` is the walk's device bytes
/// without the routed experts; `routed` is the routed experts' device bytes alone (the same
/// figure the host store holds, because a member's store bytes are its bank bytes).
///
/// The slot arithmetic mirrors `project::cuda::admit`'s demand rule: a load of `tensors`
/// bytes plus the reserve's `kv_cache + work + headroom` must fit `free`. Banking replaces
/// the routed share with `per_member * slots * layers`.
// Every figure is one named admission input; a config struct would only rename them.
#[allow(clippy::too_many_arguments)]
pub(super) fn decide(
    request: SlotRequest,
    mode: BankMode,
    dense: u64,
    routed: u64,
    layers: usize,
    n_expert: usize,
    free: u64,
    reserve: Reserve,
    host: Option<(u64, u64)>,
) -> Result<Option<BankAdmission>, EngError> {
    let hold = reserve
        .kv_cache
        .saturating_add(reserve.work)
        .saturating_add(reserve.headroom);
    let per_member = per_member_bytes(routed, layers, n_expert)?;
    let resident = dense
        .checked_add(routed)
        .ok_or_else(|| refuse("the resident figure overflows u64".into()))?;
    if mode == BankMode::Auto && resident.saturating_add(hold) <= free {
        return Ok(None);
    }
    admit_host(routed, host)?;
    // The load also leaves the resident flow's post-load floor free: the CUDA context and
    // workspaces eat bytes the figures above do not count, and the per-tensor admits after
    // the banks re-subtract the full hold (the embed table's admit is the last one).
    let budget = free
        .saturating_sub(hold)
        .saturating_sub(dense)
        .saturating_sub(
            u64::try_from(super::pinned::min_free_after_load()?)
                .map_err(|_| refuse("the free-after-load floor exceeds u64".into()))?,
        );
    let slots = match request {
        SlotRequest::Exactly(want) => {
            if want < 1 || want > n_expert {
                return Err(refuse(format!(
                    "{want} slots are outside 1 to {n_expert}, one banked expert to the experts \
                     of a layer",
                )));
            }
            let demand = per_member
                .checked_mul(want as u64)
                .and_then(|bytes| bytes.checked_mul(layers as u64))
                .ok_or_else(|| refuse("the slot demand overflows u64".into()))?;
            if demand > budget {
                return Err(refuse(format!(
                    "{} slots need {} of expert bytes, the budget after the dense weights and \
                     the reserve is {} (free {}, hold {}, dense {})",
                    want,
                    gib(demand),
                    gib(budget),
                    gib(free),
                    gib(hold),
                    gib(dense),
                )));
            }
            want
        }
        SlotRequest::Fit => {
            // One slot costs `per_member` per LAYER; the budget must cover every layer's bank.
            let slots = (budget / per_member.max(1) / layers as u64) as usize;
            if slots < 1 {
                return Err(refuse(format!(
                    "the budget after the dense weights and the reserve leaves {slots} slots, \
                     below one (budget {}, per member {})",
                    gib(budget),
                    gib(per_member),
                )));
            }
            slots.min(n_expert)
        }
    };
    let device_bytes = dense.saturating_add(
        per_member
            .saturating_mul(slots as u64)
            .saturating_mul(layers as u64),
    );
    Ok(Some(BankAdmission {
        slots,
        device_bytes,
        host_bytes: routed,
    }))
}

/// The routed-expert bytes of one expert of one layer (the three projections summed). The
/// walk guarantees one rate per projection, so the division is exact; a remainder means a
/// pack the walk did not see, and is refused rather than mis-banked.
/// The host store's admission: the routed bytes must fit the available RAM under an OS
/// margin of the larger of 2 GiB and 20% of the total. The figures come from
/// `/proc/meminfo` (`probe_host`); a host without them does not get a banked load.
fn admit_host(routed: u64, host: Option<(u64, u64)>) -> Result<(), EngError> {
    let Some((total, available)) = host else {
        return Err(refuse(
            "the host memory figures are unavailable, so the expert store cannot be admitted"
                .into(),
        ));
    };
    let margin = (2 << 30).max(total / 5);
    let budget = available.saturating_sub(margin);
    if routed > budget {
        return Err(refuse(format!(
            "the expert store needs {} of host RAM, the budget is {} (available {}, margin {})",
            gib(routed),
            gib(budget),
            gib(available),
            gib(margin),
        )));
    }
    Ok(())
}

/// The Q36 tables behind the shared seam's table half
/// ([`SlotTable`](crate::family::expert_bank::SlotTable)): one `LayerSlots` per layer, its
/// live `slot_of` row the tiered classifier reads. Only the table surface is shared — the
/// pass protocol stays Q36's own: `assign` and the refill loop are not routed through
/// `SlotPolicy`.
pub(super) struct LayerTables(pub(crate) Vec<LayerSlots>);

impl crate::family::expert_bank::SlotTable for LayerTables {
    fn slot_row(&self, layer: usize) -> &[i32] {
        self.0.get(layer).map_or(&[], |tables| tables.slot_row())
    }

    fn expert_in_slot(&self, layer: usize, slot: usize) -> Option<u16> {
        self.0.get(layer)?.expert_in_slot(slot)
    }
}

// The family-neutral store machinery moved to `family::exl3::bank`; these re-exports keep
// every existing path (the loader, the step, the probes, the pick-golden harness) stable.
#[cfg(test)]
pub(in crate::family) use crate::family::exl3::bank::{
    DECAY_CALLS, EMPTY, NO_SLOT, probe, refill_plan, trace,
};
pub(in crate::family) use crate::family::exl3::bank::{
    Exl3SlotResolver, ExpertStore, LayerResolver, LayerSlots, PartStore, per_member_bytes,
};
mod tests {
    use super::*;

    /// Dense ≈ 6 GiB, routed ≈ 10.1 GiB: the q36 EXL3 pack's shape, with the routed figure
    /// built from an exactly-divisible per-member byte count (the guard rejects a figure
    /// that does not divide over the layers and experts).
    fn figures() -> (u64, u64) {
        let per_member = 900_000u64; // ≈ 0.86 MiB per expert per layer (the three parts)
        (6 << 30, 47 * 256 * per_member)
    }

    /// A 16 GiB host with ~14 GiB available: the store fits under the 3.2 GiB margin.
    fn host() -> Option<(u64, u64)> {
        Some((16 << 30, 14 << 30))
    }

    fn reserve(hold: u64) -> Reserve {
        Reserve {
            kv_cache: hold,
            work: 0,
            headroom: 0,
        }
    }

    fn decide_auto(
        dense_routed: (u64, u64),
        free: u64,
        request: SlotRequest,
        host: Option<(u64, u64)>,
    ) -> Result<Option<BankAdmission>, EngError> {
        let (dense, routed) = dense_routed;
        decide(
            request,
            BankMode::Auto,
            dense,
            routed,
            47,
            256,
            free,
            reserve(1 << 30),
            host,
        )
    }

    #[test]
    fn a_pack_that_fits_stays_resident_in_auto_mode() {
        let (dense, routed) = figures();
        let free = dense + routed + (1 << 30) + (1 << 20);
        assert_eq!(
            decide_auto((dense, routed), free, SlotRequest::Fit, host()).unwrap(),
            None
        );
    }

    #[test]
    fn a_pack_that_does_not_fit_banks_the_fitting_slot_count() {
        let (dense, routed) = figures();
        // Free for 1 GiB of dense + hold + the post-load floor, plus 64 slots per layer.
        let per_member = routed / 47 / 256;
        let free = dense + (1 << 30) + (1 << 30) + per_member * 64 * 47;
        let got = decide_auto((dense, routed), free, SlotRequest::Fit, host())
            .unwrap()
            .expect("banks");
        assert_eq!(got.slots, 64);
        assert_eq!(got.host_bytes, routed);
        assert_eq!(got.device_bytes, dense + per_member * 64 * 47);
    }

    #[test]
    fn on_mode_banks_even_when_the_pack_fits() {
        let (dense, routed) = figures();
        let free = dense + routed + (1 << 30);
        let got = decide(
            SlotRequest::Exactly(32),
            BankMode::On,
            dense,
            routed,
            47,
            256,
            free,
            reserve(1 << 30),
            host(),
        )
        .unwrap()
        .expect("banks");
        assert_eq!(got.slots, 32);
        assert_eq!(
            got.device_bytes,
            dense + (routed / 47 / 256) * 32 * 47,
            "linear in the slot count"
        );
    }

    #[test]
    fn an_exact_count_over_budget_is_refused_with_numbers() {
        let (dense, routed) = figures();
        let free = dense + (1 << 30) + (routed / 47 / 256) * 16;
        let error = format!(
            "{:?}",
            decide_auto((dense, routed), free, SlotRequest::Exactly(200), host()).unwrap_err()
        );
        assert!(error.contains("200 slots"), "{error}");
        assert!(error.contains("budget"), "{error}");
    }

    #[test]
    fn a_host_without_room_is_refused_before_any_slot() {
        let (dense, routed) = figures();
        // On mode: banking is requested even though the pack would fit, and the host refuses.
        // Available 8 GiB under a 3.2 GiB margin (20% of 16): 4.8 GiB < ~10.1 GiB of experts.
        let error = format!(
            "{:?}",
            decide(
                SlotRequest::Fit,
                BankMode::On,
                dense,
                routed,
                47,
                256,
                1 << 40,
                reserve(1 << 30),
                Some((16 << 30, 8 << 30)),
            )
            .unwrap_err()
        );
        assert!(error.contains("host RAM"), "{error}");
        // And a host without figures at all.
        let error = format!(
            "{:?}",
            decide(
                SlotRequest::Fit,
                BankMode::On,
                dense,
                routed,
                47,
                256,
                1 << 40,
                reserve(1 << 30),
                None,
            )
            .unwrap_err()
        );
        assert!(error.contains("unavailable"), "{error}");
    }

    #[test]
    fn fit_never_returns_zero_or_above_the_expert_count() {
        let (dense, routed) = figures();
        // Barely past the dense figure + hold + the post-load floor: one slot fits.
        let per_member = routed / 47 / 256;
        let free = dense + (1 << 30) + (1 << 30) + per_member * 47;
        let got = decide_auto((dense, routed), free, SlotRequest::Fit, host())
            .unwrap()
            .expect("banks");
        assert_eq!(got.slots, 1);
        // And a huge free figure clamps at the full expert count (On mode: Auto would stay
        // resident with room to spare).
        let got = decide(
            SlotRequest::Fit,
            BankMode::On,
            dense,
            routed,
            47,
            256,
            1 << 44,
            reserve(1 << 30),
            host(),
        )
        .unwrap()
        .expect("banks");
        assert_eq!(got.slots, 256);
    }
}

#[cfg(test)]
mod exl3_bank_pick_golden_host_tests;
