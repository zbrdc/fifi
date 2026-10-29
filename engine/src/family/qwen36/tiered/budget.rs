//! The admission arithmetic of the tiered offload.
//!
//! WHAT: two pure planners that decide, in bytes, how the machine is split at
//! load. [`plan_vram`] gives the KV cache of the configured context VRAM first
//! and the per-layer expert bank the rest, spilling KV positions to the host
//! only when they do not fit or when the minimum bank needs the room;
//! [`plan_host`] gives the host RAM to the OS page cache of the experts minus a
//! host KV overflow taken from that cache one-for-one. WHY (REV4 c.4, north
//! star = coding agents with long context): a 6 GiB card cannot hold the
//! model's experts, so the split must be settled before the first token, and a
//! context that cannot fit is refused with a typed error that names the budget
//! instead of dying mid-decode. Every number is an input measured by the
//! caller; no model constant lives here. All sizes are bytes (`u64`), every
//! operation is checked (an absurd input is refused as [`BudgetError`] — never
//! wrapped, never a panic), and the split is deterministic: no I/O, no clocks,
//! no randomness, no threads. The tiered load uses its VRAM plan.

use super::TieredConfig;

/// What the card has and what is already spoken for (the caller measures these
/// at load; none of them is guessed here). All fields are bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VramBudget {
    /// VRAM the engine may use in total.
    pub usable: u64,
    /// Non-expert model weights resident in VRAM.
    pub non_expert: u64,
    /// Engine state resident in VRAM (samplers, logits, graphs).
    pub state: u64,
    /// The kernel scratch arena.
    pub arena: u64,
    /// VRAM kept free for the desktop and other processes.
    pub desktop_reserve: u64,
}

/// The KV cache of the configured context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPlan {
    /// Context positions the caller configured (positions, not bytes).
    pub positions: u64,
    /// KV bytes one position costs (8-bit KV on every backend; the number is
    /// the caller's measurement of the model's attention layers).
    pub bytes_per_position: u64,
}

/// The expert geometry: one expert size shared by every MoE layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertSize {
    /// Bytes of one quantized expert.
    pub bytes: u64,
    /// MoE layers; the bank is uniform over them, so one slot per layer costs
    /// `bytes * layers` VRAM in total.
    pub layers: usize,
}

impl ExpertSize {
    /// Takes the layer count from a [`TieredConfig`] and the expert size from
    /// the caller's measurement. WHY: the split is only meaningful for the same
    /// layer count the policy and residency tables index, so the geometry is
    /// borrowed from the one config instead of being retyped (and drifting).
    pub fn from_config(cfg: &TieredConfig, bytes: u64) -> Self {
        Self {
            bytes,
            layers: cfg.layers,
        }
    }
}

/// Why a plan was refused. Every variant names the budget that lost, with all
/// numbers in bytes, so the caller can report the shortfall or retry with a
/// smaller context.
#[derive(Debug, PartialEq, Eq)]
pub enum BudgetError {
    /// A fixed commitment exceeded its pool: `needed` > `available` bytes of
    /// `what`. A plan never borrows from a pool that is already overdrawn.
    Overcommitted {
        /// The pool that lost, human-readable.
        what: &'static str,
        /// Bytes the plan needed from the pool.
        needed: u64,
        /// Bytes the pool had.
        available: u64,
    },
    /// The KV cache did not fit: it wants `kv_bytes`, VRAM grants
    /// `vram_for_kv` of them and the host is allowed `host_allowed`, so the
    /// remainder `kv_bytes - vram_for_kv` exceeds what the host may take.
    KvDoesNotFit {
        /// Total KV bytes the configured context wants.
        kv_bytes: u64,
        /// KV bytes VRAM grants after the minimum bank took its room.
        vram_for_kv: u64,
        /// Host bytes the caller allows KV to spill to.
        host_allowed: u64,
    },
    /// A structurally invalid input (a zero size or dimension, or a sum that
    /// overflows `u64`); nothing was computed.
    BadInput(String),
}

/// The result of the VRAM split: where the KV lives and how big the bank is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VramPlan {
    /// KV positions served from VRAM.
    pub kv_vram_positions: u64,
    /// KV positions spilled to host RAM. Invariant:
    /// `kv_vram_positions + kv_host_positions == KvPlan::positions`.
    pub kv_host_positions: u64,
    /// Bank slots per layer, uniform over the layers; at least the requested
    /// minimum whenever the plan is `Ok`.
    pub bank_slots_per_layer: usize,
    /// VRAM the whole bank owns: `bank_slots_per_layer * layers * expert bytes`.
    pub bank_bytes: u64,
    /// VRAM left after KV and the bank: the floor remainder, always less than
    /// one bank slot (`layers * expert bytes`).
    pub vram_left: u64,
}

/// Splits the VRAM: KV first, the bank gets the rest (floored to whole slots,
/// uniform over the layers). `min_bank_slots_per_layer` is kept by spilling KV
/// positions to the host while `host_kv_allowed` bytes permit it, else the plan
/// is refused. WHY: REV4 c.4 orders the tiers exactly this way — KV is 8-bit on
/// every backend and takes VRAM first, the expert bank gets the remainder, and
/// KV spills to host RAM only when it does not fit; a decode that started
/// without its minimum bank would fall through to the CPU from the first token.
///
/// Refusals, all typed and byte-exact: commitments beyond `usable` (or a
/// minimum bank beyond the free VRAM) are [`BudgetError::Overcommitted`]; a KV
/// remainder the host may not take is [`BudgetError::KvDoesNotFit`]; zero sizes,
/// zero layers or overflowing products are [`BudgetError::BadInput`]. The
/// arithmetic is checked throughout, so no input wraps or panics.
pub fn plan_vram(
    v: &VramBudget,
    kv: &KvPlan,
    e: &ExpertSize,
    min_bank_slots_per_layer: usize,
    host_kv_allowed: u64,
) -> Result<VramPlan, BudgetError> {
    if kv.bytes_per_position == 0 {
        return Err(bad_input("budget: kv bytes_per_position must be nonzero"));
    }
    if e.bytes == 0 {
        return Err(bad_input("budget: expert bytes must be nonzero"));
    }
    if e.layers == 0 {
        return Err(bad_input("budget: expert layers must be nonzero"));
    }
    let layers =
        u64::try_from(e.layers).map_err(|_| bad_input("budget: expert layers does not fit u64"))?;
    let slot_cost = layers
        .checked_mul(e.bytes)
        .ok_or_else(|| bad_input("budget: expert bytes * layers overflows u64"))?;
    let min_bank_bytes = u64::try_from(min_bank_slots_per_layer)
        .map_err(|_| bad_input("budget: min_bank_slots_per_layer does not fit u64"))?
        .checked_mul(slot_cost)
        .ok_or_else(|| bad_input("budget: minimum bank bytes overflow u64"))?;
    let commitments = v
        .non_expert
        .checked_add(v.state)
        .and_then(|sum| sum.checked_add(v.arena))
        .and_then(|sum| sum.checked_add(v.desktop_reserve))
        .ok_or_else(|| bad_input("budget: vram commitments overflow u64"))?;
    let free_vram = v
        .usable
        .checked_sub(commitments)
        .ok_or(BudgetError::Overcommitted {
            what: "vram (non-expert weights + state + arena + desktop reserve)",
            needed: commitments,
            available: v.usable,
        })?;
    if min_bank_bytes > free_vram {
        return Err(BudgetError::Overcommitted {
            what: "minimum expert bank",
            needed: min_bank_bytes,
            available: free_vram,
        });
    }

    // KV takes VRAM first but never the bytes the minimum bank owns; that is
    // the only lever that pushes KV positions to the host (REV4 c.4).
    let kv_bytes = kv
        .positions
        .checked_mul(kv.bytes_per_position)
        .ok_or_else(|| bad_input("budget: kv positions * bytes_per_position overflows u64"))?;
    let vram_for_kv = kv_bytes.min(free_vram - min_bank_bytes);
    let kv_vram_positions = vram_for_kv / kv.bytes_per_position;
    let kv_vram_bytes = kv_vram_positions
        .checked_mul(kv.bytes_per_position)
        .ok_or_else(|| bad_input("budget: vram kv bytes overflow u64"))?;
    let kv_host_positions = kv.positions - kv_vram_positions;
    let host_kv_bytes = kv_host_positions
        .checked_mul(kv.bytes_per_position)
        .ok_or_else(|| bad_input("budget: host kv bytes overflow u64"))?;
    if host_kv_bytes > host_kv_allowed {
        return Err(BudgetError::KvDoesNotFit {
            kv_bytes,
            vram_for_kv: kv_vram_bytes,
            host_allowed: host_kv_allowed,
        });
    }

    // The bank gets what is left, floored to whole slots, uniform over the
    // layers; `bank_free >= min_bank_bytes` holds by construction above.
    let bank_free = free_vram - kv_vram_bytes;
    let bank_slots = bank_free / slot_cost;
    let bank_bytes = bank_slots
        .checked_mul(slot_cost)
        .ok_or_else(|| bad_input("budget: bank bytes overflow u64"))?;
    let bank_slots_per_layer = usize::try_from(bank_slots)
        .map_err(|_| bad_input("budget: bank slots does not fit usize"))?;
    Ok(VramPlan {
        kv_vram_positions,
        kv_host_positions,
        bank_slots_per_layer,
        bank_bytes,
        vram_left: bank_free - bank_bytes,
    })
}

/// What the host has and what is already spoken for before the page cache (the
/// caller measures these at load; none of them is guessed here). All fields are
/// bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostBudget {
    /// Host RAM the engine may use in total.
    pub total: u64,
    /// RAM held by the OS, the loader and other processes; never returned.
    pub non_reclaimable: u64,
    /// The engine's own heap: activations, CPU jobs, bookkeeping.
    pub engine_heap: u64,
    /// Staging buffers for expert transfers to the GPU.
    pub staging: u64,
    /// Safety margin kept free so the plan does not ride the OOM line.
    pub margin: u64,
}

/// The result of the host RAM split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostPlan {
    /// Bytes left for the OS page cache of the experts.
    pub page_cache: u64,
    /// Whole experts the cache holds: `page_cache / expert_bytes`, floored.
    pub cached_experts: u64,
    /// The host KV bytes echoed back; each was taken from `page_cache`
    /// one-for-one (REV4 c.4).
    pub host_kv: u64,
}

/// Splits the host RAM: `page_cache = total - non_reclaimable - engine_heap -
/// staging - margin - host_kv` (checked subtraction, never wraps), and
/// `cached_experts` floors that cache to whole experts of `expert_bytes`. WHY:
/// the OS page cache is what makes the CPU tier fast, so it owns the residual
/// host RAM — but every host KV byte is taken from it one-for-one (REV4 c.4),
/// which this plan makes explicit instead of accidental.
///
/// Refusals: commitments beyond `total` are [`BudgetError::Overcommitted`]
/// naming the pool and both numbers; zero `expert_bytes` (and any sum
/// overflowing `u64`) is [`BudgetError::BadInput`]. No input wraps or panics.
pub fn plan_host(h: &HostBudget, host_kv: u64, expert_bytes: u64) -> Result<HostPlan, BudgetError> {
    if expert_bytes == 0 {
        return Err(bad_input("budget: expert bytes must be nonzero"));
    }
    let spent = h
        .non_reclaimable
        .checked_add(h.engine_heap)
        .and_then(|sum| sum.checked_add(h.staging))
        .and_then(|sum| sum.checked_add(h.margin))
        .and_then(|sum| sum.checked_add(host_kv))
        .ok_or_else(|| bad_input("budget: host commitments overflow u64"))?;
    let page_cache = h
        .total
        .checked_sub(spent)
        .ok_or(BudgetError::Overcommitted {
            what: "host RAM (non-reclaimable + engine heap + staging + margin + host KV)",
            needed: spent,
            available: h.total,
        })?;
    Ok(HostPlan {
        page_cache,
        cached_experts: page_cache / expert_bytes,
        host_kv,
    })
}

/// Wraps `what` into a [`BudgetError::BadInput`]; private sugar for the many
/// refusal arms of the checked arithmetic above.
fn bad_input(what: &str) -> BudgetError {
    BudgetError::BadInput(String::from(what))
}

#[cfg(test)]
mod tests {
    //! The Qwen3.6-35B-A3B numbers live here and only here; production code
    //! takes every number as an input. GiB = 2^30 bytes throughout.

    use super::{
        BudgetError, ExpertSize, HostBudget, KvPlan, VramBudget, bad_input, plan_host, plan_vram,
    };
    use crate::family::qwen36::tiered::{PolicyMode, TieredConfig};

    /// One GiB (2^30 bytes); 8-bit KV bytes of one position (10 attention
    /// layers x 1,088 bytes); one quantized expert; the MoE layer count; and the
    /// VRAM one slot per layer costs (`EXPERT_BYTES * LAYERS`).
    const GIB: u64 = 1 << 30;
    const BYTES_PER_POSITION: u64 = 10_880;
    const EXPERT_BYTES: u64 = 1_769_472;
    const LAYERS: usize = 40;
    const SLOT_COST: u64 = EXPERT_BYTES * 40;

    /// The 6 GiB card fixture: measured weights, state, arena and a reserve.
    fn vram(desktop_reserve: u64) -> VramBudget {
        VramBudget {
            usable: 6 * GIB,
            non_expert: 2_316_995_072,
            state: 62_914_560 + 2_949_120,
            arena: 107_374_182,
            desktop_reserve,
        }
    }

    /// A card with only `usable` and `non_expert` set: the zero-case helper.
    fn vbudget(usable: u64, non_expert: u64) -> VramBudget {
        VramBudget {
            usable,
            non_expert,
            state: 0,
            arena: 0,
            desktop_reserve: 0,
        }
    }

    /// The KV cache of `positions` positions at 8-bit KV.
    fn kv(positions: u64) -> KvPlan {
        kvz(positions, BYTES_PER_POSITION)
    }

    /// The KV cache of `positions` positions at `bpp` bytes per position.
    fn kvz(positions: u64, bpp: u64) -> KvPlan {
        KvPlan {
            positions,
            bytes_per_position: bpp,
        }
    }

    /// The geometry of `bytes`-byte experts over `layers` layers.
    fn geom(bytes: u64, layers: usize) -> ExpertSize {
        ExpertSize { bytes, layers }
    }

    /// The expert geometry, layer count borrowed from the tiered config.
    fn experts() -> ExpertSize {
        ExpertSize::from_config(&TieredConfig::qwen36(1, PolicyMode::Static), EXPERT_BYTES)
    }

    /// The host RAM fixture: 14.76 GiB total with the fixed commitments.
    fn host() -> HostBudget {
        HostBudget {
            total: 15_847_931_904,
            non_reclaimable: 2 * GIB,
            engine_heap: GIB,
            staging: 214_748_365,
            margin: GIB / 2,
        }
    }

    #[test]
    fn qwen36_32k_8bit_kv_leaves_about_39_slots_per_layer() {
        assert_eq!(
            experts(),
            ExpertSize {
                bytes: EXPERT_BYTES,
                layers: LAYERS
            }
        );
        let plan = plan_vram(&vram(3 * GIB / 4), &kv(32_768), &experts(), 0, 0).unwrap();
        assert_eq!(
            (plan.kv_vram_positions, plan.kv_host_positions),
            (32_768, 0)
        );
        assert!((38..=40).contains(&plan.bank_slots_per_layer), "{plan:?}");
    }

    #[test]
    fn qwen36_128k_leaves_about_24() {
        let plan = plan_vram(&vram(3 * GIB / 4), &kv(131_072), &experts(), 0, 0).unwrap();
        assert_eq!(
            (plan.kv_vram_positions, plan.kv_host_positions),
            (131_072, 0)
        );
        assert!((23..=25).contains(&plan.bank_slots_per_layer), "{plan:?}");
    }

    #[test]
    fn qwen36_262k_in_vram_leaves_a_handful() {
        assert_eq!(262_144 * BYTES_PER_POSITION, 2_852_126_720);
        let plan = plan_vram(&vram(3 * GIB / 4), &kv(262_144), &experts(), 0, 0).unwrap();
        assert_eq!(
            (plan.kv_vram_positions, plan.kv_host_positions),
            (262_144, 0)
        );
        assert!((1..=6).contains(&plan.bank_slots_per_layer), "{plan:?}");
    }

    #[test]
    fn a_desktop_reserve_of_one_gib_at_262k_spills_kv_to_the_host_when_a_minimum_bank_is_required()
    {
        let plan = plan_vram(&vram(GIB), &kv(262_144), &experts(), 1, GIB).unwrap();
        assert_eq!(plan.kv_vram_positions + plan.kv_host_positions, 262_144);
        assert!(plan.kv_host_positions > 0, "{plan:?}");
        assert!(plan.bank_slots_per_layer >= 1, "{plan:?}");
        match plan_vram(&vram(GIB), &kv(262_144), &experts(), 1, 0) {
            Err(BudgetError::KvDoesNotFit {
                kv_bytes,
                vram_for_kv,
                host_allowed,
            }) => {
                assert_eq!((kv_bytes, host_allowed), (2_852_126_720, 0));
                assert!(vram_for_kv < kv_bytes);
            }
            other => panic!("expected KvDoesNotFit, got {other:?}"),
        }
    }

    #[test]
    fn kv_positions_split_exactly() {
        let geometry = experts();
        let mut fitted = 0;
        let mut spilled = 0;
        for reserve in [0, GIB / 2, GIB, 3 * GIB / 2] {
            let budget = vram(reserve);
            let fixed = budget.non_expert + budget.state + budget.arena + budget.desktop_reserve;
            for step in 0..64u64 {
                for min_slots in [0, 1, 4] {
                    for host_allowed in [0, 4 * GIB] {
                        let positions = step * 4_099;
                        let Ok(plan) =
                            plan_vram(&budget, &kv(positions), &geometry, min_slots, host_allowed)
                        else {
                            continue;
                        };
                        // Position counts round-trip exactly.
                        assert_eq!(plan.kv_vram_positions + plan.kv_host_positions, positions);
                        // The bank is uniform whole slots and the minimum holds.
                        assert_eq!(
                            plan.bank_bytes,
                            u64::try_from(plan.bank_slots_per_layer).unwrap() * SLOT_COST
                        );
                        assert!(plan.bank_slots_per_layer >= min_slots);
                        assert!(plan.vram_left < SLOT_COST, "{plan:?}");
                        // Every VRAM byte is accounted for: commitments + KV + bank + slack.
                        assert_eq!(
                            budget.usable,
                            fixed
                                + plan.kv_vram_positions * BYTES_PER_POSITION
                                + plan.bank_bytes
                                + plan.vram_left
                        );
                        // The host spill stays inside what the caller allowed.
                        assert!(plan.kv_host_positions * BYTES_PER_POSITION <= host_allowed);
                        if plan.kv_host_positions > 0 {
                            spilled += 1;
                        } else {
                            fitted += 1;
                        }
                    }
                }
            }
        }
        assert!(fitted > 0 && spilled > 0, "the sweep must cover both sides");
    }

    #[test]
    fn overflowing_inputs_are_refused_not_wrapped() {
        let max = u64::MAX;
        let e = experts();
        // A commitment sum that overflows u64 is BadInput, not a wrapped fit.
        let over = VramBudget {
            usable: max,
            non_expert: max / 2 + 1,
            state: max / 2 + 1,
            arena: 1,
            desktop_reserve: 1,
        };
        assert!(matches!(
            plan_vram(&over, &kv(1), &e, 0, 0),
            Err(BudgetError::BadInput(_))
        ));
        // Commitments beyond the card name both numbers.
        match plan_vram(&vbudget(GIB, 2 * GIB), &kv(1), &e, 0, 0) {
            Err(BudgetError::Overcommitted {
                needed, available, ..
            }) => {
                assert_eq!((needed, available), (2 * GIB, GIB));
            }
            other => panic!("expected Overcommitted, got {other:?}"),
        }
        // A KV product that overflows, and the three zero geometries.
        let zero_card = vbudget(max, 0);
        assert!(matches!(
            plan_vram(&zero_card, &kvz(max, 2), &e, 0, 0),
            Err(BudgetError::BadInput(_))
        ));
        assert_eq!(
            plan_vram(&zero_card, &kvz(4, 0), &e, 0, 0),
            Err(bad_input("budget: kv bytes_per_position must be nonzero"))
        );
        assert_eq!(
            plan_vram(
                &zero_card,
                &kvz(4, BYTES_PER_POSITION),
                &geom(0, LAYERS),
                0,
                0
            ),
            Err(bad_input("budget: expert bytes must be nonzero"))
        );
        assert_eq!(
            plan_vram(&zero_card, &kvz(4, BYTES_PER_POSITION), &geom(8, 0), 0, 0),
            Err(bad_input("budget: expert layers must be nonzero"))
        );
        // A minimum bank that not even a full card holds.
        match plan_vram(&vbudget(2 * GIB, 0), &kv(0), &e, 41, max) {
            Err(BudgetError::Overcommitted {
                needed, available, ..
            }) => {
                assert_eq!((needed, available), (41 * SLOT_COST, 2 * GIB));
            }
            other => panic!("expected Overcommitted, got {other:?}"),
        }
        // The host side refuses the same way: overflowing commitments, zero size.
        let host_over = HostBudget {
            total: max,
            non_reclaimable: max / 2 + 1,
            engine_heap: max / 2 + 1,
            staging: 0,
            margin: 0,
        };
        assert!(matches!(
            plan_host(&host_over, 0, EXPERT_BYTES),
            Err(BudgetError::BadInput(_))
        ));
        assert_eq!(
            plan_host(&host(), 0, 0),
            Err(bad_input("budget: expert bytes must be nonzero"))
        );
    }

    #[test]
    fn host_cache_loses_a_gib_per_host_kv_gib() {
        let base = plan_host(&host(), 0, EXPERT_BYTES).unwrap();
        // About 11.06 GiB of page cache before any host KV.
        assert!(
            (11 * GIB..11 * GIB + GIB / 8).contains(&base.page_cache),
            "{}",
            base.page_cache
        );
        assert_eq!(base.host_kv, 0);
        for extra in 0..=4u64 {
            let plan = plan_host(&host(), extra * GIB, EXPERT_BYTES).unwrap();
            assert_eq!(plan.host_kv, extra * GIB);
            // Exactly one GiB of cache per host KV GiB, one-for-one.
            assert_eq!(plan.page_cache, base.page_cache - extra * GIB);
        }
        let lost = base.cached_experts
            - plan_host(&host(), GIB, EXPERT_BYTES)
                .unwrap()
                .cached_experts;
        assert!((606..=607).contains(&lost), "{lost}"); // about 607 experts per GiB
    }

    #[test]
    fn host_plan_refuses_a_kv_larger_than_the_cache() {
        let budget = host();
        let cache = plan_host(&budget, 0, EXPERT_BYTES).unwrap().page_cache;
        match plan_host(&budget, cache + 1, EXPERT_BYTES) {
            Err(BudgetError::Overcommitted {
                what,
                needed,
                available,
            }) => {
                assert!(what.contains("host"), "{what}");
                assert_eq!(available, budget.total);
                assert!(needed > available);
            }
            other => panic!("expected Overcommitted, got {other:?}"),
        }
        // The boundary itself is legal: the cache goes to zero, never below.
        let edge = plan_host(&budget, cache, EXPERT_BYTES).unwrap();
        assert_eq!(
            (edge.page_cache, edge.cached_experts, edge.host_kv),
            (0, 0, cache)
        );
    }
}
