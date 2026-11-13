//! The admission arithmetic of a chunked-prefill session, shared by every family that arenas
//! prompt chunks. Pure host arithmetic, no CUDA call: the promoting family's host goldens pin
//! it.
//!
//! The family values — the spare minimum, the flat part charge and the chunk ladder — are
//! [`Admission`] fields set by the caller, never imports: the promoting family binds its own
//! values (byte-identical to before the promotion), and a second family binds its own without
//! this module naming either.

use crate::EngError;

/// One extra tensor of the admission sum: `fixed + bytes_per_token * M` bytes, such as the tiered
/// `partial [M][2048]` f32 (8,192 B per token) or the expert bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Extra {
    pub(crate) name: &'static str,
    pub(crate) fixed: usize,
    pub(crate) bytes_per_token: usize,
}

fn extras_bytes(extras: &[Extra], m: usize) -> Result<usize, EngError> {
    let mut total = 0usize;
    for extra in extras {
        total = extra
            .bytes_per_token
            .checked_mul(m)
            .and_then(|bytes| bytes.checked_add(extra.fixed))
            .and_then(|bytes| bytes.checked_add(total))
            .ok_or_else(|| {
                EngError::MemoryPlan(format!("prefill plan: extra {} overflows", extra.name))
            })?;
    }
    Ok(total)
}

/// What chunk size `m` costs the device beyond KV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChunkCost {
    pub(crate) m: usize,
    /// The `plan_arena` total of the chunk's tensor list.
    pub(crate) arena: usize,
    /// Bytes of the attention `part` buffer of the `M_sess` split plan, reported for a decision on
    /// the flat charge. The driver allocates the buffer at the family's flat part cap, never at
    /// this figure: a smaller chunk launched with its own split plan can need more than the
    /// `M_sess` plan, and the flat cap covers both.
    pub(crate) part: usize,
}

/// The admission inputs that do not depend on the chunk size or the window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Admission {
    /// Device bytes for the resident state, the decode arena, the prefill arena and `part`: free
    /// VRAM measured ONCE by the caller right after the weight load and BEFORE the resident state
    /// and the decode-arena allocations, less the headroom (never a literal here); the state and
    /// the decode arena are then charged as their own inputs. The snapshot must follow the loading
    /// of every kernel module: until the start loads them eagerly, a lazily loaded module can take
    /// VRAM after the snapshot, and the headroom has to cover it.
    pub(crate) device_budget: usize,
    /// `DeviceArena::bytes()` of the decode plan, read at run time.
    pub(crate) decode_arena: usize,
    /// The flat `part` charge in every row (the caller's part cap).
    pub(crate) part_charge: usize,
    /// Fallback buffers when one is active; 0 while none is.
    pub(crate) fallback: usize,
    pub(crate) device_extras: Vec<Extra>,
    /// Pinned host bytes available, a separate budget: host extras never enter the device sum.
    pub(crate) host_budget: usize,
    pub(crate) host_extras: Vec<Extra>,
    /// The spare minimum every admitted row must leave (the caller's constant).
    pub(crate) spare_min: usize,
    /// The chunk sizes a session arena may be built for, largest first (the caller's ladder).
    pub(crate) ladder: &'static [usize],
}

impl Admission {
    /// The rule with the flat `part` charge and no decode arena, fallback or extras; the spare
    /// minimum and the ladder are the caller's family constants.
    pub(crate) fn new(
        device_budget: usize,
        part_charge: usize,
        spare_min: usize,
        ladder: &'static [usize],
    ) -> Self {
        Self {
            device_budget,
            decode_arena: 0,
            part_charge,
            fallback: 0,
            device_extras: Vec::new(),
            host_budget: 0,
            host_extras: Vec::new(),
            spare_min,
            ladder,
        }
    }

    /// Whether chunk size `cost.m` is admitted with `state` bytes of resident state: the device
    /// sum (the decode arena, the state, the arena, the larger of the flat and the real `part`,
    /// the fallback and the device extras) leaves at least `spare_min` spare and the pinned host
    /// rows fit the host budget.
    pub(crate) fn passes(&self, state: usize, cost: &ChunkCost) -> Result<bool, EngError> {
        // The part term is the larger of the flat charge and the chunk's real part: the flat charge
        // is the floor, so a later lowering of `part_charge` cannot undercharge a chunk whose own
        // split plan needs more.
        let part = self.part_charge.max(cost.part);
        let parts = [
            self.decode_arena,
            state,
            cost.arena,
            part,
            self.fallback,
            extras_bytes(&self.device_extras, cost.m)?,
        ];
        let charged = parts
            .iter()
            .try_fold(0usize, |sum, &part| sum.checked_add(part));
        let spare = self
            .device_budget
            .checked_sub(charged.ok_or_else(overflow)?);
        let host = extras_bytes(&self.host_extras, cost.m)?;
        Ok(spare.is_some_and(|spare| spare >= self.spare_min) && host <= self.host_budget)
    }

    /// `M_sess`: the largest of `costs` that is admitted and at most `cap` (the caller's
    /// chunk-cap setting; `None` is no cap, the ladder's first rung). `None` means no
    /// size passes and the session keeps the per-token path. The budget is the caller's snapshot
    /// of free VRAM after the weight load and before the resident state and the decode-arena
    /// allocations, less the headroom.
    pub(crate) fn admit(
        &self,
        state: usize,
        costs: &[ChunkCost],
        cap: Option<usize>,
    ) -> Result<Option<usize>, EngError> {
        let cap = cap.unwrap_or(self.ladder[0]);
        if !self.ladder.contains(&cap) {
            return Err(EngError::BadRequest(format!(
                "prefill plan: prefill_chunk {cap} is not one of {:?}",
                self.ladder
            )));
        }
        let mut best = None;
        for cost in costs.iter().filter(|cost| cost.m <= cap) {
            if self.passes(state, cost)? {
                best = best.max(Some(cost.m));
            }
        }
        Ok(best)
    }
}

/// The admission inputs the step reads at session build: the budget the caller measured, the decode
/// arena, the state term in its two forms, and the cap of `M_sess`. Pure host values, so the host
/// rows pin the rule without a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdmitFacts {
    /// The caller's free-VRAM snapshot after the weight load and before state/decode allocation,
    /// less the selected admission headroom.
    pub(crate) budget: usize,
    /// `res.arena.bytes()` of the decode plan (`DeviceArena::bytes`).
    pub(crate) decode_arena: usize,
    /// The PLANNED state term (`state_bytes`).
    pub(crate) state_planned: usize,
    /// The state the step really allocated. The KV mode, the KV capacity and the delta slots all
    /// flow through the layout (`ResidentState::allocate`).
    pub(crate) state_allocated: usize,
    /// The session's `M_sess` cap (`set_m_sess_cap`).
    pub(crate) cap: usize,
}

/// Feeds the [`Admission`] with the REAL resident state: the planned term must equal the allocated
/// layout, so a figure from anywhere else — the KV bytes alone, about 62.8 MiB short at the 16K
/// window — is a refusal BEFORE anything else. [`Admission::admit`] then picks the largest ladder
/// size at most `cap` whose device sum leaves the spare minimum; `None` means no size fits and the
/// session keeps the per-token path.
pub(crate) fn admit_m_sess(
    facts: &AdmitFacts,
    extras: &[Extra],
    costs: &[ChunkCost],
    part_charge: usize,
    spare_min: usize,
    ladder: &'static [usize],
) -> Result<Option<usize>, EngError> {
    if facts.state_planned != facts.state_allocated {
        return Err(EngError::Checkpoint(format!(
            "prefill admission: the planned state {} B is not the allocated state {} B",
            facts.state_planned, facts.state_allocated
        )));
    }
    let admission = Admission {
        decode_arena: facts.decode_arena,
        device_extras: extras.to_vec(),
        ..Admission::new(facts.budget, part_charge, spare_min, ladder)
    };
    admission.admit(facts.state_planned, costs, Some(facts.cap))
}

fn overflow() -> EngError {
    EngError::MemoryPlan("prefill plan: a size overflows".to_owned())
}
