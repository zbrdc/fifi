//! The host bookkeeping of the device-side rollback of a Qwen3.6 verify window, as pure code.
//!
//! A window of `K` rows runs the one-row step `K` times. After row `i` (`i < K - 1`) the device
//! copies the DeltaNet state (recurrent and convolution, one contiguous range of the Delta pool)
//! into slot `i` of a dedicated buffer. A verdict that keeps `kept` rows (`1 <= kept < K`) copies
//! slot `kept - 1` back into the Delta pool and truncates the KV frontier to `base + kept`; a
//! verdict that keeps every row changes nothing. `Window::commit` refuses `kept == 0`, so no
//! pre-window slot exists.
//!
//! This file holds what needs no device: the slot arithmetic, the window state machine, the guard
//! on the state it copies, and the memory check of the case no prefill admission charges. The
//! device owner (`cuda/state/rollback.rs`) calls it before every copy, so the guard sequence is
//! tested on the host like `StateBook`'s. What this file cannot show: the byte range of the
//! copies, the slot offsets in the buffer and the stream order. Those are proven only on a device.

use crate::EngError;

/// The most rows of one window (`QWEN36_MAX_WINDOW_ROWS`, `settings.rs`).
pub(crate) const MAX_ROWS: usize = super::settings::QWEN36_MAX_WINDOW_ROWS;

fn refuse(what: &str, why: &str) -> EngError {
    EngError::BadRequest(format!("device rollback: {what}: {why}"))
}

/// The slots of one session: how many, and how many bytes each takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RollbackPlan {
    /// Rows of the longest window (the pending token included).
    pub(crate) rows: usize,
    /// Bytes of one slot: the whole Delta pool slot 0 (`layout.delta.total / layout.delta.slots`).
    pub(crate) slot_bytes: usize,
}

impl RollbackPlan {
    /// A plan for windows of up to `rows` rows over a state whose Delta slot takes `slot_bytes`.
    pub(crate) fn new(rows: usize, slot_bytes: usize) -> Result<Self, EngError> {
        if !(2..=MAX_ROWS).contains(&rows) {
            return Err(refuse(
                "plan",
                &format!("a window has 2 to {MAX_ROWS} rows, not {rows}"),
            ));
        }
        if slot_bytes == 0 {
            return Err(refuse("plan", "a slot of zero bytes"));
        }
        let plan = Self { rows, slot_bytes };
        plan.total_bytes()?;
        Ok(plan)
    }

    /// Slots held: one per row but the last.
    pub(crate) fn slots(&self) -> usize {
        self.rows - 1
    }

    /// Bytes of the whole buffer.
    pub(crate) fn total_bytes(&self) -> Result<usize, EngError> {
        self.slots().checked_mul(self.slot_bytes).ok_or_else(|| {
            EngError::MemoryPlan(format!(
                "device rollback: {} slots of {} B overflow",
                self.slots(),
                self.slot_bytes
            ))
        })
    }

    /// Byte offset of slot `slot` in the buffer.
    pub(crate) fn offset(&self, slot: usize) -> Result<usize, EngError> {
        if slot >= self.slots() {
            return Err(refuse(
                "slot",
                &format!("slot {slot} is outside {} slots", self.slots()),
            ));
        }
        Ok(slot * self.slot_bytes)
    }
}

/// What the device owner knows about the state it is about to copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StateFacts {
    /// The step's stream owns the state.
    pub(crate) owner: bool,
    /// The state is `Valid` (not `Poisoned`, no prefill open).
    pub(crate) valid: bool,
    /// No step is pending: the device words are the host's.
    pub(crate) idle: bool,
    /// The KV is the resident plane (paged and device-first KV have none).
    pub(crate) resident_kv: bool,
}

impl StateFacts {
    /// The guard of both copies. A restore must never find a `Poisoned` state acceptable:
    /// `begin_import` takes one and `finish_overwrite` would then declare it `Valid`.
    pub(crate) fn require_copyable(&self, what: &str) -> Result<(), EngError> {
        if !self.owner {
            return Err(refuse(what, "the state belongs to another stream"));
        }
        if !self.valid {
            return Err(refuse(
                what,
                "the state is not valid; reset or import first",
            ));
        }
        if !self.idle {
            return Err(refuse(what, "steps are still pending"));
        }
        if !self.resident_kv {
            return Err(refuse(what, "the KV is not the resident plane"));
        }
        Ok(())
    }
}

/// The restore a verdict asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Restore {
    /// The slot to copy back into the Delta pool.
    pub(crate) slot: usize,
    /// The committed rows after the restore: `base + kept`.
    pub(crate) frontier: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OpenWindow {
    base: usize,
    rows: usize,
    /// Slots written since the window opened: `0..captured`.
    captured: usize,
    /// Every row has run: the window now waits for its verdict.
    complete: bool,
}

/// The window state machine: which slots hold what, for the window that is open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RollbackBook {
    plan: RollbackPlan,
    window: Option<OpenWindow>,
}

impl RollbackBook {
    pub(crate) fn new(plan: RollbackPlan) -> Self {
        Self { plan, window: None }
    }

    pub(crate) fn plan(&self) -> &RollbackPlan {
        &self.plan
    }

    pub(crate) fn is_open(&self) -> bool {
        self.window.is_some()
    }

    /// Opens a window of `rows` rows over `base` committed rows. Refused while a window is open:
    /// the state would be ahead of the lineage, and every old slot mark is cleared by the new
    /// window, never carried.
    pub(crate) fn begin_window(&mut self, base: usize, rows: usize) -> Result<(), EngError> {
        if self.window.is_some() {
            return Err(refuse("begin_window", "the previous window has no verdict"));
        }
        if rows < 1 || rows > self.plan.rows {
            return Err(refuse(
                "begin_window",
                &format!("{rows} rows, the slots serve 1 to {}", self.plan.rows),
            ));
        }
        self.window = Some(OpenWindow {
            base,
            rows,
            captured: 0,
            complete: false,
        });
        Ok(())
    }

    /// Marks that every row of the open window has run. From here until the verdict the state is
    /// ahead of the lineage: any step or export through it is refused (see `awaiting_verdict`).
    pub(crate) fn mark_complete(&mut self) -> Result<(), EngError> {
        let window = self
            .window
            .as_mut()
            .ok_or_else(|| refuse("mark_complete", "no window is open"))?;
        window.complete = true;
        Ok(())
    }

    /// True while a window has run all its rows and no verdict has come. A decode step or an export
    /// through the state in that time would continue from the speculative rows.
    pub(crate) fn awaiting_verdict(&self) -> bool {
        self.window.as_ref().is_some_and(|window| window.complete)
    }

    /// The slot that the copy after row `row` writes. Rows are captured in order, and the last row
    /// is never captured (its state is the live one).
    pub(crate) fn begin_capture(&self, row: usize) -> Result<usize, EngError> {
        let window = self
            .window
            .as_ref()
            .ok_or_else(|| refuse("capture", "no window is open"))?;
        if row + 1 >= window.rows {
            return Err(refuse(
                "capture",
                &format!("row {row} of {} has no slot", window.rows),
            ));
        }
        if row != window.captured {
            return Err(refuse(
                "capture",
                &format!("row {row} out of order, {} captured", window.captured),
            ));
        }
        Ok(row)
    }

    /// Marks the copy of `row` done. Called only after the copy was enqueued.
    pub(crate) fn note_captured(&mut self, row: usize) -> Result<(), EngError> {
        self.begin_capture(row)?;
        if let Some(window) = self.window.as_mut() {
            window.captured = row + 1;
        }
        Ok(())
    }

    /// Marks slots `0..count` captured in one call: the K-row path, whose kernels stored every
    /// slot inside one graph. The caller makes the call after the window's synchronization and
    /// result read, so a slot is never marked readable on an enqueue alone. Refused unless the
    /// window is open and still has no slot marked, not complete, and `count` is exactly one slot
    /// per row but the last: a partial mark would leave a verdict that names an unwritten slot.
    /// A one-row window has no slot, so `mark_captured_range(0)` is accepted on it and, marking
    /// nothing, can be called again; that is harmless.
    pub(crate) fn mark_captured_range(&mut self, count: usize) -> Result<(), EngError> {
        let window = self
            .window
            .as_mut()
            .ok_or_else(|| refuse("mark_captured_range", "no window is open"))?;
        if window.complete || window.captured != 0 {
            return Err(refuse(
                "mark_captured_range",
                &format!(
                    "the window already has {} slots marked or is complete",
                    window.captured
                ),
            ));
        }
        if window.rows.checked_sub(1) != Some(count) {
            return Err(refuse(
                "mark_captured_range",
                &format!("{count} slots marked for a window of {} rows", window.rows),
            ));
        }
        window.captured = count;
        Ok(())
    }

    /// Closes the window and says what the verdict asks for: `None` when every row stays. The
    /// window is closed whatever the answer, so a refused verdict leaves no open window.
    /// `committed_rows` is the verdict's `base + kept` and `tokens` the book's row count now.
    pub(crate) fn finish_window(
        &mut self,
        kept: usize,
        committed_rows: usize,
        tokens: usize,
    ) -> Result<Option<Restore>, EngError> {
        let window = self
            .window
            .take()
            .ok_or_else(|| refuse("commit", "no window is open"))?;
        if kept == 0 || kept > window.rows {
            return Err(refuse(
                "commit",
                &format!("a verdict keeps {kept} of {} rows", window.rows),
            ));
        }
        if committed_rows != window.base + kept {
            return Err(refuse(
                "commit",
                &format!(
                    "the verdict commits {committed_rows} rows, the window starts at {} and keeps {kept}",
                    window.base
                ),
            ));
        }
        if tokens != window.base + window.rows {
            return Err(refuse(
                "commit",
                &format!(
                    "the state holds {tokens} rows, the window ends at {}",
                    window.base + window.rows
                ),
            ));
        }
        if kept == window.rows {
            return Ok(None);
        }
        let slot = kept - 1;
        if slot >= window.captured {
            return Err(refuse(
                "commit",
                &format!("slot {slot} was never captured ({} were)", window.captured),
            ));
        }
        Ok(Some(Restore {
            slot,
            frontier: window.base + kept,
        }))
    }

    /// Drops the open window and every slot mark (a reset, an import, a poison, a failure).
    pub(crate) fn discard(&mut self) {
        self.window = None;
    }
}

/// Refuses a window whose rows do not fit the KV capacity, before any row is fed: row `i` writes
/// cell `base + i`, so `base + rows` cells must exist.
pub(crate) fn check_window_fits(base: usize, rows: usize, capacity: usize) -> Result<(), EngError> {
    match base.checked_add(rows) {
        Some(end) if end <= capacity => Ok(()),
        _ => Err(EngError::ContextLimit(format!(
            "device rollback: a window of {rows} rows over {base} committed rows exceeds the KV capacity {capacity}"
        ))),
    }
}

/// What the step knows when the device rollback is armed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ArmingFacts {
    /// The step has already run a token.
    pub(crate) started: bool,
    /// Slots are already armed.
    pub(crate) armed: bool,
    pub(crate) tiered: bool,
    /// Sparse attention is armed: its sketch cells have a host prefix that a frontier truncation
    /// does not rewind.
    pub(crate) sparse: bool,
    /// The KV is host-paged or device-first: no resident plane to truncate.
    pub(crate) paged: bool,
    /// Experts run on branch streams: the copies are ordered on one stream only.
    pub(crate) forked_experts: bool,
}

/// The refusals of arming that need no device, in the order the step reports them. Every one is
/// answered before the plan or the allocation, with the step untouched.
pub(crate) fn check_arming(facts: ArmingFacts) -> Result<(), EngError> {
    let refuse = |why: &str| {
        EngError::BadRequest(format!(
            "draft_rollback device: {why}; set draft_rollback to host"
        ))
    };
    if facts.started {
        return Err(refuse("the step has already run"));
    }
    if facts.armed {
        return Err(refuse("the slots are already armed"));
    }
    if facts.tiered {
        return Err(refuse(
            "tiered experts keep a bank state the slots do not hold",
        ));
    }
    if facts.forked_experts {
        return Err(refuse(
            "forked experts run on branch streams and the copies are ordered on one stream",
        ));
    }
    if facts.sparse || facts.paged {
        return Err(refuse(
            "sparse attention or a paged KV cache is armed and the frontier is not proven for it",
        ));
    }
    Ok(())
}

/// The memory check of the case no prefill admission charges the slots (`arena_wanted` false: the
/// FP16 KV with chunk prefill off). Then the slot bytes and the device tip are covered only by the
/// load-time reserve `prefill_headroom(fixed_admission)` that nothing else consumes in that case,
/// and start is refused when they do not fit it. Pure, so the rule is tested on the host.
pub(crate) fn check_uncharged_slots(
    slot_bytes: usize,
    tip_bytes: usize,
    headroom: usize,
) -> Result<(), EngError> {
    let need = slot_bytes.checked_add(tip_bytes).ok_or_else(|| {
        EngError::MemoryPlan("device rollback: slot and tip bytes overflow".to_owned())
    })?;
    if need > headroom {
        return Err(EngError::MemoryPlan(format!(
            "device rollback: no prefill admission charges the rollback slots here, and the {slot_bytes} B \
             of slots plus {tip_bytes} B of device tip exceed the {headroom} B load reserve that covers \
             them; lower draft_max_rows or set draft_rollback to host"
        )));
    }
    Ok(())
}

/// How the slot bytes are covered, for the startup line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Cover {
    /// The prefill arena admission charged them as an `Extra`.
    Arena,
    /// No admission ran; the load reserve covers them.
    Reserve { headroom: usize },
}

/// The startup line of an armed rollback: the bytes, whether they were charged or covered by the
/// reserve, and whether the arena admission fell back to per-token prefill (`arena` is `Some(true)`
/// for an admitted arena, `Some(false)` for none, `None` when no admission ran).
pub(crate) fn startup_line(
    plan: &RollbackPlan,
    cover: Cover,
    arena: Option<bool>,
) -> Result<String, EngError> {
    let covered = match cover {
        Cover::Arena => "charged to the prefill arena admission as an extra".to_owned(),
        Cover::Reserve { headroom } => {
            format!("covered by the {headroom} B load reserve (no arena admission charges them)")
        }
    };
    let prefill = match arena {
        Some(true) => "prefill arena admitted",
        Some(false) => {
            "prefill arena NOT admitted: per-token prefill (the slots may have taken its room)"
        }
        None => "no prefill arena admission ran",
    };
    Ok(format!(
        "q36 cuda: device rollback, {} slots of {} B, {} B total, {covered}; {prefill}; \
         draft_rollback = host restores the default",
        plan.slots(),
        plan.slot_bytes,
        plan.total_bytes()?
    ))
}

#[cfg(test)]
#[path = "rollback_tests.rs"]
mod tests;
