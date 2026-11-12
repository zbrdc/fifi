//! The prefix-resume switch (`INFERENCE_FLASH_NEXT_PREFIX_RESUME`): `1`, `true` or `on` arms the
//! save/restore of the boundary state; unset, `0`, `false` or `off` leaves the executor exactly as
//! before (no save, no restore, the reset zeroes the whole state arena). The spelling is checked
//! at driver build with the other Flash-Next switches, so a value that would misread never arms.
//!
//! With the switch on, a job's reset zeroes only the fixed state (the GDN recurrent state, its
//! conv rings, the PLE ring, the words AND the QSA raw-key ring): the QSA K/V planes and the
//! indexer keys stay in place, because no kernel reads a position at or past the valid length,
//! so a fresh sequence never observes them. The raw-key ring is copied, not left in place: its
//! slot `p % ratio` accumulates the open block's raw keys, so the saving job's own continuation
//! overwrites the slots a mid-block boundary still needs. A restore copies the fixed state back
//! and rewinds the valid length onto the kept rows.
//!
//! The rows a boundary leaves in place are only valid while nothing else has written positions
//! below it. The driver's arena generation advances at the attempt of every fresh sequence's
//! first row (position 0 — a failed first chunk may still have queued the writes) and the entry
//! carries the generation as the prefix store's path, so a
//! restore after any intervening fresh prefill never matches the entry and falls back to a full
//! prefill: request A saves, request C prefills from 0 and overwrites the rows, request D with
//! A's prefix refuses the entry (generation mismatch) and full-prefills — correct output, no
//! stale-row read, no error. A restore at boundary b additionally drops every stored entry
//! with a LONGER key before any row is written (`drop_above`): the restoring job's own
//! continuation overwrites the rows those entries describe, and a restore does not start a
//! fresh sequence, so the generation alone never retires them.

use std::sync::OnceLock;

use super::driver::chunk::prof;

#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};

/// Forces the switch on for a test (the env value is read once and cannot be flipped in a
/// process; only the save/restore overrides read this).
#[cfg(test)]
static TEST_ON: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
fn test_on() -> bool {
    TEST_ON.load(Ordering::Relaxed)
}

#[cfg(not(test))]
fn test_on() -> bool {
    false
}

pub(in crate::family::flash_next) fn on() -> bool {
    if test_on() {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        prof::parse_switch(
            std::env::var("INFERENCE_FLASH_NEXT_PREFIX_RESUME")
                .ok()
                .as_deref(),
        )
        .unwrap_or(false)
    })
}

#[cfg(test)]
pub(in crate::family::flash_next) fn set_for_tests(on: bool) {
    TEST_ON.store(on, Ordering::Relaxed);
}

#[cfg(test)]
pub(in crate::family::flash_next) fn arm_for_tests() {
    set_for_tests(true);
}

/// The host bytes the resume store may hold across its entries: the budget a save charges
/// on the survivors after the store makes room, plus the would-be entry
/// (`survivors + one more snapshot <= budget`). The snapshots are pageable host RAM — about
/// 110.8 MiB of fixed state each on this pack plus the PLE history term — so the line, not
/// the entry count, is the machine's ceiling: a rolling two-entry store always fits (one
/// survivor plus the new entry), and any future store shape that would cross it degrades to
/// fewer entries instead of growing host RAM. A denied save is skipped; correctness never
/// depends on it.
pub(in crate::family::flash_next) const HOST_BUDGET_BYTES: u64 = 256 * 1024 * 1024;
