use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

use crate::EngError;

pub mod batch;
#[allow(
    dead_code,
    reason = "H4 wires the server to the engine handle; until then only tests call the job types"
)]
pub(crate) mod job;
#[allow(
    dead_code,
    reason = "H4 wires the server to the engine handle; until then only tests call the engine worker"
)]
pub(crate) mod worker;

pub struct SessionBudget {
    cap: u64,
    used: Rc<Cell<u64>>,
}

impl SessionBudget {
    pub fn new(cap: u64) -> Self {
        Self {
            cap,
            used: Rc::new(Cell::new(0)),
        }
    }

    pub fn used(&self) -> u64 {
        self.used.get()
    }

    /// The bytes the budget can still charge: the cap over the used bytes.
    /// Crate-visible only - this is the executors' pool-derived prompt ceiling, not public
    /// API. The only caller is the Mac-only Q36 resident executor, so a non-Mac build
    /// cannot see the use.
    #[allow(dead_code)]
    pub(crate) fn free(&self) -> u64 {
        self.cap.saturating_sub(self.used())
    }

    pub fn acquire_growth(&mut self, bytes: u64) -> Result<GrowthPermit, EngError> {
        let next = self
            .used()
            .checked_add(bytes)
            .ok_or_else(|| EngError::MemoryPlan("session growth overflows u64".into()))?;
        if next > self.cap {
            return Err(EngError::MemoryPlan(format!(
                "session growth requires {bytes} bytes, available {}",
                self.cap.saturating_sub(self.used())
            )));
        }
        self.used.set(next);
        Ok(GrowthPermit {
            shared: Rc::new(RefCell::new(Some((Rc::clone(&self.used), bytes)))),
        })
    }
}

/// One growth charge against a session budget. The charge is SHAREABLE: `share` produces a
/// co-owned permit that holds the SAME bytes, and the budget counts those bytes exactly
/// once no matter how many co-owners live - only the last one's drop releases them. The
/// shared state carries the counter and the bytes as an Option taken by the last dropper.
pub struct GrowthPermit {
    shared: SharedCharge,
}

/// The budget counter and the charged bytes, co-owned by every share of one permit.
type SharedCharge = Rc<RefCell<Option<(Rc<Cell<u64>>, u64)>>>;

impl GrowthPermit {
    /// A co-owned permit sharing this charge: the budget counts the bytes once while any
    /// co-owner lives, and the last drop releases them.
    pub fn share(&self) -> Self {
        Self {
            shared: Rc::clone(&self.shared),
        }
    }

    pub fn retained_bytes(&self) -> u64 {
        self.shared.borrow().as_ref().map(|(_, b)| *b).unwrap_or(0)
    }
}

impl Drop for GrowthPermit {
    fn drop(&mut self) {
        if Rc::strong_count(&self.shared) == 1
            && let Some((used, bytes)) = self.shared.borrow_mut().take()
        {
            used.set(used.get() - bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SessionBudget;
    use crate::EngError;

    fn memory_plan(message: &str) -> Option<EngError> {
        Some(EngError::MemoryPlan(message.to_string()))
    }

    /// A new budget has charged nothing, and each permit charges its bytes until it drops.
    ///
    /// Mutation check: acquiring without `self.used.set(next)` leaves `used` at 0 after the
    /// first grant.
    #[test]
    fn a_grant_charges_the_budget_and_the_permit_reports_its_bytes() {
        let mut budget = SessionBudget::new(100);
        assert_eq!(budget.used(), 0, "a new budget is uncharged");
        let first = budget.acquire_growth(30).expect("30 of 100");
        assert_eq!(budget.used(), 30);
        assert_eq!(first.retained_bytes(), 30);
        let second = budget.acquire_growth(50).expect("80 of 100");
        assert_eq!(budget.used(), 80, "charges add up");
        assert_eq!(second.retained_bytes(), 50);
        assert_eq!(
            first.retained_bytes(),
            30,
            "the first permit keeps its own bytes"
        );
    }

    /// The cap is inclusive: growth that lands exactly on it is granted, one byte more is
    /// refused, and the refusal names the request and what was left.
    ///
    /// Mutation check: comparing with `next >= self.cap` refuses the grant that fills the
    /// budget.
    #[test]
    fn growth_up_to_the_cap_is_granted_and_one_byte_past_it_is_refused() {
        let mut budget = SessionBudget::new(100);
        let held = budget.acquire_growth(60).expect("60 of 100");
        let error = budget.acquire_growth(41).err();
        assert_eq!(
            error,
            memory_plan("session growth requires 41 bytes, available 40")
        );
        let fill = budget
            .acquire_growth(40)
            .expect("exactly the 40 that are left");
        assert_eq!(budget.used(), 100, "the budget is full");
        assert_eq!(fill.retained_bytes() + held.retained_bytes(), 100);
        let error = budget.acquire_growth(1).err();
        assert_eq!(
            error,
            memory_plan("session growth requires 1 bytes, available 0")
        );
    }

    /// A refused request charges nothing: the live permits still release exactly their own
    /// bytes, so the budget returns to zero rather than underflowing.
    ///
    /// Mutation check: setting `used` before the cap comparison leaves `used` at 101 after
    /// the refusal.
    #[test]
    fn a_refused_request_leaves_the_budget_unchanged() {
        let mut budget = SessionBudget::new(100);
        let held = budget.acquire_growth(70).expect("70 of 100");
        assert!(budget.acquire_growth(31).is_err(), "101 of 100");
        assert_eq!(budget.used(), 70, "the refusal charged nothing");
        drop(held);
        assert_eq!(budget.used(), 0, "the one live permit released its 70");
    }

    /// Dropping a permit returns exactly its bytes, in any drop order, and the freed room
    /// can be granted again.
    ///
    /// Mutation check: a `drop` that resets `used` to 0 instead of subtracting reports 0 after
    /// the first drop instead of 70.
    #[test]
    fn dropping_a_permit_returns_its_bytes_to_the_budget() {
        let mut budget = SessionBudget::new(100);
        let low = budget.acquire_growth(20).expect("20 of 100");
        let mid = budget.acquire_growth(30).expect("50 of 100");
        let high = budget.acquire_growth(50).expect("100 of 100");
        drop(mid);
        assert_eq!(budget.used(), 70, "the middle permit released its 30");
        let reused = budget.acquire_growth(30).expect("the freed 30 fit again");
        assert_eq!(budget.used(), 100);
        drop(high);
        drop(low);
        assert_eq!(budget.used(), 30, "only the reused permit is left");
        drop(reused);
        assert_eq!(budget.used(), 0);
    }

    /// A zero-byte permit is granted even when the budget is full, and it charges and
    /// releases nothing. A zero cap grants nothing else.
    ///
    /// Mutation check: refusing when `next >= self.cap` fails the zero-byte grant on a full
    /// budget and on the zero cap.
    #[test]
    fn zero_byte_growth_is_always_granted_and_costs_nothing() {
        let mut full = SessionBudget::new(64);
        let held = full.acquire_growth(64).expect("fills the budget");
        let nothing = full.acquire_growth(0).expect("zero bytes on a full budget");
        assert_eq!(nothing.retained_bytes(), 0);
        assert_eq!(full.used(), 64);
        drop(nothing);
        assert_eq!(full.used(), 64, "a zero-byte permit releases nothing");
        drop(held);
        assert_eq!(full.used(), 0);

        let mut empty = SessionBudget::new(0);
        assert!(empty.acquire_growth(0).is_ok(), "zero of zero");
        let error = empty.acquire_growth(1).err();
        assert_eq!(
            error,
            memory_plan("session growth requires 1 bytes, available 0")
        );
    }

    /// Growth that would wrap `u64` is refused as an overflow, not as a cap miss, and
    /// charges nothing. A cap of `u64::MAX` still admits a grant that lands on it.
    ///
    /// Mutation check: `wrapping_add` instead of `checked_add` grants the request.
    #[test]
    fn growth_that_overflows_u64_is_refused_without_charging() {
        let mut budget = SessionBudget::new(u64::MAX);
        let held = budget.acquire_growth(1).expect("1 of u64::MAX");
        let error = budget.acquire_growth(u64::MAX).err();
        assert_eq!(error, memory_plan("session growth overflows u64"));
        assert_eq!(budget.used(), 1, "the overflow charged nothing");
        drop(held);

        let all = budget.acquire_growth(u64::MAX).expect("lands on the cap");
        assert_eq!(budget.used(), u64::MAX);
        assert_eq!(
            budget.acquire_growth(1).err(),
            memory_plan("session growth overflows u64")
        );
        drop(all);
        assert_eq!(budget.used(), 0);
    }

    /// A permit holds its own share of the counter, so it may outlive the budget that issued
    /// it and still release its bytes (the counter lives in the shared tuple that both the
    /// budget and every permit hold, so dropping the budget keeps the count).
    #[test]
    fn a_permit_may_outlive_its_budget() {
        let mut budget = SessionBudget::new(10);
        let permit = budget.acquire_growth(10).expect("10 of 10");
        drop(budget);
        assert_eq!(
            permit.retained_bytes(),
            10,
            "the charge outlives the budget"
        );
        // The permit holds the counter Rc beside the dropped budget, so its drop
        // releases without a dangling reference; the release itself is pinned by
        // a_shared_permit_counts_once_and_releases_on_the_last_drop below, which
        // observes the same counter through a living budget.
        drop(permit);
    }

    /// The shareable charge counts once and only the last drop releases it - the portable
    /// pin of the shared-charge linchpin the snapshot permits and the session budget
    /// arithmetic rest on.
    #[test]
    fn a_shared_permit_counts_once_and_releases_on_the_last_drop() {
        let mut budget = SessionBudget::new(10);
        let permit = budget.acquire_growth(10).expect("10 of 10");
        let shared = permit.share();
        assert_eq!(
            budget.used(),
            10,
            "a share does not double-count the charge"
        );
        drop(permit);
        assert_eq!(
            budget.used(),
            10,
            "a surviving share still holds the charge"
        );
        drop(shared);
        assert_eq!(budget.used(), 0, "the last drop releases the charge");
    }
}
