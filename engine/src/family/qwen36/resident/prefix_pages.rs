//! The page-table allowance of the state-only prefix entries (one rule):
//! an entry charges the fixed DeltaNet bytes plus the bounded page-table allowance of its
//! alias - never the KV of its positions. The allowance is a pure position-to-pages rule,
//! pinned by a portable test; the earlier PagePool/Alias bookkeeping never reached
//! production (the live snapshot charges the rule directly) and is deleted.

/// Positions per page. The stride stays a constant so the budget allowance stays predictable;
/// 2,048 matches the design note.
pub(super) const PAGE_POSITIONS: u32 = 2_048;

/// One page-table row: the cost unit the allowance charges (16 bytes: two u32 fields and a
/// usize refcount, pinned by the portable test below).
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Page {
    pub start: u32,
    pub len: u32,
    refs: usize,
}

/// THE allowance rule: the page-table rows an alias of `pos` positions covers. Exactly one
/// formula for the whole engine, never two; the snapshot charges this, nothing
/// else re-derives it.
pub(super) fn allowance_pages(pos: u32) -> u64 {
    (u64::from(pos) / PAGE_POSITIONS as u64 + 1).max(1)
}

/// The allowance in bytes: the page-table rows at the size of one `Page`.
pub(super) fn allowance_bytes(pos: u32) -> u64 {
    allowance_pages(pos) * std::mem::size_of::<Page>() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one rule, pinned: page counts at the stride boundaries and the byte cost of the
    /// page-table row (the 16-byte unit the snapshot permits multiply by).
    #[test]
    fn the_allowance_rule_is_pinned_at_the_stride_boundaries() {
        assert_eq!(PAGE_POSITIONS, 2_048, "the design-note stride");
        assert_eq!(std::mem::size_of::<Page>(), 16, "the cost unit");
        assert_eq!(
            allowance_pages(0),
            1,
            "a zero-length alias still holds one row"
        );
        assert_eq!(allowance_pages(1), 1);
        assert_eq!(
            allowance_pages(PAGE_POSITIONS - 1),
            1,
            "the last row of page one"
        );
        assert_eq!(
            allowance_pages(PAGE_POSITIONS),
            2,
            "the first row of page two"
        );
        assert_eq!(allowance_pages(2 * PAGE_POSITIONS), 3);
        assert_eq!(allowance_bytes(0), 16);
        assert_eq!(allowance_bytes(PAGE_POSITIONS), 32);
    }
}
