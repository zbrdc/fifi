//! Per-category transport counters for one process. Zero cost when
//! NEM35_TRANSPORT_ACCOUNTING is unset: every bump checks one atomic bool.
//! shortcut: counters are process-global and unsynchronized across threads;
//! the generation runner is single-threaded per process today.
#![allow(
    dead_code,
    reason = "diagnostics kit: production wires only the record_* half (cuda_projection); the enable/snapshot reporting half serves the module test and the streaming tier's upcoming report"
)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);

/// One counter cell per category; ordering by declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Category {
    Dense,
    Shared,
    Routed,
}

impl Category {
    fn index(self) -> usize {
        match self {
            Self::Dense => 0,
            Self::Shared => 1,
            Self::Routed => 2,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Shared => "shared",
            Self::Routed => "routed",
        }
    }
}

const CATEGORIES: usize = 3;
static H2D_BYTES: [AtomicU64; CATEGORIES] = [const { AtomicU64::new(0) }; CATEGORIES];
static DEVICE_CALLS: [AtomicU64; CATEGORIES] = [const { AtomicU64::new(0) }; CATEGORIES];

pub(crate) fn enable() {
    ENABLED.store(true, Ordering::Relaxed);
}

pub(crate) fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Records one host-to-device upload and one device call (the native wrappers
/// synchronize once per call) for `category`.
pub(crate) fn record_upload(category: Category, bytes: usize) {
    if !enabled() {
        return;
    }
    let bytes = u64::try_from(bytes).unwrap_or(u64::MAX);
    H2D_BYTES[category.index()].fetch_add(bytes, Ordering::Relaxed);
    DEVICE_CALLS[category.index()].fetch_add(1, Ordering::Relaxed);
}

/// Records one device call that reused resident bytes (no upload).
pub(crate) fn record_resident_call(category: Category) {
    if !enabled() {
        return;
    }
    DEVICE_CALLS[category.index()].fetch_add(1, Ordering::Relaxed);
}

/// Snapshot of one category's counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Snapshot {
    pub(crate) h2d_bytes: u64,
    pub(crate) device_calls: u64,
}

pub(crate) fn snapshot(category: Category) -> Snapshot {
    Snapshot {
        h2d_bytes: H2D_BYTES[category.index()].load(Ordering::Relaxed),
        device_calls: DEVICE_CALLS[category.index()].load(Ordering::Relaxed),
    }
}

pub(crate) fn all_snapshots() -> [(&'static str, Snapshot); CATEGORIES] {
    [
        (Category::Dense.name(), snapshot(Category::Dense)),
        (Category::Shared.name(), snapshot(Category::Shared)),
        (Category::Routed.name(), snapshot(Category::Routed)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_counters_stay_zero_and_enabled_counters_separate_categories() {
        // Not enabled: nothing records.
        record_upload(Category::Dense, 100);
        assert_eq!(snapshot(Category::Dense).h2d_bytes, 0);
        assert_eq!(snapshot(Category::Dense).device_calls, 0);
        // Enabled: bytes and calls accumulate per category, resident calls
        // count calls without bytes.
        enable();
        record_upload(Category::Routed, 10);
        record_upload(Category::Routed, 5);
        record_resident_call(Category::Shared);
        assert_eq!(
            snapshot(Category::Routed),
            Snapshot {
                h2d_bytes: 15,
                device_calls: 2
            }
        );
        assert_eq!(
            snapshot(Category::Shared),
            Snapshot {
                h2d_bytes: 0,
                device_calls: 1
            }
        );
        assert_eq!(snapshot(Category::Dense).device_calls, 0);
    }
}
