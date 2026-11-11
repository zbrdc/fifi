//! Mutation hooks of the conv prefill and narrow-block scan switches.
//!
//! [`mutated`] answers whether the mutant `id` is the active one. Without the `cuda-gdn-mutants`
//! feature it is the constant `false`, so the branches that ask it compile away and a build without
//! the feature contains no mutant. With the feature a test selects one mutant at a time with [`select`]
//! (ids 1 to 11 and 19 to 31 reach the kernels through `native/resident/gdn_mutants.h` — 23 is the
//! bound guard, 24 to 31 the chunk-parallel scan; ids 12 to 18, 22 and 32 are Rust-side branches
//! of the switch code) and the mutation tests check that the reference-pinned tests notice it.
//! The ids are listed in the header.

/// The conv launch ignores the switch and always runs the vendored kernel.
pub(crate) const ROUTE_IGNORED: u32 = 12;
/// The conv prefill threshold is one row.
pub(crate) const MIN_ROWS_ONE: u32 = 13;
/// `FIFI_GDN_CONV_PREFILL=1` selects 16 rows per thread.
pub(crate) const ONE_IS_SIXTEEN: u32 = 14;
/// The conv switch value is not trimmed.
pub(crate) const CONV_NO_TRIM: u32 = 15;
/// The conv switch accepts 8.
pub(crate) const CONV_ACCEPTS_EIGHT: u32 = 16;
/// The scan switch accepts 4.
pub(crate) const SCAN_ACCEPTS_FOUR: u32 = 17;
/// The scan switch value is not trimmed.
pub(crate) const SCAN_NO_TRIM: u32 = 18;
/// The scan threshold is one row.
pub(crate) const SCAN_MIN_ONE: u32 = 22;
/// The chunk-scan threshold is one row.
pub(crate) const CHUNK_MIN_ONE: u32 = 32;

#[cfg(not(feature = "cuda-gdn-mutants"))]
pub(crate) const fn mutated(_id: u32) -> bool {
    false
}

#[cfg(feature = "cuda-gdn-mutants")]
pub(crate) fn mutated(id: u32) -> bool {
    id != 0 && active::ACTIVE.load(std::sync::atomic::Ordering::Relaxed) == id
}

/// Whether the tests should use their short shape lists: always, inside a mutation test (see [`select`]).
#[cfg(all(test, not(feature = "cuda-gdn-mutants")))]
pub(crate) const fn quick() -> bool {
    false
}

#[cfg(all(test, feature = "cuda-gdn-mutants"))]
pub(crate) use active::{quick, select};

#[cfg(feature = "cuda-gdn-mutants")]
mod active {
    use std::sync::atomic::AtomicU32;
    #[cfg(test)]
    use std::{cell::Cell, sync::atomic::Ordering};

    #[cfg(test)]
    use crate::kernels::cuda::native::resident::gdn_conv::select_gdn_mutant;

    pub(super) static ACTIVE: AtomicU32 = AtomicU32::new(0);

    #[cfg(test)]
    thread_local! {
        static QUICK: Cell<bool> = const { Cell::new(false) };
    }

    #[cfg(test)]
    pub(crate) fn quick() -> bool {
        QUICK.with(Cell::get)
    }

    /// The mutant that is active until this guard drops; dropping selects none.
    #[cfg(test)]
    pub(crate) struct Selected(());

    /// Makes `id` the active mutant (0 is none, the control) and switches the tests of this thread to their
    /// short shape lists. The kernels read the id at launch, so select it before the call under test and
    /// run the mutation tests with `--test-threads=1`.
    #[cfg(test)]
    pub(crate) fn select(id: u32) -> Selected {
        set(id);
        QUICK.with(|quick| quick.set(true));
        Selected(())
    }

    #[cfg(test)]
    fn set(id: u32) {
        ACTIVE.store(id, Ordering::Relaxed);
        let kernel_id = if matches!(id, 1..=11 | 19..=31) {
            id
        } else {
            0
        };
        assert!(
            select_gdn_mutant(kernel_id),
            "the native mutant selector refused {kernel_id}"
        );
    }

    #[cfg(test)]
    impl Drop for Selected {
        fn drop(&mut self) {
            set(0);
            QUICK.with(|quick| quick.set(false));
        }
    }
}
