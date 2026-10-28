//! Test-only attention dispatch counters, thread-local and release-absent.
//!
//! The decode-attention slice's RED/GREEN contract is a routing counter: at
//! `rows == 1` the flagged path must dispatch the VENDOR fused SDPA and must
//! NOT dispatch the prefill-shaped float flash. Release builds never link
//! this module, so the default path is unchanged.

use std::cell::Cell;

thread_local! {
    static PREFILL_FLASH_T1: Cell<u64> = const { Cell::new(0) };
    static DECODE_VENDOR_T1: Cell<u64> = const { Cell::new(0) };
}

pub(in crate::kernels::mlx::native) fn note_prefill_flash(rows: u32) {
    if rows == 1 {
        PREFILL_FLASH_T1.with(|c| c.set(c.get() + 1));
    }
}

// Called by the shared-attention dispatch when the FIFI_FLINT_MLX_DECODE_ATTN
// predicate routes a one-row call to the vendor fused SDPA (test builds only).
pub(in crate::kernels::mlx) fn note_decode_vendor() {
    DECODE_VENDOR_T1.with(|c| c.set(c.get() + 1));
}

pub(crate) fn reset() {
    PREFILL_FLASH_T1.with(|c| c.set(0));
    DECODE_VENDOR_T1.with(|c| c.set(0));
}

/// `(float-flash calls at rows == 1, vendor fused-SDPA routes at rows == 1)`.
pub(crate) fn counts() -> (u64, u64) {
    (
        PREFILL_FLASH_T1.with(Cell::get),
        DECODE_VENDOR_T1.with(Cell::get),
    )
}
