//! Resident-decode operations: the kernels a decode step runs with its weights and state on
//! the device. Each op states its contract, refuses before any native call, and keeps its tests
//! beside it, the shape of the CUDA adapter's resident tree.
//!
//! Every op checks its ranges the same way, with [`require_range`] and [`overlaps`]: a range
//! belongs to the call's own context, holds the bytes the op reads or writes and starts on the
//! alignment its loads need, and a result shares no byte with an input.

use crate::EngError;

use super::context::{DeviceRange, Shared};

pub(crate) mod dense_gemv;
pub(crate) mod relative_attention;
pub(crate) mod rms_norm;
pub(crate) mod row_add;
pub(crate) mod short_conv;
pub(crate) mod softcap;
pub(crate) mod swiglu_rows;

/// The alignment Metal requires of a bound buffer range.
pub(super) const RANGE_ALIGN: usize = 4;

/// A range of a call: of this context, at least `need` bytes, at a multiple of `align` bytes
/// into its buffer.
pub(super) fn require_range(
    shared: &Shared,
    what: &str,
    range: &DeviceRange<'_>,
    need: usize,
    align: usize,
) -> Result<(), EngError> {
    let same = range.owner() == shared.id();
    let fits = range.bytes() >= need && range.offset().is_multiple_of(align);
    if !same || !fits {
        let whose = if same { "" } else { " of another adapter" };
        return Err(EngError::BadRequest(format!(
            "the call needs {need} bytes of {what} at a multiple of {align} in a buffer of its own Metal context; the range holds {} bytes at offset {}{whose}",
            range.bytes(),
            range.offset(),
        )));
    }
    Ok(())
}

/// Whether two ranges share an address.
pub(super) fn overlaps(a: &DeviceRange<'_>, b: &DeviceRange<'_>) -> bool {
    a.address() < b.address() + b.bytes() && b.address() < a.address() + a.bytes()
}
