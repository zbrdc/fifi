//! The pure pieces of the one-entry prefix reuse of the CUDA arm (decision 0036, addendum 1):
//! the numeric path tag a saved entry carries, and the host-byte charge of an entry. The glue
//! itself lives in `executor.rs` (the two closures around the shared session rules); the device
//! behaviour is `prefix_device_tests.rs`.

use super::state::{
    export::ExportedState,
    pinned::{PendingPinned, PinnedSnapshot},
};
use super::{dims::KvMode, state::export::plane_sizes, state_layout::StateLayout};
use crate::{
    EngError,
    family::session::{DecodeSession, SaveKind},
};

pub(super) enum CudaSavedPrefix {
    Pageable(ExportedState),
    Pinned(PinnedSnapshot),
}

pub(super) struct PendingPrefix {
    pub(super) session: DecodeSession,
    pub(super) fed: usize,
    pub(super) kind: SaveKind,
    pub(super) path: CudaPrefillPath,
    pub(super) state: PendingPinned,
}

/// The numeric prefill path that built a saved entry: a state restores only under the path that
/// built it, because a batched prefill matches one-token steps only within a tolerance (decision
/// 0036, item 7). The key also binds the actual KV format before native restore validation;
/// chunked prefill (decision 0049) must add `Chunked(n)` and pass it here, exactly like the
/// Flash-Next twin.
///
/// Neither this path nor `StateIdentity` covers the WEIGHT LAYOUT: an entry matches the device
/// buffers it was exported from only while it lives in memory, inside one step's lifetime (one
/// process, one loaded model, where the layout cannot change). Persisting an entry, or a store
/// shared by several steps, must add the weight layout to what a restore checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CudaPrefillPath {
    Steps(KvMode),
}

fn overflow() -> EngError {
    EngError::MemoryPlan("prefix entry: a size overflows".to_owned())
}

/// Host bytes of one saved entry at `positions` committed positions: the DeltaNet export
/// (recurrent plus convolution, f32, one layer's worth per DeltaNet layer) plus the exported KV
/// planes of the first `positions` rows of every attention layer. Checked arithmetic only: an
/// absurd count is a `MemoryPlan` refusal, never a panic.
pub(super) fn entry_bytes(layout: &StateLayout, positions: usize) -> Result<usize, EngError> {
    let pool = layout.delta;
    let per_layer = pool
        .rec_values
        .checked_add(pool.conv_values)
        .and_then(|values| values.checked_mul(size_of::<f32>()))
        .and_then(|bytes| bytes.checked_mul(pool.layers))
        .ok_or_else(overflow)?;
    let mut total = per_layer;
    for ordinal in 0..layout.kv.layers {
        for bytes in plane_sizes(&layout.kv, ordinal, positions)? {
            total = total.checked_add(bytes).ok_or_else(overflow)?;
        }
    }
    Ok(total)
}
