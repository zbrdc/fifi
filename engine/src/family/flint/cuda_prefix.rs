//! Flint CUDA prefix snapshots (slice P1): the device half of the shared
//! prefix store. The session framework owns token lineage, kinds, and LRU;
//! this module owns WHAT a CUDA snapshot is and WHICH numeric path it binds.
//!
//! kv_profile `f16` only (the Q8 scale planes are dead code; save and
//! restore refuse under `q8`). A snapshot is a committed-bounds host copy
//! of every layer's keys and values plus the health buffer, taken after a
//! stream quiesce at a committed boundary. Rows below the first wrap sit
//! at slots `[0, committed)` (slot = position % cache_rows), so the live
//! rows are a plane-prefix byte range; a wrapped ring needs the whole
//! plane. Copying only that range bounds the host charge to the rows the
//! conversation actually used.
//!
//! HOST-RAM BUDGET: the snapshot pins host memory equal
//! to the committed rows' plane bytes plus the health buffer for as long
//! as the entry lives. The save admits that charge against the host and
//! cgroup budget (`HostBudget`, the same admission the expert cache uses)
//! BEFORE allocating, and the allocation is fallible; a refusal carries
//! the planned bytes in a MemoryPlan error. Device KV is NOT doubled (a
//! restore refills the existing planes).

use crate::EngError;
use std::path::Path;

type Result<T> = std::result::Result<T, EngError>;

/// The EFFECTIVE (process-frozen) selector values a [`CudaPrefixPath`]
/// binds — never raw environment strings (raw values would
/// distinguish spellings that resolve to the same state). The executor
/// computes them with the same helpers the numeric path reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CudaPrefixEnv {
    pub(crate) alloc_cache: bool,
    pub(crate) late_health: bool,
    /// The packed-pointwise expert path selector (a
    /// snapshot taken on one pointwise path must not restore onto the
    /// other).
    pub(crate) packed_pointwise: bool,
    /// The midrows legacy rollback arming (`FIFI_EXL3_LEGACY=1`): the
    /// rows 64..=144 routing selector.
    pub(crate) exl3_legacy: bool,
    /// The expert host-cache opt-in (`FIFI_FLINT_CUDA_EXPERT_CACHE`):
    /// changes the load path and therefore the numerics a snapshot binds.
    pub(crate) expert_cache: bool,
}

/// The numeric-path fingerprint a stored CUDA prefix binds: context limit,
/// prefill chunk size, kv profile, the five effective selectors (alloc
/// cache, late health, packed pointwise, midrows legacy, expert cache),
/// and the model root. Mirrors the MLX `FlintPath` contract.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct CudaPrefixPath {
    limit: u32,
    chunk: u32,
    profile: String,
    env: CudaPrefixEnv,
    root: std::path::PathBuf,
}

impl CudaPrefixPath {
    /// The pure half: every input explicit, so the fingerprint contract is
    /// testable without mutating the process environment (real-env coverage
    /// stays with the per-arm test-bin processes of the qualification window).
    pub(crate) fn from_values(
        root: &Path,
        limit: u32,
        chunk: u32,
        profile: &str,
        env: CudaPrefixEnv,
    ) -> Self {
        Self {
            limit,
            chunk,
            profile: profile.to_owned(),
            env,
            root: root.to_path_buf(),
        }
    }

    /// `from_values` with the current process state, as the executor reads
    /// it at save/restore time: the frozen effective selectors, not the raw
    /// environment.
    pub(crate) fn now(
        root: &Path,
        options: &super::settings::FlintOptions,
        env: CudaPrefixEnv,
    ) -> Self {
        Self::from_values(
            root,
            options.context_limit,
            options.prefill_chunk_size,
            &options.kv_profile,
            env,
        )
    }
}

/// The f16-only scope gate: `q8` refuses with the message naming both
/// profiles. Pure in the profile string so the refusal is testable on any
/// host.
pub(crate) fn profile_allows_prefix(profile: &str) -> Result<()> {
    if profile == "q8" {
        return Err(EngError::BadRequest(
            "flint cuda prefix reuse requires kv_profile f16: the q8 scale planes are not integrated".into(),
        ));
    }
    Ok(())
}

/// One layer's committed KV: the `[0, committed)` byte range of the keys
/// and values planes as host bytes (see the module doc for why the
/// committed rows are a plane prefix).
pub(crate) struct CudaLayerPlanes {
    pub(crate) keys: Vec<u8>,
    pub(crate) values: Vec<u8>,
}

/// The device half of a stored prefix: every layer's whole KV planes (host
/// bytes), the health buffer when the plan carries one, and the committed
/// absolute position the planes end at. Geometry is validated against the
/// restoring state's plane lengths before any device write.
pub(crate) struct CudaPrefixSnapshot {
    pub(crate) layers: Vec<CudaLayerPlanes>,
    pub(crate) health: Option<Vec<u8>>,
    pub(crate) committed: usize,
}

impl CudaPrefixSnapshot {
    /// The host bytes this entry pins (the budget figure the report
    /// carries): plane bytes plus the health buffer.
    pub(crate) fn host_bytes(&self) -> u64 {
        let planes: usize = self
            .layers
            .iter()
            .map(|layer| layer.keys.len() + layer.values.len())
            .sum();
        u64::try_from(planes + self.health.as_deref().map_or(0, <[u8]>::len)).unwrap_or(u64::MAX)
    }

    /// Refuses when the snapshot does not fit the restoring state's
    /// committed-bounds plane geometry exactly (each layer's bytes at the
    /// snapshot's own committed row count). The path fingerprint already
    /// binds limit/chunk/profile, so a mismatch here means a different
    /// build or a corrupted entry: fail closed, never partial-restore.
    pub(crate) fn validate_against(
        &self,
        keys_len: &[usize],
        values_len: &[usize],
        health_len: Option<usize>,
    ) -> Result<()> {
        let geometry = self.layers.len() == keys_len.len()
            && keys_len.len() == values_len.len()
            && !self.layers.is_empty()
            && self.committed > 0
            && self.layers.iter().zip(keys_len.iter().zip(values_len)).all(
                |(layer, (keys, values))| {
                    layer.keys.len() == *keys && layer.values.len() == *values
                },
            )
            && self.health.as_deref().map_or(0, <[u8]>::len) == health_len.unwrap_or(0);
        if geometry {
            Ok(())
        } else {
            Err(EngError::BadRequest(
                "flint cuda prefix restore refused: plane geometry mismatch".into(),
            ))
        }
    }
}
