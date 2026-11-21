//! Host-side planning for MLX resident prefill. Each family supplies the workspace cost
//! for its own tensors. The chunk ladder and DeltaNet retire-memory rule are shared.

use crate::EngError;

pub(crate) const MIN_PREFILL_CHUNK_ROWS: u32 = 4;
pub(crate) const PREFILL_CANDIDATES: [u32; 11] = [
    2048,
    1024,
    512,
    256,
    128,
    64,
    32,
    16,
    8,
    MIN_PREFILL_CHUNK_ROWS,
    1,
];

/// A chunk temporarily sets the MLX retirement cap to 512 MiB. Leave 64 MiB
/// available for other temporaries when sizing the DeltaNet recurrence.
pub(crate) const PREFILL_RETIRE_CAP: u64 = 512 << 20;
pub(crate) const RETIRE_CAP_BASE: u64 = 64 << 20;

/// Group only adjacent, exactly routed expert slices for multi-row stages.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn prefill_expert_read_runs() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("FIFI_MLX_EXPERT_READ_RUNS").is_some_and(|value| value == "prefill")
    })
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn overlap_expert_reads() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("FIFI_MLX_OVERLAP_EXPERT_READS").is_some_and(|value| value == "1")
    })
}

/// Queue one independent GPU result while the worker reads the exact expert IDs.
/// Its owner passes the returned tensor into the dependent output graph.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn submit_shared_expert(
    kernels: &mut crate::kernels::MlxKernels,
    output: crate::tensor::ExecutionTensor,
) -> Result<crate::tensor::ExecutionTensor, EngError> {
    kernels.evaluate_async(&[&output])?;
    Ok(output)
}

/// Sort and deduplicate checkpoint expert IDs for shared page staging.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn sorted_expert_ids(selected: &[u32]) -> Vec<u32> {
    let mut experts = selected.to_vec();
    experts.sort_unstable();
    experts.dedup();
    experts
}

/// Index the sorted compact stack while keeping the original pair order.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn local_expert_ids(selected: &[u32], experts: &[u32]) -> Result<Vec<u32>, EngError> {
    selected
        .iter()
        .map(|id| {
            experts
                .binary_search(id)
                .ok()
                .and_then(|slot| u32::try_from(slot).ok())
                .ok_or_else(|| {
                    EngError::BadRequest(format!("expert {id} is outside the staged stack"))
                })
        })
        .collect()
}

/// F32 bytes for the replaced state, decayed state and outer product per token.
pub(crate) fn delta_retire_per_token(heads: u32, key_dim: u32, value_dim: u32) -> Option<u64> {
    u64::from(heads)
        .checked_mul(u64::from(key_dim))?
        .checked_mul(u64::from(value_dim))?
        .checked_mul(3 * 4)
        .filter(|&bytes| bytes > 0)
}

pub(crate) fn delta_subchunk(rows: u32, cap: u64, per_token: u64) -> Option<u32> {
    if rows == 0 || per_token == 0 {
        return None;
    }
    let fit = cap.saturating_sub(RETIRE_CAP_BASE) / per_token;
    Some(u32::try_from(fit).unwrap_or(u32::MAX).clamp(1, rows))
}

/// Find the largest fitting candidate. Both workspace models are monotone.
/// On the descending serving ladder, only price rows through the first fit.
pub(in crate::family) fn largest_fitting_chunk(
    candidates: impl IntoIterator<Item = u32>,
    remaining: u64,
    workspace: impl Fn(u32) -> Result<u64, EngError>,
) -> Result<Option<u32>, EngError> {
    let mut best = None;
    for rows in candidates {
        if rows == 0 {
            return Err(EngError::BadRequest(
                "prefill chunk size must exceed zero".into(),
            ));
        }
        if best.is_none_or(|previous| rows > previous) && workspace(rows)? <= remaining {
            best = Some(rows);
        }
    }
    Ok(best)
}

/// A one-row fallback needs no chunk allocation. Limit the search to sizes
/// the worker can actually submit before asking the family for a cost.
pub(crate) fn admitted_chunk(
    max_rows: u32,
    remaining: u64,
    workspace: impl Fn(u32) -> Result<u64, EngError>,
) -> Result<Option<u32>, EngError> {
    let sizes = PREFILL_CANDIDATES
        .into_iter()
        .filter(|&rows| rows > 1 && rows <= max_rows);
    largest_fitting_chunk(sizes, remaining, workspace)
}
