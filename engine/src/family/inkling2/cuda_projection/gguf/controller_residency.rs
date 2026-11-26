//! Supplement the original placement with complete F32 controller triples.

use super::{
    CudaContext, DEVICE_HEADROOM, DenseDtype, EngError, GgufDenseUpload, Projections,
    device_prefill, projection_bytes, upload_projections,
};
use crate::family::inkling2::switches::Switch;

use crate::family::mem::DEVICE;

const MAX_CHARGE: u64 = 64 << 20;
/// The resident prefill pass launches every controller of every visit from one plan, so it needs
/// the complete triples of all visits on the device; with the pass flag the charge may cover them.
const PASS_CHARGE: u64 = 640 << 20;

pub(super) fn enabled() -> bool {
    Switch::ResidentControllers.on()
}

pub(super) fn admit(
    context: &CudaContext,
    optional: &[GgufDenseUpload<'_>],
    dense: &mut Projections,
    [original_floor, expert_workspace]: [u64; 2],
) -> Result<(), EngError> {
    let before = context.memory_info()?.free as u64;
    let Some(floor) = floor(before, original_floor, expert_workspace, charge()) else {
        return Ok(());
    };
    let mut admitted = Vec::new();
    for group in optional.chunk_by(|a, b| {
        super::projection_group(a.matrix.name) == super::projection_group(b.matrix.name)
    }) {
        if !is_missing_controller(group, dense) {
            continue;
        }
        let available = (context.memory_info()?.free as u64).saturating_sub(floor);
        if projection_bytes(group) > available {
            continue;
        }
        let uploaded = match upload_projections(context, group) {
            Ok(uploaded) => uploaded,
            Err(EngError::MemoryPlan(_)) => break,
            Err(error) => return Err(error),
        };
        if (context.memory_info()?.free as u64) >= floor {
            admitted.extend(uploaded.keys().cloned());
            dense.extend(uploaded);
        }
    }
    if !admitted.is_empty() {
        let after = context.memory_info()?.free as u64;
        eprintln!(
            "Inkling CUDA extra resident controllers {}",
            serde_json::json!({
                "matrices": admitted, "device_bytes": before.saturating_sub(after),
                "free_after_bytes": after, "floor_bytes": floor,
                "max_expert_workspace_bytes": expert_workspace,
            })
        );
    }
    Ok(())
}

fn is_missing_controller(group: &[GgufDenseUpload<'_>], dense: &Projections) -> bool {
    group.len() == 3
        && group
            .iter()
            .zip(["down_weight", "forget_weight", "gain_weight"])
            .all(|(item, expected)| {
                item.dtype == DenseDtype::F32
                    && item.matrix.name.contains(".controllers.")
                    && item
                        .matrix
                        .name
                        .rsplit_once('.')
                        .is_some_and(|(_, suffix)| suffix == expected)
                    && !dense.contains_key(item.matrix.name)
            })
}

pub(super) fn workspace_floor(expert_workspace: u64) -> Option<u64> {
    expert_workspace
        .checked_mul(2)?
        .checked_add(DEVICE_HEADROOM + DEVICE.inkling_controller_margin)
}

fn charge() -> u64 {
    if crate::family::inkling2::switches::Switch::ResidentPrefillPass.on() {
        PASS_CHARGE
    } else {
        MAX_CHARGE
    }
}

fn floor(before: u64, original_floor: u64, expert_workspace: u64, charge: u64) -> Option<u64> {
    Some(
        workspace_floor(expert_workspace)?
            .max(original_floor.saturating_sub(charge))
            .max(before.saturating_sub(charge)),
    )
}

#[cfg(test)]
fn minimum_free(before: u64, original_floor: u64, expert_workspace: u64) -> Option<u64> {
    floor(before, original_floor, expert_workspace, MAX_CHARGE)
}

pub(super) fn expert_workspace(
    spec: &super::super::TextSpec,
    context_limit: usize,
    use_device: bool,
) -> Option<u64> {
    if use_device {
        device_prefill::device_workspace_bytes(spec, context_limit)
    } else {
        Some(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_controller_floor_preserves_growth_and_caps_physical_charge() {
        let mib = 1 << 20;
        assert_eq!(
            minimum_free(2800 * mib, 2799 * mib, 212 * mib),
            Some(2736 * mib)
        );
        assert_eq!(
            minimum_free(2900 * mib, 2910 * mib, 200 * mib),
            Some(2846 * mib)
        );
        assert_eq!(
            minimum_free(1800 * mib, 1799 * mib, 800 * mib),
            Some(1888 * mib)
        );
        assert_eq!(minimum_free(1000 * mib, 1000 * mib, u64::MAX), None);
    }
}
