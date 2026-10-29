//! Unit tests of the pure section-timing algebra of the in-graph section timer: which layers
//! carry timing marks and how the marked sections extrapolate to a whole token. CPU only, no
//! device call.
use super::section_timing::{MarkedGroup, section_plan};
use crate::family::AttentionKind::{DeltaNet, Full, Qsa};

#[test]
fn second_layer_of_each_run_is_marked() {
    let plan = section_plan(&[Full, Full, Full, DeltaNet, DeltaNet]);
    assert_eq!(
        plan.groups,
        vec![
            MarkedGroup { layer: 1, size: 3 },
            MarkedGroup { layer: 4, size: 2 },
        ]
    );
}

#[test]
fn singleton_run_marks_its_only_layer() {
    let plan = section_plan(&[Full, DeltaNet, DeltaNet, Qsa]);
    assert_eq!(
        plan.groups,
        vec![
            MarkedGroup { layer: 0, size: 1 },
            MarkedGroup { layer: 2, size: 2 },
            MarkedGroup { layer: 3, size: 1 },
        ]
    );
}

#[test]
fn empty_and_uniform_kinds_are_plans() {
    assert!(section_plan(&[]).groups.is_empty());
    let plan = section_plan(&[DeltaNet; 28]);
    assert_eq!(plan.groups, vec![MarkedGroup { layer: 1, size: 28 }]);
}

#[test]
fn extrapolation_scales_groups_and_reconciles_with_whole() {
    let plan = section_plan(&[Full, Full, Full, DeltaNet, DeltaNet]);
    let times = plan
        .reconcile(0.5, 0.25, &[1.0, 2.0], 7.75)
        .expect("reconcile");
    assert_eq!(times.extrapolated_ms(), 7.75);
    assert_eq!(times.whole_ratio(), 1.0);
    assert_eq!(times.group_totals_ms(), vec![3.0, 4.0]);
}

#[test]
fn reconcile_rejects_non_finite_or_negative_sections() {
    let plan = section_plan(&[Full, Full]);
    assert!(plan.reconcile(-0.1, 0.2, &[1.0], 3.0).is_err());
    assert!(plan.reconcile(0.1, 0.2, &[f32::NAN], 3.0).is_err());
    assert!(plan.reconcile(0.1, f32::INFINITY, &[1.0], 3.0).is_err());
    assert!(plan.reconcile(0.1, 0.2, &[1.0], -3.0).is_err());
    // A zero whole is refused, not divided through.
    assert!(plan.reconcile(0.1, 0.2, &[1.0], 0.0).is_err());
    // A group count that does not match the plan is refused, not silently zipped.
    assert!(plan.reconcile(0.1, 0.2, &[1.0, 2.0], 3.0).is_err());
}

#[test]
fn reconcile_rejects_overflowing_derived_totals() {
    // Finite inputs can still overflow the derived totals (the scaled sum, or its ratio
    // against a tiny whole); those are refused, not read back as infinities.
    let plan = section_plan(&[Full, Full]);
    assert!(plan.reconcile(1e38, 1e38, &[1e38], 1e-38).is_err());
    // A finite sum alone is not enough: the ratio against a tiny whole overflows too.
    assert!(plan.reconcile(1.0, 1.0, &[1.0], 1e-38).is_err());
}

/// The marked pairs are ordered so a caller can hand them to `token_ops` once, in order.
#[test]
fn groups_are_in_layer_order() {
    let plan = section_plan(&[Qsa, Qsa, Full, Full, Full, Full, DeltaNet]);
    let layers: Vec<usize> = plan.groups.iter().map(|g| g.layer).collect();
    assert_eq!(layers, vec![1, 3, 6]);
}
