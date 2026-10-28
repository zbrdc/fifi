use super::{config::FULL_LAYERS, kv_plan::KvPlan, reference::ChunkWindow};
use crate::EngError;

#[test]
fn ring_plan_separates_layer_capacity_and_compact_snapshot_shape() {
    let plan = KvPlan::new(262_144, 512, true).unwrap();
    let mut full = 0;
    let mut sliding = 0;
    for layer in 0..49 {
        if FULL_LAYERS.contains(&(layer as u64)) {
            full += 1;
            assert_eq!(plan.cache_rows(layer).unwrap(), 262_144);
        } else {
            sliding += 1;
            assert_eq!(plan.cache_rows(layer).unwrap(), 1024);
        }
    }
    assert_eq!((full, sliding), (14, 35));
    assert_eq!(plan.bytes(2048).unwrap(), 7_589_593_088);
    let snapshot = plan.snapshots().unwrap();
    assert_eq!(snapshot.rows(), 513);
    assert_eq!(snapshot.keep(), 2);
    assert_eq!(snapshot.bytes(2048, 150_144 * 4).unwrap(), 74_744_832);
}

#[test]
fn flag_off_keeps_full_capacities_and_no_ring_snapshots() {
    let plan = KvPlan::new(8192, 512, false).unwrap();
    for layer in 0..49 {
        assert_eq!(plan.cache_rows(layer).unwrap(), 8192);
    }
    assert!(plan.snapshots().is_none());
    assert_eq!(plan.bytes(2048).unwrap(), 49 * 8192 * 2048);
}

#[test]
fn short_contexts_and_chunk_limits_clip_only_physical_capacity() {
    for context in [1, 512, 513, 514, 262_144] {
        for chunk in [1, 4, 512] {
            let plan = KvPlan::new(context, chunk, true).unwrap();
            for layer in 0..49 {
                let expected = if FULL_LAYERS.contains(&(layer as u64)) {
                    context
                } else {
                    context.min(512 + chunk)
                };
                assert_eq!(plan.cache_rows(layer).unwrap(), expected);
            }
            let snapshot = plan.snapshots().unwrap();
            assert_eq!(snapshot.rows(), context.min(513));
            assert_eq!(snapshot.keep(), 2);
        }
    }
}

#[test]
fn planned_capacity_survives_the_whole_chunk_append_union() {
    for chunk in [1, 4, 512] {
        let plan = KvPlan::new(262_144, chunk, true).unwrap();
        let capacity = plan.cache_rows(2).unwrap();
        for start in [0, 512, 513, 1023, 1024, 262_144 - chunk] {
            let window = ChunkWindow::new(513, start, chunk).unwrap();
            let mut seen = vec![false; capacity];
            for position in window.live_range() {
                let slot = window.slot(position, capacity).unwrap();
                assert!(!seen[slot], "live rows alias at {position}");
                seen[slot] = true;
            }
            if chunk > 1 && start >= 512 {
                assert!(window.slot(start, 513).is_err());
            }
        }
    }
}

#[test]
fn zero_and_overflowing_byte_charges_are_refused() {
    let plan = KvPlan::new(262_144, 512, true).unwrap();
    for row_bytes in [0, usize::MAX] {
        assert!(matches!(
            plan.bytes(row_bytes),
            Err(EngError::MemoryPlan(_))
        ));
        assert!(matches!(
            plan.snapshots().unwrap().bytes(row_bytes, 0),
            Err(EngError::MemoryPlan(_))
        ));
    }
    for logits_bytes in [usize::MAX / 2, usize::MAX] {
        assert!(matches!(
            plan.snapshots().unwrap().bytes(1, logits_bytes),
            Err(EngError::MemoryPlan(_))
        ));
    }
}

#[test]
fn zero_or_overflowing_extents_and_invalid_layers_are_refused() {
    for (context, chunk) in [(0, 1), (1, 0), (1, usize::MAX)] {
        assert!(matches!(
            KvPlan::new(context, chunk, true),
            Err(EngError::BadRequest(_))
        ));
    }
    let plan = KvPlan::new(8192, 1, true).unwrap();
    for layer in [49, usize::MAX] {
        assert!(matches!(
            plan.cache_rows(layer),
            Err(EngError::BadRequest(_))
        ));
    }
}

#[test]
fn contexts_outside_the_pinned_model_are_refused_before_allocation() {
    for ring_enabled in [false, true] {
        for context in [262_145, usize::MAX] {
            assert!(matches!(
                KvPlan::new(context, 1, ring_enabled),
                Err(EngError::ContextLimit(_))
            ));
        }
    }
}
