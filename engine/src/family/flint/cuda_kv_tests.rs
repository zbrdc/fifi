use super::{
    config::FULL_LAYERS,
    cuda_kv::LayerKvAllocation,
    kv_plan::{KvPlan, KvStorage},
};
use crate::EngError;

#[test]
fn append_commit_keeps_absolute_prefix_across_the_m1_ring_wrap() {
    let plan = KvPlan::new(8192, 1, true).unwrap();
    let allocation = LayerKvAllocation::new(&plan, 2).unwrap();
    let mut cursor = allocation.cursor();
    for position in 0usize..514 {
        assert_eq!(cursor.committed_prefix(), position);
        let append = cursor.begin_append(position, 1).unwrap();
        assert_eq!(
            append.live_range(),
            position.saturating_sub(512)..position + 1
        );
        assert_eq!(append.slot(position).unwrap(), position % 513);
        append.commit();
    }
    assert_eq!(cursor.committed_prefix(), 514);
}

#[test]
fn abandoned_append_refuses_another_append() {
    let plan = KvPlan::new(8192, 1, true).unwrap();
    let allocation = LayerKvAllocation::new(&plan, 2).unwrap();
    let mut cursor = allocation.cursor();
    {
        let _append = cursor.begin_append(0, 1).unwrap();
    }
    assert_eq!(cursor.committed_prefix(), 0);
    assert!(cursor.begin_append(0, 1).is_err());
}

#[test]
fn quiescent_reset_restarts_the_prefix_without_replacing_the_allocation() {
    let plan = KvPlan::new(8192, 1, true).unwrap();
    let allocation = LayerKvAllocation::new(&plan, 2).unwrap();
    let mut cursor = allocation.cursor();
    cursor.begin_append(0, 1).unwrap().commit();
    {
        let _append = cursor.begin_append(1, 1).unwrap();
    }
    assert_eq!(cursor.committed_prefix(), 1);
    cursor.reset_after_quiescence();
    assert_eq!(cursor.committed_prefix(), 0);
    cursor.begin_append(0, 1).unwrap().commit();
    assert_eq!(cursor.committed_prefix(), 1);
}

#[test]
fn allocator_charges_the_selected_capacity_per_plane_without_changing_defaults() {
    for (enabled, expected_bytes) in [(false, 26_306_674_688), (true, 7_589_593_088)] {
        let plan = KvPlan::new(262_144, 512, enabled).unwrap();
        let mut bytes = 0;
        for layer in 0..49 {
            let allocation = LayerKvAllocation::new(&plan, layer).unwrap();
            assert_eq!(allocation.cache_rows(), plan.cache_rows(layer).unwrap());
            assert_eq!(
                allocation.plane_bytes(),
                allocation.cache_rows() * 4 * 128 * 2
            );
            bytes += 2 * allocation.plane_bytes();
        }
        assert_eq!(bytes, expected_bytes);
        assert_eq!(bytes, plan.bytes(2048).unwrap());
    }
}

#[test]
fn allocator_clips_short_contexts_and_keeps_the_whole_chunk_union() {
    for context in [1, 512, 513, 514, 262_144] {
        for chunk in [1, 4, 512] {
            for enabled in [false, true] {
                let plan = KvPlan::new(context, chunk, enabled).unwrap();
                for layer in 0..49 {
                    let allocation = LayerKvAllocation::new(&plan, layer).unwrap();
                    let full = !enabled || FULL_LAYERS.contains(&(layer as u64));
                    let rows = if full {
                        context
                    } else {
                        context.min(512 + chunk)
                    };
                    assert_eq!(allocation.cache_rows(), rows);
                    assert_eq!(allocation.plane_bytes(), rows * 1024);
                }
            }
        }
    }
}

#[test]
fn internal_storage_charge_includes_both_native_scale_planes() {
    let default = KvPlan::new(4096, 8, true).unwrap();
    assert_eq!(default.storage(), KvStorage::F16);
    for (storage, row_bytes, plane_row) in [
        (KvStorage::F16, 2048, (1024, 0)),
        (KvStorage::Q8G32, 1088, (512, 32)),
    ] {
        let plan = default.with_storage(storage);
        assert_eq!(storage.plane_row_bytes(), plane_row);
        assert_eq!(storage.row_bytes(), row_bytes);
        assert_eq!(plan.snapshots(), default.snapshots());
        let bytes: usize = (0..49)
            .map(|layer| {
                let allocation = LayerKvAllocation::new(&plan, layer).unwrap();
                let cells = allocation.cache_rows();
                assert_eq!(
                    allocation.plane_components(),
                    (cells * plane_row.0, cells * plane_row.1)
                );
                allocation.plane_bytes() * 2
            })
            .sum();
        assert_eq!(bytes, plan.bytes(row_bytes).unwrap());
    }
}

#[test]
fn append_runs_use_physical_capacity_not_encoded_plane_bytes() {
    for (storage, layer) in [
        (KvStorage::F16, 0),
        (KvStorage::F16, 2),
        (KvStorage::Q8G32, 0),
        (KvStorage::Q8G32, 2),
    ] {
        for chunk in 1..=8 {
            let plan = KvPlan::new(2048, chunk, true)
                .unwrap()
                .with_storage(storage);
            for position in [0, 511, 512, 513, 519, 520, 521, 1039, 1040, 1041] {
                assert_append_runs(&plan, layer, position, chunk);
            }
        }
    }
}

#[test]
fn append_runs_clip_short_linear_contexts_and_final_partial_chunks() {
    for (storage, layer) in [
        (KvStorage::F16, 0),
        (KvStorage::F16, 2),
        (KvStorage::Q8G32, 0),
        (KvStorage::Q8G32, 2),
    ] {
        for (context, position, enabled) in [
            (1, 0, true),
            (4, 0, false),
            (512, 508, true),
            (513, 512, true),
            (514, 512, false),
            (520, 516, true),
            (521, 519, false),
            (1043, 1039, true),
            (1043, 1039, false),
        ] {
            let plan = KvPlan::new(context, 8, enabled)
                .unwrap()
                .with_storage(storage);
            assert_append_runs(&plan, layer, position, context - position);
        }
    }
}

fn assert_append_runs(plan: &KvPlan, layer: usize, position: usize, rows: usize) {
    let allocation = LayerKvAllocation::new(plan, layer).unwrap();
    let capacity = allocation.cache_rows();
    let mut cursor = allocation.cursor();
    while cursor.committed_prefix() < position {
        let start = cursor.committed_prefix();
        cursor
            .begin_append(start, rows.min(position - start))
            .unwrap()
            .commit();
    }
    let append = cursor.begin_append(position, rows).unwrap();
    assert_eq!(append.cache_rows(), capacity);
    assert_eq!(append.storage(), plan.storage());
    let first = rows.min(capacity - position % capacity);
    let runs = append.runs().unwrap();
    assert_eq!(runs, [(position % capacity, 0..first), (0, first..rows)]);
    for (cell, source) in runs {
        assert!(cell + source.len() <= capacity);
        for (offset, row) in source.enumerate() {
            assert_eq!(cell + offset, append.slot(position + row).unwrap());
        }
    }
    append.commit();
    assert_eq!(cursor.committed_prefix(), position + rows);
}

#[test]
fn allocator_rejects_an_out_of_range_layer_before_native_allocation() {
    let plan = KvPlan::new(8192, 512, true).unwrap();
    for layer in [49, usize::MAX] {
        assert!(matches!(
            LayerKvAllocation::new(&plan, layer),
            Err(EngError::BadRequest(_))
        ));
    }
}
