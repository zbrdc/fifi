use super::*;
use crate::{
    kernels::{BinaryOp, MlxKernels},
    tensor::{Dtype, ExecutionTensor, Tensor},
};

fn row_value(head: u32, position: u32, column: u32) -> f32 {
    let byte = (position >> (8 * (column % 2))) & 255;
    let tag = byte ^ ((head * 17 + column / 2) & 255);
    (tag as f32 - 128.0) / 16.0
}

fn input(k: &mut MlxKernels, start: u32, rows: u32, value: bool) -> ExecutionTensor {
    let mut data = Vec::new();
    for head in 0..4 {
        for pos in start..start + rows {
            for column in 0..128 {
                let v = row_value(head, pos, column);
                data.extend((if value { -v - 0.25 } else { v }).to_le_bytes());
            }
        }
    }
    let host = Tensor::try_from_host(Dtype::F32, vec![4, rows, 128], data).unwrap();
    let raw = k.import_tensor(&host).unwrap();
    k.cast(&raw, Dtype::Bf16).unwrap()
}

fn check(k: &mut MlxKernels, x: &ExecutionTensor, start: u32, rows: u32, value: bool) {
    assert_eq!(x.shape(), [4, rows, 128]);
    let x = k.cast(x, Dtype::F32).unwrap();
    let host = k.readback(&x).unwrap();
    let bytes = host.host_bytes().unwrap();
    for (i, word) in bytes.as_chunks::<4>().0.iter().enumerate() {
        let head = i as u32 / (rows * 128);
        let position = start + i as u32 / 128 % rows;
        let column = i as u32 % 128;
        let v = row_value(head, position, column);
        assert_eq!(f32::from_le_bytes(*word), if value { -v - 0.25 } else { v });
    }
}

fn consumer(
    k: &mut MlxKernels,
    keys: &ExecutionTensor,
    values: &ExecutionTensor,
) -> ExecutionTensor {
    let twice = k.binary(BinaryOp::Add, keys, keys).unwrap();
    k.binary(BinaryOp::Add, &twice, values).unwrap()
}

fn put(k: &mut MlxKernels, held: &mut Option<Plane>, plan: &ChunkCachePlan) {
    let keys = input(k, plan.p0, plan.rows, false);
    let values = input(k, plan.p0, plan.rows, true);
    let (keys, values, guard) = Plane::prepare_chunk(k, held, plan, &keys, &values, false).unwrap();
    let consumed = consumer(k, &keys, &values);
    Plane::commit_chunk(k, held, guard, &consumed).unwrap();
}

#[test]
fn batch_chunk_cache_preserves_union_and_final_ring_without_row_writes() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    k.set_cache_limit(0).unwrap();
    for (layer, start, rows) in [
        (2, 0, 2),
        (2, 511, 7),
        (2, 512, 17),
        (2, 513, 512),
        (2, 514, 1024),
        (2, 1026, 17),
        (0, 0, 7),
        (0, 511, 17),
        (0, 1026, 512),
    ] {
        let mut held = None;
        let mut pos = 0;
        while pos < start {
            let n = (start - pos).min(512);
            put(
                &mut k,
                &mut held,
                &ChunkCachePlan::new(layer, pos, n, 8192).unwrap(),
            );
            pos += n;
        }
        let plan = ChunkCachePlan::new(layer, start, rows, 8192).unwrap();
        let origin = if layer == 2 {
            start.saturating_sub(512)
        } else {
            0
        };
        let keys = input(&mut k, start, rows, false);
        let values = input(&mut k, start, rows, true);
        #[cfg(feature = "native-fault-qualification")]
        let before = k.selected_operation_count();
        let (query_k, query_v, guard) =
            Plane::prepare_chunk(&mut k, &mut held, &plan, &keys, &values, false).unwrap();
        assert!(held.is_none());
        // Only the query consumer may force the union before the bulk writes.
        let consumed = consumer(&mut k, &query_k, &query_v);
        Plane::commit_chunk(&mut k, &mut held, guard, &consumed).unwrap();
        #[cfg(feature = "native-fault-qualification")]
        assert!(
            k.selected_operation_count() - before < 64,
            "cache writes must not scale with T"
        );
        assert!(held.is_some());
        check(&mut k, &query_k, origin, start + rows - origin, false);
        check(&mut k, &query_v, origin, start + rows - origin, true);
        // A subsequent block exposes the committed tail, without altering old published views.
        let next = start + rows;
        let next_plan = ChunkCachePlan::new(layer, next, 2, 8192).unwrap();
        let next_k = input(&mut k, next, 2, false);
        let next_v = input(&mut k, next, 2, true);
        let (tail_k, tail_v, guard) =
            Plane::prepare_chunk(&mut k, &mut held, &next_plan, &next_k, &next_v, false).unwrap();
        let next_origin = if layer == 2 {
            next.saturating_sub(512)
        } else {
            0
        };
        let consumed = consumer(&mut k, &tail_k, &tail_v);
        Plane::commit_chunk(&mut k, &mut held, guard, &consumed).unwrap();
        check(&mut k, &tail_k, next_origin, next + 2 - next_origin, false);
        check(&mut k, &tail_v, next_origin, next + 2 - next_origin, true);
        check(&mut k, &query_k, origin, start + rows - origin, false);
        check(&mut k, &query_v, origin, start + rows - origin, true);
    }
}

/// Growing a full plane must preserve every committed row and keep chunking
/// across the old capacity identical to a plane that started big. The grown
/// plane and the direct plane must read back equal.
#[test]
fn batch_chunk_cache_grow_preserves_rows_across_the_old_capacity() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    k.set_cache_limit(0).unwrap();
    for (start_cap, grow_to, cross) in [(8192_u32, 16384_u32, 8188_u32), (4096, 8192, 4088)] {
        let mut held = None;
        let mut pos = 0_u32;
        while pos < cross {
            let n = (cross - pos).min(512);
            put(
                &mut k,
                &mut held,
                &ChunkCachePlan::new(0, pos, n, start_cap).unwrap(),
            );
            pos += n;
        }
        let grown = held.take().unwrap().grow(&mut k, grow_to).unwrap();
        held = Some(grown);
        // One chunk straddles the old capacity boundary.
        let plan = ChunkCachePlan::new(0, cross, 12, grow_to).unwrap();
        let keys = input(&mut k, cross, 12, false);
        let values = input(&mut k, cross, 12, true);
        let (query_k, query_v, guard) =
            Plane::prepare_chunk(&mut k, &mut held, &plan, &keys, &values, false).unwrap();
        let consumed = consumer(&mut k, &query_k, &query_v);
        Plane::commit_chunk(&mut k, &mut held, guard, &consumed).unwrap();
        let end = cross + 12;
        check(&mut k, &query_k, 0, end, false);
        check(&mut k, &query_v, 0, end, true);

        // Direct control: same rows in a plane that never grew.
        let mut direct = None;
        let mut pos = 0_u32;
        while pos < end {
            let n = (end - pos).min(512);
            put(
                &mut k,
                &mut direct,
                &ChunkCachePlan::new(0, pos, n, grow_to).unwrap(),
            );
            pos += n;
        }
        let (gk, gv) = held.as_ref().unwrap().snapshot(&mut k).unwrap();
        let (dk, dv) = direct.as_ref().unwrap().snapshot(&mut k).unwrap();
        assert_eq!(gk.shape(), dk.shape(), "capacity {start_cap}");
        assert_eq!(
            gk.host_bytes().unwrap(),
            dk.host_bytes().unwrap(),
            "grown key plane rows differ from the direct plane (capacity {start_cap})"
        );
        assert_eq!(
            gv.host_bytes().unwrap(),
            dv.host_bytes().unwrap(),
            "grown value plane rows differ from the direct plane (capacity {start_cap})"
        );
    }
}

/// The cache must cross the legacy 8192 boundary cleanly at the 32768
/// default - full layers keep origin 0, ring layers clamp to the 512-row
/// window, on both sides of 8192.
#[test]
fn batch_chunk_cache_crosses_legacy_8192_at_32k() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    k.set_cache_limit(0).unwrap();
    for (layer, start, rows) in [
        (2, 8188, 8),
        (2, 8192, 512),
        (2, 8704, 512),
        (0, 8188, 8),
        (0, 8191, 1024),
        (0, 16384, 512),
    ] {
        let mut held = None;
        let mut pos = 0;
        while pos < start {
            let n = (start - pos).min(512);
            put(
                &mut k,
                &mut held,
                &ChunkCachePlan::new(layer, pos, n, 32768).unwrap(),
            );
            pos += n;
        }
        let plan = ChunkCachePlan::new(layer, start, rows, 32768).unwrap();
        let origin = if layer == 2 {
            start.saturating_sub(512)
        } else {
            0
        };
        let keys = input(&mut k, start, rows, false);
        let values = input(&mut k, start, rows, true);
        let (query_k, query_v, guard) =
            Plane::prepare_chunk(&mut k, &mut held, &plan, &keys, &values, false).unwrap();
        let consumed = consumer(&mut k, &query_k, &query_v);
        Plane::commit_chunk(&mut k, &mut held, guard, &consumed).unwrap();
        check(&mut k, &query_k, origin, start + rows - origin, false);
        check(&mut k, &query_v, origin, start + rows - origin, true);
    }
}

#[test]
fn batch_chunk_cache_rejects_bad_layout_without_releasing_history() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let mut held = None;
    put(
        &mut k,
        &mut held,
        &ChunkCachePlan::new(2, 0, 2, 8192).unwrap(),
    );
    let plan = ChunkCachePlan::new(2, 2, 7, 8192).unwrap();
    let keys = input(&mut k, 2, 7, false);
    let values = input(&mut k, 2, 7, true);
    let wrong = k.reshape(&keys, &[7, 4, 128]).unwrap();
    assert!(matches!(
        Plane::prepare_chunk(&mut k, &mut held, &plan, &wrong, &values, false),
        Err(crate::EngError::BadRequest(_))
    ));
    assert!(held.is_some());
    let mut missing = None;
    assert!(matches!(
        Plane::prepare_chunk(&mut k, &mut missing, &plan, &keys, &values, false),
        Err(crate::EngError::BadRequest(_))
    ));
    put(&mut k, &mut held, &plan);
}

fn assert_same_cache(k: &mut MlxKernels, held: &[Option<Plane>; 2], layer: usize, end: u32) {
    let snapshots = held
        .each_ref()
        .map(|plane| plane.as_ref().unwrap().snapshot(k).unwrap());
    let capacity = CacheStep::new(layer, end - 1).unwrap().capacity;
    let mut positions = vec![None; capacity as usize];
    for position in 0..end {
        positions[(position % capacity) as usize] = Some(position);
    }
    for (name, a, b) in [
        ("key", &snapshots[0].0, &snapshots[1].0),
        ("value", &snapshots[0].1, &snapshots[1].1),
    ] {
        assert_eq!(a.dtype(), Dtype::Bf16);
        assert_eq!(a.dtype(), b.dtype());
        assert_eq!(a.shape(), [4, capacity, 128]);
        assert_eq!(a.shape(), b.shape());
        let left = a.host_bytes().unwrap();
        let right = b.host_bytes().unwrap();
        assert_eq!(left.len(), right.len());
        let mismatch = left.iter().zip(right).position(|(a, b)| a != b);
        if let Some(byte) = mismatch {
            let cell = byte / 2;
            let slot = cell / 128 % capacity as usize;
            panic!(
                "layer={layer} {name} head={} slot={slot} position={:?} column={} batch={:?} T1={:?}",
                cell / (capacity as usize * 128),
                positions[slot],
                cell % 128,
                &left[cell * 2..cell * 2 + 2],
                &right[cell * 2..cell * 2 + 2]
            );
        }
    }
}

fn decode_fixture_logits(
    k: &mut MlxKernels,
    held: &mut Option<Plane>,
    plan: &ChunkCachePlan,
) -> [f32; 2] {
    use crate::kernels::mlx::{KvBlock, SharedAttention};
    let keys = input(k, plan.p0, 1, false);
    let values = input(k, plan.p0, 1, true);
    let query = k
        .concatenate(&[&keys, &keys, &keys, &keys, &keys, &keys, &keys, &keys], 0)
        .unwrap();
    let query = k.reshape(&query, &[1, 32, 1, 128]).unwrap();
    let sink_host = Tensor::try_from_host(
        Dtype::F32,
        vec![1, 32],
        (0..32)
            .flat_map(|h| (h as f32 / 16.0).to_le_bytes())
            .collect(),
    )
    .unwrap();
    let sinks = k.import_tensor(&sink_host).unwrap();
    let (keys, values, guard) = Plane::prepare_chunk(k, held, plan, &keys, &values, false).unwrap();
    let slots = keys.shape()[1];
    let keys = k.reshape(&keys, &[1, 4, slots, 128]).unwrap();
    let values = k.reshape(&values, &[1, 4, slots, 128]).unwrap();
    let window = plan.step.rope.map(|_| WINDOW);
    let output = k
        .shared_attention(SharedAttention {
            q: &query,
            kv: KvBlock::Float {
                keys: &keys,
                values: &values,
            },
            query_start: plan.p0,
            key_origin: if window.is_some() {
                plan.p0.saturating_sub(WINDOW - 1)
            } else {
                0
            },
            window,
            sinks: Some(&sinks),
            scale: 1.0 / 128.0_f32.sqrt(),
        })
        .unwrap();
    Plane::commit_chunk(k, held, guard, &output).unwrap();
    let output = k.cast(&output, Dtype::F32).unwrap();
    let host = k.readback(&output).unwrap();
    // A fixed two-column fixture head; these are not full-model logits.
    let mut logits = [0.0_f32; 2];
    for (i, word) in host
        .host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
    {
        let value = f32::from_le_bytes(*word);
        logits[0] += value * (i % 7) as f32;
        logits[1] += value * (i % 11) as f32;
    }
    assert!(logits.iter().all(|x| x.is_finite()));
    logits
}

#[test]
#[ignore = "bounded native cache diagnosis; no model weights"]
fn batch_cache_matches_t1_bytes_and_eight_decode_fixture_steps_on_every_layer() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    k.set_cache_limit(0).unwrap();
    for layer in 0..49 {
        let mut held = [None, None];
        for (path, plane) in held.iter_mut().enumerate() {
            let chunk = if path == 0 { 512 } else { 1 };
            let mut position = 0;
            while position < 2048 {
                let rows = chunk.min(2048 - position);
                put(
                    &mut k,
                    plane,
                    &ChunkCachePlan::new(layer, position, rows, 8192).unwrap(),
                );
                position += rows;
            }
        }
        assert_same_cache(&mut k, &held, layer, 2048);
        for position in 2048..2056 {
            let plan = ChunkCachePlan::new(layer, position, 1, 8192).unwrap();
            let batch = decode_fixture_logits(&mut k, &mut held[0], &plan);
            let scalar = decode_fixture_logits(&mut k, &mut held[1], &plan);
            assert_eq!(batch, scalar, "layer={layer} decode position={position}");
        }
    }
}

/// Async chain: the chained commit must EXECUTE (the dependency edge must pass the
/// adapter validators — a failure class that host-side checks cannot catch) and
/// leave the plane bytes bitwise identical to the fenced commit when the attention
/// consumer is read back after the group-root evaluate. Four planes stand in for a
/// 4-layer group; three one-row steps per plane cross the group boundary exactly
/// like the forward loop. No model weights; the counter/parity/speed contract stays
/// with d2_async_chain_counter_target.
#[test]
#[ignore = "bounded native cache diagnosis; no model weights"]
fn chained_commit_executes_and_matches_fenced_bytes() {
    let _serial = crate::native_test_lock();
    let run = |chained: bool| -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let mut k = MlxKernels::try_new().unwrap();
        k.allow_bf16(true);
        k.set_cache_limit(0).unwrap();
        let mut held = [None, None, None, None];
        let mut heads = Vec::new();
        for step in 0..3_u32 {
            let mut roots: Vec<ExecutionTensor> = Vec::new();
            for (plane, held_plane) in held.iter_mut().enumerate() {
                let position = step;
                let plan = ChunkCachePlan::new(plane, position, 1, 8192).unwrap();
                let keys = input(&mut k, position, 1, false);
                let values = input(&mut k, position, 1, true);
                let (union_keys, union_values, guard) =
                    Plane::prepare_chunk(&mut k, held_plane, &plan, &keys, &values, chained)
                        .unwrap();
                // A lazy consumer of the union views, like shared_attention:
                // if the in-place append ran before the union materialized,
                // the consumer's bytes differ — that is the ordering the two
                // fences used to guarantee.
                // In an armed process the routed full planes hand out the
                // WHOLE plane; slice the consumer's view to the visible rows
                // so both arms compare the same bytes (the commits below
                // keep the original views).
                let visible = plan.p0 + plan.rows;
                let (consumer_keys, consumer_values) = if union_keys.shape()[1] > visible {
                    (
                        k.slice(&union_keys, &[0, 0, 0], &[4, visible, 128])
                            .unwrap(),
                        k.slice(&union_values, &[0, 0, 0], &[4, visible, 128])
                            .unwrap(),
                    )
                } else {
                    (union_keys.share(), union_values.share())
                };
                let output = consumer(&mut k, &consumer_keys, &consumer_values);
                if chained {
                    Plane::commit_chunk_chained(
                        &mut k,
                        held_plane,
                        guard,
                        &union_keys,
                        &union_values,
                    )
                    .unwrap();
                } else {
                    Plane::commit_chunk(&mut k, held_plane, guard, &output).unwrap();
                }
                if chained {
                    // share() the roots so no immutable borrow of `held` spans
                    // the next plane's mutable prepare.
                    roots.extend(
                        held_plane
                            .as_ref()
                            .unwrap()
                            .append_roots()
                            .map(|t| t.share()),
                    );
                }
                heads.push(output);
            }
            if chained {
                // The group root: the last consumer plus every plane the group
                // appended into, exactly like the forward loop's boundary.
                roots.push(heads.last().unwrap().share());
                let refs: Vec<&ExecutionTensor> = roots.iter().collect();
                k.evaluate(&refs).unwrap();
            }
        }
        let planes = held
            .each_ref()
            .map(|plane| plane.as_ref().unwrap().snapshot(&mut k).unwrap())
            .iter()
            .flat_map(|(keys, values)| [keys, values])
            .map(|t| t.host_bytes().unwrap().to_vec())
            .collect();
        let outputs = heads
            .iter()
            .map(|head| {
                let wide = k.cast(head, Dtype::F32).unwrap();
                let host = k.readback(&wide).unwrap();
                host.host_bytes().unwrap().to_vec()
            })
            .collect();
        (planes, outputs)
    };
    let (fenced_planes, fenced_outputs) = run(false);
    let (chained_planes, chained_outputs) = run(true);
    for (name, left, right) in [
        ("plane", &fenced_planes, &chained_planes),
        ("consumer", &fenced_outputs, &chained_outputs),
    ] {
        assert_eq!(left.len(), right.len(), "{name} tensor count");
        for (index, (left, right)) in left.iter().zip(right).enumerate() {
            assert_eq!(left.len(), right.len(), "{name} {index} byte length");
            assert_eq!(
                left, right,
                "{name} tensor {index} bytes must be bitwise identical"
            );
        }
    }
}

#[test]
fn chained_commit_matches_fenced_bytes_across_the_swa_wrap() {
    let _serial = crate::native_test_lock();
    // Layers 2/3 are SWA layers: their ring capacity is WINDOW (513). An
    // unchained bulk prefill fills the ring to 510, then seven one-row
    // steps cross the first ring overwrite (513/514) under both commits.
    // Windows 2-3 proved flag-independence on the model; this pins the
    // same property at the plane level with no weights and no window.
    let run = |chained: bool| -> Vec<Vec<u8>> {
        let mut k = MlxKernels::try_new().unwrap();
        k.allow_bf16(true);
        k.set_cache_limit(0).unwrap();
        let mut held = [None, None, None, None];
        // Pin the ring capacity: the wrap claim only holds while the SWA
        // layers keep the 513-slot ring a topology change must trip this.
        assert_eq!(
            ChunkCachePlan::new(2, 0, 1, 8192).unwrap().step.capacity,
            WINDOW
        );
        // Unchained bulk prefill on every plane: 256 + 254 rows to 510.
        for (p0, rows) in [(0_u32, 256_u32), (256, 254)] {
            for (plane, held_plane) in held.iter_mut().enumerate() {
                let plan = ChunkCachePlan::new(plane, p0, rows, 8192).unwrap();
                put(&mut k, held_plane, &plan);
            }
        }
        let mut heads = Vec::new();
        for step in 510_u32..517 {
            let mut roots: Vec<ExecutionTensor> = Vec::new();
            for (plane, held_plane) in held.iter_mut().enumerate() {
                let plan = ChunkCachePlan::new(plane, step, 1, 8192).unwrap();
                let keys = input(&mut k, step, 1, false);
                let values = input(&mut k, step, 1, true);
                let (union_keys, union_values, guard) =
                    Plane::prepare_chunk(&mut k, held_plane, &plan, &keys, &values, chained)
                        .unwrap();
                // Same armed-process guard as the fenced-bytes test: the
                // consumer reads only the visible rows of the union.
                // In an armed process the routed full planes hand out the
                // WHOLE plane; slice the consumer's view to the visible rows
                // so both arms compare the same bytes (the commits below
                // keep the original views).
                let visible = plan.p0 + plan.rows;
                let (consumer_keys, consumer_values) = if union_keys.shape()[1] > visible {
                    (
                        k.slice(&union_keys, &[0, 0, 0], &[4, visible, 128])
                            .unwrap(),
                        k.slice(&union_values, &[0, 0, 0], &[4, visible, 128])
                            .unwrap(),
                    )
                } else {
                    (union_keys.share(), union_values.share())
                };
                let output = consumer(&mut k, &consumer_keys, &consumer_values);
                if chained {
                    Plane::commit_chunk_chained(
                        &mut k,
                        held_plane,
                        guard,
                        &union_keys,
                        &union_values,
                    )
                    .unwrap();
                    roots.extend(
                        held_plane
                            .as_ref()
                            .unwrap()
                            .append_roots()
                            .map(|t| t.share()),
                    );
                } else {
                    Plane::commit_chunk(&mut k, held_plane, guard, &output).unwrap();
                }
                heads.push(output);
            }
            if chained {
                roots.push(heads.last().unwrap().share());
                let refs: Vec<&ExecutionTensor> = roots.iter().collect();
                k.evaluate(&refs).unwrap();
            }
        }
        let mut bytes = Vec::new();
        for plane in held.each_ref() {
            let (keys, values) = plane.as_ref().unwrap().snapshot(&mut k).unwrap();
            for tensor in [keys, values] {
                bytes.push(tensor.host_bytes().unwrap().to_vec());
            }
        }
        for head in &heads {
            let wide = k.cast(head, Dtype::F32).unwrap();
            let host = k.readback(&wide).unwrap();
            bytes.push(host.host_bytes().unwrap().to_vec());
        }
        bytes
    };
    let fenced = run(false);
    let chained = run(true);
    assert_eq!(fenced.len(), chained.len());
    for (index, (left, right)) in fenced.iter().zip(&chained).enumerate() {
        assert_eq!(left.len(), right.len(), "tensor {index} byte length");
        assert_eq!(
            left, right,
            "tensor {index} bytes must be bitwise identical across the SWA wrap"
        );
    }
}

#[path = "mlx_decode_attn_tests.rs"]
mod decode_attn;
#[path = "mlx_decode_order_tests.rs"]
mod decode_order;

/// The ring-view guard must never silently survive the FENCED commit: the
/// route arms only for the chained commit, so if a prepended guard ever
/// reaches `commit_chunk` it must fail closed with the specific error and
/// leave the slot vacant (no second append). Runs meaningfully in an armed
/// process; in an unarmed process the guard is not prepended and the fenced
/// commit must still succeed, so both flag states are pinned.
#[test]
fn prepended_guard_fails_closed_at_the_fenced_commit() {
    let armed = crate::family::flint::mlx_forward::ring_view_flagged();
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    k.set_cache_limit(0).unwrap();
    let mut held: Option<Plane> = None;
    // A 511-row bulk prefill first, so the next one-row chunk is a decode.
    let plan0 = ChunkCachePlan::new(0, 0, 511, 8192).unwrap();
    put(&mut k, &mut held, &plan0);
    let plan = ChunkCachePlan::new(0, 511, 1, 8192).unwrap();
    let keys = input(&mut k, 511, 1, false);
    let values = input(&mut k, 511, 1, true);
    let (union_keys, union_values, guard) =
        Plane::prepare_chunk(&mut k, &mut held, &plan, &keys, &values, true).unwrap();
    if armed {
        assert!(held.is_none(), "prepare holds the plane until commit");
        let consumed = consumer(&mut k, &union_keys, &union_values);
        let err = Plane::commit_chunk(&mut k, &mut held, guard, &consumed)
            .expect_err("a prepended guard must fail closed at the fenced commit");
        let text = format!("{err:?}");
        assert!(
            text.contains("ring-view chunk met the fenced commit"),
            "unexpected error: {text}"
        );
        assert!(
            held.is_none(),
            "the failed commit must leave the slot vacant"
        );
    } else {
        let consumed = consumer(&mut k, &union_keys, &union_values);
        Plane::commit_chunk(&mut k, &mut held, guard, &consumed)
            .expect("unarmed: the fenced commit owns the append");
        assert!(held.is_some());
    }
}
