use super::*;
use crate::kernels::{
    attention_reference::oracle::{pinned_max_ratio, sdpa_f64},
    mlx::{KvBlock, SharedAttention},
};

fn early_oracle(start: u32, rows: u32) -> Vec<f64> {
    let origin = start.saturating_sub(512);
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for position in origin..start + rows {
        for head in 0..4 {
            for column in 0..128 {
                let key = f64::from(row_value(head, position, column));
                keys.push(key);
                values.push(-key - 0.25);
            }
        }
    }
    sdpa_f64(
        &vec![0.0; 3 * 32 * 128],
        &keys,
        &values,
        3,
        32,
        (start + rows - origin) as usize,
        128,
        8,
        1.0 / 128.0_f64.sqrt(),
        start,
        origin,
        Some(513),
        None,
    )
}

fn delayed_attention(
    k: &mut MlxKernels,
    keys: &ExecutionTensor,
    values: &ExecutionTensor,
    plan: &ChunkCachePlan,
) -> ExecutionTensor {
    let start = plan.p0;
    let rows = plan.rows;
    let host = Tensor::try_from_host(
        Dtype::F32,
        vec![1, 32, rows, 128],
        vec![0; (32 * rows * 128 * 4) as usize],
    )
    .unwrap();
    let query = k.import_tensor(&host).unwrap();
    let query = k.cast(&query, Dtype::Bf16).unwrap();
    let union_rows = keys.shape()[1];
    let keys = k.reshape(keys, &[1, 4, union_rows, 128]).unwrap();
    let values = k.reshape(values, &[1, 4, union_rows, 128]).unwrap();
    k.shared_attention(SharedAttention {
        q: &query,
        kv: KvBlock::Float {
            keys: &keys,
            values: &values,
        },
        query_start: start,
        key_origin: start.saturating_sub(512),
        window: Some(513),
        sinks: None,
        scale: 1.0 / 128.0_f32.sqrt(),
    })
    .unwrap()
}

#[test]
#[ignore = "explicit tiny Mac GPU allocation; no model; unsafe fence mutation must also fail"]
fn w513_delayed_attention_survives_commit_wrap_and_next_step() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    k.set_cache_limit(0).unwrap();
    for (start, rows) in [(512, 512), (514, 1024), (1026, 17)] {
        let mut held = [None, None];
        for (path, plane) in held.iter_mut().enumerate() {
            let mut position = 0;
            let chunk = if path == 0 { 512 } else { 1 };
            while position < start {
                let count = chunk.min(start - position);
                put(
                    &mut k,
                    plane,
                    &ChunkCachePlan::new(2, position, count, 8192).unwrap(),
                );
                position += count;
            }
        }
        let plan = ChunkCachePlan::new(2, start, rows, 8192).unwrap();
        let new_keys = input(&mut k, start, rows, false);
        let new_values = input(&mut k, start, rows, true);
        let (keys, values, guard) =
            Plane::prepare_chunk(&mut k, &mut held[0], &plan, &new_keys, &new_values, false)
                .unwrap();
        let consumed = delayed_attention(&mut k, &keys, &values, &plan);
        // No attention readback/evaluate before the commit being tested.
        Plane::commit_chunk(&mut k, &mut held[0], guard, &consumed).unwrap();
        for position in start..start + rows {
            put(
                &mut k,
                &mut held[1],
                &ChunkCachePlan::new(2, position, 1, 8192).unwrap(),
            );
        }
        let next = ChunkCachePlan::new(2, start + rows, 1, 8192).unwrap();
        let batch = decode_fixture_logits(&mut k, &mut held[0], &next);
        let scalar = decode_fixture_logits(&mut k, &mut held[1], &next);
        assert_eq!(
            batch, scalar,
            "next-step fixture head start={start} rows={rows}"
        );
        assert_same_cache(&mut k, &held, 2, start + rows + 1);
        let output = k.cast(&consumed, Dtype::F32).unwrap();
        let host = k.readback(&output).unwrap();
        let words = host.host_bytes().unwrap().as_chunks::<4>().0;
        let mut early = Vec::new();
        for head in 0..32 {
            for row in 0..3 {
                let offset = ((head * rows + row) * 128) as usize;
                early.extend(
                    words[offset..offset + 128]
                        .iter()
                        .map(|word| f32::from_le_bytes(*word)),
                );
            }
        }
        let (ratio, offender) = pinned_max_ratio(&early, &early_oracle(start, rows), 32, 3, 128);
        assert!(
            ratio <= 2.0,
            "W513 delayed consumer start={start} rows={rows}: ratio={ratio}, {offender:?}"
        );
        println!("D1_W513 start={start} rows={rows} early_causal_maxratio={ratio}");
    }
}
