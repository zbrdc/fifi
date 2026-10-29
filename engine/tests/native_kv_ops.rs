//! Native tests of three operations: the consuming `kv_append`, the prefix-view
//! `grouped_attention_prefix`, and `evaluate_async`.

#![cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]

use std::{sync::Mutex, time::Instant};

use inference_engine::{
    EngError,
    kernels::{MlxKernels, UnaryOp},
    tensor::{Dtype, ExecutionTensor, Tensor},
};

static GPU_TEST: Mutex<()> = Mutex::new(());

fn f32_input(shape: Vec<u32>, values: &[f32]) -> Tensor {
    Tensor::try_from_host(
        Dtype::F32,
        shape,
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
    .unwrap()
}

/// A deterministic F32 tensor of `shape` from a value function, without naming every value.
fn f32_pattern(shape: &[u32], value: impl Fn(usize) -> f32) -> Tensor {
    let count: usize = shape.iter().map(|&dim| dim as usize).product();
    let values: Vec<f32> = (0..count).map(value).collect();
    f32_input(shape.to_vec(), &values)
}

fn values_of(tensor: &Tensor) -> Vec<f32> {
    tensor
        .host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

/// Decodes BF16 readback bytes to F32 (each BF16 value is the top half of an F32).
fn bf16_values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .host_bytes()
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16))
        .collect()
}

fn import(kernels: &mut MlxKernels, tensor: &Tensor) -> ExecutionTensor {
    kernels.import_tensor(tensor).unwrap()
}

fn assert_near(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: {actual:?}");
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 1e-6 * b.abs().max(1.0),
            "{context} element {index}: actual {a}, expected {b}"
        );
    }
}

/// One `[1, 2, 8, 4]` cache and one `[1, 2, 1, 4]` update row with exact small-integer values.
fn kv_parts() -> (Tensor, Tensor) {
    let target = f32_pattern(&[1, 2, 8, 4], |index| ((index % 11) as f32) - 5.0);
    let update = f32_pattern(&[1, 2, 1, 4], |index| 100.0 + (index % 8) as f32);
    (target, update)
}

/// `kv_append` writes the update at `start` and leaves every other row unchanged, bit for bit
/// as `slice_update` on the same inputs.
#[test]
fn kv_append_matches_slice_update_bit_for_bit() {
    let _serial = GPU_TEST.lock().unwrap();
    let (target, update) = kv_parts();
    let start = [0u32, 0, 3, 0];
    let mut kernels = MlxKernels::try_new().unwrap();

    // The retained path: `slice_update` on a target the caller keeps.
    let kept = import(&mut kernels, &target);
    let row = import(&mut kernels, &update);
    let expected = kernels.slice_update(&kept, &row, &start).unwrap();
    let expected_read = kernels.readback(&expected).unwrap();
    let expected_bytes = expected_read.host_bytes().unwrap().to_vec();

    // The consuming path on a second copy of the same data.
    let consumed = import(&mut kernels, &target);
    let got = kernels.kv_append(consumed, &row, &start).unwrap();
    assert_eq!(
        got.shape(),
        [1, 2, 8, 4],
        "the result keeps the target shape"
    );
    assert_eq!(got.dtype(), Dtype::F32);
    assert_eq!(got.byte_len(), 256);
    let got_read = kernels.readback(&got).unwrap();
    let got_bytes = got_read.host_bytes().unwrap().to_vec();
    assert_eq!(
        got_bytes, expected_bytes,
        "kv_append must match slice_update bit for bit"
    );

    // The update landed at `start` and every other row kept the imported values: element
    // (g, t, d) of the `[1, 2, 8, 4]` layout sits at (g * 8 + t) * 4 + d.
    let (got_values, target_values) = (values(&got_bytes), values_of(&target));
    for g in 0..2usize {
        for d in 0..4usize {
            let before = target_values[(g * 8 + 2) * 4 + d];
            assert_eq!(
                got_values[(g * 8 + 2) * 4 + d],
                before,
                "row 2 of head {g} lane {d} changed"
            );
            assert_eq!(
                got_values[(g * 8 + 3) * 4 + d],
                100.0 + (((g * 4 + d) % 8) as f32),
                "row 3 of head {g} lane {d} did not take the update"
            );
        }
    }
}

/// A validation failure of `kv_append` returns `BadRequest`, frees the consumed handle once
/// through the retire list, and leaves the adapter usable: an unrelated resident still reads
/// its old values afterwards, and a later `kv_append` on the same adapter works.
#[test]
fn kv_append_validation_failure_frees_once_and_keeps_the_adapter_usable() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    // A 2 MiB target makes the retirement of the consumed handle visible in the allocator:
    // [1, 2, 1024, 256] F32 is 2 * 1024 * 256 * 4 bytes.
    let target_bytes: u64 = 2 * 1024 * 256 * 4;
    let target = import(
        &mut kernels,
        &f32_pattern(&[1, 2, 1024, 256], |index| ((index % 97) as f32) / 8.0),
    );
    let row = one_kv_row(&mut kernels);
    let unrelated = import(
        &mut kernels,
        &f32_pattern(&[1, 2, 4, 4], |index| ((index % 7) as f32) - 3.0),
    );
    let before = kernels.memory_stats().unwrap().active;

    // start 1024 is out of bounds: the update does not fit the target.
    let failed = kernels.kv_append(target, &row, &[0, 0, 1024, 0]);
    assert!(
        matches!(failed, Err(EngError::BadRequest { .. })),
        "an out-of-bounds start must be BadRequest: {:?}",
        failed.err()
    );

    // The adapter stays usable and the unrelated resident keeps its old values; the readback is
    // also the completed synchronization that flushes the retire list.
    let read = kernels.readback(&unrelated).unwrap();
    let expected_unrelated: Vec<f32> = (0..32).map(|index| ((index % 7) as f32) - 3.0).collect();
    assert_near(
        &values_of(&read),
        &expected_unrelated,
        "the unrelated resident after the failed call",
    );

    // The consumed handle was retired and freed exactly once at that flush: the active bytes
    // fell by about the target while the row and the unrelated resident stayed. A handle that
    // never freed keeps the bytes; a second free of the same wrapper fails the flush.
    let after = kernels.memory_stats().unwrap().active;
    let freed = before.saturating_sub(after);
    assert!(
        freed >= target_bytes / 2,
        "the consumed handle freed {freed} bytes of the {target_bytes}-byte target"
    );

    // A later kv_append on the same adapter still works.
    let fresh = import(
        &mut kernels,
        &f32_pattern(&[1, 2, 8, 4], |index| ((index % 11) as f32) - 5.0),
    );
    let update = import(
        &mut kernels,
        &f32_pattern(&[1, 2, 1, 4], |index| 100.0 + (index % 8) as f32),
    );
    let got = kernels.kv_append(fresh, &update, &[0, 0, 3, 0]).unwrap();
    let got_read = kernels.readback(&got).unwrap();
    let got_values = values_of(&got_read);
    for g in 0..2usize {
        for d in 0..4usize {
            assert_eq!(
                got_values[(g * 8 + 3) * 4 + d],
                100.0 + (((g * 4 + d) % 8) as f32),
                "row 3 of head {g} lane {d} did not take the update after the earlier failure"
            );
        }
    }
}

/// A caller that keeps a `share()` of the target loses the in-place donation but stays correct:
/// the appended result holds the new row, and the kept share still reads the OLD rows, because
/// the append copies the buffer instead of donating it under a held share.
#[test]
fn kv_append_with_a_kept_share_reads_the_old_rows_from_the_share() {
    let _serial = GPU_TEST.lock().unwrap();
    let (target, update) = kv_parts();
    let original = values_of(&target);
    let mut kernels = MlxKernels::try_new().unwrap();
    let target = import(&mut kernels, &target);
    let row = import(&mut kernels, &update);
    let kept = target.share();
    let got = kernels.kv_append(target, &row, &[0, 0, 3, 0]).unwrap();

    // The appended result holds the update at `start` and the old values elsewhere.
    let got_read = kernels.readback(&got).unwrap();
    let got_values = values_of(&got_read);
    for g in 0..2usize {
        for d in 0..4usize {
            assert_eq!(
                got_values[(g * 8 + 3) * 4 + d],
                100.0 + (((g * 4 + d) % 8) as f32),
                "row 3 of head {g} lane {d} did not take the update"
            );
            assert_eq!(
                got_values[(g * 8 + 2) * 4 + d],
                (((g * 8 + 2) * 4 + d) % 11) as f32 - 5.0,
                "row 2 of head {g} lane {d} changed"
            );
        }
    }

    // The kept share reads the OLD rows: no in-place write may happen under a held share.
    let kept_read = kernels.readback(&kept).unwrap();
    assert_eq!(
        values_of(&kept_read),
        original,
        "the kept share must still read the rows before the append"
    );
}

/// A lazy reader of the target that is built BEFORE `kv_append` and evaluated AFTER still sees
/// the old rows (review B finding 7): the lazy slice node holds the target array, so MLX copies
/// instead of donating, and the rows the view reads never change under it.
#[test]
fn kv_append_leaves_a_lazy_reader_built_before_it_on_the_old_rows() {
    let _serial = GPU_TEST.lock().unwrap();
    let (target, update) = kv_parts();
    let mut kernels = MlxKernels::try_new().unwrap();
    let target = import(&mut kernels, &target);
    let row = import(&mut kernels, &update);

    // The lazy reader: rows 2..6 of head 0, built and left unevaluated. Element (0, t, d) of
    // the `[1, 2, 8, 4]` target sits at t * 4 + d, so the view reads flat indices 8..24.
    let view = kernels
        .slice(&target, &[0, 0, 2, 0], &[1, 1, 6, 4])
        .unwrap();
    let expected: Vec<f32> = (8..24).map(|index| ((index % 11) as f32) - 5.0).collect();

    let got = kernels.kv_append(target, &row, &[0, 0, 3, 0]).unwrap();
    let view_read = kernels.readback(&view).unwrap();
    assert_eq!(
        values_of(&view_read),
        expected,
        "a lazy reader built before the append must still see the old rows"
    );

    // The append itself is still correct.
    let got_read = kernels.readback(&got).unwrap();
    let got_values = values_of(&got_read);
    assert_eq!(
        got_values[3 * 4],
        100.0,
        "row 3 of head 0 lane 0 did not take the update"
    );
}

/// One F32 row of the cache: `[1, 2, 1, 256]`.
fn one_kv_row(kernels: &mut MlxKernels) -> ExecutionTensor {
    import(
        kernels,
        &f32_pattern(&[1, 2, 1, 256], |index| ((index % 13) as f32) / 4.0),
    )
}

/// `kv_append` donates the buffer: 200 appends of one row into one `[1, 2, 16384, 256]` F32
/// buffer (33,554,432 bytes, at least 32 MB) grow the allocator peak by less than a quarter of
/// one buffer. A donating append writes the row in place, so the expected growth is about one
/// row (2 KiB) plus scratch; any whole-buffer copy that is live at the peak adds at least one
/// full buffer and fails the bound. The same loop with the retained `slice_update` runs under
/// the same 8 GiB retire cap and must grow by at least two buffers: the retire list holds every
/// intermediate of the eight-append window until the evaluate flushes it, which proves the
/// metric sees a copy. After the loop the buffer is read back and the written rows and an
/// untouched row are checked.
#[test]
fn kv_append_donates_the_buffer_over_200_appends() {
    let _serial = GPU_TEST.lock().unwrap();
    let buffer_bytes: u64 = 2 * 16384 * 256 * 4;
    assert!(
        buffer_bytes >= 32_000_000,
        "one buffer must be at least 32 MB"
    );
    let mut kernels = MlxKernels::try_new().unwrap();
    let row = one_kv_row(&mut kernels);

    // A consumed target that retires instead of releasing at build time stays referenced until
    // the next completed synchronization, exactly as the Q36 decode cap of 4 GiB holds a token's
    // temporaries. The donating loop therefore holds a cap that no forced flush can reach, so
    // the donation bound below cannot pass on a copy per append.
    let previous_cap = kernels.set_retire_cap(8 << 30).unwrap();

    // The donating loop: every append writes into the one donated buffer.
    let cache_value = |index: usize| ((index % 97) as f32) / 8.0;
    let row_value = |index: usize| ((index % 13) as f32) / 4.0;
    let mut cache = import(&mut kernels, &f32_pattern(&[1, 2, 16384, 256], cache_value));
    kernels.reset_peak_memory().unwrap();
    let start = kernels.memory_stats().unwrap().active;
    let began = Instant::now();
    for step in 0..200u32 {
        cache = kernels.kv_append(cache, &row, &[0, 0, step, 0]).unwrap();
        if step % 8 == 7 {
            kernels.evaluate(&[&cache]).unwrap();
        }
    }
    let append_ms = began.elapsed().as_secs_f64() / 200.0 * 1e3;
    let growth = kernels.memory_stats().unwrap().peak.saturating_sub(start);
    println!(
        "kv_append: {append_ms:.3} ms per append, peak growth {growth} bytes over a {buffer_bytes}-byte buffer"
    );
    assert!(
        growth < buffer_bytes / 4,
        "a donating append grew the peak by {growth} bytes; the bound is one quarter of a buffer ({})",
        buffer_bytes / 4
    );

    // The written rows and one untouched row: element (g, t, d) of the `[1, 2, 16384, 256]`
    // layout sits at (g * 16384 + t) * 256 + d; row t < 200 holds the update written at step t,
    // row 200 was never written.
    let read = kernels.readback(&cache).unwrap();
    let got = values_of(&read);
    for g in 0..2usize {
        for t in 0..=200usize {
            for d in 0..256usize {
                let expected = if t < 200 {
                    row_value(g * 256 + d)
                } else {
                    cache_value((g * 16384 + t) * 256 + d)
                };
                assert_eq!(
                    got[(g * 16384 + t) * 256 + d],
                    expected,
                    "head {g} row {t} lane {d} after 200 appends"
                );
            }
        }
    }
    drop(cache);
    drop(read);

    // The control loop on the same cadence and under the same cap: `slice_update` keeps its
    // target, so every intermediate retires and stays alive until the evaluate flushes it —
    // about eight buffers per window, which proves a copy is visible to the same metric.
    let mut retained = import(&mut kernels, &f32_pattern(&[1, 2, 16384, 256], cache_value));
    kernels.reset_peak_memory().unwrap();
    let start = kernels.memory_stats().unwrap().active;
    let began = Instant::now();
    for step in 0..200u32 {
        retained = kernels
            .slice_update(&retained, &row, &[0, 0, step, 0])
            .unwrap();
        if step % 8 == 7 {
            kernels.evaluate(&[&retained]).unwrap();
        }
    }
    let slice_ms = began.elapsed().as_secs_f64() / 200.0 * 1e3;
    let growth = kernels.memory_stats().unwrap().peak.saturating_sub(start);
    println!(
        "slice_update: {slice_ms:.3} ms per append, peak growth {growth} bytes over the same buffer"
    );
    assert!(
        growth >= 2 * buffer_bytes,
        "the retained control must show the copy: growth {growth} bytes, the bound is two buffers ({})",
        2 * buffer_bytes
    );
    drop(retained);
    drop(row);
    kernels.set_retire_cap(previous_cap).unwrap();
}

/// Queries, keys, and values of one attention configuration with exact small-integer patterns.
fn attention_parts(queries: u32, capacity: u32, width: u32) -> (Tensor, Tensor, Tensor) {
    let q = f32_pattern(&[1, 16, queries, width], |index| {
        (((index % 7) as f32) - 3.0) / 4.0
    });
    let k = f32_pattern(&[1, 2, capacity, width], |index| {
        (((index % 5) as f32) - 2.0) / 4.0
    });
    let v = f32_pattern(&[1, 2, capacity, width], |index| {
        (((index * 3 % 9) as f32) - 4.0) / 4.0
    });
    (q, k, v)
}

/// `grouped_attention_prefix` over a capacity buffer equals `grouped_attention` over a contiguous
/// copy of the prefix bit for bit, at short and full prefixes, GQA 16 over 2 heads (with the
/// causal `Tq = 1` decode shape and a capacity beyond 16). The causal sweep uses `len >= Tq`: a
/// shorter causal prefix masks every key of the earlier queries, so those rows hold no defined
/// value to compare.
#[test]
fn grouped_attention_prefix_matches_the_contiguous_prefix() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    for &(width, tq, causal, cap, lens) in &[
        (64u32, 1u32, false, 16u32, &[1u32, 5, 15, 16][..]),
        (64, 7, true, 16, &[7, 8, 15, 16]),
        // The causal decode shape with one query, and a capacity larger than 16.
        (64, 1, true, 16, &[1, 5, 15, 16]),
        (64, 1, true, 40, &[1, 5, 39, 40]),
        (256, 1, false, 16, &[5]),
        (256, 7, true, 16, &[16]),
    ] {
        let (q, k, v) = attention_parts(tq, cap, width);
        let (q, k, v) = (
            import(&mut kernels, &q),
            import(&mut kernels, &k),
            import(&mut kernels, &v),
        );
        let scale = 1.0 / (width as f32).sqrt();
        for &len in lens {
            let prefix_k = kernels
                .slice(&k, &[0, 0, 0, 0], &[1, 2, len, width])
                .unwrap();
            let prefix_v = kernels
                .slice(&v, &[0, 0, 0, 0], &[1, 2, len, width])
                .unwrap();
            let expected = kernels
                .grouped_attention(&q, &prefix_k, &prefix_v, scale, causal)
                .unwrap();
            let got = kernels
                .grouped_attention_prefix(&q, &k, &v, len, scale, causal)
                .unwrap();
            assert_eq!(got.shape(), [1, 16, tq, width], "width {width}, len {len}");
            let expected_read = kernels.readback(&expected).unwrap();
            let expected_bytes = expected_read.host_bytes().unwrap().to_vec();
            let got_read = kernels.readback(&got).unwrap();
            let got_bytes = got_read.host_bytes().unwrap().to_vec();
            assert_eq!(
                got_bytes, expected_bytes,
                "the prefix view must equal the contiguous prefix (width {width}, tq {tq}, causal {causal}, len {len}, cap {cap})"
            );
        }
    }
}

/// `grouped_attention_prefix` over BF16 buffers equals `grouped_attention` over the contiguous
/// BF16 prefix bit for bit after `allow_bf16`. The contiguous reference slices the F32 buffers
/// and casts each prefix, so both paths see bitwise-identical BF16 keys and values.
#[test]
fn grouped_attention_prefix_matches_the_contiguous_prefix_in_bf16() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    kernels.allow_bf16(true);
    let (q, k, v) = attention_parts(7, 16, 64);
    let (q32, k32, v32) = (
        import(&mut kernels, &q),
        import(&mut kernels, &k),
        import(&mut kernels, &v),
    );
    let (q, k, v) = (
        kernels.cast(&q32, Dtype::Bf16).unwrap(),
        kernels.cast(&k32, Dtype::Bf16).unwrap(),
        kernels.cast(&v32, Dtype::Bf16).unwrap(),
    );
    let scale = 1.0 / 8.0;
    let len = 11;
    let expected = {
        let prefix_k = kernels
            .slice(&k32, &[0, 0, 0, 0], &[1, 2, len, 64])
            .unwrap();
        let prefix_v = kernels
            .slice(&v32, &[0, 0, 0, 0], &[1, 2, len, 64])
            .unwrap();
        let (prefix_k, prefix_v) = (
            kernels.cast(&prefix_k, Dtype::Bf16).unwrap(),
            kernels.cast(&prefix_v, Dtype::Bf16).unwrap(),
        );
        kernels
            .grouped_attention(&q, &prefix_k, &prefix_v, scale, true)
            .unwrap()
    };
    let got = kernels
        .grouped_attention_prefix(&q, &k, &v, len, scale, true)
        .unwrap();
    assert_eq!(got.shape(), [1, 16, 7, 64]);
    assert_eq!(got.dtype(), Dtype::Bf16, "a BF16 prefix keeps BF16");
    let expected_bytes = kernels
        .readback(&expected)
        .unwrap()
        .host_bytes()
        .unwrap()
        .to_vec();
    let got_bytes = kernels
        .readback(&got)
        .unwrap()
        .host_bytes()
        .unwrap()
        .to_vec();
    assert_eq!(
        got_bytes, expected_bytes,
        "the BF16 prefix view must equal the contiguous BF16 prefix"
    );
}

/// The attention of `queries` over the first `len` keys with scale one, computed on the host with
/// the causal rule of `grouped_attention`: a causal mask aligned to the last key lets query `i`
/// see keys `0..=len - Tq + i`, and a non-causal mask lets every query see all `len` keys.
fn expected_attention(
    queries: &[[f32; 2]],
    keys: &[[f32; 2]],
    values: &[[f32; 2]],
    len: usize,
    causal: bool,
) -> Vec<f32> {
    let tq = queries.len();
    let mut out = Vec::new();
    for (i, query) in queries.iter().enumerate() {
        let visible = if causal { len - (tq - 1) + i } else { len };
        let logits: Vec<f64> = (0..visible)
            .map(|j| f64::from(query[0] * keys[j][0] + query[1] * keys[j][1]))
            .collect();
        let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = logits.iter().map(|logit| (logit - top).exp()).sum();
        let mut weighted = [0.0f64; 2];
        for (j, logit) in logits.iter().enumerate() {
            let weight = (logit - top).exp() / total;
            weighted[0] += weight * f64::from(values[j][0]);
            weighted[1] += weight * f64::from(values[j][1]);
        }
        out.extend(weighted.map(|value| value as f32));
    }
    out
}

/// The causal mask of the prefix aligns to the last key of the prefix, as `grouped_attention`
/// aligns it. The buffers are the fixture of
/// `native_fast::grouped_attention_aligns_the_causal_mask_to_the_last_key`, so the
/// full-capacity case reproduces that test's frozen values; a mask aligned to the first key
/// would instead give query 0 only key 0, `(10, 0)`.
#[test]
fn grouped_attention_prefix_aligns_the_causal_mask_to_the_last_key_of_the_prefix() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let queries = [[1.0f32, 0.0], [0.0, 1.0]];
    let keys = [[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]];
    let values = [[10.0f32, 0.0], [0.0, 20.0], [30.0, 30.0], [5.0, 7.0]];
    let q = import(
        &mut kernels,
        &f32_input(vec![1, 1, 2, 2], &queries.concat()),
    );
    let k = import(&mut kernels, &f32_input(vec![1, 1, 4, 2], &keys.concat()));
    let v = import(&mut kernels, &f32_input(vec![1, 1, 4, 2], &values.concat()));

    let len = 3;
    let got = kernels
        .grouped_attention_prefix(&q, &k, &v, len, 1.0, true)
        .unwrap();
    assert_eq!(got.shape(), [1, 1, 2, 2]);
    let read = kernels.readback(&got).unwrap();
    assert_near(
        &values_of(&read),
        &expected_attention(&queries, &keys, &values, len as usize, true),
        "causal prefix of 3 keys",
    );

    // The full capacity is the neighbor test's fixture.
    let full = kernels
        .grouped_attention_prefix(&q, &k, &v, 4, 1.0, true)
        .unwrap();
    let read = kernels.readback(&full).unwrap();
    assert_near(
        &values_of(&read),
        &[16.892_752, 15.776_812, 12.982_939, 19.217_759],
        "causal prefix of the full capacity",
    );
}

/// The prefix views are not copied: at CAP 8,192 the attention over the first
/// `len = CAP / 2 = 4,096` rows reads views that are strided across the head axis, so making
/// them contiguous would copy two prefixes of `2 * 4096 * 256` F32 values each:
/// `2 * 8,388,608 = 16,777,216` bytes (about 16.8 MiB) of extra allocator bytes. The view path
/// allocates only the `[1, 16, 1, 256]` F32 output (16,384 bytes) and the fused attention's
/// workspace, so its peak growth stays under 4 MiB. A full-range slice is already row
/// contiguous and cannot tell a view from a copy, which is why the length is half the
/// capacity. The contiguous control copies both prefixes and must grow by at least the two
/// copies, which proves the metric sees them. The wall times of both paths are printed for the
/// report.
#[test]
fn grouped_attention_prefix_does_not_copy_the_prefix_at_half_capacity() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let (q, k, v) = attention_parts(1, 8192, 256);
    let (q, k, v) = (
        import(&mut kernels, &q),
        import(&mut kernels, &k),
        import(&mut kernels, &v),
    );
    let scale = 1.0 / 16.0;
    let prefix_bytes: u64 = 2 * 4096 * 256 * 4;

    // The peak growth of one prefix attention over the capacity buffers.
    kernels.reset_peak_memory().unwrap();
    let base = kernels.memory_stats().unwrap().active;
    let probe = kernels
        .grouped_attention_prefix(&q, &k, &v, 4096, scale, false)
        .unwrap();
    kernels.readback(&probe).unwrap();
    drop(probe);
    let prefix_peak = kernels.memory_stats().unwrap().peak.saturating_sub(base);

    // The control: the same attention over contiguous copies of both prefixes.
    kernels.reset_peak_memory().unwrap();
    let base = kernels.memory_stats().unwrap().active;
    let control = {
        let prefix_k = kernels
            .slice(&k, &[0, 0, 0, 0], &[1, 2, 4096, 256])
            .unwrap();
        let prefix_v = kernels
            .slice(&v, &[0, 0, 0, 0], &[1, 2, 4096, 256])
            .unwrap();
        kernels
            .grouped_attention(&q, &prefix_k, &prefix_v, scale, false)
            .unwrap()
    };
    kernels.readback(&control).unwrap();
    drop(control);
    let contiguous_peak = kernels.memory_stats().unwrap().peak.saturating_sub(base);

    // The mean wall time of ten readback-complete rounds per path, after one warm-up round.
    let mut rounds = |build: &dyn Fn(&mut MlxKernels) -> ExecutionTensor| {
        let warm = build(&mut kernels);
        kernels.readback(&warm).unwrap();
        drop(warm);
        let began = Instant::now();
        for _ in 0..10 {
            let output = build(&mut kernels);
            kernels.readback(&output).unwrap();
        }
        began.elapsed().as_secs_f64() / 10.0 * 1e3
    };
    let prefix_ms = rounds(&|kernels| {
        kernels
            .grouped_attention_prefix(&q, &k, &v, 4096, scale, false)
            .unwrap()
    });
    let contiguous_ms = rounds(&|kernels| {
        let prefix_k = kernels
            .slice(&k, &[0, 0, 0, 0], &[1, 2, 4096, 256])
            .unwrap();
        let prefix_v = kernels
            .slice(&v, &[0, 0, 0, 0], &[1, 2, 4096, 256])
            .unwrap();
        kernels
            .grouped_attention(&q, &prefix_k, &prefix_v, scale, false)
            .unwrap()
    });
    println!(
        "prefix views at len 4096: {prefix_ms:.3} ms with peak growth {prefix_peak} bytes; \
         contiguous prefixes: {contiguous_ms:.3} ms with peak growth {contiguous_peak} bytes"
    );

    assert!(
        prefix_peak < 4 * 1024 * 1024,
        "the prefix views were copied: peak growth {prefix_peak} bytes, the bound is 4 MiB"
    );
    assert!(
        contiguous_peak >= 2 * prefix_bytes,
        "the control did not copy the prefixes: peak growth {contiguous_peak} bytes, \
         the bound is the two prefix copies ({})",
        2 * prefix_bytes
    );
}

/// The pipelined decode case at the sizes of the Q36 decode: 100 iterations of `kv_append` of a
/// new row into the two chain buffers, `grouped_attention_prefix` over the new length,
/// `evaluate_async` of the attention output, and the next iteration built from the still-lazy
/// chains before any wait; every tenth iteration synchronizes and reads the attention output
/// back. The chains are decode-sized (`[1, 2, 4096, 256]` F32, 8 MiB each, head dimension 256 as
/// the fused kernel of the real model), so a copy of a chain per append is far above scratch and
/// allocator rounding in the printed peak growth. Correctness is asserted at every
/// synchronization point (the attention output at that length against an independent f64 host
/// reference), so a write-after-read race that corrupts an intermediate step fails; the peak
/// growth and the wall time per iteration are printed, with no performance assertion: donation
/// may fail while the previous token is still in flight, and if so the numbers show it (a
/// donating loop grows the peak by far less than one chain, 8,388,608 bytes).
#[test]
fn kv_append_in_a_pipelined_loop() {
    let _serial = GPU_TEST.lock().unwrap();
    let steps = 100u32;
    let width = 256u32;
    let capacity = 4096u32;
    let scale = 1.0 / (width as f32).sqrt();
    let key_value =
        |step: usize, index: usize| (((step * 17 + index * 3) % 23) as f32 - 11.0) / 8.0;
    let value_value =
        |step: usize, index: usize| (((step * 13 + index * 5) % 19) as f32 - 9.0) / 8.0;
    let query_value = |index: usize| (((index * 7) % 15) as f32 - 7.0) / 8.0;

    let mut kernels = MlxKernels::try_new().unwrap();
    // One batched import for the 100 distinct key rows and one for the value rows, so the
    // measured loop performs no import synchronization.
    let key_rows: Vec<Tensor> = (0..steps as usize)
        .map(|step| f32_pattern(&[1, 2, 1, width], |index| key_value(step, index)))
        .collect();
    let value_rows: Vec<Tensor> = (0..steps as usize)
        .map(|step| f32_pattern(&[1, 2, 1, width], |index| value_value(step, index)))
        .collect();
    let key_refs: Vec<&Tensor> = key_rows.iter().collect();
    let value_refs: Vec<&Tensor> = value_rows.iter().collect();
    let key_rows = kernels.import_many(&key_refs).unwrap();
    let value_rows = kernels.import_many(&value_refs).unwrap();
    let q = import(&mut kernels, &f32_pattern(&[1, 2, 1, width], query_value));
    // The chain buffers start at zero; rows 0..100 are written by the loop, the rest stays zero
    // and is never read (the attention length is the number of written rows).
    let mut key_chain = import(
        &mut kernels,
        &f32_pattern(&[1, 2, capacity, width], |_| 0.0),
    );
    let mut value_chain = import(
        &mut kernels,
        &f32_pattern(&[1, 2, capacity, width], |_| 0.0),
    );

    // The host reference of the attention of the single query over the first `len` written rows:
    // element (g, d) of a `[1, 2, 1, width]` row sits at g * width + d, and with one query row the
    // causal mask lets it see every written key. The logits and the softmax run in f64.
    let query: Vec<f32> = (0..2 * width as usize).map(query_value).collect();
    let reference = |len: usize| -> Vec<f32> {
        let mut expected = Vec::with_capacity(2 * width as usize);
        for g in 0..2usize {
            let logits: Vec<f64> = (0..len)
                .map(|j| {
                    let dot: f64 = (0..width as usize)
                        .map(|d| {
                            f64::from(query[g * width as usize + d])
                                * f64::from(key_value(j, g * width as usize + d))
                        })
                        .sum();
                    dot * f64::from(scale)
                })
                .collect();
            let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = logits.iter().map(|logit| (logit - top).exp()).sum();
            for d in 0..width as usize {
                let weighted: f64 = logits
                    .iter()
                    .enumerate()
                    .map(|(j, logit)| {
                        ((logit - top).exp() / total)
                            * f64::from(value_value(j, g * width as usize + d))
                    })
                    .sum();
                expected.push(weighted as f32);
            }
        }
        expected
    };
    // The kernel accumulates in f32 over up to 100 keys of 256 lanes: a looser bound than the
    // exact-integer cases (1e-5 relative to max(1, |expected|)).
    let check = |read: &Tensor, len: usize, context: &str| {
        assert_eq!(read.shape(), [1, 2, 1, width], "{context}");
        let actual = values_of(read);
        let expected = reference(len);
        assert_eq!(actual.len(), expected.len(), "{context}");
        for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 * b.abs().max(1.0),
                "{context} element {index}: actual {a}, expected {b}"
            );
        }
    };

    kernels.reset_peak_memory().unwrap();
    let start = kernels.memory_stats().unwrap().active;
    let began = Instant::now();
    let mut last = None;
    for step in 0..steps {
        let index = step as usize;
        key_chain = kernels
            .kv_append(key_chain, &key_rows[index], &[0, 0, step, 0])
            .unwrap();
        value_chain = kernels
            .kv_append(value_chain, &value_rows[index], &[0, 0, step, 0])
            .unwrap();
        let attn = kernels
            .grouped_attention_prefix(&q, &key_chain, &value_chain, step + 1, scale, true)
            .unwrap();
        kernels.evaluate_async(&[&attn]).unwrap();
        if step % 10 == 9 {
            kernels.evaluate(&[&attn]).unwrap();
            let read = kernels.readback(&attn).unwrap();
            check(&read, index + 1, &format!("the attention at step {step}"));
        }
        last = Some(attn);
    }
    let read = kernels.readback(&last.unwrap()).unwrap();
    let per_iteration_ms = began.elapsed().as_secs_f64() / steps as f64 * 1e3;
    let growth = kernels.memory_stats().unwrap().peak.saturating_sub(start);
    println!(
        "pipelined loop: {per_iteration_ms:.3} ms per iteration, peak growth {growth} bytes \
         (one chain is {} bytes)",
        u64::from(capacity) * u64::from(width) * 2 * 4
    );
    check(
        &read,
        steps as usize,
        "the pipelined attention over all written rows",
    );
}

/// `evaluate_async` submits the graph, and MLX allocates the output at submission: right after
/// the call returns, before any synchronization, the active bytes grew by about the 16 MiB
/// output. A no-op submission allocates nothing and fails the bound.
#[test]
fn evaluate_async_allocates_the_output_at_submission() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let output_bytes: u64 = 4096 * 1024 * 4;
    let input = |index: usize| ((index % 17) as f32) / 8.0;
    let x = import(&mut kernels, &f32_pattern(&[1, 4096, 1024], input));
    let lazy = kernels.silu(&x).unwrap();
    let before = kernels.memory_stats().unwrap().active;
    kernels.evaluate_async(&[&lazy]).unwrap();
    let after = kernels.memory_stats().unwrap().active;
    let growth = after.saturating_sub(before);
    println!(
        "evaluate_async allocated {growth} bytes at submission for a {output_bytes}-byte output"
    );
    assert!(
        growth >= output_bytes - output_bytes / 8,
        "the submission allocated only {growth} bytes; about {output_bytes} were expected"
    );
    // The submitted values are still the SiLU of the input.
    let read = kernels.readback(&lazy).unwrap();
    let values = values_of(&read);
    assert!(
        (values[0] - silu_of(input(0))).abs() <= 1e-6,
        "element 0: actual {}, expected {}",
        values[0],
        silu_of(input(0))
    );
    assert!(
        (values[17] - silu_of(input(17))).abs() <= 1e-6,
        "element 17: actual {}, expected {}",
        values[17],
        silu_of(input(17))
    );
}

/// `evaluate_async` never flushes the retire list: a retired 16 MiB resident stays allocated
/// across the submission and frees only at the next completed synchronization, the blocking
/// `evaluate` below. A submission that flushed the list itself would free the retired bytes at
/// once and fail the first bound.
#[test]
fn evaluate_async_never_flushes_the_retire_list() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let bulk_bytes: u64 = 4096 * 1024 * 4;
    let bulk = import(
        &mut kernels,
        &f32_pattern(&[1, 4096, 1024], |index| ((index % 19) as f32) / 8.0),
    );
    let x = import(
        &mut kernels,
        &f32_pattern(&[1, 1024], |index| ((index % 11) as f32) / 4.0),
    );
    let lazy = kernels.silu(&x).unwrap();
    let with_bulk = kernels.memory_stats().unwrap().active;
    drop(bulk); // Retires 16 MiB; the 64 MB cap is not crossed, so nothing frees yet.
    let retired = kernels.memory_stats().unwrap().active;
    assert_eq!(
        retired, with_bulk,
        "the retire list must hold the bytes until a synchronization"
    );
    kernels.evaluate_async(&[&lazy]).unwrap();
    let after_async = kernels.memory_stats().unwrap().active;
    println!("active after the submission: {after_async} bytes (was {retired} before it)");
    assert!(
        after_async >= retired,
        "evaluate_async flushed the retire list: {retired} -> {after_async} bytes"
    );
    kernels.evaluate(&[&lazy]).unwrap(); // Completed synchronization: the flush frees the bulk.
    let after_sync = kernels.memory_stats().unwrap().active;
    println!("active after the blocking evaluate: {after_sync} bytes");
    assert!(
        after_sync + bulk_bytes / 2 <= after_async,
        "the blocking evaluate freed the retired bytes: {after_async} -> {after_sync} bytes"
    );
}

/// `evaluate_async` submits without waiting and a later `readback` blocks until the values are
/// complete.
#[test]
fn evaluate_async_then_readback_returns_the_right_values() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = import(&mut kernels, &f32_input(vec![4], &[0.0, 2.0, -3.0, 4.0]));
    let silu = kernels.silu(&x).unwrap();
    let gate = kernels.unary(UnaryOp::Sigmoid, &x).unwrap();

    kernels.evaluate_async(&[&silu, &gate]).unwrap();
    let silu_read = kernels.readback(&silu).unwrap();
    let gate_read = kernels.readback(&gate).unwrap();
    let expected_silu: Vec<f32> = [0.0f32, 2.0, -3.0, 4.0]
        .iter()
        .map(|&value| silu_of(value))
        .collect();
    let expected_gate: Vec<f32> = [0.0f32, 2.0, -3.0, 4.0]
        .iter()
        .map(|&value| sigmoid_of(value))
        .collect();
    assert_near(&values_of(&silu_read), &expected_silu, "async SiLU values");
    assert_near(&values_of(&gate_read), &expected_gate, "async gate values");
}

/// An async evaluation followed by dropping its outputs and a blocking evaluate of another graph
/// neither crashes nor leaks: after a forced flush the allocator returns to the start within
/// 1 MB (three 16 MiB tensors, so any array the submission or the drop leaks exceeds the
/// bound).
#[test]
fn evaluate_async_drop_then_blocking_evaluate_neither_crashes_nor_leaks() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    // A warm import, read back and dropped, then one more synchronization: the start line is
    // taken only after the retire list has flushed the warm array.
    let probe = f32_input(vec![4], &[1.0, 2.0, 3.0, 4.0]);
    let warm = import(&mut kernels, &probe);
    kernels.readback(&warm).unwrap();
    drop(warm);
    let flush = import(&mut kernels, &probe);
    kernels.readback(&flush).unwrap();
    drop(flush);
    let start = kernels.memory_stats().unwrap().active;

    {
        let x = import(
            &mut kernels,
            &f32_pattern(&[1, 4096, 1024], |index| ((index % 17) as f32) / 8.0),
        );
        let first = kernels.silu(&x).unwrap();
        let second = kernels.unary(UnaryOp::Exp, &x).unwrap();
        kernels.evaluate_async(&[&first, &second]).unwrap();
        // Drop the outputs of the pending async evaluation, and then the input.
    }

    // A blocking evaluate of another graph completes the pending work.
    let other = import(&mut kernels, &f32_input(vec![4], &[4.0, 3.0, 2.0, 1.0]));
    let blocked = kernels.silu(&other).unwrap();
    kernels.evaluate(&[&blocked]).unwrap();
    let read = kernels.readback(&blocked).unwrap();
    assert_eq!(values_of(&read)[0] as u32, 3, "the blocking evaluate ran");
    drop((other, blocked, read));
    // One more synchronization forces the retire-list flush of everything above.
    let flush = import(&mut kernels, &probe);
    kernels.readback(&flush).unwrap();
    drop(flush);

    let after = kernels.memory_stats().unwrap().active;
    let drift = after.abs_diff(start);
    println!("evaluate_async leak check: start {start}, after {after}, drift {drift} bytes");
    assert!(
        drift <= 1_000_000,
        "allocator drifted {drift} bytes from the start"
    );
}

/// The sync counter of the qualification build proves the submission is asynchronous (this is
/// the test that kills a blocking `evaluate_async`). Qualification builds only.
#[cfg(feature = "native-fault-qualification")]
#[test]
fn evaluate_async_submits_without_synchronizing() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let x = import(&mut kernels, &f32_input(vec![4], &[0.0, 2.0, -3.0, 4.0]));
    let silu = kernels.silu(&x).unwrap();
    let before = kernels.qualification_sync_count();
    kernels.evaluate_async(&[&silu]).unwrap();
    assert_eq!(
        kernels.qualification_sync_count(),
        before,
        "evaluate_async must not synchronize"
    );
    let read = kernels.readback(&silu).unwrap();
    assert_eq!(
        kernels.qualification_sync_count(),
        before + 1,
        "the readback completes the async evaluation"
    );
    let expected: Vec<f32> = [0.0f32, 2.0, -3.0, 4.0]
        .iter()
        .map(|&value| silu_of(value))
        .collect();
    assert_near(&values_of(&read), &expected, "async SiLU values");
}

/// Imports an F32 pattern and casts it to BF16 in two steps (one mutable borrow each).
fn bf16_of(kernels: &mut MlxKernels, tensor: &Tensor) -> ExecutionTensor {
    let source = import(kernels, tensor);
    kernels.cast(&source, Dtype::Bf16).unwrap()
}

/// `kv_append` keeps BF16 buffers: the byte length halves with the dtype, the dtype-equality
/// branch accepts a BF16 target and update and the values survive, and a mixed F32/BF16 pair
/// is `BadRequest` before any native call.
#[test]
fn kv_append_keeps_bf16_buffers_and_rejects_mixed_dtypes() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    kernels.allow_bf16(true);

    // A mixed dtype pair is rejected: the BF16 target with an F32 update row.
    let mixed_target = bf16_of(
        &mut kernels,
        &f32_pattern(&[1, 2, 8, 4], |index| ((index % 11) as f32) - 5.0),
    );
    let f32_row = import(
        &mut kernels,
        &f32_pattern(&[1, 2, 1, 4], |index| 100.0 + (index % 8) as f32),
    );
    let refused = kernels.kv_append(mixed_target, &f32_row, &[0, 0, 3, 0]);
    assert!(
        matches!(refused, Err(EngError::BadRequest { .. })),
        "a mixed F32/BF16 pair must be BadRequest: {:?}",
        refused.err()
    );

    // The BF16 append: small integers are exact in BF16, so the readback decodes exactly.
    let target = bf16_of(
        &mut kernels,
        &f32_pattern(&[1, 2, 8, 4], |index| ((index % 11) as f32) - 5.0),
    );
    let update = bf16_of(
        &mut kernels,
        &f32_pattern(&[1, 2, 1, 4], |index| 100.0 + (index % 8) as f32),
    );
    let got = kernels.kv_append(target, &update, &[0, 0, 3, 0]).unwrap();
    assert_eq!(got.dtype(), Dtype::Bf16, "a BF16 append keeps BF16");
    assert_eq!(
        got.byte_len(),
        128,
        "[1, 2, 8, 4] BF16 is 64 elements of 2 bytes"
    );
    let read = kernels.readback(&got).unwrap();
    let got_values = bf16_values(&read);
    for g in 0..2usize {
        for d in 0..4usize {
            assert_eq!(
                got_values[(g * 8 + 3) * 4 + d],
                100.0 + (((g * 4 + d) % 8) as f32),
                "row 3 of head {g} lane {d} did not take the BF16 update"
            );
            assert_eq!(
                got_values[(g * 8 + 2) * 4 + d],
                (((g * 8 + 2) * 4 + d) % 11) as f32 - 5.0,
                "row 2 of head {g} lane {d} changed"
            );
        }
    }
}

/// `x / (1 + e^-x)` in F64, rounded once to F32.
fn silu_of(value: f32) -> f32 {
    (f64::from(value) / (1.0 + (-f64::from(value)).exp())) as f32
}

/// `1 / (1 + e^-x)` in F64, rounded once to F32.
fn sigmoid_of(value: f32) -> f32 {
    (1.0 / (1.0 + (-f64::from(value)).exp())) as f32
}

/// Reads F32 values out of raw readback bytes.
fn values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}
