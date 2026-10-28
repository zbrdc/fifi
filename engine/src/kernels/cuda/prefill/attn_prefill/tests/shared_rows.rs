use super::*;
use crate::kernels::attention_reference::oracle::sdpa_f64;

fn shape(dim: usize, heads: (usize, usize), rows: usize, p0: i64) -> AttnPrefillShape {
    AttnPrefillShape {
        m: rows,
        n_head: heads.0,
        n_kv_head: heads.1,
        head_dim: dim,
        cache_rows: p0 as usize + rows,
        p0,
        ld: 0,
    }
}

struct Observation {
    actual: Vec<f32>,
    reference: Vec<f64>,
    bound: f64,
    status: i32,
    canaries: bool,
}

// Export device results and drop all owners before the caller asserts on them.
fn observe(
    shape: AttnPrefillShape,
    input: &Data,
    sinks: Option<&[f32]>,
    segment: Option<AttnPrefillSegment>,
    scale: f32,
) -> Result<Observation, EngError> {
    let origin = segment.as_ref().map_or(0, |s| s.key_origin);
    let split = CpfAttnSplit {
        max_splits: if segment.is_some() { 1 } else { 4 },
        min_keys: 32,
    };
    let mut fx = Fx::with_data(shape, split, false, true, input)?;
    fx.scale = scale;
    if let Some(values) = sinks {
        fx.sink_buf = Some(fx.ctx.upload(&le(values))?);
    }
    if let Some(segment) = segment {
        fx.ctx.attn_prefill_part(&fx.stream, fx.call()?, segment)?;
        fx.ctx.attn_prefill_combine(&fx.stream, fx.call()?)?;
        fx.stream.synchronize()?;
    } else {
        fx.run()?;
    }
    let actual = fx.out_values();
    let status = fx.status_word();
    let q_len = shape.m * shape.n_head * shape.head_dim;
    let kv_len = shape.cache_rows * shape.n_kv_head * shape.head_dim;
    let canaries = fx.bytes(OUT)[q_len * 4..].iter().all(|&b| b == OUT_FILL)
        && fx.bytes(PART)[fx.base[PART].2..]
            .iter()
            .all(|&b| b == PART_FILL)
        && match (&fx.sink_buf, sinks) {
            (Some(buffer), Some(values)) => {
                testkit::read_bytes(&fx.ctx, buffer, buffer.bytes()) == le(values)
            }
            (None, None) => true,
            _ => false,
        };
    drop(fx);
    let q: Vec<f64> = input.q[..q_len]
        .iter()
        .map(|&v| f64::from(f16_to_f32(f16_bits(v))))
        .collect();
    let half = |values: &[u16]| -> Vec<f64> {
        values[..kv_len]
            .iter()
            .map(|&v| f64::from(f16_to_f32(v)))
            .collect()
    };
    let sink_ref = sinks.map(|values| values.iter().copied().map(f64::from).collect::<Vec<_>>());
    let reference = sdpa_f64(
        &q,
        &half(&input.k),
        &half(&input.v),
        shape.m,
        shape.n_head,
        shape.cache_rows,
        shape.head_dim,
        shape.n_head / shape.n_kv_head,
        f64::from(scale),
        shape.p0 as u32,
        origin as u32,
        None,
        sink_ref.as_deref(),
    );
    Ok(Observation {
        actual,
        reference,
        bound: 2f64.powi(-9) * v_scale(&input.v[..kv_len]),
        status,
        canaries,
    })
}

#[test]
fn attn_prefill_shared_rows_match_the_portable_oracle() {
    if !testkit::device_present("attn_prefill_shared_rows_match_the_portable_oracle") {
        return;
    }
    // Dense full-prefix controls, not proof of physical W513 ring admission.
    let shapes = [
        shape(128, (2, 2), 2, 0),
        shape(128, (32, 4), 7, 511),
        shape(128, (32, 4), 17, 513),
        shape(256, (2, 2), 7, 512),
        shape(256, (16, 2), 17, 514),
        shape(128, (32, 4), 17, 1026),
        shape(128, (2, 2), 512, 0),
    ];
    for shape in shapes {
        let input = data(shape.m, shape.cache_rows, 0x51_0001, 0.2, 0.4);
        let sinks: Vec<f32> = (0..shape.n_head).map(|h| (h % 4) as f32 * 2.0).collect();
        for with_sinks in [false, true] {
            let result = observe(
                shape,
                &input,
                with_sinks.then_some(sinks.as_slice()),
                None,
                1.0 / (shape.head_dim as f32).sqrt(),
            )
            .expect("shared-row device phase");
            assert_eq!(result.status, 0);
            assert!(result.canaries);
            assert_eq!(result.actual.len(), result.reference.len());
            assert!(result.actual.iter().all(|v| v.is_finite()));
            let worst = result
                .actual
                .iter()
                .zip(&result.reference)
                .map(|(&a, &b)| (f64::from(a) - b).abs())
                .fold(0.0, f64::max);
            assert!(
                worst <= result.bound,
                "D{} Hq{} Hkv{} T{} p0{} sinks{with_sinks}: delta {worst} > {}",
                shape.head_dim,
                shape.n_head,
                shape.n_kv_head,
                shape.m,
                shape.p0,
                result.bound
            );
        }
    }
    let mut shape = shape(128, (32, 4), 17, 1026);
    shape.cache_rows = shape.m - 1;
    let input = data(shape.m, shape.cache_rows, 0x51_0002, 0.2, 0.4);
    for sinks in [None, Some(vec![2.0; shape.n_head])] {
        let result = observe(
            shape,
            &input,
            sinks.as_deref(),
            Some(AttnPrefillSegment {
                key_origin: shape.p0 as usize + 1,
                key_rows: shape.cache_rows,
                part_index: 0,
            }),
            1.0 / (shape.head_dim as f32).sqrt(),
        )
        .expect("fully masked first row device phase");
        assert_eq!(result.status, 0);
        assert!(result.canaries);
        assert!(result.actual.iter().all(|v| v.is_finite()));
        assert!(
            result.actual[..shape.n_head * shape.head_dim]
                .iter()
                .all(|&v| v == 0.0)
        );
        assert!(
            result
                .actual
                .iter()
                .zip(&result.reference)
                .all(|(&a, &b)| { (f64::from(a) - b).abs() <= result.bound })
        );
    }
}

#[test]
fn attn_prefill_none_sink_is_exact_at_extreme_negative_logits() {
    if !testkit::device_present("attn_prefill_none_sink_is_exact_at_extreme_negative_logits") {
        return;
    }
    let phase = |with_sink: bool| -> Result<Vec<f32>, EngError> {
        let shape = shape(128, (2, 2), 2, 0);
        let mut input = data(shape.m, shape.cache_rows, 0x51_0003, 0.0, 0.0);
        input.q.fill(-128.0);
        input.k.fill(f16_bits(128.0));
        input.v.fill(f16_bits(1.0));
        let split = CpfAttnSplit {
            max_splits: 1,
            min_keys: 32,
        };
        let mut fx = Fx::with_data(shape, split, false, true, &input)?;
        // Every real logit is -2^101: a fabricated -1e30 absent sink would win.
        fx.scale = 2.0f32.powi(80);
        if with_sink {
            fx.sink_buf = Some(fx.ctx.upload(&le(&vec![0.0; shape.n_head]))?);
        }
        fx.run()?;
        Ok(fx.out_values())
    };
    for with_sink in [false, true] {
        let actual = phase(with_sink).expect("extreme-logit device phase");
        let expected = if with_sink { 0.0 } else { 1.0 };
        assert!(actual.iter().all(|&value| value == expected));
    }
}

#[test]
fn attn_prefill_sink_ranges_refuse_short_unaligned_and_writable_aliases() {
    if !testkit::device_present(
        "attn_prefill_sink_ranges_refuse_short_unaligned_and_writable_aliases",
    ) {
        return;
    }
    let phase = || -> Result<Vec<bool>, EngError> {
        let shape = shape(128, (2, 2), 2, 0);
        let split = CpfAttnSplit {
            max_splits: 1,
            min_keys: 32,
        };
        let fx = Fx::new(shape, split, 0x51_0004, false, true)?;
        let bytes = shape.n_head * 4;
        let mut refused = Vec::new();
        for (buffer, offset, extent) in [
            (Q, 0, bytes - 4),
            (Q, 2, bytes),
            (OUT, 0, bytes),
            (STATUS, 0, bytes),
        ] {
            let mut call = fx.call()?;
            call.sinks = Some(fx.bufs[buffer].range(offset, extent)?);
            let result = fx.ctx.attn_prefill(&fx.stream, call);
            refused.push(matches!(result, Err(EngError::BadRequest(_))));
            fx.stream.synchronize()?;
        }
        fx.run()?;
        Ok(refused)
    };
    let refused = phase().expect("sink-range device phase");
    assert_eq!(refused, vec![true; 4]);
}

#[test]
fn attn_prefill_large_raw_sink_matches_equal_key_logits() {
    if !testkit::device_present("attn_prefill_large_raw_sink_matches_equal_key_logits") {
        return;
    }
    let shape = shape(128, (1, 1), 1, 0);
    let scale = 15.0 * 2.0f32.powi(20);
    let tie = 128.0 * scale;
    let mut observations = Vec::new();
    for (name, query, sink, expected) in [
        ("positive tie", 1.0, tie, 0.5),
        ("negative tie", -1.0, -tie, 0.5),
        ("largest finite sink", 1.0, f32::MAX, 0.0),
        ("smallest finite sink", 1.0, -f32::MAX, 1.0),
    ] {
        let mut input = data(shape.m, shape.cache_rows, 0x51_0010, 0.0, 0.0);
        input.q.fill(query);
        input.k.fill(f16_bits(1.0));
        input.v.fill(f16_bits(1.0));
        let result =
            observe(shape, &input, Some(&[sink]), None, scale).expect("large-sink device phase");
        observations.push((name, result, expected));
    }
    let mut failures = Vec::new();
    for (name, result, expected) in observations {
        assert_eq!(result.status, 0);
        assert!(result.canaries);
        assert!(result.reference.iter().all(|&v| v == expected));
        eprintln!("{name}: want {expected}, got[0] {}", result.actual[0]);
        if let Some((i, (&actual, &reference))) = result
            .actual
            .iter()
            .zip(&result.reference)
            .enumerate()
            .find(|(_, (a, b))| !a.is_finite() || (f64::from(**a) - **b).abs() > result.bound)
        {
            failures.push(format!("{name}[{i}]: want {reference}, got {actual}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

#[test]
fn attn_prefill_visible_score_overflow_is_not_an_empty_row() {
    if !testkit::device_present("attn_prefill_visible_score_overflow_is_not_an_empty_row") {
        return;
    }
    let mut observations = Vec::new();
    for (name, rows, masked_first_key, bad_first_row) in [
        ("single visible overflow", 1, false, true),
        ("only first row overflows", 2, false, true),
        ("future overflow is masked", 2, true, false),
    ] {
        let shape = shape(256, (8, 1), rows, 0);
        let mut input = data(shape.m, shape.cache_rows, 0x51_0011, 0.0, 0.0);
        input.q.fill(0.0);
        input.q[..shape.n_head * shape.head_dim].fill(-32768.0);
        input.k.fill(f16_bits(32768.0));
        if masked_first_key {
            input.k[..shape.n_kv_head * shape.head_dim].fill(f16_bits(0.0));
        }
        input.v.fill(f16_bits(1.0));
        let result = observe(shape, &input, None, None, 2.0f32.powi(94))
            .expect("visible-overflow device phase");
        observations.push((name, result, bad_first_row));
    }
    let mut failures = Vec::new();
    for (name, result, bad_first_row) in observations {
        assert_eq!(result.status, 0);
        assert!(result.canaries);
        assert!(result.reference.iter().all(|&v| v == 1.0));
        for (row, values) in result
            .actual
            .as_chunks::<{ 8 * 256 }>()
            .0
            .iter()
            .enumerate()
        {
            let bad_expected = bad_first_row && row == 0;
            let valid = if bad_expected {
                values.iter().all(|v| !v.is_finite())
            } else {
                values
                    .iter()
                    .all(|&v| v.is_finite() && (f64::from(v) - 1.0).abs() <= result.bound)
            };
            eprintln!(
                "{name} row{row}: failure expected {bad_expected}, got[0] {}",
                values[0]
            );
            if !valid {
                failures.push(format!(
                    "{name} row{row}: nonfinite expected {bad_expected}, got[0] {}",
                    values[0]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

#[test]
fn attn_prefill_visible_overflow_survives_mixed_keys_and_tiles() {
    if !testkit::device_present("attn_prefill_visible_overflow_survives_mixed_keys_and_tiles") {
        return;
    }
    let mut observations = Vec::new();
    for (name, keys, bad_start, max_splits) in [
        ("same tile bad first", 2, 0, 1),
        ("same tile bad last", 2, 1, 1),
        ("two tiles bad first one part", 64, 0, 1),
        ("two tiles bad first two parts", 64, 0, 2),
        ("two tiles bad last one part", 64, 32, 1),
        ("two tiles bad last two parts", 64, 32, 2),
    ] {
        let phase = || -> Result<(Vec<f32>, i32, bool), EngError> {
            let shape = shape(256, (8, 1), 1, keys as i64 - 1);
            let mut input = data(shape.m, keys, 0x51_0012, 0.0, 0.0);
            input.q.fill(-32768.0);
            input.k.fill(f16_bits(0.0));
            input.v.fill(f16_bits(0.0));
            let bad = bad_start * 256..(bad_start + keys / 2) * 256;
            input.k[bad.clone()].fill(f16_bits(32768.0));
            input.v[bad].fill(f16_bits(1.0));
            let split = CpfAttnSplit {
                max_splits,
                min_keys: 32,
            };
            let mut fx = Fx::with_data(shape, split, false, true, &input)?;
            fx.scale = 2.0f32.powi(94);
            fx.run()?;
            let actual = fx.out_values();
            let status = fx.status_word();
            let canaries = fx.bytes(OUT)[actual.len() * 4..]
                .iter()
                .all(|&b| b == OUT_FILL)
                && fx.bytes(PART)[fx.base[PART].2..]
                    .iter()
                    .all(|&b| b == PART_FILL);
            Ok((actual, status, canaries))
        };
        let (actual, status, canaries) = phase().expect("mixed-key overflow device phase");
        observations.push((name, actual, status, canaries));
    }
    let mut failures = Vec::new();
    for (name, actual, status, canaries) in observations {
        assert_eq!(status, 0);
        assert!(canaries);
        eprintln!("{name}: nonfinite expected, got[0] {}", actual[0]);
        if actual.iter().any(|v| v.is_finite()) {
            failures.push(format!("{name}: finite[0] {}", actual[0]));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}
