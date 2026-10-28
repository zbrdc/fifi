use super::*;

fn close(a: &[f64], b: &[f64]) {
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-12, "{x} != {y}");
    }
    assert_eq!(a.len(), b.len());
}

#[test]
fn sink_has_no_value_and_is_not_scaled_like_qk() {
    let mut p = AttentionPartial::new(2).unwrap();
    p.push(0.0, &[6.0, -3.0]).unwrap();
    p.push(0.0, &[3.0, 9.0]).unwrap();
    // Two real logits 0 and sink ln(2): denominator 1+1+2.
    close(&p.finish(2.0f64.ln()).unwrap(), &[2.25, 1.5]);
    let mut p = AttentionPartial::new(1).unwrap();
    // A scaled QK dot of ln(2), but raw sink ln(4): output = 2*9/(2+4).
    p.push(2.0f64.ln(), &[9.0]).unwrap();
    close(&p.finish(4.0f64.ln()).unwrap(), &[3.0]);
}

#[test]
fn large_sink_and_empty_partitions_remain_finite() {
    for sink in [-1000.0, 1000.0] {
        close(
            &AttentionPartial::new(2).unwrap().finish(sink).unwrap(),
            &[0.0, 0.0],
        );
    }
    let mut p = AttentionPartial::new(1).unwrap();
    p.push(-1000.0, &[17.0]).unwrap();
    close(&p.finish(1000.0).unwrap(), &[0.0]);
    let mut p = AttentionPartial::new(1).unwrap();
    p.push(1000.0, &[17.0]).unwrap();
    close(&p.finish(1000.0).unwrap(), &[8.5]);
}

#[test]
fn partitions_match_independent_dense_softmax_with_sink_once() {
    let scores: Vec<_> = (0..514)
        .map(|i| ((i * 43) % 97) as f64 / 17.0 - 3.0)
        .collect();
    let values: Vec<_> = (0..514)
        .map(|i| [i as f64 / 514.0, (i as f64).sin()])
        .collect();
    let sink: f64 = 0.75;
    let denominator: f64 = scores.iter().map(|x| x.exp()).sum::<f64>() + sink.exp();
    let oracle: Vec<f64> = (0..2)
        .map(|d| {
            scores
                .iter()
                .zip(&values)
                .map(|(s, v)| s.exp() * v[d])
                .sum::<f64>()
                / denominator
        })
        .collect();
    for size in [1, 7, 128, 512, 513, 514] {
        let mut merged = AttentionPartial::new(2).unwrap();
        for start in (0..scores.len()).step_by(size) {
            let mut part = AttentionPartial::new(2).unwrap();
            for i in start..(start + size).min(scores.len()) {
                part.push(scores[i], &values[i]).unwrap();
            }
            merged.merge(&part).unwrap();
        }
        merged.merge(&AttentionPartial::new(2).unwrap()).unwrap();
        close(&merged.finish(sink).unwrap(), &oracle);
    }
    // A sink per partition would attenuate this by 1/2, not the required 2/3.
    let mut merged = AttentionPartial::new(1).unwrap();
    for _ in 0..2 {
        let mut p = AttentionPartial::new(1).unwrap();
        p.push(0.0, &[1.0]).unwrap();
        merged.merge(&p).unwrap();
    }
    close(&merged.finish(0.0).unwrap(), &[2.0 / 3.0]);
}

#[test]
fn window_edges_include_current_and_exactly_512_predecessors() {
    for query in [511, 512, 513, 514, 1026, 4097] {
        let plan = ChunkWindow::new(513, query, 1).unwrap();
        let expected: Vec<_> = (0..=query).filter(|&key| query - key < 513).collect();
        assert_eq!(
            plan.query_range(query).unwrap().collect::<Vec<_>>(),
            expected
        );
        assert_eq!(plan.required_slots(), (query + 1).min(513));
        assert!(plan.query_range(query + 1).is_err());
    }
}

#[test]
fn chunk_append_ring_wrap_preserves_all_query_windows() {
    for start in [0, 512, 513, 514, 1025, 1538, 8193] {
        for rows in [1, 2, 7, 512, 513, 514] {
            let plan = ChunkWindow::new(513, start, rows).unwrap();
            let capacity = plan.required_slots();
            let mut ring = vec![usize::MAX; capacity];
            for pos in plan.live_range() {
                ring[plan.slot(pos, capacity).unwrap()] = pos;
            }
            for query in start..start + rows {
                let keys: Vec<_> = plan
                    .query_range(query)
                    .unwrap()
                    .map(|p| ring[plan.slot(p, capacity).unwrap()])
                    .collect();
                let oracle: Vec<_> = (0..=query).filter(|&p| query - p < 513).collect();
                assert_eq!(keys, oracle, "start={start} rows={rows} query={query}");
            }
            if start >= 512 && rows > 1 {
                assert_eq!(capacity, 513 + rows - 1);
                assert!(plan.slot(start, 513).is_err());
            }
        }
    }
    let plan = ChunkWindow::new(513, 513, 1).unwrap();
    assert_eq!(plan.slot(513, 513).unwrap(), 0);
    assert_eq!(plan.slot(512, 513).unwrap(), 512);
}

#[test]
fn invalid_attention_and_window_inputs_fail() {
    assert!(AttentionPartial::new(0).is_err());
    let mut p = AttentionPartial::new(1).unwrap();
    for score in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(p.push(score, &[1.0]).is_err());
        assert!(p.clone().finish(score).is_err());
    }
    assert!(p.push(0.0, &[f64::NAN]).is_err());
    assert!(p.push(0.0, &[]).is_err());
    assert!(p.merge(&AttentionPartial::new(2).unwrap()).is_err());
    assert!(ChunkWindow::new(0, 0, 1).is_err());
    assert!(ChunkWindow::new(513, 0, 0).is_err());
    assert!(ChunkWindow::new(513, usize::MAX, 1).is_err());
}

#[test]
fn three_projection_swiglu_and_unit_shared_coefficient() {
    // x=[2,-1], row-major gate/up matrices chosen with negative gates so
    // neither ReLU-squared nor swapping gate/up can pass. Down is asymmetric.
    let x = [2.0f64, -1.0];
    let gate_w = [[-1.0, 0.0], [1.0, 1.0]];
    let up_w = [[1.0, 1.0], [3.0, -2.0]];
    let dot = |w: [f64; 2]| w.iter().zip(x).map(|(a, b)| a * b).sum::<f64>();
    let g = gate_w.map(dot);
    let u = up_w.map(dot);
    // Reuse the existing portable matrix/projector seam for the actual projections.
    use crate::family::project::{DenseProjector, DenseWeight, Projector};
    let mut projector = DenseProjector::default();
    let matrix = |w: [[f64; 2]; 2]| {
        DenseWeight::matrix(2, 2, w.into_iter().flatten().map(|v| v as f32).collect())
    };
    let projected_gate = projector.linear(&matrix(gate_w), &[2.0, -1.0]).unwrap();
    let projected_up = projector.linear(&matrix(up_w), &[2.0, -1.0]).unwrap();
    let h = swiglu(&projected_gate, &projected_up).unwrap();
    let expected = [
        g[0] / (1.0 + (-g[0]).exp()) * u[0],
        g[1] / (1.0 + (-g[1]).exp()) * u[1],
    ];
    for (got, want) in h.iter().zip(expected) {
        assert!((*got as f64 - want).abs() < 6e-7);
    }
    let down = |h: &[f64]| [h[0] + 2.0 * h[1], -3.0 * h[0] + h[1]];
    let down_w = matrix([[1.0, 2.0], [-3.0, 1.0]]);
    let routed = projector.linear(&down_w, &h).unwrap();
    let shared = down(&expected);
    let output = [
        0.25 * routed[0] as f64 + shared[0],
        0.25 * routed[1] as f64 + shared[1],
    ];
    for (got, want) in output.iter().zip(shared.map(|v| v * 1.25)) {
        assert!((got - want).abs() < 1e-6);
    }
    assert!(swiglu(&[], &[]).is_err());
    assert!(swiglu(&[1.0], &[1.0, 2.0]).is_err());
    assert!(swiglu(&[f32::NAN], &[1.0]).is_err());
}

#[test]
fn qk_head_uses_direct_gamma_and_only_local_heads_rotate() {
    let values: Vec<_> = (0..128).map(|i| i as f32 / 64.0 - 1.0).collect();
    let gamma: Vec<_> = (0..128).map(|i| 0.5 + i as f32 / 128.0).collect();
    let divisor = (values.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / 128.0 + 1e-6).sqrt();
    let plain: Vec<_> = values
        .iter()
        .zip(&gamma)
        .map(|(&v, &g)| v as f64 * g as f64 / divisor)
        .collect();
    let full = qk_head(&values, &gamma, 1e-6, None).unwrap();
    for (got, want) in full.iter().zip(&plain) {
        assert!((*got as f64 - want).abs() < 3e-7);
    }
    for pos in [512, 513, 514, 8193] {
        let local = qk_head(&values, &gamma, 1e-6, Some(pos)).unwrap();
        for i in 0..64 {
            let angle = pos as f64 / 10000.0f64.powf((2 * i) as f64 / 128.0);
            let (s, c) = angle.sin_cos();
            // F32 theta exponent and trig can differ from the independent f64 oracle.
            assert!((local[i] as f64 - (plain[i] * c - plain[i + 64] * s)).abs() < 1e-3);
            assert!((local[i + 64] as f64 - (plain[i] * s + plain[i + 64] * c)).abs() < 1e-3);
        }
        assert_ne!(local, full);
    }
    assert!(qk_head(&values[..127], &gamma, 1e-6, None).is_err());
    assert!(qk_head(&values, &gamma, 0.0, None).is_err());
}
