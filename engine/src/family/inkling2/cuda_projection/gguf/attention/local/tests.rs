use super::*;
use crate::{
    family::inkling2::cuda_projection::{
        Resident,
        dense::{DensePlan, DenseStep, ResidentProjection},
        upload_tensor,
    },
    kernels::cuda::device_present,
};

#[test]
fn local_decode_updates_raw_and_processed_rows_and_refuses_stale_cursor() {
    if !device_present("Inkling CUDA chained local decode attention") {
        return;
    }
    let context = CudaContext::new().unwrap();
    let hidden = 32;
    let shape = RelativeAttentionShape {
        query_heads: 1,
        kv_heads: 1,
        relative_dims: 4,
        relative_extent: 2,
        capacity: 2,
    };
    let [_out, cache_bytes, _, _, _, _, _, _, table_bytes, ..] = shape.region_bytes().unwrap();
    let attention = Plan {
        visits: vec![Visit {
            shape,
            kv_format: crate::kernels::cuda::resident::relative_attention::RelativeKvFormat::F32,
            table_at: 0,
            table_bytes,
            cache_at: table_bytes,
            cache_bytes,
        }],
        tables: vec![0; table_bytes],
        scratch_bytes: 4096,
        local_workspace: Some(workspace(shape, None, hidden)),
        projected_rows: None,
    }
    .load(&context, false)
    .unwrap()
    .unwrap();
    let plan = DensePlan::new([128, 2 * hidden], hidden, 16).unwrap();
    let dense = DenseStep::new(&context, &plan).unwrap();
    let matrices = [128, 128, 128, 4, 1, 128].map(|rows| {
        let mut weights = vec![0.0_f32; rows * hidden];
        for (d, row) in weights.chunks_exact_mut(hidden).enumerate() {
            row[0] = (d % 5) as f32 * 0.25;
        }
        let bytes: Vec<_> = weights.into_iter().flat_map(f32::to_ne_bytes).collect();
        Resident {
            codes: upload_tensor(&context, &bytes).unwrap(),
            shape: [rows, hidden],
        }
    });
    let projection = matrices.each_ref().map(|weight| ResidentProjection {
        weight,
        dtype: DenseDtype::F32,
        input: 0,
    });
    let weights = || {
        [
            vec![0.5; 128],
            vec![1.0; 4],
            [0.5, 1.0, -1.0, 2.0].repeat(128),
            [0.0, -0.5, 0.25, 1.0].repeat(128),
        ]
    };
    let run = |pos, input: &[f32; 32], raw: [&VecDeque<Vec<f32>>; 2], weights: [Vec<f32>; 4]| {
        attention.step_local(
            &context,
            (0, pos),
            LocalPrep {
                math: PrepMath {
                    qk_limit: 0.5,
                    rel_limit: 1.0,
                    eps: 1e-5,
                    log: None,
                },
                read_weights: &|_| Ok(weights.each_ref().map(|row| Cow::Borrowed(row.as_slice()))),
                raw,
            },
            LocalProjection {
                dense: &dense,
                inputs: LocalInputs::Plain([input; 3]),
                weights: projection,
            },
            None,
        )
    };
    let empty = VecDeque::new();
    assert!(
        run(0, &[1.0; 32], [&empty; 2], weights())
            .unwrap()
            .is_none()
    );
    let zero = [0.0; HEAD_DIM];
    attention
        .step(
            &context,
            0,
            Row {
                query: &zero,
                key: &zero,
                value: &zero,
                relative: &[0.0; 4],
                sinks: &[0.0],
                gates: &zero,
            },
            History {
                position: 0,
                keys: &VecDeque::new(),
                values: &VecDeque::new(),
            },
            None,
        )
        .unwrap();
    let mut raw = [
        VecDeque::from([vec![0.5; 128]]),
        VecDeque::from([vec![-0.25; 128]]),
    ];
    for pos in 1..5 {
        let x = pos as f32;
        let result = run(pos, &[x; 32], [&raw[0], &raw[1]], weights())
            .unwrap()
            .unwrap();
        let expected_raw: Vec<_> = (0..128).map(|d| (d % 5) as f32 * 0.25 * x).collect();
        assert_eq!(result.rows[0], expected_raw);
        assert_eq!(result.rows[1], expected_raw);
        for (plane, kernel) in [[2.0, -1.0, 1.0, 0.5], [1.0, 0.25, -0.5, 0.0]]
            .into_iter()
            .enumerate()
        {
            let mut expected: Vec<_> = expected_raw
                .iter()
                .enumerate()
                .map(|(d, &x)| {
                    let mut conv = x * kernel[0];
                    for (lag, row) in raw[plane].iter().enumerate() {
                        conv += kernel[lag + 1] * row[d];
                    }
                    (conv + x) / 5.0_f32.sqrt()
                })
                .collect();
            let sum = expected.iter().map(|v| v * v).sum::<f32>();
            let scale = if plane == 0 {
                (0.5 / (sum.sqrt() + 1e-6)).min(1.0)
            } else {
                (sum / 128.0 + 1e-5).sqrt().recip()
            };
            for (v, got) in expected.iter_mut().zip(&result.rows[plane + 2]) {
                *v *= scale;
                assert!(
                    (got - *v).abs() <= 2e-6,
                    "position {pos}, plane {plane}: {got} != {v}"
                );
            }
            raw[plane].push_front(result.rows[plane].clone());
            raw[plane].truncate(3);
        }
    }
    // Adjacent visits use the mirror, so corrupt host seed weights are
    // irrelevant until a scalar step or restore invalidates that mirror.
    let mut invalid = weights();
    invalid[0][0] = f32::NAN;
    assert!(
        run(5, &[1.0; 32], [&raw[0], &raw[1]], invalid.clone())
            .unwrap()
            .is_some()
    );
    attention.invalidate_local_aux(0);
    raw[0].push_front(vec![0.0; HEAD_DIM]);
    raw[1].push_front(vec![0.0; HEAD_DIM]);
    raw[0].truncate(3);
    raw[1].truncate(3);
    assert!(run(6, &[1.0; 32], [&raw[0], &raw[1]], invalid).is_err());
    assert!(!attention.can_chain_local(0, 6));
    attention.invalidate();
    assert!(
        run(3, &[1.0; 32], [&raw[0], &raw[1]], weights())
            .unwrap()
            .is_none()
    );
}
