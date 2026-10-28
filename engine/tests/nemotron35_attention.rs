use inference_engine::{
    family::nemotron35::forward::attention_block_forward,
    tensor::{Dtype, Tensor},
};

fn tensor(shape: &[u32], values: &[f32]) -> Tensor {
    Tensor::try_from_host(
        Dtype::F32,
        shape.to_vec(),
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap()
}

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && (a - e).abs() < 3e-6, "{i}: {a} != {e}");
    }
}

fn identity(size: u32) -> Tensor {
    let mut data = vec![0.0; (size * size) as usize];
    for i in 0..size as usize {
        data[i * size as usize + i] = 1.0;
    }
    tensor(&[size, size], &data)
}

#[test]
fn attention_block_matches_hand_computed_causal_two_head_case() {
    // Each row has one 2: mean(x^2)=4/8=0.5. eps=0.5 and weight=1
    // make RMSNorm the identity. Q heads are +e_t and -e_t; K=e_t.
    // With head_dim=4, scores are 0 on the past and +/-0.5 on self.
    // For head 0, softmax rows are [1], [0.377540669, 0.622459331],
    // [0.274068619, 0.274068619, 0.451862762]. Head 1 has [1],
    // [0.622459331, 0.377540669], [0.383651731, 0.383651731, 0.232696538].
    // V rows are [2,0,0,2], [0,2,0,2], [0,0,2,2]. O has diagonal
    // [1,2,3,4,5,6,7,8] and O[0,7]=0.25. Add the ORIGINAL hidden row.
    let mut hidden = vec![0.0; 24];
    for t in 0..3 {
        hidden[t * 8 + t] = 2.0;
    }
    let norm = tensor(&[8], &[1.0; 8]);
    let mut q = vec![0.0; 64];
    let mut k = vec![0.0; 32];
    let mut v = vec![0.0; 32];
    let mut o = vec![0.0; 64];
    for i in 0..3 {
        q[i * 8 + i] = 0.5;
        q[(4 + i) * 8 + i] = -0.5;
        k[i * 8 + i] = 0.5;
        v[i * 8 + i] = 1.0;
        v[3 * 8 + i] = 1.0;
    }
    for i in 0..8 {
        o[i * 8 + i] = (i + 1) as f32;
    }
    o[7] = 0.25;
    let (q, k, v, o) = (
        tensor(&[8, 8], &q),
        tensor(&[4, 8], &k),
        tensor(&[4, 8], &v),
        tensor(&[8, 8], &o),
    );
    let run = |x: &[f32]| {
        attention_block_forward(&norm, &tensor(&[3, 8], x), &q, &k, &v, &o, 2, 1, 4, 0.5).unwrap()
    };
    let output = run(&hidden);
    assert_eq!(output.shape(), &[3, 8]);
    close(
        &values(&output),
        &[
            4.5, 0.0, 0.0, 8.0, 10.0, 0.0, 0.0, 16.0, 1.2550813, 4.489_837, 0.0, 8.0, 6.224_593,
            4.530488, 0.0, 16.0, 1.0481372, 1.0962745, 4.7111764, 8.0, 3.8365173, 4.603821,
            3.2577515, 16.0,
        ],
    );
    hidden[16..].fill(7.0);
    let changed = values(&run(&hidden));
    assert_eq!(&values(&output)[..16], &changed[..16]);
    assert_ne!(&values(&output)[16..], &changed[16..]);
}

#[test]
fn attention_block_gqa_matches_manually_expanded_kv_heads() {
    let norm = tensor(&[8], &[0.5, 1.0, 1.5, 2.0, 0.5, 1.0, 1.5, 2.0]);
    let hidden = tensor(
        &[3, 8],
        &(0..24).map(|i| i as f32 * 0.1 - 1.0).collect::<Vec<_>>(),
    );
    let q = identity(8);
    let o = identity(8);
    let k: Vec<f32> = (0..32).map(|i| (i % 7) as f32 * 0.1 - 0.2).collect();
    let v: Vec<f32> = (0..32).map(|i| (i % 5) as f32 * 0.2 - 0.3).collect();
    // Explicit head expansion: [K0a,K0b,K0a,K0b,K1a,K1b,K1a,K1b],
    // not alternating KV0 and KV1. Each head spans two matrix rows.
    let expand = |w: &[f32]| {
        [0, 1, 0, 1, 2, 3, 2, 3]
            .into_iter()
            .flat_map(|row| w[row * 8..(row + 1) * 8].iter().copied())
            .collect::<Vec<_>>()
    };
    let grouped = attention_block_forward(
        &norm,
        &hidden,
        &q,
        &tensor(&[4, 8], &k),
        &tensor(&[4, 8], &v),
        &o,
        4,
        2,
        2,
        1e-5,
    )
    .unwrap();
    let expanded = attention_block_forward(
        &norm,
        &hidden,
        &q,
        &tensor(&[8, 8], &expand(&k)),
        &tensor(&[8, 8], &expand(&v)),
        &o,
        4,
        4,
        2,
        1e-5,
    )
    .unwrap();
    close(&values(&grouped), &values(&expanded));
}

#[test]
fn attention_block_real_geometry_has_causal_f32_output() {
    let mut seed = 0x1234_5678u32;
    let mut random = |len: usize, scale: f32| {
        (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((seed >> 8) as f32 / 16_777_216.0 - 0.5) * scale
            })
            .collect::<Vec<_>>()
    };
    let mut hidden = random(3 * 4096, 1.0);
    let norm = tensor(&[4096], &vec![1.0; 4096]);
    let q = tensor(&[4096, 4096], &random(4096 * 4096, 0.01));
    let k = tensor(&[256, 4096], &random(256 * 4096, 0.01));
    let v = tensor(&[256, 4096], &random(256 * 4096, 0.01));
    let o = tensor(&[4096, 4096], &random(4096 * 4096, 0.01));
    let run = |x: &[f32]| {
        attention_block_forward(
            &norm,
            &tensor(&[3, 4096], x),
            &q,
            &k,
            &v,
            &o,
            32,
            2,
            128,
            1e-5,
        )
        .unwrap()
    };
    let output = run(&hidden);
    assert_eq!(output.shape(), &[3, 4096]);
    assert_eq!(output.dtype(), Dtype::F32);
    let original = values(&output);
    assert!(original.iter().all(|x| x.is_finite()));
    hidden[8192..].fill(0.75);
    let changed = values(&run(&hidden));
    assert_eq!(&original[..8192], &changed[..8192]);
    assert_ne!(&original[8192..], &changed[8192..]);
}

#[test]
fn attention_block_rms_uses_weight_not_one_plus_weight_and_eps_inside_sqrt() {
    // T=1, V=O=I: attention is exactly the normalized row regardless of Q/K.
    // [3,4], w=[2,3], eps=0.5: mean=12.5, denominator=sqrt(13).
    // Residual + norm = [3+6/sqrt(13), 4+12/sqrt(13)].
    // [0.003,0.004], eps=1e-5: mean=0.0000125; denominator=sqrt(0.0000225).
    // These small values make an omitted or misplaced epsilon observable.
    let norm = tensor(&[2], &[2.0, 3.0]);
    let zero = tensor(&[2, 2], &[0.0; 4]);
    let id = identity(2);
    for (hidden, eps, expected) in [
        ([3.0, 4.0], 0.5, [4.6641006, 7.3282013]),
        ([0.003, 0.004], 1e-5, [1.2679111, 2.533_822]),
    ] {
        let output = attention_block_forward(
            &norm,
            &tensor(&[1, 2], &hidden),
            &zero,
            &zero,
            &id,
            &id,
            1,
            1,
            2,
            eps,
        )
        .unwrap();
        close(&values(&output), &expected);
    }
}

#[test]
fn attention_block_softmax_is_stable_for_large_finite_scores() {
    let norm = tensor(&[2], &[1.0; 2]);
    let hidden = tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
    let qk = tensor(&[2, 2], &[1000.0, 0.0, 0.0, 1000.0]);
    let id = identity(2);
    // Diagonal logits are 1e6/sqrt(2); off-diagonal logits are zero.
    // exp(logit) overflows without max subtraction; the correct softmax selects self.
    let output = attention_block_forward(&norm, &hidden, &qk, &qk, &id, &id, 1, 1, 2, 0.5).unwrap();
    close(&values(&output), &[2.0, 0.0, 0.0, 2.0]);
}

#[test]
fn attention_block_rejects_invalid_geometry_and_epsilon() {
    use inference_engine::EngError;
    let norm = tensor(&[2], &[1.0; 2]);
    let hidden = tensor(&[1, 2], &[1.0, 2.0]);
    let id = identity(2);
    for (heads, kv_heads, dim, eps) in [
        (0, 1, 2, 1e-5),
        (1, 0, 2, 1e-5),
        (3, 2, 2, 1e-5),
        (1, 1, 0, 1e-5),
        (u32::MAX, 1, 2, 1e-5),
        (1, 1, 2, 0.0),
        (1, 1, 2, -1e-5),
        (1, 1, 2, f32::NAN),
        (1, 1, 2, f32::INFINITY),
    ] {
        assert!(matches!(
            attention_block_forward(
                &norm, &hidden, &id, &id, &id, &id, heads, kv_heads, dim, eps,
            ),
            Err(EngError::BadRequest(_))
        ));
    }
    for hidden in [
        tensor(&[2], &[1.0, 2.0]),
        tensor(&[0, 2], &[]),
        tensor(&[1, 0], &[]),
    ] {
        assert!(matches!(
            attention_block_forward(&norm, &hidden, &id, &id, &id, &id, 1, 1, 2, 1e-5,),
            Err(EngError::BadRequest(_))
        ));
    }
}

#[test]
fn attention_block_rejects_wrong_tensor_shapes_dtypes_and_nonfinite_values() {
    use inference_engine::EngError;
    let norm = tensor(&[2], &[1.0; 2]);
    let hidden = tensor(&[1, 2], &[1.0, 2.0]);
    let id = identity(2);
    let wrong_shape = tensor(&[4], &[1.0; 4]);
    let wrong_dtype = Tensor::try_from_host(Dtype::U32, vec![2, 2], vec![0; 16]).unwrap();
    let nonfinite = tensor(&[2, 2], &[0.0, f32::NAN, 0.0, 0.0]);
    for bad in [&wrong_shape, &wrong_dtype, &nonfinite] {
        // Each projection must validate independently, before performing any arithmetic.
        for index in 2..6 {
            let mut args = [&norm, &hidden, &id, &id, &id, &id];
            args[index] = bad;
            assert!(matches!(
                attention_block_forward(
                    args[0], args[1], args[2], args[3], args[4], args[5], 1, 1, 2, 1e-5,
                ),
                Err(EngError::BadRequest(_))
            ));
        }
    }
    for (bad_norm, bad_hidden) in [
        (tensor(&[1, 2], &[1.0; 2]), hidden.clone()),
        (norm.clone(), tensor(&[1, 2], &[f32::INFINITY, 0.0])),
        (tensor(&[2], &[1.0, f32::NAN]), hidden),
    ] {
        assert!(matches!(
            attention_block_forward(&bad_norm, &bad_hidden, &id, &id, &id, &id, 1, 1, 2, 1e-5,),
            Err(EngError::BadRequest(_))
        ));
    }
}

#[test]
fn attention_block_reports_nonfinite_intermediate_or_output() {
    use inference_engine::EngError;
    let norm = tensor(&[2], &[1.0; 2]);
    let huge = tensor(&[2, 2], &[f32::MAX; 4]);
    let id = identity(2);
    for (hidden, q, v, o) in [
        (tensor(&[1, 2], &[f32::MAX; 2]), &id, &id, &id), // RMS square overflow.
        (tensor(&[1, 2], &[1.0; 2]), &huge, &id, &id),    // QK score overflow.
        (tensor(&[1, 2], &[1.0; 2]), &id, &huge, &id),    // Value/output overflow.
        (tensor(&[1, 2], &[1.0; 2]), &id, &id, &huge),    // Output projection overflow.
    ] {
        assert!(matches!(
            attention_block_forward(&norm, &hidden, q, &id, v, o, 1, 1, 2, 1e-5,),
            Err(EngError::BadRequest(_))
        ));
    }
}
