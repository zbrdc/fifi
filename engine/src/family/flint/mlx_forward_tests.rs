use super::*;
use crate::kernels::{PackedWeight, QuantizationMode, QuantizationSpec, ResidentPackedWeight};

fn packed(k: &mut MlxKernels, stacked: bool, phase: u32) -> ResidentPackedWeight {
    let experts = if stacked { 256 } else { 1 };
    let mut payload = Vec::new();
    for e in 0..experts {
        for row in 0..64 {
            for word in 0..8 {
                let mut code = 0u32;
                for lane in 0..8 {
                    code |= ((phase + e + row + word * 8 + lane) % 16) << (lane * 4);
                }
                payload.extend(code.to_le_bytes());
            }
        }
    }
    let shape = |last| {
        if stacked {
            vec![256, 64, last]
        } else {
            vec![64, last]
        }
    };
    let repeated = |value: f32| {
        (0..experts * 64)
            .flat_map(|i| {
                (((value * (1 + i / 64 % 7) as f32).to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect()
    };
    let host = PackedWeight::try_new(
        Tensor::try_from_host(Dtype::U32, shape(8), payload).unwrap(),
        Tensor::try_from_host(Dtype::Bf16, shape(1), repeated(0.125)).unwrap(),
        Some(Tensor::try_from_host(Dtype::Bf16, shape(1), repeated(0.5)).unwrap()),
        shape(64),
        QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: 4,
            group_size: 64,
        },
    )
    .unwrap();
    k.import_packed(&host).unwrap()
}

fn swiglu(k: &mut MlxKernels, stacked: bool) -> super::super::mlx_model::SwiGlu {
    super::super::mlx_model::SwiGlu {
        gate: packed(k, stacked, 1),
        up: packed(k, stacked, 3),
        down: packed(k, stacked, 5),
    }
}

fn read(k: &mut MlxKernels, x: &ExecutionTensor) -> Vec<f32> {
    let x = k.cast(x, Dtype::F32).unwrap();
    k.readback(&x)
        .unwrap()
        .host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect()
}

fn row_reference(k: &mut MlxKernels, w: &Ffn, x: &ExecutionTensor) -> ExecutionTensor {
    let Ffn::Moe {
        router,
        routed,
        shared,
    } = w
    else {
        unreachable!()
    };
    let raw = k.mm(x, router).unwrap();
    let route = mlx_math::route(k, &raw).unwrap();
    let ids = Tensor::try_from_host(
        Dtype::U32,
        vec![6],
        route.indices.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap();
    let ids = k.import_tensor(&ids).unwrap();
    let weights = import_f32(k, &[6, 1, 1], &route.weights).unwrap();
    let x = k.reshape(x, &[1, 1, 64]).unwrap();
    let gate = k.gather_qmm(&x, &routed.gate, &ids).unwrap();
    let up = k.gather_qmm(&x, &routed.up, &ids).unwrap();
    let gate = k.silu(&gate).unwrap();
    let hidden = k.binary(BinaryOp::Mul, &gate, &up).unwrap();
    let down = k.gather_qmm(&hidden, &routed.down, &ids).unwrap();
    let rows = k.reshape(&down, &[6, 64]).unwrap();
    let x = k.reshape(&x, &[1, 64]).unwrap();
    let shared = shared.apply(k, &x).unwrap();
    precision_native::combine(k, &rows, &weights, &shared).unwrap()
}

#[test]
fn batch_grouped_experts_match_row_reference_without_half_overflow() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let mut router = vec![0.0; 64 * 256];
    for e in 0..256 {
        router[e] = if e < 6 { (e as f32 - 2.5) / 8.0 } else { -4.0 };
        if e < 6 {
            router[256 + e] = ((e * 3 % 6) as f32 - 2.5) / 8.0;
        }
    }
    let router = import_f32(&mut k, &[64, 256], &router).unwrap();
    let router = k.cast(&router, Dtype::Bf16).unwrap();
    let w = Ffn::Moe {
        router,
        routed: swiglu(&mut k, true),
        shared: Box::new(swiglu(&mut k, false)),
    };
    for count in [2, 7, 17] {
        let values: Vec<_> = (0..count * 64)
            .map(|i| {
                let row = i / 64;
                let value = 8.0 + row as f32 / 2.0 + (i % 3) as f32 / 16.0;
                if i % 64 == 1 && row % 2 != 0 {
                    -value
                } else {
                    value
                }
            })
            .collect();
        let x = import_f32(&mut k, &[count, 64], &values).unwrap();
        let x = k.cast(&x, Dtype::Bf16).unwrap();
        let (actual, health) = ffn(&mut k, &w, &x).unwrap();
        assert_eq!(actual.shape(), [count, 64]);
        assert_eq!(actual.dtype(), Dtype::Bf16);
        let actual = read(&mut k, &actual);
        assert!(actual.iter().all(|v| v.is_finite()));
        assert!(actual.iter().any(|v| v.abs() > 65_504.0));
        // The token loop is an independent test oracle, never the production prefill path.
        for row in 0..count {
            let input = k.slice(&x, &[row, 0], &[row + 1, 64]).unwrap();
            let expected = row_reference(&mut k, &w, &input);
            let expected = read(&mut k, &expected);
            for (&a, &b) in actual[row as usize * 64..(row as usize + 1) * 64]
                .iter()
                .zip(&expected)
            {
                assert!((a - b).abs() <= b.abs() / 64.0 + 1.0, "{a} != {b}");
            }
        }
        assert_eq!(
            mlx_route::read_step(&mut k, None, &[health.unwrap()]).unwrap(),
            None
        );
    }
}

#[test]
fn the_chain_never_routes_multi_row_chunks() {
    // The async chain routes only one-row chunks (decode steps and
    // single-row prefill pieces): prefill-group chaining would need its own
    // memory bound and design. Under the SHIPPED DEFAULT (env unset = on)
    // one-row chunks chain and every multi-row chunk runs the per-layer-
    // fence path; the fail-safe parser's polarity is pinned host-side in
    // tokenizer::tests (a set value rolls back unless it is 1/true/on).
    assert!(
        async_chain_flagged(),
        "test requires the default-on environment (unset FIFI_FLINT_MLX_ASYNC_CHAIN)"
    );
    assert!(chain_routes(1), "rows == 1 chains under the default");
    for rows in [2u32, 3, 64, 512, 1024] {
        assert!(!chain_routes(rows), "rows = {rows} must never chain");
    }
}

#[test]
fn the_prefill_chain_arms_multi_row_chunks_only_with_the_flag() {
    // Prefill-group chaining (FIFI_FLINT_MLX_PREFILL_CHAIN, default OFF):
    // the pure routing predicate must keep every multi-row chunk on the
    // fenced path unless the prefill-chain flag arms it, and one-row
    // chunks keep chaining on the async-chain flag alone whatever the
    // prefill flag says.
    assert!(
        chain_routes_with(1, false),
        "one row chains without the prefill flag (async chain unchanged)"
    );
    assert!(chain_routes_with(1, true));
    for rows in [2u32, 3, 64, 512, 513, 1024, 4096] {
        assert!(
            !chain_routes_with(rows, false),
            "rows = {rows} must stay fenced without the flag"
        );
        assert!(
            chain_routes_with(rows, true),
            "rows = {rows} must chain when armed"
        );
    }
}

#[test]
fn the_prefill_chain_flag_stays_off_under_the_default_environment() {
    // The default test environment leaves FIFI_FLINT_MLX_PREFILL_CHAIN
    // unset; with the async chain on, every multi-row chunk therefore
    // still runs the flag-off fenced commit. (Polarity of the parser is
    // pinned in tokenizer::tests.)
    assert!(async_chain_flagged());
    assert!(!prefill_chain_flagged());
    assert!(!chain_routes(512), "multi-row stays fenced by default");
}

/// The compiled-tail routing contract (graph-cache slice 1): one-row chunks route to the
/// compiled closure exactly when the flag arms the seam, and no multi-row prefill chunk
/// ever does, whatever the flag says (their row counts would multiply the compile cache
/// keys). The flag is process-cached (`compile_layer_flagged`'s OnceLock), so the armed
/// positive control runs in the armed process: the qualification ratchet's flag-on arm is
/// the forward-level routing control (its 33/33 identity gate exercises this branch); this
/// host test pins the predicate itself.
#[test]
fn the_compiled_tail_never_routes_multi_row_chunks() {
    assert_eq!(
        compiled_tail_routes(1),
        crate::kernels::mlx::compile_layer_flagged(),
        "rows == 1 routes exactly when the flag arms the seam"
    );
    for rows in [2u32, 3, 64, 512, 1024] {
        assert!(
            !compiled_tail_routes(rows),
            "rows = {rows} must never route to the compiled tail"
        );
    }
}

/// The compiled tail is bitwise the eager tail at the decode shape (graph-cache slice 1):
/// the closure replays the same adapter ops in the same order, so the outputs must agree
/// byte for byte at BOTH the F32 fixture dtype and the served BF16 activation dtype (the
/// BF16 embedding policy plus `allow_bf16`), a cached replay with CHANGED operands stays
/// bitwise, and the second call traces nothing (the body-entry count grows only on the
/// trace — the Qwen-B1 failure mode).
#[test]
fn the_compiled_tail_is_bitwise_identical_to_the_eager_tail() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    k.allow_bf16(true);
    let dense = swiglu(&mut k, false);
    let round = |k: &mut MlxKernels, shape: &[u32], scale: f32, dtype: Dtype| {
        let values: Vec<_> = (0..shape.iter().product::<u32>())
            .map(|i| 0.25 * scale - i as f32 / 512.0)
            .collect::<Vec<_>>();
        let imported = import_f32(k, shape, &values).unwrap();
        k.cast(&imported, dtype).unwrap()
    };
    for dtype in [Dtype::F32, Dtype::Bf16] {
        let x = round(&mut k, &[1, 64], 1.0, dtype);
        let attended = round(&mut k, &[1, 64], -0.5, dtype);
        let gamma = round(&mut k, &[64], 0.75, dtype);

        // Eager arm: the exact op sequence the layer loop runs after attention, in the
        // layer loop's argument order (Add(residual, ff)).
        let residual = k.binary(BinaryOp::Add, &x, &attended).unwrap();
        let normed = k.rms_norm(&residual, &gamma, 1e-6).unwrap();
        let ff = dense.apply(&mut k, &normed).unwrap();
        let eager = k.binary(BinaryOp::Add, &residual, &ff).unwrap();

        let args = crate::kernels::mlx::native::compile::FlintTailArgs {
            x: &x,
            attended: &attended,
            gamma: &gamma,
            packed: [&dense.gate, &dense.up, &dense.down],
            hidden: 64,
            eps: 1e-6,
        };
        let before = crate::kernels::mlx::native::flint_tail_body_entries();
        let compiled = k.flint_compiled_tail(&args).unwrap();
        // A cached replay with CHANGED operands: same shapes and dtypes, new values in
        // every plain operand (x, attended, gamma) — the cross-layer reuse case, where the
        // next layer's tail replays the same closure with its own tensors. The changes
        // are non-cancelling: x2 + attended2 differs from x + attended, so a trace that
        // froze the residual operands would NOT reproduce eager2.
        let x2 = round(&mut k, &[1, 64], 1.25, dtype);
        let attended2 = round(&mut k, &[1, 64], -0.25, dtype);
        let gamma2 = round(&mut k, &[64], 1.5, dtype);
        let args2 = crate::kernels::mlx::native::compile::FlintTailArgs {
            x: &x2,
            attended: &attended2,
            gamma: &gamma2,
            packed: [&dense.gate, &dense.up, &dense.down],
            hidden: 64,
            eps: 1e-6,
        };
        let replay = k.flint_compiled_tail(&args2).unwrap();
        let entries = crate::kernels::mlx::native::flint_tail_body_entries() - before;
        assert_eq!(
            entries, 1,
            "two calls must trace exactly once (cache replay, no re-trace)"
        );

        let bytes = |k: &mut MlxKernels, t: &ExecutionTensor| {
            k.readback(t).unwrap().host_bytes().unwrap().to_vec()
        };
        assert_eq!(
            bytes(&mut k, &eager),
            bytes(&mut k, &compiled),
            "compiled must be bitwise eager at {dtype:?}"
        );
        // The changed-operand replay recomputes the eager tail for the new operands.
        let residual2 = k.binary(BinaryOp::Add, &x2, &attended2).unwrap();
        let normed2 = k.rms_norm(&residual2, &gamma2, 1e-6).unwrap();
        let ff2 = dense.apply(&mut k, &normed2).unwrap();
        let eager2 = k.binary(BinaryOp::Add, &residual2, &ff2).unwrap();
        // Sensitivity: the changed operands must actually change the result — otherwise
        // the bitwise replay oracle above is vacuous (a frozen trace would pass it).
        assert_ne!(
            bytes(&mut k, &eager),
            bytes(&mut k, &eager2),
            "changed operands must change the eager tail at {dtype:?}"
        );
        assert_eq!(
            bytes(&mut k, &eager2),
            bytes(&mut k, &replay),
            "cached replay with changed operands must be bitwise eager at {dtype:?}"
        );
    }

    // The baked key pins epsilon: a different eps is a different closure, so it must trace
    // again (two body entries for two calls), not silently reuse the eps-1e-6 tape.
    let x = round(&mut k, &[1, 64], 1.0, Dtype::F32);
    let attended = round(&mut k, &[1, 64], -0.5, Dtype::F32);
    let gamma = round(&mut k, &[64], 0.75, Dtype::F32);
    let args_eps = crate::kernels::mlx::native::compile::FlintTailArgs {
        x: &x,
        attended: &attended,
        gamma: &gamma,
        packed: [&dense.gate, &dense.up, &dense.down],
        hidden: 64,
        eps: 1e-5,
    };
    let before_eps = crate::kernels::mlx::native::flint_tail_body_entries();
    k.flint_compiled_tail(&args_eps).unwrap();
    let entries_eps = crate::kernels::mlx::native::flint_tail_body_entries() - before_eps;
    assert_eq!(
        entries_eps, 1,
        "a different eps must trace a new closure, not reuse the 1e-6 tape"
    );

    // Admission rejects a hidden width the weights cannot project before any native call:
    // x is 32-wide and hidden=32 (so the input-shape guard passes) while the packed
    // weights still project 64.
    let x32 = round(&mut k, &[1, 32], 1.0, Dtype::F32);
    let attended32 = round(&mut k, &[1, 32], -0.5, Dtype::F32);
    let gamma32 = round(&mut k, &[32], 0.75, Dtype::F32);
    let args_bad = crate::kernels::mlx::native::compile::FlintTailArgs {
        x: &x32,
        attended: &attended32,
        gamma: &gamma32,
        packed: [&dense.gate, &dense.up, &dense.down],
        hidden: 32,
        eps: 1e-6,
    };
    let err = match k.flint_compiled_tail(&args_bad) {
        Err(e) => e,
        Ok(_) => panic!("width mismatch must be rejected at admission"),
    };
    assert!(
        format!("{err:?}").contains("gate/up must project"),
        "width mismatch must be rejected at admission: {err:?}"
    );
}
