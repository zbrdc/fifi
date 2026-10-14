//! Stage 1 of the compile seam: one `mlx_compile`d closure around the gated delta kernel.
//!
//! Decision 0039 item 8 keeps `mlx_compile` off, and decision 0046 A5 names Q36-G2 as the
//! measurement that can ask for an amendment. The ignored test below compiles one closure that
//! replays the gated delta kernel of decision 0050 on tiny deterministic operands, applies it
//! twice, and compares both outputs bit for bit with the uncompiled
//! [`MlxKernels::gated_delta`](crate::kernels::MlxKernels::gated_delta) result. Reaching the
//! comparison means MLX traced the custom kernel; a native failure at the compiled call means it
//! refused, and the packet stops after stage 1. A Q36 result never satisfies Phase 0B or 1B.
//!
//! ```text
//! Q36_OUT_DIR=<a path that does not exist yet> \
//!   cargo test --release --features native-mlx,native-fault-qualification --lib \
//!   family::qwen36::resident::compile_probe -- --ignored --nocapture
//! ```
//!
//! Mutation check (run on the Mac, then revert exactly): in `inference_mlx_compile_probe` in
//! `engine/src/kernels/mlx/native/guard.cpp`, swap the two `mlx_vector_array_get` output indices
//! so `y` receives the state and `state_out` receives `y`. The seam's output validation must
//! reject the swapped shapes and the test must fail; if it still passes, the test is weak. The
//! timing fields come from the same code path, so the mutation also covers them.

use std::time::Instant;

use super::ops::{f32_tensor, values};
use crate::{
    kernels::MlxKernels,
    tensor::{Dtype, ExecutionTensor, Tensor},
};

/// The required `Q36_OUT_DIR`. Panics when it is unset or when the path exists, before any GPU
/// work, like the other ignored real-GPU tests.
fn out_dir() -> String {
    let out = std::env::var("Q36_OUT_DIR").unwrap_or_else(|_| panic!("set Q36_OUT_DIR"));
    assert!(
        std::fs::symlink_metadata(&out).is_err(),
        "Q36_OUT_DIR must not exist yet: {out}"
    );
    out
}

/// Deterministic pseudo-random values in `(-scale, scale)` from one multiplicative generator, so
/// a run is reproducible and no RNG crate enters the engine.
fn lcg_values(seed: u64, count: usize, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = (state >> 33) as f64 / (1u64 << 30) as f64 - 1.0;
            (unit * f64::from(scale)) as f32
        })
        .collect()
}

/// Asserts that two readbacks hold exactly equal f32 bit patterns, element by element.
fn assert_bits(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.to_bits() == e.to_bits(),
            "{what} element {i}: compiled {a} (bits {:#x}), uncompiled {e} (bits {:#x})",
            a.to_bits(),
            e.to_bits()
        );
    }
}

/// Runs the probe once and evaluates both outputs, returning the elapsed seconds of the seam call
/// alone and of the call plus the evaluation.
fn probe_once(
    kernels: &mut MlxKernels,
    q: &ExecutionTensor,
    k: &ExecutionTensor,
    v: &ExecutionTensor,
    g: &ExecutionTensor,
    beta: &ExecutionTensor,
    state: &ExecutionTensor,
) -> (f64, f64, Vec<f32>, Vec<f32>) {
    let started = Instant::now();
    let compiled = kernels
        .compile_probe_gated_delta(q, k, v, g, beta, state)
        .unwrap();
    let call_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    kernels.evaluate(&[&compiled.y, &compiled.state]).unwrap();
    let with_eval = call_seconds + started.elapsed().as_secs_f64();
    let y = values(kernels, &compiled.y).unwrap();
    let next = values(kernels, &compiled.state).unwrap();
    (call_seconds, with_eval, y, next)
}

#[test]
#[ignore = "needs the Mac GPU and a fresh Q36_OUT_DIR; run with --ignored --nocapture"]
fn gated_delta_traces_under_compile() {
    let _serial = crate::native_test_lock();
    let out = out_dir();
    // One key head of width 32 (the kernel lane width) and two value heads over it, one token.
    let (q, k, v, g, beta, state) = (
        lcg_values(0x5147_4732, 32, 0.5),
        lcg_values(0x5147_4733, 32, 0.5),
        lcg_values(0x5147_4734, 64, 1.0),
        lcg_values(0x5147_4735, 2, 1.5),
        lcg_values(0x5147_4736, 2, 0.5),
        lcg_values(0x5147_4737, 2048, 0.05),
    );
    let hosts = [
        f32_tensor(vec![1, 1, 1, 32], &q).unwrap(),
        f32_tensor(vec![1, 1, 1, 32], &k).unwrap(),
        f32_tensor(vec![1, 1, 2, 32], &v).unwrap(),
        f32_tensor(vec![1, 1, 2], &g).unwrap(),
        f32_tensor(vec![1, 1, 2], &beta).unwrap(),
        f32_tensor(vec![1, 2, 32, 32], &state).unwrap(),
    ];
    let mut kernels = MlxKernels::try_new().unwrap();
    let refs: Vec<&Tensor> = hosts.iter().collect();
    let imported: Vec<ExecutionTensor> = kernels.import_many(&refs).unwrap();
    let [rq, rk, rv, rg, rbeta, rstate] = imported.as_slice() else {
        panic!("the import returned {} tensors", imported.len());
    };
    assert_eq!(rq.dtype(), Dtype::F32, "probe operand dtype");
    // The uncompiled reference.
    let plain = kernels.gated_delta(rq, rk, rv, rg, rbeta, rstate).unwrap();
    kernels.evaluate(&[&plain.y, &plain.state]).unwrap();
    let plain_y = values(&mut kernels, &plain.y).unwrap();
    let plain_state = values(&mut kernels, &plain.state).unwrap();
    // The first compiled call compiles (a cache miss on the fixed fun id); the second hits it.
    let (first_call, first_with_eval, y1, next1) =
        probe_once(&mut kernels, rq, rk, rv, rg, rbeta, rstate);
    let (second_call, second_with_eval, y2, next2) =
        probe_once(&mut kernels, rq, rk, rv, rg, rbeta, rstate);
    assert_eq!(y1.len(), 64, "compiled y length");
    assert_eq!(next1.len(), 2048, "compiled state length");
    assert_bits(&y1, &plain_y, "first compiled y against the uncompiled y");
    assert_bits(&next1, &plain_state, "first compiled state");
    assert_bits(&y2, &plain_y, "second compiled y");
    assert_bits(&next2, &plain_state, "second compiled state");
    let stats = kernels.memory_stats().unwrap();
    println!(
        "mlx traces the gated delta custom kernel under mlx_compile: yes; both outputs are \
         bit-identical on the first and the second compiled call"
    );
    let report = serde_json::json!({
        "traced": true, "bit_identical": true,
        "dims_b_t_hk_dk_hv_dv": [1, 1, 1, 32, 2, 32],
        "first_call_seconds": first_call, "first_call_with_eval_seconds": first_with_eval,
        "second_call_seconds": second_call, "second_call_with_eval_seconds": second_with_eval,
        "fun_id": "0x0036_5147_a5a0_0001",
        "mlx_active": stats.active, "mlx_cache": stats.cache, "mlx_peak": stats.peak,
        "rss_kb": super::real::rss_kb(),
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    std::fs::create_dir(&out).expect("Q36_OUT_DIR must not exist yet");
    std::fs::write(
        std::path::Path::new(&out).join("compile-probe.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
}
