//! The Mac gate tests of the compiled decode blocks: the warm-up, the compile scope, the program
//! limit, the teardown clear, the tiny-model numerics of BOTH blocks, the seam wiring, and the
//! operand refusals run with the plain suite; the real-checkpoint numerics, determinism,
//! trace-count, snapshot-interop, and memory tests are `#[ignore]`d like their siblings.
//!
//! Every test that introduces compiled programs zeroes the compile counters first (under the
//! native test lock, which serializes the native tests of one process), which also zeroes the
//! NATIVE closure-body counts: the traces a test reads are its own. The program registry needs
//! no reset — it lives in each adapter context and dies with it.
//!
//! | Variable | Meaning | Default |
//! | --- | --- | --- |
//! | `Q36_CKPT_DIR` | the pinned checkpoint directory (ignored tests) | required there |
//! | `IDS_FILE` | the fifi render dump with the 919 prompt ids (ignored tests) | required there |
//! | `REF_JSON` | the reference with the 3 `suffix_ids` and the token counts | required there |
//! | `Q36_OUT_DIR` | the output directory; must not exist, created at the run | required there |
//! | `Q36_ACTIVATION` | the decode activation dtype (`f32` or `bf16`) | `f32` |
//!
//! The exact Mac commands (placeholders only), one process per run, a fresh `Q36_OUT_DIR` each:
//!
//! ```text
//! Q36_CKPT_DIR=<CKPT_DIR> IDS_FILE=<IDS_FILE> REF_JSON=<REF_JSON> Q36_OUT_DIR=<OUT_DIR> \
//!   cargo test --release --features native-mlx --lib \
//!   family::qwen36::resident::compile_gate -- --ignored --nocapture
//! ```
//!
//! The fault-hook tests need `native-fault-qualification` and run each fault in a child process
//! (a native failure is sticky for the whole process); the plain suite runs them through the
//! feature set of the Mac mirror.
//!
//! Mutation mapping: the warm-up patch is killed by
//! `the_warm_up_traces_both_blocks_once_before_the_first_token`; the settings-ignored patch by
//! `the_chunked_prefill_never_compiles_and_a_one_token_step_does`; the executor-ignores-rules patch
//! by `rules_with_compile_true_warm_the_decoder_at_open`; the start-passes-false patch by the
//! `start` rules test; the limit patch by
//! `the_fifth_distinct_compiled_program_is_refused_before_any_native_call` and the Linux registry
//! test; the teardown-clear patch by `dropping_the_adapter_clears_the_compile_cache`; the eps patch
//! and the seam `eps_bits: 0` patch by `compiled_and_uncompiled_deltanet_blocks_agree_in_f32_tiny`
//! and `two_eps_values_and_two_activation_dtypes_trace_distinct_programs`; the clamp patch
//! (`experts - 2`) by the expert-`E-1` case of
//! `compiled_and_uncompiled_moe_blocks_agree_in_f32_tiny`; the biasless-check patch and the
//! plan-limit patch by the Linux `compile_plan` tests; the F32-operand patch by
//! `the_compiled_delta_block_refuses_non_f32_operands`.

use super::compile_logits::compare_logit_rows;
use super::compiled;
use super::decoder::{ResidentDecoder, StepMode};
use super::delta;
use super::executor::{Q36Executor, Q36Rules};
use super::moe;
use super::moe::Routing;
use super::ops::{Ctx, cast_to, dim, f32_tensor, values};
use super::weights::{RDelta, RMixer, RMoe};
use crate::EngError;
use crate::family::qwen36::host::HostShape;
use crate::family::qwen36::host_load::HostWeights;
use crate::family::qwen36::settings::Activation;
use crate::family::qwen36::tiny::{self, TINY_GQA};
use crate::family::session::StoreLimits;
use crate::family::spec::QWEN3_6_35B_A3B;
use crate::kernels::MlxKernels;
use crate::kernels::mlx::compile_stats;
use crate::kernels::mlx::native::compile::{CompileDeltaArgs, CompileMoeArgs};
use crate::tensor::{Dtype, ExecutionTensor, Tensor};

/// Deterministic pseudo-random values in `(-scale, scale)` from one multiplicative generator,
/// so a run is reproducible and no RNG crate enters the engine (the rule of `compile_probe`).
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

/// The lowest index of the largest value of `row` (the greedy rule of the decode harness).
fn argmax_lowest(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &value) in row.iter().enumerate() {
        if value > row[best] {
            best = i;
        }
    }
    best as u32
}

/// Zeroes the compile counters, the stored ones and the native closure-body counts. Every test
/// of this module that introduces compiled programs starts here, under the native test lock;
/// the program registry needs no reset because it lives in each adapter context.
fn reset_compile_state() {
    compile_stats::reset();
}

/// The host load of the four-layer tiny model (DeltaNet, attention, DeltaNet, attention, every
/// layer with a 4-expert MoE).
fn tiny_host() -> (crate::weights::StreamedCheckpoint, HostWeights) {
    let ckpt = tiny::checkpoint(&TINY_GQA);
    let host = HostWeights::load(&ckpt, &TINY_GQA, &tiny::config(&TINY_GQA)).unwrap();
    (ckpt, host)
}

/// The zero inputs of one DeltaNet block: `x` `[1, H]` in the activation dtype of `decoder`, the
/// zero recurrent state, and the zero convolution ring of its first DeltaNet layer, imported
/// with one synchronization.
fn zero_delta_inputs(
    kernels: &mut MlxKernels,
    shape: &HostShape,
    act: Dtype,
) -> Result<(ExecutionTensor, ExecutionTensor, ExecutionTensor), EngError> {
    let d = &shape.delta;
    let hosts = [
        f32_tensor(vec![1, dim(shape.hidden)?], &vec![0.0; shape.hidden])?,
        f32_tensor(
            vec![dim(d.value_heads)?, dim(d.value_dim)?, dim(d.key_dim)?],
            &vec![0.0; d.value_heads * d.value_dim * d.key_dim],
        )?,
        f32_tensor(
            vec![dim(d.conv_dim())?, dim(d.conv_kernel - 1)?],
            &vec![0.0; d.conv_dim() * (d.conv_kernel - 1)],
        )?,
    ];
    let refs: Vec<&Tensor> = hosts.iter().collect();
    let mut imported = kernels.import_many(&refs)?.into_iter();
    let mut next = || imported.next().ok_or_else(super::ops::short_import);
    let (x_zero, rec, ring) = (next()?, next()?, next()?);
    let x = if x_zero.dtype() == act {
        x_zero
    } else {
        kernels.cast(&x_zero, act)?
    };
    Ok((x, rec, ring))
}

/// The weights of the first DeltaNet layer. Takes the weights alone, so a mutable borrow of
/// the decoder's kernels can live beside it.
fn first_delta(weights: &super::weights::ResidentWeights) -> Result<&RDelta, EngError> {
    weights
        .layers
        .iter()
        .find_map(|layer| match &layer.mixer {
            RMixer::Delta(w) => Some(w),
            RMixer::Attn(_) => None,
        })
        .ok_or_else(|| EngError::BadRequest("no DeltaNet layer".into()))
}

/// The weights of the first MoE layer.
fn first_moe(weights: &super::weights::ResidentWeights) -> Result<&RMoe, EngError> {
    weights
        .layers
        .first()
        .ok_or_else(|| EngError::BadRequest("no layer".into()))?
        .as_moe()
}

/// The seam arguments of one compiled DeltaNet step on `w`, with an explicit epsilon and
/// activation dtype: the direct seam calls below drive both fields, which the decoder's own
/// path always takes from the model shape. `inputs` is the (x, rec, ring) triple.
fn delta_seam_args<'a>(
    shape: &HostShape,
    w: &'a RDelta,
    consts: &'a super::ops::Consts,
    inputs: (
        &'a ExecutionTensor,
        &'a ExecutionTensor,
        &'a ExecutionTensor,
    ),
    eps: f32,
    act: Dtype,
) -> CompileDeltaArgs<'a> {
    let (x, rec, ring) = inputs;
    let d = &shape.delta;
    CompileDeltaArgs {
        x,
        rec,
        ring,
        packed: [&w.qkv, &w.z, &w.b, &w.a, &w.out],
        plain: [
            &w.conv_taps,
            &w.conv_last,
            &w.neg_exp_a_log,
            &w.dt_bias,
            &w.norm,
        ],
        consts: [&consts.zero, &consts.qk_eps, &consts.sqrt_kd],
        hidden: dim(shape.hidden).unwrap(),
        dims: [
            dim(d.key_heads).unwrap(),
            dim(d.key_dim).unwrap(),
            dim(d.value_heads).unwrap(),
            dim(d.value_dim).unwrap(),
            dim(d.conv_dim()).unwrap(),
            dim(d.conv_kernel - 1).unwrap(),
        ],
        eps,
        act,
    }
}

/// Imports one independently allocated nonzero state or ring for a trajectory fixture.
fn delta_fixture_tensor(kernels: &mut MlxKernels, shape: Vec<u32>, seed: u64) -> ExecutionTensor {
    let count = shape.iter().map(|&n| n as usize).product();
    let data: Vec<_> = lcg_values(seed, count, 0.125)
        .into_iter()
        .map(|v| v + 0.25)
        .collect();
    kernels
        .import_tensor(&f32_tensor(shape, &data).unwrap())
        .unwrap()
}

/// Uses the existing format-level oracle for BF16 and the existing DeltaNet F32 bound.
fn assert_delta_trajectory(
    activation: Activation,
    label: &str,
    got: &[f32],
    want: &[f32],
) -> (f64, f64) {
    assert_eq!(got.len(), want.len(), "{label}: element count");
    assert_finite(label, &[got, want]);
    let absolute = max_abs_diff(got, want);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    let relative = f64::from(absolute / scale);
    if activation == Activation::F32 {
        assert!(
            absolute <= 5e-5,
            "{label}: absolute error {absolute} > 5e-5"
        );
    } else {
        let cosine = super::parity::cosine(got, want);
        assert!(
            moe_block_passes(activation, cosine, relative),
            "{label}: normalized error {relative}, cosine {cosine}"
        );
    }
    (f64::from(absolute), relative)
}

/// Distinct nonzero live states follow reordered slots through repeated shape-specialized calls.
/// These tiny predicates do not qualify model quality or physical MLX allocation behavior.
#[test]
fn compiled_delta_batches_keep_independent_live_trajectories_tiny() {
    let _serial = crate::native_test_lock();
    for activation in [Activation::F32, Activation::Bf16] {
        reset_compile_state();
        let mut max_absolute = 0.0f64;
        let mut max_relative = 0.0f64;
        let (ckpt, host) = tiny_host();
        let mut decoder = ResidentDecoder::new_with_activation(
            MlxKernels::try_new().unwrap(),
            &ckpt,
            host,
            64,
            activation,
        )
        .unwrap();
        let shape = decoder.weights.shape.clone();
        let d = &shape.delta;
        let h = shape.hidden;
        let act = decoder.weights.consts.act;
        let rec_shape = vec![
            dim(d.value_heads).unwrap(),
            dim(d.value_dim).unwrap(),
            dim(d.key_dim).unwrap(),
        ];
        let ring_shape = vec![dim(d.conv_dim()).unwrap(), dim(d.conv_kernel - 1).unwrap()];
        let w = first_delta(&decoder.weights).unwrap();
        for batch in [2u32, 4, 8] {
            let b = batch as usize;
            let fresh_states = |kernels: &mut MlxKernels| -> Vec<_> {
                (0..b)
                    .map(|slot| {
                        (
                            delta_fixture_tensor(kernels, rec_shape.clone(), 100 + slot as u64),
                            delta_fixture_tensor(kernels, ring_shape.clone(), 200 + slot as u64),
                        )
                    })
                    .collect()
            };
            let mut states = fresh_states(&mut decoder.kernels);
            let mut solo = fresh_states(&mut decoder.kernels);
            for step in 0..4 {
                // Move state identity with its input, not with its previous batch position.
                let order: Vec<_> = (0..b).map(|i| (i + step) % b).collect();
                let input: Vec<_> = order
                    .iter()
                    .flat_map(|&slot| lcg_values(1000 + 17 * step as u64 + slot as u64, h, 0.5))
                    .collect();
                let x = decoder
                    .kernels
                    .import_tensor(&f32_tensor(vec![batch, dim(h).unwrap()], &input).unwrap())
                    .unwrap();
                let x = decoder.kernels.cast(&x, act).unwrap();
                let first = &states[order[0]];
                let args = delta_seam_args(
                    &shape,
                    w,
                    &decoder.weights.consts,
                    (&x, &first.0, &first.1),
                    1e-6,
                    act,
                );
                let extra: Vec<_> = order[1..]
                    .iter()
                    .map(|&slot| (&states[slot].0, &states[slot].1))
                    .collect();
                let got = decoder.kernels.compiled_delta_rows(&args, &extra).unwrap();
                assert_eq!(got.out.shape(), [batch, dim(h).unwrap()]);
                assert_eq!(got.out.dtype(), act);
                assert_eq!(got.states.len(), b);
                let output = moe_seam_values(&mut decoder.kernels, &got.out);
                for (row, (rec, ring)) in got.states.into_iter().enumerate() {
                    let slot = order[row];
                    assert_eq!(rec.shape(), rec_shape);
                    assert_eq!(ring.shape(), ring_shape);
                    assert_eq!((rec.dtype(), ring.dtype()), (Dtype::F32, Dtype::F32));
                    let x1 = decoder
                        .kernels
                        .import_tensor(
                            &f32_tensor(vec![1, dim(h).unwrap()], &input[row * h..(row + 1) * h])
                                .unwrap(),
                        )
                        .unwrap();
                    let x1 = decoder.kernels.cast(&x1, act).unwrap();
                    let args = delta_seam_args(
                        &shape,
                        w,
                        &decoder.weights.consts,
                        (&x1, &solo[slot].0, &solo[slot].1),
                        1e-6,
                        act,
                    );
                    let control = decoder.kernels.compiled_delta_step(&args).unwrap();
                    for (role, actual, expected) in [
                        (
                            "out",
                            output[row * h..(row + 1) * h].to_vec(),
                            moe_seam_values(&mut decoder.kernels, &control.out),
                        ),
                        (
                            "rec",
                            values(&mut decoder.kernels, &rec).unwrap(),
                            values(&mut decoder.kernels, &control.rec).unwrap(),
                        ),
                        (
                            "ring",
                            values(&mut decoder.kernels, &ring).unwrap(),
                            values(&mut decoder.kernels, &control.ring).unwrap(),
                        ),
                    ] {
                        let label = format!(
                            "{activation:?} B={batch} step={step} row={row} slot={slot} {role}"
                        );
                        let (absolute, relative) =
                            assert_delta_trajectory(activation, &label, &actual, &expected);
                        max_absolute = max_absolute.max(absolute);
                        max_relative = max_relative.max(relative);
                    }
                    states[slot] = (rec, ring);
                    solo[slot] = (control.rec, control.ring);
                }
                let stats = compile_stats::read();
                assert_eq!(stats.delta.seen.len(), 1, "B is not a baked id");
                assert_eq!(
                    stats.delta.traces,
                    match batch {
                        2 => 2,
                        4 => 3,
                        8 => 4,
                        _ => unreachable!(),
                    },
                    "one B1 trace and one trace per batch shape; replay does not retrace"
                );
            }
        }
        eprintln!(
            "compiled_delta_rows activation={activation:?} max_absolute={max_absolute:.9e} max_relative={max_relative:.9e}"
        );
        drop(decoder);
    }
    reset_compile_state();
}

/// Invalid batch/count/shape/dtype/owner inputs cannot compile or consume a program id.
#[test]
fn compiled_delta_rows_refuse_invalid_operands_before_admission_tiny() {
    let _serial = crate::native_test_lock();
    reset_compile_state();
    let (ckpt, host) = tiny_host();
    let mut decoder = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host,
        64,
        Activation::F32,
    )
    .unwrap();
    let shape = decoder.weights.shape.clone();
    let w = first_delta(&decoder.weights).unwrap();
    let (x1, rec, ring) = zero_delta_inputs(&mut decoder.kernels, &shape, Dtype::F32).unwrap();
    let mut foreign = MlxKernels::try_new().unwrap();
    let (_, foreign_rec, foreign_ring) =
        zero_delta_inputs(&mut foreign, &shape, Dtype::F32).unwrap();
    let flat_rec = decoder
        .kernels
        .reshape(
            &rec,
            &[dim(shape.delta.value_heads * shape.delta.value_dim * shape.delta.key_dim).unwrap()],
        )
        .unwrap();
    let flat_ring = decoder
        .kernels
        .reshape(
            &ring,
            &[dim(shape.delta.conv_dim() * (shape.delta.conv_kernel - 1)).unwrap()],
        )
        .unwrap();
    decoder.kernels.allow_bf16(true);
    let bad_rec = decoder.kernels.cast(&rec, Dtype::Bf16).unwrap();
    let bad_ring = decoder.kernels.cast(&ring, Dtype::Bf16).unwrap();
    let x2 = decoder
        .kernels
        .import_tensor(
            &f32_tensor(
                vec![2, dim(shape.hidden).unwrap()],
                &vec![0.25; 2 * shape.hidden],
            )
            .unwrap(),
        )
        .unwrap();
    let args = delta_seam_args(
        &shape,
        w,
        &decoder.weights.consts,
        (&x2, &rec, &ring),
        1e-6,
        Dtype::F32,
    );
    for extras in [
        vec![],
        vec![(&rec, &ring); 2],
        vec![(&flat_rec, &ring)],
        vec![(&rec, &flat_ring)],
        vec![(&bad_rec, &ring)],
        vec![(&rec, &bad_ring)],
        vec![(&foreign_rec, &ring)],
        vec![(&rec, &foreign_ring)],
    ] {
        assert!(matches!(
            decoder.kernels.compiled_delta_rows(&args, &extras),
            Err(EngError::BadRequest(_))
        ));
    }
    let mut wrong_plain = args.plain;
    wrong_plain[1] = &flat_rec;
    let mut wrong_consts = args.consts;
    wrong_consts[0] = &rec;
    for invalid in [
        CompileDeltaArgs {
            plain: wrong_plain,
            ..args
        },
        CompileDeltaArgs {
            consts: wrong_consts,
            ..args
        },
    ] {
        assert!(matches!(
            decoder.kernels.compiled_delta_rows(&invalid, &[(&rec, &ring)]),
            Err(EngError::BadRequest(message)) if message.contains("plain operand shapes")
        ));
    }
    for batch in [1u32, 3, 5, 16] {
        let x = decoder
            .kernels
            .import_tensor(
                &f32_tensor(
                    vec![batch, dim(shape.hidden).unwrap()],
                    &vec![0.0; batch as usize * shape.hidden],
                )
                .unwrap(),
            )
            .unwrap();
        let args = delta_seam_args(
            &shape,
            w,
            &decoder.weights.consts,
            (&x, &rec, &ring),
            1e-6,
            Dtype::F32,
        );
        assert!(matches!(
            decoder.kernels.compiled_delta_rows(&args, &[(&rec, &ring)]),
            Err(EngError::BadRequest(_))
        ));
    }
    let stats = compile_stats::read();
    assert_eq!((stats.delta.calls, stats.delta.traces), (0, 0));
    assert!(stats.delta.seen.is_empty());
    let got = decoder
        .kernels
        .compiled_delta_rows(&args, &[(&rec, &ring)])
        .unwrap();
    assert_finite(
        "valid after refusal",
        &[&values(&mut decoder.kernels, &got.out).unwrap()],
    );
    for (rec, ring) in got.states {
        assert_finite(
            "valid states after refusal",
            &[
                &values(&mut decoder.kernels, &rec).unwrap(),
                &values(&mut decoder.kernels, &ring).unwrap(),
            ],
        );
    }
    let args = delta_seam_args(
        &shape,
        w,
        &decoder.weights.consts,
        (&x1, &rec, &ring),
        1e-6,
        Dtype::F32,
    );
    decoder.kernels.compiled_delta_step(&args).unwrap();
    drop(foreign);
    drop(decoder);
    reset_compile_state();
}

/// A one-tap ring may alias the full projection backing. Keep multirow shapes on solo until
/// a private-allocation operation is qualified; legacy B1 remains supported in both dtypes.
#[test]
fn compiled_delta_rows_refuse_one_tap_but_keep_solo_tiny() {
    let _serial = crate::native_test_lock();
    for activation in [Activation::F32, Activation::Bf16] {
        reset_compile_state();
        let (ckpt, host) = tiny_host();
        let mut decoder = ResidentDecoder::new_with_activation(
            MlxKernels::try_new().unwrap(),
            &ckpt,
            host,
            64,
            activation,
        )
        .unwrap();
        let mut shape = decoder.weights.shape.clone();
        shape.delta.conv_kernel = 2;
        let w = first_delta(&decoder.weights).unwrap();
        let conv_taps = decoder
            .kernels
            .slice(
                &w.conv_taps,
                &[0, 0],
                &[dim(shape.delta.conv_dim()).unwrap(), 1],
            )
            .unwrap();
        let act = decoder.weights.consts.act;
        let (x1, rec, ring) = zero_delta_inputs(&mut decoder.kernels, &shape, act).unwrap();
        assert_eq!(ring.shape()[1], 1);
        for batch in [2u32, 4, 8] {
            let x = decoder
                .kernels
                .import_tensor(
                    &f32_tensor(
                        vec![batch, dim(shape.hidden).unwrap()],
                        &vec![0.25; batch as usize * shape.hidden],
                    )
                    .unwrap(),
                )
                .unwrap();
            let x = decoder.kernels.cast(&x, act).unwrap();
            let mut args = delta_seam_args(
                &shape,
                w,
                &decoder.weights.consts,
                (&x, &rec, &ring),
                1e-6,
                act,
            );
            args.plain[0] = &conv_taps;
            let extras = vec![(&rec, &ring); batch as usize - 1];
            let error = decoder
                .kernels
                .compiled_delta_rows(&args, &extras)
                .err()
                .unwrap();
            assert!(matches!(error, EngError::BadRequest(m) if m.contains("taps > 1")));
            args.dims[5] = 0;
            let error = decoder
                .kernels
                .compiled_delta_rows(&args, &extras)
                .err()
                .unwrap();
            assert!(matches!(error, EngError::BadRequest(m) if m.contains("taps > 1")));
        }
        let stats = compile_stats::read();
        assert_eq!((stats.delta.calls, stats.delta.traces), (0, 0));
        assert!(stats.delta.seen.is_empty());
        let mut args = delta_seam_args(
            &shape,
            w,
            &decoder.weights.consts,
            (&x1, &rec, &ring),
            1e-6,
            act,
        );
        args.plain[0] = &conv_taps;
        let got = decoder.kernels.compiled_delta_step(&args).unwrap();
        assert_finite(
            "one-tap solo control",
            &[
                &moe_seam_values(&mut decoder.kernels, &got.out),
                &values(&mut decoder.kernels, &got.rec).unwrap(),
                &values(&mut decoder.kernels, &got.ring).unwrap(),
            ],
        );
        drop(decoder);
    }
    reset_compile_state();
}

/// The load-time warm-up traces both blocks exactly once, BEFORE the first token: a loaded
/// decoder with `compile = true` holds one trace per block kind in the production counters, and
/// the first decode step of a request replays the compiled program instead of tracing one.
/// Mutation check (the warm-up-skipped patch): with the warm-up removed, both counters read
/// zero traces here and the test fails.
#[test]
fn the_warm_up_traces_both_blocks_once_before_the_first_token() {
    let _serial = crate::native_test_lock();
    reset_compile_state();
    let (ckpt, host) = tiny_host();
    let mut decoder = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host,
        64,
        Activation::F32,
    )
    .unwrap();
    decoder.set_compile(true).unwrap();
    let stats = decoder.kernels.compile_counters();
    assert_eq!(stats.delta.traces, 1, "the delta warm-up traced once");
    assert_eq!(stats.moe.traces, 1, "the moe warm-up traced once");
    assert_eq!(stats.delta.calls, 1, "one warm-up call per block kind");
    assert_eq!(stats.moe.calls, 1, "one warm-up call per block kind");
    assert_eq!(
        stats.delta.call_seconds.len(),
        1,
        "the warm-up seconds are recorded"
    );
    // The first token of the first request adds NO trace: it replays the warmed program.
    let mut state = decoder.zero_state().unwrap();
    decoder.step(&mut state, 1, StepMode::Generate).unwrap();
    let stats = decoder.kernels.compile_counters();
    assert_eq!(
        stats.delta.traces, 1,
        "no delta trace after the first token"
    );
    assert_eq!(stats.moe.traces, 1, "no moe trace after the first token");
    assert!(
        stats.delta.calls > 1 && stats.moe.calls > 1,
        "the first token ran through the compiled blocks"
    );
    reset_compile_state();
}

/// The compile scope: a chunked prefill NEVER compiles (the chunk shapes vary), and the one-token
/// decode step DOES when the setting is on. Mutation check (the settings-ignored patch): with the
/// flag ignored, the single-token step leaves the counters where the chunk left them and the last
/// two asserts fail.
#[test]
fn the_chunked_prefill_never_compiles_and_a_one_token_step_does() {
    let _serial = crate::native_test_lock();
    reset_compile_state();
    let (ckpt, host) = tiny_host();
    let mut decoder = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host,
        64,
        Activation::F32,
    )
    .unwrap();
    decoder.set_chunk_rows(4).unwrap();
    decoder.set_compile(true).unwrap();
    let mut state = decoder.zero_state().unwrap();
    // Two whole chunks of four rows: the chunk path alone, which never compiles.
    decoder
        .prefill(&mut state, &[1, 2, 3, 4, 5, 6, 7, 0], StepMode::Prompt)
        .unwrap();
    let after_chunks = decoder.kernels.compile_counters();
    assert_eq!(
        (after_chunks.delta.traces, after_chunks.moe.traces),
        (1, 1),
        "only the load-time warm-up traced"
    );
    assert_eq!(
        (after_chunks.delta.calls, after_chunks.moe.calls),
        (1, 1),
        "the chunked prefill made no compiled call"
    );
    // One one-token step: the compiled decode path. The tiny model has two DeltaNet layers and
    // four MoE layers, and `Prompt` skips the head and the LAST layer's MoE.
    decoder.step(&mut state, 3, StepMode::Prompt).unwrap();
    let after_step = decoder.kernels.compile_counters();
    assert_eq!(
        (after_step.delta.calls, after_step.moe.calls),
        (3, 4),
        "the one-token step ran 2 compiled delta and 3 compiled moe blocks"
    );
    assert_eq!(
        (after_step.delta.traces, after_step.moe.traces),
        (1, 1),
        "the one-token step traced nothing new"
    );
    reset_compile_state();
}

/// The adapter teardown clears the calling thread's MLX compile cache and the test sees the MLX
/// SIDE of it, not only the Rust counter: after the drop, a second adapter on the same thread runs
/// the same program key, and if the clear had not emptied the thread-local cache the call would be
/// a cache HIT — no closure-body entry. The retrace (the native trace count rising again) proves
/// the cache was emptied. Mutation check (the cache-clear-removed patch): the Rust clear count
/// stays at zero and the second adapter's first call is a hit, so both asserts fail. The ORDER of
/// the clear against the stream release is not observable by any test: no query exposes it, and the
/// teardown order is documented in `kernels/mlx/native.rs` (`Context::drop`).
#[test]
fn dropping_the_adapter_clears_the_compile_cache() {
    let _serial = crate::native_test_lock();
    reset_compile_state();
    let (ckpt, host) = tiny_host();
    let mut first = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host,
        64,
        Activation::F32,
    )
    .unwrap();
    first.set_compile(true).unwrap();
    assert_eq!(
        first.kernels.compile_counters().cache_clears,
        0,
        "no clear while the adapter lives"
    );
    let traced = compile_stats::read().delta.traces;
    assert_eq!(traced, 1, "the first adapter traced its delta program once");
    drop(first);
    let stats = compile_stats::read();
    assert!(
        stats.cache_clears >= 1,
        "the teardown cleared the compile cache: {:?}",
        stats.cache_clears
    );
    // A second adapter on the same thread under the SAME program key: its warm-up retraces
    // only because the drop emptied the thread-local MLX cache.
    let host_second = HostWeights::load(&ckpt, &TINY_GQA, &tiny::config(&TINY_GQA)).unwrap();
    let mut second = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host_second,
        64,
        Activation::F32,
    )
    .unwrap();
    second.set_compile(true).unwrap();
    let retraced = compile_stats::read().delta.traces;
    assert_eq!(
        retraced,
        traced + 1,
        "the second adapter's first call retraced: the drop emptied the MLX cache"
    );
    drop(second);
    reset_compile_state();
}

/// The program limit of one adapter context: four seam calls with four distinct epsilon constants
/// register four programs of THIS adapter, and a call whose key is a FIFTH distinct one is refused
/// with `BadRequest` BEFORE any native call, so the counters do not move. The portable registry
/// logic carries its own Linux test; this one runs the refusal through the real seam. Mutation
/// check (the registry-limit patch): with the limit raised to 5, the fifth call succeeds and the
/// `unwrap_err` fails.
#[test]
fn the_fifth_distinct_compiled_program_is_refused_before_any_native_call() {
    let _serial = crate::native_test_lock();
    reset_compile_state();
    let (ckpt, host) = tiny_host();
    let mut decoder = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host,
        64,
        Activation::F32,
    )
    .unwrap();
    let shape = decoder.weights.shape.clone();
    let w = first_delta(&decoder.weights).unwrap();
    let (x, rec, ring) = zero_delta_inputs(&mut decoder.kernels, &shape, Dtype::F32).unwrap();
    let inputs = (&x, &rec, &ring);
    for eps in [1e-6f32, 2e-6, 3e-6, 4e-6] {
        let args = delta_seam_args(&shape, w, &decoder.weights.consts, inputs, eps, Dtype::F32);
        decoder.kernels.compiled_delta_step(&args).unwrap();
    }
    let warmed = compile_stats::read();
    assert_eq!(
        warmed.delta.traces, 4,
        "four distinct epsilons traced four programs (the native body count)"
    );
    let refused = decoder.kernels.compiled_delta_step(&delta_seam_args(
        &shape,
        w,
        &decoder.weights.consts,
        inputs,
        5e-6,
        Dtype::F32,
    ));
    let message = match refused {
        Err(EngError::BadRequest(text)) => text,
        Err(other) => panic!("expected BadRequest, got {other:?}"),
        Ok(_) => panic!("expected BadRequest, got a successful step"),
    };
    assert!(
        message.contains("at most 4 compiled programs per adapter"),
        "the refusal names the limit: {message}"
    );
    let stats = compile_stats::read();
    assert_eq!(
        stats.delta.calls, warmed.delta.calls,
        "the refused call reached no native call"
    );
    assert_eq!(
        stats.delta.traces, 4,
        "the refused call traced nothing (the native body count stayed)"
    );
    reset_compile_state();
}

/// Compiled-block numerics on the TINY model, DeltaNet half, plain suite: the compiled and
/// the uncompiled DeltaNet block on the same random inputs agree to at most 5e-5 in every
/// output, with equal shapes and dtypes. Mutation check (the eps-baked-wrong patch: the delta
/// closure reads sqrt_kd where qk_eps belongs, or the seam bakes `eps_bits: 0`): the RMS norm
/// epsilon of the closure no longer matches the uncompiled step and the 5e-5 bound fails.
#[test]
fn compiled_and_uncompiled_deltanet_blocks_agree_in_f32_tiny() {
    let _serial = crate::native_test_lock();
    reset_compile_state();
    assert!(
        !compiled::requested(),
        "the uncompiled arm must stay uncompiled (unset QWEN36_G2A_COMPILE)"
    );
    let (ckpt, host) = tiny_host();
    let mut decoder = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host,
        64,
        Activation::F32,
    )
    .unwrap();
    let ((p_out, p_rec, p_ring), (c_out, c_rec, c_ring)) =
        deltanet_block_pair(&mut decoder, Activation::F32, 0x5147_5155);
    assert_finite(
        "tiny f32 block outputs",
        &[&p_out, &p_rec, &p_ring, &c_out, &c_rec, &c_ring],
    );
    assert!(max_abs_diff(&p_out, &c_out) <= 5e-5, "tiny out differs");
    assert!(max_abs_diff(&p_rec, &c_rec) <= 5e-5, "tiny rec differs");
    assert!(max_abs_diff(&p_ring, &c_ring) <= 5e-5, "tiny ring differs");
    reset_compile_state();
}

/// The MoE half of the tiny-model gate, plain suite: the compiled and the uncompiled
/// device-routed MoE block agree — the output within 5e-5, the ids EXACTLY equal, the gates and
/// the health within 5e-5. Two inputs: a random row, and a row built to route to expert
/// `E - 1`, the LAST expert, so the ONE clamp of the expert ids is observable: with the clamp
/// bound at `E - 2` (the mutation) the top id becomes `E - 2`, the ids differ, and the gathered
/// experts differ, so the id assert and the output bound both fail.
#[test]
fn compiled_and_uncompiled_moe_blocks_agree_in_f32_tiny() {
    let _serial = crate::native_test_lock();
    reset_compile_state();
    assert!(
        !compiled::requested(),
        "the uncompiled arm must stay uncompiled (unset QWEN36_G2A_COMPILE)"
    );
    let (ckpt, host) = tiny_host();
    let mut decoder = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host,
        64,
        Activation::F32,
    )
    .unwrap();
    let shape = decoder.weights.shape.clone();
    let experts = u32::try_from(shape.moe.experts).unwrap();
    // The random row and the expert-`E-1` row: the last column of the router, scaled up, makes
    // expert `E - 1` the sharpest choice (its own column has the largest inner product with x).
    let random = lcg_values(0x5147_5156, shape.hidden, 0.5);
    let router = values(
        &mut decoder.kernels,
        &decoder.weights.layers[0].as_moe().unwrap().router_t,
    )
    .unwrap();
    let last_column: Vec<f32> = (0..shape.hidden)
        .map(|h| router[h * shape.moe.experts + (shape.moe.experts - 1)])
        .collect();
    let biased: Vec<f32> = last_column.iter().map(|v| 3.0 * v).collect();
    for (name, x_values) in [("random row", random), ("expert E-1 row", biased)] {
        let w = first_moe(&decoder.weights).unwrap();
        let host_row = f32_tensor(vec![1, dim(shape.hidden).unwrap()], &x_values).unwrap();
        let mut imported = decoder
            .kernels
            .import_many(&[&host_row])
            .unwrap()
            .into_iter();
        let x = imported.next().unwrap();
        let mut ctx = Ctx {
            kernels: &mut decoder.kernels,
            consts: &decoder.weights.consts,
            compile: false,
        };
        let (plain, plain_route) = moe::step_device(&mut ctx, &shape, w, &x).unwrap();
        let (compiled_run, compiled_route) =
            compiled::moe_step_device(&mut ctx, &shape, w, &x).unwrap();
        assert_same_metadata("moe output", &plain, &compiled_run);
        assert_same_metadata("moe gates", &plain_route.gates, &compiled_route.gates);
        let (plain_ids, compiled_ids) = (
            ctx.kernels.u32_to_f32(plain_route.ids.ids()).unwrap(),
            ctx.kernels.u32_to_f32(compiled_route.ids.ids()).unwrap(),
        );
        ctx.kernels
            .evaluate(&[
                &plain,
                &plain_route.gates,
                &plain_route.health,
                &compiled_run,
                &compiled_route.gates,
                &compiled_route.health,
                &plain_ids,
                &compiled_ids,
            ])
            .unwrap();
        let read =
            |kernels: &mut MlxKernels, tensor: &ExecutionTensor| values(kernels, tensor).unwrap();
        let (p_out, c_out) = (
            read(&mut decoder.kernels, &plain),
            read(&mut decoder.kernels, &compiled_run),
        );
        let (p_ids, c_ids) = (
            read(&mut decoder.kernels, &plain_ids),
            read(&mut decoder.kernels, &compiled_ids),
        );
        let (p_gates, c_gates) = (
            read(&mut decoder.kernels, &plain_route.gates),
            read(&mut decoder.kernels, &compiled_route.gates),
        );
        let (p_health, c_health) = (
            read(&mut decoder.kernels, &plain_route.health),
            read(&mut decoder.kernels, &compiled_route.health),
        );
        assert_finite(
            "tiny f32 moe outputs",
            &[&p_out, &c_out, &p_gates, &c_gates, &p_health, &c_health],
        );
        assert_eq!(p_ids, c_ids, "{name}: the chosen experts are identical");
        assert!(
            max_abs_diff(&p_out, &c_out) <= 5e-5,
            "{name}: moe out differs"
        );
        assert!(
            max_abs_diff(&p_gates, &c_gates) <= 5e-5,
            "{name}: gates differ"
        );
        assert!(
            (p_health[0] - c_health[0]).abs() <= 5e-5,
            "{name}: health {p_health:?} against {c_health:?}"
        );
        if name == "expert E-1 row" {
            assert_eq!(
                p_ids.first().copied(),
                Some((experts - 1) as f32),
                "the biased row routes its top expert to the LAST expert, so the clamp is live"
            );
        }
    }
    reset_compile_state();
}

/// The rules of the executor carry the setting: `Q36Rules { compile: true }` through the
/// `from_parts` path of `Q36Executor::open` admits the model and warms exactly one program per
/// block kind before the first token, and `compile: false` warms nothing. Mutation check (the
/// executor-ignores-rules patch): the `true` row warms nothing and the trace asserts fail.
#[test]
fn rules_with_compile_true_warm_the_decoder_at_open() {
    let _serial = crate::native_test_lock();
    reset_compile_state();
    assert!(
        !compiled::requested(),
        "the off arm reads the setting off (unset QWEN36_G2A_COMPILE)"
    );
    let ckpt: &'static crate::weights::StreamedCheckpoint =
        Box::leak(Box::new(tiny::checkpoint(&TINY_GQA)));
    let host = || HostWeights::load(ckpt, &TINY_GQA, &tiny::config(&TINY_GQA)).unwrap();
    // The cache growth of the tiny model per position: 2 attention layers of 2 key-value heads
    // of width 32, keys and values, F32 — 2 * 2 * 32 * 2 * 4 bytes.
    let rules = |compile| Q36Rules {
        prefix_store: StoreLimits::default(),
        eos: vec![],
        sampler_vocab: 8,
        context_limit: 8,
        session_cap: 8 * 2 * 2 * 32 * 2 * 4,
        activation: Activation::F32,
        kv_dtype: super::super::settings::KvDtype::F32,
        compile,
        max_seqs: 1,
    };
    let (off, session) =
        Q36Executor::from_parts(MlxKernels::try_new().unwrap(), ckpt, host(), rules(false))
            .unwrap();
    drop(session);
    let stats = off.compile_counters();
    assert_eq!(
        (
            stats.delta.calls,
            stats.moe.calls,
            stats.delta.traces,
            stats.moe.traces
        ),
        (0, 0, 0, 0),
        "compile: false warms nothing"
    );
    let (off_delta_seconds, off_moe_seconds) = off.compile_warmup_seconds();
    assert!(
        off_delta_seconds.is_empty() && off_moe_seconds.is_empty(),
        "no warm-up seconds with the setting off"
    );
    drop(off);
    let (on, session) =
        Q36Executor::from_parts(MlxKernels::try_new().unwrap(), ckpt, host(), rules(true)).unwrap();
    drop(session);
    let stats = on.compile_counters();
    assert_eq!(
        (stats.delta.calls, stats.moe.calls),
        (1, 1),
        "the uniform tiny model warms one call per block kind"
    );
    assert_eq!(
        (stats.delta.traces, stats.moe.traces),
        (1, 1),
        "the warm-up traced both block kinds once (the native body counts)"
    );
    let (delta_seconds, moe_seconds) = on.compile_warmup_seconds();
    assert_eq!(
        (delta_seconds.len(), moe_seconds.len()),
        (1, 1),
        "this engine's warm-up seconds are labeled per block kind"
    );
    drop(on);
    reset_compile_state();
}

/// Direct MoE seam arguments. The row count comes only from `x`, not from the baked key.
fn moe_seam_args<'a>(
    shape: &HostShape,
    w: &'a RMoe,
    x: &'a ExecutionTensor,
    act: Dtype,
    renorm: bool,
) -> CompileMoeArgs<'a> {
    CompileMoeArgs {
        x,
        router_t: &w.router_t,
        shared: [&w.shared_gate, &w.shared_up, &w.shared_down],
        shared_scalar_t: &w.shared_scalar_t,
        stacks: [&w.gate, &w.up, &w.down],
        hidden: dim(shape.hidden).unwrap(),
        experts: dim(shape.moe.experts).unwrap(),
        top_k: dim(shape.moe.top_k).unwrap(),
        renorm,
        act,
    }
}

/// Widens BF16 and the tiny expert ids exactly for host assertions.
fn moe_seam_values(kernels: &mut MlxKernels, tensor: &ExecutionTensor) -> Vec<f32> {
    let widened = match tensor.dtype() {
        Dtype::F32 => tensor.share(),
        Dtype::U32 => kernels.u32_to_f32(tensor).unwrap(),
        Dtype::Bf16 => kernels.cast(tensor, Dtype::F32).unwrap(),
        dtype => panic!("unexpected MoE seam dtype {dtype:?}"),
    };
    values(kernels, &widened).unwrap()
}

/// Maximum block error divided by the largest reference component: four BF16 epsilons.
const BF16_BLOCK_MAX_REL: f64 = 4.0 / 128.0;

/// Retains the cosine gate and checks block amplitude, which cosine alone cannot detect.
fn moe_block_passes(activation: Activation, cosine: f64, relative: f64) -> bool {
    super::within_mode::within_mode_logits_passes(activation, cosine, relative)
        && (activation != Activation::Bf16
            || (relative.is_finite() && relative <= BF16_BLOCK_MAX_REL))
}

/// Rejects a scaled vector and checks both sides of the exact BF16 error boundary.
#[test]
fn compiled_moe_bf16_block_oracle_rejects_scaled_outputs() {
    let reference = [1f32, -2.0, 0.5];
    let doubled = reference.map(|value| 2.0 * value);
    let cosine = super::parity::cosine(&doubled, &reference);
    let scale = reference
        .iter()
        .fold(0f32, |max, value| max.max(value.abs()));
    let relative = f64::from(max_abs_diff(&doubled, &reference) / scale);
    assert!(super::within_mode::within_mode_logits_passes(
        Activation::Bf16,
        cosine,
        relative
    ));
    assert!(!moe_block_passes(Activation::Bf16, cosine, relative));
    assert!(moe_block_passes(Activation::Bf16, 1.0, BF16_BLOCK_MAX_REL));
    let above = f64::from_bits(BF16_BLOCK_MAX_REL.to_bits() + 1);
    assert!(!moe_block_passes(Activation::Bf16, 1.0, above));
    assert!(moe_block_passes(Activation::Bf16, 1.0, 1.0 / 128.0));
    assert!(!moe_block_passes(Activation::Bf16, 1.0, f64::NAN));
    assert!(!moe_block_passes(Activation::Bf16, 0.97, 0.0));
}

/// Every supported batch and activation matches independent B=1 controls. Both normalization
/// modes reuse one baked id each across B=1/2/4/8. Repeated calls and a changed row do not trace.
#[test]
fn compiled_moe_batches_match_solo_rows_without_retrace_tiny() {
    let _serial = crate::native_test_lock();
    for activation in [Activation::F32, Activation::Bf16] {
        reset_compile_state();
        let (ckpt, host) = tiny_host();
        let mut decoder = ResidentDecoder::new_with_activation(
            MlxKernels::try_new().unwrap(),
            &ckpt,
            host,
            64,
            activation,
        )
        .unwrap();
        let shape = decoder.weights.shape.clone();
        let act = decoder.weights.consts.act;
        let w = first_moe(&decoder.weights).unwrap();
        let h = shape.hidden;
        let k = shape.moe.top_k;
        let e = shape.moe.experts;
        assert!(h >= e);
        // Identity columns give different, observable routes. The final zero row tests ties.
        let mut router = vec![0.0; h * e];
        for expert in 0..e {
            router[expert * e + expert] = 1.0;
        }
        let router = decoder
            .kernels
            .import_tensor(&f32_tensor(vec![dim(h).unwrap(), dim(e).unwrap()], &router).unwrap())
            .unwrap();
        let mut max_relative = 0.0f64;
        for renorm in [false, true] {
            for batch in [2u32, 4, 8] {
                let b = batch as usize;
                let mut input = lcg_values(0x3636 + u64::from(batch), b * h, 0.5);
                for (slot, row) in input.chunks_mut(h).enumerate() {
                    row[..e].fill(-1.0);
                    row[slot % e] = 3.0 + slot as f32 * 0.25;
                }
                input[(b - 1) * h..].fill(0.0);
                let host_x = f32_tensor(vec![batch, dim(h).unwrap()], &input).unwrap();
                let x = decoder.kernels.import_tensor(&host_x).unwrap();
                let x = decoder.kernels.cast(&x, act).unwrap();
                let mut args = moe_seam_args(&shape, w, &x, act, renorm);
                args.router_t = &router;
                let got = decoder.kernels.compiled_moe_step(&args).unwrap();
                assert_eq!(got.out.shape(), [batch, dim(h).unwrap()]);
                assert_eq!(got.out.dtype(), act);
                assert_eq!(got.ids.shape(), [batch, dim(k).unwrap()]);
                assert_eq!(got.ids.dtype(), Dtype::U32);
                assert_eq!(got.gates.shape(), [batch, dim(k).unwrap(), 1, 1]);
                assert_eq!(got.gates.dtype(), Dtype::F32);
                assert_eq!(got.health.shape(), [batch]);
                assert_eq!(got.health.dtype(), Dtype::F32);
                let out = moe_seam_values(&mut decoder.kernels, &got.out);
                let ids = moe_seam_values(&mut decoder.kernels, &got.ids);
                let gates = moe_seam_values(&mut decoder.kernels, &got.gates);
                let health = moe_seam_values(&mut decoder.kernels, &got.health);
                assert_finite("batched MoE", &[&out, &ids, &gates, &health]);
                for slot in 0..b {
                    let solo =
                        f32_tensor(vec![1, dim(h).unwrap()], &input[slot * h..(slot + 1) * h])
                            .unwrap();
                    let solo = decoder.kernels.import_tensor(&solo).unwrap();
                    let solo = decoder.kernels.cast(&solo, act).unwrap();
                    let mut args = moe_seam_args(&shape, w, &solo, act, renorm);
                    args.router_t = &router;
                    let control = decoder.kernels.compiled_moe_step(&args).unwrap();
                    assert_eq!(control.ids.shape(), [dim(k).unwrap()]);
                    assert_eq!(control.gates.shape(), [dim(k).unwrap(), 1, 1]);
                    assert_eq!(control.health.shape(), [1]);
                    let want = moe_seam_values(&mut decoder.kernels, &control.out);
                    let want_ids = moe_seam_values(&mut decoder.kernels, &control.ids);
                    let want_gates = moe_seam_values(&mut decoder.kernels, &control.gates);
                    let want_health = moe_seam_values(&mut decoder.kernels, &control.health);
                    let row = &out[slot * h..(slot + 1) * h];
                    assert_finite("solo MoE", &[&want, &want_gates, &want_health]);
                    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
                    if scale == 0.0 {
                        assert_eq!(row, want, "zero input stays zero");
                    } else {
                        let relative = f64::from(max_abs_diff(row, &want) / scale);
                        max_relative = max_relative.max(relative);
                        let cosine = super::parity::cosine(row, &want);
                        assert!(
                            moe_block_passes(activation, cosine, relative),
                            "MoE {activation:?} B={batch} slot={slot} renorm={renorm}: cosine={cosine}, relative={relative}"
                        );
                    }
                    assert_eq!(&ids[slot * k..(slot + 1) * k], want_ids);
                    super::ops::assert_close(
                        &gates[slot * k..(slot + 1) * k],
                        &want_gates,
                        5e-5,
                        "MoE gates",
                    );
                    assert!((health[slot] - want_health[0]).abs() <= 5e-5);
                }
                assert_eq!(
                    &ids[(b - 1) * k..],
                    &(0..k).map(|id| id as f32).collect::<Vec<_>>(),
                    "ties keep the lowest expert first"
                );
                assert_ne!(&out[..h], &out[h..2 * h], "rows must be distinguishable");
                let traces = compile_stats::read().moe.traces;
                // A warmed replay must keep every output, not only the shape, unchanged.
                let replay = decoder.kernels.compiled_moe_step(&args).unwrap();
                assert_eq!(moe_seam_values(&mut decoder.kernels, &replay.out), out);
                assert_eq!(moe_seam_values(&mut decoder.kernels, &replay.ids), ids);
                assert_eq!(moe_seam_values(&mut decoder.kernels, &replay.gates), gates);
                assert_eq!(
                    moe_seam_values(&mut decoder.kernels, &replay.health),
                    health
                );
                // Change only row zero. The unchanged rows must not read its activation or route.
                input[..h].fill(0.0);
                let changed = f32_tensor(vec![batch, dim(h).unwrap()], &input).unwrap();
                let changed = decoder.kernels.import_tensor(&changed).unwrap();
                let changed = decoder.kernels.cast(&changed, act).unwrap();
                args.x = &changed;
                let changed = decoder.kernels.compiled_moe_step(&args).unwrap();
                let changed_out = moe_seam_values(&mut decoder.kernels, &changed.out);
                assert_ne!(&changed_out[..h], &out[..h]);
                assert_eq!(&changed_out[h..], &out[h..]);
                assert_eq!(
                    &moe_seam_values(&mut decoder.kernels, &changed.ids)[k..],
                    &ids[k..]
                );
                assert_eq!(
                    &moe_seam_values(&mut decoder.kernels, &changed.gates)[k..],
                    &gates[k..]
                );
                assert_eq!(
                    &moe_seam_values(&mut decoder.kernels, &changed.health)[1..],
                    &health[1..]
                );
                assert_eq!(
                    compile_stats::read().moe.traces,
                    traces,
                    "warmed shape retraced"
                );
            }
        }
        eprintln!("compiled_moe_rows activation={activation:?} max_relative={max_relative:.9e}");
        let stats = compile_stats::read();
        assert_eq!(stats.moe.seen.len(), 2, "only renorm changes the baked key");
        assert_eq!(stats.moe.traces, 8, "four shape specializations per key");
        drop(decoder);
    }
    reset_compile_state();
}

/// A nonfinite row does not contaminate another row's health or output.
#[test]
fn compiled_moe_batches_keep_nonfinite_health_per_row_tiny() {
    let _serial = crate::native_test_lock();
    for activation in [Activation::F32, Activation::Bf16] {
        reset_compile_state();
        let (ckpt, host) = tiny_host();
        let mut decoder = ResidentDecoder::new_with_activation(
            MlxKernels::try_new().unwrap(),
            &ckpt,
            host,
            64,
            activation,
        )
        .unwrap();
        let shape = decoder.weights.shape.clone();
        let act = decoder.weights.consts.act;
        let w = first_moe(&decoder.weights).unwrap();
        for batch in [2u32, 4, 8] {
            for bad in [f32::NAN, f32::INFINITY] {
                let mut input = vec![0.25; batch as usize * shape.hidden];
                input[shape.hidden] = bad;
                let x = f32_tensor(vec![batch, dim(shape.hidden).unwrap()], &input).unwrap();
                let x = decoder.kernels.import_tensor(&x).unwrap();
                let x = decoder.kernels.cast(&x, act).unwrap();
                let args = moe_seam_args(&shape, w, &x, act, shape.moe.renorm);
                let got = decoder.kernels.compiled_moe_step(&args).unwrap();
                let health = moe_seam_values(&mut decoder.kernels, &got.health);
                assert_eq!(health.len(), batch as usize);
                for (row, value) in health.into_iter().enumerate() {
                    if row == 1 {
                        assert!(!value.is_finite(), "bad row health is {value}");
                    } else {
                        assert!((value - 1.0).abs() <= 5e-5, "row {row} health is {value}");
                    }
                }
                let output = moe_seam_values(&mut decoder.kernels, &got.out);
                let clean = &output[..shape.hidden];
                for (row, values) in output.chunks_exact(shape.hidden).enumerate() {
                    if row != 1 {
                        assert_finite("uncontaminated MoE row", &[values]);
                        assert_eq!(values, clean, "row {row} changed despite identical input");
                    }
                }
            }
        }
        drop(decoder);
    }
    reset_compile_state();
}

/// Unsupported batch sizes fail before native compilation or registry admission.
#[test]
fn compiled_moe_refuses_unsupported_batches_before_admission_tiny() {
    let _serial = crate::native_test_lock();
    reset_compile_state();
    let (ckpt, host) = tiny_host();
    let mut decoder = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host,
        64,
        Activation::F32,
    )
    .unwrap();
    let shape = decoder.weights.shape.clone();
    let w = first_moe(&decoder.weights).unwrap();
    for batch in [3u32, 5, 16] {
        let x = f32_tensor(
            vec![batch, dim(shape.hidden).unwrap()],
            &vec![0.0; batch as usize * shape.hidden],
        )
        .unwrap();
        let x = decoder.kernels.import_tensor(&x).unwrap();
        let args = moe_seam_args(&shape, w, &x, Dtype::F32, shape.moe.renorm);
        let error = decoder.kernels.compiled_moe_step(&args).err().unwrap();
        assert!(matches!(error, EngError::BadRequest(m) if m.contains("B in 1, 2, 4, or 8")));
    }
    let stats = compile_stats::read();
    assert_eq!((stats.moe.calls, stats.moe.traces), (0, 0));
    assert!(stats.moe.seen.is_empty());
    let x = f32_tensor(
        vec![2, dim(shape.hidden).unwrap()],
        &vec![0.25; 2 * shape.hidden],
    )
    .unwrap();
    let x = decoder.kernels.import_tensor(&x).unwrap();
    let args = moe_seam_args(&shape, w, &x, Dtype::F32, shape.moe.renorm);
    let valid = decoder.kernels.compiled_moe_step(&args).unwrap();
    assert_finite(
        "valid MoE after refusals",
        &[&moe_seam_values(&mut decoder.kernels, &valid.out)],
    );
    drop(decoder);
    reset_compile_state();
}

/// The seam wiring of the cache key: two epsilon values and the two activation dtypes give FOUR
/// distinct programs — distinct `fun_id`s in the counters and four native closure-body entries
/// (each new key misses the MLX cache and traces). Mutation checks: with `eps_bits` hard-coded to 0
/// the two epsilons share one id and the second call is a cache hit, so both the id count and the
/// native trace count stay at two per dtype; with the activation dtype dropped from the key the F32
/// and BF16 rows share ids the same way.
#[test]
fn two_eps_values_and_two_activation_dtypes_trace_distinct_programs() {
    let _serial = crate::native_test_lock();
    reset_compile_state();
    let (ckpt, host) = tiny_host();
    let mut f32_decoder = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host,
        64,
        Activation::F32,
    )
    .unwrap();
    let shape = f32_decoder.weights.shape.clone();
    let w = first_delta(&f32_decoder.weights).unwrap();
    let (x, rec, ring) = zero_delta_inputs(&mut f32_decoder.kernels, &shape, Dtype::F32).unwrap();
    let inputs = (&x, &rec, &ring);
    for eps in [1e-6f32, 2e-6] {
        let args = delta_seam_args(
            &shape,
            w,
            &f32_decoder.weights.consts,
            inputs,
            eps,
            Dtype::F32,
        );
        f32_decoder.kernels.compiled_delta_step(&args).unwrap();
    }
    let host_bf16 = HostWeights::load(&ckpt, &TINY_GQA, &tiny::config(&TINY_GQA)).unwrap();
    let mut bf16_decoder = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host_bf16,
        64,
        Activation::Bf16,
    )
    .unwrap();
    let w16 = first_delta(&bf16_decoder.weights).unwrap();
    let (x16, rec16, ring16) =
        zero_delta_inputs(&mut bf16_decoder.kernels, &shape, Dtype::Bf16).unwrap();
    let inputs16 = (&x16, &rec16, &ring16);
    for eps in [1e-6f32, 2e-6] {
        let args = delta_seam_args(
            &shape,
            w16,
            &bf16_decoder.weights.consts,
            inputs16,
            eps,
            Dtype::Bf16,
        );
        bf16_decoder.kernels.compiled_delta_step(&args).unwrap();
    }
    let stats = compile_stats::read();
    assert_eq!(
        stats.delta.traces, 4,
        "each of the four keys traced once (the native body count)"
    );
    assert_eq!(stats.delta.seen.len(), 4, "four distinct fun ids");
    let mut distinct = stats.delta.seen.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        stats.delta.seen.len(),
        "two of the four seam keys collide: {:?}",
        stats.delta.seen
    );
    reset_compile_state();
}

/// The operand refusals, through the real seam, all `BadRequest` BEFORE any native call (the
/// stored call counter stays empty): an activation dtype the adapter does not take, a layer
/// input that does not carry the activation dtype, and a BF16 recurrent state on an adapter
/// that allows BF16 — the seam still requires F32 there, like the uncompiled `gated_delta`.
/// Mutation check (the F32-operand-check patch): the BF16-state row runs without the refusal,
/// the call counter moves, and the last assert fails.
#[test]
fn the_compiled_delta_block_refuses_wrong_operands_before_any_native_call() {
    let _serial = crate::native_test_lock();
    reset_compile_state();
    let (ckpt, host) = tiny_host();
    // A BF16 decoder: it allows BF16 operands, and the seam still requires F32 for the state.
    let mut bf16_decoder = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host,
        64,
        Activation::Bf16,
    )
    .unwrap();
    let shape = bf16_decoder.weights.shape.clone();
    let w = first_delta(&bf16_decoder.weights).unwrap();
    let (x16, rec, ring) =
        zero_delta_inputs(&mut bf16_decoder.kernels, &shape, Dtype::Bf16).unwrap();
    let rec16 = bf16_decoder.kernels.cast(&rec, Dtype::Bf16).unwrap();
    let error = bf16_decoder
        .kernels
        .compiled_delta_step(&delta_seam_args(
            &shape,
            w,
            &bf16_decoder.weights.consts,
            (&x16, &rec16, &ring),
            1e-6,
            Dtype::Bf16,
        ))
        .err()
        .unwrap();
    assert!(
        matches!(&error, EngError::BadRequest(m) if m.contains("input dtype must be F32")),
        "the F32 operand refusal: {error:?}"
    );
    // The layer input must carry the activation dtype exactly: an F32 input under a BF16 step.
    let (x32, _, _) = zero_delta_inputs(&mut bf16_decoder.kernels, &shape, Dtype::F32).unwrap();
    let error = bf16_decoder
        .kernels
        .compiled_delta_step(&delta_seam_args(
            &shape,
            w,
            &bf16_decoder.weights.consts,
            (&x32, &rec, &ring),
            1e-6,
            Dtype::Bf16,
        ))
        .err()
        .unwrap();
    assert!(
        matches!(&error, EngError::BadRequest(m) if m.contains("the layer input carries F32")),
        "the activation-dtype refusal: {error:?}"
    );
    // An activation dtype the adapter does not take: a plain F32 adapter with a BF16 step.
    let host_f32 = HostWeights::load(&ckpt, &TINY_GQA, &tiny::config(&TINY_GQA)).unwrap();
    let mut f32_decoder = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host_f32,
        64,
        Activation::F32,
    )
    .unwrap();
    let w32 = first_delta(&f32_decoder.weights).unwrap();
    let (x, rec32, ring32) =
        zero_delta_inputs(&mut f32_decoder.kernels, &shape, Dtype::F32).unwrap();
    let error = f32_decoder
        .kernels
        .compiled_delta_step(&delta_seam_args(
            &shape,
            w32,
            &f32_decoder.weights.consts,
            (&x, &rec32, &ring32),
            1e-6,
            Dtype::Bf16,
        ))
        .err()
        .unwrap();
    assert!(
        matches!(&error, EngError::BadRequest(m) if m.contains("does not take Bf16 operands")),
        "the adapter refusal: {error:?}"
    );
    // No refusal reached a native call: the stored call counter stayed empty.
    let stats = compile_stats::read();
    assert_eq!(
        stats.delta.calls, 0,
        "every refusal happened before any native call"
    );
    reset_compile_state();
}

/// The weight-shape refusals, through the real seam, `BadRequest` before admission: a delta
/// projection whose logical shape disagrees with the block dimensions, and an expert stack
/// that does not hold the baked expert count. Mutation check: with the logical-shape checks
/// removed the calls run and the call counter moves, so the last assert fails.
#[test]
fn the_compiled_blocks_refuse_inconsistent_weight_shapes_before_admission() {
    let _serial = crate::native_test_lock();
    reset_compile_state();
    let (ckpt, host) = tiny_host();
    let mut decoder = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host,
        64,
        Activation::F32,
    )
    .unwrap();
    let shape = decoder.weights.shape.clone();
    let delta_w = first_delta(&decoder.weights).unwrap();
    let moe_w = first_moe(&decoder.weights).unwrap();
    let (x, rec, ring) = zero_delta_inputs(&mut decoder.kernels, &shape, Dtype::F32).unwrap();
    // The z projection slot gets the qkv weight: [CD, H] against the needed [VH*VD, H].
    let mut args = delta_seam_args(
        &shape,
        delta_w,
        &decoder.weights.consts,
        (&x, &rec, &ring),
        1e-6,
        Dtype::F32,
    );
    args.packed[1] = args.packed[0];
    let error = decoder.kernels.compiled_delta_step(&args).err().unwrap();
    assert!(
        matches!(&error, EngError::BadRequest(m) if m.contains("the z weight holds the logical shape")),
        "the delta shape refusal: {error:?}"
    );
    // The b and a slots get the z weight: [VH*VD, H] against the needed [VH, H]. The assert on the
    // shapes keeps the case honest: with VD = 1 the wrong weight would have the right shape.
    for (slot, what) in [(2usize, "b"), (3usize, "a")] {
        let mut args = delta_seam_args(
            &shape,
            delta_w,
            &decoder.weights.consts,
            (&x, &rec, &ring),
            1e-6,
            Dtype::F32,
        );
        assert_ne!(
            args.packed[1].logical_shape(),
            args.packed[slot].logical_shape(),
            "the z weight must differ in shape from the {what} weight in this model"
        );
        args.packed[slot] = args.packed[1];
        let error = decoder.kernels.compiled_delta_step(&args).err().unwrap();
        let needle = format!("the {what} weight holds the logical shape");
        assert!(
            matches!(&error, EngError::BadRequest(m) if m.contains(&needle)),
            "the delta {what} shape refusal: {error:?}"
        );
    }
    // The gate stack slot gets the shared gate weight: rank 2 against the needed [E, ., H].
    let stack_args = CompileMoeArgs {
        x: &x,
        router_t: &moe_w.router_t,
        shared: [&moe_w.shared_gate, &moe_w.shared_up, &moe_w.shared_down],
        shared_scalar_t: &moe_w.shared_scalar_t,
        stacks: [&moe_w.shared_gate, &moe_w.up, &moe_w.down],
        hidden: dim(shape.hidden).unwrap(),
        experts: dim(shape.moe.experts).unwrap(),
        top_k: dim(shape.moe.top_k).unwrap(),
        renorm: shape.moe.renorm,
        act: Dtype::F32,
    };
    let error = decoder
        .kernels
        .compiled_moe_step(&stack_args)
        .err()
        .unwrap();
    assert!(
        matches!(&error, EngError::BadRequest(m) if m.contains("the gate stack holds the logical shape")),
        "the moe stack refusal: {error:?}"
    );
    let stats = compile_stats::read();
    assert_eq!(
        (stats.delta.calls, stats.moe.calls),
        (0, 0),
        "every refusal happened before any native call"
    );
    reset_compile_state();
}

/// The host-routed decoder: its DeltaNet blocks compile (their step ignores the routing),
/// its MoE blocks NEVER do — the warm-up plans no MoE program and a one-token step leaves
/// the MoE counters untouched. Mutation check (a compile branch added to `moe::step`): the
/// MoE call counter moves after the step and the last assert fails.
#[test]
fn a_host_routed_decoder_compiles_its_delta_blocks_and_never_its_moe_blocks() {
    let _serial = crate::native_test_lock();
    reset_compile_state();
    let (ckpt, host) = tiny_host();
    let mut decoder = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host,
        64,
        Activation::F32,
    )
    .unwrap();
    decoder.set_routing(Routing::Host);
    decoder.set_compile(true).unwrap();
    let warmed = decoder.kernels.compile_counters();
    assert_eq!(
        (warmed.delta.traces, warmed.moe.traces),
        (1, 0),
        "the warm-up traced the DeltaNet program and planned no MoE program"
    );
    assert_eq!(
        (warmed.delta.calls, warmed.moe.calls),
        (1, 0),
        "no MoE call happened at load"
    );
    let mut state = decoder.zero_state().unwrap();
    decoder.step(&mut state, 1, StepMode::Generate).unwrap();
    let after = decoder.kernels.compile_counters();
    assert!(
        after.delta.calls > warmed.delta.calls,
        "the host-routed step ran its DeltaNet blocks compiled"
    );
    assert_eq!(
        after.delta.traces, 1,
        "no new delta trace after the warm-up (the native count)"
    );
    assert_eq!(
        (after.moe.calls, after.moe.traces),
        (0, 0),
        "the host-routed MoE block never compiled"
    );
    reset_compile_state();
}

/// Reads the prompt ids of the ignored real-checkpoint runs (the reader of `tool_forward`: the
/// `ids` field of `IDS_FILE` plus the 3 `suffix_ids` of `REF_JSON`).
fn real_ids() -> Vec<u32> {
    let (ids_path, ref_path) = (
        super::real::env_path("IDS_FILE"),
        super::real::env_path("REF_JSON"),
    );
    let ids_value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&ids_path).unwrap()).unwrap();
    let ref_value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&ref_path).unwrap()).unwrap();
    super::tool_forward::read_ids(&ids_value, &ref_value)
}

/// Opens the pinned real checkpoint and loads its host weights (the load of `compile_logits`).
fn real_host() -> (crate::weights::StreamedCheckpoint, HostWeights, Vec<u8>) {
    let ckpt_dir = super::real::env_path("Q36_CKPT_DIR");
    let real = super::real::open(&ckpt_dir);
    assert_eq!(
        super::parity::sha256(&real.config),
        super::parity::CONFIG_SHA256,
        "config.json differs from the pinned checkpoint"
    );
    let host = HostWeights::load(&real.ckpt, &QWEN3_6_35B_A3B, &real.config).unwrap();
    (real.ckpt, host, real.config)
}

/// The required `Q36_OUT_DIR` of the ignored runs (the rule of `compile_probe`).
fn out_dir() -> std::path::PathBuf {
    let out = super::tool_forward::out_dir_must_be_new();
    super::tool_forward::create_out_dir(&out)
}

/// The readback rows of one block run: out, rec, and ring.
type BlockRows = (Vec<f32>, Vec<f32>, Vec<f32>);

/// Runs one DeltaNet block — uncompiled through `delta::step` and compiled through the seam — on
/// the SAME random inputs and returns the six readback rows (out, rec, ring per run).
fn deltanet_block_pair(
    decoder: &mut ResidentDecoder<'_>,
    activation: Activation,
    seed: u64,
) -> (BlockRows, BlockRows) {
    let shape = decoder.weights.shape.clone();
    let w = first_delta(&decoder.weights).unwrap();
    let d = &shape.delta;
    let x_values = lcg_values(seed, shape.hidden, 0.5);
    let rec_values = lcg_values(seed + 1, d.value_heads * d.value_dim * d.key_dim, 0.05);
    let ring_values = lcg_values(seed + 2, d.conv_dim() * (d.conv_kernel - 1), 0.25);
    let hosts = [
        f32_tensor(vec![1, dim(shape.hidden).unwrap()], &x_values).unwrap(),
        f32_tensor(
            vec![
                dim(d.value_heads).unwrap(),
                dim(d.value_dim).unwrap(),
                dim(d.key_dim).unwrap(),
            ],
            &rec_values,
        )
        .unwrap(),
        f32_tensor(
            vec![dim(d.conv_dim()).unwrap(), dim(d.conv_kernel - 1).unwrap()],
            &ring_values,
        )
        .unwrap(),
    ];
    let refs: Vec<&Tensor> = hosts.iter().collect();
    let mut imported = decoder.kernels.import_many(&refs).unwrap().into_iter();
    let mut next = || imported.next().unwrap();
    let (x_zero, rec, ring) = (next(), next(), next());
    let act = activation.dtype();
    let mut ctx = Ctx {
        kernels: &mut decoder.kernels,
        consts: &decoder.weights.consts,
        compile: false,
    };
    let x = cast_to(&mut ctx, &x_zero, act).unwrap();
    let plain = delta::step(&mut ctx, &shape, w, &rec, &ring, &x).unwrap();
    let compiled_run = compiled::delta_step(&mut ctx, &shape, w, &rec, &ring, &x).unwrap();
    // Equal shapes and dtypes on all three outputs, before any readback.
    assert_same_metadata("block output", &plain.out, &compiled_run.out);
    assert_same_metadata("recurrent state", &plain.rec, &compiled_run.rec);
    assert_same_metadata("convolution ring", &plain.ring, &compiled_run.ring);
    decoder
        .kernels
        .evaluate(&[
            &plain.out,
            &plain.rec,
            &plain.ring,
            &compiled_run.out,
            &compiled_run.rec,
            &compiled_run.ring,
        ])
        .unwrap();
    // A BF16 output reads back through an exact widening to F32 (every BF16 value is an F32
    // value), because the host readback of the test helper is F32 only.
    let read = |kernels: &mut MlxKernels, tensor: &ExecutionTensor| {
        if tensor.dtype() == Dtype::F32 {
            values(kernels, tensor).unwrap()
        } else {
            let widened = kernels.cast(tensor, Dtype::F32).unwrap();
            values(kernels, &widened).unwrap()
        }
    };
    (
        (
            read(&mut decoder.kernels, &plain.out),
            read(&mut decoder.kernels, &plain.rec),
            read(&mut decoder.kernels, &plain.ring),
        ),
        (
            read(&mut decoder.kernels, &compiled_run.out),
            read(&mut decoder.kernels, &compiled_run.rec),
            read(&mut decoder.kernels, &compiled_run.ring),
        ),
    )
}

/// Asserts the metadata pair of one output: the same dtype and the same shape on both paths.
fn assert_same_metadata(what: &str, plain: &ExecutionTensor, compiled: &ExecutionTensor) {
    assert_eq!(plain.dtype(), compiled.dtype(), "{what}: dtype");
    assert_eq!(plain.shape(), compiled.shape(), "{what}: shape");
}

/// Every compared value must be finite (the test-quality rule): a NaN can slip past a
/// worst-case maximum, so the finiteness is its own assert.
fn assert_finite(what: &str, rows: &[&[f32]]) {
    for (r, row) in rows.iter().enumerate() {
        for (i, &value) in row.iter().enumerate() {
            assert!(
                value.is_finite(),
                "{what} row {r} element {i} is not finite: {value}"
            );
        }
    }
}

/// The largest absolute difference of two rows (both already asserted finite).
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// F32 numerics: the compiled and the uncompiled DeltaNet block on the same random inputs agree to
/// at most 5e-5 in every output (the measured F32 bound of the measurement runs: the logits of 8
/// greedy steps differ by at most 4.8e-5), with equal shapes and dtypes. The fusion reorders float
/// operations; it does not change the math.
#[test]
#[ignore = "needs the real checkpoint on the Mac; run with --ignored, Q36_ACTIVATION unset"]
fn compiled_and_uncompiled_deltanet_blocks_agree_in_f32() {
    let _serial = crate::native_test_lock();
    reset_compile_state();
    // The uncompiled arm reads the setting off; a leftover `QWEN36_G2A_COMPILE=1` from a
    // measurement run would silently compile it and make the comparison vacuous.
    assert!(
        !compiled::requested(),
        "run the f32 gate with QWEN36_G2A_COMPILE unset"
    );
    let activation = super::real::activation_from_env();
    assert_eq!(
        activation,
        Activation::F32,
        "run the f32 case with Q36_ACTIVATION unset"
    );
    let out = out_dir();
    let (ckpt, host, _config) = real_host();
    let mut decoder = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host,
        super::tool_forward::CONTEXT_LIMIT,
        activation,
    )
    .unwrap();
    let ((p_out, p_rec, p_ring), (c_out, c_rec, c_ring)) =
        deltanet_block_pair(&mut decoder, activation, 0x5147_4741);
    assert_finite(
        "f32 block outputs",
        &[&p_out, &p_rec, &p_ring, &c_out, &c_rec, &c_ring],
    );
    let (d_out, d_rec, d_ring) = (
        max_abs_diff(&p_out, &c_out),
        max_abs_diff(&p_rec, &c_rec),
        max_abs_diff(&p_ring, &c_ring),
    );
    assert!(d_out <= 5e-5, "compiled out differs by {d_out}");
    assert!(d_rec <= 5e-5, "compiled rec differs by {d_rec}");
    assert!(d_ring <= 5e-5, "compiled ring differs by {d_ring}");
    // The comparison helper of the logits gate, reused on the block output row: the metrics it prints carry
    // the same information as the three bounds above.
    let compared = compare_logit_rows(&p_out, &c_out, p_out.len()).unwrap();
    assert!(
        compared.bits_equal || compared.max_abs_diff <= 5e-5,
        "compare_logit_rows: {:?}",
        compared.max_abs_diff
    );
    let report = serde_json::json!({
        "test": "compiled_and_uncompiled_deltanet_blocks_agree_in_f32",
        "max_abs_diff": {"out": d_out, "rec": d_rec, "ring": d_ring},
        "compare_logit_rows": {
            "bits_equal": compared.bits_equal,
            "max_abs_diff": compared.max_abs_diff,
            "top5_overlap": compared.top5_overlap,
            "max_kl": compared.kl_divergence.iter().cloned().fold(0f64, f64::max),
        },
        "compile_counters": compiled::report(&decoder.kernels)["compile_counters"].clone(),
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    std::fs::write(
        out.join("compile-delta-f32.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    reset_compile_state();
}

/// BF16 numerics: the format-level result, through `compare_logit_rows`. NO elementwise BF16 bound
/// is claimed on the block output: the out projection sums `vh * vd` (2048 on the real model)
/// products, so an element that lands near zero by CANCELLATION can differ by many BF16 ulps *of
/// itself* while both paths are valid roundings of the same F32 value — a `max(|a|, |b|) * 2^-7`
/// bound on every element is not derived and would false-fail. What the format level guarantees and
/// this test asserts: the greedy choice is the same (the top-1 entry of the row is equal), and the
/// probability mass behind it agrees — with every logit within about one BF16 ulp (2^-7 relative)
/// of its partner, `|ln(p_i/q_i)| <= 2 * max|a - b| <= 2 * 2^-7 * max|row|`, so the per-row KL
/// divergence stays far below the 2^-2 headroom asserted here (cancelling elements carry only their
/// own small probability mass). The elementwise maxima of out, rec, and ring are printed as
/// information, not asserted: the F32 state under BF16 operands carries the same cancellation
/// caveat, so its bound is stated against the row scale, not the element.
#[test]
#[ignore = "needs the real checkpoint on the Mac; run with --ignored and Q36_ACTIVATION=bf16"]
fn compiled_and_uncompiled_deltanet_blocks_agree_at_the_bf16_format_level() {
    let _serial = crate::native_test_lock();
    reset_compile_state();
    // The uncompiled arm reads the setting off; a leftover `QWEN36_G2A_COMPILE=1` from a
    // measurement run would compile both arms and make the comparison vacuous.
    assert!(
        !compiled::requested(),
        "run the bf16 gate with QWEN36_G2A_COMPILE unset"
    );
    let activation = super::real::activation_from_env();
    assert_eq!(
        activation,
        Activation::Bf16,
        "run the bf16 case with Q36_ACTIVATION=bf16"
    );
    let out = out_dir();
    let (ckpt, host, _config) = real_host();
    let mut decoder = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host,
        super::tool_forward::CONTEXT_LIMIT,
        activation,
    )
    .unwrap();
    let ((p_out, p_rec, p_ring), (c_out, c_rec, c_ring)) =
        deltanet_block_pair(&mut decoder, activation, 0x5147_4742);
    assert_finite(
        "bf16 block outputs",
        &[&p_out, &p_rec, &p_ring, &c_out, &c_rec, &c_ring],
    );
    // The elementwise maxima are INFORMATION: no elementwise BF16 bound is claimed, because an
    // element that cancels can differ by many ulps of itself (see the test doc).
    let mut worst_out = 0f32;
    let mut differing = 0usize;
    for (&a, &b) in p_out.iter().zip(&c_out) {
        worst_out = worst_out.max((a - b).abs());
        differing += usize::from(a.to_bits() != b.to_bits());
    }
    // The format-level result: the greedy choice of the row is the same on both paths.
    assert_eq!(
        argmax_lowest(&p_out),
        argmax_lowest(&c_out),
        "the top-1 entry of the block output flipped"
    );
    // The probability mass behind it: the per-row KL divergence of the softmaxes, with the
    // derivation of the test doc (about one BF16 ulp per logit gives |ln p/q| far below 1).
    let compared = compare_logit_rows(&p_out, &c_out, p_out.len()).unwrap();
    for (row, kl) in compared.kl_divergence.iter().enumerate() {
        assert!(
            *kl <= 2f64.powi(-2),
            "row {row}: the softmax mass diverges by KL {kl}, above the 2^-2 headroom"
        );
    }
    // The F32 state and ring under BF16 operands: the bound is stated against the ROW SCALE,
    // because an element that cancels is bounded by the operand-level ulp of the row, not of
    // itself (each gated-delta operand shifts by about one BF16 ulp where it rounded).
    let scale = |name: &str, a: &[f32], b: &[f32]| {
        let row_scale = a.iter().chain(b).fold(0f32, |m, v| m.max(v.abs()));
        let mut worst = 0f32;
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() <= x.abs().max(y.abs()) * 2f32.powi(-6) + row_scale * 2f32.powi(-7),
                "{name} element {i}: {x} against {y} exceeds the row-scale bound"
            );
            worst = worst.max((x - y).abs());
        }
        worst
    };
    let (worst_rec, worst_ring) = (
        scale("rec", &p_rec, &c_rec),
        scale("ring", &p_ring, &c_ring),
    );
    let report = serde_json::json!({
        "test": "compiled_and_uncompiled_deltanet_blocks_agree_at_the_bf16_format_level",
        "out_elements": p_out.len(),
        "out_elements_differing": differing,
        "out_worst_abs": worst_out,
        "rec_worst_relative": worst_rec,
        "ring_worst_relative": worst_ring,
        "asserted": {
            "out": "the top-1 entry is equal and the per-row KL divergence is at most 2^-2",
            "rec_and_ring": "max(|a|,|b|) * 2^-6 + row_scale * 2^-7 (cancellation-robust)",
        },
        "note": "no elementwise BF16 bound on out: a cancelling element of the vh*vd-term sum \
                 can differ by many ulps of itself while both paths are valid roundings",
        "compare_logit_rows": {
            "bits_equal": compared.bits_equal,
            "max_abs_diff": compared.max_abs_diff,
            "top5_overlap": compared.top5_overlap,
            "max_kl": compared.kl_divergence.iter().cloned().fold(0f64, f64::max),
        },
        "compile_counters": compiled::report(&decoder.kernels)["compile_counters"].clone(),
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    std::fs::write(
        out.join("compile-delta-bf16.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    reset_compile_state();
}

/// Determinism: two compiled runs of the same 8 decode steps return IDENTICAL logit bits — compile
/// is deterministic run to run.
#[test]
#[ignore = "needs the real checkpoint on the Mac; run with --ignored"]
fn two_compiled_runs_of_the_same_steps_return_identical_logit_bits() {
    let _serial = crate::native_test_lock();
    reset_compile_state();
    let activation = super::real::activation_from_env();
    let out = out_dir();
    let ids = real_ids();
    let (ckpt, host, _config) = real_host();
    let mut decoder = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host,
        super::tool_forward::CONTEXT_LIMIT,
        activation,
    )
    .unwrap();
    decoder.set_compile(true).unwrap();
    decoder
        .set_chunk_rows(super::tool_forward::serve_chunk_setting(activation))
        .unwrap();
    let mut state = decoder.zero_state().unwrap();
    let chunk = decoder.chunk_rows as usize;
    for rows in ids[..921].chunks(chunk.max(1)) {
        if let &[token] = rows {
            decoder.step(&mut state, token, StepMode::Prompt).unwrap();
        } else {
            decoder.prefill(&mut state, rows, StepMode::Prompt).unwrap();
        }
    }
    // One snapshot, two identical runs from it: the snapshot restores the exact committed state,
    // so any bit difference of the two runs is a nondeterminism of the compiled path.
    let snapshot = decoder.snapshot(&state).unwrap();
    let run = |decoder: &mut ResidentDecoder<'_>| -> Vec<Vec<f32>> {
        let mut state = decoder.zero_state().unwrap();
        decoder.restore(&mut state, &snapshot).unwrap();
        let mut token = ids[921];
        let mut rows = Vec::with_capacity(8);
        for _ in 0..8 {
            let got = decoder.step(&mut state, token, StepMode::Logits).unwrap();
            let row = got.logits.expect("the logits step returns the head row");
            token = argmax_lowest(&row);
            rows.push(row);
        }
        rows
    };
    let (first, second) = (run(&mut decoder), run(&mut decoder));
    assert_eq!(first.len(), 8, "eight logits rows per run");
    let mut first_difference: Option<(usize, usize, f32, f32)> = None;
    for (step, (a, b)) in first.iter().zip(&second).enumerate() {
        assert_finite("determinism rows", &[a, b]);
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            if x.to_bits() != y.to_bits() && first_difference.is_none() {
                first_difference = Some((step, i, *x, *y));
            }
        }
    }
    assert!(
        first_difference.is_none(),
        "two compiled runs differ: {first_difference:?}"
    );
    let stats = decoder.kernels.compile_counters();
    // The runs above must have run THROUGH the compiled blocks: with the setting silently off
    // the bit comparison still passes, so the compiled call count is its own assert.
    assert!(
        stats.delta.calls > 1 && stats.moe.calls > 1,
        "the runs made {} delta and {} moe compiled calls",
        stats.delta.calls,
        stats.moe.calls
    );
    assert_eq!(
        stats.delta.traces, 1,
        "one delta trace at load, none during the runs"
    );
    let (calls, traces, _hits) = stats.totals();
    let report = serde_json::json!({
        "test": "two_compiled_runs_of_the_same_steps_return_identical_logit_bits",
        "activation_dtype": crate::family::qwen36::settings::activation_name(activation),
        "steps": 8, "bits_identical": true,
        "compile_counters": compiled::report(&decoder.kernels)["compile_counters"].clone(),
        "calls": calls, "traces": traces,
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    std::fs::write(
        out.join("compile-determinism.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    drop(snapshot);
    reset_compile_state();
}

/// Trace count: 2,000 compiled decode steps add NO trace after the first token. The trace counts
/// are the NATIVE closure-body counts: a body entry means MLX traced (or returned the uncompiled
/// function), so a silent retrace under the same Rust key — a shape or dtype drift,
/// `MLX_DISABLE_COMPILE`, a second thread's own cache — would raise `traces` and fail this test,
/// not pass vacuously.
#[test]
#[ignore = "needs the real checkpoint on the Mac; run with --ignored (about a minute of decode)"]
fn two_thousand_compiled_decode_steps_add_no_trace_after_the_first_token() {
    let _serial = crate::native_test_lock();
    reset_compile_state();
    let activation = super::real::activation_from_env();
    let out = out_dir();
    let ids = real_ids();
    let (ckpt, host, _config) = real_host();
    let mut decoder = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host,
        super::tool_forward::CONTEXT_LIMIT,
        activation,
    )
    .unwrap();
    decoder.set_compile(true).unwrap();
    decoder
        .set_chunk_rows(super::tool_forward::serve_chunk_setting(activation))
        .unwrap();
    let after_load = decoder.kernels.compile_counters();
    assert_eq!(
        (after_load.delta.traces, after_load.moe.traces),
        (1, 1),
        "the load-time warm-up traced both blocks once"
    );
    let mut state = decoder.zero_state().unwrap();
    let chunk = decoder.chunk_rows as usize;
    for rows in ids[..921].chunks(chunk.max(1)) {
        if let &[token] = rows {
            decoder.step(&mut state, token, StepMode::Prompt).unwrap();
        } else {
            decoder.prefill(&mut state, rows, StepMode::Prompt).unwrap();
        }
    }
    let mut token = ids[921];
    let started = std::time::Instant::now();
    for _ in 0..2_000 {
        let got = decoder.step(&mut state, token, StepMode::Generate).unwrap();
        token = got.token.expect("the generate step returns the token");
    }
    let step_seconds = started.elapsed().as_secs_f64();
    let stats = decoder.kernels.compile_counters();
    assert_eq!(
        stats.delta.traces, 1,
        "no delta trace after 2,000 compiled decode steps"
    );
    assert_eq!(
        stats.moe.traces, 1,
        "no moe trace after 2,000 compiled decode steps"
    );
    let (calls, _traces, hits) = stats.totals();
    assert!(
        calls >= 2_000 * 70,
        "2,000 steps of 30 delta and 40 moe blocks made {calls} compiled calls"
    );
    let report = serde_json::json!({
        "test": "two_thousand_compiled_decode_steps_add_no_trace_after_the_first_token",
        "activation_dtype": crate::family::qwen36::settings::activation_name(activation),
        "steps": 2_000, "step_seconds": step_seconds,
        "calls": calls, "traces": stats.delta.traces + stats.moe.traces,
        "cache_hits": hits,
        "compile_counters": compiled::report(&decoder.kernels)["compile_counters"].clone(),
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    std::fs::write(
        out.join("compile-trace-count.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    reset_compile_state();
}

/// Snapshot interop: a snapshot taken after compiled steps restores and the decode continues
/// UNCOMPILED with the same greedy tokens. The ownership rule of the adapter (the arrays of one
/// owner stay inside that owner's decoder) keeps a cross-adapter restore impossible by design, so
/// the uncompiled decoder is THIS decoder with the setting off — the honest reading of "an
/// UNCOMPILED decoder". A second, fully uncompiled decoder replays the same prefix and continues in
/// parallel: the two greedy token chains of the 16 suffix steps must be equal.
#[test]
#[ignore = "needs the real checkpoint on the Mac; run with --ignored and Q36_ACTIVATION unset"]
fn a_snapshot_after_compiled_steps_restores_into_an_uncompiled_decoder() {
    let _serial = crate::native_test_lock();
    reset_compile_state();
    // The uncompiled arms below read the setting off; a leftover `QWEN36_G2A_COMPILE=1` from a
    // measurement run would silently compile them and make the comparison vacuous.
    assert!(
        !compiled::requested(),
        "run the snapshot gate with QWEN36_G2A_COMPILE unset"
    );
    let activation = super::real::activation_from_env();
    assert_eq!(
        activation,
        Activation::F32,
        "the snapshot gate runs in F32 (run with Q36_ACTIVATION unset)"
    );
    let out = out_dir();
    let ids = real_ids();
    let (ckpt, host, config) = real_host();
    let mut compiled_decoder = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host,
        super::tool_forward::CONTEXT_LIMIT,
        activation,
    )
    .unwrap();
    compiled_decoder.set_compile(true).unwrap();
    let mut state = compiled_decoder.zero_state().unwrap();
    // 16 prompt tokens as one-token steps (compiled), then 8 compiled greedy steps. The first
    // greedy input is the 17th prompt id, exactly like the harness of `compile_logits`: N
    // prefilled, token N feeds the first decode step.
    let prefix_tokens: Vec<u32> = ids[..16].to_vec();
    for &id in &prefix_tokens {
        compiled_decoder
            .step(&mut state, id, StepMode::Prompt)
            .unwrap();
    }
    let mut token = ids[16];
    let mut fed = Vec::with_capacity(8);
    let mut compiled_greedy = Vec::with_capacity(8);
    for _ in 0..8 {
        fed.push(token);
        let got = compiled_decoder
            .step(&mut state, token, StepMode::Generate)
            .unwrap();
        token = got.token.expect("the generate step returns the token");
        compiled_greedy.push(token);
    }
    let snapshot = compiled_decoder.snapshot(&state).unwrap();
    // The input both continuations start from: the token the snapshot state expects next. The
    // continued loop below ADVANCES `token`, so the control chain must restart from `resume`,
    // not from the last continued token (the chains would differ and the final assert would
    // compare two different suffixes).
    let resume = token;
    // The UNCOMPILED continuation of the same decoder, restored from the compiled snapshot.
    compiled_decoder.compile = false;
    let mut restored = compiled_decoder.zero_state().unwrap();
    compiled_decoder.restore(&mut restored, &snapshot).unwrap();
    let mut continued = Vec::with_capacity(16);
    for _ in 0..16 {
        let got = compiled_decoder
            .step(&mut restored, token, StepMode::Generate)
            .unwrap();
        token = got.token.expect("the generate step returns the token");
        continued.push(token);
    }
    // The control: a second, fully uncompiled decoder replays the same prefix (the 16 prompt
    // tokens and the 8 compiled greedy choices) and continues 16 steps FROM THE SAME RESUME
    // TOKEN.
    let host_control = HostWeights::load(&ckpt, &QWEN3_6_35B_A3B, &config).unwrap();
    let mut control = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host_control,
        super::tool_forward::CONTEXT_LIMIT,
        activation,
    )
    .unwrap();
    let mut control_state = control.zero_state().unwrap();
    for &id in prefix_tokens.iter().chain(&fed) {
        control
            .step(&mut control_state, id, StepMode::Prompt)
            .unwrap();
    }
    let mut token = resume;
    let mut control_continued = Vec::with_capacity(16);
    for _ in 0..16 {
        let got = control
            .step(&mut control_state, token, StepMode::Generate)
            .unwrap();
        token = got.token.expect("the generate step returns the token");
        control_continued.push(token);
    }
    assert_eq!(
        continued, control_continued,
        "the uncompiled continuation after the compiled snapshot picked other tokens"
    );
    let report = serde_json::json!({
        "test": "a_snapshot_after_compiled_steps_restores_into_an_uncompiled_decoder",
        "prefix_tokens": prefix_tokens.len(),
        "compiled_greedy": compiled_greedy,
        "uncompiled_continuation": continued,
        "control_continuation": control_continued,
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    std::fs::write(
        out.join("compile-snapshot.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    drop(snapshot);
    reset_compile_state();
}

/// Memory: the `memory_stats` peak with compile on stays within +64 MB of the uncompiled run after
/// 64 decode steps at depth 16. Both numbers are printed for the report; the compiled programs hold
/// no weights, so the extra peak is the traced programs and their replay buffers.
#[test]
#[ignore = "needs the real checkpoint on the Mac; run with --ignored"]
fn compile_on_peak_stays_within_64mb_of_the_uncompiled_run() {
    let _serial = crate::native_test_lock();
    reset_compile_state();
    // The `off` arm below reads the setting off; a leftover `QWEN36_G2A_COMPILE=1` from a
    // measurement run would silently compile it and make the comparison vacuous.
    assert!(
        !compiled::requested(),
        "run the memory gate with QWEN36_G2A_COMPILE unset"
    );
    let activation = super::real::activation_from_env();
    let out = out_dir();
    let ids = real_ids();
    let run = |compile: bool| -> (u64, u64) {
        let (ckpt, host, _config) = real_host();
        let mut decoder = ResidentDecoder::new_with_activation(
            MlxKernels::try_new().unwrap(),
            &ckpt,
            host,
            super::tool_forward::CONTEXT_LIMIT,
            activation,
        )
        .unwrap();
        if compile {
            decoder.set_compile(true).unwrap();
        }
        decoder
            .set_chunk_rows(super::tool_forward::serve_chunk_setting(activation))
            .unwrap();
        // The load (and the warm-up with it) is not part of the steady-state peak: the reset
        // happens here, after the weights and the compiled programs exist.
        decoder.kernels.reset_peak_memory().unwrap();
        let mut state = decoder.zero_state().unwrap();
        let chunk = decoder.chunk_rows as usize;
        for rows in ids[..16].chunks(chunk.max(1)) {
            if let &[token] = rows {
                decoder.step(&mut state, token, StepMode::Prompt).unwrap();
            } else {
                decoder.prefill(&mut state, rows, StepMode::Prompt).unwrap();
            }
        }
        let mut token = ids[16];
        for _ in 0..64 {
            let got = decoder.step(&mut state, token, StepMode::Generate).unwrap();
            token = got.token.expect("the generate step returns the token");
        }
        let stats = decoder.kernels.memory_stats().unwrap();
        (stats.peak, stats.active)
    };
    let (off_peak, off_active) = run(false);
    let (on_peak, on_active) = run(true);
    println!(
        "compile off: peak {off_peak} bytes active {off_active}; \
         compile on: peak {on_peak} bytes active {on_active}; \
         difference {} bytes",
        on_peak.abs_diff(off_peak)
    );
    assert!(
        on_peak <= off_peak + 64 * 1024 * 1024,
        "compile on peaks at {on_peak} against {off_peak} uncompiled (+{} allowed)",
        64 * 1024 * 1024
    );
    let report = serde_json::json!({
        "test": "compile_on_peak_stays_within_64mb_of_the_uncompiled_run",
        "activation_dtype": crate::family::qwen36::settings::activation_name(activation),
        "off_peak": off_peak, "on_peak": on_peak,
        "off_active": off_active, "on_active": on_active,
        "allowed_extra": 64 * 1024 * 1024,
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    std::fs::write(
        out.join("compile-memory.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    reset_compile_state();
}

/// The fault hooks of the qualification build select the two new operations. Each fault kind runs
/// in its own child process (a native failure is sticky for the whole process), and the child
/// asserts the operation name, so a wrong fault index fails the test instead of passing weakly.
#[cfg(all(test, feature = "native-fault-qualification"))]
mod fault_tests {
    use super::*;
    use std::{
        env,
        io::Read,
        process::{Command, Stdio},
        thread,
        time::{Duration, Instant},
    };

    const CHILD: &str = "INFERENCE_COMPILE_GATE_CHILD";
    /// Set in the counting child: it prints the number of selected native operations so far,
    /// just before the call under test, and ends with success.
    const COUNT: &str = "INFERENCE_COMPILE_GATE_COUNT";
    /// Set in a probing child: the fault run prints the label of the failing operation
    /// (`FAULT_LABEL=...`) instead of asserting it, so that the parent can find the operation that
    /// it wants to strike.
    const PROBE: &str = "INFERENCE_COMPILE_GATE_PROBE";
    /// The most selected operations that the parent probes after the counted index to find the
    /// compiled delta trace of the warm-up (the zero import and the zero state come first).
    const WARM_UP_SCAN: u64 = 12;
    const DEADLINE: Duration = Duration::from_secs(240);
    /// The largest operation index that `INFERENCE_MLX_FAULT_NEXT_OPERATION` accepts.
    const FAULT_INDEX_LIMIT: u64 = 999;

    fn drain<R: Read + Send + 'static>(mut pipe: R) -> thread::JoinHandle<String> {
        thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = pipe.read_to_end(&mut bytes);
            String::from_utf8_lossy(&bytes).into_owned()
        })
    }

    /// Runs the test `name` in a child process, optionally with a fault variable, and returns its
    /// captured output. The child must pass.
    fn run_child(name: &str, count_only: bool, fault: Option<(&str, String)>) -> String {
        run_child_with(name, count_only, false, fault)
    }

    /// `run_child` with the probe switch: a probing child prints the label of the failing
    /// operation instead of asserting it.
    fn run_child_with(
        name: &str,
        count_only: bool,
        probe: bool,
        fault: Option<(&str, String)>,
    ) -> String {
        let path = module_path!().split_once("::").unwrap().1;
        let mut command = Command::new(env::current_exe().unwrap());
        command
            .args(["--exact", &format!("{path}::{name}"), "--nocapture"])
            .env(CHILD, "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if count_only {
            command.env(COUNT, "1");
        }
        if probe {
            command.env(PROBE, "1");
        }
        if let Some((variable, value)) = &fault {
            command.env(variable, value);
        }
        let mut child = command.spawn().unwrap();
        let stdout = drain(child.stdout.take().unwrap());
        let stderr = drain(child.stderr.take().unwrap());
        let started = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break Some(status);
            }
            if started.elapsed() >= DEADLINE {
                child.kill().unwrap();
                child.wait().unwrap();
                break None;
            }
            thread::sleep(Duration::from_millis(20));
        };
        let captured = format!("{} {}", stdout.join().unwrap(), stderr.join().unwrap());
        assert!(
            status.is_some_and(|status| status.success()) && captured.contains("1 passed"),
            "child did not pass {name}: {captured}"
        );
        captured
    }

    /// In the parent, runs `name` twice in child processes. The decoder setup before the call
    /// under test takes a few hundred selected native operations (the page imports of the
    /// weights), so the fault index cannot be a constant: the first child (the counting run, no
    /// fault) prints the number `n` of selected operations before the call under test, and the
    /// second child carries the fault `kind:(n + 1)`, which strikes the first selected operation
    /// of the call under test. `fault.1` names the kind (`call`, `null`, `mismatch`, or `partial`) before
    /// its colon. Returns `false` in a child, which then runs the test body itself.
    fn delegated_to_child(name: &str, fault: (&str, &str)) -> bool {
        if env::var_os(CHILD).is_some() {
            return false;
        }
        let kind = fault.1.split(':').next().unwrap();
        let count = operation_count(name);
        let index = count + 1;
        assert!(
            index <= FAULT_INDEX_LIMIT,
            "{name}: the fault index {index} exceeds the limit {FAULT_INDEX_LIMIT} of the hook"
        );
        run_child(name, false, Some((fault.0, format!("{kind}:{index}"))));
        true
    }

    /// Runs the counting child of the test `name` and returns the number of selected native
    /// operations that it printed just before the call under test.
    fn operation_count(name: &str) -> u64 {
        let counted = run_child(name, true, None);
        counted
            .split("OPCOUNT=")
            .nth(1)
            .and_then(|tail| tail.split_whitespace().next())
            .and_then(|text| text.parse().ok())
            .unwrap_or_else(|| panic!("the counting child of {name} printed no OPCOUNT: {counted}"))
    }

    /// In the counting child, prints the number of selected native operations so far and returns
    /// `true`, so that the caller returns before the call under test. In every other process it
    /// returns `false` and does nothing.
    fn announce_count(kernels: &MlxKernels) -> bool {
        if env::var_os(COUNT).is_none() {
            return false;
        }
        println!("OPCOUNT={}", kernels.selected_operation_count());
        true
    }

    /// Runs one compiled delta step on the zero inputs of the tiny model's first DeltaNet layer.
    /// The setup (the weight imports, the zero state) takes selected native operations, so the
    /// fault index comes from the counting run of `delegated_to_child`; the fault strikes the first
    /// selected operation of the compiled call, and the assertion of the operation name catches a
    /// wrong index.
    fn one_compiled_delta_step() -> Result<(), EngError> {
        let (ckpt, host) = tiny_host();
        let mut decoder = ResidentDecoder::new_with_activation(
            MlxKernels::try_new().unwrap(),
            &ckpt,
            host,
            64,
            Activation::F32,
        )
        .unwrap();
        let shape = decoder.weights.shape.clone();
        let w = first_delta(&decoder.weights).unwrap();
        let (x, rec, ring) = zero_delta_inputs(&mut decoder.kernels, &shape, Dtype::F32).unwrap();
        let mut ctx = Ctx {
            kernels: &mut decoder.kernels,
            consts: &decoder.weights.consts,
            compile: true,
        };
        if announce_count(ctx.kernels) {
            return Ok(());
        }
        let got = compiled::delta_step(&mut ctx, &shape, w, &rec, &ring, &x)?;
        ctx.kernels.evaluate(&[&got.out, &got.rec, &got.ring])
    }

    /// Runs one compiled device-routed MoE step on the zero input of the tiny model.
    fn one_compiled_moe_step() -> Result<(), EngError> {
        let (ckpt, host) = tiny_host();
        let mut decoder = ResidentDecoder::new_with_activation(
            MlxKernels::try_new().unwrap(),
            &ckpt,
            host,
            64,
            Activation::F32,
        )
        .unwrap();
        let shape = decoder.weights.shape.clone();
        let w = first_moe(&decoder.weights).unwrap();
        let (x, _rec, _ring) = zero_delta_inputs(&mut decoder.kernels, &shape, Dtype::F32).unwrap();
        let mut ctx = Ctx {
            kernels: &mut decoder.kernels,
            consts: &decoder.weights.consts,
            compile: true,
        };
        if announce_count(ctx.kernels) {
            return Ok(());
        }
        let (out, route) = compiled::moe_step_device(&mut ctx, &shape, w, &x)?;
        ctx.kernels
            .evaluate(&[&out, route.ids.ids(), &route.gates, &route.health])
    }

    /// One B2 DeltaNet seam call; count the fault index after fixture imports.
    fn one_compiled_delta_rows() -> Result<(), EngError> {
        let (ckpt, host) = tiny_host();
        let mut decoder = ResidentDecoder::new_with_activation(
            MlxKernels::try_new().unwrap(),
            &ckpt,
            host,
            64,
            Activation::F32,
        )
        .unwrap();
        let shape = decoder.weights.shape.clone();
        let w = first_delta(&decoder.weights).unwrap();
        let (_, rec, ring) = zero_delta_inputs(&mut decoder.kernels, &shape, Dtype::F32)?;
        let x = decoder.kernels.import_tensor(&f32_tensor(
            vec![2, dim(shape.hidden)?],
            &vec![0.0; 2 * shape.hidden],
        )?)?;
        let args = delta_seam_args(
            &shape,
            w,
            &decoder.weights.consts,
            (&x, &rec, &ring),
            1e-6,
            Dtype::F32,
        );
        if announce_count(&decoder.kernels) {
            return Ok(());
        }
        let got = decoder
            .kernels
            .compiled_delta_rows(&args, &[(&rec, &ring)])?;
        let mut outputs = vec![&got.out];
        for (rec, ring) in &got.states {
            outputs.extend([rec, ring]);
        }
        decoder.kernels.evaluate(&outputs)
    }

    /// One B2 MoE seam call, including row health and routing outputs.
    fn one_compiled_moe_rows() -> Result<(), EngError> {
        let (ckpt, host) = tiny_host();
        let mut decoder = ResidentDecoder::new_with_activation(
            MlxKernels::try_new().unwrap(),
            &ckpt,
            host,
            64,
            Activation::F32,
        )
        .unwrap();
        let shape = &decoder.weights.shape;
        let w = first_moe(&decoder.weights).unwrap();
        let x = decoder.kernels.import_tensor(&f32_tensor(
            vec![2, dim(shape.hidden)?],
            &vec![0.0; 2 * shape.hidden],
        )?)?;
        let args = moe_seam_args(shape, w, &x, Dtype::F32, false);
        if announce_count(&decoder.kernels) {
            return Ok(());
        }
        let got = decoder.kernels.compiled_moe_step(&args)?;
        decoder
            .kernels
            .evaluate(&[&got.out, &got.ids, &got.gates, &got.health])
    }

    /// The load-time warm-up under an injected call failure: the fault strikes the compiled delta
    /// trace inside `set_compile(true)`, so the warm-up returns the typed `Backend` load error HERE
    /// at load, never a failure in the middle of a request. The warm-up takes a few selected
    /// operations before the trace (the zero import and the zero state), so the parent probes the
    /// operations after the counted index until one fails as the compiled delta block.
    fn one_warm_up() -> Result<(), EngError> {
        let (ckpt, host) = tiny_host();
        let mut decoder = ResidentDecoder::new_with_activation(
            MlxKernels::try_new().unwrap(),
            &ckpt,
            host,
            64,
            Activation::F32,
        )
        .unwrap();
        if announce_count(&decoder.kernels) {
            return Ok(());
        }
        decoder.set_compile(true)?;
        Ok(())
    }

    #[test]
    fn a_failed_warm_up_trace_is_a_typed_load_error() {
        const NAME: &str = "a_failed_warm_up_trace_is_a_typed_load_error";
        const FAULT: (&str, &str) = ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "call:1");
        if env::var_os(CHILD).is_none() {
            let count = operation_count(NAME);
            let struck = |step: u64| (FAULT.0, format!("call:{}", count + step));
            let hit = (1..=WARM_UP_SCAN).find(|&step| {
                run_child_with(NAME, false, true, Some(struck(step)))
                    .contains("FAULT_LABEL=compiled delta block")
            });
            let step = hit.unwrap_or_else(|| {
                panic!("no operation within {WARM_UP_SCAN} of the warm-up start is the delta trace")
            });
            run_child(NAME, false, Some(struck(step)));
            return;
        }
        assert_fault(NAME, FAULT, "compiled delta block", one_warm_up);
    }

    /// Asserts that the injected fault struck the given operation and left the adapter terminal.
    /// An empty `operation` accepts any `Backend` label.
    fn assert_fault<F>(name: &str, fault: (&str, &str), operation: &'static str, run: F)
    where
        F: FnOnce() -> Result<(), EngError>,
    {
        if delegated_to_child(name, fault) {
            return;
        }
        if env::var_os(COUNT).is_some() {
            // The counting child: the run prints the count and returns Ok before the call.
            run().expect("the counting run stops before the call under test and succeeds");
            return;
        }
        let got = run();
        if env::var_os(PROBE).is_some() {
            let label = match &got {
                Err(EngError::Backend { operation, .. }) => *operation,
                Err(_) => "other",
                Ok(()) => "none",
            };
            println!("FAULT_LABEL={label}");
            return;
        }
        assert!(
            matches!(
                &got,
                Err(EngError::Backend { operation: got_op, .. })
                    if operation.is_empty() || *got_op == operation
            ),
            "expected Backend {operation} from fault {}, got {got:?}",
            fault.1
        );
        // A native failure fails the process-wide resident owner: a new adapter cannot even
        // initialize afterwards. That is the observable proof that the failing call made the
        // adapter terminal.
        let later = MlxKernels::try_new();
        assert!(
            matches!(
                &later,
                Err(EngError::Backend {
                    operation: "initialize",
                    message,
                }) if message.contains("has failed")
            ),
            "the native owner must be failed after the injected fault, got {:?}",
            later.err()
        );
    }

    #[test]
    fn a_failed_compiled_delta_call_is_terminal() {
        assert_fault(
            "a_failed_compiled_delta_call_is_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "call:1"),
            "compiled delta block",
            one_compiled_delta_step,
        );
    }

    #[test]
    fn a_null_compiled_delta_output_is_terminal() {
        assert_fault(
            "a_null_compiled_delta_output_is_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "null:1"),
            "compiled delta block",
            one_compiled_delta_step,
        );
    }

    #[test]
    fn a_mismatched_compiled_delta_output_is_terminal() {
        assert_fault(
            "a_mismatched_compiled_delta_output_is_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "mismatch:1"),
            "compiled delta block",
            one_compiled_delta_step,
        );
    }

    #[test]
    fn a_failed_compiled_moe_call_is_terminal() {
        assert_fault(
            "a_failed_compiled_moe_call_is_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "call:1"),
            "compiled moe block",
            one_compiled_moe_step,
        );
    }

    #[test]
    fn a_null_compiled_moe_output_is_terminal() {
        assert_fault(
            "a_null_compiled_moe_output_is_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "null:1"),
            "compiled moe block",
            one_compiled_moe_step,
        );
    }

    #[test]
    fn a_mismatched_compiled_moe_output_is_terminal() {
        assert_fault(
            "a_mismatched_compiled_moe_output_is_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "mismatch:1"),
            "compiled moe block",
            one_compiled_moe_step,
        );
    }
    #[test]
    fn a_failed_compiled_delta_rows_call_is_terminal() {
        assert_fault(
            "a_failed_compiled_delta_rows_call_is_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "call:1"),
            "compiled delta block",
            one_compiled_delta_rows,
        );
    }

    #[test]
    fn a_null_compiled_delta_rows_output_is_terminal() {
        assert_fault(
            "a_null_compiled_delta_rows_output_is_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "null:1"),
            "compiled delta block",
            one_compiled_delta_rows,
        );
    }

    #[test]
    fn a_mismatched_compiled_delta_rows_output_is_terminal() {
        assert_fault(
            "a_mismatched_compiled_delta_rows_output_is_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "mismatch:1"),
            "compiled delta block",
            one_compiled_delta_rows,
        );
    }

    /// The counted fault stops after one real output extraction, before the other slots fill.
    /// The child checks the successful non-null free, not only the native failure status.
    /// Removing the C rows cleanup branch must leave this counter unchanged and fail the test.
    #[test]
    fn a_partial_compiled_delta_rows_output_is_freed_before_terminal() {
        assert_fault(
            "a_partial_compiled_delta_rows_output_is_freed_before_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "partial:1"),
            "compiled delta block",
            || {
                let before = crate::kernels::mlx::native::compiled_delta_rows_cleanup_count();
                let got = one_compiled_delta_rows();
                if env::var_os(COUNT).is_none() {
                    let after = crate::kernels::mlx::native::compiled_delta_rows_cleanup_count();
                    assert_eq!(
                        after - before,
                        1,
                        "the real non-null partial output must be freed before native failure returns"
                    );
                }
                got
            },
        );
    }

    #[test]
    fn a_failed_compiled_moe_rows_call_is_terminal() {
        assert_fault(
            "a_failed_compiled_moe_rows_call_is_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "call:1"),
            "compiled moe block",
            one_compiled_moe_rows,
        );
    }

    #[test]
    fn a_null_compiled_moe_rows_output_is_terminal() {
        assert_fault(
            "a_null_compiled_moe_rows_output_is_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "null:1"),
            "compiled moe block",
            one_compiled_moe_rows,
        );
    }

    #[test]
    fn a_mismatched_compiled_moe_rows_output_is_terminal() {
        assert_fault(
            "a_mismatched_compiled_moe_rows_output_is_terminal",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "mismatch:1"),
            "compiled moe block",
            one_compiled_moe_rows,
        );
    }
}
