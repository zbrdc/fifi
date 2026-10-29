use super::super::{
    decoder::{ResidentDecoder, StepMode, StepResult},
    moe::Routing,
    state::{ResidentState, kv_storage_bytes},
};
use crate::{
    family::qwen36::{host_load::HostWeights, settings::Activation, tiny},
    kernels::MlxKernels,
};

const STEPS: usize = 6;

#[derive(Clone, Copy, Debug, serde::Serialize)]
struct RowObservation {
    batch: usize,
    request: usize,
    // The input token's zero-based position; this logits row predicts position + 1.
    position: usize,
    prefix_len: usize,
}

#[derive(Debug)]
struct Agreement {
    rows: usize,
    greedy_flips: usize,
    growth_events: usize,
    min_cosine: f64,
    max_relative: f64,
    kl_sum: f64,
    max_row_kl: f64,
    min_cosine_at: Option<RowObservation>,
    max_kl_at: Option<RowObservation>,
}

impl Agreement {
    fn kl_passes(&self) -> bool {
        self.rows != 0
            && self.kl_sum.is_finite()
            && self.kl_sum >= -1e-12
            && self.kl_sum / self.rows as f64 <= 0.004
            && (-1e-12..=0.05).contains(&self.max_row_kl)
            && 1.0 - self.greedy_flips as f64 / self.rows as f64 >= 0.99
    }
}

#[test]
fn kl_gate_accepts_boundaries_and_rejects_bad_metrics() {
    let boundary = Agreement {
        rows: 100,
        greedy_flips: 1,
        growth_events: 0,
        min_cosine: 1.0,
        max_relative: 0.0,
        kl_sum: 0.4,
        max_row_kl: 0.05,
        min_cosine_at: None,
        max_kl_at: None,
    };
    assert!(boundary.kl_passes());
    for rejected in [
        Agreement {
            kl_sum: 0.4001,
            ..boundary
        },
        Agreement {
            max_row_kl: 0.0501,
            ..boundary
        },
        Agreement {
            greedy_flips: 2,
            ..boundary
        },
        Agreement {
            rows: 0,
            ..boundary
        },
        Agreement {
            kl_sum: f64::NAN,
            ..boundary
        },
        Agreement {
            max_row_kl: f64::INFINITY,
            ..boundary
        },
    ] {
        assert!(!rejected.kl_passes(), "{rejected:?}");
    }
}

fn paired_states(
    decoder: &mut ResidentDecoder<'_>,
    prompts: &[Vec<u32>],
) -> (Vec<ResidentState>, Vec<ResidentState>, Vec<u32>) {
    let (mut solo, mut batch, mut tokens) = (Vec::new(), Vec::new(), Vec::new());
    for prompt in prompts {
        let mut a = decoder.zero_state().unwrap();
        let mut b = decoder.zero_state().unwrap();
        let expected = decoder.prefill(&mut a, prompt, StepMode::Generate).unwrap();
        let actual = decoder.prefill(&mut b, prompt, StepMode::Generate).unwrap();
        assert_eq!(actual.token, expected.token, "identical prefill controls");
        assert_eq!(a.pos(), b.pos());
        tokens.push(expected.token.unwrap());
        solo.push(a);
        batch.push(b);
    }
    (solo, batch, tokens)
}

fn compare(actual: &StepResult, expected: &StepResult, at: RowObservation, result: &mut Agreement) {
    let (got, want) = (
        actual.logits.as_ref().unwrap(),
        expected.logits.as_ref().unwrap(),
    );
    assert_eq!(got.len(), want.len(), "{at:?}");
    assert!(
        got.iter().chain(want).all(|value| value.is_finite()),
        "{at:?}"
    );
    let delta = got
        .iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - f64::from(*w)).abs())
        .fold(0.0, f64::max);
    let scale = want
        .iter()
        .map(|value| f64::from(value.abs()))
        .fold(0.0, f64::max);
    let relative = delta / scale.max(f64::MIN_POSITIVE);
    let cosine = super::super::parity::cosine(got, want);
    assert!(
        super::super::within_mode::within_mode_logits_passes(Activation::Bf16, cosine, relative),
        "{at:?}: cosine={cosine} relative={relative}"
    );
    if actual.token != expected.token {
        let control = expected.token.unwrap() as usize;
        let candidate = actual.token.unwrap() as usize;
        eprintln!(
            "GREEDY_FLIP {at:?}: expected={control} actual={candidate} control_gap={} max_abs={delta}",
            f64::from(want[control]) - f64::from(want[candidate])
        );
        result.greedy_flips += 1;
    }
    let kl = super::super::compile_logits::kl_divergence(want, got).unwrap();
    assert!(kl.is_finite() && kl >= -1e-12, "{at:?}: invalid KL={kl}");
    println!(
        "BATCH_BF16_ROW {}",
        serde_json::json!({
            "row": at,
            "cosine": cosine,
            "relative": relative,
            "kl": kl,
            "top1": actual.token == expected.token,
        })
    );
    if result.max_kl_at.is_none() || kl > result.max_row_kl {
        result.max_row_kl = kl;
        result.max_kl_at = Some(at);
    }
    if result.min_cosine_at.is_none() || cosine < result.min_cosine {
        result.min_cosine = cosine;
        result.min_cosine_at = Some(at);
    }
    result.kl_sum += kl;
    result.rows += 1;
    result.max_relative = result.max_relative.max(relative);
}

fn compiled_window(
    decoder: &mut ResidentDecoder<'_>,
    window: &mut [&mut ResidentState],
    tokens: &[u32],
) -> Vec<StepResult> {
    decoder.set_compile(true).unwrap();
    if tokens.len() > 1 {
        decoder.prepare_compiled_rows(&[tokens.len()]).unwrap();
    }
    let delta_layers = decoder
        .weights
        .layers
        .iter()
        .filter(|layer| matches!(layer.mixer, super::super::weights::RMixer::Delta(_)))
        .count() as u64;
    let moe_layers = decoder.weights.layers.len() as u64;
    assert!(delta_layers > 0 && moe_layers > 0);
    let before = decoder.kernels.compile_counters();
    let actual = decoder
        .decode_rows(window, tokens, StepMode::Logits)
        .unwrap();
    let after = decoder.kernels.compile_counters();
    assert_eq!(after.delta.calls - before.delta.calls, delta_layers);
    assert_eq!(after.moe.calls - before.moe.calls, moe_layers);
    assert_eq!(
        after.delta.traces, before.delta.traces,
        "decode retraced DeltaNet"
    );
    assert_eq!(after.moe.traces, before.moe.traces, "decode retraced MoE");
    println!(
        "BATCH_COMPILED_WITNESS {}",
        serde_json::json!({
            "batch": tokens.len(),
            "delta_calls": after.delta.calls - before.delta.calls,
            "moe_calls": after.moe.calls - before.moe.calls,
            "delta_layers": delta_layers,
            "moe_layers": moe_layers,
            "decode_traces": 0,
        })
    );
    actual
}

fn identity(
    decoder: &mut ResidentDecoder<'_>,
    prompts: &[Vec<u32>],
    compiled: bool,
    fixture: &'static str,
) -> Vec<Agreement> {
    assert_eq!(
        decoder.routing,
        if compiled {
            Routing::Device
        } else {
            Routing::Host
        }
    );
    assert!(!super::super::compiled::step_requested(decoder.compile));
    assert_eq!(prompts.len(), 8);
    assert!(prompts.windows(2).all(|pair| pair[0].len() < pair[1].len()));
    let mut results = Vec::new();
    let batches: &[usize] = if compiled { &[1, 2, 4, 8] } else { &[2, 4, 8] };
    for &rows in batches {
        let mut result = Agreement {
            rows: 0,
            greedy_flips: 0,
            growth_events: 0,
            min_cosine: 1.0,
            max_relative: 0.0,
            kl_sum: 0.0,
            max_row_kl: 0.0,
            min_cosine_at: None,
            max_kl_at: None,
        };
        if compiled {
            decoder.set_compile(false).unwrap();
        }
        let (mut solo, mut batch, mut tokens) = paired_states(decoder, &prompts[..rows]);
        for step in 0..STEPS {
            // Reuse one weight load while keeping the teacher-forced reference uncompiled.
            if compiled {
                decoder.set_compile(false).unwrap();
            }
            let control_before = decoder.kernels.compile_counters();
            let expected: Vec<_> = solo
                .iter_mut()
                .zip(&tokens)
                .map(|(state, &token)| decoder.step(state, token, StepMode::Logits).unwrap())
                .collect();
            let control_after = decoder.kernels.compile_counters();
            assert_eq!(
                (control_after.delta.calls, control_after.moe.calls),
                (control_before.delta.calls, control_before.moe.calls),
                "OFF control executed a compiled block"
            );
            let storage_before: Vec<_> = batch
                .iter()
                .map(|state| kv_storage_bytes(state.layers()))
                .collect();
            let mut window: Vec<_> = batch.iter_mut().collect();
            let before = super::BATCH_CALLS.with(std::cell::Cell::get);
            let actual = if compiled {
                compiled_window(decoder, &mut window, &tokens)
            } else {
                decoder
                    .decode_rows(&mut window, &tokens, StepMode::Logits)
                    .unwrap()
            };
            assert_eq!(
                super::BATCH_CALLS.with(std::cell::Cell::get),
                before + usize::from(rows > 1)
            );
            assert_eq!(actual.len(), rows);
            for (slot, (got, want)) in actual.iter().zip(&expected).enumerate() {
                compare(
                    got,
                    want,
                    RowObservation {
                        batch: rows,
                        request: slot,
                        position: prompts[slot].len() + step,
                        prefix_len: prompts[slot].len(),
                    },
                    &mut result,
                );
                assert_eq!(batch[slot].pos(), solo[slot].pos());
                assert_eq!(
                    batch[slot].pos(),
                    u32::try_from(prompts[slot].len() + step + 1).unwrap()
                );
                result.growth_events +=
                    usize::from(kv_storage_bytes(batch[slot].layers()) > storage_before[slot]);
            }
            // Both paths consume the control token even when a near-tie flips the batch argmax.
            tokens = expected.iter().map(|row| row.token.unwrap()).collect();
        }
        assert_eq!(result.rows, rows * STEPS);
        assert!(result.min_cosine_at.is_some() && result.max_kl_at.is_some());
        println!(
            "BF16 teacher-forced fixture={fixture} compiled={compiled} B={rows} steps={STEPS} agreement={result:?}"
        );
        println!(
            "BATCH_BF16_KL {}",
            serde_json::json!({
                "fixture": fixture,
                "batch": rows,
                "compiled_candidate": compiled,
                "reference": "compile OFF independent solo; KL(OFF || candidate)",
                "rows": result.rows,
                "mean_kl": result.kl_sum / result.rows as f64,
                "max_row_kl": result.max_row_kl,
                "min_cosine": result.min_cosine,
                "min_cosine_at": result.min_cosine_at,
                "max_kl_at": result.max_kl_at,
                "top1": result.rows - result.greedy_flips,
                "top1_rate": 1.0 - result.greedy_flips as f64 / result.rows as f64,
                "batch_windows_asserted": STEPS,
                "kl_gate_passed": result.kl_passes(),
            })
        );
        results.push(result);
    }
    results
}

fn tiny_identity(compiled: bool) -> Vec<Agreement> {
    let _serial = crate::native_test_lock();
    let ckpt = tiny::checkpoint(&tiny::TINY_KV8);
    let host = HostWeights::load(&ckpt, &tiny::TINY_KV8, &tiny::config(&tiny::TINY_KV8)).unwrap();
    let boundary = super::super::kv_plan::KV_STEP;
    let mut decoder = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &ckpt,
        host,
        u32::try_from(2 * boundary).unwrap(),
        Activation::Bf16,
    )
    .unwrap();
    decoder.routing = if compiled {
        Routing::Device
    } else {
        Routing::Host
    };
    let prompts: Vec<_> = [1, 2, 4, 8, 16, 32, boundary - 1, boundary]
        .into_iter()
        .enumerate()
        .map(|(slot, len)| (0..len).map(|pos| ((slot + pos) % 8) as u32).collect())
        .collect();
    identity(&mut decoder, &prompts, compiled, "tiny_plumbing")
}

#[test]
fn bf16_teacher_forced_identity_protocol_covers_staggered_prefixes_and_growth() {
    let result = tiny_identity(false);
    assert!(
        result.iter().all(|batch| batch.greedy_flips == 0),
        "{result:?}"
    );
    assert_eq!(
        result
            .iter()
            .map(|batch| batch.growth_events)
            .sum::<usize>(),
        2
    );
    assert!(result.iter().all(Agreement::kl_passes), "{result:?}");
}

fn real_identity(compiled: bool) {
    let _serial = crate::native_test_lock();
    use sha2::{Digest, Sha256};
    let dir = super::super::real::env_path("Q36_CKPT_DIR");
    let bytes = std::fs::read(dir.join("tokenizer.json")).unwrap();
    let tokenizer = crate::tokenizer::Tokenizer::from_bytes(&bytes).unwrap();
    let prompts: Vec<_> = (0..8)
        .map(|slot| {
            let text = format!(
                "Prompt {slot}. {}",
                "Explain how ownership keeps Rust references valid. ".repeat(slot + 1)
            );
            tokenizer.encode(&text).unwrap()
        })
        .collect();
    assert!(
        prompts
            .iter()
            .all(|prompt| !prompt.is_empty() && prompt.len() + STEPS < 512)
    );
    assert!(prompts.windows(2).all(|pair| pair[0].len() < pair[1].len()));
    println!(
        "tokenizer_sha256={:x} prompt_lengths={:?}",
        Sha256::digest(&bytes),
        prompts.iter().map(Vec::len).collect::<Vec<_>>()
    );
    let real = super::super::real::open(&dir);
    let host = HostWeights::load(
        &real.ckpt,
        &crate::family::spec::QWEN3_6_35B_A3B,
        &real.config,
    )
    .unwrap();
    let mut decoder = ResidentDecoder::new_with_activation(
        MlxKernels::try_new().unwrap(),
        &real.ckpt,
        host,
        512,
        Activation::Bf16,
    )
    .unwrap();
    decoder.routing = if compiled {
        Routing::Device
    } else {
        Routing::Host
    };
    let result = identity(&mut decoder, &prompts, compiled, "real_weights");
    println!("REAL_BF16_TEACHER_FORCED_WITHIN_EXISTING_BOUNDS compiled={compiled} {result:?}");
    if result.iter().any(|batch| batch.greedy_flips != 0) {
        println!(
            "NOT_EXACT_GREEDY_IDENTITY: inspect GREEDY_FLIP gaps before a near-tie qualification ruling"
        );
    }
    assert!(result.iter().all(Agreement::kl_passes), "{result:?}");
}

/// Plumbing-only: asserts dispatch, counters, positions and KV growth, not real-weight identity.
#[test]
fn compiled_device_teacher_forced_identity_covers_all_row_sizes_and_growth() {
    let result = tiny_identity(true);
    assert_eq!(
        result.iter().map(|batch| batch.rows).collect::<Vec<_>>(),
        [6, 12, 24, 48]
    );
    assert_eq!(
        result
            .iter()
            .map(|batch| batch.growth_events)
            .sum::<usize>(),
        2
    );
    assert!(result.iter().all(Agreement::kl_passes), "{result:?}");
}

#[test]
#[ignore = "real weights: Q36_CKPT_DIR required; correctness only"]
fn real_weights_bf16_teacher_forced_distinct_prompts() {
    real_identity(false);
}

#[test]
#[ignore = "real weights: Q36_CKPT_DIR required; compiled ON/OFF correctness only"]
fn real_weights_compiled_device_identity_vs_off() {
    real_identity(true);
}
