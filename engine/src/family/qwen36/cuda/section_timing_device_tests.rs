//! The device test of the in-graph section timer: mechanics only, on the 3090, report-only timing
//! lines (the `gdn_scan_seq` precedent: the numbers are printed, not gated).
//!
//! One test per process with the GPU held exclusively, like the other full-step device rows.
//! Isolation (the latch rule): the device phase is a plain `Result` function that returns host data
//! through `?` and drops every CUDA owner at its return; only the caller, with no device owner
//! alive, asserts.
//!
//! What is gated: the marks exist (`section_times` returns readings after a captured step),
//! every reading is finite, the whole-token pair is positive, the marked groups ARE the
//! model's section plan (`section_plan` of its layer kinds), the first token ran eager and
//! every later one the captured graph (a refused capture would report its fallback), and the
//! marks-off step of the observer-effect pair has no readings. What is reported, never
//! gated: the per-section milliseconds, the summed section GPU time, the marked fraction of
//! the whole token, the per-token wall time, the idle-gap fractions `1 - whole/wall` (the
//! measured pair) and `1 - sum/wall` (the estimate), and both steady walls of the
//! observer-effect pair (marked and marks-off) — the verdicts come from an exclusive timing
//! window.

use super::super::host::HostShape;
use super::dims::{KvMode, StepDims, StepOptions};
use super::native_device_rig::device_present;
use super::pinned::{ROTARY_DIM, head_rows_of};
use super::resources::StepResources;
use super::rules::QwenRules;
use super::run_report::CaptureOutcome;
use super::section_timing::section_plan;
use super::start::GGUF_NAME;
use super::step::{CaptureConfig, QwenResidentStep};
use crate::{
    EngError,
    family::{
        QWEN3_6_35B_A3B,
        gguf_model::{Inventory, open_qwen3_6_35b_a3b},
        project::cuda::{CudaProjector, LoadOptions, Reserve, WeightLayout},
        qwen36::settings::tokenizer_pin,
    },
    tokenizer::TokenizerBundle,
};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The context of the run: enough positions for the prompt and the generated rows.
const CONTEXT: usize = 1_024;
/// A short fixed prompt inside the sampler vocabulary, and how many tokens run in total:
/// six tokens of the prompt prefix; the sampler output is ignored (the row times the step's
/// mechanics, not generation).
const PROMPT: [u32; 8] = [128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384];
const TOKENS: usize = 6;

/// One token's section readings plus how the body ran and the wall time of the `run_token`
/// call that produced them (report-only).
struct Reading {
    token: usize,
    how: CaptureOutcome,
    embed_ms: f32,
    tail_ms: f32,
    whole_ms: f32,
    sum_ms: f32,
    fraction: f32,
    groups: Vec<(usize, usize, f32)>,
    wall_ms: f32,
}

/// One token of the marks-off pair: how the body ran and the wall time (report-only).
struct PlainToken {
    token: usize,
    how: CaptureOutcome,
    wall_ms: f32,
}

/// The device phase's result: the marked groups (so the caller can assert they are the
/// model's section plan), the marked readings, and the marks-off tokens of the
/// observer-effect pair.
struct DeviceReceipt {
    marked_groups: Vec<(usize, usize)>,
    readings: Vec<Reading>,
    plain: Vec<PlainToken>,
}

fn dims() -> Result<StepDims, EngError> {
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
    StepDims::new(&shape, &inventory)
}

fn options() -> StepOptions {
    StepOptions {
        kv_mode: KvMode::Fp16,
        context_tokens: CONTEXT,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Native,
    }
}

fn gguf_path() -> Result<PathBuf, EngError> {
    let dir = std::env::var("Q36_CUDA_MODEL_DIR")
        .unwrap_or_else(|_| "/media/dan/nvme/models/qwen3.6-35b-a3b".to_owned());
    Ok(Path::new(&dir).join(GGUF_NAME))
}

fn load_options() -> LoadOptions<'static> {
    LoadOptions {
        reserve: Reserve {
            kv_cache: 0,
            work: 0,
            headroom: 0,
        },
        drop_page_cache: None,
        layout: WeightLayout::Native,
    }
}

/// The rules of the loaded bundle (the gate's seam): eos and the sampler vocabulary come from
/// the same bundle, never the padded head width.
fn rules_for(bundle: &TokenizerBundle) -> Result<QwenRules, EngError> {
    let vocab = usize::try_from(bundle.vocab_size())
        .map_err(|_| EngError::Checkpoint("the vocabulary exceeds usize".into()))?;
    QwenRules::new(vec![0], vocab, head_rows_of()?)
}

/// The device phase: builds the pinned model's step with capture on, runs the prompt tokens
/// with marks on and then — for the observer-effect pair — the same tokens with marks off,
/// and returns each token's readings with the wall time of its `run_token` call. The marked
/// groups come back too, so the caller (with no device owner alive) can assert they are the
/// model's section plan.
fn run_pair() -> Result<DeviceReceipt, EngError> {
    let gguf = gguf_path()?;
    let bundle = TokenizerBundle::load(gguf.parent().unwrap_or(Path::new(".")), &tokenizer_pin()?)?;
    let rules = rules_for(&bundle)?;
    let projector = CudaProjector::new()?;
    let model = open_qwen3_6_35b_a3b(&gguf, None)?;
    let device = projector.load_model(&model, load_options())?.device;
    let opts = options();
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
    let res = StepResources::new(projector.context(), dims()?, &inventory, &opts)?;

    let mut step = QwenResidentStep::new(projector.context(), &res, &device, &opts, rules)?;
    step.set_capture_config(CaptureConfig {
        capture: true,
        section_marks: true,
    })?;
    let mut readings = Vec::new();
    for token in 0..TOKENS {
        let next = PROMPT.get(token).copied().unwrap_or(PROMPT[0]);
        let wall = Instant::now();
        let _ = step.run_token(next, false)?;
        let wall_ms = wall.elapsed().as_secs_f32() * 1e3;
        let times = step.section_times()?.ok_or_else(|| {
            EngError::Checkpoint("no section readings after a marked token".into())
        })?;
        let groups = times
            .group_marked_ms()
            .iter()
            .copied()
            .zip(step.marked_groups())
            .map(|(ms, group)| (group.layer, group.size, ms))
            .collect();
        readings.push(Reading {
            token: token + 1,
            how: step.last_report().capture,
            embed_ms: times.embed_ms(),
            tail_ms: times.tail_ms(),
            whole_ms: times.whole_ms(),
            sum_ms: times.extrapolated_ms(),
            fraction: times.whole_ratio(),
            groups,
            wall_ms,
        });
    }
    let marked_groups: Vec<(usize, usize)> = step
        .marked_groups()
        .iter()
        .map(|group| (group.layer, group.size))
        .collect();
    // The marks die with the step (their events and the graph that re-records them); the
    // model, the device and the resources stay for the marks-off half of the pair.
    drop(step);

    let rules = rules_for(&bundle)?;
    let mut plain = QwenResidentStep::new(projector.context(), &res, &device, &opts, rules)?;
    plain.set_capture_config(CaptureConfig {
        capture: true,
        section_marks: false,
    })?;
    if let Some(times) = plain.section_times()? {
        return Err(EngError::Checkpoint(format!(
            "a marks-off step reported section readings: {times:?}"
        )));
    }
    let mut plain_tokens = Vec::new();
    for token in 0..TOKENS {
        let next = PROMPT.get(token).copied().unwrap_or(PROMPT[0]);
        let wall = Instant::now();
        let _ = plain.run_token(next, false)?;
        let wall_ms = wall.elapsed().as_secs_f32() * 1e3;
        plain_tokens.push(PlainToken {
            token: token + 1,
            how: plain.last_report().capture,
            wall_ms,
        });
    }
    // Still no readings after the marks-off run: that step carries no marks at all.
    if let Some(times) = plain.section_times()? {
        return Err(EngError::Checkpoint(format!(
            "a marks-off step reported section readings after its run: {times:?}"
        )));
    }
    Ok(DeviceReceipt {
        marked_groups,
        readings,
        plain: plain_tokens,
    })
}

#[test]
#[ignore = "the 3090, held exclusively: loads the real model; run by name"]
fn section_marks_time_the_captured_step() {
    if !device_present("section_marks_time_the_captured_step") {
        return;
    }
    let DeviceReceipt {
        marked_groups,
        readings,
        plain: plain_tokens,
    } = run_pair().unwrap_or_else(|error| panic!("section-timer device phase: {error:?}"));
    assert_eq!(readings.len(), TOKENS);
    assert_eq!(plain_tokens.len(), TOKENS);

    // Mechanics: the marked groups ARE the model's section plan (the second layer of each
    // run of identical kinds, a singleton run its only layer).
    let expected_groups: Vec<(usize, usize)> = section_plan(&dims().unwrap().shape.kinds)
        .groups
        .iter()
        .map(|group| (group.layer, group.size))
        .collect();
    assert_eq!(marked_groups, expected_groups);

    for reading in &readings {
        // Mechanics: the marks exist and every reading is finite; the whole-token pair is
        // strictly positive (the pair around Begin..Advance measures a full decode step);
        // token 1 runs eager (the warm-up), every later one the captured graph — a refused
        // capture would report its fallback instead.
        let expected = if reading.token == 1 {
            CaptureOutcome::Eager
        } else {
            CaptureOutcome::Captured
        };
        assert_eq!(reading.how, expected, "token {}", reading.token);
        assert!(reading.embed_ms.is_finite(), "token {}", reading.token);
        assert!(reading.tail_ms.is_finite(), "token {}", reading.token);
        assert!(reading.whole_ms.is_finite(), "token {}", reading.token);
        assert!(reading.sum_ms.is_finite(), "token {}", reading.token);
        assert!(reading.whole_ms > 0.0, "token {}", reading.token);
        assert!(
            !reading.groups.is_empty(),
            "token {}: the model's layer kinds form no marked groups",
            reading.token
        );
        for (layer, size, ms) in &reading.groups {
            assert!(ms.is_finite(), "token {} layer {layer}", reading.token);
            assert!(*size > 0, "token {} layer {layer}", reading.token);
        }
        // Report-only: per-section GPU time, the summed section GPU time, the marked fraction, the
        // wall time of the token, and the idle-gap fractions — `1 - whole/wall` from the measured
        // pair, `1 - sum/wall` from the estimate. Never gated here; the claimed numbers come from
        // an exclusive timing window.
        let groups = reading
            .groups
            .iter()
            .map(|(layer, size, ms)| format!("l{layer}x{size}:{ms:.4}ms"))
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "SECTION TIMING token={} embed={:.4}ms tail={:.4}ms whole={:.4}ms groups=[{}] sum={:.4}ms fraction={:.4} wall={:.4}ms idle_gap={:.4} sum_gap={:.4} (report-only)",
            reading.token,
            reading.embed_ms,
            reading.tail_ms,
            reading.whole_ms,
            groups,
            reading.sum_ms,
            reading.fraction,
            reading.wall_ms,
            1.0 - reading.whole_ms / reading.wall_ms,
            1.0 - reading.sum_ms / reading.wall_ms,
        );
    }

    // The observer-effect pair (report-only): both steady walls from the same token filter,
    // so the receipt reads what the record nodes cost the replay. The marks-off half runs
    // second on a shared GPU, and four tokens cannot separate a small gap from noise — the
    // line prints the caveat only; the verdict belongs to the exclusive timing window.
    let steady_mean = |walls: &[f32]| walls.iter().sum::<f32>() / walls.len() as f32;
    let marked_steady: Vec<f32> = readings
        .iter()
        .filter(|reading| reading.token >= 3)
        .map(|reading| reading.wall_ms)
        .collect();
    let plain_steady: Vec<f32> = plain_tokens
        .iter()
        .filter(|token| token.token >= 3)
        .map(|token| token.wall_ms)
        .collect();
    for token in &plain_tokens {
        let expected = if token.token == 1 {
            CaptureOutcome::Eager
        } else {
            CaptureOutcome::Captured
        };
        assert_eq!(token.how, expected, "marks-off token {}", token.token);
    }
    println!(
        "SECTION TIMING steady_wall marked={:.4}ms marks_off={:.4}ms over {} tokens each (report-only; a small gap is below this resolution)",
        steady_mean(&marked_steady),
        steady_mean(&plain_steady),
        plain_steady.len(),
    );
}
