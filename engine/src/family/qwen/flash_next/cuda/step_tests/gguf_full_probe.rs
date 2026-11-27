//! Exploratory full-model staged GGUF probe; not production admission or startup.
#[path = "gguf_full_probe/admission.rs"]
mod admission;
#[path = "gguf_full_probe/router_replay.rs"]
mod router_replay;
#[path = "gguf_full_probe/trace.rs"]
mod trace;
#[path = "gguf_full_probe/upstream_intervention.rs"]
mod upstream_intervention;

use super::*;
use crate::family::qwen::flash_next::{
    cuda::{
        driver::TokenFeeds,
        staged_moe::{GgufMoeOwner, GgufStep},
    },
    load::gguf::{GgufModel, staged_trunk::GgufTrunk, token_feed::GgufTokenFeeds},
    step::GgufDecodePrecision,
};
use crate::kernels::cuda::{CudaContext, resident::step_params::StepParams};
use admission::ProbeBudget;

fn greedy(values: &[f32]) -> usize {
    assert!(!values.is_empty() && values.iter().all(|x| x.is_finite()));
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    best
}

#[test]
#[ignore = "requires verified pinned GGUF and an admitted CUDA device"]
fn pinned_guarded_one_layer_and_head() {
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("pinned first shard");
    assert_pinned_file(
        &path,
        "9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915",
    );
    let model = GgufModel::open(Path::new(&path)).unwrap().unwrap();
    ProbeBudget::check_host(&model).unwrap();
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let budget = ProbeBudget::new(&model, &ctx, 3).unwrap();
    let trunk =
        GgufTrunk::upload_with_final_guarded(&model, &ctx, &stream, 0..1, |layer, weights| {
            budget.group_guard(&ctx, layer, weights)
        })
        .unwrap();
    budget.before_step(&ctx).unwrap();
    let prior = ctx.live_bytes();
    let _step = GgufStep::new(&ctx, &stream, &trunk, 3).unwrap();
    assert_eq!(ctx.live_bytes() - prior, budget.step_payload());
    budget.before_execution(&ctx).unwrap();
    eprintln!("guarded one-layer plus final staged payload verified");
}

#[test]
#[ignore = "requires pinned GGUF, full host oracle and guarded resource admission"]
fn pinned_full_model_first_logits() {
    run_full_model(None);
}

#[test]
#[ignore = "requires pinned GGUF, host first-token trace and guarded resource admission"]
fn pinned_first_token_full_layer_trace() {
    run_full_model(Some(TraceWindow::All));
}

#[test]
#[ignore = "requires pinned GGUF, host first-token trace and guarded resource admission"]
fn pinned_first_four_layer_trace() {
    run_full_model(Some(TraceWindow::Four));
}

#[test]
#[ignore = "requires pinned GGUF, host first-token trace and guarded resource admission"]
fn pinned_first_six_layer_router_trace() {
    run_full_model(Some(TraceWindow::Six));
}

#[test]
#[ignore = "requires pinned GGUF and guarded resource admission"]
fn pinned_prompt_stateful_four_layer_trace() {
    run_full_model(Some(TraceWindow::StatefulFour));
}

#[test]
#[ignore = "requires pinned GGUF and guarded resource admission"]
fn pinned_prompt_stateful_full_layer_trace() {
    run_full_model(Some(TraceWindow::StatefulAll));
}

#[derive(Clone, Copy)]
enum TraceWindow {
    All,
    Four,
    Six,
    StatefulFour,
    StatefulAll,
}

/// One definition of what a full-model probe compares: the prompt comes from the oracle
/// file, never from a literal here, so a second fixture needs no source change.
struct ProbeInput {
    prompt: Vec<u32>,
    expected: Vec<f32>,
    /// `None` reports the logits error; `Some(b)` fails the test above `b`.
    bound: Option<f64>,
    /// Greedy steps generated after the prompt (0 = prompt only).
    continuation: usize,
    /// Every oracle logit row: the prompt's final step followed by its greedy continuation.
    /// `expected` is `oracle_rows[0]`.
    oracle_rows: Vec<Vec<f32>>,
}

impl ProbeInput {
    /// The pinned oracle keeps its content hash. Any other oracle must be paired with its own
    /// prompt by construction (the prompt is read from it), so it is accepted by shape only;
    /// `FIFI_FLASH_GGUF_HOST_ORACLE_SHA256` may pin it explicitly.
    fn open() -> Self {
        let oracle_path = std::env::var_os("FIFI_FLASH_GGUF_HOST_ORACLE").expect("host oracle");
        if let Some(expected) = std::env::var_os("FIFI_FLASH_GGUF_HOST_ORACLE_SHA256") {
            assert_pinned_file(&oracle_path, &expected.to_string_lossy());
        } else {
            assert_pinned_file(
                &oracle_path,
                "2c2133dccb289ad6f3e244d304f655189c04497bf19cd3b609779eb5af235ffa",
            );
        }
        let oracle: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&oracle_path).unwrap()).unwrap();
        let prompt: Vec<u32> =
            serde_json::from_value(oracle["prompt"].clone()).expect("oracle holds its prompt");
        assert!(!prompt.is_empty(), "oracle prompt is empty");
        let oracle_rows: Vec<Vec<f32>> =
            serde_json::from_value(oracle["logits"].clone()).expect("oracle logits");
        let expected = oracle_rows.first().cloned().expect("oracle first logits");
        assert!(expected.len() > 1 && expected.iter().all(|x| x.is_finite()));
        for row in &oracle_rows {
            assert!(row.len() == expected.len(), "oracle rows differ in width");
        }
        let bound = Self::parse_bound(
            std::env::var("FIFI_FLASH_GGUF_LOGITS_BOUND")
                .ok()
                .as_deref(),
        )
        .unwrap_or_else(|message| panic!("FIFI_FLASH_GGUF_LOGITS_BOUND: {message}"));
        let continuation = Self::parse_continuation(
            std::env::var("FIFI_FLASH_GGUF_CONTINUATION")
                .ok()
                .as_deref(),
        )
        .unwrap_or_else(|message| panic!("FIFI_FLASH_GGUF_CONTINUATION: {message}"));
        assert!(
            Self::continuation_fits_oracle(continuation, oracle_rows.len()),
            "oracle holds {} greedy steps; it cannot qualify {continuation} continuation steps",
            oracle_rows.len()
        );
        Self {
            prompt,
            expected,
            bound,
            continuation,
            oracle_rows,
        }
    }

    /// One oracle row covers the prompt's own logits; each continuation step needs one more.
    fn continuation_fits_oracle(continuation: usize, oracle_steps: usize) -> bool {
        continuation < oracle_steps
    }

    /// The token a position consumes: the prompt's own, or once the prompt is exhausted the
    /// token the device chose at the previous position (recorded in step order).
    fn fed_token(prompt: &[u32], generated: &[u32], position: usize) -> u32 {
        match prompt.get(position) {
            Some(token) => *token,
            None => generated[position - prompt.len()],
        }
    }

    /// Whether this step's row is the same computation on both sides. Each step consumes the
    /// device's own previous choice, so the first differing choice ends comparability for this
    /// step and every later one, even if the two happen to agree again.
    fn comparable_so_far(diverged: &mut bool, actual: usize, expected: usize) -> bool {
        *diverged |= actual != expected;
        !*diverged
    }

    fn parse_continuation(value: Option<&str>) -> Result<usize, &'static str> {
        let Some(text) = value else {
            return Ok(0);
        };
        text.parse::<usize>()
            .ok()
            .filter(|n| *n > 0)
            .ok_or("continuation must be a positive whole number of steps")
    }

    fn parse_bound(value: Option<&str>) -> Result<Option<f64>, &'static str> {
        let Some(text) = value else {
            return Ok(None);
        };
        let bound: f64 = text.parse().map_err(|_| "logits bound must be a number")?;
        if !bound.is_finite() || bound <= 0.0 {
            return Err("logits bound must be positive");
        }
        Ok(Some(bound))
    }

    /// `FIFI_FLASH_GGUF_PROBE_PRECISION`: absent is the production default
    /// ([`GgufDecodePrecision::default()`]); `exact` names it and `q8` selects the earlier
    /// 8-bit-activation arithmetic for a comparison.
    fn parse_precision(value: Option<&str>) -> Result<GgufDecodePrecision, &'static str> {
        match value {
            None => Ok(GgufDecodePrecision::default()),
            Some("q8") => Ok(GgufDecodePrecision::Q8),
            Some("exact") => Ok(GgufDecodePrecision::Exact),
            Some(_) => Err("probe precision must be `q8` or `exact`"),
        }
    }
}

fn run_full_model(trace_window: Option<TraceWindow>) {
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("pinned first shard");
    assert_pinned_file(
        &path,
        "9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915",
    );
    let input = ProbeInput::open();
    let (prompt, expected) = (&input.prompt, &input.expected);
    let total = prompt.len() + input.continuation;
    let model = GgufModel::open(Path::new(&path)).unwrap().unwrap();
    let shape = model.config().model_shape().unwrap();
    let layers = model.config().layers();
    let end = match trace_window {
        Some(TraceWindow::Four | TraceWindow::StatefulFour) => 4.min(layers),
        Some(TraceWindow::Six) => 6.min(layers),
        _ => layers,
    };
    let trace_stateful = matches!(
        trace_window,
        Some(TraceWindow::StatefulFour | TraceWindow::StatefulAll)
    );
    if trace_stateful && end == layers {
        assert!(
            std::env::var_os("FIFI_FLASH_GGUF_FULL_PROBE_LOGITS_OUT").is_some(),
            "full stateful trace requires a separate logits receipt"
        );
    }
    let trace_first_token = trace_window.is_some() && !trace_stateful;
    let mut stateful = trace_stateful.then(|| {
        trace::StatefulTrace::new(
            end,
            shape.hyper.hidden() * shape.hyper.branches(),
            shape.moe.top_k(),
            shape.moe.experts(),
        )
    });
    let mut trace = trace_first_token.then(|| {
        trace::Trace::open(
            layers,
            shape.hyper.hidden() * shape.hyper.branches(),
            shape.moe.top_k(),
            shape.moe.experts(),
        )
    });
    ProbeBudget::check_host(&model).expect("probe host admission");
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let budget = ProbeBudget::new(&model, &ctx, total).expect("probe preflight");
    let trunk =
        GgufTrunk::upload_with_final_guarded(&model, &ctx, &stream, 0..end, |layer, weights| {
            budget.group_guard(&ctx, layer, weights)
        })
        .expect("guarded trunk upload");
    let ngram = trunk.model().ngram().unwrap();
    let mut feed = GgufTokenFeeds::new(trunk.model(), &ngram, total).unwrap();
    budget.before_step(&ctx).unwrap();
    let before_step = ctx.live_bytes();
    let precision = match std::env::var("FIFI_FLASH_GGUF_PROBE_PRECISION") {
        Ok(value) => ProbeInput::parse_precision(Some(&value)),
        Err(std::env::VarError::NotPresent) => ProbeInput::parse_precision(None),
        Err(_) => panic!("FIFI_FLASH_GGUF_PROBE_PRECISION is not Unicode"),
    }
    .unwrap_or_else(|message| panic!("FIFI_FLASH_GGUF_PROBE_PRECISION: {message}"));
    let mut bound = GgufStep::new(&ctx, &stream, &trunk, total).unwrap();
    bound.set_decode_precision(precision);
    assert_eq!(
        ctx.live_bytes() - before_step,
        budget.step_payload(),
        "step allocation disagrees with planned payload"
    );
    let step = bound.step();
    budget.before_bank(&model, &ctx).unwrap();
    let before_bank = (ctx.live_bytes(), ctx.live_owners().pinned);
    let mut owner = GgufMoeOwner::new(&bound, shape.moe.top_k() + 1).unwrap();
    let (device_bank, pinned_bank) = (
        ctx.live_bytes() - before_bank.0,
        ctx.live_owners().pinned - before_bank.1,
    );
    if end == layers {
        assert_eq!(
            device_bank,
            budget.bank_payload().unwrap(),
            "bank device payload differs from plan"
        );
        assert_eq!(
            pinned_bank,
            budget.pinned_bytes(),
            "bank pinned payload differs from plan"
        );
    } else {
        assert!(
            device_bank < budget.bank_payload().unwrap() && pinned_bank < budget.pinned_bytes()
        );
    }
    budget.before_execution(&ctx).unwrap();
    assert!(
        input.continuation == 0 || (trace_window.is_none() && end == layers),
        "continuation requires the untraced full model"
    );
    let mut generated: Vec<u32> = Vec::with_capacity(input.continuation);
    let (mut diverged, mut comparable_steps) = (false, 0usize);
    for position in 0..total {
        let token = ProbeInput::fed_token(prompt, &generated, position);
        let start = feed.begin(token).unwrap();
        step.begin_token().unwrap();
        let (words, at) = step.words_buffer().unwrap();
        words
            .write_at(
                &stream,
                at,
                &StepParams::new(token as i32, position as i32)
                    .unwrap()
                    .to_bytes(),
            )
            .unwrap();
        let state = step.slot(Slot::Stream).unwrap();
        let write = |values: &[f32]| {
            state
                .buffer()
                .write_at(
                    &stream,
                    state.offset(),
                    &values
                        .iter()
                        .flat_map(|v| v.to_ne_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap()
        };
        write(&start.stream);
        let mut apply_ple = || {
            let mut residual = step.read_f32(state, start.stream.len()).unwrap();
            let update = feed.ple_update(token, &residual).unwrap();
            assert_eq!(residual.len(), update.len());
            for (value, delta) in residual.iter_mut().zip(update) {
                *value += delta;
            }
            write(&residual);
        };
        let ple = shape.ple_layer;
        if trace_window.is_none() {
            if position == 0 && ple > 0 {
                let err = bound.run_layers(&mut owner, 0..end, position).unwrap_err();
                assert!(
                    matches!(err, EngError::BadRequest(message) if message.contains("apply the PLE update"))
                );
                assert!(!step.is_poisoned());
            }
            bound.run_layers(&mut owner, 0..ple, position).unwrap();
            apply_ple();
            bound.run_layers(&mut owner, ple..end, position).unwrap();
        } else {
            for layer in 0..end {
                eprintln!("full probe position={position} layer={layer} begin");
                if layer == ple {
                    apply_ple();
                }
                let mut ids = None;
                {
                    let moe = owner.bind(&bound).unwrap();
                    let mut tap = trace::RouteTap::new(moe, &mut ids);
                    step.run_layer(layer, position, 1, &mut tap).unwrap();
                }
                let routed = ids.expect("router input and IDs captured inside MoE");
                if let Some(trace) = &mut trace {
                    trace.record(step, layer, routed);
                } else if let Some(trace) = &mut stateful {
                    trace.record(step, position, token, layer, routed);
                }
                let health = step.health().unwrap();
                eprintln!("full probe position={position} layer={layer} end health={health:?}");
                assert_eq!(
                    health,
                    (0, 0),
                    "unhealthy position={position} layer={layer}"
                );
            }
        }
        if trace_first_token {
            step.finish_token().unwrap();
            trace.take().unwrap().finish(end);
            return;
        }
        if position + 1 >= prompt.len() && end == layers {
            step.head(1).unwrap();
            let got = step
                .read_f32(step.slot(Slot::Logits).unwrap(), expected.len())
                .unwrap();
            assert_eq!(got.len(), expected.len());
            let actual_greedy = greedy(&got);
            // Step 0 is the prompt's own logits; later steps follow the device's own tokens.
            let step_index = position + 1 - prompt.len();
            let host_row = &input.oracle_rows[step_index];
            let expected_greedy = greedy(host_row);
            let error = relative_l2(&got, host_row);
            let comparable =
                ProbeInput::comparable_so_far(&mut diverged, actual_greedy, expected_greedy);
            comparable_steps += usize::from(comparable);
            eprintln!(
                "greedy step={step_index} position={position} comparable={comparable} precision={precision:?} logits count={} relative_l2={error:.9} gpu_greedy={actual_greedy} host_greedy={expected_greedy} gpu_max={} host_max={}",
                got.len(),
                got[actual_greedy],
                host_row[expected_greedy]
            );
            if let Some(path) = std::env::var_os("FIFI_FLASH_GGUF_FULL_PROBE_LOGITS_OUT")
                && step_index == 0
            {
                let output = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)
                    .unwrap();
                serde_json::to_writer(output, &got).unwrap();
            }
            // A measurement by default; a tolerance is imposed only when the caller chose one.
            if let Some(bound) = input.bound {
                assert!(
                    error.is_finite() && error < bound && actual_greedy == expected_greedy,
                    "greedy step {step_index} mismatch: relative_l2={error}, gpu_greedy={actual_greedy}, host_greedy={expected_greedy}, bound={bound}"
                );
            } else {
                assert!(error.is_finite(), "non-finite logits error");
            }
            if step_index < input.continuation {
                generated.push(u32::try_from(actual_greedy).expect("greedy token id fits u32"));
            }
        }
        step.finish_token().unwrap();
    }
    if input.continuation > 0 {
        eprintln!(
            "greedy continuation: {} steps measured, {comparable_steps} comparable; each fed the device's own token; set FIFI_FLASH_GGUF_LOGITS_BOUND to assert",
            input.continuation
        );
    }
    if let Some(trace) = stateful {
        trace.finish(prompt.len());
    }
}

#[cfg(test)]
mod probe_input_tests {
    use super::ProbeInput;

    #[test]
    fn logits_bound_is_opt_in_and_validated() {
        assert_eq!(ProbeInput::parse_bound(None).unwrap(), None);
        assert_eq!(ProbeInput::parse_bound(Some("0.05")).unwrap(), Some(0.05));
        for bad in ["", "0", "-1", "nan", "inf", "abc"] {
            assert!(ProbeInput::parse_bound(Some(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn continuation_is_opt_in_and_positive() {
        assert_eq!(ProbeInput::parse_continuation(None).unwrap(), 0);
        assert_eq!(ProbeInput::parse_continuation(Some("2")).unwrap(), 2);
        for bad in ["", "0", "-1", "1.5", "abc"] {
            assert!(ProbeInput::parse_continuation(Some(bad)).is_err(), "{bad}");
        }
        assert!(ProbeInput::continuation_fits_oracle(0, 1));
        assert!(ProbeInput::continuation_fits_oracle(2, 3));
        assert!(!ProbeInput::continuation_fits_oracle(3, 3));
    }

    #[test]
    fn first_generated_position_feeds_the_prompt_final_choice() {
        let prompt = [10u32, 20, 30];
        let generated = [5854u32, 11];
        assert_eq!(ProbeInput::fed_token(&prompt, &[], 0), 10);
        assert_eq!(ProbeInput::fed_token(&prompt, &[], 2), 30);
        assert_eq!(ProbeInput::fed_token(&prompt, &generated, 3), 5854);
        assert_eq!(ProbeInput::fed_token(&prompt, &generated, 4), 11);
    }

    #[test]
    fn a_divergence_ends_comparability_even_if_a_later_step_agrees() {
        let mut diverged = false;
        let flags: Vec<bool> = [(7, 7), (9, 7), (7, 7)]
            .into_iter()
            .map(|(actual, expected)| {
                ProbeInput::comparable_so_far(&mut diverged, actual, expected)
            })
            .collect();
        assert_eq!(flags, [true, false, false]);
    }

    #[test]
    fn precision_follows_the_production_default_unless_named() {
        use super::GgufDecodePrecision;
        // Unconfigured, the probe measures what a new step runs: one definition of the default.
        assert_eq!(GgufDecodePrecision::default(), GgufDecodePrecision::Exact);
        assert_eq!(
            ProbeInput::parse_precision(None),
            Ok(GgufDecodePrecision::default())
        );
        assert_eq!(
            ProbeInput::parse_precision(Some("q8")),
            Ok(GgufDecodePrecision::Q8)
        );
        assert_eq!(
            ProbeInput::parse_precision(Some("exact")),
            Ok(GgufDecodePrecision::Exact)
        );
        for bad in ["f16", "", "Exact"] {
            assert!(ProbeInput::parse_precision(Some(bad)).is_err(), "{bad}");
        }
    }
}
