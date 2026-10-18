//! Trained-head parity and warmed prediction latency on Apple Silicon.
//!
//! `HEAD_FIXTURE_DIR` supplies the manifest and raw little-endian f32 arrays.
//! Features are already normalised. `mean.bin` and `scale.bin` are training-time
//! artefacts: their shapes and lengths are checked, but they are not applied.
//! `HeadConfig` rejects zero classes, so one zero row is appended to `w2`,
//! with one zero bias appended to `b2`, for a single unused class.
//!
//! The bridge converts inputs and weights to fp16 and executes fp16 operations.
//! Its unit roundoff is 2^-11, approximately 4.9e-4. At the fixture's scale,
//! the supplied tolerance rationale allows several such rounding units across
//! the 2048-term dot products and the second layer. The 0.05 absolute threshold
//! adds loose headroom; it is a fixture-scale sanity threshold, not a proven
//! worst-case bound. Decision agreement additionally requires top-8 expert-set
//! overlap of at least 7 per vector and a mean of at least 7.5 over 32 vectors.
//!
//! All 32 parity predictions warm each backend before 200 timed predictions.
//! Construction is excluded. Sorted latency index `floor(N * q)` defines each
//! reported percentile; timings include the public prediction call and readout.
//!
//! `ane_head_dense_stage` prints all 255 selected hidden activations.
//! Its CPU oracle rounds operands to fp16 and accumulates and evaluates SiLU
//! in f64. It does not round intermediate results to fp16.
//! Reduction differences are ANE minus oracle. `max_abs_operand` reports the
//! largest operand magnitude before rounding.

#![cfg(all(feature = "native-ane", target_os = "macos", target_arch = "aarch64"))]

use std::{fs, hint::black_box, path::Path, time::Instant};

use serde::Deserialize;

use super::{DecisionHead, HeadConfig, HeadWeights};

const INPUTS: usize = 32;
const TIMED_CALLS: usize = 200;
const LOGIT_TOLERANCE: f32 = 5.0e-2;

#[derive(Deserialize)]
struct Array {
    file: String,
    shape: Vec<usize>,
    bytes: usize,
}

#[derive(Deserialize)]
struct Manifest {
    input: usize,
    hidden: usize,
    experts: usize,
    classes: usize,
    dtype: String,
    endianness: String,
    w1: Array,
    b1: Array,
    w2: Array,
    b2: Array,
    mean: Array,
    scale: Array,
}

fn read_array(directory: &Path, array: &Array, shape: &[usize]) -> Vec<f32> {
    assert_eq!(
        array.shape, shape,
        "{}: manifest shape does not match head dimensions",
        array.file
    );
    let elements = array
        .shape
        .iter()
        .try_fold(1usize, |n, &dimension| n.checked_mul(dimension))
        .unwrap_or_else(|| panic!("{}: shape element count overflow", array.file));
    let expected_bytes = elements
        .checked_mul(size_of::<f32>())
        .unwrap_or_else(|| panic!("{}: shape byte count overflow", array.file));
    assert_eq!(
        array.bytes, expected_bytes,
        "{}: manifest declares {} bytes, shape requires {} bytes",
        array.file, array.bytes, expected_bytes
    );
    let path = directory.join(&array.file);
    let bytes = fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(
        bytes.len(),
        expected_bytes,
        "{}: file contains {} bytes, shape requires {} bytes",
        array.file,
        bytes.len(),
        expected_bytes
    );
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&value| f32::from_le_bytes(value))
        .collect()
}

fn read_weights(directory: &Path) -> HeadWeights {
    let path = directory.join("manifest.json");
    let bytes = fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let manifest: Manifest = serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(manifest.dtype, "f32", "fixture dtype");
    assert_eq!(manifest.endianness, "little", "fixture endianness");
    assert!(
        manifest.classes <= 1,
        "the fixture carries at most one class row"
    );
    assert_eq!(manifest.input, 2048, "fixture input width");
    assert!(
        manifest.experts >= 8,
        "fixture needs at least eight experts"
    );
    let config = HeadConfig {
        input: manifest.input,
        hidden: manifest.hidden,
        experts: manifest.experts,
        classes: 1,
    };
    let w1 = read_array(directory, &manifest.w1, &[config.hidden, config.input]);
    let b1 = read_array(directory, &manifest.b1, &[config.hidden]);
    let rows = config.experts + manifest.classes;
    let mut w2 = read_array(directory, &manifest.w2, &[rows, config.hidden]);
    let mut b2 = read_array(directory, &manifest.b2, &[rows]);
    read_array(directory, &manifest.mean, &[1, config.input]);
    read_array(directory, &manifest.scale, &[1, config.input]);
    if manifest.classes == 0 {
        w2.extend(std::iter::repeat_n(0.0, config.hidden));
        b2.push(0.0);
    }
    HeadWeights::try_new(&config, w1, b1, w2, b2).expect("trained fixture weights must be valid")
}

fn normalised_inputs(input: usize) -> Vec<Vec<f32>> {
    let mut state = 0x6d2b79f5u32;
    (0..INPUTS)
        .map(|_| {
            (0..input)
                .map(|_| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let u = (state >> 8) as f32 / 16777216.0;
                    (2.0 * u - 1.0) * 2.0
                })
                .collect()
        })
        .collect()
}

fn top8(logits: &[f32]) -> Vec<usize> {
    let mut order: Vec<_> = (0..logits.len()).collect();
    order.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b)));
    order.truncate(8);
    order
}

fn agreement(cpu: &DecisionHead, ane: &DecisionHead, inputs: &[Vec<f32>]) -> (f32, usize, f64) {
    let mut max_abs_diff = 0.0f32;
    let mut min_overlap = 8;
    let mut total_overlap = 0;
    for (index, features) in inputs.iter().enumerate() {
        let expected = cpu
            .predict(features)
            .expect("CPU trained prediction failed");
        let actual = ane
            .predict(features)
            .expect("ANE trained prediction failed");
        assert_eq!(actual.expert_logits.len(), expected.expert_logits.len());
        assert_eq!(actual.class_logits.len(), expected.class_logits.len());
        let actual_logits = actual.expert_logits.iter().chain(&actual.class_logits);
        let expected_logits = expected.expert_logits.iter().chain(&expected.class_logits);
        for (logit, (&value, &reference)) in actual_logits.zip(expected_logits).enumerate() {
            assert!(
                value.is_finite() && reference.is_finite(),
                "input {index}, logit {logit}: ANE {value}, CPU {reference} must be finite"
            );
            let diff = (value - reference).abs();
            assert!(
                diff <= LOGIT_TOLERANCE,
                "input {index}, logit {logit}: ANE {value}, CPU {reference}, difference {diff}"
            );
            max_abs_diff = max_abs_diff.max(diff);
        }
        let expected_top8 = top8(&expected.expert_logits);
        let overlap = top8(&actual.expert_logits)
            .iter()
            .filter(|expert| expected_top8.contains(expert))
            .count();
        assert!(
            overlap >= 7,
            "input {index}: top-8 expert overlap is {overlap}, requires at least 7"
        );
        min_overlap = min_overlap.min(overlap);
        total_overlap += overlap;
    }
    let mean_overlap = total_overlap as f64 / inputs.len() as f64;
    assert!(
        mean_overlap >= 7.5,
        "mean top-8 expert overlap is {mean_overlap}, requires at least 7.5"
    );
    (max_abs_diff, min_overlap, mean_overlap)
}

fn latency_ms(head: &DecisionHead, inputs: &[Vec<f32>]) -> (f64, f64) {
    let mut times = Vec::with_capacity(TIMED_CALLS);
    for i in 0..TIMED_CALLS {
        let features = black_box(inputs[i % inputs.len()].as_slice());
        let start = Instant::now();
        let decision = head.predict(features).expect("timed prediction failed");
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        black_box(decision);
    }
    times.sort_unstable_by(f64::total_cmp);
    (times[TIMED_CALLS / 2], times[TIMED_CALLS * 99 / 100])
}

#[test]
#[ignore = "requires HEAD_FIXTURE_DIR and Apple Silicon ANE; run in release mode"]
fn ane_head_matches_cpu_on_trained_weights() {
    let directory = std::env::var_os("HEAD_FIXTURE_DIR")
        .expect("HEAD_FIXTURE_DIR must name the trained head fixture directory");
    let weights = read_weights(Path::new(&directory));
    let inputs = normalised_inputs(weights.config().input);
    let cpu = DecisionHead::cpu(weights.clone()).expect("CPU trained head construction failed");
    let ane = DecisionHead::ane(weights).unwrap_or_else(|error| match error {
        crate::EngError::Backend { message, .. } => panic!("{message}"),
        other => panic!("{other:?}"),
    });
    let (max_abs_diff, min_overlap, mean_overlap) = agreement(&cpu, &ane, &inputs);
    let (cpu_p50, _) = latency_ms(&cpu, &inputs);
    let (ane_p50, ane_p99) = latency_ms(&ane, &inputs);
    println!(
        concat!(
            "ane_head_trained={{\"inputs\":{},\"max_abs_diff\":{},",
            "\"min_top8_overlap\":{},\"mean_top8_overlap\":{},",
            "\"cpu_p50_ms\":{},\"ane_p50_ms\":{},\"ane_p99_ms\":{}}}"
        ),
        INPUTS, max_abs_diff, min_overlap, mean_overlap, cpu_p50, ane_p50, ane_p99
    );
}

fn shape_probe_weights(config: &HeadConfig, case: usize) -> HeadWeights {
    let outputs = config.experts + config.classes;
    let mut w1 = vec![0.0; config.hidden * config.input];
    let mut b1 = vec![0.0; config.hidden];
    let mut w2 = vec![0.0; outputs * config.hidden];
    let mut b2 = vec![0.0; outputs];
    match case {
        1 => {
            // Zero activations leave only the exactly representable output biases.
            for (output, bias) in b2.iter_mut().enumerate() {
                *bias = (output + 1) as f32;
            }
        }
        2 => {
            // Every hidden activation is SiLU(1); every output selects hidden 0.
            b1.fill(1.0);
            for output in 0..outputs {
                w2[output * config.hidden] = 1.0;
            }
        }
        3 => {
            // With x[0] = 1, only hidden 0 receives a nonzero first-layer dot product.
            w1[0] = 1.0;
            for output in 0..outputs {
                w2[output * config.hidden] = 1.0;
            }
        }
        4 => {
            // The expert boundary at 128 separates SiLU(1) from 2 * SiLU(1).
            // The unused class remains zero.
            b1.fill(1.0);
            for output in 0..config.experts {
                let (hidden, weight) = if output < 128 { (0, 1.0) } else { (1, 2.0) };
                w2[output * config.hidden + hidden] = weight;
            }
        }
        5 | 6 => {
            // Case 5 expects SiLU(1/256); a +64 shift gives SiLU(65/256).
            // Case 6 expects SiLU(65/256); a shift gives SiLU(129/256) or SiLU(1/256).
            for (hidden, bias) in b1.iter_mut().enumerate() {
                *bias = (hidden + 1) as f32 / 256.0;
            }
            let hidden = if case == 5 { 0 } else { 64 };
            for output in 0..outputs {
                w2[output * config.hidden + hidden] = 1.0;
            }
        }
        7 => {
            // Separate LCG seeds: weights 0xa341316c, input 0x6d2b79f5.
            let mut state = 0xa341316cu32;
            for weight in w1.iter_mut().chain(&mut w2) {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (state >> 8) as f32 / 16777216.0;
                *weight = 2.0 * u - 1.0;
            }
        }
        _ => panic!("unknown shape probe case {case}"),
    }
    HeadWeights::try_new(config, w1, b1, w2, b2).expect("shape probe weights must be valid")
}

#[test]
#[ignore = "requires Apple Silicon ANE; diagnostic only, with no parity assertions"]
fn ane_head_shape_probe() {
    let config = HeadConfig {
        input: 2048,
        hidden: 256,
        experts: 256,
        classes: 1,
    };
    let mut features = vec![0.0; config.input];
    features[0] = 1.0;
    for case in 1..=7 {
        if case == 7 {
            let mut state = 0x6d2b79f5u32;
            for feature in &mut features {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (state >> 8) as f32 / 16777216.0;
                *feature = 2.0 * u - 1.0;
            }
        }
        let weights = shape_probe_weights(&config, case);
        let cpu = DecisionHead::cpu(weights.clone())
            .expect("CPU shape probe construction failed")
            .predict(&features)
            .expect("CPU shape probe prediction failed");
        // Keep later cases observable even when ANE construction or prediction fails.
        let ane = DecisionHead::ane(weights).and_then(|head| head.predict(&features));
        match ane {
            Ok(ane) => {
                let extra = if case == 7 {
                    let max_abs_diff = cpu
                        .expert_logits
                        .iter()
                        .chain(&cpu.class_logits)
                        .zip(ane.expert_logits.iter().chain(&ane.class_logits))
                        .map(|(&expected, &actual)| (actual - expected).abs())
                        .fold(0.0f32, f32::max);
                    format!(",\"max_abs_diff\":{max_abs_diff}")
                } else {
                    String::new()
                };
                println!(
                    concat!(
                        "ane_head_probe={{\"case\":{},\"cpu0\":{},\"ane0\":{},",
                        "\"cpu127\":{},\"ane127\":{},\"cpu128\":{},\"ane128\":{},",
                        "\"cpu255\":{},\"ane255\":{},\"cpu_last\":{},\"ane_last\":{}{}}}"
                    ),
                    case,
                    cpu.expert_logits[0],
                    ane.expert_logits[0],
                    cpu.expert_logits[127],
                    ane.expert_logits[127],
                    cpu.expert_logits[128],
                    ane.expert_logits[128],
                    cpu.expert_logits[255],
                    ane.expert_logits[255],
                    // The final combined logit is the single unused class, index 256.
                    cpu.class_logits[0],
                    ane.class_logits[0],
                    extra,
                );
            }
            Err(error) => println!("ane_head_probe case {case} failed: {error:?}"),
        }
    }
}

fn alias_probe_weights(config: &HeadConfig, j: usize, jp: usize) -> HeadWeights {
    let outputs = config.experts + config.classes;
    let w1 = vec![0.0; config.hidden * config.input];
    let mut b1 = vec![0.0; config.hidden];
    let mut w2 = vec![0.0; outputs * config.hidden];
    let b2 = vec![0.0; outputs];
    b1[j] = 1.0;
    for output in 0..outputs {
        w2[output * config.hidden + jp] = 1.0;
    }
    HeadWeights::try_new(config, w1, b1, w2, b2).expect("alias probe weights must be valid")
}

#[test]
#[ignore = "requires Apple Silicon ANE; diagnostic only, with no parity assertions"]
fn ane_head_alias_matrix() {
    let config = HeadConfig {
        input: 2048,
        hidden: 256,
        experts: 256,
        classes: 1,
    };
    let indices = [0, 1, 63, 64, 128, 255];
    let features = vec![0.0; config.input];
    let target = 0.731_058_6f32;
    let mut closest = [None::<(usize, f32)>; 6];
    for j in indices {
        for (column, jp) in indices.into_iter().enumerate() {
            let weights = alias_probe_weights(&config, j, jp);
            let cpu = DecisionHead::cpu(weights.clone())
                .expect("CPU alias probe construction failed")
                .predict(&features)
                .expect("CPU alias probe prediction failed")
                .expert_logits[0];
            let ane = DecisionHead::ane(weights)
                .and_then(|head| head.predict(&features))
                .ok()
                .map(|decision| decision.expert_logits[0])
                .filter(|value| value.is_finite());
            if let Some(value) = ane
                && closest[column]
                    .is_none_or(|(_, previous)| (value - target).abs() < (previous - target).abs())
            {
                closest[column] = Some((j, value));
            }
            let ane = ane.map_or_else(|| "null".to_owned(), |value| value.to_string());
            println!("ane_head_alias={{\"j\":{j},\"jp\":{jp},\"cpu\":{cpu},\"ane\":{ane}}}");
        }
    }
    for (jp, closest) in indices.into_iter().zip(closest) {
        match closest {
            Some((j, value)) => {
                println!("ane_head_alias_map={{\"jp\":{jp},\"reads_j\":{j},\"value\":{value}}}")
            }
            None => println!("ane_head_alias_map={{\"jp\":{jp},\"reads_j\":null,\"value\":null}}"),
        }
    }
}

fn input_alias_probe_weights(config: &HeadConfig, jp: usize) -> HeadWeights {
    let outputs = config.experts + config.classes;
    let mut w1 = vec![0.0; config.hidden * config.input];
    let b1 = vec![0.0; config.hidden];
    let mut w2 = vec![0.0; outputs * config.hidden];
    let b2 = vec![0.0; outputs];
    for hidden in 0..config.hidden {
        w1[hidden * config.input + hidden] = 1.0;
    }
    for output in 0..outputs {
        w2[output * config.hidden + jp] = 1.0;
    }
    HeadWeights::try_new(config, w1, b1, w2, b2).expect("input alias probe weights must be valid")
}

#[test]
#[ignore = "requires Apple Silicon ANE; diagnostic only, with no parity assertions"]
fn ane_head_input_alias() {
    let config = HeadConfig {
        input: 2048,
        hidden: 256,
        experts: 256,
        classes: 1,
    };
    let indices = [0, 1, 63, 64, 128, 255];
    let target = 0.731_058_6f32;
    for jx in indices {
        let mut features = vec![0.0; config.input];
        features[jx] = 1.0;
        let mut closest = None::<(usize, f32)>;
        for jp in indices {
            let weights = input_alias_probe_weights(&config, jp);
            let cpu = DecisionHead::cpu(weights.clone())
                .expect("CPU input alias probe construction failed")
                .predict(&features)
                .expect("CPU input alias probe prediction failed")
                .expert_logits[0];
            let ane = DecisionHead::ane(weights)
                .and_then(|head| head.predict(&features))
                .ok()
                .map(|decision| decision.expert_logits[0])
                .filter(|value| value.is_finite());
            if let Some(value) = ane
                && closest
                    .is_none_or(|(_, previous)| (value - target).abs() < (previous - target).abs())
            {
                closest = Some((jp, value));
            }
            let ane = ane.map_or_else(|| "null".to_owned(), |value| value.to_string());
            println!(
                "ane_head_input_alias={{\"jx\":{jx},\"jp\":{jp},\"cpu\":{cpu},\"ane\":{ane}}}"
            );
        }
        match closest {
            Some((jp, value)) => {
                println!("ane_head_input_map={{\"jx\":{jx},\"feeds_jp\":{jp},\"value\":{value}}}")
            }
            None => println!("ane_head_input_map={{\"jx\":{jx},\"feeds_jp\":null,\"value\":null}}"),
        }
    }
}

#[test]
#[ignore = "requires Apple Silicon ANE; diagnostic only, with no parity assertions"]
fn ane_head_silu_sweep() {
    let config = HeadConfig {
        input: 2048,
        hidden: 256,
        experts: 256,
        classes: 1,
    };
    let outputs = config.experts + config.classes;
    let features = vec![0.0; config.input];
    let mut bad = Vec::new();
    for c in [
        0.001f32, 0.004, 0.01, 0.05, 0.25, 1.0, 2.0, 4.0, 8.0, 16.0, -0.5, -4.0,
    ] {
        let mut w2 = vec![0.0; outputs * config.hidden];
        for output in 0..outputs {
            w2[output * config.hidden] = 1.0;
        }
        let weights = HeadWeights::try_new(
            &config,
            vec![0.0; config.hidden * config.input],
            vec![c; config.hidden],
            w2,
            vec![0.0; outputs],
        )
        .expect("SiLU sweep weights must be valid");
        let cpu = DecisionHead::cpu(weights.clone())
            .expect("CPU SiLU sweep construction failed")
            .predict(&features)
            .expect("CPU SiLU sweep prediction failed")
            .expert_logits[0];
        let ane = DecisionHead::ane(weights)
            .and_then(|head| head.predict(&features))
            .expect("ANE SiLU sweep failed")
            .expert_logits[0];
        let ratio = if cpu != 0.0 {
            (ane / cpu).to_string()
        } else {
            "null".to_owned()
        };
        println!("ane_head_silu={{\"c\":{c},\"cpu\":{cpu},\"ane\":{ane},\"ratio\":{ratio}}}");
        if (ane - cpu).abs() > 0.05 * cpu.abs() {
            bad.push(c);
        }
    }
    println!(
        "ane_head_silu_bad={}",
        serde_json::json!({"count": bad.len(), "values": bad})
    );
}

fn dense_stage_fp16(value: f32) -> f64 {
    // Fixture operands are finite and in [-1, 1], within the expected [-8, 8] domain.
    // Round to the binary16 grid, including its subnormal spacing, with ties to even.
    let exponent = ((value.to_bits() >> 23) & 0xff) as i32 - 127;
    let quantum = 2.0f32.powi((exponent - 10).max(-24));
    f64::from((value / quantum).round_ties_even() * quantum)
}

fn dense_stage_oracle(weights: &HeadWeights, features: &[f32]) -> (Vec<f64>, [f64; 2]) {
    let config = weights.config();
    let features: Vec<_> = features.iter().map(|&x| dense_stage_fp16(x)).collect();
    let hidden: Vec<_> = weights
        .w1
        .chunks_exact(config.input)
        .zip(&weights.b1)
        .map(|(row, &bias)| {
            let acc = row
                .iter()
                .zip(&features)
                .fold(dense_stage_fp16(bias), |acc, (&weight, &x)| {
                    acc + dense_stage_fp16(weight) * x
                });
            acc / (1.0 + (-acc).exp())
        })
        .collect();
    let reductions = [255, 256].map(|output| {
        weights.w2[output * config.hidden..(output + 1) * config.hidden]
            .iter()
            .zip(&hidden)
            .fold(
                dense_stage_fp16(weights.b2[output]),
                |acc, (&weight, &h)| acc + dense_stage_fp16(weight) * h,
            )
    });
    (hidden, reductions)
}

#[test]
#[ignore = "requires Apple Silicon ANE; diagnostic only, with no parity assertions"]
fn ane_head_dense_stage() {
    let config = HeadConfig {
        input: 2048,
        hidden: 256,
        experts: 256,
        classes: 1,
    };
    let mut weights = shape_probe_weights(&config, 7);
    let signed = weights.w2[..config.hidden].to_vec();
    weights.w2.fill(0.0);
    for (h, &weight) in signed.iter().take(255).enumerate() {
        weights.w2[h * config.hidden + h] = 1.0;
        weights.w2[255 * config.hidden + h] = weight;
        weights.w2[256 * config.hidden + h] = weight.abs();
    }
    // Both reduction rows omit hidden 255; all output biases remain zero.
    let mut state = 0x6d2b79f5u32;
    let features: Vec<_> = (0..config.input)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = (state >> 8) as f32 / 16777216.0;
            2.0 * u - 1.0
        })
        .collect();
    let max_abs_operand = weights
        .w1
        .iter()
        .chain(&weights.b1)
        .chain(&weights.w2)
        .chain(&weights.b2)
        .chain(&features)
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let (hidden, reductions) = dense_stage_oracle(&weights, &features);
    let ane = DecisionHead::ane(weights)
        .and_then(|head| head.predict(&features))
        .expect("ANE dense stage failed");
    let mut max_abs_diff_hidden = 0.0f64;
    let mut at_h = 0;
    for (h, &cpu16) in hidden.iter().take(255).enumerate() {
        let actual = ane.expert_logits[h];
        let diff = (f64::from(actual) - cpu16).abs();
        if diff.total_cmp(&max_abs_diff_hidden).is_gt() {
            max_abs_diff_hidden = diff;
            at_h = h;
        }
        println!("ane_head_dense_stage={{\"h\":{h},\"cpu16\":{cpu16},\"ane\":{actual}}}");
    }
    for ((which, cpu16), actual) in ["signed", "abs"]
        .into_iter()
        .zip(reductions)
        .zip([ane.expert_logits[255], ane.class_logits[0]])
    {
        println!(
            "ane_head_dense_reduce={{\"which\":\"{which}\",\"cpu16\":{cpu16},\"ane\":{actual}}}"
        );
    }
    let signed_diff = f64::from(ane.expert_logits[255]) - reductions[0];
    let abs_diff = f64::from(ane.class_logits[0]) - reductions[1];
    println!(
        concat!(
            "ane_head_dense_summary={{\"max_abs_diff_hidden\":{},\"at_h\":{},",
            "\"signed_diff\":{},\"abs_diff\":{},\"max_abs_operand\":{}}}"
        ),
        max_abs_diff_hidden, at_h, signed_diff, abs_diff, max_abs_operand
    );
}

#[test]
#[ignore = "requires Apple Silicon ANE; diagnostic only, with no parity assertions"]
fn ane_head_dense_stage_padded() {
    let dense_config = HeadConfig {
        input: 2048,
        hidden: 256,
        experts: 256,
        classes: 1,
    };
    let config = HeadConfig {
        hidden: 512,
        ..dense_config
    };
    let mut dense = shape_probe_weights(&dense_config, 7);
    let signed = dense.w2[..dense_config.hidden].to_vec();
    dense.w1.resize(config.hidden * config.input, 0.0);
    dense.b1.resize(config.hidden, 0.0);
    let mut w2 = vec![0.0; (config.experts + config.classes) * config.hidden];
    for (h, &weight) in signed.iter().take(255).enumerate() {
        w2[h * config.hidden + h] = 1.0;
        w2[255 * config.hidden + h] = weight;
        w2[256 * config.hidden + h] = weight.abs();
    }
    // Both reductions omit hidden 255; all padded columns and output biases stay zero.
    let weights = HeadWeights::try_new(&config, dense.w1, dense.b1, w2, dense.b2)
        .expect("padded dense stage weights must be valid");
    // The first shared input uses the same seed, at twice the dense probe's scale.
    let features: Vec<_> = normalised_inputs(config.input)[0]
        .iter()
        .map(|&value| value * 0.5)
        .collect();
    let max_abs_operand = weights
        .w1
        .iter()
        .chain(&weights.b1)
        .chain(&weights.w2)
        .chain(&weights.b2)
        .chain(&features)
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let (hidden, reductions) = dense_stage_oracle(&weights, &features);
    let ane = DecisionHead::ane(weights)
        .and_then(|head| head.predict(&features))
        .expect("ANE padded dense stage failed");
    let mut max_abs_diff_hidden = 0.0f64;
    let mut at_h = 0;
    for (h, &cpu16) in hidden.iter().take(255).enumerate() {
        let actual = ane.expert_logits[h];
        let diff = (f64::from(actual) - cpu16).abs();
        if diff.total_cmp(&max_abs_diff_hidden).is_gt() {
            max_abs_diff_hidden = diff;
            at_h = h;
        }
        println!("ane_head_dense_padded={{\"h\":{h},\"cpu16\":{cpu16},\"ane\":{actual}}}");
    }
    for ((which, cpu16), actual) in ["signed", "abs"]
        .into_iter()
        .zip(reductions)
        .zip([ane.expert_logits[255], ane.class_logits[0]])
    {
        println!(
            "ane_head_dense_padded_reduce={{\"which\":\"{which}\",\"cpu16\":{cpu16},\"ane\":{actual}}}"
        );
    }
    let signed_diff = f64::from(ane.expert_logits[255]) - reductions[0];
    let abs_diff = f64::from(ane.class_logits[0]) - reductions[1];
    println!(
        concat!(
            "ane_head_dense_padded_summary={{\"max_abs_diff_hidden\":{},\"at_h\":{},",
            "\"signed_diff\":{},\"abs_diff\":{},\"max_abs_operand\":{}}}"
        ),
        max_abs_diff_hidden, at_h, signed_diff, abs_diff, max_abs_operand
    );
}
