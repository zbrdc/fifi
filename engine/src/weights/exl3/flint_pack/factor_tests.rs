//! The routed-expert compensation: where the factor is read from, how it is refused, and what
//! ignoring it does to the routed contribution.

use half::f16;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{
    RoutedScale,
    fixture::{PINNED_CONFIG, PINNED_CONFIG_SHA256, expect_refusal},
};
use crate::{
    EngError,
    family::{Routed, flint::math::route},
    weights::{exl3::Config, sha256_hex},
};

fn pinned() -> Value {
    serde_json::from_slice(PINNED_CONFIG).unwrap()
}

fn nested(config: &mut Value) -> &mut serde_json::Map<String, Value> {
    config["quantization_config"].as_object_mut().unwrap()
}

fn encode(config: &Value) -> Vec<u8> {
    serde_json::to_vec(config).unwrap()
}

fn pinned_text() -> &'static str {
    std::str::from_utf8(PINNED_CONFIG).unwrap()
}

/// Replaces `from` once; a missed pattern would make a duplicate-key control vacuous.
fn rewrite(from: &str, to: &str) -> Vec<u8> {
    assert_eq!(pinned_text().matches(from).count(), 1, "{from}");
    pinned_text().replace(from, to).into_bytes()
}

#[test]
fn the_pinned_config_states_128_under_quantization_config() {
    let digest: [u8; 32] = Sha256::digest(PINNED_CONFIG).into();
    assert_eq!(sha256_hex(&digest), PINNED_CONFIG_SHA256);
    assert!(pinned().get("expert_interm_div").is_none(), "not top level");
    assert_eq!(
        Config::from_json(PINNED_CONFIG)
            .unwrap()
            .expert_interm_div(),
        128
    );
    let scale = RoutedScale::from_model_config(PINNED_CONFIG).unwrap();
    assert_eq!(scale.factor(), 128.0);
}

#[test]
fn a_missing_factor_is_refused_though_the_generic_reader_defaults_to_one() {
    let mut config = pinned();
    nested(&mut config).remove("expert_interm_div");
    let bytes = encode(&config);
    assert_eq!(Config::from_json(&bytes).unwrap().expert_interm_div(), 1);
    expect_refusal(
        RoutedScale::from_model_config(&bytes),
        "expert_interm_div is missing from quantization_config",
    );
}

#[test]
fn the_factor_is_only_read_from_quantization_config() {
    let mut top_only = pinned();
    nested(&mut top_only).remove("expert_interm_div");
    top_only["expert_interm_div"] = json!(128);
    let mut both = pinned();
    both["expert_interm_div"] = json!(128);
    for config in [top_only, both] {
        expect_refusal(
            RoutedScale::from_model_config(&encode(&config)),
            "must be nested in quantization_config",
        );
    }
    let mut wrong_object = pinned();
    nested(&mut wrong_object).remove("expert_interm_div");
    wrong_object["text_config"] = json!({"expert_interm_div": 128});
    expect_refusal(
        RoutedScale::from_model_config(&encode(&wrong_object)),
        "is missing from quantization_config",
    );
}

#[test]
fn only_an_integer_128_is_accepted() {
    for value in [
        json!(0),
        json!(1),
        json!(64),
        json!(127),
        json!(129),
        json!(256),
        json!(-128),
        json!(128.0),
        json!(127.9),
        json!("128"),
        json!(null),
        json!(true),
        json!([128]),
        json!({"value": 128}),
        json!(u64::MAX),
        json!(1e300),
    ] {
        let mut config = pinned();
        nested(&mut config).insert("expert_interm_div".into(), value.clone());
        let result = RoutedScale::from_model_config(&encode(&config));
        assert!(
            matches!(&result, Err(EngError::Checkpoint(m)) if m.contains("expert_interm_div")),
            "{value}: {result:?}"
        );
    }
    let mut one = pinned();
    nested(&mut one).insert("expert_interm_div".into(), json!(1));
    expect_refusal(
        RoutedScale::from_model_config(&encode(&one)),
        "expert_interm_div must be 128",
    );
}

#[test]
fn duplicate_members_are_refused() {
    let member = "\"expert_interm_div\": 128";
    for bytes in [
        rewrite(
            member,
            "\"expert_interm_div\": 128, \"expert_interm_div\": 1",
        ),
        rewrite(
            member,
            "\"expert_interm_div\": 1, \"expert_interm_div\": 128",
        ),
        rewrite(
            "\"quantization_config\": {",
            "\"quantization_config\": {\"expert_interm_div\": 1}, \"quantization_config\": {",
        ),
    ] {
        expect_refusal(
            RoutedScale::from_model_config(&bytes),
            "duplicate JSON member",
        );
    }
}

#[test]
fn the_config_read_is_bounded_and_must_be_a_pinned_exl3_config() {
    let mut oversized = PINNED_CONFIG.to_vec();
    oversized.resize(1_048_577, b' ');
    assert!(matches!(
        RoutedScale::from_model_config(&oversized),
        Err(EngError::MemoryPlan(_))
    ));
    let mut gptq = pinned();
    nested(&mut gptq).insert("quant_method".into(), json!("gptq"));
    expect_refusal(
        RoutedScale::from_model_config(&encode(&gptq)),
        "quant_method must be exl3",
    );
}

fn logits() -> Vec<f32> {
    (0..256)
        .map(|i| ((i * 37) % 256) as f32 / 32.0 - 4.0)
        .collect()
}

#[test]
fn compensation_multiplies_the_normalized_route_by_128_and_nothing_else() {
    let routed = route(&logits()).unwrap();
    let before = routed.clone();
    let sum: f32 = routed.weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "the shared router stays normalized"
    );
    let scaled = RoutedScale::from_model_config(PINNED_CONFIG)
        .unwrap()
        .compensate(&routed)
        .unwrap();
    assert_eq!(routed, before);
    assert_eq!(scaled.indices, routed.indices);
    assert_eq!(scaled.indices.len(), 6);
    for (got, normalized) in scaled.weights.iter().zip(&routed.weights) {
        assert_eq!(got.to_bits(), (normalized * 128.0).to_bits());
    }
    let scaled_sum: f64 = scaled.weights.iter().map(|w| f64::from(*w)).sum();
    assert!((scaled_sum - 128.0).abs() < 1e-4, "{scaled_sum}");
}

#[test]
fn compensation_refuses_malformed_routes() {
    let scale = RoutedScale::from_model_config(PINNED_CONFIG).unwrap();
    let ragged = Routed {
        indices: vec![0, 1],
        weights: vec![1.0],
    };
    assert!(matches!(
        scale.compensate(&ragged),
        Err(EngError::BadRequest(m)) if m.contains("indices")
    ));
    for bad in [f32::NAN, f32::INFINITY, -0.25, f32::MAX] {
        let route = Routed {
            indices: vec![3],
            weights: vec![bad],
        };
        assert!(
            matches!(
                scale.compensate(&route),
                Err(EngError::Backend {
                    operation: "route_experts",
                    ..
                })
            ),
            "{bad}"
        );
    }
}

const D: usize = 8;
const F: usize = 4;

struct Ffn {
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

fn pseudo(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / 16_777_216.0 - 0.5) * 2.0
        })
        .collect()
}

fn ffn(id: u64) -> Ffn {
    Ffn {
        gate: pseudo(id * 3 + 1, F * D),
        up: pseudo(id * 3 + 2, F * D),
        down: pseudo(id * 3 + 3, D * F),
    }
}

/// The stored form: `up_proj` divided by the pack's factor, exact for a power of two.
fn stored(mut ffn: Ffn, divisor: f32) -> Ffn {
    ffn.up.iter_mut().for_each(|w| *w /= divisor);
    ffn
}

fn matvec(matrix: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|r| (0..cols).map(|c| matrix[r * cols + c] * x[c]).sum())
        .collect()
}

fn run(ffn: &Ffn, x: &[f32]) -> Vec<f32> {
    let gate = matvec(&ffn.gate, F, D, x);
    let up = matvec(&ffn.up, F, D, x);
    let hidden: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    matvec(&ffn.down, D, F, &hidden)
}

fn routed_output(route: &Routed, expert: impl Fn(u32) -> Ffn, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; D];
    for (id, weight) in route.indices.iter().zip(&route.weights) {
        for (o, v) in out.iter_mut().zip(run(&expert(*id), x)) {
            *o += weight * v;
        }
    }
    out
}

fn assert_close(actual: &[f32], expected: &[f32], relative: f32) {
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= relative * e.abs() + 1e-9,
            "element {i}: {a} vs {e}"
        );
    }
}

#[test]
fn ignoring_the_factor_scales_the_routed_contribution_to_one_128th_not_the_whole_moe() {
    let x = pseudo(77, D);
    let route = route(&logits()).unwrap();
    let scale = RoutedScale::from_model_config(PINNED_CONFIG).unwrap();
    let compensated = scale.compensate(&route).unwrap();
    let divided = |id: u32| stored(ffn(u64::from(id)), 128.0);
    let truth = routed_output(&route, |id| ffn(u64::from(id)), &x);
    assert_close(&routed_output(&compensated, divided, &x), &truth, 1e-6);

    // Ignoring the factor: the routed term is exactly 1/128 of the truth.
    let ignored = routed_output(&route, divided, &x);
    let truth_128th: Vec<f32> = truth.iter().map(|v| v / 128.0).collect();
    assert_close(&ignored, &truth_128th, 1e-6);

    // The unchanged shared expert is added afterwards: the whole MoE is not 1/128 of anything.
    let shared = run(&ffn(9_999), &x);
    let whole =
        |routed: &[f32]| -> Vec<f32> { shared.iter().zip(routed).map(|(s, r)| s + r).collect() };
    let (whole_truth, whole_ignored) = (whole(&truth), whole(&ignored));
    let isolated: Vec<f32> = whole_ignored
        .iter()
        .zip(&shared)
        .map(|(w, s)| w - s)
        .collect();
    // The sum with the O(1) shared output rounds at about 1e-7, far above the 1e-9 routed term
    // scale: compare absolutely.
    for (got, want) in isolated.iter().zip(&truth_128th) {
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }
    let wrongly_expected: Vec<f32> = whole_truth.iter().map(|v| v / 128.0).collect();
    let gap = whole_ignored
        .iter()
        .zip(&wrongly_expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        gap > 1e-3,
        "a whole-MoE 1/128 claim must be visibly wrong: {gap}"
    );
}

#[test]
fn a_dead_selected_expert_contributes_zero_and_poisons_nothing() {
    let x = pseudo(5, D);
    let route = route(&logits()).unwrap();
    let scale = RoutedScale::from_model_config(PINNED_CONFIG).unwrap();
    let compensated = scale.compensate(&route).unwrap();
    let dead = route.indices[2];
    let expert = |id: u32| {
        let mut f = stored(ffn(u64::from(id)), 128.0);
        if id == dead {
            f.down.fill(0.0);
        }
        f
    };
    let out = routed_output(&compensated, expert, &x);
    assert!(out.iter().all(|v| v.is_finite()));
    let mut without = compensated.clone();
    let position = route.indices.iter().position(|&i| i == dead).unwrap();
    without.indices.remove(position);
    without.weights.remove(position);
    let others = routed_output(&without, |id| stored(ffn(u64::from(id)), 128.0), &x);
    assert_close(&out, &others, 1e-6);
}

#[test]
fn dividing_up_proj_keeps_a_2_6e5_activation_inside_half_precision() {
    let activation = 2.6e5f32;
    assert!(f16::from_f32(activation).is_infinite());
    let stored_value = f16::from_f32(activation / 128.0);
    assert!(stored_value.is_finite());
    let weight = 0.3f32;
    let recovered = weight * 128.0 * stored_value.to_f32();
    assert!((recovered - weight * activation).abs() / (weight * activation) < 1e-3);
}
