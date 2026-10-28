// ANE-DECISION-01a: portable CPU adapter for the decision head.
// Tests cross DecisionHead::predict only (agreed seam 1).
use inference_engine::{
    EngError,
    kernels::decision::{DecisionHead, HeadConfig, HeadWeights},
};

fn config() -> HeadConfig {
    HeadConfig {
        input: 4,
        hidden: 2,
        experts: 3,
        classes: 2,
    }
}

fn weights() -> HeadWeights {
    HeadWeights::try_new(
        &config(),
        vec![0.1, 0.2, 0.3, 0.4, -0.1, 0.5, -0.2, 0.7],
        vec![0.1, -0.1],
        vec![0.5, -1.0, 0.2, 0.3, -0.4, 0.6, 0.7, 0.1, 0.3, -0.2],
        vec![0.05, -0.05, 0.1, 0.02, -0.01],
    )
    .unwrap()
}

#[test]
fn cpu_predict_matches_hand_computed_silu_head() {
    let head = DecisionHead::cpu(weights()).unwrap();
    let decision = head.predict(&[1.0, -2.0, 0.5, 3.0]).unwrap();
    // Hand-computed reference: y1 = W1x+b1 = [1.15, 0.8], silu = [0.873452, 0.551980],
    // out = W2*silu+b2:
    let expect_expert = [-0.065254, 0.290284, 0.081807];
    let expect_class = [0.686614, 0.141640];
    assert_eq!(decision.expert_logits.len(), 3);
    assert_eq!(decision.class_logits.len(), 2);
    for (actual, expected) in decision.expert_logits.iter().zip(expect_expert) {
        assert!(
            (actual - expected).abs() < 1e-4,
            "expert logit {actual} differs from {expected}"
        );
    }
    for (actual, expected) in decision.class_logits.iter().zip(expect_class) {
        assert!(
            (actual - expected).abs() < 1e-4,
            "class logit {actual} differs from {expected}"
        );
    }
}

#[test]
fn try_new_rejects_zero_dimensions() {
    let broken = HeadConfig {
        input: 4,
        hidden: 0,
        experts: 3,
        classes: 2,
    };
    let error = HeadWeights::try_new(&broken, vec![], vec![], vec![], vec![]).unwrap_err();
    assert!(matches!(error, EngError::BadRequest(_)));
}

#[test]
fn head_uses_the_weights_configuration() {
    let head = DecisionHead::cpu(weights()).unwrap();
    // the head derives its input width from the weights, so a wrong length fails
    let error = head.predict(&[1.0, -2.0]).unwrap_err();
    assert!(matches!(error, EngError::BadRequest(_)));
    let decision = head.predict(&[1.0, -2.0, 0.5, 3.0]).unwrap();
    assert_eq!(decision.expert_logits.len(), config().experts);
    assert_eq!(decision.class_logits.len(), config().classes);
}

#[test]
fn try_new_rejects_shape_mismatch() {
    let error =
        HeadWeights::try_new(&config(), vec![1.0, 2.0], vec![], vec![], vec![]).unwrap_err();
    assert!(matches!(error, EngError::BadRequest(_)));
}

#[test]
fn predict_rejects_wrong_feature_length() {
    let head = DecisionHead::cpu(weights()).unwrap();
    let error = head.predict(&[1.0, 2.0]).unwrap_err();
    assert!(matches!(error, EngError::BadRequest(_)));
}
#[test]
fn try_new_rejects_dimensions_beyond_the_native_int_range() {
    // Every adapter receives the dimensions as C int values, so the checked
    // weights type rejects larger values before any buffer comparison.
    let limit = i32::MAX as usize;
    for (name, config) in [
        (
            "input",
            HeadConfig {
                input: limit + 1,
                hidden: 2,
                experts: 3,
                classes: 2,
            },
        ),
        (
            "hidden",
            HeadConfig {
                input: 4,
                hidden: limit + 1,
                experts: 3,
                classes: 2,
            },
        ),
        (
            "experts",
            HeadConfig {
                input: 4,
                hidden: 2,
                experts: limit + 1,
                classes: 2,
            },
        ),
        (
            "classes",
            HeadConfig {
                input: 4,
                hidden: 2,
                experts: 3,
                classes: limit + 1,
            },
        ),
    ] {
        let error = HeadWeights::try_new(&config, vec![], vec![], vec![], vec![]).unwrap_err();
        match error {
            EngError::BadRequest(message) => assert!(
                message.contains(name) && message.contains("2147483647"),
                "the {name} rejection must name the field and the limit: {message}"
            ),
            other => panic!("expected BadRequest for {name}, received {other:?}"),
        }
    }
}

#[test]
fn try_new_rejects_an_output_count_beyond_the_native_int_range() {
    // Each dimension fits a C int, but experts + classes does not.
    let config = HeadConfig {
        input: 4,
        hidden: 2,
        experts: i32::MAX as usize,
        classes: 1,
    };
    let error = HeadWeights::try_new(&config, vec![], vec![], vec![], vec![]).unwrap_err();
    match error {
        EngError::BadRequest(message) => assert!(
            message.contains("outputs") && message.contains("2147483647"),
            "the rejection must name the output count and the limit: {message}"
        ),
        other => panic!("expected BadRequest, received {other:?}"),
    }
}

#[test]
fn try_new_accepts_the_largest_native_int_dimension_before_checking_lengths() {
    // The limit itself passes the dimension check, so a length mismatch is the
    // only remaining rejection.
    let config = HeadConfig {
        input: 4,
        hidden: 2,
        experts: 3,
        classes: i32::MAX as usize - 3,
    };
    let error = HeadWeights::try_new(&config, vec![], vec![], vec![], vec![]).unwrap_err();
    match error {
        EngError::BadRequest(message) => assert!(
            message.contains("decision weights must be"),
            "expected a length rejection, received: {message}"
        ),
        other => panic!("expected BadRequest, received {other:?}"),
    }
}

#[test]
#[ignore = "timing probe: run with --release --ignored"]
fn cpu_head_decision_tier_latency() {
    use std::time::Instant;
    let c = HeadConfig {
        input: 4096,
        hidden: 512,
        experts: 256,
        classes: 4,
    };
    let mut w1 = Vec::with_capacity(c.hidden * c.input);
    let mut w2 = Vec::with_capacity((c.experts + c.classes) * c.hidden);
    for i in 0..c.hidden * c.input {
        w1.push((i % 7) as f32 / 10.0 - 0.3);
    }
    for i in 0..(c.experts + c.classes) * c.hidden {
        w2.push((i % 11) as f32 / 100.0 - 0.05);
    }
    let weights = HeadWeights::try_new(
        &c,
        w1,
        vec![0.0; c.hidden],
        w2,
        vec![0.0; c.experts + c.classes],
    )
    .unwrap();
    let head = DecisionHead::cpu(weights).unwrap();
    let features: Vec<f32> = (0..c.input).map(|i| (i % 11) as f32 / 10.0).collect();
    // warmup
    for _ in 0..100 {
        let _ = head.predict(&features).unwrap();
    }
    const N: usize = 10_000;
    let mut lat = Vec::with_capacity(N);
    for _ in 0..N {
        let t0 = Instant::now();
        let _ = head.predict(&features).unwrap();
        lat.push(t0.elapsed());
    }
    lat.sort_unstable();
    let p = |q: f64| lat[((N as f64 * q) as usize).min(N - 1)].as_secs_f64() * 1e6;
    println!(
        "CPU decision head 4096/512/260: p50={:.1}us p90={:.1}us p99={:.1}us",
        p(0.50),
        p(0.90),
        p(0.99)
    );
}
