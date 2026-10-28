// ANE-DECISION-01b: native ANE adapter parity with the CPU adapter.
// Tests cross DecisionHead::predict only (agreed seam 2).
// Uses decision-head dims (4096/512/256/4). The CPU adapter is the oracle; the
// hand-computed literals in tests/decision_head.rs validate that adapter
// independently. The bridge applies the measured ANE layout rules
// (ANE-RE-01): one const file per tensor; matmul channel storage shifted +64
// mod width (W2c rows and the final readout compensate); second-layer width
// padded to 512 to avoid a low-channel reshape artifact.
#![cfg(all(feature = "native-ane", target_os = "macos", target_arch = "aarch64"))]

use std::sync::Mutex;

use inference_engine::{
    EngError,
    kernels::decision::{DecisionHead, HeadConfig, HeadWeights},
};

static ANE_TEST: Mutex<()> = Mutex::new(());

fn config() -> HeadConfig {
    HeadConfig {
        input: 4096,
        hidden: 512,
        experts: 256,
        classes: 4,
    }
}

fn weights() -> HeadWeights {
    let c = config();
    let mut w1 = Vec::with_capacity(c.hidden * c.input);
    for i in 0..c.hidden * c.input {
        w1.push((i % 7) as f32 / 10.0 - 0.3);
    }
    let mut w2 = Vec::with_capacity((c.experts + c.classes) * c.hidden);
    for i in 0..(c.experts + c.classes) * c.hidden {
        w2.push((i % 11) as f32 / 100.0 - 0.05);
    }
    let mut b1 = Vec::with_capacity(c.hidden);
    for i in 0..c.hidden {
        b1.push((i % 5) as f32 / 40.0 - 0.025);
    }
    let mut b2 = Vec::with_capacity(c.classes + c.experts);
    for i in 0..c.classes + c.experts {
        b2.push((i % 7) as f32 / 30.0 - 0.1);
    }
    HeadWeights::try_new(&c, w1, b1, w2, b2).unwrap()
}

fn features() -> Vec<f32> {
    (0..config().input)
        .map(|i| (i % 11) as f32 / 10.0)
        .collect()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

fn topk(values: &[f32], k: usize) -> Vec<usize> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_unstable_by(|a, b| values[*b].total_cmp(&values[*a]));
    order.truncate(k);
    order
}

#[test]
fn ane_rejects_unsupported_hidden() {
    let _serial = ANE_TEST.lock().unwrap();
    let c = HeadConfig {
        input: 64,
        hidden: 32,
        experts: 4,
        classes: 2,
    };
    let w1 = vec![0.01f32; c.hidden * c.input];
    let w2 = vec![0.01f32; (c.experts + c.classes) * c.hidden];
    let weights = HeadWeights::try_new(
        &c,
        w1,
        vec![0.0; c.hidden],
        w2,
        vec![0.0; c.experts + c.classes],
    )
    .unwrap();
    match DecisionHead::ane(weights) {
        Err(EngError::Backend { .. }) => {}
        Err(other) => panic!("expected a backend rejection, received {other:?}"),
        Ok(_) => panic!("expected the bridge to reject the shape"),
    }
}

#[test]
fn ane_rejects_excessive_outputs() {
    let _serial = ANE_TEST.lock().unwrap();
    let c = HeadConfig {
        input: 64,
        hidden: 256,
        experts: 4096,
        classes: 4,
    };
    let w1 = vec![0.01f32; c.hidden * c.input];
    let w2 = vec![0.01f32; (c.experts + c.classes) * c.hidden];
    let weights = HeadWeights::try_new(
        &c,
        w1,
        vec![0.0; c.hidden],
        w2,
        vec![0.0; c.experts + c.classes],
    )
    .unwrap();
    match DecisionHead::ane(weights) {
        Err(EngError::Backend { .. }) => {}
        Err(other) => panic!("expected a backend rejection, received {other:?}"),
        Ok(_) => panic!("expected the bridge to reject the shape"),
    }
}

fn small_weights(c: &HeadConfig) -> HeadWeights {
    HeadWeights::try_new(
        c,
        vec![0.01f32; c.hidden * c.input],
        vec![0.0; c.hidden],
        vec![0.01f32; (c.experts + c.classes) * c.hidden],
        vec![0.0; c.experts + c.classes],
    )
    .unwrap()
}

fn backend_message(result: Result<DecisionHead, EngError>) -> String {
    match result {
        Err(EngError::Backend { operation, message }) => {
            assert_eq!(operation, "ane decision head compile");
            message
        }
        Err(other) => panic!("expected a backend rejection, received {other:?}"),
        Ok(_) => panic!("expected the bridge to reject the shape"),
    }
}

#[test]
fn ane_rejections_carry_their_own_reason() {
    // The bridge keeps one error slot per thread. Each call must clear it, so
    // a later rejection never reports the text of an earlier one.
    let _serial = ANE_TEST.lock().unwrap();
    let too_narrow = HeadConfig {
        input: 64,
        hidden: 32,
        experts: 4,
        classes: 2,
    };
    let too_wide = HeadConfig {
        input: 64,
        hidden: 256,
        experts: 4096,
        classes: 4,
    };
    let first = backend_message(DecisionHead::ane(small_weights(&too_wide)));
    assert!(first.contains("4096 outputs"), "unexpected text: {first}");
    let second = backend_message(DecisionHead::ane(small_weights(&too_narrow)));
    assert!(
        second.contains("hidden must be at least 256"),
        "unexpected text: {second}"
    );
    let third = backend_message(DecisionHead::ane(small_weights(&too_wide)));
    assert_eq!(third, first);
}

#[test]
fn ane_serves_predictions_after_rejected_shapes() {
    // A rejected shape leaves the process-global bridge state usable.
    let _serial = ANE_TEST.lock().unwrap();
    let rejected = HeadConfig {
        input: 64,
        hidden: 32,
        experts: 4,
        classes: 2,
    };
    backend_message(DecisionHead::ane(small_weights(&rejected)));
    let accepted = HeadConfig {
        input: 64,
        hidden: 256,
        experts: 8,
        classes: 2,
    };
    let features: Vec<f32> = (0..accepted.input)
        .map(|i| (i % 11) as f32 / 10.0)
        .collect();
    let head = DecisionHead::ane(small_weights(&accepted)).unwrap();
    let decision = head.predict(&features).unwrap();
    assert_eq!(decision.expert_logits.len(), accepted.experts);
    assert_eq!(decision.class_logits.len(), accepted.classes);
    let expected = DecisionHead::cpu(small_weights(&accepted))
        .unwrap()
        .predict(&features)
        .unwrap();
    for (actual, reference) in decision.expert_logits.iter().zip(&expected.expert_logits) {
        assert!((actual - reference).abs() < 0.05);
    }
}

#[test]
fn ane_parity_holds_at_the_smallest_supported_hidden() {
    let _serial = ANE_TEST.lock().unwrap();
    let c = HeadConfig {
        input: 64,
        hidden: 256,
        experts: 8,
        classes: 2,
    };
    let mut w1 = Vec::new();
    for i in 0..c.hidden * c.input {
        w1.push((i % 7) as f32 / 10.0 - 0.3);
    }
    let mut w2 = Vec::new();
    for i in 0..(c.experts + c.classes) * c.hidden {
        w2.push((i % 11) as f32 / 100.0 - 0.05);
    }
    let mut b1 = Vec::new();
    for i in 0..c.hidden {
        b1.push((i % 5) as f32 / 40.0 - 0.025);
    }
    let mut b2 = Vec::new();
    for i in 0..c.experts + c.classes {
        b2.push((i % 7) as f32 / 30.0 - 0.1);
    }
    let weights =
        || HeadWeights::try_new(&c, w1.clone(), b1.clone(), w2.clone(), b2.clone()).unwrap();
    let features: Vec<f32> = (0..c.input).map(|i| (i % 11) as f32 / 10.0).collect();
    let cpu = DecisionHead::cpu(weights()).unwrap();
    let ane = DecisionHead::ane(weights()).unwrap();
    let expected = cpu.predict(&features).unwrap();
    let actual = ane.predict(&features).unwrap();
    for (actual_value, expected_value) in actual.expert_logits.iter().zip(&expected.expert_logits) {
        assert!(
            (actual_value - expected_value).abs() < 0.05,
            "expert logit {actual_value} differs from cpu {expected_value}"
        );
    }
    for (actual_value, expected_value) in actual.class_logits.iter().zip(&expected.class_logits) {
        assert!(
            (actual_value - expected_value).abs() < 0.05,
            "class logit {actual_value} differs from cpu {expected_value}"
        );
    }
}

#[test]
fn ane_predict_matches_cpu_within_fp16_tolerance() {
    let _serial = ANE_TEST.lock().unwrap();
    let cpu = DecisionHead::cpu(weights()).unwrap();
    let ane = DecisionHead::ane(weights()).unwrap();
    let expected = cpu.predict(&features()).unwrap();
    let actual = ane.predict(&features()).unwrap();
    assert_eq!(actual.expert_logits.len(), 256);
    assert_eq!(actual.class_logits.len(), 4);

    let tolerance = |reference: f32| 0.05 + 0.02 * reference.abs();
    for (actual_value, expected_value) in actual.expert_logits.iter().zip(&expected.expert_logits) {
        assert!(
            (actual_value - expected_value).abs() <= tolerance(*expected_value),
            "expert logit {actual_value} differs from cpu {expected_value}"
        );
    }
    for (actual_value, expected_value) in actual.class_logits.iter().zip(&expected.class_logits) {
        assert!(
            (actual_value - expected_value).abs() <= tolerance(*expected_value),
            "class logit {actual_value} differs from cpu {expected_value}"
        );
    }
    // fp16 noise flips near-ties among the clustered expert logits: require
    // class argmax agreement and a top-8 expert set overlap of at least 4 of 8.
    let actual_experts = topk(&actual.expert_logits, 8);
    let expected_experts = topk(&expected.expert_logits, 8);
    let overlap: usize = actual_experts
        .iter()
        .filter(|member| expected_experts.contains(member))
        .count();
    assert!(
        overlap >= 4,
        "top-8 expert sets diverge too far: {overlap} of 8 overlap"
    );
    assert_eq!(argmax(&actual.class_logits), argmax(&expected.class_logits));
}
