use inference_engine::EngError;
use inference_engine::family::{Router, route_experts};

#[test]
fn ties_break_to_the_lower_index_and_renormalize() {
    let routed = route_experts(
        &[0.1, 0.4, 0.4, 0.1],
        &Router::SoftmaxTopK { renorm: true },
        2,
    )
    .expect("routes the tie fixture");
    assert_eq!(routed.indices, vec![1, 2]);
    assert_eq!(routed.weights, vec![0.5, 0.5]);
}

#[test]
fn distinct_probabilities_renormalize_by_exact_f32_division() {
    let routed = route_experts(
        &[0.5, 0.25, 0.125, 0.0625],
        &Router::SoftmaxTopK { renorm: true },
        3,
    )
    .expect("routes the distinct fixture");
    assert_eq!(routed.indices, vec![0, 1, 2]);
    assert_eq!(
        routed.weights,
        vec![0.5f32 / 0.875f32, 0.25f32 / 0.875f32, 0.125f32 / 0.875f32]
    );
}

#[test]
fn without_renorm_the_selected_probabilities_pass_through() {
    let routed = route_experts(
        &[0.1, 0.4, 0.4, 0.1],
        &Router::SoftmaxTopK { renorm: false },
        2,
    )
    .expect("routes without renormalization");
    assert_eq!(routed.indices, vec![1, 2]);
    assert_eq!(routed.weights, vec![0.4, 0.4]);
}

#[test]
fn equal_values_across_a_gap_stay_in_index_order() {
    let routed = route_experts(
        &[0.1, 0.4, 0.35, 0.4],
        &Router::SoftmaxTopK { renorm: true },
        2,
    )
    .expect("routes the gap fixture");
    assert_eq!(routed.indices, vec![1, 3]);
    assert_eq!(routed.weights, vec![0.5, 0.5]);
}

#[test]
fn zero_top_k_is_rejected() {
    let error = route_experts(&[0.25, 0.75], &Router::SoftmaxTopK { renorm: true }, 0)
        .expect_err("top_k must exceed zero");
    assert!(matches!(error, EngError::BadRequest(_)));
}

#[test]
fn top_k_beyond_the_router_width_is_rejected() {
    let error = route_experts(&[0.25, 0.75], &Router::SoftmaxTopK { renorm: true }, 3)
        .expect_err("top_k must fit the probabilities");
    assert!(matches!(error, EngError::BadRequest(_)));
}

#[test]
fn empty_probabilities_are_rejected() {
    let error = route_experts(&[], &Router::SoftmaxTopK { renorm: true }, 1)
        .expect_err("probabilities must not be empty");
    assert!(matches!(error, EngError::BadRequest(_)));
}

fn backend_message(error: EngError) -> String {
    match error {
        EngError::Backend { operation, message } => {
            assert_eq!(operation, "route_experts");
            message
        }
        other => panic!("expected Backend, got {other:?}"),
    }
}

#[test]
fn nan_probabilities_are_a_backend_fault() {
    let error = route_experts(
        &[0.25, f32::NAN, 0.5],
        &Router::SoftmaxTopK { renorm: true },
        2,
    )
    .expect_err("probabilities must not be NaN");
    assert!(backend_message(error).contains("finite"));
}

#[test]
fn infinite_and_negative_probabilities_are_a_backend_fault() {
    for probs in [
        [f32::INFINITY, 0.5, 0.25],
        [f32::NEG_INFINITY, 0.5, 0.25],
        [0.25, 0.5, -0.125],
    ] {
        for renorm in [true, false] {
            let error = route_experts(&probs, &Router::SoftmaxTopK { renorm }, 2)
                .expect_err("probabilities must be finite and non-negative");
            assert!(backend_message(error).contains("finite"), "{probs:?}");
        }
    }
}

#[test]
fn a_zero_or_overflowing_selected_sum_cannot_renormalize() {
    for probs in [[0.0, 0.0, 0.0], [f32::MAX, f32::MAX, 0.0]] {
        let error = route_experts(&probs, &Router::SoftmaxTopK { renorm: true }, 2)
            .expect_err("the selected sum must be finite and positive");
        assert!(backend_message(error).contains("sum"), "{probs:?}");
    }
}

#[test]
fn zero_probabilities_pass_through_without_renormalization() {
    let routed = route_experts(&[0.0, 0.0, 0.0], &Router::SoftmaxTopK { renorm: false }, 2)
        .expect("no division happens");
    assert_eq!(routed.indices, vec![0, 1]);
    assert_eq!(routed.weights, vec![0.0, 0.0]);
}

#[test]
fn zeros_outside_the_top_k_do_not_block_renormalization() {
    // Top-k keeps the largest values, so a zero selected sum needs every value to be zero.
    let routed = route_experts(&[0.0, 0.5, 0.0], &Router::SoftmaxTopK { renorm: true }, 1)
        .expect("the largest value is positive");
    assert_eq!(routed.indices, vec![1]);
    assert_eq!(routed.weights, vec![1.0]);
}

#[test]
fn sigmoid_group_routing_is_a_family_defect() {
    let error = route_experts(&[0.25, 0.75], &Router::SigmoidGroup { groups: 2 }, 1)
        .expect_err("sigmoid groups are not routed by this seam");
    assert!(matches!(error, EngError::Checkpoint(_)));
}
