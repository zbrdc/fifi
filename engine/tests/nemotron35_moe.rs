use inference_engine::{
    family::nemotron35::forward::moe::{
        ExpertWeights, MlpWeights, MoeConfig, RoutedExpert, apply_routed_experts, expert_forward,
        moe_block_forward, route,
    },
    tensor::{Dtype, Tensor},
};

fn tensor(shape: &[u32], values: &[f32]) -> Tensor {
    Tensor::try_from_host(
        Dtype::F32,
        shape.to_vec(),
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap()
}

fn values(t: &Tensor) -> Vec<f32> {
    t.host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect()
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-5, "{a} != {e}");
    }
}

fn config() -> MoeConfig {
    MoeConfig {
        n_group: 2,
        topk_group: 1,
        experts_per_token: 2,
        norm_topk_prob: true,
        routed_scaling_factor: 5.0,
        eps: 3.0,
    }
}

#[test]
fn routing_uses_top_two_group_sum_then_raw_sigmoid_renormalization_and_scaling() {
    // Literal probabilities: [.9,.1,.1,.1 | .8,.7,.6,.1]. Group sums: 1 and 1.5.
    let logits = tensor(
        &[8, 1],
        &[
            2.1972246, -2.1972246, -2.1972246, -2.1972246, 1.3862944, 0.84729785, 0.4054651,
            -2.1972246,
        ],
    );
    let selected = route(
        &tensor(&[1, 1], &[1.0]),
        &logits,
        &tensor(&[8], &[0.0; 8]),
        config(),
    )
    .unwrap();
    assert_eq!(
        selected[0].iter().map(|e| e.index).collect::<Vec<_>>(),
        [4, 5]
    );
    close(
        &selected[0].iter().map(|e| e.weight).collect::<Vec<_>>(),
        &[2.6666667, 2.3333333],
    );
}

#[test]
fn correction_bias_changes_choice_but_not_mixture_weights() {
    let logits = tensor(
        &[8, 1],
        &[
            2.1972246, -2.1972246, -2.1972246, -2.1972246, 1.3862944, 0.84729785, 0.4054651,
            -2.1972246,
        ],
    );
    let selected = route(
        &tensor(&[1, 1], &[1.0]),
        &logits,
        &tensor(&[8], &[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        config(),
    )
    .unwrap();
    assert_eq!(
        selected[0].iter().map(|e| e.index).collect::<Vec<_>>(),
        [1, 0]
    );
    close(
        &selected[0].iter().map(|e| e.weight).collect::<Vec<_>>(),
        &[0.5, 4.5],
    );
}

#[test]
fn expert_uses_ungated_relu_squared() {
    let up = tensor(&[2, 2], &[1.0, 1.0, -1.0, 2.0]);
    let down = tensor(&[2, 2], &[3.0, 5.0, -2.0, 7.0]);
    let output = expert_forward(
        &tensor(&[1, 2], &[2.0, -1.0]),
        MlpWeights {
            up: &up,
            down: &down,
        },
    )
    .unwrap();
    close(&values(&output), &[3.0, -2.0]);
}

fn tiny_block(shared_only: bool) -> Tensor {
    let norm = tensor(&[2], &[1.0, 1.0]);
    let hidden = tensor(&[1, 2], &[1.0, 1.0]);
    let router = tensor(
        &[4, 2],
        &[2.1972246, 0.0, 0.0, 0.0, -2.1972246, 0.0, -4.394449, 0.0],
    );
    let bias = tensor(&[4], &[0.0; 4]);
    let latent_up = tensor(&[1, 2], &[2.0, 0.0]);
    let latent_down = tensor(&[2, 1], &[1.0, -1.0]);
    let up0 = tensor(&[1, 1], &[2.0]);
    let down0 = tensor(&[1, 1], &[if shared_only { 0.0 } else { 3.0 }]);
    let up1 = tensor(&[1, 1], &[-1.0]);
    let down1 = tensor(&[1, 1], &[9.0]);
    let shared_up = tensor(&[1, 2], &[0.0, 2.0]);
    let shared_down = tensor(&[2, 1], &[2.0, 4.0]);
    // Only two selected experts are provided, although the router has four rows.
    let experts = [
        ExpertWeights {
            index: 0,
            weights: MlpWeights {
                up: &up0,
                down: &down0,
            },
        },
        ExpertWeights {
            index: 1,
            weights: MlpWeights {
                up: &up1,
                down: &down1,
            },
        },
    ];
    moe_block_forward(
        &norm,
        &hidden,
        &router,
        &bias,
        &latent_up,
        &latent_down,
        &experts,
        MlpWeights {
            up: &shared_up,
            down: &shared_down,
        },
        config(),
    )
    .unwrap()
}

#[test]
fn shared_expert_is_added_unweighted_from_normalized_hidden() {
    close(&values(&tiny_block(true)), &[3.0, 5.0]);
}

#[test]
fn full_moe_uses_latent_experts_shared_path_and_original_residual() {
    // Norm=[.5,.5]; latent=1; selected weights=3,2; expert outputs=12,0.
    // Latent mixture=36; expanded=[36,-36]; shared=[2,4]; residual=[1,1].
    close(&values(&tiny_block(false)), &[39.0, -31.0]);
}

#[test]
fn real_router_cardinality_has_positive_weights_summing_to_five() {
    let mut c = config();
    c.n_group = 1;
    c.experts_per_token = 22;
    let selected = route(
        &tensor(&[2, 1], &[1.0, 2.0]),
        &tensor(&[512, 1], &[0.0; 512]),
        &tensor(&[512], &[0.0; 512]),
        c,
    )
    .unwrap();
    assert_eq!(selected.len(), 2);
    for row in selected {
        assert_eq!(row.len(), 22);
        assert!(row.iter().all(|e| e.weight > 0.0));
        let mut indices: Vec<_> = row.iter().map(|e| e.index).collect();
        indices.sort_unstable();
        indices.dedup();
        assert_eq!(indices.len(), 22);
        close(&[row.iter().map(|e| e.weight).sum()], &[5.0]);
    }
}

#[test]
fn routed_apply_accepts_only_selected_experts_for_multiple_tokens() {
    let up = tensor(&[1, 1], &[2.0]);
    let down = tensor(&[1, 1], &[3.0]);
    let experts = [ExpertWeights {
        index: 7,
        weights: MlpWeights {
            up: &up,
            down: &down,
        },
    }];
    let selection = vec![
        vec![RoutedExpert {
            index: 7,
            weight: 2.0,
        }],
        vec![RoutedExpert {
            index: 7,
            weight: 0.5,
        }],
    ];
    let output = apply_routed_experts(&tensor(&[2, 1], &[1.0, 2.0]), &selection, &experts).unwrap();
    close(&values(&output), &[24.0, 24.0]);
    assert!(apply_routed_experts(&tensor(&[2, 1], &[1.0, 2.0]), &selection, &[]).is_err());
}

#[test]
fn routing_without_renormalization_scales_raw_sigmoid_weights() {
    let mut c = config();
    c.n_group = 1;
    c.norm_topk_prob = false;
    let selected = route(
        &tensor(&[1, 1], &[1.0]),
        &tensor(&[2, 1], &[1.3862944, 0.84729785]),
        &tensor(&[2], &[0.0, 0.0]),
        c,
    )
    .unwrap();
    close(
        &selected[0].iter().map(|e| e.weight).collect::<Vec<_>>(),
        &[4.0, 3.5],
    );
}

#[test]
fn routing_rejects_invalid_geometry_shapes_and_nonfinite_inputs() {
    let hidden = tensor(&[1, 1], &[1.0]);
    let router = tensor(&[4, 1], &[0.0; 4]);
    let bias = tensor(&[4], &[0.0; 4]);
    let mut invalid_configs = Vec::new();
    for n_group in [0, 3, 4] {
        invalid_configs.push(MoeConfig {
            n_group,
            ..config()
        });
    }
    for topk_group in [0, 3] {
        invalid_configs.push(MoeConfig {
            topk_group,
            ..config()
        });
    }
    for experts_per_token in [0, 3] {
        invalid_configs.push(MoeConfig {
            experts_per_token,
            ..config()
        });
    }
    for routed_scaling_factor in [0.0, -1.0, f32::NAN] {
        invalid_configs.push(MoeConfig {
            routed_scaling_factor,
            ..config()
        });
    }
    for c in invalid_configs {
        assert!(route(&hidden, &router, &bias, c).is_err());
    }
    assert!(route(&tensor(&[1], &[1.0]), &router, &bias, config()).is_err());
    assert!(route(&tensor(&[0, 1], &[]), &router, &bias, config()).is_err());
    assert!(route(&hidden, &tensor(&[4, 2], &[0.0; 8]), &bias, config()).is_err());
    assert!(route(&hidden, &router, &tensor(&[3], &[0.0; 3]), config()).is_err());
    assert!(route(&tensor(&[1, 1], &[f32::NAN]), &router, &bias, config()).is_err());
    assert!(route(&hidden, &router, &tensor(&[4], &[f32::MAX; 4]), config()).is_err());
    let wrong_dtype = Tensor::try_from_host(Dtype::U32, vec![4], vec![0; 16]).unwrap();
    assert!(route(&hidden, &router, &wrong_dtype, config()).is_err());
}

#[test]
fn expert_apply_rejects_invalid_selection_shapes_and_arithmetic_overflow() {
    let hidden = tensor(&[1, 1], &[1.0]);
    let up = tensor(&[1, 1], &[2.0]);
    let down = tensor(&[1, 1], &[3.0]);
    let weights = MlpWeights {
        up: &up,
        down: &down,
    };
    let expert = ExpertWeights { index: 0, weights };
    for selection in [
        vec![],
        vec![vec![]],
        vec![vec![RoutedExpert {
            index: 0,
            weight: -1.0,
        }]],
        vec![vec![RoutedExpert {
            index: 0,
            weight: f32::NAN,
        }]],
        vec![vec![
            RoutedExpert {
                index: 0,
                weight: 1.0
            };
            2
        ]],
    ] {
        assert!(apply_routed_experts(&hidden, &selection, std::slice::from_ref(&expert)).is_err());
    }
    let selection = vec![vec![RoutedExpert {
        index: 0,
        weight: 1.0,
    }]];
    assert!(
        apply_routed_experts(
            &hidden,
            &selection,
            &[ExpertWeights { index: 0, weights }, expert]
        )
        .is_err()
    );
    assert!(
        expert_forward(
            &hidden,
            MlpWeights {
                up: &up,
                down: &tensor(&[1, 2], &[1.0, 1.0])
            }
        )
        .is_err()
    );
    assert!(expert_forward(&tensor(&[1, 1], &[f32::MAX]), weights).is_err());
    assert!(expert_forward(&tensor(&[1, 1], &[1e20]), weights).is_err());
}
