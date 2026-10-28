use super::plan::{FfnBlock, FfnShape, Grouping};

fn moe_shape() -> FfnShape {
    FfnShape::Moe {
        hidden: 64,
        intermediate: 64,
        shared_intermediate: 96,
        experts: 8,
    }
}

#[test]
fn every_requested_t_is_one_whole_projection_block() {
    for rows in [1, 7, 17, 512] {
        let plan = FfnBlock::new(rows, moe_shape()).unwrap();
        assert_eq!(plan.rows, rows);
        assert_eq!(plan.pairs, rows * 6);
        assert_eq!(plan.hidden, 64);
        let dense = FfnBlock::new(
            rows,
            FfnShape::Dense {
                hidden: 64,
                intermediate: 96,
            },
        )
        .unwrap();
        assert_eq!(dense.pairs, rows);
        assert_eq!(dense.experts, 0);
    }
}

#[test]
fn malformed_or_backend_unrepresentable_shapes_refuse_before_allocation() {
    for rows in [0, 4097, usize::MAX] {
        assert!(FfnBlock::new(rows, moe_shape()).is_err());
    }
    for (hidden, intermediate) in [(0, 64), (33, 64), (64, 0), (64, usize::MAX)] {
        assert!(
            FfnBlock::new(
                1,
                FfnShape::Dense {
                    hidden,
                    intermediate
                }
            )
            .is_err()
        );
    }
    let too_few = FfnShape::Moe {
        hidden: 64,
        intermediate: 64,
        shared_intermediate: 96,
        experts: 5,
    };
    assert!(FfnBlock::new(1, too_few).is_err());
    assert!(
        FfnBlock::new(
            4096,
            FfnShape::Moe {
                hidden: 32768,
                intermediate: 64,
                shared_intermediate: 96,
                experts: 8,
            }
        )
        .is_err()
    );
}

#[test]
fn expert_envelope_matches_the_shared_256_expert_router() {
    for experts in [6, 256] {
        let shape = FfnShape::Moe {
            hidden: 64,
            intermediate: 64,
            shared_intermediate: 96,
            experts,
        };
        assert!(FfnBlock::new(1, shape).is_ok());
        assert!(Grouping::new(&[0, 1, 2, 3, 4, 5], 1, experts).is_ok());
    }
    for experts in [257, 4096] {
        let shape = FfnShape::Moe {
            hidden: 64,
            intermediate: 64,
            shared_intermediate: 96,
            experts,
        };
        assert!(FfnBlock::new(1, shape).is_err());
        assert!(Grouping::new(&[0, 1, 2, 3, 4, 5], 1, experts).is_err());
    }
}

#[test]
fn grouping_preserves_original_selected_rank_and_builds_its_inverse() {
    let plan = Grouping::new(&[5, 1, 4, 2, 3, 0], 1, 8).unwrap();
    assert_eq!(plan.order, [5, 1, 3, 4, 2, 0]);
    assert_eq!(plan.inverse, [5, 1, 4, 2, 3, 0]);
    assert_eq!(plan.tokens, [0; 6]);
}

#[test]
fn expert_major_order_stays_stable_across_all_requested_t() {
    for rows in [1, 7, 17, 512] {
        let ids: Vec<i32> = (0..rows)
            .flat_map(|token| (0..6).map(move |rank| ((token + rank) % 8) as i32))
            .collect();
        let plan = Grouping::new(&ids, rows, 8).unwrap();
        assert_eq!(plan.order.len(), rows * 6);
        for (sorted, &original) in plan.order.iter().enumerate() {
            assert_eq!(plan.inverse[original], sorted as i32);
            assert_eq!(plan.tokens[sorted], (original / 6) as i32);
        }
        for pair in plan.order.windows(2) {
            let (left, right) = (pair[0], pair[1]);
            assert!(ids[left] < ids[right] || (ids[left] == ids[right] && left < right));
        }
    }
}

#[test]
fn invalid_route_metadata_never_builds_a_descriptor_plan() {
    let good = [0, 1, 2, 3, 4, 5];
    assert!(Grouping::new(&good[..5], 1, 8).is_err());
    assert!(Grouping::new(&good, 0, 8).is_err());
    for bad in [-1, 8] {
        let mut ids = good;
        ids[3] = bad;
        assert!(Grouping::new(&ids, 1, 8).is_err());
    }
    let mut duplicate = good;
    duplicate[4] = duplicate[0];
    assert!(Grouping::new(&duplicate, 1, 8).is_err());
}
