use super::*;

#[test]
fn packed_router_sizing_covers_each_t_without_legacy_scratch_column() {
    for m in [1, 7, 17, 512] {
        let shape = RouterShape {
            m,
            hidden: 64,
            n_expert: 256,
            top_k: 6,
        };
        let packed = router_packed_logits_bytes(shape).unwrap();
        assert_eq!(packed, m * 256 * 4);
        assert_eq!(router_logits_bytes(shape).unwrap() - packed, m * 4);
    }
}

#[test]
fn packed_router_rejects_unlaunchable_geometry_before_sizing() {
    let good = RouterShape {
        m: 1,
        hidden: 64,
        n_expert: 8,
        top_k: 6,
    };
    for bad in [
        RouterShape { m: 0, ..good },
        RouterShape {
            m: usize::MAX,
            ..good
        },
        RouterShape { hidden: 5, ..good },
        RouterShape {
            hidden: 65540,
            ..good
        },
        RouterShape {
            n_expert: 0,
            ..good
        },
        RouterShape {
            n_expert: 257,
            ..good
        },
        RouterShape { top_k: 0, ..good },
        RouterShape { top_k: 9, ..good },
    ] {
        assert!(router_packed_logits_bytes(bad).is_err(), "{bad:?}");
    }
}
