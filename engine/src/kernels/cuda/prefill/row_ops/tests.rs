use super::RowShape;
use crate::kernels::cuda::native::prefill::row_ops::{
    gather_bytes, multiply_bytes, rank_sum_bytes,
};

#[test]
fn row_ops_geometry_covers_whole_blocks_and_rejects_overflow() {
    assert_eq!(
        RowShape {
            rows: 512,
            columns: 3072
        }
        .bytes(),
        Some(512 * 3072 * 4)
    );
    assert_eq!(
        RowShape {
            rows: 65536,
            columns: 65536
        }
        .bytes(),
        Some(1usize << 34)
    );
    for shape in [
        RowShape {
            rows: 0,
            columns: 64,
        },
        RowShape {
            rows: 1,
            columns: 0,
        },
        RowShape {
            rows: 65537,
            columns: 1,
        },
        RowShape {
            rows: 1,
            columns: (1 << 22) + 1,
        },
        RowShape {
            rows: 65536,
            columns: 65537,
        },
        RowShape {
            rows: usize::MAX,
            columns: usize::MAX,
        },
    ] {
        assert_eq!(shape.bytes(), None);
    }
}

#[test]
fn row_ops_plans_broadcast_gather_and_original_rank_reduction() {
    assert_eq!(multiply_bytes(17, 64, 1), Some((17 * 64 * 4, 17 * 4)));
    assert_eq!(multiply_bytes(17, 64, 64), Some((17 * 64 * 4, 17 * 64 * 4)));
    assert_eq!(multiply_bytes(17, 64, 2), None);
    assert_eq!(
        gather_bytes(17 * 6, 64, 17),
        Some((17 * 64 * 4, 17 * 6 * 4, 17 * 6 * 64 * 4))
    );
    assert_eq!(gather_bytes(17, 64, 0), None);
    assert_eq!(
        rank_sum_bytes(17, 64, 6),
        Some((17 * 6 * 64 * 4, 17 * 64 * 4))
    );
    assert_eq!(rank_sum_bytes(17, 64, 0), None);
    assert_eq!(rank_sum_bytes(17, 64, 33), None);
    assert_eq!(rank_sum_bytes(65536, 1, 2), None);
}
