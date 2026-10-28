use super::AttnStrides;

#[test]
fn time_major_and_head_major_cover_the_same_cells() {
    for rows in [2, 7, 17, 512] {
        for heads in [1, 3, 5, 32] {
            for dim in [64, 128, 256] {
                let width = dim * 2;
                let time_major = AttnStrides {
                    time_stride_bytes: heads * width,
                    head_stride_bytes: width,
                };
                let head_major = AttnStrides {
                    time_stride_bytes: width,
                    head_stride_bytes: rows * width,
                };
                let bytes = Some(rows * heads * width);
                assert_eq!(time_major.span_bytes(rows, heads, width), bytes);
                assert_eq!(head_major.span_bytes(rows, heads, width), bytes);
            }
        }
    }
}

#[test]
fn padded_strides_end_at_the_last_live_cell() {
    let time_major = AttnStrides {
        time_stride_bytes: 1024,
        head_stride_bytes: 320,
    };
    let head_major = AttnStrides {
        time_stride_bytes: 320,
        head_stride_bytes: 1024,
    };
    assert_eq!(time_major.span_bytes(3, 2, 256), Some(2624));
    assert_eq!(head_major.span_bytes(3, 2, 256), Some(1920));
}

#[test]
fn read_only_parameter_planes_can_broadcast() {
    let shared = AttnStrides {
        time_stride_bytes: 0,
        head_stride_bytes: 0,
    };
    assert_eq!(shared.span_bytes(512, 32, 16), Some(16));
}

#[test]
fn an_empty_logical_plane_needs_no_storage() {
    let strides = AttnStrides {
        time_stride_bytes: usize::MAX,
        head_stride_bytes: usize::MAX,
    };
    assert_eq!(strides.span_bytes(0, 32, 256), Some(0));
    assert_eq!(strides.span_bytes(512, 0, 256), Some(0));
}

#[test]
fn nonempty_planes_require_a_nonzero_cell_width() {
    let strides = AttnStrides {
        time_stride_bytes: 4,
        head_stride_bytes: 4,
    };
    assert_eq!(strides.span_bytes(1, 1, 0), None);
}

#[test]
fn either_stride_product_or_the_final_extent_can_overflow() {
    let strides = AttnStrides {
        time_stride_bytes: usize::MAX,
        head_stride_bytes: usize::MAX,
    };
    assert_eq!(strides.span_bytes(3, 1, 1), None);
    assert_eq!(strides.span_bytes(1, 3, 1), None);
    assert_eq!(strides.span_bytes(2, 1, 1), None);
    assert_eq!(strides.span_bytes(2, 2, 1), None);
}
