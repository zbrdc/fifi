use crate::tensor::Dtype;

use super::storage_tests::quantized;
use super::{AttnKvExtents, AttnKvLayout, AttnKvStorage, AttnScalarLayout, AttnStrides};

#[test]
fn affine_parameter_span_uses_its_own_float_dtype_and_group_pitch() {
    for (dtype, expected) in [
        (Dtype::F16, 330_usize),
        (Dtype::Bf16, 330),
        (Dtype::F32, 332),
    ] {
        let plane = AttnScalarLayout {
            dtype,
            strides: AttnStrides {
                time_stride_bytes: 128,
                head_stride_bytes: 48,
            },
            group_stride_bytes: 8,
        };
        assert_eq!(plane.span_bytes(3, 2, 4), Some(expected), "{dtype:?}");
    }
    let plane = AttnScalarLayout {
        dtype: Dtype::F32,
        strides: AttnStrides {
            time_stride_bytes: usize::MAX,
            head_stride_bytes: usize::MAX,
        },
        group_stride_bytes: usize::MAX,
    };
    for (rows, heads, groups, expected) in [
        (0, 2, usize::MAX, Some(0)),
        (2, 0, usize::MAX, Some(0)),
        (2, 2, 0, Some(0)),
        (1, 1, 1, Some(4)),
        (1, 1, 2, None),
        (1, 1, 3, None),
        (2, 1, 1, None),
        (3, 1, 1, None),
        (1, 3, 1, None),
        (2, 2, 1, None),
    ] {
        assert_eq!(plane.span_bytes(rows, heads, groups), expected);
    }
    assert_eq!(
        AttnScalarLayout {
            dtype: Dtype::I64,
            ..plane
        }
        .span_bytes(1, 1, 1),
        None,
    );
}

fn scalar(dtype: Dtype) -> AttnScalarLayout {
    AttnScalarLayout {
        dtype,
        strides: AttnStrides {
            time_stride_bytes: 0,
            head_stride_bytes: 0,
        },
        group_stride_bytes: 0,
    }
}

fn layout(storage: AttnKvStorage) -> AttnKvLayout {
    AttnKvLayout {
        storage,
        payload_strides: AttnStrides {
            time_stride_bytes: 0,
            head_stride_bytes: 0,
        },
        scales: None,
        bias: None,
    }
}

#[test]
fn float_layouts_size_each_payload_dtype() {
    for (rows, heads, dim) in [
        (2, 1, 64),
        (7, 3, 128),
        (17, 5, 256),
        (512, 4, 128),
        (1024, 8, 256),
    ] {
        for (storage, width) in [
            (AttnKvStorage::F16, 2),
            (AttnKvStorage::Bf16, 2),
            (AttnKvStorage::F32, 4),
        ] {
            let plan = AttnKvLayout {
                payload_strides: AttnStrides {
                    time_stride_bytes: heads * dim * width,
                    head_stride_bytes: dim * width,
                },
                ..layout(storage)
            };
            assert_eq!(
                plan.extents(rows, heads, dim).unwrap(),
                AttnKvExtents {
                    payload_bytes: rows * heads * dim * width,
                    scale_bytes: None,
                    bias_bytes: None,
                }
            );
        }
    }
}

#[test]
fn affine_layouts_size_payload_and_scalars_independently() {
    for (rows, heads, dim) in [
        (2, 1, 64_usize),
        (7, 3, 128),
        (17, 5, 256),
        (512, 4, 128),
        (1024, 8, 256),
    ] {
        for group in [16, 32, 64, 128, 256]
            .into_iter()
            .filter(|group| dim.is_multiple_of(*group))
        {
            for storage in quantized(group) {
                let cell_bytes = match storage {
                    AttnKvStorage::Q4Affine { .. } => dim / 2,
                    _ => dim,
                };
                let groups = dim / group;
                let plan = AttnKvLayout {
                    payload_strides: AttnStrides {
                        time_stride_bytes: cell_bytes,
                        head_stride_bytes: rows * cell_bytes,
                    },
                    scales: Some(AttnScalarLayout {
                        strides: AttnStrides {
                            time_stride_bytes: heads * groups * 4,
                            head_stride_bytes: groups * 4,
                        },
                        group_stride_bytes: 4,
                        ..scalar(Dtype::F32)
                    }),
                    ..layout(storage)
                };
                assert_eq!(
                    plan.extents(rows, heads, dim).unwrap(),
                    AttnKvExtents {
                        payload_bytes: rows * heads * cell_bytes,
                        scale_bytes: Some(rows * heads * groups * 4),
                        bias_bytes: None,
                    }
                );
            }
        }
    }
}

#[test]
fn affine_layouts_allow_padded_and_broadcast_scalar_planes() {
    let plan = AttnKvLayout {
        payload_strides: AttnStrides {
            time_stride_bytes: 320,
            head_stride_bytes: 1024,
        },
        scales: Some(AttnScalarLayout {
            strides: AttnStrides {
                time_stride_bytes: 128,
                head_stride_bytes: 48,
            },
            group_stride_bytes: 8,
            ..scalar(Dtype::F32)
        }),
        bias: Some(scalar(Dtype::Bf16)),
        ..layout(AttnKvStorage::Q8Affine {
            group_size: 32,
            signed: false,
        })
    };
    assert_eq!(
        plan.extents(3, 2, 128).unwrap(),
        AttnKvExtents {
            payload_bytes: 1792,
            scale_bytes: Some(332),
            bias_bytes: Some(2),
        }
    );
}

#[test]
fn layouts_refuse_missing_extra_or_invalid_planes() {
    for storage in [AttnKvStorage::F16, AttnKvStorage::Bf16, AttnKvStorage::F32] {
        let plan = layout(storage);
        assert!(
            AttnKvLayout {
                scales: Some(scalar(Dtype::F16)),
                ..plan
            }
            .extents(2, 3, 64)
            .is_err()
        );
        assert!(
            AttnKvLayout {
                bias: Some(scalar(Dtype::F32)),
                ..plan
            }
            .extents(2, 3, 64)
            .is_err()
        );
    }
    for storage in quantized(32) {
        let missing = layout(storage);
        assert!(missing.extents(2, 3, 64).is_err());
        assert!(
            AttnKvLayout {
                bias: Some(scalar(Dtype::F32)),
                ..missing
            }
            .extents(2, 3, 64)
            .is_err()
        );
        for dtype in [Dtype::I16, Dtype::I32, Dtype::I64] {
            let invalid = AttnKvLayout {
                scales: Some(scalar(dtype)),
                ..missing
            };
            assert!(invalid.extents(2, 3, 64).is_err());
            let invalid = AttnKvLayout {
                scales: Some(scalar(Dtype::F16)),
                bias: Some(scalar(dtype)),
                ..missing
            };
            assert!(invalid.extents(2, 3, 64).is_err());
        }
    }
    for group in [0, 17, 65, usize::MAX] {
        for storage in quantized(group) {
            assert!(
                AttnKvLayout {
                    scales: Some(scalar(Dtype::F16)),
                    ..layout(storage)
                }
                .extents(2, 3, 64)
                .is_err()
            );
        }
    }
}

#[test]
fn layouts_preserve_empty_planes_without_hiding_invalid_metadata() {
    for storage in quantized(32) {
        let plan = AttnKvLayout {
            scales: Some(scalar(Dtype::F16)),
            bias: Some(scalar(Dtype::F32)),
            ..layout(storage)
        };
        for (rows, heads) in [(0, 3), (7, 0)] {
            assert_eq!(
                plan.extents(rows, heads, 64).unwrap(),
                AttnKvExtents {
                    payload_bytes: 0,
                    scale_bytes: Some(0),
                    bias_bytes: Some(0),
                }
            );
            assert!(layout(storage).extents(rows, heads, 64).is_err());
            assert!(
                AttnKvLayout {
                    bias: Some(scalar(Dtype::I64)),
                    ..plan
                }
                .extents(rows, heads, 64)
                .is_err()
            );
        }
        assert!(plan.extents(0, 0, 0).is_err());
    }
    assert_eq!(
        layout(AttnKvStorage::F32).extents(0, 3, 64).unwrap(),
        AttnKvExtents {
            payload_bytes: 0,
            scale_bytes: None,
            bias_bytes: None,
        }
    );
}

#[test]
fn layout_extents_reject_payload_and_parameter_overflow() {
    let plan = AttnKvLayout {
        payload_strides: AttnStrides {
            time_stride_bytes: usize::MAX,
            head_stride_bytes: usize::MAX,
        },
        ..layout(AttnKvStorage::F16)
    };
    for (rows, heads) in [(2, 1), (1, 2), (3, 1), (1, 3)] {
        assert!(plan.extents(rows, heads, 64).is_err());
    }
    assert!(plan.extents(1, 1, usize::MAX).is_err());
    let plan = AttnKvLayout {
        scales: Some(AttnScalarLayout {
            group_stride_bytes: usize::MAX,
            ..scalar(Dtype::F16)
        }),
        ..layout(AttnKvStorage::Q8Affine {
            group_size: 32,
            signed: true,
        })
    };
    assert!(plan.extents(1, 1, 64).is_err());
    let plan = AttnKvLayout {
        scales: Some(scalar(Dtype::F16)),
        bias: Some(AttnScalarLayout {
            strides: AttnStrides {
                time_stride_bytes: usize::MAX,
                head_stride_bytes: 0,
            },
            ..scalar(Dtype::F32)
        }),
        ..plan
    };
    assert!(plan.extents(3, 1, 64).is_err());
}
