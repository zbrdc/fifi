use inference_engine::EngError;
use inference_engine::family::RecurrentLayout;
use inference_engine::tensor::Dtype;

fn hand_layout() -> RecurrentLayout {
    RecurrentLayout {
        heads: 2,
        rank: 3,
        conv_dim: 5,
        conv_width: 4,
        dtype: Dtype::F32,
    }
}

#[test]
fn hand_layout_computes_shapes_and_bytes() {
    let layout = hand_layout();
    assert_eq!(layout.conv_shape().expect("conv shape"), [5, 3]);
    assert_eq!(layout.ssm_shape().expect("ssm shape"), [2, 3, 3]);
    assert_eq!(layout.conv_bytes().expect("conv bytes"), 60);
    assert_eq!(layout.ssm_bytes().expect("ssm bytes"), 72);
    assert_eq!(layout.byte_len().expect("total bytes"), 132);
}

#[test]
fn family_geometry_pins_the_fixed_state_size() {
    let layout = RecurrentLayout {
        heads: 32,
        rank: 128,
        conv_dim: 8192,
        conv_width: 4,
        dtype: Dtype::F32,
    };
    assert_eq!(layout.conv_shape().expect("conv shape"), [8192, 3]);
    assert_eq!(layout.ssm_shape().expect("ssm shape"), [32, 128, 128]);
    assert_eq!(layout.conv_bytes().expect("conv bytes"), 98304);
    assert_eq!(layout.ssm_bytes().expect("ssm bytes"), 2097152);
    assert_eq!(layout.byte_len().expect("total bytes"), 2195456);
}

#[test]
fn dtype_scales_the_byte_width() {
    let layout = RecurrentLayout {
        heads: 1,
        rank: 2,
        conv_dim: 3,
        conv_width: 2,
        dtype: Dtype::F16,
    };
    assert_eq!(layout.conv_bytes().expect("conv bytes"), 6);
    assert_eq!(layout.ssm_bytes().expect("ssm bytes"), 8);
    assert_eq!(layout.byte_len().expect("total bytes"), 14);
}

#[test]
fn zero_fields_are_a_family_defect() {
    for field in 0..4 {
        let mut layout = hand_layout();
        match field {
            0 => layout.heads = 0,
            1 => layout.rank = 0,
            2 => layout.conv_dim = 0,
            _ => layout.conv_width = 0,
        }
        assert!(matches!(layout.conv_shape(), Err(EngError::Checkpoint(_))));
        assert!(matches!(layout.ssm_shape(), Err(EngError::Checkpoint(_))));
        assert!(matches!(layout.conv_bytes(), Err(EngError::Checkpoint(_))));
        assert!(matches!(layout.ssm_bytes(), Err(EngError::Checkpoint(_))));
        assert!(matches!(layout.byte_len(), Err(EngError::Checkpoint(_))));
    }
}

#[test]
fn overflowing_state_sizes_are_a_family_defect() {
    let layout = RecurrentLayout {
        heads: u32::MAX,
        rank: u32::MAX,
        conv_dim: u32::MAX,
        conv_width: u32::MAX,
        dtype: Dtype::I64,
    };
    assert!(matches!(layout.ssm_bytes(), Err(EngError::Checkpoint(_))));
    let wide = RecurrentLayout {
        heads: 1,
        rank: 1,
        conv_dim: u32::MAX,
        conv_width: u32::MAX,
        dtype: Dtype::I64,
    };
    // The element count (2^32 - 1) * (2^32 - 2) fits in u64, but eight bytes each overflow it.
    assert!(matches!(wide.conv_bytes(), Err(EngError::Checkpoint(_))));
}

#[test]
fn every_stored_dtype_has_a_byte_width() {
    for (dtype, width) in [
        (Dtype::U8, 1u64),
        (Dtype::F16, 2),
        (Dtype::Bf16, 2),
        (Dtype::F32, 4),
        (Dtype::U32, 4),
        (Dtype::I64, 8),
    ] {
        let layout = RecurrentLayout {
            heads: 1,
            rank: 1,
            conv_dim: 1,
            conv_width: 2,
            dtype,
        };
        assert_eq!(layout.ssm_bytes().expect("ssm bytes"), width, "{dtype:?}");
        assert_eq!(layout.conv_bytes().expect("conv bytes"), width, "{dtype:?}");
    }
}
