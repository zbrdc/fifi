//! Host-only tests of the CUDA context checks: the tensor type and shape gates,
//! the Q6_K and Q8_0 scale ranges, the fp16 decoder, and the linear token count.
//!
//! The fp16 oracle below is written from the bit fields, not from the decoder,
//! so the exhaustive round trip compares two independent implementations.

use super::*;

/// Error text by its Debug form, so the helpers need no `T: Debug`.
fn error_text<T>(result: Result<T, EngError>) -> String {
    match result {
        Ok(_) => panic!("expected an error, got Ok"),
        Err(error) => format!("{error:?}"),
    }
}

/// Asserts the error variant and that its message contains `needle`.
fn assert_error<T>(result: Result<T, EngError>, kind: &str, needle: &str) {
    let text = error_text(result);
    assert!(
        text.starts_with(kind) && text.contains(needle),
        "expected {kind} containing {needle:?}, got {text}"
    );
}

/// Round-to-nearest-even f32 -> fp16, written from the bit fields. Ties go to
/// the even significand, the same rule the device conversion uses.
fn f32_to_f16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = (bits >> 23) & 0xff;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0xff {
        return if mantissa != 0 { 0x7e00 } else { sign | 0x7c00 };
    }
    // f32 zero or subnormal: far below half of the smallest fp16 subnormal.
    if exponent == 0 {
        return sign;
    }
    let e = exponent as i32 - 127;
    let full = (1u32 << 23) | mantissa;
    if e >= 16 {
        sign | 0x7c00
    } else if e >= -14 {
        normal_f16(sign, e, full)
    } else {
        subnormal_f16(sign, e, full)
    }
}

/// Keeps 11 significant bits of the 24-bit significand, ties to even.
fn normal_f16(sign: u16, e: i32, full: u32) -> u16 {
    let mut m = full >> 13;
    let rem = full & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && m % 2 == 1) {
        m += 1;
    }
    let mut e = e;
    if m == 0x800 {
        m = 0x400;
        e += 1;
    }
    if e > 15 {
        return sign | 0x7c00;
    }
    sign | (((e + 15) as u16) << 10) | (m as u16 - 0x400)
}

/// Subnormal fp16 counts units of 2^-24, so the shift is `-1 - e` (14 at e = -15).
fn subnormal_f16(sign: u16, e: i32, full: u32) -> u16 {
    let shift = (-1 - e) as u32;
    // Below 2^-25 the value rounds to zero; at exactly 2^-25 the tie rounds to even.
    if shift > 24 {
        return sign;
    }
    let m = full >> shift;
    let rem = full & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    let m = if rem > half || (rem == half && m % 2 == 1) {
        m + 1
    } else {
        m
    };
    sign | m as u16
}

/// An fp16 pattern that decodes to NaN: exponent all ones, mantissa non-zero.
fn is_nan_pattern(bits: u16) -> bool {
    (bits & 0x7c00) == 0x7c00 && (bits & 0x03ff) != 0
}

/// A Q6_K block whose 16 int8 scales all equal `scale` and whose fp16 `d` is `d_bits`.
fn q6k_block(d_bits: u16, scale: i8) -> Vec<u8> {
    let mut block = vec![0u8; 210];
    block[192..208].fill(scale.cast_unsigned());
    block[208..210].copy_from_slice(&d_bits.to_le_bytes());
    block
}

/// A Q8_0 block with fp16 `d` bits `d_bits`, quant 0 set to `q` and the rest 1.
fn q8_0_block(d_bits: u16, q: i8) -> Vec<u8> {
    let mut block = vec![1u8; 34];
    block[0..2].copy_from_slice(&d_bits.to_le_bytes());
    block[2] = q.cast_unsigned();
    block
}

#[test]
fn checked_tiled_rejects_a_type_the_matmul_does_not_execute() {
    // 8 rows of one 256-value Q5_K block.
    let blocks = [0u8; 8 * 176];
    let result = checked_tiled(QuantTensor {
        name: "w",
        ty: GgufType::Q5K,
        n_rows: 8,
        k: 256,
        source_len: blocks.len(),
    });
    assert_error(
        result,
        "Checkpoint",
        "tensor w has type Q5K, which the CUDA matmul does not execute",
    );
}

#[test]
fn checked_tiled_rejects_a_block_length_that_does_not_match_the_shape() {
    // 8 rows of one 256-value Q4_K block need 8 * 144 = 1152 bytes.
    let blocks = [0u8; 1151];
    let result = checked_tiled(QuantTensor {
        name: "w",
        ty: GgufType::Q4K,
        n_rows: 8,
        k: 256,
        source_len: blocks.len(),
    });
    assert_error(
        result,
        "Checkpoint",
        "tensor w has 1151 bytes of blocks for 8 rows of 256 values",
    );
}

#[test]
fn checked_tiled_rejects_shapes_the_tiles_cannot_hold() {
    let not_whole_tiles = QuantTensor {
        name: "w",
        ty: GgufType::Q4K,
        n_rows: 8,
        k: 300,
        source_len: 0,
    };
    assert_error(
        checked_tiled(not_whole_tiles),
        "Checkpoint",
        "tensor w has 8 rows of 300 values, which the CUDA tiles cannot hold",
    );
    let partial_group = QuantTensor {
        name: "w",
        ty: GgufType::Q4K,
        n_rows: 7,
        k: 256,
        source_len: 0,
    };
    assert_error(
        checked_tiled(partial_group),
        "Checkpoint",
        "tensor w has 7 rows of 256 values, which the CUDA tiles cannot hold",
    );
}

#[test]
fn checked_tiled_accepts_one_whole_group_of_each_tiled_type() {
    // 8 rows of one 256-value block each: 8 * 144 bytes.
    let q4k = [0u8; 8 * 144];
    let tiled = checked_tiled(QuantTensor {
        name: "w",
        ty: GgufType::Q4K,
        n_rows: 8,
        k: 256,
        source_len: q4k.len(),
    })
    .expect("one Q4_K group is a valid shape");
    assert_eq!(
        tiled,
        Tiled {
            ty: TiledType::Q4K,
            n_rows: 8,
            k: 256
        }
    );
    // 8 rows of one 256-value block each: 8 * 210 bytes.
    let q6k = [0u8; 8 * 210];
    let tiled = checked_tiled(QuantTensor {
        name: "w",
        ty: GgufType::Q6K,
        n_rows: 8,
        k: 256,
        source_len: q6k.len(),
    })
    .expect("one Q6_K group is a valid shape");
    assert_eq!(
        tiled,
        Tiled {
            ty: TiledType::Q6K,
            n_rows: 8,
            k: 256
        }
    );
    // 8 rows of eight 32-value blocks each: 8 * 8 * 34 bytes.
    let q8_0 = [0u8; 8 * 8 * 34];
    let tiled = checked_tiled(QuantTensor {
        name: "w",
        ty: GgufType::Q80,
        n_rows: 8,
        k: 256,
        source_len: q8_0.len(),
    })
    .expect("one Q8_0 group is a valid shape");
    assert_eq!(
        tiled,
        Tiled {
            ty: TiledType::Q80,
            n_rows: 8,
            k: 256
        }
    );
}

#[test]
fn check_q6k_scales_accepts_values_up_to_the_fp16_maximum() {
    // d = 1.0 with scale 127, and the largest finite fp16 product (65504 * 1 is
    // not above F16_MAX, so it is accepted), and the negative mirror of it.
    for block in [
        q6k_block(0x3c00, 127),
        q6k_block(0x7bff, 1),
        q6k_block(0xfbff, -1),
    ] {
        check_q6k_scales("w", 0, &block).expect("the product is within the fp16 range");
    }
}

#[test]
fn check_q6k_scales_rejects_a_product_past_the_fp16_maximum_and_names_the_block() {
    let blocks = [q6k_block(0x2400, 1), q6k_block(0x7bff, 127)].concat();
    assert_error(
        check_q6k_scales("w", 0, &blocks),
        "Checkpoint",
        "tensor w block 1: Q6_K scale product",
    );
    assert_error(
        check_q6k_scales("w", 0, &blocks),
        "Checkpoint",
        "exceeds the fp16 range",
    );
    // 65504 * 2 is one step past the boundary that 65504 * 1 meets.
    assert_error(
        check_q6k_scales("w", 0, &q6k_block(0x7bff, 2)),
        "Checkpoint",
        "exceeds the fp16 range",
    );
}

#[test]
fn check_q6k_scales_rejects_nan_and_infinite_d() {
    // NaN survives a zero scale (NaN * 0 is NaN), so every case is rejected.
    for d_bits in [0x7e00u16, 0x7c00] {
        for scale in [0i8, 1] {
            assert_error(
                check_q6k_scales("w", 0, &q6k_block(d_bits, scale)),
                "Checkpoint",
                "exceeds the fp16 range",
            );
        }
    }
}

#[test]
fn check_q8_0_scales_accepts_products_up_to_exactly_the_fp16_maximum() {
    // 0x5fff is 511.75 and 511.75 * 128 = 65504 exactly: q = -128 counts as
    // 128 and meets the boundary. 0x6000 is 512 and 512 * 127 = 65024.
    // 0x7bff is 65504 itself, times 1. A zero-quant block passes at any
    // finite d.
    let mut zero = q8_0_block(0x7bff, 0);
    zero[3..].fill(0);
    for block in [
        q8_0_block(0x5fff, -128),
        q8_0_block(0xdfff, -128),
        q8_0_block(0x6000, 127),
        q8_0_block(0x7bff, 1),
        zero,
    ] {
        check_q8_0_scales("w", 0, &block).expect("the product is within the fp16 range");
    }
}

#[test]
fn check_q8_0_scales_rejects_a_product_past_the_fp16_maximum_and_names_the_block() {
    // 512 * 128 = 65536 is one fp16 step of d past the 65504 boundary.
    let blocks = [q8_0_block(0x5fff, -128), q8_0_block(0x6000, -128)].concat();
    assert_error(
        check_q8_0_scales("w", 0, &blocks),
        "Checkpoint",
        "tensor w block 1: Q8_0 scale 512 times quant 128 exceeds the fp16 range",
    );
    // The largest quant of the block decides, wherever it sits.
    let mut late = q8_0_block(0x7bff, 1);
    late[33] = 2;
    assert_error(
        check_q8_0_scales("w", 0, &late),
        "Checkpoint",
        "tensor w block 0: Q8_0 scale 65504 times quant 2",
    );
}

#[test]
fn check_q8_0_scales_rejects_nan_and_infinite_d_even_with_zero_quants() {
    for d_bits in [0x7e00u16, 0x7c00, 0xfc00] {
        let mut block = q8_0_block(d_bits, 0);
        block[3..].fill(0);
        assert_error(
            check_q8_0_scales("w", 0, &block),
            "Checkpoint",
            "exceeds the fp16 range",
        );
    }
}

#[test]
fn scale_checks_name_a_block_by_its_index_from_the_tensor_start() {
    // The bad block is the second of a chunk that starts at block 40.
    let q6k = [q6k_block(0x2400, 1), q6k_block(0x7bff, 127)].concat();
    assert_error(
        check_q6k_scales("w", 40, &q6k),
        "Checkpoint",
        "tensor w block 41: Q6_K scale product",
    );
    let q8_0 = [q8_0_block(0x5fff, -128), q8_0_block(0x6000, -128)].concat();
    assert_error(
        check_q8_0_scales("w", 40, &q8_0),
        "Checkpoint",
        "tensor w block 41: Q8_0 scale 512 times quant 128",
    );
}

/// 64 rows of 512 values: 8 groups, each 8 rows of 2 blocks of 256 values.
/// Checks the chunk size for limits around one, three and all groups.
fn chunk_plan_case(ty: GgufType, block_bytes: usize, group_src: usize, group_bytes: usize) {
    let tensor = QuantTensor {
        name: "w",
        ty,
        n_rows: 64,
        k: 512,
        source_len: 64 * 2 * block_bytes,
    };
    let tiled = checked_tiled(tensor).expect("a valid shape");
    let tile_bytes = tiled.tile_vecs().expect("tile vectors") * size_of::<TileVec>();
    assert_eq!(tile_bytes, 8 * group_bytes, "{ty:?}");
    for (limit, per_chunk) in [
        (0, 1),
        (1, 1),
        (group_bytes - 1, 1),
        (group_bytes, 1),
        (3 * group_bytes + 5, 3),
        (7 * group_bytes, 7),
        (8 * group_bytes, 8),
        (usize::MAX, 8),
    ] {
        let plan = ChunkPlan::new(tensor, tiled, tile_bytes, limit);
        assert_eq!(plan.per_chunk, per_chunk, "{ty:?} limit {limit}");
        assert_eq!(plan.groups, 8, "{ty:?}");
        assert_eq!(plan.group_src, group_src, "{ty:?} limit {limit}");
        assert_eq!(plan.group_bytes, group_bytes, "{ty:?} limit {limit}");
    }
}

/// A Q4_K group has the same 2 * 1152 = 2304 bytes as source and as tiles, so
/// only Q6_K (3360 source bytes against 3584 tile bytes) tells `group_src` from
/// `group_bytes`.
#[test]
fn chunk_plans_hold_whole_groups_within_the_limit_and_at_least_one() {
    chunk_plan_case(GgufType::Q4K, 144, 2304, 2304);
    chunk_plan_case(GgufType::Q6K, 210, 3360, 3584);
}

#[test]
fn f16_to_f32_round_trips_every_non_nan_pattern_through_the_oracle() {
    for bits in 0..=u16::MAX {
        let value = f16_to_f32(bits);
        if is_nan_pattern(bits) {
            assert!(value.is_nan(), "pattern {bits:#06x} must decode to NaN");
        } else {
            assert_eq!(
                f32_to_f16_rne(value),
                bits,
                "pattern {bits:#06x} decodes to {value:e}"
            );
        }
    }
}

#[test]
fn f16_to_f32_decodes_the_anchor_values() {
    let anchors = [
        (0x3c00u16, 1.0f32),
        (0xc000, -2.0),
        (0x7bff, 65504.0),
        (0x0001, 2f32.powi(-24)),
        (0x0400, 2f32.powi(-14)),
        (0x8000, -0.0),
    ];
    for (bits, want) in anchors {
        assert_eq!(f16_to_f32(bits).to_bits(), want.to_bits(), "{bits:#06x}");
    }
    assert!(f16_to_f32(0x7c00).is_infinite() && f16_to_f32(0x7c00) > 0.0);
}

#[test]
fn f32_to_f16_oracle_rounds_ties_to_even() {
    let cases: [(f32, u16); 7] = [
        // Between 1.0 and 1 + 2^-10: the even significand (1.0) wins.
        (1.0 + 2f32.powi(-11), 0x3c00),
        // Between 1 + 2^-10 and 1 + 2^-9: the even significand (ending in 10) wins.
        (1.0 + 3.0 * 2f32.powi(-11), 0x3c02),
        // Just below the overflow midpoint 65520 rounds down to 65504.
        (65519.0, 0x7bff),
        // The midpoint 65520 ties to the even "significand" past the maximum: infinity.
        (65520.0, 0x7c00),
        (-0.0, 0x8000),
        // Half the smallest subnormal ties to zero (even), 1.5 of it ties to 2 units.
        (2f32.powi(-25), 0x0000),
        (3.0 * 2f32.powi(-25), 0x0002),
    ];
    for (value, want) in cases {
        assert_eq!(f32_to_f16_rne(value), want, "{value:e}");
    }
}

#[test]
fn token_count_takes_whole_tokens_up_to_max_dim() {
    let tiled = Tiled {
        ty: TiledType::Q4K,
        n_rows: 8,
        k: 256,
    };
    assert_eq!(token_count(tiled, 17 * 256).expect("17 tokens"), 17);
    assert_eq!(
        token_count(tiled, native::MAX_DIM * 256).expect("MAX_DIM tokens"),
        native::MAX_DIM
    );
}

#[test]
fn token_count_rejects_empty_partial_and_oversized_inputs() {
    let tiled = Tiled {
        ty: TiledType::Q4K,
        n_rows: 8,
        k: 256,
    };
    let range = format!("the weight takes 1 to {} tokens of 256", native::MAX_DIM);
    assert_error(token_count(tiled, 0), "BadRequest", &range);
    assert_error(
        token_count(tiled, 300),
        "BadRequest",
        "cuda linear input has 300 values",
    );
    assert_error(
        token_count(tiled, (native::MAX_DIM + 1) * 256),
        "BadRequest",
        &range,
    );
}

#[test]
fn row_range_offsets_whole_groups_by_their_tile_bytes() {
    // Bytes of two 8-row groups (first = 16), written out by hand: a group is
    // `k / 256` tiles of 72 (Q4_K), 112 (Q6_K) or 136 (Q8_0) vectors of 16 bytes.
    let cases = [
        (TiledType::Q4K, 256, 2 * 1152),
        (TiledType::Q4K, 512, 2 * 2304),
        (TiledType::Q6K, 256, 2 * 1792),
        (TiledType::Q6K, 512, 2 * 3584),
        (TiledType::Q80, 256, 2 * 2176),
        (TiledType::Q80, 512, 2 * 4352),
    ];
    for (ty, k, expected) in cases {
        let weight = Tiled { ty, n_rows: 32, k };
        let (view, offset) = row_range(weight, 16, 16).expect("groups 2 and 3");
        assert_eq!((view.n_rows, view.k, view.ty), (16, k, ty), "{ty:?} k={k}");
        assert_eq!(offset, expected, "{ty:?} k={k}");
        let (whole, offset) = row_range(weight, 0, 32).expect("the whole weight");
        assert_eq!((whole.n_rows, offset), (32, 0), "{ty:?} k={k}");
    }
}

#[test]
fn row_range_rejects_partial_empty_and_overlong_ranges() {
    let q4k = Tiled {
        ty: TiledType::Q4K,
        n_rows: 32,
        k: 256,
    };
    for (first, rows) in [
        (4, 8),
        (0, 12),
        (0, 0),
        (24, 16),
        (40, 8),
        (usize::MAX - 7, 16),
    ] {
        assert_error(
            row_range(q4k, first, rows),
            "BadRequest",
            "whole 8-row groups of a weight with 32 rows",
        );
    }
}
