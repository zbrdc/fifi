use super::{
    Kv8DequantCall, Kv8DequantNeeds, Kv8DequantShape, check_shape, check_source_origin,
    kv8_dequant_needs, staging_bytes,
};
use crate::EngError;
use crate::kernels::cuda::{CudaContext, testkit};

const CELLS: usize = 7;
const ROWS: usize = 3;
const HEADS: usize = 2;
const DIM: usize = 64;
const GROUPS: usize = DIM / 32;
const FILL: u8 = 0x55;

fn code(head: usize, row: usize, col: usize) -> i8 {
    (head * 64 + row * 4 + col % 5) as i8 - 50
}

fn scale_bits(head: usize, row: usize, group: usize) -> u16 {
    0x3c00 + ((head + row % 2 + group) as u16) * 0x400
}

fn native_prefix(origin: Option<usize>, rows: usize) -> Result<Vec<u8>, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    for head in 0..HEADS {
        for row in 0..CELLS {
            codes.extend((0..DIM).map(|col| code(head, row, col) as u8));
            for group in 0..GROUPS {
                scales.extend(scale_bits(head, row, group).to_le_bytes());
            }
        }
    }
    let codes = ctx.upload(&codes)?;
    let scales = ctx.upload(&scales)?;
    let out_bytes = CELLS * HEADS * DIM * 2;
    let out = ctx.upload(&vec![FILL; out_bytes + 16])?;
    let call = Kv8DequantCall {
        shape: Kv8DequantShape {
            capacity: CELLS,
            rows,
            n_kv_head: HEADS,
            head_dim: DIM,
        },
        codes: codes.range(0, codes.bytes())?,
        scales: scales.range(0, scales.bytes())?,
        out: out.range(0, out_bytes)?,
    };
    if let Some(origin) = origin {
        ctx.kv8_dequant_segment(&stream, call, origin)?;
    } else {
        ctx.kv8_dequant(&stream, call)?;
    }
    let mut result = vec![0; out_bytes + 16];
    out.read_at(&stream, 0, &mut result)?;
    Ok(result)
}

#[test]
#[ignore = "native GPU correctness; precompiled binary and GPU lock required"]
fn native_dequant_transposes_only_the_active_prefix() {
    if !testkit::device_present("native_dequant_transposes_only_the_active_prefix") {
        return;
    }
    let output = native_prefix(None, ROWS).expect("native KV8-to-FP16 active-prefix conversion");
    assert_segment_rows(&output, 0, ROWS);
}

fn assert_segment_rows(output: &[u8], origin: usize, rows: usize) {
    let active = rows * HEADS * DIM * 2;
    let mut index = 0;
    for row in 0..rows {
        for head in 0..HEADS {
            for col in 0..DIM {
                let bits = u16::from_le_bytes(output[index..index + 2].try_into().unwrap());
                let value = testkit::f16_to_f32(bits);
                let source_row = origin + row;
                let scale = testkit::f16_to_f32(scale_bits(head, source_row, col / 32));
                assert_eq!(value, f32::from(code(head, source_row, col)) * scale);
                index += 2;
            }
        }
    }
    assert_eq!(index, active);
    assert!(output[active..].iter().all(|&byte| byte == FILL));
}

#[test]
#[ignore = "native GPU correctness; precompiled binary and GPU lock required"]
fn native_dequant_segments_keep_capacity_stride_and_output_canaries() {
    if !testkit::device_present("native_dequant_segments_keep_capacity_stride_and_output_canaries")
    {
        return;
    }
    for (origin, rows) in [(0, ROWS), (1, ROWS), (4, ROWS), (6, 1)] {
        let output = native_prefix(Some(origin), rows).expect("absolute source segment");
        assert_segment_rows(&output, origin, rows);
    }
}

fn native_refusals() -> Result<Vec<bool>, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let buffer = ctx.upload(&[0u8; 4096])?;
    let shape = Kv8DequantShape {
        capacity: CELLS,
        rows: ROWS,
        n_kv_head: HEADS,
        head_dim: DIM,
    };
    let ranges = [
        [(0, 896), (1024, 56), (0, 768)],
        [(0, 896), (1024, 56), (1024, 768)],
        [(0, 895), (1024, 56), (2048, 768)],
        [(0, 896), (1024, 55), (2048, 768)],
        [(0, 896), (1024, 56), (2048, 767)],
        [(1, 896), (1024, 56), (2048, 768)],
        [(0, 896), (1025, 56), (2048, 768)],
        [(0, 896), (1024, 56), (2049, 768)],
    ];
    let mut refused = Vec::new();
    for [codes, scales, out] in ranges {
        let result = ctx.kv8_dequant(
            &stream,
            Kv8DequantCall {
                shape,
                codes: buffer.range(codes.0, codes.1)?,
                scales: buffer.range(scales.0, scales.1)?,
                out: buffer.range(out.0, out.1)?,
            },
        );
        refused.push(matches!(result, Err(EngError::BadRequest(_))));
    }
    for origin in [5, usize::MAX] {
        let result = ctx.kv8_dequant_segment(
            &stream,
            Kv8DequantCall {
                shape,
                codes: buffer.range(0, 896)?,
                scales: buffer.range(1024, 56)?,
                out: buffer.range(2048, 768)?,
            },
            origin,
        );
        refused.push(matches!(result, Err(EngError::BadRequest(_))));
    }
    ctx.ensure_usable()?;
    Ok(refused)
}

#[test]
#[ignore = "native GPU correctness; precompiled binary and GPU lock required"]
fn native_dequant_refuses_bad_ranges_without_poisoning() {
    if !testkit::device_present("native_dequant_refuses_bad_ranges_without_poisoning") {
        return;
    }
    assert_eq!(native_refusals().unwrap(), vec![true; 10]);
}

#[test]
fn checked_geometry_uses_source_capacity_and_only_active_output_rows() {
    let shape = Kv8DequantShape {
        capacity: CELLS,
        rows: ROWS,
        n_kv_head: HEADS,
        head_dim: DIM,
    };
    assert!(check_shape(&shape).is_ok());
    assert_eq!(
        kv8_dequant_needs(&shape),
        Some(Kv8DequantNeeds {
            codes: 896,
            scales: 56,
            out: 768,
        })
    );
    for bad in [
        Kv8DequantShape {
            capacity: 0,
            ..shape
        },
        Kv8DequantShape {
            capacity: 1 << 30,
            ..shape
        },
        Kv8DequantShape { rows: 0, ..shape },
        Kv8DequantShape {
            rows: CELLS + 1,
            ..shape
        },
        Kv8DequantShape {
            n_kv_head: 0,
            ..shape
        },
        Kv8DequantShape {
            n_kv_head: 65,
            ..shape
        },
        Kv8DequantShape {
            head_dim: 0,
            ..shape
        },
        Kv8DequantShape {
            head_dim: 33,
            ..shape
        },
        Kv8DequantShape {
            head_dim: 1056,
            ..shape
        },
    ] {
        assert!(check_shape(&bad).is_err());
        assert!(kv8_dequant_needs(&bad).is_none());
    }
}

#[test]
fn segment_origin_is_bounded_by_the_source_capacity() {
    let shape = Kv8DequantShape {
        capacity: CELLS,
        rows: ROWS,
        n_kv_head: HEADS,
        head_dim: DIM,
    };
    assert_eq!(check_source_origin(&shape, 0).unwrap(), 0);
    assert_eq!(check_source_origin(&shape, 4).unwrap(), 4);
    for origin in [5, CELLS, usize::MAX] {
        assert!(check_source_origin(&shape, origin).is_err());
    }
    let tail = Kv8DequantShape { rows: 1, ..shape };
    assert_eq!(check_source_origin(&tail, CELLS - 1).unwrap(), 6);
}

#[test]
fn staging_charges_both_half_planes_for_actual_capacity() {
    for (capacity, expected) in [
        (33_280, 68_157_440),
        (65_536, 134_217_728),
        (262_144, 536_870_912),
    ] {
        assert_eq!(staging_bytes(capacity, 2, 256).unwrap(), expected);
    }
}

#[test]
fn staging_size_refuses_zero_and_overflow() {
    for (capacity, heads, dim) in [
        (0, 2, 256),
        (1, 0, 256),
        (1, 2, 0),
        (usize::MAX, 2, 256),
        (1, usize::MAX, 2),
        (1, 1, usize::MAX),
    ] {
        assert!(staging_bytes(capacity, heads, dim).is_err());
    }
}
