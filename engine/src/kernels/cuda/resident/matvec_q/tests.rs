//! Device tests (3090) of `matvec_q` (seam 1: the Rust wrapper against the C entry against the
//! trimmed vendored `mul_mat_vec_q` kernel). Device tests assume `--test-threads=1`.
//!
//! The host reference decodes the native GGUF blocks exactly as `ggml-common.h` lays them out and
//! the stored `block_q8_1` values exactly as the vendored kernel reads them, and accumulates in
//! f64. The kernel accumulates f32 in a fixed warp order, so the tolerance of test 2 is
//! `64 * 2^-24 * sum(|term|)` per output (the stated bound); the max error and the max
//! error ratio are reported.
//!
//! Q5_0 note: the vendored kernel subtracts the q8_1 block's raw f32 input sum `s` (the stored
//! half) for the -16 offset, not `d * sum(qx)`; the reference mirrors that (vecdotq.cuh's
//! `vec_dot_q5_0_q8_1_impl`).

use super::{MatvecCall, QuantType, matvec_needs};
use crate::EngError;
use crate::kernels::cuda::native::resident::matvec_q;
use crate::kernels::cuda::resident::matvec_expert::ExpertMatvecCall;
use crate::kernels::cuda::resident::quantize_q8_1::{MATRIX_ROW_PADDING, QuantizeCall, q8_1_bytes};
use crate::kernels::cuda::testkit;
use crate::kernels::cuda::{CapturePlan, CudaContext, DeviceBuffer, DeviceRange, DeviceStream};

/// The shapes of tests 2 and 4: (k, rows).
pub(in crate::kernels::cuda) const SHAPES: [(usize, usize); 5] = [
    (2048, 512),
    (512, 2048),
    (4096, 2048),
    (2048, 32),
    (2048, 8192),
];

/// k 2304 (nine K-blocks, on every type's grid) pads to 2560: the activation columns of an
/// `ncols` launch are 80 q8_1 blocks apart, not 72. Every shape of `SHAPES` has k % 512 == 0,
/// where the padded and the unpadded stride coincide, so this one pins the column stride.
pub(in crate::kernels::cuda) const PADDED_SHAPE: (usize, usize) = (2304, 64);

const TYPES: [QuantType; 4] = [
    QuantType::Q4K,
    QuantType::Q50,
    QuantType::Q6K,
    QuantType::Q80,
];

/// Host-side block layout and reference of the four matvec types, shared with the expert tests.
pub(in crate::kernels::cuda) mod reference {
    use super::*;

    fn half_at(bytes: &[u8], at: usize) -> f32 {
        testkit::f16_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
    }

    /// ggml's `get_scale_min_k4`: the 6-bit scale and min of Q4_K/Q5_K group `j`.
    fn scale_min_k4(scales: &[u8], j: usize) -> (i32, i32) {
        if j < 4 {
            ((scales[j] & 63) as i32, (scales[j + 4] & 63) as i32)
        } else {
            (
                ((scales[j + 4] & 0xF) as i32) | (((scales[j - 4] >> 6) as i32) << 4),
                ((scales[j + 4] >> 4) as i32) | (((scales[j] >> 6) as i32) << 4),
            )
        }
    }

    /// Unsigned quant of Q6_K element i (ggml's dequantize_row_q6_K layout).
    fn q6_of(block: &[u8], i: usize) -> i32 {
        let (half, quarter, l) = (i / 128, (i % 128) / 32, i % 32);
        let ql = block[64 * half + l + 32 * (quarter & 1)];
        let qh = block[128 + 32 * half + l];
        let lo4 = if quarter < 2 { ql & 0xF } else { ql >> 4 };
        (lo4 as i32) | ((((qh >> (2 * quarter)) & 3) as i32) << 4)
    }

    /// Dequantized weight value `i` of one block (d, dmin, scales, qs bytes at their offsets).
    pub(super) fn weight_value(ty: QuantType, block: &[u8], i: usize) -> f64 {
        match ty {
            QuantType::Q80 => half_at(block, 0) as f64 * (block[2 + i] as i8) as f64,
            QuantType::Q50 => {
                let d = half_at(block, 0) as f64;
                let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
                let nib = if i < 16 {
                    (block[6 + i] & 0xF) as i32
                } else {
                    (block[6 + i - 16] >> 4) as i32
                };
                let q = nib | (((qh >> i) & 1) as i32) << 4;
                d * (q - 16) as f64
            }
            QuantType::Q4K => {
                let (d, dmin) = (half_at(block, 0) as f64, half_at(block, 2) as f64);
                // ggml's Q4_K layout: within each 64-value unit, the low nibbles of the
                // 32 qs bytes carry values [0, 32) and the high nibbles carry [32, 64);
                // 16 * half + l == i % 32, spelled out to name the quarter.
                let (j, half, nib, l) = (i / 64, (i % 32) / 16, (i % 64) / 32, i % 16);
                let (sc, m) = scale_min_k4(&block[4..16], i / 32);
                let q = (block[16 + 32 * j + 16 * half + l] >> (4 * nib)) & 0xF;
                d * sc as f64 * q as f64 - dmin * m as f64
            }
            QuantType::Q6K => {
                let d = half_at(block, 208) as f64;
                let sc = block[192 + i / 16] as i8;
                d * sc as f64 * (q6_of(block, i) - 32) as f64
            }
        }
    }

    /// One stored q8_1 block: (d, s, qs) of `values_per_block` 32.
    pub(super) fn q8_1_block(bytes: &[u8], at: usize) -> (f32, f32, Vec<i8>) {
        let d = half_at(bytes, at);
        let s = half_at(bytes, at + 2);
        let qs = (0..32).map(|i| bytes[at + 4 + i] as i8).collect();
        (d, s, qs)
    }

    /// The reference dot of one output row against one activation row of q8_1 blocks, with the
    /// sum of the absolute terms for the tolerance. Q5_0 uses the kernel's raw-sum offset.
    pub(in crate::kernels::cuda) fn reference_dot(
        ty: QuantType,
        w: &[u8],
        x_q8: &[u8],
        k: usize,
    ) -> (f64, f64) {
        let (qk, bb) = (ty.block_size(), ty.block_bytes());
        let blocks = k / qk;
        // One q8_1 block per 32 values: a 256-value K-quant block reads eight of them.
        let x_all: Vec<(f32, f32, Vec<i8>)> =
            (0..k / 32).map(|b| q8_1_block(x_q8, b * 36)).collect();
        let mut sum = 0.0f64;
        let mut abs = 0.0f64;
        for b in 0..blocks {
            let block = &w[b * bb..(b + 1) * bb];
            if ty == QuantType::Q50 {
                // d5 * (d8 * sum(q*qx) - 16 * s), the kernel's exact form.
                let (xd, xs, xq) = &x_all[b];
                let (xd, xs) = (*xd, *xs);
                let d = half_at(block, 0) as f64;
                let mut sq = 0i64;
                for i in 0..32 {
                    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
                    let nib = if i < 16 {
                        (block[6 + i] & 0xF) as i64
                    } else {
                        (block[6 + i - 16] >> 4) as i64
                    };
                    let q = nib | ((((qh >> i) & 1) as i64) << 4);
                    sq += q * xq[i] as i64;
                }
                let term = d * (xd as f64 * sq as f64 - 16.0 * xs as f64);
                sum += term;
                abs += term.abs();
                continue;
            }
            for i in 0..qk {
                let (xd, _, xq) = &x_all[b * (qk / 32) + i / 32];
                let term = weight_value(ty, block, i) * (*xd as f64 * xq[i % 32] as f64);
                sum += term;
                abs += term.abs();
            }
        }
        (sum, abs)
    }

    /// Builds `rows` rows of `k` values of random native blocks of `ty` from `rng`.
    pub(in crate::kernels::cuda) fn build_weight(
        ty: QuantType,
        rows: usize,
        k: usize,
        rng: &mut testkit::Rng,
    ) -> Vec<u8> {
        let (qk, bb) = (ty.block_size(), ty.block_bytes());
        let mut bytes = vec![0u8; rows * (k / qk) * bb];
        for block in bytes.chunks_mut(bb) {
            let d = 0.05 + rng.below(190) as f64 / 100.0;
            let dmin = 0.01 + rng.below(190) as f64 / 1000.0;
            let d_bits = testkit::f64_to_f16_bits(d);
            let dmin_bits = testkit::f64_to_f16_bits(dmin);
            block[0..2].copy_from_slice(&d_bits.to_le_bytes());
            block[2..4].copy_from_slice(&dmin_bits.to_le_bytes());
            for byte in &mut block[4..] {
                *byte = (rng.next_u64() & 0xFF) as u8;
            }
            if ty == QuantType::Q6K {
                // Q6_K keeps its d half at 208; a fully random one could be NaN or inf, so the
                // builder writes a sane scale there over the random fill.
                let d6 = testkit::f64_to_f16_bits(0.05 + rng.below(190) as f64 / 100.0);
                block[208..210].copy_from_slice(&d6.to_le_bytes());
            }
        }
        bytes
    }
}

/// Uploads a weight and quantizes `ncols` random rows, returning (w, x_q8, x_q8 bytes per row).
fn rig(
    ctx: &CudaContext,
    stream: &DeviceStream,
    ty: QuantType,
    k: usize,
    rows: usize,
    ncols: usize,
    seed: u64,
) -> (DeviceBuffer, DeviceBuffer, usize) {
    let mut rng = testkit::Rng::new(seed);
    let w_bytes = reference::build_weight(ty, rows, k, &mut rng);
    let w = testkit::upload_bytes(ctx, &w_bytes);
    let x: Vec<f32> = (0..ncols * k)
        .map(|_| (rng.normal() as f32) * 0.05)
        .collect();
    let x_f32 = testkit::upload_f32(ctx, &x);
    let x_bytes = q8_1_bytes(k, ncols).expect("size");
    let x_q8 = testkit::upload_filled(ctx, 0xAA, x_bytes);
    ctx.quantize_q8_1(
        stream,
        QuantizeCall {
            x: x_f32.range(0, x.len() * 4).expect("x range"),
            y: x_q8.range(0, x_bytes).expect("y range"),
            k,
            ncols,
        },
    )
    .expect("quantize");
    stream.synchronize().expect("drain");
    let row_bytes = q8_1_bytes(k, 1).expect("size");
    (w, x_q8, row_bytes)
}

fn read_f32(ctx: &CudaContext, buffer: &DeviceBuffer, len: usize) -> Vec<f32> {
    testkit::floats(&testkit::read_bytes(ctx, buffer, len))
}

#[test]
fn quant_sizes_are_pure_and_exact() {
    // Weight bytes per type: Q4_K 144 per 256, Q5_0 22 per 32, Q6_K 210 per 256, Q8_0 34 per 32.
    assert_eq!(QuantType::Q4K.block_size(), 256);
    assert_eq!(QuantType::Q4K.block_bytes(), 144);
    assert_eq!(QuantType::Q50.block_size(), 32);
    assert_eq!(QuantType::Q50.block_bytes(), 22);
    assert_eq!(QuantType::Q6K.block_size(), 256);
    assert_eq!(QuantType::Q6K.block_bytes(), 210);
    assert_eq!(QuantType::Q80.block_size(), 32);
    assert_eq!(QuantType::Q80.block_bytes(), 34);
    assert_eq!(QuantType::Q4K.weight_bytes(512, 2048), Some(512 * 8 * 144));
    assert_eq!(
        QuantType::Q80.weight_bytes(8192, 2048),
        Some(8192 * 64 * 34)
    );
    assert_eq!(QuantType::Q4K.weight_bytes(8, 100), None);
    assert_eq!(QuantType::Q4K.weight_bytes(usize::MAX, 256), None);
    assert_eq!(q8_1_bytes(2048, 1), Some(64 * 36));
    assert_eq!(MATRIX_ROW_PADDING, 512);
}

/// The byte counts `plan_matvec` checks a call against.
#[test]
fn matvec_needs_are_pure_and_exact() {
    let need = matvec_needs(QuantType::Q4K, 2048, 512, 2).expect("sizes");
    assert_eq!(
        (need.w, need.x_q8, need.y),
        (512 * 8 * 144, 64 * 36 * 2, 512 * 2 * 4)
    );
    // k 2304 pads to 2560 on the activation side only: 80 q8_1 blocks per column, 72 weight blocks.
    let need = matvec_needs(QuantType::Q80, 2304, 64, 2).expect("sizes");
    assert_eq!(
        (need.w, need.x_q8, need.y),
        (64 * 72 * 34, 80 * 36 * 2, 64 * 2 * 4)
    );
    assert!(matvec_needs(QuantType::Q4K, 100, 64, 1).is_none());
    assert!(matvec_needs(QuantType::Q4K, 2048, usize::MAX, 1).is_none());
    assert!(matvec_needs(QuantType::Q80, 2048, usize::MAX / 8 + 1, 8).is_none());
}

#[test]
fn matvec_q_matches_the_integer_reference() {
    if !testkit::device_present("matvec_q_matches_the_integer_reference") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let ncols = 2;
    let mut max_ratio = 0.0f64;
    for (k, rows) in SHAPES.into_iter().chain([PADDED_SHAPE]) {
        for (ti, ty) in TYPES.iter().enumerate() {
            let seed = 0x1000_0000 + ti as u64 * 100 + k as u64 + rows as u64;
            let (w, x_q8, row_bytes) = rig(&ctx, &stream, *ty, k, rows, ncols, seed);
            let y_bytes = rows * ncols * 4;
            let y = testkit::upload_filled(&ctx, 0xAA, y_bytes);
            ctx.matvec_q(
                &stream,
                MatvecCall {
                    ty: *ty,
                    w: w.range(0, ty.weight_bytes(rows, k).expect("size"))
                        .expect("w"),
                    x_q8: x_q8
                        .range(0, q8_1_bytes(k, ncols).expect("size"))
                        .expect("x"),
                    y: y.range(0, y_bytes).expect("y"),
                    k,
                    rows,
                    ncols,
                },
            )
            .expect("matvec");
            stream.synchronize().expect("drain");
            let got = read_f32(&ctx, &y, y_bytes);
            let w_bytes = testkit::read_bytes(&ctx, &w, ty.weight_bytes(rows, k).expect("size"));
            let x_bytes = testkit::read_bytes(&ctx, &x_q8, row_bytes * ncols);
            for col in 0..ncols {
                for row in 0..rows {
                    let (reference, abs) = reference::reference_dot(
                        *ty,
                        &w_bytes[row * (k / ty.block_size()) * ty.block_bytes()..],
                        &x_bytes[col * row_bytes..],
                        k,
                    );
                    let tolerance = 64.0 * 2f64.powi(-24) * abs;
                    let error = (got[col * rows + row] as f64 - reference).abs();
                    let ratio = if tolerance > 0.0 {
                        error / tolerance
                    } else {
                        error
                    };
                    max_ratio = max_ratio.max(ratio);
                    assert!(
                        error <= tolerance,
                        "{ty:?} k {k} rows {rows} col {col} row {row}: error {error} over tol \
                         {tolerance} (reference {reference})"
                    );
                }
            }
        }
    }
    println!("matvec_q reference: max error ratio {max_ratio:.3} of the tolerance");
}

#[test]
fn matvec_q_is_deterministic() {
    if !testkit::device_present("matvec_q_is_deterministic") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (k, rows) = (2048, 512);
    for (ti, ty) in TYPES.iter().enumerate() {
        let (w, x_q8, _) = rig(&ctx, &stream, *ty, k, rows, 1, 0x2000_0000 + ti as u64);
        let y_bytes = rows * 4;
        let y = testkit::upload_filled(&ctx, 0xAA, y_bytes);
        let run = |bytes: &[u8]| {
            y.write_at(&stream, 0, bytes).expect("reset y");
            ctx.matvec_q(
                &stream,
                MatvecCall {
                    ty: *ty,
                    w: w.range(0, ty.weight_bytes(rows, k).expect("size"))
                        .expect("w"),
                    x_q8: x_q8.range(0, q8_1_bytes(k, 1).expect("size")).expect("x"),
                    y: y.range(0, y_bytes).expect("y"),
                    k,
                    rows,
                    ncols: 1,
                },
            )
            .expect("matvec");
            stream.synchronize().expect("drain");
            testkit::read_bytes(&ctx, &y, y_bytes)
        };
        let first = run(&vec![0xAA; y_bytes]);
        for _ in 1..10 {
            let again = run(&vec![0xAA; y_bytes]);
            assert_eq!(first, again, "{ty:?}: repeated launch differs");
        }
    }
}

#[test]
fn matvec_q_columns_do_not_depend_on_ncols() {
    if !testkit::device_present("matvec_q_columns_do_not_depend_on_ncols") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let ncols_max = 8;
    println!("row-invariance table (type, k, rows, n, first differing column)");
    let mut stop_worthy = Vec::new();
    for (k, rows) in SHAPES {
        for (ti, ty) in TYPES.iter().enumerate() {
            let seed = 0x3000_0000 + ti as u64 * 100 + k as u64 + rows as u64;
            let (w, x_q8, row_bytes) = rig(&ctx, &stream, *ty, k, rows, ncols_max, seed);
            let w_range_bytes = ty.weight_bytes(rows, k).expect("size");
            let mut singles = Vec::with_capacity(ncols_max);
            for col in 0..ncols_max {
                let y_bytes = rows * 4;
                let y = testkit::upload_filled(&ctx, 0xAA, y_bytes);
                ctx.matvec_q(
                    &stream,
                    MatvecCall {
                        ty: *ty,
                        w: w.range(0, w_range_bytes).expect("w"),
                        x_q8: x_q8.range(col * row_bytes, row_bytes).expect("x col"),
                        y: y.range(0, y_bytes).expect("y"),
                        k,
                        rows,
                        ncols: 1,
                    },
                )
                .expect("single-column matvec");
                stream.synchronize().expect("drain");
                singles.push(testkit::read_bytes(&ctx, &y, y_bytes));
            }
            for n in 2..=ncols_max {
                let y_bytes = rows * n * 4;
                let y = testkit::upload_filled(&ctx, 0xAA, y_bytes);
                ctx.matvec_q(
                    &stream,
                    MatvecCall {
                        ty: *ty,
                        w: w.range(0, w_range_bytes).expect("w"),
                        x_q8: x_q8.range(0, row_bytes * n).expect("x"),
                        y: y.range(0, y_bytes).expect("y"),
                        k,
                        rows,
                        ncols: n,
                    },
                )
                .expect("n-column matvec");
                stream.synchronize().expect("drain");
                let multi = testkit::read_bytes(&ctx, &y, y_bytes);
                let mut first_diff: Option<usize> = None;
                for col in 0..n {
                    let column = &multi[col * rows * 4..(col + 1) * rows * 4];
                    if column != singles[col].as_slice() && first_diff.is_none() {
                        first_diff = Some(col);
                    }
                }
                println!(
                    "{ty:?} {k} {rows} {n} {}",
                    first_diff.map_or(-1, |c| c as i64)
                );
                if let Some(col) = first_diff
                    && (2..=4).contains(&n)
                {
                    stop_worthy.push((format!("{ty:?}"), k, rows, n, col));
                }
            }
        }
    }
    // Columns 2..4 must be bit-identical to the 1-column launch; a difference there fails the
    // test after the report. Columns 5..8 are reported only (verify windows then run the ncols-1
    // kernel once per row).
    assert!(
        stop_worthy.is_empty(),
        "row-invariance failures at n = 2..4: {stop_worthy:?}"
    );
}

#[test]
fn quantize_and_matvec_inside_a_capture_equal_eager() {
    if !testkit::device_present("quantize_and_matvec_inside_a_capture_equal_eager") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (k, rows) = (2048, 64);
    let ty = QuantType::Q4K;
    let mut rng = testkit::Rng::new(0x4000_0001);
    let w_bytes = reference::build_weight(ty, rows, k, &mut rng);
    let w = testkit::upload_bytes(&ctx, &w_bytes);
    let x: Vec<f32> = (0..k).map(|_| (rng.normal() as f32) * 0.05).collect();
    let x_dev = testkit::upload_f32(&ctx, &x);
    let x_bytes = q8_1_bytes(k, 1).expect("size");
    let y_bytes = rows * 4;
    let buffers = |ctx: &CudaContext| {
        (
            testkit::upload_filled(ctx, 0xAA, x_bytes),
            testkit::upload_filled(ctx, 0xAA, y_bytes),
        )
    };
    let call = |ctx: &CudaContext,
                stream: &DeviceStream,
                x_q8: &DeviceBuffer,
                y: &DeviceBuffer,
                x_dev: &DeviceBuffer,
                w: &DeviceBuffer|
     -> Result<(), crate::EngError> {
        ctx.quantize_q8_1(
            stream,
            QuantizeCall {
                x: x_dev.range(0, k * 4)?,
                y: x_q8.range(0, x_bytes)?,
                k,
                ncols: 1,
            },
        )?;
        ctx.matvec_q(
            stream,
            MatvecCall {
                ty,
                w: w.range(0, ty.weight_bytes(rows, k).expect("size"))?,
                x_q8: x_q8.range(0, x_bytes)?,
                y: y.range(0, y_bytes)?,
                k,
                rows,
                ncols: 1,
            },
        )
    };
    let (eager_q8, eager_y) = buffers(&ctx);
    call(&ctx, &stream, &eager_q8, &eager_y, &x_dev, &w).expect("eager");
    stream.synchronize().expect("drain eager");
    let eager = testkit::read_bytes(&ctx, &eager_y, y_bytes);
    let (cap_q8, cap_y) = buffers(&ctx);
    let plan = CapturePlan {
        pins: &[],
        holds: Vec::new(),
    };
    let (graph, ()) = ctx
        .capture(&stream, plan, || {
            call(&ctx, &stream, &cap_q8, &cap_y, &x_dev, &w)
        })
        .expect("capture");
    assert_eq!(graph.nodes(), 2, "quantize 1 node + matvec 1 node");
    for replay in 0..3 {
        cap_y
            .write_at(&stream, 0, &vec![0xAA; y_bytes])
            .expect("reset y");
        graph.launch(&stream).expect("replay");
        stream.synchronize().expect("drain the replay");
        assert_eq!(
            testkit::read_bytes(&ctx, &cap_y, y_bytes),
            eager,
            "replay {replay} differs from eager"
        );
    }
}

#[test]
fn matvec_writes_only_its_rows() {
    if !testkit::device_present("matvec_writes_only_its_rows") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (k, rows) = (2048, 100);
    let ty = QuantType::Q80;
    let (w, x_q8, _) = rig(&ctx, &stream, ty, k, rows, 1, 0x5000_0001);
    let guard = 64;
    let y_bytes = rows * 4;
    let y = testkit::upload_filled(&ctx, 0xAA, y_bytes + guard);
    ctx.matvec_q(
        &stream,
        MatvecCall {
            ty,
            w: w.range(0, ty.weight_bytes(rows, k).expect("size"))
                .expect("w"),
            x_q8: x_q8.range(0, q8_1_bytes(k, 1).expect("size")).expect("x"),
            y: y.range(0, y_bytes).expect("y"),
            k,
            rows,
            ncols: 1,
        },
    )
    .expect("matvec");
    stream.synchronize().expect("drain");
    let bytes = testkit::read_bytes(&ctx, &y, y_bytes + guard);
    for (i, byte) in bytes[y_bytes..].iter().enumerate() {
        assert_eq!(*byte, 0xAA, "canary {i} beyond y was written");
    }
}

/// The padded tail of a live column is written by the quantizer and never read by the matvec.
///
/// k 2304 is on every type's grid (9 x 256) and pads to 2560, so each live column ends in 8 padded
/// q8_1 blocks. The buffer is exactly the live columns (no unused capacity columns: those are
/// never read whatever they hold) and starts as 0xFF. After the quantize the tail of every live
/// column must be zero (the quantizer writes it), and the result of the matvec must be
/// bit-identical after that tail is overwritten with poison (a matvec that read the padded width
/// would change).
#[test]
fn matvec_q_is_not_affected_by_stale_activation_padding() {
    if !testkit::device_present("matvec_q_is_not_affected_by_stale_activation_padding") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let (k, rows) = (2304, 64);
    let ncols = 2;
    let column_bytes = q8_1_bytes(k, 1).expect("size");
    let live_bytes = (k / 32) * 36;
    assert!(live_bytes < column_bytes, "k 2304 has a padded tail");
    let x_bytes = column_bytes * ncols;
    let y_bytes = rows * ncols * 4;
    for ty in TYPES {
        let mut rng = testkit::Rng::new(0x6000_0000 + ty.ggml() as u64);
        let w_bytes = reference::build_weight(ty, rows, k, &mut rng);
        let w = testkit::upload_bytes(&ctx, &w_bytes);
        let x: Vec<f32> = (0..ncols * k)
            .map(|_| (rng.normal() as f32) * 0.05)
            .collect();
        let x_dev = testkit::upload_f32(&ctx, &x);
        let x_q8 = testkit::upload_filled(&ctx, 0xFF, x_bytes);
        ctx.quantize_q8_1(
            &stream,
            QuantizeCall {
                x: x_dev.range(0, x.len() * 4).expect("x"),
                y: x_q8.range(0, x_bytes).expect("x_q8"),
                k,
                ncols,
            },
        )
        .expect("quantize");
        stream.synchronize().expect("drain");
        let quantized = testkit::read_bytes(&ctx, &x_q8, x_bytes);
        for col in 0..ncols {
            let tail = &quantized[col * column_bytes + live_bytes..(col + 1) * column_bytes];
            assert!(
                tail.iter().all(|&byte| byte == 0),
                "{ty:?}: the quantizer left stale bytes in the padded tail of column {col}"
            );
        }
        let matvec = || {
            let y = testkit::upload_filled(&ctx, 0xAA, y_bytes);
            ctx.matvec_q(
                &stream,
                MatvecCall {
                    ty,
                    w: w.range(0, ty.weight_bytes(rows, k).expect("size"))
                        .expect("w"),
                    x_q8: x_q8.range(0, x_bytes).expect("x"),
                    y: y.range(0, y_bytes).expect("y"),
                    k,
                    rows,
                    ncols,
                },
            )
            .expect("matvec");
            stream.synchronize().expect("drain");
            testkit::read_bytes(&ctx, &y, y_bytes)
        };
        let clean = matvec();
        // Poison the padded tail of every live column: 0x7f bytes are a NaN d and s with codes of 127.
        let poison = vec![0x7f_u8; column_bytes - live_bytes];
        for col in 0..ncols {
            x_q8.write_at(&stream, col * column_bytes + live_bytes, &poison)
                .expect("poison the padded tail");
        }
        let after = testkit::read_bytes(&ctx, &x_q8, x_bytes);
        for col in 0..ncols {
            let tail = &after[col * column_bytes + live_bytes..(col + 1) * column_bytes];
            assert!(
                tail.iter().all(|&byte| byte == 0x7f),
                "{ty:?}: the poison landed"
            );
        }
        assert_eq!(
            matvec(),
            clean,
            "{ty:?}: a poisoned padded tail changed the result"
        );
    }
}

/// The flush buffer of the cold column: FLUSH_ROWS rows of FLUSH_K Q8_0 values, 272 MB.
const FLUSH_K: usize = 1024;
const FLUSH_ROWS: usize = 250_000;

/// A device buffer of `bytes` bytes (a multiple of 4) filled with the quiet NaN 0x7fc00000, the
/// poison of the bench outputs: unlike 0xAA, which is the finite -3.0e-13, it fails a non-NaN assert.
fn nan_filled(ctx: &CudaContext, bytes: usize) -> DeviceBuffer {
    let pattern = 0x7fc0_0000_u32.to_le_bytes();
    let fill: Vec<u8> = pattern.iter().copied().cycle().take(bytes).collect();
    testkit::upload_bytes(ctx, &fill)
}

/// Times `launch` `iters` times between two events and returns the microseconds per launch; the
/// warmup before the window is `iters.min(10)` launches.
fn bench(ctx: &CudaContext, stream: &DeviceStream, iters: usize, launch: &dyn Fn()) -> f32 {
    for _ in 0..iters.min(10) {
        launch();
    }
    stream.synchronize().expect("warmup drain");
    let start = ctx.bench_mark(stream).expect("start event");
    for _ in 0..iters {
        launch();
    }
    let end = ctx.bench_mark(stream).expect("end event");
    stream.synchronize().expect("bench drain");
    let warm = ctx.bench_elapsed_ms(start, end).expect("elapsed");
    ctx.bench_destroy(start).expect("destroy");
    ctx.bench_destroy(end).expect("destroy");
    warm * 1000.0 / iters as f32
}

#[test]
fn matvec_report_gbps() {
    if !testkit::device_present("matvec_report_gbps") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    // The debug cargo profile runs unoptimised device code (about 100x slower: qkv 3350 us,
    // lm_head 191 ms), so its timings are meaningless. The loops shrink to a smoke pass; every
    // correctness check stays.
    let (warm_iters, cold_iters) = if cfg!(debug_assertions) {
        (5, 3)
    } else {
        (200, 20)
    };
    if cfg!(debug_assertions) {
        println!("debug build: timing shrunk to {warm_iters} warm / {cold_iters} cold passes");
    }
    // The six dense shapes of the design table (the three expert rows are measured in
    // matvec_expert's tests, which own the expert wrapper).
    let mut shapes: Vec<(&str, QuantType, usize, usize)> = vec![
        ("qkv / attn_q", QuantType::Q80, 2048, 8192),
        ("attn_gate", QuantType::Q80, 2048, 4096),
        ("ssm_out / attn_output", QuantType::Q80, 4096, 2048),
        ("shared gate/up, attn k/v", QuantType::Q80, 2048, 512),
        ("shared down", QuantType::Q80, 512, 2048),
        ("alpha / beta", QuantType::Q80, 2048, 32),
    ];
    // The lm_head row: 417 MB of Q6_K. Skipped when less than 2 GiB stay free; the free-memory
    // query rides the test-support entry of the same .cu.
    let free = matvec_q::free_bytes_raw().unwrap_or(0);
    if free >= 2 * 1024 * 1024 * 1024 {
        shapes.push(("lm_head (Q6_K)", QuantType::Q6K, 2048, 248_320));
    } else {
        println!("lm_head row skipped: only {free} bytes free");
    }
    // The L2 flush of the cold numbers: a dense Q8_0 matvec over ALL FLUSH_ROWS rows of a 272 MB
    // weight buffer, so the launch reads every byte of it and evicts the 3090's 6 MB L2. (The
    // first version passed this buffer with rows: 1 and read 1 KB of it.) The flush owns its
    // activation and output, so it never borrows a shape's buffers.
    let flush_w_bytes = QuantType::Q80
        .weight_bytes(FLUSH_ROWS, FLUSH_K)
        .expect("size");
    let (l2_flush, flush_x, flush_y) = (
        testkit::upload_filled(&ctx, 0x5A, flush_w_bytes),
        testkit::upload_filled(&ctx, 0x5A, q8_1_bytes(FLUSH_K, 1).expect("size")),
        testkit::upload_filled(&ctx, 0x5A, FLUSH_ROWS * 4),
    );
    let flush = || {
        ctx.matvec_q(
            &stream,
            MatvecCall {
                ty: QuantType::Q80,
                w: l2_flush.range(0, flush_w_bytes).expect("flush w"),
                x_q8: flush_x
                    .range(0, q8_1_bytes(FLUSH_K, 1).expect("size"))
                    .expect("flush x"),
                y: flush_y.range(0, FLUSH_ROWS * 4).expect("flush y"),
                k: FLUSH_K,
                rows: FLUSH_ROWS,
                ncols: 1,
            },
        )
        .expect("flush matvec");
    };
    println!(
        "warm = {warm_iters} launches between two events; cold = one launch timed alone right \
         after a read of the {} MB flush buffer (L2 evicted); a shape under 6 MB is resident in \
         L2 when warm",
        flush_w_bytes / 1_000_000
    );
    println!("shape                     warm_us  cold_us   GB/s_warm GB/s_cold  %936");
    for (name, ty, k, rows) in shapes {
        let (w, x_q8, _) = rig(&ctx, &stream, ty, k, rows, 1, 0x7000_0000 + rows as u64);
        let y_bytes = rows * 4;
        let y = nan_filled(&ctx, y_bytes);
        let launch = || {
            ctx.matvec_q(
                &stream,
                MatvecCall {
                    ty,
                    w: w.range(0, ty.weight_bytes(rows, k).expect("size"))
                        .expect("w"),
                    x_q8: x_q8.range(0, q8_1_bytes(k, 1).expect("size")).expect("x"),
                    y: y.range(0, y_bytes).expect("y"),
                    k,
                    rows,
                    ncols: 1,
                },
            )
            .expect("matvec");
        };
        let warm_us = bench(&ctx, &stream, warm_iters, &launch);
        // Cold: one launch behind the L2 flush read, timed alone, median of the cold passes.
        let mut cold = Vec::new();
        for _ in 0..cold_iters {
            flush();
            let start = ctx.bench_mark(&stream).expect("start");
            launch();
            let end = ctx.bench_mark(&stream).expect("end");
            stream.synchronize().expect("drain");
            cold.push(ctx.bench_elapsed_ms(start, end).expect("elapsed") * 1000.0);
            ctx.bench_destroy(start).expect("destroy");
            ctx.bench_destroy(end).expect("destroy");
        }
        cold.sort_by(|a, b| a.total_cmp(b));
        let cold_us = cold[cold.len() / 2];
        // y started as NaN: a row the launch did not write, or wrote as NaN, fails here.
        let out = read_f32(&ctx, &y, y_bytes);
        assert!(out.iter().all(|v| !v.is_nan()), "{name}: NaN output");
        let bytes = ty.weight_bytes(rows, k).expect("size") as f64
            + q8_1_bytes(k, 1).expect("size") as f64
            + y_bytes as f64;
        let gbs = |us: f32| bytes / (us as f64 * 1e-6) / 1e9;
        println!(
            "{name:24} {warm_us:8.2} {cold_us:8.2} {:9.0} {:9.0} {:6.1}",
            gbs(warm_us),
            gbs(cold_us),
            gbs(cold_us) / 936.0 * 100.0
        );
    }
    // The three expert rows of the design table: a 256-expert stack when the device has room
    // (else 64 experts, said in the output), n_used 8, the ids rotated over all experts for the
    // cold numbers and fixed for the warm ones, both reported.
    let n_expert = if free >= 600 * 1024 * 1024 { 256 } else { 64 };
    println!("expert rows: {n_expert} experts (of the 256 of the model when memory allows)");
    for (name, shape) in [
        ("expert gate/up", (2048, 512usize)),
        ("expert down", (512, 2048)),
    ] {
        let (k, rows) = shape;
        let ty = QuantType::Q4K;
        let mut rng = testkit::Rng::new(0x7100_0000 + rows as u64);
        let stack_bytes = reference::build_weight(ty, rows * n_expert, k, &mut rng);
        let stack = testkit::upload_bytes(&ctx, &stack_bytes);
        let x: Vec<f32> = (0..k).map(|_| (rng.normal() as f32) * 0.05).collect();
        let x_dev = testkit::upload_f32(&ctx, &x);
        let x_q8_bytes = q8_1_bytes(k, 1).expect("size");
        let x_q8 = testkit::upload_filled(&ctx, 0xAA, x_q8_bytes);
        ctx.quantize_q8_1(
            &stream,
            QuantizeCall {
                x: x_dev.range(0, k * 4).expect("x"),
                y: x_q8.range(0, x_q8_bytes).expect("y"),
                k,
                ncols: 1,
            },
        )
        .expect("quantize");
        let y_bytes = rows * 8 * 4;
        let y = nan_filled(&ctx, y_bytes);
        // 32 id sets, each a different random 8-subset of the experts.
        let id_sets: Vec<DeviceBuffer> = (0..32)
            .map(|_set| {
                let ids: Vec<i32> = (0..8)
                    .map(|_slot| rng.below(n_expert as u64) as i32)
                    .collect();
                ids_buffer(&ctx, &ids)
            })
            .collect();
        let fixed = id_sets[0].range(0, 32).expect("ids");
        let launch = |ids: &DeviceRange| {
            ctx.matvec_expert(
                &stream,
                ExpertMatvecCall {
                    ty,
                    w_stack: stack.range(0, stack_bytes.len()).expect("stack"),
                    ids: *ids,
                    x_q8: x_q8.range(0, x_q8_bytes).expect("x"),
                    x_per_slot: false,
                    y: y.range(0, y_bytes).expect("y"),
                    y_slot_stride: rows,
                    k,
                    rows,
                    n_expert,
                    n_used: 8,
                },
            )
            .expect("expert matvec");
        };
        let warm_us = bench(&ctx, &stream, warm_iters, &|| launch(&fixed));
        let mut cold = Vec::new();
        for set in 0..cold_iters {
            flush();
            let ids = id_sets[set % id_sets.len()].range(0, 32).expect("ids");
            let start = ctx.bench_mark(&stream).expect("start");
            launch(&ids);
            let end = ctx.bench_mark(&stream).expect("end");
            stream.synchronize().expect("drain");
            cold.push(ctx.bench_elapsed_ms(start, end).expect("elapsed") * 1000.0);
            ctx.bench_destroy(start).expect("destroy");
            ctx.bench_destroy(end).expect("destroy");
        }
        cold.sort_by(|a, b| a.total_cmp(b));
        let cold_us = cold[cold.len() / 2];
        let out = read_f32(&ctx, &y, y_bytes);
        assert!(out.iter().all(|v| !v.is_nan()), "{name}: NaN output");
        let bytes = ty.weight_bytes(rows, k).expect("size") as f64 * 8.0
            + x_q8_bytes as f64
            + y_bytes as f64;
        let gbs = |us: f32| bytes / (us as f64 * 1e-6) / 1e9;
        println!(
            "{name:24} {warm_us:8.2} {cold_us:8.2} {:9.0} {:9.0} {:6.1}",
            gbs(warm_us),
            gbs(cold_us),
            gbs(cold_us) / 936.0 * 100.0
        );
    }
}

/// Uploads eight little-endian ids (helper of the expert bench rows).
fn ids_buffer(ctx: &CudaContext, ids: &[i32]) -> DeviceBuffer {
    let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    testkit::upload_bytes(ctx, &bytes)
}

/// Asserts a `BadRequest` whose message contains `needle`, so a refusal test fails when another
/// check refuses the call instead of the one under test.
pub(in crate::kernels::cuda) fn assert_bad_request(
    result: Result<(), EngError>,
    case: &str,
    needle: &str,
) {
    let Err(error) = result else {
        panic!("{case}: expected a BadRequest, got Ok");
    };
    assert!(
        matches!(&error, EngError::BadRequest(message) if message.contains(needle)),
        "{case}: expected BadRequest containing {needle:?}, got {error:?}"
    );
}

/// Whose buffer or stream a refusal case hands the wrapper: the rig's own context, or the second
/// adapter (two contexts is the process limit).
#[derive(Clone, Copy)]
pub(in crate::kernels::cuda) enum Owner {
    Own,
    Other,
}

/// Where `y` of a dense refusal case lies.
#[derive(Clone, Copy)]
enum YIn {
    Own,
    W,
    X,
}

/// One refusal case of the dense wrapper: the numbers of the call and the (offset, bytes) of its
/// ranges inside the rig's buffers.
#[derive(Clone, Copy)]
struct Spec {
    k: usize,
    rows: usize,
    ncols: usize,
    w: (usize, usize),
    x: (usize, usize),
    y: (usize, usize),
    y_in: YIn,
    w_of: Owner,
    stream_of: Owner,
}

/// Q4_K, k 768 (three blocks, padded to 1024 on the activation side), 64 rows, two columns.
const W_NEED: usize = 64 * 3 * 144;
const X_NEED: usize = 2 * 32 * 36;
const Y_NEED: usize = 64 * 2 * 4;
const VALID: Spec = Spec {
    k: 768,
    rows: 64,
    ncols: 2,
    w: (0, W_NEED),
    x: (0, X_NEED),
    y: (0, Y_NEED),
    y_in: YIn::Own,
    w_of: Owner::Own,
    stream_of: Owner::Own,
};

/// The rig buffers are larger than the call needs, so a case can offset, shorten or overlap a range.
const RIG_BYTES: usize = 32_768;

struct RefusalRig {
    stream: DeviceStream,
    w: DeviceBuffer,
    x: DeviceBuffer,
    y: DeviceBuffer,
    ctx: CudaContext,
}

/// The second adapter: a stream and a weight buffer that belong to another context.
struct OtherContext {
    stream: DeviceStream,
    w: DeviceBuffer,
    _ctx: CudaContext,
}

fn attempt(rig: &RefusalRig, other: &OtherContext, spec: Spec) -> Result<(), EngError> {
    let w_buffer = match spec.w_of {
        Owner::Own => &rig.w,
        Owner::Other => &other.w,
    };
    let y_buffer = match spec.y_in {
        YIn::Own => &rig.y,
        YIn::W => &rig.w,
        YIn::X => &rig.x,
    };
    let stream = match spec.stream_of {
        Owner::Own => &rig.stream,
        Owner::Other => &other.stream,
    };
    rig.ctx.matvec_q(
        stream,
        MatvecCall {
            ty: QuantType::Q4K,
            w: w_buffer.range(spec.w.0, spec.w.1)?,
            x_q8: rig.x.range(spec.x.0, spec.x.1)?,
            y: y_buffer.range(spec.y.0, spec.y.1)?,
            k: spec.k,
            rows: spec.rows,
            ncols: spec.ncols,
        },
    )
}

/// For `plan_matvec`, every refusal is a `BadRequest` with the text of its own check, nothing is
/// queued, the backend stays usable, and the valid call of the same rig launches.
#[test]
fn matvec_q_rejects_unsound_calls_before_launching() {
    if !testkit::device_present("matvec_q_rejects_unsound_calls_before_launching") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let rig = RefusalRig {
        stream: ctx.stream().expect("stream"),
        w: testkit::upload_filled(&ctx, 0, RIG_BYTES),
        x: testkit::upload_filled(&ctx, 0, 4096),
        y: testkit::upload_filled(&ctx, 0xAB, 4096),
        ctx,
    };
    let other_ctx = CudaContext::new().expect("second context");
    let other = OtherContext {
        stream: other_ctx.stream().expect("stream of the other context"),
        w: testkit::upload_filled(&other_ctx, 0, RIG_BYTES),
        _ctx: other_ctx,
    };
    let cases: [(&str, &str, Spec); 20] = [
        ("k 0", "needs k as a multiple of", Spec { k: 0, ..VALID }),
        (
            "k off the Q4_K grid",
            "needs k as a multiple of",
            Spec { k: 700, ..VALID },
        ),
        (
            "no rows",
            "positive multiple of 4",
            Spec { rows: 0, ..VALID },
        ),
        (
            "rows 62",
            "positive multiple of 4",
            Spec { rows: 62, ..VALID },
        ),
        (
            "no columns",
            "takes 1 to 8 columns",
            Spec { ncols: 0, ..VALID },
        ),
        (
            "nine columns",
            "takes 1 to 8 columns",
            Spec { ncols: 9, ..VALID },
        ),
        (
            "rows that overflow the byte count",
            "size overflows",
            Spec {
                rows: usize::MAX - 3,
                ..VALID
            },
        ),
        (
            "w one block short",
            "w range holds",
            Spec {
                w: (0, W_NEED - 144),
                ..VALID
            },
        ),
        (
            "w misaligned by 2",
            "w range starts at offset",
            Spec {
                w: (2, W_NEED),
                ..VALID
            },
        ),
        (
            "x one byte short",
            "x_q8 range holds",
            Spec {
                x: (0, X_NEED - 1),
                ..VALID
            },
        ),
        (
            "x unpadded: 24 blocks per column",
            "x_q8 range holds",
            Spec {
                x: (0, 2 * 24 * 36),
                ..VALID
            },
        ),
        (
            "x misaligned by 1",
            "x_q8 range starts at offset",
            Spec {
                x: (1, X_NEED),
                ..VALID
            },
        ),
        (
            "y one float short",
            "y range holds",
            Spec {
                y: (0, Y_NEED - 4),
                ..VALID
            },
        ),
        (
            "y misaligned by 2",
            "y range starts at offset",
            Spec {
                y: (2, Y_NEED),
                ..VALID
            },
        ),
        (
            "y over w",
            "y and w ranges overlap",
            Spec {
                y: (1024, Y_NEED),
                y_in: YIn::W,
                ..VALID
            },
        ),
        (
            "y over the end of w",
            "y and w ranges overlap",
            Spec {
                y: (W_NEED - 4, Y_NEED),
                y_in: YIn::W,
                ..VALID
            },
        ),
        (
            "y over x",
            "y and x_q8 ranges overlap",
            Spec {
                y: (1024, Y_NEED),
                y_in: YIn::X,
                ..VALID
            },
        ),
        (
            "y over the end of x",
            "y and x_q8 ranges overlap",
            Spec {
                y: (X_NEED - 4, Y_NEED),
                y_in: YIn::X,
                ..VALID
            },
        ),
        (
            "w of another context",
            "buffer belongs",
            Spec {
                w_of: Owner::Other,
                ..VALID
            },
        ),
        (
            "a stream of another context",
            "stream belongs",
            Spec {
                stream_of: Owner::Other,
                ..VALID
            },
        ),
    ];
    for (case, needle, spec) in cases {
        assert_bad_request(attempt(&rig, &other, spec), case, needle);
    }
    // No refusal queued anything: y is untouched and the backend works; then the valid call runs.
    rig.stream.synchronize().expect("drain");
    let untouched = testkit::read_bytes(&rig.ctx, &rig.y, 4096);
    assert!(untouched.iter().all(|&byte| byte == 0xAB), "y untouched");
    rig.ctx
        .ensure_usable()
        .expect("the refusals leave the backend usable");
    attempt(&rig, &other, VALID).expect("the valid call launches");
    rig.stream.synchronize().expect("drain");
    let written = testkit::read_bytes(&rig.ctx, &rig.y, Y_NEED);
    assert!(
        written.iter().all(|&byte| byte != 0xAB),
        "the valid call wrote y"
    );
}

/// The GDN input-projection fusion A/B: production launches the four Q8_0 siblings that read one
/// q8_1 row (attn_qkv 8192 rows, attn_gate 4096, ssm_beta 32, ssm_alpha 32, k 2048) as four
/// `matvec_q` calls; the fused arm concatenates the weights row-wise into one 12352-row matrix (a
/// load-time repack, no kernel change) and launches once. Gate: bitwise per segment before any
/// timing (`rows` batching cannot change a row's dot), then exclusive timing, 30 interleaved reps
/// per arm.
#[test]
#[ignore = "bench: exclusive timing slot"]
fn matvec_q_gdn_sibling_fusion_ab_bench() {
    if !testkit::device_present("matvec_q_gdn_sibling_fusion_ab_bench") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let ty = QuantType::Q80;
    let k = 2048;
    let segments: [usize; 4] = [8192, 4096, 32, 32];
    let fused_rows: usize = segments.iter().sum();

    // The load-time repack: four sibling tensors, row-concatenated into one weight buffer.
    let mut rng = testkit::Rng::new(0x5D6E_0001);
    let mut fused_host = Vec::new();
    let mut seg_off = Vec::new();
    for &rows in &segments {
        seg_off.push(fused_host.len());
        fused_host.extend_from_slice(&reference::build_weight(ty, rows, k, &mut rng));
    }
    let w = testkit::upload_bytes(&ctx, &fused_host);
    let bytes_per_row = ty.weight_bytes(1, k).expect("row size");
    assert!(seg_off.iter().all(|&off| off % bytes_per_row == 0));

    let x_bytes = q8_1_bytes(k, 1).expect("size");
    let x: Vec<f32> = (0..k).map(|_| (rng.normal() as f32) * 0.05).collect();
    let x_f32 = testkit::upload_f32(&ctx, &x);
    let x_q8 = testkit::upload_filled(&ctx, 0xAA, x_bytes);
    ctx.quantize_q8_1(
        &stream,
        QuantizeCall {
            x: x_f32.range(0, x.len() * 4).expect("x"),
            y: x_q8.range(0, x_bytes).expect("y"),
            k,
            ncols: 1,
        },
    )
    .expect("quantize");
    stream.synchronize().expect("drain");

    let four_launches = |ys: &[DeviceBuffer]| {
        for (i, &rows) in segments.iter().enumerate() {
            ctx.matvec_q(
                &stream,
                MatvecCall {
                    ty,
                    w: w.range(seg_off[i], ty.weight_bytes(rows, k).expect("size"))
                        .expect("w seg"),
                    x_q8: x_q8.range(0, x_bytes).expect("x row"),
                    y: ys[i].range(0, rows * 4).expect("y seg"),
                    k,
                    rows,
                    ncols: 1,
                },
            )
            .expect("segment matvec");
        }
    };
    let fused_launch = |y: &DeviceBuffer| {
        ctx.matvec_q(
            &stream,
            MatvecCall {
                ty,
                w: w.range(0, ty.weight_bytes(fused_rows, k).expect("size"))
                    .expect("w fused"),
                x_q8: x_q8.range(0, x_bytes).expect("x row"),
                y: y.range(0, fused_rows * 4).expect("y fused"),
                k,
                rows: fused_rows,
                ncols: 1,
            },
        )
        .expect("fused matvec");
    };

    // Bitwise gate: the fused launch's rows are the four segments' outputs, unchanged.
    let mut seg_ys = Vec::new();
    for &rows in &segments {
        seg_ys.push(testkit::upload_filled(&ctx, 0xAA, rows * 4));
    }
    four_launches(&seg_ys);
    let y_fused = testkit::upload_filled(&ctx, 0xAA, fused_rows * 4);
    fused_launch(&y_fused);
    stream.synchronize().expect("drain");
    let fused_out = testkit::read_bytes(&ctx, &y_fused, fused_rows * 4);
    for (i, &rows) in segments.iter().enumerate() {
        let single = testkit::read_bytes(&ctx, &seg_ys[i], rows * 4);
        let start = (seg_off[i] / bytes_per_row) * 4;
        let seg = &fused_out[start..start + rows * 4];
        assert_eq!(
            seg,
            single.as_slice(),
            "segment {i} (rows {rows}): fused rows differ from the single launch"
        );
        println!("segment {i} rows {rows}: bitwise OK");
    }

    // Exclusive timing, interleaved A/B, 30 samples per arm (the mma bench's host-timer
    // pattern: Instant around the launch train, sync inside the sample).
    let mut a_ms = Vec::new();
    let mut b_ms = Vec::new();
    for rep in 0..60 {
        let arm_a = rep % 2 == 0;
        let t = std::time::Instant::now();
        if arm_a {
            four_launches(&seg_ys);
        } else {
            fused_launch(&y_fused);
        }
        stream.synchronize().expect("drain");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        (if arm_a { &mut a_ms } else { &mut b_ms }).push(ms);
    }
    a_ms.sort_by(|p, q| p.total_cmp(q));
    b_ms.sort_by(|p, q| p.total_cmp(q));
    let (a_med, b_med) = (a_ms[15], b_ms[15]);
    println!(
        "gdn sibling fusion A/B: 4-launch median {a_med:.4} ms, fused median {b_med:.4} ms, ratio {:.3}; x30 layers: {:.3} -> {:.3} ms/token; spread a {:.4}-{:.4} b {:.4}-{:.4}",
        b_med / a_med,
        a_med * 30.0,
        b_med * 30.0,
        a_ms[0],
        a_ms[a_ms.len() - 1],
        b_ms[0],
        b_ms[b_ms.len() - 1]
    );
}
