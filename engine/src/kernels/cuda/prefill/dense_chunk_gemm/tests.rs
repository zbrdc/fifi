//! The tensor-core gemm of a chunk gives the rows the column-invariant matvec gives on the same
//! quantized activations, up to the f32 order of the sums, at the shapes of the Bolt projections.

use super::*;
use crate::kernels::cuda::QuantTensor;
use crate::kernels::cuda::resident::add_rmsnorm_q8::tests::as_f32;
use crate::kernels::cuda::resident::matvec_q::tests::reference;
use crate::kernels::cuda::resident::matvec_q::{MatvecCall, QuantType};
use crate::kernels::cuda::resident::quantize_q8_1::{
    QuantizeCall, q8_1_bytes as decode_q8_1_bytes,
};
use crate::kernels::cuda::testkit;

#[test]
fn the_gemm_matches_the_invariant_matvec_at_the_bolt_shapes() {
    if !testkit::device_present("the_gemm_matches_the_invariant_matvec_at_the_bolt_shapes") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let gemm = DenseChunkGemm::new(&ctx, 10_240, 64).expect("scratch");
    let mut rng = testkit::Rng::new(9);
    for (rows, k, m) in [
        (2048usize, 2560usize, 33usize),
        (512, 2560, 64),
        (2560, 2048, 17),
        (10_240, 2560, 40),
        (2560, 10_240, 33),
    ] {
        let w_bytes = reference::build_weight(QuantType::Q80, rows, k, &mut rng);
        let w_dev = testkit::upload_bytes(&ctx, &w_bytes);
        let weight = {
            let tensor = QuantTensor {
                name: "test",
                ty: crate::weights::GgufType::Q80,
                n_rows: rows,
                k,
                source_len: w_bytes.len(),
            };
            ctx.upload_native(tensor, 1 << 20, &mut |at, part| {
                part.copy_from_slice(&w_bytes[at..at + part.len()]);
                Ok(())
            })
            .expect("native weight")
        };
        let x: Vec<f32> = (0..m * k).map(|_| (rng.normal() as f32) * 0.5).collect();
        let x_dev = testkit::upload_f32(&ctx, &x);
        let q8_bytes = decode_q8_1_bytes(k, m).expect("q8 bytes");
        let q8 = testkit::upload_filled(&ctx, 0, q8_bytes);
        ctx.quantize_q8_1(
            &stream,
            QuantizeCall {
                x: x_dev.range(0, m * k * 4).expect("x"),
                y: q8.range(0, q8_bytes).expect("q8"),
                k,
                ncols: m,
            },
        )
        .expect("quantize");
        let want = testkit::upload_filled(&ctx, 0, m * rows * 4);
        for first in (0..m).step_by(8) {
            let cols = 8.min(m - first);
            let row_bytes = decode_q8_1_bytes(k, 1).expect("row bytes");
            ctx.matvec_q_invariant(
                &stream,
                MatvecCall {
                    ty: QuantType::Q80,
                    w: w_dev.range(0, w_bytes.len()).expect("w"),
                    x_q8: q8.range(first * row_bytes, cols * row_bytes).expect("x q8"),
                    y: want.range(first * rows * 4, cols * rows * 4).expect("want"),
                    k,
                    rows,
                    ncols: cols,
                },
            )
            .expect("matvec");
        }
        let got = testkit::upload_filled(&ctx, 0, m * rows * 4);
        let want = as_f32(&testkit::read_bytes(&ctx, &want, m * rows * 4));
        let scale = want.iter().fold(0f32, |acc, v| acc.max(v.abs()));
        // The packing of the activation: the reference rule, and the vendored one for comparison.
        for packing in [MmqPacking::CpuReference, MmqPacking::Vendor] {
            ctx.mmq_dense_input(
                &stream,
                MmqDenseInputCall {
                    packing,
                    weight: &weight,
                    m,
                    row_stride: k,
                    x: x_dev.range(0, m * k * 4).expect("x"),
                    y: gemm.input.range(0, gemm.input.bytes()).expect("input"),
                },
            )
            .expect("input");
            gemm.project(
                &ctx,
                &stream,
                &weight,
                m,
                got.range(0, m * rows * 4).expect("got"),
            )
            .expect("project");
            stream.synchronize().expect("drain");
            let got = as_f32(&testkit::read_bytes(&ctx, &got, m * rows * 4));
            let worst = want
                .iter()
                .zip(&got)
                .fold(0f32, |acc, (a, b)| acc.max((a - b).abs()));
            eprintln!(
                "rows {rows} k {k} m {m} {packing:?}: max |dy| {worst:.3e} of max |y| {scale:.3e}"
            );
            if packing == MmqPacking::CpuReference {
                assert!(
                    worst <= scale * 1e-3,
                    "rows {rows} k {k} m {m}: {worst} over {scale}"
                );
            }
        }
    }
}

/// The plane gemm over a weight in another owner's buffer gives the bits of the chunk gemm over
/// the same native blocks, for a plane at an offset in packed rows and a strided output, and
/// refuses a plane off the 16-byte grid.
#[test]
fn the_plane_gemm_matches_the_chunk_gemm_bit_for_bit() {
    if !testkit::device_present("the_plane_gemm_matches_the_chunk_gemm_bit_for_bit") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut rng = testkit::Rng::new(11);
    let mut planes = PlaneGemm::default();
    for (ty, gguf, rows, k, m) in [
        (QuantType::Q80, GgufType::Q80, 3072usize, 2304usize, 64usize),
        (QuantType::Q6K, GgufType::Q6K, 1024, 2304, 63),
        (QuantType::Q80, GgufType::Q80, 2304, 3072, 128),
    ] {
        let gemm = DenseChunkGemm::new(&ctx, k, m).expect("scratch");
        let w_bytes = reference::build_weight(ty, rows, k, &mut rng);
        let owned = ctx
            .upload_native(
                QuantTensor {
                    name: "test",
                    ty: gguf,
                    n_rows: rows,
                    k,
                    source_len: w_bytes.len(),
                },
                1 << 20,
                &mut |at, part| {
                    part.copy_from_slice(&w_bytes[at..at + part.len()]);
                    Ok(())
                },
            )
            .expect("native weight");
        let codes = testkit::upload_bytes(&ctx, &w_bytes);
        let x: Vec<f32> = (0..m * k).map(|_| (rng.normal() as f32) * 0.5).collect();
        // Packed rows: a plane of 256 other values, then the projected plane.
        let ld_x = k + 256;
        let packed: Vec<f32> = x
            .chunks_exact(k)
            .flat_map(|row| std::iter::repeat_n(9.0, 256).chain(row.iter().copied()))
            .collect();
        let x_dev = testkit::upload_f32(&ctx, &x);
        let packed_dev = testkit::upload_f32(&ctx, &packed);
        gemm.input(
            &ctx,
            &stream,
            &owned,
            x_dev.range(0, m * k * 4).expect("x"),
            m,
        )
        .expect("input");
        let want = testkit::upload_filled(&ctx, 0, m * rows * 4);
        gemm.project(
            &ctx,
            &stream,
            &owned,
            m,
            want.range(0, m * rows * 4).expect("want"),
        )
        .expect("project");
        let ld_y = rows + 5;
        let got = testkit::upload_filled(&ctx, 0x7f, m * ld_y * 4);
        let x_plane = packed_dev
            .range(256 * 4, ((m - 1) * ld_x + k) * 4)
            .expect("plane");
        let y = got.range(0, ((m - 1) * ld_y + rows) * 4).expect("y");
        let whole = codes.range(0, codes.bytes()).expect("codes");
        let weight = PlaneGemm::weight(whole, gguf, [rows, k], (m, &x_plane, ld_x), ld_y)
            .expect("the gemm takes the weight");
        let off_grid = packed_dev
            .range(4, ((m - 1) * ld_x + k) * 4)
            .expect("off grid");
        assert!(PlaneGemm::weight(whole, gguf, [rows, k], (m, &off_grid, ld_x), ld_y).is_none());
        let call = PlaneProjection {
            weight,
            plane: 1,
            x: x_plane,
            y,
            ld_x,
            ld_y,
        };
        planes
            .project(&ctx, &stream, &[call], m)
            .expect("plane project");
        stream.synchronize().expect("drain");
        let want = testkit::read_bytes(&ctx, &want, m * rows * 4);
        let got = testkit::read_bytes(&ctx, &got, m * ld_y * 4);
        for (j, row) in got.chunks_exact(ld_y * 4).enumerate() {
            assert!(
                row[..rows * 4] == want[j * rows * 4..(j + 1) * rows * 4],
                "{ty:?} rows {rows} k {k} m {m}: row {j} differs"
            );
            assert!(
                row[rows * 4..].iter().all(|&b| b == 0x7f),
                "row {j} padding"
            );
        }
    }
}
