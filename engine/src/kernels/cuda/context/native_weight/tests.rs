//! Host and device tests of the native-layout upload.
//!
//! Host: the shape gates of `checked_native` and the whole-row chunk arithmetic of
//! `RowChunkPlan` (no chunk splits a row, the last chunk may be short, a row wider than the
//! limit is its own chunk). Device (3090): the upload round trip of every type and a
//! 16-expert stack through a small chunk limit, the free of a failed read (the live-byte
//! count returns), and a Native weight through `matvec_q` against the integer reference.
//! Device tests assume `--test-threads=1`.

use super::*;
use crate::kernels::cuda::resident::matvec_q::{MatvecCall, tests::reference};
use crate::kernels::cuda::resident::quantize_q8_1::{QuantizeCall, q8_1_bytes};
use crate::kernels::cuda::{CHUNK_LIMIT, testkit};

/// The error text of a result, so the helpers need no `T: Debug` on `Ok`.
fn error_text<T>(result: Result<T, EngError>) -> String {
    match result {
        Ok(_) => panic!("expected an error, got Ok"),
        Err(error) => format!("{error:?}"),
    }
}

/// A Q4_K tensor of `n_rows` rows of `k` values with its exact source length.
fn q4k(n_rows: usize, k: usize) -> QuantTensor<'static> {
    QuantTensor {
        name: "w",
        ty: GgufType::Q4K,
        n_rows,
        k,
        source_len: n_rows * (k / 256) * 144,
    }
}

#[test]
fn checked_native_takes_the_four_matvec_types_and_refuses_the_rest() {
    for (ty, quant) in [
        (GgufType::Q4K, QuantType::Q4K),
        (GgufType::Q50, QuantType::Q50),
        (GgufType::Q6K, QuantType::Q6K),
        (GgufType::Q80, QuantType::Q80),
    ] {
        let tensor = QuantTensor {
            name: "w",
            ty,
            n_rows: 12,
            k: 512,
            source_len: 12 * (512 / quant.block_size()) * quant.block_bytes(),
        };
        assert!(
            matches!(checked_native(tensor), Ok(found) if found == quant),
            "{ty:?}"
        );
    }
    for ty in [GgufType::F32, GgufType::Q5K, GgufType::Q40] {
        let text = error_text(checked_native(QuantTensor {
            name: "embd",
            ty,
            n_rows: 8,
            k: 256,
            source_len: 8 * 144,
        }));
        assert!(
            text.starts_with("Checkpoint")
                && text.contains("tensor embd has type")
                && text.contains("which the CUDA native weights do not take"),
            "{ty:?}: {text}"
        );
    }
}

#[test]
fn checked_native_allows_any_row_count_but_gates_k_and_the_source_length() {
    // 12 rows: not a whole 8-row group, legal natively.
    assert!(matches!(checked_native(q4k(12, 512)), Ok(QuantType::Q4K)));
    for (tensor, needle) in [
        (q4k(0, 512), "the row count is zero"),
        (q4k(8, 100), "k is off the type's block grid"),
        (q4k(8, 0), "k is off the type's block grid"),
        (
            QuantTensor {
                k: 256 * 129,
                ..q4k(8, 256)
            },
            "k is above the matvec limit",
        ),
        (
            QuantTensor {
                source_len: 8 * 145,
                ..q4k(8, 256)
            },
            "bytes of blocks for 8 rows of 256 values",
        ),
    ] {
        let text = error_text(checked_native(tensor));
        assert!(text.contains(needle), "{needle:?}: {text}");
    }
}

#[test]
fn row_chunks_hold_whole_rows_within_the_limit_and_at_least_one() {
    // A Q4_K row of 512 values is 288 bytes; 64 rows of it, limits around one, three and all
    // rows per chunk.
    let row = 288;
    for (limit, per_chunk) in [(0, 1), (1, 1), (row - 1, 1), (row, 1), (3 * row + 5, 3)] {
        let plan = RowChunkPlan::new(64, row, limit);
        assert_eq!(plan.per_chunk, per_chunk, "limit {limit}");
        let mut first = 0;
        while first < plan.rows {
            let count = plan.count_of(first);
            assert!(count >= 1 && count <= per_chunk, "limit {limit}");
            first += count;
        }
        assert_eq!(first, 64, "limit {limit}");
    }
    // The last chunk is short: 70 rows in chunks of 16 end in a chunk of 6.
    let plan = RowChunkPlan::new(70, row, 16 * row);
    assert_eq!(plan.per_chunk, 16);
    let counts: Vec<usize> = (0..70)
        .step_by(16)
        .map(|first| plan.count_of(first))
        .collect();
    assert_eq!(counts, [16, 16, 16, 16, 6]);
    // A row wider than the limit is its own chunk.
    let wide = RowChunkPlan::new(3, 300 * 1024, 1024);
    assert_eq!(wide.per_chunk, 1);
    // The production limit splits the served lm_head rows: a Q6_K 2048-value row is 1680 bytes.
    let lm_head = RowChunkPlan::new(248_320, 1680, CHUNK_LIMIT);
    assert_eq!(lm_head.per_chunk, CHUNK_LIMIT / 1680);
}

#[test]
fn upload_native_round_trips_the_file_bytes_of_every_type_and_a_stack() {
    const NAME: &str = "upload_native_round_trips_the_file_bytes_of_every_type_and_a_stack";
    if !testkit::device_present(NAME) {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let mut rng = testkit::Rng::new(0x5258_3232);
    for (ty, quant, k) in [
        (GgufType::Q4K, QuantType::Q4K, 512),
        (GgufType::Q6K, QuantType::Q6K, 512),
        (GgufType::Q80, QuantType::Q80, 512),
        (GgufType::Q50, QuantType::Q50, 512),
    ] {
        // 10 rows: not a whole 8-row group, legal natively; a small limit forces chunks of
        // 3, 3, 3 rows and a SHORT LAST CHUNK of 1 (the device round trip must exercise a
        // non-dividing row count too, like the 70-rows-in-16s plan case).
        let n_rows = 10;
        let row_bytes = (k / quant.block_size()) * quant.block_bytes();
        let source: Vec<u8> = (0..n_rows * row_bytes)
            .map(|_| (rng.next_u64() >> 24) as u8)
            .collect();
        let tensor = QuantTensor {
            name: "w",
            ty,
            n_rows,
            k,
            source_len: source.len(),
        };
        let weight = ctx
            .upload_native(tensor, 3 * row_bytes, &mut |at, buf| {
                buf.copy_from_slice(&source[at..at + buf.len()]);
                Ok(())
            })
            .expect("upload");
        assert_eq!(weight.k(), k);
        assert_eq!(weight.n_rows(), n_rows);
        assert_eq!(weight.quant(), quant);
        assert_eq!(weight.experts(), None);
        assert_eq!(weight.buffer().bytes(), source.len());
        let mut back = vec![0u8; source.len()];
        ctx.download(weight.buffer(), &mut back).expect("download");
        assert_eq!(
            back, source,
            "{ty:?}: the bytes must land as the file holds them"
        );
    }
    // A 16-expert Q4_K stack of 8 rows each: the depth rides `with_experts`.
    let (k, experts, rows) = (512, 16, 8);
    let row_bytes = k / 256 * 144;
    let source: Vec<u8> = (0..experts * rows * row_bytes)
        .map(|_| (rng.next_u64() >> 40) as u8)
        .collect();
    let tensor = QuantTensor {
        name: "stack",
        ty: GgufType::Q4K,
        n_rows: experts * rows,
        k,
        source_len: source.len(),
    };
    let weight = ctx
        .upload_native(tensor, 2 * row_bytes, &mut |at, buf| {
            buf.copy_from_slice(&source[at..at + buf.len()]);
            Ok(())
        })
        .expect("upload the stack")
        .with_experts(Some(experts));
    assert_eq!(weight.experts(), Some(experts));
    assert_eq!(weight.n_rows(), experts * rows);
    let mut back = vec![0u8; source.len()];
    ctx.download(weight.buffer(), &mut back).expect("download");
    assert_eq!(back, source, "the stack bytes must land in file order");
}

#[test]
fn a_failed_read_of_a_native_upload_frees_the_allocation() {
    const NAME: &str = "a_failed_read_of_a_native_upload_frees_the_allocation";
    if !testkit::device_present(NAME) {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let (k, n_rows) = (512, 16);
    let row_bytes = k / 256 * 144;
    let tensor = QuantTensor {
        name: "w",
        ty: GgufType::Q4K,
        n_rows,
        k,
        source_len: n_rows * row_bytes,
    };
    let before = ctx.live_bytes();
    let mut reads = 0;
    let failed = ctx.upload_native(tensor, row_bytes, &mut |_, _| {
        reads += 1;
        if reads == 2 {
            return Err(EngError::BadRequest(
                "native read fixture failure".to_owned(),
            ));
        }
        Ok(())
    });
    assert!(matches!(
        &failed,
        Err(EngError::BadRequest(message)) if message == "native read fixture failure"
    ));
    assert_eq!(reads, 2, "the failure must come from the second chunk");
    assert_eq!(
        ctx.live_bytes(),
        before,
        "the failed upload must free its allocation"
    );
    // The backend stays usable: a second upload succeeds and frees again.
    let source = vec![7u8; n_rows * row_bytes];
    let weight = ctx
        .upload_native(tensor, row_bytes, &mut |at, buf| {
            buf.copy_from_slice(&source[at..at + buf.len()]);
            Ok(())
        })
        .expect("upload after the failed read");
    drop(weight);
    assert_eq!(ctx.live_bytes(), before, "the retry freed its allocation");
}

#[test]
fn a_native_weight_through_matvec_q_matches_the_reference() {
    const NAME: &str = "a_native_weight_through_matvec_q_matches_the_reference";
    if !testkit::device_present(NAME) {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut rng = testkit::Rng::new(0x5258_3233);
    // One shape per type: the Qwen3.6 expert gate/up shape for the K-quants and Q8_0, the
    // Mellum down shape (k 896) for Q5_0.
    for (ty, quant, k, rows) in [
        (GgufType::Q4K, QuantType::Q4K, 2048, 512),
        (GgufType::Q6K, QuantType::Q6K, 2048, 512),
        (GgufType::Q80, QuantType::Q80, 2048, 512),
        (GgufType::Q50, QuantType::Q50, 896, 64),
    ] {
        let source = reference::build_weight(quant, rows, k, &mut rng);
        let tensor = QuantTensor {
            name: "w",
            ty,
            n_rows: rows,
            k,
            source_len: source.len(),
        };
        let weight = ctx
            .upload_native(tensor, CHUNK_LIMIT, &mut |at, buf| {
                buf.copy_from_slice(&source[at..at + buf.len()]);
                Ok(())
            })
            .expect("upload");
        let x: Vec<f32> = (0..k).map(|_| (rng.normal() as f32) * 0.05).collect();
        let x_f32 = testkit::upload_f32(&ctx, &x);
        let x_bytes = q8_1_bytes(k, 1).expect("size");
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
        let y = testkit::upload_filled(&ctx, 0x55, rows * 4);
        ctx.matvec_q(
            &stream,
            MatvecCall {
                ty: quant,
                w: weight.buffer().range(0, source.len()).expect("w"),
                x_q8: x_q8.range(0, x_bytes).expect("x_q8"),
                y: y.range(0, rows * 4).expect("y"),
                k,
                rows,
                ncols: 1,
            },
        )
        .expect("matvec");
        stream.synchronize().expect("drain");
        let device = testkit::floats(&testkit::read_bytes(&ctx, &y, rows * 4));
        let x_q8_bytes = testkit::read_bytes(&ctx, &x_q8, x_bytes);
        let (want, abs) = reference::reference_dot(quant, &source, &x_q8_bytes, k);
        let tolerance = 64.0 * f64::powi(2.0, -24) * abs;
        let error = (f64::from(device[0]) - want).abs();
        assert!(
            error <= tolerance,
            "{ty:?}: device {} reference {want}, tolerance {tolerance}",
            device[0]
        );
    }
}
