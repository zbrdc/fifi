//! Real dense/shared FFN shape qualification, not caller wiring or model throughput.
//! Larger M repeats five saved rows; shared down inputs are an original-GGUF BF16 graph.

use super::super::*;
use crate::family::flint::{
    cuda_ffn::reference::{self, real::gguf},
    gguf_weights,
};
use crate::kernels::cuda::device_present;
use crate::kernels::cuda::prefill::mmq_nvfp4::{Nvfp4DenseCall, Nvfp4DenseShape};
use crate::weights::{GgufHeader, GgufType, dequantize, repack_nvfp4_into, verify_sha256};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufReader, Seek},
};

const SAVED_ROWS: usize = 5;
const INPUT_SHA: &str = "88bcdf501fe655c749344bf669f507df3b2fe8edfd40b6e1f2c3b8e64cb480d8";
const ROWS: [usize; 4] = [5, 64, 128, 144];

#[derive(Deserialize)]
struct Activation {
    layer: u32,
    rows: usize,
    model_sha256: String,
    trace_matches_untraced: bool,
    boundaries: Vec<Boundary>,
}

#[derive(Deserialize)]
struct Boundary {
    name: String,
    bits: Vec<u32>,
}

impl Activation {
    fn input(&self, name: &str, width: usize) -> Vec<f32> {
        let selected = self
            .boundaries
            .iter()
            .filter(|b| b.name == name)
            .collect::<Vec<_>>();
        assert_eq!(selected.len(), 1, "unique saved boundary {name}");
        assert_eq!(selected[0].bits.len(), SAVED_ROWS * width, "{name} shape");
        assert!(
            selected[0]
                .bits
                .iter()
                .all(|&b| b & 0xffff == 0 && f32::from_bits(b).is_finite())
        );
        selected[0]
            .bits
            .iter()
            .map(|&b| f32::from_bits(b))
            .collect()
    }
}

struct Weight {
    name: String,
    k: usize,
    n: usize,
    raw: Vec<u8>,
    packed: Vec<u8>,
    scales: Vec<u8>,
    factor: f32,
    original: Vec<f32>,
}

impl Weight {
    fn load(file: &File, header: &GgufHeader, name: &str, k: usize, n: usize) -> Self {
        let source = gguf_weights::matrix_source(header, name, [n, k]).unwrap();
        let raw = gguf::raw(file, source.data.clone()).unwrap();
        let scalar = gguf::raw(file, source.scale.clone()).unwrap();
        let factor = source.bind(&raw, &scalar).unwrap().tensor_scale;
        let original = dequantize(GgufType::Nvfp4, &raw).unwrap();
        assert_eq!(original.len(), k * n);
        let mut packed = vec![0; k * n / 2];
        let mut scales = vec![0; k * n / 16];
        repack_nvfp4_into(&raw, &mut packed, &mut scales).unwrap();
        Self {
            name: name.into(),
            k,
            n,
            raw,
            packed,
            scales,
            factor,
            original,
        }
    }

    fn expected(&self, input: &[f32]) -> Vec<f64> {
        assert!(input.len().is_multiple_of(self.k));
        input
            .chunks_exact(self.k)
            .flat_map(|x| {
                self.original.chunks_exact(self.k).map(move |w| {
                    x.iter()
                        .zip(w)
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum::<f64>()
                        * f64::from(self.factor)
                })
            })
            .collect()
    }
}

fn shared_hidden(input: &[f32], gate: &Weight, up: &Weight) -> Vec<f32> {
    gate.expected(input)
        .into_iter()
        .zip(up.expected(input))
        .map(|(gate, up)| {
            let gate = reference::bf16(gate as f32).unwrap();
            let up = reference::bf16(up as f32).unwrap();
            let probability =
                reference::bf16((1.0 / (1.0 + (-f64::from(gate)).exp())) as f32).unwrap();
            let silu = reference::bf16(gate * probability).unwrap();
            reference::bf16(silu * up).unwrap()
        })
        .collect()
}

fn repeat_rows<T: Copy>(input: &[T], width: usize, rows: usize) -> Vec<T> {
    assert_eq!(input.len(), SAVED_ROWS * width);
    (0..rows)
        .flat_map(|row| input[(row % SAVED_ROWS) * width..][..width].iter().copied())
        .collect()
}

fn read(stream: &DeviceStream, buffer: &DeviceBuffer) -> Result<Vec<f32>, EngError> {
    let mut bytes = vec![0; buffer.bytes()];
    buffer.read_at(stream, 0, &mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_ne_bytes(*b))
        .collect())
}

struct Outcome {
    raw: Vec<f32>,
    published: Vec<f32>,
    frame: Vec<f32>,
}

fn phase(weight: &Weight, input: &[f32], rows: usize) -> Result<Outcome, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let shape = Nvfp4DenseShape::new(weight.k, weight.n, rows)?;
    let raw_weight = ctx.upload(&weight.raw)?;
    let packed = ctx.upload(&weight.packed)?;
    let scales = ctx.upload(&weight.scales)?;
    let plan = ctx.mmq_nvfp4_dense_plan(
        raw_weight.range(0, raw_weight.bytes())?,
        [weight.k, weight.n, rows],
        weight.factor,
    )?;
    let original = ctx.nvfp4_weight_range(
        packed.range(0, packed.bytes())?,
        scales.range(0, scales.bytes())?,
        weight.factor,
        [weight.n, weight.k],
    )?;
    let baseline = ctx.nvfp4_grouped_plan(&vec![original; rows])?;
    let x = ctx.upload(
        &input
            .iter()
            .flat_map(|x| x.to_ne_bytes())
            .collect::<Vec<_>>(),
    )?;
    let q8 = ctx.upload(&vec![0xff; shape.needs.y])?;
    let fixup = ctx.upload(&vec![0xff; ctx.mmq_fixup_bytes()?])?;
    let input_shape = RowShape {
        rows,
        columns: weight.k,
    };
    let output_shape = RowShape {
        rows,
        columns: weight.n,
    };
    let mut candidate = Frame::new(&ctx, &stream)?;
    let input_index = candidate.input(x.range(0, x.bytes())?, input_shape)?;
    let output = candidate.allocate(output_shape)?;
    ctx.mmq_nvfp4_dense_input(
        &stream,
        &plan,
        candidate.range(input_index)?,
        q8.range(0, q8.bytes())?,
    )?;
    ctx.mmq_nvfp4_dense(
        &stream,
        Nvfp4DenseCall {
            plan: &plan,
            q8_1: q8.range(0, q8.bytes())?,
            dst: candidate.range(output)?,
            fixup: fixup.range(0, fixup.bytes())?,
        },
    )?;
    let published = candidate.publish(output, output_shape)?;
    let raw = read(&stream, &candidate.buffers[output].0)?;
    let published = read(&stream, &candidate.finish(published)?)?;
    let mut current = Frame::new(&ctx, &stream)?;
    let input_index = current.input(x.range(0, x.bytes())?, input_shape)?;
    let output = current.project(&baseline, input_index, rows, [weight.n, weight.k])?;
    let frame = read(&stream, &current.finish(output)?)?;
    Ok(Outcome {
        raw,
        published,
        frame,
    })
}

fn check_rows(actual: &[f32], expected: &[f64], width: usize, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} size");
    assert!(width > 0 && !actual.is_empty() && actual.len().is_multiple_of(width));
    for (row, (actual, expected)) in actual
        .chunks_exact(width)
        .zip(expected.chunks_exact(width))
        .enumerate()
    {
        let (mut error, mut reference_norm, mut actual_norm, mut dot) = (0.0, 0.0, 0.0, 0.0);
        for (&a, &b) in actual.iter().zip(expected) {
            assert!(a.is_finite() && b.is_finite(), "{label} nonfinite row{row}");
            let a = f64::from(a);
            error += (a - b).powi(2);
            reference_norm += b * b;
            actual_norm += a * a;
            dot += a * b;
        }
        let relative = if reference_norm == 0.0 {
            if error == 0.0 { 0.0 } else { f64::INFINITY }
        } else {
            (error / reference_norm).sqrt()
        };
        let cosine = if reference_norm == 0.0 && actual_norm == 0.0 {
            1.0
        } else if reference_norm == 0.0 || actual_norm == 0.0 {
            0.0
        } else {
            dot / (reference_norm * actual_norm).sqrt()
        };
        eprintln!("S3_ACCURACY {label} row={row} relative_l2={relative:.12e} cosine={cosine:.12e}");
        assert!(
            relative <= 0.01 && cosine >= 0.9999,
            "{label} row{row}: rel={relative} cosine={cosine}"
        );
    }
}

fn qualify(weight: &Weight, input: &[f32], provenance: &str) {
    let expected = weight.expected(input);
    for rows in ROWS {
        let x = repeat_rows(input, weight.k, rows);
        let expected = repeat_rows(&expected, weight.n, rows);
        let result = phase(weight, &x, rows).expect("S3 device phase completed");
        // Owners are dropped before assertions; a failed case cannot hide queued work.
        assert!(
            result
                .published
                .iter()
                .chain(&result.frame)
                .all(|v| v.to_bits() & 0xffff == 0)
        );
        for (name, actual) in [
            ("mmq-f32", &result.raw),
            ("mmq-bf16", &result.published),
            ("frame-bf16", &result.frame),
        ] {
            check_rows(
                actual,
                &expected,
                weight.n,
                &format!("{} M={rows} {name}-vs-original-f64", weight.name),
            );
        }
        let current = result
            .frame
            .iter()
            .map(|&x| f64::from(x))
            .collect::<Vec<_>>();
        check_rows(
            &result.published,
            &current,
            weight.n,
            &format!("{} M={rows} mmq-vs-frame", weight.name),
        );
        eprintln!(
            "S3_CELL_OK tensor={} K={} N={} M={rows} input={provenance} repeated_saved_rows=5 NO_TIMING_NO_MODEL_QUALIFICATION",
            weight.name, weight.k, weight.n
        );
    }
}

#[test]
fn row_replay_keeps_complete_rows_and_ragged_tail() {
    assert_eq!(
        repeat_rows(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 2, 7),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3]
    );
}

#[test]
fn f64_gate_rejects_amplitude_error_and_later_bad_rows() {
    assert!(
        std::panic::catch_unwind(|| check_rows(
            &[1.0, 2.0, 2.0, 4.0],
            &[1.0, 2.0, 1.0, 2.0],
            2,
            "negative-control"
        ))
        .is_err()
    );
    check_rows(&[0.0, 0.0], &[0.0, 0.0], 2, "zero");
}

mod prefill_bench;

fn with_cases(mut run: impl FnMut(&Weight, &[f32], &str)) {
    let mut file = File::open(std::env::var("FIFI_FLINT_GGUF").expect("pinned GGUF")).unwrap();
    eprintln!("S3_SETUP model_sha_begin");
    verify_sha256(
        &mut BufReader::new(&file),
        22_863_084_160,
        gguf::MODEL_SHA256,
    )
    .unwrap();
    eprintln!("S3_SETUP model_sha_ok");
    file.rewind().unwrap();
    let header = GgufHeader::parse(
        &mut BufReader::new(&file),
        file.metadata().unwrap().len(),
        64 << 20,
    )
    .unwrap();
    let bytes =
        std::fs::read(std::env::var("FIFI_FLINT_MMQ_INPUT").expect("native L0 capture")).unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), INPUT_SHA);
    let activation: Activation = serde_json::from_slice(&bytes).unwrap();
    assert_eq!((activation.layer, activation.rows), (0, SAVED_ROWS));
    assert_eq!(activation.model_sha256, gguf::MODEL_SHA256);
    assert!(activation.trace_matches_untraced);
    let post_norm = activation.input("post_norm", 2048);
    let product = activation.input("product", 6144);
    for (layer, intermediate, suffix) in [(0, 6144, ""), (2, 1024, "_shexp")] {
        let load = |projection: &str, k, n| {
            Weight::load(
                &file,
                &header,
                &format!("blk.{layer}.ffn_{projection}{suffix}"),
                k,
                n,
            )
        };
        let gate = load("gate", 2048, intermediate);
        let up = load("up", 2048, intermediate);
        let down = load("down", intermediate, 2048);
        run(&gate, &post_norm, "native-L0-post_norm-replay");
        run(&up, &post_norm, "native-L0-post_norm-replay");
        if layer == 0 {
            run(&down, &product, "native-L0-product-replay");
        } else {
            let hidden = shared_hidden(&post_norm, &gate, &up);
            run(
                &down,
                &hidden,
                "original-GGUF-BF16-shared-graph-from-L0-post_norm-NOT-L2-capture",
            );
        }
    }
}

#[test]
#[ignore = "requires admitted CUDA device, pinned GGUF and native L0 M5 activation"]
fn nvfp4_dense_shared_all_ffn_shapes_real_correctness() {
    assert!(device_present(
        "nvfp4_dense_shared_all_ffn_shapes_real_correctness"
    ));
    with_cases(qualify);
    eprintln!(
        "S3_ALL_SHAPES_OK tensors=6 cells=24 NO_TIMING_NO_MODEL_QUALIFICATION NO_SELECTOR_QUALIFICATION"
    );
}
