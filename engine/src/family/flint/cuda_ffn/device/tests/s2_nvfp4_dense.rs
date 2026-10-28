//! One original-GGUF L0 gate and its saved native post-norm input; not model qualification.
//! Correctness only. Timing requires a separately reviewed harness and coordinator window.

use super::super::*;
use crate::family::flint::{cuda_ffn::reference::real::gguf, gguf_weights};
use crate::kernels::cuda::device_present;
use crate::kernels::cuda::prefill::mmq_nvfp4::{Nvfp4DenseCall, Nvfp4DensePlan, Nvfp4DenseShape};
use crate::weights::{GgufHeader, GgufType, dequantize, repack_nvfp4_into, verify_sha256};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufReader, Seek},
};

const M: usize = 5;
const K: usize = 2048;
const N: usize = 6144;
const INPUT_SHA: &str = "88bcdf501fe655c749344bf669f507df3b2fe8edfd40b6e1f2c3b8e64cb480d8";

struct Fixture {
    raw: Vec<u8>,
    packed: Vec<u8>,
    scales: Vec<u8>,
    factor: f32,
    input: Vec<f32>,
    expected: Vec<f64>,
}

#[derive(Deserialize)]
struct Activation {
    boundaries: Vec<Boundary>,
}
#[derive(Deserialize)]
struct Boundary {
    name: String,
    bits: Vec<u32>,
}

fn fixture() -> Fixture {
    let path = std::env::var("FIFI_FLINT_GGUF").expect("pinned Flint GGUF path");
    let mut file = File::open(path).unwrap();
    eprintln!("DENSE_SETUP model_sha_begin");
    verify_sha256(
        &mut BufReader::new(&file),
        22_863_084_160,
        gguf::MODEL_SHA256,
    )
    .unwrap();
    eprintln!("DENSE_SETUP model_sha_ok");
    file.rewind().unwrap();
    let header = GgufHeader::parse(
        &mut BufReader::new(&file),
        file.metadata().unwrap().len(),
        64 << 20,
    )
    .unwrap();
    let source = gguf_weights::matrix_source(&header, "blk.0.ffn_gate", [N, K]).unwrap();
    let raw = gguf::raw(&file, source.data.clone()).unwrap();
    let scalar = gguf::raw(&file, source.scale.clone()).unwrap();
    let factor = source.bind(&raw, &scalar).unwrap().tensor_scale;
    let input_bytes = std::fs::read(
        std::env::var("FIFI_FLINT_MMQ_INPUT").expect("saved native post_norm fixture"),
    )
    .unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(&input_bytes)), INPUT_SHA);
    let mut activation: Activation = serde_json::from_slice(&input_bytes).unwrap();
    assert_eq!(
        activation
            .boundaries
            .iter()
            .filter(|b| b.name == "post_norm")
            .count(),
        1
    );
    let boundary = activation
        .boundaries
        .iter_mut()
        .find(|b| b.name == "post_norm")
        .unwrap();
    assert_eq!(boundary.bits.len(), M * K);
    assert!(
        boundary
            .bits
            .iter()
            .all(|&b| b & 0xffff == 0 && f32::from_bits(b).is_finite())
    );
    let input: Vec<_> = boundary.bits.iter().map(|&b| f32::from_bits(b)).collect();
    let original = dequantize(GgufType::Nvfp4, &raw).unwrap();
    assert_eq!(original.len(), N * K);
    let mut expected = Vec::with_capacity(M * N);
    for x in input.as_chunks::<K>().0 {
        for weight in original.as_chunks::<K>().0 {
            let dot: f64 = x
                .iter()
                .zip(weight)
                .map(|(&x, &w)| f64::from(x) * f64::from(w))
                .sum();
            expected.push(dot * f64::from(factor));
        }
    }
    let mut packed = vec![0; N * K / 2];
    let mut scales = vec![0; N * K / 16];
    repack_nvfp4_into(&raw, &mut packed, &mut scales).unwrap();
    Fixture {
        raw,
        packed,
        scales,
        factor,
        input,
        expected,
    }
}

struct Scratch {
    rows: usize,
    input: DeviceBuffer,
    q8: DeviceBuffer,
    fixup: DeviceBuffer,
}

impl Scratch {
    fn output_shape(&self) -> RowShape {
        RowShape {
            rows: self.rows,
            columns: N,
        }
    }
}

fn candidate_frame<'c, 's>(
    ctx: &'c CudaContext,
    stream: &'s DeviceStream,
    scratch: &Scratch,
    plan: &Nvfp4DensePlan<'_>,
) -> Result<(Frame<'c, 's>, usize), EngError> {
    let mut frame = Frame::new(ctx, stream)?;
    let x = frame.input(
        scratch.input.range(0, scratch.input.bytes())?,
        RowShape {
            rows: scratch.rows,
            columns: K,
        },
    )?;
    let out = frame.allocate(scratch.output_shape())?;
    ctx.mmq_nvfp4_dense_input(
        stream,
        plan,
        frame.range(x)?,
        scratch.q8.range(0, scratch.q8.bytes())?,
    )?;
    ctx.mmq_nvfp4_dense(
        stream,
        Nvfp4DenseCall {
            plan,
            q8_1: scratch.q8.range(0, scratch.q8.bytes())?,
            dst: frame.range(out)?,
            fixup: scratch.fixup.range(0, scratch.fixup.bytes())?,
        },
    )?;
    Ok((frame, out))
}

fn candidate(
    ctx: &CudaContext,
    stream: &DeviceStream,
    scratch: &Scratch,
    plan: &Nvfp4DensePlan<'_>,
) -> Result<DeviceBuffer, EngError> {
    let (mut frame, raw) = candidate_frame(ctx, stream, scratch, plan)?;
    let output = frame.publish(raw, scratch.output_shape())?;
    frame.finish(output)
}

fn baseline(
    ctx: &CudaContext,
    stream: &DeviceStream,
    scratch: &Scratch,
    plan: &Nvfp4GroupedPlan<'_>,
) -> Result<DeviceBuffer, EngError> {
    let mut frame = Frame::new(ctx, stream)?;
    let x = frame.input(
        scratch.input.range(0, scratch.input.bytes())?,
        RowShape {
            rows: scratch.rows,
            columns: K,
        },
    )?;
    let output = frame.project(plan, x, scratch.rows, [N, K])?;
    frame.finish(output)
}

fn read(stream: &DeviceStream, buffer: &DeviceBuffer) -> Result<Vec<f32>, EngError> {
    let mut bytes = vec![0; buffer.bytes()];
    buffer.read_at(stream, 0, &mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_ne_bytes(*word))
        .collect())
}

struct Outcome {
    raw: Vec<f32>,
    published: Vec<f32>,
    baseline: Vec<f32>,
    repeat: Vec<f32>,
    refusals: Vec<Option<String>>,
    untouched: bool,
}

fn phase(fixture: &Fixture) -> Result<Outcome, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let shape = Nvfp4DenseShape::new(K, N, M)?;
    let weight = ctx.upload(&fixture.raw)?;
    let canonical = ctx.upload(&fixture.packed)?;
    let scales = ctx.upload(&fixture.scales)?;
    let plan =
        ctx.mmq_nvfp4_dense_plan(weight.range(0, weight.bytes())?, [K, N, M], fixture.factor)?;
    let view = ctx.nvfp4_weight_range(
        canonical.range(0, canonical.bytes())?,
        scales.range(0, scales.bytes())?,
        fixture.factor,
        [N, K],
    )?;
    let old = ctx.nvfp4_grouped_plan(&[view; M])?;
    let input: Vec<_> = fixture.input.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let floor = ctx.mmq_fixup_bytes()?;
    let short_floor = floor.checked_sub(1).ok_or_else(|| bad("zero fixup size"))?;
    let scratch = Scratch {
        rows: M,
        input: ctx.upload(&input)?,
        q8: ctx.upload(&vec![0xff; shape.needs.y])?,
        fixup: ctx.upload(&vec![0xff; floor])?,
    };
    let (frame, output) = candidate_frame(&ctx, &stream, &scratch, &plan)?;
    let raw = read(&stream, &frame.finish(output)?)?;
    let published = read(&stream, &candidate(&ctx, &stream, &scratch, &plan)?)?;
    let baseline = read(&stream, &baseline(&ctx, &stream, &scratch, &old)?)?;
    let dst = ctx.upload(&vec![0xa7; shape.needs.dst])?;
    let snapshot = |buffer: &DeviceBuffer| -> Result<Vec<u8>, EngError> {
        let mut bytes = vec![0; buffer.bytes()];
        buffer.read_at(&stream, 0, &mut bytes)?;
        Ok(bytes)
    };
    let before = [
        snapshot(&weight)?,
        snapshot(&scratch.q8)?,
        snapshot(&scratch.fixup)?,
        snapshot(&dst)?,
    ];
    let mut refusals = Vec::new();
    refusals.push(
        ctx.mmq_nvfp4_dense_plan(weight.range(0, weight.bytes())?, [K, N, M], f32::NAN)
            .err()
            .map(|e| format!("{e:?}")),
    );
    refusals.push(
        ctx.mmq_nvfp4_dense_input(
            &stream,
            &plan,
            scratch.input.range(0, scratch.input.bytes())?,
            weight.range(0, shape.needs.y)?,
        )
        .err()
        .map(|e| format!("{e:?}")),
    );
    refusals.push(
        ctx.mmq_nvfp4_dense(
            &stream,
            Nvfp4DenseCall {
                plan: &plan,
                q8_1: scratch.q8.range(0, shape.needs.y - 1)?,
                dst: dst.range(0, dst.bytes())?,
                fixup: scratch.fixup.range(0, scratch.fixup.bytes())?,
            },
        )
        .err()
        .map(|e| format!("{e:?}")),
    );
    refusals.push(
        ctx.mmq_nvfp4_dense(
            &stream,
            Nvfp4DenseCall {
                plan: &plan,
                q8_1: scratch.q8.range(0, scratch.q8.bytes())?,
                dst: dst.range(0, dst.bytes() - 1)?,
                fixup: scratch.fixup.range(0, scratch.fixup.bytes())?,
            },
        )
        .err()
        .map(|e| format!("{e:?}")),
    );
    refusals.push(
        ctx.mmq_nvfp4_dense(
            &stream,
            Nvfp4DenseCall {
                plan: &plan,
                q8_1: scratch.q8.range(0, scratch.q8.bytes())?,
                dst: dst.range(0, dst.bytes())?,
                fixup: scratch.fixup.range(0, short_floor)?,
            },
        )
        .err()
        .map(|e| format!("{e:?}")),
    );
    stream.synchronize()?;
    let after = [
        snapshot(&weight)?,
        snapshot(&scratch.q8)?,
        snapshot(&scratch.fixup)?,
        snapshot(&dst)?,
    ];
    let repeat = read(&stream, &candidate(&ctx, &stream, &scratch, &plan)?)?;
    Ok(Outcome {
        raw,
        published,
        baseline,
        repeat,
        refusals,
        untouched: before == after,
    })
}

fn assert_rows(actual: &[f32], expected: &[f64], rows: usize, label: &str) {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(actual.len(), rows * N);
    for (row, (actual, expected)) in actual
        .as_chunks::<N>()
        .0
        .iter()
        .zip(expected.as_chunks::<N>().0)
        .enumerate()
    {
        let (mut error, mut reference, mut norm, mut dot) = (0.0, 0.0, 0.0, 0.0);
        for (&a, &b) in actual.iter().zip(expected) {
            assert!(a.is_finite() && b.is_finite(), "{label} nonfinite row{row}");
            let a = f64::from(a);
            error += (a - b) * (a - b);
            reference += b * b;
            norm += a * a;
            dot += a * b;
        }
        if reference == 0.0 {
            assert_eq!(error, 0.0, "{label} zero row{row}");
            continue;
        }
        let relative = (error / reference).sqrt();
        let cosine = if norm == 0.0 {
            0.0
        } else {
            dot / (norm * reference).sqrt()
        };
        eprintln!(
            "DENSE_ACCURACY {label} m={rows} row={row} relative_l2={relative:.12e} cosine={cosine:.12e}"
        );
        assert!(
            relative <= 0.01 && cosine >= 0.9999,
            "{label} row{row}: rel={relative}, cosine={cosine}"
        );
    }
}

fn assert_correctness(fixture: &Fixture) {
    eprintln!("DENSE_SETUP fixture_ready");
    // All assertions below occur after phase has dropped context, buffers, plans and stream.
    let result = phase(fixture).expect("dense phase completed");
    eprintln!("DENSE_DEVICE returned");
    assert!(
        result.untouched,
        "refusals wrote a live input/output/scratch range"
    );
    for (error, tag) in result
        .refusals
        .iter()
        .zip(["finite", "overlap", "q8_1", "dst", "fixup"])
    {
        assert!(
            error.as_ref().is_some_and(|e| e.contains(tag)),
            "missing refusal {tag}: {error:?}"
        );
    }
    assert_eq!(
        result
            .published
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        result
            .repeat
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert_rows(
        &result.raw,
        &fixture.expected,
        M,
        "mmq-f32-vs-original-gguf-f64",
    );
    assert_rows(
        &result.published,
        &fixture.expected,
        M,
        "mmq-bf16-vs-original-gguf-f64",
    );
    assert_rows(
        &result.baseline,
        &fixture.expected,
        M,
        "current-frame-bf16-vs-original-gguf-f64",
    );
    let old: Vec<_> = result.baseline.iter().map(|&v| f64::from(v)).collect();
    assert_rows(&result.published, &old, M, "mmq-vs-current-frame");
    eprintln!("DENSE_CORRECTNESS_OK M=5 K=2048 N=6144 NO_TIMING_CLAIM NO_MODEL_QUALIFICATION");
}

#[test]
#[ignore = "requires admitted CUDA device, pinned GGUF and saved native M5 activation"]
fn nvfp4_dense_l0_gate_real_correctness() {
    assert!(device_present("nvfp4_dense_l0_gate_real_correctness"));
    let fixture = fixture();
    assert_correctness(&fixture);
}

pub(super) struct BenchCell {
    pub(super) rows: usize,
    // Each pair is [current Frame.project, MMQ], in wall-clock nanoseconds.
    pub(super) samples_ns: Vec<[u128; 2]>,
    pub(super) output: [Vec<f32>; 2],
}

pub(super) fn bench_cell(
    stream: &DeviceStream,
    mut run: impl FnMut(bool) -> Result<DeviceBuffer, EngError>,
    rows: usize,
    measured: bool,
) -> Result<BenchCell, EngError> {
    let warm = if measured { 8 } else { 1 };
    let pairs = if measured { 30 } else { 1 };
    let mut cell = BenchCell {
        rows,
        samples_ns: Vec::new(),
        output: [Vec::new(), Vec::new()],
    };
    for pair in 0..warm + pairs {
        let mut ns = [0; 2];
        for variant in [pair % 2, 1 - pair % 2] {
            stream.synchronize()?;
            let start = (measured && pair >= warm).then(std::time::Instant::now);
            let output = run(variant == 1)?;
            stream.synchronize()?;
            if let Some(start) = start {
                ns[variant] = start.elapsed().as_nanos();
            }
            // Readback and destruction of the returned output are outside the sample.
            if pair + 1 == warm + pairs {
                cell.output[variant] = read(stream, &output)?;
            }
        }
        if measured && pair >= warm {
            cell.samples_ns.push(ns);
        }
    }
    Ok(cell)
}

fn bench_phase(fixture: &Fixture, measured: bool) -> Result<Vec<BenchCell>, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let weight = ctx.upload(&fixture.raw)?;
    let canonical = ctx.upload(&fixture.packed)?;
    let scales = ctx.upload(&fixture.scales)?;
    let view = ctx.nvfp4_weight_range(
        canonical.range(0, canonical.bytes())?,
        scales.range(0, scales.bytes())?,
        fixture.factor,
        [N, K],
    )?;
    let floor = ctx.mmq_fixup_bytes()?;
    let mut cells = Vec::new();
    for rows in [5, 64, 128, 144] {
        let shape = Nvfp4DenseShape::new(K, N, rows)?;
        let plan = ctx.mmq_nvfp4_dense_plan(
            weight.range(0, weight.bytes())?,
            [K, N, rows],
            fixture.factor,
        )?;
        let old = ctx.nvfp4_grouped_plan(&vec![view; rows])?;
        let input: Vec<_> = fixture
            .input
            .iter()
            .copied()
            .cycle()
            .take(rows * K)
            .flat_map(f32::to_ne_bytes)
            .collect();
        let scratch = Scratch {
            rows,
            input: ctx.upload(&input)?,
            q8: ctx.upload(&vec![0xff; shape.needs.y])?,
            fixup: ctx.upload(&vec![0xff; floor])?,
        };
        let run = |mmq| {
            if mmq {
                candidate(&ctx, &stream, &scratch, &plan)
            } else {
                baseline(&ctx, &stream, &scratch, &old)
            }
        };
        cells.push(bench_cell(&stream, run, rows, measured)?);
    }
    Ok(cells)
}

#[test]
#[ignore = "requires admitted CUDA, pinned fixtures; timing mode additionally requires reviewed harness and exclusive coordinator GO"]
fn nvfp4_dense_l0_gate_microbench() {
    let mode = std::env::var("FIFI_FLINT_MMQ_BENCH_MODE").expect("explicit dry or timing mode");
    let measured = match mode.as_str() {
        "dry" => false,
        "timing" => true,
        _ => panic!("FIFI_FLINT_MMQ_BENCH_MODE must be dry or timing"),
    };
    assert!(device_present("nvfp4_dense_l0_gate_microbench"));
    let fixture = fixture();
    assert_correctness(&fixture);
    let cells = bench_phase(&fixture, measured).expect("paired projection phase completed");
    // Device owners are gone before assertions and JSON construction.
    assert_eq!(
        cells.iter().map(|c| c.rows).collect::<Vec<_>>(),
        [5, 64, 128, 144]
    );
    for cell in cells {
        let expected: Vec<_> = fixture
            .expected
            .iter()
            .copied()
            .cycle()
            .take(cell.rows * N)
            .collect();
        let old: Vec<_> = cell.output[0].iter().map(|&v| f64::from(v)).collect();
        assert_rows(
            &cell.output[0],
            &expected,
            cell.rows,
            "bench-current-vs-original-f64",
        );
        assert_rows(
            &cell.output[1],
            &expected,
            cell.rows,
            "bench-mmq-vs-original-f64",
        );
        assert_rows(&cell.output[1], &old, cell.rows, "bench-mmq-vs-current");
        assert_eq!(cell.samples_ns.len(), if measured { 30 } else { 0 });
        assert!(cell.samples_ns.iter().flatten().all(|&ns| ns > 0));
        eprintln!(
            "DENSE_MICROBENCH_RECORD {}",
            serde_json::json!({
                "mode": mode.as_str(), "M":cell.rows, "K":K, "N":N,
                "input": "cyclic repeat of five pinned native post_norm rows",
                "input_sha256":INPUT_SHA,
                "pair_columns":["current Frame.project", "NVFP4 MMQ int8"],
                "warm_pairs":if measured {8} else {1},
                "samples_ns":cell.samples_ns,
                "included":"input quantization, projection, BF16 publication, frame allocation, status/read synchronization",
                "excluded":"immutable weights/plans/factors/input upload, full fixture validation, output readback and returned-output destruction",
                "build_profile":"native debug/libtest host, CUDA O3/sm86; not production model throughput",
                "no_model_or_default_qualification":true
            })
        );
    }
    eprintln!("DENSE_MICROBENCH_DONE mode={mode} ONE_KN_CELL NO_MODEL_QUALIFICATION");
}
