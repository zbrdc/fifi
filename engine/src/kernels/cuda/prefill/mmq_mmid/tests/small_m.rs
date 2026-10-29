//! The speculation-verification shape of the grouped launch: M = 2, 5 and 8 rows (decode with a
//! draft of 1-8 tokens verified per step). At this edge each expert gets 0, 1 or a few rows and
//! most of the 256 experts of the resident bank none, so the tests run the whole path (guard,
//! builder, typed scatter or gather quantize, grouped launch) at M = 2, 5 and 8 against the f64
//! reference under the frozen bounds, the HARD GATE first, with several experts empty in every
//! case:
//!
//! - the toy shape (4 experts of 128 rows, K 512, two ids per token, one expert empty and one
//!   or two with a single row),
//! - the Q8X gate/up shape (K 2,048, N 512, `ld` 1,024) and the Q8X down shape (K 512, N 2,048,
//!   gathered), over a 16-expert bank of which only 12 are live (at least four empty).
//!
//! The cost test measures, per M and per pass, the GPU time of the sort/gather side of one MoE
//! layer beside the GEMM's: the guard kernel, the builder (guard plus `mm_ids_helper`; the
//! helper's share is the difference), the scatter and gather quantize, and the three grouped
//! launches, on the resident 256-expert Qwen shape. Each pass is timed between two CUDA events
//! on its stream, 20 times, and the median is printed in microseconds. The numbers are UNPAIRED
//! and reported only; the debug profile (device code `-G -O0`) times nothing.

use super::device_facts;
use super::rig::{
    IdBufs, LaunchSpec, POISON, dequantized, errors, expert_blocks, f32_bytes, gate_dst,
    launch_once, nan_bytes, normal_rows, reference, skewed_rows,
};
use super::{IdsGuardCall, random_rows};
use crate::EngError;
use crate::kernels::cuda::prefill::mmq_mmid::{
    MmidLaunchCall, MmidQuantCall, MmidShape, MmidWeight,
};
use crate::kernels::cuda::prefill::mmq_size::{fixup_floats_max, q8_1_bytes};
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::kernels::cuda::testkit;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};

/// The frozen NMSE bounds of the gate/up and down classes.
const GATE_UP_NMSE: f64 = 1e-3;
const DOWN_NMSE: f64 = 2e-3;
/// Timed launches per pass.
const SAMPLES: usize = 20;

/// The expert column counts of a launch from its `expert_bounds`.
fn counts(bounds: &[i32], experts: usize) -> Vec<usize> {
    bounds[..=experts]
        .windows(2)
        .map(|w| usize::try_from(w[1] - w[0]).expect("gated"))
        .collect()
}

/// Runs the toy, Q8X gate/up and Q8X down cases at `m` rows and asserts them; returns nothing.
fn small_m_cases(m: usize) {
    // Toy: tokens alternate the pairs (0, 1) and (1, 2), so expert 3 is empty and, at m = 2,
    // experts 0 and 2 hold a single row.
    let toy = MmidShape {
        n_experts: 4,
        m,
        top_k: 2,
    };
    let (k, n, ld) = (512, 128, 136);
    let toy_rows: Vec<i32> = (0..m)
        .flat_map(|t| if t % 2 == 0 { [0, 1] } else { [1, 2] })
        .collect();
    let toy_w = expert_blocks(QuantType::Q4K, 4, n, k, 0x5a1 + m as u64);
    let toy_x = normal_rows(m, k, 0x5a2 + m as u64);
    let toy_spec = LaunchSpec {
        shape: toy,
        quant: QuantType::Q4K,
        k,
        n,
        ld,
        rows: &toy_rows,
        x: &toy_x,
        gather: false,
    };
    // Q8X gate/up and down over a 16-expert bank, 12 live.
    let bank = MmidShape {
        n_experts: 16,
        m,
        top_k: 8,
    };
    let rows = skewed_rows(&bank, 12, 0x5b1 + m as u64);
    let ne = bank.ne_get_rows();
    let gate_w = expert_blocks(QuantType::Q4K, 16, 512, 2048, 0x5b2 + m as u64);
    let down_w = expert_blocks(QuantType::Q4K, 16, 2048, 512, 0x5b3 + m as u64);
    let gate_x = normal_rows(m, 2048, 0x5b4 + m as u64);
    let down_x = normal_rows(ne, 512, 0x5b5 + m as u64);
    let gate = LaunchSpec {
        shape: bank,
        quant: QuantType::Q4K,
        k: 2048,
        n: 512,
        ld: 1024,
        rows: &rows,
        x: &gate_x,
        gather: false,
    };
    let down = LaunchSpec {
        shape: bank,
        quant: QuantType::Q4K,
        k: 512,
        n: 2048,
        ld: 2048,
        rows: &rows,
        x: &down_x,
        gather: true,
    };
    let phase = || -> Result<Vec<super::rig::Launched>, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let (toy_buf, gate_buf, down_buf) = (
            ctx.upload(&toy_w)?,
            ctx.upload(&gate_w)?,
            ctx.upload(&down_w)?,
        );
        Ok(vec![
            launch_once(&ctx, &stream, &toy_buf, &toy_spec)?,
            launch_once(&ctx, &stream, &gate_buf, &gate)?,
            launch_once(&ctx, &stream, &down_buf, &down)?,
        ])
    };
    let runs = phase().expect("the small-M launches");
    let specs = [
        ("toy", &toy_spec, &toy_w, GATE_UP_NMSE),
        ("Q8X gate/up", &gate, &gate_w, GATE_UP_NMSE),
        ("Q8X down", &down, &down_w, DOWN_NMSE),
    ];
    for ((name, spec, weights, bound), got) in specs.into_iter().zip(&runs) {
        let tag = format!("{name}, M {m}");
        let rows_n = spec.shape.ne_get_rows();
        gate_dst(&got.dst, (rows_n, spec.n, spec.ld), &tag);
        assert_eq!(got.status, 0, "{tag}: clean rows set no status mask");
        let per_expert = counts(&got.bounds, spec.shape.n_experts);
        let empty = per_expert.iter().filter(|&&c| c == 0).count();
        assert!(
            empty >= if name == "toy" { 1 } else { 4 },
            "{tag}: only {empty} empty experts: {per_expert:?}"
        );
        if m == 2 {
            assert!(
                per_expert.iter().all(|&c| c <= 2),
                "{tag}: an expert holds more than the two tokens: {per_expert:?}"
            );
        }
        let err = errors(
            &got.dst,
            &reference(&dequantized(QuantType::Q4K, weights), spec),
            (rows_n, spec.n, spec.ld),
        );
        println!(
            "small M {tag}: empty experts {empty}, counts {per_expert:?}, NMSE {:.3e}",
            err.nmse
        );
        assert!(
            err.nmse <= bound,
            "{tag}: NMSE {:.3e} above the frozen {bound:e}",
            err.nmse
        );
    }
}

#[test]
fn mmq_mmid_small_m_2_matches_the_f64_reference_with_empty_experts() {
    if !testkit::device_present("mmq_mmid_small_m_2_matches_the_f64_reference_with_empty_experts") {
        return;
    }
    small_m_cases(2);
}

#[test]
fn mmq_mmid_small_m_5_matches_the_f64_reference_with_empty_experts() {
    if !testkit::device_present("mmq_mmid_small_m_5_matches_the_f64_reference_with_empty_experts") {
        return;
    }
    small_m_cases(5);
}

#[test]
fn mmq_mmid_small_m_8_matches_the_f64_reference_with_empty_experts() {
    if !testkit::device_present("mmq_mmid_small_m_8_matches_the_f64_reference_with_empty_experts") {
        return;
    }
    small_m_cases(8);
}

/// The GPU time in microseconds of `pass`, between two events on `stream`.
fn gpu_us(
    ctx: &CudaContext,
    stream: &DeviceStream,
    mut pass: impl FnMut() -> Result<(), EngError>,
) -> Result<f64, EngError> {
    let start = ctx.bench_mark(stream)?;
    pass()?;
    let end = ctx.bench_mark(stream)?;
    stream.synchronize()?;
    let ms = ctx.bench_elapsed_ms(start, end)?;
    ctx.bench_destroy(start)?;
    ctx.bench_destroy(end)?;
    Ok(f64::from(ms) * 1e3)
}

/// The median of `SAMPLES` timings of `pass`, after one warm-up.
fn median_us(
    ctx: &CudaContext,
    stream: &DeviceStream,
    mut pass: impl FnMut() -> Result<(), EngError>,
) -> Result<f64, EngError> {
    gpu_us(ctx, stream, &mut pass)?;
    let mut samples = (0..SAMPLES)
        .map(|_| gpu_us(ctx, stream, &mut pass))
        .collect::<Result<Vec<_>, _>>()?;
    samples.sort_by(f64::total_cmp);
    Ok((samples[SAMPLES / 2 - 1] + samples[SAMPLES / 2]) / 2.0)
}

/// Median microseconds of each pass at one M: guard, builder, scatter, gate, up, down, gather.
struct Costs {
    m: usize,
    guard: f64,
    builder: f64,
    scatter: f64,
    launches: [f64; 3],
    gather: f64,
}

/// A Q4_K quantize call over `rows` rows of `k` floats in `x` into `y` (`y_bytes` bytes).
fn quant_call<'b>(
    k: usize,
    (x, rows): (&'b DeviceBuffer, usize),
    (y, y_bytes): (&'b DeviceBuffer, usize),
) -> Result<MmidQuantCall<'b>, EngError> {
    Ok(MmidQuantCall {
        packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
        quant: QuantType::Q4K,
        k,
        row_stride: k,
        x: x.range(0, rows * k * 4)?,
        y: y.range(0, y_bytes)?,
    })
}

/// The weights of the three stacks on the device.
struct Stacks {
    gate: DeviceBuffer,
    up: DeviceBuffer,
    down: DeviceBuffer,
}

fn measure(
    ctx: &CudaContext,
    w: &Stacks,
    m: usize,
    (kg, n_gate, kd, n_down): (usize, usize, usize, usize),
) -> Result<Costs, EngError> {
    let stream = ctx.stream()?;
    let shape = MmidShape {
        n_experts: 256,
        m,
        top_k: 8,
    };
    let ne = shape.ne_get_rows();
    let rows = random_rows(&shape, 0x7c0 + m as u64);
    let bufs = IdBufs::new(ctx, &shape, &rows, 0)?;
    let ids = bufs.build(ctx, &stream, shape)?;
    let (x_buf, act_buf) = (
        ctx.upload(&f32_bytes(&normal_rows(m, kg, 0x7c1)))?,
        ctx.upload(&f32_bytes(&normal_rows(ne, kd, 0x7c2)))?,
    );
    let (y_gate_bytes, y_down_bytes) = (q8_1_bytes(kg, ne)?, q8_1_bytes(kd, ne)?);
    let (y_gate, y_down) = (
        ctx.upload(&vec![POISON; y_gate_bytes])?,
        ctx.upload(&vec![POISON; y_down_bytes])?,
    );
    let fixup_bytes = fixup_floats_max(device_facts(ctx)?.nsm)? * 4;
    let fixup = ctx.upload(&vec![0u8; fixup_bytes])?;
    let gu = ctx.upload(&nan_bytes(ne * n_gate))?;
    let down = ctx.upload(&nan_bytes(ne * n_down))?;
    let guard = median_us(ctx, &stream, || {
        ctx.mmq_ids_guard(
            &stream,
            IdsGuardCall {
                shape,
                ids: bufs.ids.range(0, ne * 4)?,
                ids_g: bufs.ids_g.range(0, ne * 4)?,
                status: None,
            },
        )
    })?;
    let builder = median_us(ctx, &stream, || bufs.build(ctx, &stream, shape).map(|_| ()))?;
    let scatter = median_us(ctx, &stream, || {
        ctx.mmq_quantize_scatter_ids(
            &stream,
            &ids,
            quant_call(kg, (&x_buf, m), (&y_gate, y_gate_bytes))?,
        )
    })?;
    let gather = median_us(ctx, &stream, || {
        ctx.mmq_quantize_gather_ids(
            &stream,
            &ids,
            quant_call(kd, (&act_buf, ne), (&y_down, y_down_bytes))?,
        )
    })?;
    let mut launches = [0.0; 3];
    for (slot, (stack, k, n, y, y_bytes, dst, dst_len)) in [
        (
            &w.gate,
            kg,
            n_gate,
            &y_gate,
            y_gate_bytes,
            &gu,
            (ne - 1) * n_gate + n_gate,
        ),
        (
            &w.up,
            kg,
            n_gate,
            &y_gate,
            y_gate_bytes,
            &gu,
            (ne - 1) * n_gate + n_gate,
        ),
        (
            &w.down,
            kd,
            n_down,
            &y_down,
            y_down_bytes,
            &down,
            (ne - 1) * n_down + n_down,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        launches[slot] = median_us(ctx, &stream, || {
            ctx.mmq_mmid_launch(
                &stream,
                MmidLaunchCall {
                    shape,
                    weight: MmidWeight {
                        base: stack.range(0, stack.bytes())?,
                        quant: QuantType::Q4K,
                        k,
                        n,
                        stride_bytes: n * (k / 256) * 144,
                    },
                    ids: &ids,
                    y: y.range(0, y_bytes)?,
                    dst: dst.range(0, dst_len * 4)?,
                    ld: n,
                    fixup: fixup.range(0, fixup_bytes)?,
                },
            )
        })?;
    }
    Ok(Costs {
        m,
        guard,
        builder,
        scatter,
        launches,
        gather,
    })
}

#[test]
fn mmq_mmid_small_m_pass_costs_are_reported_unpaired() {
    if !testkit::device_present("mmq_mmid_small_m_pass_costs_are_reported_unpaired") {
        return;
    }
    // The correctness of the timed kernels first, in this process: the small-M cases of the tests above.
    for m in [2, 5, 8] {
        small_m_cases(m);
    }
    let release = !cfg!(debug_assertions);
    // Release: the resident Qwen shapes (256 experts; gate/up K 2,048 N 512; down K 512 N 2,048).
    // Debug (device code -G -O0): small shapes, to run the path once and say so.
    let dims = if release {
        (2048, 512, 512, 2048)
    } else {
        (256, 128, 256, 128)
    };
    let (kg, n_gate, kd, n_down) = dims;
    let blocks = (
        expert_blocks(QuantType::Q4K, 256, n_gate, kg, 0x7d1),
        expert_blocks(QuantType::Q4K, 256, n_gate, kg, 0x7d2),
        expert_blocks(QuantType::Q4K, 256, n_down, kd, 0x7d3),
    );
    let phase = || -> Result<Vec<Costs>, EngError> {
        let ctx = CudaContext::new()?;
        let w = Stacks {
            gate: ctx.upload(&blocks.0)?,
            up: ctx.upload(&blocks.1)?,
            down: ctx.upload(&blocks.2)?,
        };
        [2, 5, 8]
            .into_iter()
            .map(|m| measure(&ctx, &w, m, dims))
            .collect()
    };
    let costs = phase().expect("the pass-cost launches");
    for c in &costs {
        let gemms: f64 = c.launches.iter().sum();
        let sort = c.builder;
        println!(
            "small M {} (unpaired, median of {SAMPLES} per pass, GPU us between events{}): guard {:.1}, builder {:.1} (guard + mm_ids_helper; helper {:.1}), scatter quantize {:.1}, gather quantize {:.1}; grouped launches gate {:.1}, up {:.1}, down {:.1} (sum {:.1}); sort share of the three launches {:.1} %, of one gate launch {:.1} %",
            c.m,
            if release {
                ""
            } else {
                "; debug profile, small shapes, not a measurement"
            },
            c.guard,
            c.builder,
            c.builder - c.guard,
            c.scatter,
            c.gather,
            c.launches[0],
            c.launches[1],
            c.launches[2],
            gemms,
            100.0 * sort / gemms,
            100.0 * sort / c.launches[0]
        );
        assert!(
            c.guard > 0.0 && c.builder > 0.0 && gemms > 0.0,
            "a pass took no time"
        );
    }
}
