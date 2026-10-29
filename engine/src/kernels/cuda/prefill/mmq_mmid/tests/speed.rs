//! Speed rows of the grouped launch: UNPAIRED and REPORTED ONLY, never gated. At M = 512, 1,024 and
//! 2,048 tokens on the 3090 (release profile) they print the three mmid launches of one MoE layer
//! (gate, up, down; 256 experts, top 8, uniform random ids; Q4_K; K 2,048 / N 512, K 2,048 / N 512,
//! K 512 / N 2,048) in milliseconds per layer and times 40 layers, beside two anchors: an earlier
//! probe (gate 870 us + up 870 us + down 1,117 us = 2.86 ms per layer at T = 1,024, MEASURED) and
//! llama.cpp's expert bucket (92.4 ms at 1K, MEASURED), and the builder (guard plus
//! `mm_ids_helper`) in microseconds at each M. The numbers are host wall-clock over 20 launches of
//! a synchronized stream, after 3 warm-up launches.
//!
//! The timed kernels are the correct ones: the same process spot-checks eight slots of the gate
//! and of the down output against the f64 reference under the frozen bounds, after the HARD GATE
//! (every output finite). In the debug profile (`-G -O0` device code) nothing is timed: the rows
//! run once at M = 64 and say so.

use std::time::Instant;

use super::rig::{
    IdBufs, NAN_BITS, POISON, dequantized, expert_blocks, f32_bytes, f32s, nan_bytes, normal_rows,
    slot_experts,
};
use super::{device_facts, random_rows, read};
use crate::EngError;
use crate::kernels::cuda::prefill::mmq_mmid::{
    MmidLaunchCall, MmidQuantCall, MmidShape, MmidWeight,
};
use crate::kernels::cuda::prefill::mmq_size::{fixup_floats_max, q8_1_bytes};
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::kernels::cuda::testkit;
use crate::kernels::cuda::{CudaContext, DeviceBuffer};

/// The frozen NMSE bounds of the gate/up and down classes.
const GATE_UP_NMSE: f64 = 1e-3;
const DOWN_NMSE: f64 = 2e-3;
/// Layers of the model: the per-layer time is multiplied by this for the comparison.
const LAYERS: f64 = 40.0;
/// Timed launches after the warm-up.
const ITERS: usize = 20;

/// What one M of the speed rows measured and left on the device.
struct Timed {
    m: usize,
    /// Milliseconds per layer: gate, up and down launches.
    layer_ms: f64,
    /// Microseconds per build (guard plus `mm_ids_helper`).
    ids_us: f64,
    gu: Vec<f32>,
    down: Vec<f32>,
    rows: Vec<i32>,
    x: Vec<f32>,
    act: Vec<f32>,
}

/// The three Q4_K expert stacks as device buffers.
struct Stacks {
    gate: DeviceBuffer,
    up: DeviceBuffer,
    down: DeviceBuffer,
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

/// Times one M: builds the ids, quantizes both inputs, runs the three launches (warm-up, then
/// `iters` timed), times the builder, and reads the outputs back.
fn time_m(ctx: &CudaContext, w: &Stacks, m: usize, iters: usize) -> Result<Timed, EngError> {
    let stream = ctx.stream()?;
    let shape = MmidShape {
        n_experts: 256,
        m,
        top_k: 8,
    };
    let ne = shape.ne_get_rows();
    let rows = random_rows(&shape, 0x5bee + m as u64);
    let x = normal_rows(m, 2048, 0x5bef + m as u64);
    let act = normal_rows(ne, 512, 0x5bf0 + m as u64);
    let bufs = IdBufs::new(ctx, &shape, &rows, 0)?;
    let ids = bufs.build(ctx, &stream, shape)?;
    let (y_gate_bytes, y_down_bytes) = (q8_1_bytes(2048, ne)?, q8_1_bytes(512, ne)?);
    let (x_buf, act_buf) = (ctx.upload(&f32_bytes(&x))?, ctx.upload(&f32_bytes(&act))?);
    let y_gate = ctx.upload(&vec![POISON; y_gate_bytes])?;
    let y_down = ctx.upload(&vec![POISON; y_down_bytes])?;
    ctx.mmq_quantize_scatter_ids(
        &stream,
        &ids,
        quant_call(2048, (&x_buf, m), (&y_gate, y_gate_bytes))?,
    )?;
    ctx.mmq_quantize_gather_ids(
        &stream,
        &ids,
        quant_call(512, (&act_buf, ne), (&y_down, y_down_bytes))?,
    )?;
    let gu = ctx.upload(&nan_bytes(ne * 1024))?;
    let down = ctx.upload(&nan_bytes(ne * 2048))?;
    let fixup_bytes = fixup_floats_max(device_facts(ctx)?.nsm)? * 4;
    let fixup = ctx.upload(&vec![0u8; fixup_bytes])?;
    // One layer: gate at column 0 of `gu` and up at column 512 (ld 1,024), then the down launch.
    let layer = || -> Result<(), EngError> {
        for (stack, col, y, y_bytes, k, n, ld, dst, dst_len) in [
            (
                &w.gate,
                0,
                &y_gate,
                y_gate_bytes,
                2048,
                512,
                1024,
                &gu,
                (ne - 1) * 1024 + 512,
            ),
            (
                &w.up,
                512,
                &y_gate,
                y_gate_bytes,
                2048,
                512,
                1024,
                &gu,
                (ne - 1) * 1024 + 512,
            ),
            (
                &w.down,
                0,
                &y_down,
                y_down_bytes,
                512,
                2048,
                2048,
                &down,
                (ne - 1) * 2048 + 2048,
            ),
        ] {
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
                    dst: dst.range(col * 4, dst_len * 4)?,
                    ld,
                    fixup: fixup.range(0, fixup_bytes)?,
                },
            )?;
        }
        Ok(())
    };
    for _ in 0..3 {
        layer()?;
    }
    stream.synchronize()?;
    let start = Instant::now();
    for _ in 0..iters {
        layer()?;
    }
    stream.synchronize()?;
    let layer_ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;
    let start = Instant::now();
    for _ in 0..iters {
        let _built = bufs.build(ctx, &stream, shape)?;
    }
    stream.synchronize()?;
    let ids_us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;
    Ok(Timed {
        m,
        layer_ms,
        ids_us,
        gu: f32s(&read(ctx, &gu)?),
        down: f32s(&read(ctx, &down)?),
        rows,
        x,
        act,
    })
}

/// All Ms in one context, over the three stacks (the same bytes the later checks dequantize).
fn speed_phase(
    blocks: &(Vec<u8>, Vec<u8>, Vec<u8>),
    ms: &[usize],
    iters: usize,
) -> Result<Vec<Timed>, EngError> {
    let ctx = CudaContext::new()?;
    let w = Stacks {
        gate: ctx.upload(&blocks.0)?,
        up: ctx.upload(&blocks.1)?,
        down: ctx.upload(&blocks.2)?,
    };
    ms.iter().map(|&m| time_m(&ctx, &w, m, iters)).collect()
}

/// The NMSE of the rows of `slots` in `got` (`ld` floats per slot) against the f64 product of
/// `x` rows (`row_of(slot)`) with the dequantized expert matrices `blocks` (`n` rows of `k`
/// values per expert), dequantizing only the experts those slots use.
fn spot_nmse(
    blocks: &[u8],
    (k, n, ld): (usize, usize, usize),
    experts: &[usize],
    x_row: impl Fn(usize) -> Vec<f32>,
    got: &[f32],
    slots: &[usize],
) -> f64 {
    let stride = n * (k / 256) * 144;
    let (mut err, mut energy) = (0.0, 0.0);
    for &slot in slots {
        let expert = experts[slot];
        let deq = dequantized(
            QuantType::Q4K,
            &blocks[expert * stride..(expert + 1) * stride],
        );
        let x = x_row(slot);
        for i in 0..n {
            let want: f64 = deq[i * k..(i + 1) * k]
                .iter()
                .zip(&x)
                .map(|(&w, &x)| f64::from(w) * f64::from(x))
                .sum();
            let diff = f64::from(got[slot * ld + i]) - want;
            err += diff * diff;
            energy += want * want;
        }
    }
    err / energy
}

#[test]
fn mmq_mmid_speed_rows_are_reported_unpaired() {
    if !testkit::device_present("mmq_mmid_speed_rows_are_reported_unpaired") {
        return;
    }
    let release = !cfg!(debug_assertions);
    let (ms, iters): (&[usize], usize) = if release {
        (&[512, 1024, 2048], ITERS)
    } else {
        (&[64], 1)
    };
    let blocks = (
        expert_blocks(QuantType::Q4K, 256, 512, 2048, 0x5e1),
        expert_blocks(QuantType::Q4K, 256, 512, 2048, 0x5e2),
        expert_blocks(QuantType::Q4K, 256, 2048, 512, 0x5e3),
    );
    let timed = speed_phase(&blocks, ms, iters).expect("the speed launches");
    for t in &timed {
        let ne = t.m * 8;
        // The HARD GATE: every output of both launches is finite and the unused half of nothing
        // is left (gu holds gate and up in full rows).
        assert!(
            t.gu[..ne * 1024]
                .iter()
                .all(|v| v.is_finite() && v.to_bits() != NAN_BITS),
            "HARD GATE (speed M {}): gate/up output holds a non-finite value",
            t.m
        );
        assert!(
            t.down[..ne * 2048]
                .iter()
                .all(|v| v.is_finite() && v.to_bits() != NAN_BITS),
            "HARD GATE (speed M {}): down output holds a non-finite value",
            t.m
        );
        let shape = MmidShape {
            n_experts: 256,
            m: t.m,
            top_k: 8,
        };
        let experts = slot_experts(&shape, &t.rows);
        let slots: Vec<usize> = (0..8).map(|i| i * (ne / 8) + i).collect();
        let gate = spot_nmse(
            &blocks.0,
            (2048, 512, 1024),
            &experts,
            |s| t.x[(s / 8) * 2048..(s / 8 + 1) * 2048].to_vec(),
            &t.gu,
            &slots,
        );
        let up_shifted: Vec<f32> = t.gu.iter().skip(512).copied().collect();
        let up = spot_nmse(
            &blocks.1,
            (2048, 512, 1024),
            &experts,
            |s| t.x[(s / 8) * 2048..(s / 8 + 1) * 2048].to_vec(),
            &up_shifted,
            &slots,
        );
        let down = spot_nmse(
            &blocks.2,
            (512, 2048, 2048),
            &experts,
            |s| t.act[s * 512..(s + 1) * 512].to_vec(),
            &t.down,
            &slots,
        );
        println!(
            "speed M {}: spot-check NMSE gate {gate:.2e}, up {up:.2e}, down {down:.2e}",
            t.m
        );
        assert!(
            gate <= GATE_UP_NMSE && up <= GATE_UP_NMSE,
            "speed M {}: gate/up spot-check above the frozen bound",
            t.m
        );
        assert!(
            down <= DOWN_NMSE,
            "speed M {}: down spot-check above the frozen bound",
            t.m
        );
        if release {
            println!(
                "speed M {} (unpaired, reported only): gate+up+down {:.3} ms/layer, x40 = {:.1} ms (earlier probe 2.86 ms/layer at T 1,024; llama.cpp expert bucket 92.4 ms at 1K); ids build (guard + mm_ids_helper) {:.1} us",
                t.m,
                t.layer_ms,
                t.layer_ms * LAYERS,
                t.ids_us
            );
        } else {
            println!(
                "speed M {}: debug profile, not timed (device code is -G -O0)",
                t.m
            );
        }
    }
}
