//! Manual calibration of completed owned copies; not a model-throughput benchmark.
use std::time::Instant;

use super::*;
use crate::kernels::cuda::{PackCall, device_present, native};

const SIZES: [usize; 7] = [
    16 << 10,
    64 << 10,
    256 << 10,
    1 << 20,
    2 << 20,
    4 << 20,
    8 << 20,
];
// Five full repeats must fit the exclusive two-minute calibration turn.
const ITERATIONS: usize = 40;
const SAMPLES: usize = 5;

struct Bench {
    ctx: CudaContext,
    stream: DeviceStream,
    event: DeviceEvent,
    device: super::super::DeviceBuffer,
    pinned: PinnedBuf,
}

impl Bench {
    fn copy(&mut self, bytes: usize, upload: bool) -> Result<(), EngError> {
        let range = self.device.range(0, bytes)?;
        if upload {
            self.ctx
                .copy_from_pinned(&self.stream, &mut self.pinned, 0, range)?;
        } else {
            self.ctx
                .copy_to_pinned(&self.stream, range, &mut self.pinned, 0)?;
        }
        self.stream.record(&self.event)?;
        self.pinned.wait(&self.event)
    }

    fn sample(&mut self, bytes: usize, upload: bool, iterations: usize) -> Result<u128, EngError> {
        let start = Instant::now();
        for _ in 0..iterations {
            self.copy(bytes, upload)?;
        }
        Ok(start.elapsed().as_nanos())
    }
}

fn median_per_copy(ns: &[u128]) -> f64 {
    let mut sorted = ns.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2] as f64 / ITERATIONS as f64
}

struct BusyPack {
    stream: DeviceStream,
    x: super::super::DeviceBuffer,
    xf: super::super::DeviceBuffer,
}

impl BusyPack {
    fn new(ctx: &CudaContext) -> Result<Self, EngError> {
        let need = native::activation_bytes(8, 32768, 2048).expect("fixed valid pack geometry");
        let stream = ctx.stream()?;
        let x = ctx.allocate(need.x)?;
        let xf = ctx.allocate(need.xf)?;
        x.write_at(&stream, 0, &vec![0; need.x])?;
        Ok(Self { stream, x, xf })
    }

    fn queue(&self, ctx: &CudaContext) -> Result<(), EngError> {
        for _ in 0..4 {
            ctx.pack(
                &self.stream,
                PackCall {
                    x: self.x.range(0, self.x.bytes())?,
                    xf: self.xf.range(0, self.xf.bytes())?,
                    m: 2048,
                    k: 32768,
                },
            )?;
        }
        Ok(())
    }
}

fn overlap(bench: &mut Bench) -> Result<(serde_json::Value, bool), EngError> {
    let busy = BusyPack::new(&bench.ctx)?;
    busy.queue(&bench.ctx)?;
    busy.stream.synchronize()?;
    let bytes = *SIZES.last().expect("fixed sizes");
    let expected: Vec<u8> = (0..bytes).map(|i| (i % 251) as u8).collect();
    bench.pinned.write(0, &expected)?;
    let mut total_samples = [Vec::new(), Vec::new()];
    let mut copy_samples = [Vec::new(), Vec::new()];
    for sample in 0..SAMPLES {
        // Alternate serial/concurrent order to limit drift.
        for concurrent in [sample % 2 == 0, sample % 2 != 0] {
            let mut total_ns = 0;
            let mut copy_ns = 0;
            for _ in 0..ITERATIONS {
                let start = Instant::now();
                busy.queue(&bench.ctx)?;
                if !concurrent {
                    busy.stream.synchronize()?;
                }
                let copy_start = Instant::now();
                bench.copy(bytes, true)?;
                copy_ns += copy_start.elapsed().as_nanos();
                busy.stream.synchronize()?;
                total_ns += start.elapsed().as_nanos();
            }
            total_samples[usize::from(concurrent)].push(total_ns);
            copy_samples[usize::from(concurrent)].push(copy_ns);
        }
    }
    let serial_ns = median_per_copy(&total_samples[0]);
    let concurrent_ns = median_per_copy(&total_samples[1]);
    let copy_ns = median_per_copy(&copy_samples[1]);
    let mut packed = [0xff; 16];
    busy.xf.read_at(&busy.stream, 0, &mut packed)?;
    bench.pinned.write(0, &vec![0; bytes])?;
    bench.copy(bytes, false)?;
    let mut got = vec![0; bytes];
    bench.pinned.read(0, &mut got)?;
    Ok((
        serde_json::json!({
            "direction": "h2d_busy_pack",
            "bytes": bytes,
            "busy_kernel": "four checked f32-to-fp16 activation packs, each 2048x32768",
            "kernel_scope": "memory-bound pack, not GEMM/model throughput",
            "iterations_per_sample": ITERATIONS,
            "serial_samples_ns": total_samples[0],
            "concurrent_samples_ns": total_samples[1],
            "concurrent_copy_samples_ns": copy_samples[1],
            "median_serial_total_ns": serial_ns,
            "median_concurrent_total_ns": concurrent_ns,
            "median_concurrent_copy_ns": copy_ns,
            "median_gb_s": bytes as f64 / copy_ns,
            "matched_serial_over_concurrent": serial_ns / concurrent_ns,
            "scope": "CPU wall time; serial and concurrent both complete copy and busy stream",
        }),
        packed == [0; 16] && got == expected && copy_ns > 0.0 && concurrent_ns > 0.0,
    ))
}

fn measure() -> Result<(Vec<serde_json::Value>, bool), EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let event = ctx.event()?;
    let device = ctx.allocate(*SIZES.last().expect("fixed sizes"))?;
    let pinned = ctx.pinned_buf(device.bytes())?;
    let mut bench = Bench {
        ctx,
        stream,
        event,
        device,
        pinned,
    };
    let mut results = Vec::new();
    let mut exact = true;
    for bytes in SIZES {
        let expected: Vec<u8> = (0..bytes).map(|i| (i % 251) as u8).collect();
        bench.pinned.write(0, &expected)?;
        for upload in [true, false] {
            if !upload {
                bench.pinned.write(0, &vec![0; bytes])?;
            }
            bench.sample(bytes, upload, 3)?;
            let mut ns = Vec::new();
            for _ in 0..SAMPLES {
                ns.push(bench.sample(bytes, upload, ITERATIONS)?);
            }
            let median_ns = median_per_copy(&ns);
            let gib_s = bytes as f64 / median_ns * 1e9 / (1_u64 << 30) as f64;
            exact &= median_ns > 0.0 && gib_s.is_finite();
            results.push(serde_json::json!({
                "direction": if upload { "h2d" } else { "d2h" },
                "bytes": bytes,
                "iterations_per_sample": ITERATIONS,
                "samples_ns": ns,
                "median_copy_ns": median_ns,
                "median_gib_s": gib_s,
                "median_gb_s": bytes as f64 / median_ns,
                "scope": "CPU wall time: checked enqueue + event record + completion wait",
                "includes": "one event completion per copy",
                "excludes": "allocation, host fill/read, warmups, model computation",
            }));
        }
        let mut got = vec![0; bytes];
        bench.pinned.read(0, &mut got)?;
        exact &= got == expected;
    }
    let (overlap_row, overlap_exact) = overlap(&mut bench)?;
    results.push(overlap_row);
    Ok((results, exact && overlap_exact))
}

#[test]
#[ignore = "manual exclusive-GPU pinned transfer calibration"]
fn owned_pinned_transfer_calibration() {
    if !device_present("owned_pinned_transfer_calibration") {
        return;
    }
    let (results, exact) = measure().expect("real owned pinned transfers");
    assert_eq!(results.len(), 2 * SIZES.len() + 1);
    assert!(
        exact,
        "completed copies must roundtrip exactly and have valid timing"
    );
    for row in results {
        println!("PINNED_TRANSFER_BENCH {row}");
    }
    println!("PINNED_TRANSFER_BENCH exact_roundtrip=true");
}
