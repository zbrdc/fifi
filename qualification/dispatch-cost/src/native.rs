//! The native measurement. This module exists only on Apple Silicon with `native-mlx`.

use std::{
    collections::BTreeMap,
    process::Command,
    time::{Duration, Instant},
};

use inference_engine::{
    EngError,
    kernels::{
        MlxKernels, PackedWeight, QuantizationMode, QuantizationSpec, ResidentPackedWeight,
        RopeStep,
    },
    tensor::{Dtype, ExecutionTensor, Tensor},
};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
    Options,
    host::parse_load_averages,
    plan::{self, Config},
    stats::{Summary, summarize},
    synth::{activation_le, biases_le, dense_le, norm_le, scales_le, words_le},
};

#[derive(Serialize, Clone)]
struct Host {
    load_1m: Option<f64>,
    load_5m: Option<f64>,
    load_15m: Option<f64>,
    gpu_device_utilization_pct: Option<u32>,
}

fn command_text(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Host load and GPU utilization. Run outside every timed region.
fn host_sample() -> Host {
    let [load_1m, load_5m, load_15m] = command_text("sysctl", &["-n", "vm.loadavg"])
        .map(|text| parse_load_averages(&text))
        .unwrap_or([None; 3]);
    let gpu =
        command_text("ioreg", &["-r", "-d", "1", "-w0", "-c", "IOAccelerator"]).and_then(|t| {
            let tail = t.split("\"Device Utilization %\"=").nth(1)?;
            let digits: String = tail.chars().take_while(char::is_ascii_digit).collect();
            digits.parse().ok()
        });
    Host {
        load_1m,
        load_5m,
        load_15m,
        gpu_device_utilization_pct: gpu,
    }
}

#[derive(Serialize)]
struct SeriesHost {
    before: Host,
    after: Host,
}

type Res<T> = Result<T, EngError>;

const OPS: [&str; 14] = [
    "import",
    "qmm",
    "mm",
    "rms_norm",
    "silu",
    "mul",
    "add",
    "softmax",
    "rope",
    "readback",
    "release",
    "release_unread",
    "release_unread_input",
    "release_evaluated_input",
];

fn f32_tensor(shape: &[u32], bytes: Vec<u8>) -> Res<Tensor> {
    Tensor::try_from_host(Dtype::F32, shape.to_vec(), bytes)
}

fn packed(seed: u32, rows: u32, cols: u32) -> Res<PackedWeight> {
    let words = (cols * plan::BITS / 32) as usize;
    let groups = (cols / plan::GROUP_SIZE) as usize;
    let payload = Tensor::try_from_host(
        Dtype::U32,
        vec![rows, words as u32],
        words_le(seed, rows as usize * words),
    )?;
    let scale_seed = seed + 1;
    let count = rows as usize * groups;
    let shape = vec![rows, groups as u32];
    let scales = f32_tensor(&shape, scales_le(scale_seed, count))?;
    let biases = f32_tensor(&shape, biases_le(scale_seed, count))?;
    PackedWeight::try_new(
        payload,
        scales,
        Some(biases),
        vec![rows, cols],
        QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: plan::BITS,
            group_size: plan::GROUP_SIZE,
        },
    )
}

struct Layer {
    w1: ExecutionTensor,
    w2: ExecutionTensor,
    v: ExecutionTensor,
    score: ResidentPackedWeight,
    gate: ResidentPackedWeight,
    up: ResidentPackedWeight,
    down: ResidentPackedWeight,
    q_bytes: Vec<u8>,
    k_bytes: Vec<u8>,
}

#[derive(Default)]
struct Recorder {
    samples: BTreeMap<&'static str, Vec<f64>>,
}

impl Recorder {
    fn record(&mut self, name: &'static str, duration: Duration) {
        self.samples
            .entry(name)
            .or_default()
            .push(duration.as_secs_f64() * 1.0e6);
    }

    fn time<T>(&mut self, name: &'static str, call: impl FnOnce() -> Res<T>) -> Res<T> {
        let start = Instant::now();
        let value = call();
        self.record(name, start.elapsed());
        value
    }
}

fn build_layer(kernels: &mut MlxKernels, cfg: &Config, index: u32) -> Res<Layer> {
    let base = 100 * (index + 1);
    let (h, i, s) = (cfg.hidden, cfg.inter, cfg.seq);
    let w1 = kernels.import_tensor(&f32_tensor(&[h], norm_le(base + 1, h as usize))?)?;
    let w2 = kernels.import_tensor(&f32_tensor(&[h], norm_le(base + 5, h as usize))?)?;
    let v = kernels.import_tensor(&f32_tensor(&[s, h], dense_le(base + 4, (s * h) as usize))?)?;
    let score = kernels.import_packed(&packed(base + 2, s, h)?)?;
    let gate = kernels.import_packed(&packed(base + 6, i, h)?)?;
    let up = kernels.import_packed(&packed(base + 8, i, h)?)?;
    let down = kernels.import_packed(&packed(base + 10, h, i)?)?;
    let q_len = (plan::HEADS_Q * plan::HEAD_DIM) as usize;
    let k_len = (plan::HEADS_KV * plan::HEAD_DIM) as usize;
    Ok(Layer {
        w1,
        w2,
        v,
        score,
        gate,
        up,
        down,
        q_bytes: activation_le(base + 12, q_len),
        k_bytes: activation_le(base + 13, k_len),
    })
}

#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
struct ReleaseCounts {
    direct_resident_handles_per_chain: usize,
    direct_release_sync_calls_per_chain: usize,
    never_evaluated_resident_handles_per_chain: usize,
    never_evaluated_release_sync_calls_per_chain: usize,
    evaluated_source_release_sync_calls_per_chain: usize,
    rope_source_release_sync_calls_per_chain: usize,
}

struct Chain {
    wall_us: f64,
    output_sha256: String,
    finite: bool,
    release_counts: ReleaseCounts,
}

fn run_chain(
    kernels: &mut MlxKernels,
    cfg: &Config,
    layers: &[Layer],
    x_bytes: &[u8],
    rec: &mut Recorder,
) -> Res<Chain> {
    let h = cfg.hidden;
    let start = Instant::now();
    let x_host = f32_tensor(&[1, h], x_bytes.to_vec())?;
    let mut x = rec.time("import", || kernels.import_tensor(&x_host))?;
    let mut direct_release_handles = 0usize;
    let mut direct_release_sync_calls = 0usize;
    let mut never_evaluated_handles = 0usize;
    let mut never_evaluated_sync_calls = 0usize;
    let mut evaluated_source_release_sync_calls = 0usize;
    let mut rope_source_release_sync_calls = 0usize;
    for (layer_index, layer) in layers.iter().enumerate() {
        let q_host = f32_tensor(&[plan::HEADS_Q, 1, plan::HEAD_DIM], layer.q_bytes.clone())?;
        let k_host = f32_tensor(&[plan::HEADS_KV, 1, plan::HEAD_DIM], layer.k_bytes.clone())?;
        let mut q = rec.time("import", || kernels.import_tensor(&q_host))?;
        let mut k = rec.time("import", || kernels.import_tensor(&k_host))?;
        let n = rec.time("rms_norm", || {
            kernels.rms_norm(&x, &layer.w1, plan::RMS_EPS)
        })?;
        let s = rec.time("qmm", || kernels.qmm(&n, &layer.score))?;
        let p = rec.time("softmax", || kernels.softmax(&s))?;
        let a = rec.time("mm", || kernels.mm(&p, &layer.v))?;
        let x1 = rec.time("add", || kernels.add(&x, &a))?;
        let n2 = rec.time("rms_norm", || {
            kernels.rms_norm(&x1, &layer.w2, plan::RMS_EPS)
        })?;
        let g = rec.time("qmm", || kernels.qmm(&n2, &layer.gate))?;
        let u = rec.time("qmm", || kernels.qmm(&n2, &layer.up))?;
        let sg = rec.time("silu", || kernels.silu(&g))?;
        let act = rec.time("mul", || kernels.mul(&sg, &u))?;
        let d = rec.time("qmm", || kernels.qmm(&act, &layer.down))?;
        let x_new = rec.time("add", || kernels.add(&x1, &d))?;
        let step = RopeStep {
            pos: plan::ROPE_POS,
            theta: plan::ROPE_THETA,
            part: plan::ROPE_PART,
        };
        let rope_syncs_before = kernels.qualification_sync_count();
        rec.time("rope", || kernels.rope(&mut q, &mut k, &step))?;
        let rope_syncs = kernels.qualification_sync_count() - rope_syncs_before;
        assert_eq!(
            rope_syncs, 2,
            "rope releases its two replaced input residents"
        );
        rope_source_release_sync_calls += usize::try_from(rope_syncs).unwrap();

        // Split the 14 resident drops so the never-evaluated arrays have their own time series.
        let release_syncs_before = kernels.qualification_sync_count();
        let release_start = Instant::now();
        let unread_start = Instant::now();
        let unread_syncs_before = kernels.qualification_sync_count();
        drop((q, k, n, s, p, a, x1, n2, g, u, sg, act, d));
        let unread_duration = unread_start.elapsed();
        let unread_syncs = kernels.qualification_sync_count() - unread_syncs_before;
        assert_eq!(unread_syncs, UNREAD_INTERMEDIATE_DROPS as u64);

        let source_release_start = Instant::now();
        let source_syncs_before = kernels.qualification_sync_count();
        drop(std::mem::replace(&mut x, x_new));
        let source_release_duration = source_release_start.elapsed();
        let source_syncs = kernels.qualification_sync_count() - source_syncs_before;
        assert_eq!(
            source_syncs, 1,
            "each replaced x resident synchronizes once"
        );
        let release_duration = release_start.elapsed();
        let release_syncs = kernels.qualification_sync_count() - release_syncs_before;
        assert_eq!(release_syncs, RELEASE_DROPS as u64);

        rec.record("release", release_duration);
        rec.record("release_unread", unread_duration);
        direct_release_handles += RELEASE_DROPS;
        direct_release_sync_calls += usize::try_from(release_syncs).unwrap();
        never_evaluated_handles += UNREAD_INTERMEDIATE_DROPS;
        never_evaluated_sync_calls += usize::try_from(unread_syncs).unwrap();
        if layer_index == 0 {
            rec.record("release_evaluated_input", source_release_duration);
            evaluated_source_release_sync_calls += usize::try_from(source_syncs).unwrap();
        } else {
            rec.record("release_unread_input", source_release_duration);
            never_evaluated_handles += 1;
            never_evaluated_sync_calls += usize::try_from(source_syncs).unwrap();
        }
    }
    let out = rec.time("readback", || kernels.readback(&x))?;
    let wall_us = start.elapsed().as_secs_f64() * 1.0e6;
    let bytes = out.host_bytes().unwrap_or(&[]);
    let finite = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .all(|c| f32::from_le_bytes(*c).is_finite());
    let release_counts = ReleaseCounts {
        direct_resident_handles_per_chain: direct_release_handles,
        direct_release_sync_calls_per_chain: direct_release_sync_calls,
        never_evaluated_resident_handles_per_chain: never_evaluated_handles,
        never_evaluated_release_sync_calls_per_chain: never_evaluated_sync_calls,
        evaluated_source_release_sync_calls_per_chain: evaluated_source_release_sync_calls,
        rope_source_release_sync_calls_per_chain: rope_source_release_sync_calls,
    };
    assert_eq!(direct_release_handles, RELEASE_DROPS * layers.len());
    assert_eq!(direct_release_sync_calls, direct_release_handles);
    assert_eq!(never_evaluated_handles, direct_release_handles - 1);
    assert_eq!(never_evaluated_sync_calls, never_evaluated_handles);
    assert_eq!(evaluated_source_release_sync_calls, 1);
    assert_eq!(rope_source_release_sync_calls, 2 * layers.len());
    Ok(Chain {
        wall_us,
        output_sha256: format!("{:x}", Sha256::digest(bytes)),
        finite,
        release_counts,
    })
}

/// The L0 report used this pre-L2 synchronization formula. It is historical, not a count
/// of synchronizations in the current lazy chain.
fn l0_baseline_synchronizations_per_chain(layers: usize) -> usize {
    plan::CALLS_FIXED + L0_SYNCS_PER_LAYER * layers
}

/// Drops in the direct release interval: q, k, n, s, p, a, x1, n2, g, u, sg, act, d, and old x.
const RELEASE_DROPS: usize = 14;
/// The first 13 drops are lazy operation outputs; old x is counted separately.
const UNREAD_INTERMEDIATE_DROPS: usize = RELEASE_DROPS - 1;
/// Pre-L2 L0 synchronization estimate retained only for historical comparison.
const L0_ROPE_EXTRA_SYNCS: usize = 3;
const L0_SYNCS_PER_LAYER: usize = plan::CALLS_PER_LAYER + L0_ROPE_EXTRA_SYNCS + RELEASE_DROPS;
const _: () = assert!(L0_SYNCS_PER_LAYER == 32 && plan::CALLS_PER_LAYER == 15);

#[cfg(test)]
mod tests {
    use super::l0_baseline_synchronizations_per_chain;

    #[test]
    fn l0_sync_formula_matches_recorded_baseline_scope() {
        assert_eq!(l0_baseline_synchronizations_per_chain(1), 34);
        assert_eq!(l0_baseline_synchronizations_per_chain(8), 258);
    }
}

#[derive(Serialize)]
struct ChainResult {
    layers: usize,
    calls_per_chain: usize,
    repetitions: usize,
    chain: Summary,
    per_layer_us_median: f64,
    per_call_us_from_chain_median: f64,
    per_op: BTreeMap<&'static str, Summary>,
    output_sha256: String,
    output_stable: bool,
    output_finite: bool,
    l0_baseline_synchronizations_per_chain: usize,
    release_counts: ReleaseCounts,
    timed_release_samples_per_chain: usize,
    series_host: SeriesHost,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    engine_crate: &'static str,
    config: Config,
    warmup: usize,
    repetitions: usize,
    setup_us: f64,
    release_us: f64,
    release_per_layer_us: f64,
    adapter_release_us: f64,
    host_start: Host,
    host_end: Host,
    timing: &'static str,
    results: Vec<ChainResult>,
}

pub fn run(options: &Options) -> Result<String, String> {
    run_inner(options).map_err(|error| format!("{error:?}"))
}

fn run_inner(options: &Options) -> Res<String> {
    let cfg =
        Config::new(options.hidden, options.inter, options.seq).map_err(EngError::BadRequest)?;
    let host_start = host_sample();
    let max_layers = options.layers.iter().copied().max().unwrap_or(1);
    let mut kernels = MlxKernels::try_new()?;
    let setup_start = Instant::now();
    let mut layers = Vec::with_capacity(max_layers);
    for index in 0..max_layers {
        layers.push(build_layer(&mut kernels, &cfg, index as u32)?);
    }
    let setup_us = setup_start.elapsed().as_secs_f64() * 1.0e6;
    let x_bytes = activation_le(7, cfg.hidden as usize);
    let mut results = Vec::new();
    for &count in &options.layers {
        let slice = &layers[..count];
        for _ in 0..options.warmup {
            run_chain(
                &mut kernels,
                &cfg,
                slice,
                &x_bytes,
                &mut Recorder::default(),
            )?;
        }
        let mut rec = Recorder::default();
        let mut walls = Vec::with_capacity(options.reps);
        let mut hashes = Vec::new();
        let mut finite = true;
        let mut observed_release_counts = None;
        // Host sampling runs outside the timed repetitions: once before and once after the series.
        let before = host_sample();
        for _ in 0..options.reps {
            let chain = run_chain(&mut kernels, &cfg, slice, &x_bytes, &mut rec)?;
            walls.push(chain.wall_us);
            finite &= chain.finite;
            hashes.push(chain.output_sha256);
            if let Some(previous) = observed_release_counts {
                assert_eq!(
                    previous, chain.release_counts,
                    "release counts must stay fixed"
                );
            } else {
                observed_release_counts = Some(chain.release_counts);
            }
        }
        let release_counts = observed_release_counts.expect("repetitions are nonzero");
        let series_host = SeriesHost {
            before,
            after: host_sample(),
        };
        let summary = summarize(&walls);
        let calls = cfg.calls_per_chain(count);
        let per_op = OPS
            .iter()
            .filter_map(|name| rec.samples.get(name).map(|s| (*name, summarize(s))))
            .collect();
        results.push(ChainResult {
            layers: count,
            calls_per_chain: calls,
            repetitions: options.reps,
            per_layer_us_median: summary.median_us / count as f64,
            per_call_us_from_chain_median: summary.median_us / calls as f64,
            chain: summary,
            per_op,
            output_stable: hashes.windows(2).all(|w| w[0] == w[1]),
            output_sha256: hashes.first().cloned().unwrap_or_default(),
            output_finite: finite,
            l0_baseline_synchronizations_per_chain: l0_baseline_synchronizations_per_chain(count),
            release_counts,
            timed_release_samples_per_chain: count,
            series_host,
        });
    }
    let release_start = Instant::now();
    drop(layers);
    let release_us = release_start.elapsed().as_secs_f64() * 1.0e6;
    let adapter_start = Instant::now();
    drop(kernels);
    let adapter_release_us = adapter_start.elapsed().as_secs_f64() * 1.0e6;
    let report = Report {
        schema: "inference-engine.dispatch-cost.v1",
        engine_crate: "inference-engine native-mlx native-fault-qualification MlxKernels",
        config: cfg,
        warmup: options.warmup,
        repetitions: options.reps,
        setup_us,
        release_us,
        release_per_layer_us: release_us / max_layers as f64,
        adapter_release_us,
        host_start,
        host_end: host_sample(),
        timing: concat!(
            "Per-call Instant measurements surround public MlxKernels calls. Current resident methods build lazy graphs; readback has a separate timing series. This run does not qualify evaluation synchronization counts. ",
            "The direct release series times 14 resident drops per layer before readback and verifies 14 synchronization calls per layer. Thirteen are lazy outputs; the old x input is evaluated only in the first layer and lazy in later layers. The release_unread and release_unread_input series time the never-evaluated handles separately from release_evaluated_input. ",
            "The rope call releases two replaced, previously imported q/k inputs per layer; its two synchronization calls are counted, while their release time remains inside the rope call series. The L0 baseline synchronization formula is reported only as historical context. ",
            "The final x, output, and host input drops follow wall_us. Host sampling runs outside timed repetitions. Chain wall includes imports and final readback. release_us times resident layer release after the last chain, outside every chain."
        ),
        results,
    };
    serde_json::to_string_pretty(&report).map_err(|e| EngError::BadRequest(e.to_string()))
}
