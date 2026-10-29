//! Real-checkpoint evidence for device-side expert routing.
//!
//! Two ignored tests run on the Mac with the qualification counters.
//!
//! 1. `device_routing_picks_the_host_experts_on_the_real_checkpoint` runs the 5-token prompt of the
//!    frozen host-reference evidence and 31 greedy steps twice in `Record` mode: once with host
//!    routing and once with device routing, on the same tokens. It requires equal expert ids (rank
//!    order) in every layer of every step, equal tokens, and logits with cosine at least 0.99999 in
//!    the `f32` control. It also measures the synchronizations of each mode, and the forced retire
//!    flushes of the device run.
//!    The ids of an exact tie are equal too, because both rules take the lower index first. A
//!    difference at an exact tie is a failure of the gate.
//!
//!    The id rule follows the activation dtype. In the `f32` control the rule above is exact and
//!    unchanged. In BF16 the two routings rank the SAME probability node but renormalize the
//!    selected gates with different F32 summation orders, so their residual streams diverge in the
//!    last BF16 bit after the first layer and near-tied experts can swap rank or exchange the last
//!    expert of the top 8. The rule (`route::compare_top_k`): a set difference or a rank swap
//!    passes only when the host-row probability gap to the nearest exchanged (or swapped) expert is
//!    <= `BF16_TIE_EPSILON` (5e-3, an absolute probability gap); a missing exchange partner is a
//!    fault; ties at exactly equal probability must agree with the lower index; any other
//!    difference fails. The report prints the absolute and relative gap distributions of the
//!    mismatching cells. The co-conditions of the BF16 run: no cell differs at layers 0 to 2 of
//!    step 0 (`route::origin_fault`, the summation-order origin), no differing cell has a gap
//!    relative to max(p_a, p_b) above `BF16_RELATIVE_GAP_LIMIT` (`route::route_cell_deltas`), the
//!    fraction of differing cells stays at or under `BF16_DIFFERING_FRACTION_LIMIT`, and the
//!    absolute signed mean of the per-cell host-minus-device probability sums stays at or under
//!    `BF16_SIGNED_MEAN_LIMIT`. Greedy ids equal in all 36 steps is a gate of both dtypes, and the
//!    logits-cosine floor follows the dtype too: `MIN_COSINE` (0.99999) for `f32` and
//!    `BF16_MIN_COSINE` (0.997) for BF16, or `BF16_COMPILED_MIN_COSINE` (0.995) for BF16 with the
//!    compiled decode steps (`route::cosine_passes`), so a cosine under the floor of the run still
//!    fails the gate. Every failed check is collected into the report's `failures` list before the
//!    test fails.
//! 2. `device_routing_speed_table` times greedy decoding after a 16-token prompt (and, on request,
//!    after an 8,192-token prompt) with the prompt ids of the mlx-lm reference run.
//!
//! ```text
//! Q36_CKPT_DIR=<model_dir> Q36_OUT_DIR=<a path that does not exist yet> \
//!   cargo test --release --features native-mlx,native-fault-qualification --lib \
//!   family::qwen36::resident::route_parity::device_routing_picks -- --ignored --nocapture
//! Q36_CKPT_DIR=... Q36_PROMPT_IDS=<8,192 little-endian u32 ids> Q36_OUT_DIR=<new path> \
//!   cargo test --release --features native-mlx,native-fault-qualification --lib \
//!   family::qwen36::resident::route_parity::device_routing_speed -- --ignored --nocapture
//! ```
//!
//! `Q36_OUT_DIR` is required in both tests. `Q36_SPEED_8K=1` adds the 8,192-token row, which runs on
//! device routing only. Any other set value of `Q36_SPEED_8K` fails the test.
//!
//! `QWEN36_G2A_COMPILE=1` runs the DeltaNet and device-routed MoE decode steps of the device
//! routing block through `mlx_compile`d closures in the activation dtype of `Q36_ACTIVATION` (the
//! dtype is part of the compile cache key); host routing, the parity reference, never compiles.
//! Both reports then carry the `compile` row (the switch value and the `compile_counters`: traces,
//! cache hits, calls, per block the cache key and the seconds of the first calls) beside
//! `mlx_peak`, the active and cache bytes, and the retired-byte peak. Run each test with and
//! without the variable, in each activation dtype, and compare the rows.
//!
//! The speed test has four more knobs for profile runs. The defaults give the plain speed table.
//! A set value outside the range fails the test.
//!
//! | Variable | Values | Default |
//! | --- | --- | --- |
//! | `Q36_SPEED_TRIALS` | 1 to 3 timed trials per routing | 3 |
//! | `Q36_SPEED_TOKENS` | 2 to 256 new tokens per timed trial | 256 |
//! | `Q36_SPEED_ROUTING` | `device` runs the device routing block only | both blocks |
//! | `Q36_SPEED_GAP_MS` | 1 to 10,000 ms of idle time | none |
//!
//! `Q36_ACTIVATION` selects the decode activation dtype of the speed test through the product
//! lever: unset or `f32` is the table above and is unchanged, `bf16` runs BF16 activations with the
//! stored BF16 scales and the converted norms. Every row and the report carry `activation_dtype`,
//! and the decode rows carry the first 32 tokens. The routing test does not read it.
//!
//! `Q36_SPEED_GAP_MS` sleeps before the prompt phase and before the greedy phase of every timed
//! trial (the warm-up trial has no sleep). The sleeps are outside the timed regions. The GPU is
//! idle during them, so a trace shows each phase as its own cluster of GPU work. The idle GPU can
//! lower its clock, so a run with the gap is not a speed result.
//!
//! Both tests pin `config.json` and `model.safetensors.index.json` by SHA-256. The allocator peak
//! and the retired-byte peak are reset at the start of every routing block and recorded per block,
//! so the device run and the host run can be compared.

use sha2::{Digest, Sha256};

use super::decoder::{DEFAULT_CONTEXT_LIMIT, ResidentDecoder, StepMode};
use super::moe::Routing;
use super::real::{env_path, open, rss_kb};
use super::scale_knob::{hash_f32s, new_decoder};
use crate::{
    family::{
        qwen36::host_load::HostWeights,
        qwen36::settings::{Activation, activation_name},
        route::{
            compare_top_k, cosine_passes, differing_fraction_passes, origin_fault,
            route_cell_deltas, signed_mean_passes,
        },
        spec::QWEN3_6_35B_A3B,
    },
    kernels::{MlxKernels, QualificationTiming},
};

/// The prompt ids of the frozen host-reference evidence ("The capital of France is"), from
/// `host-ref.json`.
const PROMPT: [u32; 5] = [760, 6511, 314, 9338, 369];
/// Five prompt steps and 31 greedy steps.
const STEPS: usize = 36;
/// SHA-256 of the pinned `config.json`.
const CONFIG_SHA256: &str = "a822a9e48b0aafbe3144ec37d4fb067e178ed96615ce6e4420b3149893cc5767";
/// SHA-256 of the pinned `model.safetensors.index.json`.
const INDEX_SHA256: &str = "e359cd95565275b923fed8df019dd8e869d9af5dc4a621b4114ab7a8e6c534d1";
/// The logits-cosine floor of the `f32` control. The BF16 run of this gate uses
/// `BF16_MIN_COSINE` instead.
const MIN_COSINE: f64 = 0.99999;
/// The logits-cosine floor of the BF16 run of this gate. The same summation-order noise that moves
/// the router probabilities moves the logits: the reference BF16 gate run saw a minimum cosine of
/// 0.99864 over its 36 steps against the 0.99999 of the unchanged F32 control. 0.97 would sit on
/// the observed minimum, so the floor is 0.997, with a 1.64e-3 margin under it, and a logits cosine
/// under it still fails the gate (`route::cosine_passes`).
const BF16_MIN_COSINE: f64 = 0.997;
/// The logits-cosine floor of the BF16 run with the compiled decode steps. Compile fuses the
/// elementwise chains of the DeltaNet and MoE blocks, which is another valid BF16 rounding. The
/// compiled run saw a minimum cosine of 0.99697 over its 36 steps (at step 1) against the 0.99864
/// of the uncompiled run. The floor is 0.995, with a 1.97e-3 margin under the compiled minimum. A
/// cosine under it still fails the gate.
const BF16_COMPILED_MIN_COSINE: f64 = 0.995;
/// The near-tie epsilon of the BF16 routing rule, an absolute probability gap. The two routings of
/// one token renormalize their selected gates with different F32 summation orders (the host sums
/// the top 8 sequentially in `route::weights_for`; the device reduces them as one graph node), so
/// after the first layer their residual streams differ in the last BF16 bit and the router
/// probabilities of near-tied experts move by up to the 2.55e-3 maximum of the 600 mismatching
/// cells of 1,440 that the reference BF16 gate run saw (median 2.4e-4; greedy ids equal in all 36
/// steps, so the swaps are that ULP noise and not a bug). The number 5e-3 is chosen above the
/// observed maximum 2.55e-3 of 600 differing cells over 36 steps of a 5-token prompt, a thin
/// sample; the epsilon is to be re-derived from the 922-token prompt run and the longer-decode run.
/// Every observed near-tie passes it, and any differing cell with a gap above it fails the gate.
/// The report prints the gap distribution of every mismatching cell, so each Mac run re-checks the
/// split.
const BF16_TIE_EPSILON: f32 = 5e-3;
/// The derivation of `BF16_TIE_EPSILON` for the report field.
const BF16_TIE_EPSILON_BASIS: &str = "chosen above the observed maximum 2.55e-3 of 600 differing cells over 36 steps of a 5-token \
     prompt, a thin sample; the epsilon is to be re-derived from the 922-token prompt run and \
     the longer-decode run";
/// The relative-gap gate of one differing BF16 cell: the cell's largest `gap / max(p_a, p_b)`
/// (`route::route_cell_deltas`) may not exceed it. The reference evidence records absolute gaps
/// only, so 0.5 is the first gate: the report prints the observed maximum relative gap, and the
/// limit is to be tightened once a Mac run measures it.
const BF16_RELATIVE_GAP_LIMIT: f32 = 0.5;
/// The largest fraction of compared cells that may differ: 60% against the 42% (600 of 1,440)
/// that the reference BF16 gate run observed.
const BF16_DIFFERING_FRACTION_LIMIT: f64 = 0.6;
/// The largest absolute signed mean of the per-cell host-minus-device probability sums over the
/// differing cells: `BF16_TIE_EPSILON` / 5, 1e-3, so a one-sided bias beyond the near-tie noise
/// fails the gate.
const BF16_SIGNED_MEAN_LIMIT: f64 = 1e-3;
/// SHA-256 of the little-endian `u32` ids of the first 16 and the first 8,192 prompt tokens of the
/// mlx-lm reference run (`ref-ops.json` and `ref-kernel.json`, `prompt_sha`).
const REF_PROMPT_16_SHA256: &str =
    "ae6509043c143e3713c8f52183d16585717e7cda81fe5dfabba4bd2a588bb5a2";
const REF_PROMPT_8192_SHA256: &str =
    "f7941743c150d4b37710af78eecdff5488abbb39b8b54f04fc6dd42240a122bb";

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Cosine of two equal-length vectors, accumulated in `f64`.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut dot, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    dot / (aa.sqrt() * bb.sqrt())
}

/// The smaller of two cosines. A NaN in either input gives NaN, so a NaN cosine cannot hide
/// behind a larger value.
fn min_cosine(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

/// A report number. JSON has no NaN, so a NaN becomes the string `"NaN"` instead of `null`.
fn number_or_nan(value: f64) -> serde_json::Value {
    if value.is_nan() {
        serde_json::json!("NaN")
    } else {
        serde_json::json!(value)
    }
}

/// Refuses a checkpoint directory whose `model.safetensors.index.json` differs from the pin.
fn assert_pinned_index(ckpt_dir: &std::path::Path) {
    let path = ckpt_dir.join("model.safetensors.index.json");
    let index = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(
        sha256(&index),
        INDEX_SHA256,
        "model.safetensors.index.json differs from the pinned checkpoint"
    );
}

/// The ids as little-endian bytes, the layout that the reference `prompt_sha` hashes.
fn id_bytes(ids: &[u32]) -> Vec<u8> {
    ids.iter().flat_map(|id| id.to_le_bytes()).collect()
}

/// The required `Q36_OUT_DIR`. Panics when it is unset or when the path exists, before the long run.
fn out_dir() -> String {
    let out = std::env::var("Q36_OUT_DIR").unwrap_or_else(|_| panic!("set Q36_OUT_DIR"));
    assert!(
        std::fs::symlink_metadata(&out).is_err(),
        "Q36_OUT_DIR must not exist yet: {out}"
    );
    out
}

/// Reads `Q36_SPEED_8K`: unset is off, `1` is on, any other value panics.
fn speed_8k_requested() -> bool {
    match std::env::var("Q36_SPEED_8K") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(value) if value == "1" => true,
        other => panic!("Q36_SPEED_8K must be unset or 1, got {other:?}"),
    }
}

/// The value of `raw` as a decimal integer in `min..=max`, or the reason that it is not one.
fn parse_count(name: &str, raw: &str, min: usize, max: usize) -> Result<usize, String> {
    match raw.parse::<usize>() {
        Ok(value) if (min..=max).contains(&value) => Ok(value),
        _ => Err(format!(
            "{name} must be a decimal integer from {min} to {max}, got {raw:?}"
        )),
    }
}

/// Reads a count knob: unset gives `default`, any value outside `min..=max` panics.
fn count_knob(name: &str, min: usize, max: usize, default: usize) -> usize {
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => default,
        Ok(raw) => parse_count(name, &raw, min, max).unwrap_or_else(|e| panic!("{e}")),
        Err(e) => panic!("{name}: {e}"),
    }
}

/// Reads `Q36_SPEED_ROUTING`: unset runs the device block and the host block, `device` runs the
/// device block only, any other value panics.
fn speed_routings() -> Vec<Routing> {
    match std::env::var("Q36_SPEED_ROUTING") {
        Err(std::env::VarError::NotPresent) => vec![Routing::Device, Routing::Host],
        Ok(value) if value == "device" => vec![Routing::Device],
        other => panic!("Q36_SPEED_ROUTING must be unset or device, got {other:?}"),
    }
}

/// Writes `report` to `<out>/<name>`, creating `out`.
fn write_report(out: &str, name: &str, report: &serde_json::Value) {
    println!("{}", serde_json::to_string_pretty(report).unwrap());
    std::fs::create_dir(out).expect("Q36_OUT_DIR must not exist yet");
    std::fs::write(
        std::path::Path::new(out).join(name),
        serde_json::to_vec_pretty(report).unwrap(),
    )
    .unwrap();
}

/// The allocator peak and the retired-byte peak of one block, since the last `reset_peaks`.
#[derive(Clone, Copy, Default)]
struct Peaks {
    mlx_peak: u64,
    peak_retired_bytes: u64,
}

impl Peaks {
    /// Starts a block: both peaks restart from the current level.
    fn reset(kernels: &mut MlxKernels) {
        kernels.reset_peak_memory().unwrap();
        kernels.qualification_reset_retire_peak();
    }

    /// Reads both peaks of the block that `reset` started.
    fn read(kernels: &MlxKernels) -> Self {
        Self {
            mlx_peak: kernels.memory_stats().unwrap().peak,
            peak_retired_bytes: kernels.qualification_retire_stats().1,
        }
    }

    /// The larger of each peak of two blocks.
    fn max(self, other: Self) -> Self {
        Self {
            mlx_peak: self.mlx_peak.max(other.mlx_peak),
            peak_retired_bytes: self.peak_retired_bytes.max(other.peak_retired_bytes),
        }
    }
}

#[test]
#[ignore = "needs the real checkpoint on the Mac; run with --ignored"]
fn device_routing_picks_the_host_experts_on_the_real_checkpoint() {
    let _serial = crate::native_test_lock();
    let ckpt_dir = env_path("Q36_CKPT_DIR");
    let out = out_dir();
    assert_pinned_index(&ckpt_dir);
    let real = open(&ckpt_dir);
    assert_eq!(
        sha256(&real.config),
        CONFIG_SHA256,
        "config.json differs from the pinned checkpoint"
    );
    let host = HostWeights::load(&real.ckpt, &QWEN3_6_35B_A3B, &real.config).unwrap();
    let (mut decoder, activation) =
        new_decoder(MlxKernels::try_new().unwrap(), &real.ckpt, host, 256);
    decoder.kernels.reset_compile_counters();
    decoder.kernels.clear_compile_cache().unwrap();
    let mut host_state = decoder.zero_state().unwrap();
    let mut device_state = decoder.zero_state().unwrap();
    let mut failures = Vec::new();
    let (mut compared, mut min_logit_cosine, mut tokens) = (0usize, 1.0f64, Vec::new());
    let flushes_before = decoder.kernels.qualification_retire_stats().0;
    // The peaks restart before every step, so each step reads the peaks of its own routing. The
    // largest step peak of each routing is the peak of that routing's block.
    let (mut host_peaks, mut device_peaks) = (Peaks::default(), Peaks::default());
    // The identity of two runs (`Q36_SCALE_DTYPE` bf16 against f32) is the equal hash of the
    // logits of all 36 steps, per routing, and the equal greedy ids.
    let (mut host_hash, mut device_hash) = (Sha256::new(), Sha256::new());
    let mut fed = PROMPT[0];
    // The BF16 near-tie bookkeeping: the gap that each mismatching cell bridged, and the
    // relative gaps and the signed host-minus-device sums of the same cells.
    let mut cell_gaps: Vec<f64> = Vec::new();
    let mut cell_relative_gaps: Vec<f64> = Vec::new();
    let mut cell_signed_gaps: Vec<f64> = Vec::new();
    // Greedy ids equal in all 36 steps is a gate of both dtypes.
    let mut greedy_ids_equal = true;
    // The logits-cosine floor of this run: the exact floor for `f32`, 0.997 for BF16, and 0.995 for
    // BF16 with the compiled decode steps.
    let (cosine_floor, floor_basis) = if activation == Activation::F32 {
        (MIN_COSINE, "the exact gate of the f32 control")
    } else if super::compiled::requested() {
        (
            BF16_COMPILED_MIN_COSINE,
            "0.995, a 1.97e-3 margin under the observed BF16 minimum 0.99697 of the 36 steps \
             with the compiled decode steps",
        )
    } else {
        (
            BF16_MIN_COSINE,
            "0.997, a 1.64e-3 margin under the observed BF16 minimum 0.99864 of the 36 steps",
        )
    };
    for step in 0..STEPS {
        if let Some(&prompt_token) = PROMPT.get(step) {
            fed = prompt_token;
        }
        decoder.set_routing(Routing::Host);
        Peaks::reset(&mut decoder.kernels);
        let want = decoder
            .step(&mut host_state, fed, StepMode::Record)
            .unwrap();
        host_peaks = host_peaks.max(Peaks::read(&decoder.kernels));
        decoder.set_routing(Routing::Device);
        Peaks::reset(&mut decoder.kernels);
        let got = decoder
            .step(&mut device_state, fed, StepMode::Record)
            .unwrap();
        device_peaks = device_peaks.max(Peaks::read(&decoder.kernels));
        hash_f32s(&mut host_hash, want.logits.as_ref().unwrap());
        hash_f32s(&mut device_hash, got.logits.as_ref().unwrap());
        let (want_routes, got_routes) = (want.routes.unwrap(), got.routes.unwrap());
        // The shape checks are collected, not asserted, so every failed check of the run lands
        // in the report's `failures` list before the test fails.
        if want_routes.len() != 40 {
            failures.push(format!("step {step}: {} host layers", want_routes.len()));
        }
        if got_routes.len() != 40 {
            failures.push(format!("step {step}: {} device layers", got_routes.len()));
        }
        // The host probability rows: the oracle the near-tie gaps are measured on. `Record` with
        // host routing holds one row per layer.
        let want_probs = want.probs.unwrap();
        if want_probs.len() != 40 {
            failures.push(format!(
                "step {step}: {} host router rows",
                want_probs.len()
            ));
        }
        for (layer, ((a, b), probs)) in want_routes
            .iter()
            .zip(&got_routes)
            .zip(&want_probs)
            .enumerate()
        {
            if a.len() != 8 {
                failures.push(format!("step {step} layer {layer}: {} host top-k", a.len()));
                continue;
            }
            if probs.len() != 256 {
                failures.push(format!(
                    "step {step} layer {layer}: {} router width",
                    probs.len()
                ));
                continue;
            }
            compared += 1;
            if activation == Activation::F32 {
                // The F32 control keeps the exact rule: the ids match in rank order, and an exact
                // tie already agrees because both rules take the lower index.
                if a != b {
                    failures.push(format!(
                        "step {step} layer {layer}: host {a:?}, device {b:?}"
                    ));
                }
                continue;
            }
            // The BF16 rule: a set difference or a rank swap passes only when the host-row
            // probability gap to the nearest exchanged (or swapped) expert is <= BF16_TIE_EPSILON;
            // a missing exchange partner is a fault; ties at exactly equal probability must agree
            // with the lower index; any other difference fails.
            let comparison = compare_top_k(a, b, probs, BF16_TIE_EPSILON);
            if a != b {
                cell_gaps.push(f64::from(comparison.max_gap));
                let deltas = route_cell_deltas(a, b, probs);
                cell_relative_gaps.push(f64::from(deltas.max_relative_gap));
                cell_signed_gaps.push(f64::from(deltas.signed_gap));
                if !(deltas.max_relative_gap.is_finite()
                    && deltas.max_relative_gap <= BF16_RELATIVE_GAP_LIMIT)
                {
                    failures.push(format!(
                        "step {step} layer {layer}: relative gap {}",
                        deltas.max_relative_gap
                    ));
                }
                if origin_fault(step, layer) {
                    failures.push(format!(
                        "step {step} layer {layer}: a cell differs within layers 0 to 2 of \
                         step 0, before the summation-order origin"
                    ));
                }
            }
            for fault in &comparison.faults {
                failures.push(match fault.rank {
                    Some(rank) => format!(
                        "step {step} layer {layer}: rank {rank} holds host {} and device {}, \
                         probability gap {:?}",
                        fault.oracle, fault.routed, fault.gap
                    ),
                    None if fault.oracle == fault.routed => format!(
                        "step {step} layer {layer}: expert {} appears on one side with no \
                         exchanged expert in the probability row",
                        fault.oracle
                    ),
                    None => format!(
                        "step {step} layer {layer}: host {} against device {}, set difference \
                         with probability gap {:?}",
                        fault.oracle, fault.routed, fault.gap
                    ),
                });
            }
        }
        if want.token != got.token {
            failures.push(format!(
                "step {step}: host token {:?}, device token {:?}",
                want.token, got.token
            ));
            greedy_ids_equal = false;
        }
        let cos = cosine(want.logits.as_ref().unwrap(), got.logits.as_ref().unwrap());
        min_logit_cosine = min_cosine(min_logit_cosine, cos);
        if !cosine_passes(cos, cosine_floor) {
            failures.push(format!("step {step}: logits cosine {cos}"));
        }
        fed = want.token.unwrap();
        tokens.push(fed);
    }
    let record_flushes = decoder.kernels.qualification_retire_stats().0 - flushes_before;
    if record_flushes != 0 {
        failures.push(format!(
            "{record_flushes} forced retire flushes during the recorded steps"
        ));
    }
    // The synchronization budget of every mode, on fresh states.
    let mut budgets = Vec::new();
    let modes = [
        StepMode::Prompt,
        StepMode::Generate,
        StepMode::Logits,
        StepMode::Record,
    ];
    for (routing, expected) in [
        (Routing::Device, [2u64, 3, 4, 6]),
        (Routing::Host, [80, 83, 84, 85]),
    ] {
        decoder.set_routing(routing);
        for (mode, want) in modes.into_iter().zip(expected) {
            let mut state = decoder.zero_state().unwrap();
            let before = decoder.kernels.qualification_sync_count();
            Peaks::reset(&mut decoder.kernels);
            decoder.step(&mut state, PROMPT[0], mode).unwrap();
            let peaks = Peaks::read(&decoder.kernels);
            let got = decoder.kernels.qualification_sync_count() - before;
            if got != want {
                failures.push(format!(
                    "{routing:?} {mode:?}: {got} synchronizations, expected {want}"
                ));
            }
            budgets.push(serde_json::json!({
                "routing": format!("{routing:?}"), "mode": format!("{mode:?}"),
                "synchronizations": got, "expected": want,
                "mlx_peak": peaks.mlx_peak, "peak_retired_bytes": peaks.peak_retired_bytes,
            }));
        }
    }
    // The flush count never resets, so the delta covers the recorded steps and the budget steps.
    let flushes = decoder.kernels.qualification_retire_stats().0 - flushes_before;
    if flushes != record_flushes {
        failures.push(format!(
            "{} forced retire flushes during the budget steps",
            flushes - record_flushes
        ));
    }
    let stats = decoder.kernels.memory_stats().unwrap();
    // The gap distribution of the mismatching cells: the evidence behind the epsilon.
    let (gap_max, gap_median) = if cell_gaps.is_empty() {
        (0.0, 0.0)
    } else {
        (
            cell_gaps.iter().copied().fold(0.0, f64::max),
            median(cell_gaps.clone()),
        )
    };
    let mut gap_buckets = std::collections::BTreeMap::new();
    for gap in &cell_gaps {
        *gap_buckets.entry(gap_bucket(*gap)).or_insert(0u64) += 1;
    }
    // The relative-gap distribution and the bias aggregates of the same cells: the
    // evidence behind the relative gate, the differing-fraction gate, and the signed-mean gate.
    let (relative_max, relative_median) = if cell_relative_gaps.is_empty() {
        (0.0, 0.0)
    } else {
        (
            cell_relative_gaps.iter().copied().fold(0.0, f64::max),
            median(cell_relative_gaps.clone()),
        )
    };
    let mut relative_buckets = std::collections::BTreeMap::new();
    for relative in &cell_relative_gaps {
        *relative_buckets
            .entry(relative_bucket(*relative))
            .or_insert(0u64) += 1;
    }
    let differing_fraction = if compared == 0 {
        0.0
    } else {
        cell_gaps.len() as f64 / compared as f64
    };
    let signed_mean = if cell_signed_gaps.is_empty() {
        0.0
    } else {
        cell_signed_gaps.iter().sum::<f64>() / cell_signed_gaps.len() as f64
    };
    if !differing_fraction_passes(compared, cell_gaps.len(), BF16_DIFFERING_FRACTION_LIMIT) {
        failures.push(format!(
            "{} of {compared} cells differ, above the fraction gate \
             {BF16_DIFFERING_FRACTION_LIMIT}",
            cell_gaps.len()
        ));
    }
    if !signed_mean_passes(signed_mean, BF16_SIGNED_MEAN_LIMIT) {
        failures.push(format!(
            "signed mean {signed_mean} of the per-cell host-minus-device probability sums, \
             above the gate {BF16_SIGNED_MEAN_LIMIT}"
        ));
    }
    let report = serde_json::json!({
        "steps": STEPS, "layers_compared": compared,
        "min_logits_cosine": number_or_nan(min_logit_cosine),
        "min_logits_cosine_floor": cosine_floor,
        "min_logits_cosine_floor_basis": floor_basis,
        "greedy_ids": tokens, "failures": failures, "budgets": budgets,
        "route_rule": if activation == Activation::F32 {
            "f32-exact"
        } else {
            "bf16-set-difference-or-rank-swap-within-tie-epsilon"
        },
        "bf16_tie_epsilon": BF16_TIE_EPSILON,
        "bf16_tie_epsilon_basis": BF16_TIE_EPSILON_BASIS,
        "cells_differing": cell_gaps.len(),
        "cell_gap_max": number_or_nan(gap_max),
        "cell_gap_median": number_or_nan(gap_median),
        "cell_gap_buckets": gap_buckets,
        "cell_relative_gap_max": number_or_nan(relative_max),
        "cell_relative_gap_median": number_or_nan(relative_median),
        "cell_relative_gap_buckets": relative_buckets,
        "cell_relative_gap_gate": BF16_RELATIVE_GAP_LIMIT,
        "cells_differing_fraction": differing_fraction,
        "cells_differing_fraction_gate": BF16_DIFFERING_FRACTION_LIMIT,
        "cell_signed_gap_mean": number_or_nan(signed_mean),
        "cell_signed_gap_mean_gate": BF16_SIGNED_MEAN_LIMIT,
        "greedy_ids_equal_in_all_steps": greedy_ids_equal,
        "forced_retire_flushes": flushes, "forced_retire_flushes_recorded_steps": record_flushes,
        "peak_retired_bytes": host_peaks.peak_retired_bytes.max(device_peaks.peak_retired_bytes),
        "mlx_active": stats.active, "mlx_cache": stats.cache,
        "mlx_peak": host_peaks.mlx_peak.max(device_peaks.mlx_peak),
        "host_peaks": {
            "mlx_peak": host_peaks.mlx_peak, "peak_retired_bytes": host_peaks.peak_retired_bytes,
        },
        "device_peaks": {
            "mlx_peak": device_peaks.mlx_peak,
            "peak_retired_bytes": device_peaks.peak_retired_bytes,
        },
        "rss_kb": rss_kb(),
        "activation_dtype": activation_name(activation),
        "host_logits_sha256": format!("{:x}", host_hash.finalize()),
        "device_logits_sha256": format!("{:x}", device_hash.finalize()),
        "compile": super::compiled::report(&decoder.kernels),
    });
    write_report(&out, "route-parity.json", &report);
    assert_eq!(
        compared,
        STEPS * 40,
        "every layer of every step was compared"
    );
    assert!(failures.is_empty(), "routing gate failed: {failures:#?}");
}

/// One timed decode: the prompt steps, then `new_tokens` greedy tokens.
struct Trial {
    /// Greedy tokens per second after the first generated token, `(new_tokens - 1) / seconds`.
    generation_tps: f64,
    /// Prompt tokens per second for the prompt steps, including the one that yields the first token.
    prompt_tps: f64,
    tokens: Vec<u32>,
    /// Imports, readbacks, and evaluates of `zero_state` and the prompt steps.
    prompt_timing: QualificationTiming,
    /// Imports, readbacks, and evaluates of the greedy steps after the first generated token.
    decode_timing: QualificationTiming,
}

/// The sum of the counts and times of `timings`.
fn sum_timing(timings: impl Iterator<Item = QualificationTiming>) -> QualificationTiming {
    timings.fold(QualificationTiming::default(), |a, b| QualificationTiming {
        import_calls: a.import_calls + b.import_calls,
        import_ns: a.import_ns + b.import_ns,
        readback_calls: a.readback_calls + b.readback_calls,
        readback_ns: a.readback_ns + b.readback_ns,
        evaluate_calls: a.evaluate_calls + b.evaluate_calls,
        evaluate_ns: a.evaluate_ns + b.evaluate_ns,
    })
}

/// The timing keys of a report row, with `prefix` before each key.
fn timing_fields(row: &mut serde_json::Value, prefix: &str, timing: QualificationTiming) {
    let fields = [
        ("import_calls", timing.import_calls.into()),
        ("readback_calls", timing.readback_calls.into()),
        ("evaluate_calls", timing.evaluate_calls.into()),
        ("evaluate_ms", (timing.evaluate_ns as f64 / 1e6).into()),
        ("readback_ms", (timing.readback_ns as f64 / 1e6).into()),
        ("import_ms", (timing.import_ns as f64 / 1e6).into()),
    ];
    for (key, value) in fields {
        row[format!("{prefix}{key}")] = value;
    }
}

/// Runs `prompt` (all but the last token as `Prompt` steps, the last as `Generate`) and then
/// `new_tokens - 1` more `Generate` steps on a fresh state. It sleeps `gap` before the prompt phase
/// and between the two phases, outside both timed regions.
fn trial(
    decoder: &mut ResidentDecoder<'_>,
    prompt: &[u32],
    new_tokens: usize,
    gap: std::time::Duration,
) -> Trial {
    let (&last, head) = prompt.split_last().unwrap();
    if !gap.is_zero() {
        std::thread::sleep(gap);
    }
    decoder.kernels.qualification_reset_timing();
    let mut state = decoder.zero_state().unwrap();
    let started = std::time::Instant::now();
    for &token in head {
        decoder.step(&mut state, token, StepMode::Prompt).unwrap();
    }
    let first = decoder
        .step(&mut state, last, StepMode::Generate)
        .unwrap()
        .token
        .unwrap();
    let prompt_seconds = started.elapsed().as_secs_f64();
    let prompt_timing = decoder.kernels.qualification_timing();
    decoder.kernels.qualification_reset_timing();
    if !gap.is_zero() {
        std::thread::sleep(gap);
    }
    let mut tokens = vec![first];
    let started = std::time::Instant::now();
    for _ in 1..new_tokens {
        let next = decoder
            .step(&mut state, *tokens.last().unwrap(), StepMode::Generate)
            .unwrap()
            .token
            .unwrap();
        tokens.push(next);
    }
    let seconds = started.elapsed().as_secs_f64();
    let decode_timing = decoder.kernels.qualification_timing();
    Trial {
        generation_tps: (new_tokens - 1) as f64 / seconds,
        prompt_tps: prompt.len() as f64 / prompt_seconds,
        tokens,
        prompt_timing,
        decode_timing,
    }
}

/// The median of the values.
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// The bucket of one mismatching cell's gap, for the report distribution. The boundaries sit at
/// 1e-4, 1e-3, and 1e-2 around `BF16_TIE_EPSILON` (5e-3, inside `le_1e-2`), so the report shows
/// whether the tolerated gaps and the failing gaps separate cleanly.
fn gap_bucket(gap: f64) -> &'static str {
    if gap <= 1e-4 {
        "le_1e-4"
    } else if gap <= 1e-3 {
        "le_1e-3"
    } else if gap <= 1e-2 {
        "le_1e-2"
    } else {
        "gt_1e-2"
    }
}

/// The bucket of one differing cell's relative gap, for the report distribution. The boundaries sit
/// at 1e-2, 1e-1, and 5e-1 around `BF16_RELATIVE_GAP_LIMIT` (0.5, inside `le_5e-1`), so the report
/// shows whether the tolerated relative gaps and the failing ones separate cleanly.
fn relative_bucket(relative: f64) -> &'static str {
    if relative <= 1e-2 {
        "le_1e-2"
    } else if relative <= 1e-1 {
        "le_1e-1"
    } else if relative <= 5e-1 {
        "le_5e-1"
    } else {
        "gt_5e-1"
    }
}

#[test]
#[ignore = "needs the real checkpoint and the reference prompt ids on the Mac; run with --ignored"]
fn device_routing_speed_table() {
    let _serial = crate::native_test_lock();
    let ckpt_dir = env_path("Q36_CKPT_DIR");
    let out = out_dir();
    // The ids of `Q36_PROMPT_IDS`: the one shared reader of the little-endian `u32` dump
    // format, `decode_timing::read_prompt_ids` (whose pure parse has its own unit test).
    let ids = super::decode_timing::read_prompt_ids();
    assert!(ids.len() >= 16, "Q36_PROMPT_IDS holds at least 16 ids");
    assert_eq!(
        sha256(&id_bytes(&ids[..16])),
        REF_PROMPT_16_SHA256,
        "the first 16 ids differ from the reference prompt"
    );
    let with_8k = speed_8k_requested();
    if with_8k {
        assert!(ids.len() >= 8192, "Q36_PROMPT_IDS holds 8,192 ids");
        assert_eq!(
            sha256(&id_bytes(&ids[..8192])),
            REF_PROMPT_8192_SHA256,
            "the first 8,192 ids differ from the reference prompt"
        );
    }
    let trial_count = count_knob("Q36_SPEED_TRIALS", 1, 3, 3);
    let new_tokens = count_knob("Q36_SPEED_TOKENS", 2, 256, 256);
    let routings = speed_routings();
    let gap_ms = count_knob("Q36_SPEED_GAP_MS", 1, 10_000, 0);
    let gap = std::time::Duration::from_millis(gap_ms as u64);
    assert_pinned_index(&ckpt_dir);
    let real = open(&ckpt_dir);
    assert_eq!(sha256(&real.config), CONFIG_SHA256, "config.json differs");
    let host = HostWeights::load(&real.ckpt, &QWEN3_6_35B_A3B, &real.config).unwrap();
    let (mut decoder, act) = new_decoder(
        MlxKernels::try_new().unwrap(),
        &real.ckpt,
        host,
        DEFAULT_CONTEXT_LIMIT,
    );
    decoder.kernels.reset_compile_counters();
    decoder.kernels.clear_compile_cache().unwrap();
    // The flush count never resets, so each block reports a delta from this baseline.
    let flushes_before = decoder.kernels.qualification_retire_stats().0;
    let mut rows = Vec::new();
    let mut device_tokens = Vec::new();
    for routing in routings {
        decoder.set_routing(routing);
        Peaks::reset(&mut decoder.kernels);
        let block_flushes_before = decoder.kernels.qualification_retire_stats().0;
        trial(&mut decoder, &ids[..16], 8, std::time::Duration::ZERO);
        let trials: Vec<Trial> = (0..trial_count)
            .map(|_| trial(&mut decoder, &ids[..16], new_tokens, gap))
            .collect();
        let peaks = Peaks::read(&decoder.kernels);
        if routing == Routing::Device {
            device_tokens = trials[0].tokens.clone();
        }
        let mut row = serde_json::json!({
            "case": "decode_16", "routing": format!("{routing:?}"),
            "activation_dtype": activation_name(act),
            "first_tokens": trials[0].tokens.iter().take(32).collect::<Vec<_>>(),
            "trials": trial_count, "new_tokens": new_tokens, "gap_ms": gap_ms,
            "generation_tps": trials.iter().map(|t| t.generation_tps).collect::<Vec<_>>(),
            "median_generation_tps": median(trials.iter().map(|t| t.generation_tps).collect()),
            "prompt_tps": trials.iter().map(|t| t.prompt_tps).collect::<Vec<_>>(),
            "tokens_sha256": sha256(&id_bytes(&trials[0].tokens)),
            "same_tokens_in_all_trials": trials.iter().all(|t| t.tokens == trials[0].tokens),
            "same_tokens_as_device": routing == Routing::Device || trials[0].tokens == device_tokens,
            "mlx_peak": peaks.mlx_peak, "peak_retired_bytes": peaks.peak_retired_bytes,
            "forced_retire_flushes":
                decoder.kernels.qualification_retire_stats().0 - block_flushes_before,
        });
        // The unprefixed timing keys cover the greedy steps of the three timed trials only.
        timing_fields(
            &mut row,
            "",
            sum_timing(trials.iter().map(|t| t.decode_timing)),
        );
        timing_fields(
            &mut row,
            "prompt_",
            sum_timing(trials.iter().map(|t| t.prompt_timing)),
        );
        rows.push(row);
    }
    if with_8k {
        decoder.set_routing(Routing::Device);
        Peaks::reset(&mut decoder.kernels);
        let block_flushes_before = decoder.kernels.qualification_retire_stats().0;
        let one = trial(&mut decoder, &ids[..8192], 128, std::time::Duration::ZERO);
        let peaks = Peaks::read(&decoder.kernels);
        let mut row = serde_json::json!({
            "case": "decode_8192", "routing": "Device",
            "activation_dtype": activation_name(act),
            "first_tokens": one.tokens.iter().take(32).collect::<Vec<_>>(),
            "generation_tps": one.generation_tps, "prompt_tps": one.prompt_tps,
            "tokens_sha256": sha256(&id_bytes(&one.tokens)),
            "mlx_peak": peaks.mlx_peak, "peak_retired_bytes": peaks.peak_retired_bytes,
            "forced_retire_flushes":
                decoder.kernels.qualification_retire_stats().0 - block_flushes_before,
        });
        timing_fields(&mut row, "", one.decode_timing);
        timing_fields(&mut row, "prompt_", one.prompt_timing);
        rows.push(row);
    }
    let stats = decoder.kernels.memory_stats().unwrap();
    let flushes = decoder.kernels.qualification_retire_stats().0 - flushes_before;
    let peak_retired = rows
        .iter()
        .filter_map(|row| row["peak_retired_bytes"].as_u64())
        .max()
        .unwrap_or(0);
    let peak = rows
        .iter()
        .filter_map(|row| row["mlx_peak"].as_u64())
        .max()
        .unwrap_or(0);
    let report = serde_json::json!({
        "activation_dtype": activation_name(act),
        "rows": rows, "forced_retire_flushes": flushes, "peak_retired_bytes": peak_retired,
        "mlx_active": stats.active, "mlx_cache": stats.cache, "mlx_peak": peak, "rss_kb": rss_kb(),
        "compile": super::compiled::report(&decoder.kernels),
        "reference": {
            "decode_16": {"ops": 101.7, "kernel": 117.5},
            "decode_8192": {"ops": 92.3, "kernel": 106.2},
            "source": "mlx-lm reference run, mlx 0.30.6, mlx-lm 0.31.0",
        },
    });
    write_report(&out, "speed.json", &report);
}

#[test]
fn a_count_knob_accepts_its_range_and_rejects_every_other_value() {
    // The range 2 to 256 of Q36_SPEED_TOKENS: both bounds pass.
    assert_eq!(parse_count("Q36_SPEED_TOKENS", "2", 2, 256), Ok(2));
    assert_eq!(parse_count("Q36_SPEED_TOKENS", "256", 2, 256), Ok(256));
    for bad in [
        "1",
        "257",
        "0",
        "",
        "-3",
        "12x",
        " 8",
        "1e2",
        "99999999999999999999999",
    ] {
        let err = parse_count("Q36_SPEED_TOKENS", bad, 2, 256).unwrap_err();
        assert_eq!(
            err,
            format!("Q36_SPEED_TOKENS must be a decimal integer from 2 to 256, got {bad:?}")
        );
    }
}
