//! The per-layer parity gate of the resident decoder on the real checkpoint.
//!
//! The ignored test runs the resident decoder on the 5-token prompt of the frozen host-reference
//! evidence and compares it with that evidence: the residual stream after every layer for every
//! prompt token, the last-row logits and their top 5, and 32 greedy tokens. The reference is the
//! host forward, which shares no kernel with MLX. mlx-lm is not involved.
//!
//! `QWEN36_G2A_COMPILE=1` runs the DeltaNet and device-routed MoE decode steps through `mlx_compile`d
//! closures in the activation dtype of `Q36_ACTIVATION` (unset or `f32` is the F32 control,
//! `bf16` runs the BF16 decoder — the dtype is part of the compile cache key) and adds the
//! `compile` row to the report: the switch value and the `compile_counters` (traces, cache hits,
//! calls, per block the cache key and the seconds of the first calls). Run the gate twice, once
//! without the variable (the control) and once with it, in each activation dtype: bit identity of
//! the two `layers_sha256` values is the first expectation, and the cosine gates below are the
//! stated fallback tolerance when fusion changes the numerics.
//!
//! ```text
//! Q36_CKPT_DIR=<model_dir> Q36_REF_DIR=<copy of the evidence directory ref/> \
//!   Q36_OUT_DIR=<a path that does not exist yet> \
//!   cargo test --release --features native-mlx,native-fault-qualification --lib \
//!   family::qwen36::resident::parity -- --ignored --nocapture
//! ```
//!
//! `Q36_ACTIVATION=bf16` runs the decode activations in BF16 through the product
//! `activation_dtype` lever: the stored BF16 scales and the converted norms. Unset or `f32` is the
//! F32 control above and its gates are unchanged. With BF16 the 0.99999 per-layer rule stays
//! report-only, but the run asserts its floor: no cosine is NaN, the minimum layer cosine is at
//! least 0.95, and it does not regress by more than 0.01 from the anchor 0.9799 - a report-only
//! result is not allowed to ratchet. The report names the dtype in `activation_dtype`.

use sha2::{Digest, Sha256};

use super::decoder::StepMode;
use super::real::{env_path, open, rss_kb};
use super::scale_knob::{hash_f32s, new_decoder};
use super::within_mode::{
    BF16_LAYER_COSINE_FLOOR, BF16_LAYER_COSINE_REGRESSION, D1_BF16_MIN_LAYER_COSINE, passes,
};
use crate::{
    family::{
        qwen36::host_load::HostWeights,
        qwen36::settings::{Activation, activation_name},
        route::keeps_floored_anchor,
        spec::QWEN3_6_35B_A3B,
    },
    kernels::MlxKernels,
};

/// SHA-256 of the frozen evidence files (the `SHA256SUMS` of the evidence directory).
pub(super) const HOST_REF_JSON_SHA256: &str =
    "27acedbe149a50eab685299f3faa4202de0558689b681f383c19d2e16234a076";
pub(super) const LAYERS_SHA256: &str =
    "5bc310d7accdc56fbdc464d86e50fc3e655773d338e173d5d964bbbf6b7aef02";
pub(super) const LOGITS_SHA256: &str =
    "6b4973e208ec44aa91efb9d61ae72de941a3ac775e82b771a275227378052044";
/// SHA-256 of the pinned `config.json`.
pub(super) const CONFIG_SHA256: &str =
    "a822a9e48b0aafbe3144ec37d4fb067e178ed96615ce6e4420b3149893cc5767";

/// The smaller of two cosines. A NaN in either input gives NaN, so a NaN cosine cannot hide
/// behind a larger value.
pub(super) fn min_cosine(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

pub(super) fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(super) fn read_checked(dir: &std::path::Path, name: &str, want: &str) -> Vec<u8> {
    let bytes = std::fs::read(dir.join(name)).unwrap();
    assert_eq!(
        sha256(&bytes),
        want,
        "evidence file {name} differs from the frozen hash"
    );
    bytes
}

pub(super) fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Cosine of two equal-length vectors, accumulated in `f64`.
pub(super) fn cosine(a: &[f32], b: &[f32]) -> f64 {
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

/// The five largest logits as token ids, the lowest token first on ties (the host reference's
/// rule).
pub(super) fn top5(logits: &[f32]) -> Vec<u32> {
    let mut ranked: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, *v))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked.into_iter().take(5).map(|r| r.0).collect()
}

#[test]
#[ignore = "needs the real checkpoint and the frozen host-reference evidence on the Mac"]
fn real_checkpoint_matches_the_host_reference() {
    let _serial = crate::native_test_lock();
    let (ckpt_dir, ref_dir) = (env_path("Q36_CKPT_DIR"), env_path("Q36_REF_DIR"));
    // Refuse an existing output path before the long run. `create_dir` refuses it again at the end.
    let out_dir = std::env::var("Q36_OUT_DIR").ok();
    if let Some(out) = &out_dir {
        assert!(
            std::fs::symlink_metadata(out).is_err(),
            "Q36_OUT_DIR must not exist yet: {out}"
        );
    }
    let report_json = read_checked(&ref_dir, "host-ref.json", HOST_REF_JSON_SHA256);
    let layers_ref = f32s(&read_checked(&ref_dir, "layers.f32", LAYERS_SHA256));
    let logits_ref = f32s(&read_checked(&ref_dir, "logits-last.f32", LOGITS_SHA256));
    let reference: serde_json::Value = serde_json::from_slice(&report_json).unwrap();
    let ids: Vec<u32> = serde_json::from_value(reference["prompt_ids"].clone()).unwrap();
    let greedy: Vec<u32> = serde_json::from_value(reference["greedy_ids"].clone()).unwrap();
    let want_top5: Vec<u32> = reference["steps"][ids.len() - 1]["top5"]
        .as_array()
        .unwrap()
        .iter()
        .map(|pair| pair[0].as_u64().unwrap() as u32)
        .collect();
    assert_eq!(
        (ids.len(), greedy.len()),
        (5, 32),
        "the frozen prompt and greedy lengths"
    );
    let (layers, hidden, vocab) = (40usize, 2048usize, 248_320usize);
    assert_eq!(layers_ref.len(), ids.len() * layers * hidden);
    assert_eq!(logits_ref.len(), vocab);

    let recorded_index = reference["index_sha256"]
        .as_str()
        .expect("host-ref.json records index_sha256");
    let index = std::fs::read(ckpt_dir.join("model.safetensors.index.json")).unwrap();
    assert_eq!(
        sha256(&index),
        recorded_index,
        "model.safetensors.index.json differs from the checkpoint of the evidence"
    );
    let real = open(&ckpt_dir);
    assert_eq!(
        sha256(&real.config),
        CONFIG_SHA256,
        "config.json differs from the pinned checkpoint"
    );
    let mut kernels = MlxKernels::try_new().unwrap();
    kernels.reset_peak_memory().unwrap();
    kernels.reset_compile_counters();
    kernels.clear_compile_cache().unwrap();
    let started = std::time::Instant::now();
    let host = HostWeights::load(&real.ckpt, &QWEN3_6_35B_A3B, &real.config).unwrap();
    let (mut decoder, act) = new_decoder(kernels, &real.ckpt, host, 256);
    let load_seconds = started.elapsed().as_secs_f64();
    let mut state = decoder.zero_state().unwrap();

    // Gate 1: every layer of every prompt token.
    let mut failures = Vec::new();
    let mut table = Vec::new();
    let (mut min_cos, mut nan_cosines, mut seconds) = (1.0f64, 0usize, Vec::new());
    let mut last = None;
    // The identity of two runs (`Q36_SCALE_DTYPE` bf16 against f32) is the equal hash of every
    // recorded layer row and of the last-row logits, plus equal greedy ids.
    let mut layers_hash = Sha256::new();
    for (i, &token) in ids.iter().enumerate() {
        let started = std::time::Instant::now();
        let got = decoder.step(&mut state, token, StepMode::Record).unwrap();
        seconds.push(started.elapsed().as_secs_f64());
        let rows = got.layer_out.as_ref().unwrap();
        assert_eq!(rows.len(), layers);
        let mut row_cos = Vec::new();
        for (l, row) in rows.iter().enumerate() {
            hash_f32s(&mut layers_hash, row);
            let at = (i * layers + l) * hidden;
            let cos = cosine(row, &layers_ref[at..at + hidden]);
            min_cos = min_cosine(min_cos, cos);
            nan_cosines += usize::from(cos.is_nan());
            if !passes(cos) {
                failures.push(format!("token {i} layer {l}: cosine {cos}"));
            }
            row_cos.push(cos);
        }
        table.push(row_cos);
        last = Some(got);
    }
    let last = last.unwrap();
    // Gate 2: the last-row logits and their top 5.
    let logits = last.logits.as_ref().unwrap();
    let logits_cos = cosine(logits, &logits_ref);
    if !passes(logits_cos) {
        failures.push(format!("last-row logits: cosine {logits_cos}"));
    }
    let mut logits_hasher = Sha256::new();
    hash_f32s(&mut logits_hasher, logits);
    let logits_hash = format!("{:x}", logits_hasher.finalize());
    let got_top5 = top5(logits);
    if got_top5 != want_top5 {
        failures.push(format!(
            "top 5 differs: resident {got_top5:?}, reference {want_top5:?}"
        ));
    }
    // Gate 3: 31 more greedy steps give 32 tokens with the one from the last prompt step.
    let mut tokens = vec![last.token.unwrap()];
    for _ in 1..greedy.len() {
        let started = std::time::Instant::now();
        let got = decoder
            .step(&mut state, *tokens.last().unwrap(), StepMode::Generate)
            .unwrap();
        seconds.push(started.elapsed().as_secs_f64());
        tokens.push(got.token.unwrap());
    }
    if tokens != greedy {
        failures.push(format!(
            "greedy tokens differ: resident {tokens:?}, reference {greedy:?}"
        ));
    }
    let stats = decoder.kernels.memory_stats().unwrap();
    // The agreement numbers that a BF16 run reports beside the gate.
    let top5_overlap = got_top5.iter().filter(|id| want_top5.contains(id)).count();
    let greedy_equal_prefix = tokens
        .iter()
        .zip(&greedy)
        .take_while(|(a, b)| a == b)
        .count();
    let greedy_equal_positions = tokens.iter().zip(&greedy).filter(|(a, b)| a == b).count();
    // JSON has no NaN: a NaN cosine appears as null, and `nan_layer_cosines` counts the NaN rows.
    let report = serde_json::json!({
        "min_layer_cosine": min_cos, "nan_layer_cosines": nan_cosines, "logits_cosine": logits_cos, "top5": got_top5,
        "greedy_ids": tokens, "failures": failures, "layer_cosines": table,
        "load_seconds": load_seconds, "step_seconds": seconds,
        "mlx_active": stats.active, "mlx_cache": stats.cache, "mlx_peak": stats.peak, "rss_kb": rss_kb(),
        "kv_charge_bytes": decoder.budget.used(), "state_bytes": state.byte_len().unwrap(),
        "activation_dtype": activation_name(act),
        "layers_sha256": format!("{:x}", layers_hash.finalize()),
        "logits_sha256": logits_hash,
        "reference_top5": want_top5,
        "top5_overlap": top5_overlap, "reference_greedy_ids": greedy,
        "greedy_equal_prefix": greedy_equal_prefix,
        "greedy_equal_positions": greedy_equal_positions,
        "compile": super::compiled::report(&decoder.kernels),
    });
    // The retired-byte peak exists only in qualification builds, like its counter.
    #[cfg(feature = "native-fault-qualification")]
    let report = {
        let mut extended = report;
        extended["peak_retired_bytes"] =
            serde_json::json!(decoder.kernels.qualification_retire_stats().1);
        extended
    };
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if let Some(out) = &out_dir {
        std::fs::create_dir(out).expect("Q36_OUT_DIR must not exist yet");
        std::fs::write(
            std::path::Path::new(out).join("parity.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
    }
    if act == Activation::F32 {
        assert!(failures.is_empty(), "parity gate failed: {failures:#?}");
    } else {
        assert_eq!(nan_cosines, 0, "a NaN cosine is a broken run, not a result");
        assert!(
            keeps_floored_anchor(
                min_cos,
                BF16_LAYER_COSINE_FLOOR,
                D1_BF16_MIN_LAYER_COSINE,
                BF16_LAYER_COSINE_REGRESSION
            ),
            "BF16 parity floor: min layer cosine {min_cos} is under 0.95 or regresses more than \
             0.01 below the anchor 0.9799"
        );
        println!("measurement run: {} gate misses recorded", failures.len());
    }
}
