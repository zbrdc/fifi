//! The chunk gates on the real checkpoint (packet Q36-E8). Ignored tests; the Mac coordinator runs them.
//!
//! 1. `real_checkpoint_chunk_matches_the_host_reference`: the 5-token prompt of the frozen Q36-HR
//!    evidence runs as one chunk. The gate of decision 0046 item 4 applies: the residual stream of
//!    the last row after each of the 40 layers, the last-row logits and their top 5, and 32 greedy
//!    tokens (one from the chunk and 31 one-token steps) against the frozen evidence.
//! 2. `real_checkpoint_chunks_match_the_one_token_path`: the 37 tokens of the prompt and the greedy
//!    text, repeated to 296 tokens, run in chunks of 128 and as one-token steps. The last-row
//!    residuals, the logits, and the next 8 greedy tokens must agree. The bounds are the frozen
//!    bounds of decision 0044 G2, as in Flash-Next H8b, not proposals. Each of the 40 last-row
//!    layer outputs and the logits need a cosine of at least 0.99999 and a relative error
//!    `max|got - want| / max|want|` of at most 1e-4. The top 5 and the 8 greedy tokens must be equal.
//!    The one-token path runs with `Routing::Host`, because every chunk routes on the host. Only the
//!    accumulation of M > 1 against M = 1 differs between the two paths. Device against host routing
//!    parity is gated by Q36-DR4, not here. The report records the experts of the stepped last token.
//! 3. `real_checkpoint_prompt_speed`: a 2,048-token prompt in chunks of each size in `Q36_CHUNKS`.
//!    It reports seconds, tokens per second, memory, and the peak retired bytes of each run. With
//!    the `native-fault-qualification` feature it also reports synchronizations and forced
//!    flushes. Without the feature those fields are null. It asserts no speed. A Q36 result never
//!    satisfies Phase 0B or 1B.
//!
//! The activation dtype (packet Q36-BF16-b): `Q36_ACTIVATION=bf16` runs every test of this file
//! through the product `activation_dtype` lever (`scale_knob::new_decoder`), and the chunk path is
//! dtype aware, so the tests run rather than refuse. Unset or `f32` is the F32 control above and
//! its gates are unchanged. In BF16 the gates follow the within-mode rule of the packet: gate 2
//! (the chunk against the one-token path of the same decoder) gates the top 5, the 8 greedy
//! tokens, a finite cosine on every layer row and the logits cosine at the provisional floor 0.98
//! (fix5 of packet Q36-BF16-b: a layer cosine fell to 0.716 with the logits at 0.990), reports
//! the layer cosines with the layer of the minimum and the max relative error, and
//! reads the optional variables Q36_G2_TOKENS and Q36_G2_CHUNK. The same test also reads
//! Q36_G2_DUMP_DIR (an F32 run writes its last-row vectors and a `meta.txt` that records the run)
//! and Q36_G2_REF_DIR (a BF16 run reads them and reports the triangulation of git-db: BF16 chunk
//! against F32, BF16 steps against F32, chunk against steps; the dump is validated before any
//! native object exists — the same tokens and chunk, the `f32` activation, whole 40-layer files,
//! and one logits row of the vocabulary). Gate 1 (the
//! chunk against the frozen F32 evidence) keeps the 0.99999 rule report-only and asserts its
//! floor instead (decision 0046 A6 fix5): no NaN cosine, a minimum layer cosine of at least 0.95,
//! and no regression of more than 0.01 below the d1 value 0.9799 without a recorded decision. The
//! report names the dtype in `activation_dtype`.
//!
//! Run each test alone with its own `Q36_OUT_DIR`, which is required and must not exist yet:
//!
//! ```text
//! Q36_CKPT_DIR=<model_dir> Q36_REF_DIR=<copy of the evidence directory ref/> \
//!   Q36_OUT_DIR=<a path that does not exist yet> \
//!   cargo test --release --features native-mlx,native-fault-qualification --lib \
//!   family::qwen36::resident::prefill::parity::<test name> -- --ignored --exact --nocapture
//! ```

use super::super::decoder::{MAX_CHUNK_ROWS, ResidentDecoder, StepMode};
use super::super::moe::Routing;
use super::super::parity::{
    CONFIG_SHA256, HOST_REF_JSON_SHA256, LAYERS_SHA256, LOGITS_SHA256, cosine, f32s, min_cosine,
    read_checked, sha256, top5,
};
use super::super::real::{env_path, open, rss_kb};
use super::super::scale_knob::new_decoder;
use super::super::within_mode::{
    BF16_LOGITS_COSINE_FLOOR, MAX_REL_ERROR, MIN_COSINE, chunk_gate1_verdict, env_number,
    min_with_index, passes, within_mode_logits_passes, within_mode_passes, write_f32s,
};
use crate::{
    family::{
        qwen36::{
            host_load::HostWeights,
            settings::{Activation, activation_name},
        },
        spec::QWEN3_6_35B_A3B,
    },
    kernels::MlxKernels,
};

/// Layer, hidden, and vocabulary sizes of the real model.
const LAYERS: usize = 40;
const HIDDEN: usize = 2048;
const VOCAB: usize = 248_320;

/// The largest prompt the one-token-path test accepts (fix4 of packet Q36-BF16-b): the 1024-token
/// context that its `decoder_for` call names, minus the 8 greedy tokens the test generates after
/// the prompt, minus a margin of 16 tokens so the prompt and its greedy tail stay inside the
/// context. 1024 - 8 - 16 = 1000, the bound the test accepted before the named constant.
const G2_TOKENS_HIGH: usize = 1024 - 8 - 16;

/// `max|got - want| / max|want|`. A NaN in `got` or `want` gives NaN, so it cannot pass. All-zero
/// `want` values give infinity.
fn max_rel(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (mut diff, mut top) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (f64::from(g), f64::from(w));
        if g.is_nan() || w.is_nan() {
            return f64::NAN;
        }
        diff = diff.max((g - w).abs());
        top = top.max(w.abs());
    }
    if top > 0.0 { diff / top } else { f64::INFINITY }
}

/// The frozen evidence and the opened checkpoint.
struct Evidence {
    ids: Vec<u32>,
    greedy: Vec<u32>,
    layers_ref: Vec<f32>,
    logits_ref: Vec<f32>,
    want_top5: Vec<u32>,
}

/// Reads and hash-checks the evidence files and the checkpoint identity.
fn evidence(ref_dir: &std::path::Path, ckpt_dir: &std::path::Path) -> Evidence {
    let report_json = read_checked(ref_dir, "host-ref.json", HOST_REF_JSON_SHA256);
    let layers_ref = f32s(&read_checked(ref_dir, "layers.f32", LAYERS_SHA256));
    let logits_ref = f32s(&read_checked(ref_dir, "logits-last.f32", LOGITS_SHA256));
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
    assert_eq!(layers_ref.len(), ids.len() * LAYERS * HIDDEN);
    assert_eq!(logits_ref.len(), VOCAB);
    let recorded_index = reference["index_sha256"]
        .as_str()
        .expect("host-ref.json records index_sha256");
    let index = std::fs::read(ckpt_dir.join("model.safetensors.index.json")).unwrap();
    assert_eq!(
        sha256(&index),
        recorded_index,
        "model.safetensors.index.json differs from the checkpoint of the evidence"
    );
    Evidence {
        ids,
        greedy,
        layers_ref,
        logits_ref,
        want_top5,
    }
}

/// Opens the checkpoint, checks `config.json`, and builds a decoder with `context_limit` in the
/// dtype that `Q36_ACTIVATION` names (packet Q36-BF16-b; `scale_knob::new_decoder` also checks the
/// scale census and prints the load report). Unset is `f32`, the qualification control of the
/// frozen gates below.
fn decoder_for<'c>(
    real: &'c super::super::real::RealCheckpoint,
    context_limit: u32,
) -> (ResidentDecoder<'c>, Activation) {
    assert_eq!(
        sha256(&real.config),
        CONFIG_SHA256,
        "config.json differs from the pinned checkpoint"
    );
    let mut kernels = MlxKernels::try_new().unwrap();
    kernels.reset_peak_memory().unwrap();
    let host = HostWeights::load(&real.ckpt, &QWEN3_6_35B_A3B, &real.config).unwrap();
    new_decoder(kernels, &real.ckpt, host, context_limit)
}

/// Returns the required `Q36_OUT_DIR` and refuses an existing path before the long run.
fn out_dir_must_be_new() -> String {
    let out = std::env::var("Q36_OUT_DIR")
        .expect("Q36_OUT_DIR is required: a path that does not exist yet, one per test");
    assert!(
        std::fs::symlink_metadata(&out).is_err(),
        "Q36_OUT_DIR must not exist yet: {out}"
    );
    out
}

/// Reads one file of a G2 dump and panics naming the path when it is missing or unreadable.
fn read_dump(path: &std::path::Path) -> Vec<u8> {
    std::fs::read(path)
        .unwrap_or_else(|_| panic!("the G2 dump file {} is unreadable", path.display()))
}

/// Reads `meta.txt` of a G2 dump (fix4 of packet Q36-BF16-b): the three lines `tokens=<n>`,
/// `chunk=<c>`, and `activation=<f32|bf16>` that the dump run wrote next to its four vector
/// files. A missing line names the file; a value that is not a number names the line.
fn dump_meta(dir: &std::path::Path) -> (usize, usize, String) {
    let path = dir.join("meta.txt");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("the G2 dump file {} is unreadable", path.display()));
    let (mut tokens, mut chunk, mut activation) = (None, None, None);
    for line in text.lines() {
        if let Some(value) = line.strip_prefix("tokens=") {
            tokens =
                Some(value.parse::<usize>().unwrap_or_else(|_| {
                    panic!("{line:?} of {} is not a token count", path.display())
                }));
        } else if let Some(value) = line.strip_prefix("chunk=") {
            chunk =
                Some(value.parse::<usize>().unwrap_or_else(|_| {
                    panic!("{line:?} of {} is not a chunk size", path.display())
                }));
        } else if let Some(value) = line.strip_prefix("activation=") {
            activation = Some(value.to_string());
        }
    }
    (
        tokens.unwrap_or_else(|| panic!("{} records no tokens= line", path.display())),
        chunk.unwrap_or_else(|| panic!("{} records no chunk= line", path.display())),
        activation.unwrap_or_else(|| panic!("{} records no activation= line", path.display())),
    )
}

/// Validates the F32 dump that `Q36_G2_REF_DIR` names against this run before any native object
/// exists (fix4 of packet Q36-BF16-b): `meta.txt` must record this run's `tokens` and `chunk` — a
/// dump of another shape cannot be compared — and the dump's activation must be `f32`, because
/// the reference is the F32 run; each layer file must hold a whole number of `LAYERS`-layer rows,
/// and each logits file must hold exactly one row of `QWEN3_6_35B_A3B.vocab` floats, the length
/// of the report's `want.logits`. A missing file names its path.
fn check_ref_dump(dir: &std::path::Path, tokens: usize, chunk: usize) {
    let (dump_tokens, dump_chunk, activation) = dump_meta(dir);
    assert_eq!(
        (dump_tokens, dump_chunk),
        (tokens, chunk),
        "the Q36_G2_REF_DIR dump ran tokens={dump_tokens} chunk={dump_chunk}, \
         this run asks tokens={tokens} chunk={chunk}"
    );
    assert_eq!(
        activation, "f32",
        "the Q36_G2_REF_DIR dump must be the F32 run, its meta.txt says activation={activation}"
    );
    for name in ["steps-layers.f32", "chunk-layers.f32"] {
        let path = dir.join(name);
        let floats = read_dump(&path).len() / 4;
        assert_eq!(
            floats % LAYERS,
            0,
            "{} holds {floats} floats, not a whole number of {LAYERS}-layer rows",
            path.display()
        );
    }
    for name in ["steps-logits.f32", "chunk-logits.f32"] {
        let path = dir.join(name);
        assert_eq!(
            read_dump(&path).len(),
            QWEN3_6_35B_A3B.vocab as usize * 4,
            "{} must hold one logits row of {} floats, the length of want.logits",
            path.display(),
            QWEN3_6_35B_A3B.vocab
        );
    }
}

/// Prints the report and writes `<name>` into the new output directory. `create_dir` refuses a
/// directory that appeared during the run.
fn write_report(out: &str, name: &str, report: &serde_json::Value) {
    println!("{}", serde_json::to_string_pretty(report).unwrap());
    let dir = std::path::Path::new(out);
    if let Some(parent) = dir.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::create_dir(dir).expect("Q36_OUT_DIR must not exist yet");
    std::fs::write(dir.join(name), serde_json::to_vec_pretty(report).unwrap()).unwrap();
}

/// Synchronizations and forced retire flushes of the process, in qualification builds.
#[cfg(feature = "native-fault-qualification")]
fn counters(kernels: &MlxKernels) -> Option<(u64, u64)> {
    Some((
        kernels.qualification_sync_count(),
        kernels.qualification_retire_stats().0,
    ))
}

/// Without the qualification feature the counters are not available: the report holds null.
#[cfg(not(feature = "native-fault-qualification"))]
fn counters(_kernels: &MlxKernels) -> Option<(u64, u64)> {
    None
}

/// Resets the peak of retired bytes before a run (qualification builds only).
#[cfg(feature = "native-fault-qualification")]
fn reset_retire_peak(kernels: &mut MlxKernels) {
    kernels.qualification_reset_retire_peak();
}

#[cfg(not(feature = "native-fault-qualification"))]
fn reset_retire_peak(_kernels: &mut MlxKernels) {}

/// The peak of retired bytes since the last reset (decision 0044 item 8), or `None` without the
/// qualification feature.
#[cfg(feature = "native-fault-qualification")]
fn peak_retired_bytes(kernels: &MlxKernels) -> Option<u64> {
    Some(kernels.qualification_retire_stats().1)
}

#[cfg(not(feature = "native-fault-qualification"))]
fn peak_retired_bytes(_kernels: &MlxKernels) -> Option<u64> {
    None
}

#[test]
#[ignore = "needs the real checkpoint and the frozen Q36-HR evidence on the Mac; run with --ignored"]
fn real_checkpoint_chunk_matches_the_host_reference() {
    let _serial = crate::native_test_lock();
    let (ckpt_dir, ref_dir) = (env_path("Q36_CKPT_DIR"), env_path("Q36_REF_DIR"));
    let out = out_dir_must_be_new();
    let ev = evidence(&ref_dir, &ckpt_dir);
    let real = open(&ckpt_dir);
    let (mut decoder, act) = decoder_for(&real, 256);
    let mut state = decoder.zero_state().unwrap();
    let mut failures = Vec::new();
    // Gate 1: the last row after each layer, from one chunk of the whole prompt.
    let started = std::time::Instant::now();
    let got = decoder
        .prefill(&mut state, &ev.ids, StepMode::Record)
        .unwrap();
    let chunk_seconds = started.elapsed().as_secs_f64();
    let rows = got.layer_out.as_ref().unwrap();
    assert_eq!(rows.len(), LAYERS);
    let last = ev.ids.len() - 1;
    let (mut min_cos, mut nan_cosines, mut table) = (1.0f64, 0usize, Vec::new());
    let mut first_failing_layer: Option<usize> = None;
    for (l, row) in rows.iter().enumerate() {
        let at = (last * LAYERS + l) * HIDDEN;
        let cos = cosine(row, &ev.layers_ref[at..at + HIDDEN]);
        min_cos = min_cosine(min_cos, cos);
        nan_cosines += usize::from(cos.is_nan());
        if !passes(cos) {
            first_failing_layer.get_or_insert(l);
            failures.push(format!("token {last} layer {l}: cosine {cos}"));
        }
        table.push(cos);
    }
    // Gate 2: the last-row logits and their top 5.
    let logits = got.logits.as_ref().unwrap();
    let logits_cos = cosine(logits, &ev.logits_ref);
    if !passes(logits_cos) {
        failures.push(format!("token {last} logits: cosine {logits_cos}"));
    }
    let got_top5 = top5(logits);
    if got_top5 != ev.want_top5 {
        failures.push(format!(
            "top 5 differs: resident {got_top5:?}, reference {:?}",
            ev.want_top5
        ));
    }
    // Gate 3: 31 one-token steps after the chunk give 32 tokens with the one from the chunk.
    let mut tokens = vec![got.token.unwrap()];
    for _ in 1..ev.greedy.len() {
        let next = decoder
            .step(&mut state, *tokens.last().unwrap(), StepMode::Generate)
            .unwrap();
        tokens.push(next.token.unwrap());
    }
    if tokens != ev.greedy {
        failures.push(format!(
            "greedy tokens differ: resident {tokens:?}, reference {:?}",
            ev.greedy
        ));
    }
    // Packet Q36-BF16-b fix1: the agreement counters of the decode parity, recorded for both
    // dtypes so a BF16 report shows how close the frozen F32 reference came.
    let top5_overlap = got_top5
        .iter()
        .filter(|id| ev.want_top5.contains(id))
        .count();
    let greedy_equal_prefix = tokens
        .iter()
        .zip(&ev.greedy)
        .take_while(|(a, b)| a == b)
        .count();
    let report = serde_json::json!({
        "min_layer_cosine": min_cos, "nan_layer_cosines": nan_cosines, "logits_cosine": logits_cos,
        "top5": got_top5, "greedy_ids": tokens, "failures": failures, "layer_cosines": table,
        "first_failing_layer": first_failing_layer,
        "top5_overlap": top5_overlap, "greedy_equal_prefix": greedy_equal_prefix,
        "chunk_seconds": chunk_seconds, "rss_kb": rss_kb(),
        "kv_charge_bytes": decoder.budget.used(), "state_bytes": state.byte_len().unwrap(),
        "activation_dtype": activation_name(act),
    });
    write_report(&out, "chunk-parity.json", &report);
    if act != Activation::F32 {
        // BF16 (packet Q36-BF16-b): the chunk runs BF16 activations against the frozen F32
        // evidence, so the 0.99999 rule is report-only with its floor (decision 0046 A6 fix5).
        // The d1 anchor 0.9799 is the decode parity's minimum over all five prompt tokens
        // (evidence 9dec1b7c); this gate measures the last row of the chunk only, so the
        // no-ratchet check is the weaker sample of the two.
        assert!(
            logits_cos.is_finite(),
            "a NaN logits cosine is a broken run, not a result"
        );
        println!("measurement run: {} gate misses recorded", failures.len());
    }
    chunk_gate1_verdict(act, &failures, min_cos, nan_cosines).unwrap();
}

#[test]
#[ignore = "needs the real checkpoint and the frozen Q36-HR evidence on the Mac; run with --ignored"]
fn real_checkpoint_chunks_match_the_one_token_path() {
    let _serial = crate::native_test_lock();
    let (ckpt_dir, ref_dir) = (env_path("Q36_CKPT_DIR"), env_path("Q36_REF_DIR"));
    let out = out_dir_must_be_new();
    let dump_dir = std::env::var("Q36_G2_DUMP_DIR").ok();
    let ref_dump = std::env::var("Q36_G2_REF_DIR").ok();
    let ev = evidence(&ref_dir, &ckpt_dir);
    let n = env_number("Q36_G2_TOKENS", 296, 16, G2_TOKENS_HIGH);
    let chunk = env_number("Q36_G2_CHUNK", 128, 1, MAX_CHUNK_ROWS as usize);
    // Fix4 of packet Q36-BF16-b: the triangulation inputs are checked and the dump directory is
    // created here, before any native object exists — the checkpoint open and the decoder below.
    if let Some(dir) = ref_dump.as_deref() {
        check_ref_dump(std::path::Path::new(dir), n, chunk);
    }
    if let Some(dir) = dump_dir.as_deref() {
        assert!(
            dir != out,
            "Q36_G2_DUMP_DIR must differ from Q36_OUT_DIR: both are {dir}"
        );
        std::fs::create_dir(dir).unwrap_or_else(|error| {
            panic!(
                "Q36_G2_DUMP_DIR must be a new directory (it may hold files already): \
                 create_dir {dir}: {error}"
            )
        });
    }
    // The prompt and its greedy text, 37 coherent tokens, repeated.
    let text: Vec<u32> = ev.ids.iter().chain(&ev.greedy).copied().collect();
    let tokens: Vec<u32> = text.iter().copied().cycle().take(n).collect();
    let real = open(&ckpt_dir);
    let (mut decoder, act) = decoder_for(&real, 1024);
    decoder
        .set_chunk_rows(u32::try_from(chunk).unwrap())
        .unwrap();
    let mut chunked = decoder.zero_state().unwrap();
    let mut stepped = decoder.zero_state().unwrap();
    let started = std::time::Instant::now();
    let got = decoder
        .prefill(&mut chunked, &tokens, StepMode::Record)
        .unwrap();
    let chunk_seconds = started.elapsed().as_secs_f64();
    let started = std::time::Instant::now();
    let (last, before) = tokens.split_last().unwrap();
    // Every chunk routes on the host, so the one-token path routes on the host too. Only the
    // accumulation of M > 1 against M = 1 differs. Device routing parity is gated by Q36-DR4.
    decoder.set_routing(Routing::Host);
    for &t in before {
        decoder.step(&mut stepped, t, StepMode::Prompt).unwrap();
    }
    let want = decoder.step(&mut stepped, *last, StepMode::Record).unwrap();
    let step_seconds = started.elapsed().as_secs_f64();
    let token = tokens.len() - 1;
    let mut failures = Vec::new();
    let mut first_failing_layer: Option<usize> = None;
    let (mut min_cos, mut max_layer_rel, mut cosines, mut rels) =
        (1.0f64, 0.0f64, Vec::new(), Vec::new());
    for (l, (g, w)) in got
        .layer_out
        .as_ref()
        .unwrap()
        .iter()
        .zip(want.layer_out.as_ref().unwrap())
        .enumerate()
    {
        let (cos, rel) = (cosine(g, w), max_rel(g, w));
        min_cos = min_cosine(min_cos, cos);
        max_layer_rel = if rel.is_nan() || max_layer_rel.is_nan() {
            f64::NAN
        } else {
            max_layer_rel.max(rel)
        };
        if !within_mode_passes(act, cos, rel) {
            first_failing_layer.get_or_insert(l);
            failures.push(format!(
                "token {token} layer {l}: cosine {cos}, relative error {rel}"
            ));
        }
        cosines.push(cos);
        rels.push(rel);
    }
    let (got_logits, want_logits) = (got.logits.as_ref().unwrap(), want.logits.as_ref().unwrap());
    let (logits_cos, logits_rel) = (
        cosine(got_logits, want_logits),
        max_rel(got_logits, want_logits),
    );
    if !within_mode_logits_passes(act, logits_cos, logits_rel) {
        failures.push(format!(
            "token {token} logits: cosine {logits_cos}, relative error {logits_rel}"
        ));
    }
    let (got_top5, want_top5) = (top5(got_logits), top5(want_logits));
    if got_top5 != want_top5 {
        failures.push(format!(
            "token {token} top 5 differs: chunk {got_top5:?}, steps {want_top5:?}"
        ));
    }
    // The next 8 greedy tokens from each state.
    let (mut a, mut b) = (vec![got.token.unwrap()], vec![want.token.unwrap()]);
    for _ in 1..8 {
        let next = decoder
            .step(&mut chunked, *a.last().unwrap(), StepMode::Generate)
            .unwrap();
        a.push(next.token.unwrap());
        let next = decoder
            .step(&mut stepped, *b.last().unwrap(), StepMode::Generate)
            .unwrap();
        b.push(next.token.unwrap());
    }
    if a != b {
        let at = a.iter().zip(&b).position(|(x, y)| x != y).unwrap();
        failures.push(format!(
            "greedy token {at} after the prompt differs: chunk {a:?}, steps {b:?}"
        ));
    }
    // The triangulation of git-db against an F32 run of this test: with Q36_G2_DUMP_DIR this run
    // writes its last-row vectors for a later BF16 run, and with Q36_G2_REF_DIR it reads an F32
    // dump and reports this dtype's chunk and steps against it.
    if let Some(dir) = dump_dir.as_deref() {
        // Created new before the runs (fix4): only this run's four dumps and its meta.txt land
        // in it.
        let dir = std::path::Path::new(dir);
        for (name, rows) in [
            ("steps-layers.f32", &want.layer_out),
            ("chunk-layers.f32", &got.layer_out),
        ] {
            let mut all = Vec::new();
            for layer in rows.as_ref().unwrap() {
                all.extend_from_slice(layer);
            }
            write_f32s(&dir.join(name), &all);
        }
        write_f32s(&dir.join("steps-logits.f32"), want.logits.as_ref().unwrap());
        write_f32s(&dir.join("chunk-logits.f32"), got.logits.as_ref().unwrap());
        std::fs::write(
            dir.join("meta.txt"),
            format!(
                "tokens={}\nchunk={}\nactivation={}\n",
                tokens.len(),
                chunk,
                activation_name(act)
            ),
        )
        .unwrap();
    }
    let (c_min, c_min_layer) = min_with_index(&cosines);
    let tri = ref_dump.as_deref().map(|dir| {
        let dir = std::path::Path::new(dir);
        let layer_count = got.layer_out.as_ref().unwrap().len();
        let ref_layers = f32s(&std::fs::read(dir.join("steps-layers.f32")).unwrap());
        let ref_logits = f32s(&std::fs::read(dir.join("steps-logits.f32")).unwrap());
        assert_eq!(
            ref_layers.len() % layer_count,
            0,
            "the F32 layer dump does not hold a whole number of {layer_count}-layer rows"
        );
        let width = ref_layers.len() / layer_count;
        let ref_layer = |l: usize| &ref_layers[l * width..(l + 1) * width];
        let a: Vec<f64> = (0..layer_count)
            .map(|l| cosine(&got.layer_out.as_ref().unwrap()[l], ref_layer(l)))
            .collect();
        let b: Vec<f64> = (0..layer_count)
            .map(|l| cosine(&want.layer_out.as_ref().unwrap()[l], ref_layer(l)))
            .collect();
        (
            min_with_index(&a),
            min_with_index(&b),
            cosine(got_logits, &ref_logits),
            cosine(want_logits, &ref_logits),
            a,
            b,
        )
    });
    let report = serde_json::json!({
        "tokens": tokens.len(), "chunk_rows": chunk, "routing_steps": "host",
        "activation_dtype": activation_name(act),
        "bounds": {"min_cosine": MIN_COSINE, "max_relative_error": MAX_REL_ERROR, "max_relative_error_gated": act == Activation::F32, "min_cosine_gated": act == Activation::F32, "bf16_layer_cosine_gated": false, "bf16_logits_cosine_floor": BF16_LOGITS_COSINE_FLOOR},
        "bound_note": "f32 gates cosine and max relative error; bf16 gates the top 5, the 8 greedy tokens, a finite cosine on every layer row and the logits cosine at bf16_logits_cosine_floor, and reports the layer cosines (with the layer of the minimum) and the max relative error (packet Q36-BF16-b; the floor is provisional until Q36-BF16-SENS)",
        "min_layer_cosine": min_cos, "max_layer_rel": max_layer_rel,
        "logits_cosine": logits_cos, "logits_rel": logits_rel,
        "top5_chunk": got_top5, "top5_steps": want_top5,
        "greedy_chunk": a, "greedy_steps": b, "failures": failures,
        "first_failing_layer": first_failing_layer,
        "layer_cosines": cosines, "layer_rels": rels,
        "triangulation": {
            "a_chunk_vs_f32_layers": tri.as_ref().map(|t| &t.4),
            "b_steps_vs_f32_layers": tri.as_ref().map(|t| &t.5),
            "a_min": tri.as_ref().map(|t| t.0 .0),
            "a_min_layer": tri.as_ref().map(|t| t.0 .1),
            "b_min": tri.as_ref().map(|t| t.1 .0),
            "b_min_layer": tri.as_ref().map(|t| t.1 .1),
            "c_min": c_min, "c_min_layer": c_min_layer,
            "a_logits": tri.as_ref().map(|t| t.2),
            "b_logits": tri.as_ref().map(|t| t.3),
            "note": "a = this dtype's chunk against the F32 steps of the dump, b = this dtype's steps against the same F32 steps, c = the chunk against the steps of this dtype (the existing layer_cosines)"
        },
        "steps_routes": want.routes,
        "chunk_seconds": chunk_seconds, "step_seconds": step_seconds, "rss_kb": rss_kb(),
    });
    write_report(&out, "chunk-vs-steps.json", &report);
    assert!(
        failures.is_empty(),
        "chunk against steps failed: {failures:#?}"
    );
}

#[test]
#[ignore = "needs the real checkpoint on the Mac; run with --ignored"]
fn real_checkpoint_prompt_speed() {
    let _serial = crate::native_test_lock();
    let (ckpt_dir, ref_dir) = (env_path("Q36_CKPT_DIR"), env_path("Q36_REF_DIR"));
    let out = out_dir_must_be_new();
    let ev = evidence(&ref_dir, &ckpt_dir);
    let sizes: Vec<u32> = std::env::var("Q36_CHUNKS")
        .unwrap_or_else(|_| "512,256".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let text: Vec<u32> = ev.ids.iter().chain(&ev.greedy).copied().collect();
    let tokens: Vec<u32> = text.iter().copied().cycle().take(2_048).collect();
    let real = open(&ckpt_dir);
    let (mut decoder, act) = decoder_for(&real, 4_096);
    let mut runs = Vec::new();
    // Two passes per size: the first includes shader compilation, the second is warm.
    for (index, &size) in sizes.iter().chain(&sizes).enumerate() {
        let pass = index / sizes.len() + 1;
        decoder.set_chunk_rows(size).unwrap();
        decoder.kernels.reset_peak_memory().unwrap();
        reset_retire_peak(&mut decoder.kernels);
        let mut state = decoder.zero_state().unwrap();
        let before = counters(&decoder.kernels);
        let started = std::time::Instant::now();
        let mut per_chunk = Vec::new();
        let parts: Vec<&[u32]> = tokens.chunks(size as usize).collect();
        let mut first = None;
        for (i, part) in parts.iter().enumerate() {
            let mode = if i + 1 == parts.len() {
                StepMode::Generate
            } else {
                StepMode::Prompt
            };
            let at = std::time::Instant::now();
            let got = decoder.prefill(&mut state, part, mode).unwrap();
            per_chunk.push(at.elapsed().as_secs_f64());
            first = got.token.or(first);
        }
        let seconds = started.elapsed().as_secs_f64();
        let after = counters(&decoder.kernels);
        let peak_retired = peak_retired_bytes(&decoder.kernels);
        // Null without the qualification feature: a zero would read as a clean result.
        let (syncs, flushes) = match (before, after) {
            (Some((s0, f0)), Some((s1, f1))) => (Some(s1 - s0), Some(f1 - f0)),
            _ => (None, None),
        };
        let stats = decoder.kernels.memory_stats().unwrap();
        runs.push(serde_json::json!({
            "pass": pass, "chunk_rows": size, "tokens": tokens.len(), "seconds": seconds,
            "prompt_tokens_per_second": tokens.len() as f64 / seconds,
            "per_chunk_seconds": per_chunk, "first_token": first,
            "synchronizations": syncs, "forced_flushes": flushes, "peak_retired_bytes": peak_retired,
            "mlx_active": stats.active, "mlx_peak": stats.peak, "rss_kb": rss_kb(),
            "kv_charge_bytes": decoder.budget.used(),
        }));
    }
    // The first token must be equal in all runs.
    let firsts: Vec<&serde_json::Value> = runs.iter().map(|run| &run["first_token"]).collect();
    let first_token_equal = firsts.windows(2).all(|pair| pair[0] == pair[1]);
    let report = serde_json::json!({
        "gate_tokens_per_second": 60, "first_token_equal": first_token_equal, "runs": runs,
        "activation_dtype": activation_name(act),
        "note": "pass 1 includes shader compilation; pass 2 is warm; counters are null without native-fault-qualification",
    });
    write_report(&out, "prompt-speed.json", &report);
}
