//! The token driver on the real pack, on a card: the assembly the engine start uses
//! (`start::build_driver`), against the exllamav3 oracle. Three ignored device tests, each with its
//! own environment. Every test needs a GPU (about 21 GiB), about 25 GiB of pinned host memory and
//! `--test-threads=1`.
//!
//! `the_first_real_tokens_match_the_oracle`: one free greedy run on one oracle prompt.
//! - `FIFI_FLASH_NEXT_EXL3_PACK`: the full local pack directory (required).
//! - `FIFI_FLASH_NEXT_DRIVER_ORACLE`: the oracle run JSON (required).
//! - `FIFI_FLASH_NEXT_DRIVER_PROMPT`: the prompt id in it (default `chat-short`); its ids are the
//!   oracle's.
//! - `FIFI_FLASH_NEXT_DRIVER_STEPS`: greedy tokens to draw (default 16).
//! - `FIFI_FLASH_NEXT_DRIVER_CONTEXT`: positions of the caches (default 512, a multiple of 4).
//! - `FIFI_FLASH_NEXT_DRIVER_SLOTS`: expert slots per layer (default: the load plan's).
//! - `FIFI_FLASH_NEXT_DRIVER_OUT`: a directory for `engine.json` (one run record) and
//!   `step0.logits.f32` (the first row); optional.
//!
//! Reported: the generated ids against the oracle's, the stage table, and the step-0 relative L2,
//! largest difference, argmax and top-20 overlap against the oracle's full row.
//!
//! `teacher_forced_steps_follow_the_oracle_on_every_prompt`: every oracle prompt, fed the oracle's
//! tokens, in one process.
//! - `FIFI_FLASH_NEXT_EXL3_PACK`, `_DRIVER_ORACLE`, `_DRIVER_CONTEXT` (default: the largest prompt
//!   plus its steps plus 8, rounded up to 4) and `_DRIVER_SLOTS` as above.
//! - `FIFI_FLASH_NEXT_DRIVER_PROMPTS`: a comma-separated list of prompt ids (default: all).
//! - `FIFI_FLASH_NEXT_DRIVER_BUDGET_SEC`: wall seconds after which a prompt that would not finish
//!   is skipped and named (default 700).
//! - `FIFI_FLASH_NEXT_DRIVER_OUT`: a directory for `follow.json` (summaries), `engine-follow.json`
//!   (the run in the oracle's schema) and `engine-follow.parts/<id>.logits.npy` (the first full
//!   rows of each prompt); rewritten after every prompt. Optional.
//!
//! `single_row_kernel_receipt_run`: one greedy run with a logits hash per step, for one process
//! configuration of the EXL3 kernel switches.
//! - `FIFI_FLASH_NEXT_EXL3_PACK`, `_DRIVER_ORACLE`, `_DRIVER_PROMPT` (default `chat-short`),
//!   `_DRIVER_CONTEXT` (default 512) and `_DRIVER_SLOTS` as above.
//! - `FIFI_FLASH_NEXT_DRIVER_STEPS`: steps (default 64).
//! - `FIFI_FLASH_NEXT_DRIVER_CONFIG` (a label) and `FIFI_FLASH_NEXT_DRIVER_ROUND` (default `1`):
//!   name the record. Both are file-name parts: letters, digits, `.`, `_` and `-` only.
//! - `FIFI_FLASH_NEXT_DRIVER_OUT`: the directory of `ab-<config>-r<round>.json` (required).
//! - It also reports `FIFI_EXL3_SINGLE_GATHER`, `FIFI_EXL3_DP4A` and `FIFI_EXL3_INDEXED_PAIRS`.
//!
//! Nothing here asserts a speed.

use std::path::{Path, PathBuf};

use std::time::Duration;

use super::{
    driver::TokenTimes,
    start::{StepDriver, build_driver},
};
use crate::{
    family::{
        flash_next::settings::{CudaSettings, KvFormatName},
        session::TokenStepper,
    },
    kernels::cuda::device_present,
};

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// A file-name part from the oracle file or the environment: letters, digits, `.`, `_` and `-`
/// only, no `..`, so a name cannot leave the output directory.
fn safe_name(what: &str, value: &str) -> String {
    assert!(
        !value.is_empty()
            && !value.contains("..")
            && value
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')),
        "{what} {value:?} is not a plain file-name part ([A-Za-z0-9._-])"
    );
    value.to_string()
}

/// The `[rows, width]` float32 rows of a `.npy` file (version 1 or 2, C order).
fn read_npy(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&bytes[..6], b"\x93NUMPY", "not an npy file");
    let (header_len, start) = match bytes[6] {
        1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
        _ => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
    };
    bytes[start + header_len..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (index, value) in row.iter().enumerate() {
        if *value > row[best] {
            best = index;
        }
    }
    best
}

fn top(row: &[f32], count: usize) -> Vec<(usize, f32)> {
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|a, b| row[*b].total_cmp(&row[*a]).then(a.cmp(b)));
    order.into_iter().take(count).map(|i| (i, row[i])).collect()
}

/// One token's measurements: the driver's phases, the host argmax and the bank misses it caused.
#[derive(Clone, Copy)]
struct Sample {
    times: TokenTimes,
    wall: Duration,
    argmax: Duration,
    misses: u64,
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn mean_and_p50(values: &mut [f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    values.sort_by(f64::total_cmp);
    (
        values.iter().sum::<f64>() / values.len() as f64,
        values[values.len() / 2],
    )
}

/// Prints the stage table of `samples`: mean and p50 in ms per token.
fn table(title: &str, samples: &[Sample]) {
    type Field = fn(&Sample) -> f64;
    let rows: [(&str, Field); 11] = [
        ("embed + n-gram rows", |s| ms(s.times.rows)),
        ("start write", |s| ms(s.times.start)),
        ("layers before PLE (queued)", |s| ms(s.times.layers_before)),
        ("stream read-back (device wait)", |s| ms(s.times.ple_sync)),
        ("host PLE step", |s| ms(s.times.ple_host)),
        ("stream write-back", |s| ms(s.times.ple_write)),
        ("layers from PLE on (queued)", |s| ms(s.times.layers_after)),
        ("head + logits read-back", |s| ms(s.times.head)),
        ("health read", |s| ms(s.times.health)),
        ("host argmax", |s| ms(s.argmax)),
        ("TOTAL wall", |s| ms(s.wall)),
    ];
    println!(
        "stage table, {title}, {} tokens (ms per token: mean / p50)",
        samples.len()
    );
    for (name, field) in rows {
        let mut values: Vec<f64> = samples.iter().map(field).collect();
        let (mean, p50) = mean_and_p50(&mut values);
        println!("  {name:<32} {mean:>9.3} {p50:>9.3}");
    }
    let mut misses: Vec<f64> = samples.iter().map(|s| s.misses as f64).collect();
    let (mean, p50) = mean_and_p50(&mut misses);
    let total: u64 = samples.iter().map(|s| s.misses).sum();
    println!("  bank misses per token            {mean:>9.2} {p50:>9.1}  (total {total})");
}

fn sample(driver: &StepDriver, wall: Duration, argmax: Duration, before: u64) -> Sample {
    Sample {
        times: driver.last_times(),
        wall,
        argmax,
        misses: driver.bank_misses().saturating_sub(before),
    }
}

/// `prefill_rows: 4` builds the chunked driver: the prompt runs as device chunks of four rows
/// through `feed_rows`, then decode continues one row at a time, and the greedy tokens still
/// match the oracle run. This is the settings-to-driver wiring proof; the m-row kernel math is
/// proven by `step_tests::a_chunk_of_rows_matches_the_same_rows_one_at_a_time`.
#[test]
#[ignore = "needs a GPU (about 21 GiB), the full pack in FIFI_FLASH_NEXT_EXL3_PACK and the oracle run JSON"]
fn the_chunked_prefill_driver_matches_the_oracle() {
    assert!(
        device_present("Flash-Next chunked driver"),
        "this test needs a CUDA card"
    );
    let pack = PathBuf::from(var("FIFI_FLASH_NEXT_EXL3_PACK").expect("FIFI_FLASH_NEXT_EXL3_PACK"));
    let oracle_path =
        PathBuf::from(var("FIFI_FLASH_NEXT_DRIVER_ORACLE").expect("FIFI_FLASH_NEXT_DRIVER_ORACLE"));
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&oracle_path).expect("oracle json")).expect("json");
    let actual_vocab = oracle["vocab"]["actual_vocab"]
        .as_u64()
        .expect("actual_vocab") as usize;
    let steps: usize =
        var("FIFI_FLASH_NEXT_DRIVER_STEPS").map_or(16, |v| v.parse().expect("steps"));
    let record = oracle["prompts"]
        .as_array()
        .expect("prompts")
        .iter()
        .find(|p| p["id"] == "chat-short")
        .expect("the chat-short prompt");
    let ids = |key: &str| -> Vec<u32> {
        record[key]
            .as_array()
            .unwrap_or_else(|| panic!("no {key}"))
            .iter()
            .map(|v| v.as_u64().expect("an id") as u32)
            .collect()
    };
    let prompt = ids("prompt_ids");
    let expected = ids("generated");
    let context: u32 = 512;
    assert!(
        prompt.len() + steps < context as usize,
        "the context is too small"
    );
    let wanted = CudaSettings {
        context,
        kv_format: KvFormatName::F16,
        slots: None,
        expert_stats: None,
        prefill_rows: var("FIFI_FLASH_NEXT_DRIVER_ROWS").map_or(4, |v| v.parse().expect("rows")),
    };
    let mut driver = build_driver(&|_| {}, &pack, &wanted, actual_vocab, None).expect("driver");

    // The prompt (its last token included) through the chunked feed: chunks of four rows.
    let began = std::time::Instant::now();
    driver.feed_rows(&prompt).expect("feed the prompt");
    println!(
        "chunked prompt: {} tokens in {:.2} s",
        prompt.len(),
        began.elapsed().as_secs_f64()
    );

    // Decode from the last prompt token, one row per call, against the oracle's greedy tokens.
    let mut input = *prompt.last().expect("a prompt");
    let mut generated = Vec::with_capacity(steps);
    for step in 0..steps {
        let row = driver
            .logits(input)
            .unwrap_or_else(|e| panic!("step {step}: {e:?}"));
        let pick = argmax(&row[..actual_vocab]) as u32;
        assert!(
            row[..actual_vocab].iter().all(|v| v.is_finite()),
            "step {step}: non-finite logits"
        );
        generated.push(pick);
        input = pick;
    }
    println!("chunked generated {generated:?}");
    println!(
        "oracle           {:?}",
        &expected[..steps.min(expected.len())]
    );
    let agree = generated
        .iter()
        .zip(&expected)
        .take_while(|(a, b)| a == b)
        .count();
    assert!(
        agree >= 8.min(steps),
        "the chunked prefill driver agrees with the oracle on only {agree} of {steps}"
    );
}

#[test]
#[ignore = "needs a GPU (about 21 GiB), the full pack in FIFI_FLASH_NEXT_EXL3_PACK and the oracle run JSON"]
fn the_first_real_tokens_match_the_oracle() {
    assert!(
        device_present("Flash-Next token driver"),
        "a GPU is required"
    );
    let pack = PathBuf::from(var("FIFI_FLASH_NEXT_EXL3_PACK").expect("FIFI_FLASH_NEXT_EXL3_PACK"));
    let steps: usize =
        var("FIFI_FLASH_NEXT_DRIVER_STEPS").map_or(16, |v| v.parse().expect("steps"));
    let oracle_path =
        PathBuf::from(var("FIFI_FLASH_NEXT_DRIVER_ORACLE").expect("FIFI_FLASH_NEXT_DRIVER_ORACLE"));
    let wanted_id = var("FIFI_FLASH_NEXT_DRIVER_PROMPT").unwrap_or_else(|| "chat-short".into());
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&oracle_path).expect("oracle json")).expect("json");
    let actual_vocab = oracle["vocab"]["actual_vocab"]
        .as_u64()
        .expect("actual_vocab") as usize;
    let record = oracle["prompts"]
        .as_array()
        .expect("prompts")
        .iter()
        .find(|p| p["id"] == wanted_id.as_str())
        .unwrap_or_else(|| panic!("no prompt {wanted_id}"));
    let ids = |key: &str| -> Vec<u32> {
        record[key]
            .as_array()
            .unwrap_or_else(|| panic!("no {key}"))
            .iter()
            .map(|v| v.as_u64().expect("an id") as u32)
            .collect()
    };
    let prompt = ids("prompt_ids");
    let expected = ids("generated");
    let context: u32 =
        var("FIFI_FLASH_NEXT_DRIVER_CONTEXT").map_or(512, |v| v.parse().expect("context"));
    assert!(
        prompt.len() + steps < context as usize,
        "the context is too small"
    );
    let wanted = CudaSettings {
        context,
        kv_format: KvFormatName::F16,
        slots: var("FIFI_FLASH_NEXT_DRIVER_SLOTS").map(|v| v.parse().expect("slots")),
        expert_stats: None,
        prefill_rows: 1,
    };
    let began = std::time::Instant::now();
    let mut driver = build_driver(
        &|stage| println!("stage: {stage}"),
        &pack,
        &wanted,
        actual_vocab,
        None,
    )
    .expect("build the driver");
    println!("load: {:.1} s", began.elapsed().as_secs_f64());

    let (last, head) = prompt.split_last().expect("a prompt");
    let fed = std::time::Instant::now();
    let mut prompt_samples = Vec::new();
    for &token in head {
        let before = driver.bank_misses();
        let opened = std::time::Instant::now();
        driver.feed(token).expect("feed a prompt token");
        prompt_samples.push(sample(&driver, opened.elapsed(), Duration::ZERO, before));
    }
    println!(
        "prompt: {} tokens fed in {:.2} s",
        head.len(),
        fed.elapsed().as_secs_f64()
    );
    let mut input = *last;
    let mut generated: Vec<u32> = Vec::new();
    let mut records = Vec::new();
    let mut first_row: Vec<f32> = Vec::new();
    let mut step_samples = Vec::new();
    for step in 0..steps {
        let before = driver.bank_misses();
        let began = std::time::Instant::now();
        let row = driver
            .logits(input)
            .unwrap_or_else(|e| panic!("step {step}: {e:?}"));
        let wall = began.elapsed();
        let visible = &row[..actual_vocab];
        let picking = std::time::Instant::now();
        let pick = argmax(visible);
        step_samples.push(sample(&driver, wall, picking.elapsed(), before));
        let wall_ms = ms(wall);
        let finite = visible.iter().all(|v| v.is_finite());
        let want = expected.get(step).copied();
        println!(
            "step {step}: input {input} -> argmax {pick} (oracle {want:?}), top-1 logit {:.4}, finite {finite}, {wall_ms:.1} ms, {} bank misses",
            visible[pick],
            step_samples.last().map_or(0, |s| s.misses)
        );
        assert!(finite, "step {step}: non-finite logits");
        records.push(serde_json::json!({
            "i": step, "token": pick, "argmax": pick,
            "top": top(visible, 20).iter().map(|(i, v)| serde_json::json!([i, v])).collect::<Vec<_>>(),
        }));
        if step == 0 {
            first_row = row.clone();
        }
        generated.push(pick as u32);
        input = pick as u32;
    }
    table("prompt tokens (no head)", &prompt_samples);
    table("generated tokens", &step_samples);
    println!("generated {generated:?}");
    println!("oracle    {:?}", &expected[..steps.min(expected.len())]);
    let agree = generated
        .iter()
        .zip(&expected)
        .take_while(|(a, b)| a == b)
        .count();
    println!("greedy prefix agreement: {agree} of {steps}");
    if agree < steps.min(expected.len()) {
        let oracle_steps = record["steps"].as_array().expect("steps");
        let at = &oracle_steps[agree];
        let top = at["top"].as_array().expect("top");
        println!(
            "first divergence at step {agree}: ours {} oracle {}; oracle top-1 {} top-2 {} margin {}",
            generated[agree], expected[agree], top[0][1], top[1][1], at["margin"]
        );
    }

    // Step 0 against the oracle's full row.
    let oracle_rows = read_npy(
        &oracle_path.parent().expect("dir").join(
            record["full_logits"]["file"]
                .as_str()
                .expect("full_logits file"),
        ),
    );
    let width = record["full_logits"]["width"].as_u64().expect("width") as usize;
    let oracle0 = &oracle_rows[..width];
    let (got, want) = (&first_row[..actual_vocab], &oracle0[..actual_vocab]);
    let (mut diff, mut norm, mut max_abs) = (0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let d = f64::from(*g) - f64::from(*w);
        diff += d * d;
        norm += f64::from(*w).powi(2);
        max_abs = max_abs.max(d.abs());
    }
    let log_sum_exp = |row: &[f32]| -> f64 {
        let top = f64::from(row[argmax(row)]);
        top + row
            .iter()
            .map(|v| (f64::from(*v) - top).exp())
            .sum::<f64>()
            .ln()
    };
    let (lse_got, lse_want) = (log_sum_exp(got), log_sum_exp(want));
    let kl: f64 = got
        .iter()
        .zip(want)
        .map(|(g, w)| {
            let log_want = f64::from(*w) - lse_want;
            log_want.exp() * (log_want - (f64::from(*g) - lse_got))
        })
        .sum();
    println!("step 0 KL(oracle || ours) {kl:.4e}");
    let got_top: Vec<usize> = top(got, 20).iter().map(|t| t.0).collect();
    let want_top: Vec<usize> = top(want, 20).iter().map(|t| t.0).collect();
    let overlap = got_top.iter().filter(|i| want_top.contains(i)).count();
    println!(
        "step 0 vs the oracle row: relative L2 {:.3e}, max abs {max_abs:.3e}, argmax {} vs {}, top-20 overlap {overlap}/20",
        (diff / norm).sqrt(),
        argmax(got),
        argmax(want)
    );
    if let Some(out) = var("FIFI_FLASH_NEXT_DRIVER_OUT") {
        let out = PathBuf::from(out);
        std::fs::create_dir_all(&out).expect("out dir");
        let run = serde_json::json!({
            "schema": "fifi-flash-next-oracle/1", "role": "engine", "prompt": wanted_id,
            "prompt_ids": prompt, "generated": generated, "steps": records,
            "agree_prefix": agree,
        });
        std::fs::write(
            out.join("engine.json"),
            serde_json::to_vec_pretty(&run).expect("json"),
        )
        .expect("write engine.json");
        let raw: Vec<u8> = first_row.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out.join("step0.logits.f32"), raw).expect("write logits");
    }
}

// ---------------------------------------------------------------------------------------------
// Teacher-forced follow mode over the oracle prompts.

fn softmax(logits: &[f64]) -> Vec<f64> {
    let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = logits.iter().map(|v| (v - top).exp()).sum();
    logits.iter().map(|v| (v - top).exp() / sum).collect()
}

/// KL(p || q) of the softmaxes of two logit vectors over one support.
fn kl_of_logits(p: &[f64], q: &[f64]) -> f64 {
    let (p, q) = (softmax(p), softmax(q));
    p.iter()
        .zip(&q)
        .filter(|(p, _)| **p > 0.0)
        .map(|(p, q)| p * (p / q).ln())
        .sum()
}

fn widen(row: &[f32]) -> Vec<f64> {
    row.iter().map(|v| f64::from(*v)).collect()
}

/// One step of a teacher-forced prompt.
struct FollowStep {
    oracle_argmax: usize,
    our_argmax: usize,
    oracle_margin: f64,
    our_margin: f64,
    /// Our logit of the oracle's token and of ours.
    our_logit_oracle_token: f64,
    our_logit_top: f64,
    /// KL over the oracle's stored top-20 tokens, both sides renormalized on those 20.
    kl_top20: f64,
    /// KL over the whole vocabulary, where the oracle stored the full row.
    kl_full: Option<f64>,
    overlap: usize,
}

/// Writes `rows` x `width` float32 values as a version 1.0 `.npy` file, C order.
fn write_npy(path: &Path, values: &[f32], rows: usize, width: usize) {
    assert_eq!(values.len(), rows * width);
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {width}), }}");
    // Magic, version and length take 10 bytes; the header ends in a newline and the whole is a
    // multiple of 64.
    while !(10 + header.len() + 1).is_multiple_of(64) {
        header.push(' ');
    }
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    std::fs::write(path, bytes).expect("write npy");
}

fn mean_max(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    (
        values.iter().sum::<f64>() / values.len() as f64,
        values.iter().copied().fold(0.0, f64::max),
    )
}

/// All the oracle prompts, teacher-forced with the oracle's tokens, in one process: the bank is
/// loaded once and the driver is reset between prompts. Prompts run shortest first, and a prompt
/// that would not finish inside `FIFI_FLASH_NEXT_DRIVER_BUDGET_SEC` (default 700) of wall time is
/// skipped and named, so a slow run still ends with a complete report of what it did.
///
/// Reported per prompt: steps, top-1 agreement, mean and max KL(oracle || ours) over the oracle's
/// stored top-20 tokens (every step) and over the whole vocabulary (the steps with a stored full
/// row), top-20 overlap, and every disagreement with both margins.
#[test]
#[ignore = "needs a GPU (about 21 GiB), the full pack in FIFI_FLASH_NEXT_EXL3_PACK and the oracle run JSON"]
fn teacher_forced_steps_follow_the_oracle_on_every_prompt() {
    assert!(
        device_present("Flash-Next token driver follow mode"),
        "a GPU is required"
    );
    let opened = std::time::Instant::now();
    let pack = PathBuf::from(var("FIFI_FLASH_NEXT_EXL3_PACK").expect("FIFI_FLASH_NEXT_EXL3_PACK"));
    let oracle_path =
        PathBuf::from(var("FIFI_FLASH_NEXT_DRIVER_ORACLE").expect("FIFI_FLASH_NEXT_DRIVER_ORACLE"));
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&oracle_path).expect("oracle json")).expect("json");
    let actual_vocab = oracle["vocab"]["actual_vocab"]
        .as_u64()
        .expect("actual_vocab") as usize;
    let filter: Option<Vec<String>> = var("FIFI_FLASH_NEXT_DRIVER_PROMPTS")
        .map(|v| v.split(',').map(|id| id.trim().to_string()).collect());
    let budget: f64 =
        var("FIFI_FLASH_NEXT_DRIVER_BUDGET_SEC").map_or(700.0, |v| v.parse().expect("budget"));
    let mut records: Vec<&serde_json::Value> = oracle["prompts"]
        .as_array()
        .expect("prompts")
        .iter()
        .filter(|p| {
            filter
                .as_ref()
                .is_none_or(|ids| ids.iter().any(|id| p["id"] == id.as_str()))
        })
        .collect();
    records.sort_by_key(|p| p["prompt_len"].as_u64().expect("prompt_len"));
    // Every id becomes a file name: refuse a bad one now, not after the load and the first prompts.
    for record in &records {
        safe_name("prompt id", record["id"].as_str().expect("id"));
    }
    let need = records
        .iter()
        .map(|p| p["prompt_len"].as_u64().unwrap() + p["steps"].as_array().unwrap().len() as u64)
        .max()
        .expect("a prompt") as u32
        + 8;
    let context: u32 = var("FIFI_FLASH_NEXT_DRIVER_CONTEXT")
        .map_or(need.next_multiple_of(4), |v| v.parse().expect("context"));
    assert!(
        context >= need,
        "the context {context} is below the {need} needed"
    );
    let wanted = CudaSettings {
        context,
        kv_format: KvFormatName::F16,
        slots: var("FIFI_FLASH_NEXT_DRIVER_SLOTS").map(|v| v.parse().expect("slots")),
        expert_stats: None,
        prefill_rows: 1,
    };
    let mut driver = build_driver(
        &|stage| println!("stage: {stage}"),
        &pack,
        &wanted,
        actual_vocab,
        None,
    )
    .expect("build the driver");
    println!(
        "load: {:.1} s; context {context}; {} prompts",
        opened.elapsed().as_secs_f64(),
        records.len()
    );

    let (mut feed_ms, mut step_ms) = (60.0f64, 50.0f64);
    let mut summaries = Vec::new();
    // The run in the oracle's schema, for its comparer: one record per prompt, the first
    // `full_steps` full rows beside it.
    let mut engine_prompts = Vec::new();
    let (mut all_steps, mut all_agree) = (0usize, 0usize);
    let (mut all_kl20, mut all_klfull) = (Vec::new(), Vec::new());
    let dir = oracle_path.parent().expect("dir");
    for record in records {
        let id = safe_name("prompt id", record["id"].as_str().expect("id"));
        let ids = |key: &str| -> Vec<u32> {
            record[key]
                .as_array()
                .unwrap_or_else(|| panic!("no {key}"))
                .iter()
                .map(|v| v.as_u64().expect("an id") as u32)
                .collect()
        };
        let (prompt, generated) = (ids("prompt_ids"), ids("generated"));
        let oracle_steps = record["steps"].as_array().expect("steps");
        let count = oracle_steps.len();
        if count == 0 {
            println!("skipped {id}: the oracle has no steps for it");
            summaries.push(serde_json::json!({"id": id, "skipped": true, "estimate_sec": 0}));
            if let Some(out) = var("FIFI_FLASH_NEXT_DRIVER_OUT") {
                write_follow_reports(
                    &PathBuf::from(out),
                    context,
                    &summaries,
                    &engine_prompts,
                    (all_steps, all_agree, &all_kl20, &all_klfull),
                    false,
                );
            }
            continue;
        }
        let estimate = (prompt.len() as f64 * feed_ms + count as f64 * step_ms) * 1.25 / 1e3;
        if opened.elapsed().as_secs_f64() + estimate > budget {
            println!(
                "skipped {id}: about {estimate:.0} s more would pass the {budget:.0} s budget"
            );
            summaries
                .push(serde_json::json!({"id": id, "skipped": true, "estimate_sec": estimate}));
            if let Some(out) = var("FIFI_FLASH_NEXT_DRIVER_OUT") {
                write_follow_reports(
                    &PathBuf::from(out),
                    context,
                    &summaries,
                    &engine_prompts,
                    (all_steps, all_agree, &all_kl20, &all_klfull),
                    false,
                );
            }
            continue;
        }
        let rows = read_npy(&dir.join(record["full_logits"]["file"].as_str().expect("file")));
        let width = record["full_logits"]["width"].as_u64().expect("width") as usize;
        let stored_steps = record["full_logits"]["steps"].as_u64().expect("steps") as usize;
        let full_steps = stored_steps.min(count);
        assert!(
            rows.len() >= full_steps * width && width >= actual_vocab,
            "{id}: the stored full rows ({} values) do not hold {full_steps} rows of {width}",
            rows.len()
        );
        driver.reset().expect("reset");
        let began = std::time::Instant::now();
        let (last, head) = prompt.split_last().expect("a prompt");
        for &token in head {
            driver.feed(token).expect("feed a prompt token");
        }
        feed_ms = began.elapsed().as_secs_f64() * 1e3 / head.len().max(1) as f64;
        let began = std::time::Instant::now();
        let mut steps: Vec<FollowStep> = Vec::with_capacity(count);
        let mut engine_steps = Vec::with_capacity(count);
        let mut full_rows: Vec<f32> = Vec::new();
        for i in 0..count {
            let input = if i == 0 { *last } else { generated[i - 1] };
            let row = driver
                .logits(input)
                .unwrap_or_else(|e| panic!("{id} step {i}: {e:?}"));
            assert_eq!(row.len(), width, "{id} step {i}: the logits row width");
            let ours = &row[..actual_vocab];
            assert!(
                ours.iter().all(|v| v.is_finite()),
                "{id} step {i}: non-finite logits"
            );
            let ours_top = top(ours, 20);
            let order = &ours_top[..2];
            let oracle = &oracle_steps[i];
            let oracle_top = oracle["top"].as_array().expect("top");
            let oracle_ids: Vec<usize> = oracle_top
                .iter()
                .map(|t| t[0].as_u64().expect("id") as usize)
                .collect();
            let oracle_logits: Vec<f64> = oracle_top
                .iter()
                .map(|t| t[1].as_f64().expect("logit"))
                .collect();
            let our_at_oracle: Vec<f64> = oracle_ids.iter().map(|&t| f64::from(ours[t])).collect();
            let our_top20: Vec<usize> = ours_top.iter().map(|t| t.0).collect();
            engine_steps.push(serde_json::json!({
                "i": i, "token": generated[i], "argmax": order[0].0,
                "top": ours_top.iter().map(|(t, v)| serde_json::json!([t, v])).collect::<Vec<_>>(),
                "margin": f64::from(order[0].1 - order[1].1),
            }));
            if i < full_steps {
                full_rows.extend_from_slice(&row);
            }
            let kl_full = (i < full_steps).then(|| {
                let oracle_row = &rows[i * width..i * width + actual_vocab];
                kl_of_logits(&widen(oracle_row), &widen(ours))
            });
            steps.push(FollowStep {
                oracle_argmax: oracle["argmax"].as_u64().expect("argmax") as usize,
                our_argmax: order[0].0,
                oracle_margin: oracle["margin"].as_f64().expect("margin"),
                our_margin: f64::from(order[0].1 - order[1].1),
                our_logit_oracle_token: f64::from(
                    ours[oracle["argmax"].as_u64().unwrap() as usize],
                ),
                our_logit_top: f64::from(order[0].1),
                kl_top20: kl_of_logits(&oracle_logits, &our_at_oracle),
                kl_full,
                overlap: oracle_ids.iter().filter(|t| our_top20.contains(t)).count(),
            });
        }
        step_ms = began.elapsed().as_secs_f64() * 1e3 / count.max(1) as f64;
        if let Some(out) = var("FIFI_FLASH_NEXT_DRIVER_OUT") {
            let parts = PathBuf::from(out).join("engine-follow.parts");
            std::fs::create_dir_all(&parts).expect("parts dir");
            write_npy(
                &parts.join(format!("{id}.logits.npy")),
                &full_rows,
                full_steps,
                width,
            );
        }
        engine_prompts.push(serde_json::json!({
            "id": id, "kind": record["kind"], "mode": "follow", "prompt_ids": prompt,
            "prompt_len": prompt.len(), "generated": generated, "forced_tokens": generated,
            "n_generated": count, "steps": engine_steps,
            "full_logits": {"file": format!("engine-follow.parts/{id}.logits.npy"),
                            "steps": full_steps, "width": width, "dtype": "float32"},
        }));
        let agree = steps
            .iter()
            .filter(|s| s.oracle_argmax == s.our_argmax)
            .count();
        let (kl20_mean, kl20_max) = mean_max(&steps.iter().map(|s| s.kl_top20).collect::<Vec<_>>());
        let full: Vec<f64> = steps.iter().filter_map(|s| s.kl_full).collect();
        let (klf_mean, klf_max) = mean_max(&full);
        let overlap = steps.iter().map(|s| s.overlap).sum::<usize>() as f64 / count as f64;
        println!(
            "{id}: {count} steps, top-1 {agree}/{count} ({:.4}), KL top-20 mean {kl20_mean:.3e} max {kl20_max:.3e}, \
             KL full-vocab ({} rows) mean {klf_mean:.3e} max {klf_max:.3e}, top-20 overlap mean {overlap:.2}/20, \
             prompt {:.1} ms/token, steps {step_ms:.1} ms/token",
            agree as f64 / count as f64,
            full.len(),
            feed_ms
        );
        let mut flips = Vec::new();
        for (i, s) in steps.iter().enumerate() {
            if s.oracle_argmax != s.our_argmax {
                println!(
                    "  disagreement {id} step {i}: oracle {} (margin {:.4}, our logit {:.4}) vs ours {} (margin {:.4}, our logit {:.4})",
                    s.oracle_argmax,
                    s.oracle_margin,
                    s.our_logit_oracle_token,
                    s.our_argmax,
                    s.our_margin,
                    s.our_logit_top
                );
                flips.push(serde_json::json!({
                    "step": i, "oracle": s.oracle_argmax, "ours": s.our_argmax,
                    "oracle_margin": s.oracle_margin, "our_margin": s.our_margin,
                    "our_logit_of_oracle_token": s.our_logit_oracle_token,
                    "our_logit_top": s.our_logit_top,
                }));
            }
        }
        all_steps += count;
        all_agree += agree;
        all_kl20.extend(steps.iter().map(|s| s.kl_top20));
        all_klfull.extend(full.iter().copied());
        summaries.push(serde_json::json!({
            "id": id, "steps": count, "top1_agree": agree, "kl_top20_mean": kl20_mean,
            "kl_top20_max": kl20_max, "kl_full_rows": full.len(), "kl_full_mean": klf_mean,
            "kl_full_max": klf_max, "top20_overlap_mean": overlap, "disagreements": flips,
            "prompt_ms_per_token": feed_ms, "step_ms_per_token": step_ms,
        }));
        if let Some(out) = var("FIFI_FLASH_NEXT_DRIVER_OUT") {
            write_follow_reports(
                &PathBuf::from(out),
                context,
                &summaries,
                &engine_prompts,
                (all_steps, all_agree, &all_kl20, &all_klfull),
                false,
            );
        }
    }
    let (m20, x20) = mean_max(&all_kl20);
    let (mf, xf) = mean_max(&all_klfull);
    println!(
        "ALL: {all_agree}/{all_steps} top-1 ({:.4}); KL top-20 mean {m20:.3e} max {x20:.3e} over {} steps; \
         KL full-vocab mean {mf:.3e} max {xf:.3e} over {} rows; {:.0} s",
        all_agree as f64 / all_steps.max(1) as f64,
        all_kl20.len(),
        all_klfull.len(),
        opened.elapsed().as_secs_f64()
    );
    if let Some(out) = var("FIFI_FLASH_NEXT_DRIVER_OUT") {
        write_follow_reports(
            &PathBuf::from(out),
            context,
            &summaries,
            &engine_prompts,
            (all_steps, all_agree, &all_kl20, &all_klfull),
            engine_prompts.len() == oracle["prompts"].as_array().unwrap().len(),
        );
    }
}

/// Writes `bytes` to `path` through a temporary name in the same directory and a rename, so a kill
/// during the write leaves the previous file whole.
fn write_atomic(path: &Path, bytes: &[u8]) {
    let name = path.file_name().expect("a file name").to_string_lossy();
    let temp = path.with_file_name(format!(".{name}.tmp"));
    std::fs::write(&temp, bytes).unwrap_or_else(|e| panic!("{}: {e}", temp.display()));
    std::fs::rename(&temp, path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// Writes `follow.json` (the summaries) and `engine-follow.json` (the run in the oracle's schema).
/// Called after every prompt, so a run that dies keeps what it finished; `complete` is false until
/// the last prompt.
fn write_follow_reports(
    out: &Path,
    context: u32,
    summaries: &[serde_json::Value],
    engine_prompts: &[serde_json::Value],
    (steps, agree, kl20, klfull): (usize, usize, &[f64], &[f64]),
    complete: bool,
) {
    let (m20, x20) = mean_max(kl20);
    let (mf, xf) = mean_max(klfull);
    std::fs::create_dir_all(out).expect("out dir");
    write_atomic(
        &out.join("follow.json"),
        &serde_json::to_vec_pretty(&serde_json::json!({
            "mode": "teacher-forced", "context": context, "prompts": summaries,
            "all": {"steps": steps, "top1_agree": agree, "kl_top20_mean": m20,
                    "kl_top20_max": x20, "kl_full_mean": mf, "kl_full_max": xf},
        }))
        .expect("json"),
    );
    write_atomic(
        &out.join("engine-follow.json"),
        &serde_json::to_vec(&serde_json::json!({
            "schema": "fifi-flash-next-oracle/1", "role": "engine",
            "complete": complete, "prompts": engine_prompts,
        }))
        .expect("json"),
    );
}

// ---------------------------------------------------------------------------------------------
// The single-row EXL3 kernel receipt: one process, one configuration.

/// One greedy run of `STEPS` tokens on the oracle prompt, with the sha256 of every step's whole
/// logits row and the wall time of every step, written to
/// `$OUT/ab-<CONFIG>-r<ROUND>.json`. The EXL3 kernel switches are process-wide and read once, so
/// a script runs this test once per configuration, with the `FIFI_EXL3_*` variables of that
/// configuration in the environment, and compares the hashes across the runs.
///
/// Environment: `FIFI_FLASH_NEXT_EXL3_PACK`, `FIFI_FLASH_NEXT_DRIVER_ORACLE`, `_PROMPT` (default
/// `chat-short`), `_STEPS` (default 64), `_CONTEXT` (default 512), `_OUT`, `_CONFIG` (a label) and
/// `_ROUND`.
#[test]
#[ignore = "needs a GPU (about 21 GiB) and the full pack; run by the kernel receipt script"]
fn single_row_kernel_receipt_run() {
    use sha2::{Digest, Sha256};

    assert!(
        device_present("Flash-Next kernel receipt"),
        "a GPU is required"
    );
    let pack = PathBuf::from(var("FIFI_FLASH_NEXT_EXL3_PACK").expect("FIFI_FLASH_NEXT_EXL3_PACK"));
    let oracle_path =
        PathBuf::from(var("FIFI_FLASH_NEXT_DRIVER_ORACLE").expect("FIFI_FLASH_NEXT_DRIVER_ORACLE"));
    let config = safe_name(
        "config",
        &var("FIFI_FLASH_NEXT_DRIVER_CONFIG").expect("FIFI_FLASH_NEXT_DRIVER_CONFIG"),
    );
    let round = safe_name(
        "round",
        &var("FIFI_FLASH_NEXT_DRIVER_ROUND").unwrap_or_else(|| "1".into()),
    );
    let out = PathBuf::from(var("FIFI_FLASH_NEXT_DRIVER_OUT").expect("FIFI_FLASH_NEXT_DRIVER_OUT"));
    let steps: usize =
        var("FIFI_FLASH_NEXT_DRIVER_STEPS").map_or(64, |v| v.parse().expect("steps"));
    let context: u32 =
        var("FIFI_FLASH_NEXT_DRIVER_CONTEXT").map_or(512, |v| v.parse().expect("context"));
    let wanted_id = var("FIFI_FLASH_NEXT_DRIVER_PROMPT").unwrap_or_else(|| "chat-short".into());
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&oracle_path).expect("oracle json")).expect("json");
    let actual_vocab = oracle["vocab"]["actual_vocab"]
        .as_u64()
        .expect("actual_vocab") as usize;
    let record = oracle["prompts"]
        .as_array()
        .expect("prompts")
        .iter()
        .find(|p| p["id"] == wanted_id.as_str())
        .unwrap_or_else(|| panic!("no prompt {wanted_id}"));
    let prompt: Vec<u32> = record["prompt_ids"]
        .as_array()
        .expect("prompt_ids")
        .iter()
        .map(|v| v.as_u64().expect("an id") as u32)
        .collect();
    assert!(
        prompt.len() + steps < context as usize,
        "the context is too small"
    );
    // The engine key path of the warm-start slice: FIFI_FLASH_NEXT_DRIVER_STATS points the
    // driver's expert_stats at a file; a missing one is the cold start, and the bank's drop
    // writes the usage dump there for the next run.
    let stats = var("FIFI_FLASH_NEXT_DRIVER_STATS").map(std::path::PathBuf::from);
    let wanted = CudaSettings {
        context,
        kv_format: KvFormatName::F16,
        slots: var("FIFI_FLASH_NEXT_DRIVER_SLOTS").map(|v| v.parse().expect("slots")),
        expert_stats: stats,
        prefill_rows: 1,
    };
    let mut driver = build_driver(&|_| {}, &pack, &wanted, actual_vocab, None).expect("driver");
    let kernel = driver.single_row_kernel_name().expect("kernel name");
    let env = |name: &str| std::env::var(name).ok();
    println!(
        "config {config} round {round}: single-row kernel {kernel}; SINGLE_GATHER {:?} DP4A {:?} INDEXED_PAIRS {:?}",
        env("FIFI_EXL3_SINGLE_GATHER"),
        env("FIFI_EXL3_DP4A"),
        env("FIFI_EXL3_INDEXED_PAIRS")
    );
    let (last, head) = prompt.split_last().expect("a prompt");
    for &token in head {
        driver.feed(token).expect("feed a prompt token");
    }
    let mut input = *last;
    let mut records = Vec::with_capacity(steps);
    // The launch proof: every decode step adds the same launches to the counters. The grouped
    // QSA k and v projections are one wide launch per QSA layer (12), the shared expert two per
    // layer (96); gather must stay flat: the shared expert rides the context default, not the
    // gather kernel.
    let mut before = driver.kernel_launches();
    let mut step_wide: Vec<u64> = Vec::with_capacity(steps);
    for step in 0..steps {
        let began = std::time::Instant::now();
        let row = driver
            .logits(input)
            .unwrap_or_else(|e| panic!("step {step}: {e:?}"));
        let wall = began.elapsed();
        let after = driver.kernel_launches();
        let wide = after.wide1 - before.wide1;
        assert_eq!(
            after.gather, before.gather,
            "step {step}: a grouped launch ran on the gather kernel"
        );
        if let Some(&first) = step_wide.first() {
            assert_eq!(
                wide, first,
                "step {step}: {wide} wide launches, step 0 ran {first}"
            );
        }
        // Ungrouped k and v would add 12 launches and still be constant: pin the count. 253 =
        // 145 singles (36x3 GDN + 12x3 QSA + lm_head) + 12 QSA grouped + 96 shared expert.
        assert_eq!(
            wide, 253,
            "step {step}: {wide} wide launches, the receipt config runs 253"
        );
        step_wide.push(wide);
        before = after;
        assert!(
            row[..actual_vocab].iter().all(|v| v.is_finite()),
            "step {step}: non-finite logits"
        );
        let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        let hash = format!("{:x}", Sha256::digest(&bytes));
        if step == 0
            && let Some(path) = var("FIFI_FLASH_NEXT_DRIVER_STEP0_BIN")
        {
            std::fs::write(&path, &bytes).expect("write step0 logits");
        }
        let pick = argmax(&row[..actual_vocab]);
        records.push(serde_json::json!({
            "i": step, "input": input, "token": pick, "sha256": hash, "ms": ms(wall),
            "misses": driver.bank_misses(),
            "wide1_launches": wide,
        }));
        input = pick as u32;
    }
    std::fs::create_dir_all(&out).expect("out dir");
    // The warm-start e2e: with the engine key set, the bank's drop must have written the
    // usage dump to exactly that path, in the format the next start's reader takes.
    drop(driver);
    if let Some(stats) = wanted.expert_stats.as_ref() {
        let body =
            std::fs::read(stats).unwrap_or_else(|e| panic!("usage dump {}: {e}", stats.display()));
        let parsed: serde_json::Value = serde_json::from_slice(&body).expect("dump json");
        assert_eq!(parsed["schema"], "fifi-usage/1");
        assert_eq!(
            parsed["layers"].as_object().map(|m| m.len()),
            Some(48),
            "one layer row per bank layer"
        );
    }
    let path = out.join(format!("ab-{config}-r{round}.json"));
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "config": config, "round": round, "kernel": kernel, "prompt": wanted_id,
            "env": {"FIFI_EXL3_SINGLE_GATHER": env("FIFI_EXL3_SINGLE_GATHER"),
                    "FIFI_EXL3_DP4A": env("FIFI_EXL3_DP4A"),
                    "FIFI_EXL3_INDEXED_PAIRS": env("FIFI_EXL3_INDEXED_PAIRS")},
            "steps": records,
        }))
        .expect("json"),
    )
    .expect("write the run");
    println!("wrote {}", path.display());
    println!(
        "launches per step: wide1 {} (grouped QSA k+v 12 + shared expert 96 + singles), gather 0",
        step_wide.first().copied().unwrap_or(0)
    );
}
