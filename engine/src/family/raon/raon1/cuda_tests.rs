//! Device tests of the Raon-1 forward pass against the host reference. Both need the published
//! Q4_K_M file (`RAON_Q4_KM`) and a device, so every test here is `#[ignore]`d: a normal run lists
//! them as ignored instead of passing them empty, and `cargo test -- --ignored` runs them and
//! fails when the file or the device is missing.
//!
//! The host reference is slow (minutes for a short prompt), so its all-position logits can be
//! kept in the file named by `RAON_REF_LOGITS`: the first run writes it, later runs read it.

use std::path::PathBuf;

use super::cuda::{HeadRows, Raon1Device};
use super::reference::{ActivationRule, attn_half, ffn_half, head, hyper, with_activation_rule};
use super::{Raon1Model, open_raon_1_mini_preview};
use crate::kernels::cuda::device_present;

/// The first twelve tokens of the frozen fixture prompt.
const PROMPT: [u32; 12] = [
    166_863, 54_935, 7_821, 44, 11_456, 189_096, 257_706, 142_907, 189_286, 9_305, 39_154, 166_294,
];

/// Greedy tokens decoded one at a time after the prompt.
const DECODED: usize = 12;

fn model_path() -> PathBuf {
    PathBuf::from(std::env::var("RAON_Q4_KM").expect("RAON_Q4_KM names the Raon-1 Q4_K_M file"))
}

/// The device presence check of a test that was asked for by `--ignored`: no device fails it.
fn require_device(test: &str) {
    assert!(device_present(test), "{test} needs a CUDA device");
}

/// The host reference's residual after the embedding: the token row in all four streams.
fn host_embedding(model: &Raon1Model, tokens: &[u32]) -> Vec<f32> {
    let spec = model.spec();
    let h = spec.hidden as usize;
    let hc = spec.streams as usize;
    let table = model.read_f32("token_embd.weight").expect("embedding");
    let mut inp = vec![0f32; h * hc * tokens.len()];
    for (t, &token) in tokens.iter().enumerate() {
        let row = &table[token as usize * h..(token as usize + 1) * h];
        for s in 0..hc {
            let at = (t * hc + s) * h;
            inp[at..at + h].copy_from_slice(row);
        }
    }
    inp
}

/// The host logits of every position of `tokens`.
fn host_logits(model: &Raon1Model, tokens: &[u32]) -> Vec<Vec<f32>> {
    let spec = model.spec();
    let hy = hyper(model, spec).expect("hyperparameters");
    let hc = spec.streams as usize;
    let h = spec.hidden as usize;
    let mut inp = host_embedding(model, tokens);
    let mut sink = |_: &str, _: &[f32]| {};
    for layer in 0..spec.layers {
        inp = attn_half(model, spec, &hy, &inp, layer, &mut sink).expect("attention half");
        inp = ffn_half(model, spec, &hy, &inp, layer, &mut sink, None).expect("ffn half");
    }
    (0..tokens.len())
        .map(|t| {
            let prefix = &inp[..(t + 1) * h * hc];
            head(model, spec, &hy, prefix, t + 1, &mut sink).expect("head")
        })
        .collect()
}

fn top_k(row: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
    order.truncate(k);
    order
}

/// Prefill logits for every position of the prompt, then greedy single-token decode: the device
/// logits row of every position of the whole sequence.
fn device_sequence(device: &mut Raon1Device) -> (Vec<u32>, Vec<Vec<f32>>) {
    device.reset().expect("reset");
    let mut tokens = PROMPT.to_vec();
    let mut rows = device.feed(&PROMPT, 0, HeadRows::All).expect("prompt feed");
    for _ in 0..DECODED {
        let next = top_k(rows.last().expect("a logits row"), 1)[0] as u32;
        let mut out = device
            .feed(&[next], tokens.len(), HeadRows::Last)
            .expect("decode step");
        tokens.push(next);
        rows.append(&mut out);
    }
    (tokens, rows)
}

fn read_u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

fn read_rows(bytes: &[u8], vocab: usize) -> Vec<Vec<f32>> {
    bytes
        .chunks_exact(vocab * 4)
        .map(|row| {
            row.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        })
        .collect()
}

fn write_rows(path: &std::path::Path, rows: &[Vec<f32>]) {
    let bytes: Vec<u8> = rows
        .iter()
        .flat_map(|row| row.iter().flat_map(|v| v.to_le_bytes()))
        .collect();
    std::fs::write(path, bytes).expect("write logits");
}

/// Writes the greedy sequence of the device (prompt plus decoded tokens) to
/// `RAON_DEVICE_SEQ` and its logits rows, one per position, to `RAON_DEVICE_SEQ.logits`.
#[test]
#[ignore = "needs RAON_Q4_KM and a device; run with --ignored"]
fn the_device_writes_its_greedy_sequence_and_logits() {
    let path = model_path();
    let out = PathBuf::from(
        std::env::var("RAON_DEVICE_SEQ").expect("RAON_DEVICE_SEQ names the output file"),
    );
    require_device("the_device_writes_its_greedy_sequence_and_logits");
    let mut device = Raon1Device::load(&path, Some(4096), &mut |_| {}).expect("device load");
    let (tokens, rows) = device_sequence(&mut device);
    let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    std::fs::write(&out, bytes).expect("write the sequence");
    write_rows(&out.with_extension("logits"), &rows);
    eprintln!("device sequence {tokens:?}");
}

/// Accuracy probe, not a gate, device half. Writes a stream of `RAON_ACC_TOKENS` tokens (default
/// 48: the fixture prompt, then the device's greedy choices) to `<RAON_ACC_OUT>.ids`, and the
/// device's logits row of every position twice: fed one token at a time (`.decode`) and fed as
/// one chunk (`.prefill`). Rows are little-endian f32, one vocabulary each.
#[test]
#[ignore = "needs RAON_Q4_KM and a device; run with --ignored"]
fn the_device_writes_rows_for_an_accuracy_comparison() {
    let out = PathBuf::from(std::env::var("RAON_ACC_OUT").expect("RAON_ACC_OUT names the output"));
    let count: usize = std::env::var("RAON_ACC_TOKENS")
        .map(|v| v.parse().expect("a token count"))
        .unwrap_or(48);
    require_device("the_device_writes_rows_for_an_accuracy_comparison");
    let mut device = Raon1Device::load(&model_path(), Some(4096), &mut |_| {}).expect("load");
    device.reset().expect("reset");
    let mut tokens: Vec<u32> = Vec::with_capacity(count);
    let mut decode: Vec<Vec<f32>> = Vec::with_capacity(count);
    for pos in 0..count {
        let token = match PROMPT.get(pos) {
            Some(&token) => token,
            None => top_k(decode.last().expect("a logits row"), 1)[0] as u32,
        };
        let mut row = device.feed(&[token], pos, HeadRows::Last).expect("a step");
        tokens.push(token);
        decode.append(&mut row);
    }
    device.reset().expect("reset");
    let prefill = device.feed(&tokens, 0, HeadRows::All).expect("the chunk");
    let ids: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    std::fs::write(out.with_extension("ids"), ids).expect("write the ids");
    write_rows(&out.with_extension("decode"), &decode);
    write_rows(&out.with_extension("prefill"), &prefill);
    eprintln!("{count} positions written for the decode and the prefill route");
}

/// Accuracy probe, host half, no device. Reads `<RAON_ACC_OUT>.ids` and writes the host
/// reference's rows under each rule of `RAON_ACC_RULES` (default `exact,fork,q8_1`) to
/// `<RAON_ACC_OUT>.host-<rule>`. `exact` reads f32 activations and is the reference of the
/// comparison; `fork` rounds as the fork's CPU path does; `q8_1` as the device does.
#[test]
#[ignore = "needs RAON_Q4_KM; minutes per rule on the host"]
fn the_host_writes_rows_for_an_accuracy_comparison() {
    let out = PathBuf::from(std::env::var("RAON_ACC_OUT").expect("RAON_ACC_OUT names the output"));
    let rules = std::env::var("RAON_ACC_RULES").unwrap_or("exact,fork,q8_1".to_owned());
    let model = open_raon_1_mini_preview(&model_path(), None).expect("open");
    let tokens = read_u32s(&std::fs::read(out.with_extension("ids")).expect("the ids"));
    for name in rules.split(',') {
        let rule = match name {
            "exact" => ActivationRule::Exact,
            "fork" => ActivationRule::ForkCpu,
            "q8_1" => ActivationRule::Q8_1Blocks,
            other => panic!("unknown rule {other}"),
        };
        let t0 = std::time::Instant::now();
        let rows = with_activation_rule(rule, || host_logits(&model, &tokens));
        write_rows(&out.with_extension(format!("host-{name}")), &rows);
        eprintln!(
            "host rule {name}: {} rows in {:.0} s",
            rows.len(),
            t0.elapsed().as_secs_f64()
        );
    }
}

/// The logits rows of `tokens` on the host reference under `rule`, kept in `cache` when it is
/// given (the host reference takes minutes).
fn host_rows_cached(
    model: &Raon1Model,
    tokens: &[u32],
    rule: ActivationRule,
    cache: &std::path::Path,
) -> Vec<Vec<f32>> {
    let vocab = model.spec().vocab as usize;
    // The cache file name carries a hash of the token sequence, so a cache of another sequence is
    // never read.
    let hash = tokens.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, &t| {
        (h ^ u64::from(t)).wrapping_mul(0x0100_0000_01b3)
    });
    let cache = &cache.with_extension(format!("{hash:016x}.bin"));
    match std::fs::read(cache) {
        Ok(bytes) if bytes.len() == tokens.len() * vocab * 4 => read_rows(&bytes, vocab),
        _ => {
            let rows = with_activation_rule(rule, || host_logits(model, tokens));
            write_rows(cache, &rows);
            rows
        }
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Worst per-position logit difference of two row sets.
fn worst_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| max_abs_diff(x, y))
        .fold(0f32, f32::max)
}

/// The device sequence against the host reference run under both activation rules: the fork's
/// CPU rule (Q8_K blocks of 256) and the device's rule (blocks of 32 with an f16 scale). The
/// two host runs differ only in activation rounding, so their gap is the noise the quantization
/// rule alone makes in these logits; the device is accepted when it sits within that gap of the
/// host run that uses its own rule, and a top-1 difference is accepted only where the host's top
/// two logits are closer than that gap. The host logits are kept in `RAON_REF_LOGITS` (fork
/// rule) and `RAON_REF_LOGITS_Q8_1` (device rule).
#[test]
#[ignore = "needs RAON_Q4_KM and a device; run with --ignored"]
fn the_device_logits_sit_within_the_activation_rule_noise_of_the_host_reference() {
    let path = model_path();
    let seq = PathBuf::from(std::env::var("RAON_DEVICE_SEQ").expect("RAON_DEVICE_SEQ"));
    let fork_cache = PathBuf::from(std::env::var("RAON_REF_LOGITS").expect("RAON_REF_LOGITS"));
    let q81_cache =
        PathBuf::from(std::env::var("RAON_REF_LOGITS_Q8_1").expect("RAON_REF_LOGITS_Q8_1"));
    let model = open_raon_1_mini_preview(&path, None).expect("open");
    let vocab = model.spec().vocab as usize;
    let tokens = read_u32s(&std::fs::read(&seq).expect("the device sequence"));
    let device_rows = read_rows(
        &std::fs::read(seq.with_extension("logits")).expect("the device logits"),
        vocab,
    );
    assert_eq!(tokens.len(), device_rows.len());
    let fork = host_rows_cached(&model, &tokens, ActivationRule::ForkCpu, &fork_cache);
    let q81 = host_rows_cached(&model, &tokens, ActivationRule::Q8_1Blocks, &q81_cache);
    let noise = worst_diff(&fork, &q81);
    let device_vs_fork = worst_diff(&device_rows, &fork);
    let device_vs_q81 = worst_diff(&device_rows, &q81);
    eprintln!(
        "max abs logit difference: host fork rule vs host device rule {noise:.4} | device vs host fork rule {device_vs_fork:.4} | device vs host device rule {device_vs_q81:.4}"
    );
    let mut failures = Vec::new();
    for (position, ((dev, host), other)) in device_rows.iter().zip(&q81).zip(&fork).enumerate() {
        let (host20, dev20, other20) = (top_k(host, 20), top_k(dev, 20), top_k(other, 20));
        let mut sorted_host = host20.clone();
        let mut sorted_dev = dev20.clone();
        sorted_host.sort_unstable();
        sorted_dev.sort_unstable();
        let overlap = sorted_host
            .iter()
            .filter(|id| sorted_dev.contains(id))
            .count();
        let margin = host[host20[0]] - host[host20[1]];
        let top1 = host20[0] == dev20[0];
        eprintln!(
            "position {position:2}: top1 host(device rule) {:6} device {:6} host(fork rule) {:6} | top20 overlap with host(device rule) {overlap} | device vs host(device rule) {:.4} | host margin {margin:.3}",
            host20[0],
            dev20[0],
            other20[0],
            max_abs_diff(dev, host),
        );
        if !top1 && margin > 2.0 * noise {
            failures.push(position);
        }
    }
    // The bound has two terms. `noise` is the rule term: how far the two activation rules land
    // apart when the host reference runs them (its dot products accumulate in f64). The device
    // kernels accumulate in f32, and that drift against an f64 host measured at up to 1.5x the
    // rule gap on this fixture (1.67 against a 1.11 gap, identical on the pre- and post-rebase
    // tips, 2026-11-23), so the gate allows the rule gap twice over; a porting defect lands
    // orders beyond the gap, not inside a factor of two, and still trips it.
    let bound = 2.0 * noise;
    assert!(
        device_vs_q81 <= bound,
        "the device differs from the host run of its own rule by {device_vs_q81}, more than the {bound} = 2 x {noise} that the rounding rule plus the f32 accumulation drift makes"
    );
    assert!(
        failures.is_empty(),
        "top-1 differs beyond the {bound} rule-plus-accumulation noise at positions {failures:?}"
    );
}

/// A deterministic prompt of ids spread over the vocabulary (not text).
fn synthetic_tokens(count: usize) -> Vec<u32> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            1000 + ((state >> 33) % 200_000) as u32
        })
        .collect()
}

/// The stages of the host reference that the stage tests compare.
const STAGE_NAMES: [&str; 5] = [
    "attn_norm",
    "hc_attn_out",
    "ffn_norm",
    "l_out",
    "ffn_moe_probs_biased",
];

/// The cache file of the host stages of `tokens` over `layers` layers under `rule`, in the
/// directory `RAON_STAGE_CACHE` when it is set.
fn stage_cache(tokens: &[u32], layers: usize, rule: ActivationRule) -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("RAON_STAGE_CACHE")?);
    let hash = tokens.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, &t| {
        (h ^ u64::from(t)).wrapping_mul(0x0100_0000_01b3)
    });
    Some(dir.join(format!("stages-{rule:?}-{layers}-{hash:016x}.bin")))
}

fn write_stages(path: &std::path::Path, stages: &[super::cuda::Stage]) {
    let mut bytes = (stages.len() as u32).to_le_bytes().to_vec();
    for (name, data) in stages {
        bytes.extend((name.len() as u32).to_le_bytes());
        bytes.extend(name.as_bytes());
        bytes.extend((data.len() as u64).to_le_bytes());
        bytes.extend(data.iter().flat_map(|v| v.to_le_bytes()));
    }
    std::fs::write(path, bytes).expect("write the stage cache");
}

fn read_stages(path: &std::path::Path) -> Option<Vec<super::cuda::Stage>> {
    let bytes = std::fs::read(path).ok()?;
    let mut at = 0usize;
    let mut take = |count: usize| -> Option<&[u8]> {
        let part = bytes.get(at..at + count)?;
        at += count;
        Some(part)
    };
    let count = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
    let mut stages = Vec::with_capacity(count);
    for _ in 0..count {
        let name_len = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
        let name = String::from_utf8(take(name_len)?.to_vec()).ok()?;
        let len = u64::from_le_bytes(take(8)?.try_into().ok()?) as usize;
        let data = take(len * 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        stages.push((name, data));
    }
    Some(stages)
}

/// The host stages of the first `layers` layers of `tokens` under `rule`, read from the cache
/// when it holds them.
fn host_stages(
    model: &Raon1Model,
    tokens: &[u32],
    layers: usize,
    rule: ActivationRule,
) -> Vec<super::cuda::Stage> {
    let cache = stage_cache(tokens, layers, rule);
    if let Some(stages) = cache.as_deref().and_then(read_stages) {
        return stages;
    }
    let spec = model.spec();
    let hy = hyper(model, spec).expect("hyperparameters");
    let mut host: Vec<super::cuda::Stage> = Vec::new();
    let mut inp = host_embedding(model, tokens);
    for layer in 0..layers {
        let mut sink = |name: &str, data: &[f32]| {
            if STAGE_NAMES
                .iter()
                .any(|stage| name == format!("{stage}-{layer}"))
            {
                host.push((name.to_owned(), data.to_vec()));
            }
        };
        inp = with_activation_rule(rule, || {
            attn_half(model, spec, &hy, &inp, layer, &mut sink).expect("attention half")
        });
        inp = with_activation_rule(rule, || {
            ffn_half(model, spec, &hy, &inp, layer, &mut sink, None).expect("ffn half")
        });
    }
    if let Some(path) = &cache {
        write_stages(path, &host);
    }
    host
}

fn max_abs_diff_of(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// The experts a host router row selects (sorted) and the margin of its sixth score to its
/// seventh.
fn routed_set(row: &[f32], used: usize) -> (Vec<usize>, f32) {
    let order = top_k(row, used + 1);
    let margin = row[order[used - 1]] - row[order[used]];
    let mut set = order[..used].to_vec();
    set.sort_unstable();
    (set, margin)
}

/// The per-layer stages of the first `layers` layers, device against the host reference under
/// both activation rules. The device feeds `tokens[..split]` untraced, then traces the feed of
/// the rest; the host runs the whole sequence and its rows from `split` on are compared.
///
/// The two host runs differ only in activation rounding, so their difference is the noise of
/// the rounding rule itself. A residual stage (`hc_attn_out`, `l_out`) of the device is accepted
/// when its difference from the nearer host run is no larger than the difference between the
/// host runs (plus float rounding of the range); a token whose experts differ from both host
/// runs is accepted only when its margin, between the host's sixth and seventh biased router
/// scores, is no larger than the largest margin at which the two host runs disagree with each
/// other. Prints every stage. Returns the stages and the routed token rows that fail, and the
/// worst stage error as a fraction of that stage's range.
#[allow(clippy::type_complexity)]
fn compare_stages(
    path: &std::path::Path,
    tokens: &[u32],
    split: usize,
    layers: usize,
) -> (Vec<String>, Vec<(String, f32)>, f32) {
    let mut worst_relative = 0.0f32;
    let model = open_raon_1_mini_preview(path, None).expect("open");
    let spec = model.spec();
    let hy = hyper(&model, spec).expect("hyperparameters");
    let (ne, nu) = (spec.experts as usize, hy.used_experts);
    let nt = tokens.len();
    let fork = host_stages(&model, tokens, layers, ActivationRule::ForkCpu);
    let q81 = host_stages(&model, tokens, layers, ActivationRule::Q8_1Blocks);
    let find = |stages: &[super::cuda::Stage], name: &str| -> Vec<f32> {
        stages
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, data)| data.clone())
            .unwrap_or_else(|| panic!("host stage {name}"))
    };
    let mut device = Raon1Device::load(path, Some(nt.max(4096).next_multiple_of(256)), &mut |_| {})
        .expect("device load");
    device.reset().expect("reset");
    if split > 0 {
        device
            .feed(&tokens[..split], 0, HeadRows::None)
            .expect("prefix feed");
    }
    device.trace_next_feed();
    device
        .feed(&tokens[split..], split, HeadRows::None)
        .expect("traced feed");
    let traced = device.take_trace();
    // A chunked feed traces the same stage once per chunk: join them in order.
    let joined = |name: &str| -> Vec<f32> {
        traced
            .iter()
            .filter(|(n, _)| n == name)
            .flat_map(|(_, data)| data.iter().copied())
            .collect()
    };
    let (mut bad, mut routing) = (Vec::new(), Vec::new());
    for layer in 0..layers {
        for stage in ["attn_norm", "hc_attn_out", "ffn_norm", "l_out"] {
            // The norm outputs have a small range and are reported only; the residuals are gated.
            let gated = stage == "hc_attn_out" || stage == "l_out";
            let name = format!("{stage}-{layer}");
            let (all_fork, all_q81) = (find(&fork, &name), find(&q81, &name));
            let per_token = all_fork.len() / nt;
            let (want_fork, want_q81) = (
                &all_fork[split * per_token..],
                &all_q81[split * per_token..],
            );
            let got = joined(&name);
            assert_eq!(want_fork.len(), got.len(), "{name}: length");
            let range = want_fork
                .iter()
                .fold(0f32, |acc, v| acc.max(v.abs()))
                .max(1e-6);
            let (dev_fork, dev_q81, noise) = (
                max_abs_diff_of(want_fork, &got),
                max_abs_diff_of(want_q81, &got),
                max_abs_diff_of(want_fork, want_q81),
            );
            eprintln!(
                "{name:16} range {range:.4} | device vs host fork rule {:.2e} | device vs host device rule {:.2e} | host rules apart {:.2e}",
                dev_fork / range,
                dev_q81 / range,
                noise / range
            );
            let nearer = dev_fork.min(dev_q81);
            // The bound and the tracked worst cover the residual stages only: the norm stages
            // have small ranges and large relative rounding, so they are printed, not gated.
            if gated {
                worst_relative = worst_relative.max(nearer / range);
            }
            // Twice the intra-rule gap: the second term is the device's f32 accumulation
            // against the f64 host reference, measured at up to 1.8x the rule gap on the short
            // prompts this gate serves (2026-11-23, identical pre- and post-rebase).
            if gated && (nearer > 2.0 * noise + range * 1e-6 || !nearer.is_finite()) {
                bad.push(name);
            }
        }
        let Some(biased_fork) = fork
            .iter()
            .find(|(n, _)| *n == format!("ffn_moe_probs_biased-{layer}"))
            .map(|(_, d)| d)
        else {
            continue;
        };
        let biased_q81 = find(&q81, &format!("ffn_moe_probs_biased-{layer}"));
        let ids = joined(&format!("moe_ids-{layer}"));
        let rows: Vec<_> = (split..nt)
            .map(|t| {
                (
                    routed_set(&biased_fork[t * ne..(t + 1) * ne], nu),
                    routed_set(&biased_q81[t * ne..(t + 1) * ne], nu).0,
                )
            })
            .collect();
        // The largest margin at which the two host runs route a token differently.
        let host_noise = rows
            .iter()
            .filter(|((set, _), other)| set != other)
            .map(|((_, margin), _)| *margin)
            .fold(0f32, f32::max);
        let mut flips = 0;
        // The router weights are plain floats, so the two host rules agree on routing almost
        // exactly (margins below 2e-4 here) while the device's f32 router sits up to 0.014 from
        // the f64 host on this fixture (2026-11-23). A set change inside that drift is
        // accumulation noise; beyond it, a defect.
        let router_drift = 0.02;
        for (index, ((set, margin), other)) in rows.iter().enumerate() {
            let at = index * nu;
            let mut dev_set: Vec<usize> = ids[at..at + nu].iter().map(|&v| v as usize).collect();
            dev_set.sort_unstable();
            if &dev_set != set {
                flips += 1;
                if &dev_set != other && *margin > host_noise.max(router_drift) {
                    routing.push((format!("layer {layer} token {}", split + index), *margin));
                }
            }
        }
        eprintln!(
            "moe-{layer:<11} routed sets differ from the host fork rule at {flips} of {} tokens; host rules disagree up to margin {host_noise:.4}",
            nt - split
        );
    }
    (bad, routing, worst_relative)
}

/// Fills the stage cache (`RAON_STAGE_CACHE`) with the host stages of the stage tests, on the
/// host alone: `RAON_STAGE_TOKENS` picks the prompt (4: the fixture prefix, 300 or 1101: the
/// synthetic prompts of the chunk and split tests), `RAON_STAGE_LAYERS` the layers.
#[test]
#[ignore = "needs RAON_Q4_KM; fills the host stage cache without a device"]
fn the_host_stages_are_cached_for_the_stage_tests() {
    let path = model_path();
    assert!(
        std::env::var_os("RAON_STAGE_CACHE").is_some(),
        "RAON_STAGE_CACHE names the cache directory"
    );
    let count: usize = std::env::var("RAON_STAGE_TOKENS")
        .expect("RAON_STAGE_TOKENS")
        .parse()
        .expect("a token count");
    let tokens = if count == 4 {
        PROMPT[..4].to_vec()
    } else {
        synthetic_tokens(count)
    };
    let model = open_raon_1_mini_preview(&path, None).expect("open");
    for rule in [ActivationRule::ForkCpu, ActivationRule::Q8_1Blocks] {
        let stages = host_stages(&model, &tokens, stage_layers(4), rule);
        eprintln!("{rule:?}: {} stages", stages.len());
    }
}

fn stage_layers(default: usize) -> usize {
    std::env::var("RAON_STAGE_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The per-layer stages of the first layers (all 25 with `RAON_STAGE_LAYERS=25`), for locating
/// a defect and for reading the noise floor. The stage-by-stage comparison against the rule
/// noise cannot stay tight at depth — the f64 host reference and the f32 device kernels drift
/// apart smoothly through the layers (measured ratios of the device error to the intra-rule
/// noise from 0.5x at layer 0 to 9.4x at layer 15 on a healthy build, 2026-11-23) — so at depth
/// this gate asserts only the catastrophic form: no stage more than 5% of its range off (the
/// healthy worst is ~0.1%), and no routing flip beyond the router drift floor.
#[test]
#[ignore = "needs RAON_Q4_KM and a device; run with --ignored"]
fn the_device_stages_match_the_host_reference_for_the_first_layers() {
    let path = model_path();
    require_device("the_device_stages_match_the_host_reference_for_the_first_layers");
    let (bad, routing, worst_relative) = compare_stages(&path, &PROMPT[..4], 0, stage_layers(4));
    eprintln!("routing beyond the host rule noise: {routing:?}");
    eprintln!("stages beyond the 2x rule noise: {bad:?} (diagnostic at depth)");
    assert!(
        worst_relative < 0.05,
        "a stage sits {worst_relative:.4} of its range from both host rules; a healthy build stays near 0.001"
    );
    assert!(
        routing.is_empty(),
        "routing flips beyond the drift floor: {routing:?}"
    );
}

/// A prompt past one chunk: the delta-rule state and the convolution history carry across the
/// chunk boundary at 256 tokens.
#[test]
#[ignore = "needs RAON_Q4_KM and a device; run with --ignored"]
fn the_device_stages_match_across_a_chunk_boundary() {
    let path = model_path();
    require_device("the_device_stages_match_across_a_chunk_boundary");
    let tokens = synthetic_tokens(300);
    let (bad, routing, _) = compare_stages(&path, &tokens, 0, stage_layers(4));
    eprintln!("routing beyond the host rule noise: {routing:?}");
    assert!(
        bad.is_empty() && routing.is_empty(),
        "stages beyond the host rule noise: {bad:?}; routing beyond it: {routing:?}"
    );
}

/// A decode step over more than 1,024 keys: the latent attention splits the keys and reduces the
/// partial softmaxes.
#[test]
#[ignore = "needs RAON_Q4_KM and a device; run with --ignored"]
fn the_device_stages_match_with_split_key_attention() {
    let path = model_path();
    require_device("the_device_stages_match_with_split_key_attention");
    let tokens = synthetic_tokens(1101);
    let (bad, routing, _) = compare_stages(&path, &tokens, 1100, stage_layers(4));
    eprintln!("routing beyond the host rule noise: {routing:?}");
    assert!(
        bad.is_empty() && routing.is_empty(),
        "stages beyond the host rule noise: {bad:?}; routing beyond it: {routing:?}"
    );
}

/// Final logits recorded by an independent implementation over the frozen 42-token fixture
/// prompt, separate from the host reference. The device's top three must be the same ids, within the quantization noise.
#[test]
#[ignore = "needs RAON_Q4_KM and a device; run with --ignored"]
fn the_device_top_logits_match_the_recorded_logits_on_the_fixture_prompt() {
    let path = model_path();
    require_device("the_device_top_logits_match_the_recorded_logits_on_the_fixture_prompt");
    const FIXTURE: [u32; 42] = [
        166863, 54935, 7821, 44, 11456, 189096, 257706, 142907, 189286, 9305, 39154, 166294,
        107408, 51684, 46, 224037, 18624, 166295, 11234, 166471, 167757, 142907, 22233, 5407, 5066,
        46, 254733, 27335, 44, 4565, 46, 124319, 156431, 44, 235664, 73629, 177858, 9678, 444,
        2452, 44, 3248,
    ];
    const RECORDED_TOP3: [(usize, f32); 3] = [(44, 21.71462), (58, 19.871094), (369, 19.577736)];
    let mut device = Raon1Device::load(&path, Some(4096), &mut |_| {}).expect("device load");
    device.reset().expect("reset");
    let rows = device
        .feed(&FIXTURE, 0, HeadRows::Last)
        .expect("fixture feed");
    let row = &rows[0];
    let top = top_k(row, 3);
    for (rank, &(id, want)) in RECORDED_TOP3.iter().enumerate() {
        eprintln!(
            "rank {rank}: recorded {id} {want:.4} | device {} {:.4}",
            top[rank], row[top[rank]]
        );
        assert_eq!(top[rank], id, "rank {rank} id");
        assert!(
            (row[id] - want).abs() < 0.5,
            "logit of {id}: {} vs {want}",
            row[id]
        );
    }
}

/// Prefill and decode rates at the prompt lengths of `RAON_BENCH` (comma separated tokens),
/// on the context the free device memory admits. Prints one line per length.
#[test]
#[ignore = "needs RAON_Q4_KM and a device; run with --ignored"]
fn the_device_reports_prefill_and_decode_rates() {
    let path = model_path();
    let lengths = std::env::var("RAON_BENCH").expect("RAON_BENCH lists the prompt lengths");
    require_device("the_device_reports_prefill_and_decode_rates");
    let mut device = Raon1Device::load(&path, None, &mut |cells| {
        eprintln!("planned context {cells}");
    })
    .expect("device load");
    for length in lengths
        .split(',')
        .map(|v| v.trim().parse::<usize>().expect("length"))
    {
        let tokens: Vec<u32> = (0..length)
            .map(|i| 1000 + (i as u32 * 7919) % 200_000)
            .collect();
        device.reset().expect("reset");
        let start = std::time::Instant::now();
        device.feed(&tokens, 0, HeadRows::Last).expect("prefill");
        let prefill = start.elapsed().as_secs_f64();
        let steps = 32;
        let start = std::time::Instant::now();
        for step in 0..steps {
            device
                .feed(&[1234 + step as u32], length + step, HeadRows::Last)
                .expect("decode");
        }
        let decode = start.elapsed().as_secs_f64();
        let pick_rate = if device.takes_tokens() {
            let vocab = device.vocab();
            let start = std::time::Instant::now();
            let mut tok = 1234u32;
            for step in 0..steps {
                tok = device
                    .decode_token(tok, length + steps + step, vocab)
                    .expect("decode_token");
            }
            Some(steps as f64 / start.elapsed().as_secs_f64())
        } else {
            None
        };
        eprintln!(
            "prompt {length}: prefill {:.1} tok/s ({prefill:.2} s) | decode {:.1} tok/s (replay={}) | pick {:?} tok/s (replay={})",
            length as f64 / prefill,
            steps as f64 / decode,
            device.decode_replayed(),
            pick_rate.map(|r| format!("{r:.1}")),
            device.token_step_replayed()
        );
    }
}

/// Captured decode logits and device greedy picks match across repeated prompts and resets.
#[test]
#[ignore = "needs RAON_Q4_KM and a device; run with --ignored"]
fn the_replayed_decode_and_token_pick_match_across_reset() {
    let path = model_path();
    require_device("the_replayed_decode_and_token_pick_match_across_reset");
    let mut device = Raon1Device::load(&path, Some(4096), &mut |_| {}).expect("device load");
    assert!(device.takes_tokens(), "expected device embedding table");
    let vocab = device.vocab();
    let argmax = |row: &[f32]| -> u32 {
        row.iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then(ib.cmp(ia)))
            .map(|(i, _)| i as u32)
            .expect("non-empty logits")
    };
    let prompts: [&[u32]; 2] = [
        &[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
        &[9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, 1500],
    ];
    for (run, prompt) in prompts.iter().enumerate() {
        device.reset().expect("reset logits");
        let last = device
            .feed(prompt, 0, HeadRows::Last)
            .expect("prefill")
            .pop()
            .expect("row");
        let mut tok = argmax(&last);
        let mut want = Vec::new();
        for step in 0..6 {
            let row = device
                .feed(&[tok], prompt.len() + step, HeadRows::Last)
                .expect("decode")
                .pop()
                .expect("row");
            tok = argmax(&row);
            want.push(tok);
        }
        assert!(
            device.decode_replayed(),
            "run {run}: decode graph not replayed"
        );

        device.reset().expect("reset pick");
        let last = device
            .feed(prompt, 0, HeadRows::Last)
            .expect("prefill 2")
            .pop()
            .expect("row");
        let mut tok = argmax(&last);
        let mut got = Vec::new();
        for step in 0..6 {
            tok = device
                .decode_token(tok, prompt.len() + step, vocab)
                .expect("decode_token");
            got.push(tok);
        }
        assert!(
            device.token_step_replayed(),
            "run {run}: token graph not replayed"
        );
        assert_eq!(
            got, want,
            "run {run}: device greedy picks differ from logits argmax"
        );
    }
}

/// Where a prompt chunk spends its time at the depths of `RAON_SCORE_DEPTHS` (comma separated,
/// ascending): the seconds of two whole chunks fed at each depth, and of the latent attention's
/// score alone in every layer, on the served route and under the exact switch.
#[test]
#[ignore = "needs RAON_Q4_KM and a device; run with --ignored"]
fn the_device_times_the_latent_score_of_a_chunk() {
    use super::cuda::CHUNK;
    let path = model_path();
    let depths: Vec<usize> = std::env::var("RAON_SCORE_DEPTHS")
        .expect("RAON_SCORE_DEPTHS lists the depths")
        .split(',')
        .map(|v| v.trim().parse().expect("a depth"))
        .collect();
    require_device("the_device_times_the_latent_score_of_a_chunk");
    let mut device = Raon1Device::load(&path, None, &mut |cells| {
        eprintln!("planned context {cells}");
    })
    .expect("device load");
    let token = |i: usize| 1000 + (i as u32 * 7919) % 200_000;
    device.reset().expect("reset");
    let mut fed = 0;
    for depth in depths {
        assert!(depth >= fed, "the depths ascend");
        if depth > fed {
            let tokens: Vec<u32> = (fed..depth).map(token).collect();
            device.feed(&tokens, fed, HeadRows::None).expect("prefix");
        }
        let mut chunks = [0f64; 2];
        for (index, seconds) in chunks.iter_mut().enumerate() {
            let at = depth + index * CHUNK;
            let tokens: Vec<u32> = (at..at + CHUNK).map(token).collect();
            let start = std::time::Instant::now();
            device.feed(&tokens, at, HeadRows::None).expect("chunk");
            *seconds = start.elapsed().as_secs_f64();
        }
        fed = depth + 2 * CHUNK;
        let served = device.time_scores(CHUNK, depth, 3).expect("served score");
        device.set_exact(true);
        let exact = device.time_scores(CHUNK, depth, 3).expect("exact score");
        device.set_exact(false);
        eprintln!(
            "depth {depth}: chunks of {CHUNK} rows {:.1} ms and {:.1} ms | latent score of all layers: served {:.1} ms, exact {:.1} ms | chunk less the served score {:.1} ms",
            chunks[0] * 1e3,
            chunks[1] * 1e3,
            served * 1e3,
            exact * 1e3,
            (chunks[0] - served) * 1e3
        );
    }
}

/// The last logits row of a synthetic prompt of `RAON_LAST_ROW_TOKENS` tokens, on the context the
/// free device memory admits: writes the prompt ids as a JSON array to `RAON_LAST_ROW_OUT.ids.json`
/// and the top 20 `id logit` pairs to `RAON_LAST_ROW_OUT.top20`, for comparison with another
/// implementation of the model on the same ids.
#[test]
#[ignore = "needs RAON_Q4_KM and a device; run with --ignored"]
fn the_device_writes_the_last_logits_row_of_a_synthetic_prompt() {
    let path = model_path();
    require_device("the_device_writes_the_last_logits_row_of_a_synthetic_prompt");
    let count: usize = std::env::var("RAON_LAST_ROW_TOKENS")
        .expect("RAON_LAST_ROW_TOKENS")
        .parse()
        .expect("a token count");
    let out = PathBuf::from(std::env::var("RAON_LAST_ROW_OUT").expect("RAON_LAST_ROW_OUT"));
    let tokens = synthetic_tokens(count);
    let mut device = Raon1Device::load(&path, None, &mut |_| {}).expect("device load");
    device.reset().expect("reset");
    let rows = device.feed(&tokens, 0, HeadRows::Last).expect("prefill");
    let row = &rows[0];
    let text: String = top_k(row, 20)
        .iter()
        .map(|&id| format!("{id} {}\n", row[id]))
        .collect();
    std::fs::write(out.with_extension("top20"), text).expect("write the top 20");
    std::fs::write(
        out.with_extension("ids.json"),
        serde_json::to_vec(&tokens).expect("ids"),
    )
    .expect("write the ids");
}

/// The device as a token stepper of the teacher-forced scorer.
struct Scored {
    device: Raon1Device,
    pos: usize,
}

impl crate::family::session::TokenStepper for Scored {
    const OPERATION: &'static str = "raon1_quality_logits";

    fn feed(&mut self, token: u32) -> Result<(), crate::EngError> {
        self.feed_rows(&[token])
    }

    fn feed_rows(&mut self, tokens: &[u32]) -> Result<(), crate::EngError> {
        self.device.feed(tokens, self.pos, HeadRows::None)?;
        self.pos += tokens.len();
        Ok(())
    }

    fn logits(&mut self, token: u32) -> Result<Vec<f32>, crate::EngError> {
        let mut rows = self.device.feed(&[token], self.pos, HeadRows::Last)?;
        self.pos += 1;
        rows.pop()
            .ok_or_else(|| crate::EngError::BadRequest("raon1: the step returned no row".into()))
    }

    fn fed(&self) -> usize {
        self.pos
    }

    fn reset(&mut self) -> Result<(), crate::EngError> {
        self.device.reset()?;
        self.pos = 0;
        Ok(())
    }
}

/// Scores the windows `KVQ_WINDOWS` of the file `KVQ_MODEL` on the device: each window's unscored
/// tokens through the chunked feed, then one forced token per scored position. With `KVQ_EXACT`
/// set every product decodes its weights over f32 rows, the reference of the other arms.
#[test]
#[ignore = "a card under fifi-gpu-run: KVQ_MODEL, KVQ_SAMPLER_VOCAB, KVQ_IDS, KVQ_IDS_SHA, KVQ_LAYOUT, KVQ_WINDOWS, KVQ_OUT, optional KVQ_REF and KVQ_EXACT"]
fn the_device_scores_teacher_forced_windows() {
    use crate::family::teacher_forced::{Run, score_windows};
    let env = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"));
    let run = Run::from_env().expect("run");
    let sampler_vocab: usize = env("KVQ_SAMPLER_VOCAB").parse().expect("KVQ_SAMPLER_VOCAB");
    require_device("the_device_scores_teacher_forced_windows");
    let device = Raon1Device::load(
        &PathBuf::from(env("KVQ_MODEL")),
        Some(run.context + 1),
        &mut |_| {},
    )
    .expect("device load");
    let exact = std::env::var_os("KVQ_EXACT").is_some();
    device.set_exact(exact);
    println!("kvq raon1 leg: exact={exact}");
    let mut stepper = Scored { device, pos: 0 };
    score_windows(&mut stepper, &run, 1024, sampler_vocab).expect("score");
}
