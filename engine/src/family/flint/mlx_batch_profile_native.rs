//! One authorized model load for teacher-forced fifi-vs-fifi 2K/32 identity and pp4096.

use super::super::{env_path, scratch};
use super::{Model, State, clean_observation, git, metadata, qualification_paths};
use crate::family::{
    flint::mlx_qualification::{
        identity_metrics, layer_capture, parse_tokens, trace_close, trace_open, write_row,
    },
    sampler::Sampler,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io, path::Path, time::Instant};

/// Which env set the probe piece size + the serving knob state, in every
/// probe's run metadata.
fn env_snapshot() -> Value {
    json!({
        "FLINT_QUAL_CHUNK_ROWS": crate::family::flint::factory::os_setting(
            std::env::var_os("FLINT_QUAL_CHUNK_ROWS").as_deref()),
        "FIFI_FLINT_MLX_BATCH_PREFILL": crate::family::flint::factory::os_setting(
            std::env::var_os("FIFI_FLINT_MLX_BATCH_PREFILL").as_deref()),
        "FIFI_FLINT_MLX_DECODE_ATTN": crate::family::flint::factory::os_setting(
            std::env::var_os("FIFI_FLINT_MLX_DECODE_ATTN").as_deref()),
        "FIFI_FLINT_MLX_RING_VIEW": crate::family::flint::factory::os_setting(
            std::env::var_os("FIFI_FLINT_MLX_RING_VIEW").as_deref()),
    })
}

/// The probes' piece size: FLINT_QUAL_CHUNK_ROWS when set (64..=4096), else
/// whatever the serving path would run, so the gates qualify the default the
/// engine actually ships.
fn probe_chunk_rows() -> io::Result<usize> {
    let resolved = probe_chunk_rows_inner()?;
    let source = if std::env::var("FLINT_QUAL_CHUNK_ROWS").is_ok() {
        "FLINT_QUAL_CHUNK_ROWS"
    } else {
        "serving default (env unset or non-UTF-8)"
    };
    eprintln!("flint_qual: probe piece size {resolved} rows ({source})");
    Ok(resolved)
}

fn probe_chunk_rows_inner() -> io::Result<usize> {
    match std::env::var("FLINT_QUAL_CHUNK_ROWS") {
        Ok(v) => {
            let n: usize = v
                .parse()
                .map_err(|_| io::Error::other("FLINT_QUAL_CHUNK_ROWS must be a usize"))?;
            if !(64..=4096).contains(&n) {
                return Err(io::Error::other("FLINT_QUAL_CHUNK_ROWS must be 64..=4096"));
            }
            Ok(n)
        }
        Err(_) => {
            let rows = crate::family::flint::factory::prefill_chunk_rows(
                crate::family::flint::factory::os_setting(
                    std::env::var_os("FIFI_FLINT_MLX_BATCH_PREFILL").as_deref(),
                ),
            );
            // 1-row pieces make the ratchet pass trivially; refuse them.
            if rows < 64 {
                return Err(io::Error::other(
                    "FIFI_FLINT_MLX_BATCH_PREFILL=0 (1-row pieces) cannot gate the probes",
                ));
            }
            Ok(rows)
        }
    }
}

/// Context-32K depth probe: mean decode ms/token and engine memory at
/// 8K/16K/32K prefix depths on the tiled frozen corpus, plus the
/// chunked-prefill last-row logits digest per depth as the ring/wrap
/// fingerprint. BITWISE parity across the chain arms at deep prefixes is the
/// multirow parity guard's job — run it with
/// FLINT_QUAL_PROBE_PREFIXES=8704,16384,30720 in the same window.
#[test]
#[ignore = "context-32K depth probe; loads Flint; exclusive parent-owned Mac window"]
fn ctx32k_depth_probe() -> io::Result<()> {
    assert_eq!(
        std::env::var("FLINT_QUAL_GO").as_deref(),
        Ok("parent-owned-gpu-window"),
    );
    clean_observation()?;
    let (root, token_path, out) = qualification_paths()?;
    let raw = fs::read(&token_path)?;
    let digest = format!("{:x}", Sha256::digest(&raw));
    if digest != "75c47e34ae9b6a3fc5190ec0e01273b88fee06fb6fdb933745ede28f1362dae0" {
        return Err(io::Error::other(
            "depth probe IDs differ from the frozen 2049-ID corpus",
        ));
    }
    let ids = parse_tokens(&raw)?;
    // The corpus tiles to the stream length the sweep needs: 32K by default
    // (the historic probe), raised via FLINT_QUAL_STREAM_TOKENS. Stream
    // construction only - depths above 32K still need the context-limit
    // slice (factory::context_limit admits only 8192/32768).
    let stream_tokens = match std::env::var("FLINT_QUAL_STREAM_TOKENS") {
        Ok(v) => v
            .trim()
            .parse::<usize>()
            .map_err(|_| io::Error::other("FLINT_QUAL_STREAM_TOKENS must be a usize"))?,
        Err(std::env::VarError::NotPresent) => 32 * 1024,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(io::Error::other(
                "FLINT_QUAL_STREAM_TOKENS must be UTF-8 (unset means the 32K default)",
            ));
        }
    };
    if stream_tokens < ids.len() || stream_tokens > 1024 * 1024 {
        return Err(io::Error::other(
            "FLINT_QUAL_STREAM_TOKENS must be within corpus_len..=1Mi",
        ));
    }
    let stream: Vec<u32> = (0..)
        .flat_map(|_| ids.iter().copied())
        .take(stream_tokens)
        .collect();
    let depths: Vec<usize> = match std::env::var("FLINT_QUAL_DEPTHS") {
        Ok(list) => list
            .split(',')
            .map(|v| {
                v.trim()
                    .parse::<usize>()
                    .map_err(|_| io::Error::other("FLINT_QUAL_DEPTHS must be usizes"))
            })
            .collect::<io::Result<_>>()?,
        Err(_) => vec![8192, 16384, 30720],
    };
    let limit = crate::family::flint::factory::context_limit() as usize;
    // Short depths (22/512) measure the plane-allocation floor at low prompt
    // lengths; the lower bound is 1, not 1024.
    if depths.iter().any(|&d| d < 1 || d > stream.len().min(limit)) {
        return Err(io::Error::other(format!(
            "depths must be within 1..={} (serving limit via FIFI_FLINT_MLX_CONTEXT_LIMIT)",
            stream.len().min(limit)
        )));
    }
    let chunk_rows = probe_chunk_rows()?;
    fs::create_dir(&out)?;
    let mut info = json!({
        "source_commit": git("HEAD")?,
        "probe": "ctx32k-depth",
        "token_ids_sha256": digest,
        "corpus": "frozen-2049-tiled",
        "stream_tokens": stream_tokens,
        "chunk_rows": chunk_rows,
        "env_snapshot": {
            "FLINT_QUAL_CHUNK_ROWS": crate::family::flint::factory::os_setting(
                std::env::var_os("FLINT_QUAL_CHUNK_ROWS").as_deref()),
            "FIFI_FLINT_MLX_BATCH_PREFILL": crate::family::flint::factory::os_setting(
                std::env::var_os("FIFI_FLINT_MLX_BATCH_PREFILL").as_deref()) },
        "depths": {},
    });
    metadata(&out, &info)?;
    let _serial = crate::native_test_lock();
    let mut model = Model::open(&root).map_err(|e| io::Error::other(format!("{e:?}")))?;
    for &d in &depths {
        // Per-depth peak: reset the allocator's high-water mark so each
        // depth's memory reflects its own footprint.
        model
            .kernels
            .reset_peak_memory()
            .map_err(|e| io::Error::other(format!("reset peak memory before depth {d}: {e:?}")))?;
        let mut state = State::new();
        let mut last = None;
        let prefill_start = Instant::now();
        for (index, piece) in stream[..d].chunks(chunk_rows).enumerate() {
            let last_piece = (index + 1) * chunk_rows >= d;
            last = model
                .prefill_rows(&mut state, piece, last_piece)
                .map_err(|e| io::Error::other(format!("{e:?}")))?;
        }
        let prefill_seconds = prefill_start.elapsed().as_secs_f64();
        let logits = last.ok_or_else(|| io::Error::other("depth probe returned no logits"))?;
        let mut bytes = Vec::new();
        write_row(&mut bytes, &logits)?;
        let logits_sha256 = format!("{:x}", Sha256::digest(&bytes));
        let start = Instant::now();
        let (tokens, _) = greedy_rows(&mut model, &mut state, logits, 64)?;
        let decode_seconds = start.elapsed().as_secs_f64();
        // On the tiled corpus the greedy continuation should follow the
        // pattern - the fraction of the 64 decoded tokens that equal the
        // corpus continuation is the induction score (asserted >= 0.95).
        let induction_matches = tokens
            .iter()
            .enumerate()
            // Index the corpus by position modulo its length, so a depth at
            // the stream's end still scores against the true continuation
            // instead of comparing against nothing (stream.get -> None).
            .filter(|&(i, tok)| ids.get((d + i) % ids.len()) == Some(tok))
            .count();
        let induction_fraction = induction_matches as f64 / tokens.len().max(1) as f64;
        // The >= 0.95 bar applies only once at least one full tiling of the
        // corpus is in context; below that (short depths) the fraction is
        // recorded but not gated.
        assert!(
            d < ids.len() || induction_fraction >= 0.95,
            "induction broke at depth {d}: {induction_matches}/{} tokens follow the corpus",
            tokens.len()
        );
        let stats = model
            .kernels
            .memory_stats()
            .map_err(|e| io::Error::other(format!("memory stats at depth {d}: {e:?}")))?;
        let entry = json!({
            "prefill_tokens": d,
            "prefill_seconds": prefill_seconds,
            "ttft_ms_per_ktoken": prefill_seconds * 1000.0 / (d as f64 / 1024.0),
            "decode_tokens": tokens.len(),
            "decode_seconds": decode_seconds,
            "decode_ms_per_token": decode_seconds * 1000.0 / tokens.len().max(1) as f64,
            "token_ids_sha256": format!("{:x}", Sha256::digest(
                tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>())),
            "logits_sha256": logits_sha256,
            "induction_matches": induction_matches,
            "induction_fraction": induction_fraction,
            "memory": {
                "active": stats.active, "cache": stats.cache, "peak": stats.peak,
            },
        });
        info["depths"][&d.to_string()] = entry;
        metadata(&out, &info)?;
        eprintln!(
            "flint_ctx32k_depth d={} decode_ms_per_token={:.3}",
            d,
            decode_seconds * 1000.0 / tokens.len().max(1) as f64
        );
    }
    Ok(())
}
#[test]
#[ignore = "requires an exclusive parent-owned Mac model/timing window"]
fn batched_prefill_identity_and_pp4096() -> io::Result<()> {
    assert_eq!(
        std::env::var("FLINT_QUAL_GO").as_deref(),
        Ok("parent-owned-gpu-window")
    );
    clean_observation()?;
    let capture_layers = match std::env::var_os("FLINT_QUAL_LAYER_CAPTURE") {
        None => false,
        Some(value) if value == "1" => true,
        Some(_) => return Err(io::Error::other("FLINT_QUAL_LAYER_CAPTURE accepts only 1")),
    };
    let (root, token_path, out) = qualification_paths()?;
    let raw = fs::read(&token_path)?;
    let digest = format!("{:x}", Sha256::digest(&raw));
    if digest != "75c47e34ae9b6a3fc5190ec0e01273b88fee06fb6fdb933745ede28f1362dae0" {
        return Err(io::Error::other(
            "identity IDs differ from the frozen 2049-ID corpus",
        ));
    }
    let pp_path = env_path("FLINT_QUAL_PP_TOKENS")?.canonicalize()?;
    if !scratch(&pp_path) || fs::metadata(&pp_path)?.len() > 1024 * 1024 {
        return Err(io::Error::other("pp4096 IDs must use bounded task scratch"));
    }
    let pp_raw = fs::read(&pp_path)?;
    let ids = parse_tokens(&raw)?;

    let ratchet_chunk = probe_chunk_rows()?;
    let pp_ids = parse_tokens(&pp_raw)?;
    if ids.len() != 2049 || pp_ids.len() != 4097 || pp_ids[..2049] != ids {
        return Err(io::Error::other(
            "pp4096 IDs must extend the frozen 2049-ID prefix",
        ));
    }
    fs::create_dir(&out)?;
    let mut info = json!({"source_commit":git("HEAD")?,"source_tree":git("HEAD^{tree}")?,
        "model_revision":"3b93f9053b530488405ae9fca51d4900d6ce21f0",
        "identity_authority":"01a1676b-56d8-7146-8d01-bc2207ef990b",
        "corpus_authority":"01a16623-6180-758f-9fcf-87ff916f88fd",
        "identity_protocol":"teacher-forced-t1-reference","identity_execution_order":["per-token","batched"],
        "token_ids_sha256":digest,"pp_token_ids_sha256":format!("{:x}",Sha256::digest(&pp_raw)),
        "identity_prefix_tokens":2048,"identity_decode_tokens":32,"identity_rows_expected":33,
        "identity_logit_positions":[2047,2080],"identity_generated_positions":[2048,2080],
        "row_semantics":"half-open positions: last prefill row plus 32 decode rows on the per-token control's greedy token sequence",
        "identity":{},"identity_reference":"same loaded fifi model, T=1 control; no external reference",
        "pp_tokens":4096,"chunk_rows": ratchet_chunk,"warm_decode_tokens":128,"completed":false,
        "env_snapshot": env_snapshot(),
        "pp_warmup":"identical untimed prefill before every timed prefill; KV reset",
        "profile_build_not_production_throughput":true,
        "layer_capture_enabled":capture_layers,"capture_invalidates_timing":capture_layers});
    metadata(&out, &info)?;
    let result = observe(&root, &out, &pp_ids, ratchet_chunk, &mut info);
    info["completed"] = json!(result.is_ok());
    if let Err(error) = &result {
        info["error"] = json!(error.to_string());
    }
    metadata(&out, &info)?;
    result
}

fn observe(
    root: &Path,
    out: &Path,
    ids: &[u32],
    chunk_rows: usize,
    info: &mut Value,
) -> io::Result<()> {
    let _serial = crate::native_test_lock();
    let mut model = Model::open(root).map_err(|e| io::Error::other(format!("{e:?}")))?;
    identity(&mut model, out, ids, chunk_rows, info)?;
    if info["layer_capture_enabled"] == json!(true) {
        return Ok(());
    }
    let mut state = State::new();
    // pp attribution probe (FLINT_QUAL_PP_TRACE): per-stage span rows for the
    // timed pp4096 pass (warm + measured rows both land in the trace; the
    // analyzer takes the last occurrence per stage). With
    // FLINT_QUAL_STAGE_SYNC the forward also fences each stage so its
    // elapsed time is GPU-inclusive (perturbs dispatch: the async chain's
    // whole point). Both modes record; neither changes any computed value.
    let traced = std::env::var_os("FLINT_QUAL_PP_TRACE").is_some();
    if traced {
        trace_open(&out.join("pp-trace.jsonl"), 8192)?;
    }
    let traced_result = warm_then_measure_rows(&mut model, &mut state, &ids[..4096], chunk_rows);
    if traced {
        trace_close();
    }
    let (row, stats) = traced_result?;
    assert_eq!(state.pos, 4096);
    let mut bytes = Vec::new();
    write_row(&mut bytes, &row)?;
    info["pp4096"] = stats;
    info["pp_logits_sha256"] = json!(format!("{:x}", Sha256::digest(&bytes)));
    fs::write(out.join("pp4096-last-row.f32le"), bytes)?;
    let started = Instant::now();
    let (tokens, _) = greedy_rows(&mut model, &mut state, row, 128)?;
    let seconds = started.elapsed().as_secs_f64();
    assert_eq!(tokens.len(), 128);
    assert_eq!(state.pos, 4224);
    info["warm_decode128"] = json!({"tokens":128,"seconds":seconds,
        "tokens_per_second":128.0/seconds,"prefix_tokens":4096,
        "includes_greedy_sampling_and_logit_readback":true});
    let mut short = State::new();
    let (_, stats) = warm_then_measure_rows(&mut model, &mut short, &ids[..512], chunk_rows)?;
    assert_eq!(short.pos, 512);
    info["pp512"] = stats;
    Ok(())
}

fn identity(
    model: &mut Model,
    out: &Path,
    ids: &[u32],
    chunk_rows: usize,
    info: &mut Value,
) -> io::Result<()> {
    let mut runs = Vec::new();
    let mut reference = Vec::new();
    let capture_layers = info["layer_capture_enabled"] == json!(true);
    let capture_root = out.join("layers");
    if capture_layers {
        fs::create_dir(&capture_root)?;
    }
    for (label, chunk_rows) in [("per-token", 1), ("batched", chunk_rows)] {
        let mut state = State::new();
        let capture = if capture_layers {
            Some(layer_capture::Scope::enter(&capture_root, label)?)
        } else {
            None
        };
        let (first, stats) = warm_then_measure_rows(model, &mut state, &ids[..2048], chunk_rows)?;
        assert_eq!(state.pos, 2048);
        info["identity"][label] = json!({"prefix":stats,"completed":false,
            "token_source":if chunk_rows == 1 { "greedy-reference" } else { "teacher-forced-reference" }});
        metadata(out, info)?;
        let (tokens, rows) = if chunk_rows == 1 {
            let run = greedy_rows(model, &mut state, first, 32)?;
            reference = run.0.clone();
            run
        } else {
            assert_eq!(reference.len(), 32);
            (
                reference.clone(),
                teacher_forced_rows(model, &mut state, first, &reference)?,
            )
        };
        assert_eq!(tokens.len(), 32);
        assert_eq!(rows.len(), 33);
        assert_eq!(state.pos, 2080);
        let mut bytes = Vec::new();
        for row in &rows {
            write_row(&mut bytes, row)?;
        }
        let digest = format!("{:x}", Sha256::digest(&bytes));
        fs::write(out.join(format!("identity-{label}.f32le")), bytes)?;
        info["identity"][label]["tokens"] = json!(tokens);
        info["identity"][label]["logits_sha256"] = json!(digest);
        info["identity"][label]["completed"] = json!(true);
        metadata(out, info)?;
        if let Some(capture) = capture {
            info["identity"][label]["layer_capture"] = capture.finish()?;
            metadata(out, info)?;
        }
        info["identity"][label]["warm_decode"] =
            warm_decode128(model, &mut state, rows.last().unwrap().clone())?;
        metadata(out, info)?;
        runs.push((tokens, rows));
    }
    let metrics = identity_metrics(&runs[1].1, &runs[0].1)?;
    let same_context = runs[0].0 == runs[1].0;
    let passed = same_context && metrics["passed"] == json!(true);
    // Gating mode: the per-token-vs-batched drift is flag-INDEPENDENT
    // - fenced and chained arms fail it with metrics identical to every
    // digit and per-token/batched row digests bitwise equal across flags -
    // so the CHAIN is gated by the in-tree chained_fenced_bitwise_
    // identity_2040 guard (bitwise per-token rows across flags). This gate
    // ratchets the drift against the fenced baseline: exceeding it means
    // NEW numerics, not the known T=1 vs batched tile-rounding class, which
    // is tracked separately.
    // Applied bars on every run (pass or fail). For 1024-row pieces a run
    // measured identity drift mean 0.050387 / max row 1.046713 / top-1 33
    // of 33, ~23% above the 512 magnitude, and the bars carry ~3% headroom
    // over that.
    let (applied_mean_bar, applied_max_bar) = match chunk_rows {
        1024 => (0.0520, 1.0800),
        _ => (0.0410, 1.008),
    };
    info["identity_gate"] = json!({
        "teacher_forcing_inputs_identical": same_context,
        "metrics": metrics,
        "ratchet_bars": {
            "mean": applied_mean_bar,
            "max_row": applied_max_bar,
            "top1": 33,
        },
        "passed": passed,
        "gating_mode": "ratchet vs the fenced baseline, per piece size (512: mean_kl <= 0.0410, max_row_kl <= 1.008; 1024: mean_kl <= 0.0520, max_row_kl <= 1.0800; top1 >= 33/33, fatal); the chain itself is gated by the in-tree chained_fenced_bitwise_identity_2040 guard",
    });
    metadata(out, info)?;
    if !passed {
        // Ratchet against the fenced baseline: the flag-independent drift
        // sits at mean 0.0410 / max row 1.007 / top1 33 of 33, identical
        // under both flags, so exceeding it means NEW numerics, not the known
        // T=1-vs-batched drift. same_context cannot fire as a check - the
        // batched run teacher-forces the per-token run's own tokens by
        // construction.
        let mean = metrics["mean_kl"].as_f64().unwrap_or(f64::INFINITY);
        let max = metrics["max_row_kl"].as_f64().unwrap_or(f64::INFINITY);
        let top1 = metrics["top1_matches"].as_i64().unwrap_or(-1);
        // Bars are per piece size: the baseline limits were calibrated on
        // 512-row pieces (mean 0.040916 / max row 1.007338, ~0.2% headroom);
        // 1024-row pieces carry the same tile-rounding class at a larger
        // magnitude (mean 0.050387 / max 1.046713, top-1 still 33/33, ~23%
        // more drift than 512 - the reason the 1024 bars are separate). Any
        // other piece size keeps the 512 bars.
        if mean > applied_mean_bar || max > applied_max_bar || top1 < 33 {
            return Err(io::Error::other(format!(
                "2K/32 drift exceeded the ratchet for {chunk_rows}-row pieces: mean_kl {mean:.6} (<= {applied_mean_bar}), max_row_kl {max:.6} (<= {applied_max_bar}), top1 {top1} of 33"
            )));
        }
    }
    Ok(())
}

fn warm_decode128(model: &mut Model, state: &mut State, first: Vec<f32>) -> io::Result<Value> {
    let start_position = state.pos;
    let start = Instant::now();
    let (tokens, _) = greedy_rows(model, state, first, 128)?;
    let seconds = start.elapsed().as_secs_f64();
    let tokens_per_second = tokens.len() as f64 / seconds;
    Ok(json!({
        "tokens":tokens.len(), "start_position":start_position,
        "warmup_decode_tokens":32, "seconds":seconds,
        // The greedy IDs themselves: the per-token-vs-batched first-divergence
        // index is the flag-off calibration for real-prompt stream diffs.
        "token_ids":tokens,
        "tokens_per_second":tokens_per_second,
        "includes_sampler_and_logit_readback":true,
        "llama_cpp_tg128_tokens_per_second":95.8,
        "ratio_to_supplied_llama_cpp_tg128":tokens_per_second/95.8
    }))
}

/// In-tree guard: the chained and fenced commits must produce
/// BITWISE-identical one-row logits at long context. Re-execs the test binary
/// as two children (the tokenizer child-process-table pattern): child A runs
/// the probe with the chain default (flag unset), child B with
/// FIFI_FLINT_MLX_ASYNC_CHAIN=0, both under the parent's GO, model, corpus
/// and output root at prefix 2040. The parent compares every row digest and
/// fails on any difference, so a future change to the chain cannot silently
/// regress the long-context identity. Spawns one re-exec child of the test
/// binary running the probe under the given flag state and returns its
/// per-token row digests at prefix 2040.
fn spawn_probe_child(
    binary: &Path,
    probe: &str,
    label: &str,
    rollback: bool,
    model: &Path,
    tokens: &Path,
    child_out: &Path,
) -> io::Result<Vec<String>> {
    use std::process::Command;
    let mut command = Command::new(binary);
    command
        .args(["--exact", probe, "--ignored", "--test-threads", "1"])
        .env("FLINT_QUAL_GO", "parent-owned-gpu-window")
        .env("FLINT_QUAL_MODEL", model)
        .env("FLINT_QUAL_TOKENS", tokens)
        .env("FLINT_QUAL_OUTPUT", child_out)
        .env("FLINT_QUAL_PROBE_PREFIXES", "2040")
        .env_remove("FLINT_QUAL_LAYER_CAPTURE");
    if rollback {
        command.env("FIFI_FLINT_MLX_ASYNC_CHAIN", "0");
    } else {
        command.env_remove("FIFI_FLINT_MLX_ASYNC_CHAIN");
    }
    let output = command
        .output()
        .map_err(|e| io::Error::other(format!("spawn the {label} child: {e}")))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "the {label} child exited with {}\nstdout:\n{stdout}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let meta: Value = serde_json::from_str(&fs::read_to_string(child_out.join("metadata.json"))?)?;
    assert_eq!(
        meta["chain_flag"],
        json!(label),
        "the {label} child ran the wrong flag state"
    );
    Ok(meta["prefixes"]["2040"]["row_sha256"]
        .as_array()
        .map(|rows| {
            rows.iter()
                .map(|v| v.as_str().unwrap_or_default().to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default())
}

#[test]
#[ignore = "in-tree chain guard; loads Flint twice through child processes; exclusive parent-owned window"]
fn chained_fenced_bitwise_identity_2040() -> io::Result<()> {
    assert_eq!(
        std::env::var("FLINT_QUAL_GO").as_deref(),
        Ok("parent-owned-gpu-window")
    );
    clean_observation()?;
    let (model, tokens, out) = qualification_paths()?;
    let raw = fs::read(&tokens)?;
    if format!("{:x}", Sha256::digest(&raw))
        != "75c47e34ae9b6a3fc5190ec0e01273b88fee06fb6fdb933745ede28f1362dae0"
    {
        return Err(io::Error::other(
            "guard IDs differ from the frozen 2049-ID corpus",
        ));
    }
    fs::create_dir(&out)?;
    let probe =
        "family::flint::mlx_qualification::native::profile::batch::prefix_parity_bisect_probe";
    let binary = std::env::current_exe().expect("test binary");
    let mut digests = Vec::with_capacity(2);
    for (label, rollback) in [("chained", false), ("fenced", true)] {
        let child_out = out.join(label);
        digests.push(spawn_probe_child(
            &binary, probe, label, rollback, &model, &tokens, &child_out,
        )?);
    }
    assert_eq!(digests[0].len(), 13, "the chained child recorded no rows");
    assert_eq!(
        digests[0], digests[1],
        "chained and fenced per-token rows differ at prefix 2040"
    );
    Ok(())
}

/// Long-context parity probe: warm every prefix through the batched path
/// path (identical code under both flags), run the trailing 12 ids one row
/// at a time - the only code the flag changes - and record per-row
/// digests for a controller (or the in-tree guard above) to compare
/// between the chained and fenced arms. Windows 2-3 proved the two
/// flags bitwise-identical at every position tested 98..2044 in both the
/// hybrid and the full-2K structures; the prefix list is env-overridable
/// so later bands reuse the same binary.
#[test]
#[ignore = "diagnostic bisect; loads Flint; exclusive parent-owned window; the controller runs the ASYNC_CHAIN=0 and unset arms and compares row digests"]
fn prefix_parity_bisect_probe() -> io::Result<()> {
    assert_eq!(
        std::env::var("FLINT_QUAL_GO").as_deref(),
        Ok("parent-owned-gpu-window")
    );
    let ratchet_chunk = probe_chunk_rows()?;

    clean_observation()?;
    let (root, token_path, out) = qualification_paths()?;
    let raw = fs::read(&token_path)?;
    let digest = format!("{:x}", Sha256::digest(&raw));
    if digest != "75c47e34ae9b6a3fc5190ec0e01273b88fee06fb6fdb933745ede28f1362dae0" {
        return Err(io::Error::other(
            "bisect IDs differ from the frozen 2049-ID corpus",
        ));
    }
    let ids = parse_tokens(&raw)?;
    // The controller may override the prefix list (FLINT_QUAL_PROBE_PREFIXES,
    // comma-separated u32s); the default localizes the first divergence in
    // the 98..1028 band, the extension reuses the same binary for the
    // 1280..2040 band.
    let prefixes: Vec<u32> = match std::env::var("FLINT_QUAL_PROBE_PREFIXES") {
        Ok(list) => list
            .split(',')
            .map(|v| {
                v.trim()
                    .parse::<u32>()
                    .map_err(|_| io::Error::other("FLINT_QUAL_PROBE_PREFIXES must be u32s"))
            })
            .collect::<io::Result<_>>()?,
        Err(std::env::VarError::NotPresent) => vec![256, 512, 513, 514, 600, 1024],
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(io::Error::other("FLINT_QUAL_PROBE_PREFIXES must be UTF-8"));
        }
    };
    if prefixes.is_empty()
        || prefixes
            .iter()
            .any(|&p| p < 16 || p as usize + 4 > ids.len())
    {
        return Err(io::Error::other(
            "prefixes must be >= 16 and end at least 4 ids before the corpus end",
        ));
    }
    fs::create_dir(&out)?;
    let mut info = json!({
        "source_commit": git("HEAD")?,
        "chunk_rows": ratchet_chunk,
        "env_snapshot": env_snapshot(),
        "probe": "prefix-parity-bisect",
        "token_ids_sha256": digest,
        "chain_flag": if crate::family::flint::mlx_forward::async_chain_flagged() {
            "chained"
        } else {
            "fenced"
        },
        "tail_rows": 12,
        "prefixes": {},
    });
    info["prefix_list"] = json!(&prefixes);
    metadata(&out, &info)?;
    let _serial = crate::native_test_lock();
    let mut model = Model::open(&root).map_err(|e| io::Error::other(format!("{e:?}")))?;
    let rows_dir = out.join("rows");
    fs::create_dir(&rows_dir)?;
    for p in prefixes {
        let mut state = State::new();
        let (row, _stats) = measured_rows(
            &mut model,
            &mut state,
            &ids[..p as usize - 8],
            ratchet_chunk,
        )?;
        let tail = &ids[p as usize - 8..p as usize + 4];
        let rows = teacher_forced_rows(&mut model, &mut state, row, tail)?;
        assert_eq!(rows.len(), 13);
        assert_eq!(state.pos, p + 4);
        let mut shas = Vec::with_capacity(rows.len());
        for (i, r) in rows.iter().enumerate() {
            let mut bytes = Vec::new();
            write_row(&mut bytes, r)?;
            fs::write(rows_dir.join(format!("{p}-{i:02}.f32le")), &bytes)?;
            shas.push(format!("{:x}", Sha256::digest(&bytes)));
        }
        info["prefixes"][format!("{p}")] = json!({"rows": rows.len(), "row_sha256": shas});
        metadata(&out, &info)?;
    }
    Ok(())
}

fn teacher_forced_rows(
    model: &mut Model,
    state: &mut State,
    first: Vec<f32>,
    tokens: &[u32],
) -> io::Result<Vec<Vec<f32>>> {
    let mut rows = vec![first];
    for &token in tokens {
        let next = model
            .step(state, token, true)
            .map_err(|e| io::Error::other(format!("{e:?}")))?
            .ok_or_else(|| io::Error::other("teacher-forced step returned no logits"))?;
        rows.push(next);
    }
    Ok(rows)
}

fn greedy_rows(
    model: &mut Model,
    state: &mut State,
    first: Vec<f32>,
    steps: usize,
) -> io::Result<(Vec<u32>, Vec<Vec<f32>>)> {
    let mut sampler = Sampler::greedy();
    let mut tokens = Vec::with_capacity(steps);
    let mut rows = vec![first];
    for _ in 0..steps {
        let token = sampler
            .sample(rows.last().unwrap())
            .map_err(|e| io::Error::other(format!("{e:?}")))?;
        let next = model
            .step(state, token, true)
            .map_err(|e| io::Error::other(format!("{e:?}")))?
            .ok_or_else(|| io::Error::other("greedy step returned no logits"))?;
        tokens.push(token);
        rows.push(next);
    }
    Ok((tokens, rows))
}

fn warm_then_measure_rows(
    model: &mut Model,
    state: &mut State,
    ids: &[u32],
    chunk_rows: usize,
) -> io::Result<(Vec<f32>, Value)> {
    {
        let _pause = layer_capture::suspend();
        let mut warm = State::new();
        measured_rows(model, &mut warm, ids, chunk_rows)?;
    }
    let (row, mut stats) = measured_rows(model, state, ids, chunk_rows)?;
    stats["warmup_tokens"] = json!(ids.len());
    stats["warmup_completed"] = json!(true);
    Ok((row, stats))
}

fn measured_rows(
    model: &mut Model,
    state: &mut State,
    ids: &[u32],
    chunk_rows: usize,
) -> io::Result<(Vec<f32>, Value)> {
    model.kernels.qualification_reset_timing();
    model.kernels.qualification_reset_flush_stats();
    let sync_before = model.kernels.qualification_sync_count();
    let ops_before = model.kernels.selected_operation_count();
    let start = Instant::now();
    let mut row = None;
    for (index, chunk) in ids.chunks(chunk_rows).enumerate() {
        let last = (index + 1) * chunk_rows >= ids.len();
        row = model
            .prefill_rows(state, chunk, last)
            .map_err(|error| io::Error::other(format!("{error:?}")))?;
        if last || ((index + 1) * chunk_rows).is_multiple_of(128) {
            eprintln!(
                "flint_identity_prefill chunk_rows={chunk_rows} tokens={}",
                state.pos
            );
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    let timing = model.kernels.qualification_timing();
    let flush = model.kernels.qualification_flush_stats();
    let syncs = model
        .kernels
        .qualification_sync_count()
        .checked_sub(sync_before)
        .ok_or_else(|| io::Error::other("synchronization counter regressed"))?;
    let ops = model
        .kernels
        .selected_operation_count()
        .checked_sub(ops_before)
        .ok_or_else(|| io::Error::other("operation counter regressed"))?;
    Ok((
        row.ok_or_else(|| io::Error::other("prefill returned no last-row logits"))?,
        json!({"seconds":seconds,"tokens":ids.len(),"tokens_per_second":ids.len() as f64 / seconds,
        "chunk_rows":chunk_rows,"final_logits":true,"import_calls":timing.import_calls,"import_ns":timing.import_ns,"env_snapshot": env_snapshot(),
        "readback_calls":timing.readback_calls,"readback_ns":timing.readback_ns,
        "evaluate_calls":timing.evaluate_calls,"evaluate_ns":timing.evaluate_ns,
        "native_synchronizations":syncs,"selected_operations":ops,"forced_retire_flushes":flush.flushes,
        "flush_sync_ns":flush.sync_ns,"flush_free_ns":flush.free_ns}),
    ))
}

/// Multi-row chunk parity probe (prefill-group chain slice): run each
/// prefix as chunk_rows pieces through the REAL forward under whatever
/// FIFI_FLINT_MLX_PREFILL_CHAIN state the controller armed, and record the
/// last-row logits digest plus a per-layer KV-plane digest. The in-tree
/// guard below compares armed vs default arms for bitwise identity; the
/// A/B bench reuses the same probe shape for timing.
#[test]
#[ignore = "diagnostic probe; loads Flint; exclusive parent-owned window; the controller runs the PREFILL_CHAIN armed/unset arms and compares digests"]
fn multirow_parity_probe() -> io::Result<()> {
    assert_eq!(
        std::env::var("FLINT_QUAL_GO").as_deref(),
        Ok("parent-owned-gpu-window"),
    );
    clean_observation()?;
    let (root, token_path, out) = qualification_paths()?;
    let raw = fs::read(&token_path)?;
    let digest = format!("{:x}", Sha256::digest(&raw));
    if digest != "75c47e34ae9b6a3fc5190ec0e01273b88fee06fb6fdb933745ede28f1362dae0" {
        return Err(io::Error::other(
            "probe IDs differ from the frozen 2049-ID corpus",
        ));
    }
    let base_ids = parse_tokens(&raw)?;
    let prefixes: Vec<u32> = match std::env::var("FLINT_QUAL_PROBE_PREFIXES") {
        Ok(list) => list
            .split(',')
            .map(|v| {
                v.trim()
                    .parse::<u32>()
                    .map_err(|_| io::Error::other("FLINT_QUAL_PROBE_PREFIXES must be u32s"))
            })
            .collect::<io::Result<_>>()?,
        Err(_) => vec![512, 513, 1024],
    };
    let chunk_rows = probe_chunk_rows()?;
    // Deep prefixes tile the frozen corpus to 32K (same deterministic tiling
    // as chunk_kl_probe/ctx32k_depth_probe) so the ring/wrap boundary work
    // beyond the legacy 2049-ID corpus; shallow prefixes keep it verbatim.
    let deep = prefixes.iter().any(|&p| p as usize > base_ids.len());
    let limit = crate::family::flint::factory::context_limit() as usize;
    let cap = (32 * 1024).min(limit);
    let ids: Vec<u32> = if deep {
        base_ids.iter().copied().cycle().take(cap).collect()
    } else {
        base_ids.clone()
    };
    if prefixes.is_empty() || prefixes.iter().any(|&p| p < 2 || p as usize > ids.len()) {
        return Err(io::Error::other(
            "prefixes must cover at least one multi-row chunk",
        ));
    }
    let info_corpus = if deep {
        "frozen-2049-tiled-32k"
    } else {
        "frozen-2049"
    };
    fs::create_dir(&out)?;
    let mut info = json!({
        "source_commit": git("HEAD")?,
        "probe": "multirow-parity",
        "corpus": info_corpus,
        "token_ids_sha256": digest,
        "prefill_chain_flag": if crate::family::flint::mlx_forward::prefill_chain_flagged() {
            "prefill-chained"
        } else {
            "prefill-fenced"
        },
        "chunk_rows": chunk_rows,
        "env_snapshot": env_snapshot(),
        "prefixes": {},
    });
    metadata(&out, &info)?;
    let _serial = crate::native_test_lock();
    let mut model = Model::open(&root).map_err(|e| io::Error::other(format!("{e:?}")))?;
    for &p in &prefixes {
        let mut state = State::new();
        let mut row = None;
        for (index, chunk) in ids[..p as usize].chunks(chunk_rows).enumerate() {
            let last = (index + 1) * chunk_rows >= p as usize;
            row = model
                .prefill_rows(&mut state, chunk, last)
                .map_err(|e| io::Error::other(format!("{e:?}")))?;
        }
        let row = row.ok_or_else(|| io::Error::other("probe prefix returned no logits"))?;
        let mut bytes = Vec::new();
        write_row(&mut bytes, &row)?;
        let mut plane_digests = Vec::with_capacity(state.planes.len());
        for plane in &state.planes {
            let plane = plane
                .as_ref()
                .ok_or_else(|| io::Error::other("probe prefix left a layer without a plane"))?;
            let (key, value) = plane
                .snapshot(&mut model.kernels)
                .map_err(|e| io::Error::other(format!("{e:?}")))?;
            let mut hasher = Sha256::new();
            hasher.update(key.host_bytes().unwrap_or_default());
            hasher.update(value.host_bytes().unwrap_or_default());
            plane_digests.push(format!("{:x}", hasher.finalize()));
        }
        let mut plane_hasher = Sha256::new();
        for d in &plane_digests {
            plane_hasher.update(d.as_bytes());
        }
        info["prefixes"][&p.to_string()] = json!({
            "logits_sha256": format!("{:x}", Sha256::digest(&bytes)),
            "planes_sha256": format!("{:x}", plane_hasher.finalize()),
            "plane_digests": plane_digests,
            "positions": state.pos,
        });
        metadata(&out, &info)?;
    }
    Ok(())
}

/// Chunk-size KL probe: teacher-forced prefill of the frozen corpus tiled to 32K
/// tokens, chunked at 512 vs 1024. Reports last-row logits KL, top-1 agreement
/// and top-5 overlap per prefix, plus a 512-vs-512 self-run noise floor.
#[test]
#[ignore = "KL probe; loads Flint; exclusive parent-owned window"]
fn chunk_kl_probe() -> io::Result<()> {
    assert_eq!(
        std::env::var("FLINT_QUAL_GO").as_deref(),
        Ok("parent-owned-gpu-window"),
    );
    clean_observation()?;
    let (root, token_path, out) = qualification_paths()?;
    let raw = fs::read(&token_path)?;
    let digest = format!("{:x}", Sha256::digest(&raw));
    if digest != "75c47e34ae9b6a3fc5190ec0e01273b88fee06fb6fdb933745ede28f1362dae0" {
        return Err(io::Error::other(
            "probe IDs differ from the frozen 2049-ID corpus",
        ));
    }
    let ids = parse_tokens(&raw)?;
    // Deterministic tiling of the frozen corpus to 32K tokens: the shape effect
    // (chunk boundary accumulation) is what is measured, not corpus novelty.
    let stream: Vec<u32> = (0..)
        .flat_map(|_| ids.iter().copied())
        .take(32 * 1024)
        .collect();
    let prefixes: Vec<usize> = match std::env::var("FLINT_QUAL_KL_PREFIXES") {
        Ok(list) => list
            .split(',')
            .map(|v| {
                v.trim()
                    .parse::<usize>()
                    .map_err(|_| io::Error::other("FLINT_QUAL_KL_PREFIXES must be usizes"))
            })
            .collect::<io::Result<_>>()?,
        Err(_) => vec![2048, 8192, 30720],
    };
    let limit = crate::family::flint::factory::context_limit() as usize;
    if prefixes
        .iter()
        .any(|&p| p < 1024 || p > stream.len().min(limit))
    {
        return Err(io::Error::other(format!(
            "KL prefixes must be within 1024..={} (serving limit via FIFI_FLINT_MLX_CONTEXT_LIMIT)",
            stream.len().min(limit)
        )));
    }
    fs::create_dir(&out)?;
    let mut info = json!({
        "source_commit": git("HEAD")?,
        "probe": "chunk-kl",
        "token_ids_sha256": digest,
        "corpus": "frozen-2049-tiled-32k",
        "prefixes": {},
    });
    metadata(&out, &info)?;
    let _serial = crate::native_test_lock();
    let mut model = Model::open(&root).map_err(|e| io::Error::other(format!("{e:?}")))?;
    let mut prefill = |chunk_rows: usize, p: usize| -> io::Result<Vec<f32>> {
        let mut state = State::new();
        let mut row = None;
        for (index, piece) in stream[..p].chunks(chunk_rows).enumerate() {
            let last = (index + 1) * chunk_rows >= p;
            row = model
                .prefill_rows(&mut state, piece, last)
                .map_err(|e| io::Error::other(format!("{e:?}")))?;
        }
        row.ok_or_else(|| io::Error::other("kl probe prefix returned no logits"))
    };
    // REAL KL via the tested row_kl (a closure that never exponentiates
    // reports logit-sum noise, not KL).
    let compare = |a: &[f32], b: &[f32]| -> (f64, bool, usize) {
        let kl = crate::family::flint::mlx_qualification::row_kl(b, a);
        let (ea, eb): (Vec<f64>, Vec<f64>) = (
            a.iter().map(|&v| (v.max(-30.0)) as f64).collect(),
            b.iter().map(|&v| (v.max(-30.0)) as f64).collect(),
        );
        let top = |v: &[f64], n: usize| -> Vec<usize> {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|&x, &y| v[y].total_cmp(&v[x]));
            idx.truncate(n);
            idx
        };
        let (t1a, t1b) = (
            a.iter()
                .enumerate()
                .max_by(|x, y| x.1.total_cmp(y.1))
                .map(|(i, _)| i),
            b.iter()
                .enumerate()
                .max_by(|x, y| x.1.total_cmp(y.1))
                .map(|(i, _)| i),
        );
        let same_top1 = t1a.is_some() && t1a == t1b;
        let (fa, fb) = (top(&ea, 5), top(&eb, 5));
        let overlap = fa.iter().filter(|i| fb.contains(i)).count();
        (kl, same_top1, overlap)
    };
    for &p in &prefixes {
        let a = prefill(512, p)?;
        let b = prefill(1024, p)?;
        let (kl, top1, top5) = compare(&a, &b);
        // Noise floor: 512 twice (deterministic build => expect exactly 0).
        let a2 = prefill(512, p)?;
        let (kl0, top10, top50) = compare(&a, &a2);
        // Induction check: the corpus is a deterministic tiling, so the
        // model's top-1 at each prefix should be the next repeated token.
        let next_token = stream.get(p).copied();
        let top1_id = a
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.total_cmp(y.1))
            .map(|(i, _)| i as u32);
        let induction = next_token.is_some() && next_token == top1_id;
        // Gate only when a continuation token exists: at the corpus end
        // (p == stream.len()) there is nothing to induct on.
        assert!(
            next_token.is_none() || induction,
            "induction failed at prefix {p}: top-1 is not the next tiled token"
        );
        info["prefixes"][&p.to_string()] = json!({
            // row_kl(1024-arm, 512-arm): KL(1024||512)
            "kl_1024_vs_512": kl, "top1_equal": top1, "top5_overlap": top5,
            "floor_kl_512_vs_512": kl0, "floor_top1_equal": top10, "floor_top5_overlap": top50,
            "induction_top1_next_token": induction,
        });
        eprintln!(
            "flint_chunk_kl p={} kl={:.6} top1={} top5={}/5 floor_kl={:.6}",
            p, kl, top1, top5, kl0
        );
        metadata(&out, &info)?;
    }
    Ok(())
}

/// Ring/wrap cross-arm parity beyond the legacy 8192 boundary: the same
/// child-digest comparison as chained_fenced_bitwise_identity_2040, at deep
/// prefixes (default 8704/16384/30720) on the tiled corpus. Any row digest
/// difference between the chain-default and FIFI_FLINT_MLX_ASYNC_CHAIN=0
/// arms is a hard failure.
#[test]
#[ignore = "deep parity guard; loads Flint; exclusive parent-owned Mac window"]
fn multirow_chained_fenced_parity_deep() -> io::Result<()> {
    assert_eq!(
        std::env::var("FLINT_QUAL_GO").as_deref(),
        Ok("parent-owned-gpu-window"),
    );
    let binary = std::env::current_exe()?;
    let (root, _token_path, out) = qualification_paths()?;
    fs::create_dir(&out)?;
    let base = env_path("FLINT_QUAL_TOKENS")?.canonicalize()?;
    let deep: String = std::env::var("FLINT_QUAL_PROBE_PREFIXES")
        .unwrap_or_else(|_| "8704,16384,30720".to_string());
    let digests = spawn_multirow_child(
        &binary,
        true,
        &root,
        &base,
        &out.join("child-chain"),
        Some(&deep),
    )?;
    let digests_off = spawn_multirow_child(
        &binary,
        false,
        &root,
        &base,
        &out.join("child-fenced"),
        Some(&deep),
    )?;
    if digests != digests_off {
        return Err(io::Error::other(format!(
            "deep chain/fenced parity failed: {digests:?} vs {digests_off:?}"
        )));
    }
    Ok(())
}

/// Multi-row chain bench probe: warm one pp4096 pass, then run timed
/// pp4096 reps on fresh states, reporting tokens_per_second per rep for
/// the controller's counterbalanced armed/fenced arms. Uses the frozen
/// 2049-ID identity corpus plus FLINT_QUAL_PP_TOKENS (the 4097-ID pp
/// corpus).
#[test]
#[ignore = "bench probe; loads Flint; exclusive parent-owned window; the controller alternates PREFILL_CHAIN arms"]
fn multirow_bench_probe() -> io::Result<()> {
    assert_eq!(
        std::env::var("FLINT_QUAL_GO").as_deref(),
        Ok("parent-owned-gpu-window"),
    );
    clean_observation()?;
    let (root, token_path, out) = qualification_paths()?;
    let raw = fs::read(&token_path)?;
    let digest = format!("{:x}", Sha256::digest(&raw));
    if digest != "75c47e34ae9b6a3fc5190ec0e01273b88fee06fb6fdb933745ede28f1362dae0" {
        return Err(io::Error::other(
            "bench IDs differ from the frozen 2049-ID corpus",
        ));
    }
    let pp_path = env_path("FLINT_QUAL_PP_TOKENS")?.canonicalize()?;
    if !scratch(&pp_path) || fs::metadata(&pp_path)?.len() > 1024 * 1024 {
        return Err(io::Error::other("pp4096 IDs must use bounded task scratch"));
    }
    let pp_ids = parse_tokens(&fs::read(&pp_path)?)?;
    if pp_ids.len() != 4097 {
        return Err(io::Error::other("bench requires the 4097-ID pp corpus"));
    }
    let reps: usize = std::env::var("FLINT_QUAL_BENCH_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&r| (1..=8).contains(&r))
        .unwrap_or(3);
    let chunk_rows = probe_chunk_rows()?;
    fs::create_dir(&out)?;
    let mut info = json!({
        "source_commit": git("HEAD")?,
        "probe": "multirow-bench",
        "token_ids_sha256": digest,
        "pp_tokens": 4096,
        "chunk_rows": chunk_rows,
        "env_snapshot": env_snapshot(),
        "reps": reps,
        "prefill_chain_flag": if crate::family::flint::mlx_forward::prefill_chain_flagged() {
            "prefill-chained"
        } else {
            "prefill-fenced"
        },
        "warm": {},
        "timed": [],
    });
    metadata(&out, &info)?;
    let _serial = crate::native_test_lock();
    let mut model = Model::open(&root).map_err(|e| io::Error::other(format!("{e:?}")))?;
    {
        let mut warm = State::new();
        let (_, stats) =
            warm_then_measure_rows(&mut model, &mut warm, &pp_ids[..4096], chunk_rows)?;
        info["warm"] = stats;
        metadata(&out, &info)?;
    }
    if let Ok(stats) = model.kernels.memory_stats() {
        info["memory"] = json!({"active": stats.active, "cache": stats.cache, "peak": stats.peak});
        metadata(&out, &info)?;
    }
    for rep in 0..reps {
        let mut state = State::new();
        let (_, stats) =
            warm_then_measure_rows(&mut model, &mut state, &pp_ids[..4096], chunk_rows)?;
        info["timed"].as_array_mut().unwrap().push(stats.clone());
        eprintln!(
            "flint_bench rep={} tok_s={}",
            rep + 1,
            stats["tokens_per_second"]
        );
        metadata(&out, &info)?;
    }
    Ok(())
}

fn spawn_multirow_child(
    binary: &Path,
    armed: bool,
    model: &Path,
    tokens: &Path,
    child_out: &Path,
    prefixes: Option<&str>,
) -> io::Result<Vec<String>> {
    use std::process::Command;
    let label = if armed { "chained" } else { "fenced" };
    let mut command = Command::new(binary);
    command
        .args([
            "--exact",
            "family::flint::mlx_qualification::native::profile::batch::multirow_parity_probe",
            "--ignored",
            "--test-threads",
            "1",
        ])
        .env("FLINT_QUAL_GO", "parent-owned-gpu-window")
        .env("FLINT_QUAL_MODEL", model)
        .env("FLINT_QUAL_TOKENS", tokens)
        .env("FLINT_QUAL_OUTPUT", child_out)
        .env_remove("FLINT_QUAL_LAYER_CAPTURE")
        .env(
            "FLINT_QUAL_PROBE_PREFIXES",
            prefixes.unwrap_or("512,513,1024"),
        )
        // Pin the piece size: at the 1024 engine default the 512/513/1024
        // prefixes would all be single-chunk and the guard would stop
        // crossing chunk boundaries. Never inherit the parent's env.
        .env("FLINT_QUAL_CHUNK_ROWS", "512");
    if armed {
        command.env("FIFI_FLINT_MLX_PREFILL_CHAIN", "1");
    } else {
        command.env_remove("FIFI_FLINT_MLX_PREFILL_CHAIN");
    }
    let output = command
        .output()
        .map_err(|e| io::Error::other(format!("spawn the multirow {label} child: {e}")))?;
    assert!(
        output.status.success(),
        "the multirow {label} child exited with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let meta: Value = serde_json::from_str(&fs::read_to_string(child_out.join("metadata.json"))?)?;
    assert_eq!(
        meta["chunk_rows"],
        json!(512),
        "the guard child must run 512-row pieces"
    );
    assert_eq!(
        meta["prefill_chain_flag"],
        json!(format!("prefill-{label}")),
        "the multirow {label} child ran the wrong flag state"
    );
    Ok(meta["prefixes"]
        .as_object()
        .map(|prefixes| {
            prefixes
                .values()
                .map(|v| {
                    format!(
                        "{}|{}",
                        v["logits_sha256"].as_str().unwrap_or_default(),
                        v["planes_sha256"].as_str().unwrap_or_default()
                    )
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default())
}

/// In-tree multi-row chain guard: the prefill-group chain must be BITWISE
/// identical to the fenced default at every probed prefix — logits rows and
/// every layer's KV-plane bytes — on the frozen 2049-ID corpus (prefixes
/// 512/513/1024 cross the first chunk boundary, a 513-row odd remainder,
/// and a multi-chunk sequence).
#[test]
#[ignore = "in-tree multirow chain guard; loads Flint twice through child processes; exclusive parent-owned window"]
fn multirow_chained_fenced_bitwise_identity() -> io::Result<()> {
    assert_eq!(
        std::env::var("FLINT_QUAL_GO").as_deref(),
        Ok("parent-owned-gpu-window"),
    );
    clean_observation()?;
    let (model, tokens, out) = qualification_paths()?;
    fs::create_dir(&out)?;
    let binary = std::env::current_exe().expect("test binary");
    let mut digests = Vec::with_capacity(2);
    for (label, armed) in [("chained", true), ("fenced", false)] {
        let child_out = out.join(label);
        digests.push(spawn_multirow_child(
            &binary, armed, &model, &tokens, &child_out, None,
        )?);
    }
    assert_eq!(
        digests[0].len(),
        3,
        "the chained child recorded no prefixes"
    );
    assert_eq!(
        digests[0], digests[1],
        "chained and fenced multi-row prefills differ (logits or KV planes)"
    );
    Ok(())
}

/// Flag-off chunking calibration on real served prompt texts. The same prompts that
/// produced the armed streams are tokenized with the checkpoint's own tokenizer (the
/// production wrapper, so the id stream is the serving convention), prefilled twice at
/// flag-off - chunk_rows=1 vs the serving piece size - and 256 greedy tokens are
/// generated from each arm. The first divergence index per prompt is the engine-EXISTING
/// chunking noise the armed streams must be judged against. Prefix cap keeps the
/// per-token prefill inside the window budget.
#[test]
#[ignore = "requires an exclusive parent-owned Mac model/timing window"]
fn chunking_stream_calibration() -> io::Result<()> {
    assert_eq!(
        std::env::var("FLINT_QUAL_GO").as_deref(),
        Ok("parent-owned-gpu-window")
    );
    let ratchet_chunk = probe_chunk_rows()?;

    if crate::kernels::mlx::decode_attn_flagged() {
        return Err(io::Error::other(
            "chunking calibration is a flag-off measurement \
             (set FIFI_FLINT_MLX_DECODE_ATTN=0 to disable the vendor decode route)",
        ));
    }
    let _serial = crate::native_test_lock();
    let (root, _token_path, out) = qualification_paths()?;
    let prompts_dir = env_path("FLINT_QUAL_PROMPTS_DIR")?.canonicalize()?;
    if !scratch(&prompts_dir) {
        return Err(io::Error::other("prompts dir must use task scratch"));
    }
    let mut prompts: Vec<_> = fs::read_dir(&prompts_dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().is_some_and(|e| e == "txt"))
        .collect();
    prompts.sort();
    if prompts.is_empty() {
        return Err(io::Error::other("no .txt prompts found"));
    }
    const PREFIX_CAP: usize = 768;
    const DECODE: usize = 256;
    let tokenizer_bytes = fs::read(root.join("tokenizer.json"))?;
    let tokenizer = crate::tokenizer::Tokenizer::from_bytes(&tokenizer_bytes)
        .map_err(|error| io::Error::other(format!("{error:?}")))?;
    let mut model = Model::open(&root).map_err(|e| io::Error::other(format!("{e:?}")))?;
    fs::create_dir_all(&out)?;
    let mut info = json!({
        "git": git("HEAD")?, "prompt_count": prompts.len(),
        "prefix_cap": PREFIX_CAP, "decode_tokens": DECODE,
        "arms": ["chunk_rows=1", format!("chunk_rows={ratchet_chunk}")],
        "env_snapshot": env_snapshot(),
        "results": {},
    });
    metadata(&out, &info)?;
    for path in &prompts {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let text = fs::read_to_string(path)?;
        let ids: Vec<u32> = tokenizer
            .encode(&text)
            .map_err(|e| io::Error::other(format!("{e:?}")))?;
        if ids.len() < 8 {
            info["results"][&name] = json!({"skipped": "fewer than 8 prompt tokens"});
            continue;
        }
        let ids = &ids[..ids.len().min(PREFIX_CAP)];
        let mut per_token_state = State::new();
        let (row, _) = measured_rows(&mut model, &mut per_token_state, ids, 1)?;
        let (per_token, _) = greedy_rows(&mut model, &mut per_token_state, row, DECODE)?;
        let mut batched_state = State::new();
        let (row, _) = measured_rows(&mut model, &mut batched_state, ids, ratchet_chunk)?;
        let (batched, _) = greedy_rows(&mut model, &mut batched_state, row, DECODE)?;
        let first = per_token.iter().zip(&batched).position(|(a, b)| a != b);
        info["results"][&name] = json!({
            "prompt_tokens": ids.len(),
            "first_divergence": first,
            "identical_tokens": first.unwrap_or(per_token.len().min(batched.len())),
            "per_token_ids": per_token, "batched_ids": batched,
        });
        metadata(&out, &info)?;
    }
    Ok(())
}
