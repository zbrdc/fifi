//! Representative fixed-work host-wall diagnostics, not production throughput or GPU timing.
use super::{Model, State, clean_observation, metadata, qualification_paths};
use crate::family::sampler::Sampler;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, Read},
    path::Path,
    time::Instant,
};

fn err(error: impl std::fmt::Debug) -> io::Error {
    io::Error::other(format!("{error:?}"))
}

#[test]
#[ignore = "requires exclusive parent-owned Mac window and native-fault-qualification"]
fn four_k_decode_sync_profile() -> io::Result<()> {
    assert_eq!(
        std::env::var("FLINT_QUAL_GO").as_deref(),
        Ok("parent-owned-gpu-window")
    );
    clean_observation()?;
    if std::env::var_os("FIFI_FLINT_MLX_BATCH_PREFILL").is_some()
        || !crate::family::flint::mlx_forward::coalesce_residual_flagged()
    {
        return Err(io::Error::other(
            "profile requires the serving piece-size env unset (it pins its own S4 512-row worker pieces) and D1 ON",
        ));
    }
    let chained = crate::family::flint::mlx_forward::async_chain_flagged();
    let (root, token_path, out) = qualification_paths()?;
    if !fs::metadata(&token_path)?.is_file() {
        return Err(io::Error::other("profile input must be a regular file"));
    }
    let mut raw = Vec::new();
    fs::File::open(&token_path)?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut raw)?;
    let digest = format!("{:x}", Sha256::digest(&raw));
    if digest != "d72c7b49525c2c5fda9e60f6b6e5b1573a83c52199380524315d62a189185843" {
        return Err(io::Error::other(
            "profile requires the frozen S4 4097-ID speed stimulus",
        ));
    }
    let ids: Vec<u32> = serde_json::from_slice(&raw)?;
    if ids.len() != 4097 || ids.iter().any(|&id| id >= 150_144) {
        return Err(io::Error::other("invalid frozen speed IDs"));
    }
    fs::create_dir(&out)?;
    let mut info = json!({"source_commit":super::git("HEAD")?,"source_tree":super::git("HEAD^{tree}")?,
        "model_revision":"3b93f9053b530488405ae9fca51d4900d6ce21f0","token_ids_sha256":digest,
        "prefill_piece":512,"prefix_fed_untimed":4095,"requested_forwards":128,"start_position":4095,
        "discarded_warm_decode_forwards":128,
        "first_feed_id":ids[4095],"unused_trailing_ids":1,"vocab":150144,"ignores_eos":true,
        "d1_coalescing":true,"async_chain":chained,"profile_build_not_production_throughput":true,
        "evaluate_is_host_wall_not_gpu_execution_or_occupancy":true,"completed":false});
    metadata(&out, &info)?;
    let result = observe(&root, &out, &ids, &mut info);
    info["completed"] = json!(result.is_ok());
    if let Err(error) = &result {
        info["error"] = json!(error.to_string());
    }
    metadata(&out, &info)?;
    result
}

fn prefix(model: &mut Model, state: &mut State, ids: &[u32]) -> io::Result<()> {
    // The S4 worker shapes, NOT the serving default: pinned [511, 512, ...,
    // 512] pieces so the profile reproduces the S4 measurement exactly.
    for start in (0usize..4096).step_by(512) {
        let begin = start.saturating_sub(1);
        let end = (start + 512).min(4096) - 1;
        if model
            .prefill_rows(state, &ids[begin..end], false)
            .map_err(err)?
            .is_some()
        {
            return Err(io::Error::other("no-head prefix returned logits"));
        }
    }
    if state.pos != 4095 {
        return Err(io::Error::other("prefix position mismatch"));
    }
    Ok(())
}

fn decode(
    model: &mut Model,
    state: &mut State,
    first: u32,
) -> io::Result<(Vec<Vec<f32>>, Vec<u32>, f64)> {
    let mut pending = first;
    let mut sampler = Sampler::greedy();
    let mut rows = Vec::with_capacity(128);
    let mut tokens = Vec::with_capacity(128);
    let start = Instant::now();
    for _ in 0..128 {
        let row = model
            .step(state, pending, true)
            .map_err(err)?
            .ok_or_else(|| io::Error::other("profile returned no logits"))?;
        if row.len() != 150_144 {
            return Err(io::Error::other("invalid decode logits"));
        }
        pending = sampler.sample(&row).map_err(err)?;
        rows.push(row);
        tokens.push(pending);
    }
    Ok((rows, tokens, start.elapsed().as_secs_f64()))
}

fn observe(root: &Path, out: &Path, ids: &[u32], info: &mut Value) -> io::Result<()> {
    // The OnceLock-cached production parser: the same value the outer test
    // read and recorded, so the per-arm pin below cannot drift from metadata.
    let chained = crate::family::flint::mlx_forward::async_chain_flagged();
    let _serial = crate::native_test_lock();
    let mut model = Model::open(root).map_err(err)?;
    {
        let mut warm = State::new();
        prefix(&mut model, &mut warm, ids)?;
        let _ = decode(&mut model, &mut warm, ids[4095])?;
    }
    let mut state = State::new();
    prefix(&mut model, &mut state, ids)?;
    model.kernels.qualification_reset_timing();
    model.kernels.qualification_reset_flush_stats();
    let sync_before = model.kernels.qualification_sync_count();
    let ops_before = model.kernels.selected_operation_count();
    let labels_snapshot = model.kernels.operation_label_counts();
    let capture = model
        .kernels
        .qualification_capture_evaluate_wall()
        .map_err(err)?;
    let start = Instant::now();
    let result = decode(&mut model, &mut state, ids[4095]);
    let seconds = result
        .as_ref()
        .map_or(start.elapsed().as_secs_f64(), |r| r.2);
    let samples = capture.finish();
    let timing = model.kernels.qualification_timing();
    let flush = model.kernels.qualification_flush_stats();
    let sum = samples
        .ns
        .iter()
        .try_fold(0u64, |sum, &ns| sum.checked_add(ns))
        .ok_or_else(|| io::Error::other("evaluate wall sum overflow"))?;
    let syncs = model
        .kernels
        .qualification_sync_count()
        .checked_sub(sync_before);
    let ops = model
        .kernels
        .selected_operation_count()
        .checked_sub(ops_before);
    // Per-MLX-primitive split of the same selected-operations accounting: the
    // delta over the timed decode, ranked descending by count.
    let labels_after = model.kernels.operation_label_counts();
    let labels_before = labels_snapshot;
    let label_delta: Vec<(String, u64)> = labels_after
        .into_iter()
        .filter_map(|(name, count)| {
            let before = labels_before
                .iter()
                .find(|(n, _)| n == &name)
                .map_or(0, |(_, c)| *c);
            (count > before).then_some((name, count - before))
        })
        .collect();
    let label_sum: u64 = label_delta.iter().map(|(_, c)| c).sum();
    info["profile"] = json!({"seconds":seconds,"ms_per_forward":seconds*1000.0/128.0,
        "import_calls":timing.import_calls,"import_ns":timing.import_ns,
        "readback_calls":timing.readback_calls,"readback_ns":timing.readback_ns,
        "evaluate_calls":timing.evaluate_calls,"evaluate_ns":timing.evaluate_ns,
        "native_synchronizations":syncs,"selected_operations":ops,
        "forced_retire_flushes":flush.flushes,"flush_sync_ns":flush.sync_ns,"flush_free_ns":flush.free_ns,
        "evaluate_samples":samples.ns.len(),"dropped_samples":samples.dropped,
        "buckets_upper_inclusive_ns_and_count":samples.histogram(),
        "nearest_rank_p50_p90_p99_ns":samples.percentiles(),"captured_evaluate_ns":sum,
        "selected_operation_labels":label_delta.clone()});
    fs::write(
        out.join("evaluate-wall-ns.json"),
        serde_json::to_vec(&samples.ns)?,
    )?;
    let (rows, tokens, _) = result?;
    if syncs.is_none()
        || ops.is_none()
        || state.pos != 4223
        || rows.len() != 128
        || tokens.len() != 128
        || seconds <= 0.0
        || samples.dropped != 0
        || samples.ns.len() as u64 != timing.evaluate_calls
        || sum != timing.evaluate_ns
        || timing.evaluate_calls != 128 * if chained { 13 } else { 98 }
        // The label choke point (produce) and the native counter do not see
        // exactly the same call set (compound entries count once natively but
        // build through produce repeatedly), so accept a +-20% band: the
        // ranking is what the table is for.
        || ops.is_some_and(|ops| {
            label_sum < ops * 4 / 5 || label_sum > ops * 6 / 5
        })
    {
        return Err(io::Error::other(
            "decode work, per-arm evaluate counts, label split, or capture mismatch",
        ));
    }
    let mut hashes = Vec::with_capacity(128);
    for (index, row) in rows.iter().enumerate() {
        let mut bytes = Vec::with_capacity(row.len() * 4);
        super::write_row(&mut bytes, row)?;
        hashes.push(format!("{:x}", Sha256::digest(&bytes)));
        fs::write(out.join(format!("logits-{index:03}.f32le")), bytes)?;
    }
    info["decode128"] = json!({"forwards":128,"outputs":tokens,"start_position":4095,
        "end_position":state.pos,"first_feed_id":ids[4095],"final_sampled_id_unfed":tokens[127],
        "seconds":seconds,"ms_per_forward":seconds*1000.0/128.0,"logits_sha256_f32le":hashes,
        "ignores_eos":true,"production_eos_throughput":false,"prefix_fed_untimed":4095});
    Ok(())
}
// KV-plane copy-scale measurement (2026-10-25, window 9): how many plane rows
// the decode path copies per step through concatenate/kv-append, split by
// site and window (SWA) vs full-attention layers, at a 4k and an 8k prefix
// (the model context limit). One row of one plane is 1 KiB (4x128 BF16), so
// rows/step is KiB/step moved; every site records per keys/values call
// separately. One `decode` call is 128 forwards; the timed window counts
// only those. Synthetic deterministic ids (no frozen stimulus: no logits are
// asserted). Mac GPU allocation via the parent-owned window protocol.

/// The four_k `prefix` helper hardcodes the 4096-token stimulus; this generic
/// version feeds any length in the same [511, 512, ..., pending-last] worker
/// pattern so the copy-scale measurement can run at 4k and 8k.
fn chunked_prefix(model: &mut Model, state: &mut State, ids: &[u32]) -> io::Result<()> {
    let last = ids.len().saturating_sub(1);
    let mut fed = 0usize;
    while fed < last {
        let take = if fed == 0 {
            (last - fed).min(511)
        } else {
            (last - fed).min(512)
        };
        if model
            .prefill_rows(state, &ids[fed..fed + take], false)
            .map_err(err)?
            .is_some()
        {
            return Err(io::Error::other("no-head prefix returned logits"));
        }
        fed += take;
    }
    if state.pos as usize != last {
        return Err(io::Error::other("generic prefix position mismatch"));
    }
    Ok(())
}

#[test]
#[ignore = "requires exclusive parent-owned Mac window and native-fault-qualification"]
fn kv_plane_copy_scale() -> io::Result<()> {
    assert_eq!(
        std::env::var("FLINT_QUAL_GO").as_deref(),
        Ok("parent-owned-gpu-window")
    );
    let _serial = crate::native_test_lock();
    let (root, _, out) = qualification_paths()?;
    fs::create_dir(&out)?;
    let mut info = json!({"source_commit": super::git("HEAD")?,"measurement":"kv_plane_copy_scale",
        "one_row_bytes":1024,"model_revision":"3b93f9053b530488405ae9fca51d4900d6ce21f0",
        "profile_build_not_production_throughput":true,"completed":false});
    let mut model = Model::open(&root).map_err(err)?;
    let id = |i: u32| (i.wrapping_mul(7919).wrapping_add(11)) % 150_000;
    let mut table = Vec::new();
    for prefix_len in [4000_u32, 8000] {
        let ids: Vec<u32> = (0..=prefix_len).map(id).collect();
        {
            let mut warm = State::new();
            chunked_prefix(&mut model, &mut warm, &ids)?;
            let _ = decode(&mut model, &mut warm, ids[prefix_len as usize])?;
        }
        let mut state = State::new();
        chunked_prefix(&mut model, &mut state, &ids)?;
        // Snapshot after the prefix flood: the timed window below is exactly
        // one decode call (128 forwards).
        let start = crate::family::flint::mlx_state::plane_copy_samples().len();
        let _ = decode(&mut model, &mut state, ids[prefix_len as usize])?;
        let samples = crate::family::flint::mlx_state::plane_copy_samples();
        let timed = &samples[start..];
        let steps = 128_u64;
        let mut rows: std::collections::BTreeMap<(&'static str, bool), (u64, u64)> =
            std::collections::BTreeMap::new();
        for (site, windowed, count, _) in timed {
            let e = rows.entry((site, *windowed)).or_insert((0, 0));
            e.0 += u64::from(*count);
            e.1 += 1;
        }
        let per_step: Vec<(String, u64)> = rows
            .iter()
            .map(|((site, windowed), (row_sum, _calls))| {
                (
                    format!("{site}{}", if *windowed { "/swa" } else { "/full" }),
                    row_sum / steps,
                )
            })
            .collect();
        let kib_per_step: u64 = per_step.iter().map(|(_, r)| r).sum();
        table.push(json!({"prefix":prefix_len,"steps":steps,
            "rows_per_step_by_site":per_step,"kib_per_step_all_sites":kib_per_step}));
    }
    info["kv_plane_copy_scale"] = json!(table);
    info["completed"] = json!(true);
    metadata(&out, &info)
}

/// Ring-view gate (2026-10-25, base-failing RED): with FIFI_FLINT_MLX_RING_VIEW
/// forced on, a 4k-prefix decode must not materialize ANY full-plane union
/// copy (union_concat/full and union_head_tail/full rows/step == 0) while the
/// SWA planes keep their bounded two-slice union (> 0 rows/step). Pins the
/// window-12 baseline (299,628 KiB/step total at 4k) until the slice lands.
#[test]
#[ignore = "requires exclusive parent-owned Mac window and native-fault-qualification"]
fn ring_view_eliminates_full_plane_union_copies() -> io::Result<()> {
    use crate::family::flint::mlx_state::plane_copy_samples;
    assert_eq!(
        std::env::var("FLINT_QUAL_GO").as_deref(),
        Ok("parent-owned-gpu-window")
    );
    // The flag is set by the run wrapper (edition 2024 forbids set_var; the
    // T1_FLASH two-process convention): this process must come in armed.
    assert!(
        crate::family::flint::mlx_forward::ring_view_flagged(),
        "gate process must start with FIFI_FLINT_MLX_RING_VIEW armed"
    );
    let _serial = crate::native_test_lock();
    let (root, _, out) = qualification_paths()?;
    fs::create_dir(&out)?;
    let mut model = Model::open(&root).map_err(err)?;
    let mut state = State::new();
    let ids: Vec<u32> = (0..=4000u32)
        .map(|i| (i.wrapping_mul(7919).wrapping_add(11)) % 150_000)
        .collect();
    chunked_prefix(&mut model, &mut state, &ids)?;
    let start = plane_copy_samples().len();
    let _ = decode(&mut model, &mut state, ids[4000])?;
    let samples = plane_copy_samples();
    let timed = &samples[start..];
    let steps = 128_u64;
    let mut full_rows = 0_u64;
    let mut swa_rows = 0_u64;
    for (site, windowed, rows, _) in timed {
        if !site.starts_with("union_concat") && !site.starts_with("union_head_tail") {
            continue;
        }
        if *windowed {
            swa_rows += u64::from(*rows);
        } else {
            full_rows += u64::from(*rows);
        }
    }
    // The report lands BEFORE the asserts so a failing run still leaves
    // its evidence on disk (Opus SF-D).
    fs::write(
        out.join("ring-view-gate.json"),
        serde_json::to_vec_pretty(&json!({
            "full_plane_union_rows_per_step": full_rows / steps,
            "swa_union_rows_per_step": swa_rows / steps,
        }))?,
    )?;
    assert!(
        crate::family::flint::mlx_forward::async_chain_flagged(),
        "the gate must run in the chained mode it routes for"
    );
    assert_eq!(
        full_rows, 0,
        "ring view must eliminate every full-plane union copy (raw rows, not per-step)"
    );
    assert_eq!(
        full_rows / steps,
        0,
        "ring view must eliminate every full-plane union copy"
    );
    assert!(
        swa_rows > 0,
        "SWA planes keep their bounded two-slice union in this slice"
    );
    Ok(())
}

/// Ring-view boundary sweep (2026-10-25 window 16): one model load per arm;
/// the run wrapper executes this once flag-off and once flag-on and the
/// comparison is host-side. Covers: per-prefix greedy logits digests at the
/// ring/chunk/context boundaries (p0 = 0, 511-514, 2048, 4095-4097, 8064 =
/// the last 128 rows to 8191, 8191, and the legacy 8192 context limit
/// (sweep coverage predates the 32768 default)); a
/// continuation (4k prefix -> decode -> 256-row multi-row chunk -> decode);
/// a 600-token token-by-token prefill; per-phase device peak memory and
/// process retire/flush/sync stats (the Opus O-2 watch item); and one
/// 128-decode timing sample at 4k and at the 8k boundary.
#[test]
#[ignore = "requires exclusive parent-owned Mac window and native-fault-qualification"]
fn ring_view_boundary_sweep() -> io::Result<()> {
    use crate::family::flint::mlx_forward::ring_view_flagged;
    assert_eq!(
        std::env::var("FLINT_QUAL_GO").as_deref(),
        Ok("parent-owned-gpu-window")
    );
    let _serial = crate::native_test_lock();
    let (root, _, out) = qualification_paths()?;
    fs::create_dir(&out)?;
    let mut info = json!({
        "source_commit": super::git("HEAD")?,
        "measurement": "ring_view_boundary_sweep",
        "ring_view_flagged": ring_view_flagged(),
        "completed": false,
    });
    metadata(&out, &info)?;
    let mut model = Model::open(&root).map_err(err)?;
    let id = |i: u32| (i.wrapping_mul(7919).wrapping_add(11)) % 150_000;

    // n greedy decodes from `first`; returns (sha over the concatenated
    // logits rows, chosen tokens, wall seconds).
    let decode_n = |model: &mut Model, state: &mut State, first: u32, n: u32| {
        let mut pending = first;
        let mut sampler = Sampler::greedy();
        let mut hasher = Sha256::new();
        let mut tokens = Vec::new();
        let start = Instant::now();
        for _ in 0..n {
            let row = model
                .step(state, pending, true)
                .map_err(err)?
                .ok_or_else(|| io::Error::other("sweep decode returned no logits"))?;
            for v in &row {
                hasher.update(v.to_le_bytes());
            }
            pending = sampler.sample(&row).map_err(err)?;
            tokens.push(pending);
        }
        Ok::<_, io::Error>((
            format!("{:x}", hasher.finalize()),
            tokens,
            start.elapsed().as_secs_f64(),
        ))
    };

    let mut sweep = Vec::new();
    for prefix_len in [
        1_u32, 511, 512, 513, 514, 2048, 4095, 4096, 4097, 8064, 8191, 8192,
    ] {
        let mut state = State::new();
        let ids: Vec<u32> = (0..prefix_len).map(id).collect();
        chunked_prefix(&mut model, &mut state, &ids)?;
        // The legacy 8192 budget left room for 8193 - L decodes after a
        // prefix of L ids (chunked_prefix parks the last id pending); the
        // sweep predates the 32768 default and still probes the legacy
        // boundary, so the 8k timing sample runs
        // at 8064 - the last-128-rows-to-8191 boundary.
        let timed = matches!(prefix_len, 4096 | 8064);
        let n = (if timed { 128 } else { 4 }).min(8193 - prefix_len); // legacy-8192 sweep bound
        model.kernels.reset_peak_memory().map_err(err)?;
        model.kernels.qualification_reset_retire_peak();
        let (flushes_before, _) = model.kernels.qualification_retire_stats();
        let sync_before = model.kernels.qualification_sync_count();
        let (sha, tokens, seconds) =
            decode_n(&mut model, &mut state, ids[(prefix_len - 1) as usize], n)?;
        let memory = model.kernels.memory_stats().map_err(err)?;
        let (flushes, retired_peak) = model.kernels.qualification_retire_stats();
        sweep.push(json!({
            "prefix_len": prefix_len, "decodes": n, "logits_sha256": sha,
            "first_tokens": tokens[..4.min(tokens.len())],
            "seconds": seconds,
            "ms_per_forward": seconds * 1000.0 / n as f64,
            "peak_active_bytes": memory.active, "peak_bytes": memory.peak,
            "forced_flushes": flushes - flushes_before,
            "retired_peak_bytes": retired_peak,
            "syncs": model.kernels.qualification_sync_count() - sync_before,
        }));
        metadata(&out, &info.clone())?;
        info["sweep"] = json!(sweep);
        metadata(&out, &info)?;
    }

    // Continuation: 4k prefix, decode, a 256-row multi-row chunk, decode.
    let mut state = State::new();
    let ids: Vec<u32> = (0..4096_u32).map(id).collect();
    chunked_prefix(&mut model, &mut state, &ids)?;
    let (sha1, toks1, _) = decode_n(&mut model, &mut state, ids[4095], 4)?;
    // A multi-row chunk fed mid-conversation: chunked_prefix asserts a
    // fresh state, so feed `more` from the CURRENT position directly,
    // leaving its last id pending exactly like the prefix helpers do.
    let more: Vec<u32> = (4096_u32..4352).map(id).collect();
    let last = more.len() - 1;
    let pos_before = state.pos as usize;
    let mut fed = 0usize;
    while fed < last {
        let take = (last - fed).min(512);
        if model
            .prefill_rows(&mut state, &more[fed..fed + take], false)
            .map_err(err)?
            .is_some()
        {
            return Err(io::Error::other("continuation chunk returned logits"));
        }
        fed += take;
    }
    if state.pos as usize != pos_before + last {
        return Err(io::Error::other("continuation position mismatch"));
    }
    let (sha2, _, _) = decode_n(&mut model, &mut state, more[last], 4)?;
    info["continuation"] = json!({
        "after_prefix_sha256": sha1, "first_tokens": toks1[..4.min(toks1.len())],
        "after_multi_row_chunk_sha256": sha2, "multi_row_len": more.len(),
    });
    metadata(&out, &info)?;

    // 600-token token-by-token prefill (one-row chunks from p0 = 0).
    let mut state = State::new();
    let ids: Vec<u32> = (0..600_u32).map(id).collect();
    for (i, token) in ids.iter().enumerate() {
        if model
            .step(&mut state, *token, false)
            .map_err(err)?
            .is_some()
        {
            return Err(io::Error::other("token-by-token prefill returned logits"));
        }
        if state.pos as usize != i + 1 {
            return Err(io::Error::other("token-by-token position mismatch"));
        }
    }
    let (sha3, _, _) = decode_n(&mut model, &mut state, ids[599], 4)?;
    info["token_by_token_600"] = json!({"logits_sha256": sha3});
    info["completed"] = json!(true);
    metadata(&out, &info)?;
    Ok(())
}
