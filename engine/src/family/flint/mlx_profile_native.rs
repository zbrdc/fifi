//! Private model observations. The default product does not include this module.
use super::{Model, State, git, metadata, qualification_paths};
use crate::family::flint::mlx_qualification::{profile_tokens, write_row};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, io, time::Instant};

#[path = "mlx_attribution_profile_native.rs"]
mod attribution;
#[path = "mlx_batch_profile_native.rs"]
mod batch;
#[path = "mlx_decode_profile_native.rs"]
mod decode;

fn clean_observation() -> io::Result<()> {
    for key in [
        "FLINT_QUAL_TENSORS",
        "FLINT_QUAL_INPUTS",
        "FLINT_QUAL_RESIDUALS",
    ] {
        if std::env::var_os(key).is_some() {
            return Err(io::Error::other(
                "profile forbids capture or injected inputs",
            ));
        }
    }
    if std::env::vars_os().any(|(key, _)| {
        key.to_str()
            .is_some_and(|key| key.starts_with("INFERENCE_MLX_FAULT"))
    }) {
        return Err(io::Error::other(
            "profile requires fault injection disabled",
        ));
    }
    Ok(())
}

#[test]
#[ignore = "requires exclusive parent-owned Mac window and native-fault-qualification"]
fn one_token_sync_profile() -> io::Result<()> {
    assert_eq!(
        std::env::var("FLINT_QUAL_GO").as_deref(),
        Ok("parent-owned-gpu-window")
    );
    clean_observation()?;
    let (root, token_path, out) = qualification_paths()?;
    let raw = fs::read(&token_path)?;
    let digest = format!("{:x}", Sha256::digest(&raw));
    if digest != "75c47e34ae9b6a3fc5190ec0e01273b88fee06fb6fdb933745ede28f1362dae0" {
        return Err(io::Error::other(
            "profile input IDs differ from the qualified corpus",
        ));
    }
    let ids = profile_tokens(&raw)?;
    fs::create_dir(&out)?;
    let (async_chain, coalesce_residual) = (
        crate::family::flint::mlx_forward::async_chain_flagged(),
        crate::family::flint::mlx_forward::coalesce_residual_flagged(),
    );
    let mut info = json!({"source_commit":git("HEAD")?,"source_tree":git("HEAD^{tree}")?,
        "model_revision":"3b93f9053b530488405ae9fca51d4900d6ce21f0",
        "token_ids_sha256":digest,"warm_tokens":98,"measured_tokens":1,
        "position":98,"input_token":ids[98],"target_token":ids[99],"completed":false,
        "async_chain":async_chain,"coalesce_residual":coalesce_residual});
    metadata(&out, &info)?;
    let result = observe(&root, &out, &ids, &mut info);
    info["completed"] = json!(result.is_ok());
    if let Err(error) = &result {
        info["error"] = json!(error.to_string());
    }
    metadata(&out, &info)?;
    result
}

fn observe(
    root: &std::path::Path,
    out: &std::path::Path,
    ids: &[u32],
    info: &mut serde_json::Value,
) -> io::Result<()> {
    let _serial = crate::native_test_lock();
    let err = |error| io::Error::other(format!("{error:?}"));
    let mut model = Model::open(root).map_err(err)?;
    let mut state = State::new();
    for &token in &ids[..98] {
        model.step(&mut state, token, false).map_err(err)?;
    }
    model.kernels.qualification_reset_timing();
    model.kernels.qualification_reset_flush_stats();
    let sync_before = model.kernels.qualification_sync_count();
    let ops_before = model.kernels.selected_operation_count();
    let start = Instant::now();
    let row = model
        .step(&mut state, ids[98], true)
        .map_err(err)?
        .ok_or_else(|| io::Error::other("profile returned no logits"))?;
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
    info["profile"] = json!({"step_seconds":seconds,"import_calls":timing.import_calls,
        "import_ns":timing.import_ns,"readback_calls":timing.readback_calls,
        "readback_ns":timing.readback_ns,"evaluate_calls":timing.evaluate_calls,
        "evaluate_ns":timing.evaluate_ns,"native_synchronizations":syncs,"selected_operations":ops,
        "forced_retire_flushes":flush.flushes,"flush_sync_ns":flush.sync_ns,"flush_free_ns":flush.free_ns,
        "profile_build_not_production_throughput":true});
    let mut bytes = Vec::new();
    write_row(&mut bytes, &row)?;
    info["logits_sha256"] = json!(format!("{:x}", Sha256::digest(&bytes)));
    fs::write(out.join("logits.f32le"), bytes)
}

// The original D1 RED target (`d1_decode_step_counter_target`, 49 evaluates
// once per layer) retired at the flip fold: its expectation matches no
// current regime (chain on = 13 group roots, unchained coalesced = 98, all
// fences = 147) and the coalesce target below pins the live counts per arm.

#[test]
#[ignore = "D1 first slice; loads Flint; run once with FIFI_FLINT_MLX_COALESCE_RESIDUAL=0 and once with =1 (default on), BOTH with FIFI_FLINT_MLX_ASYNC_CHAIN=0 (the chain subsumes the coalesce counts at 13 group roots); the harness compares the two metadata.json logits hashes for bitwise parity"]
fn d1_residual_coalesce_counter_target() -> io::Result<()> {
    let switched = crate::family::flint::mlx_forward::coalesce_residual_flagged();
    let chained = crate::family::flint::mlx_forward::async_chain_flagged();
    one_token_sync_profile()?;
    let (_, _, out) = qualification_paths()?;
    let info: serde_json::Value = serde_json::from_slice(&fs::read(out.join("metadata.json"))?)?;
    let profile = &info["profile"];
    println!("D1_RESIDUAL_COALESCE switched={switched} chained={chained} {profile}");
    assert_eq!(
        profile["evaluate_calls"],
        if chained {
            13
        } else if switched {
            98
        } else {
            147
        },
        "the chain (13 group roots) subsumes the coalesce counts; unchained the switch \
         drops the 49 residual roots (98) and both cache fences stay without it (147)"
    );
    assert_eq!(profile["import_calls"], 1);
    assert_eq!(profile["readback_calls"], 1);
    Ok(())
}

/// D2 target (design scrch/7c82644c/flint-async-chain/DESIGN.md): the async chain
/// evaluates layers N..N+3 as ONE graph, so `commit_chunk` loses its two host-ordered
/// fences per layer (49x2 = 98 evaluates) and the route-health readbacks batch into the
/// group roots (48 readbacks -> 1). Run once with FIFI_FLINT_MLX_ASYNC_CHAIN=0
/// (the rollback env; the default is ON since the 2026-10-25 window) and once
/// with =1 or unset; the harness compares the two metadata.json logits hashes
/// for bitwise parity.
#[test]
#[ignore = "D2 target; loads Flint and requires a separate parent-owned model window"]
fn d2_async_chain_counter_target() -> io::Result<()> {
    let chained = crate::family::flint::mlx_forward::async_chain_flagged();
    one_token_sync_profile()?;
    let (_, _, out) = qualification_paths()?;
    let info: serde_json::Value = serde_json::from_slice(&fs::read(out.join("metadata.json"))?)?;
    let profile = &info["profile"];
    println!("D2_ASYNC_CHAIN chained={chained} {profile}");
    if chained {
        assert_eq!(
            profile["evaluate_calls"], 13,
            "D2 roots one evaluate per 4-layer group: 49 layers -> 13 groups, not 98 fences"
        );
        assert_eq!(
            profile["readback_calls"], 1,
            "D2 batches the route-health readbacks into the group roots: 48 -> 1"
        );
        // No import_calls assert: the 144 imports are D1-remaining scope; D2 owns
        // evaluates, readbacks, and syncs only.
        assert!(
            profile["native_synchronizations"]
                .as_u64()
                .is_some_and(|syncs| syncs <= D2_SYNC_BOUND),
            "D2 sync bound (run3-calibrated); unchained baseline is 290"
        );
    } else {
        assert_eq!(
            profile["evaluate_calls"], 98,
            "the unchained baseline keeps the two commit_chunk fences per layer (49x2)"
        );
    }
    assert!(profile["forced_retire_flushes"].is_u64());
    assert_eq!(
        info["async_chain"].as_bool(),
        Some(chained),
        "metadata records the production parser's effective state"
    );
    Ok(())
}

/// The unchained profile at the D2 design baseline measured 290 native synchronizations
/// (98 evaluate fences + 48 health readbacks + 144 imports). The chain removes the fence
/// and readback syncs (98 + 47) and adds 13 group roots + 1 readback, leaving the import
/// syncs untouched: 290 - 145 + 14 = 159. Calibrated against the mac-attr3 run3
/// per-class sync sums; revisit if run3 revises the 290 decomposition.
const D2_SYNC_BOUND: u64 = 160;
