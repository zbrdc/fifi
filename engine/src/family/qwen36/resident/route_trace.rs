//! The router traces of the real Q36 model (Mac, ignored, real checkpoint).
//!
//! The question: which experts does the real router select, per layer and per token, over a real
//! prompt? The decision head's only role is predicting experts for speculative SSD
//! prefetch, and every later accuracy claim needs these traces plus a lookahead baseline.
//! Nothing exists yet, so this harness records the ground truth once.
//!
//! NO ROUTER IS REIMPLEMENTED and no production path changes. The harness runs the production
//! one-token step `ResidentDecoder::step` in `Record` mode with `Routing::Host`, the seam the
//! parity gates already use: `StepResult.routes` holds the expert ids of every layer in rank
//! order and `StepResult.probs` the router probability row `[experts]` behind each recorded
//! route. The gates come from `route::route_experts`, the function
//! the step itself routed with, run on the exact recorded rows, so the derived ids equal the
//! reported ids by construction; the report carries that equality as the summary field
//! `ids_equal_step_routes` instead of an assert.
//!
//! THE BATCHED CHUNK PATH CANNOT BE REACHED FROM HERE without a production change: the stage
//! functions of `prefill` (`prefill::attn`, `prefill::delta`, `prefill::moe`) are private to
//! `prefill`, and `ResidentDecoder::prefill` drops the host routes of every chunk
//! (`routes: None`), so a chunked trace would need a mirror inside `prefill` (the wiring
//! `prefill/timing.rs` mirrors) or a visibility change. The one-token step IS the serving path
//! without an admitted chunk (the serving fallback feeds one token per `decoder.step`), and
//! `Record` routes EVERY layer including the last, where a `Prompt` chunk skips the last
//! layer's MoE. Two honest differences from a chunked prefill stay: the router matmul runs
//! `[1, H]` instead of `[T, H]` (a different F32 accumulation may flip a near tie), and
//! `Record` also reads back the logits row and the layer rows of every token (about 1.3 MB per
//! token on this model), so a trace names a depth the run can afford; the routing rows add no
//! synchronization of their own.
//!
//! Inputs by environment, the contract of `prefill/timing.rs` (whose parse helpers are private
//! and frozen, so the rules are mirrored here as `timing.rs` mirrors `decode_timing`):
//!
//! | Variable | Meaning | Default |
//! | --- | --- | --- |
//! | `Q36_CKPT_DIR` | the pinned checkpoint directory | required |
//! | `Q36_OUT_DIR` | the output directory; created after every input is validated, right before the load | required |
//! | `Q36_ACTIVATION` | the activation dtype: `bf16` or `f32` | required |
//! | `Q36_KV_DTYPE` | the attention cache form: `f32` or `q8` | `f32` |
//! | `DT_IDS_SOURCE` | must be `file`: this harness reads `Q36_PROMPT_IDS`, the only deep source | required `file` |
//! | `Q36_PROMPT_IDS` | a file of little-endian `u32` ids, the format `route_parity` reads | required |
//! | `DT_DEPTHS` | ONE prompt depth in tokens; the timing default names two depths and is refused | set one depth |
//!
//! A depth must fit the ids source and the context limit (262,144).
//!
//! The session budget covers the whole depth through `chunk_growth_bytes`, the same product the
//! per-token charges sum to, raised over the shipped cap when the depth needs more, exactly as
//! `prefill/timing.rs` raises it; the report records the cap the run used.
//!
//! Output: `router_traces.json` in `Q36_OUT_DIR` — the config identity (family, checkpoint
//! path, `config_sha256`, the dtypes), the prompt ids sha256 over the little-endian `u32` bytes
//! of the traced ids, one row per layer holding the `top_k` expert ids in choice order and
//! their renormalized gates for every token in prompt order (row `t` of every layer is token
//! `t`; the per-token arrays make the file grow linearly with the depth, so it is written
//! COMPACT, not pretty), and a summary: total tokens, layers, top-k, the expert-usage
//! histogram over every selection, the count of experts ever selected, and the mean and maximum
//! gate per layer. Plus the one printed line
//! `router_traces={"layers":N,"tokens":N,"top_k":N,"experts_seen":N,"rows":N}`. The test
//! asserts only that the run reaches the depth and that every token of every layer selects
//! exactly `top_k` experts (ids and gates); it asserts no accuracy or distribution bound.
//!
//! The exact Mac command:
//!
//! ```text
//! Q36_CKPT_DIR=<CKPT_DIR> \
//! Q36_OUT_DIR=<OUT_DIR> \
//! Q36_ACTIVATION=bf16 \
//! DT_IDS_SOURCE=file \
//! Q36_PROMPT_IDS=<PROMPT_IDS> \
//! DT_DEPTHS=922 \
//! cargo test --release --features native-mlx --lib \
//!   family::qwen36::resident::route_trace::capture_router_traces -- --ignored --exact --nocapture
//! ```
//!
//! `capture_layer_states` uses the same environment contract. It writes `layer_states.bin` as
//! little-endian F32 in row-major `[token][layer][hidden]` order, without a header or padding.
//! `layer_states.json` records the shape, ordering, identities, binary hash, and expert ids in
//! `[token][layer][choice]` order. The projected binary size must not exceed 512 MiB.
//! For same-token prediction, pair residual `layer_out[i]` with experts `routes[i + 1]`.
//! That residual precedes the next layer's mixer and post norm, not just its router.
//! The last layer has no next-layer target. This capture asserts no prediction accuracy.

use std::io::Write;

use super::decode_timing::{read_prompt_ids, required_activation_from_env};
use super::decoder::{ResidentDecoder, StepMode, StepResult};
use super::moe::Routing;
use super::parity::{CONFIG_SHA256, sha256};
use super::real::{env_path, kv_dtype_from_env, open};
use super::state::chunk_growth_bytes;
use super::tool_forward::{
    CACHE_LIMIT, CONTEXT_LIMIT, SESSION_CAP, create_out_dir, out_dir_must_be_new,
};
use crate::{
    family::{
        qwen36::{
            host_load::HostWeights,
            settings::{activation_name, kv_dtype_name},
        },
        route::route_experts,
        spec::{QWEN3_6_35B_A3B, Router},
    },
    kernels::MlxKernels,
    sched::SessionBudget,
};

/// Requires `DT_IDS_SOURCE=file`: this harness reads `Q36_PROMPT_IDS` only, the deep ids source
/// of `decode_timing` (the rule of `prefill/timing.rs`, whose private helper is mirrored here).
fn require_file_ids_source() {
    match std::env::var("DT_IDS_SOURCE") {
        Ok(text) if text == "file" => {}
        Ok(text) => panic!("DT_IDS_SOURCE must be file for this harness, got {text:?}"),
        Err(std::env::VarError::NotPresent) => panic!("set DT_IDS_SOURCE=file"),
        Err(error) => panic!("DT_IDS_SOURCE: {error}"),
    }
}

/// Parses `DT_DEPTHS` exactly as `prefill/timing.rs` does (that helper is private, so the
/// contract is mirrored here as `timing.rs` mirrors `decode_timing`): a comma list of depths
/// from 1 to `max_depth`, entries trimmed, no duplicates.
fn parse_depths(text: &str, max_depth: usize) -> Result<Vec<usize>, String> {
    if text.trim().is_empty() {
        return Err("DT_DEPTHS must name at least one depth".to_string());
    }
    let mut depths = Vec::new();
    for part in text.split(',') {
        let depth: usize = part
            .trim()
            .parse()
            .map_err(|_| format!("DT_DEPTHS must be a comma list of numbers, got {text:?}"))?;
        if depth == 0 || depth > max_depth {
            return Err(format!(
                "DT_DEPTHS: {depth} is not a depth from 1 to {max_depth} tokens (a depth cannot \
                 exceed the ids of Q36_PROMPT_IDS or the context limit {CONTEXT_LIMIT})"
            ));
        }
        if depths.contains(&depth) {
            return Err(format!(
                "DT_DEPTHS: the depth {depth} appears twice (a duplicate would overwrite its \
                 own result row)"
            ));
        }
        depths.push(depth);
    }
    Ok(depths)
}

/// The depths of `DT_DEPTHS`, default `16,<ids of the source>` through the same validation as
/// an explicit list (the rule of `prefill/timing.rs`, mirrored), with the context limit as the
/// second upper bound.
fn depths_from_env(total_ids: usize) -> Vec<usize> {
    let max = total_ids.min(usize::try_from(CONTEXT_LIMIT).unwrap_or(usize::MAX));
    let text = match std::env::var("DT_DEPTHS") {
        Ok(text) => text,
        Err(std::env::VarError::NotPresent) => format!("16,{total_ids}"),
        Err(error) => panic!("DT_DEPTHS: {error}"),
    };
    parse_depths(&text, max).unwrap_or_else(|error| panic!("{error}"))
}

/// The ONE depth a trace prefills: the same list rule as `prefill/timing.rs`, then exactly one
/// entry — the timing default `16,<ids>` names two depths and is refused here, because the
/// trace and its summary describe one prefill.
fn depth_from_env(total_ids: usize) -> usize {
    let depths = depths_from_env(total_ids);
    if depths.len() != 1 {
        panic!("DT_DEPTHS must name exactly one depth for a router trace, got {depths:?}");
    }
    depths[0]
}

/// The router traces of the real model over the real prompt: one ignored Mac-only run through
/// the production one-token step, one JSON report, one printed line. See the module doc for the
/// environment contract, the seam, and the differences from the chunked prefill.
#[test]
#[ignore = "needs the real checkpoint on the Mac; run with --ignored"]
fn capture_router_traces() {
    let _serial = crate::native_test_lock();
    // Every input is validated first — the checkpoint, the ids source, the ids, the depth, and
    // the activation — and the output directory is created LAST, right before the load, so a
    // configuration error leaves no empty directory that blocks the rerun (the rule of
    // `decode_timing`).
    let ckpt_dir = env_path("Q36_CKPT_DIR");
    require_file_ids_source();
    let ids = read_prompt_ids();
    let depth = depth_from_env(ids.len());
    let activation = required_activation_from_env();
    let out = out_dir_must_be_new();
    let out_dir = create_out_dir(&out);

    // The load of `prefill/timing.rs`, unchanged, plus the host routing the trace needs (the
    // constructor default is `Routing::Device`).
    let real = open(&ckpt_dir);
    assert_eq!(
        sha256(&real.config),
        CONFIG_SHA256,
        "config.json differs from the pinned checkpoint"
    );
    let mut kernels = MlxKernels::try_new().unwrap();
    kernels.set_cache_limit(CACHE_LIMIT).unwrap();
    let host = HostWeights::load(&real.ckpt, &QWEN3_6_35B_A3B, &real.config).unwrap();
    let kv = kv_dtype_from_env();
    let mut decoder =
        ResidentDecoder::new_with_kv(kernels, &real.ckpt, host, CONTEXT_LIMIT, activation, kv)
            .unwrap();
    decoder.set_routing(Routing::Host);
    let (layers, experts, top_k, renorm) = {
        let m = &decoder.weights.shape.moe;
        (decoder.weights.layers.len(), m.experts, m.top_k, m.renorm)
    };
    let top_k_u32 = u32::try_from(top_k).unwrap();
    // The session cap of the shipped config, raised only when the whole depth charges more than
    // it (`chunk_growth_bytes` is the product the per-token charges of the depth sum to).
    let depth_rows = u32::try_from(depth).unwrap();
    let cap = SESSION_CAP.max(chunk_growth_bytes(&decoder.weights, depth_rows).unwrap());
    decoder.budget = SessionBudget::new(cap);
    let mut state = decoder.zero_state().unwrap();
    println!(
        "router-trace plan: activation {}, kv {}, routing Host, {layers} layers, {experts} \
         experts, top_k {top_k}, depth {depth} tokens of {} ids, session cap {cap} bytes",
        activation_name(activation),
        kv_dtype_name(kv),
        ids.len()
    );

    // One production Record step per token. `routes` holds the rank-ordered ids of every layer
    // and `probs` the probability row each selection read; the gates come from `route_experts`
    // on those rows, the function the step itself used, so the derived ids equal the reported
    // ids by construction. Indexing (not zipping) keeps a short report an honest panic instead
    // of a silently short trace.
    let prompt = &ids[..depth];
    let router = Router::SoftmaxTopK { renorm };
    let mut ids_by_layer = vec![Vec::with_capacity(depth); layers];
    let mut gates_by_layer = vec![Vec::with_capacity(depth); layers];
    let mut ids_equal_step_routes = true;
    for (token_index, &token) in prompt.iter().enumerate() {
        let StepResult { routes, probs, .. } =
            decoder.step(&mut state, token, StepMode::Record).unwrap();
        let (reported, prob_rows) = match (routes, probs) {
            (Some(reported), Some(prob_rows)) => (reported, prob_rows),
            _ => panic!("token {token_index}: a host-routed Record step reported no routes"),
        };
        for layer in 0..layers {
            let routed = route_experts(&prob_rows[layer], &router, top_k_u32).unwrap();
            ids_equal_step_routes &= routed.indices == reported[layer];
            ids_by_layer[layer].push(routed.indices);
            gates_by_layer[layer].push(routed.weights);
        }
    }
    assert_eq!(
        usize::try_from(state.pos()).unwrap(),
        depth,
        "the traced steps did not reach the depth"
    );

    // The summary: the usage histogram over every selection, the experts ever selected, and the
    // mean and maximum gate per layer (over every gate of the layer, `depth * top_k` values).
    let mut histogram = std::collections::BTreeMap::new();
    for rows in &ids_by_layer {
        for row in rows {
            for &expert in row {
                *histogram.entry(expert).or_insert(0u64) += 1;
            }
        }
    }
    let experts_seen = histogram.len();
    let histogram_json: serde_json::Map<String, serde_json::Value> = histogram
        .iter()
        .map(|(expert, count)| (expert.to_string(), serde_json::json!(count)))
        .collect();
    let per_layer_gate: Vec<serde_json::Value> = gates_by_layer
        .iter()
        .enumerate()
        .map(|(layer, rows)| {
            let (mut sum, mut max, mut count) = (0f64, f64::NEG_INFINITY, 0usize);
            for row in rows {
                for &gate in row {
                    let gate = f64::from(gate);
                    sum += gate;
                    max = max.max(gate);
                    count += 1;
                }
            }
            serde_json::json!({
                "layer": layer,
                "mean_gate": sum / count as f64,
                "max_gate": max,
            })
        })
        .collect();
    let layer_rows: Vec<serde_json::Value> = ids_by_layer
        .iter()
        .zip(&gates_by_layer)
        .enumerate()
        .map(|(layer, (ids, gates))| {
            serde_json::json!({
                "layer": layer,
                "ids": ids,
                "gates": gates,
            })
        })
        .collect();
    let mut ids_bytes = Vec::with_capacity(depth * 4);
    for &token in prompt {
        ids_bytes.extend_from_slice(&token.to_le_bytes());
    }

    let report = serde_json::json!({
        "engine": "fifi",
        "packet": "router_traces",
        "checkpoint": ckpt_dir.display().to_string(),
        "family": QWEN3_6_35B_A3B.name,
        "config_sha256": sha256(&real.config),
        "activation_dtype": activation_name(activation),
        "kv_dtype": kv_dtype_name(kv),
        "routing": "Host",
        "step_mode": "Record",
        "context_limit": CONTEXT_LIMIT,
        "session_cap_bytes": cap,
        "shipped_session_cap_bytes": SESSION_CAP,
        "ids": {
            "source": "file",
            "prompt_ids_file": env_path("Q36_PROMPT_IDS").display().to_string(),
            "ids_in_source": ids.len(),
            "depth": depth,
            "prompt_ids_sha256": sha256(&ids_bytes),
            "prompt_ids_sha256_note": "sha256 of the little-endian u32 bytes of the traced ids",
        },
        "path_note": "one production ResidentDecoder::step per token in Record mode with host \
                      routing; the chunked serving prefill batches the router matmul and skips \
                      the last layer's MoE in Prompt chunks, so a chunked run may flip near \
                      ties, and this trace is the one-token path",
        "layer_rows_note": "row t of every layer holds token t of the prompt in choice order; \
                            the per-token arrays make the file grow linearly with the depth, \
                            so the file is written compact, not pretty",
        "layer_rows": layer_rows,
        "summary": {
            "tokens": depth,
            "layers": layers,
            "top_k": top_k,
            "experts": experts,
            "expert_histogram": histogram_json,
            "experts_seen": experts_seen,
            "ids_equal_step_routes": ids_equal_step_routes,
            "per_layer_gate": per_layer_gate,
        },
    });
    std::fs::write(
        out_dir.join("router_traces.json"),
        serde_json::to_vec(&report).unwrap(),
    )
    .unwrap();
    println!("wrote {}", out_dir.join("router_traces.json").display());
    let rows = layers * depth;
    println!(
        "router_traces={{\"layers\":{layers},\"tokens\":{depth},\"top_k\":{top_k},\
         \"experts_seen\":{experts_seen},\"rows\":{rows}}}"
    );

    // The only routing assert, and only after the report is on disk (the rule of
    // `decode_timing`): every token of every layer selected exactly `top_k` experts, in ids and
    // in gates.
    for (layer, layer_ids) in ids_by_layer.iter().enumerate() {
        for (token_index, row) in layer_ids.iter().enumerate() {
            assert_eq!(
                row.len(),
                top_k,
                "layer {layer} token {token_index}: expert count"
            );
            assert_eq!(
                gates_by_layer[layer][token_index].len(),
                top_k,
                "layer {layer} token {token_index}: gate count"
            );
        }
    }
}

/// Appends one token's layer rows as little-endian F32, with hidden coordinates contiguous.
/// Only one row of encoded bytes is retained between writes.
fn write_layer_state_rows(writer: &mut impl Write, rows: &[Vec<f32>]) -> std::io::Result<()> {
    let mut bytes = Vec::new();
    for row in rows {
        bytes.clear();
        for value in row {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        writer.write_all(&bytes)?;
    }
    Ok(())
}

/// Captures residuals and routes through the production Record step, with one layer of lead
/// time available to an offline predictor. See the module doc for the environment and layout.
#[test]
#[ignore = "needs the real checkpoint on the Mac; run with --ignored"]
fn capture_layer_states() {
    let _serial = crate::native_test_lock();
    let ckpt_dir = env_path("Q36_CKPT_DIR");
    require_file_ids_source();
    let ids = read_prompt_ids();
    let depth = depth_from_env(ids.len());
    let activation = required_activation_from_env();
    let kv = kv_dtype_from_env();
    let out = out_dir_must_be_new();

    // HostWeights::load checks the config against this spec before any step. Refuse an oversized
    // capture before creating output or loading weights, including for a valid deep prompt.
    let layers = usize::try_from(QWEN3_6_35B_A3B.layers).unwrap();
    let hidden = usize::try_from(QWEN3_6_35B_A3B.hidden).unwrap();
    let projected_bytes = u64::try_from(depth)
        .unwrap()
        .checked_mul(u64::from(QWEN3_6_35B_A3B.layers))
        .and_then(|count| count.checked_mul(u64::from(QWEN3_6_35B_A3B.hidden)))
        .and_then(|count| count.checked_mul(4))
        .unwrap_or_else(|| panic!("layer_states.bin size overflows u64 at depth {depth}"));
    const MAX_BYTES: u64 = 512 * 1024 * 1024;
    if projected_bytes > MAX_BYTES {
        panic!(
            "layer_states.bin projected size {projected_bytes} bytes at depth {depth} \
             exceeds 512 MiB ({MAX_BYTES} bytes)"
        );
    }
    let out_dir = create_out_dir(&out);
    let real = open(&ckpt_dir);
    let mut kernels = MlxKernels::try_new().unwrap();
    kernels.set_cache_limit(CACHE_LIMIT).unwrap();
    let host = HostWeights::load(&real.ckpt, &QWEN3_6_35B_A3B, &real.config).unwrap();
    let mut decoder =
        ResidentDecoder::new_with_kv(kernels, &real.ckpt, host, CONTEXT_LIMIT, activation, kv)
            .unwrap();
    decoder.set_routing(Routing::Host);
    let depth_rows = u32::try_from(depth).unwrap();
    let cap = SESSION_CAP.max(chunk_growth_bytes(&decoder.weights, depth_rows).unwrap());
    decoder.budget = SessionBudget::new(cap);
    let mut state = decoder.zero_state().unwrap();

    let binary_path = out_dir.join("layer_states.bin");
    let mut writer = std::io::BufWriter::new(std::fs::File::create(&binary_path).unwrap());
    let prompt = &ids[..depth];
    let mut expert_ids = Vec::with_capacity(depth);
    for (token_index, &token) in prompt.iter().enumerate() {
        let StepResult {
            layer_out, routes, ..
        } = decoder.step(&mut state, token, StepMode::Record).unwrap();
        let rows = layer_out.expect("a Record step must report layer residuals");
        assert_eq!(
            rows.len(),
            layers,
            "token {token_index}: residual layer count"
        );
        for (layer, row) in rows.iter().enumerate() {
            assert_eq!(
                row.len(),
                hidden,
                "token {token_index} layer {layer}: residual hidden size"
            );
        }
        write_layer_state_rows(&mut writer, &rows).unwrap();
        expert_ids.push(routes.expect("a host-routed Record step must report expert ids"));
    }
    writer.flush().unwrap();
    drop(writer);
    assert_eq!(
        usize::try_from(state.pos()).unwrap(),
        depth,
        "the captured steps did not reach the depth"
    );
    let bytes = std::fs::metadata(&binary_path).unwrap().len();
    assert_eq!(
        bytes, projected_bytes,
        "layer_states.bin byte count must equal tokens * layers * hidden * 4"
    );
    let binary_sha256 = sha256(&std::fs::read(&binary_path).unwrap());
    let mut ids_bytes = Vec::new();
    for &token in prompt {
        ids_bytes.extend_from_slice(&token.to_le_bytes());
    }
    let report = serde_json::json!({
        "engine": "fifi",
        "packet": "layer_states",
        "checkpoint": ckpt_dir.display().to_string(),
        "family": QWEN3_6_35B_A3B.name,
        "config_sha256": sha256(&real.config),
        "activation_dtype": activation_name(activation),
        "kv_dtype": kv_dtype_name(kv),
        "routing": "Host",
        "step_mode": "Record",
        "context_limit": CONTEXT_LIMIT,
        "session_cap_bytes": cap,
        "tokens": depth,
        "layers": layers,
        "hidden": hidden,
        "dtype": "f32",
        "endianness": "little",
        "ordering": "[token][layer][hidden]",
        "layout_note": "row-major; zero-based token and layer indices; no header or padding",
        "binary_file": "layer_states.bin",
        "bytes": bytes,
        "sha256": binary_sha256,
        "ids": {
            "source": "file",
            "prompt_ids_file": env_path("Q36_PROMPT_IDS").display().to_string(),
            "ids_in_source": ids.len(),
            "depth": depth,
            "prompt_ids_sha256": sha256(&ids_bytes),
            "prompt_ids_sha256_note": "sha256 of the little-endian u32 bytes of the traced ids",
        },
        "expert_ids_ordering": "[token][layer][choice]",
        "expert_ids": expert_ids,
        "prediction_note": "for the same token, pair layer_out[i] with expert_ids[t][i + 1]; \
                            layer_out[i] is after layer i's MoE and before layer i + 1's mixer \
                            and post norm; the last layer has no next-layer target",
    });
    std::fs::write(
        out_dir.join("layer_states.json"),
        serde_json::to_vec(&report).unwrap(),
    )
    .unwrap();
    println!(
        "layer_states={{\"tokens\":{depth},\"layers\":{layers},\"hidden\":{hidden},\
         \"bytes\":{bytes},\"sha256\":\"{binary_sha256}\"}}"
    );
}
