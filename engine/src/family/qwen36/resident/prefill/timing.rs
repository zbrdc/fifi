//! Where the wall of one deep Q36 prefill goes, by stage (Mac, ignored, real checkpoint).
//!
//! The question: at 65K to 131K prompt depth, which stage of the prefill chunk step dominates
//! the wall? `decode_timing` splits the DECODE step with the process-wide qualification
//! counters, but a chunk step builds one lazy graph across its layers, so those counters cannot
//! attribute a chunk to its stages. This harness reaches the stages directly: it mirrors the
//! private wiring of `prefill::step::run` (as `tool_forward` mirrors the private `Q36Step`) and
//! puts a wall clock around the REAL stage calls — `attn::chunk`, `delta::chunk`, and
//! `moe::chunk`/`moe::chunk_device` — never a copy of their bodies.
//!
//! Inputs by environment, the contract of `decode_timing`:
//!
//! | Variable | Meaning | Default |
//! | --- | --- | --- |
//! | `Q36_CKPT_DIR` | the pinned checkpoint directory | required |
//! | `Q36_OUT_DIR` | the output directory; created after every input is validated, right before the load | required |
//! | `Q36_ACTIVATION` | the activation dtype: `bf16` or `f32` | required |
//! | `Q36_KV_DTYPE` | the attention cache form: `f32` or `q8` | `f32` |
//! | `DT_IDS_SOURCE` | must be `file`: this harness reads `Q36_PROMPT_IDS`, the only source deep enough | required `file` |
//! | `Q36_PROMPT_IDS` | a file of little-endian `u32` ids, the format `route_parity` reads | required |
//! | `DT_DEPTHS` | comma list of prompt depths in tokens; each depth is prefilled from token 0 | `16,<ids of the source>` |
//!
//! A depth must fit the ids source and the context limit (262,144), and the ids file must hold
//! at least as many ids as the largest depth.
//!
//! How one depth runs: a fresh state, then the prompt in worker pieces of the planned chunk (or
//! one row at a time when the plan admits no chunk), each piece split into chunk parts by
//! `kv_plan::depth_ranges` exactly as `ResidentDecoder::prefill` splits it — the depth-aware
//! halving that turns a 256-row chunk into a 128-row chunk past 65,536 positions. Every part
//! runs the timed chunk step below in `Prompt` mode, the mode of the serving prefill. The
//! harness does not reproduce the one-token pending shift of `DecodeSession` (the first serving
//! piece feeds one row less), which moves piece boundaries by one token and leaves the stage
//! shares unchanged.
//!
//! The timed chunk step mirrors `prefill::step::run` for the `Prompt` path: the same embed read,
//! the same layer loop, the same last-layer MoE skip, the same health readback and commit. The
//! mirror omits the `Record` head and the state dtype guards (host-only checks) and subsumes
//! the production cadence evaluate (`EVAL_CADENCE` 1) into the stage barriers. `embed_rows`
//! below mirrors the private `step::embed_rows`; a change there must be mirrored here.
//!
//! THE BARRIER: every stage clock closes with an `evaluate` of the stage's outputs, so the GPU
//! work of a stage completes inside the stage being timed. The barrier inflates each stage
//! slightly — it adds one stream synchronization per stage where the serving chunk makes one per
//! layer, and it also flushes the small graph built since the previous barrier (the norms and
//! the residual adds). The numbers RANK stages; they do not sum to the true total wall of the
//! serving chunk. `other` is the residual of the chunk wall minus the measured stages.
//!
//! THE CONV STAGE CANNOT BE TIMED without a production change: the causal convolution is built
//! INSIDE `delta::chunk` (`prefill/conv.rs::causal_conv` is its entry point), so its graph cannot
//! be separated from the projections and the recurrence without splitting that function or
//! duplicating its body. `conv_s` is therefore `null` in every row and line, and `delta_s`
//! INCLUDES the convolution. Splitting `delta::chunk` at the conv boundary is the smallest
//! production change that would fill the column.
//!
//! The session budget covers the deepest requested depth: the shipped cap of 4 GiB charges only
//! 104,857 positions of the F32 cache (a 131,072-position F32 run charges about 5.4 GB; a Q8 run
//! about 1.4 GB), the charge is the same function the serving path uses, and the report records
//! the cap the run used.
//!
//! Output: `prefill_stage_breakdown.json` in `Q36_OUT_DIR` — per depth the chunk count, one row
//! per chunk (start position, row count, stage seconds, wall), the stage aggregate in seconds
//! and as a share of the measured stage time, and the notes above — plus one printed line per
//! depth, `prefill_stage_breakdown={"depth":N,"chunks":N,"attention_s":x,"delta_s":x,
//! "conv_s":null,"moe_s":x,"other_s":x,"total_s":x}`. The test asserts only that each depth
//! completes (the state reaches the depth) and that the measured stages sum to at most the chunk
//! wall plus a small margin (1 percent); it asserts no speed bound.
//!
//! The exact Mac command (the ids file must hold at least 131,072 ids for this depth list):
//!
//! ```text
//! Q36_CKPT_DIR=<CKPT_DIR> \
//! Q36_OUT_DIR=<OUT_DIR> \
//! Q36_ACTIVATION=bf16 \
//! DT_IDS_SOURCE=file \
//! Q36_PROMPT_IDS=<PROMPT_IDS> \
//! DT_DEPTHS=65536,131072 \
//! cargo test --release --features native-mlx --lib \
//!   family::qwen36::resident::prefill::timing::prefill_stage_breakdown -- --ignored --exact --nocapture
//! ```

use super::super::chunk_plan::{
    ChunkCost, PREFILL_RETIRE_CAP, PrefillTerms, admitted_chunk, delta_retire_per_token,
    delta_subchunk,
};
use super::super::decode_timing::{read_prompt_ids, required_activation_from_env};
use super::super::decoder::ResidentDecoder;
use super::super::kv_plan::{SCORE_BUDGET_BYTES, depth_ranges};
use super::super::moe::Routing;
use super::super::ops::{Ctx, dim, short_import, to_act, values};
use super::super::parity::{CONFIG_SHA256, sha256};
use super::super::real::{env_path, kv_dtype_from_env, open};
use super::super::state::{LayerState, ResidentState, chunk_growth_bytes};
use super::super::step::check_dtype;
use super::super::tool_forward::{
    CACHE_LIMIT, CONTEXT_LIMIT, PREFILL_BUDGET, PREFILL_RESERVE, SESSION_CAP, create_out_dir,
    out_dir_must_be_new, serve_chunk_setting,
};
use super::super::weights::RMixer;
use super::{attn, delta, moe};
use crate::{
    EngError,
    family::{
        flash_next::stored::{self, SlicedWeight},
        qwen36::{
            host::HostShape,
            host_load::HostWeights,
            settings::{activation_name, chunked_prefill_allowed, kv_dtype_name},
        },
        route::check_health,
        spec::QWEN3_6_35B_A3B,
    },
    kernels::{BinaryOp, MlxKernels, PackedWeight},
    sched::{GrowthPermit, SessionBudget},
    tensor::ExecutionTensor,
    weights::StreamedCheckpoint,
};
use std::time::{Duration, Instant};

fn bad(message: &str) -> EngError {
    EngError::BadRequest(message.into())
}

/// The stage clocks of one chunk: the wall spent inside `attn::chunk`, `delta::chunk`, and
/// `moe::chunk`/`moe::chunk_device`, each closed by its barrier evaluate, and the wall of the
/// whole chunk (the span of `timed_chunk`: the growth permit, `begin`, the retire-cap switch,
/// the graph, the final evaluate, the health readback, and the commit).
struct ChunkTimes {
    attention_s: f64,
    delta_s: f64,
    moe_s: f64,
    wall_s: f64,
}

/// One chunk of the report: where it started, how many rows it ran, and its stage clocks.
struct ChunkRecord {
    start_pos: u32,
    rows: usize,
    times: ChunkTimes,
}

/// Requires `DT_IDS_SOURCE=file`: this harness reads `Q36_PROMPT_IDS` only, the deep ids source
/// of `decode_timing` (the 922-id tool prompt cannot reach the depth band).
fn require_file_ids_source() {
    match std::env::var("DT_IDS_SOURCE") {
        Ok(text) if text == "file" => {}
        Ok(text) => panic!("DT_IDS_SOURCE must be file for this harness, got {text:?}"),
        Err(std::env::VarError::NotPresent) => panic!("set DT_IDS_SOURCE=file"),
        Err(error) => panic!("DT_IDS_SOURCE: {error}"),
    }
}

/// Parses `DT_DEPTHS` exactly as `decode_timing::parse_depths` does (that helper is private and
/// `decode_timing.rs` is frozen for this task, so the contract is mirrored here): a comma list
/// of depths from 1 to `max_depth`, entries trimmed, no duplicates.
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

/// The depths of `DT_DEPTHS`, default `16,<ids of the source>` through the same validation as an
/// explicit list (the rule of `decode_timing`), with the context limit as the second upper
/// bound.
fn depths_from_env(total_ids: usize) -> Vec<usize> {
    let max = total_ids.min(usize::try_from(CONTEXT_LIMIT).unwrap_or(usize::MAX));
    let text = match std::env::var("DT_DEPTHS") {
        Ok(text) => text,
        Err(std::env::VarError::NotPresent) => format!("16,{total_ids}"),
        Err(error) => panic!("DT_DEPTHS: {error}"),
    };
    parse_depths(&text, max).unwrap_or_else(|error| panic!("{error}"))
}

/// The share of `stage` of the measured stage time: `stage / measured` when anything was
/// measured, zero otherwise, so the shares of a degenerate run stay finite.
fn share_of(stage: f64, measured: f64) -> f64 {
    if measured > 0.0 {
        stage / measured
    } else {
        0.0
    }
}

/// The query repeats per key-value head, the `rep` of `depth_ranges`: the mirror of the private
/// `step::repeat_count` (8 on the real model).
fn repeat_count(shape: &HostShape) -> u64 {
    let (heads_q, heads_kv) = (shape.attn.heads_q, shape.attn.heads_kv);
    assert!(
        heads_kv != 0 && heads_q % heads_kv == 0,
        "the query heads are not a multiple of the key-value heads"
    );
    u64::try_from(heads_q / heads_kv).unwrap()
}

/// The mirror of the private `step::embed_rows`: the token-start read of every row, one packed
/// import, the dequantize, the concatenate, and the cast to the activation dtype. A change of
/// the production function must be mirrored here.
fn embed_rows(
    ctx: &mut Ctx<'_>,
    ckpt: &StreamedCheckpoint,
    embed: &SlicedWeight,
    tokens: &[u32],
) -> Result<ExecutionTensor, EngError> {
    let host = tokens
        .iter()
        .map(|&token| stored::read_slice(ckpt, embed, token))
        .collect::<Result<Vec<PackedWeight>, _>>()?;
    let refs: Vec<&PackedWeight> = host.iter().collect();
    let residents = ctx.kernels.import_packed_many(&refs)?;
    if residents.len() != host.len() {
        return Err(short_import());
    }
    let rows = residents
        .iter()
        .map(|row| ctx.kernels.dequantize(row))
        .collect::<Result<Vec<_>, _>>()?;
    let parts: Vec<&ExecutionTensor> = rows.iter().collect();
    let rows = ctx.kernels.concatenate(&parts, 0)?;
    to_act(ctx, &rows)
}

/// One timed chunk: `prefill::step::chunk_with_cap` with clocks. The per-chunk check, the growth
/// permit, `begin`, the `PREFILL_RETIRE_CAP` switch, the run, and the cap restore mirror the
/// production function line for line. The whole-prompt bound of `prefill` needs no own check
/// here: the `DT_DEPTHS` parse bounds every depth by the context limit, and the per-chunk check
/// below refuses any chunk that would pass it.
fn timed_chunk(
    decoder: &mut ResidentDecoder<'_>,
    state: &mut ResidentState,
    tokens: &[u32],
) -> Result<ChunkTimes, EngError> {
    let rows = u32::try_from(tokens.len())
        .ok()
        .filter(|&rows| rows > 0)
        .ok_or_else(|| bad("a prefill needs between one and u32::MAX tokens"))?;
    state.check_chunk(&decoder.weights, decoder.context_limit, rows)?;
    let started = Instant::now();
    let growth = chunk_growth_bytes(&decoder.weights, rows)?;
    let permit = decoder.budget.acquire_growth(growth)?;
    let old = state.begin();
    match decoder.kernels.set_retire_cap(PREFILL_RETIRE_CAP) {
        Ok(previous) => {
            let stages = run_timed(decoder, state, old, permit, tokens, rows);
            let restored = decoder.kernels.set_retire_cap(previous);
            let (attention, delta, moe) = stages?;
            restored?;
            Ok(ChunkTimes {
                attention_s: attention.as_secs_f64(),
                delta_s: delta.as_secs_f64(),
                moe_s: moe.as_secs_f64(),
                wall_s: started.elapsed().as_secs_f64(),
            })
        }
        Err(error) => Err(error),
    }
}

/// The `Prompt`-mode mirror of `prefill::step::run` with a wall clock around every stage call.
/// The stage calls are the production ones; only the wiring is mirrored (see the module doc).
fn run_timed(
    decoder: &mut ResidentDecoder<'_>,
    state: &mut ResidentState,
    old: Vec<LayerState>,
    permit: GrowthPermit,
    tokens: &[u32],
    rows: u32,
) -> Result<(Duration, Duration, Duration), EngError> {
    let ckpt = decoder.ckpt;
    let weights = &decoder.weights;
    let shape = &weights.shape;
    let pos = state.pos();
    let per_token = delta_retire_per_token(
        dim(shape.delta.value_heads)?,
        dim(shape.delta.key_dim)?,
        dim(shape.delta.value_dim)?,
    )?;
    let sub = delta_subchunk(rows, PREFILL_RETIRE_CAP, per_token)?;
    let last = weights.layers.len().saturating_sub(1);
    let mut ctx = Ctx {
        kernels: &mut decoder.kernels,
        consts: &weights.consts,
        compile: false,
    };
    let mut h = embed_rows(&mut ctx, ckpt, &weights.embed, tokens)?;
    let mut layers = Vec::with_capacity(old.len());
    let mut healths: Vec<ExecutionTensor> = Vec::new();
    let (mut attention_time, mut delta_time, mut moe_time) =
        (Duration::ZERO, Duration::ZERO, Duration::ZERO);
    for (index, (layer, old_layer)) in weights.layers.iter().zip(old).enumerate() {
        let normed = ctx.kernels.rms_norm(&h, &layer.input_norm, shape.eps)?;
        check_dtype("input norm output", &normed, ctx.consts.act)?;
        let (mixed, next) = match (&layer.mixer, old_layer) {
            (RMixer::Delta(w), LayerState::Delta { rec, ring }) => {
                let stage = Instant::now();
                let s = delta::chunk(&mut ctx, shape, &w, &rec, &ring, &normed, sub)?;
                // The stage barrier: the evaluate waits for the GPU work of THIS stage inside
                // its own clock (and flushes the small graph built since the previous barrier).
                // The barrier inflates each stage slightly; the numbers rank stages rather
                // than sum to the true total.
                ctx.kernels.evaluate(&[&s.out, &s.rec, &s.ring])?;
                delta_time += stage.elapsed();
                (
                    s.out,
                    LayerState::Delta {
                        rec: s.rec,
                        ring: s.ring,
                    },
                )
            }
            (RMixer::Attn(w), LayerState::Attn(cache)) => {
                let stage = Instant::now();
                let s = attn::chunk(&mut ctx, shape, &w, cache, pos, &normed)?;
                // The stage barrier, as above: the appended cache rows are dependencies of
                // the attention output, so this evaluate completes the cache write too.
                ctx.kernels.evaluate(&[&s.out])?;
                attention_time += stage.elapsed();
                (s.out, LayerState::Attn(s.cache))
            }
            _ => return Err(bad("layer state kind does not match its weights")),
        };
        layers.push(next);
        let mid = ctx.kernels.binary(BinaryOp::Add, &h, &mixed)?;
        if index == last {
            // The `Prompt` rule of `run`: the last layer's MoE feeds only the head, which
            // `Prompt` skips, so the chunk stops at its `mid`.
            check_dtype("last layer mid", &mid, ctx.consts.act)?;
            h = mid;
            break;
        }
        let normed = ctx.kernels.rms_norm(&mid, &layer.post_norm, shape.eps)?;
        check_dtype("post norm output", &normed, ctx.consts.act)?;
        let routed = match decoder.routing {
            Routing::Host => moe::chunk(&mut ctx, shape, &layer.moe, &normed)?.0,
            Routing::Device => {
                let stage = Instant::now();
                let device = moe::chunk_device(&mut ctx, shape, &layer.moe, &normed)?;
                // The stage barrier, as above; the health node stays lazy for the chunk tail.
                ctx.kernels.evaluate(&[&device.out])?;
                moe_time += stage.elapsed();
                healths.push(device.health);
                device.out
            }
        };
        h = ctx.kernels.binary(BinaryOp::Add, &mid, &routed)?;
        check_dtype("layer output", &h, ctx.consts.act)?;
    }
    commit_timed(&mut ctx, state, layers, healths, rows, permit)?;
    Ok((attention_time, delta_time, moe_time))
}

/// The `Prompt` tail of `run`: the health stack, the final evaluate of the new state, the one
/// health readback, and the commit. The per-stage barriers have already run the state tensors,
/// so this evaluate is nearly free and lands in `other`.
fn commit_timed(
    ctx: &mut Ctx<'_>,
    state: &mut ResidentState,
    layers: Vec<LayerState>,
    healths: Vec<ExecutionTensor>,
    rows: u32,
    permit: GrowthPermit,
) -> Result<(), EngError> {
    let health_stack = if healths.is_empty() {
        None
    } else {
        let parts: Vec<&ExecutionTensor> = healths.iter().collect();
        Some(ctx.kernels.concatenate(&parts, 0)?)
    };
    let mut outputs: Vec<&ExecutionTensor> = Vec::new();
    for layer in &layers {
        match layer {
            LayerState::Delta { rec, ring } => outputs.extend([rec, ring]),
            LayerState::Attn(cache) => outputs.extend(cache.tensors()),
        }
    }
    if let Some(health) = &health_stack {
        outputs.push(health);
    }
    ctx.kernels.evaluate(&outputs)?;
    if let Some(health) = &health_stack {
        check_health(&values(ctx.kernels, health)?)?;
    }
    state.commit_rows(rows, layers, permit)?;
    Ok(())
}

/// The stage breakdown of the deep prefill: one ignored Mac-only run over the real checkpoint,
/// one row per chunk, one aggregate and one printed line per depth, one JSON report. See the
/// module doc for the environment contract, the barrier caveat, and the conv limitation.
#[test]
#[ignore = "needs the real checkpoint on the Mac; run with --ignored"]
fn prefill_stage_breakdown() {
    let _serial = crate::native_test_lock();
    // Every input is validated first — the checkpoint, the ids source, the ids, the depths, and
    // the activation — and the output directory is created LAST, right before the load, so a
    // configuration error leaves no empty directory that blocks the rerun (the rule of
    // `decode_timing`).
    let ckpt_dir = env_path("Q36_CKPT_DIR");
    require_file_ids_source();
    let ids = read_prompt_ids();
    let depths = depths_from_env(ids.len());
    let activation = required_activation_from_env();
    let out = out_dir_must_be_new();
    let out_dir = create_out_dir(&out);

    // The load of `decode_timing`, unchanged: the pinned checkpoint, the shipped cache limit,
    // and the product constructor of the `Q36_ACTIVATION` and `Q36_KV_DTYPE` levers.
    let real = open(&ckpt_dir);
    assert_eq!(
        sha256(&real.config),
        CONFIG_SHA256,
        "config.json differs from the pinned checkpoint"
    );
    let mut kernels = MlxKernels::try_new().unwrap();
    kernels.set_cache_limit(CACHE_LIMIT).unwrap();
    let host = HostWeights::load(&real.ckpt, &QWEN3_6_35B_A3B, &real.config).unwrap();
    let heads_q = u64::try_from(host.shape().attn.heads_q).unwrap();
    let kv = kv_dtype_from_env();
    let mut decoder =
        ResidentDecoder::new_with_kv(kernels, &real.ckpt, host, CONTEXT_LIMIT, activation, kv)
            .unwrap();
    // The session cap of the shipped config, raised only when the deepest requested depth
    // charges more than it (the charge is `chunk_growth_bytes`, the function the serving path
    // uses); the report records both caps.
    let deepest = u32::try_from(*depths.iter().max().unwrap()).unwrap();
    let cap = SESSION_CAP.max(chunk_growth_bytes(&decoder.weights, deepest).unwrap());
    decoder.budget = SessionBudget::new(cap);
    let mut state = decoder.zero_state().unwrap();

    // The serving-chunk admission of `decode_timing`, unchanged: the planned chunk of the
    // shipped config, or no chunk when the plan admits none.
    let terms = PrefillTerms {
        budget: PREFILL_BUDGET,
        weights: decoder.weights.byte_len().unwrap(),
        session_reserve: PREFILL_RESERVE,
    };
    let cost = ChunkCost::new(heads_q, CONTEXT_LIMIT).unwrap();
    let serve_setting = serve_chunk_setting(activation);
    let chunked = chunked_prefill_allowed(decoder.weights.consts.act);
    let chunk = if chunked {
        admitted_chunk(&cost, terms.remaining(), serve_setting).unwrap()
    } else {
        None
    };
    if let Some(rows) = chunk {
        decoder.set_chunk_rows(rows).unwrap();
    }
    // Without an admitted chunk the serving path feeds one token per `decoder.step`, which has
    // no stage boundaries to time, so this harness feeds one row at a time through the chunk
    // path instead (a chunk of one row runs the same blocks).
    let piece = chunk
        .map(|rows| usize::try_from(rows).unwrap())
        .unwrap_or(1);
    let rep = repeat_count(&decoder.weights.shape);
    println!(
        "prefill-stage plan: activation {}, kv {}, chunked prefill {chunked}, serving chunk \
         {chunk:?}, piece {piece}, session cap {cap} bytes, depths {depths:?}; {}",
        activation_name(activation),
        kv_dtype_name(kv),
        terms.to_string().lines().collect::<Vec<_>>().join(" ")
    );

    let mut depth_rows = Vec::new();
    let mut sums = Vec::new();
    for &depth in &depths {
        decoder.reset(&mut state).unwrap();
        let prompt = &ids[..depth];
        let mut records: Vec<ChunkRecord> = Vec::new();
        for fed in prompt.chunks(piece) {
            let rows = u32::try_from(fed.len()).unwrap();
            let parts = depth_ranges(
                state.pos(),
                rows,
                decoder.chunk_rows,
                rep,
                SCORE_BUDGET_BYTES,
            )
            .unwrap();
            for &(start, stop) in &parts {
                let tokens = &fed[usize::try_from(start).unwrap()..usize::try_from(stop).unwrap()];
                let start_pos = state.pos();
                let times = timed_chunk(&mut decoder, &mut state, tokens).unwrap();
                records.push(ChunkRecord {
                    start_pos,
                    rows: tokens.len(),
                    times,
                });
            }
        }
        assert_eq!(
            usize::try_from(state.pos()).unwrap(),
            depth,
            "depth {depth}: the timed chunks did not reach the depth"
        );
        let (mut attention_s, mut delta_s, mut moe_s, mut wall_s) = (0.0, 0.0, 0.0, 0.0);
        for record in &records {
            attention_s += record.times.attention_s;
            delta_s += record.times.delta_s;
            moe_s += record.times.moe_s;
            wall_s += record.times.wall_s;
        }
        let other_s = (wall_s - attention_s - delta_s - moe_s).max(0.0);
        let measured_s = attention_s + delta_s + moe_s + other_s;
        let chunk_rows: Vec<serde_json::Value> = records
            .iter()
            .enumerate()
            .map(|(index, record)| {
                let t = &record.times;
                serde_json::json!({
                    "chunk": index,
                    "start_pos": record.start_pos,
                    "rows": record.rows,
                    "attention_s": t.attention_s,
                    "delta_s": t.delta_s,
                    "conv_s": null,
                    "moe_s": t.moe_s,
                    "other_s": (t.wall_s - t.attention_s - t.delta_s - t.moe_s).max(0.0),
                    "wall_s": t.wall_s,
                })
            })
            .collect();
        println!(
            "prefill_stage_breakdown={}",
            serde_json::json!({
                "depth": depth,
                "chunks": records.len(),
                "attention_s": attention_s,
                "delta_s": delta_s,
                "conv_s": null,
                "moe_s": moe_s,
                "other_s": other_s,
                "total_s": wall_s,
            })
        );
        sums.push((depth, attention_s + delta_s + moe_s, wall_s));
        depth_rows.push(serde_json::json!({
            "depth": depth,
            "chunks": records.len(),
            "rows_fed": depth,
            "stages_s": {
                "attention_s": attention_s,
                "delta_s": delta_s,
                "conv_s": null,
                "moe_s": moe_s,
                "other_s": other_s,
                "measured_stage_s": measured_s,
                "chunk_wall_s": wall_s,
            },
            "share_of_measured_stage_time": {
                "attention": share_of(attention_s, measured_s),
                "delta": share_of(delta_s, measured_s),
                "moe": share_of(moe_s, measured_s),
                "other": share_of(other_s, measured_s),
            },
            "chunk_rows": chunk_rows,
        }));
    }

    let report = serde_json::json!({
        "engine": "fifi",
        "packet": "prefill_stage_breakdown",
        "checkpoint": ckpt_dir.display().to_string(),
        "config_sha256": sha256(&real.config),
        "activation_dtype": activation_name(activation),
        "kv_dtype": kv_dtype_name(kv),
        "context_limit": CONTEXT_LIMIT,
        "ids": {
            "source": "file",
            "prompt_ids_file": env_path("Q36_PROMPT_IDS").display().to_string(),
            "ids_in_source": ids.len(),
        },
        "depths": depths,
        "serving_chunk_requested": serve_setting,
        "serving_chunk": chunk,
        "chunk_piece": piece,
        "session_cap_bytes": cap,
        "shipped_session_cap_bytes": SESSION_CAP,
        "routing": format!("{:?}", decoder.routing),
        "barrier_note": "every stage clock closes with an evaluate of the stage's outputs, so \
                         the barrier inflates each stage slightly (one extra stream \
                         synchronization per stage and the flush of the small graph built since \
                         the previous barrier); the numbers rank stages rather than sum to the \
                         true total wall of the serving chunk",
        "conv_note": "conv_s is null: the causal convolution is built inside delta::chunk, so \
                      it cannot be timed separately without splitting that production function \
                      or duplicating its body; delta_s includes the convolution",
        "depth_rows": depth_rows,
    });
    std::fs::write(
        out_dir.join("prefill_stage_breakdown.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!(
        "wrote {}",
        out_dir.join("prefill_stage_breakdown.json").display()
    );

    // The only timing assert, and only after the report is on disk (the rule of
    // `decode_timing`): the stage clocks are nested inside the chunk clock, so only timer
    // overhead and scheduling jitter can push their sum past the wall. The margin is small and
    // relative (1 percent, at least 1 ms); no speed bound is asserted anywhere.
    for (depth, stages, wall) in sums {
        let margin = (wall * 0.01).max(0.001);
        assert!(
            stages <= wall + margin,
            "depth {depth}: the measured stages sum to {stages:.3} s, above the chunk wall \
             {wall:.3} s plus the margin {margin:.3} s"
        );
    }
}
