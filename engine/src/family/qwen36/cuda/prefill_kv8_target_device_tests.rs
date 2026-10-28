//! Private CPF target row: one m16 body versus sixteen production Int8G32 steps.
//!
//! This child of `step.rs` uses private fields and the existing chunk protocol. It does not
//! open the public KV8 gate or change the Q6_K guard. The comparison is KV8 route equivalence,
//! not FP16 G-KV8 acceptance, a full-body E3 bound, or a 256-row qualification.

use super::super::{
    book::Health,
    common_tests::{dims as step_dims, inventory},
    prefill::{ChunkKernelClass, ChunkViews},
    prefill_driver::admit_m_sess,
    prefill_plan::ChunkSlot,
    run_report::CaptureOutcome,
    state::export::f32_from_le,
};
use super::*;
use crate::family::gguf_model::open_qwen3_6_35b_a3b;
use crate::family::project::cuda::{CudaProjector, LoadOptions, Reserve};
use crate::kernels::cuda::prefill::attn_prefill::CPF_ATTN_PART_CAP;
use sha2::{Digest, Sha256};

type RawRows = Vec<Vec<u8>>;

const ROWS: usize = 16;
const ARENA_ROWS: usize = 128;
const SAMPLER_VOCAB: usize = 248_077;
const HEAD_ROWS: usize = 248_320;
const CONTEXT_TOKENS: usize = 1024;
const PROMPT_SHA256: &str = "8589556ee1e4dff410f4a3d16b0a266be3324fa9b8603714bf62cdec1be8f7f2";
// Historical report-only statistics; the 16-row route-to-route gate was retired by Dan.
const ROUTE_MEAN_NLL_LIMIT: f64 = 0.02;
const ROUTE_MARGIN: f64 = 0.511;

fn bad(message: &str) -> EngError {
    EngError::BadRequest(format!("private CPF KV8 target: {message}"))
}

/// The parent binds the immutable canonical P5 file at this existing fixture location.
fn canonical_ids() -> Result<Vec<u32>, EngError> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../qualification/cuda-rd/oracle/prompts/P5.ids.json");
    let bytes =
        std::fs::read(&path).map_err(|error| bad(&format!("{}: {error}", path.display())))?;
    if format!("{:x}", Sha256::digest(&bytes)) != PROMPT_SHA256 {
        return Err(bad("the canonical P5 file hash differs"));
    }
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|error| bad(&error.to_string()))?;
    if value["name"].as_str() != Some("P5") {
        return Err(bad("the canonical prompt is not P5"));
    }
    let ids = value["ids"]
        .as_array()
        .ok_or_else(|| bad("P5 has no ids"))?;
    if ids.len() <= ROWS {
        return Err(bad(
            "P5 needs sixteen inputs and the seventeenth teacher id",
        ));
    }
    ids.iter()
        .take(ROWS + 1)
        .map(|id| {
            id.as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .filter(|&id| (id as usize) < SAMPLER_VOCAB)
                .ok_or_else(|| bad("a P5 id is outside the sampler vocabulary"))
        })
        .collect()
}

fn options() -> StepOptions {
    StepOptions {
        kv_mode: KvMode::Int8G32,
        context_tokens: CONTEXT_TOKENS,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Native,
    }
}

fn rules() -> Result<QwenRules, EngError> {
    QwenRules::new(vec![248_046, 248_044], SAMPLER_VOCAB, HEAD_ROWS)
}

#[derive(Debug, PartialEq, Eq)]
struct Boundary {
    tokens: usize,
    health: Health,
}

fn boundary(step: &QwenResidentStep<'_>) -> Boundary {
    Boundary {
        tokens: step.tokens(),
        health: step.state.health(),
    }
}

struct ProductionRows {
    logits: Vec<Vec<u8>>,
    chosen: Vec<u32>,
    committed: Vec<usize>,
    capture: Vec<CaptureOutcome>,
    end: Boundary,
}

struct Observed {
    kernel_class: ChunkKernelClass,
    model_path: std::path::PathBuf,
    production: ProductionRows,
    hidden: Vec<Vec<u8>>,
    logits: Vec<Vec<u8>>,
    before: Boundary,
    after_chunk: Boundary,
    after_tail: Boundary,
    records: Vec<StepRecord>,
    n_prefilled: usize,
    n_fed: usize,
    chunk_q6k: bool,
    layers: usize,
    delta_layers: usize,
    attn_layers: usize,
    hidden_width: usize,
    arena_rows: usize,
    arena_bytes: usize,
    part_bytes: usize,
    budget: usize,
    length_word: Vec<u8>,
}

fn read_range(stream: &DeviceStream, range: DeviceRange<'_>) -> Result<Vec<u8>, EngError> {
    let mut bytes = vec![0u8; range.bytes()];
    range.buffer().read_at(stream, range.offset(), &mut bytes)?;
    Ok(bytes)
}

/// Use the normal admission equality and budget sum, but do not change `needs_arena` or its gate.
fn private_arena(step: &mut QwenResidentStep<'_>, budget: usize) -> Result<(), EngError> {
    admit_guard(step.started, step.prefill.is_some())?;
    if step.chunk_q6k {
        return Err(bad(
            "the target unexpectedly has a Q6_K chunk weight; retain its refusal",
        ));
    }
    let layout = step.state.layout();
    let admission = AdmitFacts {
        budget,
        decode_arena: step.res.arena.bytes(),
        state_planned: state_bytes(
            &step.res.dims,
            layout.kv.mode,
            layout.kv.capacity,
            layout.delta.slots,
        )?,
        state_allocated: layout.bytes()?,
        cap: ARENA_ROWS,
    };
    let facts = step.context.mmq_device_facts()?;
    let costs = chunk_costs(&step.res.dims, facts.nsm)?;
    let admitted = admit_m_sess(&admission, &[], &costs)?;
    if admitted != Some(ARENA_ROWS) {
        return Err(EngError::MemoryPlan(format!(
            "private CPF KV8 target: the 128-row arena was not admitted: {admitted:?}"
        )));
    }
    step.prefill = Some(PrefillArena::allocate(
        step.context,
        &step.stream,
        &step.res.dims,
        ARENA_ROWS,
        facts,
        budget,
    )?);
    step.refill_prefill_report();
    Ok(())
}

/// Read all final hidden rows, then use exactly the production final norm and head on each.
/// Only decode scratch changes: no token body, argmax, advance, or session publication runs.
fn all_row_tails(step: &QwenResidentStep<'_>) -> Result<(RawRows, RawRows), EngError> {
    let arena = step
        .prefill
        .as_ref()
        .ok_or_else(|| bad("no private arena"))?;
    let views = ChunkViews::new(arena.list(), arena.arena())?;
    let hidden = views.get(ChunkSlot::Hidden)?;
    let row_bytes = step
        .res
        .geometry
        .hidden
        .checked_mul(F32)
        .ok_or_else(|| bad("hidden row size overflows"))?;
    let total = row_bytes
        .checked_mul(ROWS)
        .ok_or_else(|| bad("hidden rows overflow"))?;
    if total > hidden.bytes() {
        return Err(bad("the hidden slot does not hold sixteen rows"));
    }
    let bytes = read_range(&step.stream, hidden.buffer().range(hidden.offset(), total)?)?;
    let rows: Vec<Vec<u8>> = bytes.chunks_exact(row_bytes).map(<[u8]>::to_vec).collect();
    let scratch = step.view(final_resid(step.res.table.len()))?;
    if scratch.bytes() != row_bytes {
        return Err(bad("the production tail scratch is not one hidden row"));
    }
    let mut logits = Vec::with_capacity(ROWS);
    for row in &rows {
        scratch
            .buffer()
            .write_at(&step.stream, scratch.offset(), row)?;
        step.final_norm()?;
        step.lm_head()?;
        logits.push(read_range(&step.stream, step.view(Slot::Logits)?)?);
    }
    Ok((rows, logits))
}

/// Plain Result-only device phase. Owners stay scoped and drop before any host assertion.
fn device_phase(ids: &[u32], kernel_class: ChunkKernelClass) -> Result<Observed, EngError> {
    // These deterministic fixture helpers contain expects; call them before acquiring CUDA owners.
    let dims = step_dims();
    let inv = inventory();
    let opts = options();
    let dir = std::env::var("Q36_CUDA_MODEL_DIR")
        .unwrap_or_else(|_| "/media/dan/nvme/models/qwen3.6-35b-a3b".to_owned());
    let gguf_path = std::path::Path::new(&dir).join("Qwen3.6-35B-A3B-Q4_K_M.gguf");
    let model = open_qwen3_6_35b_a3b(&gguf_path, None)?;
    let projector = CudaProjector::new()?;
    let weights = projector
        .load_model(
            &model,
            LoadOptions {
                reserve: Reserve::default(),
                drop_page_cache: None,
                layout: WeightLayout::Native,
            },
        )?
        .device;
    // Same snapshot point as production: after fullweights, before decode arena and state.
    let free = projector.memory_info()?.free;
    let budget = free.checked_sub(1usize << 30).ok_or_else(|| {
        EngError::MemoryPlan("private CPF KV8 target: no 1 GiB headroom".to_owned())
    })?;
    let context = projector.context();
    let res = StepResources::new(context, dims, &inv, &opts)?;
    let production = {
        let mut step = QwenResidentStep::new(context, &res, &weights, &opts, rules()?)?;
        step.set_capture_config(CaptureConfig {
            capture: false,
            section_marks: false,
        })?;
        let mut logits = Vec::with_capacity(ROWS);
        let mut chosen = Vec::with_capacity(ROWS);
        let mut committed = Vec::with_capacity(ROWS);
        let mut capture = Vec::with_capacity(ROWS);
        for &id in &ids[..ROWS] {
            chosen.push(step.run_token(id, true)?);
            logits.push(step.row_bytes().to_vec());
            committed.push(step.tokens());
            capture.push(step.last_report().capture);
        }
        ProductionRows {
            logits,
            chosen,
            committed,
            capture,
            end: boundary(&step),
        }
    };
    // The reference state has dropped. The two steps share only immutable weights and reusable
    // decode scratch, never mutable session state, and never execute concurrently.
    let mut step = QwenResidentStep::new(context, &res, &weights, &opts, rules()?)?;
    step.set_capture_config(CaptureConfig {
        capture: false,
        section_marks: false,
    })?;
    step.chunk_kernel_class = kernel_class;
    step.set_m_sess_cap(ARENA_ROWS as u32);
    private_arena(&mut step, budget)?;
    let before = boundary(&step);
    let arena = step
        .prefill
        .as_ref()
        .ok_or_else(|| bad("no admitted arena"))?;
    let p0 = step.tokens();
    let length = p0
        .checked_add(1)
        .and_then(|value| i32::try_from(value).ok())
        .ok_or_else(|| bad("p0 + 1 does not fit i32"))?;
    let length_offset = CPF_ATTN_PART_CAP
        .checked_sub(size_of::<i32>())
        .ok_or_else(|| bad("the part tail has no length word"))?;
    let length_range = arena.part().range(length_offset, size_of::<i32>())?;
    length_range
        .buffer()
        .write_at(&step.stream, length_range.offset(), &length.to_le_bytes())
        .inspect_err(|_| step.state.poison())?;
    // This existing binding opens the book, clears status, uploads ids, invokes exactly ONE
    // all-layer run_chunk_body, reads status, records the step, and revalidates the state.
    step.run_chunk(p0, &ids[..ROWS])?;
    let after_chunk = boundary(&step);
    let (hidden, logits) = all_row_tails(&step).inspect_err(|_| step.state.poison())?;
    let arena = step
        .prefill
        .as_ref()
        .ok_or_else(|| bad("the private arena disappeared"))?;
    let length_word = read_range(
        &step.stream,
        arena.part().range(length_offset, size_of::<i32>())?,
    )
    .inspect_err(|_| step.state.poison())?;
    let observed = Observed {
        kernel_class,
        model_path: gguf_path,
        production,
        hidden,
        logits,
        before,
        after_chunk,
        after_tail: boundary(&step),
        records: step.prefill_report.steps().to_vec(),
        n_prefilled: step.prefill_report.n_prefilled(),
        n_fed: step.prefill_report.n_fed(),
        chunk_q6k: step.chunk_q6k,
        layers: res.table.len(),
        delta_layers: step.state.layout().delta.layers,
        attn_layers: step.state.layout().kv.layers,
        hidden_width: res.geometry.hidden,
        arena_rows: arena.m_sess(),
        arena_bytes: arena.bytes(),
        part_bytes: arena.part_bytes(),
        budget,
        length_word,
    };
    // Surface completion errors as Result; preserve earlier failures through ordinary `?` returns.
    step.stream
        .synchronize()
        .inspect_err(|_| step.state.poison())?;
    Ok(observed)
}

#[derive(Debug)]
struct Score {
    top: usize,
    margin: f64,
    nll: f64,
}

/// Host-only, sampler-domain log-sum-exp; ties select the lowest token id.
/// The entire head must be finite, but padding never participates in selection or NLL.
fn score(head: &[f32], sampler_width: usize, teacher: usize) -> Result<Score, &'static str> {
    if sampler_width < 2 || sampler_width > head.len() {
        return Err("the sampler domain does not fit the head shape");
    }
    if teacher >= sampler_width {
        return Err("the teacher is outside the sampler domain");
    }
    if !head.iter().all(|value| value.is_finite()) {
        return Err("the head contains a non-finite value");
    }
    let row = &head[..sampler_width];
    let mut top = 0;
    let mut second = f64::NEG_INFINITY;
    let mut best = f64::from(row[0]);
    for (id, &value) in row.iter().enumerate().skip(1) {
        let value = f64::from(value);
        if value > best {
            second = best;
            best = value;
            top = id;
        } else if value > second {
            second = value;
        }
    }
    let sum = row
        .iter()
        .map(|&v| (f64::from(v) - best).exp())
        .sum::<f64>();
    Ok(Score {
        top,
        margin: best - second,
        nll: sum.ln() - (f64::from(row[teacher]) - best),
    })
}

#[test]
fn kv8_target_nll_has_a_deterministic_host_check() {
    let uniform = score(&[0.0, 0.0, 0.0, 0.0], 4, 2).expect("uniform");
    assert_eq!(uniform.top, 0);
    assert_eq!(uniform.margin, 0.0);
    assert!((uniform.nll - 4.0f64.ln()).abs() < 1e-14);
    // exp(0) : exp(ln(3)) is 1 : 3, so the first token's NLL is ln(4).
    let row = [0.0, 3.0f32.ln()];
    let weighted = score(&row, 2, 0).expect("weighted");
    assert_eq!(weighted.top, 1);
    assert!((weighted.margin - 3.0f64.ln()).abs() < 1e-7);
    assert!((weighted.nll - 4.0f64.ln()).abs() < 1e-7);
    assert!((score(&[1000.0, 1000.0], 2, 1).expect("shifted").nll - 2.0f64.ln()).abs() < 1e-14);
    assert_eq!(score(&[2.0, 1.0, 2.0], 3, 1).expect("tie").top, 0);
    let zero = score(&[-0.0, 0.0], 2, 1).expect("signed zero");
    assert_eq!(zero.top, 0);
    assert_eq!(zero.margin, 0.0);
    assert!((zero.nll - 2.0f64.ln()).abs() < 1e-14);
    let padded = score(&[0.0, 1.0, 1000.0], 2, 0).expect("padding");
    let unpadded = score(&[0.0, 1.0], 2, 0).expect("no padding");
    assert_eq!(padded.top, unpadded.top);
    assert_eq!(padded.nll, unpadded.nll);
    assert!(score(&[], 0, 0).is_err());
    assert!(score(&[0.0], 2, 0).is_err());
    assert!(score(&[0.0, 1.0], 3, 0).is_err());
    assert!(score(&[0.0, 1.0, 1000.0], 2, 2).is_err());
    for non_finite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(score(&[non_finite, 0.0], 2, 0).is_err());
        assert!(score(&[0.0, 1.0, non_finite], 2, 0).is_err());
    }
}

fn write_json(path: &std::path::Path, value: &serde_json::Value) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    let bytes = serde_json::to_vec_pretty(value).map_err(std::io::Error::other)?;
    file.write_all(&bytes)?;
    file.sync_all()
}

/// Persist raw evidence after all CUDA owners drop, before numerical acceptance assertions.
/// The parent maps this process's temporary directory to its reviewed NVMe scratch.
fn save_rows(
    observed: &Observed,
    ids: &[u32],
    artifact_name: &str,
) -> std::io::Result<std::path::PathBuf> {
    let dir = std::env::temp_dir().join(artifact_name);
    std::fs::create_dir(&dir)?;
    for (name, rows) in [
        ("production-logits.f32le", &observed.production.logits),
        ("chunk-logits.f32le", &observed.logits),
        ("chunk-hidden.f32le", &observed.hidden),
    ] {
        use std::io::Write;
        let path = std::path::Path::new(&dir).join(name);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        for row in rows {
            file.write_all(row)?;
        }
        file.sync_all()?;
        println!(
            "private CPF KV8 target raw {} rows={} row_bytes={:?}",
            path.display(),
            rows.len(),
            rows.first().map(Vec::len)
        );
    }
    let mut manifest = serde_json::json!({
        "schema": "cpf-kv8-target16/2", "model_path": observed.model_path,
        "reference": { "route": "production_run_token", "want_logits": true, "kv_mode": "Int8G32" },
        "candidate": { "route": "private_chunk", "kv_mode": "Int8G32", "tail": "production_final_norm_lm_head_per_row" },
        "scope": "KV8 route equivalence, NOT FP16 G-KV8, full-body E3, or full256 acceptance",
        "capture": false, "tiered": false, "layout": "Native",
        "prompt": "P5", "prompt_sha256": PROMPT_SHA256, "canonical_first17_ids": ids,
        "rows": ROWS, "p0": 0, "layers": observed.layers,
        "head_width": HEAD_ROWS, "sampler_width": SAMPLER_VOCAB, "hidden_width": observed.hidden_width,
        "encoding": "row-major little-endian f32 without headers",
        "production_row_bytes": observed.production.logits.iter().map(Vec::len).collect::<Vec<_>>(),
        "chunk_row_bytes": observed.logits.iter().map(Vec::len).collect::<Vec<_>>(),
        "hidden_row_bytes": observed.hidden.iter().map(Vec::len).collect::<Vec<_>>(),
        "production_argmax": observed.production.chosen,
        "production_committed_counts": observed.production.committed,
        "chunk_tokens": observed.after_chunk.tokens, "tail_tokens": observed.after_tail.tokens,
        "chunk_health": format!("{:?}", observed.after_chunk.health),
        "tail_health": format!("{:?}", observed.after_tail.health),
        "prefilled": observed.n_prefilled, "fed": observed.n_fed,
        "chunk_records": observed.records.iter().map(|record| serde_json::json!({
            "p0": record.p0, "rows": record.rows, "status": record.status,
            "kind": format!("{:?}", record.kind)
        })).collect::<Vec<_>>(),
        "delta_layers": observed.delta_layers, "attention_layers": observed.attn_layers,
        "chunk_q6k": observed.chunk_q6k,
        "arena_rows": observed.arena_rows, "arena_bytes": observed.arena_bytes,
        "part_bytes": observed.part_bytes, "admission_budget": observed.budget,
        "kv8_length_word_bytes": observed.length_word,
        "mean_signed_delta_nll_limit": ROUTE_MEAN_NLL_LIMIT,
        "kv8_reference_top1_margin_threshold": ROUTE_MARGIN
    });
    manifest["retired_route_gate_is_report_only"] = true.into();
    manifest["chunk_kernel_class"] = format!("{:?}", observed.kernel_class).into();
    manifest["diagnostic_confounds"] = "chunk normalization, routing, swiglu, scan and attention are unchanged; projection-class control is not model acceptance".into();
    write_json(&dir.join("manifest.json"), &manifest)?;
    Ok(dir)
}

fn inspect_rows(
    observed: &Observed,
    ids: &[u32],
    artifact_dir: &std::path::Path,
) -> (f64, Vec<usize>) {
    assert_eq!(
        observed.before,
        Boundary {
            tokens: 0,
            health: Health::Valid
        }
    );
    assert_eq!(
        observed.after_chunk,
        Boundary {
            tokens: ROWS,
            health: Health::Valid
        }
    );
    assert_eq!(
        observed.after_tail, observed.after_chunk,
        "stateless tails must not publish rows"
    );
    assert_eq!(observed.production.end, observed.after_chunk);
    assert_eq!(
        observed.production.committed,
        (1..=ROWS).collect::<Vec<_>>()
    );
    assert_eq!(
        observed.production.capture,
        vec![CaptureOutcome::Eager; ROWS]
    );
    assert!(
        !observed.chunk_q6k,
        "the actual target must retain its independent Q6_K scan"
    );
    assert_eq!(
        (observed.layers, observed.delta_layers, observed.attn_layers),
        (40, 30, 10)
    );
    assert_eq!(observed.arena_rows, ARENA_ROWS);
    assert!(observed.arena_bytes > 0);
    assert_eq!(observed.part_bytes, CPF_ATTN_PART_CAP);
    assert_eq!(observed.length_word, 1i32.to_le_bytes());
    assert_eq!((observed.n_prefilled, observed.n_fed), (ROWS, 0));
    assert_eq!(observed.records.len(), 1);
    let record = observed.records[0];
    assert_eq!(
        (record.p0, record.rows, record.kind, record.status),
        (0, ROWS, StepKind::Chunk, 0)
    );
    assert_eq!(observed.production.logits.len(), ROWS);
    assert_eq!(observed.production.chosen.len(), ROWS);
    assert_eq!(observed.logits.len(), ROWS);
    assert_eq!(observed.hidden.len(), ROWS);
    println!(
        "private CPF KV8 target reference=production_run_token_true kv=Int8G32 layout=Native \
         capture=off tiered=off prompt=P5 sha256={PROMPT_SHA256} rows={ROWS} p0=0 layers={} \
         arena_rows={} arena_bytes={} part_bytes={} budget={} state={:?} prefilled={} fed={} \
         status={} length_word=1; KV8 route equivalence only; no FP16 G-KV8/full-body E3/full256 claim",
        observed.layers,
        observed.arena_rows,
        observed.arena_bytes,
        observed.part_bytes,
        observed.budget,
        observed.after_chunk,
        observed.n_prefilled,
        observed.n_fed,
        record.status,
    );
    let mut agree = 0;
    let mut exact_rows = 0;
    let mut margin_failures = Vec::new();
    let mut row_reports = Vec::with_capacity(ROWS);
    let mut sum_delta_nll = 0.0;
    let mut sum_abs_delta_nll = 0.0;
    let mut max_abs_delta_nll = 0.0f64;
    for (at, &teacher) in ids.iter().skip(1).take(ROWS).enumerate() {
        let raw_reference = &observed.production.logits[at];
        let raw_candidate = &observed.logits[at];
        let raw_hidden = &observed.hidden[at];
        assert_eq!(raw_reference.len(), HEAD_ROWS * F32);
        assert_eq!(raw_candidate.len(), HEAD_ROWS * F32);
        assert_eq!(raw_hidden.len(), observed.hidden_width * F32);
        let reference = f32_from_le(raw_reference);
        let candidate = f32_from_le(raw_candidate);
        let hidden = f32_from_le(raw_hidden);
        let finite_reference = reference.iter().filter(|v| v.is_finite()).count();
        let finite_candidate = candidate.iter().filter(|v| v.is_finite()).count();
        let finite_hidden = hidden.iter().filter(|v| v.is_finite()).count();
        assert_eq!(finite_reference, HEAD_ROWS, "production head row {at}");
        assert_eq!(finite_candidate, HEAD_ROWS, "chunk head row {at}");
        assert_eq!(
            finite_hidden, observed.hidden_width,
            "chunk hidden row {at}"
        );
        let reference_score =
            score(&reference, SAMPLER_VOCAB, teacher as usize).expect("production score");
        let candidate_score =
            score(&candidate, SAMPLER_VOCAB, teacher as usize).expect("chunk score");
        assert_eq!(reference_score.top, observed.production.chosen[at] as usize);
        let delta = candidate_score.nll - reference_score.nll;
        assert!(
            reference_score.nll.is_finite() && candidate_score.nll.is_finite() && delta.is_finite()
        );
        sum_delta_nll += delta;
        sum_abs_delta_nll += delta.abs();
        max_abs_delta_nll = max_abs_delta_nll.max(delta.abs());
        let same = reference_score.top == candidate_score.top;
        agree += usize::from(same);
        exact_rows += usize::from(raw_reference == raw_candidate);
        if !same && reference_score.margin > ROUTE_MARGIN {
            margin_failures.push(at);
        }
        let max_logit_delta = reference
            .iter()
            .zip(&candidate)
            .map(|(&r, &c)| (f64::from(c) - f64::from(r)).abs())
            .fold(0.0f64, f64::max);
        let equal_values = reference
            .iter()
            .zip(&candidate)
            .filter(|(r, c)| r.to_bits() == c.to_bits())
            .count();
        row_reports.push(serde_json::json!({
            "row": at, "input": ids[at], "teacher": teacher,
            "reference_top1": reference_score.top, "candidate_top1": candidate_score.top,
            "kv8_reference_margin": reference_score.margin, "kv8_candidate_margin": candidate_score.margin,
            "reference_nll": reference_score.nll, "candidate_nll": candidate_score.nll,
            "signed_delta_nll": delta, "absolute_delta_nll": delta.abs(),
            "finite_reference_head": finite_reference, "finite_candidate_head": finite_candidate,
            "finite_reference_sampler": reference[..SAMPLER_VOCAB].iter().filter(|v| v.is_finite()).count(),
            "finite_candidate_sampler": candidate[..SAMPLER_VOCAB].iter().filter(|v| v.is_finite()).count(),
            "finite_hidden": finite_hidden,
            "raw_equal_head_values": equal_values, "raw_equal_row": raw_reference == raw_candidate,
            "max_abs_logit_delta": max_logit_delta,
            "reference_sha256": format!("{:x}", Sha256::digest(raw_reference)),
            "candidate_sha256": format!("{:x}", Sha256::digest(raw_candidate)),
            "hidden_sha256": format!("{:x}", Sha256::digest(raw_hidden)),
        }));
        println!(
            "private CPF KV8 target row={at} input={} teacher={teacher} \
             reference_top={} chunk_top={} reference_margin={:.9} chunk_margin={:.9} \
             reference_nll={:.12} chunk_nll={:.12} delta_nll={delta:.12} \
             finite_head={finite_reference}/{finite_candidate}/{HEAD_ROWS} finite_hidden={finite_hidden}/{} \
             raw_equal_values={equal_values}/{HEAD_ROWS} raw_row_equal={} max_abs_logit_delta={max_logit_delta:.9} \
             reference_sha256={:x} chunk_sha256={:x} hidden_sha256={:x}",
            ids[at],
            reference_score.top,
            candidate_score.top,
            reference_score.margin,
            candidate_score.margin,
            reference_score.nll,
            candidate_score.nll,
            observed.hidden_width,
            raw_reference == raw_candidate,
            Sha256::digest(raw_reference),
            Sha256::digest(raw_candidate),
            Sha256::digest(raw_hidden),
        );
    }
    let mean_delta_nll = sum_delta_nll / ROWS as f64;
    println!(
        "private CPF KV8 target KV8-route-equivalence rows={ROWS} top1_agreement={agree}/{ROWS} \
         raw_equal_rows={exact_rows}/{ROWS} mean_delta_nll={mean_delta_nll:.12} \
         mean_abs_delta_nll={:.12} max_abs_delta_nll={max_abs_delta_nll:.12} \
         high_margin_flips={margin_failures:?}; reference is Int8G32 production, not FP16",
        sum_abs_delta_nll / ROWS as f64,
    );
    write_json(
        &artifact_dir.join("scores.json"),
        &serde_json::json!({
            "reference": { "route": "production_run_token", "kv_mode": "Int8G32" },
            "candidate": { "route": "private_chunk", "kv_mode": "Int8G32", "kernel_class": format!("{:?}", observed.kernel_class) },
            "scope": "KV8 route equivalence, NOT FP16 G-KV8 or full256 acceptance",
            "rows": row_reports, "top1_agreement": agree, "raw_equal_rows": exact_rows,
            "mean_signed_delta_nll": mean_delta_nll,
            "mean_absolute_delta_nll": sum_abs_delta_nll / ROWS as f64,
            "max_absolute_delta_nll": max_abs_delta_nll,
            "kv8_reference_high_margin_flips": margin_failures,
            "mean_signed_delta_nll_limit": ROUTE_MEAN_NLL_LIMIT,
            "kv8_reference_margin_threshold": ROUTE_MARGIN,
        }),
    )
    .expect("save route scores before numerical acceptance assertions");
    (mean_delta_nll, margin_failures)
}

#[test]
#[ignore = "full Q36 Native weights on the 3090 under reviewed controls; run by exact name"]
fn private_kv8_m16_all_layer_rows_match_production_p5() {
    if !super::super::native_device_rig::device_present(
        "private_kv8_m16_all_layer_rows_match_production_p5",
    ) {
        return;
    }
    let ids = canonical_ids().expect("the pinned canonical P5 ids");
    let observed = match device_phase(&ids, ChunkKernelClass::ProductionMmq) {
        Ok(observed) => observed,
        Err(error) => panic!("the private CPF KV8 target device phase failed: {error:?}"),
    };
    // Host-only from here. Even a failed assertion cannot drop a CUDA owner during unwinding.
    let artifact_dir =
        save_rows(&observed, &ids, "cpf-target16").expect("save exclusive raw target artifacts");
    let _ = inspect_rows(&observed, &ids, &artifact_dir);
}

#[test]
#[ignore = "full Q36 weights; projection-class diagnostics only, not G-KV8 acceptance"]
fn private_kv8_m16_projection_kernel_class_diagnostic_p5() {
    if !super::super::native_device_rig::device_present(
        "private_kv8_m16_projection_kernel_class_diagnostic_p5",
    ) {
        return;
    }
    let ids = canonical_ids().expect("the pinned canonical P5 ids");
    for (class, artifact_name) in [
        (ChunkKernelClass::ProductionMmq, "cpf-target16-mmq"),
        (ChunkKernelClass::DenseMmvq, "cpf-target16-dense-mmvq"),
        (ChunkKernelClass::AllMmvq, "cpf-target16-all-mmvq"),
    ] {
        let observed = device_phase(&ids, class)
            .unwrap_or_else(|error| panic!("private CPF kernel-class {class:?}: {error:?}"));
        // Device owners have dropped; persist all raw rows before host assertions.
        let dir =
            save_rows(&observed, &ids, artifact_name).expect("save exclusive diagnostic rows");
        let (delta, flips) = inspect_rows(&observed, &ids, &dir);
        println!(
            "private CPF projection-class diagnostic class={class:?} mean_signed_delta_nll={delta:.12} \
             KV8_reference_high_margin_flips={flips:?}; diagnostic completion is NOT model acceptance"
        );
    }
}

include!("prefill_kv8_cohort_device_tests.rs");
