//! The device loop of the stage 1 gate (RD-10a-17): one `QwenResidentStep` per call, one untimed
//! prompt pass (the warm-up), then the 128 timed decode steps of every run. The loop is thin:
//! `rows`/`steps`/`phases`/`raw` compute the numbers, and the step's own `RunReport` supplies
//! the capture outcome and the per-token sync and host-write counts, never the setting. On the
//! merged RD-9d step API `replay` brackets the `run_token` call (the device step with its ONE
//! synchronize; the pair and row copies the same call performs sit inside it), `readback` the
//! copy of the 993 KB row into the preallocated host buffer, `host` the loop bookkeeping; the
//! top-32, hash and log-sum-exp run after the clock stops (`analysis_ns`). The ~127 MB row
//! buffers are allocated BEFORE the timer starts, so no allocation sits inside a timed region.
//! The tokenizer bundle beside the GGUF supplies `QwenRules` (Q-M7): the eos ids and the
//! sampler vocabulary come from the SAME loaded bundle, and the bundle is loaded BEFORE any
//! device work (a missing tokenizer file fails in seconds, not after a 20 GB upload). The raw
//! identities echo the settings' full-file `gguf_sha256` for the evaluator, and
//! `INFERENCE_CUDA_SYNC_EACH=1` is refused unconditionally (RD-10a-17b): it changes the timing
//! the gate measures. Design guidance only, no code copied (no MegaCapybara-derived design).

use super::{
    GateReport, GateSettings,
    phases::{MonotonicClock, Phase, PhaseTimer, Timing},
    raw::{RunMeta, SCHEMA, run_json, sha256_hex, step_json},
    rows::summarize,
    steps::{Mode, N_STEPS, input_for, logit_of},
};
use crate::{
    EngError,
    family::{
        QWEN3_6_35B_A3B,
        gguf_model::{Inventory, open_qwen3_6_35b_a3b},
        project::cuda::{CudaProjector, CudaWeights, LoadOptions, Reserve},
        qwen36::{
            cuda::{
                dims::{KvMode, StepDims, StepOptions},
                readback::{SYNC_EACH_VAR, parse_sync_each},
                resources::StepResources,
                rules::QwenRules,
                run_report::CaptureOutcome,
                step::{CaptureConfig, QwenResidentStep},
            },
            host::HostShape,
            settings::tokenizer_pin,
        },
    },
    kernels::cuda::CudaContext,
    tokenizer::TokenizerBundle,
};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Rotary dimension of the pinned model (head_dim 256 times the partial-rotary factor 0.25);
/// the GGUF carries no config.json, and the RD-9b fixtures name the same 64.
const ROTARY_DIM: u32 = 64;
/// Less than this many bytes free after the load is a typed refusal, not an error (0040 A2).
const MIN_FREE_AFTER_LOAD: usize = 1 << 30;

fn bad(text: &str) -> EngError {
    EngError::BadRequest(format!("gate: {text}"))
}

/// A phase-timer misuse is a host bug, reported as a typed error instead of a wrong number.
fn check<T>(step: Result<T, &'static str>) -> Result<T, EngError> {
    step.map_err(bad)
}

fn unix_ns() -> Result<u64, EngError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .map_err(|_| bad("the wall clock is before the epoch"))
}

/// The mode the settings ask for: forced reference lists present means forced.
fn mode_of(forced: &[Vec<u32>]) -> Mode {
    if forced.is_empty() {
        Mode::Free
    } else {
        Mode::Forced
    }
}

/// The consistency `FromStr` cannot see: stage-1 terms (fp16 KV, 128 rows), agreeing lists, a
/// forced stream long enough for every row, non-empty prompts.
fn check_settings(s: &GateSettings) -> Result<Mode, EngError> {
    if !s.kv_fp16 {
        return Err(bad("stage 1 runs an fp16 KV cache: kv_fp16 must be true"));
    }
    if s.gen_tokens as usize != N_STEPS {
        return Err(bad("the gate times exactly 128 decode steps"));
    }
    let (mode, prompts) = (mode_of(&s.forced), s.prompts.len());
    if s.names.len() != prompts || prompts == 0 || s.prompts.iter().any(Vec::is_empty) {
        return Err(bad("names and prompts disagree or a prompt is empty"));
    }
    let forced_ok = mode == Mode::Forced
        && s.forced.len() == prompts
        && s.forced_stream_sha256.len() == prompts
        && s.forced.iter().all(|stream| stream.len() >= N_STEPS);
    let free_ok = mode == Mode::Free && s.forced.is_empty() && s.forced_stream_sha256.is_empty();
    if forced_ok || free_ok {
        return Ok(mode);
    }
    Err(bad(
        "forced and forced_stream_sha256 disagree with the prompt list",
    ))
}

/// A gate run is never a debug run: `INFERENCE_CUDA_SYNC_EACH=1` synchronizes after every
/// launch, which changes the timing the gate measures, so the gate refuses it unconditionally,
/// capture or not (RD-10a-17b item 2).
fn refuse_sync_each(sync_each: bool) -> Result<(), EngError> {
    if sync_each {
        return Err(bad(&format!(
            "{SYNC_EACH_VAR}=1 synchronizes after every launch, which changes the timing the gate measures"
        )));
    }
    Ok(())
}

/// KV positions the session must hold: the longest prompt plus the 128 decode rows.
fn context_tokens(longest: usize) -> Result<usize, EngError> {
    longest
        .checked_add(N_STEPS)
        .ok_or_else(|| bad("the KV context size overflows"))
}

/// `ids_sha256` of the raw file: the SHA-256 of the comma-joined decimal ids.
fn ids_digest(ids: &[u32]) -> String {
    let joined = ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    sha256_hex(joined.as_bytes())
}

/// The logits row of one step as values: little-endian f32 bytes, a ragged tail is dropped.
fn row_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// The per-run capture flags from the step's own outcome, never the setting: only a captured
/// run says `captured` and had a graph before the timed region; Eager with graph mode requested
/// never ran the mode at all (the capture setting did not reach the step), which stops the call.
fn capture_flags(outcome: CaptureOutcome, requested: bool) -> Result<(bool, bool), EngError> {
    match outcome {
        CaptureOutcome::Captured => Ok((true, true)),
        CaptureOutcome::RefusedFellBackToEager => Ok((false, false)),
        CaptureOutcome::Eager if requested => Err(bad(
            "graph mode never ran: the step reports an eager outcome although capture was \
             requested (the capture setting did not reach the step)",
        )),
        CaptureOutcome::Eager => Ok((false, false)),
    }
}

/// The top-level raw document of schema `cuda-rd-raw/2`; the binary adds the identities only it
/// knows (pid, exe, commit, GPU, driver).
fn gate_doc(mode: Mode, identities: Value, prompts: Vec<Value>) -> Value {
    json!({
        "schema": SCHEMA,
        "mode": if mode == Mode::Forced { "forced" } else { "free" },
        "incomplete": false,
        "refused": Option::<Value>::None,
        "identities": identities,
        "prompts": prompts,
    })
}

/// The identity fields of the memory admission plus the settings' full-file GGUF hash, echoed
/// for the evaluator (0040 A2; `rawcheck.py` requires `identities.gguf_sha256`).
fn identities_of(s: &GateSettings, total: usize, before: usize, after: usize) -> Value {
    json!({
        "capture_requested": if s.capture { "graph" } else { "eager" },
        "gguf_sha256": s.gguf_sha256,
        "kv_mode": "fp16",
        "mem_total": total,
        "mem_free_before": before,
        "mem_free_after_load": after,
    })
}

fn report(doc: Value) -> Result<GateReport, EngError> {
    Ok(GateReport {
        raw: serde_json::to_string(&doc)
            .map_err(|e| bad(&format!("the raw gate file does not serialize: {e}")))?,
    })
}

/// The session options of the call: fp16 KV, one committed DeltaNet slot, no forked experts, a
/// cache that holds every prompt plus the decode rows.
fn session_options(s: &GateSettings) -> Result<StepOptions, EngError> {
    Ok(StepOptions {
        kv_mode: KvMode::Fp16,
        context_tokens: context_tokens(s.prompts.iter().map(Vec::len).max().unwrap_or(0))?,
        delta_slots: 1,
        forked_experts: false,
    })
}

/// Everything a session builds once: the checked dimensions of the pinned model and the arena.
fn build_resources(context: &CudaContext, s: &GateSettings) -> Result<StepResources, EngError> {
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
    let dims = StepDims::new(&shape, &inventory)?;
    StepResources::new(context, dims, &inventory, &session_options(s)?)
}

/// The head width of the pinned model from the host-side dimensions: no context, no device, so
/// the rules can be checked before any device work (RD-10a-17b item 4).
fn head_rows_of() -> Result<usize, EngError> {
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
    Ok(StepDims::new(&shape, &inventory)?.shape.vocab)
}

/// The directory the pinned tokenizer bundle sits in: the GGUF's parent directory, never the
/// GGUF path itself (the settings carry no tokenizer path; the pinned files sit beside the
/// GGUF on the run host).
fn bundle_dir(gguf: &Path) -> Result<PathBuf, EngError> {
    gguf.parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .ok_or_else(|| bad("the gguf path has no directory to read the tokenizer from"))
}

/// The pure seam of the tokenizer wiring (Q-M7, RD-10a-17b item 5): the eos ids and the sampler
/// vocabulary come from the SAME loaded bundle — never hard-coded ids, never the padded head
/// width — and the tokenizer directory is the GGUF's parent directory, never the GGUF path
/// itself. `device_gate` calls this seam with the loaded bundle's own numbers.
fn rules_from_bundle(
    eos: &[u32],
    vocab_size: usize,
    head_rows: usize,
    gguf: &Path,
) -> Result<(QwenRules, PathBuf), EngError> {
    Ok((
        QwenRules::new(eos.to_vec(), vocab_size, head_rows)?,
        bundle_dir(gguf)?,
    ))
}

/// The resident step over the loaded weights and the rules built from the tokenizer bundle
/// BEFORE any device work (KUBDAN-MSG-56; Q-M7: same bundle, no constants). `tokenizer_pin`
/// checked the bundle's digests when it was loaded. The gate's capture setting is handed to
/// the step here, before its first token, so a graph-mode call actually captures.
fn build_step<'a>(
    projector: &'a CudaProjector,
    res: &'a StepResources,
    weights: &'a CudaWeights,
    rules: QwenRules,
    s: &GateSettings,
) -> Result<QwenResidentStep<'a>, EngError> {
    let mut step = QwenResidentStep::new(
        projector.context(),
        res,
        weights,
        &session_options(s)?,
        rules,
    )?;
    step.set_capture_config(CaptureConfig { capture: s.capture })?;
    Ok(step)
}

/// The per-prompt inputs of the timed runs (`check_settings` has agreed the lists).
struct RunInputs<'a> {
    mode: Mode,
    requested: bool,
    prompt: &'a [u32],
    forced: &'a [u32],
}

/// What one timed region produced, before the JSON assembly.
struct Timed {
    timing: Timing,
    rows: Vec<Vec<u8>>,
    fed: Vec<u32>,
    syncs: u64,
    host_writes: u64,
    captured: bool,
    graph_ready: bool,
    prompt_synced: u64,
    timed_start: u64,
    timed_end: u64,
}

enum Region {
    Complete(Timed),
    Expired { steps_done: usize },
}

/// The untimed prompt pass: every prompt token but the last runs the step (the logits are
/// computed, the row is not read). This pass is the warm-up (REPLY-72 Q8).
fn prefill(step: &mut QwenResidentStep<'_>, x: &RunInputs<'_>) -> Result<u32, EngError> {
    step.reset_state()?;
    let last = *x
        .prompt
        .last()
        .ok_or_else(|| bad("a prompt has no tokens"))?;
    for &token in &x.prompt[..x.prompt.len() - 1] {
        step.run_token(token, false)?;
    }
    Ok(last)
}

/// The row buffers and `fed`, allocated BEFORE `PhaseTimer::start` so no allocation sits
/// inside a timed region (RD-10a-17b item 3): one zeroed logits row per timed step and the
/// fed vector at its full capacity. The caller passes the row length (`step.row_bytes()`),
/// so the helper runs and is checked on the CPU, with no device.
fn alloc_rows(row_len: usize) -> (Vec<Vec<u8>>, Vec<u32>) {
    let rows: Vec<Vec<u8>> = (0..N_STEPS).map(|_| vec![0u8; row_len]).collect();
    (rows, Vec::with_capacity(N_STEPS))
}

/// One run: the prefill, then the 128 timed steps; on deadline expiry the region reports how
/// many steps completed and the caller writes the incomplete document.
fn timed_region(
    step: &mut QwenResidentStep<'_>,
    x: &RunInputs<'_>,
    started: Instant,
    secs: u32,
) -> Result<Region, EngError> {
    let last_prompt = prefill(step, x)?;
    let prompt_synced = unix_ns()?;
    let (mut rows, mut fed) = alloc_rows(step.row_bytes().len());
    let mut timer = PhaseTimer::start(MonotonicClock::new());
    let timed_start = unix_ns()?;
    let (mut prev, mut syncs, mut host_writes) = (0u32, 0u64, 0u64);
    for (j, row) in rows.iter_mut().enumerate() {
        let token = input_for(x.mode, j, last_prompt, x.forced, prev).map_err(bad)?;
        check(timer.step_begin())?;
        fed.push(token);
        check(timer.begin(Phase::Replay))?;
        let chosen = step.run_token(token, true)?;
        check(timer.end(Phase::Replay))?;
        check(timer.begin(Phase::Readback))?;
        row.copy_from_slice(step.row_bytes());
        check(timer.end(Phase::Readback))?;
        check(timer.begin(Phase::Host))?;
        prev = chosen;
        let run = step.last_report();
        syncs += u64::from(run.syncs);
        host_writes += u64::from(run.host_writes);
        check(timer.end(Phase::Host))?;
        check(timer.step_end())?;
        if secs != 0 && started.elapsed().as_secs() >= u64::from(secs) {
            return Ok(Region::Expired { steps_done: j + 1 });
        }
    }
    let timed_end = unix_ns()?;
    let timing = check(timer.finish())?;
    let (captured, graph_ready) = capture_flags(step.last_report().capture, x.requested)?;
    Ok(Region::Complete(Timed {
        timing,
        rows,
        fed,
        syncs,
        host_writes,
        captured,
        graph_ready,
        prompt_synced,
        timed_start,
        timed_end,
    }))
}

/// The run JSON: the analysis (after the clock stops) and the step list of `raw`.
fn run_json_of(t: &Timed, x: &RunInputs<'_>) -> Result<Value, EngError> {
    let analysis = Instant::now();
    let mut steps = Vec::with_capacity(N_STEPS);
    for (j, bytes) in t.rows.iter().enumerate() {
        let values = row_values(bytes);
        let summary = summarize(&values);
        let forced_logit = x.forced.get(j).and_then(|&token| logit_of(&values, token));
        steps.push(step_json(t.fed[j], &summary, forced_logit));
    }
    let meta = RunMeta {
        captured: t.captured,
        graph_ready_before_timed: t.graph_ready,
        analysis_ns: analysis.elapsed().as_nanos() as u64,
        syncs: t.syncs,
        host_writes: t.host_writes,
        prompt_synced_unix_ns: t.prompt_synced,
        timed_start_unix_ns: t.timed_start,
        timed_end_unix_ns: t.timed_end,
    };
    Ok(run_json(&meta, &t.timing, steps))
}

/// Every prompt and every run, then the raw document; the deadline expiry ends the call with
/// the incomplete document and what was finished.
fn run_all(
    step: &mut QwenResidentStep<'_>,
    s: &GateSettings,
    mode: Mode,
    identities: Value,
) -> Result<GateReport, EngError> {
    let started = Instant::now();
    let mut prompts: Vec<Value> = Vec::with_capacity(s.prompts.len());
    for index in 0..s.prompts.len() {
        let x = RunInputs {
            mode,
            requested: s.capture,
            prompt: &s.prompts[index],
            forced: if mode == Mode::Forced {
                &s.forced[index]
            } else {
                &[]
            },
        };
        let mut runs = Vec::with_capacity(s.runs as usize);
        for _ in 0..s.runs {
            match timed_region(step, &x, started, s.deadline_secs)? {
                Region::Expired { steps_done } => {
                    let mut doc = gate_doc(mode, identities, prompts);
                    doc["incomplete"] = json!(true);
                    doc["elapsed_s"] = json!(started.elapsed().as_secs());
                    doc["partial"] = json!({ "steps_done": steps_done, "elapsed_ns": started.elapsed().as_nanos() as u64 });
                    return report(doc);
                }
                Region::Complete(t) => runs.push(run_json_of(&t, &x)?),
            }
        }
        prompts.push(json!({
            "id": s.names[index],
            "ids_sha256": ids_digest(x.prompt),
            "n_prompt": x.prompt.len(),
            "forced_stream_sha256": if x.mode == Mode::Forced { s.forced_stream_sha256[index].as_str() } else { "" },
            "runs": runs,
        }));
    }
    report(gate_doc(mode, identities, prompts))
}

/// Runs the whole gate call: the settings checks and the sync-each refusal, the tokenizer
/// bundle and its rules (before any device work), the context, the load, the memory admission,
/// the step, the prompt and run loop, and the raw JSON of schema `cuda-rd-raw/2`.
pub(super) fn device_gate(s: &GateSettings) -> Result<GateReport, EngError> {
    let mode = check_settings(s)?;
    let sync_each = parse_sync_each(std::env::var(SYNC_EACH_VAR).ok().as_deref())?;
    refuse_sync_each(sync_each)?;
    // The tokenizer bundle and the rules come BEFORE any device work (KUBDAN-MSG-56): a
    // missing tokenizer file fails in seconds, not after a 20 GB upload.
    let head_rows = head_rows_of()?;
    let dir = bundle_dir(&s.gguf)?;
    let bundle = TokenizerBundle::load(&dir, &tokenizer_pin()?)?;
    let vocab = usize::try_from(bundle.vocab_size())
        .map_err(|_| bad("the tokenizer vocabulary exceeds usize"))?;
    let (rules, _) = rules_from_bundle(bundle.eos_token_ids(), vocab, head_rows, &s.gguf)?;
    let projector = CudaProjector::new()?;
    let before = projector.memory_info()?;
    let model = open_qwen3_6_35b_a3b(&s.gguf, None)?;
    let loaded = projector.load_model(
        &model,
        LoadOptions {
            reserve: Reserve {
                kv_cache: 0,
                work: 0,
                headroom: 0,
            },
            drop_page_cache: None,
        },
    )?;
    let after = projector.memory_info()?;
    let identities = identities_of(s, after.total, before.free, after.free);
    if after.free < MIN_FREE_AFTER_LOAD {
        let mut doc = gate_doc(mode, identities, Vec::new());
        doc["refused"] = json!({ "kind": "memory", "free_after_load": after.free });
        return report(doc);
    }
    let res = build_resources(projector.context(), s)?;
    let mut step = build_step(&projector, &res, &loaded.device, rules, s)?;
    run_all(&mut step, s, mode, identities)
}
#[cfg(test)]
mod tests {
    use super::*;

    const FREE: &str = r#"{"gguf":"/m/q.gguf","gguf_sha256":"ab","names":["P1"],"prompts":[[7,8,9]],
        "forced":[],"forced_stream_sha256":[],"gen_tokens":128,"runs":1,
        "capture":false,"kv_fp16":true,"deadline_secs":270}"#;

    /// The settings text of a call whose reference stream has `len` tokens (0 = the free shape).
    fn call_text(len: usize) -> String {
        let stream: Vec<String> = (0..len).map(|id| id.to_string()).collect();
        let (forced, sha) = if len == 0 {
            ("[]".to_owned(), "[]".to_owned())
        } else {
            (format!("[[{}]]", stream.join(",")), r#"["cd"]"#.to_owned())
        };
        format!(
            r#"{{"gguf":"/m/q.gguf","gguf_sha256":"ab","names":["P1"],"prompts":[[7,8,9]],"forced":{forced},"forced_stream_sha256":{sha},"gen_tokens":128,"runs":1,"capture":false,"kv_fp16":true,"deadline_secs":270}}"#
        )
    }

    fn as_call(text: &str) -> Result<Mode, EngError> {
        check_settings(&text.parse().unwrap())
    }

    #[test]
    fn forced_lists_make_a_forced_call_and_empty_lists_a_free_one() {
        assert_eq!(as_call(&call_text(N_STEPS)), Ok(Mode::Forced));
        assert_eq!(as_call(FREE), Ok(Mode::Free));
        assert_eq!(
            (mode_of(&[]), mode_of(&[vec![1]])),
            (Mode::Free, Mode::Forced)
        );
    }

    #[test]
    fn the_stage_one_terms_are_refused_before_anything_runs() {
        for bad_text in [
            FREE.replace("\"kv_fp16\":true", "\"kv_fp16\":false"),
            FREE.replace("\"gen_tokens\":128", "\"gen_tokens\":64"),
        ] {
            assert!(as_call(&bad_text).is_err(), "{bad_text}");
        }
        // No device is touched: the refusal comes before the context.
        let refused = FREE.replace("\"kv_fp16\":true", "\"kv_fp16\":false");
        assert!(matches!(
            run_step_gate_of(&refused),
            Err(EngError::BadRequest(_))
        ));
    }

    fn run_step_gate_of(text: &str) -> Result<GateReport, EngError> {
        crate::family::qwen36::cuda::gate::run_step_gate(&text.parse().unwrap())
    }

    #[test]
    fn list_disagreements_are_refused() {
        // A names/prompts disagreement is FromStr's own check and fails before this module.
        let two_names = FREE.replace(r#""names":["P1"]"#, r#""names":["P1","P2"]"#);
        assert!(two_names.parse::<GateSettings>().is_err());
        for bad_text in [
            call_text(N_STEPS).replace(r#"["cd"]"#, "[]"),
            FREE.replace(
                r#""forced_stream_sha256":[]"#,
                r#""forced_stream_sha256":["cd"]"#,
            ),
            call_text(1),
            FREE.replace("[[7,8,9]]", "[[]]"),
        ] {
            assert!(as_call(&bad_text).is_err(), "{bad_text}");
        }
    }

    #[test]
    fn the_digest_is_the_sha256_of_the_comma_joined_ids() {
        assert_eq!(
            ids_digest(&[1, 2, 3]),
            "8a6ae15122001229edb8866f56e342af12ae8187203c3e3b33931743e7c0c48d"
        );
        assert_eq!(ids_digest(&[]).len(), 64);
    }

    #[test]
    fn row_values_reads_little_endian_and_drops_a_ragged_tail() {
        assert_eq!(row_values(&[0x00, 0x00, 0x80, 0x3F]), vec![1.0f32]);
        assert_eq!(row_values(&[0x00, 0x00, 0x80, 0x3F, 0xFF]), vec![1.0f32]);
        assert!(row_values(&[]).is_empty());
    }

    #[test]
    fn the_document_carries_the_schema_the_mode_and_the_refused_slot() {
        let doc = gate_doc(
            Mode::Forced,
            identities_of(&FREE.parse().unwrap(), 10, 4, 2),
            Vec::new(),
        );
        let mut keys: Vec<_> = doc.as_object().unwrap().keys().cloned().collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "identities",
                "incomplete",
                "mode",
                "prompts",
                "refused",
                "schema"
            ]
        );
        assert_eq!(doc["schema"], json!("cuda-rd-raw/2"));
        assert_eq!(doc["mode"], json!("forced"));
        assert_eq!(doc["incomplete"], json!(false));
        assert!(doc["refused"].is_null());
        assert_eq!(
            doc["identities"],
            json!({
                "capture_requested": "eager",
                "gguf_sha256": "ab",
                "kv_mode": "fp16",
                "mem_total": 10,
                "mem_free_before": 4,
                "mem_free_after_load": 2,
            })
        );
    }

    #[test]
    fn the_identities_echo_the_settings_gguf_sha256() {
        let s: GateSettings = FREE.parse().unwrap();
        assert_eq!(
            identities_of(&s, 10, 4, 2)["gguf_sha256"],
            json!(s.gguf_sha256)
        );
        assert_eq!(identities_of(&s, 10, 4, 2)["gguf_sha256"], json!("ab"));
    }

    #[test]
    fn sync_each_is_refused_whether_or_not_a_capture_was_requested() {
        assert!(matches!(
            refuse_sync_each(true),
            Err(EngError::BadRequest(_))
        ));
        assert_eq!(refuse_sync_each(false), Ok(()));
    }

    #[test]
    fn the_rules_and_the_tokenizer_dir_come_from_the_bundle_and_the_ggufs_parent() {
        let (rules, dir) = rules_from_bundle(
            &[248_046, 248_044],
            248_077,
            248_320,
            Path::new("/m/dir/model.gguf"),
        )
        .unwrap();
        assert_eq!(rules.eos.to_vec(), [248_046, 248_044]);
        assert_eq!(rules.sampler_vocab, 248_077);
        assert_eq!(dir, PathBuf::from("/m/dir"));
    }

    #[test]
    fn capture_flags_come_from_the_outcome_never_the_setting() {
        assert_eq!(
            capture_flags(CaptureOutcome::Captured, true),
            Ok((true, true))
        );
        assert_eq!(
            capture_flags(CaptureOutcome::RefusedFellBackToEager, true),
            Ok((false, false))
        );
        assert_eq!(
            capture_flags(CaptureOutcome::Eager, false),
            Ok((false, false))
        );
        // The STOP of a graph-mode run that stayed eager: the merged step exists, so the text
        // must name the real failure (the setting never reached the step), not an absent
        // packet.
        match capture_flags(CaptureOutcome::Eager, true) {
            Err(EngError::BadRequest(message)) => {
                assert!(
                    !message.contains("RD-11a"),
                    "the STOP text still names RD-11a after the merge: {message}"
                );
                assert!(
                    message.contains("capture was requested"),
                    "the STOP text does not say capture was requested: {message}"
                );
            }
            other => panic!("Eager with capture requested is a STOP, not {other:?}"),
        }
    }

    #[test]
    fn build_step_hands_the_capture_setting_to_the_step() {
        // The step cannot run on the CPU, so the wiring is pinned on this file's own text
        // (the c17 style): comments are stripped and whitespace is removed entirely, so only
        // the real call with its `?` matches, and a comment or a dead spelling cannot.
        let body = code_only(fn_section("fn build_step"))
            .split_whitespace()
            .collect::<String>();
        let length = body.len();
        let call = body
            .find("set_capture_config(CaptureConfig{capture:s.capture})?")
            .unwrap_or_else(|| {
                panic!("build_step ({length} bytes) never hands the setting to the step")
            });
        let built = body
            .find("QwenResidentStep::new(")
            .unwrap_or_else(|| panic!("build_step ({length} bytes) never constructs the step"));
        let returned = body
            .rfind("Ok(")
            .unwrap_or_else(|| panic!("build_step ({length} bytes) never returns the step"));
        assert!(
            built < call && call < returned,
            "the capture setting ({call}) must travel after construction ({built}) and before \
             the return ({returned})"
        );
        for wrong in [
            "capture:false",
            "CaptureConfig::default",
            "capture:!s.capture",
        ] {
            assert!(
                !body.contains(wrong),
                "build_step ({length} bytes) carries the wrong capture spelling {wrong}"
            );
        }
    }

    #[test]
    fn the_session_holds_the_longest_prompt_plus_the_decode_rows() {
        assert_eq!(context_tokens(1024).unwrap(), 1152);
        assert!(context_tokens(usize::MAX).is_err());
    }

    /// The text of one `fn` item of this file's own production code, from the line that holds
    /// `start` to the next line that opens a `fn` item at the same indentation or the
    /// `#[cfg(test)]` line, whatever the `pub` prefix (the hardened c17 `step_tests.rs` style,
    /// pinned on `run.rs` itself): the step cannot run on the CPU, so a mutant of a wiring must
    /// still fail a named test.
    fn fn_section(start: &str) -> &'static str {
        let source = include_str!("run.rs");
        let at = source
            .find(start)
            .unwrap_or_else(|| panic!("run.rs has no {start:?}"));
        let line_start = source[..at].rfind('\n').map_or(0, |nl| nl + 1);
        let mut end = source.len();
        let mut line = source[at..]
            .find('\n')
            .map_or(source.len(), |nl| at + nl + 1);
        while line < source.len() {
            let next = source[line..]
                .find('\n')
                .map_or(source.len(), |nl| line + nl + 1);
            let text = source[line..next].trim_start();
            if starts_fn_item(text) || text.starts_with("#[cfg(test)]") {
                end = line;
                break;
            }
            line = next;
        }
        &source[line_start..end]
    }

    /// True when `line` opens a `fn` item behind any `pub` prefix.
    fn starts_fn_item(line: &str) -> bool {
        let rest = ["pub(crate) ", "pub(super) ", "pub "]
            .iter()
            .find_map(|prefix| line.strip_prefix(prefix))
            .unwrap_or(line);
        rest.starts_with("fn ")
    }

    /// A copy of the section text without comments and without string or char literals:
    /// every `//` line comment, every nested `/* ... */` block comment and every `"..."` or
    /// `'...'` literal (with `\` escapes) becomes one space, so a comment cannot satisfy a
    /// pin and a `/*` inside a literal starts no comment (lifetimes like `'a` are kept).
    fn code_only(text: &str) -> String {
        let bytes = text.as_bytes();
        let mut out = String::with_capacity(text.len());
        let (mut plain, mut at) = (0, 0);
        while at < bytes.len() {
            let skipped = match bytes[at] {
                b'/' if bytes.get(at + 1) == Some(&b'/') => Some(skip_line_comment(bytes, at)),
                b'/' if bytes.get(at + 1) == Some(&b'*') => {
                    Some(skip_block_comment(bytes, at + 2, 1))
                }
                b'"' => Some(skip_quoted(bytes, at + 1, b'"')),
                b'\'' if opens_char_literal(bytes, at) => Some(skip_quoted(bytes, at + 1, b'\'')),
                _ => None,
            };
            if let Some(end) = skipped {
                out.push_str(&text[plain..at]);
                out.push(' ');
                at = end;
                plain = at;
            } else {
                at += 1;
            }
        }
        out.push_str(&text[plain..]);
        out
    }

    /// The index after the `//` line comment starting at `at` (the newline itself stays).
    fn skip_line_comment(bytes: &[u8], mut at: usize) -> usize {
        while at < bytes.len() && bytes[at] != b'\n' {
            at += 1;
        }
        at
    }

    /// The index after the nested block comment whose body starts at `at` (depth 1 on
    /// entry): `/*` nests, a `//` inside is plain text, and an unterminated comment ends at
    /// the end of the text instead of a panic.
    fn skip_block_comment(bytes: &[u8], mut at: usize, mut depth: usize) -> usize {
        while depth > 0 && at < bytes.len() {
            match (bytes.get(at), bytes.get(at + 1)) {
                (Some(b'/'), Some(b'*')) => {
                    depth += 1;
                    at += 2;
                }
                (Some(b'*'), Some(b'/')) => {
                    depth -= 1;
                    at += 2;
                }
                _ => at += 1,
            }
        }
        at
    }

    /// The index after the quoted literal (string or char) whose body starts at `at`.
    fn skip_quoted(bytes: &[u8], mut at: usize, quote: u8) -> usize {
        while at < bytes.len() {
            match bytes[at] {
                b'\\' => at = (at + 2).min(bytes.len()),
                b if b == quote => return at + 1,
                _ => at += 1,
            }
        }
        at
    }

    /// True when the `'` at `at` opens a char literal rather than a lifetime: `'x'`, `'''`
    /// never occurs, and an escape (`'\\..'`) always does.
    fn opens_char_literal(bytes: &[u8], at: usize) -> bool {
        matches!(
            (bytes.get(at + 1), bytes.get(at + 2)),
            (Some(b'\\'), _) | (Some(_), Some(b'\''))
        )
    }

    /// The pinned form of a section: code only, whitespace collapsed to one space, so a
    /// rustfmt re-wrap cannot break a pin.
    fn pinned_text(section: &str) -> String {
        code_only(section)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn code_only_strips_block_comments_and_keeps_the_rest_alive() {
        // Pinned text inside a block comment is not found: the comment becomes one space.
        let stripped = code_only(r#"let a = 1; /* refuse_sync_each(sync_each)?; */ let b = 2;"#);
        assert!(stripped.contains("let a = 1;"));
        assert!(stripped.contains("let b = 2;"));
        assert!(!stripped.contains("refuse_sync_each"));
        // Block comments nest: everything up to the OUTER close goes.
        let nested = code_only("/* outer /* inner */ still a comment */ let c = 3;");
        assert!(nested.contains("let c = 3;"));
        assert!(!nested.contains("still a comment"));
        // A `/*` inside a string literal starts no comment: the code after it survives.
        let quoted = code_only(r#"let s = "/*"; let d = 4;"#);
        assert!(quoted.contains("let d = 4;"));
        assert!(!quoted.contains("/*"));
        // A `//` inside a block comment does not end it: the close still closes.
        let mixed = code_only("/* // not a line end */ let e = 5;");
        assert!(mixed.contains("let e = 5;"));
        assert!(!mixed.contains("not a line end"));
        // An unterminated block comment strips to the end of the text and does not panic.
        let open = code_only("let f = 6; /* never closed");
        assert!(open.contains("let f = 6;"));
        assert!(!open.contains("never closed"));
    }

    #[test]
    fn code_only_keeps_lifetimes_and_skips_char_literals() {
        // The pinned sections carry lifetimes (`QwenResidentStep<'_>`): not literals.
        let lifetime = code_only("fn f(step: &mut QwenResidentStep<'_>) {}");
        assert!(lifetime.contains("QwenResidentStep<'_>"));
        // A quote inside a char literal opens no string: the code after it survives.
        let quoted = code_only("let q = '\"'; let g = 7;");
        assert!(quoted.contains("let g = 7;"));
        assert!(!quoted.contains("'\"'"));
        // An escaped quote inside a string ends nothing early.
        let escaped = code_only(r#"let h = "a\"b"; let i = 8;"#);
        assert!(escaped.contains("let i = 8;"));
        assert!(!escaped.contains(r#""a\"b""#));
    }

    #[test]
    fn code_only_gives_exact_output_for_escapes_comments_and_a_trailing_backslash() {
        // Exact outputs: each stripped piece becomes ONE space and nothing else changes.
        let table: [(&str, &str); 7] = [
            ("a/* x */b", "a b"),
            ("a// c\nb", "a \nb"),
            ("a\"s\"b", "a b"),
            (r#"let n = '\"'; let o = 12;"#, "let n =  ; let o = 12;"),
            (r#"let b = '\\'; let c = 13;"#, "let b =  ; let c = 13;"),
            ("let p = \"abc\\", "let p =  "),
            ("let q = 'x'; let r = 14;", "let q =  ; let r = 14;"),
        ];
        for (input, expected) in table {
            assert_eq!(code_only(input), expected, "code_only({input:?})");
        }
    }

    #[test]
    fn alloc_rows_builds_one_zeroed_row_per_step_and_an_empty_fed_at_capacity() {
        let row_len = 12;
        let (rows, fed) = alloc_rows(row_len);
        assert_eq!(
            rows.len(),
            N_STEPS,
            "the helper must build one row per timed step"
        );
        for (index, row) in rows.iter().enumerate() {
            assert_eq!(
                row.len(),
                row_len,
                "row {index} is not {row_len} bytes long"
            );
            assert!(
                row.iter().all(|&byte| byte == 0),
                "row {index} is not zeroed"
            );
        }
        assert!(fed.is_empty(), "fed starts empty");
        assert!(fed.capacity() >= N_STEPS, "fed is not at its full capacity");
    }

    #[test]
    fn the_row_buffers_are_allocated_before_the_timer_starts() {
        // The allocations live in the helper: one zeroed row per timed step and the fed
        // vector, both produced before the timer starts and the timed_start stamp.
        let helper = pinned_text(fn_section("fn alloc_rows("));
        let helper_len = helper.len();
        for allocation in ["vec![0u8; row_len]", "Vec::with_capacity(N_STEPS)"] {
            assert!(
                helper.contains(allocation),
                "alloc_rows ({helper_len} bytes) does not allocate {allocation}"
            );
        }
        let body = pinned_text(fn_section("fn timed_region("));
        let length = body.len();
        let helper_call = body
            .find("alloc_rows(step.row_bytes().len())")
            .unwrap_or_else(|| {
                panic!(
                    "timed_region ({length} bytes) never takes alloc_rows(step.row_bytes().len())"
                )
            });
        let timer = body
            .find("PhaseTimer::start")
            .unwrap_or_else(|| panic!("timed_region ({length} bytes) has no PhaseTimer::start"));
        let stamp = body
            .find("timed_start = unix_ns")
            .unwrap_or_else(|| panic!("timed_region ({length} bytes) never stamps timed_start"));
        assert!(
            helper_call < timer && helper_call < stamp,
            "the alloc_rows call ({helper_call}) sits inside the timed region (the timer \
             starts at {timer}, the timed_start stamp is at {stamp})"
        );
        // The loop must use the helper's own bindings: the raw forms live in alloc_rows only.
        for used in ["rows.iter_mut()", "fed.push("] {
            let at = body
                .find(used)
                .unwrap_or_else(|| panic!("the timed loop ({length} bytes) does not use {used}"));
            assert!(
                helper_call < at,
                "the loop uses {used} ({at}) before alloc_rows produced it ({helper_call})"
            );
        }
        assert!(
            !body[timer..].contains("alloc_rows"),
            "the timed region calls alloc_rows again after the timer starts"
        );
    }

    #[test]
    fn the_timed_region_text_spells_no_allocation() {
        let body = pinned_text(fn_section("fn timed_region("));
        let length = body.len();
        // What is actually checked: a substring denylist over the STRIPPED timed_region
        // text. It catches the known allocating spellings (a per-step to_vec copy, a
        // rebound Vec::from(..).collect(), a fresh Vec::new); a genuinely new allocation
        // whose spelling shares no substring with this list would still pass.
        for stray in [
            "to_vec",
            "Vec::new",
            "Vec::from",
            ".collect",
            ".clone()",
            "vec!",
            "Box::new",
            "format!",
            "String::from",
            ".to_owned",
            ".into_boxed",
            "Vec::with_capacity",
            "rows =",
            "fed =",
        ] {
            assert!(
                !body.contains(stray),
                "the timed region ({length} bytes) contains the allocating form {stray}"
            );
        }
        // The per-step copy is the pinned statement itself (quoted from run.rs).
        assert!(
            body.contains("row.copy_from_slice(step.row_bytes())"),
            "the per-step copy of the {length}-byte body is not row.copy_from_slice(step.row_bytes())"
        );
    }

    #[test]
    fn the_sync_each_refusal_is_propagated_before_any_device_call() {
        let body = pinned_text(fn_section("fn device_gate("));
        let length = body.len();
        // The pin demands the propagated call itself: a commented-out call (a line or a
        // block comment) is stripped with its comment, and a dead `let _ =
        // refuse_sync_each(sync_each);` binding carries no `?` behind the closing
        // parenthesis, so neither of them satisfies the pin.
        let refusal = body
            .find("refuse_sync_each(sync_each)?")
            .unwrap_or_else(|| {
                panic!("device_gate ({length} bytes) never propagates the refuse_sync_each refusal")
            });
        let device = body
            .find("CudaProjector::new")
            .unwrap_or_else(|| panic!("device_gate ({length} bytes) never opens the device"));
        assert!(
            refusal < device,
            "the sync-each refusal ({refusal}) runs after the device work ({device})"
        );
        // The call must also sit at the TOP LEVEL of the function body: an `if false { .. }`
        // wrapper, a `match` arm or a closure all move it off the unconditional path, and a
        // rebind of `sync_each` between the env read and the call silences the refusal.
        let before = &body[..refusal];
        let depth = before.matches('{').count() as isize - before.matches('}').count() as isize;
        assert_eq!(
            depth, 1,
            "the refusal ({refusal}) sits {depth} brace levels into device_gate, not at the top level"
        );
        let boundary = before.trim_end().chars().next_back();
        assert!(
            matches!(boundary, Some(';') | Some('{') | Some('}')),
            "the refusal ({refusal}) does not start a standalone statement of device_gate"
        );
        let env_read = body
            .find("std::env::var(SYNC_EACH_VAR)")
            .unwrap_or_else(|| panic!("device_gate ({length} bytes) never reads {SYNC_EACH_VAR}"));
        assert!(
            !body[env_read..refusal].contains("let sync_each"),
            "sync_each is rebound between the env read ({env_read}) and the refusal ({refusal})"
        );
    }

    #[test]
    fn the_bundle_and_the_rules_precede_every_device_call_of_the_gate() {
        let body = pinned_text(fn_section("fn device_gate("));
        let length = body.len();
        let load = body.find("TokenizerBundle::load").unwrap_or_else(|| {
            panic!("device_gate ({length} bytes) never loads the tokenizer bundle")
        });
        let rules = body
            .find("rules_from_bundle")
            .unwrap_or_else(|| panic!("device_gate ({length} bytes) never calls the rules seam"));
        for device in [
            "CudaProjector::new",
            "memory_info",
            "open_qwen3_6_35b_a3b",
            "load_model",
        ] {
            let at = body
                .find(device)
                .unwrap_or_else(|| panic!("device_gate ({length} bytes) has no {device}"));
            assert!(
                load < at && rules < at,
                "the tokenizer work (load {load}, rules {rules}) runs after the device call \
                 {device} ({at})"
            );
        }
        let seam = pinned_text(fn_section("fn rules_from_bundle("));
        let seam_len = seam.len();
        assert!(
            seam.contains("QwenRules::new("),
            "the rules seam ({seam_len} bytes) does not build QwenRules"
        );
    }
}
