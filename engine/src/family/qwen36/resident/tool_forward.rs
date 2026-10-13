//! The tool-forward diagnostic: the next-token distribution of the 922-token tool prompt on
//! the product `activation_dtype` setting of packet Q36-BF16-a, with the format-token gate.
//!
//! The mlx-lm 0.31.0 reference on the same int4 weights gives P("\\n") = 0.999555 after the 919
//! prompt ids of `IDS_FILE` plus the 3 suffix ids of `"</think>\\n\\n<tool_call>"` (922 ids), but
//! the served engine gave "\\n" 0.418 and "=read" 0.370. Prompt
//! rendering and the sampler are cleared, so the number is a property of the forward path. This
//! test measures the serve path itself on the real checkpoint, one mode per fresh session (each
//! mode resets the same loaded decoder, as the engine worker resets the session between jobs):
//!
//! - M1 `m1_serving_path`: the serving path as the executor plans it for the activation of the
//!   run (`Q36Executor::plan_prefill` on the shipped config
//!   `engine/config/qwen36-mac-studio-36gb.json`: `prefill_budget_bytes` 32,212,254,720,
//!   `prefill_session_reserve_bytes` 8,286,699,520, `cache_limit_bytes` 4,294,967,296), then
//!   the final decode step in `StepMode::Logits`. The largest chunk is NOT a copied constant:
//!   `serve_chunk_setting` below reads it through the same `Q36Settings` path the server uses,
//!   so it follows the `prefill_chunk` member of the config and otherwise
//!   `DEFAULT_PREFILL_CHUNK`/`BF16_DEFAULT_PREFILL_CHUNK` of `settings.rs` (256 for both
//!   activations today). The chunk path is dtype aware (`chunked_prefill_allowed` holds for F32
//!   and BF16), so a bf16 run plans and runs the same chunks as an f32 run; only a budget that
//!   keeps every chunk out leaves the serving path token by token with the `NO_BUDGET_PIECE`
//!   piece.
//! - M2 `m2_chunk_16`: the same with a largest chunk of 16, in both activations.
//! - M3 `m3_token_by_token`: no chunked prefill (no planned chunk): every prompt token is a
//!   one-token step through the resident decode path, as the engine runs without a prefill budget.
//! - M4 `m4_host_reference`: the Q36-D host forward of the crate, only with `Q36_HOST_REF=1`.
//!   `Q36_HOST_REF` accepts unset, 0, or 1 (an earlier defect rejected a set `0`).
//!   It is skipped by default: 922 CPU forward steps are not known to finish in reasonable time
//!   on the Mac.
//! - M5 (chunk 256 with the MK1 gated delta kernel disabled) does not exist: no switch in the
//!   crate disables the kernel of decision 0050. `resident/delta.rs` (the decode step) and
//!   `resident/prefill/delta.rs` (the prefill chunk) call `MlxKernels::gated_delta`
//!   unconditionally, so the fallback cannot be selected.
//!
//! The modes differ only by the chunk setting: each loads one checkpoint the way the server does
//! (`real::open` plus `HostWeights::load`, `MlxKernels::set_cache_limit`, the product constructor
//! of `scale_knob::new_decoder`, and the session cap that `Q36Executor::from_parts` sets) and ends
//! with one `StepMode::Logits` step over the pending token. The step side of `Q36Executor` is
//! private to `resident::executor`, so the harness mirrors `Q36Step::feed`, `feed_rows`, and
//! `logits` line for line and drives the real `DecodeSession` with the worker's piece split
//! (`sched/worker.rs::start` resets the session, `prefill_chunks` feeds pieces of the planned
//! chunk, and `decode` runs the logits step of the pending token).
//!
//! The activation comes from `Q36_ACTIVATION` through the product constructor: `bf16` keeps the
//! stored BF16 scales and the converted norms, `f32` keeps the G0 widening. The variable is
//! REQUIRED here, in the bank run below, and in `decode_timing` — one rule for the whole chain:
//! an unset run panics before the load with a message that names the variable and its two
//! accepted values, `bf16` and `f32`. The F32 control must equal the recorded results: "\\n"
//! 0.418 (m1, chunk 256), 0.490 (m2, chunk 16), 0.490 (m3, token by token); a bf16 run plans the
//! same chunks today (the serving chunk 256 in m1, 16 in m2) and its rows are read from the
//! report.
//!
//! Output in `Q36_OUT_DIR` (which must not exist yet): `report.json` and `logits-<mode>.f32`, the
//! full next-token row of each resident mode as little-endian f32. The report holds, per mode, the
//! top 20 ids with decoded strings, temperature-1.0 probabilities, and raw logits, the probability
//! of the format token (`FORMAT_TOKEN_ID`, the parameter of the gate, default 198: the token id
//! after the 922-token prompt), the difference against the raw logits of `REF_JSON` where the
//! reference lists the id, the max absolute difference and the cosine of the full row against the
//! M3 row, the elapsed seconds, `fed_row_runs` — the row counts each call fed, in order, the
//! output of `within_mode::run_pieces` (the pending-token shift of `DecodeSession::prefill`, the
//! chunk split of a long run, and the final one-token logits step included; the counts sum to
//! the prompt length) — and, for the chunk-shaped modes, `requested_chunk` and
//! `effective_chunk` (the planner admits the largest candidate of at most the request, so
//! `x_chunk_100` runs 64) with `fallback: "token_by_token"` when the plan admits none.
//!
//! With `Q36_FORMAT_GATE=<probability>` the test asserts, as its last step, that the serving mode
//! (M1) gives P(format token) at or above the gate, panicking with the numbers otherwise. The
//! panic marks the shared native owner failed, so run this test in its own process with
//! `--exact`.
//!
//! ```text
//! Q36_CKPT_DIR=<CKPT_DIR> \
//! IDS_FILE=<IDS_FILE> \
//! REF_JSON=<REF_JSON> \
//! TOKENIZER_DIR=<CKPT_DIR> \
//! Q36_OUT_DIR=<OUT_DIR> \
//! Q36_ACTIVATION=f32 \
//! Q36_HOST_REF=0 \
//! cargo test --release --features native-mlx,native-fault-qualification --lib \
//!   family::qwen36::resident::tool_forward -- --ignored --nocapture
//! ```
//!
//! The bf16 gate run is the same command with a fresh `Q36_OUT_DIR`, `Q36_ACTIVATION=bf16`, and
//! `Q36_FORMAT_GATE=0.95`; the f32 control run above sets no gate:
//!
//! ```text
//! Q36_CKPT_DIR=<CKPT_DIR> \
//! IDS_FILE=<IDS_FILE> \
//! REF_JSON=<REF_JSON> \
//! TOKENIZER_DIR=<CKPT_DIR> \
//! Q36_OUT_DIR=<OUT_DIR> \
//! Q36_HOST_REF=0 \
//! Q36_ACTIVATION=bf16 \
//! Q36_FORMAT_GATE=0.95 \
//! cargo test --release --features native-mlx,native-fault-qualification --lib \
//!   family::qwen36::resident::tool_forward -- --ignored --exact --nocapture
//! ```
//!
//! The ids of `IDS_FILE` come from the field `ids` (the fifi render dump), and the 3 suffix ids
//! come from the field `suffix_ids` of `REF_JSON`; the lengths are checked against `prompt_tokens`
//! and `total_tokens` of `REF_JSON`.
//!
//! The second ignored test, `bank_format_token_by_config` (packet BF16-EVAL), runs the same
//! measurement over a BANK of prompts instead of the one 922-token prompt: it reads
//! `Q36_BANK_FILE`, a JSON bank file of rendered prompts (one `ids` list per prompt, built
//! outside this repository) with the schema
//! `{"prompts": [{"ids": [<token ids>], "source": "<file name>", "index": <number>}, ...]}`
//! — every `ids` list ends with the 3 suffix ids 248069, 271, 248058 — and runs every
//! configuration of `Q36_BANK_CONFIGS` (a comma list of `steps`, `chunkN`, `tailN`; the default
//! `steps,chunk256,chunk64,tail8,tail20,tail40,tail60,tail100,tail150,tail300`) on every prompt
//! from a fresh session, writing `bank-fifi.json` into `Q36_OUT_DIR` (created after every
//! input is validated, right before the load): per prompt and config the probability of the
//! format token 198 at temperature 1,
//! the top-1 id and whether it IS the format token, the format-token logit, the seconds, and the
//! `fed_row_runs` of the config. `steps` feeds every prompt token as its own one-token step;
//! `chunkN` runs the serving prefill with the chunk the planner admits for N (the largest
//! candidate of at most N); `tailT` feeds the first N - T ids in head chunks of the serving
//! chunk plus the last T ids as one piece. `within_mode::run_pieces` is the authority for the
//! rows each config feeds: the pending-token shift of `DecodeSession::prefill` (the first head
//! chunk feeds 255 rows at a 256 chunk, then 256, 256, ...), the split of a fed run at the
//! chunk (a tail longer than the serving chunk is NOT one call: `tail300` at chunk 256 feeds
//! 256 + 44 rows), and the final one-token logits step of the last pending id. Every chunk row
//! records `requested_chunk` and `effective_chunk` (null when none is admitted) with
//! `fallback: "token_by_token"` when the plan admits none; a tail at least as long as the
//! prompt, or any chunk or tail config when the plan admits no serving chunk at all, is skipped
//! and recorded as `null`. The checkpoint loads once (`Q36_CKPT_DIR`), the activation dtype
//! comes from `Q36_ACTIVATION` as above, and the per-prompt state frees before the next prompt.
//! The exact Mac commands:
//!
//! ```text
//! Q36_CKPT_DIR=<CKPT_DIR> \
//! Q36_BANK_FILE=<BANK_FILE> \
//! Q36_OUT_DIR=<OUT_DIR> \
//! Q36_ACTIVATION=bf16 \
//! cargo test --release --features native-mlx,native-fault-qualification --lib \
//!   family::qwen36::resident::tool_forward::bank_format_token_by_config -- --ignored --exact --nocapture
//! ```
//!
//! The F32 control is the same command with a fresh `Q36_OUT_DIR` and `Q36_ACTIVATION=f32`.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::Instant,
};

use super::chunk_plan::{ChunkCost, PrefillTerms, admitted_chunk};
use super::decode_timing::required_activation_from_env;
use super::decoder::{ResidentDecoder, StepMode};
use super::parity::{CONFIG_SHA256, cosine, sha256};
use super::real::{RealCheckpoint, env_path, open, rss_kb};
use super::scale_knob::new_decoder;
use super::state::ResidentState;
use super::within_mode::{
    BankConfig, bank_config_name, default_bank_configs, parse_bank_configs, run_pieces,
};
use crate::{
    EngError,
    family::{
        qwen36::{
            host::{HostState, step as host_step},
            host_cpu::CpuProjector,
            host_load::HostWeights,
            settings::{
                Activation, Backend, Q36Env, Q36Options, Q36Settings, activation_name,
                chunked_prefill_allowed,
            },
        },
        session::{DecodeSession, TokenStepper},
        spec::QWEN3_6_35B_A3B,
    },
    kernels::MlxKernels,
    sched::SessionBudget,
    tokenizer::Tokenizer,
};

type Result<T> = std::result::Result<T, EngError>;

/// `prefill_budget_bytes` of the shipped config `engine/config/qwen36-mac-studio-36gb.json`.
/// Shared with `decode_timing`.
pub(super) const PREFILL_BUDGET: u64 = 32_212_254_720;
/// `prefill_session_reserve_bytes` of the shipped config. Shared with `decode_timing`.
pub(super) const PREFILL_RESERVE: u64 = 8_286_699_520;
/// `cache_limit_bytes` of the shipped config: the MLX allocator cache limit. Shared with
/// `decode_timing`.
pub(super) const CACHE_LIMIT: u64 = 4_294_967_296;
/// `session_cap_bytes` of the shipped config (4 GiB, the default of `settings.rs` on the MLX
/// arm). Shared with `decode_timing`.
pub(super) const SESSION_CAP: u64 = 4 << 30;
/// The context limit of the shipped config (262,144, the default of `settings.rs` on the MLX
/// arm). Shared with `decode_timing`.
pub(super) const CONTEXT_LIMIT: u32 = 262_144;
/// The largest chunk of M2.
const SMALL_CHUNK: u32 = 16;
/// The worker piece without a plan (16, the engine's `NO_BUDGET_PIECE`): M3's piece, and M1's
/// piece in a bf16 run where the plan admits no chunk. Shared with `decode_timing`.
pub(super) const NO_BUDGET_PIECE: usize = 16;
/// The ranked ids per mode in the report.
const TOP_K: usize = 20;
/// The ids of the whole tool prompt of the finding: 919 render ids plus the 3 suffix ids.
/// Shared with `decode_timing`.
pub(super) const TOTAL_IDS: usize = 922;
/// The format token of the finding: the token id 198 after the 922-token
/// tool prompt. The parameter of the `format_token` report rows and of the `Q36_FORMAT_GATE`
/// assert; the default is the captured id and every gate asserts on it. Shared with
/// `long_context`.
pub(super) const FORMAT_TOKEN_ID: u32 = 198;

/// The required `Q36_OUT_DIR`. Panics when it is unset or when the path exists, before the long run.
/// Shared with `decode_timing`.
pub(super) fn out_dir_must_be_new() -> String {
    let out = std::env::var("Q36_OUT_DIR").unwrap_or_else(|_| panic!("set Q36_OUT_DIR"));
    assert!(
        std::fs::symlink_metadata(&out).is_err(),
        "Q36_OUT_DIR must not exist yet: {out}"
    );
    out
}

/// Creates `out` after every input of the run is validated, so a configuration error leaves
/// no empty directory behind that blocks the rerun, and a missing parent or a write error
/// stops the run before the load, not after hours of measurement. Shared with
/// `decode_timing`.
pub(super) fn create_out_dir(out: &str) -> PathBuf {
    std::fs::create_dir(out).unwrap_or_else(|error| panic!("create Q36_OUT_DIR {out}: {error}"));
    Path::new(out).to_path_buf()
}

/// The largest serving chunk of the run, read through the same `Q36Settings` path the server
/// uses: the `qwen36` object of the shipped config
/// `engine/config/qwen36-mac-studio-36gb.json` parsed as `Q36Options`, with the run's
/// `Q36_ACTIVATION` in place of the config's `activation_dtype` member (the lever this harness
/// varies), through `Q36Settings::new`, whose `prefill_chunk` is the config member when the
/// config sets one and otherwise `DEFAULT_PREFILL_CHUNK` (f32) or `BF16_DEFAULT_PREFILL_CHUNK`
/// (bf16) of `settings.rs`. No copied constant remains, so a re-pin of either default keeps the
/// `serving` label of M1 true. Shared with `decode_timing`.
pub(super) fn serve_chunk_setting(activation: Activation) -> u32 {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("config/qwen36-mac-studio-36gb.json");
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let entry = config["engines"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["family"] == QWEN3_6_35B_A3B.name)
        .unwrap_or_else(|| {
            panic!(
                "the shipped config names no {} engine",
                QWEN3_6_35B_A3B.name
            )
        });
    let mut options: Q36Options = serde_json::from_value(entry["qwen36"].clone()).unwrap();
    options.activation_dtype = Some(activation_name(activation).to_string());
    Q36Settings::new(
        Path::new("."),
        Path::new("."),
        Some(&options),
        &Q36Env::default(),
        Backend::Mlx,
    )
    .unwrap()
    .prefill_chunk
}

/// The `u32` array at `field` of `value`, with every element checked.
fn u32_array(value: &serde_json::Value, field: &str) -> Vec<u32> {
    value[field]
        .as_array()
        .unwrap_or_else(|| panic!("the field {field} is missing or holds no array"))
        .iter()
        .map(|entry| {
            u32::try_from(
                entry
                    .as_u64()
                    .unwrap_or_else(|| panic!("the field {field} holds a non-integer id")),
            )
            .unwrap_or_else(|_| panic!("the field {field} holds an id above u32"))
        })
        .collect()
}

/// The number at `field` of `entry`.
fn f64_field(entry: &serde_json::Value, field: &str) -> f64 {
    entry[field]
        .as_f64()
        .unwrap_or_else(|| panic!("the field {field} is missing or holds no number"))
}

/// The whole tool prompt: the `ids` field of the ids file plus the `suffix_ids` field of the
/// reference, checked against `prompt_tokens` and `total_tokens`. Shared with `decode_timing`.
pub(super) fn read_ids(ids_value: &serde_json::Value, ref_value: &serde_json::Value) -> Vec<u32> {
    let ids = u32_array(ids_value, "ids");
    assert_eq!(
        ids.len(),
        ref_value["prompt_tokens"].as_u64().unwrap() as usize,
        "the ids field does not hold the prompt_tokens of the reference"
    );
    let suffix = u32_array(ref_value, "suffix_ids");
    assert_eq!(suffix.len(), 3, "the tool suffix is 3 ids");
    let mut all = ids;
    all.extend(suffix);
    assert_eq!(
        all.len(),
        ref_value["total_tokens"].as_u64().unwrap() as usize,
        "the prompt ids plus the suffix do not reach total_tokens"
    );
    assert_eq!(all.len(), TOTAL_IDS, "the captured tool prompt length");
    all
}

/// The `topk` of `REF_JSON` as id to (probability, raw logit).
fn reference_top(ref_value: &serde_json::Value) -> HashMap<u32, (f64, f64)> {
    ref_value["topk"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| {
            let id = u32::try_from(
                entry["id"]
                    .as_u64()
                    .unwrap_or_else(|| panic!("a topk entry holds no integer id")),
            )
            .unwrap();
            (id, (f64_field(entry, "prob"), f64_field(entry, "logit")))
        })
        .collect()
}

/// True when `Q36_HOST_REF` asks for the M4 host reference. Unset, `0`, and `1` are accepted
/// (an earlier defect rejected a set `0`); any other value panics.
fn host_ref_requested() -> bool {
    match std::env::var("Q36_HOST_REF") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(value) if value == "0" => false,
        Ok(value) if value == "1" => true,
        other => panic!("Q36_HOST_REF must be unset, 0, or 1, got {other:?}"),
    }
}

/// The `Q36_FORMAT_GATE` of the run: unset is no gate (a plain diagnostic run), any other value
/// is the probability that the serving mode must reach. A value that is not a probability in
/// `[0, 1]` panics before the long run.
fn format_gate() -> Option<f64> {
    match std::env::var("Q36_FORMAT_GATE") {
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => panic!("Q36_FORMAT_GATE: {error}"),
        Ok(value) => match value.parse::<f64>() {
            Ok(gate) if gate.is_finite() && (0.0..=1.0).contains(&gate) => Some(gate),
            _ => panic!("Q36_FORMAT_GATE must be a probability in [0, 1], got {value:?}"),
        },
    }
}

/// The `(logit, temperature-1.0 probability)` of `FORMAT_TOKEN_ID` in the full logits row, with
/// the head width checked first: the parameter is assert-able, and a head that does not cover it
/// fails with its own message instead of indexing past the row.
fn format_token_value(logits: &[f32]) -> (f32, f64) {
    let index = usize::try_from(FORMAT_TOKEN_ID)
        .unwrap_or_else(|_| panic!("FORMAT_TOKEN_ID exceeds the platform size"));
    let logit = *logits.get(index).unwrap_or_else(|| {
        panic!("the head row is narrower than FORMAT_TOKEN_ID {FORMAT_TOKEN_ID}")
    });
    let (max, sum) = softmax_terms(logits);
    (logit, f64::from((logit - max).exp()) / sum)
}

/// The model side of one serve-path call: a line-for-line mirror of the executor's private
/// `Q36Step` (`resident/executor.rs`), so the test drives the real `DecodeSession` and the real
/// `ResidentDecoder` exactly as the engine worker does. Shared with `decode_timing`, which
/// drives the same serving prefill.
pub(super) struct ServeStep<'a, 'c> {
    /// The decoder of the loaded executor.
    pub(super) decoder: &'a mut ResidentDecoder<'c>,
    /// The native state of the session.
    pub(super) state: &'a mut ResidentState,
    /// The planned chunk size, or `None` for one-token steps.
    pub(super) chunk: Option<u32>,
    /// The lengths of the row runs fed so far, one entry per prompt step or chunk.
    pub(super) fed: &'a mut Vec<usize>,
}

impl TokenStepper for ServeStep<'_, '_> {
    const OPERATION: &'static str = "qwen36";

    /// Feeds one prompt token. No head runs and nothing is read back.
    fn feed(&mut self, token: u32) -> Result<()> {
        self.decoder
            .step(self.state, token, StepMode::Prompt)
            .map(|_| ())?;
        self.fed.push(1);
        Ok(())
    }

    /// Feeds `tokens` as `Prompt` rows: with a planned chunk size, each run of that many rows is
    /// one chunk and a single row is a one-token step; without one, every row is a step.
    fn feed_rows(&mut self, tokens: &[u32]) -> Result<()> {
        let Some(size) = self.chunk.filter(|&size| size > 1) else {
            return tokens.iter().try_for_each(|&token| self.feed(token));
        };
        let width = usize::try_from(size)
            .map_err(|_| EngError::MemoryPlan("the chunk size exceeds the platform size".into()))?;
        for rows in tokens.chunks(width) {
            if let &[token] = rows {
                self.feed(token)?;
            } else {
                self.decoder.prefill(self.state, rows, StepMode::Prompt)?;
                self.fed.push(rows.len());
            }
        }
        Ok(())
    }

    /// Feeds `token`, runs the head, and returns the full logits row of the next token.
    fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.decoder
            .step(self.state, token, StepMode::Logits)?
            .logits
            .ok_or_else(|| EngError::Backend {
                operation: Self::OPERATION,
                message: "the logits step returned no logits".to_string(),
            })
    }

    /// The number of positions fed so far.
    fn fed(&self) -> usize {
        usize::try_from(self.state.pos()).unwrap_or(usize::MAX)
    }

    /// Clears the model state of the session.
    fn reset(&mut self) -> Result<()> {
        self.decoder.reset(self.state)
    }
}

/// One measured resident mode: the full next-token logits row and its timings.
struct ModeRun {
    /// The logits row of the step after the last prompt id (the width of the model head).
    logits: Vec<f32>,
    /// The lengths of the row runs the run fed, in order, including the final one-token logits
    /// step (the counts sum to the prompt length).
    fed_rows: Vec<usize>,
    /// The seconds of the piece loop (the chunked prefill).
    prefill_seconds: f64,
    /// The seconds of the final `StepMode::Logits` step.
    logits_seconds: f64,
}

/// One extra sweep mode of `Q36_TOOLFWD_CHUNKS` or `Q36_TOOLFWD_TAILS`: its row key, its chunk
/// label (requested against admitted), its worker piece, and the run.
struct ExtraMode {
    /// The report row key: `x_chunk_<n>` or `x_tail_<t>`.
    name: String,
    /// The requested and effective chunk of the mode.
    label: ChunkLabel,
    /// The worker piece the mode fed.
    piece: usize,
    /// The measured run.
    run: ModeRun,
}

/// One resident mode from a fresh session: reset the state, apply the chunk like
/// `Q36Executor::apply_chunk`, feed the worker pieces through the real `DecodeSession`
/// (`prefill_chunks` of `sched/worker.rs`), then take the `StepMode::Logits` step of the pending
/// token, which is what `DecodeSession::decode` does before the sampler draws.
fn serve_run(
    decoder: &mut ResidentDecoder<'_>,
    state: &mut ResidentState,
    ids: &[u32],
    piece: usize,
    chunk: Option<u32>,
) -> ModeRun {
    let pieces: Vec<&[u32]> = ids.chunks(piece).collect();
    serve_run_pieces(decoder, state, ids.len(), &pieces, chunk)
}

/// `serve_run` over explicit pieces (packet Q36-BF16-SENS): the pieces are fed in order, so a short
/// piece after full ones is a short chunk appended to a cached state, as a tool result is in a
/// multi-turn session. `total` is the number of ids in all pieces.
fn serve_run_pieces(
    decoder: &mut ResidentDecoder<'_>,
    state: &mut ResidentState,
    total: usize,
    pieces: &[&[u32]],
    chunk: Option<u32>,
) -> ModeRun {
    decoder.reset(state).unwrap();
    if let Some(rows) = chunk {
        decoder.set_chunk_rows(rows).unwrap();
    }
    let limit = usize::try_from(CONTEXT_LIMIT).unwrap();
    let mut session = DecodeSession::default();
    let mut fed = Vec::new();
    let started = Instant::now();
    let mut step = ServeStep {
        decoder,
        state,
        chunk,
        fed: &mut fed,
    };
    for rows in pieces {
        session.prefill(&mut step, rows, limit).unwrap();
    }
    assert_eq!(
        session.history().len(),
        total,
        "the session commits every prompt token"
    );
    let pending = session.pending().unwrap();
    let prefill_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let logits = step.logits(pending).unwrap();
    let logits_seconds = started.elapsed().as_secs_f64();
    assert_eq!(step.fed(), total, "positions after the logits step");
    fed.push(1); // the logits step fed the pending token: one row, always the last entry
    ModeRun {
        logits,
        fed_rows: fed,
        prefill_seconds,
        logits_seconds,
    }
}

/// The expected `fed_rows` of one run: `within_mode::run_pieces` over the run's shape. The
/// observed rows must equal it, so the pending-token shift of `DecodeSession::prefill`
/// and the chunk split of `ServeStep::feed_rows` cannot drift from the definition the report
/// prints.
fn expected_row_runs(ids_len: usize, config: BankConfig, chunk: Option<u32>) -> Vec<usize> {
    run_pieces(
        ids_len,
        config,
        chunk.map(|rows| usize::try_from(rows).unwrap_or(usize::MAX)),
    )
}

/// The chunk fields of one mode row: the requested largest chunk and the chunk the
/// plan admitted for it. A mode that runs no chunk path carries `None` in both.
#[derive(Clone, Copy)]
struct ChunkLabel {
    requested: Option<u32>,
    effective: Option<u32>,
}

impl ChunkLabel {
    /// The label of a chunk-shaped mode: `requested` against the admitted `effective`.
    fn of(requested: u32, effective: Option<u32>) -> Self {
        Self {
            requested: Some(requested),
            effective,
        }
    }

    /// The label of a mode that requests no chunk.
    fn none() -> Self {
        Self {
            requested: None,
            effective: None,
        }
    }
}

/// The Q36-D host reference over `ids`: a second weight load (the resident import consumed the
/// first), one CPU step per id, and the logits row of the step after the last id.
fn host_reference_run(real: &RealCheckpoint, ids: &[u32]) -> (Vec<f32>, f64, f64) {
    let started = Instant::now();
    let host = HostWeights::load(&real.ckpt, &QWEN3_6_35B_A3B, &real.config).unwrap();
    let load_seconds = started.elapsed().as_secs_f64();
    let (shape, model) = (host.shape(), host.view());
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let mut projector = CpuProjector::new(&real.ckpt, threads);
    let mut state = HostState::new(shape);
    let started = Instant::now();
    let (&last, head) = ids.split_last().unwrap();
    for &token in head {
        host_step(&mut projector, shape, &model, &mut state, token, false).unwrap();
    }
    let logits = host_step(&mut projector, shape, &model, &mut state, last, false)
        .unwrap()
        .logits;
    (logits, load_seconds, started.elapsed().as_secs_f64())
}

/// The `TOP_K` largest logits as `(id, logit)`, the largest first, the lowest id first on ties.
fn top_k(logits: &[f32]) -> Vec<(u32, f32)> {
    let mut ranked: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, *v))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked.truncate(TOP_K);
    ranked
}

/// The temperature-1.0 softmax terms of the full row: its maximum and its normalizer.
fn softmax_terms(logits: &[f32]) -> (f32, f64) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&v| f64::from((v - max).exp())).sum();
    (max, sum)
}

/// The largest absolute difference of two equal-length rows, accumulated in `f64`.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x - y).abs())
        .fold(0.0, f64::max)
}

/// The report row of one mode: the top 20 ids with decoded strings, temperature-1.0 probabilities,
/// and raw logits, the probability of the format token (`FORMAT_TOKEN_ID`), the difference against
/// the raw logits of the reference where it lists the id, and the max absolute difference and
/// cosine of the full row against the M3 row. The caller adds the `activation_dtype` label of the
/// run; a chunk-shaped mode carries its requested and effective chunk in `label`, and a
/// requested chunk the plan cannot admit is marked `fallback: "token_by_token"`.
fn mode_row(
    name: &str,
    label: ChunkLabel,
    piece: usize,
    run: &ModeRun,
    m3: &[f32],
    tokenizer: &Tokenizer,
    reference: &HashMap<u32, (f64, f64)>,
) -> serde_json::Value {
    let (max, sum) = softmax_terms(&run.logits);
    let top20: Vec<serde_json::Value> = top_k(&run.logits)
        .into_iter()
        .map(|(id, logit)| {
            let prob = f64::from((logit - max).exp()) / sum;
            let want = reference.get(&id);
            serde_json::json!({
                "id": id,
                "token": tokenizer.decode(&[id]).ok(),
                "prob": prob,
                "logit": logit,
                "ref_prob": want.map(|&(prob, _)| prob),
                "ref_logit": want.map(|&(_, ref_logit)| ref_logit),
                "logit_minus_ref": want.map(|&(_, ref_logit)| f64::from(logit) - ref_logit),
            })
        })
        .collect();
    let (format_logit, format_prob) = format_token_value(&run.logits);
    let format_ref = reference.get(&FORMAT_TOKEN_ID);
    let mut row = serde_json::json!({
        "mode": name,
        "chunk": label.effective,
        "requested_chunk": label.requested,
        "effective_chunk": label.effective,
        "piece": piece,
        "temperature": 1.0,
        "fed_row_runs": run.fed_rows,
        "prefill_seconds": run.prefill_seconds,
        "logits_step_seconds": run.logits_seconds,
        "elapsed_seconds": run.prefill_seconds + run.logits_seconds,
        "head_width": run.logits.len(),
        "top20": top20,
        "format_token": {
            "id": FORMAT_TOKEN_ID,
            "prob": format_prob,
            "logit": format_logit,
            "ref_prob": format_ref.map(|&(prob, _)| prob),
            "ref_logit": format_ref.map(|&(_, ref_logit)| ref_logit),
            "logit_minus_ref": format_ref.map(|&(_, ref_logit)| f64::from(format_logit) - ref_logit),
        },
        "vs_m3": {
            "max_abs_logit_diff": max_abs_diff(&run.logits, m3),
            "cosine": cosine(&run.logits, m3),
        },
    });
    if label.requested.is_some() && label.effective.is_none() {
        row["fallback"] = serde_json::json!("token_by_token");
    }
    row
}

#[test]
#[ignore = "needs the real checkpoint on the Mac; run with --ignored"]
fn next_token_logits_by_forward_path_on_the_922_token_tool_prompt() {
    let _serial = crate::native_test_lock();
    // Every input is validated first — the activation, the paths, the format gate, the ids
    // reads, and the tokenizer — and the output directory is created LAST, right before the
    // load, so a configuration error leaves no empty directory that blocks the rerun.
    let activation = required_activation_from_env();
    let (ckpt_dir, ids_path, ref_path, tok_dir) = (
        env_path("Q36_CKPT_DIR"),
        env_path("IDS_FILE"),
        env_path("REF_JSON"),
        env_path("TOKENIZER_DIR"),
    );
    let gate = format_gate();
    let ids_value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&ids_path).unwrap()).unwrap();
    let ref_value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&ref_path).unwrap()).unwrap();
    let ids = read_ids(&ids_value, &ref_value);
    let reference = reference_top(&ref_value);
    let tokenizer =
        Tokenizer::from_bytes(&std::fs::read(tok_dir.join("tokenizer.json")).unwrap()).unwrap();
    let out = out_dir_must_be_new();
    let out_dir = create_out_dir(&out);

    // The load of the server: the pinned checkpoint, the shipped cache limit, the product
    // constructor of the `Q36_ACTIVATION` lever, and the session cap that `Q36Executor::from_parts`
    // sets.
    let real = open(&ckpt_dir);
    assert_eq!(
        sha256(&real.config),
        CONFIG_SHA256,
        "config.json differs from the pinned checkpoint"
    );
    let mut kernels = MlxKernels::try_new().unwrap();
    kernels.set_cache_limit(CACHE_LIMIT).unwrap();
    let started = Instant::now();
    let host = HostWeights::load(&real.ckpt, &QWEN3_6_35B_A3B, &real.config).unwrap();
    let heads_q = u64::try_from(host.shape().attn.heads_q).unwrap();
    let (mut decoder, act) = new_decoder(kernels, &real.ckpt, host, CONTEXT_LIMIT);
    assert_eq!(
        act, activation,
        "the product constructor built the activation Q36_ACTIVATION names"
    );
    decoder.budget = SessionBudget::new(SESSION_CAP);
    let act_name = activation_name(act);
    let serve_setting = serve_chunk_setting(act);
    let mut state = decoder.zero_state().unwrap();
    let load_seconds = started.elapsed().as_secs_f64();

    // The admission of the shipped config (`Q36Executor::plan_prefill` without an executor):
    // both activation dtypes plan a chunk (`chunked_prefill_allowed`), the largest chunk comes
    // from the settings path (`serve_chunk_setting`), and only the budget can keep a chunk out.
    let terms = PrefillTerms {
        budget: PREFILL_BUDGET,
        weights: decoder.weights.byte_len().unwrap(),
        session_reserve: PREFILL_RESERVE,
    };
    let cost = ChunkCost::new(heads_q, CONTEXT_LIMIT).unwrap();
    let chunked = chunked_prefill_allowed(decoder.weights.consts.act);
    let (m1_chunk, m2_chunk) = if chunked {
        (
            admitted_chunk(&cost, terms.remaining(), serve_setting).unwrap(),
            admitted_chunk(&cost, terms.remaining(), SMALL_CHUNK).unwrap(),
        )
    } else {
        (None, None)
    };
    println!(
        "q36 tool-forward plan: chunked prefill {chunked}, serve chunk setting {serve_setting}, \
         m1 chunk {m1_chunk:?}, m2 chunk {m2_chunk:?}; {}",
        terms.to_string().lines().collect::<Vec<_>>().join(" ")
    );

    // M1, M2, M3: fresh sessions of one decoder, differing only by the chunk setting. M1 is the
    // serving path as the executor plans it: the planned chunk, or one-token steps with the
    // NO_BUDGET_PIECE piece where the plan admits no chunk. M2 is chunk 16 and M3 token by
    // token, in both activations. The fed rows must equal `run_pieces`, the pure authority the
    // report prints.
    let m1_piece = m1_chunk
        .map(|rows| usize::try_from(rows).unwrap())
        .unwrap_or(NO_BUDGET_PIECE);
    let m2_piece = m2_chunk
        .map(|rows| usize::try_from(rows).unwrap())
        .unwrap_or(NO_BUDGET_PIECE);
    let m1 = serve_run(&mut decoder, &mut state, &ids, m1_piece, m1_chunk);
    assert_eq!(
        m1.fed_rows,
        expected_row_runs(ids.len(), BankConfig::Chunk(serve_setting), m1_chunk),
        "m1 fed rows"
    );
    let m2 = if chunked {
        let run = serve_run(&mut decoder, &mut state, &ids, m2_piece, m2_chunk);
        assert_eq!(
            run.fed_rows,
            expected_row_runs(ids.len(), BankConfig::Chunk(SMALL_CHUNK), m2_chunk),
            "m2 fed rows"
        );
        Some(run)
    } else {
        None
    };
    let m3 = serve_run(&mut decoder, &mut state, &ids, NO_BUDGET_PIECE, None);
    assert_eq!(
        m3.fed_rows,
        expected_row_runs(ids.len(), BankConfig::Steps, None),
        "m3 fed rows"
    );
    // The memory, the budget charge, and the RSS are read right after M3, the last resident
    // mode, before the extra sweep modes can move them; the routing of the report is read from
    // the decoder, not asserted in a literal string.
    let stats = decoder.kernels.memory_stats().unwrap();
    let kv_charge_bytes = decoder.budget.used();
    let rss = rss_kb();
    let routing = format!("{:?}", decoder.routing);
    // Q36-BF16-SENS: more chunk sizes (`Q36_TOOLFWD_CHUNKS`, a comma list from 2 to 512, default
    // none) and appended tails (`Q36_TOOLFWD_TAILS`, a comma list, default none: the first
    // `N - t` ids in chunks of the serving chunk, then the last `t` ids as ONE short piece).
    let list = |name: &str, low: usize, high: usize| -> Vec<usize> {
        std::env::var(name)
            .ok()
            .map(|text| {
                text.split(',')
                    .map(|part| {
                        let n: usize = part
                            .trim()
                            .parse()
                            .unwrap_or_else(|_| panic!("{name} must be a comma list of numbers"));
                        assert!(
                            (low..=high).contains(&n),
                            "{name}: {n} is not from {low} to {high}"
                        );
                        n
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut extra_modes: Vec<ExtraMode> = Vec::new();
    if chunked {
        for n in list("Q36_TOOLFWD_CHUNKS", 2, 512) {
            let requested = u32::try_from(n).unwrap();
            let planned = admitted_chunk(&cost, terms.remaining(), requested).unwrap();
            let piece = planned
                .map(|rows| usize::try_from(rows).unwrap())
                .unwrap_or(NO_BUDGET_PIECE);
            let run = serve_run(&mut decoder, &mut state, &ids, piece, planned);
            assert_eq!(
                run.fed_rows,
                expected_row_runs(ids.len(), BankConfig::Chunk(requested), planned),
                "x_chunk_{n} fed rows"
            );
            extra_modes.push(ExtraMode {
                name: format!("x_chunk_{n}"),
                label: ChunkLabel::of(requested, planned),
                piece,
                run,
            });
        }
        for tail in list("Q36_TOOLFWD_TAILS", 1, 900) {
            let head = ids.len() - tail;
            let mut pieces: Vec<&[u32]> = ids[..head].chunks(m1_piece).collect();
            pieces.push(&ids[head..]);
            let run = serve_run_pieces(&mut decoder, &mut state, ids.len(), &pieces, m1_chunk);
            assert_eq!(
                run.fed_rows,
                expected_row_runs(
                    ids.len(),
                    BankConfig::Tail(u32::try_from(tail).unwrap()),
                    m1_chunk
                ),
                "x_tail_{tail} fed rows"
            );
            extra_modes.push(ExtraMode {
                name: format!("x_tail_{tail}"),
                label: ChunkLabel::of(serve_setting, m1_chunk),
                piece: m1_piece,
                run,
            });
        }
    }
    for ExtraMode { name, run, .. } in &extra_modes {
        let (_, format_prob) = format_token_value(&run.logits);
        let (max, sum) = softmax_terms(&run.logits);
        let (id, logit) = top_k(&run.logits)[0];
        let prob = f64::from((logit - max).exp()) / sum;
        println!(
            "{name}: top1 id {id} prob {prob:.6} logit {logit:.4} P(format token) {format_prob:.6}"
        );
    }
    for (name, run) in [("m1", Some(&m1)), ("m2", m2.as_ref()), ("m3", Some(&m3))]
        .into_iter()
        .filter_map(|(name, run)| run.map(|run| (name, run)))
    {
        let (max, sum) = softmax_terms(&run.logits);
        let (id, logit) = top_k(&run.logits)[0];
        let prob = f64::from((logit - max).exp()) / sum;
        let token = tokenizer.decode(&[id]).ok().unwrap_or_default();
        let (_, format_prob) = format_token_value(&run.logits);
        println!(
            "{name}: top1 id {id} {token:?} prob {prob:.6} logit {logit:.4} \
             P(format token) {format_prob:.6} (prefill {:.1} s, logits step {:.1} s)",
            run.prefill_seconds, run.logits_seconds
        );
    }

    // M4 only on request: the resident weights drop first, so the host load fits in memory.
    let m4 = if host_ref_requested() {
        drop(state);
        drop(decoder);
        let (logits, load_seconds, forward_seconds) = host_reference_run(&real, &ids);
        Some((logits, load_seconds, forward_seconds))
    } else {
        None
    };

    let mut modes = serde_json::Map::new();
    modes.insert("m1_serving_path".into(), {
        let mut row = mode_row(
            "m1_serving_path",
            ChunkLabel::of(serve_setting, m1_chunk),
            m1_piece,
            &m1,
            &m3.logits,
            &tokenizer,
            &reference,
        );
        row["activation_dtype"] = serde_json::json!(act_name);
        row
    });
    modes.insert(
        "m2_chunk_16".into(),
        match &m2 {
            Some(run) => {
                let mut row = mode_row(
                    "m2_chunk_16",
                    ChunkLabel::of(SMALL_CHUNK, m2_chunk),
                    m2_piece,
                    run,
                    &m3.logits,
                    &tokenizer,
                    &reference,
                );
                row["activation_dtype"] = serde_json::json!(act_name);
                row
            }
            None => serde_json::json!({
                "mode": "m2_chunk_16",
                "activation_dtype": act_name,
                "skipped": "the plan admits no chunk: chunked_prefill_allowed is false for this \
                            activation, so every prompt token is a one-token step",
            }),
        },
    );
    modes.insert("m3_token_by_token".into(), {
        let mut row = mode_row(
            "m3_token_by_token",
            ChunkLabel::none(),
            NO_BUDGET_PIECE,
            &m3,
            &m3.logits,
            &tokenizer,
            &reference,
        );
        row["activation_dtype"] = serde_json::json!(act_name);
        row
    });
    let mut format_by_mode = Vec::new();
    for (name, run) in [("m1_serving_path", &m1), ("m3_token_by_token", &m3)] {
        let (logit, prob) = format_token_value(&run.logits);
        format_by_mode.push(serde_json::json!({"mode": name, "prob": prob, "logit": logit}));
    }
    if let Some(run) = &m2 {
        let (logit, prob) = format_token_value(&run.logits);
        format_by_mode
            .push(serde_json::json!({"mode": "m2_chunk_16", "prob": prob, "logit": logit}));
    }
    for ExtraMode {
        name,
        label,
        piece,
        run,
    } in &extra_modes
    {
        let mut row = mode_row(
            name, *label, *piece, run, &m3.logits, &tokenizer, &reference,
        );
        row["activation_dtype"] = serde_json::json!(act_name);
        let (logit, prob) = format_token_value(&run.logits);
        format_by_mode.push(serde_json::json!({"mode": name, "prob": prob, "logit": logit}));
        modes.insert(name.clone(), row);
    }
    let m4_row = match &m4 {
        Some((logits, load_seconds, forward_seconds)) => {
            let run = ModeRun {
                logits: logits.clone(),
                fed_rows: Vec::new(),
                prefill_seconds: *forward_seconds,
                logits_seconds: 0.0,
            };
            let mut row = mode_row(
                "m4_host_reference",
                ChunkLabel::none(),
                1,
                &run,
                &m3.logits,
                &tokenizer,
                &reference,
            );
            row["activation_dtype"] = serde_json::json!(act_name);
            row["host_load_seconds"] = serde_json::json!(load_seconds);
            row["note"] =
                serde_json::json!("one CPU step per id; prefill_seconds holds the forward seconds");
            // The host reference runs no decoder prefill (one CPU step per id on the host
            // path), so the row OMITS `fed_row_runs`: an empty list would read as “nothing
            // fed” while the mode did feed every id — one host step at a time.
            row.as_object_mut().unwrap().remove("fed_row_runs");
            row
        }
        None => serde_json::json!({
            "skipped": "Q36_HOST_REF is not 1; 922 host forward steps are not known to finish in reasonable time"
        }),
    };
    modes.insert("m4_host_reference".into(), m4_row);

    let report = serde_json::json!({
        "ids": {
            "ids_file": ids_path.display().to_string(),
            "ref_json": ref_path.display().to_string(),
            "prompt_tokens": ids.len() - 3,
            "suffix_ids": &ids[ids.len() - 3..],
            "total_tokens": ids.len(),
        },
        "checkpoint": ckpt_dir.display().to_string(),
        "config_sha256": sha256(&real.config),
        "tokenizer_dir": tok_dir.display().to_string(),
        "shipped_config": {
            "prefill_budget_bytes": PREFILL_BUDGET,
            "prefill_session_reserve_bytes": PREFILL_RESERVE,
            "cache_limit_bytes": CACHE_LIMIT,
            "session_cap_bytes": SESSION_CAP,
            "context_limit": CONTEXT_LIMIT,
            "prefill_chunk_setting": serve_setting,
        },
        "plan": {
            "weights_bytes": terms.weights,
            "remaining_bytes": terms.remaining(),
            "chunked_prefill_allowed": chunked,
            "m1_requested_chunk": serve_setting,
            "m1_planned_chunk": m1_chunk,
            "m2_planned_chunk": m2_chunk,
            "cost_per_token_bytes": cost.per_token,
            "cost_fixed_bytes": cost.fixed,
        },
        "activation_dtype": act_name,
        "format_token_id": FORMAT_TOKEN_ID,
        "format_gate": gate,
        "serving_mode": "m1_serving_path",
        "decode_routing": {
            "decode_path": routing,
            "chunk_path": routing,
            "note": "the decoder holds one routing setting and both paths follow it",
        },
        "load_seconds": load_seconds,
        "mlx_active_after_m3": stats.active,
        "mlx_peak_after_m3": stats.peak,
        "mlx_peak_scope": "the process-wide allocator peak since the weight load, read after \
                            M3: monotone across the modes, dominated by the load — not a \
                            per-mode or per-case peak",
        "rss_kb_after_m3": rss,
        "kv_charge_bytes_after_m3": {
            "note": "read right after M3, the last resident mode, before the extra sweep modes",
            "bytes": kv_charge_bytes,
        },
        "modes": modes,
        "format_token_by_mode": format_by_mode,
        "m5_gated_delta_fallback":
            "skipped: no switch exists; resident/delta.rs and resident/prefill/delta.rs call \
             MlxKernels::gated_delta (decision 0050) unconditionally"
                .to_string(),
    });
    std::fs::write(
        out_dir.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    for (name, run) in [
        ("m1_serving_path", Some(&m1)),
        ("m2_chunk_16", m2.as_ref()),
        ("m3_token_by_token", Some(&m3)),
    ]
    .into_iter()
    .filter_map(|(name, run)| run.map(|run| (name, run)))
    {
        let bytes: Vec<u8> = run.logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out_dir.join(format!("logits-{name}.f32")), &bytes).unwrap();
    }
    if let Some((logits, _, _)) = &m4 {
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out_dir.join("logits-m4_host_reference.f32"), &bytes).unwrap();
    }
    println!("wrote {}", out_dir.join("report.json").display());

    // The format-token gate, the last step of the test: with `Q36_FORMAT_GATE` set,
    // the serving mode (M1) must give P(format token) at or above it. A miss panics with the
    // numbers, which marks the shared native owner failed; the Mac gate runs this test in its own
    // process so that the panic cannot cascade onto any other test.
    if let Some(gate) = gate {
        let (logit, prob) = format_token_value(&m1.logits);
        assert!(
            prob >= gate,
            "Q36_FORMAT_GATE {gate}: the serving mode m1_serving_path gives \
             P(format token {FORMAT_TOKEN_ID}) = {prob} (logit {logit})"
        );
        println!(
            "Q36_FORMAT_GATE {gate}: P(format token {FORMAT_TOKEN_ID}) = {prob} of \
             m1_serving_path passes"
        );
    }
}

/// The `Q36_BANK_CONFIGS` of the bank run: the parsed variable when set (any parse error
/// panics with the variable named), the default list of the eval run when unset.
fn bank_configs() -> Vec<BankConfig> {
    match std::env::var("Q36_BANK_CONFIGS") {
        Ok(text) => parse_bank_configs(&text).unwrap_or_else(|error| panic!("{error}")),
        Err(std::env::VarError::NotPresent) => default_bank_configs(),
        Err(error) => panic!("Q36_BANK_CONFIGS: {error}"),
    }
}

/// The result row of one bank (prompt, config) run (packet BF16-EVAL): the probability of the
/// format token at temperature 1, the top-1 id and whether it IS the format token, the
/// format-token logit, the seconds of the run, the `fed_row_runs` of the config, and, for a
/// chunk-shaped config, `requested_chunk` with `effective_chunk` and the `fallback` marker of a
/// request the plan cannot admit. The caller drops the `ModeRun` (its logits row)
/// right after, so the per-prompt state frees before the next prompt.
fn bank_row(run: &ModeRun, requested: Option<u32>, effective: Option<u32>) -> serde_json::Value {
    let (logit_format, p_format) = format_token_value(&run.logits);
    let (top1_id, _) = top_k(&run.logits)[0];
    let mut row = serde_json::json!({
        "p_format": p_format,
        "top1_id": top1_id,
        "top1_is_format": top1_id == FORMAT_TOKEN_ID,
        "logit_format": logit_format,
        "seconds": run.prefill_seconds + run.logits_seconds,
        "fed_row_runs": run.fed_rows,
        "requested_chunk": requested,
        "effective_chunk": effective,
    });
    if requested.is_some() && effective.is_none() {
        row["fallback"] = serde_json::json!("token_by_token");
    }
    row
}

#[test]
#[ignore = "needs the real checkpoint on the Mac; run with --ignored"]
fn bank_format_token_by_config() {
    let _serial = crate::native_test_lock();
    // Every input is validated first — the activation, the paths, the configs, and the bank
    // file — and the output directory is created LAST, right before the load, so a
    // configuration error leaves no empty directory that blocks the rerun.
    let ckpt_dir = env_path("Q36_CKPT_DIR");
    let bank_path = env_path("Q36_BANK_FILE");
    let activation = required_activation_from_env();
    let configs = bank_configs();
    let bank: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&bank_path).unwrap()).unwrap();
    let prompts = bank["prompts"]
        .as_array()
        .unwrap_or_else(|| panic!("the bank file holds no prompts array"));
    let out = out_dir_must_be_new();
    let out_dir = create_out_dir(&out);

    // The load of the server, as the first test: the pinned checkpoint, the shipped cache
    // limit, the product constructor of the `Q36_ACTIVATION` lever, the session cap, and the
    // serving-chunk admission of the shipped config.
    let real = open(&ckpt_dir);
    assert_eq!(
        sha256(&real.config),
        CONFIG_SHA256,
        "config.json differs from the pinned checkpoint"
    );
    let mut kernels = MlxKernels::try_new().unwrap();
    kernels.set_cache_limit(CACHE_LIMIT).unwrap();
    let started = Instant::now();
    let host = HostWeights::load(&real.ckpt, &QWEN3_6_35B_A3B, &real.config).unwrap();
    let heads_q = u64::try_from(host.shape().attn.heads_q).unwrap();
    let (mut decoder, act) = new_decoder(kernels, &real.ckpt, host, CONTEXT_LIMIT);
    assert_eq!(
        act, activation,
        "the product constructor built the activation Q36_ACTIVATION names"
    );
    decoder.budget = SessionBudget::new(SESSION_CAP);
    let act_name = activation_name(act);
    let mut state = decoder.zero_state().unwrap();
    let load_seconds = started.elapsed().as_secs_f64();
    let terms = PrefillTerms {
        budget: PREFILL_BUDGET,
        weights: decoder.weights.byte_len().unwrap(),
        session_reserve: PREFILL_RESERVE,
    };
    let cost = ChunkCost::new(heads_q, CONTEXT_LIMIT).unwrap();
    let chunked = chunked_prefill_allowed(decoder.weights.consts.act);
    let serve_setting = serve_chunk_setting(act);
    let serve_chunk = if chunked {
        admitted_chunk(&cost, terms.remaining(), serve_setting).unwrap()
    } else {
        None
    };
    let serve_piece = serve_chunk
        .map(|rows| usize::try_from(rows).unwrap())
        .unwrap_or(NO_BUDGET_PIECE);
    println!(
        "bank plan: chunked prefill {chunked}, serving chunk {serve_chunk:?}, {} configs, {} \
         prompts, activation {act_name}",
        configs.len(),
        prompts.len()
    );

    let mut rows = Vec::new();
    for prompt in prompts {
        let ids = u32_array(prompt, "ids");
        let source = prompt["source"]
            .as_str()
            .unwrap_or_else(|| panic!("a bank prompt holds no source file name"));
        let index = prompt["index"]
            .as_u64()
            .unwrap_or_else(|| panic!("a bank prompt holds no index"));
        assert!(
            ids.len() >= 4 && ids[ids.len() - 3..] == [248069, 271, 248058],
            "prompt {index} does not end with the three suffix ids"
        );
        let mut results = serde_json::Map::new();
        let mut line = format!("prompt {index} ({} tokens, {source}):", ids.len());
        for config in &configs {
            let name = bank_config_name(config);
            let row = match config {
                BankConfig::Steps => {
                    let run = serve_run(&mut decoder, &mut state, &ids, NO_BUDGET_PIECE, None);
                    assert_eq!(
                        run.fed_rows,
                        expected_row_runs(ids.len(), BankConfig::Steps, None),
                        "steps fed rows of prompt {index}"
                    );
                    Some(bank_row(&run, None, None))
                }
                BankConfig::Chunk(n) => match serve_chunk {
                    // No serving chunk at all: the config is skipped and recorded as null below.
                    None => None,
                    Some(_) => {
                        // The planner admits the largest candidate of at most n; a request it
                        // cannot admit runs token by token with the fallback marked.
                        let planned = admitted_chunk(&cost, terms.remaining(), *n).unwrap();
                        let piece = planned
                            .map(|rows| usize::try_from(rows).unwrap())
                            .unwrap_or(NO_BUDGET_PIECE);
                        let run = serve_run(&mut decoder, &mut state, &ids, piece, planned);
                        assert_eq!(
                            run.fed_rows,
                            expected_row_runs(ids.len(), BankConfig::Chunk(*n), planned),
                            "chunk{n} fed rows of prompt {index}"
                        );
                        Some(bank_row(&run, Some(*n), planned))
                    }
                },
                BankConfig::Tail(t) => {
                    let tail = usize::try_from(*t).unwrap_or(usize::MAX);
                    match (serve_chunk, tail < ids.len()) {
                        (Some(chunk), true) => {
                            let head = ids.len() - tail;
                            let mut pieces: Vec<&[u32]> = ids[..head].chunks(serve_piece).collect();
                            pieces.push(&ids[head..]);
                            let run = serve_run_pieces(
                                &mut decoder,
                                &mut state,
                                ids.len(),
                                &pieces,
                                Some(chunk),
                            );
                            assert_eq!(
                                run.fed_rows,
                                expected_row_runs(ids.len(), *config, serve_chunk),
                                "tail{t} fed rows of prompt {index}"
                            );
                            Some(bank_row(&run, Some(serve_setting), serve_chunk))
                        }
                        _ => None, // a tail longer than the prompt, or no serving chunk
                    }
                }
            };
            match &row {
                Some(value) => {
                    let p = value["p_format"].as_f64().unwrap();
                    let top1 = value["top1_id"].as_u64().unwrap();
                    let tag = if value["top1_is_format"].as_bool().unwrap() {
                        "fmt".to_string()
                    } else {
                        top1.to_string()
                    };
                    line.push_str(&format!(" {name}={p:.4}/{tag}"));
                }
                None => line.push_str(&format!(" {name}=null")),
            }
            results.insert(name, row.unwrap_or(serde_json::Value::Null));
        }
        println!("{line}");
        rows.push(serde_json::json!({
            "index": index,
            "n_tokens": ids.len(),
            "source": source,
            "results": results,
        }));
    }

    let report = serde_json::json!({
        "engine": "fifi",
        "bank_file": bank_path.display().to_string(),
        "checkpoint": ckpt_dir.display().to_string(),
        "config_sha256": sha256(&real.config),
        "activation_dtype": act_name,
        "format_token_id": FORMAT_TOKEN_ID,
        "configs": configs.iter().map(bank_config_name).collect::<Vec<_>>(),
        "chunked_prefill_allowed": chunked,
        "serving_chunk_requested": serve_setting,
        "serving_chunk": serve_chunk,
        "load_seconds": load_seconds,
        "prompts": rows,
    });
    std::fs::write(
        out_dir.join("bank-fifi.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("wrote {}", out_dir.join("bank-fifi.json").display());
}
