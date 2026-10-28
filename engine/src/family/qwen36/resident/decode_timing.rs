//! What the served Q36 decode path costs per token, by phase.
//!
//! The Q36 speed numbers against mlx-lm were measured in a greedy mode (`StepMode::Generate`:
//! the head runs and one token id is read back). The served path is different:
//! `DecodeSession::decode` feeds the pending token through `TokenStepper::logits`, which runs
//! `StepMode::Logits` and reads back the WHOLE logits row (vocab 248,320 F32, about 1 MB), and
//! the host `Sampler` then samples (top-k 20, temperature 1.0, top-p 0.95, a bounded heap over
//! 248K logits) before the next step can start. This test times that path per token and splits
//! it into phases, so the choice between device-side sampling and pipelining has numbers.
//!
//! One ignored Mac-only real-checkpoint test, `decode_phase_timing_by_mode`, plus the portable
//! unit tests of the pure parts below it. Inputs by environment:
//!
//! | Variable | Meaning | Default |
//! | --- | --- | --- |
//! | `Q36_CKPT_DIR` | the pinned checkpoint directory | required |
//! | `IDS_FILE` | the fifi render dump with the 919 prompt ids (`DT_IDS_SOURCE=tool` only) | required with `tool` |
//! | `REF_JSON` | the reference with the 3 `suffix_ids` and the token counts (`tool` only) | required with `tool` |
//! | `Q36_OUT_DIR` | the output directory; created after every input is validated, right before the load | required |
//! | `Q36_ACTIVATION` | the decode activation dtype | required: `bf16` or `f32`, no default |
//! | `Q36_KV_DTYPE` | the attention cache form: `f32` or `q8` | `f32` |
//! | `DT_IDS_SOURCE` | `tool` (the 922-id tool prompt) or `file` (the `Q36_PROMPT_IDS` ids) | `tool` |
//! | `Q36_PROMPT_IDS` | a file of little-endian `u32` ids, the format `route_parity` reads (`file` only) | required with `file` |
//! | `DT_TOKENS` | decode tokens per case | 64 |
//! | `DT_DEPTHS` | comma list of context depths in tokens | `16,<ids of the source>` |
//!
//! The depth is the number of prompt tokens prefilled before the timed tokens: the first N ids
//! of the ids source — the 922-token tool prompt (919 render ids plus the 3 suffix ids) with
//! `DT_IDS_SOURCE=tool`, or the ids of `Q36_PROMPT_IDS` with `DT_IDS_SOURCE=file`, the only
//! source deep enough for the depths above 922 that the Mac speed gates need (up to 8,192) —
//! prefilled with the serving chunk path exactly as `tool_forward::serve_run` feeds it (the
//! planned chunk of the shipped config, read through the `Q36Settings` path like the server, or
//! one-token steps when the plan admits no chunk); a chunk of 1 runs the token-by-token path.
//! Every depth must be at most the number of ids the source holds, and the failure message says
//! so.
//!
//! The timed tokens do NOT run at context `depth`: prefill leaves the last prompt id
//! pending, the two warm-up tokens and each timed token advance the position, so timed token `i`
//! (0-based) attends over `depth + WARMUP_TOKENS + i` tokens, its own input included. Each row
//! records `context_first` and `context_last` (with 64 timed tokens at depth 16: 18 and 81, a
//! mean context of about 50).
//!
//! Each case and depth starts from a fresh state, the same prompt, and runs 2 untimed warm-up
//! tokens in the case's own loop:
//!
//! - A `generate`: the greedy token read back on the device (`StepMode::Generate`).
//! - B `logits_argmax`: `StepMode::Logits` with no host sampling; the full row is read back and
//!   the next token is the argmax computed on the host from the row (this isolates the readback
//!   plus a host scan).
//! - C `served_sample`: the served path; `StepMode::Logits`, then `Sampler::sample` under
//!   `SamplingParams::new(1.0, 0.95, 20)` with the fixed seed `SAMPLER_SEED`, and the sampled
//!   token feeds the next step.
//!
//! Per token and case the run records, in milliseconds: the whole token wall (the step plus the
//! host token choice), the wall of `ResidentDecoder::step` (build + evaluate + readback as ONE
//! number — the decoder API hides the boundaries), the counter-timed evaluate, readback, and
//! import walls inside the step, the residual build, and the host sampling wall (the argmax scan
//! in B, the sampler in C, zero in A). It reports p50 and p90 over the timed tokens, tok/s from
//! the p50 wall, and the MLX active memory after each case plus the process-wide allocator peak
//! since load (`mlx_peak_bytes_process`: monotone across the cases of a run, dominated by the
//! weight load — NOT a per-case peak), and writes `decode_timing.json` plus one printed line per
//! case and depth into `Q36_OUT_DIR`.
//!
//! Two guarantees the phases carry: the greedy token chains of A and B must be
//! EQUAL — A reads the argmax token back on the device and B computes the argmax on the host,
//! so a difference is a defect of the measurement. Every row records `chains_equal` and the
//! first differing index, and the test asserts the equality only AFTER `decode_timing.json`
//! is written, so one divergence does not discard the rows of every earlier depth. Case B
//! runs the same finite check over the row that the sampler of case C runs before it draws
//! (`check_logits` of `sampler.rs`: non-empty and finite), so the host walls of B and C are
//! comparable.
//!
//! What the test could NOT split, and why: `ResidentDecoder::step` is one call that builds the
//! graph, evaluates it, and reads back, and no production code was added for this measurement.
//! The phase numbers come from the existing qualification counters
//! (`MlxKernels::qualification_timing`, process-wide wall sums of the synchronizing calls)
//! around wall clocks, so the split columns exist only in a build with
//! `native-fault-qualification`; without it they are `null` and the step wall alone is reported.
//! The build figure is the RESIDUAL of the step wall minus the counter-timed evaluate and
//! readback walls: it holds the host graph construction, the imports (also reported on their
//! own), and everything between the synchronizing calls. `evaluate` measures the wait for the
//! GPU as the qualification timer counts it around the synchronizing evaluate call — the host
//! wall of the call that blocks until the submitted graph finishes. It is NOT a GPU-side kernel
//! time (the GPU may finish earlier than the host observes, and without the feature nothing is
//! reported). The sampler of case C sees
//! the FULL head row (248,320 logits): this run loads no tokenizer, while the served path
//! narrows the row to the tokenizer vocabulary first (248,077 of 248,320), a negligible
//! difference for the timing. The served loop's incremental text decode is not part of any
//! phase (no tokenizer is loaded), and the sampler's two warm-up draws advance the fixed seed.
//!
//! The exact Mac command (both dtypes name `Q36_ACTIVATION`; an unset run stops with a message
//! that names the variable and its two values, the same rule as `tool_forward`):
//!
//! ```text
//! Q36_CKPT_DIR=<CKPT_DIR> \
//! IDS_FILE=<IDS_FILE> \
//! REF_JSON=<REF_JSON> \
//! Q36_OUT_DIR=<OUT_DIR> \
//! Q36_ACTIVATION=bf16 \
//! DT_TOKENS=64 \
//! DT_DEPTHS=16,922 \
//! cargo test --release --features native-mlx,native-fault-qualification --lib \
//!   family::qwen36::resident::decode_timing::decode_phase_timing_by_mode -- --ignored --exact --nocapture
//! ```
//!
//! The f32 control is the same command with `Q36_ACTIVATION=f32` and a fresh `Q36_OUT_DIR`. A
//! deep-depth run reads the prompt ids from a file instead — the little-endian `u32` format of
//! `Q36_PROMPT_IDS` — by adding `DT_IDS_SOURCE=file Q36_PROMPT_IDS=<PROMPT_IDS>
//! DT_DEPTHS=16,8192`; the file must hold at least as many ids as the largest depth.
//!
//! `QWEN36_G2A_COMPILE` turns the compiled DeltaNet and MoE decode
//! blocks on in this test build; unset is off and the run is the control. Both runs add the
//! `compile` field (the value of the switch) and the `compile_counters` field (traces, cache
//! hits, calls) to `decode_timing.json`. The off/on measurement pair at the three depths, with
//! the ids file source (placeholders only):
//!
//! ```text
//! Q36_CKPT_DIR=<CKPT_DIR> IDS_FILE=<IDS_FILE> REF_JSON=<REF_JSON> Q36_OUT_DIR=<OUT_DIR_OFF> \
//! Q36_ACTIVATION=bf16 DT_IDS_SOURCE=file Q36_PROMPT_IDS=<PROMPT_IDS> DT_DEPTHS=16,8192,30000 \
//! cargo test --release --features native-mlx,native-fault-qualification --lib \
//!   family::qwen36::resident::decode_timing::decode_phase_timing_by_mode -- --ignored --exact --nocapture
//! Q36_CKPT_DIR=<CKPT_DIR> IDS_FILE=<IDS_FILE> REF_JSON=<REF_JSON> Q36_OUT_DIR=<OUT_DIR_ON> \
//! Q36_ACTIVATION=bf16 QWEN36_G2A_COMPILE=1 DT_IDS_SOURCE=file Q36_PROMPT_IDS=<PROMPT_IDS> \
//! DT_DEPTHS=16,8192,30000 \
//! cargo test --release --features native-mlx,native-fault-qualification --lib \
//!   family::qwen36::resident::decode_timing::decode_phase_timing_by_mode -- --ignored --exact --nocapture
//! ```
//!
//! The f32 pair is the same two commands with `Q36_ACTIVATION=f32` and two more fresh output
//! directories.

use super::super::settings::{Activation, parse_activation};

/// The highest `DT_TOKENS` the run accepts: the context limit (32,768) minus the deepest prompt
/// the gates run (8,192 ids of a `Q36_PROMPT_IDS` file) leaves room for this many timed tokens
/// plus the warm-up.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
const MAX_TIMED_TOKENS: usize = 4096;
/// The untimed warm-up tokens before the timed window of every case.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
const WARMUP_TOKENS: usize = 2;
/// The seed of the case-C sampler. The executor's seeded tests use 7 too; the two warm-up draws
/// advance it before the first timed draw.
const SAMPLER_SEED: u64 = 7;

/// Parses `DT_DEPTHS`, a comma list of context depths in tokens, each from 1 to `max_depth`
/// (the number of ids the source holds). Entries are trimmed. An empty list, an entry that is
/// not a number, a depth out of range, and a duplicate depth (which would overwrite its own
/// result row) are errors naming the variable.
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
                 exceed the {max_depth} ids of the ids source)"
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

/// The nearest-rank percentile `p` (0 to 100) of `values`: the smallest value whose rank,
/// `ceil(p/100 * n)`, reaches it, so p50 of an even count takes the lower of the middle two.
/// The input needs no order. Panics on an empty slice; every caller feeds the timed window,
/// which holds at least one token.
fn percentile(values: &[f64], p: u32) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (p.min(100) as usize * sorted.len()).div_ceil(100).max(1);
    sorted[rank - 1]
}

/// The index of the largest entry of `row`: a tie goes to the lower index and a NaN never
/// replaces the running maximum, the greedy rule and tie behavior of the sampler's own argmax,
/// computed on the host for case B.
fn host_argmax(row: &[f32]) -> u32 {
    let mut best_index = 0_usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in row.iter().enumerate() {
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }
    u32::try_from(best_index).unwrap_or(u32::MAX)
}

/// How one case picks the next token after a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
enum NextToken {
    /// A `Generate` step reads the greedy token id back on the device.
    DeviceGreedy,
    /// A `Logits` step reads the whole row; the host argmax of the row is the next token.
    HostArgmax,
    /// A `Logits` step reads the whole row; the seeded `Sampler` draws the next token.
    HostSampler,
}

/// One row of the case table.
#[derive(Clone, Copy)]
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
struct Case {
    /// The report key of the case.
    key: &'static str,
    /// The `StepMode` name the case drives, for the report.
    mode_name: &'static str,
    /// How the next token is picked.
    next: NextToken,
    /// True when the case reads the whole logits row back (B and C).
    reads_row: bool,
}

/// The case table, in run order: A `generate`, B `logits_argmax`, C `served_sample`.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn case_table() -> [Case; 3] {
    [
        Case {
            key: "generate",
            mode_name: "Generate",
            next: NextToken::DeviceGreedy,
            reads_row: false,
        },
        Case {
            key: "logits_argmax",
            mode_name: "Logits",
            next: NextToken::HostArgmax,
            reads_row: true,
        },
        Case {
            key: "served_sample",
            mode_name: "Logits",
            next: NextToken::HostSampler,
            reads_row: true,
        },
    ]
}

/// The activation the run REQUIRES from the raw `Q36_ACTIVATION` text: the variable
/// must name `bf16` or `f32`, the words of `settings::parse_activation`, in this harness and in
/// `tool_forward` alike — an unset variable is an error that names the variable and the two
/// accepted values, so one command line cannot quietly mean different dtypes in the two
/// harnesses (the older rule defaulted an unset run to `bf16` here and to `f32` there).
pub(super) fn required_activation(raw: Option<&str>) -> Result<Activation, String> {
    match raw {
        None => Err("Q36_ACTIVATION is unset; set it to bf16 or f32".to_string()),
        Some(text) => parse_activation(text)
            .map_err(|_| format!("Q36_ACTIVATION must be bf16 or f32, got {text:?}")),
    }
}

/// The ids source of `DT_IDS_SOURCE` (the Mac speed gates need depths up to 8,192, which the
/// 922-id tool prompt cannot reach): `tool`, the prompt of `IDS_FILE` plus `REF_JSON`, or
/// `file`, the little-endian `u32` ids of `Q36_PROMPT_IDS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IdsSource {
    Tool,
    File,
}

/// Parses the raw `DT_IDS_SOURCE` text: unset is `tool`, and any word other than `tool` and
/// `file` is an error naming the variable and the two accepted words.
fn parse_ids_source(raw: Option<&str>) -> Result<IdsSource, String> {
    match raw {
        None => Ok(IdsSource::Tool),
        Some("tool") => Ok(IdsSource::Tool),
        Some("file") => Ok(IdsSource::File),
        Some(text) => Err(format!("DT_IDS_SOURCE must be tool or file, got {text:?}")),
    }
}

/// The ids of a `Q36_PROMPT_IDS` file: little-endian `u32` words, the format `route_parity`
/// reads. A partial trailing word is an error that says what the file must hold.
fn ids_from_le_bytes(bytes: &[u8]) -> Result<Vec<u32>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "Q36_PROMPT_IDS holds {} bytes: little-endian u32 words need a multiple of 4",
            bytes.len()
        ));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect())
}

/// Reads the ids of the REQUIRED `Q36_PROMPT_IDS` file through `ids_from_le_bytes`: the one
/// env-bound reader of the little-endian `u32` dump format, shared by the `file` ids source
/// of this harness and by `route_parity` (whose own private reader it replaces), so the
/// format keeps one definition with its unit test next to it.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(super) fn read_prompt_ids() -> Vec<u32> {
    let path = env_path("Q36_PROMPT_IDS");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    ids_from_le_bytes(&bytes).unwrap_or_else(|error| panic!("{error}"))
}

/// The row check of case B: the same pass the sampler of case C runs over the row
/// before it draws (`check_logits` of `sampler.rs`, which is private) — non-empty, no longer
/// than `u32::MAX`, and finite values only — so the host wall of B holds the same full scan of
/// the row that the sampler of C pays before its heap. A unit test ties this mirror to the
/// real `Sampler::sample`, which rejects exactly the rows this check rejects.
fn row_passes_the_sampler_check(row: &[f32]) -> bool {
    !row.is_empty() && u32::try_from(row.len()).is_ok() && row.iter().all(|v| v.is_finite())
}

/// The native cfg of the ignored run: `tool_forward`'s own gate, so the test compiles exactly
/// where the loader helpers it reuses do.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
use super::tool_forward::{
    CACHE_LIMIT, CONTEXT_LIMIT, NO_BUDGET_PIECE, PREFILL_BUDGET, PREFILL_RESERVE, SESSION_CAP,
    ServeStep, create_out_dir, out_dir_must_be_new, read_ids, serve_chunk_setting,
};

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
use {
    super::super::settings::{activation_name, chunked_prefill_allowed, kv_dtype_name},
    super::chunk_plan::{ChunkCost, PrefillTerms, admitted_chunk},
    super::decoder::{ResidentDecoder, StepMode, StepResult},
    super::parity::{CONFIG_SHA256, sha256},
    super::real::{env_path, kv_dtype_from_env, open},
    super::state::ResidentState,
    super::within_mode::{BankConfig, env_number, run_pieces},
    crate::family::{
        qwen36::host_load::HostWeights,
        sampler::{Sampler, SamplingParams},
        session::DecodeSession,
        spec::QWEN3_6_35B_A3B,
    },
    crate::{kernels::MlxKernels, sched::SessionBudget},
    std::time::Instant,
};

/// The default depth list of a run that sets no `DT_DEPTHS`: `16` and the whole ids source,
/// built as text and parsed by the same `parse_depths` validation as an explicit list, so an
/// ids source with fewer than 16 ids (the first default depth exceeds the source), a source
/// of exactly 16 ids (the two default entries duplicate), and an empty source are clear
/// errors instead of a panic or a duplicated result row.
fn default_depths(total_ids: usize) -> Result<Vec<usize>, String> {
    parse_depths(&format!("16,{total_ids}"), total_ids)
}

/// The depths of `DT_DEPTHS`, default `16,<total_ids>` through `default_depths`.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn depths_from_env(total_ids: usize) -> Vec<usize> {
    match std::env::var("DT_DEPTHS") {
        Ok(text) => parse_depths(&text, total_ids).unwrap_or_else(|error| panic!("{error}")),
        Err(std::env::VarError::NotPresent) => {
            default_depths(total_ids).unwrap_or_else(|error| panic!("{error}"))
        }
        Err(error) => panic!("DT_DEPTHS: {error}"),
    }
}

/// Reads the REQUIRED `Q36_ACTIVATION` of the run through `required_activation`: the one env
/// wrapper of this harness and of `tool_forward` (both its tests), so the harnesses cannot
/// drift.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(super) fn required_activation_from_env() -> Activation {
    let raw = match std::env::var("Q36_ACTIVATION") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => panic!("Q36_ACTIVATION: {error}"),
    };
    required_activation(raw.as_deref()).unwrap_or_else(|error| panic!("{error}"))
}

/// Reads `DT_IDS_SOURCE` through `parse_ids_source`: unset is the tool prompt.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn ids_source_from_env() -> IdsSource {
    match std::env::var("DT_IDS_SOURCE") {
        Err(std::env::VarError::NotPresent) => IdsSource::Tool,
        Ok(text) => parse_ids_source(Some(&text)).unwrap_or_else(|error| panic!("{error}")),
        Err(error) => panic!("DT_IDS_SOURCE: {error}"),
    }
}

/// A duration in milliseconds.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn ms(elapsed: std::time::Duration) -> f64 {
    elapsed.as_secs_f64() * 1000.0
}

/// The process-wide qualification counters at one instant:
/// `[import_ns, readback_ns, evaluate_ns, import_calls, readback_calls, evaluate_calls]`.
/// Qualification builds only.
/// The class-partition marker `run_case` threads through its arms: without the
/// qualification feature the module is compiled out and the marker can never be
/// `Some` (the DT_CLASS_PARTITION driver panics instead).
#[cfg(all(
    test,
    feature = "native-mlx",
    feature = "native-fault-qualification",
    target_os = "macos",
    target_arch = "aarch64"
))]
type ClassMarker<'a> = Option<&'a mut super::decode_class_partition::Partitioner>;
#[cfg(not(all(
    test,
    feature = "native-mlx",
    feature = "native-fault-qualification",
    target_os = "macos",
    target_arch = "aarch64"
)))]
type ClassMarker<'a> = Option<&'a mut ()>;

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
#[cfg(feature = "native-fault-qualification")]
fn counters(kernels: &MlxKernels) -> [u64; 6] {
    let timing = kernels.qualification_timing();
    [
        timing.import_ns,
        timing.readback_ns,
        timing.evaluate_ns,
        timing.import_calls,
        timing.readback_calls,
        timing.evaluate_calls,
    ]
}

/// One timed token's phases, in milliseconds.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
struct TokenTimes {
    /// The whole token wall: the step plus the host token choice.
    wall_ms: f64,
    /// The wall of `ResidentDecoder::step`: build + evaluate + readback as one number.
    step_ms: f64,
    /// The host token choice: the argmax scan (B), the sampler (C), or zero (A).
    host_ms: f64,
    /// The counter-timed evaluate wall inside the step. Qualification builds only.
    evaluate_ms: Option<f64>,
    /// The counter-timed readback wall inside the step. Qualification builds only.
    readback_ms: Option<f64>,
    /// The counter-timed import wall inside the step. Qualification builds only.
    import_ms: Option<f64>,
}

/// The run of one case: the per-token phases, the counter call counts per timed token, the
/// MLX memory after the case, and the row runs the prefill fed.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
struct CaseRun {
    /// One entry per timed token, in order.
    times: Vec<TokenTimes>,
    /// The evaluate calls per timed token. Qualification builds only.
    evaluate_calls: Option<f64>,
    /// The readback calls per timed token. Qualification builds only.
    readback_calls: Option<f64>,
    /// The MLX active bytes after the case.
    active: u64,
    /// The MLX peak bytes after the case (process-wide, monotone across the cases of a run).
    peak: u64,
    /// The tokens the case drew, in order (the warm-up tokens first). A and B are both greedy
    /// and must draw the same chain; C draws from the seeded sampler.
    chain: Vec<u32>,
    /// The row runs the case OBSERVED its prefill feed, in order: one entry per prompt step or
    /// chunk `ServeStep` fed, recorded by `ServeStep` itself (not derived from the plan).
    prefill_fed: Vec<usize>,
}

/// The `StepMode` of a case: `Generate` for A, `Logits` for B and C.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn step_mode(case: &Case) -> StepMode {
    match case.next {
        NextToken::DeviceGreedy => StepMode::Generate,
        NextToken::HostArgmax | NextToken::HostSampler => StepMode::Logits,
    }
}

/// The next token after the step and the host milliseconds of choosing it: the device greedy id
/// of a `Generate` step (A), the host argmax of the full row (B), or the seeded sampler draw
/// over the full row (C). The row readback itself happens inside the step, so the host wall
/// holds only the host scan or the sampler.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn host_choice(result: StepResult, case: &Case, sampler: &mut Option<Sampler>) -> (u32, f64) {
    match case.next {
        NextToken::DeviceGreedy => (result.token.unwrap(), 0.0),
        NextToken::HostArgmax => {
            let row = result.logits.unwrap();
            let started = Instant::now();
            // The same finite pass the sampler of case C runs (`check_logits` of `sampler.rs`),
            // inside the timed host wall, so B and C pay the same scan of the row.
            assert!(
                row_passes_the_sampler_check(&row),
                "case logits_argmax: the row fails the sampler's own check (empty or non-finite \
                 logits), a defect of the measurement"
            );
            let id = host_argmax(&row);
            (id, ms(started.elapsed()))
        }
        NextToken::HostSampler => {
            let row = result.logits.unwrap();
            let started = Instant::now();
            let id = sampler.as_mut().unwrap().sample(&row).unwrap();
            (id, ms(started.elapsed()))
        }
    }
}

/// Runs one token of `case`: the step, then the case's host token choice, with the phase clocks
/// and counters around them. The warm-up tokens call this too and drop the record.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn one_token(
    decoder: &mut ResidentDecoder<'_>,
    state: &mut ResidentState,
    input: u32,
    case: &Case,
    sampler: &mut Option<Sampler>,
) -> (u32, TokenTimes) {
    let token_started = Instant::now();
    #[cfg(feature = "native-fault-qualification")]
    let before = counters(&decoder.kernels);
    let step_started = Instant::now();
    let result = decoder.step(state, input, step_mode(case)).unwrap();
    let step_ms = ms(step_started.elapsed());
    #[cfg(feature = "native-fault-qualification")]
    let after = counters(&decoder.kernels);
    let (next, host_ms) = host_choice(result, case, sampler);
    let (evaluate_ms, readback_ms, import_ms) = {
        #[cfg(feature = "native-fault-qualification")]
        {
            let wall = |delta: u64| delta as f64 / 1e6;
            (
                Some(wall(after[2] - before[2])),
                Some(wall(after[1] - before[1])),
                Some(wall(after[0] - before[0])),
            )
        }
        #[cfg(not(feature = "native-fault-qualification"))]
        {
            (None, None, None)
        }
    };
    (
        next,
        TokenTimes {
            wall_ms: ms(token_started.elapsed()),
            step_ms,
            host_ms,
            evaluate_ms,
            readback_ms,
            import_ms,
        },
    )
}

/// One case at one depth, from a fresh state: the serving prefill of `prompt` (the planned
/// chunk and the worker piece, exactly as `tool_forward::serve_run` feeds it), `WARMUP_TOKENS`
/// untimed tokens of the case's own loop, then `timed` timed tokens, and the MLX memory after
/// the case. The counter call counts cover the timed window only.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
fn run_case(
    decoder: &mut ResidentDecoder<'_>,
    state: &mut ResidentState,
    prompt: &[u32],
    piece: usize,
    chunk: Option<u32>,
    case: &Case,
    timed: usize,
    mut marker: ClassMarker<'_>,
) -> CaseRun {
    decoder.reset(state).unwrap();
    // Without the feature the marker type exists but no arm reads it.
    #[cfg(not(feature = "native-fault-qualification"))]
    let _ = &mut marker;
    if let Some(rows) = chunk {
        decoder.set_chunk_rows(rows).unwrap();
    }
    let limit = usize::try_from(CONTEXT_LIMIT).unwrap();
    let mut session = DecodeSession::default();
    // The row runs the prefill OBSERVES, recorded by `ServeStep` (the harness reports this
    // list and asserts it against `within_mode::run_pieces` minus the trailing pending-token
    // step: in this harness that id is fed by the first warm-up token, not by the prefill, so
    // the observed prefill rows sum to depth - 1).
    let mut prefill_fed = Vec::new();
    {
        let mut step = ServeStep {
            decoder,
            state,
            chunk,
            fed: &mut prefill_fed,
        };
        for rows in prompt.chunks(piece) {
            session.prefill(&mut step, rows, limit).unwrap();
        }
    }
    assert_eq!(
        session.history().len(),
        prompt.len(),
        "the session commits every prompt token"
    );
    let mut input = session.pending().unwrap();
    let mut sampler = (case.next == NextToken::HostSampler)
        .then(|| Sampler::new(SamplingParams::new(1.0, 0.95, 20).unwrap(), SAMPLER_SEED));
    let mut chain = Vec::with_capacity(WARMUP_TOKENS + timed);
    for _ in 0..WARMUP_TOKENS {
        #[cfg(all(
            test,
            feature = "native-mlx",
            feature = "native-fault-qualification",
            target_os = "macos",
            target_arch = "aarch64"
        ))]
        {
            input = match marker.as_deref_mut() {
                Some(part) => {
                    super::decode_class_partition::one_marked_token(decoder, state, input, part)
                        .unwrap()
                }
                None => one_token(decoder, state, input, case, &mut sampler).0,
            };
        }
        #[cfg(not(feature = "native-fault-qualification"))]
        {
            input = one_token(decoder, state, input, case, &mut sampler).0;
        }
        chain.push(input);
    }
    #[cfg(all(
        test,
        feature = "native-mlx",
        feature = "native-fault-qualification",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    if let Some(part) = marker.as_deref_mut() {
        // The timed window starts clean: the warm-up (and the compile traces it
        // pays) stays out of the class sums.
        part.reset_sums();
    }
    #[cfg(feature = "native-fault-qualification")]
    let calls_before = counters(&decoder.kernels);
    let mut times = Vec::with_capacity(timed);
    for _ in 0..timed {
        #[cfg(all(
            test,
            feature = "native-mlx",
            feature = "native-fault-qualification",
            target_os = "macos",
            target_arch = "aarch64"
        ))]
        let (next, record) = match marker.as_deref_mut() {
            Some(part) => {
                let started = std::time::Instant::now();
                let t =
                    super::decode_class_partition::one_marked_token(decoder, state, input, part)
                        .unwrap();
                (
                    t,
                    TokenTimes {
                        wall_ms: ms(started.elapsed()),
                        step_ms: 0.0,
                        host_ms: 0.0,
                        evaluate_ms: None,
                        readback_ms: None,
                        import_ms: None,
                    },
                )
            }
            None => one_token(decoder, state, input, case, &mut sampler),
        };
        #[cfg(not(all(
            test,
            feature = "native-mlx",
            feature = "native-fault-qualification",
            target_os = "macos",
            target_arch = "aarch64"
        )))]
        let (next, record) = one_token(decoder, state, input, case, &mut sampler);
        input = next;
        chain.push(input);
        times.push(record);
    }
    #[cfg(feature = "native-fault-qualification")]
    let calls_after = counters(&decoder.kernels);
    let (evaluate_calls, readback_calls) = {
        #[cfg(feature = "native-fault-qualification")]
        {
            let per_token = u64::try_from(timed).unwrap_or(1) as f64;
            (
                Some((calls_after[5] - calls_before[5]) as f64 / per_token),
                Some((calls_after[4] - calls_before[4]) as f64 / per_token),
            )
        }
        #[cfg(not(feature = "native-fault-qualification"))]
        {
            (None, None)
        }
    };
    let memory = decoder.kernels.memory_stats().unwrap();
    CaseRun {
        times,
        evaluate_calls,
        readback_calls,
        active: memory.active,
        peak: memory.peak,
        chain,
        prefill_fed,
    }
}

/// The `p` percentile of one phase over the timed tokens, or `None` when any token lacks the
/// value (a build without the qualification counters).
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn percentile_of(
    times: &[TokenTimes],
    p: u32,
    pick: impl Fn(&TokenTimes) -> Option<f64>,
) -> Option<f64> {
    let values = times.iter().map(pick).collect::<Option<Vec<_>>>()?;
    Some(percentile(&values, p))
}

/// One report phase: its name and its per-token value picker.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
type PhasePicker<'a> = (&'a str, &'a dyn Fn(&TokenTimes) -> Option<f64>);

/// The p50 and p90 of every phase over the timed tokens, for the report row. `build` is the
/// residual of the step wall minus the counter-timed evaluate and readback walls.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn phase_percentiles(times: &[TokenTimes]) -> serde_json::Value {
    let phases: [PhasePicker<'_>; 7] = [
        ("wall", &|t: &TokenTimes| Some(t.wall_ms)),
        ("step", &|t: &TokenTimes| Some(t.step_ms)),
        ("build", &|t: &TokenTimes| {
            t.evaluate_ms
                .zip(t.readback_ms)
                .map(|(evaluate, readback)| t.step_ms - evaluate - readback)
        }),
        ("evaluate", &|t: &TokenTimes| t.evaluate_ms),
        ("readback", &|t: &TokenTimes| t.readback_ms),
        ("import", &|t: &TokenTimes| t.import_ms),
        ("host", &|t: &TokenTimes| Some(t.host_ms)),
    ];
    let mut p50 = serde_json::Map::new();
    let mut p90 = serde_json::Map::new();
    for (name, pick) in phases {
        p50.insert(
            (*name).to_string(),
            serde_json::json!(percentile_of(times, 50, pick)),
        );
        p90.insert(
            (*name).to_string(),
            serde_json::json!(percentile_of(times, 90, pick)),
        );
    }
    serde_json::json!({ "p50_ms": p50, "p90_ms": p90 })
}

/// An optional phase figure for the printed line: the number, or `n/a` in a build without the
/// qualification counters.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn ms_text(value: Option<f64>) -> String {
    value.map_or_else(|| "n/a".to_string(), |figure| format!("{figure:.2}"))
}

#[test]
#[ignore = "needs the real checkpoint on the Mac; run with --ignored"]
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn decode_phase_timing_by_mode() {
    let _serial = crate::native_test_lock();
    // Every input is validated first — the case count, the ids of the source, the depths, and
    // the required activation — and the output directory is created LAST, right before the
    // load, so a configuration error leaves no empty directory that blocks the rerun.
    let ckpt_dir = env_path("Q36_CKPT_DIR");
    let source = ids_source_from_env();
    let timed = env_number("DT_TOKENS", 64, 1, MAX_TIMED_TOKENS);
    // The ids of the source the run reads: the 922-id tool prompt by default, or the
    // little-endian u32 ids of `Q36_PROMPT_IDS` with `DT_IDS_SOURCE=file`, the only source deep
    // enough for the 8,192-token speed gates. The report records which source ran.
    let (ids, ids_report) = match source {
        IdsSource::Tool => {
            let (ids_path, ref_path) = (env_path("IDS_FILE"), env_path("REF_JSON"));
            let ids_value: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&ids_path).unwrap()).unwrap();
            let ref_value: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&ref_path).unwrap()).unwrap();
            let ids = read_ids(&ids_value, &ref_value);
            let report = serde_json::json!({
                "source": "tool",
                "ids_file": ids_path.display().to_string(),
                "ref_json": ref_path.display().to_string(),
                "prompt_tokens": ids.len() - 3,
                "total_tokens": ids.len(),
            });
            (ids, report)
        }
        IdsSource::File => {
            let ids_path = env_path("Q36_PROMPT_IDS");
            let ids = read_prompt_ids();
            let report = serde_json::json!({
                "source": "file",
                "prompt_ids_file": ids_path.display().to_string(),
                "ids_in_source": ids.len(),
            });
            (ids, report)
        }
    };
    let depths = depths_from_env(ids.len());
    let activation = required_activation_from_env();
    let out = out_dir_must_be_new();
    let out_dir = create_out_dir(&out);

    // The load of the server, as `tool_forward`'s first test: the pinned checkpoint, the shipped
    // cache limit, the session cap that `Q36Executor::from_parts` sets, and the product
    // constructor of the `Q36_ACTIVATION` lever (here with the product default, bf16).
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
    decoder.budget = SessionBudget::new(SESSION_CAP);
    // The compile counters of the run start clean: with the switch on,
    // the warm-up tokens trace each block kind once per dtype and every later call hits the
    // MLX cache, so the `compile_counters` row of the report separates the one-time compile
    // from the timed decode calls.
    decoder.kernels.reset_compile_counters();
    decoder.kernels.clear_compile_cache().unwrap();
    let mut state = decoder.zero_state().unwrap();
    // The routing of the report is read from the decoder, never asserted as a literal string:
    // one setting serves the decode steps and the chunk prefill alike.
    let routing = format!("{:?}", decoder.routing);

    // The serving-chunk admission of the shipped config (`Q36Executor::plan_prefill` without an
    // executor), as `tool_forward`'s M1: the planned chunk, or one-token steps with the
    // `NO_BUDGET_PIECE` piece when the plan admits no chunk.
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
    let piece = chunk
        .map(|rows| usize::try_from(rows).unwrap())
        .unwrap_or(NO_BUDGET_PIECE);
    println!(
        "decode-timing plan: activation {}, chunked prefill {chunked}, serving chunk {chunk:?}, \
         piece {piece}, {timed} timed tokens per case plus {WARMUP_TOKENS} warm-up, depths \
         {depths:?}; {}",
        activation_name(activation),
        terms.to_string().lines().collect::<Vec<_>>().join(" ")
    );

    let mut rows = Vec::new();
    let mut chain_checks = Vec::new();
    // The class-partition arm (DT_CLASS_PARTITION=1): after the three plain
    // cases, one marked pass per depth attributes the step by op class; its
    // chain must equal case A's (the marked replica is the same computation).
    let class_partition = std::env::var("DT_CLASS_PARTITION").is_ok_and(|v| v == "1");
    #[cfg(not(all(
        test,
        feature = "native-mlx",
        feature = "native-fault-qualification",
        target_os = "macos",
        target_arch = "aarch64"
    )))]
    if class_partition {
        panic!("DT_CLASS_PARTITION=1 needs the native-fault-qualification feature");
    }
    #[cfg(all(
        test,
        feature = "native-mlx",
        feature = "native-fault-qualification",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    let mut partitioner =
        class_partition.then(|| super::decode_class_partition::partitioner_of(&mut decoder));
    #[cfg(not(all(
        test,
        feature = "native-mlx",
        feature = "native-fault-qualification",
        target_os = "macos",
        target_arch = "aarch64"
    )))]
    let _partitioner: Option<()> = None;
    #[cfg(feature = "native-fault-qualification")]
    let mut class_rows: Vec<serde_json::Value> = Vec::new();
    #[cfg(not(feature = "native-fault-qualification"))]
    let class_rows: Vec<serde_json::Value> = Vec::new();
    #[cfg(all(
        test,
        feature = "native-mlx",
        feature = "native-fault-qualification",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    if let Some(part) = partitioner.as_mut() {
        part.calibrate(&mut decoder, 64);
    }
    for &depth in &depths {
        let prompt = &ids[..depth];
        // The prefill row runs the definition promises for this depth: `run_pieces` WITHOUT
        // its trailing one-token entry. That entry is the logits step of the last pending id,
        // which in this harness is fed by the first WARM-UP token (a decoder step, not a
        // prefill row), so the OBSERVED prefill rows sum to depth - 1: every prompt id except
        // the one `DecodeSession::prefill` holds pending.
        let expected = run_pieces(
            depth,
            BankConfig::Chunk(serve_setting),
            chunk.map(|rows| usize::try_from(rows).unwrap_or(usize::MAX)),
        );
        let expected_prefill = &expected[..expected.len() - 1];
        let mut case_runs = Vec::new();
        for case in case_table() {
            let run = run_case(
                &mut decoder,
                &mut state,
                prompt,
                piece,
                chunk,
                &case,
                timed,
                None,
            );
            assert_eq!(
                run.prefill_fed, expected_prefill,
                "depth {depth} case {}: the observed prefill row runs differ from the \
                 definition (run_pieces minus the trailing pending-token step)",
                case.key
            );
            case_runs.push((case, run));
        }
        // A reads the argmax token back on the device and B computes it on the host, so the
        // two greedy chains must be equal — a difference is a defect of the measurement, not
        // a model difference. The verdict is recorded in each row of the depth and asserted
        // only after the report is written, so one divergence does not discard the rows of
        // every earlier depth.
        let chain = |key: &str| {
            case_runs
                .iter()
                .find(|(case, _)| case.key == key)
                .map(|(_, run)| run.chain.clone())
                .unwrap_or_default()
        };
        let chain_b = chain("logits_argmax");
        let chain_a = chain("generate");
        let first_diff = chain_a
            .iter()
            .zip(&chain_b)
            .position(|(a, b)| a != b)
            .or_else(|| {
                (chain_a.len() != chain_b.len()).then_some(chain_a.len().min(chain_b.len()))
            });
        chain_checks.push((depth, first_diff));
        for (case, run) in case_runs {
            let summary = phase_percentiles(&run.times);
            let p50 = |name: &str| summary["p50_ms"][name].as_f64();
            let p90 = |name: &str| summary["p90_ms"][name].as_f64();
            let wall50 = p50("wall").unwrap();
            let tok_s = 1000.0 / wall50;
            println!(
                "depth {depth} case {}: p50 wall {:.2} ms (step {} = build {} + evaluate {} + \
                 readback {}, host {}), p90 wall {}, {:.1} tok/s, mlx active {:.2} GB peak \
                 {:.2} GB",
                case.key,
                wall50,
                ms_text(p50("step")),
                ms_text(p50("build")),
                ms_text(p50("evaluate")),
                ms_text(p50("readback")),
                ms_text(p50("host")),
                ms_text(p90("wall")),
                tok_s,
                run.active as f64 / 1e9,
                run.peak as f64 / 1e9
            );
            rows.push(serde_json::json!({
                "depth": depth,
                "case": case.key,
                "step_mode": case.mode_name,
                "timed_tokens": timed,
                "warmup_tokens": WARMUP_TOKENS,
                "context_first": depth + WARMUP_TOKENS,
                "context_last": depth + WARMUP_TOKENS + timed - 1,
                // The OBSERVED row runs of the prefill (recorded by `ServeStep`), not the
                // plan: they are `run_pieces` of this depth minus the trailing one-token
                // step and sum to depth - 1, as asserted above.
                "prefill_fed_row_runs": run.prefill_fed,
                "chains_equal": first_diff.is_none(),
                "chain_first_diff": first_diff,
                "percentiles": summary,
                "evaluate_calls_per_token": run.evaluate_calls,
                "readback_calls_per_token": run.readback_calls,
                "tok_s_from_p50_wall": tok_s,
                "mlx_active_bytes": run.active,
                "mlx_peak_bytes_process": run.peak,
            }));
        }
        #[cfg(all(
            test,
            feature = "native-mlx",
            feature = "native-fault-qualification",
            target_os = "macos",
            target_arch = "aarch64"
        ))]
        if let Some(part) = partitioner.as_mut() {
            let gen_case = case_table()[0];
            let run = run_case(
                &mut decoder,
                &mut state,
                prompt,
                piece,
                chunk,
                &gen_case,
                timed,
                Some(part),
            );
            let marked_chain = run.chain;
            assert_eq!(
                marked_chain, chain_a,
                "depth {depth}: the marked replica's greedy chain differs from the plain \
                 Generate chain — the class partition would attribute a different computation"
            );
            let wall50 = {
                let walls: Vec<f64> = run.times.iter().map(|t| t.wall_ms).collect();
                percentile(&walls, 50)
            };
            class_rows.push(serde_json::json!({
                "depth": depth,
                "timed_tokens": timed,
                "loop_wall_ms_p50": wall50,
                "classes": part.class_rows(timed),
            }));
            println!(
                "depth {depth} class partition: {}",
                serde_json::to_string(&class_rows.last().unwrap()).unwrap()
            );
        }
    }
    // `QWEN36_G2A_COMPILE` switch and the compile counters after every case and depth.
    let (compile_switch, compile_counters) = super::compiled::switch_and_counters(&decoder.kernels);
    let report = serde_json::json!({
        "engine": "fifi",
        "packet": "decode_phase_timing_by_mode",
        "checkpoint": ckpt_dir.display().to_string(),
        "config_sha256": sha256(&real.config),
        "ids": ids_report,
        "activation_dtype": activation_name(activation),
        "kv_dtype": kv_dtype_name(kv),
        "timed_tokens_per_case": timed,
        "warmup_tokens": WARMUP_TOKENS,
        "depths": depths,
        "chunked_prefill_allowed": chunked,
        "serving_chunk_requested": serve_setting,
        "serving_chunk": chunk,
        "chunk_piece": piece,
        "decode_routing": {
            "decode_path": routing,
            "chunk_path": routing,
            "note": "the decoder holds one routing setting and both paths follow it",
        },
        "sampler": {
            "seed": SAMPLER_SEED,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "row": "the full head row (no tokenizer is loaded); the served path narrows the row \
                    to the tokenizer vocabulary first, 248,077 of 248,320",
        },
        "phase_split": "the decoder API hides the build-to-evaluate boundary, so no production \
                        code was added: the split columns are the process-wide qualification \
                        counters (native-fault-qualification builds) and build is the residual \
                        of the step wall minus the counter-timed evaluate and readback walls; \
                        evaluate is the wait for the GPU as the qualification timer counts it \
                        around the synchronizing evaluate call, not a GPU-side kernel time",
        "compile": compile_switch,
        "compile_counters": compile_counters,
        "class_partition": class_rows,
        "rows": rows,
    });
    std::fs::write(
        out_dir.join("decode_timing.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("wrote {}", out_dir.join("decode_timing.json").display());

    // The A==B chain assert runs only now, with `decode_timing.json` already on disk: a
    // divergence is a defect of the measurement, and the rows of every depth survive it.
    for (depth, first_diff) in chain_checks {
        assert!(
            first_diff.is_none(),
            "the greedy chains of generate (device argmax) and logits_argmax (host argmax) \
             differ at depth {depth}, first difference at chain index {first_diff:?}: both \
             draw the argmax token, so a difference is a defect of the measurement, not a \
             model difference"
        );
    }
}

/// Prefill rate by depth segment (Mac, ignored, real checkpoint): the like-for-like counterpart of
/// the segment rates that an mlx-lm baseline reports. One growing state is prefilled through
/// the serving path (the same pieces, chunks and `DecodeSession` as `decode_phase_timing_by_mode`)
/// up to each depth of `PT_DEPTHS` (a comma list, default `16,8192,30000`); the wall time of
/// each segment (from the previous depth to this one) is measured around the whole segment, so a
/// segment's rate includes the attention over the growing context, as a prompt of that length
/// would pay it. Only the prefill runs: no token is decoded. The ids come from the same sources
/// as the decode timing (`DT_IDS_SOURCE=file` with `Q36_PROMPT_IDS` for depths above the
/// 922-id tool prompt). Run it like the decode timing with this test name.
#[test]
#[ignore = "needs the real checkpoint on the Mac; run with --ignored"]
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn prefill_rate_by_depth_segment() {
    let _serial = crate::native_test_lock();
    let ckpt_dir = env_path("Q36_CKPT_DIR");
    let source = ids_source_from_env();
    let ids = match source {
        IdsSource::Tool => {
            let (ids_path, ref_path) = (env_path("IDS_FILE"), env_path("REF_JSON"));
            let ids_value: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&ids_path).unwrap()).unwrap();
            let ref_value: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&ref_path).unwrap()).unwrap();
            read_ids(&ids_value, &ref_value)
        }
        IdsSource::File => read_prompt_ids(),
    };
    let marks = match std::env::var("PT_DEPTHS") {
        Ok(text) => parse_depths(&text, ids.len()).unwrap_or_else(|why| panic!("PT_DEPTHS: {why}")),
        Err(_) => parse_depths("16,8192,30000", ids.len()).unwrap_or_else(|why| panic!("{why}")),
    };
    let activation = required_activation_from_env();
    let out = out_dir_must_be_new();
    let out_dir = create_out_dir(&out);

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
    decoder.budget = SessionBudget::new(SESSION_CAP);
    let mut state = decoder.zero_state().unwrap();
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
    let piece = chunk
        .map(|rows| usize::try_from(rows).unwrap())
        .unwrap_or(NO_BUDGET_PIECE);
    if let Some(rows) = chunk {
        decoder.set_chunk_rows(rows).unwrap();
    }
    let limit = usize::try_from(CONTEXT_LIMIT).unwrap();
    let mut session = DecodeSession::default();
    let mut fed = Vec::new();
    let mut segments = Vec::new();
    let mut done = 0usize;
    decoder.kernels.reset_peak_memory().unwrap();
    for &mark in &marks {
        let mut step = ServeStep {
            decoder: &mut decoder,
            state: &mut state,
            chunk,
            fed: &mut fed,
        };
        let started = Instant::now();
        for rows in ids[done..mark].chunks(piece) {
            session.prefill(&mut step, rows, limit).unwrap();
        }
        // The last chunk's state is evaluated inside `prefill`; nothing is left pending, but
        // a read of one pending value would add a synchronization to the timed region, so the
        // wall of the loop above is the segment time.
        let seconds = started.elapsed().as_secs_f64();
        let tokens = mark - done;
        let memory = decoder.kernels.memory_stats().unwrap();
        println!(
            "PREFILL segment {done} -> {mark}: {tokens} tokens in {seconds:.2} s = {:.0} tok/s, \
             mlx active {:.2} GB peak {:.2} GB",
            tokens as f64 / seconds,
            memory.active as f64 / 1e9,
            memory.peak as f64 / 1e9
        );
        segments.push(serde_json::json!({
            "from": done,
            "to": mark,
            "tokens": tokens,
            "seconds": seconds,
            "tok_s": tokens as f64 / seconds,
            "mlx_active_bytes": memory.active,
            "mlx_peak_bytes_process": memory.peak,
        }));
        done = mark;
    }
    let report = serde_json::json!({
        "engine": "fifi",
        "test": "prefill_rate_by_depth_segment",
        "checkpoint": ckpt_dir.display().to_string(),
        "activation_dtype": activation_name(activation),
        "kv_dtype": kv_dtype_name(kv),
        "serving_chunk_requested": serve_setting,
        "serving_chunk": chunk,
        "chunk_piece": piece,
        "ids_in_source": ids.len(),
        "segments": segments,
        "fed_row_runs_total": fed.len(),
    });
    std::fs::write(
        out_dir.join("prefill_timing.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("wrote {}", out_dir.join("prefill_timing.json").display());
}

#[cfg(test)]
mod tests {
    use super::super::within_mode::{BankConfig, run_pieces};
    use super::*;
    use crate::EngError;
    use crate::family::qwen36::settings::NO_BUDGET_PIECE;
    use crate::family::sampler::{Sampler, SamplingParams};
    use crate::family::session::{DecodeSession, TokenStepper};

    type Result<T> = std::result::Result<T, EngError>;

    /// The context limit of the fake-stepper runs below: every prompt of the tie test is far
    /// under it.
    const STEPPER_LIMIT: usize = 4096;

    /// `parse_depths` takes the accepted depth-list form (`16,922`, entries trimmed, single
    /// depths allowed, the whole prompt as the inclusive upper bound) and rejects every malformed
    /// form with the variable named: an empty list, a trailing or lone comma, a depth of zero,
    /// a depth above the prompt, text, a negative number, and a duplicate depth.
    #[test]
    fn depths_parse_the_accepted_forms_and_reject_the_rest() {
        assert_eq!(parse_depths("16,922", 922).unwrap(), vec![16, 922]);
        assert_eq!(
            parse_depths(" 16 , 922 ", 922).unwrap(),
            vec![16, 922],
            "entries are trimmed"
        );
        assert_eq!(parse_depths("1", 922).unwrap(), vec![1], "one prompt token");
        assert_eq!(
            parse_depths("922", 922).unwrap(),
            vec![922],
            "the whole prompt is the inclusive upper bound"
        );
        for bad in [
            "", " ", ",", "16,", ",16", "0", "923", "16,x", "-4", "16,16",
        ] {
            assert!(
                parse_depths(bad, 922).is_err(),
                "{bad:?} must not parse as a depth list"
            );
        }
        assert_eq!(
            parse_depths("0", 922).unwrap_err(),
            "DT_DEPTHS: 0 is not a depth from 1 to 922 tokens (a depth cannot exceed the 922 ids \
             of the ids source)"
        );
        assert_eq!(
            parse_depths("16,16", 922).unwrap_err(),
            "DT_DEPTHS: the depth 16 appears twice (a duplicate would overwrite its own result \
             row)"
        );
        assert_eq!(
            parse_depths("16,x", 922).unwrap_err(),
            "DT_DEPTHS must be a comma list of numbers, got \"16,x\""
        );
    }

    /// The DEFAULT depth list goes through the same validation as an explicit one: `16` and
    /// the whole source parse for the 922-id tool prompt, while an ids source with fewer than
    /// 16 ids (the first default depth exceeds the source), a source of exactly 16 ids (the
    /// two default entries duplicate), and an empty source are clear errors that name the
    /// variable — not a panic, a duplicated result row, or a zero depth.
    #[test]
    fn the_default_depths_go_through_the_same_validation() {
        assert_eq!(default_depths(922).unwrap(), vec![16, 922]);
        assert_eq!(default_depths(17).unwrap(), vec![16, 17]);
        assert_eq!(
            default_depths(10).unwrap_err(),
            "DT_DEPTHS: 16 is not a depth from 1 to 10 tokens (a depth cannot exceed the 10 ids \
             of the ids source)",
            "a source with fewer than 16 ids"
        );
        assert_eq!(
            default_depths(16).unwrap_err(),
            "DT_DEPTHS: the depth 16 appears twice (a duplicate would overwrite its own result \
             row)",
            "a source of exactly 16 ids duplicates the two default depths"
        );
        assert_eq!(
            default_depths(0).unwrap_err(),
            "DT_DEPTHS: 16 is not a depth from 1 to 0 tokens (a depth cannot exceed the 0 ids \
             of the ids source)",
            "an empty ids source"
        );
    }

    /// `percentile` is the nearest-rank method: p50 of one to ten is 5, p90 is 9, p0 clamps to
    /// the first and p100 is the last, an unsorted input is sorted, an even count takes the
    /// lower of the middle two, and a single sample is every percentile.
    #[test]
    fn percentile_uses_the_nearest_rank_method() {
        let one_to_ten: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_eq!(percentile(&one_to_ten, 50), 5.0, "ceil(0.5 * 10) = 5");
        assert_eq!(percentile(&one_to_ten, 90), 9.0, "ceil(0.9 * 10) = 9");
        assert_eq!(percentile(&one_to_ten, 0), 1.0, "p0 clamps to rank 1");
        assert_eq!(percentile(&one_to_ten, 100), 10.0, "the last value");
        assert_eq!(
            percentile(&[3.0, 1.0, 2.0], 50),
            2.0,
            "the input is sorted, not assumed sorted"
        );
        assert_eq!(
            percentile(&[1.0, 2.0], 50),
            1.0,
            "an even count takes the lower of the middle two"
        );
        assert_eq!(percentile(&[7.5], 90), 7.5, "a single sample");
    }

    /// A `TokenStepper` that records the row runs it is fed and runs no model: `feed` records
    /// a one-token step, `feed_rows` splits a run at the chunk exactly like the harness's
    /// `ServeStep::feed_rows` (a chunk of `None` or 1 feeds token by token, a single row is a
    /// one-token step, a longer run is one chunked call per chunk-width rows), and `logits`
    /// records the final one-token step. Driven by the REAL `DecodeSession::prefill`, this
    /// pins the pending-token shift and the chunk split of `run_pieces` on Linux: the session,
    /// not a copy of it, decides what each piece feeds.
    struct RecordingStepper {
        /// The planned chunk size, or `None` for one-token steps.
        chunk: Option<u32>,
        /// The row runs fed so far, one entry per step or chunk.
        fed: Vec<usize>,
        /// The positions fed so far.
        positions: usize,
    }

    impl RecordingStepper {
        fn new(chunk: Option<u32>) -> Self {
            Self {
                chunk,
                fed: Vec::new(),
                positions: 0,
            }
        }
    }

    impl TokenStepper for RecordingStepper {
        const OPERATION: &'static str = "recording";

        fn feed(&mut self, _token: u32) -> Result<()> {
            self.fed.push(1);
            self.positions += 1;
            Ok(())
        }

        fn feed_rows(&mut self, tokens: &[u32]) -> Result<()> {
            let Some(size) = self.chunk.filter(|&size| size > 1) else {
                return tokens.iter().try_for_each(|&token| self.feed(token));
            };
            let width = usize::try_from(size).unwrap_or(usize::MAX);
            for rows in tokens.chunks(width) {
                if let &[token] = rows {
                    self.feed(token)?;
                } else {
                    self.fed.push(rows.len());
                    self.positions += rows.len();
                }
            }
            Ok(())
        }

        fn logits(&mut self, _token: u32) -> Result<Vec<f32>> {
            self.fed.push(1);
            self.positions += 1;
            Ok(Vec::new())
        }

        fn fed(&self) -> usize {
            self.positions
        }

        fn reset(&mut self) -> Result<()> {
            Ok(())
        }
    }

    /// Feeds `ids_len` ids of `config` through the real `DecodeSession::prefill` with the
    /// chunking of the Mac tests (the head in pieces of the planned chunk — or the
    /// `NO_BUDGET_PIECE` piece without one — plus the tail as ONE piece for `Tail`), then the
    /// one-token logits step of the pending id, and returns the row runs the stepper observed.
    fn recorded_row_runs(ids_len: usize, config: BankConfig, chunk: Option<usize>) -> Vec<usize> {
        let fallback = usize::try_from(NO_BUDGET_PIECE).unwrap_or(1);
        let piece = chunk.unwrap_or(fallback);
        let ids: Vec<u32> = (0..u32::try_from(ids_len).unwrap()).collect();
        let mut stepper = RecordingStepper::new(chunk.and_then(|width| u32::try_from(width).ok()));
        let mut session = DecodeSession::default();
        let pieces: Vec<&[u32]> = match config {
            BankConfig::Tail(t) => {
                let head = ids_len - t as usize;
                let mut pieces: Vec<&[u32]> = ids[..head].chunks(piece).collect();
                pieces.push(&ids[head..]);
                pieces
            }
            BankConfig::Steps | BankConfig::Chunk(_) => ids.chunks(piece).collect(),
        };
        for rows in pieces {
            session.prefill(&mut stepper, rows, STEPPER_LIMIT).unwrap();
        }
        assert_eq!(
            session.history().len(),
            ids_len,
            "the session commits every prompt token"
        );
        stepper.logits(session.pending().unwrap()).unwrap();
        stepper.fed
    }

    /// `run_pieces` matches what the REAL `DecodeSession::prefill` feeds for a handful of (N,
    /// T, chunk) shapes: the 922-id tool prompt with a small tail at the serving chunk, a tail
    /// LONGER than the chunk (300 at 256: 256 + 44 rows, not one call), a short prompt, a
    /// one-id first piece (its id stays pending and the tail feeds it plus its own first ids),
    /// a tail with NO chunk (token by token), a chunk of 1 (token by token), and a chunk
    /// larger than the prompt (one piece of N - 1 rows).
    #[test]
    fn run_pieces_matches_the_real_prefill_of_a_recording_stepper() {
        for (len, tail, chunk) in [
            (922_usize, 8_usize, Some(256_usize)),
            (922, 300, Some(256)),
            (30, 20, Some(256)),
            (9, 8, Some(256)),
            (40, 20, None),
            (5, 2, Some(1)),
            (20, 5, Some(512)),
        ] {
            let config = BankConfig::Tail(u32::try_from(tail).unwrap());
            assert_eq!(
                recorded_row_runs(len, config, chunk),
                run_pieces(len, config, chunk),
                "N {len}, tail {tail}, chunk {chunk:?}"
            );
        }
    }

    /// `host_argmax` returns the lowest index of the largest value (the sampler's greedy rule),
    /// works on negative rows, and a NaN never replaces the running maximum, so an all-NaN row
    /// keeps the seed index.
    #[test]
    fn host_argmax_takes_the_lowest_index_of_the_largest_value() {
        assert_eq!(host_argmax(&[0.0, 3.0, 3.0, 1.0]), 1, "a tie goes lower");
        assert_eq!(host_argmax(&[-2.0, -1.0, -3.0]), 1, "negative rows");
        assert_eq!(host_argmax(&[5.0]), 0, "a single entry");
        assert_eq!(
            host_argmax(&[f32::NAN, 1.0]),
            1,
            "a NaN never replaces the running maximum"
        );
        assert_eq!(
            host_argmax(&[f32::NAN]),
            0,
            "an all-NaN row keeps the seed index, like the sampler's argmax"
        );
    }

    /// The activation is REQUIRED: an unset `Q36_ACTIVATION` is an error whose
    /// message names the variable and its two accepted values — one rule for this harness and
    /// `tool_forward`, so the same command line cannot quietly mean different dtypes — and any
    /// text other than the two words is an error naming them.
    #[test]
    fn the_activation_is_required_and_names_the_variable_and_both_values() {
        assert_eq!(required_activation(Some("bf16")), Ok(Activation::Bf16));
        assert_eq!(required_activation(Some("f32")), Ok(Activation::F32));
        assert_eq!(
            required_activation(None).unwrap_err(),
            "Q36_ACTIVATION is unset; set it to bf16 or f32",
            "an unset run must stop with the variable and both values"
        );
        assert_eq!(
            required_activation(Some("f64")).unwrap_err(),
            "Q36_ACTIVATION must be bf16 or f32, got \"f64\""
        );
    }

    /// The ids source parse (the Mac speed gates need depths above 922): unset is `tool`, the
    /// two words are accepted, and any other word is an error naming the variable and both
    /// words.
    #[test]
    fn the_ids_source_parses_tool_and_file_and_defaults_to_tool() {
        assert_eq!(parse_ids_source(None), Ok(IdsSource::Tool));
        assert_eq!(parse_ids_source(Some("tool")), Ok(IdsSource::Tool));
        assert_eq!(parse_ids_source(Some("file")), Ok(IdsSource::File));
        assert_eq!(
            parse_ids_source(Some("json")).unwrap_err(),
            "DT_IDS_SOURCE must be tool or file, got \"json\""
        );
    }

    /// The `Q36_PROMPT_IDS` reader: little-endian u32 words — the first byte is the low byte —
    /// and a partial trailing word is an error that says what the file must hold.
    #[test]
    fn ids_from_le_bytes_reads_little_endian_words_and_rejects_a_partial_word() {
        assert_eq!(
            ids_from_le_bytes(&[0x01, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00]).unwrap(),
            vec![1, 255]
        );
        assert_eq!(
            ids_from_le_bytes(&[0x34, 0x12, 0x00, 0x00]).unwrap(),
            vec![0x1234],
            "the first byte is the low byte"
        );
        assert!(ids_from_le_bytes(&[]).unwrap().is_empty(), "an empty file");
        assert_eq!(
            ids_from_le_bytes(&[0x00, 0x00, 0x00, 0x00, 0x01]).unwrap_err(),
            "Q36_PROMPT_IDS holds 5 bytes: little-endian u32 words need a multiple of 4"
        );
    }

    /// The row check of case B mirrors the sampler's own check over the row: a finite row
    /// passes, and an empty row or one NaN or infinite logit fails.
    #[test]
    fn the_row_check_of_case_b_mirrors_the_sampler_check() {
        assert!(row_passes_the_sampler_check(&[0.0, -1.5, 3.25]));
        assert!(!row_passes_the_sampler_check(&[]), "an empty row");
        assert!(
            !row_passes_the_sampler_check(&[1.0, f32::NAN]),
            "a NaN logit fails"
        );
        assert!(
            !row_passes_the_sampler_check(&[f32::INFINITY]),
            "an infinite logit fails"
        );
    }

    /// The row check of case B agrees with the REAL sampler on the rows its `check_logits`
    /// rejects: the seeded `Sampler` of case C accepts exactly the finite non-empty rows and
    /// fails on the empty, NaN, and infinite rows, and `row_passes_the_sampler_check` must
    /// give the same verdict on each, so a change of the sampler's own check breaks this
    /// mirror (the check itself is private to `sampler.rs`, but `sample` runs it first).
    #[test]
    fn the_row_check_of_case_b_agrees_with_the_real_sampler() {
        let mut sampler = Sampler::new(SamplingParams::new(1.0, 0.95, 20).unwrap(), SAMPLER_SEED);
        for row in [
            vec![0.0, -1.5, 3.25],
            vec![],
            vec![1.0, f32::NAN],
            vec![f32::INFINITY, 0.0],
            vec![f32::NEG_INFINITY],
        ] {
            assert_eq!(
                row_passes_the_sampler_check(&row),
                sampler.sample(&row).is_ok(),
                "the row check and the sampler's own check disagree on {row:?}"
            );
        }
    }
    /// The cohort gate capture (reusable, R4-6): the per-position logits for the default-flip gate. Env: Q36_CKPT_DIR,
    /// Q36_GATE_IDS (the cohort tokens.json), Q36_GATE_OUT, Q36_ACTIVATION=bf16, Q36_FLASH_DELTA (selects the kernel),
    /// Q36_KV_DTYPE=q8. Two chunks of 2048; per chunk: prefill the first 1024 rows in Prompt
    /// mode, then EVERY scored position 1024..2046 runs as a ONE-ROW prefill chunk in Logits
    /// mode. The [vocab] f32 logits of the 1023 scored positions append to chunk{c}.f32.
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    #[ignore = "run with --ignored and Q36_CKPT_DIR/Q36_GATE_IDS/Q36_GATE_OUT set"]
    #[test]
    fn cohort_gate_capture() {
        let _serial = crate::native_test_lock();
        let ckpt_dir = env_path("Q36_CKPT_DIR");
        let ids_path = env_path("Q36_GATE_IDS");
        let out_dir = env_path("Q36_GATE_OUT");
        std::fs::create_dir_all(&out_dir).unwrap();
        #[derive(serde::Deserialize)]
        struct CohortFile {
            ids: Vec<u32>,
        }
        let ids: Vec<u32> =
            serde_json::from_str::<CohortFile>(&std::fs::read_to_string(&ids_path).unwrap())
                .unwrap()
                .ids;
        assert_eq!(ids.len(), 4380, "the cohort text");
        let real = open(&ckpt_dir);
        let mut kernels = MlxKernels::try_new().unwrap();
        kernels.set_cache_limit(CACHE_LIMIT).unwrap();
        let host = HostWeights::load(
            &real.ckpt,
            &crate::family::spec::QWEN3_6_35B_A3B,
            &real.config,
        )
        .unwrap();
        let mut decoder = ResidentDecoder::new_with_kv(
            kernels,
            &real.ckpt,
            host,
            CONTEXT_LIMIT,
            required_activation_from_env(),
            kv_dtype_from_env(),
        )
        .unwrap();
        decoder.budget = SessionBudget::new(SESSION_CAP);
        for c in 0..2usize {
            let chunk = &ids[c * 2048..(c + 1) * 2048];
            let mut state = decoder.zero_state().unwrap();
            decoder
                .prefill(&mut state, &chunk[..1024], StepMode::Prompt)
                .unwrap();
            let mut vocab = 0usize;
            let mut out: Vec<f32> = Vec::new();
            for p in 1024..2047usize {
                let r = decoder
                    .prefill(&mut state, &chunk[p..p + 1], StepMode::Logits)
                    .unwrap();
                let l = r.logits.expect("Logits mode");
                vocab = l.len();
                out.extend_from_slice(&l);
            }
            assert_eq!(out.len(), 1023 * vocab, "1023 scored positions");
            let bytes: Vec<u8> = out.iter().flat_map(|f| f.to_le_bytes()).collect();
            std::fs::write(out_dir.join(format!("chunk{}.f32", c)), &bytes).unwrap();
            eprintln!("WROTE chunk{}: 1023 x vocab {}", c, vocab);
        }
    }

    /// The robust-depth gate capture (reusable, R4-6): the decode logits at the robust depths. Same env plus
    /// DT_DEPTHS semantics: per depth in 16384/33600/40960, prefill to the depth then eight
    /// Logits steps; the [8 x vocab] pairs land in depth{d}.f32.
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    #[ignore = "run with --ignored and Q36_CKPT_DIR/Q36_GATE_IDS/Q36_GATE_OUT set"]
    #[test]
    fn robust_depth_gate_capture() {
        let _serial = crate::native_test_lock();
        let ckpt_dir = env_path("Q36_CKPT_DIR");
        let ids_path = env_path("Q36_GATE_IDS");
        let out_dir = env_path("Q36_GATE_OUT");
        std::fs::create_dir_all(&out_dir).unwrap();
        #[derive(serde::Deserialize)]
        struct CohortFile {
            ids: Vec<u32>,
        }
        let base: Vec<u32> =
            serde_json::from_str::<CohortFile>(&std::fs::read_to_string(&ids_path).unwrap())
                .unwrap()
                .ids;
        let mut ids = base.clone();
        while ids.len() < 41_000 {
            ids.extend_from_slice(&base);
        }
        ids.truncate(41_000);
        let real = open(&ckpt_dir);
        let mut kernels = MlxKernels::try_new().unwrap();
        kernels.set_cache_limit(CACHE_LIMIT).unwrap();
        let host = HostWeights::load(
            &real.ckpt,
            &crate::family::spec::QWEN3_6_35B_A3B,
            &real.config,
        )
        .unwrap();
        let mut decoder = ResidentDecoder::new_with_kv(
            kernels,
            &real.ckpt,
            host,
            CONTEXT_LIMIT,
            required_activation_from_env(),
            kv_dtype_from_env(),
        )
        .unwrap();
        decoder.budget = SessionBudget::new(SESSION_CAP);
        for &d in &[16_384usize, 33_600usize, 40_960usize] {
            let mut state = decoder.zero_state().unwrap();
            decoder
                .prefill(&mut state, &ids[..d], StepMode::Prompt)
                .unwrap();
            let mut out: Vec<f32> = Vec::new();
            for i in 0..8usize {
                let r = decoder
                    .step(&mut state, ids[d + i], StepMode::Logits)
                    .unwrap();
                out.extend_from_slice(&r.logits.expect("Logits mode"));
            }
            let bytes: Vec<u8> = out.iter().flat_map(|f| f.to_le_bytes()).collect();
            std::fs::write(out_dir.join(format!("depth{}.f32", d)), &bytes).unwrap();
            eprintln!("WROTE depth{}: 8 x vocab {}", d, out.len() / 8);
        }
    }
}
