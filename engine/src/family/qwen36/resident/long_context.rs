//! The long-context gate of the 8-bit KV cache.
//!
//! A coding agent sends long prompts, so the context length is the target. This file
//! holds the Mac test that measures, per cache form and depth, in one run: whether the model
//! still finds a needle and still writes a valid tool call at that depth, how long the prefill and
//! the decode steps take, and how much memory the run holds. A second test compares the final
//! prompt position of two runs (the F32 control and the Q8 cache) as full-vocabulary log-softmax
//! rows. The tests report numbers and assert only structural facts; the numbers are read
//! against the gate.
//!
//! The prompt has the shape of a coding-agent request:
//!
//! - a system message and two tool definitions (`read_file` and `run`) in the form the chat
//!   template renders;
//! - one user message with the haystack, the Rust files of a source directory as `=== path ===`
//!   blocks that repeat in order until the prompt has the target length, one needle line
//!   ("The release passphrase for service orion-7 is ...") at a line boundary at the requested
//!   fraction, and a task that asks for a `run` call that echoes the passphrase;
//! - the generation prompt with thinking off, so the model answers with a tool call.
//!
//! The prompt length is the largest number of haystack lines whose rendered prompt has at most the
//! target number of tokens. A line is the unit, so the real length is within one line of the
//! target, and the report names the real length. The passphrase is joined at run time from its
//! words and a haystack file that holds the whole needle sentence is left out, so the needle
//! appears once.
//!
//! What a depth reports (`long_context.json` in `Q36_OUT_DIR`, one row per depth):
//!
//! - the prompt: needle position in percent, real token count, haystack lines, needle line, and the
//!   SHA-256 of its ids;
//! - the prefill: seconds, rows fed (one fewer than the head holds: the last id stays pending),
//!   rows per second, the distinct row counts of the chunks that ran, and `cold` on the first
//!   depth of the process (its timings carry one-time costs);
//! - the decode (a task prompt without `LC_FORCE`): the greedy steps (the first two read back the
//!   full row, the rest read back only the token), the generated ids and text, a decode error if
//!   any, the first step, and the p50, mean, and maximum step time over the steps after the first
//!   four; the tokens per second are null under 8 timed steps. `greedy_text_contains_passphrase`
//!   is true when the greedy text holds the passphrase, which a format failure also makes false;
//!   `needle_pass` (a valid `run` call that holds the passphrase in its `cmd` argument) and
//!   `format_pass` (a valid tool call) are null unless the answer ended with EOS;
//! - the forced answer (`LC_FORCE=1`): the reference tool call is fed instead of the greedy
//!   decode, and `forced` reports the negative log-likelihood of the whole call and of the
//!   passphrase tokens, the smallest passphrase token probability, and how many tokens were the
//!   argmax. This measures the retrieval without the near tie of the first token;
//! - the format token: the probability of `<tool_call>` at the final prompt position, and the
//!   probability of the newline token after the first generated token, which is the format token
//!   of `tool_forward` only when that token was the tool tag (null otherwise, and null in a forced
//!   run);
//! - the cache fields (position, logical and stored bytes, state bytes) and the memory fields
//!   (MLX active, peak since the start of the depth, allocator cache, process resident set), both
//!   sampled at the end of the depth, after the window and the decode;
//! - the top five tokens of the final prompt position, and the file
//!   `logits_<depth>_p<percent>_<kv>.f32` with the log-softmax row of that position (little-endian
//!   `f32`); with a window, `window_<depth>_p<percent>_<kv>.f32` holds the rows of the window.
//!
//! The comparison test reads the reports of two run directories and refuses runs that differ in
//! activation, chunk, window, or prompt kind, that share a cache form, or whose prompts differ.
//! With `LC_COMPARE_NOISE=1` it takes two runs of ONE cache form at different chunks instead (the
//! noise floor of the schedule). With `LC_COMPARE_BUILD=1` it takes two runs of one schedule and
//! one cache form from two builds instead (an implementation A/B): the whole settings object
//! must match with every compared field present, and the checkpoint, tokenizer, and haystack
//! identity must be present and equal (physical weight identity stays with the external run
//! provenance). A key counts as present when it holds null: an unchunked run records
//! `"chunk": null`. The whole settings object includes values that the build computes or fixes
//! itself: the planned chunk, the prefill memory terms with the weight and remaining byte counts,
//! the shipped session cap, the cache limit, and the routing. A build that changes any of them
//! is refused even when both runs use the same chunk, because the A/B then did not hold the
//! schedule fixed. Setting both flags is an error. Its output names the mode and the settings
//! of both runs.
//!
//! A depth that the memory plan refuses reports the typed error as a result row and the run
//! continues with the next depth. The plan itself is the shipped one, except for the levers
//! below. A report records each lever.
//!
//! | Variable | Meaning | Default |
//! | --- | --- | --- |
//! | `Q36_CKPT_DIR` | the pinned checkpoint directory | required |
//! | `Q36_OUT_DIR` | the output directory, created after every input is validated | required |
//! | `Q36_ACTIVATION` | the decode activation dtype | required: `bf16` or `f32` |
//! | `Q36_KV_DTYPE` | the cache form | required: `f32` or `q8` |
//! | `LC_HAYSTACK_DIR` | a directory with `.rs` files, read in sorted path order | required |
//! | `LC_TOKENIZER_DIR` | the tokenizer files of the checkpoint | `Q36_CKPT_DIR` |
//! | `LC_DEPTHS` | comma list of target prompt lengths in tokens | `32768,131072` |
//! | `LC_DECODE_TOKENS` | the most greedy tokens per depth | 64 |
//! | `LC_WINDOW` | the last prompt positions run as teacher-forced one-token steps whose rows are kept (0: none) | 0 (task), 128 (text) |
//! | `LC_DUMP_DIR` | (the dry run only) a directory that receives `ids_<depth>_p<percent>.u32`, the prompt ids as little-endian `u32`, for a reference run in another stack | unset |
//! | `LC_PROMPT` | `task` (the coding-agent prompt with the needle) or `text` (the haystack alone as raw ids: a code continuation whose window rows are natural text deep in the context) | `task` |
//! | `LC_FORCE` | 1: feed the reference tool call instead of the greedy decode and report the negative log-likelihood of the passphrase tokens (the retrieval, without the near tie of the first token) | 0 |
//! | `LC_NEEDLE_PERCENTS` | comma list of needle positions in percent of the haystack lines; every depth runs at every position | 50 |
//! | `LC_CONTEXT_LIMIT` | the context limit of the decoder | deepest depth + decode tokens + 16 |
//! | `LC_SESSION_CAP` | the session byte cap of the decoder | the shipped cap (2 GiB) |
//! | `LC_CHUNK` | prefill chunk rows instead of the planned chunk | unset: the plan |
//!
//! The prompt build needs only the tokenizer files. The dry run prints the real token counts,
//! the build times, and the head and tail of each prompt without loading a model:
//!
//! ```text
//! LC_TOKENIZER_DIR=<CKPT_DIR> LC_HAYSTACK_DIR=<HAYSTACK_DIR> LC_DEPTHS=32768,131072,262000 \
//! cargo test --release --lib \
//!   family::qwen36::resident::long_context::long_context_prompts -- --ignored --exact --nocapture
//! ```
//!
//! The commands of the gate (placeholders for the paths; run each cache form once). The planned
//! prefill chunk depends on the context limit, so a run that compares two cache forms fixes the
//! chunk with `LC_CHUNK` and the rows are comparable:
//!
//! ```text
//! Q36_CKPT_DIR=<CKPT_DIR> Q36_OUT_DIR=<OUT_DIR> Q36_ACTIVATION=bf16 Q36_KV_DTYPE=q8 \
//! LC_HAYSTACK_DIR=<HAYSTACK_DIR> LC_DEPTHS=32768,131072,262000 LC_SESSION_CAP=4294967296 \
//! LC_CHUNK=256 \
//! cargo test --release --features native-mlx --lib \
//!   family::qwen36::resident::long_context::mac::long_context_gate -- --ignored --exact --nocapture
//! ```
//!
//! The F32 control is the same command with `Q36_KV_DTYPE=f32`, a new `Q36_OUT_DIR`, and
//! `LC_DEPTHS=32768`. The comparison reads the two output directories and needs no Mac:
//!
//! ```text
//! LC_COMPARE_CONTROL=<F32_OUT_DIR> LC_COMPARE_CANDIDATE=<Q8_OUT_DIR> Q36_OUT_DIR=<COMPARE_DIR> \
//! cargo test --release --lib \
//!   family::qwen36::resident::long_context::long_context_compare -- --ignored --exact --nocapture
//! ```

use serde_json::json;
use std::path::Path;

/// The words of the passphrase. They are joined at run time, so no source file holds the whole
/// needle sentence.
const PASSPHRASE_WORDS: [&str; 3] = ["quasar", "lantern", "4412"];

/// The system message of the prompt.
const SYSTEM_TEXT: &str = "You are a coding agent. Use the tools to act on the repository.";

/// The text before the haystack.
const INTRO: &str = "These are the files of the repository.\n\n";

/// The task after the haystack.
const TASK: &str = "\nTask: the release passphrase for service orion-7 appears once in the \
files above. Call the `run` tool with the command `echo <passphrase>`, where <passphrase> is that \
passphrase. Answer with the tool call only.";

/// The most haystack lines a prompt search tries. A target that needs more is an error.
const MAX_LINES: usize = 1 << 24;

/// The logits rows that a compare holds the top tokens of.
const TOP: usize = 5;

/// Strings that the tokenizer turns into one structural token wherever they stand in a text: the
/// chat markers, the tool-call and tool-response tags, and the think tags. A haystack file that
/// spells one of them would inject a structural token into the user message, so such a file is
/// left out of the haystack.
const CONTROL_STRINGS: [&str; 10] = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    "<|vision_start|>",
];

/// The answer that the forced probe feeds: the tool call that echoes the passphrase.
fn forced_answer_text() -> String {
    format!(
        "<tool_call>\n<function=run>\n<parameter=cmd>\necho {}\n</parameter>\n</function>\n</tool_call>",
        passphrase()
    )
}

/// What a prompt is made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PromptKind {
    /// The coding-agent task: system message with tools, the haystack with the needle, the task,
    /// and the generation prompt.
    Task,
    /// The haystack text alone, as raw ids, with no template and no needle: a code continuation
    /// whose last positions are natural text deep in the context.
    Text,
}

/// The kind of `LC_PROMPT`: `task` (the default) or `text`.
fn parse_prompt_kind(text: &str) -> Result<PromptKind, String> {
    match text {
        "task" => Ok(PromptKind::Task),
        "text" => Ok(PromptKind::Text),
        other => Err(format!("LC_PROMPT must be task or text, got {other:?}")),
    }
}

/// The passphrase of the needle.
fn passphrase() -> String {
    PASSPHRASE_WORDS.join("-")
}

/// The needle line.
fn needle_sentence() -> String {
    format!(
        "The release passphrase for service orion-7 is {}.",
        passphrase()
    )
}

/// One file of the haystack: the path shown in its header and its text.
#[derive(Clone, Debug)]
struct HaystackFile {
    path: String,
    text: String,
}

/// The lines of the haystack: for each file a header line and then its lines, in the order of the
/// files. The lines repeat in this order for as many lines as a prompt needs.
struct Haystack {
    cycle: Vec<String>,
}

impl Haystack {
    /// Builds the cycle. A file that holds the needle sentence and an empty list of files are
    /// errors. A file that spells a control string is an error here too; `read_haystack_dir`
    /// leaves such files out before this check.
    fn new(files: &[HaystackFile]) -> Result<Self, String> {
        let needle = needle_sentence();
        let mut cycle = Vec::new();
        for file in files {
            if file.text.contains(&needle) {
                return Err(format!(
                    "the haystack file {} holds the needle sentence",
                    file.path
                ));
            }
            if let Some(control) = CONTROL_STRINGS.iter().find(|c| file.text.contains(**c)) {
                return Err(format!(
                    "the haystack file {} spells the control string {control}",
                    file.path
                ));
            }
            cycle.push(format!("=== {} ===", file.path));
            cycle.extend(file.text.lines().map(str::to_string));
        }
        if cycle.is_empty() {
            return Err("the haystack holds no lines".to_string());
        }
        Ok(Self { cycle })
    }

    /// The first `lines` haystack lines and nothing else.
    fn plain_text(&self, lines: usize) -> String {
        let mut text = String::new();
        for index in 0..lines {
            text.push_str(&self.cycle[index % self.cycle.len()]);
            text.push('\n');
        }
        text
    }

    /// The user message: the intro, `lines` haystack lines, the needle line before haystack line
    /// `needle_line` (after the last line when `needle_line` equals `lines`), and the task.
    fn user_text(&self, lines: usize, needle_line: usize) -> String {
        let needle = needle_sentence();
        let mut text = String::from(INTRO);
        for index in 0..lines {
            if index == needle_line {
                text.push_str(&needle);
                text.push('\n');
            }
            text.push_str(&self.cycle[index % self.cycle.len()]);
            text.push('\n');
        }
        if needle_line >= lines {
            text.push_str(&needle);
            text.push('\n');
        }
        text.push_str(TASK);
        text
    }
}

/// The index of the line before which the needle stands: `fraction` of `lines`, rounded down.
fn needle_line_for(lines: usize, fraction: f64) -> usize {
    let line = (lines as f64 * fraction).floor();
    if line >= lines as f64 {
        lines
    } else {
        line as usize
    }
}

/// One built prompt.
#[derive(Debug)]
struct BuiltPrompt {
    /// The prompt ids as the chat template and the tokenizer produce them.
    ids: Vec<u32>,
    /// The haystack lines in the prompt.
    lines: usize,
    /// The line before which the needle stands.
    needle_line: usize,
}

/// Builds the prompt with the most haystack lines whose ids number at most `target`. `encode` maps
/// the text of the prompt kind (the user message of a task, the plain haystack text of a text
/// prompt) to the ids of the whole prompt. The search doubles the lines until the prompt exceeds
/// `target` and then bisects. The result always fits, and one more line does not fit (both ends
/// are real renders); that it is the largest fitting count assumes that more lines never give
/// fewer tokens, which holds at line boundaries.
fn build_prompt(
    haystack: &Haystack,
    target: usize,
    fraction: f64,
    kind: PromptKind,
    encode: &mut dyn FnMut(&str) -> Result<Vec<u32>, String>,
) -> Result<BuiltPrompt, String> {
    if !(0.0..=1.0).contains(&fraction) {
        return Err(format!("the needle fraction {fraction} is outside 0 to 1"));
    }
    let mut render = |lines: usize| -> Result<Vec<u32>, String> {
        match kind {
            PromptKind::Task => {
                encode(&haystack.user_text(lines, needle_line_for(lines, fraction)))
            }
            PromptKind::Text => encode(&haystack.plain_text(lines)),
        }
    };
    let mut fits = 0usize;
    let mut fit_ids = render(0)?;
    if fit_ids.len() > target {
        return Err(format!(
            "the target of {target} tokens is below the {} tokens of the prompt without a haystack",
            fit_ids.len()
        ));
    }
    let mut over = 1usize;
    loop {
        if over > MAX_LINES {
            return Err(format!(
                "the target of {target} tokens needs more than {MAX_LINES} haystack lines"
            ));
        }
        let ids = render(over)?;
        if ids.len() > target {
            break;
        }
        fits = over;
        fit_ids = ids;
        over = over.saturating_mul(2);
    }
    while over - fits > 1 {
        let middle = fits + (over - fits) / 2;
        let ids = render(middle)?;
        if ids.len() > target {
            over = middle;
        } else {
            fits = middle;
            fit_ids = ids;
        }
    }
    Ok(BuiltPrompt {
        ids: fit_ids,
        lines: fits,
        needle_line: match kind {
            PromptKind::Task => needle_line_for(fits, fraction),
            PromptKind::Text => 0,
        },
    })
}

/// The number of ids at or above `floor`: the added tokens of the tokenizer (chat markers, tool
/// and think tags, and the rest of its structural tokens) all sit above its first added id.
fn count_at_least(ids: &[u32], floor: u32) -> usize {
    ids.iter().filter(|&&id| id >= floor).count()
}

/// The (fed id, predicted id) pairs of the teacher-forced window: the last `window` ids of the
/// prompt, each predicted from the id before it. The model has fed `ids[..head - 1]` after the
/// chunked prefill of `ids[..head]` (the last id of the chunk stays pending), so the first pair
/// feeds `ids[head - 1]`. After the last pair the next id to feed is `ids[len - 1]`.
fn window_pairs(ids: &[u32], head: usize, window: usize) -> Vec<(u32, u32)> {
    (head..head + window)
        .map(|position| (ids[position - 1], ids[position]))
        .collect()
}

/// The forced answer: its ids, and the lengths of the token prefix that ends before the
/// passphrase and of the token prefix that ends after it. `encode` maps a text to ids.
#[derive(Clone, Debug, PartialEq)]
struct ForcedAnswer {
    ids: Vec<u32>,
    /// The ids before the first passphrase token (the call up to and including `echo`).
    prefix_len: usize,
    /// The ids up to and including the last passphrase token.
    body_len: usize,
}

/// Encodes the forced answer and finds the span of the passphrase tokens. The span exists only
/// when the encoding of the call up to `echo` and the encoding of the call up to the passphrase
/// are both prefixes of the encoding of the whole answer; otherwise it returns `None` and the
/// caller reports the whole-answer figures only.
fn forced_answer(encode: &dyn Fn(&str) -> Option<Vec<u32>>) -> Option<ForcedAnswer> {
    let answer = encode(&forced_answer_text())?;
    let head = "<tool_call>\n<function=run>\n<parameter=cmd>\necho";
    let prefix = encode(head)?;
    let body = encode(&format!("{head} {}", passphrase()))?;
    if answer.starts_with(&body) && body.starts_with(&prefix) && prefix.len() < body.len() {
        Some(ForcedAnswer {
            prefix_len: prefix.len(),
            body_len: body.len(),
            ids: answer,
        })
    } else {
        None
    }
}

/// The depths of `LC_DEPTHS`: a comma list of positive integers, each at least 1,024, without a
/// repeat. The order is the order of the run.
fn parse_depth_list(text: &str) -> Result<Vec<usize>, String> {
    let mut depths = Vec::new();
    for entry in text.split(',') {
        let entry = entry.trim();
        let depth: usize = entry
            .parse()
            .map_err(|_| format!("LC_DEPTHS entry {entry:?} is not a positive integer"))?;
        if depth < 1_024 {
            return Err(format!("LC_DEPTHS entry {depth} is below 1,024 tokens"));
        }
        if depths.contains(&depth) {
            return Err(format!("LC_DEPTHS repeats the depth {depth}"));
        }
        depths.push(depth);
    }
    Ok(depths)
}

/// The median of `values`, or `None` for an empty list. A list of even length gives the mean of
/// the two middle values.
fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Some(if sorted.len() % 2 == 1 {
        sorted[middle]
    } else {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    })
}

/// The log-softmax of `row`, computed in `f64` and stored as `f32`. An empty row or a row with a
/// value that is not finite is an error.
fn log_softmax(row: &[f32]) -> Result<Vec<f32>, String> {
    if row.is_empty() {
        return Err("the logits row is empty".to_string());
    }
    if row.iter().any(|value| !value.is_finite()) {
        return Err("the logits row holds a value that is not finite".to_string());
    }
    let max = row
        .iter()
        .fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
    let sum: f64 = row.iter().map(|&v| (f64::from(v) - max).exp()).sum();
    let log_sum = sum.ln();
    Ok(row
        .iter()
        .map(|&v| (f64::from(v) - max - log_sum) as f32)
        .collect())
}

/// The `k` largest values of `row` as `(id, value)`, largest first. A tie keeps the lower id
/// first.
fn top_k(row: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut ranked: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked.truncate(k);
    ranked
}

/// What a comparison of two log-softmax rows reports.
#[derive(Debug, PartialEq)]
struct Comparison {
    /// True when both rows have the same largest token.
    argmax_equal: bool,
    /// How many of the top tokens of the control are among the top tokens of the candidate.
    top_overlap: usize,
    /// The number of top tokens compared: five, or fewer for a row narrower than five.
    top_count: usize,
    /// KL(control || candidate) in nats over the whole row, in `f64`.
    kl_nats: f64,
    /// The largest absolute log-probability difference over the top tokens of the control.
    max_top_logprob_diff: f64,
}

/// Compares the candidate row with the control row. Both rows are log-softmax rows of the same
/// width.
fn compare_log_probs(control: &[f32], candidate: &[f32]) -> Result<Comparison, String> {
    if control.len() != candidate.len() {
        return Err(format!(
            "the rows differ in width: {} and {}",
            control.len(),
            candidate.len()
        ));
    }
    if control.is_empty() {
        return Err("the rows are empty".to_string());
    }
    let count = TOP.min(control.len());
    let control_top = top_k(control, count);
    let candidate_top = top_k(candidate, count);
    let top_overlap = control_top
        .iter()
        .filter(|(id, _)| candidate_top.iter().any(|(other, _)| other == id))
        .count();
    let kl_nats = control
        .iter()
        .zip(candidate)
        .map(|(&a, &b)| {
            let a = f64::from(a);
            a.exp() * (a - f64::from(b))
        })
        .sum();
    let max_top_logprob_diff = control_top
        .iter()
        .map(|&(id, _)| (f64::from(control[id]) - f64::from(candidate[id])).abs())
        .fold(0.0, f64::max);
    Ok(Comparison {
        argmax_equal: control_top[0].0 == candidate_top[0].0,
        top_overlap,
        top_count: count,
        kl_nats,
        max_top_logprob_diff,
    })
}

/// What a comparison of two teacher-forced windows reports.
#[derive(Debug, PartialEq)]
struct WindowSummary {
    /// The number of positions compared.
    rows: usize,
    /// The mean of KL(control || candidate) over the positions, in nats.
    mean_kl_nats: f64,
    /// The nearest-rank 90th percentile of the KL over the positions.
    p90_kl_nats: f64,
    /// The largest KL over the positions.
    max_kl_nats: f64,
    /// The share of positions where both rows have the same largest token.
    argmax_agreement: f64,
    /// The mean top-5 overlap (at most 5, fewer for a narrower row).
    mean_top_overlap: f64,
}

/// Compares two windows of `width`-wide log-softmax rows, position by position. Both windows
/// hold the same whole number of rows.
fn summarize_window(
    control: &[f32],
    candidate: &[f32],
    width: usize,
) -> Result<WindowSummary, String> {
    if width == 0 || control.len() != candidate.len() || !control.len().is_multiple_of(width) {
        return Err(format!(
            "windows of {} and {} values are not the same whole number of rows of {width}",
            control.len(),
            candidate.len()
        ));
    }
    let rows = control.len() / width;
    if rows == 0 {
        return Err("the windows are empty".to_string());
    }
    let (mut kls, mut agree, mut overlap) = (Vec::with_capacity(rows), 0usize, 0usize);
    for row in 0..rows {
        let span = row * width..(row + 1) * width;
        let comparison = compare_log_probs(&control[span.clone()], &candidate[span])?;
        kls.push(comparison.kl_nats);
        agree += usize::from(comparison.argmax_equal);
        overlap += comparison.top_overlap;
    }
    kls.sort_by(f64::total_cmp);
    let rank = (rows * 9).div_ceil(10);
    Ok(WindowSummary {
        rows,
        mean_kl_nats: kls.iter().sum::<f64>() / rows as f64,
        p90_kl_nats: kls[rank - 1],
        max_kl_nats: kls[rows - 1],
        argmax_agreement: agree as f64 / rows as f64,
        mean_top_overlap: overlap as f64 / rows as f64,
    })
}

/// The file bytes of a log-softmax row: each value as little-endian `f32`.
fn log_probs_to_bytes(row: &[f32]) -> Vec<u8> {
    row.iter().flat_map(|value| value.to_le_bytes()).collect()
}

/// The row of a `logits_<depth>_<kv>.f32` file. A length that is not a multiple of four is an
/// error.
fn log_probs_from_bytes(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "a logits file of {} bytes is not a whole number of f32 values",
            bytes.len()
        ));
    }
    let (values, _) = bytes.as_chunks::<4>();
    Ok(values
        .iter()
        .map(|value| f32::from_le_bytes(*value))
        .collect())
}

/// The file name of the final-position row of a depth, a needle position in percent, and a cache
/// form.
fn logits_file_name(depth: usize, percent: usize, kv: &str) -> String {
    format!("logits_{depth}_p{percent}_{kv}.f32")
}

/// The file name of the teacher-forced window rows of a depth, a needle position, and a cache
/// form: `window` rows of log-softmax, one after the other, each as wide as a final-position row.
fn window_file_name(depth: usize, percent: usize, kv: &str) -> String {
    format!("window_{depth}_p{percent}_{kv}.f32")
}

/// The depth and the needle percent in the name of a `logits_<depth>_p<percent>_<kv>.f32` file,
/// or `None` for another name.
fn parse_logits_file_name(name: &str) -> Option<(usize, usize)> {
    parse_row_file_name(name, "logits_")
}

/// The depth and the needle percent in the name of a `window_<depth>_p<percent>_<kv>.f32` file,
/// or `None` for another name.
fn parse_window_file_name(name: &str) -> Option<(usize, usize)> {
    parse_row_file_name(name, "window_")
}

fn parse_row_file_name(name: &str, prefix: &str) -> Option<(usize, usize)> {
    let rest = name.strip_prefix(prefix)?.strip_suffix(".f32")?;
    let mut parts = rest.split('_');
    let (depth, percent, kv) = (parts.next()?, parts.next()?, parts.next()?);
    if kv.is_empty() || parts.next().is_some() {
        return None;
    }
    Some((
        depth.parse().ok()?,
        percent.strip_prefix('p')?.parse().ok()?,
    ))
}

/// The needle positions of `LC_NEEDLE_PERCENTS`: a comma list of integers from 0 to 100 without a
/// repeat. The order is the order of the run.
fn parse_percent_list(text: &str) -> Result<Vec<usize>, String> {
    let mut percents = Vec::new();
    for entry in text.split(',') {
        let entry = entry.trim();
        let percent: usize = entry
            .parse()
            .map_err(|_| format!("LC_NEEDLE_PERCENTS entry {entry:?} is not an integer"))?;
        if percent > 100 {
            return Err(format!("LC_NEEDLE_PERCENTS entry {percent} is above 100"));
        }
        if percents.contains(&percent) {
            return Err(format!("LC_NEEDLE_PERCENTS repeats the position {percent}"));
        }
        percents.push(percent);
    }
    Ok(percents)
}

/// The `.rs` files under `dir` in sorted relative-path order, split into the files that can serve
/// as haystack and the paths left out because they hold the needle sentence or spell a control
/// string of the tokenizer (`CONTROL_STRINGS`).
fn read_haystack_dir(dir: &Path) -> Result<(Vec<HaystackFile>, Vec<String>), String> {
    fn walk(dir: &Path, base: &Path, out: &mut Vec<String>) -> Result<(), String> {
        let entries = std::fs::read_dir(dir).map_err(|e| format!("read {}: {e}", dir.display()))?;
        for entry in entries {
            let path = entry.map_err(|e| e.to_string())?.path();
            if path.is_dir() {
                walk(&path, base, out)?;
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let relative = path.strip_prefix(base).map_err(|e| e.to_string())?;
                out.push(relative.to_string_lossy().replace('\\', "/"));
            }
        }
        Ok(())
    }
    let mut paths = Vec::new();
    walk(dir, dir, &mut paths)?;
    paths.sort();
    let needle = needle_sentence();
    let (mut files, mut left_out) = (Vec::new(), Vec::new());
    for path in paths {
        let text =
            std::fs::read_to_string(dir.join(&path)).map_err(|e| format!("read {path}: {e}"))?;
        if text.contains(&needle)
            || CONTROL_STRINGS
                .iter()
                .any(|control| text.contains(*control))
        {
            left_out.push(path);
        } else {
            files.push(HaystackFile { path, text });
        }
    }
    Ok((files, left_out))
}

/// The two tool definitions of the prompt, in the OpenAI form the template receives.
fn tool_definitions() -> Vec<serde_json::Value> {
    vec![
        json!({"type": "function", "function": {
            "name": "read_file",
            "description": "Read a file of the repository.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "The path of the file."}},
                "required": ["path"]}}}),
        json!({"type": "function", "function": {
            "name": "run",
            "description": "Run a shell command in the repository.",
            "parameters": {"type": "object", "properties": {
                "cmd": {"type": "string", "description": "The command line."}},
                "required": ["cmd"]}}}),
    ]
}

/// The verdict of the generated text: the number of valid calls, the number of malformed call
/// bodies, and whether a `run` call holds the passphrase.
#[derive(Debug, PartialEq)]
struct TextVerdict {
    calls: usize,
    malformed: usize,
    needle_pass: bool,
}

/// Parses `text` as the answer of a prompt that ended with an empty think block and judges it.
fn judge_text(text: &str, tools: &[crate::tokenizer::ToolSpec]) -> TextVerdict {
    use crate::tokenizer::toolcall::{Start, ToolSchemas, parse_complete};
    let parsed = parse_complete(
        text,
        Start::Content,
        ToolSchemas::from_tools(tools),
        "call_",
    );
    let passphrase = passphrase();
    TextVerdict {
        calls: parsed.tool_calls.len(),
        malformed: parsed.malformed.len(),
        // The call must name the `run` tool and carry the passphrase in its `cmd` argument,
        // the one parameter the tool declares; a call with another parameter name would be
        // rejected by a real client.
        needle_pass: parsed.tool_calls.iter().any(|call| {
            call.name == "run"
                && serde_json::from_str::<serde_json::Value>(&call.arguments)
                    .ok()
                    .and_then(|arguments| {
                        arguments
                            .get("cmd")
                            .and_then(serde_json::Value::as_str)
                            .map(|cmd| cmd.contains(&passphrase))
                    })
                    .unwrap_or(false)
        }),
    }
}

/// What two run directories are compared for: the two cache forms of one schedule, two chunks
/// of one cache form (the noise floor of the schedule), or two builds of one schedule and one
/// cache form (an implementation A/B).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompareMode {
    Cache,
    Noise,
    Build,
}

impl CompareMode {
    /// The name the comparison report carries for the mode.
    fn label(self) -> &'static str {
        match self {
            CompareMode::Cache => "cache",
            CompareMode::Noise => "noise floor",
            CompareMode::Build => "build",
        }
    }
}

/// Picks the comparison mode from the `LC_COMPARE_NOISE` and `LC_COMPARE_BUILD` request flags.
/// Both set at once is ambiguous, so it is an error instead of a silent winner.
fn compare_mode_from_flags(noise: bool, build: bool) -> Result<CompareMode, String> {
    match (noise, build) {
        (false, false) => Ok(CompareMode::Cache),
        (true, false) => Ok(CompareMode::Noise),
        (false, true) => Ok(CompareMode::Build),
        (true, true) => Err(
            "LC_COMPARE_NOISE and LC_COMPARE_BUILD are both set; pick one comparison mode"
                .to_string(),
        ),
    }
}

/// The reports of two runs must describe comparable runs: the settings the mode holds
/// equal (a build comparison also needs every one of them present, the whole settings object
/// equal, and the checkpoint, tokenizer, and haystack identity present and equal), the
/// difference the mode needs (a cache form for a cache comparison, a chunk for a noise floor),
/// and (checked per prompt by the caller) the same prompt ids. Returns the prompt hashes by
/// depth and needle position.
fn check_comparable(
    control: &Path,
    candidate: &Path,
    mode: CompareMode,
) -> Result<Comparable, String> {
    fn report(dir: &Path) -> Result<serde_json::Value, String> {
        let path = dir.join("long_context.json");
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))
    }
    let (a, b) = (report(control)?, report(candidate)?);
    // A noise-floor comparison puts two runs of ONE cache form that differ in the chunk side by
    // side: it measures how far the schedule alone moves the rows. A cache comparison puts two
    // cache forms at one schedule. A build comparison puts two builds of one schedule and one
    // cache form side by side.
    let same_fields: &[&str] = match mode {
        CompareMode::Noise => &["activation", "window", "prompt_kind", "kv_dtype"],
        CompareMode::Cache => &["activation", "chunk", "window", "prompt_kind"],
        CompareMode::Build => &["activation", "chunk", "window", "prompt_kind", "kv_dtype"],
    };
    // A missing setting never compares as equal in a build comparison, even when both reports
    // lack it: the key must exist on both sides. Presence is the key, not the value: the writer
    // records `"chunk": null` for a run whose plan admits no chunk (a depth of 4,096 or less),
    // and two such runs hold the same schedule.
    if mode == CompareMode::Build {
        for &field in same_fields {
            for (side, report) in [("control", &a), ("candidate", &b)] {
                if report["settings"].get(field).is_none() {
                    return Err(format!("the {side} run has no {field} setting"));
                }
            }
        }
    }
    for &field in same_fields {
        if a["settings"][field] != b["settings"][field] {
            return Err(format!(
                "the runs differ in {field}: {} against {}",
                a["settings"][field], b["settings"][field]
            ));
        }
    }
    if mode == CompareMode::Noise && a["settings"]["chunk"] == b["settings"]["chunk"] {
        return Err(format!(
            "a noise-floor comparison needs two chunks, both runs use {}",
            a["settings"]["chunk"]
        ));
    }
    if mode == CompareMode::Cache && a["settings"]["kv_dtype"] == b["settings"]["kv_dtype"] {
        return Err(format!(
            "both runs use the cache form {}",
            a["settings"]["kv_dtype"]
        ));
    }
    // The rest of the build preconditions: the whole settings object, so members beyond the
    // named fields (piece, routing, force, caps, ...) cannot drift, and the checkpoint,
    // tokenizer, and haystack identity, present and equal. Physical weight identity stays
    // with the external run provenance of the two directories.
    if mode == CompareMode::Build {
        if a["settings"] != b["settings"] {
            return Err(format!(
                "the runs differ in settings: {} against {}",
                a["settings"], b["settings"]
            ));
        }
        for (section, field) in [
            ("checkpoint", "config_sha256"),
            ("tokenizer", "tokenizer_json_sha256"),
            ("haystack", "sha256"),
        ] {
            for (side, report) in [("control", &a), ("candidate", &b)] {
                if report[section][field].is_null() {
                    return Err(format!("the {side} run has no {section}.{field} identity"));
                }
            }
            if a[section][field] != b[section][field] {
                return Err(format!(
                    "the runs differ in {section}.{field}: {} against {}",
                    a[section][field], b[section][field]
                ));
            }
        }
    }
    let hashes =
        |report: &serde_json::Value| -> std::collections::BTreeMap<(usize, usize), String> {
            report["rows"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|row| {
                    Some((
                        (
                            usize::try_from(row["depth_target"].as_u64()?).ok()?,
                            usize::try_from(row["needle_percent"].as_u64()?).ok()?,
                        ),
                        row["prompt_sha256"].as_str()?.to_string(),
                    ))
                })
                .collect()
        };
    let (control_hashes, candidate_hashes) = (hashes(&a), hashes(&b));
    let mut paired = std::collections::BTreeMap::new();
    for (key, control_hash) in control_hashes {
        if let Some(candidate_hash) = candidate_hashes.get(&key) {
            if &control_hash != candidate_hash {
                return Err(format!(
                    "depth {} at {} percent: the runs built different prompts",
                    key.0, key.1
                ));
            }
            paired.insert(key, control_hash);
        }
    }
    let summary = |report: &serde_json::Value| {
        let settings = &report["settings"];
        json!({
            "kv_dtype": settings["kv_dtype"],
            "activation": settings["activation"],
            "chunk": settings["chunk"],
            "window": settings["window"],
            "prompt_kind": settings["prompt_kind"],
        })
    };
    Ok(Comparable {
        hashes: paired,
        control: summary(&a),
        candidate: summary(&b),
    })
}

/// What `check_comparable` returns: the prompt hash of every prompt both reports hold, and the
/// settings of each run that the comparison depends on.
struct Comparable {
    hashes: std::collections::BTreeMap<(usize, usize), String>,
    control: serde_json::Value,
    candidate: serde_json::Value,
}

/// The comparison of two run directories: for each depth and needle position with a
/// `logits_<depth>_p<percent>_<kv>.f32` file in both, the argmax flag, the top-token overlap,
/// KL(control || candidate), and the largest log-probability difference over the top tokens, and
/// the summary of the teacher-forced windows when both runs wrote one. A prompt in one directory
/// only gets a row that names the missing side. Two files of one prompt in one directory, runs
/// that differ in activation, chunk, window, or prompt kind (or share a cache form), prompts with
/// different ids, and a comparison with no paired prompt are errors. A noise comparison must
/// differ in nothing but the chunk, a build comparison in nothing at all.
fn compare_directories(
    control: &Path,
    candidate: &Path,
    mode: CompareMode,
) -> Result<serde_json::Value, String> {
    let comparable = check_comparable(control, candidate, mode)?;
    let hashes = &comparable.hashes;
    fn files_by_prompt(
        dir: &Path,
        parse: fn(&str) -> Option<(usize, usize)>,
    ) -> Result<std::collections::BTreeMap<(usize, usize), String>, String> {
        let mut found = std::collections::BTreeMap::new();
        let entries = std::fs::read_dir(dir).map_err(|e| format!("read {}: {e}", dir.display()))?;
        for entry in entries {
            let name = entry
                .map_err(|e| e.to_string())?
                .file_name()
                .to_string_lossy()
                .into_owned();
            if let Some(key) = parse(&name)
                && let Some(other) = found.insert(key, name.clone())
            {
                return Err(format!(
                    "{} holds two row files for the depth {} at {} percent: {other} and {name}",
                    dir.display(),
                    key.0,
                    key.1
                ));
            }
        }
        Ok(found)
    }
    let control_files = files_by_prompt(control, parse_logits_file_name)?;
    let candidate_files = files_by_prompt(candidate, parse_logits_file_name)?;
    let control_windows = files_by_prompt(control, parse_window_file_name)?;
    let candidate_windows = files_by_prompt(candidate, parse_window_file_name)?;
    let prompts: std::collections::BTreeSet<(usize, usize)> = control_files
        .keys()
        .chain(candidate_files.keys())
        .copied()
        .collect();
    let mut rows = Vec::new();
    for key in prompts {
        let (depth, percent) = key;
        let (Some(control_name), Some(candidate_name)) =
            (control_files.get(&key), candidate_files.get(&key))
        else {
            let missing = if control_files.contains_key(&key) {
                "candidate"
            } else {
                "control"
            };
            rows.push(json!({"depth": depth, "needle_percent": percent, "missing": missing}));
            continue;
        };
        let read = |dir: &Path, name: &str| -> Result<Vec<f32>, String> {
            let bytes = std::fs::read(dir.join(name)).map_err(|e| format!("read {name}: {e}"))?;
            log_probs_from_bytes(&bytes).map_err(|e| format!("{name}: {e}"))
        };
        let (control_row, candidate_row) = (
            read(control, control_name)?,
            read(candidate, candidate_name)?,
        );
        let comparison = compare_log_probs(&control_row, &candidate_row)
            .map_err(|e| format!("depth {depth} at {percent} percent: {e}"))?;
        let mut row = json!({
            "depth": depth,
            "needle_percent": percent,
            "prompt_sha256": hashes.get(&key).ok_or_else(|| {
                format!("depth {depth} at {percent} percent: a row file has no row in the reports")
            })?,
            "control_file": control_name,
            "candidate_file": candidate_name,
            "argmax_equal": comparison.argmax_equal,
            "top_overlap": comparison.top_overlap,
            "top_count": comparison.top_count,
            "kl_nats": comparison.kl_nats,
            "max_top_logprob_diff": comparison.max_top_logprob_diff,
        });
        // The teacher-forced window of the same prompt, when both runs wrote one: the rows of the
        // last positions of the prompt, each as wide as the final-position row.
        if let (Some(control_window), Some(candidate_window)) =
            (control_windows.get(&key), candidate_windows.get(&key))
        {
            let summary = summarize_window(
                &read(control, control_window)?,
                &read(candidate, candidate_window)?,
                control_row.len(),
            )
            .map_err(|e| format!("depth {depth} at {percent} percent window: {e}"))?;
            row["window"] = json!({
                "control_file": control_window,
                "candidate_file": candidate_window,
                "rows": summary.rows,
                "mean_kl_nats": summary.mean_kl_nats,
                "p90_kl_nats": summary.p90_kl_nats,
                "max_kl_nats": summary.max_kl_nats,
                "argmax_agreement": summary.argmax_agreement,
                "mean_top_overlap": summary.mean_top_overlap,
            });
        } else if control_windows.contains_key(&key) != candidate_windows.contains_key(&key) {
            row["window_missing"] = json!(if control_windows.contains_key(&key) {
                "candidate"
            } else {
                "control"
            });
        }
        rows.push(row);
    }
    if !rows.iter().any(|row| row.get("kl_nats").is_some()) {
        return Err("the directories hold no paired final-position rows".to_string());
    }
    Ok(json!({
        "mode": mode.label(),
        "control": comparable.control,
        "candidate": comparable.candidate,
        "rows": rows,
    }))
}

/// Compares the final-position rows of two `long_context_gate` runs. It reads files only, so it
/// runs on any host.
#[test]
#[ignore = "needs the output directories of two long_context_gate runs; run with --ignored"]
fn long_context_compare() {
    let var = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("set {name}"));
    let (control, candidate) = (var("LC_COMPARE_CONTROL"), var("LC_COMPARE_CANDIDATE"));
    let out = var("Q36_OUT_DIR");
    assert!(
        std::fs::symlink_metadata(&out).is_err(),
        "Q36_OUT_DIR must not exist yet: {out}"
    );
    let noise = std::env::var("LC_COMPARE_NOISE").is_ok_and(|value| value == "1");
    let build = std::env::var("LC_COMPARE_BUILD").is_ok_and(|value| value == "1");
    let mode = compare_mode_from_flags(noise, build).unwrap_or_else(|error| panic!("{error}"));
    let report = compare_directories(Path::new(&control), Path::new(&candidate), mode)
        .unwrap_or_else(|error| panic!("{error}"));
    std::fs::create_dir(&out).unwrap_or_else(|error| panic!("create Q36_OUT_DIR {out}: {error}"));
    let text = serde_json::to_string_pretty(&report).unwrap();
    std::fs::write(Path::new(&out).join("compare.json"), &text).unwrap();
    println!("{text}");
}

/// The SHA-256 of `bytes` as lower-case hex.
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

/// The prompts of a run, with the tokenizer that built them.
struct PromptSet {
    bundle: crate::tokenizer::TokenizerBundle,
    tools: Vec<crate::tokenizer::ToolSpec>,
    files: Vec<HaystackFile>,
    /// The paths left out of the haystack because they hold the needle sentence.
    left_out: Vec<String>,
    /// The lines of one pass over the haystack files; a prompt with more lines repeats them.
    cycle_lines: usize,
    /// One prompt per depth and needle position, depth-major.
    prompts: Vec<PlannedPrompt>,
}

/// One built prompt with the depth and the needle position it was built for.
struct PlannedPrompt {
    /// The target token count.
    depth: usize,
    /// The needle position in percent of the haystack lines.
    percent: usize,
    built: BuiltPrompt,
}

/// The bytes that identify a haystack: each path and text, each followed by a zero byte.
fn haystack_bytes(files: &[HaystackFile]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for file in files {
        bytes.extend_from_slice(file.path.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(file.text.as_bytes());
        bytes.push(0);
    }
    bytes
}

/// Builds the prompt of every depth at every needle position with the pinned tokenizer and chat
/// template of the checkpoint (thinking off), on the CPU. `progress` receives one line per prompt.
fn build_prompts(
    tokenizer_dir: &Path,
    haystack_dir: &Path,
    depths: &[usize],
    percents: &[usize],
    kind: PromptKind,
    progress: &mut dyn FnMut(String),
) -> Result<PromptSet, String> {
    use crate::tokenizer::{PromptMessage, Role, Thinking, TokenizerBundle, ToolSpec};
    let (files, left_out) = read_haystack_dir(haystack_dir)?;
    let haystack = Haystack::new(&files)?;
    let pin = super::super::settings::tokenizer_pin().map_err(|error| format!("{error:?}"))?;
    let bundle =
        TokenizerBundle::load(tokenizer_dir, &pin).map_err(|error| format!("{error:?}"))?;
    let tools: Vec<ToolSpec> = tool_definitions().into_iter().map(ToolSpec).collect();
    let message = |role: Role, text: &str| PromptMessage {
        role,
        content: Some(text.to_string()),
        reasoning_content: None,
        name: None,
        tool_call_id: None,
        tool_calls: Vec::new(),
    };
    let mut encode = |text: &str| -> Result<Vec<u32>, String> {
        match kind {
            PromptKind::Task => {
                let messages = [
                    message(Role::System, SYSTEM_TEXT),
                    message(Role::User, text),
                ];
                bundle
                    .encode_prompt(&messages, &tools, Thinking::Off)
                    .map_err(|error| format!("{error:?}"))
            }
            PromptKind::Text => bundle
                .tokenizer()
                .encode(text)
                .map_err(|error| format!("{error:?}")),
        }
    };
    // The count of `<|im_start|>` ids in a task prompt is 3 (system, user, assistant) and in a text
    // prompt 0. Another count means that the template changed or that the haystack injected a
    // structural token, and the measurement would not be the one its labels name.
    let im_start = bundle.tokenizer().token_id("<|im_start|>");
    // Every added token of the tokenizer sits at or above `<|endoftext|>`. The ids of a prompt
    // that are that high must be exactly the ones the template writes: the count of a prompt
    // without haystack lines for a task, none for a text prompt. A haystack that spells any
    // added token (not only the ten of `CONTROL_STRINGS`) raises the count and is refused.
    let special_floor = bundle.tokenizer().token_id("<|endoftext|>");
    let special_baseline = match (kind, special_floor) {
        (PromptKind::Task, Some(floor)) => {
            Some(count_at_least(&encode(&haystack.user_text(0, 0))?, floor))
        }
        (PromptKind::Text, Some(_)) => Some(0),
        (_, None) => None,
    };
    let expected_starts = match kind {
        PromptKind::Task => 3,
        PromptKind::Text => 0,
    };
    let mut prompts = Vec::new();
    for &depth in depths {
        for &percent in percents {
            let started = std::time::Instant::now();
            let built = build_prompt(&haystack, depth, percent as f64 / 100.0, kind, &mut encode)
                .map_err(|error| format!("depth {depth} at {percent} percent: {error}"))?;
            let starts = built.ids.iter().filter(|&&id| Some(id) == im_start).count();
            if starts != expected_starts {
                return Err(format!(
                    "depth {depth} at {percent} percent: the prompt holds {starts} <|im_start|> \
                     ids, expected {expected_starts}"
                ));
            }
            if let (Some(floor), Some(baseline)) = (special_floor, special_baseline) {
                let specials = count_at_least(&built.ids, floor);
                if specials != baseline {
                    return Err(format!(
                        "depth {depth} at {percent} percent: the prompt holds {specials} added-token \
                         ids, expected {baseline}: the haystack spells an added token"
                    ));
                }
            }
            progress(format!(
                "long-context prompt: target {depth}, needle at {percent} percent, {} tokens, \
                 {} lines, needle before line {}, built in {:.1} s",
                built.ids.len(),
                built.lines,
                built.needle_line,
                started.elapsed().as_secs_f64()
            ));
            prompts.push(PlannedPrompt {
                depth,
                percent,
                built,
            });
        }
    }
    Ok(PromptSet {
        bundle,
        tools,
        files,
        left_out,
        cycle_lines: haystack.cycle.len(),
        prompts,
    })
}

/// The needle positions of `LC_NEEDLE_PERCENTS`, or the middle of the haystack when it is unset.
fn percents_from_env() -> Vec<usize> {
    let text = match std::env::var("LC_NEEDLE_PERCENTS") {
        Ok(text) => text,
        Err(std::env::VarError::NotPresent) => "50".to_string(),
        Err(error) => panic!("LC_NEEDLE_PERCENTS: {error}"),
    };
    parse_percent_list(&text).unwrap_or_else(|error| panic!("{error}"))
}

/// The prompt kind of `LC_PROMPT`: `task` unless it names `text`.
fn prompt_kind_from_env() -> PromptKind {
    match std::env::var("LC_PROMPT") {
        Ok(text) => parse_prompt_kind(&text).unwrap_or_else(|error| panic!("{error}")),
        Err(std::env::VarError::NotPresent) => PromptKind::Task,
        Err(error) => panic!("LC_PROMPT: {error}"),
    }
}

/// The depths of `LC_DEPTHS`, or the default list when it is unset.
fn depths_from_env() -> Vec<usize> {
    let text = match std::env::var("LC_DEPTHS") {
        Ok(text) => text,
        Err(std::env::VarError::NotPresent) => "32768,131072".to_string(),
        Err(error) => panic!("LC_DEPTHS: {error}"),
    };
    parse_depth_list(&text).unwrap_or_else(|error| panic!("{error}"))
}

/// The dry run of the prompt build: it needs only the tokenizer files, the haystack directory,
/// and `LC_DEPTHS`, loads no model, and runs on any host. It prints, per depth, the real token
/// count and the build time, and the first and last tokens of the prompt as text, so the template
/// shape (the end of the generation prompt with thinking off) is visible before a long run.
#[test]
#[ignore = "needs the tokenizer files of the checkpoint; run with --ignored"]
fn long_context_prompts() {
    let var = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("set {name}"));
    let tokenizer_dir = std::path::PathBuf::from(var("LC_TOKENIZER_DIR"));
    let haystack_dir = std::path::PathBuf::from(var("LC_HAYSTACK_DIR"));
    let depths = depths_from_env();
    let kind = prompt_kind_from_env();
    let percents = match kind {
        PromptKind::Task => percents_from_env(),
        PromptKind::Text => vec![0],
    };
    let set = build_prompts(
        &tokenizer_dir,
        &haystack_dir,
        &depths,
        &percents,
        kind,
        &mut |line| println!("{line}"),
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let tokenizer = set.bundle.tokenizer();
    for planned in &set.prompts {
        let built = &planned.built;
        let head = tokenizer.decode(&built.ids[..24]).unwrap_or_default();
        let tail = tokenizer
            .decode(&built.ids[built.ids.len() - 24..])
            .unwrap_or_default();
        // The checks the template path needs: the prompt fits its target and is not far below
        // it, and a task prompt ends in the empty think block of the thinking-off generation
        // prompt.
        assert!(
            built.ids.len() <= planned.depth,
            "the prompt exceeds its target"
        );
        assert!(
            built.ids.len() + 300 > planned.depth,
            "the prompt is more than 300 tokens below its target"
        );
        if kind == PromptKind::Task {
            assert!(
                tail.ends_with("<think>\n\n</think>\n\n"),
                "the task prompt does not end in the empty think block: {tail:?}"
            );
        }
        println!(
            "prompt of {} tokens:\n  head: {head:?}\n  tail: {tail:?}",
            built.ids.len()
        );
        // The ids can be written for a reference run in another stack: little-endian `u32`.
        if let Ok(dir) = std::env::var("LC_DUMP_DIR") {
            std::fs::create_dir_all(&dir).unwrap();
            let name = format!("ids_{}_p{}.u32", planned.depth, planned.percent);
            let bytes: Vec<u8> = built.ids.iter().flat_map(|id| id.to_le_bytes()).collect();
            std::fs::write(Path::new(&dir).join(name), bytes).unwrap();
        }
    }
    println!(
        "haystack: {} files, {} left out for the needle, sha256 {}",
        set.files.len(),
        set.left_out.len(),
        sha256_hex(&haystack_bytes(&set.files))
    );
}

/// The Mac harness of the gate. It reuses the serving prefill of `tool_forward` and the load of
/// `decode_timing`, so the prefill chunk, the worker piece, and the memory plan are the ones the
/// server uses, except for the levers of the table above.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod mac {
    use super::super::super::settings::{KvDtype, chunked_prefill_allowed, kv_dtype_name};
    use super::super::chunk_plan::{ChunkCost, PrefillTerms, admitted_chunk};
    use super::super::decode_timing::required_activation_from_env;
    use super::super::decoder::{ResidentDecoder, StepMode};
    use super::super::parity::{CONFIG_SHA256, sha256};
    use super::super::real::{env_path, open, parse_kv_dtype, rss_kb};
    use super::super::state::{ResidentState, kv_growth_bytes, kv_storage_bytes};
    use super::super::tool_forward::{
        CACHE_LIMIT, FORMAT_TOKEN_ID, NO_BUDGET_PIECE, PREFILL_BUDGET, PREFILL_RESERVE,
        SESSION_CAP, ServeStep, create_out_dir, out_dir_must_be_new, serve_chunk_setting,
    };
    use super::super::within_mode::env_number;
    use super::*;
    use crate::{
        EngError,
        family::{qwen36::host_load::HostWeights, session::DecodeSession, spec::QWEN3_6_35B_A3B},
        kernels::MlxKernels,
        sched::SessionBudget,
        tokenizer::{TokenizerBundle, ToolSpec},
    };
    use std::time::Instant;

    /// Greedy steps that read back the whole row: the first two, for the format probes.
    const ROW_STEPS: usize = 2;
    /// The steps before the timed window of the decode speed.
    const SPEED_SKIP: usize = 4;
    /// The fewest timed steps for which a decode speed is reported; a shorter answer leaves it null.
    const MIN_TIMED_STEPS: usize = 8;

    /// Reads the REQUIRED `Q36_KV_DTYPE`: there is no default, so a run states its cache form.
    fn required_kv_dtype_from_env() -> KvDtype {
        match std::env::var("Q36_KV_DTYPE") {
            Ok(text) => parse_kv_dtype(Some(&text)).unwrap_or_else(|error| panic!("{error}")),
            Err(std::env::VarError::NotPresent) => panic!("set Q36_KV_DTYPE to f32 or q8"),
            Err(error) => panic!("Q36_KV_DTYPE: {error}"),
        }
    }

    /// True for the typed errors of the memory plan and the over-length refusal: a depth that
    /// hits one is a result row, not a failed run.
    fn is_refusal(error: &EngError) -> bool {
        matches!(
            error,
            EngError::MemoryPlan(_) | EngError::ContextLimit(_) | EngError::PromptTooLong { .. }
        )
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn ms(elapsed: std::time::Duration) -> f64 {
        elapsed.as_secs_f64() * 1000.0
    }

    /// What one run holds across its depths.
    struct Run<'a, 'c> {
        decoder: &'a mut ResidentDecoder<'c>,
        state: &'a mut ResidentState,
        bundle: &'a TokenizerBundle,
        tools: &'a [ToolSpec],
        kind: PromptKind,
        /// The prefill chunk rows, or `None` for one-token steps.
        chunk: Option<u32>,
        /// The worker piece: the rows per `DecodeSession::prefill` call.
        piece: usize,
        context_limit: u32,
        decode_tokens: usize,
        /// The positions at the end of the prompt that run as teacher-forced one-token steps
        /// whose rows are kept (zero: the whole prompt is prefilled in chunks).
        window: usize,
        /// The forced answer to feed instead of the greedy decode (task prompts only).
        forced: Option<ForcedAnswer>,
        out: &'a Path,
        kv_name: &'static str,
        /// True until the first depth of the process has run: its timings carry one-time costs.
        cold: bool,
    }

    /// The memory fields of a row: the MLX allocator's active, peak (since the start of the
    /// depth), and cache bytes, and the resident set of the process.
    fn memory_row(run: &mut Run<'_, '_>) -> serde_json::Value {
        let memory = run.decoder.kernels.memory_stats().unwrap();
        json!({
            "active_bytes": memory.active,
            "peak_bytes_since_depth_start": memory.peak,
            "allocator_cache_bytes": memory.cache,
            "process_rss_kb": rss_kb(),
        })
    }

    /// The cache fields of a row, sampled at the end of the depth.
    fn cache_row(run: &Run<'_, '_>) -> serde_json::Value {
        let position = run.state.pos();
        let logical = kv_growth_bytes(&run.decoder.weights)
            .unwrap()
            .checked_mul(u64::from(position))
            .unwrap();
        json!({
            "position": position,
            "logical_bytes": logical,
            "stored_bytes": kv_storage_bytes(run.state.layers()),
            "state_bytes": run.state.byte_len().unwrap(),
        })
    }

    /// The log-softmax row of a logits row, or the count of values that are not finite.
    fn log_probs(logits: &[f32]) -> Result<Vec<f32>, usize> {
        log_softmax(logits).map_err(|_| logits.iter().filter(|value| !value.is_finite()).count())
    }

    /// One depth: prefill, the teacher-forced window, and the greedy decode or the forced answer,
    /// and the report row. A memory refusal or a logits row that is not finite gives a row with
    /// a `refused` or a `non_finite_logits` member and no verdict.
    fn run_depth(run: &mut Run<'_, '_>, planned: &PlannedPrompt) -> serde_json::Value {
        let (depth, percent, built) = (planned.depth, planned.percent, &planned.built);
        let ids_bytes: Vec<u8> = built.ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let window = run.window.min(built.ids.len().saturating_sub(2));
        let head = built.ids.len() - window;
        let mut row = json!({
            "depth_target": depth,
            "needle_percent": percent,
            "prompt_kind": format!("{:?}", run.kind),
            "prompt_tokens": built.ids.len(),
            "haystack_lines": built.lines,
            "needle_line": built.needle_line,
            "prompt_sha256": sha256_hex(&ids_bytes),
            "cold": run.cold,
            "verdict_path": if window > 0 {
                "chunked prefill of the head, then the last window tokens as teacher-forced one-token steps"
            } else {
                "chunked prefill of all but the last token; the last token is the first decode step"
            },
        });
        run.cold = false;
        run.decoder.reset(run.state).unwrap();
        if let Some(rows) = run.chunk {
            run.decoder.set_chunk_rows(rows).unwrap();
        }
        run.decoder.kernels.reset_peak_memory().unwrap();
        let limit = usize::try_from(run.context_limit).unwrap();
        let mut session = DecodeSession::default();
        let mut fed = Vec::new();
        let started = Instant::now();
        let prefill = {
            let mut step = ServeStep {
                decoder: &mut *run.decoder,
                state: &mut *run.state,
                chunk: run.chunk,
                fed: &mut fed,
            };
            built.ids[..head]
                .chunks(run.piece)
                .try_for_each(|rows| session.prefill(&mut step, rows, limit))
        };
        let prefill_seconds = started.elapsed().as_secs_f64();
        if let Err(error) = prefill {
            assert!(
                is_refusal(&error),
                "depth {depth}: prefill failed: {error:?}"
            );
            row["refused"] = json!(format!("{error:?}"));
            row["prefill_seconds_before_refusal"] = json!(prefill_seconds);
            row["memory"] = memory_row(run);
            return row;
        }
        let mut chunk_rows = fed.clone();
        chunk_rows.sort_unstable();
        chunk_rows.dedup();
        // The last id of the head stays pending in the session, so the model has fed one row
        // fewer than the head holds.
        let rows_fed: usize = fed.iter().sum();
        row["prefill"] = json!({
            "seconds": prefill_seconds,
            "rows_fed": rows_fed,
            "tokens_per_second": rows_fed as f64 / prefill_seconds,
            "distinct_chunk_rows": chunk_rows,
            "calls": fed.len(),
        });

        // The teacher-forced window: each pair feeds an id and keeps the row that predicts the
        // next prompt id. The row of the final prompt position is the first step after it.
        let mut input = session.pending().unwrap();
        let pairs = window_pairs(&built.ids, head, window);
        // The prefill fed `head - 1` rows and the pending id is the last id of the head: the
        // first window pair feeds exactly that id.
        assert_eq!(
            usize::try_from(run.state.pos()).unwrap(),
            head - 1,
            "the chunked prefill fed the head but one id"
        );
        assert_eq!(
            input,
            built.ids[head - 1],
            "the pending id is the last id of the head"
        );
        assert!(
            window == 0 || pairs[0].0 == input,
            "the first window pair must feed the pending id"
        );
        let mut window_rows: Vec<f32> = Vec::new();
        let window_started = Instant::now();
        for (index, &(fed_id, predicted)) in pairs.iter().enumerate() {
            let result = match run.decoder.step(run.state, fed_id, StepMode::Logits) {
                Ok(result) => result,
                Err(error) if is_refusal(&error) => {
                    row["refused"] = json!(format!("window: {error:?}"));
                    row["memory"] = memory_row(run);
                    return row;
                }
                Err(error) => panic!("depth {depth}: window step failed: {error:?}"),
            };
            match log_probs(&result.logits.unwrap()) {
                Ok(lp) => window_rows.extend(lp),
                Err(count) => {
                    row["non_finite_logits"] =
                        json!({"where": "window", "step": index, "values": count});
                    row["memory"] = memory_row(run);
                    return row;
                }
            }
            input = predicted;
        }
        if window > 0 {
            let name = window_file_name(depth, percent, run.kv_name);
            std::fs::write(run.out.join(&name), log_probs_to_bytes(&window_rows)).unwrap();
            row["window"] = json!({
                "tokens": window,
                "file": name,
                "seconds": window_started.elapsed().as_secs_f64(),
            });
        }

        let tokenizer = run.bundle.tokenizer();
        let tool_call = tokenizer.token_id("<tool_call>");
        let probability = |lp: &[f32], id: u32| {
            usize::try_from(id)
                .ok()
                .and_then(|index| lp.get(index))
                .map(|value| f64::from(*value).exp())
        };
        let forced = run.forced.clone();
        let mut generated: Vec<u32> = Vec::new();
        let mut step_ms: Vec<f64> = Vec::new();
        let mut rows: Vec<Vec<f32>> = Vec::new();
        let mut stop = "limit".to_string();
        if let Some(forced) = &forced {
            // The forced answer: feed the reference tool call and keep the log-probability of
            // every token. The negative log-likelihood of the passphrase tokens measures the
            // retrieval without the near tie of the first token (code fence or tool tag).
            let mut token_lp = Vec::new();
            let mut matches = 0usize;
            let mut first_row_done = false;
            for &target in &forced.ids {
                let result = match run.decoder.step(run.state, input, StepMode::Logits) {
                    Ok(result) => result,
                    Err(error) if is_refusal(&error) => {
                        row["refused"] = json!(format!("forced answer: {error:?}"));
                        row["memory"] = memory_row(run);
                        return row;
                    }
                    Err(error) => panic!("depth {depth}: forced step failed: {error:?}"),
                };
                let logits = result.logits.unwrap();
                let lp = match log_probs(&logits) {
                    Ok(lp) => lp,
                    Err(count) => {
                        row["non_finite_logits"] = json!({"where": "forced answer", "step": token_lp.len(), "values": count});
                        row["memory"] = memory_row(run);
                        return row;
                    }
                };
                if !first_row_done {
                    rows.push(lp.clone());
                    first_row_done = true;
                }
                token_lp.push(f64::from(lp[usize::try_from(target).unwrap()]));
                matches += usize::from(top_k(&lp, 1)[0].0 == usize::try_from(target).unwrap());
                input = target;
            }
            let nll = |range: std::ops::Range<usize>| -> f64 {
                token_lp[range].iter().map(|value| -value).sum()
            };
            row["forced"] = json!({
                "tokens": forced.ids.len(),
                "nll_total": nll(0..token_lp.len()),
                "nll_passphrase": nll(forced.prefix_len..forced.body_len),
                "passphrase_tokens": forced.body_len - forced.prefix_len,
                "min_passphrase_token_probability": token_lp[forced.prefix_len..forced.body_len]
                    .iter()
                    .map(|value| value.exp())
                    .fold(f64::INFINITY, f64::min),
                "argmax_matches": matches,
                "token_log_probabilities": token_lp,
            });
        } else {
            // The greedy decode. The first two steps read back the whole row: the first is the
            // row of the final prompt position, the second the row after the first generated
            // token. A text prompt takes the first step only.
            let eos = run.bundle.eos_token_ids().to_vec();
            let steps = if run.kind == PromptKind::Text {
                1
            } else {
                run.decode_tokens
            };
            for index in 0..steps {
                let mode = if index < ROW_STEPS {
                    StepMode::Logits
                } else {
                    StepMode::Generate
                };
                let step_started = Instant::now();
                let result = match run.decoder.step(run.state, input, mode) {
                    Ok(result) => result,
                    Err(error) if is_refusal(&error) => {
                        stop = format!("refused: {error:?}");
                        break;
                    }
                    Err(error) => panic!("depth {depth}: decode step {index} failed: {error:?}"),
                };
                step_ms.push(ms(step_started.elapsed()));
                let next = if index < ROW_STEPS {
                    let logits = result.logits.unwrap();
                    let id = u32::try_from(top_k(&logits, 1)[0].0).unwrap();
                    match log_probs(&logits) {
                        Ok(lp) => rows.push(lp),
                        Err(count) => {
                            row["non_finite_logits"] =
                                json!({"where": "decode", "step": index, "values": count});
                            row["memory"] = memory_row(run);
                            return row;
                        }
                    }
                    id
                } else {
                    result.token.unwrap()
                };
                if eos.contains(&next) {
                    stop = "eos".to_string();
                    break;
                }
                generated.push(next);
                input = next;
            }
            if run.kind == PromptKind::Task {
                let (text, decode_error) = match tokenizer.decode(&generated) {
                    Ok(text) => (text, None),
                    Err(error) => (String::new(), Some(format!("{error:?}"))),
                };
                let timed = step_ms.get(SPEED_SKIP..).unwrap_or(&[]);
                let p50 = median(timed);
                let enough = timed.len() >= MIN_TIMED_STEPS;
                row["decode"] = json!({
                    "tokens": generated.len(),
                    "stop": stop,
                    "generated_ids": generated,
                    "decode_error": decode_error,
                    "first_step_ms": step_ms.first(),
                    "p50_step_ms_after_4": p50,
                    "mean_step_ms_after_4": (!timed.is_empty())
                        .then(|| timed.iter().sum::<f64>() / timed.len() as f64),
                    "max_step_ms_after_4": timed.iter().copied().reduce(f64::max),
                    "timed_steps": timed.len(),
                    "tokens_per_second": p50.filter(|_| enough).map(|value| 1000.0 / value),
                });
                row["generated_text"] = json!(text);
                // The verdicts mean something only for an answer that ended: a refused or cut
                // answer gets none.
                let finished = stop == "eos" && decode_error.is_none();
                let verdict = judge_text(&text, run.tools);
                row["greedy_text_contains_passphrase"] = json!(text.contains(&passphrase()));
                row["needle_pass"] = json!(finished.then_some(verdict.needle_pass));
                row["format_pass"] =
                    json!(finished.then_some(verdict.calls >= 1 && verdict.malformed == 0));
                row["tool_calls"] = json!(verdict.calls);
                row["malformed_calls"] = json!(verdict.malformed);
            }
        }
        if let Some(first) = rows.first() {
            let first_is_tool_call = tool_call.is_some() && generated.first().copied() == tool_call;
            // A forced answer starts with the tool tag by construction, so the two fields that
            // depend on the first generated token are not measured there.
            let measured = forced.is_none();
            row["format_token"] = json!({
                "tool_call_id": tool_call,
                "p_tool_call_at_final_position": tool_call.and_then(|id| probability(first, id)),
                "first_generated_is_tool_call": measured.then_some(first_is_tool_call),
                "newline_id": FORMAT_TOKEN_ID,
                // The row after the first generated token is the format token of `tool_forward`
                // only when that token is the tool tag; otherwise it is null.
                "p_newline_after_first_generated": rows
                    .get(1)
                    .filter(|_| measured && first_is_tool_call)
                    .and_then(|r| probability(r, FORMAT_TOKEN_ID)),
            });
            row["top5"] = json!(
                top_k(first, TOP)
                    .into_iter()
                    .map(|(id, value)| json!({
                        "id": id,
                        "probability": f64::from(value).exp(),
                        "log_probability": value,
                        "text": u32::try_from(id).ok().and_then(|id| tokenizer.decode(&[id]).ok()).unwrap_or_default(),
                    }))
                    .collect::<Vec<_>>()
            );
            let name = logits_file_name(depth, percent, run.kv_name);
            std::fs::write(run.out.join(&name), log_probs_to_bytes(first)).unwrap();
            row["logits_file"] = json!(name);
        }
        // The cache and the memory at the end of the depth, after the window and the decode.
        row["cache"] = cache_row(run);
        row["memory"] = memory_row(run);
        row
    }

    /// The long-context gate: the needle, the tool-call format, memory, and speed of one cache
    /// form at the depths of `LC_DEPTHS`.
    #[test]
    #[ignore = "needs the real checkpoint on the Mac; run with --ignored"]
    fn long_context_gate() {
        let _serial = crate::native_test_lock();
        // Every input is validated first. The output directory is created last, right before
        // the load, so a configuration error leaves no empty directory that blocks the rerun.
        let ckpt_dir = env_path("Q36_CKPT_DIR");
        let tokenizer_dir = std::env::var_os("LC_TOKENIZER_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| ckpt_dir.clone());
        let haystack_dir = env_path("LC_HAYSTACK_DIR");
        let activation = required_activation_from_env();
        let kv = required_kv_dtype_from_env();
        let kv_name = kv_dtype_name(kv);
        let depths = depths_from_env();
        let decode_tokens = env_number("LC_DECODE_TOKENS", 64, 8, 512);
        let kind = prompt_kind_from_env();
        let percents = match kind {
            PromptKind::Task => percents_from_env(),
            PromptKind::Text => vec![0],
        };
        let force = env_number("LC_FORCE", 0, 0, 1) == 1;
        assert!(
            !(force && kind == PromptKind::Text),
            "LC_FORCE needs the task prompt: the forced answer is a tool call"
        );
        let deepest = depths.iter().copied().max().unwrap();
        let context_limit = u32::try_from(env_number(
            "LC_CONTEXT_LIMIT",
            deepest + decode_tokens + 16,
            deepest + decode_tokens + 1,
            u32::MAX as usize,
        ))
        .unwrap();
        let session_cap = u64::try_from(env_number(
            "LC_SESSION_CAP",
            usize::try_from(SESSION_CAP).unwrap(),
            1 << 20,
            1 << 40,
        ))
        .unwrap();
        let chunk_override = env_number("LC_CHUNK", 0, 0, 2_048);
        // A task prompt takes its verdicts from the whole-prompt serving path unless asked for a
        // window; a text prompt exists to measure the window, so it takes 128 positions.
        let window = env_number(
            "LC_WINDOW",
            match kind {
                PromptKind::Task => 0,
                PromptKind::Text => 128,
            },
            0,
            512,
        );
        assert!(
            kind == PromptKind::Task || window > 0,
            "a text prompt needs a window: LC_WINDOW above zero"
        );

        // The prompts, built on the CPU before the load.
        let set = build_prompts(
            &tokenizer_dir,
            &haystack_dir,
            &depths,
            &percents,
            kind,
            &mut |line| println!("{line}"),
        )
        .unwrap_or_else(|error| panic!("{error}"));

        let out = out_dir_must_be_new();
        let out_dir = create_out_dir(&out);

        // The load of the server: the pinned checkpoint, the shipped cache limit, the session
        // cap (a lever here), and the product constructor for the cache form.
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
        let mut decoder =
            ResidentDecoder::new_with_kv(kernels, &real.ckpt, host, context_limit, activation, kv)
                .unwrap();
        decoder.budget = SessionBudget::new(session_cap);
        let mut state = decoder.zero_state().unwrap();
        let routing = format!("{:?}", decoder.routing);

        let terms = PrefillTerms {
            budget: PREFILL_BUDGET,
            weights: decoder.weights.byte_len().unwrap(),
            session_reserve: PREFILL_RESERVE,
        };
        let cost = ChunkCost::new(heads_q, context_limit).unwrap();
        let serve_setting = serve_chunk_setting(activation);
        let chunked = chunked_prefill_allowed(decoder.weights.consts.act);
        let planned = if chunked {
            admitted_chunk(&cost, terms.remaining(), serve_setting).unwrap()
        } else {
            None
        };
        let (chunk, chunk_source) = if chunk_override > 0 {
            (Some(u32::try_from(chunk_override).unwrap()), "override")
        } else {
            (planned, "plan")
        };
        assert!(
            chunk.is_some() || deepest <= 4_096,
            "the plan admits no prefill chunk at the context limit {context_limit}; one-token \
             steps would take hours at depth {deepest}. Set LC_CHUNK to measure with chunks."
        );
        let piece = chunk
            .map(|rows| usize::try_from(rows).unwrap())
            .unwrap_or(NO_BUDGET_PIECE);
        println!(
            "long-context plan: activation {activation:?}, kv {kv_name}, context limit \
             {context_limit}, session cap {session_cap}, chunk {chunk:?} ({chunk_source}), planned \
             {planned:?}, piece {piece}, depths {depths:?}; {}",
            terms.to_string().lines().collect::<Vec<_>>().join(" ")
        );

        let mut report = json!({
            "settings": {
                "activation": format!("{activation:?}"),
                "kv_dtype": kv_name,
                "context_limit": context_limit,
                "session_cap_bytes": session_cap,
                "shipped_session_cap_bytes": SESSION_CAP,
                "decode_tokens": decode_tokens,
                "needle_percents": percents,
                "prompt_kind": format!("{kind:?}"),
                "window": window,
                "force": force,
                "chunk": chunk,
                "chunk_source": chunk_source,
                "planned_chunk": planned,
                "piece": piece,
                "routing": routing,
                "cache_limit_bytes": CACHE_LIMIT,
                "prefill_terms": terms.to_string().lines().collect::<Vec<_>>(),
            },
            "checkpoint": {"config_sha256": CONFIG_SHA256},
            "tokenizer": {"tokenizer_json_sha256": hex(&set.bundle.identity().tokenizer_json)},
            "haystack": {
                "files": set.files.len(),
                "bytes": set.files.iter().map(|file| file.text.len()).sum::<usize>(),
                "sha256": sha256_hex(&haystack_bytes(&set.files)),
                "cycle_lines": set.cycle_lines,
                "left_out_for_the_needle_or_a_control_string": set.left_out,
            },
            "rows": [],
        });
        // The forced answer is computed once, before the load: a tokenizer under which the call
        // prefix is no prefix of the answer would give no passphrase span, and the run stops
        // here instead of falling back to a greedy decode that the settings do not name.
        let forced = force.then(|| {
            forced_answer(&|text| set.bundle.tokenizer().encode(text).ok())
                .expect("the forced answer has no passphrase span under this tokenizer")
        });
        let mut run = Run {
            decoder: &mut decoder,
            state: &mut state,
            bundle: &set.bundle,
            tools: &set.tools,
            kind,
            chunk,
            piece,
            context_limit,
            decode_tokens,
            window,
            forced,
            out: &out_dir,
            kv_name,
            cold: true,
        };
        for planned in &set.prompts {
            let row = run_depth(&mut run, planned);
            println!(
                "long-context depth {} at {} percent: {}",
                planned.depth,
                planned.percent,
                serde_json::to_string(&row).unwrap()
            );
            report["rows"].as_array_mut().unwrap().push(row);
            // The report is rewritten after every depth, so a later failure keeps the earlier rows.
            std::fs::write(
                out_dir.join("long_context.json"),
                serde_json::to_string_pretty(&report).unwrap(),
            )
            .unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::ToolSpec;

    /// A tokenizer for the tests: the prompt is 5 fixed words plus one token per whitespace word
    /// of the user message. The id of a token is a hash of its word.
    fn fake_encode(user_text: &str) -> Result<Vec<u32>, String> {
        let mut ids: Vec<u32> = (0..5).collect();
        ids.extend(user_text.split_whitespace().map(|word| {
            word.bytes().fold(2_166_136_261u32, |h, b| {
                (h ^ u32::from(b)).wrapping_mul(16_777_619)
            })
        }));
        Ok(ids)
    }

    fn files() -> Vec<HaystackFile> {
        vec![
            HaystackFile {
                path: "a.rs".to_string(),
                text: "fn a() {}\nfn b() {}\n".to_string(),
            },
            HaystackFile {
                path: "b.rs".to_string(),
                text: "struct S;\n".to_string(),
            },
        ]
    }

    /// The cycle holds, for each file, its header and its lines: 3 + 2 = 5 lines.
    #[test]
    fn the_cycle_lists_each_file_as_a_header_and_its_lines() {
        let haystack = Haystack::new(&files()).unwrap();
        assert_eq!(
            haystack.cycle,
            [
                "=== a.rs ===",
                "fn a() {}",
                "fn b() {}",
                "=== b.rs ===",
                "struct S;"
            ]
        );
    }

    /// Seven lines of a five-line cycle repeat the first two lines. The needle stands before
    /// haystack line 3 (index 3 of 7 lines at a fraction of 0.5 is 3).
    #[test]
    fn the_haystack_repeats_in_order_and_the_needle_stands_at_a_line_boundary() {
        let haystack = Haystack::new(&files()).unwrap();
        let text = haystack.user_text(7, needle_line_for(7, 0.5));
        let body: Vec<&str> = text.lines().skip(2).collect();
        assert_eq!(
            &body[..8],
            [
                "=== a.rs ===",
                "fn a() {}",
                "fn b() {}",
                "The release passphrase for service orion-7 is quasar-lantern-4412.",
                "=== b.rs ===",
                "struct S;",
                "=== a.rs ===",
                "fn a() {}",
            ]
        );
        assert_eq!(text.matches(&needle_sentence()).count(), 1);
    }

    #[test]
    fn the_needle_goes_after_the_last_line_when_the_fraction_is_one() {
        assert_eq!(needle_line_for(10, 1.0), 10);
        assert_eq!(needle_line_for(10, 0.0), 0);
        assert_eq!(needle_line_for(10, 0.25), 2);
        let haystack = Haystack::new(&files()).unwrap();
        let text = haystack.user_text(3, 3);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines[lines.len() - 3],
            "The release passphrase for service orion-7 is quasar-lantern-4412."
        );
    }

    #[test]
    fn a_file_that_holds_the_needle_sentence_is_refused() {
        let mut list = files();
        list[1].text = format!("// {}\n", needle_sentence());
        let error = Haystack::new(&list).err().unwrap();
        assert!(error.contains("b.rs"), "{error}");
        assert!(Haystack::new(&[]).is_err());
        let empty = [HaystackFile {
            path: String::new(),
            text: String::new(),
        }];
        assert!(Haystack::new(&empty).is_ok(), "a header line is a line");
    }

    /// The prompt for a target of 100 tokens has the most lines that fit: one line more exceeds it.
    #[test]
    fn the_prompt_has_the_most_lines_that_fit_the_target() {
        let haystack = Haystack::new(&files()).unwrap();
        let mut encode = fake_encode;
        for target in [60, 100, 333] {
            let built =
                build_prompt(&haystack, target, 0.5, PromptKind::Task, &mut encode).unwrap();
            assert!(built.ids.len() <= target, "{} > {target}", built.ids.len());
            let one_more = fake_encode(
                &haystack.user_text(built.lines + 1, needle_line_for(built.lines + 1, 0.5)),
            )
            .unwrap();
            assert!(one_more.len() > target, "a further line still fits");
            assert_eq!(
                built.ids,
                fake_encode(&haystack.user_text(built.lines, built.needle_line)).unwrap()
            );
            assert_eq!(built.needle_line, needle_line_for(built.lines, 0.5));
        }
    }

    /// The needle position follows the fraction: at 10 and at 90 percent the same haystack gives
    /// the same line count and different needle lines, and the ids differ.
    #[test]
    fn the_needle_position_follows_the_fraction() {
        let haystack = Haystack::new(&files()).unwrap();
        let mut encode = fake_encode;
        let early = build_prompt(&haystack, 200, 0.1, PromptKind::Task, &mut encode).unwrap();
        let late = build_prompt(&haystack, 200, 0.9, PromptKind::Task, &mut encode).unwrap();
        assert_eq!(early.lines, late.lines);
        assert!(early.needle_line < late.needle_line);
        assert_ne!(early.ids, late.ids);
    }

    /// Files of five lines cycle for a target that needs many more lines.
    #[test]
    fn short_files_cycle_for_a_long_target() {
        let haystack = Haystack::new(&files()).unwrap();
        let mut encode = fake_encode;
        let built = build_prompt(&haystack, 500, 0.5, PromptKind::Task, &mut encode).unwrap();
        assert!(built.lines > haystack.cycle.len() * 5, "{}", built.lines);
        let text = haystack.user_text(built.lines, built.needle_line);
        assert_eq!(
            text.matches("=== a.rs ===").count(),
            built.lines.div_ceil(5)
        );
        assert_eq!(text.matches(&needle_sentence()).count(), 1);
    }

    #[test]
    fn the_needle_stands_near_the_requested_fraction_of_the_lines() {
        let haystack = Haystack::new(&files()).unwrap();
        let mut encode = fake_encode;
        for (fraction, target) in [(0.1, 400), (0.5, 400), (0.9, 400)] {
            let built =
                build_prompt(&haystack, target, fraction, PromptKind::Task, &mut encode).unwrap();
            let text = haystack.user_text(built.lines, built.needle_line);
            let body: Vec<&str> = text.lines().skip(2).collect();
            let at = body
                .iter()
                .position(|line| *line == needle_sentence())
                .unwrap();
            let want = fraction * built.lines as f64;
            assert!((at as f64 - want).abs() < 1.0, "{at} against {want}");
        }
    }

    #[test]
    fn a_target_below_the_empty_prompt_and_a_bad_fraction_are_refused() {
        let haystack = Haystack::new(&files()).unwrap();
        let mut encode = fake_encode;
        let error = build_prompt(&haystack, 10, 0.5, PromptKind::Task, &mut encode)
            .err()
            .unwrap();
        assert!(error.contains("below"), "{error}");
        assert!(build_prompt(&haystack, 500, 1.5, PromptKind::Task, &mut encode).is_err());
        assert!(build_prompt(&haystack, 500, -0.1, PromptKind::Task, &mut encode).is_err());
    }

    #[test]
    fn an_encoder_error_stops_the_build() {
        let haystack = Haystack::new(&files()).unwrap();
        let mut encode =
            |_: &str| -> Result<Vec<u32>, String> { Err("template failed".to_string()) };
        assert_eq!(
            build_prompt(&haystack, 500, 0.5, PromptKind::Task, &mut encode)
                .err()
                .unwrap(),
            "template failed"
        );
    }

    #[test]
    fn the_depth_list_takes_positive_distinct_depths_of_at_least_1024() {
        assert_eq!(
            parse_depth_list("32768, 131072").unwrap(),
            [32_768, 131_072]
        );
        assert_eq!(parse_depth_list("262000").unwrap(), [262_000]);
        for bad in ["", "1000", "a", "2048,2048", "2048,,4096", "-5"] {
            assert!(parse_depth_list(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_median_is_the_middle_value_or_the_mean_of_the_middle_two() {
        assert_eq!(median(&[]), None);
        assert_eq!(median(&[3.0]), Some(3.0));
        assert_eq!(median(&[9.0, 1.0, 5.0]), Some(5.0));
        assert_eq!(median(&[4.0, 1.0, 2.0, 10.0]), Some(3.0));
    }

    /// Logits 0, ln 2, ln 4, ln 8 give probabilities 1/15, 2/15, 4/15, 8/15.
    #[test]
    fn the_log_softmax_gives_the_logs_of_the_probabilities() {
        let row = [0.0f32, 2.0f32.ln(), 4.0f32.ln(), 8.0f32.ln()];
        let out = log_softmax(&row).unwrap();
        for (value, want) in out.iter().zip([1.0, 2.0, 4.0, 8.0]) {
            let want = (want / 15.0f64).ln();
            assert!((f64::from(*value) - want).abs() < 1e-6, "{value} {want}");
        }
        // A shift of every logit changes nothing.
        let shifted: Vec<f32> = row.iter().map(|v| v + 1000.0).collect();
        let again = log_softmax(&shifted).unwrap();
        for (a, b) in out.iter().zip(&again) {
            assert!((a - b).abs() < 2e-4, "{a} {b}");
        }
        assert!(log_softmax(&[]).is_err());
        assert!(log_softmax(&[1.0, f32::NAN]).is_err());
        assert!(log_softmax(&[1.0, f32::INFINITY]).is_err());
    }

    #[test]
    fn the_top_tokens_come_largest_first_and_a_tie_keeps_the_lower_id() {
        let row = [0.5, 3.0, 3.0, -1.0, 2.0];
        assert_eq!(top_k(&row, 3), [(1, 3.0), (2, 3.0), (4, 2.0)]);
        assert_eq!(top_k(&row, 9).len(), 5);
    }

    fn logs(probabilities: &[f64]) -> Vec<f32> {
        probabilities.iter().map(|p| p.ln() as f32).collect()
    }

    /// Identical rows: no divergence, the same top tokens.
    #[test]
    fn identical_rows_have_no_divergence() {
        let row = logs(&[0.4, 0.3, 0.1, 0.1, 0.05, 0.03, 0.02]);
        let comparison = compare_log_probs(&row, &row).unwrap();
        assert!(comparison.argmax_equal);
        assert_eq!(comparison.top_overlap, 5);
        assert_eq!(comparison.top_count, 5);
        assert_eq!(comparison.kl_nats, 0.0);
        assert_eq!(comparison.max_top_logprob_diff, 0.0);
    }

    /// P = (0.7, 0.2, 0.1) and Q = (0.4, 0.4, 0.2). By hand:
    /// KL(P||Q) = 0.7 ln 1.75 + 0.2 ln 0.5 + 0.1 ln 0.5 = 0.1837869 (KL(Q||P) is 0.1920420).
    /// The control tokens 0 to 2 are all in the candidate; the largest log difference is
    /// |ln 0.2 - ln 0.4| = |ln 0.1 - ln 0.2| = ln 2 = 0.6931472 against |ln 0.7 - ln 0.4| = 0.5596158.
    #[test]
    fn the_divergence_runs_from_the_control_to_the_candidate() {
        let control = logs(&[0.7, 0.2, 0.1]);
        let candidate = logs(&[0.4, 0.4, 0.2]);
        let comparison = compare_log_probs(&control, &candidate).unwrap();
        assert!(
            (comparison.kl_nats - 0.183_786_9).abs() < 2e-6,
            "{comparison:?}"
        );
        assert!(
            (comparison.max_top_logprob_diff - std::f64::consts::LN_2).abs() < 2e-6,
            "{comparison:?}"
        );
        assert_eq!(comparison.top_count, 3);
        assert_eq!(comparison.top_overlap, 3);
        assert!(comparison.argmax_equal);
        let reverse = compare_log_probs(&candidate, &control).unwrap();
        assert!((reverse.kl_nats - 0.192_042_0).abs() < 2e-6, "{reverse:?}");
    }

    /// P = (0.7, 0.2, 0.1) and R = (0.3, 0.5, 0.2): token 1 is the candidate's largest. By hand:
    /// KL(P||R) = 0.7 ln(7/3) + 0.2 ln 0.4 + 0.1 ln 0.5 = 0.3405356. The largest log difference is
    /// |ln 0.2 - ln 0.5| = 0.9162907.
    #[test]
    fn a_changed_largest_token_shows_in_the_argmax_flag() {
        let control = logs(&[0.7, 0.2, 0.1]);
        let candidate = logs(&[0.3, 0.5, 0.2]);
        let comparison = compare_log_probs(&control, &candidate).unwrap();
        assert!(!comparison.argmax_equal);
        assert_eq!(comparison.top_overlap, 3);
        assert!(
            (comparison.kl_nats - 0.340_535_6).abs() < 2e-6,
            "{comparison:?}"
        );
        assert!(
            (comparison.max_top_logprob_diff - 0.916_290_7).abs() < 2e-6,
            "{comparison:?}"
        );
    }

    /// Six tokens: the control's top five are 0 to 4; the candidate swaps token 4 for token 5.
    #[test]
    fn the_overlap_counts_the_control_top_tokens_that_the_candidate_keeps() {
        let control = logs(&[0.30, 0.25, 0.2, 0.1, 0.09, 0.06]);
        let candidate = logs(&[0.30, 0.25, 0.2, 0.1, 0.06, 0.09]);
        let comparison = compare_log_probs(&control, &candidate).unwrap();
        assert_eq!(comparison.top_overlap, 4);
        assert_eq!(comparison.top_count, 5);
        assert!(comparison.argmax_equal);
    }

    #[test]
    fn rows_of_unequal_or_zero_width_are_refused() {
        assert!(compare_log_probs(&[0.0], &[0.0, -1.0]).is_err());
        assert!(compare_log_probs(&[], &[]).is_err());
    }

    #[test]
    fn a_row_round_trips_through_its_file_bytes() {
        let row = [0.0f32, -1.5, -22.25, f32::MIN_POSITIVE];
        let bytes = log_probs_to_bytes(&row);
        assert_eq!(bytes.len(), 16);
        assert_eq!(&bytes[4..8], (-1.5f32).to_le_bytes());
        assert_eq!(log_probs_from_bytes(&bytes).unwrap(), row);
        assert!(log_probs_from_bytes(&bytes[..15]).is_err());
    }

    #[test]
    fn the_logits_file_name_carries_the_depth_the_needle_position_and_the_cache_form() {
        assert_eq!(
            logits_file_name(32_768, 50, "q8"),
            "logits_32768_p50_q8.f32"
        );
        assert_eq!(
            parse_logits_file_name("logits_32768_p50_q8.f32"),
            Some((32_768, 50))
        );
        assert_eq!(
            parse_logits_file_name("logits_131072_p0_f32.f32"),
            Some((131_072, 0))
        );
        for other in [
            "long_context.json",
            "logits_x_p50_q8.f32",
            "logits_1024_q8.f32",
            "logits_1024_p50_.f32",
            "logits_1024_50_q8.f32",
            "logits_1024_pq_q8.f32",
            "logits_1024_p50_q8_extra.f32",
            "logits_1024_p50_q8.bin",
        ] {
            assert_eq!(parse_logits_file_name(other), None, "{other}");
        }
    }

    #[test]
    fn the_percent_list_takes_distinct_integers_from_0_to_100() {
        assert_eq!(parse_percent_list("50").unwrap(), [50]);
        assert_eq!(
            parse_percent_list("10, 90,0,100").unwrap(),
            [10, 90, 0, 100]
        );
        for bad in ["", "101", "a", "10,10", "10,,20", "-1", "5.5"] {
            assert!(parse_percent_list(bad).is_err(), "{bad:?}");
        }
    }

    /// Files read from a directory come in sorted path order, nested paths use `/`, a file that
    /// is not Rust is skipped, and a file that holds the needle sentence is left out and named.
    #[test]
    fn the_haystack_directory_is_read_in_sorted_order_without_the_needle_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("z.rs"), "fn z() {}\n").unwrap();
        std::fs::write(dir.path().join("sub/m.rs"), "fn m() {}\n").unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn a() {}\n").unwrap();
        std::fs::write(dir.path().join("notes.md"), "not rust\n").unwrap();
        std::fs::write(
            dir.path().join("needle.rs"),
            format!("// {}\n", needle_sentence()),
        )
        .unwrap();
        let (files, left_out) = read_haystack_dir(dir.path()).unwrap();
        let paths: Vec<&str> = files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, ["a.rs", "sub/m.rs", "z.rs"]);
        assert_eq!(left_out, ["needle.rs"]);
        assert_eq!(files[0].text, "fn a() {}\n");
        assert!(read_haystack_dir(&dir.path().join("missing")).is_err());
    }

    fn write_row(dir: &Path, name: &str, probabilities: &[f64]) {
        std::fs::write(dir.join(name), log_probs_to_bytes(&logs(probabilities))).unwrap();
    }

    /// The settings of a run report: cache form, activation, chunk, window, and prompt kind.
    fn settings(kv: &str, window: usize) -> serde_json::Value {
        json!({
            "kv_dtype": kv, "activation": "Bf16", "chunk": 256, "window": window,
            "prompt_kind": "Task",
        })
    }

    /// Writes the `long_context.json` of a run with the given settings and one row per (depth,
    /// needle percent, prompt hash).
    fn write_report_with(dir: &Path, settings: serde_json::Value, rows: &[(usize, usize, &str)]) {
        let rows: Vec<serde_json::Value> = rows
            .iter()
            .map(|&(depth, percent, hash)| {
                json!({"depth_target": depth, "needle_percent": percent, "prompt_sha256": hash})
            })
            .collect();
        let report = json!({
            "settings": settings,
            "checkpoint": {"config_sha256": "cfg"},
            "tokenizer": {"tokenizer_json_sha256": "tok"},
            "haystack": {"sha256": "hay"},
            "rows": rows,
        });
        std::fs::write(
            dir.join("long_context.json"),
            serde_json::to_string(&report).unwrap(),
        )
        .unwrap();
    }

    /// Writes the report of a run with the default settings of a cache form.
    fn write_report(dir: &Path, kv: &str, window: usize, rows: &[(usize, usize, &str)]) {
        write_report_with(dir, settings(kv, window), rows);
    }

    /// Two run directories with matching reports of the given prompts.
    fn two_runs(
        control_rows: &[(usize, usize, &str)],
        candidate_rows: &[(usize, usize, &str)],
        window: usize,
    ) -> (tempfile::TempDir, tempfile::TempDir) {
        let (control, candidate) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        write_report(control.path(), "f32", window, control_rows);
        write_report(candidate.path(), "q8", window, candidate_rows);
        (control, candidate)
    }

    /// Depth 1024 at 50 percent is in both directories (a swapped top pair), depth 1024 at 10
    /// percent is in both with equal rows, depth 2048 only in the control, and depth 4096 only in
    /// the candidate. The rows come in depth order, then needle position.
    #[test]
    fn the_directories_are_paired_by_prompt_and_a_lone_prompt_names_its_missing_side() {
        let (control, candidate) = two_runs(
            &[(1024, 10, "a"), (1024, 50, "b"), (2048, 50, "c")],
            &[(1024, 10, "a"), (1024, 50, "b"), (4096, 50, "d")],
            0,
        );
        write_row(control.path(), "logits_1024_p10_f32.f32", &[0.7, 0.2, 0.1]);
        write_row(candidate.path(), "logits_1024_p10_q8.f32", &[0.7, 0.2, 0.1]);
        write_row(control.path(), "logits_1024_p50_f32.f32", &[0.7, 0.2, 0.1]);
        write_row(candidate.path(), "logits_1024_p50_q8.f32", &[0.3, 0.5, 0.2]);
        write_row(control.path(), "logits_2048_p50_f32.f32", &[0.7, 0.2, 0.1]);
        write_row(candidate.path(), "logits_4096_p50_q8.f32", &[0.7, 0.2, 0.1]);
        let report =
            compare_directories(control.path(), candidate.path(), CompareMode::Cache).unwrap();
        let rows = report["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 4);
        assert_eq!(
            (&rows[0]["depth"], &rows[0]["needle_percent"]),
            (&json!(1024), &json!(10))
        );
        assert_eq!(rows[0]["prompt_sha256"], "a");
        assert_eq!(rows[0]["argmax_equal"], true);
        assert_eq!(rows[0]["kl_nats"].as_f64().unwrap(), 0.0);
        assert_eq!(
            (&rows[1]["depth"], &rows[1]["needle_percent"]),
            (&json!(1024), &json!(50))
        );
        assert_eq!(rows[1]["control_file"], "logits_1024_p50_f32.f32");
        assert_eq!(rows[1]["candidate_file"], "logits_1024_p50_q8.f32");
        assert_eq!(rows[1]["argmax_equal"], false);
        assert_eq!(rows[1]["top_overlap"], 3);
        assert!((rows[1]["kl_nats"].as_f64().unwrap() - 0.340_535_6).abs() < 2e-6);
        assert_eq!(
            rows[2],
            json!({"depth": 2048, "needle_percent": 50, "missing": "candidate"})
        );
        assert_eq!(
            rows[3],
            json!({"depth": 4096, "needle_percent": 50, "missing": "control"})
        );
    }

    #[test]
    fn two_row_files_of_one_prompt_in_one_directory_are_an_error() {
        let (control, candidate) = two_runs(&[(1024, 50, "a")], &[(1024, 50, "a")], 0);
        write_row(control.path(), "logits_1024_p50_f32.f32", &[0.5, 0.5]);
        write_row(control.path(), "logits_1024_p50_q8.f32", &[0.5, 0.5]);
        let error = compare_directories(control.path(), candidate.path(), CompareMode::Cache)
            .err()
            .unwrap();
        assert!(error.contains("two row files"), "{error}");
    }

    #[test]
    fn rows_of_different_width_and_unreadable_directories_are_errors() {
        let (control, candidate) = two_runs(&[(1024, 50, "a")], &[(1024, 50, "a")], 0);
        write_row(control.path(), "logits_1024_p50_f32.f32", &[0.5, 0.5]);
        write_row(candidate.path(), "logits_1024_p50_q8.f32", &[0.4, 0.3, 0.3]);
        let error = compare_directories(control.path(), candidate.path(), CompareMode::Cache)
            .err()
            .unwrap();
        assert!(
            error.contains("depth 1024 at 50 percent") && error.contains("differ in width"),
            "{error}"
        );
        assert!(
            compare_directories(
                &control.path().join("missing"),
                candidate.path(),
                CompareMode::Cache
            )
            .is_err()
        );
    }

    /// Compares two runs with one setting changed on the candidate side and returns the error.
    fn refusal_for(change: &str, value: serde_json::Value, noise: bool) -> Option<String> {
        let rows = [(1024, 50, "a")];
        let (control, candidate) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        write_report_with(control.path(), settings("f32", 0), &rows);
        let mut changed = settings(if noise { "f32" } else { "q8" }, 0);
        changed[change] = value;
        write_report_with(candidate.path(), changed, &rows);
        write_row(control.path(), "logits_1024_p50_f32.f32", &[0.5, 0.5]);
        write_row(candidate.path(), "logits_1024_p50_q8.f32", &[0.5, 0.5]);
        compare_directories(
            control.path(),
            candidate.path(),
            if noise {
                CompareMode::Noise
            } else {
                CompareMode::Cache
            },
        )
        .err()
    }

    /// Runs that differ in what they computed, share a cache form, built different prompts, or
    /// left no report cannot be compared, and a comparison with no paired prompt is no result.
    #[test]
    fn runs_that_are_not_comparable_are_refused() {
        let rows = [(1024, 50, "a")];
        // Each setting that a cache comparison holds equal.
        for (field, value) in [
            ("activation", json!("F32")),
            ("chunk", json!(128)),
            ("window", json!(48)),
            ("prompt_kind", json!("Text")),
        ] {
            let error = refusal_for(field, value, false).unwrap();
            assert!(error.contains(&format!("differ in {field}")), "{error}");
        }
        // Different prompt ids at one key.
        let (control, candidate) = two_runs(&rows, &[(1024, 50, "b")], 0);
        write_row(control.path(), "logits_1024_p50_f32.f32", &[0.5, 0.5]);
        write_row(candidate.path(), "logits_1024_p50_q8.f32", &[0.5, 0.5]);
        let error = compare_directories(control.path(), candidate.path(), CompareMode::Cache)
            .err()
            .unwrap();
        assert!(error.contains("different prompts"), "{error}");
        // The same cache form on both sides.
        let (control, candidate) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        write_report(control.path(), "q8", 0, &rows);
        write_report(candidate.path(), "q8", 0, &rows);
        let error = compare_directories(control.path(), candidate.path(), CompareMode::Cache)
            .err()
            .unwrap();
        assert!(error.contains("both runs use the cache form"), "{error}");
        // A directory without a report.
        let (control, candidate) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        write_report(control.path(), "f32", 0, &rows);
        let error = compare_directories(control.path(), candidate.path(), CompareMode::Cache)
            .err()
            .unwrap();
        assert!(error.contains("long_context.json"), "{error}");
        // Reports that pair no prompt with a row file: no result.
        let (control, candidate) = two_runs(&rows, &rows, 0);
        let error = compare_directories(control.path(), candidate.path(), CompareMode::Cache)
            .err()
            .unwrap();
        assert!(error.contains("no paired final-position rows"), "{error}");
        // A row file that no report row describes.
        let (control, candidate) = two_runs(&rows, &rows, 0);
        write_row(control.path(), "logits_2048_p50_f32.f32", &[0.5, 0.5]);
        write_row(candidate.path(), "logits_2048_p50_q8.f32", &[0.5, 0.5]);
        write_row(control.path(), "logits_1024_p50_f32.f32", &[0.5, 0.5]);
        write_row(candidate.path(), "logits_1024_p50_q8.f32", &[0.5, 0.5]);
        let error = compare_directories(control.path(), candidate.path(), CompareMode::Cache)
            .err()
            .unwrap();
        assert!(error.contains("no row in the reports"), "{error}");
    }

    /// The noise-floor mode compares two runs of ONE cache form that differ in the chunk; the
    /// strict mode refuses the same pair, and the noise mode refuses a pair that shares the chunk
    /// or differs in the cache form. The output names the mode and the settings of both runs.
    #[test]
    fn a_noise_floor_comparison_takes_one_cache_form_and_two_chunks() {
        let rows = [(1024, 50, "a")];
        let (control, candidate) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        write_report_with(control.path(), settings("f32", 0), &rows);
        let mut half = settings("f32", 0);
        half["chunk"] = json!(128);
        write_report_with(candidate.path(), half, &rows);
        write_row(control.path(), "logits_1024_p50_f32.f32", &[0.7, 0.2, 0.1]);
        write_row(
            candidate.path(),
            "logits_1024_p50_f32.f32",
            &[0.4, 0.4, 0.2],
        );
        let error = compare_directories(control.path(), candidate.path(), CompareMode::Cache)
            .err()
            .unwrap();
        assert!(error.contains("differ in chunk"), "{error}");
        let report =
            compare_directories(control.path(), candidate.path(), CompareMode::Noise).unwrap();
        assert_eq!(report["mode"], "noise floor");
        assert_eq!(report["control"]["chunk"], 256);
        assert_eq!(report["candidate"]["chunk"], 128);
        assert!((report["rows"][0]["kl_nats"].as_f64().unwrap() - 0.183_786_9).abs() < 2e-6);
        // The same chunk twice is no noise floor.
        let error = refusal_for("chunk", json!(256), true).unwrap();
        assert!(error.contains("needs two chunks"), "{error}");
        // Another cache form is not allowed either.
        let (control, candidate) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        write_report_with(control.path(), settings("f32", 0), &rows);
        let mut other = settings("q8", 0);
        other["chunk"] = json!(128);
        write_report_with(candidate.path(), other, &rows);
        let error = compare_directories(control.path(), candidate.path(), CompareMode::Noise)
            .err()
            .unwrap();
        assert!(error.contains("differ in kv_dtype"), "{error}");
    }

    /// The build mode compares two runs of one schedule and one cache form (an implementation
    /// A/B); the cache mode refuses the same pair, and the build mode refuses a settings
    /// difference or different prompts. The output names the mode.
    #[test]
    fn a_build_comparison_takes_one_schedule_and_one_cache_form() {
        let rows = [(1024, 50, "a")];
        let (control, candidate) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        write_report_with(control.path(), settings("q8", 0), &rows);
        write_report_with(candidate.path(), settings("q8", 0), &rows);
        write_row(control.path(), "logits_1024_p50_q8.f32", &[0.7, 0.2, 0.1]);
        write_row(candidate.path(), "logits_1024_p50_q8.f32", &[0.4, 0.4, 0.2]);
        let error = compare_directories(control.path(), candidate.path(), CompareMode::Cache)
            .err()
            .unwrap();
        assert!(error.contains("both runs use the cache form"), "{error}");
        let report =
            compare_directories(control.path(), candidate.path(), CompareMode::Build).unwrap();
        assert_eq!(report["mode"], "build");
        assert_eq!(report["control"]["kv_dtype"], "q8");
        assert_eq!(report["candidate"]["kv_dtype"], "q8");
        // Both rows argmax to the first id (a tie keeps the lower id), so the flag stays set.
        assert_eq!(report["rows"][0]["argmax_equal"], true);
        assert!((report["rows"][0]["kl_nats"].as_f64().unwrap() - 0.183_786_9).abs() < 2e-6);
        // A settings difference refuses the pair.
        let (control, candidate) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        write_report_with(control.path(), settings("q8", 0), &rows);
        let mut other = settings("q8", 0);
        other["chunk"] = json!(128);
        write_report_with(candidate.path(), other, &rows);
        let error = compare_directories(control.path(), candidate.path(), CompareMode::Build)
            .err()
            .unwrap();
        assert!(error.contains("differ in chunk"), "{error}");
        // Different prompt ids at one key refuse the pair.
        let (control, candidate) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        write_report_with(control.path(), settings("q8", 0), &rows);
        write_report_with(candidate.path(), settings("q8", 0), &[(1024, 50, "b")]);
        let error = compare_directories(control.path(), candidate.path(), CompareMode::Build)
            .err()
            .unwrap();
        assert!(error.contains("different prompts"), "{error}");
    }

    /// Every setting the build mode holds equal must exist in both reports: an absent field is
    /// a refusal even when both reports lack it.
    #[test]
    fn a_build_comparison_refuses_a_missing_setting() {
        let rows = [(1024, 50, "a")];
        let mut partial = settings("q8", 0);
        partial.as_object_mut().unwrap().remove("window");
        for (control_settings, candidate_settings) in [
            (partial.clone(), settings("q8", 0)),
            (partial.clone(), partial.clone()),
        ] {
            let (control, candidate) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
            write_report_with(control.path(), control_settings, &rows);
            write_report_with(candidate.path(), candidate_settings, &rows);
            let error = compare_directories(control.path(), candidate.path(), CompareMode::Build)
                .err()
                .unwrap();
            assert!(error.contains("has no window setting"), "{error}");
        }
    }

    /// A setting that is present with the value null is not a missing setting: the report writer
    /// records `"chunk": null` for an unchunked run, and two unchunked runs of one schedule are
    /// comparable. A null chunk against a numeric chunk is a schedule difference.
    #[test]
    fn a_build_comparison_accepts_matching_null_chunks() {
        let rows = [(1024, 50, "a")];
        let mut unchunked = settings("q8", 0);
        unchunked["chunk"] = json!(null);
        let (control, candidate) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        write_report_with(control.path(), unchunked.clone(), &rows);
        write_report_with(candidate.path(), unchunked.clone(), &rows);
        write_row(control.path(), "logits_1024_p50_q8.f32", &[0.7, 0.2, 0.1]);
        write_row(candidate.path(), "logits_1024_p50_q8.f32", &[0.4, 0.4, 0.2]);
        let report =
            compare_directories(control.path(), candidate.path(), CompareMode::Build).unwrap();
        assert_eq!(report["mode"], "build");
        assert!(report["control"]["chunk"].is_null());
        assert!(report["candidate"]["chunk"].is_null());
        // One side unchunked and the other chunked is a different schedule.
        let (control, candidate) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        write_report_with(control.path(), unchunked, &rows);
        write_report_with(candidate.path(), settings("q8", 0), &rows);
        let error = compare_directories(control.path(), candidate.path(), CompareMode::Build)
            .err()
            .unwrap();
        assert!(error.contains("differ in chunk"), "{error}");
    }

    /// A build comparison holds the cache form equal: two cache forms are a cache comparison,
    /// and the named-field check refuses them in build mode with the cache form in its message.
    #[test]
    fn a_build_comparison_refuses_a_different_cache_form() {
        let rows = [(1024, 50, "a")];
        let (control, candidate) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        write_report_with(control.path(), settings("q8", 0), &rows);
        write_report_with(candidate.path(), settings("f32", 0), &rows);
        let error = compare_directories(control.path(), candidate.path(), CompareMode::Build)
            .err()
            .unwrap();
        assert!(error.contains("differ in kv_dtype"), "{error}");
    }

    /// A build comparison needs the whole settings object equal: a different piece or routing
    /// is a schedule difference even when the five named fields match.
    #[test]
    fn a_build_comparison_refuses_any_settings_difference() {
        let rows = [(1024, 50, "a")];
        for (field, value) in [("piece", json!(128)), ("routing", json!("Host"))] {
            let (control, candidate) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
            write_report_with(control.path(), settings("q8", 0), &rows);
            let mut changed = settings("q8", 0);
            changed[field] = value;
            write_report_with(candidate.path(), changed, &rows);
            let error = compare_directories(control.path(), candidate.path(), CompareMode::Build)
                .err()
                .unwrap();
            assert!(error.contains("differ in settings"), "{error}");
        }
    }

    /// A build comparison needs the model, tokenizer, and haystack identity of both runs:
    /// every identity field must exist in both reports and be equal.
    #[test]
    fn a_build_comparison_needs_the_run_identity() {
        let rows = [(1024, 50, "a")];
        let rewrite = |dir: &Path, change: &dyn Fn(&mut serde_json::Value)| {
            let report_path = dir.join("long_context.json");
            let mut report: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&report_path).unwrap()).unwrap();
            change(&mut report);
            std::fs::write(&report_path, serde_json::to_string(&report).unwrap()).unwrap();
        };
        for (section, field) in [
            ("checkpoint", "config_sha256"),
            ("tokenizer", "tokenizer_json_sha256"),
            ("haystack", "sha256"),
        ] {
            let (control, candidate) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
            write_report_with(control.path(), settings("q8", 0), &rows);
            write_report_with(candidate.path(), settings("q8", 0), &rows);
            rewrite(candidate.path(), &|report| {
                report[section][field] = json!("different")
            });
            let error = compare_directories(control.path(), candidate.path(), CompareMode::Build)
                .err()
                .unwrap();
            assert!(error.contains(section), "{error}");
        }
        // A missing identity section is a refusal, not an absent proof.
        let (control, candidate) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        write_report_with(control.path(), settings("q8", 0), &rows);
        write_report_with(candidate.path(), settings("q8", 0), &rows);
        rewrite(candidate.path(), &|report| {
            report.as_object_mut().unwrap().remove("checkpoint");
        });
        let error = compare_directories(control.path(), candidate.path(), CompareMode::Build)
            .err()
            .unwrap();
        assert!(error.contains("checkpoint"), "{error}");
    }

    /// An identity field that both reports lack is a refusal as well: null equals null, so the
    /// equality check alone would accept two runs whose model, tokenizer, or haystack is unknown.
    #[test]
    fn a_build_comparison_refuses_an_identity_missing_from_both_runs() {
        let rows = [(1024, 50, "a")];
        for (section, field) in [
            ("checkpoint", "config_sha256"),
            ("tokenizer", "tokenizer_json_sha256"),
            ("haystack", "sha256"),
        ] {
            let (control, candidate) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
            for dir in [control.path(), candidate.path()] {
                write_report_with(dir, settings("q8", 0), &rows);
                let report_path = dir.join("long_context.json");
                let mut report: serde_json::Value =
                    serde_json::from_str(&std::fs::read_to_string(&report_path).unwrap()).unwrap();
                report.as_object_mut().unwrap().remove(section);
                std::fs::write(&report_path, serde_json::to_string(&report).unwrap()).unwrap();
            }
            let error = compare_directories(control.path(), candidate.path(), CompareMode::Build)
                .err()
                .unwrap();
            assert!(
                error.contains(&format!("has no {section}.{field} identity")),
                "{error}"
            );
        }
    }

    /// The mode flags pick one mode; both set at once is ambiguous.
    #[test]
    fn the_mode_flags_pick_one_mode() {
        assert_eq!(
            compare_mode_from_flags(false, false).unwrap(),
            CompareMode::Cache
        );
        assert_eq!(
            compare_mode_from_flags(true, false).unwrap(),
            CompareMode::Noise
        );
        assert_eq!(
            compare_mode_from_flags(false, true).unwrap(),
            CompareMode::Build
        );
        let error = compare_mode_from_flags(true, true).unwrap_err();
        assert!(error.contains("both set"), "{error}");
    }

    /// An id is an added token when it sits at or above the floor.
    #[test]
    fn the_added_token_count_uses_the_floor() {
        assert_eq!(count_at_least(&[1, 2, 100, 101, 99, 100], 100), 3);
        assert_eq!(count_at_least(&[], 100), 0);
        assert_eq!(count_at_least(&[5, 6], 100), 0);
    }

    /// A row of width 8 whose top five tokens agree but whose tail differs: the KL runs over the
    /// whole row, so it is the tail's 0.0223805 (by hand, in nats: 0.05 ln(5/2) + 0.03 ln(3/5) +
    /// 0.02 ln(2/3)), while the top-five overlap is 5 and the top log-probability gap is 0.
    #[test]
    fn the_divergence_runs_over_the_whole_row_not_the_top_tokens() {
        let control = logs(&[0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.03, 0.02]);
        let candidate = logs(&[0.4, 0.2, 0.1, 0.1, 0.1, 0.02, 0.05, 0.03]);
        let comparison = compare_log_probs(&control, &candidate).unwrap();
        assert!(
            (comparison.kl_nats - 0.022_380_5).abs() < 2e-6,
            "{comparison:?}"
        );
        assert_eq!((comparison.top_overlap, comparison.top_count), (5, 5));
        assert!(comparison.max_top_logprob_diff < 1e-7);
    }

    /// Two positions of width 8: equal, and one whose candidate swaps the fifth token for the
    /// sixth (control top five ids 0 to 4, candidate top five ids 0 to 3 and 5). The mean top-5
    /// overlap is (5 + 4) / 2 = 4.5.
    #[test]
    fn the_window_overlap_is_the_mean_of_the_position_overlaps() {
        let control_row = logs(&[0.3, 0.2, 0.15, 0.12, 0.1, 0.06, 0.04, 0.03]);
        let swapped = logs(&[0.3, 0.2, 0.15, 0.12, 0.05, 0.1, 0.04, 0.04]);
        let control: Vec<f32> = [&control_row, &control_row]
            .into_iter()
            .flatten()
            .copied()
            .collect();
        let candidate: Vec<f32> = [&control_row, &swapped]
            .into_iter()
            .flatten()
            .copied()
            .collect();
        let summary = summarize_window(&control, &candidate, 8).unwrap();
        assert!(
            (summary.mean_top_overlap - 4.5).abs() < 1e-12,
            "{summary:?}"
        );
        assert_eq!(summary.argmax_agreement, 1.0);
    }

    #[test]
    fn the_window_file_name_carries_the_same_key_as_the_logits_file() {
        assert_eq!(
            window_file_name(32_768, 50, "q8"),
            "window_32768_p50_q8.f32"
        );
        assert_eq!(
            parse_window_file_name("window_32768_p50_q8.f32"),
            Some((32_768, 50))
        );
        assert_eq!(parse_window_file_name("logits_32768_p50_q8.f32"), None);
        assert_eq!(parse_logits_file_name("window_32768_p50_q8.f32"), None);
    }

    /// Three positions of width 3. Position 0: equal rows (KL 0). Position 1: P = (0.7, 0.2, 0.1)
    /// against R = (0.3, 0.5, 0.2): KL 0.3405356, a different largest token. Position 2: P
    /// against Q = (0.4, 0.4, 0.2): KL 0.1837869, the same largest token (the tie keeps the lower
    /// id). By hand: mean KL = (0 + 0.3405356 + 0.1837869) / 3 = 0.1747742; the nearest-rank 90th
    /// percentile of 3 values is the third (ceil(2.7) = 3), which is the largest; the argmax
    /// agreement is 2 / 3; every position overlaps in all 3 tokens.
    #[test]
    fn a_window_summary_averages_the_positions() {
        let p = logs(&[0.7, 0.2, 0.1]);
        let control: Vec<f32> = [&p, &p, &p].into_iter().flatten().copied().collect();
        let candidate: Vec<f32> = [&p, &logs(&[0.3, 0.5, 0.2]), &logs(&[0.4, 0.4, 0.2])]
            .into_iter()
            .flatten()
            .copied()
            .collect();
        let summary = summarize_window(&control, &candidate, 3).unwrap();
        assert_eq!(summary.rows, 3);
        assert!(
            (summary.mean_kl_nats - 0.174_774_2).abs() < 2e-6,
            "{summary:?}"
        );
        assert!(
            (summary.p90_kl_nats - 0.340_535_6).abs() < 2e-6,
            "{summary:?}"
        );
        assert!(
            (summary.max_kl_nats - 0.340_535_6).abs() < 2e-6,
            "{summary:?}"
        );
        assert!(
            (summary.argmax_agreement - 2.0 / 3.0).abs() < 1e-12,
            "{summary:?}"
        );
        assert!(
            (summary.mean_top_overlap - 3.0).abs() < 1e-12,
            "{summary:?}"
        );
    }

    /// Ten positions where only the last differs: the nearest-rank 90th percentile is the ninth
    /// value, which is 0, while the maximum is the tenth.
    #[test]
    fn the_window_percentile_is_the_nearest_rank() {
        let p = logs(&[0.7, 0.2, 0.1]);
        let control: Vec<f32> = (0..10).flat_map(|_| p.iter().copied()).collect();
        let mut candidate = control.clone();
        candidate[27..30].copy_from_slice(&logs(&[0.3, 0.5, 0.2]));
        let summary = summarize_window(&control, &candidate, 3).unwrap();
        assert_eq!(summary.p90_kl_nats, 0.0);
        assert!((summary.max_kl_nats - 0.340_535_6).abs() < 2e-6);
        assert!((summary.argmax_agreement - 0.9).abs() < 1e-12);
    }

    #[test]
    fn windows_of_unequal_size_or_that_are_not_whole_rows_are_refused() {
        let row = logs(&[0.5, 0.5]);
        assert!(summarize_window(&row, &[], 2).is_err());
        assert!(summarize_window(&row, &row, 0).is_err());
        assert!(summarize_window(&row[..1], &row[..1], 2).is_err());
        assert!(summarize_window(&[], &[], 2).is_err());
    }

    /// A window beside the final-position rows of both directories gets its summary in the row of
    /// its prompt; a prompt whose directories lack a window gets none; a window in one directory
    /// only is marked.
    #[test]
    fn the_comparison_adds_the_window_summary_of_a_prompt_that_both_runs_wrote() {
        let rows = [(1024, 50, "a"), (2048, 50, "b"), (4096, 50, "c")];
        let (control, candidate) = two_runs(&rows, &rows, 2);
        let (p, r) = (logs(&[0.7, 0.2, 0.1]), logs(&[0.3, 0.5, 0.2]));
        for depth in [1024, 2048, 4096] {
            write_row(
                control.path(),
                &format!("logits_{depth}_p50_f32.f32"),
                &[0.7, 0.2, 0.1],
            );
            write_row(
                candidate.path(),
                &format!("logits_{depth}_p50_q8.f32"),
                &[0.7, 0.2, 0.1],
            );
        }
        let window_rows = |parts: [&Vec<f32>; 2]| -> Vec<u8> {
            log_probs_to_bytes(&parts.into_iter().flatten().copied().collect::<Vec<f32>>())
        };
        std::fs::write(
            control.path().join("window_1024_p50_f32.f32"),
            window_rows([&p, &p]),
        )
        .unwrap();
        std::fs::write(
            candidate.path().join("window_1024_p50_q8.f32"),
            window_rows([&p, &r]),
        )
        .unwrap();
        // 4096 has a window in the candidate only.
        std::fs::write(
            candidate.path().join("window_4096_p50_q8.f32"),
            window_rows([&p, &p]),
        )
        .unwrap();
        let report =
            compare_directories(control.path(), candidate.path(), CompareMode::Cache).unwrap();
        let rows = report["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 3);
        let window = &rows[0]["window"];
        assert_eq!(window["rows"], 2);
        assert_eq!(window["control_file"], "window_1024_p50_f32.f32");
        assert!((window["mean_kl_nats"].as_f64().unwrap() - 0.170_267_8).abs() < 2e-6);
        assert_eq!(window["argmax_agreement"], 0.5);
        assert!(rows[1].get("window").is_none() && rows[1].get("window_missing").is_none());
        assert_eq!(rows[2]["window_missing"], "control");
    }

    /// The window pairs feed the id before each of the last ids and predict that id: with a head
    /// of 3 over 6 ids and a window of 3, the pairs are (ids[2], ids[3]), (ids[3], ids[4]), and
    /// (ids[4], ids[5]); the next id to feed afterwards is ids[5], the last prompt id.
    #[test]
    fn the_window_feeds_each_id_to_predict_the_next_one() {
        let ids = [10, 11, 12, 13, 14, 15];
        assert_eq!(window_pairs(&ids, 3, 3), [(12, 13), (13, 14), (14, 15)]);
        assert_eq!(window_pairs(&ids, 5, 1), [(14, 15)]);
        assert!(window_pairs(&ids, 6, 0).is_empty());
    }

    /// The forced answer: with an encoder that maps each word to one id, the answer text splits
    /// into words and the passphrase span follows the call prefix; an encoder under which the
    /// prefix encodings are not prefixes of the answer gives no span.
    #[test]
    fn the_forced_answer_finds_the_passphrase_span() {
        let words = |text: &str| -> Option<Vec<u32>> {
            Some(
                text.split_whitespace()
                    .map(|word| word.bytes().map(u32::from).sum::<u32>())
                    .collect(),
            )
        };
        let answer = forced_answer(&words).unwrap();
        // "<tool_call> <function=run> <parameter=cmd> echo QUASAR </parameter> </function>
        // </tool_call>" as whitespace words: 4 words up to `echo`, one passphrase word, then 3.
        assert_eq!(answer.ids.len(), 8);
        assert_eq!((answer.prefix_len, answer.body_len), (4, 5));
        // An encoder that tokenizes `echo` differently when the passphrase follows it (a merge
        // across the word boundary): the call prefix is then no prefix of the body, so the span
        // is unknown.
        let merging = |text: &str| -> Option<Vec<u32>> {
            let mut ids = words(text)?;
            if text.contains(&passphrase()) {
                let echo = words("echo")?[0];
                for id in &mut ids {
                    if *id == echo {
                        *id = 9_999;
                    }
                }
            }
            Some(ids)
        };
        assert_eq!(forced_answer(&merging), None);
        assert_eq!(forced_answer(&|_| None), None);
    }

    #[test]
    fn the_prompt_kind_is_task_or_text() {
        assert_eq!(parse_prompt_kind("task"), Ok(PromptKind::Task));
        assert_eq!(parse_prompt_kind("text"), Ok(PromptKind::Text));
        assert!(parse_prompt_kind("Task").is_err());
    }

    /// A text prompt is the haystack lines alone: no intro, no needle, no task.
    #[test]
    fn a_text_prompt_holds_only_haystack_lines() {
        let haystack = Haystack::new(&files()).unwrap();
        let mut seen = Vec::new();
        let mut encode = |text: &str| -> Result<Vec<u32>, String> {
            seen.push(text.to_string());
            Ok(vec![0; text.split_whitespace().count()])
        };
        let built = build_prompt(&haystack, 30, 0.5, PromptKind::Text, &mut encode).unwrap();
        assert!(built.ids.len() <= 30);
        let text = haystack.plain_text(built.lines);
        assert!(!text.contains(&needle_sentence()) && !text.contains("Task:"));
        assert_eq!(text.lines().count(), built.lines);
        assert_eq!(built.needle_line, 0);
        assert!(seen.iter().all(|t| !t.contains("These are the files")));
    }

    /// A file that spells a control string would inject a structural token into the user message:
    /// the cycle refuses it and the directory reader leaves it out.
    #[test]
    fn a_file_that_spells_a_control_string_is_left_out() {
        for control in CONTROL_STRINGS {
            let mut list = files();
            list[0].text = format!("let s = \"{control}\";\n");
            let error = Haystack::new(&list).err().unwrap();
            assert!(error.contains(control), "{error}");
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn a() {}\n").unwrap();
        std::fs::write(dir.path().join("b.rs"), "// <tool_call>\n").unwrap();
        let (kept, left_out) = read_haystack_dir(dir.path()).unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(left_out, ["b.rs"]);
    }

    /// The identity bytes keep file boundaries, so moving a byte from one file to the next
    /// changes them. The hash is the known SHA-256 of the empty input and of "abc".
    #[test]
    fn the_haystack_identity_keeps_the_file_boundaries() {
        let file = |path: &str, text: &str| HaystackFile {
            path: path.to_string(),
            text: text.to_string(),
        };
        assert_eq!(
            haystack_bytes(&[file("a.rs", "x"), file("b.rs", "yz")]),
            b"a.rs\0x\0b.rs\0yz\0"
        );
        assert_ne!(
            haystack_bytes(&[file("a.rs", "xy"), file("b.rs", "z")]),
            haystack_bytes(&[file("a.rs", "x"), file("b.rs", "yz")])
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    fn tool_specs() -> Vec<ToolSpec> {
        tool_definitions().into_iter().map(ToolSpec).collect()
    }

    const CALL: &str = "<tool_call>\n<function=run>\n<parameter=cmd>\necho quasar-lantern-4412\n\
</parameter>\n</function>\n</tool_call>";

    #[test]
    fn a_run_call_with_the_passphrase_passes_both_checks() {
        let verdict = judge_text(CALL, &tool_specs());
        assert_eq!(
            verdict,
            TextVerdict {
                calls: 1,
                malformed: 0,
                needle_pass: true
            }
        );
    }

    #[test]
    fn a_valid_call_without_the_passphrase_passes_the_format_only() {
        let text = CALL.replace("quasar-lantern-4412", "quasar-lantern-4413");
        let verdict = judge_text(&text, &tool_specs());
        assert_eq!((verdict.calls, verdict.malformed), (1, 0));
        assert!(!verdict.needle_pass);
        // The passphrase in another tool does not count.
        let other = CALL.replace("<function=run>", "<function=read_file>");
        assert!(!judge_text(&other, &tool_specs()).needle_pass);
    }

    /// The tool declares `cmd`: a `run` call that carries the passphrase in another parameter is
    /// a valid call tag but not a pass, since a real client would reject it.
    #[test]
    fn a_run_call_with_another_parameter_name_is_not_a_pass() {
        let text = CALL.replace("<parameter=cmd>", "<parameter=command>");
        let verdict = judge_text(&text, &tool_specs());
        assert_eq!((verdict.calls, verdict.malformed), (1, 0));
        assert!(!verdict.needle_pass);
    }

    #[test]
    fn plain_text_and_a_broken_call_pass_neither_check() {
        let plain = judge_text("The passphrase is quasar-lantern-4412.", &tool_specs());
        assert_eq!(
            (plain.calls, plain.malformed, plain.needle_pass),
            (0, 0, false)
        );
        let broken = judge_text(
            "<tool_call>\n<function=run>\n<parameter=cmd>\necho quasar-lantern-4412",
            &tool_specs(),
        );
        assert_eq!((broken.calls, broken.malformed), (0, 1));
        assert!(!broken.needle_pass);
    }

    #[test]
    fn the_tool_definitions_name_read_file_and_run() {
        let names: Vec<String> = tool_definitions()
            .iter()
            .map(|tool| tool["function"]["name"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(names, ["read_file", "run"]);
    }
}
