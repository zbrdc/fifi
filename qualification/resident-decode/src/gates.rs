//! Report parsing and gate rules for the resident decode qualification.
//!
//! The engine tests in `family/flash_next/resident/parity.rs` measure and write `key=value`
//! reports. This module is the only authority for the C5 gate thresholds. A counter that a build
//! does not have is written as `-`.

use std::collections::BTreeMap;

/// Generated tokens in the decode run.
pub const GENERATED: usize = 512;
/// `Prompt` steps before the first `Generate` step: the five-token prompt minus its last token.
pub const PROMPT_STEPS: usize = 4;
/// Leading generated tokens that must equal the host path exactly (decision 0039 gate 5).
pub const MATCH: usize = 32;
/// Generated tokens before the flat-memory window starts (decision 0039 gate 5).
pub const WARMUP: usize = 16;
/// Largest synchronization count of one generated token (decision 0039 gate 6).
pub const SYNC_MAX: u64 = 110;
/// Largest relative spread `(max - min) / min` of the per-token MLX peak in the window.
pub const FLAT_MAX: f64 = 0.05;
/// Teacher-forced check count (`fx/mlx.rs` line 479; the ledger correction of 2026-10-09).
pub const TEACHER_CHECKS: u64 = 1460;
/// Layer lines in the end-to-end report.
pub const LAYERS: usize = 48;
/// Host-path rate of the W5-MAC run (the ledger, first Rust run of Flash-Next).
pub const HOST_BASELINE_TOKENS_PER_SECOND: f64 = 1.33;
/// Column order of every `stepNNNN` line of the decode report.
pub const STEP_COLUMNS: &str =
    "mode,input,output,seconds,syncs,flushes,retired_peak,peak,active,cache,read_bytes,read_ns";

/// Decode report keys that the record copies unchanged.
const COPIED: [&str; 18] = [
    "generated_tokens",
    "cache_limit_bytes",
    "cache_limit_previous",
    "load_seconds",
    "import_seconds",
    "import_syncs",
    "zero_state_syncs",
    "weights_bytes",
    "state_bytes_start",
    "state_bytes_end",
    "expert_bytes_per_token",
    "rss_t16",
    "rss_t128",
    "rss_t256",
    "rss_t384",
    "rss_t512",
    "host_seconds",
    "host_margins",
];

/// The two builds of coordinator decision D10.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Build {
    /// `native-mlx,native-fault-qualification`: all three tests, with the counters.
    Q,
    /// `native-mlx` only: the decode test, for the headline rate.
    P,
}

impl Build {
    /// Parses `q` or `p`.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "q" => Some(Self::Q),
            "p" => Some(Self::P),
            _ => None,
        }
    }

    /// The file-name prefix and the decode report `build` value.
    pub fn name(self) -> &'static str {
        match self {
            Self::Q => "q",
            Self::P => "p",
        }
    }

    /// The engine features of the build.
    pub fn features(self) -> &'static str {
        match self {
            Self::Q => "native-mlx,native-fault-qualification",
            Self::P => "native-mlx",
        }
    }
}

/// One gate result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
    /// The build has no counter for the gate.
    Unavailable,
    /// Some required report values are unavailable.
    Partial,
}

impl Verdict {
    /// The record spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::Unavailable => "unavailable",
            Self::Partial => "partial",
        }
    }

    fn from_bool(pass: bool) -> Self {
        if pass { Self::Pass } else { Self::Fail }
    }
}

/// A parsed report. `keys` holds the `key=value` lines whose key has no space.
pub struct Report {
    keys: BTreeMap<String, String>,
    lines: Vec<String>,
}

impl Report {
    /// Parses report text. Lines such as `L00 t0 attn out rel=1e-6 cos=1.0` stay in `lines` only.
    pub fn parse(text: &str) -> Self {
        let mut keys = BTreeMap::new();
        for line in text.lines() {
            if let Some((key, value)) = line.split_once('=')
                && !key.contains(' ')
            {
                keys.insert(key.to_string(), value.to_string());
            }
        }
        Self {
            keys,
            lines: text.lines().map(str::to_string).collect(),
        }
    }

    /// The value of `key`.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.keys.get(key).map(String::as_str)
    }

    fn number(&self, key: &str) -> Option<u64> {
        self.get(key)?.parse().ok()
    }

    fn count_prefix(&self, prefix: &str) -> usize {
        self.lines.iter().filter(|l| l.starts_with(prefix)).count()
    }
}

/// One measured step of the decode report, in `STEP_COLUMNS` order.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub generate: bool,
    pub output: Option<u32>,
    pub seconds: f64,
    pub syncs: Option<u64>,
    pub flushes: Option<u64>,
    pub retired_peak: Option<u64>,
    pub peak: u64,
    pub active: u64,
    pub cache: u64,
    pub read_bytes: u64,
    pub read_ns: u64,
}

/// Parses a counter column: `-` is unavailable.
fn counter(text: &str) -> Result<Option<u64>, String> {
    if text == "-" {
        return Ok(None);
    }
    text.parse()
        .map(Some)
        .map_err(|_| format!("bad counter {text:?}"))
}

/// Parses a column that every build has.
fn required(name: &str, text: &str) -> Result<u64, String> {
    text.parse().map_err(|_| format!("bad {name} {text:?}"))
}

fn parse_step(text: &str) -> Result<Step, String> {
    let cols: Vec<&str> = text.split(',').collect();
    let [
        mode,
        _input,
        output,
        seconds,
        syncs,
        flushes,
        retired_peak,
        peak,
        active,
        cache,
        read_bytes,
        read_ns,
    ] = cols.as_slice()
    else {
        return Err(format!("step has {} columns, not 12", cols.len()));
    };
    let generate = match *mode {
        "generate" => true,
        "prompt" => false,
        other => return Err(format!("bad mode {other:?}")),
    };
    let output = match counter(output)? {
        Some(token) => Some(u32::try_from(token).map_err(|_| "output exceeds u32".to_string())?),
        None => None,
    };
    Ok(Step {
        generate,
        output,
        seconds: seconds
            .parse()
            .map_err(|_| format!("bad seconds {seconds:?}"))?,
        syncs: counter(syncs)?,
        flushes: counter(flushes)?,
        retired_peak: counter(retired_peak)?,
        peak: required("peak", peak)?,
        active: required("active", active)?,
        cache: required("cache", cache)?,
        read_bytes: required("read_bytes", read_bytes)?,
        read_ns: required("read_ns", read_ns)?,
    })
}

/// Reads `step0000` to `step0515`. The first `PROMPT_STEPS` are `prompt` without output; the rest
/// are `generate` with an output token. A missing, extra, or malformed step is an error.
pub fn steps(report: &Report) -> Result<Vec<Step>, String> {
    if report.get("step_columns") != Some(STEP_COLUMNS) {
        return Err("step_columns differ from the runner".into());
    }
    let total = PROMPT_STEPS + GENERATED;
    let mut steps = Vec::with_capacity(total);
    for n in 0..total {
        let key = format!("step{n:04}");
        let text = report
            .get(&key)
            .ok_or_else(|| format!("{key} is missing"))?;
        let step = parse_step(text).map_err(|e| format!("{key}: {e}"))?;
        if step.generate != (n >= PROMPT_STEPS) || step.generate != step.output.is_some() {
            return Err(format!("{key} has the wrong mode or output"));
        }
        steps.push(step);
    }
    if report.get(&format!("step{total:04}")).is_some() {
        return Err("the report has more steps than the runner".into());
    }
    Ok(steps)
}

/// `(max - min) / min`, or `None` for no values or a zero minimum.
pub fn flatness(peaks: &[u64]) -> Option<f64> {
    let min = *peaks.iter().min()?;
    let max = *peaks.iter().max()?;
    if min == 0 {
        return None;
    }
    Some((max - min) as f64 / min as f64)
}

/// Gate 5 memory: the MLX peak of generated tokens `WARMUP + 1` to `GENERATED` stays within
/// `FLAT_MAX`. Each step's peak was reset just before that step (coordinator decision D6).
pub fn gate_memory(steps: &[Step]) -> (Verdict, Option<f64>) {
    let peaks: Vec<u64> = steps
        .iter()
        .filter(|s| s.generate)
        .skip(WARMUP)
        .map(|s| s.peak)
        .collect();
    match flatness(&peaks) {
        Some(flat) if peaks.len() == GENERATED - WARMUP => {
            (Verdict::from_bool(flat <= FLAT_MAX), Some(flat))
        }
        other => (Verdict::Fail, other),
    }
}

/// Gate 6: every generated token synchronizes at most `SYNC_MAX` times. `Unavailable` without the
/// qualification counter.
pub fn gate_syncs(steps: &[Step]) -> Verdict {
    let counts: Option<Vec<u64>> = steps
        .iter()
        .filter(|s| s.generate)
        .map(|s| s.syncs)
        .collect();
    match counts {
        None => Verdict::Unavailable,
        Some(counts) if counts.is_empty() => Verdict::Fail,
        Some(counts) => Verdict::from_bool(counts.iter().all(|&n| n <= SYNC_MAX)),
    }
}

/// Gate 6 report: synchronizations, flushes, and peak retired bytes exist for every step.
pub fn gate_counters(steps: &[Step]) -> Verdict {
    let complete = !steps.is_empty()
        && steps
            .iter()
            .all(|s| s.syncs.is_some() && s.flushes.is_some() && s.retired_peak.is_some());
    if complete {
        Verdict::Pass
    } else {
        Verdict::Partial
    }
}

/// Gate 5 tokens: the first `MATCH` resident tokens equal the host tokens exactly. Also returns
/// the first mismatch index.
pub fn gate_tokens(resident: &[u32], host: &[u32]) -> (Verdict, Option<usize>) {
    if resident.len() < MATCH || host.len() < MATCH {
        return (Verdict::Fail, None);
    }
    let mismatch = resident
        .iter()
        .zip(host)
        .take(MATCH)
        .position(|(r, h)| r != h);
    (Verdict::from_bool(mismatch.is_none()), mismatch)
}

/// Gate 3 teacher-forced: exactly `TEACHER_CHECKS` checks and `failures=0`.
pub fn gate_teacher(report: &Report) -> Verdict {
    Verdict::from_bool(
        report.number("checks") == Some(TEACHER_CHECKS) && report.number("failures") == Some(0),
    )
}

/// Gate 3 end-to-end: `LAYERS` layer lines, one `logits t4` line, the device token, and `failures=0`.
/// The engine test records the top-5, argmax, and device token failures in `failures`.
pub fn gate_end_to_end(report: &Report) -> Verdict {
    Verdict::from_bool(
        report.count_prefix("layer_out L") == LAYERS
            && report.count_prefix("logits t4 ") == 1
            && report.get("device_token_t4").is_some()
            && report.number("failures") == Some(0),
    )
}

/// Parses a comma-separated token list. Any malformed entry gives an empty list.
pub fn tokens(text: Option<&str>) -> Vec<u32> {
    text.map(|t| {
        t.split(',')
            .map(str::parse)
            .collect::<Result<Vec<u32>, _>>()
            .unwrap_or_default()
    })
    .unwrap_or_default()
}

fn join(values: &[u32]) -> String {
    values
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// Minimum, mean, and maximum, or `-` three times when a value is unavailable or none exists.
fn stats(values: Option<Vec<u64>>) -> [String; 3] {
    match values {
        Some(v) if !v.is_empty() => {
            let min = v.iter().min().copied().unwrap_or_default();
            let max = v.iter().max().copied().unwrap_or_default();
            let mean = v.iter().sum::<u64>() as f64 / v.len() as f64;
            [min.to_string(), mean.to_string(), max.to_string()]
        }
        _ => ["-".to_string(), "-".to_string(), "-".to_string()],
    }
}

/// Derived rates, counters, memory, and reads of the decode steps, as record lines.
pub fn summary(steps: &[Step]) -> Vec<(String, String)> {
    let generated: Vec<&Step> = steps.iter().filter(|s| s.generate).collect();
    let prompt: Vec<&Step> = steps.iter().filter(|s| !s.generate).collect();
    let prompt_seconds: f64 = prompt.iter().map(|s| s.seconds).sum();
    let decode_seconds: f64 = generated.iter().map(|s| s.seconds).sum();
    let after: Vec<&&Step> = generated.iter().skip(WARMUP).collect();
    let after_seconds: f64 = after.iter().map(|s| s.seconds).sum();
    let window: Vec<u64> = after.iter().map(|s| s.peak).collect();
    let gen_syncs = stats(generated.iter().map(|s| s.syncs).collect());
    let prompt_syncs = stats(prompt.iter().map(|s| s.syncs).collect());
    let flushes: Option<u64> = steps.iter().map(|s| s.flushes).sum();
    let retired: Option<Vec<u64>> = steps.iter().map(|s| s.retired_peak).collect();
    let read_bytes: u64 = generated.iter().map(|s| s.read_bytes).sum();
    let read_ns: u64 = generated.iter().map(|s| s.read_ns).sum();
    let count = generated.len() as f64;
    let read_seconds = read_ns as f64 / 1e9;
    let dash = || "-".to_string();
    vec![
        ("prompt_seconds".into(), prompt_seconds.to_string()),
        ("decode_seconds".into(), decode_seconds.to_string()),
        (
            "tokens_per_second".into(),
            (count / decode_seconds).to_string(),
        ),
        (
            "tokens_per_second_after16".into(),
            (after.len() as f64 / after_seconds).to_string(),
        ),
        (
            "host_baseline_tokens_per_second".into(),
            HOST_BASELINE_TOKENS_PER_SECOND.to_string(),
        ),
        ("syncs_prompt_min".into(), prompt_syncs[0].clone()),
        ("syncs_prompt_max".into(), prompt_syncs[2].clone()),
        ("syncs_generate_min".into(), gen_syncs[0].clone()),
        ("syncs_generate_mean".into(), gen_syncs[1].clone()),
        ("syncs_generate_max".into(), gen_syncs[2].clone()),
        (
            "forced_flushes".into(),
            flushes.map_or_else(dash, |f| f.to_string()),
        ),
        (
            "peak_retired_bytes".into(),
            retired
                .and_then(|r| r.into_iter().max())
                .map_or_else(dash, |r| r.to_string()),
        ),
        (
            "mlx_peak_window_min".into(),
            window.iter().min().map_or_else(dash, u64::to_string),
        ),
        (
            "mlx_peak_window_max".into(),
            window.iter().max().map_or_else(dash, u64::to_string),
        ),
        (
            "mlx_active_end".into(),
            steps.last().map_or_else(dash, |s| s.active.to_string()),
        ),
        (
            "mlx_cache_max".into(),
            steps
                .iter()
                .map(|s| s.cache)
                .max()
                .map_or_else(dash, |c| c.to_string()),
        ),
        (
            "read_bytes_per_token_mean".into(),
            (read_bytes as f64 / count).to_string(),
        ),
        (
            "read_seconds_per_token_mean".into(),
            (read_seconds / count).to_string(),
        ),
        (
            "read_gib_per_second".into(),
            (read_bytes as f64 / read_seconds / f64::from(1u32 << 30)).to_string(),
        ),
    ]
}

/// The inputs of one record. A report that was not written is `None`.
pub struct Inputs<'a> {
    pub build: Build,
    pub blocks: Option<&'a str>,
    pub prompt: Option<&'a str>,
    pub decode: Option<&'a str>,
    /// Test name and exit code; `None` when a signal ended the test.
    pub exits: &'a [(String, Option<i32>)],
}

/// Writes the record text and returns whether every gate that the build requires passes.
/// Build Q requires every gate. Build P does not require the two counter gates of gate 6.
pub fn record(inputs: &Inputs<'_>) -> (String, bool) {
    let all_required = inputs.build == Build::Q;
    let mut lines: Vec<(String, String)> = vec![("build".into(), inputs.build.name().into())];
    let mut gates: Vec<(&str, Verdict, bool)> = Vec::new();
    for (name, code) in inputs.exits {
        let code = code.map_or_else(|| "signal".to_string(), |c| c.to_string());
        lines.push((format!("test_exit_{name}"), code));
    }
    let exits_ok = !inputs.exits.is_empty() && inputs.exits.iter().all(|(_, c)| *c == Some(0));
    gates.push(("gate_tests_exit", Verdict::from_bool(exits_ok), true));
    if inputs.build == Build::Q {
        let teacher = inputs
            .blocks
            .map_or(Verdict::Fail, |t| gate_teacher(&Report::parse(t)));
        gates.push(("gate_g3_teacher_forced", teacher, true));
        let end = inputs
            .prompt
            .map_or(Verdict::Fail, |t| gate_end_to_end(&Report::parse(t)));
        gates.push(("gate_g3_end_to_end", end, true));
    }
    let report = inputs.decode.map(Report::parse);
    let parsed = match &report {
        Some(report) => steps(report),
        None => Err("the decode report is missing".to_string()),
    };
    if let Some(report) = &report {
        let build_ok = report.get("build") == Some(inputs.build.name());
        gates.push(("gate_build_matches", Verdict::from_bool(build_ok), true));
        for key in COPIED {
            let value = report.get(key).unwrap_or("missing");
            lines.push((key.to_string(), value.to_string()));
        }
    }
    match parsed {
        Ok(steps) => {
            lines.extend(summary(&steps));
            let resident: Vec<u32> = steps.iter().filter_map(|s| s.output).collect();
            let host = tokens(report.as_ref().and_then(|r| r.get("host_tokens")));
            let (tokens_ok, mismatch) = gate_tokens(&resident, &host);
            lines.push((
                "resident_first32".into(),
                join(resident.get(..MATCH).unwrap_or(&[])),
            ));
            lines.push((
                "host_first32".into(),
                join(host.get(..MATCH).unwrap_or(&[])),
            ));
            lines.push((
                "first32_mismatch_index".into(),
                mismatch.map_or_else(|| "none".to_string(), |i| i.to_string()),
            ));
            let (memory, flat) = gate_memory(&steps);
            lines.push((
                "mlx_peak_flat".into(),
                flat.map_or_else(|| "-".to_string(), |f| f.to_string()),
            ));
            gates.push(("gate_g5_first32", tokens_ok, true));
            gates.push(("gate_g5_memory_flat", memory, true));
            gates.push(("gate_g6_syncs", gate_syncs(&steps), all_required));
            gates.push(("gate_g6_counters", gate_counters(&steps), all_required));
        }
        Err(error) => {
            lines.push(("decode_error".into(), error));
            gates.push(("gate_g5_first32", Verdict::Fail, true));
            gates.push(("gate_g5_memory_flat", Verdict::Fail, true));
            gates.push(("gate_g6_syncs", Verdict::Fail, all_required));
            gates.push(("gate_g6_counters", Verdict::Partial, all_required));
        }
    }
    let pass = gates
        .iter()
        .all(|(_, verdict, required)| !required || *verdict == Verdict::Pass);
    let mut text = String::new();
    for (key, value) in lines {
        text.push_str(&format!("{key}={value}\n"));
    }
    for (name, verdict, _) in &gates {
        text.push_str(&format!("{name}={}\n", verdict.as_str()));
    }
    text.push_str(&format!("verdict={}\n", if pass { "pass" } else { "fail" }));
    (text, pass)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A decode report: 4 prompt steps and 512 generated steps. Generated token `t` (1-based)
    /// outputs `100 + t`, synchronizes `syncs` times, and has MLX peak `peak(t)`. Every step takes
    /// 0.5 s (prompt) or 0.25 s (generate). The host tokens equal the first 32 outputs.
    fn decode_text(build: &str, syncs: &str, peak: impl Fn(usize) -> u64) -> String {
        let mut text = format!("build={build}\nstep_columns={STEP_COLUMNS}\n");
        for n in 0..PROMPT_STEPS {
            text.push_str(&format!(
                "step{n:04}=prompt,{n},-,0.5,{syncs},0,4096,{},10,5,100,1000\n",
                peak(1)
            ));
        }
        for t in 1..=GENERATED {
            let n = PROMPT_STEPS + t - 1;
            text.push_str(&format!(
                "step{n:04}=generate,7,{},0.25,{syncs},0,4096,{},10,5,100,1000\n",
                100 + t,
                peak(t)
            ));
        }
        let host: Vec<u32> = (1..=MATCH as u32).map(|t| 100 + t).collect();
        text.push_str(&format!("host_tokens={}\n", join(&host)));
        text
    }

    fn parsed(text: &str) -> Vec<Step> {
        steps(&Report::parse(text)).unwrap()
    }

    /// A teacher-forced report with one check line, then `checks` and `failures`.
    fn blocks_text(checks: u64, failures: u64) -> String {
        format!(
            "L00 t0 attn out rel=1.000e-6 cos=1.0000000\nchecks={checks}\nfailures={failures}\n"
        )
    }

    /// An end-to-end report with `layers` layer lines, the logits line, the device token, and `failures`.
    fn prompt_text(layers: usize, failures: u64) -> String {
        let mut text = String::from("token0_seconds=0.100\n");
        for l in 0..layers {
            text.push_str(&format!("layer_out L{l:02} rel=1.000e-6 cos=1.0000000\n"));
        }
        text.push_str("logits t4 rel=1.000e-6 cos=1.0000000\ndevice_token_t4=11751\n");
        text.push_str(&format!("failures={failures}\n"));
        text
    }

    #[test]
    fn syncs_at_110_pass_and_111_fail() {
        assert_eq!(
            gate_syncs(&parsed(&decode_text("q", "110", |_| 1000))),
            Verdict::Pass
        );
        assert_eq!(
            gate_syncs(&parsed(&decode_text("q", "111", |_| 1000))),
            Verdict::Fail
        );
        assert_eq!(
            gate_syncs(&parsed(&decode_text("p", "-", |_| 1000))),
            Verdict::Unavailable
        );
    }

    #[test]
    fn flatness_at_five_percent_passes_and_above_fails() {
        assert_eq!(flatness(&[10_000, 10_500]), Some(0.05));
        let at = parsed(&decode_text("q", "99", |t| {
            if t == 300 { 10_500 } else { 10_000 }
        }));
        assert_eq!(gate_memory(&at), (Verdict::Pass, Some(0.05)));
        let above = parsed(&decode_text("q", "99", |t| {
            if t == 300 { 10_501 } else { 10_000 }
        }));
        assert_eq!(gate_memory(&above).0, Verdict::Fail, "flatness 0.0501");
    }

    #[test]
    fn memory_window_skips_the_first_sixteen_tokens() {
        let warm = parsed(&decode_text("q", "99", |t| {
            if t == 16 { 20_000 } else { 10_000 }
        }));
        assert_eq!(
            gate_memory(&warm),
            (Verdict::Pass, Some(0.0)),
            "token 16 is warm-up"
        );
        let late = parsed(&decode_text("q", "99", |t| {
            if t == 17 { 20_000 } else { 10_000 }
        }));
        assert_eq!(
            gate_memory(&late),
            (Verdict::Fail, Some(1.0)),
            "token 17 is in the window"
        );
    }

    #[test]
    fn first_32_tokens_must_match_exactly() {
        let resident: Vec<u32> = (0..512).collect();
        let host: Vec<u32> = (0..32).collect();
        assert_eq!(gate_tokens(&resident, &host), (Verdict::Pass, None));
        let mut at31 = host.clone();
        at31[31] = 999;
        assert_eq!(gate_tokens(&resident, &at31), (Verdict::Fail, Some(31)));
        let mut at32: Vec<u32> = (0..33).collect();
        at32[32] = 999;
        assert_eq!(
            gate_tokens(&resident, &at32),
            (Verdict::Pass, None),
            "index 32 is outside"
        );
        assert_eq!(
            gate_tokens(&resident, &host[..31]),
            (Verdict::Fail, None),
            "31 host tokens"
        );
    }

    #[test]
    fn teacher_forced_gate_needs_1460_checks_and_no_failures() {
        assert_eq!(
            gate_teacher(&Report::parse(&blocks_text(1460, 0))),
            Verdict::Pass
        );
        assert_eq!(
            gate_teacher(&Report::parse(&blocks_text(1459, 0))),
            Verdict::Fail
        );
        assert_eq!(
            gate_teacher(&Report::parse(&blocks_text(1460, 1))),
            Verdict::Fail
        );
        let no_failures_line = Report::parse("checks=1460\n");
        assert_eq!(
            gate_teacher(&no_failures_line),
            Verdict::Fail,
            "missing failures line"
        );
    }

    #[test]
    fn end_to_end_gate_needs_48_layer_lines_and_no_failures() {
        assert_eq!(
            gate_end_to_end(&Report::parse(&prompt_text(48, 0))),
            Verdict::Pass
        );
        assert_eq!(
            gate_end_to_end(&Report::parse(&prompt_text(47, 0))),
            Verdict::Fail
        );
        assert_eq!(
            gate_end_to_end(&Report::parse(&prompt_text(48, 1))),
            Verdict::Fail
        );
    }

    #[test]
    fn p_build_passes_without_counters_and_q_build_does_not() {
        let decode = decode_text("p", "-", |_| 1000);
        let exits = [("decode".to_string(), Some(0))];
        let p = Inputs {
            build: Build::P,
            blocks: None,
            prompt: None,
            decode: Some(&decode),
            exits: &exits,
        };
        let (text, pass) = record(&p);
        assert!(pass, "{text}");
        assert!(text.contains("gate_g6_syncs=unavailable\n"), "{text}");
        assert!(text.contains("gate_g6_counters=partial\n"), "{text}");
        let decode = decode_text("q", "-", |_| 1000);
        let (blocks, prompt) = (blocks_text(1460, 0), prompt_text(48, 0));
        let q = Inputs {
            build: Build::Q,
            blocks: Some(&blocks),
            prompt: Some(&prompt),
            decode: Some(&decode),
            exits: &exits,
        };
        let (text, pass) = record(&q);
        assert!(!pass, "{text}");
        assert!(text.contains("verdict=fail\n"), "{text}");
    }

    #[test]
    fn a_failed_test_exit_fails_the_record() {
        let decode = decode_text("p", "-", |_| 1000);
        let exits = [("decode".to_string(), Some(101))];
        let inputs = Inputs {
            build: Build::P,
            blocks: None,
            prompt: None,
            decode: Some(&decode),
            exits: &exits,
        };
        let (text, pass) = record(&inputs);
        assert!(!pass, "{text}");
        assert!(text.contains("test_exit_decode=101\n"), "{text}");
        assert!(text.contains("gate_tests_exit=fail\n"), "{text}");
    }

    #[test]
    fn steps_read_every_column_and_reject_a_missing_step() {
        let text = decode_text("q", "99", |t| 5000 + t as u64);
        let all = parsed(&text);
        assert_eq!(all.len(), 516, "4 prompt and 512 generated steps");
        let first = &all[4];
        assert!(first.generate, "step 4 generates");
        assert_eq!(first.output, Some(101), "output");
        assert_eq!(first.seconds, 0.25, "seconds");
        assert_eq!(first.syncs, Some(99), "syncs");
        assert_eq!(first.flushes, Some(0), "flushes");
        assert_eq!(first.retired_peak, Some(4096), "retired peak");
        assert_eq!(first.peak, 5001, "peak");
        assert_eq!(first.active, 10, "active");
        assert_eq!(first.cache, 5, "cache");
        assert_eq!(first.read_bytes, 100, "read bytes");
        assert_eq!(first.read_ns, 1000, "read ns");
        assert_eq!(all[0].output, None, "prompt steps have no output");
        let missing: String = text
            .lines()
            .filter(|l| !l.starts_with("step0100="))
            .map(|l| format!("{l}\n"))
            .collect();
        let error = steps(&Report::parse(&missing)).unwrap_err();
        assert_eq!(error, "step0100 is missing");
    }

    #[test]
    fn rates_use_only_the_generated_steps() {
        let lines = summary(&parsed(&decode_text("q", "99", |_| 1000)));
        let get = |key: &str| {
            lines
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
                .unwrap()
        };
        assert_eq!(get("prompt_seconds"), "2", "4 prompt steps of 0.5 s");
        assert_eq!(
            get("decode_seconds"),
            "128",
            "512 generated steps of 0.25 s"
        );
        assert_eq!(get("tokens_per_second"), "4");
        assert_eq!(get("tokens_per_second_after16"), "4");
        assert_eq!(get("syncs_generate_max"), "99");
        assert_eq!(get("read_bytes_per_token_mean"), "100");
    }
}
