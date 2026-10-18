//! Reads `trace.tsv` (schema `s3-route-trace/1`) and `tokens.tsv` of the T0 route trace and
//! checks T0-P1 on every row (S3 plan sections 3.4 and 3.7).

use std::collections::{BTreeMap, BTreeSet};

/// The schema that `engine/src/family/flash_next/resident/trace.rs` writes.
pub const SCHEMA: &str = "s3-route-trace/1";

/// Gates of a renormalized selection sum to 1 within this bound (T0-P1).
const GATE_SUM_TOLERANCE: f64 = 1e-5;

/// One token step: the selection of every layer, `top_k` ids per layer in rank order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    /// `true` for a generated (`g`) step, `false` for a prompt (`p`) step.
    pub decode: bool,
    /// `layers * top_k` expert ids.
    pub sel: Vec<u16>,
}

/// The steps of one prompt in position order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptTrace {
    pub id: u32,
    pub steps: Vec<Step>,
}

/// A checked trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trace {
    /// The `# key=value` header lines in file order.
    pub header: Vec<(String, String)>,
    pub layers: usize,
    pub experts: usize,
    pub top_k: usize,
    /// Stored bytes of one expert in each layer (decision 0038: not one fixed size).
    pub expert_bytes: Vec<u64>,
    pub prompts: Vec<PromptTrace>,
}

impl Trace {
    /// The ids of `layer` in `step`, in rank order.
    pub fn sel<'a>(&self, step: &'a Step, layer: usize) -> &'a [u16] {
        &step.sel[layer * self.top_k..(layer + 1) * self.top_k]
    }

    /// Bytes of one step's selection over all layers: `top_k` experts per layer.
    pub fn step_bytes(&self) -> u64 {
        self.expert_bytes.iter().sum::<u64>() * self.top_k as u64
    }

    /// Parses and checks a trace. Every row needs `top_k` distinct ids below `experts`, finite
    /// probabilities and gates, and gates that sum to 1 when `renorm` is set. Rows come in
    /// complete steps of `layers` rows in layer order. Positions start at 0 in each prompt and
    /// rise by 1, and prompt steps precede generated steps.
    pub fn parse(text: &str) -> Result<Trace, String> {
        let mut header = Vec::new();
        let mut lines = text.lines().enumerate();
        let mut columns = None;
        for (number, line) in lines.by_ref() {
            if let Some(entry) = line.strip_prefix("# ") {
                let (key, value) = entry
                    .split_once('=')
                    .ok_or_else(|| format!("line {}: header without '='", number + 1))?;
                header.push((key.to_string(), value.to_string()));
            } else {
                columns = Some((number, line));
                break;
            }
        }
        let get = |key: &str| {
            header
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.as_str())
                .ok_or_else(|| format!("header has no {key}"))
        };
        if get("schema")? != SCHEMA {
            return Err(format!("schema {} is not {SCHEMA}", get("schema")?));
        }
        let number = |key: &str| -> Result<usize, String> {
            get(key)?
                .parse()
                .map_err(|_| format!("header {key} is not a count"))
        };
        let (layers, experts, top_k) = (number("layers")?, number("experts")?, number("top_k")?);
        if layers == 0 || top_k == 0 || top_k > experts || experts > usize::from(u16::MAX) {
            return Err(format!(
                "header counts layers {layers}, experts {experts}, top_k {top_k} are invalid"
            ));
        }
        let renorm = match get("renorm")? {
            "true" => true,
            "false" => false,
            other => return Err(format!("header renorm {other} is not a boolean")),
        };
        let expert_bytes: Vec<u64> = get("expert_bytes")?
            .split(',')
            .map(|v| v.parse().map_err(|_| format!("expert_bytes value {v}")))
            .collect::<Result<_, _>>()?;
        if expert_bytes.len() != layers || expert_bytes.contains(&0) {
            return Err(format!(
                "expert_bytes has {} positive values for {layers} layers",
                expert_bytes.len()
            ));
        }
        let (number, line) = columns.ok_or("the trace has no column line")?;
        if line != columns_line(top_k) {
            return Err(format!("line {}: unexpected column line", number + 1));
        }
        let mut parser = Rows {
            layers,
            experts,
            top_k,
            renorm,
            prompts: Vec::new(),
            seen: BTreeSet::new(),
            open: None,
        };
        for (number, line) in lines {
            parser
                .row(line)
                .map_err(|message| format!("line {}: {message}", number + 1))?;
        }
        if let Some((_, _, _, sel)) = &parser.open {
            return Err(format!(
                "the last step has {} of {layers} layers",
                sel.len() / top_k
            ));
        }
        Ok(Trace {
            header,
            layers,
            experts,
            top_k,
            expert_bytes,
            prompts: parser.prompts,
        })
    }
}

/// The column line for `top_k` ids per row, as the engine writes it.
pub fn columns_line(top_k: usize) -> String {
    let mut columns = vec!["prompt", "mode", "pos", "layer"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    for prefix in ["e", "p", "g"] {
        columns.extend((0..top_k).map(|rank| format!("{prefix}{rank}")));
    }
    columns.join("\t")
}

/// Row parser state.
struct Rows {
    layers: usize,
    experts: usize,
    top_k: usize,
    renorm: bool,
    prompts: Vec<PromptTrace>,
    seen: BTreeSet<u32>,
    /// The step being read: prompt, position, mode, and the ids so far.
    open: Option<(u32, u32, bool, Vec<u16>)>,
}

impl Rows {
    fn row(&mut self, line: &str) -> Result<(), String> {
        let fields: Vec<&str> = line.split('\t').collect();
        let k = self.top_k;
        if fields.len() != 4 + 3 * k {
            return Err(format!("{} fields, expected {}", fields.len(), 4 + 3 * k));
        }
        let prompt: u32 = fields[0].parse().map_err(|_| "prompt is not a count")?;
        let decode = match fields[1] {
            "p" => false,
            "g" => true,
            other => return Err(format!("mode {other} is not p or g")),
        };
        let pos: u32 = fields[2].parse().map_err(|_| "pos is not a count")?;
        let layer: usize = fields[3].parse().map_err(|_| "layer is not a count")?;
        let ids: Vec<u16> = fields[4..4 + k]
            .iter()
            .map(|v| v.parse::<u16>().map_err(|_| format!("id {v}")))
            .collect::<Result<_, _>>()?;
        let distinct: BTreeSet<u16> = ids.iter().copied().collect();
        if distinct.len() != k || ids.iter().any(|&id| usize::from(id) >= self.experts) {
            return Err(format!("ids {ids:?} are not {k} distinct expert ids"));
        }
        let floats = |range: std::ops::Range<usize>| -> Result<Vec<f64>, String> {
            fields[range]
                .iter()
                .map(|v| match v.parse::<f64>() {
                    Ok(x) if x.is_finite() => Ok(x),
                    _ => Err(format!("value {v} is not finite")),
                })
                .collect()
        };
        floats(4 + k..4 + 2 * k)?;
        let gates = floats(4 + 2 * k..4 + 3 * k)?;
        let sum: f64 = gates.iter().sum();
        if self.renorm && (sum - 1.0).abs() > GATE_SUM_TOLERANCE {
            return Err(format!("gates sum to {sum}"));
        }
        if layer == 0 {
            if let Some((_, _, _, sel)) = &self.open {
                return Err(format!(
                    "a new step starts after {} of {} layers",
                    sel.len() / k,
                    self.layers
                ));
            }
            self.start(prompt, pos, decode)?;
        }
        let Some((open_prompt, open_pos, open_decode, sel)) = self.open.as_mut() else {
            return Err(format!("layer {layer} row outside a step"));
        };
        if (prompt, pos, decode) != (*open_prompt, *open_pos, *open_decode)
            || layer != sel.len() / k
        {
            return Err(format!(
                "row prompt {prompt} pos {pos} layer {layer} breaks step prompt {open_prompt} pos {open_pos} at layer {}",
                sel.len() / k
            ));
        }
        sel.extend_from_slice(&ids);
        if layer + 1 == self.layers
            && let Some((_, _, _, sel)) = self.open.take()
        {
            let last = self.prompts.last_mut().ok_or("no prompt for the step")?;
            let step = last.steps.last_mut().ok_or("no step to complete")?;
            step.sel = sel;
        }
        Ok(())
    }

    /// Opens a step at layer 0 and checks the prompt, position, and mode order.
    fn start(&mut self, prompt: u32, pos: u32, decode: bool) -> Result<(), String> {
        let same = self.prompts.last().is_some_and(|p| p.id == prompt);
        if same {
            let last = self.prompts.last().and_then(|p| p.steps.last());
            let Some(last_step) = last else {
                return Err("prompt without steps".into());
            };
            let expected = u32::try_from(self.prompts.last().map_or(0, |p| p.steps.len()))
                .map_err(|_| "too many steps")?;
            if pos != expected {
                return Err(format!(
                    "prompt {prompt}: position {pos}, expected {expected}"
                ));
            }
            if last_step.decode && !decode {
                return Err(format!(
                    "prompt {prompt}: a prompt step after a generated step"
                ));
            }
        } else {
            if !self.seen.insert(prompt) {
                return Err(format!("prompt {prompt} appears twice"));
            }
            if pos != 0 {
                return Err(format!("prompt {prompt} starts at position {pos}"));
            }
            self.prompts.push(PromptTrace {
                id: prompt,
                steps: Vec::new(),
            });
        }
        if let Some(p) = self.prompts.last_mut() {
            p.steps.push(Step {
                decode,
                sel: Vec::new(),
            });
        }
        self.open = Some((
            prompt,
            pos,
            decode,
            Vec::with_capacity(self.layers * self.top_k),
        ));
        Ok(())
    }
}

/// The token log of one prompt from `tokens.tsv`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TokenLog {
    /// Prompt tokens: the `p` steps plus the last prompt token, which runs as the first `g` step.
    pub prompt_tokens: usize,
    /// The generated tokens in order.
    pub outputs: Vec<u32>,
}

/// Reads `tokens.tsv`: `prompt`, `pos`, `mode`, `input`, `output`, and further columns.
pub fn parse_tokens(text: &str) -> Result<BTreeMap<u32, TokenLog>, String> {
    let mut logs: BTreeMap<u32, TokenLog> = BTreeMap::new();
    let mut prompt_steps: BTreeMap<u32, usize> = BTreeMap::new();
    for (number, line) in text.lines().enumerate().skip(1) {
        let fields: Vec<&str> = line.split('\t').collect();
        let bad = || format!("tokens line {}: {line}", number + 1);
        if fields.len() < 5 {
            return Err(bad());
        }
        let prompt: u32 = fields[0].parse().map_err(|_| bad())?;
        let log = logs.entry(prompt).or_default();
        match fields[2] {
            "p" => *prompt_steps.entry(prompt).or_default() += 1,
            "g" => log.outputs.push(fields[4].parse().map_err(|_| bad())?),
            _ => return Err(bad()),
        }
    }
    for (prompt, log) in &mut logs {
        log.prompt_tokens = prompt_steps.get(prompt).copied().unwrap_or(0) + 1;
    }
    Ok(logs)
}

/// The share of the 4-grams of `tokens` that repeat an earlier 4-gram of the same sequence
/// (S3 plan section 3.5, "Repetition check"). Zero for fewer than 4 tokens.
pub fn repetition(tokens: &[u32]) -> f64 {
    let mut seen = BTreeSet::new();
    let (mut total, mut repeated) = (0usize, 0usize);
    for gram in tokens.windows(4) {
        total += 1;
        if !seen.insert(gram) {
            repeated += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        repeated as f64 / total as f64
    }
}
