//! Host-side token sampling behind the executor seam (decision 0033, item 5).
//!
//! A `Sampler` lives in the per-job `GenerationState`. The executor draws the next token from it
//! inside `decode`, after it has read back the next-token logits. The sampler is pure host code:
//! it never calls a kernel and it needs no random-number crate.
//!
//! Contract mapping (`api/openapi.yaml`, `ChatCompletionRequest`):
//!
//! | Request field | Sampler behavior |
//! | --- | --- |
//! | `temperature` absent | the engine default from `generation_config.json`, else 0.7 (`DEFAULT_TEMPERATURE`) |
//! | `temperature` 0 | greedy: the lowest index among the largest logits, no random draw |
//! | `temperature` above 0 | softmax of `logit / temperature` |
//! | `top_k` absent or 0 | off |
//! | `top_k` above 0 | only the `top_k` largest logits stay (ties go to the lower index) |
//! | `top_p` absent | the engine default from `generation_config.json`, else 1.0 (`DEFAULT_TOP_P`), which keeps everything |
//! | `top_p` below 1 | the smallest set of tokens whose probability mass reaches `top_p` stays |
//!
//! The order is `top_k`, then temperature, then `top_p`. Each sampled token consumes exactly one
//! random draw. The contract has no seed, so production seeds come from `entropy_seed`.
//!
//! The defaults of an absent field belong to the engine, not to this module: every engine entry
//! reads `temperature`, `top_p`, and `top_k` from `generation_config.json` in its model directory
//! at start (`SamplingDefaults`). An absent file, an absent key, or an explicit JSON `null` for
//! a key keeps the contract default, and a file that is malformed, out of range, not a regular
//! file, or larger than 1 MiB stops the engine start.

use crate::EngError;
use crate::checkpoint_json::UniqueValue;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs;
use std::hash::{BuildHasher, Hasher};
use std::io::{ErrorKind, Read};
use std::path::Path;

/// The contract default for `temperature`.
pub(crate) const DEFAULT_TEMPERATURE: f64 = 0.7;

/// The contract default for `top_p`.
pub(crate) const DEFAULT_TOP_P: f64 = 1.0;

/// The contract maximum for `temperature`.
const MAX_TEMPERATURE: f64 = 2.0;

/// The size cap of `generation_config.json`: 1 MiB.
const MAX_FILE_BYTES: usize = 1024 * 1024;

/// The sampling defaults of one engine entry: the `temperature`, `top_p`, and `top_k` that a
/// chat request takes when it omits the field. They come from `generation_config.json` in the
/// model directory of the engine; an absent file or key keeps the contract default of that key.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SamplingDefaults {
    pub(crate) temperature: f64,
    pub(crate) top_p: f64,
    pub(crate) top_k: u32,
}

impl SamplingDefaults {
    /// The contract defaults: `temperature` 0.7, `top_p` 1.0, and `top_k` off.
    pub(crate) const CONTRACT: Self = Self {
        temperature: DEFAULT_TEMPERATURE,
        top_p: DEFAULT_TOP_P,
        top_k: 0,
    };

    /// Reads the defaults from `generation_config.json` in `model_dir` and reports whether the
    /// file supplied them. An absent file, an absent key, or an explicit JSON `null` for a key
    /// keeps the contract default of that key (a Hugging Face config carries `null` for an
    /// unset field). A file that cannot be read, is not a regular file, is larger than 1 MiB,
    /// is not JSON, or holds a value outside the contract bounds is an error, so a broken
    /// checkpoint stops the engine start instead of sampling at silently different settings.
    pub(crate) fn from_model_dir(model_dir: &Path) -> Result<LoadedDefaults, EngError> {
        let path = model_dir.join("generation_config.json");
        let metadata = match fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                return Ok(LoadedDefaults {
                    values: Self::CONTRACT,
                    from_file: false,
                });
            }
            Err(error) => {
                return Err(EngError::Checkpoint(format!(
                    "generation_config.json could not be read: {error}"
                )));
            }
        };
        // Only a regular file may feed the defaults: a FIFO, a directory, or a device is an
        // error that names the path, never a silent fallback.
        if !metadata.is_file() {
            return Err(EngError::Checkpoint(format!(
                "generation_config.json must be a regular file: {}",
                path.display()
            )));
        }
        let bytes = read_capped(&path)?;
        let root = serde_json::from_slice::<UniqueValue>(&bytes)
            .map_err(|error| {
                EngError::Checkpoint(format!("generation_config.json parse failed: {error}"))
            })?
            .0;
        if !root.is_object() {
            return Err(EngError::Checkpoint(
                "generation_config.json must hold a JSON object".to_string(),
            ));
        }
        Ok(LoadedDefaults {
            values: Self {
                temperature: file_number(
                    &root,
                    "temperature",
                    DEFAULT_TEMPERATURE,
                    0.0,
                    MAX_TEMPERATURE,
                )?,
                top_p: file_number(&root, "top_p", DEFAULT_TOP_P, 0.0, 1.0)?,
                top_k: file_top_k(&root)?,
            },
            from_file: true,
        })
    }
}

/// Reads at most `MAX_FILE_BYTES + 1` bytes of the file, so a longer file cannot inflate memory
/// or reach the parser. A file over the cap is an error that names the path and the cap.
fn read_capped(path: &Path) -> Result<Vec<u8>, EngError> {
    let could_not_read = |error: std::io::Error| {
        EngError::Checkpoint(format!("generation_config.json could not be read: {error}"))
    };
    let mut bytes = Vec::new();
    fs::File::open(path)
        .map_err(could_not_read)?
        .take(MAX_FILE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(could_not_read)?;
    if bytes.len() > MAX_FILE_BYTES {
        return Err(EngError::Checkpoint(format!(
            "generation_config.json is larger than 1 MiB ({} bytes): {}",
            MAX_FILE_BYTES,
            path.display()
        )));
    }
    Ok(bytes)
}

/// Reads one number member of `generation_config.json` and checks the contract bounds
/// `[low, high]`. An absent key keeps `fallback`; so does an explicit JSON `null`, because a
/// Hugging Face config carries `null` for an unset field. A wrong type names the JSON type.
fn file_number(
    root: &serde_json::Value,
    key: &str,
    fallback: f64,
    low: f64,
    high: f64,
) -> Result<f64, EngError> {
    let Some(value) = root.get(key) else {
        return Ok(fallback);
    };
    if value.is_null() {
        return Ok(fallback);
    }
    let found = value.as_f64().ok_or_else(|| {
        EngError::Checkpoint(format!(
            "generation_config.json {key} must be a number, found {}",
            json_type(value)
        ))
    })?;
    if !(low..=high).contains(&found) {
        return Err(EngError::Checkpoint(format!(
            "generation_config.json {key} must be between {low} and {high}, found {found}"
        )));
    }
    Ok(found)
}

/// The JSON type name of `value` for an error message: `string`, `number`, `boolean`, `array`,
/// `object`, or `null`. The loader echoes only the type of a wrong-typed value, never the value.
fn json_type(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Reads the `top_k` member of `generation_config.json`: a non-negative integer that fits a
/// `u32`. An absent key is off; so is an explicit JSON `null`. A wrong type names the JSON type.
fn file_top_k(root: &serde_json::Value) -> Result<u32, EngError> {
    let Some(value) = root.get("top_k") else {
        return Ok(0);
    };
    if value.is_null() {
        return Ok(0);
    }
    let found = value.as_u64().ok_or_else(|| {
        EngError::Checkpoint(format!(
            "generation_config.json top_k must be a non-negative integer, found {}",
            json_type(value)
        ))
    })?;
    u32::try_from(found).map_err(|_| {
        EngError::Checkpoint(format!(
            "generation_config.json top_k must be at most 4294967295, found {found}"
        ))
    })
}

/// The sampling defaults of one engine entry and their origin: whether a
/// `generation_config.json` supplied them. The start log line takes its "from file" label from
/// this flag, so the label cannot disagree with the read (no second `exists` call, no race).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LoadedDefaults {
    /// The effective defaults.
    pub(crate) values: SamplingDefaults,
    /// True when a `generation_config.json` supplied the values; false when the file is absent.
    pub(crate) from_file: bool,
}

/// Checked sampling settings of one request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SamplingParams {
    temperature: f64,
    top_p: f64,
    top_k: u32,
}

impl SamplingParams {
    /// Checks the contract bounds: `temperature` in [0, 2] and `top_p` in [0, 1].
    /// The error message names the field.
    pub(crate) fn new(temperature: f64, top_p: f64, top_k: u32) -> Result<Self, EngError> {
        if !(0.0..=MAX_TEMPERATURE).contains(&temperature) {
            return Err(EngError::BadRequest(format!(
                "temperature must be between 0 and 2, got {temperature}"
            )));
        }
        if !(0.0..=1.0).contains(&top_p) {
            return Err(EngError::BadRequest(format!(
                "top_p must be between 0 and 1, got {top_p}"
            )));
        }
        Ok(Self {
            temperature,
            top_p,
            top_k,
        })
    }

    /// Settings for greedy decoding.
    pub(crate) fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
        }
    }

    /// Builds the settings of a request. An absent field takes its default from `defaults`:
    /// the values that the engine entry loaded from the `generation_config.json` of its model
    /// directory. A `top_k` above `u32::MAX` keeps every token, so it saturates.
    pub(crate) fn from_request(
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<u64>,
        defaults: &SamplingDefaults,
    ) -> Result<Self, EngError> {
        Self::new(
            temperature.unwrap_or(defaults.temperature),
            top_p.unwrap_or(defaults.top_p),
            u32::try_from(top_k.unwrap_or(u64::from(defaults.top_k))).unwrap_or(u32::MAX),
        )
    }

    /// True when the settings select the largest logit without a random draw.
    pub(crate) fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }
}

/// A seed for one job from system randomness and the clock. It uses only `std`.
pub(crate) fn entropy_seed() -> u64 {
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    if let Ok(elapsed) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        hasher.write_u128(elapsed.as_nanos());
    }
    hasher.finish()
}

/// SplitMix64 (Steele, Lea, and Flood, 2014): a 64-bit generator with one word of state.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in [0, 1) from the top 53 bits of the next output.
    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0
    }
}

/// Draws token ids from logits under checked settings and a seeded generator.
pub(crate) struct Sampler {
    params: SamplingParams,
    rng: SplitMix64,
}

impl Sampler {
    /// A sampler with the given settings and an explicit seed. Tests and replays use this.
    pub(crate) fn new(params: SamplingParams, seed: u64) -> Self {
        Self {
            params,
            rng: SplitMix64::new(seed),
        }
    }

    /// A sampler seeded once from system randomness. The server uses this for each job.
    pub(crate) fn from_entropy(params: SamplingParams) -> Self {
        Self::new(params, entropy_seed())
    }

    /// A greedy sampler. It never draws from its generator.
    pub(crate) fn greedy() -> Self {
        Self::new(SamplingParams::greedy(), 0)
    }

    /// The settings in force. The server's end-to-end probe records them for its tests.
    pub(crate) fn params(&self) -> SamplingParams {
        self.params
    }

    /// Returns the next token id. `logits` holds one finite value per vocabulary entry.
    /// Empty, oversized, or non-finite logits fail with `Backend`, because they mean that the
    /// forward pass produced invalid numbers.
    pub(crate) fn sample(&mut self, logits: &[f32]) -> Result<u32, EngError> {
        Ok(self.sample_with_keep(logits)?.0)
    }

    /// The next token id and the number of kept candidates. `sample` keeps only the token; the
    /// equivalence tests compare both against the reference implementation, because a wrong kept
    /// count can hide behind the same token for a whole row.
    fn sample_with_keep(&mut self, logits: &[f32]) -> Result<(u32, usize), EngError> {
        check_logits(logits)?;
        if self.params.is_greedy() {
            return Ok((argmax(logits), 1));
        }
        let temperature = self.params.temperature;
        let top = top_logit(logits);
        let top_k = usize::try_from(self.params.top_k).unwrap_or(usize::MAX);
        // The ranked candidates. With `top_k` only the best `limit` entries are needed and a
        // bounded heap pulls them in one pass over the row without the full candidate allocation;
        // the open vocabulary takes the whole ranked order so that the normaliser can add the
        // weights exactly like the reference's `weights.iter().sum()`.
        let ranked = if top_k == 0 {
            ranked_row(logits)
        } else if top_k <= HEAP_TOP_K {
            top_ranked(logits, top_k.min(logits.len()))
        } else {
            let mut ranked = ranked_row(logits);
            ranked.truncate(top_k);
            ranked
        };
        self.pick(&ranked, top, temperature)
    }

    /// Draws from the ranked candidates exactly like the reference kept in the tests: the keep
    /// loop walks the ranked weights left to right and keeps the smallest prefix whose mass
    /// reaches `top_p` times the total, one random draw scales the kept mass, and the cumulative
    /// walk returns the first kept token whose cumulative weight is above the target. Returns the
    /// token and the number of kept candidates.
    fn pick(
        &mut self,
        ranked: &[(u32, f32)],
        top: f64,
        temperature: f64,
    ) -> Result<(u32, usize), EngError> {
        let (weights, total) = ranked_weights(ranked, top, temperature);
        let threshold = self.params.top_p * total;
        let mut keep = 0_usize;
        let mut mass = 0.0_f64;
        for &w in &weights {
            mass += w;
            keep += 1;
            if mass >= threshold {
                break;
            }
        }
        let target = self.rng.next_unit() * mass;
        let mut cumulative = 0.0_f64;
        for (entry, &w) in ranked.iter().zip(&weights).take(keep) {
            cumulative += w;
            if cumulative > target {
                return Ok((entry.0, keep));
            }
        }
        // Rounding can leave `target` at the total. The last kept token is the answer then.
        ranked
            .get(keep.saturating_sub(1))
            .map(|entry| (entry.0, keep))
            .ok_or_else(|| invalid_logits("no candidate token"))
    }
}

fn invalid_logits(message: &str) -> EngError {
    EngError::Backend {
        operation: "sample",
        message: message.to_string(),
    }
}

fn check_logits(logits: &[f32]) -> Result<(), EngError> {
    if logits.is_empty() {
        return Err(invalid_logits("logits are empty"));
    }
    if u32::try_from(logits.len()).is_err() {
        return Err(invalid_logits("logits are longer than u32::MAX"));
    }
    if logits.iter().any(|logit| !logit.is_finite()) {
        return Err(invalid_logits("logits hold a non-finite value"));
    }
    Ok(())
}

/// The index of the largest logit. A tie goes to the lower index.
fn argmax(logits: &[f32]) -> u32 {
    let mut best_index = 0_usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in logits.iter().enumerate() {
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }
    best_index as u32
}

/// The largest `top_k` the bounded heap takes. Above it the heap's `O(n log top_k)` churn costs
/// more than building the whole ranked order once.
const HEAP_TOP_K: usize = 4096;

/// The largest logit as the f64 shift `top` that makes the largest weight exactly 1.
fn top_logit(logits: &[f32]) -> f64 {
    let mut top = f32::NEG_INFINITY;
    for &logit in logits {
        if logit > top {
            top = logit;
        }
    }
    f64::from(top)
}

/// The unnormalised softmax weight `exp((logit - top) / temperature)` in the present arithmetic.
fn weight(logit: f32, top: f64, temperature: f64) -> f64 {
    ((f64::from(logit) - top) / temperature).exp()
}

/// The vocabulary as `(index, logit)` pairs, ready to be ordered by `compare_ranked`.
fn candidate_pairs(logits: &[f32]) -> Vec<(u32, f32)> {
    logits
        .iter()
        .enumerate()
        .map(|(index, &logit)| (index as u32, logit))
        .collect()
}

/// The present sort's order: larger logit first, a tie to the lower index.
fn compare_ranked(a: &(u32, f32), b: &(u32, f32)) -> Ordering {
    b.1.partial_cmp(&a.1)
        .unwrap_or(Ordering::Equal)
        .then(a.0.cmp(&b.0))
}

/// A heap entry. Its `Ord` is the ranked order, so the heap's maximum is the ranked worst entry
/// and a bounded heap pops away exactly the entries the ranked order would truncate.
#[derive(Clone, Copy, Debug)]
struct Entry {
    index: u32,
    logit: f32,
}

impl Entry {
    /// The entry as the `(index, logit)` pair the ranked helpers use.
    fn pair(self) -> (u32, f32) {
        (self.index, self.logit)
    }
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Entry {}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        compare_ranked(&self.pair(), &other.pair())
    }
}

/// The ranked `limit` entries of `logits` from one pass with a bounded heap: `limit` entries of
/// memory, no full candidate allocation, and the reference's tie-break (a tie goes to the lower
/// index). The result is the same set and order as the reference's sort-then-truncate.
fn top_ranked(logits: &[f32], limit: usize) -> Vec<(u32, f32)> {
    let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(limit);
    for (index, &logit) in logits.iter().enumerate() {
        let entry = Entry {
            index: index as u32,
            logit,
        };
        if heap.len() < limit {
            heap.push(entry);
        } else if heap
            .peek()
            .copied()
            .is_some_and(|worst| compare_ranked(&entry.pair(), &worst.pair()) == Ordering::Less)
        {
            heap.pop();
            heap.push(entry);
        }
    }
    let mut ranked: Vec<(u32, f32)> = heap.into_iter().map(Entry::pair).collect();
    ranked.sort_unstable_by(compare_ranked);
    ranked
}

/// A 32-bit key that orders like the logit values: `order_key(a) < order_key(b)` exactly when
/// `a < b` as reals, with -0.0 and 0.0 equal, which is how `compare_ranked` compares them.
fn order_key(logit: f32) -> u32 {
    let canonical = if logit == 0.0 { 0.0 } else { logit };
    if canonical.is_sign_negative() {
        !canonical.to_bits()
    } else {
        canonical.to_bits() | 0x8000_0000
    }
}

/// The whole row in ranked order: larger logit first, a tie to the lower index, exactly the
/// reference sort's order. The composite key (logit key, reversed index) is a total order, so a
/// key sort reproduces the reference's comparator sort at a fraction of its cost.
fn ranked_row(logits: &[f32]) -> Vec<(u32, f32)> {
    let mut pairs = candidate_pairs(logits);
    pairs.sort_unstable_by_key(|&(index, logit)| {
        (u64::from(order_key(logit)) << 32) | u64::from(!index)
    });
    // Ascending keys give (logit up, index down); reversed they give the ranked order.
    pairs.reverse();
    pairs
}

/// The unnormalised weights of `ranked` in ranked order and their softmax normaliser `total`,
/// added left to right exactly like the reference's `weights.iter().sum()`.
///
/// The ranked order is required for an exact total: f64 addition is order-dependent, so no
/// shortcut can produce the reference's sum without the reference's order (a row of 248,076
/// weights of about 1e-17 before the top weight totals 1.0000000000025 in row order and exactly
/// 1.0 in ranked order). The one bit-exact saving is a tail that provably vanishes: once an
/// addition leaves the total unchanged and the next ranked weight is no larger, every later
/// weight vanishes too, so the rest of the row is dropped without computing its weights.
///
/// Ulp-level boundary, the one case this early exit can miss: exp's rounding may make neighbouring
/// weights non-monotone in the logit order by one ulp. If a vanishing weight is followed by a
/// weight just large enough to change the total again (only within one ulp of exactly half an ulp
/// of it), the reference's total can come out one ulp larger. Every cutoff comparison keeps the
/// reference's direction for exactly-representable cases (`mass >= threshold` to keep,
/// `cumulative > target` to return), so no other observable sampling behaviour changes.
fn ranked_weights(ranked: &[(u32, f32)], top: f64, temperature: f64) -> (Vec<f64>, f64) {
    let mut weights = Vec::with_capacity(ranked.len());
    let mut total = 0.0_f64;
    let mut stalled: Option<f64> = None;
    for &(_, logit) in ranked {
        let w = weight(logit, top, temperature);
        if stalled.is_some_and(|previous| w <= previous) {
            break;
        }
        let before = total;
        total += w;
        weights.push(w);
        stalled = (total == before).then_some(w);
    }
    (weights, total)
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_TEMPERATURE, LoadedDefaults, Sampler, SamplingDefaults, SamplingParams, SplitMix64,
        argmax, check_logits, entropy_seed, invalid_logits, ranked_row, ranked_weights,
    };
    use crate::EngError;
    use std::cmp::Ordering;

    fn params(temperature: f64, top_p: f64, top_k: u32) -> SamplingParams {
        SamplingParams::new(temperature, top_p, top_k).expect("valid params")
    }

    fn draws(sampler: &mut Sampler, logits: &[f32], count: usize) -> Vec<u32> {
        draws_with_keep(sampler, logits, count)
            .into_iter()
            .map(|(token, _)| token)
            .collect()
    }

    fn draws_with_keep(sampler: &mut Sampler, logits: &[f32], count: usize) -> Vec<(u32, usize)> {
        (0..count)
            .map(|_| sampler.sample_with_keep(logits).expect("sample"))
            .collect()
    }

    fn message(result: Result<SamplingParams, EngError>) -> String {
        match result {
            Err(EngError::BadRequest(text)) => text,
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    // The implementation before the partial-selection change, copied verbatim from `sample` with
    // `self` renamed to `sampler`. Every equivalence test compares against this reference. It
    // returns the token and the number of kept candidates, so a diverging kept count is visible
    // even when the token happens to match.
    fn reference_sample(sampler: &mut Sampler, logits: &[f32]) -> Result<(u32, usize), EngError> {
        check_logits(logits)?;
        if sampler.params.is_greedy() {
            return Ok((argmax(logits), 1));
        }
        let mut ranked: Vec<(u32, f64)> = logits
            .iter()
            .enumerate()
            .map(|(index, &logit)| (index as u32, f64::from(logit)))
            .collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        if sampler.params.top_k > 0 {
            ranked.truncate(usize::try_from(sampler.params.top_k).unwrap_or(usize::MAX));
        }
        let top = ranked.first().map_or(0.0, |entry| entry.1);
        let weights: Vec<f64> = ranked
            .iter()
            .map(|entry| ((entry.1 - top) / sampler.params.temperature).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let mut keep = 0_usize;
        let mut mass = 0.0_f64;
        for weight in &weights {
            mass += weight;
            keep += 1;
            if mass >= sampler.params.top_p * total {
                break;
            }
        }
        let target = sampler.rng.next_unit() * mass;
        let mut cumulative = 0.0_f64;
        for (entry, weight) in ranked.iter().zip(&weights).take(keep) {
            cumulative += weight;
            if cumulative > target {
                return Ok((entry.0, keep));
            }
        }
        // Rounding can leave `target` at the total. The last kept token is the answer then.
        ranked
            .get(keep.saturating_sub(1))
            .map(|entry| (entry.0, keep))
            .ok_or_else(|| invalid_logits("no candidate token"))
    }

    fn reference_draws(sampler: &mut Sampler, logits: &[f32], count: usize) -> Vec<u32> {
        reference_draws_with_keep(sampler, logits, count)
            .into_iter()
            .map(|(token, _)| token)
            .collect()
    }

    fn reference_draws_with_keep(
        sampler: &mut Sampler,
        logits: &[f32],
        count: usize,
    ) -> Vec<(u32, usize)> {
        (0..count)
            .map(|_| reference_sample(sampler, logits).expect("reference sample"))
            .collect()
    }

    fn random_row(rng: &mut SplitMix64, size: usize) -> Vec<f32> {
        (0..size)
            .map(|_| ((rng.next_unit() * 16.0) - 8.0) as f32)
            .collect()
    }

    // The (temperature, top_p, top_k) grid: very small and large temperatures, top_p 1.0 and 0.0,
    // and top_k 0 (off), 1, 3, and far above any vocabulary in use.
    fn grid() -> Vec<(f64, f64, u32)> {
        let mut combos = Vec::new();
        for temperature in [0.05, 0.7, 2.0] {
            for top_p in [0.0, 0.5, 0.95, 1.0] {
                for top_k in [0, 1, 3, 1_000_000] {
                    combos.push((temperature, top_p, top_k));
                }
            }
        }
        combos
    }

    // Draws `count` tokens twice from one seed: partial selection against the reference. The token
    // ids must be identical draw for draw, which also proves the random draws are consumed alike.
    fn assert_same_token_ids(
        logits: &[f32],
        temperature: f64,
        top_p: f64,
        top_k: u32,
        seed: u64,
        count: usize,
    ) {
        let settings = params(temperature, top_p, top_k);
        let mut partial = Sampler::new(settings, seed);
        let mut reference = Sampler::new(settings, seed);
        let got = draws(&mut partial, logits, count);
        let want = reference_draws(&mut reference, logits, count);
        assert_eq!(
            got,
            want,
            "vocab {}, temperature {temperature}, top_p {top_p}, top_k {top_k}, seed {seed}",
            logits.len()
        );
    }

    // Draws `count` (token, kept count) pairs twice from one seed: the sampler against the
    // reference. Both must be identical draw for draw: the kept count catches a wrong normaliser
    // even on rows where every draw still lands on the same token.
    fn assert_same_draws(
        label: &str,
        logits: &[f32],
        temperature: f64,
        top_p: f64,
        top_k: u32,
        seed: u64,
        count: usize,
    ) {
        let settings = params(temperature, top_p, top_k);
        let mut sampler = Sampler::new(settings, seed);
        let mut reference = Sampler::new(settings, seed);
        let got = draws_with_keep(&mut sampler, logits, count);
        let want = reference_draws_with_keep(&mut reference, logits, count);
        assert_eq!(
            got,
            want,
            "{label}, vocab {}, temperature {temperature}, top_p {top_p}, top_k {top_k}, seed {seed}",
            logits.len()
        );
    }

    // The review's concrete adversarial input: `size - 1` tail weights of about 1e-17 in row order
    // before the top logit 1.0. Each tail weight vanishes when added to 1.0 in ranked order, so
    // the ranked total is exactly 1.0; summed in row order the tail adds up first and the total
    // becomes about 1.0000000000025.
    fn tiny_tail_row(size: usize) -> Vec<f32> {
        let mut row = vec![-26.4_f32; size - 1];
        row.push(1.0);
        row
    }

    // A row whose weights span hundreds of magnitudes: the weight of index i is about 10^-(i % 321)
    // at temperature 0.7 (the top logit 1.0 is at index 0). Repeated exponents give ties as well.
    fn spread_magnitude_row(size: usize) -> Vec<f32> {
        (0..size)
            .map(|index| {
                let decades = (index % 321) as f64;
                (1.0_f64 - 0.7 * std::f64::consts::LN_10 * decades) as f32
            })
            .collect()
    }

    // The same row with its entries in a shuffled order. The order is seeded, so every run sees
    // the same row; the ranked order is untouched, but the row-order sums differ.
    fn shuffled(row: &[f32], seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        let mut out = row.to_vec();
        for index in (1..out.len()).rev() {
            let swap = (rng.next_unit() * (index + 1) as f64) as usize;
            out.swap(index, swap);
        }
        out
    }

    fn tie_pairs(size: usize) -> Vec<f32> {
        (0..size)
            .map(|index| ((size - 1 - index / 2) as f32) * 0.5)
            .collect()
    }

    fn one_spike_first(size: usize) -> Vec<f32> {
        let mut row = vec![0.0; size];
        row[0] = 8.0;
        row
    }

    fn one_spike_last(size: usize) -> Vec<f32> {
        let mut row = vec![0.0; size];
        row[size - 1] = 8.0;
        row
    }

    fn two_equal_spikes(size: usize) -> Vec<f32> {
        let mut row = vec![0.0; size];
        row[0] = 8.0;
        row[size - 1] = 8.0;
        row
    }

    // Reference values: the published SplitMix64 test vectors, and the same recurrence evaluated
    // with 64-bit shell arithmetic (`/home/dan/fifi/scrch/ee264a3/server-packets/splitmix.sh`).
    #[test]
    fn splitmix64_matches_the_reference_vectors_for_two_seeds() {
        let mut zero = SplitMix64::new(0);
        let first: Vec<u64> = (0..3).map(|_| zero.next_u64()).collect();
        assert_eq!(
            first,
            [
                16_294_208_416_658_607_535,
                7_960_286_522_194_355_700,
                487_617_019_471_545_679
            ]
        );
        let mut seeded = SplitMix64::new(1_234_567);
        let second: Vec<u64> = (0..5).map(|_| seeded.next_u64()).collect();
        assert_eq!(
            second,
            [
                6_457_827_717_110_365_317,
                3_203_168_211_198_807_973,
                9_817_491_932_198_370_423,
                4_593_380_528_125_082_431,
                16_408_922_859_458_223_821
            ]
        );
    }

    #[test]
    fn next_unit_is_the_top_53_bits_over_2_pow_53() {
        let mut rng = SplitMix64::new(0);
        let units: Vec<f64> = (0..3).map(|_| rng.next_unit()).collect();
        let expected = [
            0.883_310_808_213_642_6,
            0.431_527_997_048_509_97,
            0.026_433_771_592_597_743,
        ];
        for (got, want) in units.iter().zip(expected) {
            assert!((got - want).abs() < 1e-15, "unit {got} differs from {want}");
        }
    }

    #[test]
    fn params_accept_the_contract_bounds_and_reject_values_outside_them() {
        assert!(SamplingParams::new(0.0, 0.0, 0).is_ok());
        assert!(SamplingParams::new(2.0, 1.0, u32::MAX).is_ok());
        for bad in [2.000_1, -0.1, f64::NAN] {
            let text = message(SamplingParams::new(bad, 1.0, 0));
            assert!(text.contains("temperature"), "message was {text}");
        }
        for bad in [1.000_1, -0.1, f64::NAN] {
            let text = message(SamplingParams::new(1.0, bad, 0));
            assert!(text.contains("top_p"), "message was {text}");
        }
    }

    #[test]
    fn absent_request_fields_take_the_contract_defaults() {
        assert_eq!(
            SamplingParams::from_request(None, None, None, &SamplingDefaults::CONTRACT),
            Ok(params(0.7, 1.0, 0))
        );
        assert_eq!(DEFAULT_TEMPERATURE, 0.7);
    }

    #[test]
    fn a_top_k_above_u32_max_saturates() {
        assert_eq!(
            SamplingParams::from_request(
                Some(1.0),
                Some(1.0),
                Some(u64::MAX),
                &SamplingDefaults::CONTRACT
            ),
            Ok(params(1.0, 1.0, u32::MAX))
        );
    }

    /// The `generation_config.json` of the pinned edge0-35b checkpoint, quoted from the
    /// coordinator's copy of `/Users/dan/edge0/models/edge0-35b/generation_config.json`.
    const EDGE0_GENERATION_CONFIG: &str = r#"{
    "bos_token_id": 248044,
    "do_sample": true,
    "eos_token_id": [
        248046,
        248044
    ],
    "pad_token_id": 248044,
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95
}"#;

    /// A model directory that holds `text` as its `generation_config.json`.
    fn dir_with_generation_config(text: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(dir.path().join("generation_config.json"), text)
            .expect("write generation_config.json");
        dir
    }

    fn checkpoint_message(result: Result<LoadedDefaults, EngError>) -> String {
        match result {
            Err(EngError::Checkpoint(text)) => text,
            other => panic!("expected Checkpoint, got {other:?}"),
        }
    }

    #[test]
    fn omitted_request_fields_resolve_to_the_generation_config_values() {
        let dir = dir_with_generation_config(EDGE0_GENERATION_CONFIG);
        let loaded = SamplingDefaults::from_model_dir(dir.path()).expect("defaults");
        assert_eq!(
            loaded,
            LoadedDefaults {
                values: SamplingDefaults {
                    temperature: 1.0,
                    top_p: 0.95,
                    top_k: 20
                },
                from_file: true
            }
        );
        assert_eq!(
            SamplingParams::from_request(None, None, None, &loaded.values),
            Ok(params(1.0, 0.95, 20))
        );
    }

    #[test]
    fn from_model_dir_reports_whether_the_file_supplied_the_values() {
        // The start log line takes its "from file" label from this flag, so an absent file and
        // a file that happens to hold the contract defaults must stay distinguishable.
        let absent = tempfile::tempdir().expect("temp dir");
        assert_eq!(
            SamplingDefaults::from_model_dir(absent.path()),
            Ok(LoadedDefaults {
                values: SamplingDefaults::CONTRACT,
                from_file: false
            })
        );
        let dir = dir_with_generation_config(EDGE0_GENERATION_CONFIG);
        assert!(
            SamplingDefaults::from_model_dir(dir.path())
                .expect("defaults")
                .from_file
        );
    }

    #[test]
    fn a_request_temperature_of_zero_stays_greedy_over_the_generation_config_default() {
        let dir = dir_with_generation_config(EDGE0_GENERATION_CONFIG);
        let defaults = SamplingDefaults::from_model_dir(dir.path())
            .expect("defaults")
            .values;
        let set = SamplingParams::from_request(Some(0.0), None, None, &defaults).expect("set");
        assert_eq!(set, params(0.0, 0.95, 20));
        assert!(set.is_greedy(), "temperature 0 must stay greedy: {set:?}");
    }

    #[test]
    fn a_set_top_p_keeps_its_value_while_temperature_takes_the_generation_config_default() {
        let dir = dir_with_generation_config(EDGE0_GENERATION_CONFIG);
        let defaults = SamplingDefaults::from_model_dir(dir.path())
            .expect("defaults")
            .values;
        assert_eq!(
            SamplingParams::from_request(None, Some(0.5), None, &defaults),
            Ok(params(1.0, 0.5, 20))
        );
    }

    #[test]
    fn an_explicit_top_k_zero_overrides_the_file_default() {
        let dir = dir_with_generation_config(EDGE0_GENERATION_CONFIG);
        let defaults = SamplingDefaults::from_model_dir(dir.path())
            .expect("defaults")
            .values;
        // The file sets top_k 20; the request turns the filter off, and the request must win.
        assert_eq!(
            SamplingParams::from_request(None, None, Some(0), &defaults),
            Ok(params(1.0, 0.95, 0))
        );
    }

    #[test]
    fn an_explicit_top_p_one_overrides_the_file_default() {
        let dir = dir_with_generation_config(EDGE0_GENERATION_CONFIG);
        let defaults = SamplingDefaults::from_model_dir(dir.path())
            .expect("defaults")
            .values;
        // The file sets top_p 0.95; the request keeps every token, and the request must win.
        assert_eq!(
            SamplingParams::from_request(None, Some(1.0), None, &defaults),
            Ok(params(1.0, 1.0, 20))
        );
    }

    #[test]
    fn a_model_dir_without_generation_config_keeps_the_contract_defaults() {
        let dir = tempfile::tempdir().expect("temp dir");
        assert_eq!(
            SamplingDefaults::from_model_dir(dir.path()),
            Ok(LoadedDefaults {
                values: SamplingDefaults::CONTRACT,
                from_file: false
            })
        );
        // A file that lacks a key keeps the contract default of that key alone.
        let partial = dir_with_generation_config("{\"top_k\": 7}");
        assert_eq!(
            SamplingDefaults::from_model_dir(partial.path()),
            Ok(LoadedDefaults {
                values: SamplingDefaults {
                    temperature: 0.7,
                    top_p: 1.0,
                    top_k: 7
                },
                from_file: true
            })
        );
    }

    #[test]
    fn a_malformed_generation_config_is_an_error() {
        let broken = dir_with_generation_config("{");
        let text = checkpoint_message(SamplingDefaults::from_model_dir(broken.path()));
        assert!(text.contains("generation_config.json"), "{text}");
        assert!(text.contains("parse failed"), "{text}");
        let array = dir_with_generation_config("[1]");
        let text = checkpoint_message(SamplingDefaults::from_model_dir(array.path()));
        assert!(text.contains("must hold a JSON object"), "{text}");
        let typed = dir_with_generation_config("{\"temperature\": \"hot\"}");
        let text = checkpoint_message(SamplingDefaults::from_model_dir(typed.path()));
        assert!(text.contains("temperature must be a number"), "{text}");
    }

    #[test]
    fn an_out_of_range_generation_config_value_is_an_error() {
        let hot = dir_with_generation_config("{\"temperature\": 2.5}");
        let text = checkpoint_message(SamplingDefaults::from_model_dir(hot.path()));
        assert!(
            text.contains("temperature must be between 0 and 2"),
            "{text}"
        );
        let open = dir_with_generation_config("{\"top_p\": 1.5}");
        let text = checkpoint_message(SamplingDefaults::from_model_dir(open.path()));
        assert!(text.contains("top_p must be between 0 and 1"), "{text}");
        let negative = dir_with_generation_config("{\"top_k\": -3}");
        let text = checkpoint_message(SamplingDefaults::from_model_dir(negative.path()));
        assert!(
            text.contains("top_k must be a non-negative integer"),
            "{text}"
        );
        let huge = dir_with_generation_config("{\"top_k\": 4294967296}");
        let text = checkpoint_message(SamplingDefaults::from_model_dir(huge.path()));
        assert!(text.contains("top_k must be at most 4294967295"), "{text}");
    }

    #[test]
    fn a_wrong_typed_generation_config_value_names_the_json_type_only() {
        for (text, expected, leaked) in [
            (
                r#"{"temperature": "hot"}"#,
                "temperature must be a number, found string",
                "hot",
            ),
            (
                r#"{"temperature": [1]}"#,
                "temperature must be a number, found array",
                "[1]",
            ),
            (
                r#"{"top_p": {"a": 1}}"#,
                "top_p must be a number, found object",
                "\"a\"",
            ),
            (
                r#"{"top_p": true}"#,
                "top_p must be a number, found boolean",
                "true",
            ),
            (
                r#"{"top_k": "20"}"#,
                "top_k must be a non-negative integer, found string",
                "20",
            ),
            (
                r#"{"top_k": 1.5}"#,
                "top_k must be a non-negative integer, found number",
                "1.5",
            ),
        ] {
            let typed = dir_with_generation_config(text);
            let message = checkpoint_message(SamplingDefaults::from_model_dir(typed.path()));
            assert!(message.contains(expected), "{text}: {message}");
            // Only the JSON type name is echoed, never the value itself.
            assert!(
                !message.contains(leaked),
                "{text} leaked {leaked}: {message}"
            );
        }
    }

    #[test]
    fn a_json_null_in_generation_config_means_an_absent_key() {
        // Hugging Face configs carry an explicit null for an unset field, so null must not be a
        // type error: every one of the three keys falls back to its contract default.
        let dir =
            dir_with_generation_config("{\"temperature\": null, \"top_p\": null, \"top_k\": null}");
        assert_eq!(
            SamplingDefaults::from_model_dir(dir.path()),
            Ok(LoadedDefaults {
                values: SamplingDefaults::CONTRACT,
                from_file: true
            })
        );
    }

    #[test]
    fn a_directory_named_generation_config_is_an_error_that_names_the_path() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("generation_config.json");
        std::fs::create_dir(&path).expect("mkdir");
        let text = checkpoint_message(SamplingDefaults::from_model_dir(dir.path()));
        assert!(text.contains("must be a regular file"), "{text}");
        assert!(text.contains(path.to_str().expect("utf-8 path")), "{text}");
    }

    #[test]
    fn an_oversized_generation_config_is_an_error_that_names_the_path() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("generation_config.json");
        // One byte over the 1 MiB cap; the content never reaches the parser.
        std::fs::write(&path, vec![b'{'; (1 << 20) + 1]).expect("write generation_config.json");
        let text = checkpoint_message(SamplingDefaults::from_model_dir(dir.path()));
        assert!(text.contains("larger than 1 MiB"), "{text}");
        assert!(text.contains(path.to_str().expect("utf-8 path")), "{text}");
    }

    #[test]
    fn a_generation_config_of_exactly_one_mebibyte_still_loads() {
        // A valid object padded with spaces to exactly the 1 MiB cap: the cap rejects only a
        // file larger than the cap, so this one loads and its temperature wins.
        let dir = tempfile::tempdir().expect("temp dir");
        let prefix = r#"{"temperature":0.5,"pad":""#;
        let suffix = "\"}";
        let text = format!(
            "{prefix}{}{suffix}",
            " ".repeat((1 << 20) - prefix.len() - suffix.len())
        );
        assert_eq!(
            text.len(),
            1 << 20,
            "the padded file must hit the cap exactly"
        );
        std::fs::write(dir.path().join("generation_config.json"), text)
            .expect("write generation_config.json");
        assert_eq!(
            SamplingDefaults::from_model_dir(dir.path()),
            Ok(LoadedDefaults {
                values: SamplingDefaults {
                    temperature: 0.5,
                    top_p: 1.0,
                    top_k: 0
                },
                from_file: true
            })
        );
    }

    #[test]
    fn temperature_zero_picks_the_lowest_index_maximum_and_draws_nothing() {
        let mut sampler = Sampler::new(params(0.0, 0.1, 1), 7);
        let picked = draws(&mut sampler, &[1.0, 3.0, 3.0, 2.0], 3);
        assert_eq!(picked, [1, 1, 1]);
        assert_eq!(sampler.rng, SplitMix64::new(7), "greedy sampling drew");
    }

    #[test]
    fn a_greedy_sampler_picks_the_largest_logit() {
        let mut sampler = Sampler::greedy();
        assert_eq!(draws(&mut sampler, &[-5.0, -1.0, -3.0], 1), [1]);
        assert_eq!(draws(&mut sampler, &[0.5], 1), [0]);
    }

    // Seed 0 gives the units 0.8833, 0.4315, and 0.0264. Four equal logits split [0, 1) into four
    // quarters, so the units fall in quarters 3, 1, and 0.
    #[test]
    fn equal_logits_follow_the_seed_0_draws() {
        let mut sampler = Sampler::new(params(1.0, 1.0, 0), 0);
        assert_eq!(draws(&mut sampler, &[0.0; 4], 3), [3, 1, 0]);
    }

    #[test]
    fn a_top_k_above_the_vocabulary_size_keeps_every_token() {
        let mut sampler = Sampler::new(params(1.0, 1.0, 100), 0);
        assert_eq!(draws(&mut sampler, &[0.0; 4], 3), [3, 1, 0]);
    }

    // The logits are 2 ln 3 and 0. The weights are 1 and 1/9 at temperature 1. The first unit
    // 0.8833 times the total 1.111 is 0.981, which is below the first weight: token 0.
    #[test]
    fn temperature_one_favors_the_large_logit() {
        let mut sampler = Sampler::new(params(1.0, 1.0, 0), 0);
        assert_eq!(draws(&mut sampler, &[2.197_224_6, 0.0], 1), [0]);
    }

    // At temperature 2 the weights are 1 and 1/3. The first unit times the total 1.333 is 1.178,
    // which is above the first weight: token 1.
    #[test]
    fn temperature_two_flattens_the_distribution() {
        let mut sampler = Sampler::new(params(2.0, 1.0, 0), 0);
        assert_eq!(draws(&mut sampler, &[2.197_224_6, 0.0], 1), [1]);
    }

    #[test]
    fn top_k_one_always_picks_the_largest_logit() {
        let mut sampler = Sampler::new(params(1.0, 1.0, 1), 0);
        assert_eq!(draws(&mut sampler, &[0.0, 0.1, 0.2, 0.3], 20), [3; 20]);
    }

    #[test]
    fn top_k_two_draws_only_the_two_largest_logits() {
        let mut sampler = Sampler::new(params(1.0, 1.0, 2), 0);
        let picked = draws(&mut sampler, &[0.0, 0.1, 0.2, 0.3], 60);
        assert!(
            picked.iter().all(|&id| id == 2 || id == 3),
            "got {picked:?}"
        );
        assert!(picked.contains(&2) && picked.contains(&3), "got {picked:?}");
    }

    // Probabilities 0.4, 0.3, 0.2, and 0.1 as logits. The weights relative to the top are 1,
    // 0.75, 0.5, and 0.25 (total 2.5). A top_p of 0.6 needs mass 1.5, so tokens 0 and 1 stay
    // (mass 1.75). The units 0.8833, 0.4315, and 0.0264 times 1.75 are 1.546, 0.755, and 0.046:
    // tokens 1, 0, and 0. Without top_p the same units times 2.5 give tokens 2, 1, and 0.
    #[test]
    fn top_p_keeps_the_smallest_set_that_reaches_the_mass() {
        let logits = [0.4_f32.ln(), 0.3_f32.ln(), 0.2_f32.ln(), 0.1_f32.ln()];
        let mut sampler = Sampler::new(params(1.0, 0.6, 0), 0);
        assert_eq!(draws(&mut sampler, &logits, 3), [1, 0, 0]);
    }

    // A top_p of 0.85 needs mass 2.125, so tokens 0, 1, and 2 stay (mass 2.25). The units times
    // 2.25 are 1.987, 0.971, and 0.059: tokens 2, 0, and 0. Without top_p the second token is 1.
    #[test]
    fn a_larger_top_p_keeps_a_larger_set() {
        let logits = [0.4_f32.ln(), 0.3_f32.ln(), 0.2_f32.ln(), 0.1_f32.ln()];
        let mut sampler = Sampler::new(params(1.0, 0.85, 0), 0);
        assert_eq!(draws(&mut sampler, &logits, 3), [2, 0, 0]);
    }

    #[test]
    fn a_top_p_of_zero_keeps_only_the_largest_token() {
        let mut sampler = Sampler::new(params(1.0, 0.0, 0), 0);
        assert_eq!(draws(&mut sampler, &[0.1, 0.5, 0.2], 20), [1; 20]);
    }

    #[test]
    fn the_same_seed_repeats_and_another_seed_differs() {
        let logits = [0.0; 8];
        let sample = |seed: u64| draws(&mut Sampler::new(params(1.0, 1.0, 0), seed), &logits, 16);
        assert_eq!(sample(0), sample(0));
        assert_ne!(sample(0), sample(1));
    }

    #[test]
    fn invalid_logits_fail_with_a_backend_error() {
        let mut sampler = Sampler::new(params(1.0, 1.0, 0), 0);
        for logits in [&[][..], &[0.0, f32::NAN][..], &[f32::INFINITY, 0.0][..]] {
            match sampler.sample(logits) {
                Err(EngError::Backend { operation, .. }) => assert_eq!(operation, "sample"),
                other => panic!("expected Backend for {logits:?}, got {other:?}"),
            }
        }
        let mut greedy = Sampler::greedy();
        assert!(greedy.sample(&[f32::NAN]).is_err(), "greedy took NaN");
    }

    #[test]
    fn entropy_seeds_differ_between_calls() {
        assert_ne!(entropy_seed(), entropy_seed());
    }

    // Two samplers from `from_entropy` take different seeds, so their draws differ. Equal draws by
    // chance have probability 16^-32: each of the 32 draws matches with probability 1/16.
    #[test]
    fn from_entropy_samplers_draw_different_sequences() {
        let logits = [0.0_f32; 16];
        let sample = || draws(&mut Sampler::from_entropy(params(1.0, 1.0, 0)), &logits, 32);
        assert_ne!(sample(), sample());
    }

    // The order is top_k, renormalise, then top_p. Probabilities 0.4, 0.3, 0.2, and 0.1 give the
    // weights 1, 0.75, 0.5, and 0.25 against the largest. A top_k of 3 keeps the first three
    // (truncated total 2.25). A top_p of 0.75 needs mass 0.75 times 2.25, which is 1.6875, so
    // tokens 0 and 1 stay (mass 1.75). The seed-0 unit 0.8833 times 1.75 is 1.546: token 1.
    // If top_p used the full-vocabulary total 2.5 (needed mass 1.875), tokens 0 to 2 would stay
    // (mass 2.25), and the draw 0.8833 times 2.25 is 1.987: token 2.
    #[test]
    fn top_p_applies_to_the_mass_that_top_k_leaves() {
        let logits = [0.4_f32.ln(), 0.3_f32.ln(), 0.2_f32.ln(), 0.1_f32.ln()];
        let mut sampler = Sampler::new(params(1.0, 0.75, 3), 0);
        assert_eq!(draws(&mut sampler, &logits, 1), [1]);
    }

    // The ranked order is the row order here, and the weights are 1 and, three times,
    // w = exp(-37.352...) ≈ 6e-17: each w vanishes when added to 1.0 on its own (w < 2^-53), so
    // the left-to-right sum is exactly 1.0 in every row order. Adding the other way would give
    // 1.0 + 3w = 1.0000000000000002 and must not happen here; the review's adversarial rows below
    // pin the same property at 248,077 entries.
    #[test]
    fn the_normalizer_adds_the_ranked_weights_left_to_right() {
        let v = -37.352_19_f32;
        for logits in [[0.0, v, v, v], [v, v, v, 0.0], [v, 0.0, v, v]] {
            let (_, got) = ranked_weights(&ranked_row(&logits), 0.0, 1.0);
            assert_eq!(
                got.to_bits(),
                1.0_f64.to_bits(),
                "normalizer for {logits:?} was {got}, want exactly 1.0"
            );
        }
    }

    // Four equal logits give the weights 1, 1, 1, 1 (total 4). A top_p of 0.5 needs the mass 2,
    // and the loop stops when the mass REACHES 2, so exactly two tokens stay (mass 2). The seed-0
    // units 0.8833, 0.4315, and 0.0264 times the mass 2 are 1.767, 0.863, and 0.053: the draws are
    // the tokens 1, 0, and 0. Comparing only the mass would keep three tokens and draw 2, 1, 0.
    #[test]
    fn top_p_keeps_the_token_whose_weight_reaches_the_mass() {
        let logits = [0.0; 4];
        let mut sampler = Sampler::new(params(1.0, 0.5, 0), 0);
        let picked = draws(&mut sampler, &logits, 3);
        assert_eq!(picked, [1, 0, 0], "boundary draws were {picked:?}");
        let mut reference = Sampler::new(params(1.0, 0.5, 0), 0);
        let want = reference_draws(&mut reference, &logits, 3);
        assert_eq!(picked, want, "reference drew {want:?}");
    }

    // A flat 5,000-entry row at top_p 0.95 keeps the 4,750 tokens whose mass reaches 0.95 * 5,000,
    // far past the first window of 64, so the nucleus may not be capped. The seed-0 units 0.8833,
    // 0.4315, and 0.0264 times the kept mass 4,750 are 4,195.73, 2,049.76, and 125.56: the draws
    // are the tokens 4,195, 2,049, and 125. A cap at 64 would draw 56, 27, and 1.
    #[test]
    fn a_flat_distribution_keeps_the_whole_nucleus_without_a_cap() {
        let logits = vec![0.0; 5_000];
        let mut sampler = Sampler::new(params(1.0, 0.95, 0), 0);
        let picked = draws(&mut sampler, &logits, 3);
        assert_eq!(picked, [4_195, 2_049, 125], "flat draws were {picked:?}");
        let mut reference = Sampler::new(params(1.0, 0.95, 0), 0);
        let want = reference_draws(&mut reference, &logits, 3);
        assert_eq!(picked, want, "reference drew {want:?}");
    }

    // Adversarial rows over the whole grid: all-equal values, ties in pairs, one spike first and
    // last, and two equal spikes (a tie at the largest logit). Every draw must match the reference.
    // The -inf, NaN, and empty rows fail with the exact same error in both, as they do today.
    #[test]
    fn partial_selection_matches_the_reference_on_adversarial_rows() {
        let combos = grid();
        let rows: [(&str, Vec<f32>); 5] = [
            ("all equal", vec![0.0; 128]),
            ("tie pairs", tie_pairs(64)),
            ("one spike first", one_spike_first(128)),
            ("one spike last", one_spike_last(128)),
            ("two equal spikes", two_equal_spikes(128)),
        ];
        for (name, logits) in &rows {
            for (step, &(temperature, top_p, top_k)) in combos.iter().enumerate() {
                let settings = params(temperature, top_p, top_k);
                let seed = step as u64;
                let mut partial = Sampler::new(settings, seed);
                let mut reference = Sampler::new(settings, seed);
                let got = draws(&mut partial, logits, 8);
                let want = reference_draws(&mut reference, logits, 8);
                assert_eq!(
                    got, want,
                    "{name}, temperature {temperature}, top_p {top_p}, top_k {top_k}, seed {seed}"
                );
            }
        }
        let settings = params(0.7, 0.95, 0);
        for logits in [
            &[][..],
            &[0.0, f32::NEG_INFINITY][..],
            &[f32::NAN, 1.0][..],
            &[f32::INFINITY][..],
        ] {
            let mut partial = Sampler::new(settings, 1);
            let mut reference = Sampler::new(settings, 1);
            assert_eq!(
                partial.sample_with_keep(logits),
                reference_sample(&mut reference, logits),
                "logits {logits:?}"
            );
        }
    }

    // Seeded random rows over vocabulary sizes 1 to 248,077, a few thousand of them, each over six
    // grid points and several draws: the token ids must match the reference for the same seed.
    #[test]
    fn partial_selection_matches_the_reference_on_random_rows_over_the_grid() {
        let mut rng = SplitMix64::new(0x5EED_F1F1);
        let combos = grid();
        let mut rows: Vec<Vec<f32>> = Vec::new();
        for &vocab in &[1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 1024] {
            for _ in 0..240 {
                rows.push(random_row(&mut rng, vocab));
            }
        }
        for _ in 0..24 {
            rows.push(random_row(&mut rng, 4096));
        }
        for _ in 0..3 {
            rows.push(random_row(&mut rng, 248_077));
        }
        for (row_index, logits) in rows.iter().enumerate() {
            let count = if logits.len() > 10_000 { 2 } else { 3 };
            for step in 0..6 {
                let (temperature, top_p, top_k) = combos[(row_index + step * 17) % combos.len()];
                let seed = ((row_index as u64) << 8) | step as u64;
                assert_same_token_ids(logits, temperature, top_p, top_k, seed, count);
            }
        }
    }

    // The review's concrete input at full scale: 248,076 tail weights of about 1e-17 in row order
    // before the top logit 1.0, and the same row shuffled. In ranked order each tail weight
    // vanishes when added to 1.0, so the total is exactly 1.0, `mass >= top_p * total` is reached
    // at the first candidate, and the reference keeps one candidate and returns the top token for
    // every draw. Summing the weights in row order totals about 1.0000000000025 and keeps all
    // 248,077 candidates instead.
    #[test]
    fn the_adversarial_mass_row_keeps_only_the_top_token() {
        for (name, logits) in [
            ("row order", tiny_tail_row(248_077)),
            ("shuffled", shuffled(&tiny_tail_row(248_077), 0x0051_1F17)),
        ] {
            let top = argmax(&logits);
            for temperature in [0.05, 0.5, 0.7] {
                let mut sampler = Sampler::new(params(temperature, 1.0, 0), 0);
                let picked = draws_with_keep(&mut sampler, &logits, 3);
                assert_eq!(
                    picked,
                    vec![(top, 1); 3],
                    "{name}, temperature {temperature} drew {picked:?}"
                );
            }
        }
    }

    // The same rows against the reference at the points where this bug class lives: the token AND
    // the kept count must match draw for draw, in the review's row order and shuffled.
    #[test]
    fn adversarial_mass_rows_match_the_reference_at_full_scale() {
        let combos = [
            (0.7, 1.0, 0),
            (0.7, 0.95, 0),
            (0.7, 0.5, 0),
            (1.0, 1.0, 0),
            (0.05, 1.0, 0),
            (2.0, 0.95, 0),
            (0.7, 1.0, 20),
            (0.7, 0.95, 20),
        ];
        for (name, logits) in [
            ("row order", tiny_tail_row(248_077)),
            ("shuffled", shuffled(&tiny_tail_row(248_077), 0x0051_1F17)),
        ] {
            for (step, &(temperature, top_p, top_k)) in combos.iter().enumerate() {
                assert_same_draws(name, &logits, temperature, top_p, top_k, step as u64, 2);
            }
        }
    }

    // A row whose weights span hundreds of magnitudes, in its own order and shuffled: the total is
    // the ranked-order sum, and the early exit may only stop once the tail provably vanishes.
    #[test]
    fn spread_magnitude_rows_match_the_reference_at_full_scale() {
        let combos = [
            (0.7, 1.0, 0),
            (0.7, 0.95, 0),
            (0.7, 0.5, 0),
            (1.0, 1.0, 0),
            (0.05, 1.0, 0),
            (2.0, 0.95, 0),
            (0.7, 1.0, 20),
            (0.7, 0.95, 20),
        ];
        for (name, logits) in [
            ("row order", spread_magnitude_row(248_077)),
            (
                "shuffled",
                shuffled(&spread_magnitude_row(248_077), 0x0051_1F17),
            ),
        ] {
            for (step, &(temperature, top_p, top_k)) in combos.iter().enumerate() {
                assert_same_draws(name, &logits, temperature, top_p, top_k, step as u64, 2);
            }
        }
    }

    // The server default (top_k 20, top_p 0.95, temperature 1.0 and 0.7) and its neighbours: the
    // bounded heap must return the same ranked top-k as the reference's sort, including the
    // tie-break that sends a tie to the lower index.
    #[test]
    fn the_top_k_twenty_default_matches_the_reference() {
        let mut rng = SplitMix64::new(0x007E_5720);
        let rows: [(&str, Vec<f32>); 5] = [
            ("random 248k", random_row(&mut rng, 248_077)),
            ("all equal 4096", vec![0.0; 4_096]),
            ("tie pairs", tie_pairs(64)),
            ("one spike first", one_spike_first(512)),
            ("one spike last", one_spike_last(512)),
        ];
        let combos = [
            (0.7, 0.95, 20),
            (1.0, 0.95, 20),
            (0.7, 1.0, 20),
            (0.7, 0.5, 20),
        ];
        for (name, logits) in &rows {
            for (step, &(temperature, top_p, top_k)) in combos.iter().enumerate() {
                assert_same_draws(name, logits, temperature, top_p, top_k, step as u64, 4);
            }
        }
    }

    // Timing harness, not part of the normal suite. Run:
    //   cargo test --release --offline timing_248077_row -- --ignored --nocapture
    // It prints microseconds per sample for a 248,077-entry row, the reference against the
    // sampler, at the four points this fix round measures: (T0.7, P0.95, K0), (T1.0, P0.95, K20),
    // (T0.7, P1.0, K0), and (T0.7, P0.95, K20). The server defaults verified in
    // server/validate.rs:46-50 and family/sampler.rs are temperature 0.7 and top_p 1.0; the
    // packet requested these points, so they are what the harness measures. The top_k 20 rows
    // must stay under 1.5 ms per sample on this host.
    #[test]
    #[ignore = "timing harness; run with --ignored --nocapture"]
    fn timing_248077_row_reference_against_partial_selection() {
        let mut rng = SplitMix64::new(7);
        let logits = random_row(&mut rng, 248_077);
        let count = 64_usize;
        for &(temperature, top_p, top_k) in &[
            (0.7, 0.95, 0),
            (1.0, 0.95, 20),
            (0.7, 1.0, 0),
            (0.7, 0.95, 20),
        ] {
            let settings = params(temperature, top_p, top_k);
            let start = std::time::Instant::now();
            let mut reference = Sampler::new(settings, 1);
            let want = reference_draws_with_keep(&mut reference, &logits, count);
            let reference_seconds = start.elapsed().as_secs_f64();
            let start = std::time::Instant::now();
            let mut sampler = Sampler::new(settings, 1);
            let got = draws_with_keep(&mut sampler, &logits, count);
            let sampler_seconds = start.elapsed().as_secs_f64();
            assert_eq!(got, want, "timing run diverged from the reference");
            let per_sample = sampler_seconds / count as f64;
            println!(
                "vocab 248077, temperature {temperature}, top_p {top_p}, top_k {top_k}, \
                 {count} draws each: reference {:.1} µs/sample, sampler {:.1} µs/sample",
                reference_seconds * 1e6 / count as f64,
                per_sample * 1e6
            );
            if top_k == 20 {
                assert!(
                    per_sample < 1.5e-3,
                    "top_k 20 took {:.1} µs/sample, want under 1500",
                    per_sample * 1e6
                );
            }
        }
    }
}
