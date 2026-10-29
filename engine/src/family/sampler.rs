//! Host-side token sampling behind the executor seam.
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

/// The contract maximum for `temperature` (the canonical request encoding of `kernels/sample_canon`
/// names the same bound; a unit test pins the two together until they share one definition).
pub(crate) const MAX_TEMPERATURE: f64 = 2.0;

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

    /// The softmax temperature; 0 selects greedy decoding.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "staged for the CUDA device sampler")
    )]
    pub(crate) fn temperature(&self) -> f64 {
        self.temperature
    }

    /// The cumulative mass kept by nucleus sampling.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "staged for the CUDA device sampler")
    )]
    pub(crate) fn top_p(&self) -> f64 {
        self.top_p
    }

    /// The number of best candidates kept; 0 keeps every token.
    #[cfg_attr(
        not(any(test, all(feature = "native-cuda", target_os = "linux"))),
        expect(dead_code, reason = "staged for the CUDA device sampler")
    )]
    pub(crate) fn top_k(&self) -> u32 {
        self.top_k
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
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "staged for the CUDA device sampler")
    )]
    seed: u64,
    rng: SplitMix64,
}

impl Sampler {
    /// A sampler with the given settings and an explicit seed. Tests and replays use this.
    pub(crate) fn new(params: SamplingParams, seed: u64) -> Self {
        Self {
            params,
            seed,
            rng: SplitMix64::new(seed),
        }
    }

    /// A sampler seeded once from system randomness. The server uses this for each job.
    pub(crate) fn from_entropy(params: SamplingParams) -> Self {
        Self::new(params, entropy_seed())
    }

    /// True when the settings select greedily, so this sampler draws nothing.
    pub(crate) fn is_greedy(&self) -> bool {
        self.params.is_greedy()
    }

    /// A greedy sampler. It never draws from its generator.
    pub(crate) fn greedy() -> Self {
        Self::new(SamplingParams::greedy(), 0)
    }

    /// The settings in force. The server's end-to-end probe records them for its tests.
    pub(crate) fn params(&self) -> SamplingParams {
        self.params
    }

    /// The seed that the generator started from. The generator state is not exposed.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "staged for the CUDA device sampler")
    )]
    pub(crate) fn seed(&self) -> u64 {
        self.seed
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
        let rng = &mut self.rng;
        pick_row(self.params, logits, || rng.next_unit())
    }

    /// Draws from the complete stable top-20 of a device-validated finite vocabulary.
    /// The original f64 weight and SplitMix64 walk keep the full-row sampler's token stream.
    #[cfg_attr(
        not(any(test, all(feature = "native-cuda", target_os = "linux"))),
        expect(dead_code, reason = "used by the CUDA device shortlist")
    )]
    pub(crate) fn sample_shortlist(
        &mut self,
        ranked: &[(u32, f32)],
        vocab: usize,
    ) -> Result<u32, EngError> {
        if self.params.is_greedy() || !(1..=20).contains(&self.params.top_k) {
            return Err(EngError::BadRequest(
                "device shortlist requires non-greedy top_k 1 through 20".to_owned(),
            ));
        }
        if vocab == 0 || u32::try_from(vocab).is_err() || ranked.len() != vocab.min(20) {
            return Err(invalid_logits("device shortlist has the wrong length"));
        }
        for (index, &(id, value)) in ranked.iter().enumerate() {
            if id as usize >= vocab
                || !value.is_finite()
                || ranked[..index].iter().any(|entry| entry.0 == id)
                || (index > 0
                    && compare_ranked(&ranked[index - 1], &ranked[index]) != Ordering::Less)
            {
                return Err(invalid_logits("device shortlist is invalid or not ranked"));
            }
        }
        let ranked = &ranked[..vocab.min(self.params.top_k as usize)];
        let top = f64::from(ranked[0].1);
        let (weights, total) = ranked_weights(ranked, top, self.params.temperature);
        let rng = &mut self.rng;
        Ok(draw_kept(self.params, ranked, &weights, total, || rng.next_unit())?.0)
    }

    /// The token and the number of kept candidates that `sample_with_keep` returns when the
    /// generator's next draw is `unit`. The generator does not move, so the call is repeatable.
    /// `unit` must lie in [0, 1); any other value, NaN included, fails with `BadRequest` before
    /// the row is read. A greedy row ignores a valid `unit`, and invalid logits fail as in
    /// `sample_with_keep`.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "staged for the CUDA device sampler")
    )]
    pub(crate) fn pick_with_unit(
        &self,
        logits: &[f32],
        unit: f64,
    ) -> Result<(u32, usize), EngError> {
        if !(0.0..1.0).contains(&unit) {
            return Err(EngError::BadRequest(format!(
                "the sampling unit must be in [0, 1), got {unit}"
            )));
        }
        pick_row(self.params, logits, || unit)
    }
}

/// The token and kept count of one row under `params`. The path is the sampler's whole decision:
/// `check_logits`, the greedy argmax, the prefix path, and the ranked paths. `draw` supplies the
/// call's one random unit and runs exactly once, inside `draw_kept` after the path is decided;
/// a greedy row and an error path never call it.
fn pick_row(
    params: SamplingParams,
    logits: &[f32],
    draw: impl FnOnce() -> f64,
) -> Result<(u32, usize), EngError> {
    check_logits(logits)?;
    if params.is_greedy() {
        return Ok((argmax(logits), 1));
    }
    let temperature = params.temperature;
    let top = top_logit(logits);
    let top_k = usize::try_from(params.top_k).unwrap_or(usize::MAX);
    // The open-vocabulary prefix path: with `top_k` off on a row longer than `PREFIX_ROWS`,
    // rank only the prefix first; when its weights provably vanish inside it (`closed`), the
    // full row's weight walk would stop at the same weight and total, so the prefix answers
    // without the full-row sort. Otherwise the full path below runs exactly as before. The
    // random draw is taken only inside `draw_kept`, after the path is decided, so a fallback
    // never spends a second draw.
    if top_k == 0 && logits.len() > PREFIX_ROWS {
        let prefix = ranked_prefix(logits, PREFIX_ROWS);
        let (weights, total, closed) = ranked_weights_prefix(&prefix, top, temperature);
        if closed {
            #[cfg(test)]
            sample_path::note_prefix();
            return draw_kept(params, &prefix, &weights, total, draw);
        }
        #[cfg(test)]
        sample_path::note_fallback();
    }
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
    let (weights, total) = ranked_weights(&ranked, top, temperature);
    draw_kept(params, &ranked, &weights, total, draw)
}

/// The one keep and cumulative walk every path draws through: `weights` are the weights of
/// `ranked` in ranked order and `total` their sum. The walk is the reference's: the keep loop
/// walks the ranked weights left to right and keeps the smallest prefix whose mass reaches
/// `top_p` times the total, the call's single `draw()` scales the kept mass, and the cumulative
/// walk returns the first kept token whose cumulative weight is above the target. Every path
/// consumes exactly one draw. Returns the token and the number of kept candidates.
fn draw_kept(
    params: SamplingParams,
    ranked: &[(u32, f32)],
    weights: &[f64],
    total: f64,
    draw: impl FnOnce() -> f64,
) -> Result<(u32, usize), EngError> {
    let threshold = params.top_p * total;
    let mut keep = 0_usize;
    let mut mass = 0.0_f64;
    for &w in weights {
        mass += w;
        keep += 1;
        if mass >= threshold {
            break;
        }
    }
    let target = draw() * mass;
    let mut cumulative = 0.0_f64;
    for (entry, &w) in ranked.iter().zip(weights).take(keep) {
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

/// The number of ranked entries the prefix path ranks before it falls back to the full ranked
/// row: with `top_k` off on a row longer than this, the prefix answers only when its weights
/// provably vanish inside it, which the observed nuclei do by a wide margin at every sampling
/// temperature in use.
const PREFIX_ROWS: usize = 1_024;

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

/// The present sort's composite rank key of an entry: `(order_key(logit) << 32) | !index`.
/// Larger keys rank first and the reversed index breaks ties toward the lower index, exactly
/// as `compare_ranked` orders them; distinct indices give distinct keys, so the key is a total
/// order and a key sort reproduces the comparator sort.
fn ranked_key(entry: &(u32, f32)) -> u64 {
    (u64::from(order_key(entry.1)) << 32) | u64::from(!entry.0)
}

/// The whole row in ranked order: larger logit first, a tie to the lower index, exactly the
/// reference sort's order. The composite key (logit key, reversed index) is a total order, so a
/// key sort reproduces the reference's comparator sort at a fraction of its cost.
fn ranked_row(logits: &[f32]) -> Vec<(u32, f32)> {
    let mut pairs = candidate_pairs(logits);
    pairs.sort_unstable_by_key(ranked_key);
    // Ascending keys give (logit up, index down); reversed they give the ranked order.
    pairs.reverse();
    pairs
}

/// The first `rows` entries of the ranked order, in ranked order, found in O(n): a partial
/// selection moves the `rows` largest composite keys to the back of the row without the full
/// sort, and the selected entries are sorted by the same key afterwards, because the order of
/// the elements around the selection point is unspecified. The entries, keys, and tie-breaks
/// are exactly `ranked_row`'s, so the result is the head of the full ranked row.
/// `rows >= logits.len()` returns the whole ranked row.
fn ranked_prefix(logits: &[f32], rows: usize) -> Vec<(u32, f32)> {
    let rows = rows.min(logits.len());
    let mut pairs = candidate_pairs(logits);
    if rows < pairs.len() {
        if rows == 0 {
            pairs.clear();
            return pairs;
        }
        // Ascending keys put the `rows` largest entries in the suffix past index `len - rows`;
        // the selection leaves the order on both sides unspecified, so the suffix is sorted by
        // the same key afterwards.
        let at = pairs.len() - rows;
        pairs.select_nth_unstable_by_key(at, ranked_key);
        pairs.drain(..at);
    }
    pairs.sort_unstable_by_key(ranked_key);
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
    let (weights, total, _) = ranked_weights_prefix(ranked, top, temperature);
    (weights, total)
}

/// The weights, the normaliser, and whether the tail provably vanished: `closed` is true when
/// the loop stopped at a weight no larger than a stalled previous one — the same walk over a
/// longer ranked row stops at the same weight with the same total — and false when every entry
/// of `ranked` was consumed, so a longer row may still change the total and only the full row
/// can answer. This is the one loop `ranked_weights` runs.
fn ranked_weights_prefix(
    ranked: &[(u32, f32)],
    top: f64,
    temperature: f64,
) -> (Vec<f64>, f64, bool) {
    let mut weights = Vec::with_capacity(ranked.len());
    let mut total = 0.0_f64;
    let mut stalled: Option<f64> = None;
    let mut closed = false;
    for &(_, logit) in ranked {
        let w = weight(logit, top, temperature);
        if stalled.is_some_and(|previous| w <= previous) {
            closed = true;
            break;
        }
        let before = total;
        total += w;
        weights.push(w);
        stalled = (total == before).then_some(w);
    }
    (weights, total, closed)
}

/// Test-only bookkeeping of which sampling path answered, so a test can assert that the ranked
/// prefix answered a peaked row and that the full ranked row answered a flat one. The counters
/// are per thread and the harness runs each test on its own thread, so parallel tests cannot
/// pollute each other's counts.
#[cfg(test)]
mod sample_path {
    use std::cell::Cell;

    thread_local! {
        static PREFIX: Cell<u64> = const { Cell::new(0) };
        static FALLBACK: Cell<u64> = const { Cell::new(0) };
    }

    /// Records that the ranked prefix answered a draw.
    pub(super) fn note_prefix() {
        PREFIX.with(|count| count.set(count.get() + 1));
    }

    /// Records that the prefix did not close and the full ranked row answered.
    pub(super) fn note_fallback() {
        FALLBACK.with(|count| count.set(count.get() + 1));
    }

    /// Takes both counters and resets them: (prefix answers, fallback answers).
    pub(super) fn take() -> (u64, u64) {
        let prefix = PREFIX.with(|count| count.replace(0));
        let fallback = FALLBACK.with(|count| count.replace(0));
        (prefix, fallback)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_TEMPERATURE, LoadedDefaults, PREFIX_ROWS, Sampler, SamplingDefaults,
        SamplingParams, SplitMix64, argmax, check_logits, entropy_seed, invalid_logits,
        ranked_prefix, ranked_row, ranked_weights, ranked_weights_prefix, sample_path, top_logit,
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

    fn assert_compact_tokens(row: &[f32], settings: SamplingParams, seed: u64) {
        let mut ranked = ranked_row(row);
        ranked.truncate(20);
        let mut full = Sampler::new(settings, seed);
        let mut compact = Sampler::new(settings, seed);
        for _ in 0..64 {
            assert_eq!(
                compact
                    .sample_shortlist(&ranked, row.len())
                    .expect("compact draw"),
                full.sample(row).expect("full-row draw"),
                "seed {seed}, settings {settings:?}"
            );
        }
    }

    #[test]
    fn compact_shortlist_matches_full_row_tokens_across_settings_and_seeds() {
        let rows = [
            random_row(&mut SplitMix64::new(31), 1_031),
            vec![0.0; 37],
            vec![-0.0, 0.0, -0.0],
            one_spike_first(41),
            spread_magnitude_row(257),
        ];
        for row in rows {
            for (temperature, top_p) in [
                (0.05, 0.95),
                (0.7, 0.95),
                (1.0, 0.95),
                (2.0, 0.95),
                (1.0, 0.0),
                (1.0, 1.0),
            ] {
                for top_k in 1..=20 {
                    for seed in [0, 7, u64::MAX] {
                        assert_compact_tokens(&row, params(temperature, top_p, top_k), seed);
                    }
                }
            }
        }
    }

    #[test]
    fn compact_shortlist_rejects_bad_candidates_without_spending_randomness() {
        let row = random_row(&mut SplitMix64::new(41), 127);
        let mut good = ranked_row(&row);
        good.truncate(20);
        let mut wrong_id = good.clone();
        wrong_id[19].0 = row.len() as u32;
        let mut duplicate = good.clone();
        duplicate[19].0 = duplicate[0].0;
        let mut non_finite = good.clone();
        non_finite[19].1 = f32::NAN;
        let mut unsorted = good.clone();
        unsorted.swap(0, 1);
        for bad in [
            good[..19].to_vec(),
            wrong_id,
            duplicate,
            non_finite,
            unsorted,
        ] {
            let settings = params(1.0, 0.95, 20);
            let mut full = Sampler::new(settings, 73);
            let mut compact = Sampler::new(settings, 73);
            assert!(compact.sample_shortlist(&bad, row.len()).is_err());
            assert!(compact.sample_shortlist(&good, 0).is_err());
            for _ in 0..32 {
                assert_eq!(
                    compact.sample_shortlist(&good, row.len()).unwrap(),
                    full.sample(&row).unwrap(),
                );
            }
        }
    }

    #[test]
    fn compact_shortlist_refuses_unsupported_settings_without_spending_randomness() {
        let row = random_row(&mut SplitMix64::new(51), 127);
        let mut ranked = ranked_row(&row);
        ranked.truncate(20);
        for settings in [
            SamplingParams::greedy(),
            params(1.0, 0.95, 0),
            params(1.0, 0.95, 21),
            params(1.0, 0.95, u32::MAX),
        ] {
            let mut full = Sampler::new(settings, 83);
            let mut compact = Sampler::new(settings, 83);
            assert!(compact.sample_shortlist(&ranked, row.len()).is_err());
            for _ in 0..32 {
                assert_eq!(compact.sample(&row).unwrap(), full.sample(&row).unwrap());
            }
        }
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
    // with 64-bit shell arithmetic (`<scratch>/ee264a3/server-packets/splitmix.sh`).
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

    /// The `generation_config.json` of the pinned checkpoint, quoted from a copy of
    /// `<model_dir>/generation_config.json`.
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

    // The grid over rows on both ranked paths (a one-entry row, small rows, a flat 1,025-entry row
    // that falls back, a peaked 2,000-entry row that the prefix answers, and tie pairs): a second
    // sampler of the same seed draws with `sample_with_keep`, `reference_sample` draws the same
    // unit from a third sampler, and a separate `SplitMix64` hands the unit to `pick_with_unit`.
    // All three give the same (token, keep) for four consecutive draws.
    //
    // Mutation check: in `pick_with_unit`, replace `|| unit` by `|| 0.5`. The unit then differs
    // from the generator's draws, so the rows of 64 and more entries give other tokens and the
    // test fails.
    #[test]
    fn pick_with_unit_equals_sample_with_keep_and_the_reference_over_the_grid() {
        let mut row_rng = SplitMix64::new(0xD5_0001);
        let rows = [
            random_row(&mut row_rng, 1),
            random_row(&mut row_rng, 5),
            random_row(&mut row_rng, 64),
            random_row(&mut row_rng, 1_025),
            one_spike_first(2_000),
            tie_pairs(64),
        ];
        for (row_index, logits) in rows.iter().enumerate() {
            for (step, &(temperature, top_p, top_k)) in grid().iter().enumerate() {
                let settings = params(temperature, top_p, top_k);
                let seed = ((row_index as u64) << 8) | step as u64;
                let mut sampled = Sampler::new(settings, seed);
                let mut reference = Sampler::new(settings, seed);
                let picker = Sampler::new(settings, seed);
                let mut units = SplitMix64::new(seed);
                for draw in 0..4 {
                    let want = sampled.sample_with_keep(logits).expect("sample_with_keep");
                    let reference_want =
                        reference_sample(&mut reference, logits).expect("reference");
                    let got = picker
                        .pick_with_unit(logits, units.next_unit())
                        .expect("pick_with_unit");
                    let label = format!(
                        "row {row_index} ({}), temperature {temperature}, top_p {top_p}, \
                         top_k {top_k}, seed {seed}, draw {draw}",
                        logits.len()
                    );
                    assert_eq!(got, want, "against sample_with_keep: {label}");
                    assert_eq!(got, reference_want, "against the reference: {label}");
                }
            }
        }
        // The grid must keep covering both ranked paths, or the equality above proves less.
        let (prefix_answers, fallback_answers) = sample_path::take();
        assert!(
            prefix_answers > 0,
            "no draw of the grid took the prefix path"
        );
        assert!(
            fallback_answers > 0,
            "no draw of the grid took the fallback path"
        );
    }

    // Four equal logits at temperature 1 weigh 1, 1, 1, 1 in the ranked order 0, 1, 2, 3 (a tie
    // goes to the lower index). With top_p 1 all four stay (mass 4) and the unit u picks the token
    // whose cumulative weight first exceeds 4u: 0.1, 0.3, 0.6, and 0.9 give 0.4, 1.2, 2.4, and
    // 3.6, so the tokens 0, 1, 2, and 3 with keep 4. With top_p 0.5 the threshold is 2, so two
    // tokens stay (mass 2) and the units 0.4 and 0.9 give 0.8 and 1.8: tokens 0 and 1, keep 2.
    //
    // Mutation check: in `draw_kept`, replace `draw() * mass` by `draw() * total`. At top_p 0.5
    // the unit 0.4 then gives 1.6 instead of 0.8 and returns token 1 instead of 0, so the test
    // fails.
    #[test]
    fn a_given_unit_picks_the_token_whose_cumulative_weight_passes_its_target() {
        let sampler = Sampler::new(params(1.0, 1.0, 0), 0);
        for (unit, token) in [(0.1, 0), (0.3, 1), (0.6, 2), (0.9, 3)] {
            assert_eq!(
                sampler.pick_with_unit(&[0.0; 4], unit),
                Ok((token, 4)),
                "unit {unit}"
            );
        }
        let half = Sampler::new(params(1.0, 0.5, 0), 0);
        assert_eq!(half.pick_with_unit(&[0.0; 4], 0.4), Ok((0, 2)));
        assert_eq!(half.pick_with_unit(&[0.0; 4], 0.9), Ok((1, 2)));
    }

    // The constructor values come back through the accessors. The three settings differ from
    // each other, so a swapped accessor shows. A greedy sampler has seed 0.
    //
    // Mutation check: in `SamplingParams::top_p`, return `f64::from(self.top_k)`. The test then
    // reads 20.0 where it expects 0.95 and fails.
    #[test]
    fn the_seed_and_the_settings_come_back_through_their_accessors() {
        let settings = params(0.7, 0.95, 20);
        assert_eq!(settings.temperature(), 0.7);
        assert_eq!(settings.top_p(), 0.95);
        assert_eq!(settings.top_k(), 20);
        let sampler = Sampler::new(settings, 1_234_567);
        assert_eq!(sampler.seed(), 1_234_567);
        assert_eq!(sampler.params(), settings);
        assert_eq!(Sampler::greedy().seed(), 0);
    }

    // Four equal logits and top_p 1: the unit 0.0 gives target 0 and the unit -0.0 gives target
    // -0.0; both are below the first cumulative weight 1, so token 0. The unit 0.999999 gives
    // 3.999996, below the last cumulative weight 4: token 3. The units 1.0, 1.5, -0.5, and NaN
    // fail with `BadRequest` that prints the value. A valid call gives the unit 0.6 the target 2.4:
    // token 2.
    //
    // Mutation check: in `pick_with_unit`, replace the range `0.0..1.0` by `0.0..=1.0`. The
    // unit 1.0 is then accepted and the test fails on it.
    #[test]
    fn pick_with_unit_accepts_units_below_one_and_refuses_the_others() {
        let sampler = Sampler::new(params(1.0, 1.0, 0), 0);
        let row = [0.0_f32; 4];
        for (unit, token) in [(0.0, 0), (-0.0, 0), (0.999_999, 3)] {
            assert_eq!(
                sampler.pick_with_unit(&row, unit),
                Ok((token, 4)),
                "unit {unit}"
            );
        }
        assert_eq!(sampler.pick_with_unit(&row, 0.6), Ok((2, 4)));
        for bad in [1.0, 1.5, -0.5, f64::NAN] {
            match sampler.pick_with_unit(&row, bad) {
                Err(EngError::BadRequest(text)) => {
                    assert!(text.contains(&format!("{bad}")), "message was {text}");
                }
                other => panic!("expected BadRequest for {bad}, got {other:?}"),
            }
        }
    }

    // A greedy sampler picks the largest logit, the lowest index on a tie, and keeps one entry,
    // whatever valid unit it receives. Here the logits 1, 3, 3, 2 give token 1.
    //
    // Mutation check: in `pick_row`, replace `argmax(logits)` by `0`. The token is then 0 and the
    // test fails.
    #[test]
    fn a_greedy_sampler_ignores_the_unit() {
        let sampler = Sampler::greedy();
        for unit in [0.0, 0.25, 0.5, 0.999_999] {
            assert_eq!(
                sampler.pick_with_unit(&[1.0, 3.0, 3.0, 2.0], unit),
                Ok((1, 1)),
                "unit {unit}"
            );
        }
    }

    // `pick_with_unit` reads no draw from the generator. Two samplers start from seed 5. One
    // takes five calls of `pick_with_unit` (the same unit twice, then four others): the repeated
    // call returns the same answer, its generator equals the untouched one, and its next
    // `sample_with_keep` equals the untouched one's.
    //
    // Mutation check: give `pick_with_unit` a `&mut self` receiver and add
    // `let _ = self.rng.next_unit();` before its `pick_row` call (the bindings of the tests then
    // need `mut`). The generator then moves five draws and the test fails on the generator
    // equality.
    #[test]
    fn pick_with_unit_does_not_consume_a_generator_draw() {
        let settings = params(1.0, 1.0, 0);
        let row = [0.0_f32; 16];
        let mut touched = Sampler::new(settings, 5);
        let mut untouched = Sampler::new(settings, 5);
        let first = touched.pick_with_unit(&row, 0.3);
        assert_eq!(
            touched.pick_with_unit(&row, 0.3),
            first,
            "same unit, same answer"
        );
        for unit in [0.0, 0.1, 0.5, 0.9] {
            touched.pick_with_unit(&row, unit).expect("pick_with_unit");
        }
        assert_eq!(
            touched.rng, untouched.rng,
            "pick_with_unit moved the generator"
        );
        assert_eq!(
            touched.sample_with_keep(&row),
            untouched.sample_with_keep(&row)
        );
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

    // The prefix round against the reference: sizes below, just above, and far above
    // PREFIX_ROWS (1,000 takes the full path only; 1,025 and up try the prefix first), every
    // temperature around and beyond the default, all three top_p modes, several seeds, and
    // consecutive draws from one sampler so the draw sequence stays in step. The kept count is
    // compared too, so a prefix that answers with a different nucleus than the reference fails
    // even when the token happens to match.
    #[test]
    fn the_prefix_path_matches_the_reference_over_sizes_parameters_and_seeds() {
        let mut rng = SplitMix64::new(0x5EED_C0FF);
        for &size in &[1_000_usize, 1_025, 4_096, 248_077] {
            let logits = random_row(&mut rng, size);
            // The 248,077-entry vocabulary costs the reference a full sort per draw, so it takes
            // fewer seeds and draws than the small rows.
            let (seeds, count) = if size > 100_000 {
                (4_u64, 4_usize)
            } else {
                (8, 8)
            };
            for &temperature in &[0.2, 0.7, 1.0, 2.0] {
                for &top_p in &[0.5, 0.95, 1.0] {
                    for seed in 0..seeds {
                        assert_same_draws(
                            "prefix grid",
                            &logits,
                            temperature,
                            top_p,
                            0,
                            seed,
                            count,
                        );
                    }
                }
            }
        }
    }

    // The named adversarial rows at the full vocabulary size through the prefix path: ties in
    // pairs, one spike first and last, two equal spikes, the tiny-tail row, and the
    // spread-magnitude row. Seed 0's first unit is 0.8833 (pinned above), so with the two equal
    // spikes at a closing temperature the draw lands past the first spike's weight and a
    // reversed tie-break in the prefix would return the other spike; seed 42 spreads the draws
    // over the other rows.
    #[test]
    fn the_prefix_path_matches_the_reference_on_the_named_adversarial_rows() {
        let rows: [(&str, Vec<f32>); 6] = [
            ("tie pairs", tie_pairs(248_077)),
            ("one spike first", one_spike_first(248_077)),
            ("one spike last", one_spike_last(248_077)),
            ("two equal spikes", two_equal_spikes(248_077)),
            ("tiny tail", tiny_tail_row(248_077)),
            ("spread magnitude", spread_magnitude_row(248_077)),
        ];
        for (name, logits) in &rows {
            for &temperature in &[0.2, 0.7, 1.0, 2.0] {
                for &top_p in &[0.5, 0.95, 1.0] {
                    for seed in [0_u64, 42] {
                        assert_same_draws(name, logits, temperature, top_p, 0, seed, 4);
                    }
                }
            }
        }
    }

    // A flat 248,077-entry row at temperature 1.0: every weight is exactly 1.0, so the running
    // total grows by 1 on each of the 1,024 prefix entries and never stalls — `closed` stays
    // false and the full ranked row must answer, because the mass keeps growing past the prefix.
    // The path counter proves the fallback answered, and the draws still match the reference
    // draw for draw.
    #[test]
    fn a_flat_row_at_the_vocabulary_size_takes_the_fallback_and_matches_the_reference() {
        let logits = vec![0.0_f32; 248_077];
        let settings = params(1.0, 1.0, 0);
        let mut sampler = Sampler::new(settings, 0);
        let got = draws_with_keep(&mut sampler, &logits, 3);
        assert_eq!(
            sample_path::take(),
            (0, 3),
            "a flat row must fall back to the full ranked row"
        );
        let mut reference = Sampler::new(settings, 0);
        let want = reference_draws_with_keep(&mut reference, &logits, 3);
        assert_eq!(got, want, "flat row diverged from the reference");
    }

    // A peaked row: logit 0.0 at index 0 and -40.0 everywhere else. At temperature 1.0 every
    // tail weight is exp(-40) ≈ 4.2e-18, far below half an ulp of the running total 1.0, so the
    // second weight stalls the total and the third breaks the loop inside the prefix: `closed`
    // is true and the prefix must answer. The counter proves the prefix path ran (a fallback
    // here would leave the fast path dead); the draws match the reference as everywhere else.
    #[test]
    fn a_peaked_row_at_the_vocabulary_size_takes_the_prefix_and_matches_the_reference() {
        let mut logits = vec![-40.0_f32; 248_077];
        logits[0] = 0.0;
        let settings = params(1.0, 1.0, 0);
        let mut sampler = Sampler::new(settings, 0);
        let got = draws_with_keep(&mut sampler, &logits, 3);
        assert_eq!(
            sample_path::take(),
            (3, 0),
            "a peaked row must answer from the ranked prefix"
        );
        let mut reference = Sampler::new(settings, 0);
        let want = reference_draws_with_keep(&mut reference, &logits, 3);
        assert_eq!(got, want, "peaked row diverged from the reference");
    }

    // A row whose prefix weights stall exactly at a chosen entry of the ranked prefix, derived
    // by hand. Index 0 holds the top logit 0.0, whose weight is exactly 1. `dense` entries hold
    // ln(1e-10) ≈ -23.025851, whose weight is about 1e-10 — far above half an ulp (1.1e-16) of a
    // running total near 1, so no addition among them can leave the total unchanged. `boundary`
    // entries hold ln(1e-20) ≈ -46.051702, whose weight is about 1e-20 — far below that half
    // ulp, so the first of them stalls the total and a second, no larger, breaks the loop. Every
    // other entry holds ln(1e-30), which ranks below all of them. With PREFIX_ROWS 1,024 the
    // ranked prefix is the top entry, then the `dense` entries, then the `boundary` entries:
    // `dense` 1,021 with `boundary` 2 puts the stall at entry 1,023 and the break at entry 1,024
    // (closed, 1,023 weights), and `dense` 1,022 with `boundary` 1 puts the only stall at entry
    // 1,024, where the prefix runs out before any break (not closed, so the fallback answers).
    fn boundary_stall_row(dense: usize, boundary: usize, size: usize) -> Vec<f32> {
        let tail = (-30.0 * std::f64::consts::LN_10) as f32;
        let dense_logit = (-10.0 * std::f64::consts::LN_10) as f32;
        let boundary_logit = (-20.0 * std::f64::consts::LN_10) as f32;
        let mut row = vec![tail; size];
        for slot in &mut row[1..1 + dense] {
            *slot = dense_logit;
        }
        for slot in &mut row[1 + dense..1 + dense + boundary] {
            *slot = boundary_logit;
        }
        row[0] = 0.0;
        row
    }

    // The stall at entry 1,023 with the break at entry 1,024: the prefix closes with exactly
    // 1,023 weights, its total is bit-for-bit the total the full ranked row produces (the full
    // row breaks at the same weight), and the prefix answers the draw. This is the last position
    // at which the prefix can still prove that the tail vanished.
    #[test]
    fn a_stall_at_entry_1023_breaks_at_entry_1024_and_the_prefix_answers() {
        let logits = boundary_stall_row(1_021, 2, 248_077);
        let prefix = ranked_prefix(&logits, PREFIX_ROWS);
        assert_eq!(prefix.len(), PREFIX_ROWS, "the prefix must be full size");
        let top = top_logit(&logits);
        let (prefix_weights, prefix_total, closed) = ranked_weights_prefix(&prefix, top, 1.0);
        assert!(closed, "a stall at 1,023 plus a break at 1,024 must close");
        assert_eq!(
            prefix_weights.len(),
            1_023,
            "weights must stop at the stall"
        );
        let (full_weights, full_total) = ranked_weights(&ranked_row(&logits), top, 1.0);
        assert_eq!(
            full_weights.len(),
            1_023,
            "the full row must break at the same weight"
        );
        assert_eq!(
            prefix_total.to_bits(),
            full_total.to_bits(),
            "prefix total {prefix_total} differs from full total {full_total}"
        );
        let settings = params(1.0, 1.0, 0);
        let mut sampler = Sampler::new(settings, 0);
        let got = draws_with_keep(&mut sampler, &logits, 3);
        assert_eq!(sample_path::take(), (3, 0), "the prefix must answer");
        let mut reference = Sampler::new(settings, 0);
        let want = reference_draws_with_keep(&mut reference, &logits, 3);
        assert_eq!(got, want, "boundary row diverged from the reference");
    }

    // The stall at entry 1,024 itself: the prefix runs out of entries before any weight can
    // break the loop, so `closed` is false even though the very next entry of the full row would
    // break it, and the fallback must answer. The prefix cannot know what the 1,025th weight is,
    // so falling back here is the contract, not a missed optimisation; the totals still agree
    // bit for bit, which documents that only the proof, not the arithmetic, was missing.
    #[test]
    fn a_stall_at_entry_1024_exhausts_the_prefix_and_the_fallback_answers() {
        let logits = boundary_stall_row(1_022, 1, 248_077);
        let prefix = ranked_prefix(&logits, PREFIX_ROWS);
        let top = top_logit(&logits);
        let (prefix_weights, prefix_total, closed) = ranked_weights_prefix(&prefix, top, 1.0);
        assert!(!closed, "a stall at the last entry cannot break the loop");
        assert_eq!(
            prefix_weights.len(),
            PREFIX_ROWS,
            "every entry was consumed"
        );
        let (full_weights, full_total) = ranked_weights(&ranked_row(&logits), top, 1.0);
        assert_eq!(
            full_weights.len(),
            PREFIX_ROWS,
            "the full row breaks one later"
        );
        assert_eq!(
            prefix_total.to_bits(),
            full_total.to_bits(),
            "prefix total {prefix_total} differs from full total {full_total}"
        );
        let settings = params(1.0, 1.0, 0);
        let mut sampler = Sampler::new(settings, 0);
        let got = draws_with_keep(&mut sampler, &logits, 3);
        assert_eq!(sample_path::take(), (0, 3), "the fallback must answer");
        let mut reference = Sampler::new(settings, 0);
        let want = reference_draws_with_keep(&mut reference, &logits, 3);
        assert_eq!(got, want, "boundary row diverged from the reference");
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
    // It prints microseconds per sample for a 248,077-entry row, the reference against the sampler,
    // at four points: (T0.7, P0.95, K0), (T1.0, P0.95, K20), (T0.7, P1.0, K0), and
    // (T0.7, P0.95, K20). The server defaults verified in server/validate.rs:46-50 and
    // family/sampler.rs are temperature 0.7 and top_p 1.0. The top_k 20 rows must stay under
    // 1.5 ms per sample on this host.
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

    // Timing harness for the ranked-prefix round. Run:
    //   cargo test --release --offline --lib -- --ignored --nocapture sample_timing_at_the_vocabulary_size
    // Three warm-up draws, then thirty timed draws, on a seeded row of 248,077 logits uniform in
    // [-10, 10), at the five measured points: greedy, the top_k 20 nucleus, and the open-
    // vocabulary top_k 0 points at temperature 0.2 (top_p 0.95 and 1.0) and 1.0. It prints the
    // milliseconds per draw and asserts nothing: the numbers are read by a human, before and
    // after the change under measurement.
    #[test]
    #[ignore = "timing harness; run with --ignored --nocapture"]
    fn sample_timing_at_the_vocabulary_size() {
        let mut rng = SplitMix64::new(0x0710_2026);
        let logits: Vec<f32> = (0..248_077)
            .map(|_| ((rng.next_unit() * 20.0) - 10.0) as f32)
            .collect();
        for &(temperature, top_p, top_k) in &[
            (0.0, 1.0, 0_u32),
            (0.2, 0.95, 20),
            (0.2, 0.95, 0),
            (0.2, 1.0, 0),
            (1.0, 0.95, 0),
        ] {
            let settings = params(temperature, top_p, top_k);
            let mut sampler = Sampler::new(settings, 1);
            for _ in 0..3 {
                sampler.sample(&logits).expect("warm-up sample");
            }
            let start = std::time::Instant::now();
            for _ in 0..30 {
                sampler.sample(&logits).expect("timed sample");
            }
            let per_draw_ms = start.elapsed().as_secs_f64() / 30.0 * 1e3;
            println!(
                "vocab 248077, temperature {temperature}, top_p {top_p}, top_k {top_k}: \
                 {per_draw_ms:.2} ms per draw"
            );
        }
    }

    // ---- Statistical audit of the host sampler: 200,000 draws per configuration ----

    /// The draws of one audit configuration.
    const AUDIT_DRAWS: usize = 200_000;

    /// The draws of an audit whose outcome is deterministic (one kept token): no statistic is
    /// computed, so a thousand draws are enough to see a leak.
    const DETERMINISTIC_DRAWS: usize = 1_000;

    /// A peaked row of `len` tokens: token 0 holds probability `head` and the other tokens share
    /// `tail` with a 0.9 tilt.
    fn peaked_row(len: usize, head: f64, tail: f64) -> Vec<f32> {
        let len = i32::try_from(len).expect("a short row");
        let tilt: Vec<f64> = (0..len - 1).map(|i| 0.9_f64.powi(i)).collect();
        let sum: f64 = tilt.iter().sum();
        let mut logits = vec![head.ln()];
        logits.extend(tilt.iter().map(|share| (tail * share / sum).ln()));
        logits.iter().map(|&logit| logit as f32).collect()
    }

    /// A peaked row: token 0 holds probability 0.99 and the other 49 tokens share 0.01 with a
    /// 0.9 tilt, so every weight stays far above one ulp of the total.
    fn peaked_logits() -> Vec<f32> {
        peaked_row(50, 0.99, 0.01)
    }

    /// A flat top: 20 equal logits over 30 tail logits of -3.
    fn flat_logits() -> Vec<f32> {
        let mut logits = vec![0.0_f32; 20];
        logits.resize(50, -3.0);
        logits
    }

    /// A long geometric tail: the logit of token `i` is `-0.1 * i`, for 100 tokens.
    fn tail_logits() -> Vec<f32> {
        (0..100).map(|i| -0.1 * i as f32).collect()
    }

    /// The exact truncated and renormalized distribution that the audit tests against: rank by
    /// logit descending with ties to the lower index, keep the `top_k` largest, weight
    /// `exp((logit - max) / temperature)` in f64, keep the smallest descending prefix whose mass
    /// reaches `top_p` of the truncated total, and normalize over that prefix. That is the
    /// Hugging Face order and semantics (`top_k` first, then `top_p` on the renormalized top-k
    /// softmax), evaluated in f64 on the same f32 logits. Returns the probabilities and the kept
    /// count.
    fn reference_distribution(logits: &[f32], settings: SamplingParams) -> (Vec<f64>, usize) {
        let mut ranked: Vec<(usize, f64)> = logits
            .iter()
            .enumerate()
            .map(|(index, &logit)| (index, f64::from(logit)))
            .collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        if settings.top_k() > 0 {
            ranked.truncate(settings.top_k() as usize);
        }
        let top = ranked[0].1;
        let weights: Vec<f64> = ranked
            .iter()
            .map(|entry| ((entry.1 - top) / settings.temperature()).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let mut keep = 0_usize;
        let mut mass = 0.0_f64;
        for weight in &weights {
            mass += weight;
            keep += 1;
            if mass >= settings.top_p() * total {
                break;
            }
        }
        let kept_mass: f64 = weights.iter().take(keep).sum();
        let mut probs = vec![0.0_f64; logits.len()];
        for (entry, weight) in ranked.iter().zip(&weights).take(keep) {
            probs[entry.0] = weight / kept_mass;
        }
        (probs, keep)
    }

    /// The Pearson chi-square statistic of observed id counts against the expected counts, with
    /// categories of expected count below 5 pooled into one tail bucket (the Cochran rule).
    /// Returns the statistic and its degrees of freedom (buckets minus one).
    fn chi_square(observed: &[u64], expected: &[f64], draws: usize) -> (f64, usize) {
        let mut statistic = 0.0_f64;
        let mut buckets = 0_usize;
        let mut tail_observed = 0_u64;
        let mut tail_expected = 0.0_f64;
        for (&observed, &probability) in observed.iter().zip(expected) {
            if probability <= 0.0 {
                continue;
            }
            let expected_count = draws as f64 * probability;
            if expected_count >= 5.0 {
                let difference = observed as f64 - expected_count;
                statistic += difference * difference / expected_count;
                buckets += 1;
            } else {
                tail_observed += observed;
                tail_expected += expected_count;
            }
        }
        if tail_expected > 0.0 {
            let difference = tail_observed as f64 - tail_expected;
            statistic += difference * difference / tail_expected;
            buckets += 1;
        }
        (statistic, buckets.saturating_sub(1))
    }

    /// What an audit expects of its chi-square test: the degrees of freedom that the pooled
    /// categories give, the upper 0.1% critical value at that many degrees, and the number of
    /// draws. The audit checks the statistic's degrees against the stated ones, so a change to a
    /// row that moves the category count cannot leave a critical value for the wrong
    /// distribution in place.
    struct Expect {
        freedom: usize,
        critical: f64,
        draws: usize,
    }

    impl Expect {
        /// The expectation of an audit with the standard `AUDIT_DRAWS`.
        fn new(freedom: usize, critical: f64) -> Self {
            Self {
                freedom,
                critical,
                draws: AUDIT_DRAWS,
            }
        }

        /// The same expectation for a different number of draws.
        fn with_draws(self, draws: usize) -> Self {
            Self { draws, ..self }
        }
    }

    /// Runs one audit configuration: the exact reference distribution, `expect.draws` draws from
    /// the real sampler at a fixed seed, no draw outside the truncated set, the degrees of
    /// freedom that the caller states, and a chi-square statistic below the stated critical
    /// value. The level is 0.1%: the critical values are the exact upper 0.1% points of the
    /// chi-square distribution (regularized incomplete gamma function).
    fn audit(settings: SamplingParams, seed: u64, logits: &[f32], expect: Expect) {
        let (expected, keep) = reference_distribution(logits, settings);
        assert!(
            (expected.iter().sum::<f64>() - 1.0).abs() <= 1e-9,
            "the reference probabilities must sum to 1"
        );
        let picked = draws_with_keep(&mut Sampler::new(settings, seed), logits, expect.draws);
        let mut observed = vec![0_u64; expected.len()];
        for &(token, kept) in &picked {
            assert_eq!(
                kept, keep,
                "a draw kept {kept} candidates, the reference keeps {keep}"
            );
            let probability = expected[token as usize];
            assert!(
                probability > 0.0,
                "token {token} is outside the truncated set of {keep} candidates"
            );
            observed[token as usize] += 1;
        }
        let (statistic, freedom) = chi_square(&observed, &expected, expect.draws);
        assert_eq!(
            freedom, expect.freedom,
            "the pooled categories give {freedom} degrees of freedom, the audit states {}",
            expect.freedom
        );
        assert!(
            statistic <= expect.critical,
            "chi-square {statistic} exceeds the critical value {} at {freedom} degrees of freedom",
            expect.critical
        );
    }

    // A flat top of 20 equal logits. `top_k` 20 keeps exactly the equal block, and `top_p` 0.95 of
    // the truncated total 20 needs 19 entries (0.95 * 20 is exactly 19.0 in f64), so the sampled
    // distribution is uniform over ids 0 to 18: 19 categories, 18 degrees of freedom, and the
    // upper 0.1% critical value 42.312.
    #[test]
    fn audit_flat_top_k_20_top_p_0_95_is_uniform_over_19_tokens() {
        let logits = flat_logits();
        let (expected, keep) = reference_distribution(&logits, params(1.0, 0.95, 20));
        assert_eq!(keep, 19, "the truncation keeps 19 of the 20 equal tokens");
        assert!(
            (expected[18] - 1.0 / 19.0).abs() <= 1e-12,
            "kept: {expected:?}"
        );
        assert_eq!(expected[19], 0.0, "the top-k tail is unsampleable");
        audit(
            params(1.0, 0.95, 20),
            0xA11CE,
            &logits,
            Expect::new(18, 42.312),
        );
    }

    // A long geometric tail. `top_k` 20 keeps 20 of 100 tokens, and `top_p` 0.95 of the truncated
    // total keeps 18 of them: 18 categories, 17 degrees of freedom, upper 0.1% critical value
    // 40.790.
    #[test]
    fn audit_long_tail_top_k_20_top_p_0_95_matches_the_renormalized_distribution() {
        let logits = tail_logits();
        let (expected, keep) = reference_distribution(&logits, params(1.0, 0.95, 20));
        assert_eq!(keep, 18, "top_k 20 and top_p 0.95 keep 18 tokens");
        assert_eq!(
            expected.iter().filter(|&&p| p > 0.0).count(),
            18,
            "exactly 18 tokens are sampleable"
        );
        audit(
            params(1.0, 0.95, 20),
            0xB0B,
            &logits,
            Expect::new(17, 40.790),
        );
    }

    // The same tail without `top_k`: `top_p` 0.9 of the full total keeps the smallest prefix that
    // reaches the mass, 24 tokens: 23 degrees of freedom, upper 0.1% critical value 49.728.
    #[test]
    fn audit_long_tail_top_p_0_9_without_top_k_matches_the_truncated_distribution() {
        let logits = tail_logits();
        let (_, keep) = reference_distribution(&logits, params(1.0, 0.9, 0));
        assert_eq!(keep, 24, "top_p 0.9 alone keeps 24 tokens");
        audit(
            params(1.0, 0.9, 0),
            0xC0FFEE,
            &logits,
            Expect::new(23, 49.728),
        );
    }

    // A peaked row at temperature 0.7 without truncation: `top_p` 1.0 keeps every token. The 44
    // smallest categories have an expected count below 5 and pool into one tail bucket, so at
    // most 7 buckets remain: at most 6 degrees of freedom, and the upper 0.1% critical value at
    // 6 degrees of freedom is 22.458 (a lower bucket count only lowers the true critical value).
    #[test]
    fn audit_peaked_temperature_0_7_without_truncation_matches_the_softmax() {
        let logits = peaked_logits();
        let (_, keep) = reference_distribution(&logits, params(0.7, 1.0, 0));
        assert_eq!(keep, 50, "no truncation keeps every token");
        audit(params(0.7, 1.0, 0), 0xD1CE, &logits, Expect::new(6, 22.458));
    }

    // The geometric tail at temperature 2 with `top_k` 20 and no `top_p`: the kept weights are
    // exp(-0.05 * i) over ids 0 to 19, so the probability falls from id 0 to id 19 by a factor of
    // exp(-0.95); at temperature 1 the same row would fall by exp(-1.9). Unequal logits make the
    // temperature scaling visible, which an equal block would not: 19 degrees of freedom, upper
    // 0.1% critical value 43.820.
    #[test]
    fn audit_long_tail_temperature_2_top_k_20_matches_the_tempered_softmax() {
        let logits = tail_logits();
        let (expected, keep) = reference_distribution(&logits, params(2.0, 1.0, 20));
        assert_eq!(keep, 20, "top_k 20 without top_p keeps 20 tokens");
        assert_eq!(expected[20], 0.0, "the top-k tail is unsampleable");
        let fall = expected[19] / expected[0];
        assert!(
            (fall - (-0.95_f64).exp()).abs() <= 1e-6,
            "the tempered probabilities fall by exp(-0.95), not {fall}"
        );
        audit(
            params(2.0, 1.0, 20),
            0xE77,
            &logits,
            Expect::new(19, 43.820),
        );
    }

    // A peaked row longer than `PREFIX_ROWS` with `top_k` off: this is the open-vocabulary prefix
    // path, which ranks only the first 1,024 entries and answers when their weights vanish inside
    // them. Token 0 holds 0.6 and the other 1,024 tokens share 0.4 with a 0.9 tilt. `top_p` 0.95
    // keeps 21 tokens (the cumulative mass is 0.9460 at 20 and 0.9514 at 21): 21 categories, 20
    // degrees of freedom, upper 0.1% critical value 45.315. Each draw ranks the prefix, so the
    // audit takes 10,000 draws; the smallest expected count is 56. The path counters must show
    // that the prefix answered every draw and the full row none.
    #[test]
    fn audit_long_peaked_row_without_top_k_is_answered_by_the_prefix_path() {
        let logits = peaked_row(PREFIX_ROWS + 1, 0.6, 0.4);
        assert!(
            logits.len() > PREFIX_ROWS,
            "the row must reach the prefix path"
        );
        let (_, keep) = reference_distribution(&logits, params(1.0, 0.95, 0));
        assert_eq!(
            keep, 21,
            "top_p 0.95 of the open-vocabulary row keeps 21 tokens"
        );
        let draws = 10_000;
        audit(
            params(1.0, 0.95, 0),
            0x5EED,
            &logits,
            Expect::new(20, 45.315).with_draws(draws),
        );
        assert_eq!(
            sample_path::take(),
            (draws as u64, 0),
            "the ranked prefix must answer every draw"
        );
    }

    // A peaked row at temperature 1, `top_p` 0.95 and `top_k` 20: `top_k` 20 keeps the top 20 and
    // `top_p` 0.95 of the truncated mass stops at the first token, because 0.99 alone reaches
    // 0.95, so the kept set is the single top token and every draw must be token 0. The outcome
    // is deterministic, so it needs no chi-square sample: a sampler that leaked even 1% of the
    // mass would leave the token in 1,000 draws with probability 0.99^1000, below 5e-5.
    #[test]
    fn audit_peaked_top_k_20_top_p_0_95_never_leaves_the_single_kept_token() {
        let logits = peaked_logits();
        let (expected, keep) = reference_distribution(&logits, params(1.0, 0.95, 20));
        assert_eq!(keep, 1, "0.99 of the mass reaches top_p alone");
        assert!((expected[0] - 1.0).abs() <= 1e-12, "kept: {expected:?}");
        let picked = draws_with_keep(
            &mut Sampler::new(params(1.0, 0.95, 20), 0xF00D),
            &logits,
            DETERMINISTIC_DRAWS,
        );
        assert!(
            picked.iter().all(|&draw| draw == (0, 1)),
            "a draw left the single kept token or kept more than one candidate"
        );
    }
}
