use crate::EngError;
use std::sync::Arc;

/// Byte (start, end) offsets per token, as reported by an encode.
pub type ByteOffsets = Vec<(usize, usize)>;

pub struct Tokenizer {
    inner: Arc<tokenizers::Tokenizer>,
    vocab_size: u32,
    /// Boundary-derivation eligibility, computed once at load: no
    /// truncation, no padding, a null-or-zero BPE `dropout` (deterministic
    /// encoding), a load-time BEHAVIORAL probe of the deserialized
    /// tokenizer (the loaded post-processor leaves a synthetic encoding's
    /// ids untouched at add_special_tokens=false, and independently
    /// encoding chunks at the seams the warm path actually cuts — right
    /// chunk starting with a plain added token — concatenates to the
    /// whole-string encoding), and every added token matches without
    /// contextual or text-altering flags (lstrip/rstrip/single_word/
    /// normalized all false). Under these conditions the added-token split
    /// is byte-faithful — no matching rule can change text outside a
    /// token's exact span, so independently encoding the chunks reproduces
    /// the whole-string encoding. Any violation makes the bundle fall back
    /// to the three-pass boundary everywhere.
    boundary_eligible: bool,
}

/// A synthetic encoding with distinct ids, used to observe the loaded
/// post-processor's id rewriting DIRECTLY (Opus delta7): any insert,
/// drop, duplication, or reorder shows up as an ids change at
/// add_special_tokens=false — regardless of vocabulary coverage, which
/// finite probe strings can never guarantee (Sol9's vacuous-vocab
/// duplication bypass).
// Soundness of this observation depends on the CLOSED five-variant
// post-processor set of tokenizers 0.21.4 (Bert/Roberta/ByteLevel/
// Template/Sequence): none of them reads ids or token content at
// add_special_tokens=false. A crate upgrade must re-audit this.
fn synthetic_encoding() -> tokenizers::Encoding {
    const N: usize = 6;
    tokenizers::Encoding::new(
        (0..N as u32).collect(),
        vec![0; N],
        (0..N).map(|i| format!("x{i}")).collect(),
        vec![None; N],
        (0..N).map(|i| (i, i + 1)).collect(),
        vec![0; N],
        vec![1; N],
        vec![],
        Default::default(),
    )
}

/// The load-time behavioral probe backing `boundary_eligible`:
///
/// 1. The loaded post-processor must leave a synthetic encoding's ids
///    untouched at add_special_tokens=false (byte-faithful concat
///    requires an id-preserving post-processor; ByteLevel/Bert only trim
///    offsets or add specials when specials are requested, so they pass).
/// 2. At the seams the warm path ACTUALLY cuts — the right-hand chunk
///    starting with a plain added token (template turn boundary) —
///    independently encoding the parts at add_special_tokens=false must
///    concatenate to the whole-string encoding. Plain-text seams are
///    deliberately NOT probed: SentencePiece-style normalizers prepend a
///    marker to any standalone encode, so they would be wrongly excluded
///    (Opus delta7 SF) even though the warm seam itself is exact.
/// 3. Encoding must be deterministic run-to-run (stochastic models).
///
/// Returns false on any mismatch — the tokenizer then always takes the
/// full three-pass path (still correct, never warm).
fn warm_concatenation_probe(inner: &tokenizers::Tokenizer) -> bool {
    use tokenizers::PostProcessor as _;
    let post_processor_ok = match inner.get_post_processor() {
        None => true,
        Some(pp) => match pp.process(synthetic_encoding(), None, false) {
            Ok(enc) => enc.get_ids() == [0, 1, 2, 3, 4, 5],
            Err(_) => false,
        },
    };
    // Every plain special added token, sorted by id for determinism
    // (the decoder map iterates in hash order).
    let mut plain_specials: Vec<(u32, String)> = inner
        .get_added_vocabulary()
        .get_added_tokens_decoder()
        .iter()
        .filter(|(_, t)| {
            t.special
                && !t.lstrip
                && !t.rstrip
                && !t.single_word
                && !t.normalized
                && !t.content.is_empty()
        })
        .map(|(id, t)| (*id, t.content.clone()))
        .collect();
    plain_specials.sort_by_key(|(id, _)| *id);
    let ids = |text: &str| -> Option<Vec<u32>> {
        inner.encode(text, false).ok().map(|e| e.get_ids().to_vec())
    };
    let seams_ok = plain_specials.iter().all(|(_, content)| {
        let whole = format!("alpha{content}beta");
        let head = "alpha".to_string();
        let tail = format!("{content}beta");
        match (ids(&whole), ids(&head), ids(&tail)) {
            (Some(w), Some(l), Some(r)) => [l, r].concat() == w,
            _ => false,
        }
    });
    // Run-to-run determinism: a stochastic model (dropout, seeded
    // samplers) re-rolls and eventually disagrees with itself. Low-rate
    // stochasticity can slip this, which is why the dropout field
    // itself is also checked at load.
    let deterministic = ids("alpha beta")
        .zip(ids("alpha beta"))
        .is_some_and(|(a, b)| a == b);
    post_processor_ok && seams_ok && deterministic
}

impl Tokenizer {
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, EngError> {
        let inner = tokenizers::Tokenizer::from_bytes(bytes).map_err(|error| {
            EngError::Checkpoint(format!("tokenizer.json parse failed: {error}"))
        })?;
        // `get_vocab_size(true)` builds the whole vocabulary map on each call.
        // The size is fixed after parsing, so the wrapper queries it once.
        let vocab_size = u32::try_from(inner.get_vocab_size(true)).map_err(|_| {
            EngError::Checkpoint("tokenizer vocabulary size does not fit in u32".to_string())
        })?;
        // Load-time boundary eligibility (see the struct doc): truncation
        // and padding rewrite the id stream, and any added token with a
        // contextual/text-altering matching flag can make an independent
        // chunk split disagree with the whole-string encoding.
        // The one extra parse: a deterministic check for BPE dropout,
        // which the behavioral probe below cannot catch reliably at low
        // dropout rates (a 0.001 dropout agrees with itself ~92% of the
        // time, so only the JSON field is certain).
        let raw = serde_json::from_slice::<serde_json::Value>(bytes).ok();
        // A BPE `dropout` makes encoding stochastic: identical bytes can
        // tokenize differently on every call, so no derived boundary and
        // no warm reuse may engage (Sol MF: stochastic eligibility).
        let deterministic = raw.as_ref().is_some_and(|v| {
            v.get("model")
                .and_then(|m| m.get("dropout"))
                .and_then(|d| d.as_f64())
                .is_none_or(|d| d == 0.0)
        });
        // BEHAVIORAL post-processor/stochasticity gate (COO 01a17c64,
        // after Sol8's mistag bypass): the warm path's exactness rests on
        // chunk concatenation — encoding pieces independently at
        // add_special_tokens=false and concatenating must reproduce the
        // whole-string encoding. Structural inspection of the raw JSON
        // is unreliable (PostProcessorWrapper is untagged, so the
        // "type" tag does not name the deserialized variant), so we
        // verify the property itself on the DESERIALIZED tokenizer at
        // load: a duplicating post-processor doubles every part's ids
        // and breaks the concat equality; a prefix insert or stochastic
        // model breaks it or the run-to-run determinism check. Costs a
        // handful of tiny encodes once at load.
        // A malformed template can make tokenizers PANIC inside encode
        // (e.g. Sequence B in a single-input template indexes past the
        // encoding pair). The probe runs under catch_unwind, and a panic
        // here REJECTS THE LOAD: such a tokenizer would panic again on
        // every request encode, ineligible or not, so returning it
        // (Sol9 MEDIUM) would hand out a known-crashing encoder.
        let concatenation_holds =
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                warm_concatenation_probe(&inner)
            })) {
                Ok(v) => v,
                Err(payload) => {
                    let reason = payload
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "unknown panic".to_string());
                    return Err(EngError::Checkpoint(format!(
                        "tokenizer panics on encode during the warm probe ({reason}) — \
                     malformed post-processor?"
                    )));
                }
            };
        let added_flags_ok = inner
            .get_added_vocabulary()
            .get_added_tokens_decoder()
            .values()
            .all(|token| !token.lstrip && !token.rstrip && !token.single_word && !token.normalized);
        // One line at load whenever warm turns are refused, so a
        // silently-ineligible tokenizer is diagnosable from the log.
        if !deterministic
            || !concatenation_holds
            || inner.get_truncation().is_some()
            || inner.get_padding().is_some()
            || !added_flags_ok
        {
            let mut reasons: Vec<&str> = Vec::new();
            if !deterministic {
                reasons.push("stochastic model (dropout)");
            }
            if inner.get_truncation().is_some() {
                reasons.push("truncation");
            }
            if inner.get_padding().is_some() {
                reasons.push("padding");
            }
            if !concatenation_holds {
                reasons.push("behavioral probe (post-processor, seam, or determinism)");
            }
            if !added_flags_ok {
                reasons.push("contextual added-token flags");
            }
            eprintln!(
                "fifi: warm turns and the derived boundary disabled: {}",
                reasons.join(", ")
            );
        }
        let boundary_eligible = deterministic
            && concatenation_holds
            && inner.get_truncation().is_none()
            && inner.get_padding().is_none()
            && added_flags_ok;
        Ok(Self {
            inner: Arc::new(inner),
            vocab_size,
            boundary_eligible,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, EngError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|error| EngError::Checkpoint(format!("prompt encode failed: {error}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, EngError> {
        if ids.iter().any(|&id| id >= self.vocab_size) {
            return Err(EngError::Checkpoint(
                "token id outside tokenizer vocab".to_string(),
            ));
        }
        self.inner
            .decode(ids, true)
            .map_err(|error| EngError::Checkpoint(format!("token decode failed: {error}")))
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Whether an id belongs to the added-token vocabulary (special markers
    /// and other exact-text tokens). Lookup by reference via the vocabulary
    /// (the Tokenizer-level accessor clones the whole map).
    pub fn is_added_token_id(&self, id: u32) -> bool {
        self.inner
            .get_added_vocabulary()
            .get_added_tokens_decoder()
            .contains_key(&id)
    }

    /// Whether any added token's exact text sits at the start of `text`.
    /// Under boundary eligibility (all matching flags false) an added first
    /// token REQUIRES such a prefix, so a negative answer lets callers skip
    /// the suffix encode on the fallback path.
    pub(crate) fn any_added_text_starts(&self, text: &str) -> bool {
        self.inner
            .get_added_vocabulary()
            .get_added_tokens_decoder()
            .values()
            .any(|token| text.starts_with(token.content.as_str()))
    }

    /// True when `text` ends with a proper nonempty prefix of some added
    /// token — the warm-encoder's junction guard: an added token whose
    /// text SPANS a head/delta seam (starting inside the head, ending in
    /// the delta) would match in a full encode but not in the pieces, so
    /// such a head must not be reused.
    pub(crate) fn ends_inside_added_token(&self, text: &str) -> bool {
        self.inner
            .get_added_vocabulary()
            .get_added_tokens_decoder()
            .values()
            .any(|token| {
                // Proper nonempty prefixes only (a full added token at
                // the end is a complete split point, not a spanning
                // risk), cut at char boundaries of the token's own text.
                token
                    .content
                    .char_indices()
                    .skip(1)
                    .map(|(end, _)| &token.content[..end])
                    .any(|prefix| text.ends_with(prefix))
            })
    }

    /// Encode with byte offsets, so callers can verify that a token starts
    /// at an exact byte position in the input.
    pub(crate) fn encode_with_offsets(
        &self,
        text: &str,
    ) -> Result<(Vec<u32>, ByteOffsets), EngError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|error| EngError::Checkpoint(format!("prompt encode failed: {error}")))?;
        Ok((encoding.get_ids().to_vec(), encoding.get_offsets().to_vec()))
    }

    /// The load-time eligibility computed in `from_bytes` (see the struct
    /// doc). Callers use it to gate the derived boundary path.
    pub fn boundary_eligible(&self) -> bool {
        self.boundary_eligible
    }

    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    pub(super) fn shared(&self) -> Arc<tokenizers::Tokenizer> {
        Arc::clone(&self.inner)
    }

    #[cfg(test)]
    pub(super) fn raw(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}
