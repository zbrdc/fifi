use super::{
    ChatTemplate, PromptMessage, Role, Thinking, Tokenizer, ToolSpec, encode::ByteOffsets,
};
use crate::{EngError, checkpoint_json::UniqueValue};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::ErrorKind;
use std::path::Path;

#[derive(Clone, Copy, Hash)]
pub struct TokenizerIdentity {
    pub tokenizer_json: [u8; 32],
    pub tokenizer_config_json: [u8; 32],
    pub chat_template_jinja: [u8; 32],
    pub generation_config_json: [u8; 32],
}

#[derive(Clone, Copy)]
pub struct TokenizerPin {
    pub tokenizer_json: [u8; 32],
    pub tokenizer_config_json: [u8; 32],
    pub chat_template_jinja: [u8; 32],
    pub generation_config_json: [u8; 32],
    pub vocab_size: u32,
}

/// The message-boundary token counts of one encoded prompt.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PromptBoundaries {
    /// The number of prompt tokens that end at the last message boundary, when the prompt ids
    /// extend them: the prompt without its generation tail. `None` turns prefix reuse off.
    pub tip: Option<usize>,
    /// The number of prompt tokens that end at the first system message, when the prompt ids
    /// extend them and a tip follows them. `None` when the first message is not a system
    /// message, the render of the system block fails or does not end in the text of an empty
    /// user turn, there is no tip, or the two boundaries meet.
    pub root: Option<usize>,
}

pub struct TokenizerBundle {
    identity: TokenizerIdentity,
    tokenizer: Tokenizer,
    eos_token_ids: Vec<u32>,
    vocab_size: u32,
    template: ChatTemplate,
    /// Bundle-local render/encode counter for the work comparison between
    /// the derived and three-pass boundary paths (the work tests pin
    /// derived <= reference on the deriving, ineligible-fallback,
    /// pre-screen-rejected, and forced-alignment legs). Instance-scoped so
    /// parallel tests cannot cross-contaminate the counts.
    #[cfg(test)]
    counts: std::sync::Mutex<(usize, usize)>,
}

/// Per-session warm-turn cache: the previous turn's head render and its
/// ids, keyed by the system opener, the tool list, and the thinking mode.
/// A warm turn re-renders (full and head), tokenizes ONLY the appended
/// delta and the generation prompt, and concatenates — exact because
/// every junction sits at a complete added token (the AddedVocabulary
/// splits added tokens out of the raw text atomically, so no BPE merge
/// crosses them). Any doubt — key change, edited history (prefix
/// mismatch), a delta without an added token at the seam, a head ending
/// inside a partial added token, an ineligible tokenizer, or a seeding
/// failure — falls back to the full derived path and is COUNTED, never
/// guessed.
///
/// Bounded memory: `accounted_bytes()` is the live cost — the head
/// render string, the cached head ids, and the completed turn's kept
/// result and digest (`LastTurn`: a second ids copy a resend reuses),
/// roughly 1.5 MB at 128K tokens per session. The caller owns the cache
/// (one per session) and its eviction.
pub struct WarmTurnCache {
    /// Test-only: digest invocations charged to THIS cache — pins that
    /// the call site computes at most one digest and only for an
    /// eligible, key-matched request (per-cache so parallel tests
    /// cannot pollute each other's counts).
    #[cfg(test)]
    pub(crate) digest_calls: usize,
    key: Option<WarmKey>,
    head_render: String,
    head_ids: Vec<u32>,
    fallbacks: usize,
    reuses: usize,
    /// The fixed cause of the LAST fallback — one of "seed", "key",
    /// "ineligible", "head-partial-token", "other-conversation",
    /// "unresolved", "resend", "truncated", "render", "seam" — a &'static str, so
    /// recording it never allocates. Set on every counted
    /// fallback, read by the serving path's fallback log line.
    last_fallback: Option<&'static str>,
    /// The byte length of the boundary render (leading system messages plus
    /// the first non-system message) inside the cached head (0 = unknown).
    /// A byte divergence BEFORE it means the conversation itself differs
    /// (an unrelated request, or a slot another conversation reseeded), not
    /// an append the seam could have served. While unknown, a divergence
    /// reports "unresolved" — never a guessed "seam".
    first_head_len: usize,
    /// The `first_message_end` value `first_head_len` was derived at —
    /// the endpoint's derivation basis. A hit whose raw first-turn
    /// selector MOVED (render-preserving role edits can do this) must
    /// refresh or clear the endpoint, never keep it stale.
    first_head_end: Option<usize>,
    /// True when the cached endpoint covers the WHOLE cached head
    /// (`first_head_len == head_render.len()`): such an endpoint is
    /// re-derivable from the hit's own head render, so every hit whose
    /// boundary covers the whole current head refreshes it — all-system
    /// heads advancing turn by turn and content growth with an unchanged
    /// selector included.
    boundary_whole_head: bool,
    self_check: bool,
    /// The last completed turn's ids and boundaries plus the SHA-256 of
    /// the exact request (messages, tools, thinking) that produced them —
    /// one unit, dropped together whenever the slot reseeds or empties,
    /// so a stale result can never outlive its fingerprint.
    /// A resend whose digest and key match reuses the stored ids
    /// outright (see `encode_prompt_warm`); any edit, reorder, tool,
    /// thinking, or identity change lands in the digest or the key and
    /// misses.
    last_turn: Option<LastTurn>,
}

/// One completed warm turn: the answer a resend reuses and the digest
/// of the request that produced it.
struct LastTurn {
    ids: Vec<u32>,
    bounds: PromptBoundaries,
    inputs: [u8; 32],
}

#[derive(PartialEq, Eq)]
struct WarmKey {
    system: Option<String>,
    tools: String,
    thinking: u8,
    /// Fold of the bundle's `TokenizerIdentity` digests AND its effective
    /// render flags (replay, notice wrap): a cache filled by
    /// one pinned tokenizer/template never serves another (conservative
    /// miss on any bundle change, content-based — a reloaded identical
    /// pin re-folds to the same value and keeps serving).
    identity: u64,
}

impl Default for WarmTurnCache {
    fn default() -> Self {
        Self::new()
    }
}

/// How `warm_turn` refused, so the fallback can name a cause without
/// re-rendering anything: `Prefix` carries where the cached head and the
/// new head first differ (their common byte length); `Resend` and
/// `Truncated` are the two prefix shapes a bare position cannot tell
/// apart.
enum WarmRefusal {
    Render,
    Prefix { diverge: usize },
    Resend,
    Truncated,
    HeadPartial,
    Seam,
}

impl WarmTurnCache {
    /// An empty cache; the first turn always takes the full path.
    pub fn new() -> Self {
        #[cfg(test)]
        let digest_calls = 0;
        Self {
            key: None,
            head_render: String::new(),
            head_ids: Vec::new(),
            fallbacks: 0,
            reuses: 0,
            last_fallback: None,
            first_head_len: 0,
            first_head_end: None,
            boundary_whole_head: false,
            self_check: false,
            last_turn: None,
            #[cfg(test)]
            digest_calls,
        }
    }

    /// A clone of the last completed warm turn's result. Cloned, not
    /// taken: a resend-HIT reuses this result, so the cache KEEPS the
    /// last turn's ids and bounds as standing state (counted by
    /// `accounted_bytes`).
    fn last_result(&self) -> (Vec<u32>, PromptBoundaries) {
        let last = self
            .last_turn
            .as_ref()
            .expect("a completed turn always stores its result");
        (last.ids.clone(), last.bounds)
    }

    /// Same as [`Self::new`], but every warm turn cross-checks its ids
    /// against a full encode and panics on mismatch — the debug knob
    /// (test builds always cross-check).
    pub fn with_self_check() -> Self {
        Self {
            self_check: true,
            ..Self::new()
        }
    }

    /// How many turns took the full path (cache miss, key change, or any
    /// refused seam) — every doubt is counted, never guessed.
    pub fn fallbacks(&self) -> usize {
        self.fallbacks
    }

    /// The fixed cause string of the last counted fallback, if any.
    pub fn last_fallback_reason(&self) -> Option<&'static str> {
        self.last_fallback
    }

    /// Maps a warm refusal to its fixed cause string. A divergence before
    /// the boundary render's end means the conversation itself differs —
    /// an unrelated request, or a slot another conversation reseeded —
    /// which the serving log must not count as a seam miss.
    fn cause_of(&self, refusal: &WarmRefusal) -> &'static str {
        match refusal {
            WarmRefusal::Render => "render",
            WarmRefusal::Seam => "seam",
            WarmRefusal::Resend => "resend",
            WarmRefusal::Truncated => "truncated",
            WarmRefusal::HeadPartial => "head-partial-token",
            WarmRefusal::Prefix { diverge } => {
                if self.first_head_len == 0 {
                    // The boundary render was never derived (a head that
                    // grew past the first turn, or a template reflow): a
                    // bare position cannot classify the divergence, and a
                    // guess would make the label depend on history — the
                    // serving log gets an honest bucket instead.
                    "unresolved"
                } else if *diverge < self.first_head_len {
                    "other-conversation"
                } else {
                    "seam"
                }
            }
        }
    }

    /// How many turns reused the cache: appended turns reuse the head and
    /// tokenize only the delta; an identical resend returns the stored
    /// result with no work at all.
    pub fn reuses(&self) -> usize {
        self.reuses
    }

    /// Test-only: digest invocations charged to this cache so far.
    #[cfg(test)]
    pub fn digest_calls(&self) -> usize {
        self.digest_calls
    }

    /// Test-only: the EXACT `accounted_bytes` contribution of the stored
    /// last turn (ids capacity + the 32-byte digest), so a test can pin
    /// it apart from head and key allocations.
    #[cfg(test)]
    pub fn last_turn_bytes(&self) -> usize {
        self.last_turn
            .as_ref()
            .map_or(0, |t| 4 * t.ids.capacity() + 32)
    }

    /// Test-only: the head-and-key part of the footprint, recomputed
    /// INDEPENDENTLY of `accounted_bytes` so a test can assert the exact
    /// split (an undercount in either half breaks the equality).
    #[cfg(test)]
    pub fn head_and_key_bytes(&self) -> usize {
        self.head_render.capacity()
            + 4 * self.head_ids.capacity()
            + self.key.as_ref().map_or(0, |k| {
                k.system.as_ref().map_or(0, |s| s.capacity()) + k.tools.capacity()
            })
    }

    /// The live cache footprint in bytes: the head render's and key
    /// strings' CAPACITY plus four bytes per id CAPACITY — capacity, not
    /// len, so a cleared-but-retained buffer cannot report zero while
    /// holding its allocation.
    pub fn accounted_bytes(&self) -> usize {
        self.head_render.capacity()
            + 4 * self.head_ids.capacity()
            + self.key.as_ref().map_or(0, |k| {
                k.system.as_ref().map_or(0, |s| s.capacity()) + k.tools.capacity()
            })
            + self
                .last_turn
                .as_ref()
                .map_or(0, |t| 4 * t.ids.capacity() + 32)
    }
}

impl TokenizerBundle {
    pub fn load(dir: &Path, pin: &TokenizerPin) -> Result<Self, EngError> {
        let tokenizer_bytes = read_pinned(dir, "tokenizer.json", pin.tokenizer_json)?;
        let config_bytes = read_pinned(dir, "tokenizer_config.json", pin.tokenizer_config_json)?;
        let template_bytes = read_pinned(dir, "chat_template.jinja", pin.chat_template_jinja)?;
        let generation_bytes =
            read_pinned(dir, "generation_config.json", pin.generation_config_json)?;
        let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)?;
        let vocab_size = checked_vocab_size(&tokenizer, pin.vocab_size)?;
        let config = parse_json(&config_bytes, "tokenizer_config.json")?;
        let generation = parse_json(&generation_bytes, "generation_config.json")?;
        let eos_token_ids = eos_lookup(&config, &generation, &tokenizer)?;
        let template = parse_template(&template_bytes)?;
        Ok(Self {
            identity: TokenizerIdentity {
                tokenizer_json: pin.tokenizer_json,
                tokenizer_config_json: pin.tokenizer_config_json,
                chat_template_jinja: pin.chat_template_jinja,
                generation_config_json: pin.generation_config_json,
            },
            tokenizer,
            eos_token_ids,
            vocab_size,
            template,
            #[cfg(test)]
            counts: std::sync::Mutex::new((0, 0)),
        })
    }

    /// Output grammar bound to the exact pinned chat template. Unknown templates retain
    /// the legacy grammar; generated text never changes parser selection.
    pub fn tool_grammar(&self) -> super::toolcall::ToolGrammar {
        self.template.grammar
    }

    pub fn identity(&self) -> &TokenizerIdentity {
        &self.identity
    }

    /// The bundle's pinned digests folded to one u64 for the warm key: a
    /// cheap, content-based identity so a cache filled by one pinned
    /// tokenizer/template never serves a different one. The effective
    /// render flags (replay / notice wrap) fold in too: same pins with
    /// different effective rendering must never share a slot — the
    /// resend shortcut bypasses the render/prefix comparison that used
    /// to catch this.
    fn identity_fold(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.identity.hash(&mut hasher);
        self.template.think_replay.hash(&mut hasher);
        self.template.notice_toolwrap.hash(&mut hasher);
        hasher.finish()
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// The id of `eos_token` in `tokenizer_config.json`. It is the first id of
    /// `eos_token_ids`.
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_ids[0]
    }

    /// Every id that ends generation, without duplicates. The `eos_token` id
    /// comes first. The ids of `eos_token_id` in `generation_config.json`
    /// follow in file order. The reference loader builds the same set.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    /// Whether this bundle's template wraps demoted notices in the
    /// tool-response shape (only meaningful when replay is on).
    pub fn notice_toolwrap(&self) -> bool {
        self.template.notice_toolwrap
    }

    /// Whether this bundle's template replays resent assistant thinking.
    /// The serving surface drops reasoning for templates without it so every
    /// template keeps the bytes it rendered before the replay feature.
    pub fn think_replay(&self) -> bool {
        self.template.think_replay
    }

    /// Test-only: forces the template's replay flags off so bundled test
    /// fixtures do not depend on the ambient environment.
    #[cfg(test)]
    pub fn force_flags_off(&mut self) {
        self.template.force_flags_off();
    }

    /// Test-only: forces both replay flags on (see the template method).
    #[cfg(test)]
    pub fn force_flags_on(&mut self) {
        self.template.force_flags_on();
    }

    /// Per-entry config opt-out of replay: disables replay and the notice
    /// wrap together (the wrap never rides without replay). `ENGINESRV_THINK_REPLAY=0`
    /// forces both off in every case; config, too, can only turn these off —
    /// never on, and never past template eligibility.
    pub(crate) fn disable_replay(&mut self) {
        self.template.think_replay = false;
        self.template.notice_toolwrap = false;
    }

    /// Per-entry config opt-out of the notice wrap alone, replay staying on.
    /// A no-op when replay is already off (the wrap never rides without it).
    pub(crate) fn disable_notice_wrap(&mut self) {
        if self.template.think_replay {
            self.template.notice_toolwrap = false;
        }
    }

    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    pub fn encode_prompt(
        &self,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
    ) -> Result<Vec<u32>, EngError> {
        let rendered = self.counted_render(msgs, tools, thinking)?;
        if rendered.is_empty() {
            return Err(EngError::BadRequest("rendered prompt is empty".to_string()));
        }
        self.counted_encode(&rendered)
    }

    /// Encodes the prompt like `encode_prompt` and also returns its message boundaries: the
    /// `tip` is the prompt length at the last message boundary, the prompt without its
    /// generation tail, and the `root` is the prompt length at the end of the first system
    /// message. A boundary is `None` unless the ids of its render are the first ids of the
    /// prompt. A set boundary always has `0 < n < ids.len()`, and the root is always shorter
    /// than the tip.
    pub fn encode_prompt_with_boundary(
        &self,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
    ) -> Result<(Vec<u32>, PromptBoundaries), EngError> {
        let ids = self.encode_prompt(msgs, tools, thinking)?;
        let head = self.counted_head_render(msgs, tools, thinking)?;
        let head_ids = self.counted_encode(&head)?;
        let tip = Some(head_ids.len())
            .filter(|&len| len > 0 && len < ids.len() && ids.starts_with(&head_ids));
        let root = self.root_boundary(msgs, tools, thinking, &ids, tip);
        Ok((ids, PromptBoundaries { tip, root }))
    }

    /// Test-only: the per-request suffix ids through the full derivation
    /// path (renders this request's full and head strings, then applies
    /// every precondition in request_suffix_ids).
    #[cfg(test)]
    pub(crate) fn generation_suffix_ids_for(
        &self,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
    ) -> Option<Vec<u32>> {
        let full = self.template.render(msgs, tools, thinking).ok()?;
        let head = self
            .template
            .render_without_generation_prompt(msgs, tools, thinking)
            .ok()?;
        self.request_suffix_ids(&full, &head)
    }

    #[cfg(test)]
    fn bump_counts(&self, slot: usize) {
        let mut counts = self.counts.lock().expect("counts lock");
        match slot {
            0 => counts.0 += 1,
            _ => counts.1 += 1,
        }
    }

    /// No-op outside tests so the counted helpers stay callable from
    /// production paths without paying for the mutex.
    #[cfg(not(test))]
    #[inline]
    fn bump_counts(&self, _slot: usize) {}

    #[cfg(test)]
    pub(crate) fn counts(&self) -> (usize, usize) {
        *self.counts.lock().expect("counts lock")
    }

    #[cfg(test)]
    pub(crate) fn reset_counts(&self) {
        *self.counts.lock().expect("counts lock") = (0, 0);
    }

    fn counted_render(
        &self,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
    ) -> Result<String, EngError> {
        self.bump_counts(0);
        self.template.render(msgs, tools, thinking)
    }

    fn counted_head_render(
        &self,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
    ) -> Result<String, EngError> {
        self.bump_counts(0);
        self.template
            .render_without_generation_prompt(msgs, tools, thinking)
    }

    fn counted_encode(&self, text: &str) -> Result<Vec<u32>, EngError> {
        self.bump_counts(1);
        self.tokenizer.encode(text)
    }

    /// Same single encode as [`Self::counted_encode`], keeping the byte
    /// offsets so the derived tip can verify its seam position.
    fn counted_encode_with_offsets(&self, text: &str) -> Result<(Vec<u32>, ByteOffsets), EngError> {
        self.bump_counts(1);
        self.tokenizer.encode_with_offsets(text)
    }

    /// The single-pass boundary: derives the tip from the generation-prompt
    /// suffix of THIS request and NEVER approximates. Soundness: the
    /// tokenizer-wide load-time eligibility (no truncation, no padding,
    /// every added token with single_word/lstrip/rstrip/normalized all
    /// false) guarantees that an added-token split is byte-faithful, so
    /// `encode(full) == encode(head) ++ encode(suffix)` when the head is a
    /// byte-prefix of the full render (`strip_prefix` proof) and the first
    /// suffix token is an added token; the ids then end with the suffix ids
    /// exactly and `len - k == head_ids.len()` — the three-pass tip derived
    /// WITHOUT re-tokenizing the head (the head still renders once; a cheap
    /// byte-offset check verifies the token at the candidate starts at
    /// `head.len()`). Any violated precondition (ineligible tokenizer,
    /// conditional template, wrapped generation prompt, plain-text suffix)
    /// falls back to encoding the already-rendered head — the same renders
    /// and the same head encode as the three-pass path — and reports
    /// `tip_derived == false`. One exception: an added-text suffix that the
    /// cheap pre-screen lets through but a guard refuses costs one extra
    /// suffix encode before the fallback. The ids and root semantics are
    /// identical to [`Self::encode_prompt_with_boundary`] on every path.
    pub fn encode_prompt_with_derived_boundary(
        &self,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
    ) -> Result<(Vec<u32>, PromptBoundaries, bool), EngError> {
        self.derive_boundary(msgs, tools, thinking, None)
    }

    /// Incremental warm-turn encoding on a per-session cache: byte-exact
    /// ids and boundaries, tokenizing only the appended delta and the
    /// generation prompt when the previous turn's head is reusable, and
    /// falling back to the full derived path (counted) on any doubt. See
    /// [`WarmTurnCache`] for the exactness argument and the memory bound.
    pub fn encode_prompt_warm(
        &self,
        cache: &mut WarmTurnCache,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
    ) -> Result<(Vec<u32>, PromptBoundaries), EngError> {
        // At most one tools serialization and one digest per request: the
        // key reuses the JSON string, the digest hashes it, and `warm_turn`
        // stores the already-computed digest instead of re-deriving it
        // (two whole-conversation passes cost 1.5-3 ms at 108K tokens
        // against a 3.3 ms measured warm turn).
        let tools_json = serde_json::to_string(tools).unwrap_or_default();
        let key = warm_key(msgs, thinking, self.identity_fold(), &tools_json);
        // The digest only matters when the key already matches AND the
        // tokenizer is eligible (an ineligible slot can never store or
        // serve a last turn, so its same-key requests would only compute
        // and discard the whole-conversation hash). Exact accounting:
        // cold, key-changed, and ineligible requests compute NO digest;
        // a key-matched request computes exactly one — an append misses
        // it and proceeds, a resend hits it and returns the stored ids.
        let digest =
            (cache.key.as_ref() == Some(&key) && self.tokenizer.boundary_eligible()).then(|| {
                #[cfg(test)]
                {
                    cache.digest_calls += 1;
                }
                resend_digest(msgs, &tools_json, thinking)
            });
        // Resend-HIT (an agent's repeat turn): byte-identical
        // render inputs to the request that filled `last_turn`. The key
        // above pins system + tools + thinking + bundle identity and
        // effective flags; the digest pins every message through an
        // unambiguous JSON serialization, so identical inputs render to
        // identical text and the stored ids ARE the answer — no render,
        // no encode, no fallback count. Anything else (a one-token edit,
        // a reorder, an earlier-message edit, an absent-vs-empty option,
        // a shifted tool-call field, a tool or thinking change, another
        // bundle or another effective flag set) misses the digest or the
        // key and takes the normal append/refuse path.
        let resend_hit = digest.is_some_and(|d| {
            cache
                .last_turn
                .as_ref()
                .is_some_and(|last| last.inputs == d)
        });
        if resend_hit {
            cache.reuses += 1;
            let (ids, bounds) = cache.last_result();
            // Production defaults do NOT re-verify: byte-exactness rests
            // on the deterministic tokenizer + fingerprint arguments. The
            // diagnostic cross-check (test builds always; production via
            // opt-in self_check) re-checks what it serves, so a
            // fingerprint bug cannot hide behind the shortcut in any
            // test or diagnostic build.
            if resend_verify_enabled(cache.self_check, cfg!(test))
                && self
                    .encode_prompt(msgs, tools, thinking)
                    .is_ok_and(|reference| reference != ids)
            {
                panic!("resend shortcut diverged from the full encode");
            }
            return Ok((ids, bounds));
        }
        // The fallback cause, fixed at the exact gate that refused (a
        // &'static str: no allocation, stable in the log line). Order
        // matters: an empty slot reports "seed" — which also covers a slot
        // whose seed never proved a tip and one a concurrent request took
        // before this turn — and a changed key reports "key" (also what a
        // same-slot foreign request looks like). A refusal inside the warm
        // attempt itself asks the cache to map the refusal: a divergence
        // before the boundary render ends is another conversation, a bare
        // repeat is a resend, a shrinking head is a truncation, and the
        // rest are seam-class misses of THIS conversation.
        let eligible = self.tokenizer.boundary_eligible();
        // The head-partial vocab scan runs only past the cheap gates (key,
        // eligibility) — the cold, key-change, and ineligible refusals must
        // not pay for it.
        let mut head_partial = false;
        let attempt = if cache.key.as_ref() == Some(&key) && eligible {
            head_partial = self.tokenizer.ends_inside_added_token(&cache.head_render);
            if head_partial {
                Err(WarmRefusal::HeadPartial)
            } else {
                // warm_turn runs only past the key gate, so the digest is
                // present; the closure is an unreachable safety net.
                self.warm_turn(
                    cache,
                    msgs,
                    tools,
                    thinking,
                    digest.unwrap_or_else(|| resend_digest(msgs, &tools_json, thinking)),
                )
            }
        } else if cache.key.is_none() {
            Err(WarmRefusal::Render)
        } else {
            Err(WarmRefusal::Seam)
        };
        let refusal = match attempt {
            Ok(()) => {
                cache.reuses += 1;
                return Ok(cache.last_result());
            }
            Err(refusal) => refusal,
        };
        cache.fallbacks += 1;
        cache.last_fallback = Some(if cache.key.is_none() {
            "seed"
        } else if cache.key.as_ref() != Some(&key) {
            "key"
        } else if !eligible {
            "ineligible"
        } else if head_partial {
            "head-partial-token"
        } else {
            cache.cause_of(&refusal)
        });
        let (ids, bounds) = self.warm_fallback(cache, msgs, tools, thinking, key)?;
        Ok((ids, bounds))
    }

    /// The warm attempt itself: renders full and head, proves both
    /// junctions with the boundary machinery's own suffix proof, and
    /// stores the result on the cache for `last_result`.
    fn warm_turn(
        &self,
        cache: &mut WarmTurnCache,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
        resend_inputs: [u8; 32],
    ) -> Result<(), WarmRefusal> {
        let full = match self.counted_render(msgs, tools, thinking) {
            Ok(full) if !full.is_empty() => full,
            _ => return Err(WarmRefusal::Render),
        };
        let head = match self.counted_head_render(msgs, tools, thinking) {
            Ok(head) => head,
            _ => return Err(WarmRefusal::Render),
        };
        // The cached head must be a byte-prefix of the new head, and the
        // new head a byte-prefix of the full render; both deltas must be
        // non-empty (the suffix proof below re-derives and checks them).
        // A cached head that is NOT a prefix reports where the bytes first
        // differ, so the cause can tell a foreign conversation from an
        // edit inside this one; an empty delta or a SHRINKING head is a
        // resend or a truncation, which a bare position cannot name.
        match head.strip_prefix(cache.head_render.as_str()) {
            Some(delta) if !delta.is_empty() => {}
            Some(_) => return Err(WarmRefusal::Resend),
            None => {
                if head.len() < cache.head_render.len()
                    && cache.head_render.starts_with(head.as_str())
                {
                    return Err(WarmRefusal::Truncated);
                }
                let diverge = head
                    .as_bytes()
                    .iter()
                    .zip(cache.head_render.as_bytes())
                    .take_while(|(new, old)| new == old)
                    .count();
                return Err(WarmRefusal::Prefix { diverge });
            }
        }
        if full.strip_prefix(head.as_str()).is_none_or(str::is_empty) {
            return Err(WarmRefusal::Seam);
        }
        // Junction B needs its own guard: the NEW head must end outside
        // any partial added token. The suffix proof only shows the suffix
        // starts with a complete added token; it cannot see one that
        // starts INSIDE the head and ends inside the suffix.
        if self.tokenizer.ends_inside_added_token(head.as_str()) {
            return Err(WarmRefusal::HeadPartial);
        }
        // Both junctions must pass the full suffix proof (eligibility,
        // added-token start at byte 0, offset-0 added first id).
        let (Some(head_delta_ids), Some(gen_ids)) = (
            self.request_suffix_ids(head.as_str(), cache.head_render.as_str()),
            self.request_suffix_ids(&full, head.as_str()),
        ) else {
            return Err(WarmRefusal::Seam);
        };
        let mut head_ids = cache.head_ids.clone();
        head_ids.extend_from_slice(&head_delta_ids);
        let mut ids = head_ids.clone();
        ids.extend_from_slice(&gen_ids);
        let tip = Some(head_ids.len()).filter(|&tip| tip > 0 && tip < ids.len());
        let bounds = PromptBoundaries {
            tip,
            // Re-derive the root for THIS prompt — the cached root is the
            // seed's, and a system-only seed stored None even though the
            // appended prompt derives Some(root). Same oracle as the
            // derived path; ids are unchanged.
            root: self.root_boundary(msgs, tools, thinking, &ids, tip),
        };
        if (cache.self_check || cfg!(test))
            && self
                .encode_prompt(msgs, tools, thinking)
                .is_ok_and(|reference| reference != ids)
        {
            panic!("warm encode diverged from the full encode");
        }
        cache.head_render = head;
        cache.head_ids = head_ids;
        cache.last_turn = Some(LastTurn {
            ids,
            bounds,
            inputs: resend_inputs,
        });
        // A system-only seed's boundary covered its whole head with no
        // turn. This hit appends one and re-derives the boundary from what
        // it already rendered — no extra render on the hit path.
        // An ALL-SYSTEM head: the template demotes every non-initial system
        // to a user turn, so the whole head STAYS the boundary region and
        // the endpoint must ADVANCE with the head, or a later divergence in
        // the demoted region mislabels as seam while a direct seed of the
        // same history says other-conversation.
        // Endpoint maintenance, three cases. (1) The boundary covers the
        // whole current head (all-system histories advancing turn by turn;
        // the appended turn being the first non-system row; content growth
        // that byte-extends the head with an unchanged selector; role edits
        // that alias the render and move the selector): the hit's own head
        // render IS the boundary render, exactly what directly seeding this
        // history would store. (2) The selector MOVED and the boundary is
        // now interior: re-derivation needs a render the hit path does not
        // have — honestly unknown, and later Prefix refusals label
        // "unresolved", never seam. (3) An interior endpoint with an
        // unchanged selector needs NO refresh: its bytes are a strict
        // prefix of the old head, and a warm hit only ever byte-extends the
        // head (the strip-prefix proof), so they are still the boundary
        // render's bytes.
        let end_now = first_message_end(msgs);
        if end_now == Some(msgs.len()) {
            cache.first_head_len = cache.head_render.len();
            cache.boundary_whole_head = true;
        } else {
            if end_now != cache.first_head_end {
                cache.first_head_len = 0;
            }
            cache.boundary_whole_head = false;
        }
        cache.first_head_end = end_now;
        Ok(())
    }

    /// The full-path fallback: exact by construction, and it re-seeds the
    /// cache when the head proof holds so the next warm turn can reuse.
    fn warm_fallback(
        &self,
        cache: &mut WarmTurnCache,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
        key: WarmKey,
    ) -> Result<(Vec<u32>, PromptBoundaries), EngError> {
        let (ids, bounds, _exact) =
            self.encode_prompt_with_derived_boundary(msgs, tools, thinking)?;
        match bounds.tip {
            Some(tip) if tip > 0 && tip < ids.len() => {
                if let Ok(head_render) = self.counted_head_render(msgs, tools, thinking) {
                    // The cause split's boundary: the render of the messages
                    // up to AND INCLUDING the first non-system one. The warm
                    // key already pins the system text, so a foreign
                    // conversation that shares it still diverges AT its first
                    // user turn — the boundary must therefore sit past that
                    // turn, or every agent (system-first traffic) would
                    // classify as a seam miss. An edit INSIDE this
                    // conversation's first user turn is byte-indistinguishable
                    // from a foreign turn and also reports other-conversation
                    // — the cheap length compare's known limit.
                    // A SYSTEM-ONLY seed has no turn: the boundary covers
                    // the whole head for now (flagged), and the first warm
                    // hit that appends a turn re-derives it — without a
                    // render on the hit path.
                    // When the boundary is the whole head (end == len), the
                    // head render IS the boundary render — no second render.
                    // Honest cost otherwise: this render runs on EVERY
                    // re-seeding fallback (most non-warm turns), not only a
                    // cold seed — one template render and one String. By the
                    // measured warm-turn scale (3.3 ms at 108K tokens)
                    // the render is small but nonzero; never measured in
                    // isolation on this path.
                    let end = first_message_end(msgs);
                    let (first_head_len, whole_head) = if end == Some(msgs.len()) {
                        (head_render.len(), true)
                    } else {
                        let boundary = end
                            .and_then(|end| {
                                self.counted_head_render(&msgs[..end], tools, thinking).ok()
                            })
                            .map(|first| {
                                if head_render.starts_with(first.as_str()) {
                                    first.len()
                                } else {
                                    0
                                }
                            })
                            .unwrap_or(0);
                        (boundary, false)
                    };
                    cache.first_head_len = first_head_len;
                    cache.first_head_end = end;
                    // whole_head alone: the flag means "endpoint covers the
                    // whole cached head", whatever the head's roles — an
                    // all-system head is one case of it, not the definition.
                    cache.boundary_whole_head = whole_head;
                    cache.head_render = head_render;
                    cache.head_ids = ids[..tip].to_vec();
                    cache.key = Some(key);
                    // The resend state belongs to the hit that filled it;
                    // a reseed starts a fresh conversation — result AND
                    // digest drop together.
                    cache.last_turn = None;
                }
            }
            _ => {
                // No proven head boundary: leave the cache empty so the
                // next turn takes the full path again. The buffers are
                // DROPPED, not cleared — clear() keeps the allocation
                // while accounted_bytes would keep reporting it.
                cache.key = None;
                cache.head_render = String::new();
                cache.last_turn = None;
                cache.head_ids = Vec::new();
            }
        }
        // No last_turn store here: the fallback's own return carries the
        // ids, and only a warm success (or a resend of one) ever reads
        // the stored copy.
        Ok((ids, bounds))
    }

    /// Test-only variant that forces the suffix ids, bypassing the
    /// per-request provenance (`cfg(test)`, so no production caller can
    /// skip the `strip_prefix` proof). The eligibility and alignment
    /// guards still apply.
    #[cfg(test)]
    pub(crate) fn encode_prompt_with_derived_boundary_forced(
        &self,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
        suffix_override: &[u32],
    ) -> Result<(Vec<u32>, PromptBoundaries, bool), EngError> {
        self.derive_boundary(msgs, tools, thinking, Some(suffix_override))
    }

    fn derive_boundary(
        &self,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
        suffix_override: Option<&[u32]>,
    ) -> Result<(Vec<u32>, PromptBoundaries, bool), EngError> {
        // The full prompt renders once; the ids, the suffix step and the
        // fallback all reuse it. The head renders once, reused by both the
        // suffix step and the fallback encode — the fallback's renders and
        // head encode match the three-pass path; a suffix that survives the
        // pre-screen but is refused by a guard costs one extra suffix
        // encode first.
        let full = self.counted_render(msgs, tools, thinking)?;
        if full.is_empty() {
            return Err(EngError::BadRequest("rendered prompt is empty".to_string()));
        }
        let (ids, offsets) = self.counted_encode_with_offsets(&full)?;
        let head = self.counted_head_render(msgs, tools, thinking)?;
        let suffix = match suffix_override {
            Some(override_ids)
                if self.tokenizer.boundary_eligible()
                    && override_ids
                        .first()
                        .is_some_and(|&id| self.tokenizer.is_added_token_id(id)) =>
            {
                override_ids.to_vec()
            }
            _ => self.request_suffix_ids(&full, &head).unwrap_or_default(),
        };
        let mut tip_derived = false;
        let mut tip = None;
        if !suffix.is_empty() && ids.ends_with(&suffix) {
            let candidate = ids.len() - suffix.len();
            // Byte-offset proof: the token AT the candidate must start at
            // head.len(), i.e. the split sits exactly at the head seam —
            // not merely at an id alignment a merge could have crossed.
            let seam_ok = candidate > 0
                && candidate < ids.len()
                && offsets
                    .get(candidate)
                    .is_some_and(|&(start, _)| start == head.len());
            if seam_ok {
                tip = Some(candidate);
                tip_derived = true;
            }
        }
        if !tip_derived {
            let head_ids = self.counted_encode(&head)?;
            tip = Some(head_ids.len())
                .filter(|&len| len > 0 && len < ids.len() && ids.starts_with(&head_ids));
        }
        let root = self.root_boundary(msgs, tools, thinking, &ids, tip);
        Ok((ids, PromptBoundaries { tip, root }, tip_derived))
    }

    /// The per-request suffix ids with every derivation precondition:
    /// the tokenizer is boundary-eligible (checked here as a fast exit),
    /// the head is a byte-prefix of the full text (`strip_prefix`), the
    /// suffix is non-empty, some added token's text starts it (cheap
    /// pre-screen), and its first encoded token is that added token.
    fn request_suffix_ids(&self, full: &str, head: &str) -> Option<Vec<u32>> {
        // Tokenizer-wide eligibility first (load-time: no truncation, no
        // padding, every added token free of contextual/text-altering
        // flags) — under it, an added-token split is byte-faithful.
        if !self.tokenizer.boundary_eligible() {
            return None;
        }
        let suffix = full.strip_prefix(head)?;
        if suffix.is_empty() {
            return None;
        }
        // Cheap text pre-screen: under eligibility an added FIRST token
        // requires some added token's text at the suffix start. A negative
        // answer skips the suffix encode, keeping the eligible fallback at
        // the three-pass encode count.
        if !self.tokenizer.any_added_text_starts(suffix) {
            return None;
        }
        // A first EMITTED id that is added does not prove an added-token
        // match at suffix byte 0 — normalizers can drop leading bytes (they
        // emit nothing). Require the added first token to start at offset 0
        // of the suffix text.
        let (ids, offsets) = self.counted_encode_with_offsets(suffix).ok()?;
        ids.first()
            .zip(offsets.first())
            .is_some_and(|(&id, &(start, _))| start == 0 && self.tokenizer.is_added_token_id(id))
            .then_some(ids)
    }

    /// The token count at the end of the first system message. The pinned Qwen template raises
    /// `No user query found in messages.` for every render without a user turn, so the first
    /// message cannot be rendered alone. This renders the first message plus an empty user turn
    /// (same tools and thinking setting, no generation prompt) and cuts the text of an empty user
    /// turn, which the template writes last, from the end. The system block (the tools block
    /// included) comes first in the pinned template and does not depend on later messages, so
    /// the rest is the text of the first system message. The ids of that text must be the first
    /// ids of the prompt, shorter than the tip. When the tip is `None`, the root is dropped too.
    /// Every error of the root render gives `None`: the root only helps reuse, and the full
    /// prompt rendered already.
    fn root_boundary(
        &self,
        msgs: &[PromptMessage],
        tools: &[ToolSpec],
        thinking: Thinking,
        ids: &[u32],
        tip: Option<usize>,
    ) -> Option<usize> {
        let tip = tip?;
        let first = msgs.first().filter(|m| {
            matches!(m.role, Role::System)
                || (matches!(m.role, Role::Developer)
                    && self.tool_grammar() == super::toolcall::ToolGrammar::Qwen)
        })?;
        let empty_user = PromptMessage {
            role: Role::User,
            content: Some(String::new()),
            reasoning_content: None,
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        };
        let stub = self
            .template
            .render_without_generation_prompt(std::slice::from_ref(&empty_user), &[], thinking)
            .ok()?;
        let render = self
            .template
            .render_without_generation_prompt(&[first.clone(), empty_user], tools, thinking)
            .ok()?;
        let root_text = render.strip_suffix(stub.as_str())?;
        let root_ids = self.tokenizer.encode(root_text).ok()?;
        Some(root_ids.len())
            .filter(|&len| len > 0 && len < ids.len() && len < tip && ids.starts_with(&root_ids))
    }
}

/// How many leading messages the cause split's boundary render covers:
/// every leading system/developer message plus the FIRST non-system one.
/// `None` for empty input (the caller then leaves the boundary unknown).
/// System-like roles mirror `warm_key`, whose system pinning is the reason
/// the boundary must not stop at a system message.
fn first_message_end(msgs: &[PromptMessage]) -> Option<usize> {
    let mut end = None;
    for (index, message) in msgs.iter().enumerate() {
        match message.role {
            Role::System | Role::Developer => continue,
            _ => {
                end = Some(index + 1);
                break;
            }
        }
    }
    // A system-only history has no non-system message: the whole head is the
    // boundary region (every divergence is then a foreign conversation).
    end.or(Some(msgs.len())).filter(|end| *end > 0)
}

/// The cache key: the first system message's text, the serialized tool
/// list, the thinking mode, and the bundle's identity fold. A different
/// system, tools, thinking, or pinned tokenizer/template on the same
/// cache object never reuses the stored head.
fn warm_key(
    msgs: &[PromptMessage],
    thinking: Thinking,
    identity: u64,
    tools_json: &str,
) -> WarmKey {
    let system = msgs.first().and_then(|m| match m.role {
        Role::System | Role::Developer => m.content.clone(),
        _ => None,
    });
    WarmKey {
        system,
        tools: tools_json.to_string(),
        thinking: match thinking {
            Thinking::Default => 0,
            Thinking::On => 1,
            Thinking::Off => 2,
        },
        identity,
    }
}

/// When the resend shortcut re-verifies what it serves: opt-in
/// diagnostic mode or test builds. Production defaults pay nothing —
/// byte-exactness there rests on the deterministic tokenizer and the
/// fingerprint arguments, not on a stored cross-check.
pub(crate) fn resend_verify_enabled(self_check: bool, test_build: bool) -> bool {
    self_check || test_build
}

/// The resend digest: SHA-256 over an UNAMBIGUOUS serialization of the
/// exact render inputs — the message list as serde JSON (absent and
/// empty strings stay distinct; tool-call id, name, and arguments stay
/// framed by JSON structure, never concatenated), the key's tool-list
/// JSON, then the thinking discriminant byte. Two requests with the same
/// digest render to byte-identical prompt text, so the stored ids are
/// the answer. Streams into the hasher: no intermediate buffer.
fn resend_digest(msgs: &[PromptMessage], tools_json: &str, thinking: Thinking) -> [u8; 32] {
    let mut hasher = Sha256::new();
    // Writing to a hash sink is infallible; failure would be a bug in
    // the Serialize impls of these plain data types.
    serde_json::to_writer(&mut hasher, msgs).expect("messages serialize");
    hasher.update(tools_json.as_bytes());
    hasher.update([thinking as u8]);
    hasher.finalize().into()
}

fn read_pinned(dir: &Path, name: &str, expected: [u8; 32]) -> Result<Vec<u8>, EngError> {
    let bytes = fs::read(dir.join(name)).map_err(|error| {
        if error.kind() == ErrorKind::NotFound {
            EngError::Checkpoint(format!("{name} is missing from the checkpoint directory"))
        } else {
            EngError::Checkpoint(format!("{name} could not be read: {error}"))
        }
    })?;
    if digest(&bytes) != expected {
        return Err(EngError::Checkpoint(format!(
            "{name} hash does not match the pinned revision"
        )));
    }
    Ok(bytes)
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

fn parse_json(bytes: &[u8], name: &str) -> Result<serde_json::Value, EngError> {
    serde_json::from_slice::<UniqueValue>(bytes)
        .map(|value| value.0)
        .map_err(|error| EngError::Checkpoint(format!("{name} parse failed: {error}")))
}

fn checked_vocab_size(tokenizer: &Tokenizer, expected: u32) -> Result<u32, EngError> {
    let vocab_size = tokenizer.vocab_size();
    if vocab_size != expected {
        return Err(EngError::Checkpoint(format!(
            "tokenizer vocab size {vocab_size} does not match the pinned vocab size {expected}"
        )));
    }
    Ok(vocab_size)
}

fn eos_lookup(
    config: &serde_json::Value,
    generation: &serde_json::Value,
    tokenizer: &Tokenizer,
) -> Result<Vec<u32>, EngError> {
    let name = config
        .get("eos_token")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            EngError::Checkpoint("tokenizer_config.json does not define eos_token".to_string())
        })?;
    let primary = tokenizer.token_id(name).ok_or_else(|| {
        EngError::Checkpoint(format!(
            "eos_token {name} is not in the tokenizer vocabulary"
        ))
    })?;
    let mut ids = vec![primary];
    for id in generation_eos_ids(generation, tokenizer.vocab_size())? {
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    Ok(ids)
}

// `eos_token_id` in generation_config.json is one integer or a list of
// integers. An absent key adds no id.
fn generation_eos_ids(
    generation: &serde_json::Value,
    vocab_size: u32,
) -> Result<Vec<u32>, EngError> {
    let Some(value) = generation.get("eos_token_id") else {
        return Ok(Vec::new());
    };
    let items = match value {
        serde_json::Value::Array(items) => items.as_slice(),
        single => std::slice::from_ref(single),
    };
    items
        .iter()
        .map(|item| {
            let id = item
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| {
                    EngError::Checkpoint(format!(
                        "generation_config.json eos_token_id must hold non-negative integers, found {item}"
                    ))
                })?;
            if id >= vocab_size {
                return Err(EngError::Checkpoint(format!(
                    "generation_config.json eos_token_id {id} is outside the tokenizer vocabulary of {vocab_size} ids"
                )));
            }
            Ok(id)
        })
        .collect()
}

fn parse_template(bytes: &[u8]) -> Result<ChatTemplate, EngError> {
    let source = std::str::from_utf8(bytes)
        .map_err(|_| EngError::Checkpoint("chat_template.jinja is not UTF-8".to_string()))?;
    ChatTemplate::from_str(source)
}

#[cfg(test)]
mod tests {
    use super::first_message_end;
    use crate::tokenizer::{PromptMessage, Role, WarmTurnCache};

    #[test]
    fn a_render_failure_labels_itself_render_not_seam() {
        // The fallback cause feeds the serving log's reason= field; a render
        // failure (empty or erroring template render) is not a junction
        // divergence and must not be counted as one.
        let cache = WarmTurnCache::new();
        assert_eq!(cache.cause_of(&super::WarmRefusal::Render), "render");
        assert_eq!(cache.cause_of(&super::WarmRefusal::Seam), "seam");
    }

    fn message(role: Role, text: &str) -> PromptMessage {
        PromptMessage {
            role,
            content: Some(text.to_string()),
            reasoning_content: None,
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    /// The boundary rule as a table: a Developer opener counts as
    /// system-like; several leading system messages all skip; a system-only
    /// history makes the WHOLE head the boundary; empty input is None (the
    /// caller keeps the boundary unknown).
    #[test]
    fn first_message_end_covers_leading_system_messages_plus_one_turn() {
        let cases: &[(&str, Vec<PromptMessage>, Option<usize>)] = &[
            (
                "developer first",
                vec![message(Role::Developer, "d"), message(Role::User, "u")],
                Some(2),
            ),
            (
                "several leading system",
                vec![
                    message(Role::System, "a"),
                    message(Role::System, "b"),
                    message(Role::User, "u"),
                    message(Role::Assistant, "m"),
                ],
                Some(3),
            ),
            (
                "user first, no system",
                vec![message(Role::User, "u"), message(Role::Assistant, "m")],
                Some(1),
            ),
            (
                "tool is a turn",
                vec![message(Role::System, "s"), message(Role::Tool, "t")],
                Some(2),
            ),
            (
                "system only",
                vec![message(Role::System, "a"), message(Role::System, "b")],
                Some(2),
            ),
            ("empty", Vec::new(), None),
        ];
        for (what, msgs, expected) in cases {
            assert_eq!(&first_message_end(msgs), expected, "{what}");
        }
    }
}
