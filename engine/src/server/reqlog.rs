//! The request log: one JSON line on standard error for each chat request that has a prompt.
//!
//! Rules: the server design ("Request log"). The log shows counts, timings, and codes. It
//! never shows prompt text, reply text, a key, or a header value. `RequestLog` follows one request.
//! The line goes out once, when the request ends. A request that the client abandons ends when its
//! `RunningJob` drops, and its line says `cancelled`.

use super::error::ServerError;
use super::wire::{FinishReason, Usage};
use crate::EngError;
use crate::sched::job::PrefixReport;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::io::Write;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// Where the lines go: standard error in a server, a buffer in a test.
pub(crate) type Sink = Arc<Mutex<dyn Write + Send>>;

/// The sink of a new server state: standard error. The unit tests of this crate get a sink that
/// discards, so that the requests of other tests print no lines. A test that reads the log sets
/// its own sink.
pub(crate) fn default_sink() -> Sink {
    if cfg!(test) {
        Arc::new(Mutex::new(std::io::sink()))
    } else {
        Arc::new(Mutex::new(std::io::stderr()))
    }
}

/// The clock that stamps the moments of a request: a function that reads the current moment. A
/// server request holds the wall clock; a test request holds a clock with fixed steps.
pub(crate) type Clock = Box<dyn FnMut() -> Instant + Send>;

/// The clock of a new request: the wall clock in a server, and in a test build a clock with fixed
/// 2 ms steps, so that every span a test reads is exact and no test waits on the wall clock.
pub(crate) fn default_clock() -> Clock {
    if cfg!(test) {
        stepped_clock(Duration::from_millis(2))
    } else {
        Box::new(Instant::now)
    }
}

/// A clock with fixed steps for a test: read `n` gives `start + n * step`, so the span between two
/// reads is exact and no real time passes.
fn stepped_clock(step: Duration) -> Clock {
    let start = Instant::now();
    let mut reads = 0_u32;
    Box::new(move || {
        let at = start + step.saturating_mul(reads);
        reads = reads.saturating_add(1);
        at
    })
}

/// How a request ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    /// The job finished and the client got the reply.
    Ok,
    /// The request failed. `error_code` names the failure.
    Error,
    /// The client left, or the engine cancelled the job, before the reply was complete.
    Cancelled,
}

impl Outcome {
    fn as_str(self) -> &'static str {
        match self {
            Outcome::Ok => "ok",
            Outcome::Error => "error",
            Outcome::Cancelled => "cancelled",
        }
    }
}

/// The facts of one finished request. `render` turns them into the log line.
struct Entry<'a> {
    method: Option<&'a str>,
    endpoint: Option<&'a str>,
    digests: Option<&'a PromptDigest>,
    request_id: Option<&'a str>,
    model: &'a str,
    stream: bool,
    /// The HTTP status that the client got. `None` when the client got no response.
    status: Option<u16>,
    outcome: Outcome,
    error_code: Option<&'a str>,
    finish_reason: Option<FinishReason>,
    prompt_tokens: u32,
    completion_tokens: u32,
    reused_tokens: u32,
    /// The time from admission to the first batch with a token.
    first_token: Option<Duration>,
    /// The tokens after the first batch and the time from the first to the last batch.
    decode: Option<(u32, Duration)>,
    /// The prefix report of the job (packet SERVE-2-c item 4, fix M3). It is `None` until the
    /// pump reads the `Finished` event of the request's own job, and `None` keeps the six
    /// `prefix_*` keys out of the line — a cancelled, failed, refused, or stop-string request
    /// logs no prefix keys, and neither does an executor without a prefix report.
    prefix: Option<PrefixReport>,
}

/// Milliseconds with one decimal.
fn millis(time: Duration) -> f64 {
    (time.as_secs_f64() * 10_000.0).round() / 10.0
}

/// Tokens per second with two decimals. A rate needs at least one token and a span above zero.
fn rate(tokens: u32, span: Duration) -> Option<f64> {
    if tokens == 0 || span.is_zero() {
        return None;
    }
    Some((f64::from(tokens) / span.as_secs_f64() * 100.0).round() / 100.0)
}

/// The log line without the final newline. `queue_wait_ms` is always `null`: the engine thread
/// reports no start event, so the server cannot split the queue wait from the prefill. The four
/// `prefix_*` keys appear only on a line whose request carried a prefix boundary (decision 0036,
/// addendum 2, item 7; packet SERVE-2-c item 4).
fn render(entry: &Entry<'_>) -> String {
    let mut line = json!({
        "request_id": entry.request_id,
        "method": entry.method,
        "endpoint": entry.endpoint,
        "model": entry.model,
        "stream": entry.stream,
        "status": entry.status,
        "outcome": entry.outcome.as_str(),
        "error_code": entry.error_code,
        "finish_reason": entry.finish_reason.map(FinishReason::as_str),
        "prompt_tokens": entry.prompt_tokens,
        "completion_tokens": entry.completion_tokens,
        "reused_tokens": entry.reused_tokens,
        "queue_wait_ms": null,
        "time_to_first_token_ms": entry.first_token.map(millis),
        "decode_tokens_per_second": entry.decode.and_then(|(tokens, span)| rate(tokens, span)),
    });
    if let Some(digests) = entry.digests
        && let Some(object) = line.as_object_mut()
    {
        object.insert("prompt_hash_1024".to_string(), json!(digests.first_1024));
        object.insert("prompt_hash_4096".to_string(), json!(digests.first_4096));
        object.insert("prompt_hash_full".to_string(), json!(digests.full));
        object.insert("prompt_hash_anchors".to_string(), json!(digests.anchors));
        object.insert(
            "lcp_prev_request_tokens".to_string(),
            json!(digests.lcp_prev_request),
        );
    }
    if let Some(prefix) = entry.prefix
        && let Some(object) = line.as_object_mut()
    {
        object.insert("prefix_hit".to_string(), json!(prefix.hit.as_str()));
        object.insert("prefix_entries".to_string(), json!(prefix.entries));
        object.insert("prefix_bytes".to_string(), json!(prefix.bytes));
        // SF1 attribution: LIFETIME totals; ONE sample per request line (taken after
        // the request's own restore/prefill; lines land in completion order). Attribute
        // a request's drops against the sample of the restore that ran just before it -
        // ordering samples by value recovers restore order - never against the previous
        // line. Requests that write no prefix keys (failed, refused, cancelled, or
        // stop-string) surface their drops in a LATER request's sample.
        object.insert(
            "prefix_restore_drops".to_string(),
            json!(prefix.restore_drops),
        );
        object.insert("prefix_evictions".to_string(), json!(prefix.evictions));
        object.insert(
            "prefix_uncached_shared".to_string(),
            json!(prefix.uncached_shared),
        );
        // B2 attribution: the SAVE side of the store - LIFETIME totals with the same
        // one-sample-per-request-line rule as the SF1 pair above. `prefix_save_room_removals`
        // is the churn `prefix_evictions` cannot see: entries silently removed to admit
        // OTHER saves (the room loop and the denial retry inside save). A lineage whose
        // marks vanish between turns shows a rising room_removals, not evictions; a save
        // that stored nothing shows in `prefix_save_refusals`; a save that was only a use
        // shows in `prefix_save_duplicate_uses`; a Block cut that rolled the lineage's
        // previous cut away shows in `prefix_save_block_rolls`; a tip that replaced the
        // restored entry in place shows in `prefix_save_restored_replaces`. Saves rise at
        // saves, so ordering samples by value recovers save order.
        object.insert(
            "prefix_save_room_removals".to_string(),
            json!(prefix.save_side.room_removals),
        );
        object.insert(
            "prefix_save_refusals".to_string(),
            json!(prefix.save_side.save_refusals),
        );
        object.insert(
            "prefix_save_duplicate_uses".to_string(),
            json!(prefix.save_side.duplicate_uses),
        );
        object.insert(
            "prefix_save_block_rolls".to_string(),
            json!(prefix.save_side.block_rolls),
        );
        object.insert(
            "prefix_save_restored_replaces".to_string(),
            json!(prefix.save_side.restored_replaces),
        );
        // The shadow block-LCP observation (KV-v1 slice 1), only on a report whose lookup
        // ran with the shadow on: what a block-granular restore could have used, beside the
        // exact-key facts above and `reused_tokens`.
        if let Some(shadow) = prefix.shadow
            && let Some(object) = line.as_object_mut()
        {
            object.insert(
                "shadow_block_lcp_128_tokens".to_string(),
                json!(shadow.block_lcp_128),
            );
            object.insert(
                "shadow_block_lcp_256_tokens".to_string(),
                json!(shadow.block_lcp_256),
            );
            object.insert(
                "shadow_usable_checkpoint_tokens".to_string(),
                json!(shadow.usable_checkpoint_tokens),
            );
            object.insert(
                "shadow_divergence_offset_tokens".to_string(),
                json!(shadow.divergence_offset),
            );
            object.insert("shadow_reason".to_string(), json!(shadow.reason.as_str()));
            object.insert(
                "shadow_token_comparisons".to_string(),
                json!(shadow.token_comparisons),
            );
        }
    }
    line.to_string()
}

/// Follows one request from its prompt to its end and writes its line once.
/// Fixed non-secret digest key. The prompt digests must stay stable across
/// restarts so logs of different sessions compare, but they are obfuscation,
/// not secrecy: anyone with the source can reproduce them from known text.
/// Every digest is 16 hex characters.
const DIGEST_KEY: u64 = 0x5f1f_3a7c_d2e1_9b04;

/// The keys a ring row exposes: an ALLOWLIST, so the digest keys,
/// `prefix_*`, and `shadow_*` (prefix-overlap detail) never reach the
/// panel, and a key a future `render` adds cannot leak automatically
/// (Opus SF2). `queue_wait_ms` stays out — the line writes it as a
/// permanent null (the engine thread reports no start event), and a
/// forever-null column is noise. `reused_tokens` stays in: Dan wants
/// reuse visible on the panel.
const RING_KEYS: [&str; 14] = [
    "request_id",
    "method",
    "endpoint",
    "model",
    "stream",
    "status",
    "outcome",
    "error_code",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "reused_tokens",
    "time_to_first_token_ms",
    "decode_tokens_per_second",
];

/// How many rows the ring keeps.
pub(crate) const RING_CAP: usize = 64;

/// The monitor's recent-request ring, owned by `AppState` like the log
/// sink (Opus SF1): every server — and every test server — gets its own,
/// so the parallel suite never shares rows. It holds the last `RING_CAP`
/// written lines that passed the endpoint filter, newest last.
#[derive(Clone, Default)]
pub(crate) struct RecentRing(Arc<Mutex<VecDeque<Value>>>);

impl RecentRing {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Pushes one written line's row: only the allowlisted keys. A
    /// count_tokens line never lands (Claude Code floods it, and it would
    /// inflate the panel's finished-ok count — Opus SF4); a line that
    /// fails to parse as a JSON object is skipped, never fatal.
    fn push(&self, line: &str) {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            return;
        };
        let Some(object) = value.as_object() else {
            return;
        };
        if object.get("endpoint").and_then(Value::as_str) == Some("/v1/messages/count_tokens") {
            return;
        }
        let mut row = serde_json::Map::new();
        for key in RING_KEYS {
            if let Some(value) = object.get(key) {
                row.insert(key.to_string(), value.clone());
            }
        }
        let mut ring = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if ring.len() == RING_CAP {
            ring.pop_front();
        }
        ring.push_back(Value::Object(row));
    }

    /// The rows, newest first, for the monitor route.
    pub(crate) fn rows_newest_first(&self) -> Vec<Value> {
        let ring = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        ring.iter().rev().cloned().collect()
    }
}

/// The keyed digest of a token slice, truncated to 16 hex characters.
/// The naive reference the pins compare against; kept independent of the
/// single-walk implementation on purpose.
#[cfg(test)]
fn digest(ids: &[u32]) -> String {
    let mut state = DIGEST_KEY;
    for &id in ids {
        state ^= u64::from(id);
        state = state.wrapping_mul(0x0100_0000_01b3);
    }
    finalize(state)
}

/// The finalizer of the rolling digest state.
fn finalize(state: u64) -> String {
    let mut z = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    format!("{:016x}", z ^ (z >> 31))
}

/// The prompt digests of one request: the first-1024 and first-4096 token
/// digests, the full-prompt digest, an anchor digest at every 2048-token
/// boundary (prefix digests, so a shared anchor list pins the common prefix
/// length), and the longest common token prefix with the previous request
/// of the same model (the server's own memory, not the engine store).
#[derive(Clone, Debug, Default)]
pub(crate) struct PromptDigest {
    first_1024: String,
    first_4096: String,
    full: String,
    anchors: Vec<String>,
    lcp_prev_request: u32,
}

impl PromptDigest {
    pub(crate) fn of(ids: &[u32], lcp_prev_request: u32) -> Self {
        // One walk: the rolling state mixes each token exactly once and
        // every boundary finalizes only a copy of it, so the anchors stay
        // linear in the prompt length even near the context limit. A
        // boundary at exactly the end still counts (a 2048-token prompt
        // has one anchor).
        let mut state = DIGEST_KEY;
        let mut first_1024 = None;
        let mut first_4096 = None;
        let mut anchors = Vec::new();
        for (i, &id) in ids.iter().enumerate() {
            state ^= u64::from(id);
            state = state.wrapping_mul(0x0100_0000_01b3);
            let n = i + 1;
            if n == 1024 {
                first_1024 = Some(finalize(state));
            }
            if n == 4096 {
                first_4096 = Some(finalize(state));
            }
            if n >= 2048 && n % 2048 == 0 {
                anchors.push(finalize(state));
            }
        }
        Self {
            first_1024: first_1024.unwrap_or_else(|| finalize(state)),
            first_4096: first_4096.unwrap_or_else(|| finalize(state)),
            full: finalize(state),
            anchors,
            lcp_prev_request,
        }
    }
}

pub(crate) struct RequestLog {
    sink: Sink,
    /// Stamps the moments of the request.
    clock: Clock,
    model: String,
    stream: bool,
    request_id: Option<String>,
    prompt_tokens: u32,
    /// Set when the queue admits the job. A stream sends its headers at that moment.
    admitted: Option<Instant>,
    /// The arrival of the first batch with tokens, and its token count.
    first_batch: Option<(Instant, u32)>,
    last_batch: Option<Instant>,
    completion_tokens: u32,
    written: bool,
    /// True after the handler marked a prefix boundary on this request: the line then carries
    /// the four `prefix_*` keys (packet SERVE-2-c item 4) when the job's report arrived.
    prefix_boundary: bool,
    /// The prefix report of the job, set only when the pump read the request's `Finished`
    /// event (fix M3). It stays `None` on a cancelled, failed, refused, or stop-string request.
    prefix: Option<PrefixReport>,
    /// The HTTP method and path that produced this request, and the prompt
    /// digests of the rendered prompt. Set by the handlers before any line
    /// is written; a log built by a test may leave them unset.
    method: Option<&'static str>,
    endpoint: Option<&'static str>,
    digests: Option<PromptDigest>,
    /// The monitor's ring this request's line feeds, when the handler
    /// wired one (the served routes do; a test-built log may skip it).
    ring: Option<RecentRing>,
}

impl RequestLog {
    /// Starts the log of a request whose prompt has `prompt_tokens` tokens. `clock` stamps the
    /// moments of the request: `default_clock` in a server, a fixed-step clock in a test.
    pub(crate) fn new(
        sink: Sink,
        clock: Clock,
        model: &str,
        stream: bool,
        prompt_tokens: u32,
        method: &'static str,
        endpoint: &'static str,
    ) -> Self {
        Self {
            sink,
            clock,
            model: model.to_string(),
            stream,
            request_id: None,
            prompt_tokens,
            admitted: None,
            first_batch: None,
            last_batch: None,
            completion_tokens: 0,
            written: false,
            prefix_boundary: false,
            prefix: None,
            method: Some(method),
            endpoint: Some(endpoint),
            digests: None,
            ring: None,
        }
    }

    /// Wires the monitor's ring: this request's line then feeds it.
    pub(crate) fn set_ring(&mut self, ring: RecentRing) {
        self.ring = Some(ring);
    }

    /// Records the prompt digests of the rendered prompt. `lcp_prev_request`
    /// is the longest common token prefix with the previous request of the
    /// same model, as measured by the server's own memory.
    pub(crate) fn set_prompt_digests(&mut self, ids: &[u32], lcp_prev_request: u32) {
        self.digests = Some(PromptDigest::of(ids, lcp_prev_request));
    }

    /// Records the completion id of the reply.
    pub(crate) fn set_id(&mut self, id: &str) {
        self.request_id = Some(id.to_string());
    }

    /// Records that the queue admitted the job.
    pub(crate) fn admit(&mut self) {
        self.admitted = Some((self.clock)());
    }

    /// Records the arrival of a batch. A batch without tokens counts for nothing.
    pub(crate) fn batch(&mut self, tokens: u32) {
        if tokens == 0 {
            return;
        }
        let now = (self.clock)();
        self.first_batch.get_or_insert((now, tokens));
        self.last_batch = Some(now);
        self.completion_tokens = self.completion_tokens.saturating_add(tokens);
    }

    /// Marks the request as one with a prefix boundary: its line then carries the four
    /// `prefix_*` keys with the prefix report of its job (packet SERVE-2-c item 4), when the
    /// job's `Finished` event is read. A request without a boundary keeps the old line byte for
    /// byte.
    pub(crate) fn set_prefix_boundary(&mut self) {
        self.prefix_boundary = true;
    }

    /// Writes the line of a reply that ended with `finish` and carried the prefix report
    /// `prefix`, which the pump read from the job's `Finished` event. The report lands in the
    /// line's `prefix_*` keys only when the request had a boundary and the report arrived
    /// (fix M3); `None` omits the keys.
    pub(crate) fn done(
        &mut self,
        finish: FinishReason,
        usage: &Usage,
        prefix: Option<&PrefixReport>,
    ) {
        self.prefix = prefix.copied();
        self.write(
            Some(200),
            Outcome::Ok,
            None,
            Some(finish),
            usage.reused_tokens(),
        );
    }

    /// Writes the line of a count_tokens reply: a served request that ran
    /// no job. The count is the whole work, so the line has neither finish
    /// reason nor timings, and no completion tokens.
    pub(crate) fn counted(&mut self) {
        self.write(Some(200), Outcome::Ok, None, None, 0);
    }

    /// Writes the line of a request that failed. After the headers of a stream, the client got
    /// status 200 and an error event. In every other case it got the status of the error.
    pub(crate) fn fail(&mut self, error: &ServerError) {
        let status = if self.headers_sent() {
            200
        } else {
            error.status().as_u16()
        };
        // The log keeps its own vocabulary: the over-length refusal logs as
        // prompt_too_long even though the OpenAI wire code says
        // context_length_exceeded, so the two stay distinguishable in daily.log.
        let code = match error {
            ServerError::Eng(EngError::PromptTooLong { .. }) => "prompt_too_long",
            other => other.code(),
        };
        self.write(Some(status), Outcome::Error, Some(code), None, 0);
    }

    fn headers_sent(&self) -> bool {
        self.stream && self.admitted.is_some()
    }

    fn write(
        &mut self,
        status: Option<u16>,
        outcome: Outcome,
        error_code: Option<&str>,
        finish_reason: Option<FinishReason>,
        reused_tokens: u32,
    ) {
        if self.written {
            return;
        }
        self.written = true;
        let first_token = match (self.admitted, self.first_batch) {
            (Some(admitted), Some((first, _))) => Some(first.duration_since(admitted)),
            _ => None,
        };
        let decode = match (self.first_batch, self.last_batch) {
            (Some((first, first_tokens)), Some(last)) => Some((
                self.completion_tokens.saturating_sub(first_tokens),
                last.duration_since(first),
            )),
            _ => None,
        };
        let entry = Entry {
            method: self.method,
            endpoint: self.endpoint,
            digests: self.digests.as_ref(),
            request_id: self.request_id.as_deref(),
            model: &self.model,
            stream: self.stream,
            status,
            outcome,
            error_code,
            finish_reason,
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            reused_tokens,
            first_token,
            decode,
            prefix: self.prefix.filter(|_| self.prefix_boundary),
        };
        let mut line = render(&entry);
        line.push('\n');
        if let Some(ring) = &self.ring {
            ring.push(&line);
        }
        let mut sink = self.sink.lock().unwrap_or_else(PoisonError::into_inner);
        // A failed write must not fail the request.
        let _ = sink.write_all(line.as_bytes());
    }
}

impl Drop for RequestLog {
    /// A request that ends with no line to its name was abandoned.
    fn drop(&mut self) {
        let status = self.headers_sent().then_some(200);
        self.write(status, Outcome::Cancelled, None, None, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::digest;

    /// R2 pin: every FULL 2048 boundary produces an anchor, including the
    /// boundary that lands exactly on the prompt end.
    /// The head digests of short prompts fall back to the whole-ids
    /// digest, exactly like the naive oracle's min(K, len) — including
    /// the empty prompt.
    #[test]
    fn the_head_digests_of_short_prompts_match_the_whole_ids_digest() {
        for len in [0usize, 1, 1023, 1024, 1025, 4095] {
            let ids: Vec<u32> = (0..len).map(|i| 7 + i as u32).collect();
            let d = PromptDigest::of(&ids, 0);
            let oracle = digest(&ids[..len.min(1024)]);
            assert_eq!(d.first_1024, oracle, "len {len}: {d:?}");
            let oracle = digest(&ids[..len.min(4096)]);
            assert_eq!(d.first_4096, oracle, "len {len}: {d:?}");
        }
    }

    #[test]
    fn a_prompt_of_exactly_2048_tokens_still_carries_one_anchor() {
        let ids: Vec<u32> = (0..2048).map(|i| 1000 + i).collect();
        let d = PromptDigest::of(&ids, 0);
        assert_eq!(d.anchors, vec![digest(&ids)], "exact-2048: {d:?}");
    }

    #[test]
    fn a_prompt_of_exactly_4096_tokens_carries_two_anchors() {
        let ids: Vec<u32> = (0..4096).map(|i| 1000 + i).collect();
        let d = PromptDigest::of(&ids, 0);
        assert_eq!(
            d.anchors,
            vec![digest(&ids[..2048]), digest(&ids)],
            "exact-4096: {d:?}"
        );
    }

    /// R3 pin: anchors are CUMULATIVE prefix digests — the second anchor is
    /// the digest of the whole first 4096 tokens, and an early divergence
    /// changes every subsequent anchor, not just the first.
    #[test]
    fn every_anchor_is_the_digest_of_its_full_prefix_and_early_divergence_spreads() {
        let ids: Vec<u32> = (0..6144).map(|i| 1000 + i).collect();
        let d = PromptDigest::of(&ids, 0);
        assert_eq!(
            d.anchors,
            vec![digest(&ids[..2048]), digest(&ids[..4096]), digest(&ids)],
            "cumulative: {d:?}"
        );
        assert_eq!(d.first_1024, digest(&ids[..1024]));
        assert_eq!(d.first_4096, digest(&ids[..4096]));
        let mut diverged = ids.clone();
        diverged[10] ^= 1;
        let e = PromptDigest::of(&diverged, 0);
        assert_ne!(d.anchors[1], e.anchors[1]);
    }

    use super::{
        Entry, Outcome, PromptDigest, RecentRing, RequestLog, default_clock, millis, rate, render,
        stepped_clock,
    };
    use crate::EngError;
    use crate::sched::job::{PrefixHit, PrefixReport};
    use crate::sched::worker::{Engine, EngineState};
    use crate::server::router::router;
    use crate::server::state::AppState;
    use crate::server::testkit::{self, Ending};
    use crate::server::wire::{FinishReason, Usage};
    use axum::Router;
    use axum::http::StatusCode;
    use futures::StreamExt;
    use serde_json::{Value, json};
    use std::io::Write;
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    const CALL: &str = "<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>";

    /// The keys of every line without a prefix boundary, sorted.
    /// The keys of a route-produced row: the serving path always sets the
    /// prompt digests after the prompt renders.
    const ROUTE_KEYS: [&str; 20] = [
        "completion_tokens",
        "decode_tokens_per_second",
        "endpoint",
        "error_code",
        "finish_reason",
        "lcp_prev_request_tokens",
        "method",
        "model",
        "outcome",
        "prompt_hash_1024",
        "prompt_hash_4096",
        "prompt_hash_anchors",
        "prompt_hash_full",
        "prompt_tokens",
        "queue_wait_ms",
        "request_id",
        "reused_tokens",
        "status",
        "stream",
        "time_to_first_token_ms",
    ];

    const KEYS: [&str; 15] = [
        "completion_tokens",
        "decode_tokens_per_second",
        "endpoint",
        "error_code",
        "finish_reason",
        "method",
        "model",
        "outcome",
        "prompt_tokens",
        "queue_wait_ms",
        "request_id",
        "reused_tokens",
        "status",
        "stream",
        "time_to_first_token_ms",
    ];

    /// The keys of a line whose request carried a prefix boundary (packet SERVE-2-c item 4):
    /// the thirteen keys above plus the four prefix keys, sorted.
    const PREFIX_KEYS: [&str; 31] = [
        "completion_tokens",
        "decode_tokens_per_second",
        "endpoint",
        "error_code",
        "finish_reason",
        "lcp_prev_request_tokens",
        "method",
        "model",
        "outcome",
        "prefix_bytes",
        "prefix_entries",
        "prefix_evictions",
        "prefix_hit",
        "prefix_restore_drops",
        "prefix_save_block_rolls",
        "prefix_save_duplicate_uses",
        "prefix_save_refusals",
        "prefix_save_restored_replaces",
        "prefix_save_room_removals",
        "prefix_uncached_shared",
        "prompt_hash_1024",
        "prompt_hash_4096",
        "prompt_hash_anchors",
        "prompt_hash_full",
        "prompt_tokens",
        "queue_wait_ms",
        "request_id",
        "reused_tokens",
        "status",
        "stream",
        "time_to_first_token_ms",
    ];

    /// A sink that keeps every `write` call apart, so that a test sees how many writes a line took.
    #[derive(Clone, Default)]
    struct Capture(Arc<Mutex<Vec<Vec<u8>>>>);

    impl Write for Capture {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("capture lock").push(bytes.to_vec());
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Capture {
        /// The text of each write.
        fn writes(&self) -> Vec<String> {
            let writes = self.0.lock().expect("capture lock");
            writes
                .iter()
                .map(|bytes| String::from_utf8(bytes.clone()).expect("utf-8 write"))
                .collect()
        }

        /// The lines as JSON. Each write must be one complete line whose keys are exactly the
        /// thirteen keys of a request without a prefix boundary, so a stray prefix key fails
        /// here. A caller whose requests carried a boundary and finished calls
        /// `lines_with_prefix` instead.
        fn lines(&self) -> Vec<Value> {
            self.lines_with_keys(&KEYS)
        }

        /// The lines as JSON of requests that carried a prefix boundary and finished: each
        /// write must be one complete line whose keys are exactly the seventeen keys of a
        /// boundary request.
        fn lines_with_prefix(&self) -> Vec<Value> {
            self.lines_with_keys(&PREFIX_KEYS)
        }

        /// A written line parsed as JSON with the observability keys (method,
        /// endpoint, prompt digests) removed: the legacy exact-object pins
        /// compare the operational fields; the observability keys have their
        /// own dedicated pins.
        fn row_like(write: &str) -> Value {
            let mut line: Value = serde_json::from_str(write).expect("json line");
            let object = line.as_object_mut().expect("object");
            for key in [
                "method",
                "endpoint",
                "prompt_hash_1024",
                "prompt_hash_4096",
                "prompt_hash_full",
                "prompt_hash_anchors",
                "lcp_prev_request_tokens",
            ] {
                object.remove(key);
            }
            line
        }

        /// The lines as JSON, each with exactly the `expected` key set, sorted.
        fn lines_with_keys(&self, expected: &[&str]) -> Vec<Value> {
            self.writes()
                .iter()
                .map(|write| {
                    assert!(write.ends_with('\n'), "{write:?}");
                    assert_eq!(write.matches('\n').count(), 1, "{write:?}");
                    let line: Value = serde_json::from_str(write).expect("json line");
                    let mut keys: Vec<&str> = line
                        .as_object()
                        .expect("object")
                        .keys()
                        .map(String::as_str)
                        .collect();
                    keys.sort_unstable();
                    assert_eq!(keys, expected, "the keys are not the expected set");
                    line
                })
                .collect()
        }
    }

    /// A router over one engine named `e1` that serves `model-a`, and the capture of its request log.
    fn app(engine: &Engine, context_limit: u32) -> (Router, Capture) {
        testkit::wait_for_state(&engine.handle(), EngineState::Ready);
        let capture = Capture::default();
        let entry = testkit::entry("e1", "model-a", engine, context_limit);
        let state = AppState::new(testkit::API_KEY, vec![entry]).with_request_log(capture.clone());
        (router(Arc::new(state)), capture)
    }

    /// A server over one engine with a UNIQUE model name plus its own ring
    /// and request-log capture, so ring tests read exactly their own rows
    /// (the ring is per-`AppState`, Opus SF1 — no process-global state).
    fn ring_app(engine: &Engine, model: &'static str) -> (Router, RecentRing, Capture) {
        testkit::wait_for_state(&engine.handle(), EngineState::Ready);
        let capture = Capture::default();
        let entry = testkit::entry("e1", model, engine, 4_096);
        let state = AppState::new(testkit::API_KEY, vec![entry]).with_request_log(capture.clone());
        let ring = state.recent_ring();
        (router(Arc::new(state)), ring, capture)
    }

    #[tokio::test]
    async fn the_recent_requests_route_serves_only_the_allowlisted_keys() {
        let engine = testkit::engine(testkit::script("a"));
        let model = "monitor-ring-only-1";
        let (app, ring, _capture) = ring_app(&engine, model);
        let body = json!({
            "model": model,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        });
        let (status, _headers, _bytes) =
            testkit::send(&app, testkit::post_json("/v1/messages", &body)).await;
        assert_eq!(status, StatusCode::OK);
        // The route is bearer-authed like every /v1 route.
        let (status, _headers, _bytes) =
            testkit::send(&app, testkit::get("/v1/recent-requests", None)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let request = testkit::get("/v1/recent-requests", Some(testkit::API_KEY));
        let (status, _headers, bytes) = testkit::send(&app, request).await;
        assert_eq!(status, StatusCode::OK);
        let reply = testkit::json_of(&bytes);
        let rows = reply["requests"].as_array().expect("requests array");
        assert_eq!(rows.len(), 1, "this server's ring holds exactly its row");
        let row = &rows[0];
        // Opus SF2: the exact allowlist is the contract — the digest,
        // prefix_*, and shadow_* keys never reach the panel, and a key a
        // future render() adds cannot leak automatically.
        let mut keys: Vec<&str> = row
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        let mut expected = super::RING_KEYS;
        expected.sort_unstable();
        assert_eq!(
            keys, expected,
            "the row exposes exactly the allowlisted keys"
        );
        assert_eq!(row["model"], model);
        assert_eq!(row["endpoint"], "/v1/messages");
        assert_eq!(row["outcome"], "ok");
        // The route's answer and the state's ring agree (newest first).
        assert_eq!(&ring.rows_newest_first()[0], row);
    }

    #[test]
    fn the_ring_filter_drops_count_tokens_lines_in_push() {
        // N1: exercise the filter where it lives — push() itself. (The
        // count_tokens route wires no ring, so driving the route proved
        // nothing: the test passed even with the filter removed.)
        let ring = RecentRing::new();
        ring.push(
            "{\"endpoint\": \"/v1/messages/count_tokens\", \"model\": \"m\", \"prompt_tokens\": 3}",
        );
        assert!(
            ring.rows_newest_first().is_empty(),
            "Claude Code floods count_tokens; it must not inflate the panel"
        );
        ring.push("{\"endpoint\": \"/v1/messages\", \"model\": \"m\", \"prompt_tokens\": 3}");
        assert_eq!(ring.rows_newest_first().len(), 1, "a served line lands");
    }

    /// Opus SF5: a ring row is EXACTLY the written log line reduced to the
    /// allowlist — the capture sink holds the line, the state holds the row.
    #[tokio::test]
    async fn a_ring_row_equals_the_log_line_reduced_to_the_allowlist() {
        let engine = testkit::engine(testkit::script("a"));
        let model = "monitor-ring-only-3";
        let (app, ring, capture) = ring_app(&engine, model);
        let body = json!({
            "model": model,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        });
        let (status, _headers, _bytes) =
            testkit::send(&app, testkit::post_json("/v1/messages", &body)).await;
        assert_eq!(status, StatusCode::OK);
        let line: Value =
            serde_json::from_str(capture.writes().first().expect("one line")).expect("json");
        let mut expected = serde_json::Map::new();
        for key in super::RING_KEYS {
            if let Some(value) = line.get(key) {
                expected.insert(key.to_string(), value.clone());
            }
        }
        assert_eq!(
            ring.rows_newest_first()[0],
            Value::Object(expected),
            "the row is the line minus everything the allowlist does not name"
        );
    }

    #[test]
    fn the_ring_keeps_only_the_last_ring_cap_rows() {
        // This ring is the test's own; nothing else can push to it.
        let ring = RecentRing::new();
        for i in 0..70u32 {
            ring.push(
                &format!(
                    "{{\"endpoint\": \"/v1/messages\", \"model\": \"ring-cap-only\", \"prompt_tokens\": {i}}}"
                ),
            );
        }
        let rows = ring.rows_newest_first();
        assert_eq!(rows.len(), super::RING_CAP, "the ring is capped");
        let tokens: Vec<&Value> = rows.iter().map(|row| &row["prompt_tokens"]).collect();
        assert_eq!(tokens[0], 69, "newest first");
        assert_eq!(
            tokens.last().copied(),
            Some(&Value::from(6)),
            "a row older than the cap is gone"
        );
        assert!(!tokens.iter().any(|value| *value == 5), "row 5 was evicted");
    }

    fn request(extra: Value) -> Value {
        let mut body = json!({"model": "model-a", "messages": [{"role": "user", "content": "hi"}]});
        for (key, value) in extra.as_object().expect("object").iter() {
            body[key] = value.clone();
        }
        body
    }

    #[tokio::test]
    async fn count_tokens_writes_one_ok_line() {
        let engine = testkit::engine(Vec::new());
        let (app, capture) = app(&engine, 4_096);
        let body = request(json!({}));
        let (status, _headers, bytes) =
            testkit::send(&app, testkit::post_json("/v1/messages/count_tokens", &body)).await;
        assert_eq!(status, StatusCode::OK);
        let reply = testkit::json_of(&bytes);
        let lines = capture.lines_with_keys(&ROUTE_KEYS);
        assert_eq!(lines.len(), 1, "one line per count_tokens request");
        let line = &lines[0];
        assert_eq!(line["status"], 200);
        assert_eq!(line["outcome"], "ok");
        assert_eq!(line["prompt_tokens"], reply["input_tokens"]);
        assert_eq!(line["completion_tokens"], 0);
        assert_eq!(line["finish_reason"], Value::Null);
        assert_eq!(line["stream"], false);
        assert_eq!(line["method"], "POST");
        assert_eq!(line["endpoint"], "/v1/messages/count_tokens");
        assert!(
            line["prompt_hash_full"].is_string(),
            "the count carries the prompt digests too: a count_tokens prompt is exactly the next generation prompt"
        );
        // The same count again: the second row's LCP is the whole prompt.
        let (_status, _headers, _bytes) =
            testkit::send(&app, testkit::post_json("/v1/messages/count_tokens", &body)).await;
        let lines = capture.lines_with_keys(&ROUTE_KEYS);
        assert_eq!(lines.len(), 2);
        assert_eq!(
            lines[1]["lcp_prev_request_tokens"],
            lines[1]["prompt_tokens"]
        );
    }

    #[test]
    fn prompt_digests_are_stable_truncated_and_anchored() {
        let ids: Vec<u32> = (0..5000).map(|i| u32::try_from(i).unwrap()).collect();
        let first = PromptDigest::of(&ids, 7);
        let second = PromptDigest::of(&ids, 7);
        assert_eq!(first.first_1024, second.first_1024, "same ids, same digest");
        assert_eq!(first.first_1024.len(), 16, "truncated to 16 hex chars");
        assert_ne!(first.first_1024, first.first_4096, "different heads differ");
        assert_ne!(first.full, first.first_4096, "full differs from any head");
        // Anchors at 2048 and 4096, not at 5000 (only full blocks).
        assert_eq!(first.anchors.len(), 2, "one anchor per full 2048 block");
        // A different tail changes the full digest and the last anchor, not
        // the first head.
        let mut other = ids.clone();
        other[4999] = 99;
        let changed = PromptDigest::of(&other, 0);
        assert_eq!(changed.first_1024, first.first_1024, "head unaffected");
        assert_ne!(changed.full, first.full, "full digest follows the tail");
        // A divergent prefix at token 5: heads, anchors, and full all move.
        let mut early = ids.clone();
        early[5] = 99;
        let early_digest = PromptDigest::of(&early, 0);
        assert_ne!(early_digest.first_1024, first.first_1024);
        assert_ne!(early_digest.anchors.first(), first.anchors.first());
        // The LCP is carried verbatim.
        assert_eq!(PromptDigest::of(&ids, 4242).lcp_prev_request, 4242);
    }

    #[test]
    fn a_line_carries_the_route_and_the_prompt_digests() {
        let capture = Capture::default();
        let ids: Vec<u32> = (0..3000).map(|i| u32::try_from(i + 100).unwrap()).collect();
        let mut log = RequestLog::new(
            Arc::new(Mutex::new(capture.clone())),
            default_clock(),
            "m",
            false,
            3000,
            "POST",
            "/v1/messages",
        );
        log.set_prompt_digests(&ids, 2999);
        log.done(FinishReason::Stop, &Usage::new(3000, 0, 0), None);
        let line = &capture.lines_with_keys(&ROUTE_KEYS)[0];
        assert_eq!(line["method"], "POST");
        assert_eq!(line["endpoint"], "/v1/messages");
        assert_eq!(line["prompt_hash_1024"].as_str().map(str::len), Some(16));
        assert_eq!(line["lcp_prev_request_tokens"], 2999);
        assert_eq!(
            line["prompt_hash_anchors"].as_array().map(Vec::len),
            Some(1),
            "one anchor for 3000 tokens"
        );
        assert!(line["prompt_hash_full"].is_string());
    }

    #[tokio::test]
    async fn a_messages_route_request_logs_its_endpoint() {
        let engine = testkit::engine(testkit::script("a"));
        let (app, capture) = app(&engine, 4_096);
        let body = json!({
            "model": "model-a",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        });
        let (status, _headers, _bytes) =
            testkit::send(&app, testkit::post_json("/v1/messages", &body)).await;
        assert_eq!(status, StatusCode::OK);
        let lines = capture.lines_with_keys(&ROUTE_KEYS);
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert_eq!(lines[0]["method"], "POST");
        assert_eq!(lines[0]["endpoint"], "/v1/messages");
        assert!(
            lines[0]["prompt_hash_full"].is_string(),
            "digests flow on the messages route"
        );
    }

    fn empty_entry() -> Entry<'static> {
        Entry {
            method: None,
            endpoint: None,
            digests: None,
            request_id: None,
            model: "m",
            stream: false,
            status: None,
            outcome: Outcome::Cancelled,
            error_code: None,
            finish_reason: None,
            prompt_tokens: 0,
            completion_tokens: 0,
            reused_tokens: 0,
            first_token: None,
            decode: None,
            prefix: None,
        }
    }

    #[test]
    fn the_line_holds_every_key_with_the_given_values() {
        let entry = Entry {
            request_id: Some("chatcmpl-0000000000000007"),
            model: "demo",
            stream: true,
            status: Some(200),
            outcome: Outcome::Ok,
            finish_reason: Some(FinishReason::Length),
            prompt_tokens: 21,
            completion_tokens: 10,
            reused_tokens: 3,
            first_token: Some(Duration::from_millis(1500)),
            decode: Some((9, Duration::from_secs(2))),
            ..empty_entry()
        };
        let line: Value = serde_json::from_str(&render(&entry)).expect("json");
        assert_eq!(
            line,
            json!({
                "method": null,
                "endpoint": null,
                "request_id": "chatcmpl-0000000000000007",
                "model": "demo",
                "stream": true,
                "status": 200,
                "outcome": "ok",
                "error_code": null,
                "finish_reason": "length",
                "prompt_tokens": 21,
                "completion_tokens": 10,
                "reused_tokens": 3,
                "queue_wait_ms": null,
                "time_to_first_token_ms": 1500.0,
                "decode_tokens_per_second": 4.5,
            })
        );
    }

    #[test]
    fn the_line_gives_null_for_every_value_that_the_request_never_had() {
        let line: Value = serde_json::from_str(&render(&empty_entry())).expect("json");
        for key in [
            "request_id",
            "status",
            "error_code",
            "finish_reason",
            "queue_wait_ms",
            "time_to_first_token_ms",
            "decode_tokens_per_second",
        ] {
            assert_eq!(line[key], Value::Null, "{key} in {line}");
        }
        assert_eq!(line["outcome"], "cancelled");
        assert_eq!(line["prompt_tokens"], 0);
    }

    #[test]
    fn a_rate_needs_a_token_and_a_span_and_times_keep_one_decimal() {
        assert_eq!(rate(0, Duration::from_secs(1)), None);
        assert_eq!(rate(5, Duration::ZERO), None);
        assert_eq!(rate(1, Duration::from_secs(3)), Some(0.33));
        assert_eq!(millis(Duration::from_micros(1234)), 1.2);
        assert_eq!(millis(Duration::from_micros(1999)), 2.0);
    }

    /// The line of a request whose `batches` batches of one token arrive `step` apart on an
    /// injected fixed-step clock: 21 prompt tokens, and a reply that ends with `Length`.
    fn stamped_line(step: Duration, batches: u32) -> Value {
        let capture = Capture::default();
        let mut log = RequestLog::new(
            Arc::new(Mutex::new(capture.clone())),
            stepped_clock(step),
            "model-a",
            false,
            21,
            "GET",
            "/v1/test",
        );
        log.admit();
        for _ in 0..batches {
            log.batch(1);
        }
        log.done(FinishReason::Length, &Usage::new(21, batches, 0), None);
        drop(log);
        let mut lines = capture.lines();
        assert_eq!(lines.len(), 1, "{lines:?}");
        lines.remove(0)
    }

    /// The helper of the timing tests writes one line with the counts of its batches.
    #[test]
    fn a_stamped_line_holds_the_counts_of_its_batches() {
        let line = stamped_line(Duration::from_millis(2), 3);
        assert_eq!(line["prompt_tokens"], 21, "{line}");
        assert_eq!(line["completion_tokens"], 3, "{line}");
        assert_eq!(line["reused_tokens"], 0, "{line}");
    }

    /// The one bound check of the rate computation, on fixed synthetic timings: 10 decode tokens
    /// over 20 ms give exactly the 500 tokens per second bound, and a zero duration gives the
    /// documented fallback of no rate.
    #[test]
    fn the_rate_computation_holds_its_bound_on_fixed_timings_and_a_zero_duration_gives_no_rate() {
        let fast = stamped_line(Duration::from_millis(2), 11);
        assert_eq!(fast["decode_tokens_per_second"], 500.0, "{fast}");
        let flat = stamped_line(Duration::ZERO, 11);
        assert_eq!(flat["decode_tokens_per_second"], Value::Null, "{flat}");
    }

    /// A line whose request carried a prefix boundary carries the four prefix keys with the
    /// report of its job (packet SERVE-2-c item 4). The keys land after the thirteen old keys.
    ///
    /// Mutation check (the keys-without-a-boundary patch): rendering the prefix keys
    /// unconditionally makes the byte-for-byte test of a boundary-free request fail.
    #[test]
    fn a_line_of_a_request_with_a_boundary_carries_the_four_prefix_keys() {
        let entry = Entry {
            prefix: Some(PrefixReport {
                entries: 3,
                bytes: 422_379_520,
                hit: PrefixHit::Root,
                uncached_shared: 6,
                shadow: None,
                restore_drops: 0,
                evictions: 0,
                save_side: crate::family::session::SaveSideTotals::default(),
            }),
            ..empty_entry()
        };
        let line: Value = serde_json::from_str(&render(&entry)).expect("json");
        assert_eq!(
            line,
            json!({
                "request_id": null,
                "method": null,
                "endpoint": null,
                "model": "m",
                "stream": false,
                "status": null,
                "outcome": "cancelled",
                "error_code": null,
                "finish_reason": null,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reused_tokens": 0,
                "queue_wait_ms": null,
                "time_to_first_token_ms": null,
                "decode_tokens_per_second": null,
                "prefix_hit": "root",
                "prefix_entries": 3,
                "prefix_bytes": 422379520,
                "prefix_restore_drops": 0,
                "prefix_evictions": 0,
                "prefix_save_room_removals": 0,
                "prefix_save_refusals": 0,
                "prefix_save_duplicate_uses": 0,
                "prefix_save_block_rolls": 0,
                "prefix_save_restored_replaces": 0,
                "prefix_uncached_shared": 6,
            })
        );
    }

    /// A report with a shadow section renders the six shadow keys after the prefix keys
    /// (KV-v1 slice 1): what a block-granular restore could have used beside what the
    /// exact-key restore took. A report without one renders none of them.
    #[test]
    fn a_line_of_a_shadowed_report_carries_the_shadow_keys() {
        let entry = Entry {
            prefix: Some(PrefixReport {
                entries: 1,
                bytes: 1_000,
                hit: PrefixHit::Miss,
                uncached_shared: 6,
                shadow: Some(crate::family::session::ShadowLcp {
                    block_lcp_128: 128,
                    block_lcp_256: 0,
                    usable_checkpoint_tokens: 0,
                    divergence_offset: 200,
                    reason: crate::family::session::ShadowReason::NoUsableCheckpoint,
                    token_comparisons: 201,
                }),
                restore_drops: 0,
                evictions: 0,
                save_side: crate::family::session::SaveSideTotals::default(),
            }),
            ..empty_entry()
        };
        let line: Value = serde_json::from_str(&render(&entry)).expect("json");
        assert_eq!(line["shadow_block_lcp_128_tokens"], 128, "{line}");
        assert_eq!(line["shadow_block_lcp_256_tokens"], 0, "{line}");
        assert_eq!(line["shadow_usable_checkpoint_tokens"], 0, "{line}");
        assert_eq!(line["shadow_divergence_offset_tokens"], 200, "{line}");
        assert_eq!(line["shadow_reason"], "no-usable-checkpoint", "{line}");
        assert_eq!(line["shadow_token_comparisons"], 201, "{line}");
        // A report without a shadow section renders no shadow key.
        let plain = Entry {
            prefix: Some(PrefixReport {
                entries: 1,
                bytes: 1_000,
                hit: PrefixHit::Miss,
                uncached_shared: 6,
                ..PrefixReport::default()
            }),
            ..empty_entry()
        };
        let plain_line = render(&plain);
        assert!(!plain_line.contains("shadow_"), "{plain_line}");
    }

    #[test]
    fn a_line_goes_out_in_one_write_and_the_drop_adds_none() {
        let capture = Capture::default();
        let sink = Arc::new(Mutex::new(capture.clone()));
        let mut log = RequestLog::new(sink, default_clock(), "m", false, 4, "GET", "/v1/test");
        log.set_id("chatcmpl-9");
        log.admit();
        log.batch(0);
        log.batch(1);
        log.batch(2);
        log.done(FinishReason::Stop, &Usage::new(4, 3, 1), None);
        drop(log);
        let lines = capture.lines();
        assert_eq!(lines.len(), 1, "{:?}", capture.writes());
        assert_eq!(lines[0]["request_id"], "chatcmpl-9");
        assert_eq!(lines[0]["completion_tokens"], 3);
        assert_eq!(lines[0]["reused_tokens"], 1);
        assert_eq!(lines[0]["finish_reason"], "stop");
    }

    struct Broken;

    impl Write for Broken {
        fn write(&mut self, _bytes: &[u8]) -> std::io::Result<usize> {
            Err(std::io::ErrorKind::BrokenPipe.into())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_sink_that_fails_loses_the_line_and_nothing_else() {
        let mut log = RequestLog::new(
            Arc::new(Mutex::new(Broken)),
            default_clock(),
            "m",
            false,
            4,
            "GET",
            "/v1/test",
        );
        log.done(FinishReason::Stop, &Usage::new(4, 0, 0), None);
        drop(log);
    }

    #[tokio::test]
    async fn a_non_streaming_reply_writes_one_line_with_its_counts_and_timings() {
        let probes = testkit::probes(Duration::ZERO);
        let (app, capture) = app(&probes.engine, 100_000);
        let body = request(json!({"temperature": 0, "max_tokens": 5}));
        let post = testkit::post_json("/v1/chat/completions", &body);
        let (status, _headers, bytes) = testkit::send(&app, post).await;
        assert_eq!(status, StatusCode::OK);
        let reply = testkit::json_of(&bytes);
        let lines = capture.lines_with_keys(&ROUTE_KEYS);
        assert_eq!(lines.len(), 1, "{lines:?}");
        let line = &lines[0];
        assert_eq!(line["request_id"], reply["id"]);
        assert_eq!(line["model"], "model-a");
        assert_eq!(line["stream"], false);
        assert_eq!(line["status"], 200);
        assert_eq!(line["outcome"], "ok");
        assert_eq!(line["error_code"], Value::Null);
        assert_eq!(line["finish_reason"], "length");
        assert_eq!(line["prompt_tokens"], reply["usage"]["prompt_tokens"]);
        assert_eq!(line["prompt_tokens"], 21);
        assert_eq!(line["completion_tokens"], 5);
        assert_eq!(line["reused_tokens"], 0);
        assert_eq!(line["queue_wait_ms"], Value::Null);
        // The clock of a test request stamps every moment 2 ms after the last, so the timings are
        // exact: the first token needs 2 ms, and the 4 later tokens need 8 ms, which gives exactly
        // 500 tokens per second, the cap that the wall-clock test flaked on.
        assert_eq!(line["time_to_first_token_ms"], 2.0, "{line}");
        assert_eq!(line["decode_tokens_per_second"], 500.0, "{line}");
    }

    #[tokio::test]
    async fn the_line_of_a_request_that_reused_a_prefix_reports_the_reused_tokens() {
        let engine = testkit::engine([testkit::script("a"), testkit::script("b")].concat());
        testkit::wait_for_state(&engine.handle(), EngineState::Ready);
        let capture = Capture::default();
        let mut entry = testkit::entry("e1", "model-a", &engine, 100_000);
        entry.prefix_reuse = true;
        let state = AppState::new(testkit::API_KEY, vec![entry]).with_request_log(capture.clone());
        let app = router(Arc::new(state));
        let first = request(json!({}));
        let mut second = request(json!({}));
        second["messages"] = json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "more"},
        ]);
        let mut replies = Vec::new();
        for body in [first, second] {
            let post = testkit::post_json("/v1/chat/completions", &body);
            let (status, _headers, bytes) = testkit::send(&app, post).await;
            assert_eq!(status, StatusCode::OK);
            replies.push(testkit::json_of(&bytes));
        }
        let lines = capture.lines_with_prefix();
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert_eq!(lines[0]["reused_tokens"], 0, "first line");
        assert_eq!(lines[1]["reused_tokens"], 10, "second line");
        assert_eq!(
            lines[1]["reused_tokens"],
            replies[1]["usage"]["reused_tokens"]
        );
    }

    /// A router over one engine at the addendum 2 prefix limits (two roots, six entries) whose
    /// entry has `prefix_reuse` on, and the capture of its request log.
    fn prefix_app(engine: &Engine) -> (Router, Capture) {
        testkit::wait_for_state(&engine.handle(), EngineState::Ready);
        let capture = Capture::default();
        let mut entry = testkit::entry("e1", "model-a", engine, 4_096);
        entry.prefix_reuse = true;
        let state = AppState::new(testkit::API_KEY, vec![entry]).with_request_log(capture.clone());
        (router(Arc::new(state)), capture)
    }

    /// Two conversations that share a system prompt of 2,100 "a" bytes; the first user turn is
    /// "hi", the second "yo". Under the fixture tokenizer every ASCII byte is one id and every
    /// special marker is one id, so the system message renders as `<|im_start|>` (1) +
    /// "system\n" (7) + 2,100 + `<|im_end|>` (1) + "\n" (1) = 2,110 ids (the root), each user
    /// turn as `<|im_start|>` (1) + "user\n" + content + `<|im_end|>` (1) + "\n" (1) = 10 ids,
    /// and the generation tail `<|im_start|>assistant\n` as 11 ids: every prompt holds 2,131
    /// ids with its boundary (the tip) at 2,120. The two user turns share the six header ids
    /// 2,110..2,115 (`<|im_start|>` + "user\n") and diverge at the seventh ("y" against
    /// "h"), so the second conversation restores the root, shares 6 ids with the stored tip
    /// beyond it, and restores no tip.
    fn same_system_conversations() -> [Value; 2] {
        let system = "a".repeat(2_100);
        let turn = |content: &str| {
            request(json!({"messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ]}))
        };
        [turn("hi"), turn("yo")]
    }

    #[tokio::test]
    async fn the_first_request_with_a_boundary_logs_a_miss_and_the_two_entries() {
        let engine = testkit::engine_with_prefix_limits(
            [testkit::script("a"), testkit::script("b")].concat(),
            2,
            6,
        );
        let (app, capture) = prefix_app(&engine);
        let first = same_system_conversations();
        let post = testkit::post_json("/v1/chat/completions", &first[0]);
        let (status, _headers, bytes) = testkit::send(&app, post).await;
        assert_eq!(status, StatusCode::OK);
        let reply = testkit::json_of(&bytes);
        let lines = capture.lines_with_prefix();
        assert_eq!(lines.len(), 1, "{lines:?}");
        let line = &lines[0];
        assert_eq!(line["prompt_tokens"], 2_131, "{line}");
        // The store was empty at the restore and holds the root and the tip after the prefill:
        // the worker takes the report after the job's save points.
        assert_eq!(line["prefix_hit"], "miss", "{line}");
        assert_eq!(line["prefix_entries"], 2, "the root and the tip, {line}");
        // The fake charges a flat 1,000 bytes per stored entry (fix L3), and the value flows
        // worker → `JobUsage` → the reply pump → this line.
        assert_eq!(
            line["prefix_bytes"], 2_000,
            "two entries at 1,000 charged bytes each, {line}"
        );
        assert_eq!(line["prefix_uncached_shared"], 0, "{line}");
        // The rendered line itself, not only its parsed keys (fix L4): the thirteen old keys in
        // their render order, then the four prefix keys in theirs, in one write.
        let want = json!({
            "request_id": reply["id"],
            "model": "model-a",
            "stream": false,
            "status": 200,
            "outcome": "ok",
            "error_code": null,
            "finish_reason": "stop",
            "prompt_tokens": 2_131,
            "completion_tokens": 2,
            "reused_tokens": 0,
            "queue_wait_ms": null,
            "time_to_first_token_ms": 2.0,
            "decode_tokens_per_second": 500.0,
            "prefix_hit": "miss",
            "prefix_entries": 2,
            "prefix_bytes": 2_000,
            "prefix_restore_drops": 0,
            "prefix_evictions": 0,
            "prefix_save_room_removals": 0,
            "prefix_save_refusals": 0,
            "prefix_save_duplicate_uses": 0,
            "prefix_save_block_rolls": 0,
            "prefix_save_restored_replaces": 0,
            "prefix_uncached_shared": 0,
        });
        assert_eq!(Capture::row_like(&capture.writes()[0]), want);
    }

    #[tokio::test]
    async fn the_second_conversation_with_the_same_system_prompt_logs_the_root_hit() {
        let engine = testkit::engine_with_prefix_limits(
            [testkit::script("a"), testkit::script("b")].concat(),
            2,
            6,
        );
        let (app, capture) = prefix_app(&engine);
        for body in same_system_conversations() {
            let post = testkit::post_json("/v1/chat/completions", &body);
            let (status, _headers, _bytes) = testkit::send(&app, post).await;
            assert_eq!(status, StatusCode::OK);
        }
        let lines = capture.lines_with_prefix();
        assert_eq!(lines.len(), 2, "{lines:?}");
        let line = &lines[1];
        assert_eq!(
            line["reused_tokens"], 2_110,
            "the root covers the system prompt"
        );
        assert_eq!(line["prefix_hit"], "root", "the root restored, not a tip");
        // The tip of the first conversation shares the template header
        // `<|im_start|>user\n` — the ids 2,110..2,115 — with the second conversation's user
        // turn "yo" before the two diverge ("y" against "h"), so the longest common prefix
        // with the stored tip is 2,116 tokens: 6 beyond the restored root are shared with a
        // stored key that no checkpoint covered (fix M2).
        assert_eq!(
            line["prefix_uncached_shared"], 6,
            "the header of the user turn is shared but uncached, {line}"
        );
        assert_eq!(
            line["prefix_entries"], 3,
            "the root and the two tips, {line}"
        );
        assert_eq!(
            line["prefix_bytes"], 3_000,
            "three entries at 1,000 charged bytes each (fix L3), {line}"
        );
    }

    /// A line of a request that never read its job's `Finished` event carries no prefix keys
    /// (fix M3): a stream that the client drops is cancelled, so the pump reads no report even
    /// though the request had a boundary and the engine had a store.
    ///
    /// Mutation check: emitting the four keys whenever the boundary flag is set (with a
    /// zeroed report) fails the keys row.
    #[tokio::test]
    async fn a_cancelled_request_with_a_boundary_logs_no_prefix_keys() {
        let engine = testkit::engine_with_prefix_limits(testkit::script(&"a".repeat(2_100)), 2, 6);
        let (app, capture) = prefix_app(&engine);
        let first = same_system_conversations();
        let mut body = request(json!({"stream": true}));
        body["messages"] = first[0]["messages"].clone();
        let post = testkit::post_json("/v1/chat/completions", &body);
        let response = tower::ServiceExt::oneshot(app.clone(), post)
            .await
            .expect("response");
        let mut frames = response.into_body().into_data_stream();
        // The role chunk and the first content chunk show that the pump read a batch.
        for _ in 0..2 {
            frames.next().await.expect("a frame").expect("frame bytes");
        }
        drop(frames);
        testkit::wait_until("the cancelled line", || !capture.writes().is_empty());
        let lines = capture.lines_with_keys(&ROUTE_KEYS);
        assert_eq!(lines.len(), 1, "{lines:?}");
        let line = &lines[0];
        assert_eq!(line["outcome"], "cancelled", "{line}");
        let mut keys: Vec<&str> = line
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys, ROUTE_KEYS,
            "a cancelled request logs the route keys and no prefix key"
        );
    }

    /// A request that the server refused before admission never ran a job, so it carries no
    /// prefix keys either (fix M3), although its engine entry turns `prefix_reuse` on.
    ///
    /// Mutation check: emitting the four keys whenever the boundary flag is set (with a
    /// zeroed report) fails the keys row.
    #[tokio::test]
    async fn a_refused_request_with_a_boundary_engine_logs_no_prefix_keys() {
        let engine = testkit::engine_with_prefix_limits(Vec::new(), 2, 6);
        testkit::wait_for_state(&engine.handle(), EngineState::Ready);
        let capture = Capture::default();
        let mut entry = testkit::entry("e1", "model-a", &engine, 21);
        entry.prefix_reuse = true;
        let state = AppState::new(testkit::API_KEY, vec![entry]).with_request_log(capture.clone());
        let app = router(Arc::new(state));
        let post = testkit::post_json("/v1/chat/completions", &request(json!({})));
        let (status, _headers, _bytes) = testkit::send(&app, post).await;
        assert_eq!(status, StatusCode::CONFLICT);
        let lines = capture.lines_with_keys(&ROUTE_KEYS);
        assert_eq!(lines.len(), 1, "{lines:?}");
        let line = &lines[0];
        assert_eq!(line["outcome"], "error", "{line}");
        assert_eq!(line["error_code"], "context_limit", "{line}");
        let mut keys: Vec<&str> = line
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys, ROUTE_KEYS,
            "a refused request logs the route keys and no prefix key"
        );
    }

    #[tokio::test]
    async fn the_line_of_a_request_without_a_boundary_stays_byte_for_byte() {
        let probes = testkit::probes(Duration::ZERO);
        let (app, capture) = app(&probes.engine, 100_000);
        let body = request(json!({"temperature": 0, "max_tokens": 5}));
        let post = testkit::post_json("/v1/chat/completions", &body);
        let (status, _headers, bytes) = testkit::send(&app, post).await;
        assert_eq!(status, StatusCode::OK);
        let reply = testkit::json_of(&bytes);
        // The exact bytes of the old line: the same thirteen keys, in the render order, with
        // the values of this request; the stepped test clock makes the timings exact. A line
        // that gained the prefix keys of a request without a boundary fails here.
        let want = json!({
            "request_id": reply["id"],
            "model": "model-a",
            "stream": false,
            "status": 200,
            "outcome": "ok",
            "error_code": null,
            "finish_reason": "length",
            "prompt_tokens": 21,
            "completion_tokens": 5,
            "reused_tokens": 0,
            "queue_wait_ms": null,
            "time_to_first_token_ms": 2.0,
            "decode_tokens_per_second": 500.0,
        });
        assert_eq!(Capture::row_like(&capture.writes()[0]), want);
    }

    #[tokio::test]
    async fn a_stream_with_a_tool_call_writes_one_line_that_matches_its_usage_chunk() {
        let engine = testkit::engine(testkit::script(&format!("Adding.{CALL}")));
        let (app, capture) = app(&engine, 4_096);
        let tool = json!([{"type": "function", "function": {"name": "add", "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "number"}}}}}]);
        let user = json!([{"role": "user", "content": "zebra-prompt"}]);
        let body = request(json!({"stream": true, "tools": tool, "messages": user}));
        let post = testkit::post_json("/v1/chat/completions", &body);
        let (status, _headers, bytes) = testkit::send(&app, post).await;
        assert_eq!(status, StatusCode::OK);
        let events = testkit::sse_events(&bytes);
        let last: Value = serde_json::from_str(&events[events.len() - 2].data).expect("last chunk");
        let lines = capture.lines_with_keys(&ROUTE_KEYS);
        assert_eq!(lines.len(), 1, "{lines:?}");
        let line = &lines[0];
        assert_eq!(line["request_id"], last["id"]);
        assert_eq!(line["stream"], true);
        assert_eq!(line["status"], 200);
        assert_eq!(line["outcome"], "ok");
        assert_eq!(line["finish_reason"], "tool_calls");
        assert_eq!(line["prompt_tokens"], last["usage"]["prompt_tokens"]);
        assert_eq!(
            line["completion_tokens"],
            last["usage"]["completion_tokens"]
        );
        // One token for each byte of the reply, and the end-of-sequence token.
        let expected = "Adding.".len() + CALL.len() + 1;
        assert_eq!(line["completion_tokens"], expected);
        let raw = &capture.writes()[0];
        for secret in ["zebra-prompt", "Adding", "function", testkit::API_KEY] {
            assert!(!raw.contains(secret), "{secret} reached the log: {raw}");
        }
    }

    #[tokio::test]
    async fn a_stream_that_the_client_drops_writes_one_cancelled_line() {
        let probes = testkit::probes(Duration::ZERO);
        let (app, capture) = app(&probes.engine, 100_000);
        let body = request(json!({"stream": true, "max_tokens": 50_000}));
        let post = testkit::post_json("/v1/chat/completions", &body);
        let response = tower::ServiceExt::oneshot(app.clone(), post)
            .await
            .expect("response");
        let mut frames = response.into_body().into_data_stream();
        // The role chunk and the first content chunk show that the pump read a batch.
        for _ in 0..2 {
            frames.next().await.expect("a frame").expect("frame bytes");
        }
        assert!(capture.writes().is_empty(), "a line came before the end");
        drop(frames);
        testkit::wait_until("the cancelled line", || !capture.writes().is_empty());
        let handle = probes.engine.handle();
        testkit::wait_until("the job to stop", || handle.state() == EngineState::Ready);
        assert!(probes.decodes.load(Ordering::SeqCst) < 50_000);
        let lines = capture.lines_with_keys(&ROUTE_KEYS);
        assert_eq!(lines.len(), 1, "{lines:?}");
        let line = &lines[0];
        assert_eq!(line["stream"], true);
        assert_eq!(line["status"], 200);
        assert_eq!(line["outcome"], "cancelled");
        assert_eq!(line["error_code"], Value::Null);
        assert_eq!(line["finish_reason"], Value::Null);
        let produced = line["completion_tokens"].as_u64().expect("count");
        assert!((1..50_000).contains(&produced), "{produced}");
        assert_eq!(line["time_to_first_token_ms"], 2.0, "{line}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_dropped_non_streaming_request_writes_a_cancelled_line_without_a_status() {
        // The probe delay keeps the job mid-generation until the abort: the non-streaming pump
        // runs eagerly, and a fast probe could finish the 50_000 tokens first. It feeds no
        // asserted duration.
        let probes = testkit::probes(Duration::from_millis(2));
        let (app, capture) = app(&probes.engine, 100_000);
        let body = request(json!({"max_tokens": 50_000}));
        let pending = tokio::spawn(async move {
            let post = testkit::post_json("/v1/chat/completions", &body);
            testkit::send(&app, post).await
        });
        testkit::wait_until("the first decode", || {
            probes.decodes.load(Ordering::SeqCst) > 0
        });
        pending.abort();
        testkit::wait_until("the cancelled line", || !capture.writes().is_empty());
        let lines = capture.lines_with_keys(&ROUTE_KEYS);
        assert_eq!(lines.len(), 1, "{lines:?}");
        let line = &lines[0];
        assert_eq!(line["stream"], false);
        assert_eq!(line["status"], Value::Null);
        assert_eq!(line["outcome"], "cancelled");
        assert_eq!(line["finish_reason"], Value::Null);
    }

    #[tokio::test]
    async fn a_prompt_above_the_context_limit_writes_a_409_line_without_an_id() {
        let engine = testkit::engine(Vec::new());
        let (app, capture) = app(&engine, 21);
        let post = testkit::post_json("/v1/chat/completions", &request(json!({})));
        let (status, _headers, _bytes) = testkit::send(&app, post).await;
        assert_eq!(status, StatusCode::CONFLICT);
        let lines = capture.lines_with_keys(&ROUTE_KEYS);
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert_eq!(
            lines[0],
            json!({
                "request_id": null,
                "method": "POST",
                "endpoint": "/v1/chat/completions",
                "model": "model-a",
                "stream": false,
                "status": 409,
                "outcome": "error",
                "error_code": "context_limit",
                "finish_reason": null,
                "prompt_tokens": 21,
                "completion_tokens": 0,
                "reused_tokens": 0,
                "queue_wait_ms": null,
                "time_to_first_token_ms": null,
                "decode_tokens_per_second": null,
                "prompt_hash_1024": "ea09b61c2b659ce4",
                "prompt_hash_4096": "ea09b61c2b659ce4",
                "prompt_hash_full": "ea09b61c2b659ce4",
                "prompt_hash_anchors": [],
                "lcp_prev_request_tokens": 0,
            })
        );
    }

    #[tokio::test]
    async fn a_streaming_request_refused_before_admission_logs_the_status_of_the_error() {
        let engine = testkit::engine(Vec::new());
        let (app, capture) = app(&engine, 21);
        let body = request(json!({"stream": true}));
        let post = testkit::post_json("/v1/chat/completions", &body);
        let (status, _headers, _bytes) = testkit::send(&app, post).await;
        assert_eq!(status, StatusCode::CONFLICT);
        let lines = capture.lines_with_keys(&ROUTE_KEYS);
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert_eq!(lines[0]["stream"], true);
        assert_eq!(lines[0]["status"], 409);
        assert_eq!(lines[0]["error_code"], "context_limit");
    }

    #[tokio::test]
    async fn an_engine_refusal_after_admission_writes_the_error_line_with_the_id() {
        let engine = testkit::refusing_engine(EngError::MemoryPlan("no room".to_string()));
        let (app, capture) = app(&engine, 4_096);
        let post = testkit::post_json("/v1/chat/completions", &request(json!({})));
        let (status, _headers, _bytes) = testkit::send(&app, post).await;
        assert_eq!(status, StatusCode::CONFLICT);
        let lines = capture.lines_with_keys(&ROUTE_KEYS);
        assert_eq!(lines.len(), 1, "{lines:?}");
        let line = &lines[0];
        assert_eq!(line["request_id"], "chatcmpl-0000000000000001");
        assert_eq!(line["status"], 409);
        assert_eq!(line["outcome"], "error");
        assert_eq!(line["error_code"], "memory_plan");
        assert_eq!(line["prompt_tokens"], 21);
        assert_eq!(line["completion_tokens"], 0);
        assert_eq!(line["time_to_first_token_ms"], Value::Null);
        assert_eq!(line["method"], "POST");
        assert_eq!(line["endpoint"], "/v1/chat/completions");
    }

    #[tokio::test]
    async fn an_over_length_refusal_logs_prompt_too_long_not_the_wire_code() {
        // The log keeps its own vocabulary: the wire says
        // context_length_exceeded, the log line says prompt_too_long, so the
        // two stay distinguishable in daily.log.
        let engine = testkit::refusing_engine(EngError::PromptTooLong {
            prompt_tokens: 118_930,
            limit_tokens: 110_000,
        });
        let (app, capture) = app(&engine, 262_144);
        let post = testkit::post_json("/v1/chat/completions", &request(json!({})));
        let (status, _headers, _bytes) = testkit::send(&app, post).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let lines = capture.lines_with_keys(&ROUTE_KEYS);
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert_eq!(lines[0]["status"], 400);
        assert_eq!(lines[0]["outcome"], "error");
        assert_eq!(lines[0]["error_code"], "prompt_too_long");
    }

    #[tokio::test]
    async fn an_over_length_refusal_on_a_stream_logs_status_200_and_prompt_too_long() {
        // The 0018 pattern: the pool denial arrives after the 200 headers, so
        // the log line keeps status 200 while the error code names the refusal.
        let engine = testkit::refusing_engine(EngError::PromptTooLong {
            prompt_tokens: 118_930,
            limit_tokens: 110_000,
        });
        let (app, capture) = app(&engine, 262_144);
        let body = request(json!({"stream": true}));
        let post = testkit::post_json("/v1/chat/completions", &body);
        let (status, _headers, bytes) = testkit::send(&app, post).await;
        assert_eq!(status, StatusCode::OK);
        let events = testkit::sse_events(&bytes);
        assert_eq!(events[events.len() - 1].name.as_deref(), Some("error"));
        let lines = capture.lines_with_keys(&ROUTE_KEYS);
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert_eq!(lines[0]["stream"], true);
        assert_eq!(lines[0]["status"], 200);
        assert_eq!(lines[0]["outcome"], "error");
        assert_eq!(lines[0]["error_code"], "prompt_too_long");
    }

    #[tokio::test]
    async fn an_error_event_after_the_headers_writes_status_200_and_hides_the_diagnostic() {
        let failure = EngError::Backend {
            operation: "qmm",
            message: "Metal command buffer 7 failed".to_string(),
        };
        let engine = testkit::ending_engine("ab", Ending::Error(failure));
        let (app, capture) = app(&engine, 4_096);
        let body = request(json!({"stream": true}));
        let post = testkit::post_json("/v1/chat/completions", &body);
        let (status, _headers, bytes) = testkit::send(&app, post).await;
        assert_eq!(status, StatusCode::OK);
        let events = testkit::sse_events(&bytes);
        assert_eq!(events[events.len() - 1].name.as_deref(), Some("error"));
        let lines = capture.lines_with_keys(&ROUTE_KEYS);
        assert_eq!(lines.len(), 1, "{lines:?}");
        let line = &lines[0];
        assert_eq!(line["stream"], true);
        assert_eq!(line["status"], 200);
        assert_eq!(line["outcome"], "error");
        assert_eq!(line["error_code"], "backend_error");
        assert_eq!(line["finish_reason"], Value::Null);
        assert_eq!(line["completion_tokens"], 2);
        assert_eq!(line["time_to_first_token_ms"], 2.0, "{line}");
        let raw = &capture.writes()[0];
        assert!(
            !raw.contains("Metal"),
            "the diagnostic reached the log: {raw}"
        );
    }
}
