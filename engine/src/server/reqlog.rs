//! The request log: one JSON line on standard error for each chat request that has a prompt.
//!
//! Rules: the server design ("Request log"). The log shows counts, timings, and codes. It
//! never shows prompt text, reply text, a key, or a header value. `RequestLog` follows one request.
//! The line goes out once, when the request ends. A request that the client abandons ends when its
//! `RunningJob` drops, and its line says `cancelled`.

use super::error::ServerError;
use super::wire::{FinishReason, Usage};
use crate::sched::job::PrefixReport;
use serde_json::json;
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
    /// pump reads the `Finished` event of the request's own job, and `None` keeps the four
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
    if let Some(prefix) = entry.prefix
        && let Some(object) = line.as_object_mut()
    {
        object.insert("prefix_hit".to_string(), json!(prefix.hit.as_str()));
        object.insert("prefix_entries".to_string(), json!(prefix.entries));
        object.insert("prefix_bytes".to_string(), json!(prefix.bytes));
        object.insert(
            "prefix_uncached_shared".to_string(),
            json!(prefix.uncached_shared),
        );
    }
    line.to_string()
}

/// Follows one request from its prompt to its end and writes its line once.
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
        }
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

    /// Writes the line of a request that failed. After the headers of a stream, the client got
    /// status 200 and an error event. In every other case it got the status of the error.
    pub(crate) fn fail(&mut self, error: &ServerError) {
        let status = if self.headers_sent() {
            200
        } else {
            error.status().as_u16()
        };
        self.write(Some(status), Outcome::Error, Some(error.code()), None, 0);
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
    use super::{Entry, Outcome, RequestLog, default_clock, millis, rate, render, stepped_clock};
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
    const KEYS: [&str; 13] = [
        "completion_tokens",
        "decode_tokens_per_second",
        "error_code",
        "finish_reason",
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
    const PREFIX_KEYS: [&str; 17] = [
        "completion_tokens",
        "decode_tokens_per_second",
        "error_code",
        "finish_reason",
        "model",
        "outcome",
        "prefix_bytes",
        "prefix_entries",
        "prefix_hit",
        "prefix_uncached_shared",
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

    fn request(extra: Value) -> Value {
        let mut body = json!({"model": "model-a", "messages": [{"role": "user", "content": "hi"}]});
        for (key, value) in extra.as_object().expect("object").iter() {
            body[key] = value.clone();
        }
        body
    }

    fn empty_entry() -> Entry<'static> {
        Entry {
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
            }),
            ..empty_entry()
        };
        let line: Value = serde_json::from_str(&render(&entry)).expect("json");
        assert_eq!(
            line,
            json!({
                "request_id": null,
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
                "prefix_uncached_shared": 6,
            })
        );
    }

    #[test]
    fn a_line_goes_out_in_one_write_and_the_drop_adds_none() {
        let capture = Capture::default();
        let sink = Arc::new(Mutex::new(capture.clone()));
        let mut log = RequestLog::new(sink, default_clock(), "m", false, 4);
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
        let mut log = RequestLog::new(Arc::new(Mutex::new(Broken)), default_clock(), "m", false, 4);
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
        let lines = capture.lines();
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
            "prefix_uncached_shared": 0,
        });
        assert_eq!(capture.writes(), vec![format!("{want}\n")]);
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
        let lines = capture.lines();
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
            keys, KEYS,
            "a cancelled request logs the thirteen old keys and no prefix key"
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
        let lines = capture.lines();
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
            keys, KEYS,
            "a refused request logs the thirteen old keys and no prefix key"
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
        assert_eq!(capture.writes(), vec![format!("{}\n", want)]);
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
        let lines = capture.lines();
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
        let lines = capture.lines();
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
        let lines = capture.lines();
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
        let lines = capture.lines();
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert_eq!(
            lines[0],
            json!({
                "request_id": null,
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
        let lines = capture.lines();
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
        let lines = capture.lines();
        assert_eq!(lines.len(), 1, "{lines:?}");
        let line = &lines[0];
        assert_eq!(line["request_id"], "chatcmpl-0000000000000001");
        assert_eq!(line["status"], 409);
        assert_eq!(line["outcome"], "error");
        assert_eq!(line["error_code"], "memory_plan");
        assert_eq!(line["prompt_tokens"], 21);
        assert_eq!(line["completion_tokens"], 0);
        assert_eq!(line["time_to_first_token_ms"], Value::Null);
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
        let lines = capture.lines();
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
