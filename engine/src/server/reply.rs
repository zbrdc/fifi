//! Reads the events of one job and turns them into reply parts, a finish reason, and usage.
//!
//! Both reply writers use this pump. It applies the stop strings, which `OutputAssembler` matches
//! on decoded text. It splits the text into answer, think text, and tool calls. It does not count
//! tokens: the executor enforces the output limit of the job through `GenerationState` and ends the
//! job with `Length`. Dropping the pump drops the `JobHandle`, which cancels the job.

use super::error::ServerError;
use super::pieces::{Piece, ReplyParser};
use super::reqlog::RequestLog;
use super::wire::{FinishReason, ReplyParts, ToolCall, Usage};
use crate::EngError;
use crate::sched::job::{JobEvent, JobHandle, JobUsage, PrefixReport};
use crate::tokenizer::{DecodeBatch, DecodeFinish, OutputAssembler};

/// What the pump learned from one event.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// New parts for the client. The job continues. The list is never empty.
    Pieces(Vec<Piece>),
    /// The job ended. `pieces` are the last parts for the client.
    Done {
        pieces: Vec<Piece>,
        finish: FinishReason,
        usage: Usage,
    },
    /// The engine ended the job without a finish: a cancellation that the pump did not request,
    /// or an output channel that overflowed. The client gets no more data.
    Aborted,
}

/// A submitted job and the output state of its reply. Dropping it drops the `JobHandle`,
/// which cancels the job. It also owns the request log of the job: the pump writes the line of a
/// finished reply and of a failed step, and the drop writes the line of an abandoned request.
pub(crate) struct RunningJob {
    handle: JobHandle,
    assembler: OutputAssembler,
    parser: ReplyParser,
    prompt_tokens: u32,
    produced: u32,
    /// The prompt tokens that the engine served from a saved prefix state. It is known only from
    /// the `Finished` event, which the pump reads after a job that ended by itself. A reply that a
    /// stop string ended reports 0, because the pump does not wait for the cancelled job.
    reused: u32,
    /// The prefix-store report of the job, read from the same `Finished` event as `reused`, with
    /// the same fallback: it stays `None` unless the pump read the event, so a cancelled, failed,
    /// refused, or stop-string reply logs no prefix keys.
    prefix: Option<PrefixReport>,
    calls: u32,
    /// The log of the request. A writer that ends the request without a step writes through it.
    pub(crate) log: RequestLog,
}

fn finish_reason(finish: DecodeFinish) -> Option<FinishReason> {
    match finish {
        DecodeFinish::Stop => Some(FinishReason::Stop),
        DecodeFinish::Length => Some(FinishReason::Length),
        DecodeFinish::Cancelled => None,
    }
}

impl RunningJob {
    /// Wraps a submitted job. The job carries its own output limit.
    pub(crate) fn new(
        handle: JobHandle,
        assembler: OutputAssembler,
        parser: ReplyParser,
        prompt_tokens: u32,
        log: RequestLog,
    ) -> Self {
        Self {
            handle,
            assembler,
            parser,
            prompt_tokens,
            produced: 0,
            reused: 0,
            prefix: None,
            calls: 0,
            log,
        }
    }

    /// The stop string that ended the reply, when one did. Read after `collect`
    /// or a `Done` step: the assembler keeps the recorded match.
    pub(crate) fn matched_stop(&self) -> Option<String> {
        self.assembler.matched_stop().map(str::to_string)
    }

    /// The prompt token count of the request, known when the job was accepted.
    /// The stream reports it in message_start, before any step arrives.
    pub(crate) fn prompt_tokens(&self) -> u32 {
        self.prompt_tokens
    }

    /// Waits for the next step. After `Done`, `Aborted`, or an error, the caller stops.
    /// `Done` and an error write the line of the request log.
    pub(crate) async fn next(&mut self) -> Result<Step, ServerError> {
        let step = self.read_step().await;
        if let Err(error) = &step {
            self.log.fail(error);
        }
        step
    }

    async fn read_step(&mut self) -> Result<Step, ServerError> {
        loop {
            let Some(event) = self.handle.events.recv().await else {
                // The channel closed with no terminal event. A cancelled job means a slow client
                // or a disconnect. Otherwise a submit raced with a failure or a stop.
                if self.handle.is_cancelled() {
                    return Ok(Step::Aborted);
                }
                return Err(ServerError::Eng(EngError::Backend {
                    operation: "engine",
                    message: "the engine closed the job without a result".to_string(),
                }));
            };
            match event {
                JobEvent::Batch(batch) => {
                    if batch.finish == Some(DecodeFinish::Cancelled) {
                        return Ok(Step::Aborted);
                    }
                    let committed = u32::try_from(batch.tokens.len()).unwrap_or(u32::MAX);
                    self.produced = self.produced.saturating_add(committed);
                    self.log.batch(committed);
                    if matches!(
                        batch.finish,
                        Some(DecodeFinish::Stop | DecodeFinish::Length)
                    ) {
                        let usage = self.read_finished().await?;
                        self.reused = usage.reused_tokens;
                        self.prefix = usage.prefix;
                    }
                    if let Some(step) = self.on_batch(&batch)? {
                        return Ok(step);
                    }
                }
                JobEvent::Error(error) => return Err(error.into()),
                JobEvent::Finished(_) => {}
            }
        }
    }

    /// Reads the event that follows the last batch of a job that ended by itself. The engine sends
    /// the `Finished` event right after that batch, and the channel always has room for it. Any
    /// explicit engine failure propagates. Other events, or a closed channel, give zero usage.
    async fn read_finished(&mut self) -> Result<JobUsage, ServerError> {
        match self.handle.events.recv().await {
            Some(JobEvent::Finished(usage)) => Ok(usage),
            Some(JobEvent::Error(error)) => Err(error.into()),
            _ => Ok(JobUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
                reused_tokens: 0,
                prefix: None,
            }),
        }
    }

    fn on_batch(&mut self, batch: &DecodeBatch) -> Result<Option<Step>, ServerError> {
        let delta = self.assembler.push_batch(batch)?;
        let mut pieces = self.parser.push(&delta.text);
        if let Some(finish) = delta.finish.and_then(finish_reason) {
            if batch.finish.is_none() {
                // A stop string ended the reply while the job still runs.
                self.handle.cancel();
            }
            pieces.extend(self.parser.finish(finish == FinishReason::Length));
            return Ok(Some(self.done(pieces, finish)));
        }
        self.calls += count_calls(&pieces);
        Ok((!pieces.is_empty()).then_some(Step::Pieces(pieces)))
    }

    fn done(&mut self, pieces: Vec<Piece>, finish: FinishReason) -> Step {
        self.calls += count_calls(&pieces);
        let finish = if self.calls > 0 {
            FinishReason::ToolCalls
        } else {
            finish
        };
        let usage = Usage::new(self.prompt_tokens, self.produced, self.reused);
        self.log.done(finish, &usage, self.prefix.as_ref());
        Step::Done {
            pieces,
            finish,
            usage,
        }
    }
}

fn count_calls(pieces: &[Piece]) -> u32 {
    let calls = pieces
        .iter()
        .filter(|piece| matches!(piece, Piece::Call(_)))
        .count();
    u32::try_from(calls).unwrap_or(u32::MAX)
}

/// Reads the job to its end and returns the pieces in arrival order, the finish
/// reason, and the usage. The Anthropic route walks this so its content blocks
/// stay interleaved as the model produced them. A job that ends without a finish
/// is a backend failure for a reply that has no stream.
pub(crate) async fn collect_pieces(
    job: &mut RunningJob,
) -> Result<(Vec<Piece>, FinishReason, Usage), ServerError> {
    let mut pieces = Vec::new();
    loop {
        match job.next().await? {
            Step::Pieces(step) => pieces.extend(step),
            Step::Done {
                pieces: mut last,
                finish,
                usage,
            } => {
                pieces.append(&mut last);
                return Ok((pieces, finish, usage));
            }
            Step::Aborted => {
                let error = ServerError::Eng(EngError::Backend {
                    operation: "engine",
                    message: "the engine ended the job without a finish".to_string(),
                });
                job.log.fail(&error);
                return Err(error);
            }
        }
    }
}

/// Reads the job to its end and returns the reply parts, the finish reason, and the usage.
/// A job that ends without a finish is a backend failure for a reply that has no stream.
pub(crate) async fn collect(
    job: &mut RunningJob,
) -> Result<(ReplyParts, FinishReason, Usage), ServerError> {
    let (pieces, finish, usage) = collect_pieces(job).await?;
    let mut parts = ReplyParts::default();
    add_pieces(&mut parts, pieces);
    Ok((parts, finish, usage))
}

fn add_pieces(parts: &mut ReplyParts, pieces: Vec<Piece>) {
    for piece in pieces {
        match piece {
            Piece::Content(text) => parts.content.push_str(&text),
            Piece::Reasoning(text) => parts.reasoning.push_str(&text),
            Piece::Call(call) => {
                parts
                    .tool_calls
                    .push(ToolCall::new(call.id, call.name, call.arguments));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RunningJob, Step, collect};
    use crate::EngError;
    use crate::family::executor::GenerationState;
    use crate::sched::job::{Job, JobEvent};
    use crate::sched::worker::Engine;
    use crate::server::error::ServerError;
    use crate::server::pieces::{Piece, ReplyParser};
    use crate::server::reqlog::{RequestLog, default_clock};
    use crate::server::testkit;
    use crate::server::wire::{FinishReason, ReplyHead, Usage};
    use crate::tokenizer::toolcall::{Start, ToolCall as ParsedCall};
    use crate::tokenizer::{DecodeBatch, DecodeFinish, IncrementalDecoder, OutputAssembler};
    use serde_json::json;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    const PROMPT: [u32; 3] = [104, 105, 33];

    /// A request log that writes nowhere. The tests of the pump do not read the log.
    fn silent() -> RequestLog {
        RequestLog::new(
            Arc::new(Mutex::new(std::io::sink())),
            default_clock(),
            "m",
            false,
            3,
            "GET",
            "/v1/test",
        )
    }

    /// A log sink that keeps every write, for the tests that read the request-log line.
    #[derive(Clone, Default)]
    struct Capture(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for Capture {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .expect("capture lock")
                .extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Capture {
        /// Everything written so far, as one line of text.
        fn text(&self) -> String {
            let bytes = self.0.lock().expect("capture lock");
            String::from_utf8(bytes.clone()).expect("utf-8 write")
        }
    }

    fn head() -> ReplyHead {
        ReplyHead {
            id: "chatcmpl-0000000000000001".to_string(),
            created: 0,
            model: "m".to_string(),
            engine: "e1".to_string(),
        }
    }

    /// Submits a three-token prompt with the output limit `budget` and wraps the job in a pump.
    fn run(engine: &Engine, stops: &[&str], budget: u32) -> (RunningJob, Arc<AtomicBool>) {
        submit(engine, stops, budget, None)
    }

    /// Like `run`, with the first `boundary` prompt tokens marked as the reusable prefix.
    fn submit(
        engine: &Engine,
        stops: &[&str],
        budget: u32,
        boundary: Option<usize>,
    ) -> (RunningJob, Arc<AtomicBool>) {
        submit_logged(engine, stops, budget, boundary, silent())
    }

    /// `submit` with the pump's request log given by the caller.
    fn submit_logged(
        engine: &Engine,
        stops: &[&str],
        budget: u32,
        boundary: Option<usize>,
        log: RequestLog,
    ) -> (RunningJob, Arc<AtomicBool>) {
        let bundle = testkit::bundle();
        let generation = GenerationState::new(IncrementalDecoder::new(bundle.tokenizer()))
            .with_max_tokens(budget);
        let cancel = Arc::clone(&generation.cancel);
        let (job, handle) = Job::new(PROMPT.to_vec(), generation).expect("job");
        let job = match boundary {
            Some(boundary) => job.with_prefix_boundary(boundary).expect("boundary"),
            None => job,
        };
        engine.handle().try_submit(job).expect("submit");
        let stops = stops.iter().map(|stop| (*stop).to_string()).collect();
        let assembler = OutputAssembler::new(stops).expect("assembler");
        let parser = ReplyParser::new(
            Start::Content,
            &[testkit::add_tool()],
            &head().call_prefix(),
        );
        (RunningJob::new(handle, assembler, parser, 3, log), cancel)
    }

    async fn steps(job: &mut RunningJob) -> Vec<Step> {
        let mut found = Vec::new();
        loop {
            let step = job.next().await.expect("step");
            let last = !matches!(step, Step::Pieces(_));
            found.push(step);
            if last {
                return found;
            }
        }
    }

    fn content(text: &str) -> Piece {
        Piece::Content(text.to_string())
    }

    /// All content text of the steps, in order.
    fn content_of(found: &[Step]) -> String {
        found
            .iter()
            .flat_map(|step| match step {
                Step::Pieces(pieces) | Step::Done { pieces, .. } => pieces.as_slice(),
                Step::Aborted => &[],
            })
            .filter_map(|piece| match piece {
                Piece::Content(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    const CALL: &str = "<tool_call><function=add><parameter=a>1</parameter><parameter=b>2</parameter></function></tool_call>";

    #[tokio::test]
    async fn text_arrives_per_batch_and_the_last_step_carries_the_usage() {
        let engine = testkit::engine(testkit::script("ab"));
        let (mut job, _cancel) = run(&engine, &[], 100);
        assert_eq!(
            steps(&mut job).await,
            vec![
                Step::Pieces(vec![content("a")]),
                Step::Pieces(vec![content("b")]),
                Step::Done {
                    pieces: Vec::new(),
                    finish: FinishReason::Stop,
                    usage: Usage::new(3, 3, 0),
                },
            ]
        );
    }

    /// The pump hands the prefix report of the `Finished` event to the request log:
    /// the value flows worker → `JobUsage` → this pump → the log line. The fake charges a flat
    /// 1,000 bytes per stored entry, and this boundary job stores one tip of two tokens, so
    /// the line of the finished request carries the four prefix keys with `prefix_bytes` 1000.
    ///
    /// Mutation check: zeroing `bytes` in the fake's report makes the `prefix_bytes` row fail.
    #[tokio::test]
    async fn a_finished_job_hands_its_prefix_report_to_the_request_log() {
        let engine = testkit::engine(testkit::script("a"));
        let capture = Capture::default();
        let mut log = RequestLog::new(
            Arc::new(Mutex::new(capture.clone())),
            default_clock(),
            "m",
            false,
            3,
            "GET",
            "/v1/test",
        );
        log.set_prefix_boundary();
        let (mut job, _cancel) = submit_logged(&engine, &[], 100, Some(2), log);
        let found = steps(&mut job).await;
        assert!(
            matches!(found.last(), Some(Step::Done { .. })),
            "the reply finished"
        );
        let line: serde_json::Value =
            serde_json::from_str(capture.text().trim_end()).expect("the log line");
        assert_eq!(line["prefix_hit"], "miss", "{line}");
        assert_eq!(line["prefix_entries"], 1, "one stored tip, {line}");
        assert_eq!(
            line["prefix_bytes"], 1_000,
            "one entry at 1,000 bytes, {line}"
        );
        assert_eq!(line["prefix_uncached_shared"], 0, "{line}");
    }

    #[tokio::test]
    async fn a_stop_string_ends_the_reply_hides_the_stop_text_and_cancels_the_job() {
        let engine = testkit::engine(testkit::script("xxENDyy"));
        let (mut job, cancel) = run(&engine, &["END"], 100);
        let found = steps(&mut job).await;
        assert_eq!(content_of(&found), "xx");
        assert!(
            found
                .iter()
                .all(|step| !matches!(step, Step::Pieces(pieces) if pieces.is_empty())),
            "an empty step reached the caller: {found:?}"
        );
        assert_eq!(
            found.last(),
            Some(&Step::Done {
                pieces: Vec::new(),
                finish: FinishReason::Stop,
                usage: Usage::new(3, 5, 0),
            })
        );
        assert!(
            cancel.load(Ordering::SeqCst),
            "the running job was not cancelled"
        );
    }

    #[tokio::test]
    async fn the_job_limit_ends_the_reply_with_length_and_releases_held_text() {
        let engine = testkit::engine(testkit::script("abcdef"));
        let (mut job, cancel) = run(&engine, &["cd"], 3);
        let found = steps(&mut job).await;
        assert_eq!(
            found.last(),
            Some(&Step::Done {
                pieces: vec![content("c")],
                finish: FinishReason::Length,
                usage: Usage::new(3, 3, 0),
            })
        );
        assert!(
            !cancel.load(Ordering::SeqCst),
            "the pump cancelled a job that the executor had ended"
        );
    }

    #[tokio::test]
    async fn think_text_and_tool_calls_leave_the_answer_and_change_the_finish_reason() {
        let text = format!("<think>plan</think>Adding.{CALL}");
        let engine = testkit::engine(testkit::script(&text));
        let (mut job, _cancel) = run(&engine, &[], 1_000);
        let (parts, finish, usage) = collect(&mut job).await.expect("collect");
        assert_eq!(parts.reasoning, "plan");
        assert_eq!(parts.content, "Adding.");
        assert_eq!(finish, FinishReason::ToolCalls);
        assert_eq!(
            usage,
            Usage::new(3, u32::try_from(text.len()).expect("len") + 1, 0)
        );
        let calls = serde_json::to_value(&parts.tool_calls).expect("calls");
        assert_eq!(
            calls,
            json!([{"id": "call_0000000000000001_0", "type": "function",
                    "function": {"name": "add", "arguments": "{\"a\":1,\"b\":2}"}}])
        );
    }

    #[tokio::test]
    async fn a_call_streams_as_a_piece_only_when_it_is_complete() {
        let engine = testkit::engine(testkit::script(CALL));
        let (mut job, _cancel) = run(&engine, &[], 1_000);
        let found = steps(&mut job).await;
        let call_steps: Vec<usize> = found
            .iter()
            .enumerate()
            .filter(|(_, step)| match step {
                Step::Pieces(pieces) | Step::Done { pieces, .. } => {
                    pieces.iter().any(|piece| matches!(piece, Piece::Call(_)))
                }
                Step::Aborted => false,
            })
            .map(|(index, _)| index)
            .collect();
        assert_eq!(call_steps.len(), 1, "{found:?}");
        let expected = ParsedCall {
            id: "call_0000000000000001_0".to_string(),
            name: "add".to_string(),
            arguments: "{\"a\":1,\"b\":2}".to_string(),
        };
        let last_pieces = match &found[call_steps[0]] {
            Step::Pieces(pieces) | Step::Done { pieces, .. } => pieces,
            Step::Aborted => unreachable!("filtered above"),
        };
        assert_eq!(last_pieces, &vec![Piece::Call(expected)]);
        assert_eq!(content_of(&found), "");
    }

    #[tokio::test]
    async fn a_call_left_open_at_the_job_limit_is_dropped_and_the_reason_stays_length() {
        let text = "hi <tool_call><function=add><parameter=a>1</parameter></function></tool_call>";
        let engine = testkit::engine(testkit::script(text));
        let (mut job, _cancel) = run(&engine, &[], 20);
        let found = steps(&mut job).await;
        assert_eq!(content_of(&found), "hi");
        let Some(Step::Done { finish, .. }) = found.last() else {
            panic!("expected Done, got {found:?}");
        };
        assert_eq!(*finish, FinishReason::Length);
    }

    #[tokio::test]
    async fn a_call_left_open_when_the_reply_stops_stays_as_text() {
        let text = "ok <tool_call>\n<function=add>";
        let engine = testkit::engine(testkit::script(text));
        let (mut job, _cancel) = run(&engine, &[], 1_000);
        let found = steps(&mut job).await;
        assert_eq!(content_of(&found), "ok<tool_call>\n<function=add>");
        let Some(Step::Done { finish, .. }) = found.last() else {
            panic!("expected Done, got {found:?}");
        };
        assert_eq!(*finish, FinishReason::Stop);
    }

    #[tokio::test]
    async fn an_unrequested_cancellation_from_the_engine_is_an_abort() {
        let engine = testkit::ending_engine("ab", testkit::Ending::Cancel);
        let (mut job, _cancel) = run(&engine, &[], 100);
        let found = steps(&mut job).await;
        assert_eq!(found.last(), Some(&Step::Aborted));
        assert_eq!(content_of(&found), "ab");
    }

    #[tokio::test]
    async fn an_engine_error_becomes_the_error_of_the_step() {
        let engine = testkit::refusing_engine(EngError::MemoryPlan("no room".to_string()));
        let (mut job, _cancel) = run(&engine, &[], 100);
        let error = job.next().await.expect_err("error");
        assert_eq!(
            error,
            ServerError::Eng(EngError::MemoryPlan("no room".to_string()))
        );
    }

    #[tokio::test]
    async fn a_job_cancelled_before_it_starts_is_aborted() {
        let (engine, gate) = testkit::gated_engine(testkit::script("ab"));
        let (mut job, cancel) = run(&engine, &[], 100);
        cancel.store(true, Ordering::SeqCst);
        drop(gate);
        assert_eq!(job.next().await, Ok(Step::Aborted));
    }

    #[tokio::test]
    async fn a_cancelled_job_whose_channel_closes_is_aborted() {
        let bundle = testkit::bundle();
        let generation = GenerationState::new(IncrementalDecoder::new(bundle.tokenizer()));
        let (job, handle) = Job::new(PROMPT.to_vec(), generation).expect("job");
        drop(job);
        handle.cancel();
        let assembler = OutputAssembler::new(Vec::new()).expect("assembler");
        let parser = ReplyParser::new(Start::Content, &[], "call_");
        let mut running = RunningJob::new(handle, assembler, parser, 3, silent());
        assert_eq!(running.next().await, Ok(Step::Aborted));
    }

    #[tokio::test]
    async fn a_job_that_closes_without_an_event_is_a_backend_failure() {
        let bundle = testkit::bundle();
        let generation = GenerationState::new(IncrementalDecoder::new(bundle.tokenizer()));
        let (job, handle) = Job::new(PROMPT.to_vec(), generation).expect("job");
        drop(job);
        let assembler = OutputAssembler::new(Vec::new()).expect("assembler");
        let parser = ReplyParser::new(Start::Content, &[], "call_");
        let mut running = RunningJob::new(handle, assembler, parser, 3, silent());
        let error = running.next().await.expect_err("error");
        assert!(
            matches!(error, ServerError::Eng(EngError::Backend { .. })),
            "{error:?}"
        );
    }

    #[test]
    fn dropping_the_pump_cancels_the_job() {
        // The gate keeps the job queued, so the engine thread cannot touch the cancel flag.
        let (engine, gate) = testkit::gated_engine(testkit::script("ab"));
        let (job, cancel) = run(&engine, &[], 100);
        assert!(!cancel.load(Ordering::SeqCst));
        drop(job);
        assert!(
            cancel.load(Ordering::SeqCst),
            "the drop did not cancel the job"
        );
        drop(gate);
    }

    #[tokio::test]
    async fn text_streamed_before_an_engine_error_is_kept_and_then_the_step_fails() {
        let lost = || EngError::Backend {
            operation: "decode",
            message: "device lost".to_string(),
        };
        let engine = testkit::ending_engine("ab", testkit::Ending::Error(lost()));
        let (mut job, _cancel) = run(&engine, &[], 100);
        let mut text = String::new();
        let failure = loop {
            match job.next().await {
                Ok(Step::Pieces(pieces)) => {
                    for piece in pieces {
                        if let Piece::Content(part) = piece {
                            text.push_str(&part);
                        }
                    }
                }
                other => break other,
            }
        };
        assert_eq!(text, "ab");
        assert_eq!(failure, Err(ServerError::Eng(lost())));
    }

    #[tokio::test]
    async fn errors_after_terminal_batches_are_not_successful_replies() {
        let failure = || EngError::Backend {
            operation: "prefix",
            message: "deferred transfer failed".to_string(),
        };
        for finish in [DecodeFinish::Length, DecodeFinish::Stop] {
            for collected in [false, true] {
                let bundle = testkit::bundle();
                let generation = GenerationState::new(IncrementalDecoder::new(bundle.tokenizer()));
                let (_job, mut handle) = Job::new(PROMPT.to_vec(), generation).expect("job");
                let (sender, receiver) = tokio::sync::mpsc::channel(2);
                handle.events = receiver;
                sender
                    .try_send(JobEvent::Batch(DecodeBatch {
                        tokens: vec![97],
                        text: "a".to_string(),
                        finish: Some(finish),
                    }))
                    .expect("terminal batch");
                sender
                    .try_send(JobEvent::Error(failure()))
                    .expect("deferred failure");
                drop(sender);
                let assembler = OutputAssembler::new(Vec::new()).expect("assembler");
                let parser = ReplyParser::new(Start::Content, &[], "call_");
                let capture = Capture::default();
                let mut log = RequestLog::new(
                    Arc::new(Mutex::new(capture.clone())),
                    default_clock(),
                    "m",
                    !collected,
                    3,
                    "GET",
                    "/v1/test",
                );
                log.admit();
                let mut running = RunningJob::new(handle, assembler, parser, 3, log);
                let result = if collected {
                    collect(&mut running).await.map(|_| ())
                } else {
                    running.next().await.map(|_| ())
                };
                assert_eq!(result, Err(ServerError::Eng(failure())));
                let line: serde_json::Value =
                    serde_json::from_str(capture.text().trim_end()).expect("error log");
                assert_eq!(line["completion_tokens"], 1);
                assert!(line["time_to_first_token_ms"].is_number());
                assert_eq!(line["outcome"], "error");
                assert_eq!(line["status"], if collected { 500 } else { 200 });
            }
        }
    }

    #[tokio::test]
    async fn the_usage_counts_every_token_of_a_batch() {
        let engine = testkit::engine(vec![
            testkit::ids("abc"),
            testkit::ids("de"),
            vec![testkit::EOS],
        ]);
        let (mut job, _cancel) = run(&engine, &[], 100);
        let (parts, finish, usage) = collect(&mut job).await.expect("collect");
        assert_eq!(parts.content, "abcde");
        assert_eq!(finish, FinishReason::Stop);
        assert_eq!(usage, Usage::new(3, 6, 0));
    }

    #[tokio::test]
    async fn the_usage_reports_the_prompt_tokens_that_a_saved_prefix_served() {
        let engine = testkit::engine([testkit::script("a"), testkit::script("b")].concat());
        let mut reused = Vec::new();
        for _ in 0..2 {
            // The first 2 of the 3 prompt tokens are the reusable prefix.
            let (mut job, _cancel) = submit(&engine, &[], 100, Some(2));
            let (_, _, usage) = collect(&mut job).await.expect("collect");
            reused.push(usage.reused_tokens());
        }
        assert_eq!(reused, [0, 2]);
    }

    #[tokio::test]
    async fn the_usage_reports_the_reused_tokens_of_a_reply_that_ended_at_the_job_limit() {
        let engine = testkit::engine(testkit::script("abcdef"));
        let mut found = Vec::new();
        for _ in 0..2 {
            // A budget of 2 tokens ends each reply with `Length` before the script ends.
            let (mut job, _cancel) = submit(&engine, &[], 2, Some(2));
            let (_, finish, usage) = collect(&mut job).await.expect("collect");
            found.push((finish, usage.reused_tokens()));
        }
        assert_eq!(
            found,
            [(FinishReason::Length, 0), (FinishReason::Length, 2)]
        );
    }

    #[tokio::test]
    async fn collect_joins_the_text_and_returns_the_reason_and_usage() {
        let engine = testkit::engine(testkit::script("Hi!"));
        let (mut job, _cancel) = run(&engine, &[], 100);
        let (parts, finish, usage) = collect(&mut job).await.expect("collect");
        assert_eq!(parts.content, "Hi!");
        assert_eq!(parts.reasoning, "");
        assert!(parts.tool_calls.is_empty());
        assert_eq!(finish, FinishReason::Stop);
        assert_eq!(usage, Usage::new(3, 4, 0));
    }

    #[tokio::test]
    async fn collect_reports_a_job_that_the_engine_aborted() {
        let (engine, gate) = testkit::gated_engine(testkit::script("ab"));
        let (mut job, cancel) = run(&engine, &[], 100);
        cancel.store(true, Ordering::SeqCst);
        drop(gate);
        let error = collect(&mut job).await.expect_err("error");
        assert!(
            matches!(error, ServerError::Eng(EngError::Backend { .. })),
            "{error:?}"
        );
    }
}
