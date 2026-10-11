//! Server-sent events for a streaming chat completion.
//!
//! Format: `docs/design/server.md` ("Streaming format"). The response headers and the role chunk
//! go out when the queue admits the job, before prefill starts. A keep-alive comment follows every
//! quiet interval, so that a long prefill does not trip a client timeout. Think text goes to
//! `reasoning_content`. A tool call goes out as two entries when it is complete. The last chunk
//! carries the finish reason and the usage, and a successful stream ends with `[DONE]`.
//! After the headers, an error sends one `event: error` and closes the stream with no usage and no
//! `[DONE]`. A job that the engine aborted closes the stream with no event, as on a disconnect.

use super::error::ServerError;
use super::pieces::Piece;
use super::reply::{RunningJob, Step};
use super::wire::{ChatCompletionChunk, ReplyHead};
use crate::EngError;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use std::convert::Infallible;
use std::time::Duration;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// The role chunk is next.
    Role,
    /// The job is running.
    Running,
    /// The stream has ended.
    Finished,
}

/// The state of one stream between two events.
struct Streaming {
    head: ReplyHead,
    running: RunningJob,
    phase: Phase,
    /// The number of tool calls sent so far.
    calls: u32,
}

/// Builds the streaming response of an admitted job. The stream starts with the role chunk at
/// once. `keep_alive` is the quiet interval after which a comment goes out.
/// Dropping the response drops the job, which cancels it.
pub(crate) fn respond(head: ReplyHead, running: RunningJob, keep_alive: Duration) -> Response {
    let state = Streaming {
        head,
        running,
        phase: Phase::Role,
        calls: 0,
    };
    let events = futures::stream::unfold(state, advance)
        .flat_map(|events| futures::stream::iter(events.into_iter().map(Ok::<Event, Infallible>)));
    Sse::new(events)
        .keep_alive(KeepAlive::new().interval(keep_alive).text("keep-alive"))
        .into_response()
}

/// Produces the events of the next step. It returns no events for a step without news.
async fn advance(mut state: Streaming) -> Option<(Vec<Event>, Streaming)> {
    let events = match state.phase {
        Phase::Finished => return None,
        Phase::Role => {
            state.phase = Phase::Running;
            vec![chunk_event(&ChatCompletionChunk::role(&state.head))]
        }
        Phase::Running => match state.running.next().await {
            Ok(Step::Pieces(pieces)) => piece_events(&mut state, pieces),
            Ok(Step::Done {
                pieces,
                finish,
                usage,
            }) => {
                state.phase = Phase::Finished;
                let mut events = piece_events(&mut state, pieces);
                let last = ChatCompletionChunk::last(&state.head, finish, usage);
                events.push(chunk_event(&last));
                events.push(Event::default().data("[DONE]"));
                events
            }
            Ok(Step::Aborted) => {
                state.phase = Phase::Finished;
                Vec::new()
            }
            Err(error) => {
                state.phase = Phase::Finished;
                vec![error_event(&error)]
            }
        },
    };
    Some((events, state))
}

/// Turns reply parts into chunks. A tool call becomes a start entry and an arguments entry.
fn piece_events(state: &mut Streaming, pieces: Vec<Piece>) -> Vec<Event> {
    let mut events = Vec::new();
    for piece in pieces {
        match piece {
            Piece::Content(text) => {
                events.push(chunk_event(&ChatCompletionChunk::content(
                    &state.head,
                    text,
                )));
            }
            Piece::Reasoning(text) => {
                events.push(chunk_event(&ChatCompletionChunk::reasoning(
                    &state.head,
                    text,
                )));
            }
            Piece::Call(call) => {
                let index = state.calls;
                state.calls += 1;
                let start =
                    ChatCompletionChunk::tool_call_start(&state.head, index, call.id, call.name);
                let arguments =
                    ChatCompletionChunk::tool_call_arguments(&state.head, index, call.arguments);
                events.push(chunk_event(&start));
                events.push(chunk_event(&arguments));
            }
        }
    }
    events
}

fn chunk_event(chunk: &ChatCompletionChunk) -> Event {
    Event::default().json_data(chunk).unwrap_or_else(|error| {
        error_event(&ServerError::Eng(EngError::Backend {
            operation: "serialize",
            message: error.to_string(),
        }))
    })
}

fn error_event(error: &ServerError) -> Event {
    error.log();
    Event::default()
        .event("error")
        .data(error.body().to_string())
}

#[cfg(test)]
mod tests {
    use crate::EngError;
    use crate::sched::worker::{Engine, EngineState};
    use crate::server::contract::assert_schema;
    use crate::server::router::router;
    use crate::server::state::AppState;
    use crate::server::testkit::{self, Ending, SseEvent};
    use axum::Router;
    use axum::http::{StatusCode, header};
    use futures::StreamExt;
    use serde_json::{Value, json};
    use std::sync::Arc;
    use std::time::Duration;

    const CALL: &str = "<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>";

    fn app_with_keep_alive(engine: &Engine, keep_alive: Duration) -> Router {
        let entry = testkit::entry("e1", "model-a", engine, 4_096);
        let state = AppState::new(testkit::API_KEY, vec![entry]).with_keep_alive(keep_alive);
        router(Arc::new(state))
    }

    fn app(engine: &Engine) -> Router {
        testkit::wait_for_state(&engine.handle(), EngineState::Ready);
        app_with_keep_alive(engine, Duration::from_secs(15))
    }

    fn stream_request(extra: Value) -> Value {
        let mut body = json!({
            "model": "model-a",
            "stream": true,
            "tools": [{"type": "function", "function": {
                "name": "add",
                "parameters": {"type": "object", "properties": {
                    "a": {"type": "integer"}, "b": {"type": "number"}}}}}],
            "messages": [{"role": "user", "content": "hi"}]
        });
        for (key, value) in extra.as_object().expect("object").iter() {
            body[key] = value.clone();
        }
        body
    }

    async fn stream(app: &Router, body: &Value) -> (StatusCode, String, Vec<SseEvent>) {
        let (status, headers, bytes) =
            testkit::send(app, testkit::post_json("/v1/chat/completions", body)).await;
        let content_type = headers[header::CONTENT_TYPE]
            .to_str()
            .expect("type")
            .to_string();
        (status, content_type, testkit::sse_events(&bytes))
    }

    fn chunk(event: &SseEvent) -> Value {
        assert_eq!(event.name, None, "{event:?}");
        serde_json::from_str(&event.data).expect("chunk json")
    }

    /// The `delta` objects of the data events, without the final `[DONE]`.
    fn deltas(events: &[SseEvent]) -> Vec<Value> {
        events
            .iter()
            .filter(|event| event.data != "[DONE]" && event.name.is_none())
            .map(|event| chunk(event)["choices"][0]["delta"].clone())
            .collect()
    }

    fn joined(deltas: &[Value], field: &str) -> String {
        deltas
            .iter()
            .filter_map(|delta| delta[field].as_str())
            .collect()
    }

    #[tokio::test]
    async fn stream_ends_with_usage_event_then_done() {
        let engine = testkit::engine(testkit::script("Hi!"));
        let (status, content_type, events) =
            stream(&app(&engine), &stream_request(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, "text/event-stream");
        assert_eq!(events.len(), 6, "{events:?}");
        assert_eq!(events[5].data, "[DONE]");
        let texts: Vec<Value> = events[1..4]
            .iter()
            .map(|event| chunk(event)["choices"][0]["delta"]["content"].clone())
            .collect();
        assert_eq!(texts, [json!("H"), json!("i"), json!("!")]);
        let last = chunk(&events[4]);
        assert_eq!(last["choices"][0]["finish_reason"], "stop");
        assert_eq!(last["choices"][0]["delta"], json!({}));
        assert_eq!(last["usage"]["prompt_tokens"], 21);
        assert_eq!(last["usage"]["completion_tokens"], 4);
        assert_eq!(last["usage"]["total_tokens"], 25);
        for event in &events[..5] {
            assert_schema("ChatCompletionChunk", &chunk(event));
        }
    }

    #[tokio::test]
    async fn only_the_first_chunk_carries_the_role_and_only_the_last_carries_usage() {
        let engine = testkit::engine(testkit::script("abc"));
        let (_, _, events) = stream(&app(&engine), &stream_request(json!({}))).await;
        let chunks: Vec<Value> = events[..events.len() - 1].iter().map(chunk).collect();
        assert_eq!(
            chunks[0]["choices"][0]["delta"],
            json!({"role": "assistant", "content": ""})
        );
        for later in &chunks[1..] {
            assert!(
                later["choices"][0]["delta"].get("role").is_none(),
                "{later}"
            );
        }
        for early in &chunks[..chunks.len() - 1] {
            assert!(early.get("usage").is_none(), "{early}");
            assert!(
                early["choices"][0].get("finish_reason").is_none(),
                "{early}"
            );
        }
        assert!(
            chunks[0]["id"]
                .as_str()
                .expect("id")
                .starts_with("chatcmpl-")
        );
        assert_eq!(chunks[0]["id"], chunks[3]["id"]);
        assert_eq!(chunks[0]["model"], "model-a");
    }

    #[tokio::test]
    async fn the_usage_event_of_a_stream_reports_the_reused_tokens() {
        let engine = testkit::engine([testkit::script("a"), testkit::script("b")].concat());
        testkit::wait_for_state(&engine.handle(), EngineState::Ready);
        let mut entry = testkit::entry("e1", "model-a", &engine, 4_096);
        entry.prefix_reuse = true;
        let app = router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])));
        let mut reused = Vec::new();
        for messages in [
            json!([{"role": "user", "content": "hi"}]),
            json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "more"},
            ]),
        ] {
            let body = stream_request(json!({"messages": messages}));
            let (_, _, events) = stream(&app, &body).await;
            let usage = chunk(&events[events.len() - 2])["usage"].clone();
            reused.push(usage["reused_tokens"].clone());
        }
        // The key of the first prompt is `<|im_start|>user\nhi<|im_end|>\n`: 10 ids.
        assert_eq!(reused, [json!(0), json!(10)]);
    }

    #[tokio::test]
    async fn a_stop_string_ends_the_stream_and_its_text_never_reaches_the_client() {
        let engine = testkit::engine(testkit::script("one END two"));
        let body = stream_request(json!({"stop": ["END"]}));
        let (_, _, events) = stream(&app(&engine), &body).await;
        let deltas = deltas(&events);
        assert_eq!(joined(&deltas, "content"), "one");
        for delta in &deltas[1..] {
            assert!(delta["content"] != "", "an empty content chunk: {delta}");
        }
        let last = chunk(&events[events.len() - 2]);
        assert_eq!(last["choices"][0]["finish_reason"], "stop");
    }

    #[tokio::test]
    async fn max_tokens_ends_the_stream_with_length() {
        let engine = testkit::engine(testkit::script("abcdef"));
        let body = stream_request(json!({"max_tokens": 2}));
        let (_, _, events) = stream(&app(&engine), &body).await;
        let last = chunk(&events[events.len() - 2]);
        assert_eq!(last["choices"][0]["finish_reason"], "length");
        assert_eq!(last["usage"]["completion_tokens"], 2);
        assert_eq!(events[events.len() - 1].data, "[DONE]");
    }

    #[tokio::test]
    async fn think_text_goes_to_reasoning_content_and_never_to_content() {
        let engine = testkit::engine(testkit::script("<think>plan</think>Done."));
        let (_, _, events) = stream(&app(&engine), &stream_request(json!({}))).await;
        let deltas = deltas(&events);
        assert_eq!(joined(&deltas, "reasoning_content"), "plan");
        assert_eq!(joined(&deltas, "content"), "Done.");
        let first_content = deltas
            .iter()
            .position(|d| d["content"] == "D")
            .expect("content");
        let last_reasoning = deltas
            .iter()
            .rposition(|delta| delta.get("reasoning_content").is_some())
            .expect("reasoning");
        assert!(last_reasoning < first_content, "reasoning must come first");
    }

    #[tokio::test]
    async fn a_complete_tool_call_streams_as_two_entries_and_finishes_with_tool_calls() {
        let text = format!("Adding.{CALL}");
        let engine = testkit::engine(testkit::script(&text));
        let (_, _, events) = stream(&app(&engine), &stream_request(json!({}))).await;
        let deltas = deltas(&events);
        assert_eq!(joined(&deltas, "content"), "Adding.");
        let calls: Vec<&Value> = deltas
            .iter()
            .filter(|delta| delta.get("tool_calls").is_some())
            .collect();
        assert_eq!(calls.len(), 2, "{deltas:?}");
        assert_eq!(
            calls[0]["tool_calls"],
            json!([{"index": 0, "id": "call_0000000000000001_0", "type": "function",
                    "function": {"name": "add", "arguments": ""}}])
        );
        assert_eq!(
            calls[1]["tool_calls"],
            json!([{"index": 0, "function": {"arguments": "{\"a\":1,\"b\":2}"}}])
        );
        let all = events
            .iter()
            .map(|event| event.data.as_str())
            .collect::<String>();
        assert!(!all.contains("tool_call>"), "markup leaked: {all}");
        let last = chunk(&events[events.len() - 2]);
        assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
        assert_schema("ChatCompletionChunk", &last);
    }

    #[tokio::test]
    async fn two_tool_calls_carry_the_indices_0_0_1_1() {
        let text = format!("{CALL}{CALL}");
        let engine = testkit::engine(testkit::script(&text));
        let (_, _, events) = stream(&app(&engine), &stream_request(json!({}))).await;
        let indices: Vec<Value> = deltas(&events)
            .iter()
            .filter_map(|delta| delta.get("tool_calls"))
            .map(|entries| entries[0]["index"].clone())
            .collect();
        assert_eq!(indices, [json!(0), json!(0), json!(1), json!(1)]);
        let last = chunk(&events[events.len() - 2]);
        assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn text_that_a_stop_string_might_still_match_is_sent_when_the_reply_ends() {
        // "EN" is a prefix of the stop string "END", so the parser holds it back until the end.
        let engine = testkit::engine(testkit::script("one EN"));
        let body = stream_request(json!({"stop": ["END"]}));
        let (_, _, events) = stream(&app(&engine), &body).await;
        let deltas = deltas(&events);
        assert_eq!(joined(&deltas, "content"), "one EN");
        let last = chunk(&events[events.len() - 2]);
        assert_eq!(last["choices"][0]["finish_reason"], "stop");
        assert_eq!(events[events.len() - 1].data, "[DONE]");
    }

    #[tokio::test]
    async fn a_stream_request_to_a_stopped_engine_returns_503_json_not_a_stream() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine);
        engine.shutdown();
        let (status, headers, bytes) = testkit::send(
            &app,
            testkit::post_json("/v1/chat/completions", &stream_request(json!({}))),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(headers[header::CONTENT_TYPE], "application/json");
        let body = testkit::json_of(&bytes);
        assert_schema("Error", &body);
        assert_eq!(body["error"]["code"], "overloaded");
    }

    #[tokio::test]
    async fn a_malformed_call_streams_as_text_and_the_reason_stays_stop() {
        let broken = "<tool_call><function=add><parameter=a>1</parameter></tool_call>";
        let engine = testkit::engine(testkit::script(broken));
        let (_, _, events) = stream(&app(&engine), &stream_request(json!({}))).await;
        let deltas = deltas(&events);
        assert_eq!(joined(&deltas, "content"), broken);
        let last = chunk(&events[events.len() - 2]);
        assert_eq!(last["choices"][0]["finish_reason"], "stop");
    }

    #[tokio::test]
    async fn pre_header_error_returns_json_even_when_stream_true() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine);
        let unknown = stream_request(json!({"model": "other"}));
        let (status, headers, bytes) =
            testkit::send(&app, testkit::post_json("/v1/chat/completions", &unknown)).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(headers[header::CONTENT_TYPE], "application/json");
        assert_schema("Error", &testkit::json_of(&bytes));
        let invalid = stream_request(json!({"max_tokens": 0}));
        let (status, headers, _) =
            testkit::send(&app, testkit::post_json("/v1/chat/completions", &invalid)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(headers[header::CONTENT_TYPE], "application/json");
    }

    #[tokio::test]
    async fn a_failure_during_prefill_is_an_error_event_after_the_role_chunk() {
        let engine = testkit::refusing_engine(EngError::MemoryPlan("no room".to_string()));
        let (status, content_type, events) =
            stream(&app(&engine), &stream_request(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, "text/event-stream");
        assert_eq!(events.len(), 2, "{events:?}");
        assert_eq!(
            chunk(&events[0])["choices"][0]["delta"]["role"],
            "assistant"
        );
        assert_eq!(events[1].name.as_deref(), Some("error"));
        let body: Value = serde_json::from_str(&events[1].data).expect("error json");
        assert_schema("Error", &body);
        assert_eq!(body["error"]["code"], "memory_plan");
    }

    #[tokio::test]
    async fn error_after_headers_emits_one_error_event_and_closes() {
        let failure = EngError::Backend {
            operation: "qmm",
            message: "Metal command buffer 7 failed".to_string(),
        };
        let engine = testkit::ending_engine("ab", Ending::Error(failure));
        let (status, _, events) = stream(&app(&engine), &stream_request(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(events.len(), 4, "{events:?}");
        assert_eq!(chunk(&events[1])["choices"][0]["delta"]["content"], "a");
        assert_eq!(chunk(&events[2])["choices"][0]["delta"]["content"], "b");
        assert_eq!(events[3].name.as_deref(), Some("error"));
        let body: Value = serde_json::from_str(&events[3].data).expect("error json");
        assert_schema("BackendError", &body);
        assert!(!events[3].data.contains("Metal"), "{:?}", events[3]);
        assert!(events.iter().all(|event| event.data != "[DONE]"));
        assert!(events.iter().all(|event| !event.data.contains("usage")));
    }

    #[tokio::test]
    async fn a_job_that_the_engine_aborts_closes_the_stream_without_done_or_error() {
        let engine = testkit::ending_engine("ab", Ending::Cancel);
        let (status, _, events) = stream(&app(&engine), &stream_request(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(events.len(), 3, "{events:?}");
        assert!(events.iter().all(|event| event.name.is_none()));
        assert!(events.iter().all(|event| event.data != "[DONE]"));
    }

    #[tokio::test]
    async fn the_headers_and_the_role_chunk_go_out_before_prefill_starts() {
        let (engine, gate) = testkit::gated_engine(testkit::script("ok"));
        let app = app_with_keep_alive(&engine, Duration::from_secs(15));
        let request = testkit::post_json("/v1/chat/completions", &stream_request(json!({})));
        let response = tokio::time::timeout(
            Duration::from_secs(5),
            tower::ServiceExt::oneshot(app, request),
        )
        .await
        .expect("the headers before prefill")
        .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        assert_eq!(engine.handle().state(), EngineState::Loading);
        let mut frames = response.into_body().into_data_stream();
        let first = tokio::time::timeout(Duration::from_secs(5), frames.next())
            .await
            .expect("the role chunk before prefill")
            .expect("a frame")
            .expect("bytes");
        let text = String::from_utf8(first.to_vec()).expect("utf-8");
        assert!(text.starts_with("data: {"), "{text}");
        assert!(text.contains(r#""role":"assistant""#), "{text}");
        drop(gate);
        let mut rest = String::new();
        while let Some(frame) = frames.next().await {
            rest.push_str(std::str::from_utf8(&frame.expect("bytes")).expect("utf-8"));
        }
        assert!(rest.ends_with("data: [DONE]\n\n"), "{rest}");
    }

    #[tokio::test]
    async fn a_keep_alive_comment_follows_a_quiet_interval() {
        let (engine, gate) = testkit::gated_engine(testkit::script("ok"));
        let app = app_with_keep_alive(&engine, Duration::from_millis(40));
        let request = testkit::post_json("/v1/chat/completions", &stream_request(json!({})));
        let response = tokio::time::timeout(
            Duration::from_secs(5),
            tower::ServiceExt::oneshot(app, request),
        )
        .await
        .expect("the headers before prefill")
        .expect("response");
        let mut frames = response.into_body().into_data_stream();
        let _role = frames.next().await.expect("the role chunk").expect("bytes");
        let comment = tokio::time::timeout(Duration::from_secs(5), frames.next())
            .await
            .expect("a keep-alive within five seconds")
            .expect("a frame")
            .expect("bytes");
        assert_eq!(comment.as_ref(), b": keep-alive\n\n");
        drop(gate);
    }
}
