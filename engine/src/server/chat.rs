//! The chat completion handler: `POST /v1/chat/completions`.
//!
//! Request flow: the server design ("Request lifecycle"). The handler reads the body, validates
//! it, picks the engine, renders and tokenizes the prompt, submits the job, and writes the reply.
//! The model runs on the engine thread. This module never touches model state.

use super::error::ServerError;
use super::messages::bad;
use super::pieces::ReplyParser;
use super::reply::{RunningJob, collect};
use super::reqlog::{RequestLog, default_clock};
use super::sse;
use super::state::{AppState, unix_seconds};
use super::validate::{ChatInput, parse_chat};
use super::wire::{ChatCompletion, ReplyHead};
use crate::EngError;
use crate::family::executor::GenerationState;
use crate::family::sampler::{Sampler, SamplingParams};
use crate::sched::job::Job;
use crate::sched::worker::SubmitError;
use crate::tokenizer::toolcall::Start;
use crate::tokenizer::{IncrementalDecoder, OutputAssembler, PromptBoundaries, TokenizerBundle};
use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

/// The body cap of the contract: 16 MiB.
pub(crate) const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

/// A request that passed every check and whose job the engine queue holds.
struct Submitted {
    head: ReplyHead,
    running: RunningJob,
}

/// The route handler. Every failure before the reply is a JSON error.
pub(crate) async fn chat(State(state): State<Arc<AppState>>, body: Body) -> Response {
    match reply(&state, body).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

async fn reply(state: &AppState, body: Body) -> Result<Response, ServerError> {
    let bytes = axum::body::to_bytes(body, MAX_BODY_BYTES)
        .await
        .map_err(|_| bad("the request body is larger than 16 MiB or cannot be read"))?;
    let input = parse_chat(&bytes)?;
    // The stream flag is read before `submit` consumes the input. The headers go out at admission.
    let stream = input.stream;
    let mut submitted = submit(state, input).await?;
    if stream {
        let keep_alive = state.keep_alive();
        return Ok(sse::respond(submitted.head, submitted.running, keep_alive));
    }
    let (parts, finish, usage) = collect(&mut submitted.running).await?;
    let completion = ChatCompletion::new(&submitted.head, parts, finish, usage);
    Ok(Json(completion).into_response())
}

/// Resolves the engine, builds the prompt, and queues the job.
async fn submit(state: &AppState, input: ChatInput) -> Result<Submitted, ServerError> {
    let stream = input.stream;
    let entry = state.resolve(
        input.session.as_deref(),
        input.engine.as_deref(),
        input.model.as_deref(),
    )?;
    // The engine entry owns the defaults: the values of its `generation_config.json`. A field
    // the request set keeps its value, so an explicit temperature 0 stays greedy. Note that
    // `parse_chat` checked the bounds of the request fields against `SamplingDefaults::CONTRACT`
    // only, because no engine was resolved yet; this re-resolution, at submit, applies the
    // defaults of the engine entry that will actually serve the request.
    let sampling = SamplingParams::from_request(
        input.sampling.temperature,
        input.sampling.top_p,
        input.sampling.top_k,
        &entry.defaults,
    )?;
    let assembler = OutputAssembler::new(input.stops)?;
    let bundle = Arc::clone(&entry.bundle);
    let (messages, thinking) = (input.messages, input.thinking);
    let template_tools = input.tools.clone();
    let prefix_reuse = entry.prefix_reuse;
    let (prompt, boundaries, start) = tokio::task::spawn_blocking(move || {
        // Without the flag the handler marks no boundary, so the engine neither saves nor restores.
        let (prompt, boundaries) = if prefix_reuse {
            bundle.encode_prompt_with_boundary(&messages, &template_tools, thinking)?
        } else {
            (
                bundle.encode_prompt(&messages, &template_tools, thinking)?,
                PromptBoundaries::default(),
            )
        };
        let start = reply_start(&bundle, &prompt);
        Ok::<_, EngError>((prompt, boundaries, start))
    })
    .await
    .map_err(|error| {
        ServerError::Eng(EngError::Backend {
            operation: "tokenize",
            message: error.to_string(),
        })
    })??;
    let prompt_tokens = u32::try_from(prompt.len()).unwrap_or(u32::MAX);
    // From here on, every end of the request writes one line of the request log.
    let mut log = RequestLog::new(
        state.request_log(),
        default_clock(),
        &entry.model,
        stream,
        prompt_tokens,
    );
    if prompt_tokens >= entry.context_limit {
        return Err(failed(
            &mut log,
            EngError::ContextLimit(format!(
                "the prompt has {prompt_tokens} tokens and the context limit is {} tokens",
                entry.context_limit
            )),
        ));
    }
    // The executor enforces this limit. The server counts no tokens.
    let max_tokens = input.max_tokens.min(entry.context_limit - prompt_tokens);
    let head = ReplyHead {
        id: state.next_completion_id(),
        created: unix_seconds(),
        model: entry.model.clone(),
        engine: entry.id.clone(),
    };
    log.set_id(&head.id);
    let parser = ReplyParser::new(start, &input.tools, &head.call_prefix());
    let decoder = IncrementalDecoder::new(entry.bundle.tokenizer());
    let sampler = Sampler::from_entropy(sampling);
    let generation = GenerationState::with_sampler(decoder, sampler).with_max_tokens(max_tokens);
    let (mut job, handle) =
        Job::new(prompt, generation).map_err(|error| failed(&mut log, error))?;
    if let Some(boundary) = boundaries.tip {
        // The tokenizer returns a boundary inside the prompt, so a refusal here is a defect.
        job = job.with_prefix_boundary(boundary).map_err(|error| {
            failed(
                &mut log,
                EngError::Backend {
                    operation: "prefix_boundary",
                    message: format!("{error:?}"),
                },
            )
        })?;
        // The tokenizer puts the root inside the tip, so a refusal here is a defect, too.
        if let Some(root) = boundaries.root {
            job = job.with_root_boundary(root).map_err(|error| {
                failed(
                    &mut log,
                    EngError::Backend {
                        operation: "root_boundary",
                        message: format!("{error:?}"),
                    },
                )
            })?;
        }
        // The line of this request carries the prefix fields (packet SERVE-2-c item 4); a
        // request without a boundary keeps the old line byte for byte.
        log.set_prefix_boundary();
    }
    entry
        .handle
        .try_submit(job)
        .map_err(refusal)
        .map_err(|error| failed(&mut log, error))?;
    log.admit();
    Ok(Submitted {
        head,
        running: RunningJob::new(handle, assembler, parser, prompt_tokens, log),
    })
}

/// Writes the request log line of a request that failed after its prompt was known.
/// It returns the error for the caller to return.
fn failed(log: &mut RequestLog, error: impl Into<ServerError>) -> ServerError {
    let error = error.into();
    log.fail(&error);
    error
}

/// Chooses where the parser starts in the reply. When the prompt ends inside an open think block,
/// the template asked for thinking and the reply starts inside the reasoning. The template is the
/// one authority: the check reads the end of the rendered prompt, not the request setting.
fn reply_start(bundle: &TokenizerBundle, prompt: &[u32]) -> Start {
    const OPEN: &str = "<think>\n";
    // Every token decodes to at least one byte, so this many tokens cover the marker.
    let tail = &prompt[prompt.len().saturating_sub(OPEN.len())..];
    match bundle.tokenizer().decode(tail) {
        Ok(text) if text.ends_with(OPEN) => Start::Reasoning,
        _ => Start::Content,
    }
}

/// Maps a refused submission to its HTTP error.
fn refusal(error: SubmitError) -> ServerError {
    match error {
        SubmitError::QueueFull => ServerError::Overloaded("the engine queue is full".to_string()),
        SubmitError::Unavailable => {
            ServerError::Overloaded("the engine is not available".to_string())
        }
        SubmitError::NativeFailed => ServerError::Eng(EngError::Backend {
            operation: "engine",
            message: "a native failure stopped the engine".to_string(),
        }),
        SubmitError::LoadFailed => ServerError::Eng(EngError::Backend {
            operation: "engine",
            message: "the engine failed to load".to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use crate::EngError;
    use crate::family::sampler::{SamplingDefaults, SamplingParams};
    use crate::sched::job::Job;
    use crate::sched::worker::{Engine, EngineState};
    use crate::server::contract::{assert_pointer, assert_schema};
    use crate::server::router::router;
    use crate::server::state::AppState;
    use crate::server::testkit;
    use crate::tokenizer::TokenizerBundle;
    use axum::Router;
    use axum::http::StatusCode;
    use serde_json::{Value, json};
    use std::sync::Arc;

    /// The body cap of the contract, written out so that a change of the constant shows.
    const SIXTEEN_MIB: usize = 16 * 1024 * 1024;

    const COMPLETION: &str =
        "/paths/~1v1~1chat~1completions/post/responses/200/content/application~1json/schema";

    /// A router over one engine named `e1` that serves `model-a`.
    fn app(engine: &Engine, context_limit: u32) -> Router {
        testkit::wait_for_state(&engine.handle(), EngineState::Ready);
        let entry = testkit::entry("e1", "model-a", engine, context_limit);
        router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])))
    }

    /// A router over one engine entry that carries its own sampling defaults.
    fn app_with_defaults(engine: &Engine, defaults: SamplingDefaults) -> Router {
        testkit::wait_for_state(&engine.handle(), EngineState::Ready);
        let entry = testkit::entry_with_defaults("e1", "model-a", engine, 4_096, defaults);
        router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])))
    }

    /// The sampling defaults of a model directory whose `generation_config.json` holds `text`.
    fn file_defaults(text: &str) -> SamplingDefaults {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(dir.path().join("generation_config.json"), text)
            .expect("write generation_config.json");
        SamplingDefaults::from_model_dir(dir.path())
            .expect("defaults")
            .values
    }

    fn request(extra: Value) -> Value {
        let mut body = json!({"model": "model-a", "messages": [{"role": "user", "content": "hi"}]});
        for (key, value) in extra.as_object().expect("object").iter() {
            body[key] = value.clone();
        }
        body
    }

    async fn post(app: &Router, body: &Value) -> (StatusCode, Value) {
        let (status, _headers, bytes) =
            testkit::send(app, testkit::post_json("/v1/chat/completions", body)).await;
        // A body-less reply gives `Null`, so that a wrong status fails at its own assertion.
        let body = if bytes.is_empty() {
            Value::Null
        } else {
            testkit::json_of(&bytes)
        };
        (status, body)
    }

    fn error_code(body: &Value) -> &str {
        body["error"]["code"].as_str().expect("error code")
    }

    #[tokio::test]
    async fn non_streaming_reply_reports_usage_with_the_eos_token() {
        let engine = testkit::engine(testkit::script("Hello!"));
        let (status, body) = post(&app(&engine, 4_096), &request(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_pointer(COMPLETION, &body);
        assert_eq!(body["choices"][0]["message"]["role"], "assistant");
        assert_eq!(body["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["usage"]["prompt_tokens"], 21);
        // "Hello!" is six tokens. The count includes the EOS token.
        assert_eq!(body["usage"]["completion_tokens"], 7);
        assert_eq!(body["usage"]["total_tokens"], 28);
    }

    #[tokio::test]
    async fn the_reply_names_the_model_the_engine_and_a_fresh_id() {
        let engine = testkit::engine([testkit::script("a"), testkit::script("b")].concat());
        let app = app(&engine, 4_096);
        let (_, first) = post(&app, &request(json!({}))).await;
        let (_, second) = post(&app, &request(json!({}))).await;
        assert_eq!(first["model"], "model-a");
        assert_eq!(first["engine"], "e1");
        assert_eq!(first["object"], "chat.completion");
        assert_ne!(first["id"], second["id"]);
        assert!(first["created"].as_u64().expect("created") > 1_700_000_000);
    }

    #[tokio::test]
    async fn the_engine_receives_the_rendered_and_tokenized_prompt() {
        let (engine, seen) = testkit::probe_engine();
        let app = app(&engine, 4_096);
        post(&app, &request(json!({"max_tokens": 1}))).await;
        let mut expected = vec![257];
        expected.extend(testkit::ids("user\nhi"));
        expected.push(256);
        expected.extend(testkit::ids("\n"));
        expected.push(257);
        expected.extend(testkit::ids("assistant\n"));
        assert_eq!(*seen.lock().expect("prompt lock"), expected);
    }

    #[tokio::test]
    async fn a_developer_message_and_array_content_reach_the_engine_prompt() {
        let (engine, seen) = testkit::probe_engine();
        let app = app(&engine, 4_096);
        let body = json!({
            "model": "model-a",
            "messages": [
                {"role": "developer", "content": [
                    {"type": "text", "text": "Answer "},
                    {"type": "text", "text": "briefly."}
                ]},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]}
            ],
            "max_tokens": 1
        });
        let (status, reply) = post(&app, &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        let mut expected = vec![257];
        expected.extend(testkit::ids("system\nAnswer briefly."));
        expected.push(256);
        expected.extend(testkit::ids("\n"));
        expected.push(257);
        expected.extend(testkit::ids("user\nhi"));
        expected.push(256);
        expected.extend(testkit::ids("\n"));
        expected.push(257);
        expected.extend(testkit::ids("assistant\n"));
        assert_eq!(*seen.lock().expect("prompt lock"), expected);
    }

    #[tokio::test]
    async fn max_tokens_ends_the_reply_with_length() {
        let engine = testkit::engine(testkit::script("abcdefgh"));
        let (status, body) = post(&app(&engine, 4_096), &request(json!({"max_tokens": 3}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["choices"][0]["message"]["content"], "abc");
        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert_eq!(body["usage"]["completion_tokens"], 3);
    }

    #[tokio::test]
    async fn the_context_limit_caps_the_reply_with_length() {
        let engine = testkit::engine(testkit::script("abcdefgh"));
        let (status, body) = post(&app(&engine, 24), &request(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["choices"][0]["message"]["content"], "abc");
        assert_eq!(body["choices"][0]["finish_reason"], "length");
    }

    #[tokio::test]
    async fn a_stop_string_ends_the_reply_and_the_stop_text_stays_hidden() {
        let engine = testkit::engine(testkit::script("Hello END tail"));
        let (status, body) = post(&app(&engine, 4_096), &request(json!({"stop": ["END"]}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["choices"][0]["message"]["content"], "Hello");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["usage"]["completion_tokens"], 9);
    }

    const CALL: &str = "<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>";

    fn add_tool() -> Value {
        json!([{"type": "function", "function": {
            "name": "add",
            "parameters": {"type": "object", "properties": {
                "a": {"type": "integer"}, "b": {"type": "number"}}}}}])
    }

    #[tokio::test]
    async fn think_text_and_a_tool_call_are_split_in_a_non_streaming_reply() {
        let text = format!("<think>plan</think>Adding.{CALL}");
        let engine = testkit::engine(testkit::script(&text));
        let body = request(json!({"tools": add_tool()}));
        let (status, reply) = post(&app(&engine, 4_096), &body).await;
        assert_eq!(status, StatusCode::OK);
        assert_pointer(COMPLETION, &reply);
        let choice = &reply["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls");
        assert_eq!(choice["message"]["reasoning_content"], "plan");
        assert_eq!(choice["message"]["content"], "Adding.");
        assert_eq!(
            choice["message"]["tool_calls"],
            json!([{"id": "call_0000000000000001_0", "type": "function",
                    "function": {"name": "add", "arguments": "{\"a\":1,\"b\":2}"}}])
        );
    }

    #[tokio::test]
    async fn a_reply_with_only_a_tool_call_has_null_content() {
        let engine = testkit::engine(testkit::script(CALL));
        let body = request(json!({"tools": add_tool()}));
        let (_, reply) = post(&app(&engine, 4_096), &body).await;
        assert_eq!(reply["choices"][0]["message"]["content"], Value::Null);
        assert_pointer(COMPLETION, &reply);
    }

    #[tokio::test]
    async fn a_request_without_tools_keeps_a_call_as_text() {
        let engine = testkit::engine(testkit::script(CALL));
        let (status, reply) = post(&app(&engine, 4_096), &request(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(reply["choices"][0]["message"]["content"], CALL);
        assert_eq!(reply["choices"][0]["finish_reason"], "stop");
        assert!(reply["choices"][0]["message"].get("tool_calls").is_none());
    }

    #[tokio::test]
    async fn tool_choice_none_keeps_a_call_as_text() {
        let engine = testkit::engine(testkit::script(CALL));
        let body = request(json!({"tools": add_tool(), "tool_choice": "none"}));
        let (_, reply) = post(&app(&engine, 4_096), &body).await;
        assert_eq!(reply["choices"][0]["message"]["content"], CALL);
        assert_eq!(reply["choices"][0]["finish_reason"], "stop");
    }

    #[tokio::test]
    async fn an_empty_stop_string_returns_400() {
        let engine = testkit::engine(Vec::new());
        let (status, body) = post(&app(&engine, 4_096), &request(json!({"stop": [""]}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error_code(&body), "bad_request");
    }

    #[tokio::test]
    async fn temperature_zero_samples_the_largest_logit_every_time() {
        let (engine, _seen) = testkit::probe_engine();
        let body = request(json!({"temperature": 0, "max_tokens": 50}));
        let (_, reply) = post(&app(&engine, 4_096), &body).await;
        let text = reply["choices"][0]["message"]["content"]
            .as_str()
            .expect("content");
        assert_eq!(text.chars().count(), 50);
        assert!(
            text.chars().all(|c| c == '\u{1}'),
            "greedy must pick id 1 only"
        );
    }

    #[tokio::test]
    async fn an_omitted_temperature_takes_the_engine_default_through_the_chat_path() {
        // The model directory sets temperature 0, so a request that omits the field must reach
        // the job's sampler as greedy: the probe's two logits then always draw id 1.
        let defaults = file_defaults("{\"temperature\": 0.0}");
        assert_eq!(defaults.temperature, 0.0);
        let (engine, _seen) = testkit::probe_engine();
        let body = request(json!({"max_tokens": 200}));
        let (status, reply) = post(&app_with_defaults(&engine, defaults), &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        let text = reply["choices"][0]["message"]["content"]
            .as_str()
            .expect("content");
        assert_eq!(text.chars().count(), 200);
        assert!(
            text.chars().all(|c| c == '\u{1}'),
            "the engine default temperature must reach the sampler, text was all-{:?}",
            text.chars().take(4).collect::<String>()
        );
    }

    #[tokio::test]
    async fn a_set_temperature_overrides_a_greedy_engine_default() {
        // The engine default is greedy, but the request sets temperature 2, so the sampler draws
        // both ids of the probe's two-logit row instead of always picking id 1.
        let defaults = file_defaults("{\"temperature\": 0.0}");
        let (engine, _seen) = testkit::probe_engine();
        let body = request(json!({"temperature": 2, "max_tokens": 200}));
        let (status, reply) = post(&app_with_defaults(&engine, defaults), &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        let text = reply["choices"][0]["message"]["content"]
            .as_str()
            .expect("content");
        assert!(text.contains('\u{0}'), "both ids must appear, got {text:?}");
        assert!(text.contains('\u{1}'), "both ids must appear, got {text:?}");
    }

    #[tokio::test]
    async fn an_explicit_top_k_zero_overrides_the_file_default_through_the_chat_path() {
        // The file sets top_k 20; the request turns the filter off. The probe records the
        // settings the chat handler resolved, so the observed sampler must hold top_k 0 with
        // the file's temperature 1.0 and top_p 0.95.
        let defaults = file_defaults(r#"{"temperature": 1.0, "top_p": 0.95, "top_k": 20}"#);
        let probes = testkit::probe_engine_with_sampling();
        let body = request(json!({"top_k": 0, "max_tokens": 1}));
        let (status, reply) = post(&app_with_defaults(&probes.engine, defaults), &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        let observed = *probes.sampling.lock().expect("sampling lock");
        assert_eq!(
            observed,
            Some(SamplingParams::new(1.0, 0.95, 0).expect("expected params"))
        );
    }

    #[tokio::test]
    async fn temperature_two_samples_both_tokens() {
        let (engine, _seen) = testkit::probe_engine();
        let body = request(json!({"temperature": 2, "max_tokens": 200}));
        let (_, reply) = post(&app(&engine, 4_096), &body).await;
        let text = reply["choices"][0]["message"]["content"]
            .as_str()
            .expect("content");
        assert!(text.contains('\u{0}') && text.contains('\u{1}'), "{text:?}");
    }

    #[tokio::test]
    async fn thinking_on_opens_a_think_block_at_the_end_of_the_prompt() {
        let (engine, seen) = testkit::probe_engine();
        post(
            &app(&engine, 4_096),
            &request(json!({"max_tokens": 1, "x-thinking": "on"})),
        )
        .await;
        let prompt = seen.lock().expect("prompt lock").clone();
        assert!(prompt.ends_with(&testkit::ids("<think>\n")), "{prompt:?}");
    }

    #[tokio::test]
    async fn a_reply_to_a_prompt_with_an_open_think_block_starts_inside_the_reasoning() {
        let engine = testkit::engine(testkit::script("plan</think>Done."));
        let body = request(json!({"x-thinking": "on"}));
        let (status, reply) = post(&app(&engine, 4_096), &body).await;
        assert_eq!(status, StatusCode::OK);
        let message = &reply["choices"][0]["message"];
        assert_eq!(message["reasoning_content"], "plan");
        assert_eq!(message["content"], "Done.");
    }

    #[tokio::test]
    async fn a_reply_to_a_prompt_without_an_open_think_block_starts_as_content() {
        let engine = testkit::engine(testkit::script("plan</think>Done."));
        let (_, reply) = post(&app(&engine, 4_096), &request(json!({}))).await;
        let message = &reply["choices"][0]["message"];
        assert_eq!(message["content"], "plan</think>Done.");
        assert!(message.get("reasoning_content").is_none(), "{message}");
    }

    #[tokio::test]
    async fn the_last_free_position_of_the_context_holds_one_reply_token() {
        let engine = testkit::engine(testkit::script("abc"));
        let (status, reply) = post(&app(&engine, 22), &request(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(reply["choices"][0]["message"]["content"], "a");
        assert_eq!(reply["choices"][0]["finish_reason"], "length");
        assert_eq!(reply["usage"]["completion_tokens"], 1);
    }

    /// A router over one engine whose entry has the `prefix_reuse` flag set to `flag`.
    fn reuse_app(engine: &Engine, flag: bool) -> Router {
        testkit::wait_for_state(&engine.handle(), EngineState::Ready);
        let mut entry = testkit::entry("e1", "model-a", engine, 4_096);
        entry.prefix_reuse = flag;
        router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])))
    }

    /// A router over one engine whose entry reuses prefixes and renders prompts with `bundle`.
    fn reuse_app_with_bundle(engine: &Engine, bundle: Arc<TokenizerBundle>) -> Router {
        testkit::wait_for_state(&engine.handle(), EngineState::Ready);
        let mut entry = testkit::entry("e1", "model-a", engine, 4_096);
        entry.prefix_reuse = true;
        entry.bundle = bundle;
        router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])))
    }

    /// A request with the given messages as (role, content) pairs.
    fn conversation(messages: &[(&str, &str)]) -> Value {
        let messages: Vec<Value> = messages
            .iter()
            .map(|(role, content)| json!({"role": role, "content": content}))
            .collect();
        json!({"model": "model-a", "messages": messages})
    }

    /// Posts each conversation in order and returns `(prompt_tokens, reused_tokens)` of each reply.
    async fn usage_of(app: &Router, conversations: &[Value]) -> Vec<(u64, u64)> {
        let mut usage = Vec::new();
        for body in conversations {
            let (status, reply) = post(app, body).await;
            assert_eq!(status, StatusCode::OK, "{reply}");
            usage.push((
                reply["usage"]["prompt_tokens"]
                    .as_u64()
                    .expect("prompt tokens"),
                reply["usage"]["reused_tokens"]
                    .as_u64()
                    .expect("reused tokens"),
            ));
        }
        usage
    }

    /// The first request of the prefix tests, and the same conversation with a reply and a new
    /// question. The prompt of the second request starts with the prompt of the first without its
    /// generation tail.
    fn first_and_follow_up() -> [Value; 2] {
        [
            conversation(&[("user", "hi")]),
            conversation(&[("user", "hi"), ("assistant", "a"), ("user", "more")]),
        ]
    }

    #[tokio::test]
    async fn with_the_flag_off_a_second_request_reuses_nothing() {
        let engine = testkit::engine([testkit::script("a"), testkit::script("b")].concat());
        let usage = usage_of(&reuse_app(&engine, false), &first_and_follow_up()).await;
        assert_eq!(usage, [(21, 0), (47, 0)]);
    }

    #[tokio::test]
    async fn with_the_flag_on_a_request_that_extends_the_first_reuses_its_boundary() {
        let engine = testkit::engine([testkit::script("a"), testkit::script("b")].concat());
        let usage = usage_of(&reuse_app(&engine, true), &first_and_follow_up()).await;
        // The saved key is the first prompt without its generation tail:
        // `<|im_start|>` (1 id), `user\nhi` (7 bytes), `<|im_end|>` (1 id), and `\n` (1 byte).
        let key_tokens = 1 + "user\nhi".len() + 1 + "\n".len();
        assert_eq!(key_tokens, 10, "hand count of the key");
        // The follow-up adds `<|im_start|>assistant\na<|im_end|>\n` (14 ids) and
        // `<|im_start|>user\nmore<|im_end|>\n` (12 ids) to the 10 key ids, and the generation tail
        // `<|im_start|>assistant\n` (11 ids) ends the prompt.
        assert_eq!(usage, [(21, 0), (47, 10)]);
    }

    #[tokio::test]
    async fn with_the_flag_on_an_unrelated_request_reuses_nothing() {
        let engine = testkit::engine(
            [
                testkit::script("a"),
                testkit::script("b"),
                testkit::script("c"),
            ]
            .concat(),
        );
        let a = conversation(&[("user", "hi")]);
        let b = conversation(&[("user", "yo")]);
        // B replaces the saved key of A, so the second A finds nothing to restore.
        let usage = usage_of(&reuse_app(&engine, true), &[a.clone(), b, a]).await;
        assert_eq!(usage, [(21, 0), (21, 0), (21, 0)]);
    }

    #[tokio::test]
    async fn with_the_flag_on_a_second_conversation_restores_the_shared_root() {
        // Two conversations share a system prompt of 2,100 "a" bytes. Under the fixture
        // tokenizer every ASCII byte is one id, so the first system message renders as
        // `<|im_start|>` (1 id) + "system\n" (7 ids) + 2,100 ids + `<|im_end|>` (1 id) + "\n"
        // (1 id) = 2,110 ids: the root. Every prompt adds `<|im_start|>user\nhi<|im_end|>\n`
        // (10 ids) and the generation tail `<|im_start|>assistant\n` (11 ids), so both prompts
        // hold 2,131 ids. The second conversation's user turn differs from the first, so its
        // tip key matches nothing and only the root key matches.
        let engine = testkit::engine_with_prefix_limits(
            [testkit::script("a"), testkit::script("b")].concat(),
            2,
            6,
        );
        let system = "a".repeat(2_100);
        let first = conversation(&[("system", system.as_str()), ("user", "hi")]);
        let second = conversation(&[("system", system.as_str()), ("user", "yo")]);
        let usage = usage_of(&reuse_app(&engine, true), &[first, second]).await;
        assert_eq!(usage, [(2_131, 0), (2_131, 2_110)]);
    }

    #[tokio::test]
    async fn with_the_pinned_template_a_second_conversation_restores_the_shared_root() {
        // The pinned Qwen template raises `No user query found in messages.` for every render
        // without a user turn, so a root render of the system message alone would always fail
        // and no second conversation would restore anything. The fixture tokenizer gives one id
        // per byte and one id per special marker. The first system message renders as
        // `<|im_start|>` (1) + "system\n" (7) + 2,100 "a" + `<|im_end|>` (1) + "\n" (1) = 2,110
        // ids: the root. The user turn `<|im_start|>user\nhi<|im_end|>\n` adds 10 ids. The
        // generation tail `<|im_start|>assistant\n<think>\n` adds 1 + 10 + 8 = 19 ids, so
        // both prompts hold 2,139 ids. The second user turn differs, so only the root matches.
        let engine = testkit::engine_with_prefix_limits(
            [testkit::script("a"), testkit::script("b")].concat(),
            2,
            6,
        );
        let app = reuse_app_with_bundle(&engine, testkit::bundle_with_pinned_template());
        let system = "a".repeat(2_100);
        let first = conversation(&[("system", system.as_str()), ("user", "hi")]);
        let second = conversation(&[("system", system.as_str()), ("user", "yo")]);
        let usage = usage_of(&app, &[first, second]).await;
        assert_eq!(usage, [(2_139, 0), (2_139, 2_110)]);
    }

    #[tokio::test]
    async fn a_session_header_still_returns_not_found_with_the_flag_on() {
        let engine = testkit::engine(testkit::script("a"));
        let session = request(json!({"x-session": "s1"}));
        let (status, _) = post(&reuse_app(&engine, true), &session).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn missing_bearer_returns_401() {
        let engine = testkit::engine(Vec::new());
        let body = request(json!({})).to_string().into_bytes();
        let request = testkit::post_bytes("/v1/chat/completions", None, body);
        let (status, _headers, bytes) = testkit::send(&app(&engine, 4_096), request).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_schema("Error", &testkit::json_of(&bytes));
    }

    #[tokio::test]
    async fn body_over_16_mib_returns_badrequest() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine, 4_096);
        let frame = |pad: usize| {
            format!(
                r#"{{"model":"model-a","messages":[],"pad":"{}"}}"#,
                "x".repeat(pad)
            )
            .into_bytes()
        };
        let overhead = frame(0).len();
        let at_limit = testkit::post_bytes(
            "/v1/chat/completions",
            Some(testkit::API_KEY),
            frame(SIXTEEN_MIB - overhead),
        );
        let (status, _h, bytes) = testkit::send(&app, at_limit).await;
        let message = testkit::json_of(&bytes)["error"]["message"]
            .as_str()
            .expect("message")
            .to_string();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            message.contains("messages"),
            "a body at the cap reaches validation: {message}"
        );
        let over = testkit::post_bytes(
            "/v1/chat/completions",
            Some(testkit::API_KEY),
            frame(SIXTEEN_MIB - overhead + 1),
        );
        let (status, _h, bytes) = testkit::send(&app, over).await;
        let message = testkit::json_of(&bytes)["error"]["message"]
            .as_str()
            .expect("message")
            .to_string();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(message.contains("16 MiB"), "{message}");
    }

    #[tokio::test]
    async fn max_tokens_below_1_returns_400_that_names_the_field() {
        let engine = testkit::engine(Vec::new());
        let (status, body) = post(&app(&engine, 4_096), &request(json!({"max_tokens": 0}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body["error"]["message"]
                .as_str()
                .expect("message")
                .contains("max_tokens")
        );
        assert_schema("Error", &body);
    }

    #[tokio::test]
    async fn routing_failures_return_the_status_of_the_design() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine, 4_096);
        let session = request(json!({"x-session": "s1"}));
        assert_eq!(post(&app, &session).await.0, StatusCode::NOT_FOUND);
        let model = request(json!({"model": "other"}));
        assert_eq!(post(&app, &model).await.0, StatusCode::NOT_FOUND);
        let unrouted = json!({"messages": [{"role": "user", "content": "hi"}]});
        let (status, body) = post(&app, &unrouted).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error_code(&body), "bad_request");
    }

    #[tokio::test]
    async fn x_engine_takes_precedence_over_model_on_the_chat_route() {
        let engine = testkit::engine(testkit::script("ok"));
        let body = json!({"model": "unknown", "x-engine": "e1", "messages": [{"role": "user", "content": "hi"}]});
        let (status, reply) = post(&app(&engine, 4_096), &body).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(reply["engine"], "e1");
    }

    #[tokio::test]
    async fn full_queue_returns_503_overloaded() {
        let (engine, gate) = testkit::gated_engine(Vec::new());
        let handle = engine.handle();
        let mut held = Vec::new();
        for _ in 0..32 {
            let generation = testkit::generation();
            let (job, receiver) = Job::new(vec![104], generation).expect("job");
            handle.try_submit(job).expect("the queue has room");
            held.push(receiver);
        }
        let entry = testkit::entry("e1", "model-a", &engine, 4_096);
        let app = router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])));
        let (status, body) = post(&app, &request(json!({}))).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error_code(&body), "overloaded");
        assert_schema("Error", &body);
        drop(gate);
    }

    #[tokio::test]
    async fn an_engine_that_failed_to_load_returns_500_backend_error() {
        let engine = testkit::failing_engine();
        let entry = testkit::entry("e1", "model-a", &engine, 4_096);
        testkit::wait_for_state(&entry.handle, EngineState::Error);
        let app = router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])));
        let (status, body) = post(&app, &request(json!({}))).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_schema("BackendError", &body);
        assert!(!body.to_string().contains("fixture load failure"), "{body}");
    }

    #[tokio::test]
    async fn a_stopped_engine_returns_503_overloaded() {
        let engine = testkit::engine(Vec::new());
        let entry = testkit::entry("e1", "model-a", &engine, 4_096);
        testkit::wait_for_state(&entry.handle, EngineState::Ready);
        let app = router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])));
        engine.shutdown();
        let (status, body) = post(&app, &request(json!({}))).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error_code(&body), "overloaded");
    }

    #[tokio::test]
    async fn memoryplan_and_contextlimit_return_409_with_distinct_codes() {
        let short = testkit::engine(Vec::new());
        let (status, context) = post(&app(&short, 21), &request(json!({}))).await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_schema("Error", &context);
        let refusing =
            testkit::refusing_engine(EngError::MemoryPlan("no room for the session".to_string()));
        let (status, memory) = post(&app(&refusing, 4_096), &request(json!({}))).await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_schema("Error", &memory);
        assert_eq!(error_code(&context), "context_limit");
        assert_eq!(error_code(&memory), "memory_plan");
        assert_eq!(memory["error"]["message"], "no room for the session");
    }

    #[tokio::test]
    async fn a_prompt_over_the_limit_is_refused_with_no_prefill_on_the_engine() {
        // The rendered prompt of "hi" is 21 tokens, so an entry limit of 21 refuses it at
        // admission: the engine never sees the job.
        let (engine, seen) = testkit::probe_engine();
        let (status, body) = post(&app(&engine, 21), &request(json!({}))).await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(error_code(&body), "context_limit");
        assert_schema("Error", &body);
        assert!(
            seen.lock().expect("prompt lock").is_empty(),
            "the engine prefilled a refused prompt"
        );
    }

    #[tokio::test]
    async fn a_streaming_prompt_over_the_limit_gets_the_error_response_not_a_stream() {
        // The refusal happens before the stream is built, so the client gets the same JSON error
        // as a non-streaming request, with no SSE byte and no role chunk.
        let (engine, seen) = testkit::probe_engine();
        let app = app(&engine, 21);
        let (status, headers, bytes) = testkit::send(
            &app,
            testkit::post_json("/v1/chat/completions", &request(json!({"stream": true}))),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        let content_type = headers[axum::http::header::CONTENT_TYPE]
            .to_str()
            .expect("content type");
        assert_eq!(content_type, "application/json", "not an SSE stream");
        assert_eq!(error_code(&testkit::json_of(&bytes)), "context_limit");
        assert!(
            seen.lock().expect("prompt lock").is_empty(),
            "the engine prefilled a refused prompt"
        );
    }

    #[tokio::test]
    async fn a_backend_failure_returns_500_hides_the_diagnostic_and_stops_the_engine() {
        let failure = EngError::Backend {
            operation: "qmm",
            message: "Metal command buffer 7 failed".to_string(),
        };
        let engine = testkit::refusing_engine(failure);
        let app = app(&engine, 4_096);
        let (status, body) = post(&app, &request(json!({}))).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_schema("BackendError", &body);
        assert!(!body.to_string().contains("Metal"), "{body}");
        testkit::wait_for_state(&engine.handle(), EngineState::NativeFailed);
        let (status, again) = post(&app, &request(json!({}))).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_schema("BackendError", &again);
    }
}
