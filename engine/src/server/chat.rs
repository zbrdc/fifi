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
use crate::tokenizer::toolcall::{Start, ToolGrammar};
use crate::tokenizer::{IncrementalDecoder, OutputAssembler, PromptBoundaries, TokenizerBundle};
use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

/// The body cap of the contract: 16 MiB.
pub(crate) const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

/// A request that passed every check and whose job the engine queue holds.
pub(crate) struct Submitted {
    pub(crate) head: ReplyHead,
    pub(crate) running: RunningJob,
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
    let mut submitted = submit(state, input, "POST", "/v1/chat/completions").await?;
    if stream {
        let keep_alive = state.keep_alive();
        return Ok(sse::respond(submitted.head, submitted.running, keep_alive));
    }
    let (parts, finish, usage) = collect(&mut submitted.running).await?;
    let completion = ChatCompletion::new(&submitted.head, parts, finish, usage);
    Ok(Json(completion).into_response())
}

/// Resolves the engine, builds the prompt, and queues the job. The Anthropic route
/// shares this path: its request became a `ChatInput` before it reached here.
pub(crate) async fn submit(
    state: &AppState,
    input: ChatInput,
    method: &'static str,
    endpoint: &'static str,
) -> Result<Submitted, ServerError> {
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
    let warm = std::sync::Arc::clone(&entry.warm);
    let (prompt, boundaries, start) = tokio::task::spawn_blocking(move || {
        // Without the flag the handler marks no boundary, so the engine neither saves nor restores.
        // The derived boundary skips only the head RE-TOKENIZE: the full
        // prompt renders once and its ids prove the tip exactly through the
        // generation-prompt suffix; otherwise it encodes the already-rendered
        // head — the same cost as the three-pass path.
        // ENGINESRV_WARM_TURNS layers the per-engine warm cache on top
        // (default ON): the previous turn's head ids are reused and only
        // the appended delta is tokenized (byte-exact, counted fallbacks
        // on any doubt). ENGINESRV_WARM_TURNS=0 rolls back to this path.
        let warm_turns = prefix_reuse && warm_flag_on();
        let (prompt, boundaries) = if warm_turns {
            // Take the cache out for the encode so concurrent requests on
            // one engine never queue behind each other's tokenization, and
            // one poisoned lock cannot fail every later request:
            // a poison recovery keeps the DATA (the cache holds no
            // in-memory invariant, only reusable artifacts).
            let mut cache = warm
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
                .unwrap_or_else(crate::tokenizer::WarmTurnCache::new);
            let fallbacks_before = cache.fallbacks();
            let result =
                bundle.encode_prompt_warm(&mut cache, &messages, &template_tools, thinking);
            let fell_back = cache.fallbacks() > fallbacks_before;
            // Snapshot the cause with the counter, immediately at the call
            // site — after the cache returns to the lock, a concurrent
            // request on the same engine could re-seed it and overwrite
            // the reason before the log line runs.
            let reason = cache.last_fallback_reason();
            *warm
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(cache);
            let (prompt, boundaries) = result?;
            if fell_back {
                eprintln!(
                    "warm tokenization: fell back to the derived path ({} ids, reason={})",
                    prompt.len(),
                    reason.unwrap_or("unknown")
                );
            }
            (prompt, boundaries)
        } else if prefix_reuse {
            let (prompt, boundaries, _derived) =
                bundle.encode_prompt_with_derived_boundary(&messages, &template_tools, thinking)?;
            (prompt, boundaries)
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
        method,
        endpoint,
    );
    // The monitor's ring rides the same lines (count_tokens is filtered
    // out inside the ring, so the count route needs no wiring).
    log.set_ring(state.recent_ring());
    // The prompt digests and the LCP with this model's previous request: the
    // server's own view of render stability, independent of the engine store.
    let lcp = state.note_prompt(&entry.model, &prompt);
    log.set_prompt_digests(&prompt, lcp);
    if prompt_tokens >= entry.context_limit {
        // The wire names the largest prompt this entry admits — one below
        // the limit the gate compares against — so every statement on the
        // wire is true and the invariant prompt > limit holds at the
        // boundary too.
        return Err(failed(
            &mut log,
            EngError::ContextLimit(EngError::prompt_too_long_message(
                u64::from(prompt_tokens),
                u64::from(entry.context_limit.saturating_sub(1)),
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
    let parser = ReplyParser::with_grammar(
        start,
        &input.tools,
        &head.call_prefix(),
        entry.bundle.tool_grammar(),
    );
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
        // The line of this request carries the prefix fields; a request without a boundary
        // keeps the old line byte for byte.
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

/// `ENGINESRV_WARM_TURNS`: the repo's fail-safe default-on switch
/// (`env_allows_default_on`). Absent keeps the new default ON; once set,
/// only "1"/"true"/"on" (trimmed, case-folded) keep it on, and ANYTHING
/// else — "0", "false", "off", "no", "0 ", empty, typos, non-UTF-8 —
/// rolls back to the derived path. A rollback can never silently fail.
const WARM_TURNS_VAR: &str = "ENGINESRV_WARM_TURNS";

/// In test builds, lets the server tests force the branch without
/// mutating the process-global env var (the crate denies unsafe env
/// mutation). 0 follows the env rule, 1 forces ON, -1 forces OFF.
/// Production builds never read it.
#[cfg(test)]
static TEST_WARM_TURNS: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(0);

/// The per-request warm flag: the fail-safe env rule, overridable in tests.
/// `run_cli`'s boot line reads the same function so the banner can never
/// drift from what the requests actually do.
pub(crate) fn warm_flag_on() -> bool {
    #[cfg(test)]
    match TEST_WARM_TURNS.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return true,
        -1 => return false,
        _ => {}
    }
    crate::tokenizer::env_allows_default_on(WARM_TURNS_VAR)
}

/// Chooses where the parser starts in the reply. When the prompt ends inside an open think block,
/// the template asked for thinking and the reply starts inside the reasoning. The template is the
/// one authority: the check reads the end of the rendered prompt, not the request setting.
fn reply_start(bundle: &TokenizerBundle, prompt: &[u32]) -> Start {
    // The exact Flint template appends only <|assistant|>, never an open think block.
    // Tail decoding skips that special role token and could expose a historical <think>.
    if bundle.tool_grammar() == ToolGrammar::Flint {
        return Start::Content;
    }
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

    /// The fail-safe rule (`env_allows_value`, the pure half of
    /// `env_allows_default_on`): absent keeps the new default ON; once
    /// set, only the explicit enable words keep it on, and every other
    /// value — disable words, case games, whitespace, empty — rolls back.
    #[test]
    fn only_the_enable_words_keep_warm_turns_on() {
        use crate::tokenizer::env_allows_value;
        assert!(env_allows_value(None), "unset: the default is ON");
        assert!(!env_allows_value(Some("0")), "the plain rollback");
        assert!(env_allows_value(Some("1")), "explicit enable");
        assert!(env_allows_value(Some("true")), "enable word");
        assert!(env_allows_value(Some(" on ")), "trimmed, case-folded");
        assert!(!env_allows_value(Some("false")), "disable word rolls back");
        assert!(!env_allows_value(Some("off")), "disable word rolls back");
        assert!(!env_allows_value(Some("no")), "disable word rolls back");
        assert!(
            !env_allows_value(Some("0 ")),
            "whitespace does not hide the zero"
        );
        assert!(!env_allows_value(Some("x")), "typos roll back");
        assert!(!env_allows_value(Some("")), "empty set value rolls back");
        assert!(
            env_allows_value(Some("ON")),
            "folding matches the enable word"
        );
        assert!(!env_allows_value(Some("enabled")), "synonyms roll back");
    }

    /// One sequential test for both flag states (a std lock across the
    /// POST awaits trips await_holding_lock, and one test cannot race
    /// itself): phase one pins flag OFF — byte-identical ids, no cache —
    /// then phase two forces the flag ON via the test-only override and
    /// pins a seed + an actual warm reuse, both still exact.
    #[tokio::test]
    async fn flag_off_then_flag_on_is_exact_end_to_end() {
        assert!(
            std::env::var_os("ENGINESRV_WARM_TURNS").is_none(),
            "no other test may set the env flag"
        );
        // The guard is created BEFORE any override is stored, so even an
        // assert in phase one cannot leak a forced state to the rest of
        // the suite.
        struct ResetWarmFlag;
        impl Drop for ResetWarmFlag {
            fn drop(&mut self) {
                super::TEST_WARM_TURNS.store(0, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let _reset_warm_flag = ResetWarmFlag;
        // Step 1: flag OFF — the rollback state, forced via the test
        // override (the default is ON).
        super::TEST_WARM_TURNS.store(-1, std::sync::atomic::Ordering::Relaxed);
        let probes = testkit::probe_engine_with_sampling();
        let bundle = testkit::bundle_with_pinned_template();
        let (state, warm_slot) = reuse_state_with_bundle(&probes.engine, Arc::clone(&bundle));
        let app = router(Arc::clone(&state));
        let body = conversation(&[
            ("system", "You are careful."),
            ("user", "hello"),
            ("assistant", "hi"),
            ("user", "and again"),
        ]);
        let (status, reply) = post(&app, &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        let reference = derived_ids(&bundle, &body);
        // The probe engine records every prefilled token: the handler's
        // submitted ids must be the derived reference ids byte for byte.
        testkit::wait_until("the prompt is prefilled", || {
            !probes.prompt.lock().expect("prompt lock").is_empty()
        });
        let submitted = probes.prompt.lock().expect("prompt lock").clone();
        assert_eq!(submitted, reference, "flag-off ids must be the derived ids");
        assert!(
            warm_slot.lock().expect("warm lock").is_none(),
            "flag off must never create the warm cache"
        );

        // Step 2: flag ON (the override guard from step 1 still holds).
        super::TEST_WARM_TURNS.store(1, std::sync::atomic::Ordering::Relaxed);
        // Reuse the SAME engine and cache slot: request one seeds, the
        // appended request two is a warm reuse.
        let conversations = [
            conversation(&[("system", "You are careful."), ("user", "hello")]),
            conversation(&[
                ("system", "You are careful."),
                ("user", "hello"),
                ("assistant", "hi"),
                ("user", "and again"),
            ]),
        ];
        for (turn, body) in conversations.iter().enumerate() {
            probes.prompt.lock().expect("prompt lock").clear();
            let (status, reply) = post(&app, body).await;
            assert_eq!(status, StatusCode::OK, "{reply}");
            testkit::wait_until("the prompt is prefilled", || {
                !probes.prompt.lock().expect("prompt lock").is_empty()
            });
            let submitted = probes.prompt.lock().expect("prompt lock").clone();
            let reference = derived_ids(&bundle, body);
            assert_eq!(
                submitted, reference,
                "flag-on turn {turn} ids must stay exact"
            );
        }
        {
            let guard = warm_slot.lock().expect("warm lock");
            let cache = guard.as_ref().expect("flag on creates the cache");
            assert!(cache.reuses() >= 1, "turn two must actually reuse");
        }
    }

    /// Scope pin: the warm branch needs the engine entry's `prefix_reuse`;
    /// with the flag off, the default-ON warm rule never engages the cache
    /// (serving is verified on Qwen3.6 only — see the README scope note).
    /// No test override is written: the process-global static races the
    /// parallel e2e phases, and the default is already ON while the env
    /// stays unset.
    #[tokio::test]
    async fn warm_needs_prefix_reuse_even_when_the_flag_is_on() {
        assert!(
            std::env::var_os("ENGINESRV_WARM_TURNS").is_none(),
            "no other test may set the env flag"
        );
        let probes = testkit::probe_engine_with_sampling();
        let bundle = testkit::bundle_with_pinned_template();
        testkit::wait_for_state(&probes.engine.handle(), EngineState::Ready);
        let mut entry = testkit::entry("e1", "model-a", &probes.engine, 4_096);
        entry.prefix_reuse = false;
        entry.bundle = Arc::clone(&bundle);
        let warm = Arc::clone(&entry.warm);
        let app = router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])));
        let body = conversation(&[("user", "hello")]);
        let (status, reply) = post(&app, &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        testkit::wait_until("the prompt is prefilled", || {
            !probes.prompt.lock().expect("prompt lock").is_empty()
        });
        let reference = derived_ids(&bundle, &body);
        let submitted = probes.prompt.lock().expect("prompt lock").clone();
        assert_eq!(submitted, reference, "ids stay on the plain path");
        assert!(
            warm.lock().expect("warm lock").is_none(),
            "no prefix_reuse means no warm cache, flag or not"
        );
    }

    /// End-to-end with the REAL env var, in fresh child processes (the
    /// crate denies unsafe env mutation): `ENGINESRV_WARM_TURNS=0` must keep
    /// the derived path with no cache; unset must default ON and seed the
    /// cache. Both children pin id exactness against the derived reference.
    #[tokio::test]
    async fn the_real_env_rollback_and_default_run_end_to_end_in_children() {
        const CHILD: &str = "ENGINESRV_TEST_WARM_ENV_CHILD";
        let selector =
            "server::chat::tests::the_real_env_rollback_and_default_run_end_to_end_in_children";
        if std::env::var_os(CHILD).is_none() {
            for value in [Some("0"), Some("false"), Some(""), None] {
                let mut command = std::process::Command::new(std::env::current_exe().unwrap());
                command
                    .args(["--exact", selector, "--nocapture"])
                    .env(CHILD, "1");
                match value {
                    Some(v) => {
                        command.env("ENGINESRV_WARM_TURNS", v);
                    }
                    None => {
                        command.env_remove("ENGINESRV_WARM_TURNS");
                    }
                }
                let output = command.output().expect("child test run");
                let stderr = String::from_utf8_lossy(&output.stderr);
                let text = format!("{}{}", String::from_utf8_lossy(&output.stdout), stderr);
                assert!(output.status.success(), "{value:?} child failed: {text}");
                let sentinel = format!("child-warm-env:{value:?}");
                assert!(text.contains(&sentinel), "sentinel missing: {text}");
                if value.is_none() {
                    assert!(
                        stderr.contains("reason=other-conversation"),
                        "the unset child's foreign post must name its cause on stderr: {text}"
                    );
                }
            }
            return;
        }
        let mode = std::env::var_os("ENGINESRV_WARM_TURNS");
        let probes = testkit::probe_engine_with_sampling();
        let bundle = testkit::bundle_with_pinned_template();
        let (state, warm_slot) = reuse_state_with_bundle(&probes.engine, Arc::clone(&bundle));
        let app = router(state);
        let conversations = [
            conversation(&[("system", "You are careful."), ("user", "hello")]),
            conversation(&[
                ("system", "You are careful."),
                ("user", "hello"),
                ("assistant", "hi"),
                ("user", "and again"),
            ]),
        ];
        for body in &conversations {
            probes.prompt.lock().expect("prompt lock").clear();
            let (status, reply) = post(&app, body).await;
            assert_eq!(status, StatusCode::OK, "env={mode:?} {reply}");
            testkit::wait_until("the prompt is prefilled", || {
                !probes.prompt.lock().expect("prompt lock").is_empty()
            });
            let submitted = probes.prompt.lock().expect("prompt lock").clone();
            assert_eq!(submitted, derived_ids(&bundle, body), "ids stay exact");
        }
        // Any SET value in the child loop rolls back: the boot line and the
        // request path must agree that a mis-set rollback disables.
        if mode.is_some() {
            assert!(
                warm_slot.lock().expect("warm lock").is_none(),
                "a set value must keep the derived path with no cache"
            );
        } else {
            let guard = warm_slot.lock().expect("warm lock");
            let cache = guard
                .as_ref()
                .expect("unset must default ON and seed the cache");
            assert!(cache.reuses() >= 1, "the unset child must actually reuse");
        }
        // A same-key system-first foreign conversation (the replay traffic's
        // side-request shape): the warm path refuses, and the serving
        // fallback LINE carries the cause. Printed to the parent's captured
        // text below, the only place a per-process stderr line is pinnable.
        if mode.is_none() {
            let foreign = conversation(&[("system", "You are careful."), ("user", "unrelated")]);
            let (status, reply) = post(&app, &foreign).await;
            assert_eq!(status, StatusCode::OK, "env={mode:?} {reply}");
        }
        println!("child-warm-env:{mode:?}");
    }

    /// The derived-path reference ids of a request body, computed directly.
    fn derived_ids(bundle: &Arc<TokenizerBundle>, body: &Value) -> Vec<u32> {
        let request =
            crate::server::validate::parse_chat(&serde_json::to_vec(body).unwrap()).expect("parse");
        let messages = request.messages;
        let tools: Vec<_> = request
            .tools
            .iter()
            .map(|t| crate::tokenizer::ToolSpec(t.0.clone()))
            .collect();
        bundle
            .encode_prompt_with_derived_boundary(&messages, &tools, request.thinking)
            .expect("derived")
            .0
    }

    /// The shared state of one prefix-reuse engine that renders prompts with
    /// `bundle`, plus the entry's warm-cache slot (shared `Arc`, so tests
    /// can assert on it without an accessor).
    fn reuse_state_with_bundle(
        engine: &Engine,
        bundle: Arc<TokenizerBundle>,
    ) -> (
        Arc<AppState>,
        Arc<std::sync::Mutex<Option<crate::tokenizer::WarmTurnCache>>>,
    ) {
        testkit::wait_for_state(&engine.handle(), EngineState::Ready);
        let mut entry = testkit::entry("e1", "model-a", engine, 4_096);
        entry.prefix_reuse = true;
        entry.bundle = bundle;
        let warm = Arc::clone(&entry.warm);
        (Arc::new(AppState::new(testkit::API_KEY, vec![entry])), warm)
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
        // The probe engine never emits EOS, so an unset max_tokens decodes to the
        // context limit (~4K job events per request). Under the full parallel suite
        // the client task can fall behind draining them, the 1024-slot job event
        // channel fills, and the engine's slow-client cancel (by design: a stuck
        // client must not stall decode) aborts the job - a load-order 500 that
        // flakes the suite. A bounded decode cannot overflow the channel. Tests
        // that need another bound pass their own max_tokens, which replaces this.
        json!({"model": "model-a", "messages": messages, "max_tokens": 8})
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

    /// R4 integration: on a warm conversation the chat usage carries
    /// prompt_tokens_details.cached_tokens == reused while prompt_tokens
    /// keeps the whole prompt (OpenAI semantics).
    #[tokio::test]
    async fn a_warm_conversation_reports_the_cached_tokens_in_the_details() {
        let engine = testkit::engine_with_prefix_limits(
            [testkit::script("a"), testkit::script("b")].concat(),
            2,
            6,
        );
        let app = reuse_app_with_bundle(&engine, testkit::bundle_with_pinned_template());
        let system = "a".repeat(2_100);
        let first = conversation(&[("system", system.as_str()), ("user", "hi")]);
        let second = conversation(&[("system", system.as_str()), ("user", "yo")]);
        let (_, cold) = post(&app, &first).await;
        let (_, warm) = post(&app, &second).await;
        assert_eq!(warm["usage"]["prompt_tokens"], 2_139, "{warm}");
        assert_eq!(warm["usage"]["reused_tokens"], 2_110, "{warm}");
        // The details key follows the switch: present with cached_tokens
        // == reused when the split is on (the default), absent when the
        // switch forces legacy. The on bytes are pinned exactly by the
        // wire.rs child test.
        if crate::server::wire::cache_fields_on() {
            assert_eq!(
                cold["usage"]["prompt_tokens_details"]["cached_tokens"], 0,
                "{cold}"
            );
            assert_eq!(
                warm["usage"]["prompt_tokens_details"]["cached_tokens"], 2_110,
                "{warm}"
            );
        } else {
            assert!(
                cold["usage"]["prompt_tokens_details"].is_null()
                    && warm["usage"]["prompt_tokens_details"].is_null(),
                "details must stay absent when the switch forces legacy: {cold} {warm}"
            );
        }
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
