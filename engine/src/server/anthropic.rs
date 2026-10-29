//! The Anthropic Messages endpoints: `POST /v1/messages` and
//! `POST /v1/messages/count_tokens`.
//!
//! The handler implements both paths in `api/openapi.yaml`. The messages
//! handler converts an Anthropic request body into the `ChatInput` of the
//! chat pipeline, reuses the chat submit path unchanged, and writes the
//! reply in the Messages shape. The count_tokens handler parses the same
//! body and tokenizes the rendered prompt through the chat pipeline's
//! tokenizer bundle; no engine job runs for it.
//!
//! The module covers the non-streaming reply and the SSE event stream.

use super::chat::{self, MAX_BODY_BYTES};
use super::error::{BACKEND_MESSAGE, ServerError};
use super::messages::{bad, optional_string, required_string};
use super::pieces::Piece;
use super::reply::{RunningJob, Step, collect_pieces};
use super::reqlog::{RequestLog, default_clock};
use super::state::AppState;
use super::validate::{ChatInput, RequestedSampling};
use super::wire::{FinishReason, ReplyHead, Usage, cache_fields_on};
use crate::EngError;
use crate::family::sampler::{SamplingDefaults, SamplingParams};
use crate::tokenizer::{PromptMessage, Role, Thinking, ToolCallSpec, ToolSpec};
use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use http_body_util::LengthLimitError;
use serde_json::{Map, Value, json};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

/// The route handler. Every pre-header failure is an Anthropic JSON error;
/// after the headers, the stream sends `event: error`.
pub(crate) async fn messages(state: State<Arc<AppState>>, body: Body) -> Response {
    match reply(&state, body).await {
        Ok(response) => response,
        Err(error) => error_response(&error),
    }
}

/// The count_tokens handler: POST /v1/messages/count_tokens.
/// It parses the same Anthropic body, resolves the engine the same way,
/// renders and tokenizes the prompt, and returns the token count.
/// No engine job is submitted.
pub(crate) async fn count_tokens(state: State<Arc<AppState>>, body: Body) -> Response {
    match count_reply(&state, body).await {
        Ok(response) => response,
        Err(error) => error_response(&error),
    }
}

/// Walks the Anthropic body, resolves the engine, tokenizes the prompt,
/// and returns the count. The body limit and parsing errors are the same
/// as the messages route.
async fn count_reply(state: &AppState, body: Body) -> Result<Response, ServerError> {
    let bytes = match axum::body::to_bytes(body, MAX_BODY_BYTES).await {
        Ok(bytes) => bytes,
        Err(error) if is_length_limit(&error) => return Ok(too_large_response()),
        Err(_) => return Err(bad("the request body could not be read")),
    };
    let mut input = parse_count_tokens(&bytes)?.input;
    let entry = state.resolve(None, input.engine.as_deref(), input.model.as_deref())?;
    let bundle = Arc::clone(&entry.bundle);
    // The same template gate as the messages route: a template without
    // think replay counts the prompt without resent reasoning.
    gate_resent_reasoning(&bundle, &mut input.messages);
    let messages = input.messages;
    let template_tools = input.tools;
    let thinking = input.thinking;
    let prompt = tokio::task::spawn_blocking(move || {
        bundle.encode_prompt(&messages, &template_tools, thinking)
    })
    .await
    .map_err(|error| {
        ServerError::Eng(EngError::Backend {
            operation: "tokenize",
            message: error.to_string(),
        })
    })??;
    let prompt_tokens = u32::try_from(prompt.len()).unwrap_or(u32::MAX);
    // One request-log line like every other served request. No job follows,
    // so the line carries the count and neither finish reason nor timings;
    // a tokenize failure stays unlogged, exactly as on the chat path.
    let mut log = RequestLog::new(
        state.request_log(),
        default_clock(),
        &entry.model,
        false,
        prompt_tokens,
        "POST",
        "/v1/messages/count_tokens",
    );
    let lcp = state.note_prompt(&entry.model, &prompt);
    log.set_prompt_digests(&prompt, lcp);
    log.set_id(&state.next_completion_id());
    log.counted();
    Ok(Json(json!({"input_tokens": u64::from(prompt_tokens)})).into_response())
}

/// Walks the error chain for the body limit: `to_bytes` reports a body over the
/// cap as a `LengthLimitError`; every other read failure is something else.
fn is_length_limit(error: &axum::Error) -> bool {
    let mut source: Option<&dyn std::error::Error> = Some(error);
    while let Some(error) = source {
        if error.downcast_ref::<LengthLimitError>().is_some() {
            return true;
        }
        source = error.source();
    }
    false
}

/// The rollback gate shared by both routes: a serving template without
/// think replay never sees resent reasoning, so every template — Qwen,
/// Flint, legacy — keeps the exact bytes it rendered before the replay
/// feature, whatever the env flags say. With replay on, the field passes
/// and the template's own keep/drop rule decides which turns show it — and
/// /v1/chat/completions reasoning_content also replays and wraps then, the
/// same template serving both surfaces.
fn gate_resent_reasoning(
    bundle: &crate::tokenizer::TokenizerBundle,
    messages: &mut [crate::tokenizer::PromptMessage],
) {
    if !bundle.think_replay() {
        for message in messages {
            message.reasoning_content = None;
        }
    }
}

async fn reply(state: &AppState, body: Body) -> Result<Response, ServerError> {
    let bytes = match axum::body::to_bytes(body, MAX_BODY_BYTES).await {
        Ok(bytes) => bytes,
        // A body over the cap is its own status in the Messages contract (§5.3),
        // so it never flows through the ServerError table. Any other read
        // failure (a dropped connection, bad chunk framing) is not about size,
        // so it stays a bad request.
        Err(error) if is_length_limit(&error) => return Ok(too_large_response()),
        Err(_) => return Err(bad("the request body could not be read")),
    };
    let mut input = parse_anthropic(&bytes)?.input;
    let stream = input.stream;
    // The serving template decides whether resent thinking can replay; a
    // template without think replay (every legacy or Flint one) keeps the
    // exact bytes it rendered before the replay feature existed.
    let entry = state.resolve(None, input.engine.as_deref(), input.model.as_deref())?;
    gate_resent_reasoning(&entry.bundle, &mut input.messages);
    let mut submitted = chat::submit(state, input, "POST", "/v1/messages").await?;
    if stream {
        // The event stream takes over after admission: every failure past
        // this point is one `event: error`, not a JSON status.
        return Ok(sse_respond(
            submitted.head,
            submitted.running,
            state.keep_alive(),
        ));
    }
    let (pieces, finish, usage) = collect_pieces(&mut submitted.running).await?;
    let stop = submitted.running.matched_stop();
    let body = message_body(&submitted.head, &pieces, finish, usage, stop);
    Ok(Json(body).into_response())
}

/// The body-over-cap failure. `ServerError` has no 413; this writes the contract
/// status and type directly, outside the shared error table.
fn too_large_response() -> Response {
    let body = json!({
        "type": "error",
        "error": {
            "type": "request_too_large",
            "message": "the request body is larger than 16 MiB",
        },
    });
    (StatusCode::PAYLOAD_TOO_LARGE, Json(body)).into_response()
}

/// Writes a `ServerError` in the Anthropic error shape. The Messages contract has no
/// 409 or 422: capacity failures are invalid requests, and a checkpoint failure is a
/// server fault (500 api_error). The auth layer of this route also writes through
/// here, so its 401 carries the same shape.
pub(crate) fn error_response(error: &ServerError) -> Response {
    let status = match error {
        ServerError::Eng(EngError::MemoryPlan(_) | EngError::ContextLimit(_)) => {
            StatusCode::BAD_REQUEST
        }
        ServerError::Eng(EngError::PromptTooLong { .. }) => StatusCode::BAD_REQUEST,
        ServerError::Eng(EngError::Checkpoint(_)) => StatusCode::INTERNAL_SERVER_ERROR,
        other => other.status(),
    };
    (status, Json(anthropic_error_body(error))).into_response()
}

/// Logs the failure and builds the Anthropic error body. The stream's
/// `event: error` sends the same shape after the headers.
fn anthropic_error_body(error: &ServerError) -> Value {
    error.log();
    // A checkpoint failure is a server-side fault (a tier mapping that did not
    // load), so this route reports it as a 500 api_error with a generic
    // message; the detail stays in the local log. The chat path keeps its own
    // mapping untouched.
    if let ServerError::Eng(EngError::Checkpoint(detail)) = error {
        eprintln!("checkpoint failure on /v1/messages: {detail}");
    }
    let kind = match error {
        ServerError::Eng(EngError::ModelOutput(_)) => "api_error",
        ServerError::Eng(EngError::Backend { .. } | EngError::Checkpoint(_)) => "api_error",
        ServerError::Unauthorized => "authentication_error",
        ServerError::NotFound(_) => "not_found_error",
        ServerError::Overloaded(_) => "overloaded_error",
        ServerError::Eng(
            EngError::BadRequest(_)
            | EngError::MemoryPlan(_)
            | EngError::ContextLimit(_)
            | EngError::PromptTooLong { .. },
        ) => "invalid_request_error",
    };
    // The shared 401 message names the bearer form; this route also takes x-api-key.
    // Every api_error sends the generic message: no internal detail (paths,
    // operations) reaches the client; the log carries it.
    let message = match error {
        ServerError::Unauthorized => "missing or invalid credential".to_string(),
        ServerError::Eng(EngError::ModelOutput(_)) => BACKEND_MESSAGE.to_string(),
        ServerError::Eng(EngError::Backend { .. } | EngError::Checkpoint(_)) => {
            BACKEND_MESSAGE.to_string()
        }
        _ => error.message(),
    };
    json!({
        "type": "error",
        "error": {"type": kind, "message": message},
    })
}

/// The Messages id form: the chat completion id without its prefix.
fn message_id(head: &ReplyHead) -> String {
    format!("msg_{}", head.id.trim_start_matches("chatcmpl-"))
}

/// The stop reason and stop sequence of the reply. A matched stop string
/// wins over the plain end turn.
fn stop_fields(finish: FinishReason, matched_stop: Option<&str>) -> (&'static str, Value) {
    match finish {
        FinishReason::Stop => match matched_stop {
            Some(stop) => ("stop_sequence", json!(stop)),
            None => ("end_turn", Value::Null),
        },
        FinishReason::Length => ("max_tokens", Value::Null),
        FinishReason::ToolCalls => ("tool_use", Value::Null),
    }
}

/// The parser builds the arguments as a JSON object; the fallback keeps a
/// shape guarantee even if that ever changes. Both the non-streaming input
/// and the stream's partial_json go through here, so both are valid JSON.
fn tool_input(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| json!({}))
}

/// One block of the reply content array, in arrival order.
enum OutBlock {
    Thinking(String),
    Text(String),
    ToolUse(Value),
}

/// Builds the non-streaming Message reply from the chat pipeline pieces.
/// Consecutive pieces of one kind join into one block, so a reply that alternates
/// text and tool calls keeps the order the model produced.
fn message_body(
    head: &ReplyHead,
    pieces: &[Piece],
    finish: FinishReason,
    usage: Usage,
    matched_stop: Option<String>,
) -> Value {
    let mut blocks: Vec<OutBlock> = Vec::new();
    for piece in pieces {
        let block = match piece {
            Piece::Reasoning(text) => OutBlock::Thinking(text.clone()),
            Piece::Content(text) => OutBlock::Text(text.clone()),
            Piece::Call(call) => OutBlock::ToolUse(json!({
                "type": "tool_use", "id": call.id, "name": call.name,
                "input": tool_input(&call.arguments),
            })),
        };
        match (blocks.last_mut(), block) {
            (Some(OutBlock::Text(held)), OutBlock::Text(more)) => held.push_str(&more),
            (Some(OutBlock::Thinking(held)), OutBlock::Thinking(more)) => held.push_str(&more),
            (_, block) => blocks.push(block),
        }
    }
    let content: Vec<Value> = blocks
        .into_iter()
        .map(|block| match block {
            OutBlock::Thinking(text) => {
                json!({"type": "thinking", "thinking": text, "signature": ""})
            }
            OutBlock::Text(text) => json!({"type": "text", "text": text}),
            OutBlock::ToolUse(value) => value,
        })
        .collect();
    let (stop_reason, stop_sequence) = stop_fields(finish, matched_stop.as_deref());
    json!({
        "id": message_id(head),
        "type": "message",
        "role": "assistant",
        "model": head.model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": anthropic_usage(&usage, cache_fields_on()),
    })
}

/// An Anthropic request that passed validation, expressed in the vocabulary of the chat
/// pipeline. The reply translation reads the matched stop from the running job.
#[derive(Debug)]
pub(crate) struct AnthropicRequest {
    /// The converted request for the chat pipeline.
    pub(crate) input: ChatInput,
}

/// Parses and validates one Anthropic request body.
///
/// The Anthropic vocabulary is checked here; every failure is a `BadRequest` that names
/// the field. The result feeds the chat submit path unchanged.
pub(crate) fn parse_anthropic(body: &[u8]) -> Result<AnthropicRequest, ServerError> {
    parse_body(body, ParseMode::Messages)
}

/// Where a parsed Anthropic body is headed. The routes share every field;
/// only `max_tokens` (required on /v1/messages, accepted-and-ignored on
/// count_tokens) and `stream` (count_tokens never streams) differ.
#[derive(Clone, Copy)]
enum ParseMode {
    Messages,
    CountTokens,
}

/// Parses a count_tokens body: every shared field behaves exactly as on
/// /v1/messages; `max_tokens` may be omitted (a provided value is accepted
/// and ignored) and `stream` is validated then ignored. Feeds `count_reply`.
fn parse_count_tokens(body: &[u8]) -> Result<AnthropicRequest, ServerError> {
    parse_body(body, ParseMode::CountTokens)
}

fn parse_body(body: &[u8], mode: ParseMode) -> Result<AnthropicRequest, ServerError> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|error| bad(format!("request body is not valid JSON: {error}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| bad("request body must be a JSON object"))?;
    let route = match mode {
        ParseMode::Messages => "/v1/messages",
        ParseMode::CountTokens => "/v1/messages/count_tokens",
    };
    for key in object.keys() {
        if !TOP_LEVEL_FIELDS.contains(&key.as_str()) {
            warn_unknown_field(route, key);
        }
    }
    let model = required_string(object, "model", "")?;
    let max_tokens = match mode {
        ParseMode::Messages => {
            count(object, "max_tokens", 1)?.ok_or_else(|| bad("max_tokens is required"))?
        }
        // Not a count_tokens parameter: accepted and ignored. The ChatInput
        // still needs a bound, so an omitted value defaults to 1.
        ParseMode::CountTokens => count(object, "max_tokens", 1)?.unwrap_or(1),
    };
    let mut messages = anthropic_messages(route, object.get("messages"))?;
    if let Some(system) = object.get("system").filter(|system| !system.is_null()) {
        let content = system_content(system, "system")?;
        messages.insert(
            0,
            PromptMessage {
                role: Role::System,
                content: Some(content),
                reasoning_content: None,
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            },
        );
    }
    let mut tools = anthropic_tools(object.get("tools"))?;
    anthropic_tool_choice(object.get("tool_choice"), &mut tools)?;
    let sampling = RequestedSampling {
        temperature: number(object, "temperature")?,
        top_p: number(object, "top_p")?,
        top_k: count(object, "top_k", 0)?,
    };
    // The same bounds as the chat path: the defaults of the engine that serves the
    // request apply in the handler.
    SamplingParams::from_request(
        sampling.temperature,
        sampling.top_p,
        sampling.top_k,
        &SamplingDefaults::CONTRACT,
    )?;
    Ok(AnthropicRequest {
        input: ChatInput {
            model: Some(model),
            engine: None,
            session: None,
            messages,
            tools,
            thinking: anthropic_thinking(object.get("thinking"))?,
            stops: stop_sequences(object.get("stop_sequences"))?,
            sampling,
            max_tokens: u32::try_from(max_tokens).unwrap_or(u32::MAX),
            // A non-boolean stream is a 400 on both routes; on count_tokens
            // the value itself is ignored — it never streams.
            stream: match (mode, flag(object, "stream")?) {
                (ParseMode::Messages, stream) => stream,
                (ParseMode::CountTokens, _) => false,
            },
        },
    })
}

/// The top-level members this endpoint reads. Every other member is dropped with one
/// warning per name per process.
const TOP_LEVEL_FIELDS: &[&str] = &[
    "model",
    "messages",
    "max_tokens",
    "system",
    "stop_sequences",
    "temperature",
    "top_p",
    "top_k",
    "stream",
    "thinking",
    "tools",
    "tool_choice",
    "metadata",
];

/// How many distinct unknown field names one process remembers. Past the cap, new
/// names are dropped silently: a client that sends many unique names inside 16 MiB
/// bodies must not grow process memory or the log without bound.
const FIELD_WARN_CAP: usize = 128;
/// How many characters of one field name the warning prints, so a crafted name
/// cannot forge log lines with newlines or escape codes.
const FIELD_WARN_CHARS: usize = 64;

fn warn_unknown_field(route: &str, name: &str) {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> =
        std::sync::OnceLock::new();
    let seen = SEEN.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
    let mut seen = seen.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if seen.len() >= FIELD_WARN_CAP {
        return;
    }
    let short: String = name.chars().take(FIELD_WARN_CHARS).collect();
    if seen.insert(short.clone()) {
        eprintln!("{route}: ignoring unknown request field {short:?}");
    }
}

fn anthropic_messages(
    route: &str,
    value: Option<&Value>,
) -> Result<Vec<PromptMessage>, ServerError> {
    let items = value
        .and_then(Value::as_array)
        .ok_or_else(|| bad("messages must be a non-empty array"))?;
    if items.is_empty() {
        return Err(bad("messages must be a non-empty array"));
    }
    let mut messages = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let at = format!("messages[{index}]");
        let object = item
            .as_object()
            .ok_or_else(|| bad(format!("{at} must be an object")))?;
        for key in object.keys() {
            if !MESSAGE_FIELDS.contains(&key.as_str()) {
                warn_unknown_field(route, key);
            }
        }
        let role = required_string(object, "role", &format!("{at}."))?;
        let content = object
            .get("content")
            .filter(|content| !content.is_null())
            .ok_or_else(|| bad(format!("{at}.content is required")))?;
        match role.as_str() {
            "user" => {
                let mut added = user_messages(content, &at)?;
                if added.is_empty() {
                    return Err(bad(format!("{at}.content must hold at least one block")));
                }
                messages.append(&mut added);
            }
            "assistant" => messages.push(assistant_message(content, &at)?),
            "system" => messages.push(PromptMessage {
                role: Role::System,
                content: Some(system_content(content, &at)?),
                reasoning_content: None,
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            }),
            other => {
                return Err(bad(format!(
                    "{at}.role must be user, assistant, or system, not {other}"
                )));
            }
        }
    }
    Ok(messages)
}

/// The members of one message object that this endpoint reads.
const MESSAGE_FIELDS: &[&str] = &["role", "content"];

/// System content: a string, or an array of text blocks. Plain string
/// items are accepted as an extension and lift to text blocks here, so the
/// shared walker stays strict for every other content kind.
/// `cache_control` is accepted and ignored on every block.
fn system_content(value: &Value, at: &str) -> Result<String, ServerError> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Array(items) => {
            let blocks: Vec<Value> = items
                .iter()
                .map(|item| match item {
                    Value::String(text) => json!({"type": "text", "text": text}),
                    other => other.clone(),
                })
                .collect();
            text_blocks(&blocks, at)
        }
        _ => Err(bad(format!(
            "{at} must be a string or an array of text blocks"
        ))),
    }
}

/// An empty block holds nothing. It never joins the list, so joining
/// cannot manufacture content from separators.
fn push_text(texts: &mut Vec<String>, text: String) {
    if !text.is_empty() {
        texts.push(text);
    }
}

/// Joins text blocks with a blank line, as the conformance list requires. Media
/// blocks become notes; nothing else exists in system content.
fn text_blocks(blocks: &[Value], at: &str) -> Result<String, ServerError> {
    let mut texts = Vec::new();
    for block in blocks {
        let object = block
            .as_object()
            .ok_or_else(|| bad(format!("{at} blocks must be objects")))?;
        match optional_string(object, "type", &format!("{at}."))?.as_deref() {
            Some("text") => {
                let text = required_string(object, "text", &format!("{at}."))?;
                push_text(&mut texts, text);
            }
            Some("image") => push_text(&mut texts, IMAGE_NOTE.to_string()),
            Some("document") => {
                push_text(&mut texts, document_note(object, at)?);
            }
            Some(other) => return Err(bad(format!("{at} has an unsupported block type {other}"))),
            None => return Err(bad(format!("{at} blocks need a type"))),
        }
    }
    Ok(texts.join("\n\n"))
}

const IMAGE_NOTE: &str = "[image omitted: this model reads text only]";
const DOCUMENT_NOTE: &str = "[document omitted: this model reads text only]";

/// A document with a text source carries its text; every other source becomes a note.
fn document_note(object: &Map<String, Value>, at: &str) -> Result<String, ServerError> {
    let source = object
        .get("source")
        .and_then(Value::as_object)
        .ok_or_else(|| bad(format!("{at} document source must be an object")))?;
    if optional_string(source, "type", &format!("{at} document source."))?.as_deref()
        == Some("text")
    {
        return required_string(source, "data", &format!("{at} document source."));
    }
    Ok(DOCUMENT_NOTE.to_string())
}

/// User content: a string, or an array of text, image, document, and `tool_result`
/// blocks. A `tool_result` becomes its own message in the chat vocabulary, in order.
fn user_messages(content: &Value, at: &str) -> Result<Vec<PromptMessage>, ServerError> {
    let blocks = match content {
        Value::String(text) => {
            if text.is_empty() {
                return Err(bad(format!("{at}.content must not be empty")));
            }
            return Ok(vec![plain_message(Role::User, text.clone())]);
        }
        Value::Array(blocks) => blocks,
        _ => {
            return Err(bad(format!(
                "{at}.content must be a string or an array of blocks"
            )));
        }
    };
    let mut messages = Vec::new();
    let mut texts = Vec::new();
    for block in blocks {
        let object = block
            .as_object()
            .ok_or_else(|| bad(format!("{at} blocks must be objects")))?;
        let block_at = format!("{at}.");
        match optional_string(object, "type", &block_at)?.as_deref() {
            Some("text") => {
                let text = required_string(object, "text", &block_at)?;
                push_text(&mut texts, text);
            }
            Some("image") => push_text(&mut texts, IMAGE_NOTE.to_string()),
            Some("document") => {
                push_text(&mut texts, document_note(object, at)?);
            }
            Some("tool_result") => {
                if !texts.is_empty() {
                    messages.push(plain_message(Role::User, texts.join("\n\n")));
                    texts.clear();
                }
                messages.push(tool_result_message(object, at)?);
            }
            Some(other) => return Err(bad(format!("{at} has an unsupported block type {other}"))),
            None => return Err(bad(format!("{at} blocks need a type"))),
        }
    }
    if !texts.is_empty() {
        messages.push(plain_message(Role::User, texts.join("\n\n")));
    }
    Ok(messages)
}

fn tool_result_message(
    object: &Map<String, Value>,
    at: &str,
) -> Result<PromptMessage, ServerError> {
    let tool_use_id = required_string(object, "tool_use_id", &format!("{at}."))?;
    let content = match object.get("content") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(blocks)) => text_blocks(blocks, &format!("{at}.content"))?,
        Some(_) => {
            return Err(bad(format!(
                "{at}.tool_result content must be a string or an array of blocks"
            )));
        }
    };
    let content = match object.get("is_error") {
        None | Some(Value::Null) | Some(Value::Bool(false)) => content,
        Some(Value::Bool(true)) => format!("Error: {content}"),
        Some(_) => return Err(bad(format!("{at}.is_error must be a boolean"))),
    };
    Ok(PromptMessage {
        role: Role::Tool,
        content: Some(content),
        reasoning_content: None,
        name: None,
        tool_call_id: Some(tool_use_id),
        tool_calls: Vec::new(),
    })
}

/// Assistant content: a string, or an array of text, `thinking`, and `tool_use` blocks.
/// Thinking blocks ride as `reasoning_content`; the serving template decides
/// which turns replay them. A template without `think_replay` never sees the
/// field — the route gate drops it before the render — and the Qwen grammar
/// strips it at the template as a second layer (`ENGINESRV_THINK_REPLAY=0`
/// rolls the replay default back to the old bytes).
fn assistant_message(content: &Value, at: &str) -> Result<PromptMessage, ServerError> {
    let blocks = match content {
        Value::String(text) => {
            if text.is_empty() {
                return Err(bad(format!("{at}.content must not be empty")));
            }
            return Ok(plain_message(Role::Assistant, text.clone()));
        }
        Value::Array(blocks) => blocks,
        _ => {
            return Err(bad(format!(
                "{at}.content must be a string or an array of blocks"
            )));
        }
    };
    let mut texts = Vec::new();
    let mut thinking = String::new();
    let mut tool_calls = Vec::new();
    for block in blocks {
        let object = block
            .as_object()
            .ok_or_else(|| bad(format!("{at} blocks must be objects")))?;
        let block_at = format!("{at}.");
        match optional_string(object, "type", &block_at)?.as_deref() {
            Some("text") => {
                let text = required_string(object, "text", &block_at)?;
                push_text(&mut texts, text);
            }
            Some("thinking" | "redacted_thinking") => {
                // A missing or non-string thinking field is simply absent —
                // the pre-replay contract ignored such blocks without
                // reading them, and flags off must reject nothing new.
                let text = object.get("thinking").and_then(Value::as_str);
                if let Some(text) = text
                    && !text.is_empty()
                {
                    if !thinking.is_empty() {
                        // Foreign clients may fold several thinking blocks;
                        // the engine emits one, so a paragraph break keeps
                        // their boundaries readable.
                        thinking.push_str("\n\n");
                    }
                    thinking.push_str(text);
                }
            }
            Some("tool_use") => {
                let id = required_string(object, "id", &block_at)?;
                let name = required_string(object, "name", &block_at)?;
                let input = match object.get("input") {
                    None | Some(Value::Null) => "{}".to_string(),
                    Some(value @ Value::Object(_)) => value.to_string(),
                    Some(_) => return Err(bad(format!("{at}.tool_use input must be an object"))),
                };
                tool_calls.push(ToolCallSpec {
                    id,
                    name,
                    arguments: input,
                });
            }
            Some(other) => return Err(bad(format!("{at} has an unsupported block type {other}"))),
            None => return Err(bad(format!("{at} blocks need a type"))),
        }
    }
    // Empty text blocks never join the list, so an array can hold nothing
    // usable (an empty array, only empty text). That is no assistant turn at
    // all, so it is rejected instead of sending a message with neither
    // content nor tool calls.
    if tool_calls.is_empty() && texts.is_empty() {
        return Err(bad(format!(
            "{at}.content must hold at least one text or tool_use block"
        )));
    }
    Ok(PromptMessage {
        role: Role::Assistant,
        content: if texts.is_empty() {
            None
        } else {
            Some(texts.join("\n\n"))
        },
        reasoning_content: (!thinking.is_empty()).then_some(thinking),
        name: None,
        tool_call_id: None,
        tool_calls,
    })
}

fn plain_message(role: Role, content: String) -> PromptMessage {
    PromptMessage {
        role,
        content: Some(content),
        reasoning_content: None,
        name: None,
        tool_call_id: None,
        tool_calls: Vec::new(),
    }
}

/// Anthropic tools wrap into the chat template shape: `input_schema` becomes
/// `function.parameters`, and the description rides along when the client sent one.
fn anthropic_tools(value: Option<&Value>) -> Result<Vec<ToolSpec>, ServerError> {
    // Anthropic clients send explicit nulls for omitted optional fields.
    let value = value.filter(|value| !value.is_null());
    let Some(items) = value else {
        return Ok(Vec::new());
    };
    let array = items
        .as_array()
        .ok_or_else(|| bad("tools must be an array"))?;
    let mut tools = Vec::new();
    for item in array {
        let object = item
            .as_object()
            .ok_or_else(|| bad("tools entries must be objects"))?;
        let name = required_string(object, "name", "tools.")?;
        let schema = object
            .get("input_schema")
            .and_then(Value::as_object)
            .ok_or_else(|| bad("tools.input_schema must be an object"))?;
        let mut function = Map::new();
        function.insert("name".into(), json!(name));
        let description = match object.get("description") {
            None | Some(Value::Null) => None,
            Some(Value::String(text)) => Some(text.clone()),
            Some(_) => return Err(bad("tools.description must be a string")),
        };
        if let Some(description) = description {
            function.insert("description".into(), json!(description));
        }
        function.insert("parameters".into(), Value::Object(schema.clone()));
        tools.push(ToolSpec(json!({
            "type": "function",
            "function": function,
        })));
    }
    Ok(tools)
}

/// `auto` keeps the tools, `none` drops them, and `any` or `tool` needs forced tool
/// calls, which the engine cannot do yet. `disable_parallel_tool_use` is accepted and
/// ignored: the engine has no parallel-call behavior to disable.
fn anthropic_tool_choice(
    value: Option<&Value>,
    tools: &mut Vec<ToolSpec>,
) -> Result<(), ServerError> {
    let value = value.filter(|value| !value.is_null());
    let Some(choice) = value else {
        return Ok(());
    };
    let object = choice
        .as_object()
        .ok_or_else(|| bad("tool_choice must be an object"))?;
    if object
        .get("disable_parallel_tool_use")
        .filter(|flag| !flag.is_null())
        .is_some_and(|flag| !flag.is_boolean())
    {
        return Err(bad(
            "tool_choice.disable_parallel_tool_use must be a boolean",
        ));
    }
    match optional_string(object, "type", "tool_choice.")?.as_deref() {
        Some("auto") => Ok(()),
        Some("none") => {
            tools.clear();
            Ok(())
        }
        Some("any" | "tool") => Err(bad(
            "tool_choice any and tool need forced tool calls, which are not available",
        )),
        Some(other) => Err(bad(format!(
            "tool_choice.type must be auto, none, any, or tool, not {other}"
        ))),
        None => Err(bad("tool_choice.type is required")),
    }
}

/// `thinking: {type: enabled|adaptive|disabled}`. Adaptive is Claude Code's
/// default and maps to enabled. The budget is checked for shape only; the
/// engine's own thinking control decides what the budget means.
fn anthropic_thinking(value: Option<&Value>) -> Result<Thinking, ServerError> {
    let value = value.filter(|value| !value.is_null());
    let Some(value) = value else {
        return Ok(Thinking::Default);
    };
    let object = value
        .as_object()
        .ok_or_else(|| bad("thinking must be an object"))?;
    if object
        .get("budget_tokens")
        .filter(|budget| !budget.is_null())
        .is_some_and(|budget| budget.as_u64().map(|tokens| tokens == 0).unwrap_or(true))
    {
        return Err(bad(
            "thinking.budget_tokens must be an integer greater than zero",
        ));
    }
    match optional_string(object, "type", "thinking.")?.as_deref() {
        // Claude Code's default mode is adaptive: the model decides. The
        // engine's own thinking control owns the budget either way.
        Some("enabled" | "adaptive") => Ok(Thinking::On),
        Some("disabled") => Ok(Thinking::Off),
        Some(other) => Err(bad(format!(
            "thinking.type must be enabled, adaptive, or disabled, not {other}"
        ))),
        None => Err(bad("thinking.type is required")),
    }
}

fn stop_sequences(value: Option<&Value>) -> Result<Vec<String>, ServerError> {
    let value = value.filter(|value| !value.is_null());
    let Some(items) = value else {
        return Ok(Vec::new());
    };
    let array = items
        .as_array()
        .ok_or_else(|| bad("stop_sequences must be an array of strings"))?;
    let mut stops = Vec::new();
    for item in array {
        let text = item
            .as_str()
            .ok_or_else(|| bad("stop_sequences must be an array of strings"))?;
        if text.is_empty() {
            return Err(bad("stop_sequences must not include an empty string"));
        }
        stops.push(text.to_string());
    }
    Ok(stops)
}

/// Reads an optional number member.
fn number(object: &Map<String, Value>, field: &str) -> Result<Option<f64>, ServerError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => Ok(number.as_f64()),
        Some(_) => Err(bad(format!("{field} must be a number"))),
    }
}

/// Reads an optional whole number member that must be at least `minimum`.
fn count(
    object: &Map<String, Value>,
    field: &str,
    minimum: u64,
) -> Result<Option<u64>, ServerError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => match number.as_u64() {
            Some(found) if found >= minimum => Ok(Some(found)),
            _ => Err(bad(format!(
                "{field} must be an integer of at least {minimum}"
            ))),
        },
        Some(_) => Err(bad(format!(
            "{field} must be an integer of at least {minimum}"
        ))),
    }
}

/// Reads the optional `stream` flag.
fn flag(object: &Map<String, Value>, field: &str) -> Result<bool, ServerError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Bool(value)) => Ok(*value),
        Some(_) => Err(bad(format!("{field} must be a boolean"))),
    }
}

/// The state of one Messages stream between two event batches.
struct Streaming {
    head: ReplyHead,
    running: RunningJob,
    phase: Phase,
    /// The index and kind of the open content block, if one is open.
    open: Option<(u64, BlockKind)>,
    /// The index of the next content block.
    next_index: u64,
}

/// The phases of one stream: message_start once, the piece events, then the
/// closing events.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Start,
    Running,
    Finished,
}

/// The kind of the open content block. A tool_use block opens and closes with
/// its call, so it never stays open in `Streaming.open`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Thinking,
    Text,
}

/// The SSE response of a streamed reply: message_start, the content_block
/// events of every piece, message_delta with the stop and the usage totals,
/// message_stop. A failure after the headers is one `event: error`.
fn sse_respond(head: ReplyHead, running: RunningJob, keep_alive: Duration) -> Response {
    let state = Streaming {
        head,
        running,
        phase: Phase::Start,
        open: None,
        next_index: 0,
    };
    let events = futures::stream::unfold(state, advance)
        .flat_map(|events| futures::stream::iter(events.into_iter().map(Ok::<Event, Infallible>)));
    Sse::new(events)
        .keep_alive(KeepAlive::new().interval(keep_alive).text("keep-alive"))
        .into_response()
}

/// One batch of events per poll. The stream ends once the phase is Finished.
async fn advance(mut state: Streaming) -> Option<(Vec<Event>, Streaming)> {
    let events = match state.phase {
        Phase::Finished => return None,
        Phase::Start => {
            state.phase = Phase::Running;
            vec![named(
                "message_start",
                message_start(&state.head, state.running.prompt_tokens()),
            )]
        }
        Phase::Running => match state.running.next().await {
            Ok(Step::Pieces(pieces)) => piece_events(&mut state, &pieces),
            Ok(Step::Done {
                pieces,
                finish,
                usage,
            }) => {
                state.phase = Phase::Finished;
                let mut events = piece_events(&mut state, &pieces);
                close_block(&mut state, &mut events);
                let stop = state.running.matched_stop();
                let (reason, sequence) = stop_fields(finish, stop.as_deref());
                events.push(named(
                    "message_delta",
                    json!({
                        "type": "message_delta",
                        "delta": {"stop_reason": reason, "stop_sequence": sequence},
                        "usage": anthropic_usage(&usage, cache_fields_on()),
                    }),
                ));
                events.push(named("message_stop", json!({"type": "message_stop"})));
                events
            }
            Ok(Step::Aborted) => {
                // The engine ended the job without a finish (a cancel, an
                // overflowed output): the client gets no more events.
                state.phase = Phase::Finished;
                Vec::new()
            }
            Err(error) => {
                // The headers are out, so the failure is one event in the
                // Anthropic error shape; the stream then ends.
                state.phase = Phase::Finished;
                vec![named("error", anthropic_error_body(&error))]
            }
        },
    };
    Some((events, state))
}

/// Maps pieces to content_block events. Consecutive pieces of one kind share
/// one block, like the blocks of the non-streaming reply.
fn piece_events(state: &mut Streaming, pieces: &[Piece]) -> Vec<Event> {
    let mut events = Vec::new();
    for piece in pieces {
        match piece {
            Piece::Reasoning(text) => {
                let index = ensure_block(state, &mut events, BlockKind::Thinking);
                events.push(named(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "thinking_delta", "thinking": text},
                    }),
                ));
            }
            Piece::Content(text) => {
                let index = ensure_block(state, &mut events, BlockKind::Text);
                events.push(named(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": text},
                    }),
                ));
            }
            Piece::Call(call) => {
                // A tool call is one whole block: it opens, streams its
                // arguments as one delta, and closes with this piece.
                close_block(state, &mut events);
                let index = state.next_index;
                state.next_index += 1;
                events.push(named(
                    "content_block_start",
                    json!({
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use", "id": call.id, "name": call.name,
                            "input": {},
                        },
                    }),
                ));
                events.push(named(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": tool_input(&call.arguments).to_string(),
                        },
                    }),
                ));
                events.push(block_stop(index));
            }
        }
    }
    events
}

/// Opens a block of `kind`, closing the open one first when the kind changes.
fn ensure_block(state: &mut Streaming, events: &mut Vec<Event>, kind: BlockKind) -> u64 {
    if let Some((index, open)) = state.open {
        if open == kind {
            return index;
        }
        events.push(block_stop(index));
    }
    let index = state.next_index;
    state.next_index += 1;
    state.open = Some((index, kind));
    let content_block = match kind {
        BlockKind::Thinking => json!({
            "type": "thinking", "thinking": "", "signature": "",
        }),
        BlockKind::Text => json!({"type": "text", "text": ""}),
    };
    events.push(named(
        "content_block_start",
        json!({
            "type": "content_block_start",
            "index": index,
            "content_block": content_block,
        }),
    ));
    index
}

/// Closes the open block, if one is open.
fn close_block(state: &mut Streaming, events: &mut Vec<Event>) {
    if let Some((index, _)) = state.open.take() {
        events.push(block_stop(index));
    }
}

fn block_stop(index: u64) -> Event {
    named(
        "content_block_stop",
        json!({"type": "content_block_stop", "index": index}),
    )
}

/// The Anthropic usage shape. `input_tokens` EXCLUDES the cache-read
/// tokens (the Anthropic contract), and input + cache_read +
/// cache_creation sums to the whole prompt whenever the engine reports
/// reused <= prompt — which the workers guarantee (worker.rs and
/// batch_worker.rs report reuse only from a prefix of the rendered
/// prompt), so a client that sums the three (Claude Code does) sees the
/// same context size as the legacy single field; a fabricated
/// over-report is clamped, never let through to break the sum. A reader
/// that consumes `input_tokens` ALONE now sees less than the whole
/// prompt — that is the contract, not a bug. The split is the shipped
/// default (wire::CACHE_USAGE_DEFAULT; the redone Claude Code receipts
/// are recorded in this slice's flip commit message); every value other
/// than the exact 1/true/on forces the legacy two-field shape, empty
/// and non-Unicode included.
fn anthropic_usage(usage: &Usage, cache_fields: bool) -> serde_json::Value {
    let output_tokens = usage.completion_tokens();
    if !cache_fields {
        return serde_json::json!({
            "input_tokens": usage.prompt_tokens(),
            "output_tokens": output_tokens,
        });
    }
    serde_json::json!({
        "input_tokens": usage.prompt_tokens().saturating_sub(usage.reused_tokens()),
        "cache_read_input_tokens": usage
            .reused_tokens()
            .min(usage.prompt_tokens()),
        "cache_creation_input_tokens": 0,
        "output_tokens": output_tokens,
    })
}

/// The message_start usage: the cache split is not known until the job
/// finishes (reused arrives with the final usage), so the start event
/// carries the full prompt with zero cache fields — input + 0 + 0 equals
/// the legacy input exactly. The flag is a parameter (like
/// `anthropic_usage`) so both shapes are testable without the process
/// environment.
fn message_start_usage(input_tokens: u32, cache_fields: bool) -> serde_json::Value {
    if !cache_fields {
        return serde_json::json!({"input_tokens": input_tokens, "output_tokens": 0});
    }
    serde_json::json!({
        "input_tokens": input_tokens,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "output_tokens": 0,
    })
}

/// The message_start event. The input count is known when the job is
/// accepted; the output count and the final totals arrive with message_delta.
fn message_start(head: &ReplyHead, input_tokens: u32) -> Value {
    json!({
        "type": "message_start",
        "message": {
            "id": message_id(head),
            "type": "message",
            "role": "assistant",
            "model": head.model,
            "content": [],
            "stop_reason": null,
            "stop_sequence": null,
            "usage": message_start_usage(input_tokens, cache_fields_on()),
        },
    })
}

/// One named SSE event with a JSON body. The bodies are `json!` values, so
/// serialization cannot fail in practice; the fallback keeps the event shape
/// and logs like any other engine failure.
fn named(name: &str, body: Value) -> Event {
    match Event::default().event(name).json_data(body) {
        Ok(event) => event,
        Err(error) => {
            let failure = ServerError::Eng(EngError::Backend {
                operation: "serialize",
                message: error.to_string(),
            });
            Event::default()
                .event("error")
                .data(anthropic_error_body(&failure).to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Usage;
    use super::anthropic_usage;
    use super::message_start_usage;
    use super::{AnthropicRequest, is_length_limit, parse_anthropic};

    use crate::EngError;
    use crate::sched::worker::Engine;
    use crate::server::error::ServerError;
    use crate::server::router::router;
    use crate::server::state::AppState;
    use crate::server::testkit::{self, Ending, SseEvent};
    use crate::tokenizer::{PromptMessage, Role, Thinking};
    use axum::Router;
    use axum::body::Body;
    use axum::http::StatusCode;
    use axum::http::header;
    use serde_json::{Value, json};
    use std::sync::Arc;

    #[test]
    fn model_output_is_a_sanitized_anthropic_api_error() {
        let error = ServerError::Eng(EngError::ModelOutput("private nonfinite logits".into()));
        assert_eq!(
            super::error_response(&error).status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            super::anthropic_error_body(&error),
            json!({
                "type": "error",
                "error": {"type": "api_error", "message": super::BACKEND_MESSAGE},
            })
        );
    }

    /// A router whose entry reuses prefixes and renders with the pinned
    /// template bundle, so a second conversation restores the root. Same
    /// construction as chat.rs's `reuse_app_with_bundle`; that one takes
    /// the bundle as a parameter, this one always pins the test bundle.
    fn reuse_app(engine: &Engine) -> Router {
        testkit::wait_for_state(&engine.handle(), crate::sched::worker::EngineState::Ready);
        let mut entry = testkit::entry("e1", "model-a", engine, 4_096);
        entry.prefix_reuse = true;
        entry.bundle = testkit::bundle_with_pinned_template();
        router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])))
    }

    /// R1 pin: the non-stream Messages usage splits the cached tokens out
    /// of input_tokens. input + cache_read + cache_creation sums to the
    /// whole prompt (21) on every row the workers produce, so a client
    /// that sums the three fields sees the same context size as the
    /// legacy input_tokens.
    #[test]
    fn the_messages_usage_splits_the_cache_read_tokens() {
        let usage = Usage::new(21, 7, 14);
        let value = anthropic_usage(&usage, true);
        assert_eq!(value["input_tokens"], 7, "{value}");
        assert_eq!(value["cache_read_input_tokens"], 14, "{value}");
        assert_eq!(value["cache_creation_input_tokens"], 0, "{value}");
        assert_eq!(value["output_tokens"], 7, "{value}");
        let sum = value["input_tokens"].as_u64().unwrap()
            + value["cache_read_input_tokens"].as_u64().unwrap()
            + value["cache_creation_input_tokens"].as_u64().unwrap();
        assert_eq!(sum, 21, "{value}");
    }

    /// R3 pin: a cold row has no reuse — cache_read 0 and input_tokens
    /// exactly the legacy value.
    #[test]
    fn the_messages_usage_of_a_cold_row_is_unchanged_but_carries_the_zero() {
        let usage = Usage::new(21, 7, 0);
        let value = anthropic_usage(&usage, true);
        assert_eq!(value["input_tokens"], 21, "{value}");
        assert_eq!(value["cache_read_input_tokens"], 0, "{value}");
    }

    /// R1 integration (child, switch ON): a warm conversation through
    /// /v1/messages splits the cached tokens out of input_tokens; the
    /// three fields sum to the whole prompt (2,139), and the cold row is
    /// unchanged with a zero. Production only reaches the ON shapes
    /// through the switch or the shipped default, so the body runs in a
    /// fresh child with ENGINESRV_CACHE_USAGE=1 (sentinel + 1-passed
    /// evidence) — the in-process default is pinned elsewhere.
    #[tokio::test]
    async fn a_warm_messages_conversation_splits_the_cache_read_tokens() {
        const CHILD: &str = "ENGINESRV_TEST_WARM_MESSAGES_CHILD";
        if std::env::var_os(CHILD).is_none() {
            testkit::child_env_check(
                &format!(
                    "{}::a_warm_messages_conversation_splits_the_cache_read_tokens",
                    module_path!().split_once("::").unwrap().1
                ),
                (CHILD, "1"),
                Some("1"),
                "child-warm-messages-ok",
            );
            return;
        }
        let engine = testkit::engine_with_prefix_limits(
            [testkit::script("a"), testkit::script("b")].concat(),
            2,
            6,
        );
        let app = reuse_app(&engine);
        let system = "a".repeat(2_100);
        let mut first = request(json!({"system": system.clone(), "max_tokens": 1}));
        first["messages"][0]["content"] = "hi".into();
        let mut second = request(json!({"system": system, "max_tokens": 1}));
        second["messages"][0]["content"] = "yo".into();
        // cache_control stays accepted-and-ignored on every block: the
        // second request carries the system prompt in block form with a
        // marker, and the usage split is unaffected.
        second["system"] = json!([{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"}
        }]);
        let (_, cold) = post(&app, &first).await;
        let (_, warm) = post(&app, &second).await;
        assert_eq!(cold["usage"]["input_tokens"], 2_139, "{cold}");
        assert_eq!(cold["usage"]["cache_read_input_tokens"], 0, "{cold}");
        assert_eq!(warm["usage"]["input_tokens"], 29, "{warm}");
        assert_eq!(warm["usage"]["cache_read_input_tokens"], 2_110, "{warm}");
        let sum = warm["usage"]["input_tokens"].as_u64().unwrap()
            + warm["usage"]["cache_read_input_tokens"].as_u64().unwrap()
            + warm["usage"]["cache_creation_input_tokens"]
                .as_u64()
                .unwrap();
        assert_eq!(sum, 2_139, "{warm}");
        println!("child-warm-messages-ok");
    }

    /// R7 pin: with the switch forcing legacy (any value but the exact
    /// 1/true/on) the usage is the legacy two-field shape byte-for-byte
    /// (to_string, not Value equality — object order is part of the pin).
    #[test]
    fn the_messages_usage_stays_legacy_when_the_switch_is_off() {
        let usage = Usage::new(21, 7, 14);
        assert_eq!(
            anthropic_usage(&usage, false).to_string(),
            r#"{"input_tokens":21,"output_tokens":7}"#,
        );
    }

    /// message_start's usage carries the whole prompt with zero cache
    /// fields on both shapes; the split arrives only in message_delta
    /// (input + cache_read + creation sums to the same whole prompt at
    /// both sites).
    #[test]
    fn the_message_start_usage_carries_the_whole_prompt_on_both_shapes() {
        assert_eq!(
            message_start_usage(2_139, true).to_string(),
            r#"{"input_tokens":2139,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":0}"#,
        );
        assert_eq!(
            message_start_usage(2_139, false).to_string(),
            r#"{"input_tokens":2139,"output_tokens":0}"#,
        );
    }

    /// R9 pin: a fabricated over-report (reused > prompt) saturates and
    /// clamps — cache_read reports at most the whole prompt, so the
    /// three-field sum never exceeds the legacy input. The workers never
    /// produce such a row (reuse comes from a prefix of the rendered
    /// prompt); this pin holds the boundary anyway.
    #[test]
    fn an_over_reported_reuse_saturates_instead_of_underflowing() {
        let value = anthropic_usage(&Usage::new(5, 1, 9), true);
        assert_eq!(value["input_tokens"], 0, "{value}");
        assert_eq!(value["cache_read_input_tokens"], 5, "{value}");
        let sum = value["input_tokens"].as_u64().unwrap()
            + value["cache_read_input_tokens"].as_u64().unwrap()
            + value["cache_creation_input_tokens"].as_u64().unwrap();
        assert_eq!(sum, 5, "{value}");
    }

    /// Integration (child, switch ON): a warm streamed conversation
    /// reports the whole prompt in message_start's usage (cache fields
    /// zero) and the split in message_delta's; the three fields sum to
    /// the same whole prompt (2,139) at both sites, so a client that
    /// merges start then delta (Claude Code) sees one consistent context
    /// size. Runs in a fresh child with ENGINESRV_CACHE_USAGE=1 (sentinel
    /// + 1-passed evidence), as above.
    #[tokio::test]
    async fn a_warm_stream_reports_whole_prompt_in_start_and_the_split_in_delta() {
        const CHILD: &str = "ENGINESRV_TEST_WARM_STREAM_CHILD";
        if std::env::var_os(CHILD).is_none() {
            testkit::child_env_check(
                &format!(
                    "{}::a_warm_stream_reports_whole_prompt_in_start_and_the_split_in_delta",
                    module_path!().split_once("::").unwrap().1
                ),
                (CHILD, "1"),
                Some("1"),
                "child-warm-stream-ok",
            );
            return;
        }
        let engine = testkit::engine_with_prefix_limits(
            [testkit::script("a"), testkit::script("b")].concat(),
            2,
            6,
        );
        let app = reuse_app(&engine);
        let system = "a".repeat(2_100);
        let mut first = stream_request(json!({"max_tokens": 1}));
        first["messages"][0]["content"] = "hi".into();
        first["system"] = system.clone().into();
        let mut second = stream_request(json!({"max_tokens": 1}));
        second["messages"][0]["content"] = "yo".into();
        second["system"] = system.into();
        let _ = stream(&app, &first).await;
        let (_, _, events) = stream(&app, &second).await;
        let start = of(&events, "message_start");
        assert_eq!(start.len(), 1);
        let start_usage = &start[0]["message"]["usage"];
        assert_eq!(start_usage["input_tokens"], 2_139, "{start_usage}");
        assert_eq!(start_usage["cache_read_input_tokens"], 0, "{start_usage}");
        assert_eq!(
            start_usage["cache_creation_input_tokens"], 0,
            "{start_usage}"
        );
        assert_eq!(start_usage["output_tokens"], 0, "{start_usage}");
        let delta = of(&events, "message_delta");
        assert_eq!(delta.len(), 1);
        let delta_usage = &delta[0]["usage"];
        assert_eq!(delta_usage["input_tokens"], 29, "{delta_usage}");
        assert_eq!(
            delta_usage["cache_read_input_tokens"], 2_110,
            "{delta_usage}"
        );
        assert_eq!(
            delta_usage["cache_creation_input_tokens"], 0,
            "{delta_usage}"
        );
        for site in [start_usage, delta_usage] {
            let sum = site["input_tokens"].as_u64().unwrap()
                + site["cache_read_input_tokens"].as_u64().unwrap()
                + site["cache_creation_input_tokens"].as_u64().unwrap();
            assert_eq!(sum, 2_139, "{site}");
        }
        println!("child-warm-stream-ok");
    }

    /// The switch's byte effect on the ANTHROPIC surface through the real
    /// environment, in fresh children — the non-stream message body,
    /// message_start, and message_delta. Hardwiring `true` at any of the
    /// three production `cache_fields_on()` call sites fails the off
    /// child; hardwiring `false` fails the on child. Each child prints a
    /// sentinel; each parent requires the runner's `1 passed` line, so a
    /// child that ran zero tests cannot pass.
    #[tokio::test]
    async fn the_switch_shapes_the_anthropic_usage_bytes_through_the_environment() {
        const CHILD: &str = "ENGINESRV_TEST_ANTHROPIC_SHAPES_CHILD";
        if let Some(mode) = std::env::var_os(CHILD) {
            // Two scripted replies: the non-stream post consumes one, the
            // stream consumes the other (exec_fake refuses an exhausted
            // script), and both must agree on the same token counts.
            let engine = testkit::engine(
                [testkit::script("Recorded."), testkit::script("Recorded.")].concat(),
            );
            let app = app(&engine, 4_096);
            let (_, message) = post(&app, &request(json!({"max_tokens": 1_024}))).await;
            let (_, _, events) = stream(&app, &stream_request(json!({}))).await;
            let input = message["usage"]["input_tokens"]
                .as_u64()
                .expect("input tokens");
            let output = message["usage"]["output_tokens"]
                .as_u64()
                .expect("output tokens");
            assert!(input > 0 && output > 0, "{message}");
            let on = mode == "on";
            let body_expected = if on {
                format!(
                    r#"{{"input_tokens":{input},"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":{output}}}"#
                )
            } else {
                format!(r#"{{"input_tokens":{input},"output_tokens":{output}}}"#)
            };
            assert_eq!(
                message["usage"].to_string(),
                body_expected,
                "message body bytes"
            );
            let start = &of(&events, "message_start")[0]["message"]["usage"];
            let delta = &of(&events, "message_delta")[0]["usage"];
            let start_expected = if on {
                format!(
                    r#"{{"input_tokens":{input},"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":0}}"#
                )
            } else {
                format!(r#"{{"input_tokens":{input},"output_tokens":0}}"#)
            };
            let delta_expected = if on {
                format!(
                    r#"{{"input_tokens":{input},"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":{output}}}"#
                )
            } else {
                format!(r#"{{"input_tokens":{input},"output_tokens":{output}}}"#)
            };
            assert_eq!(start.to_string(), start_expected, "message_start bytes");
            assert_eq!(delta.to_string(), delta_expected, "message_delta bytes");
            println!("child-anthropic-{mode:?}-ok");
            return;
        }
        for (mode, cache) in [("off", "0"), ("on", "1")] {
            // "0"/"1" explicit, never env_remove: these children pin the
            // OFF and ON shapes, not the default (see the wire.rs note).
            testkit::child_env_check(
                &format!(
                    "{}::the_switch_shapes_the_anthropic_usage_bytes_through_the_environment",
                    module_path!().split_once("::").unwrap().1
                ),
                (CHILD, mode),
                Some(cache),
                &format!("child-anthropic-\"{mode}\"-ok"),
            );
        }
    }

    const CALL: &str = "<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>";

    /// A router over one engine named `e1` that serves `model-a`.
    fn app(engine: &Engine, context_limit: u32) -> Router {
        app_serving(engine, context_limit, "model-a")
    }

    /// A router over one engine named `e1` that serves `model`. The captured
    /// client fixtures name their own model; the replay binds it here.
    fn app_serving(engine: &Engine, context_limit: u32, model: &str) -> Router {
        testkit::wait_for_state(&engine.handle(), crate::sched::worker::EngineState::Ready);
        let entry = testkit::entry("e1", model, engine, context_limit);
        router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])))
    }

    /// The base request body with the fields of `extra` merged over it.
    fn request(extra: Value) -> Value {
        let mut body = json!({"model": "model-a", "max_tokens": 1_024,
            "messages": [{"role": "user", "content": "hi"}]});
        for (key, value) in extra.as_object().expect("object").iter() {
            body[key] = value.clone();
        }
        body
    }

    async fn post(app: &Router, body: &Value) -> (StatusCode, Value) {
        let (status, _headers, bytes) =
            testkit::send(app, testkit::post_json("/v1/messages", body)).await;
        let body = if bytes.is_empty() {
            Value::Null
        } else {
            testkit::json_of(&bytes)
        };
        (status, body)
    }

    /// Posts the body with `x-api-key` instead of the bearer header.
    async fn post_with_api_key(app: &Router, body: &Value) -> (StatusCode, Value) {
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("x-api-key", testkit::API_KEY)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("request");
        let (status, _headers, bytes) = testkit::send(app, request).await;
        let body = if bytes.is_empty() {
            Value::Null
        } else {
            testkit::json_of(&bytes)
        };
        (status, body)
    }

    fn error_type(body: &Value) -> &str {
        body["error"]["type"].as_str().expect("error type")
    }

    fn parsed(body: &Value) -> AnthropicRequest {
        parse_anthropic(body.to_string().as_bytes()).expect("parse")
    }

    /// Multiple thinking blocks in one resent assistant turn (the client
    /// folds consecutive blocks) and a redacted block (data, no text) both
    /// translate to one reasoning string joined by a paragraph break, no
    /// markers. Parse-level pin of `assistant_message`'s fold.
    #[test]
    fn multi_block_and_redacted_thinking_translate_to_one_reasoning() {
        let request = parsed(&serde_json::json!({
            "engine": "e",
            "model": "m",
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "first part"},
                    {"type": "thinking", "thinking": "second part"},
                    {"type": "redacted_thinking", "data": "ZWc="},
                    {"type": "thinking", "thinking": null},
                    {"type": "text", "text": "done"}
                ]}
            ]
        }));
        let assistant = request
            .input
            .messages
            .last()
            .expect("the assistant row is present");
        assert_eq!(
            assistant.reasoning_content.as_deref(),
            Some("first part\n\nsecond part"),
            "consecutive thinking blocks join with a paragraph break, redacted adds nothing"
        );
        assert_eq!(assistant.content.as_deref(), Some("done"));
    }

    fn parse_error(body: &Value) -> ServerError {
        parse_anthropic(body.to_string().as_bytes()).expect_err("parse error")
    }

    fn bad_request(error: &ServerError) {
        assert!(
            matches!(error, ServerError::Eng(EngError::BadRequest(_))),
            "expected a bad request, got {error:?}"
        );
    }

    /// One message of the converted input.
    fn message(request: &AnthropicRequest, index: usize) -> &PromptMessage {
        &request.input.messages[index]
    }

    // ── parse: envelope and top-level fields ───────────────────────────────

    #[test]
    fn system_sampling_and_stop_sequences_map_through() {
        let body = request(json!({
            "system": [{"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}],
            "temperature": 0.5, "top_p": 0.9, "top_k": 40,
            "stop_sequences": ["STOP", "END"]
        }));
        let parsed = parsed(&body);
        assert_eq!(parsed.input.messages.len(), 2);
        assert!(matches!(message(&parsed, 0).role, Role::System));
        assert_eq!(message(&parsed, 0).content.as_deref(), Some("be brief"));
        assert_eq!(
            parsed.input.stops,
            vec!["STOP".to_string(), "END".to_string()]
        );
        assert_eq!(parsed.input.sampling.temperature, Some(0.5));
        assert_eq!(parsed.input.sampling.top_p, Some(0.9));
        assert_eq!(parsed.input.sampling.top_k, Some(40));
        assert_eq!(parsed.input.max_tokens, 1_024);
        assert!(!parsed.input.stream);
    }

    #[test]
    fn system_accepts_plain_string_items() {
        // Plain string items in the system array are an accepted extension;
        // they join like text blocks.
        let body = request(json!({
            "system": ["Be brief.", {"type": "text", "text": "stay calm"}]
        }));
        let parsed = parsed(&body);
        assert_eq!(
            message(&parsed, 0).content.as_deref(),
            Some("Be brief.\n\nstay calm")
        );
    }

    #[test]
    fn tool_result_content_stays_strict_about_string_items() {
        // The string lift is a system-only extension: a tool_result content
        // array of plain strings still fails, like every non-system kind.
        let error = parse_error(&request(json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "add", "input": {}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": ["plain string"]}
                ]}
            ]
        })));
        assert!(
            error.message().contains("blocks must be objects"),
            "message names the shape: {}",
            error.message()
        );
    }

    #[test]
    fn max_tokens_is_required_and_positive() {
        for body in [
            json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
            request(json!({"max_tokens": 0})),
            request(json!({"max_tokens": -1})),
            request(json!({"max_tokens": "big"})),
        ] {
            let error = parse_error(&body);
            bad_request(&error);
            assert!(
                error.message().contains("max_tokens"),
                "message names max_tokens: {}",
                error.message()
            );
        }
    }

    #[test]
    fn the_stream_flag_maps_through() {
        let streamed = parsed(&request(json!({"stream": true})));
        assert!(streamed.input.stream);
        assert!(!parsed(&request(json!({}))).input.stream);
    }

    // ── parse: content blocks ──────────────────────────────────────────────

    #[test]
    fn text_blocks_join_with_a_blank_line() {
        let parsed = parsed(&request(json!({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "a"}, {"type": "text", "text": "b"}
            ]}]
        })));
        assert!(matches!(message(&parsed, 0).role, Role::User));
        assert_eq!(message(&parsed, 0).content.as_deref(), Some("a\n\nb"));
    }

    #[test]
    fn tool_use_and_tool_results_map_through() {
        let body = request(json!({
            "messages": [
                {"role": "user", "content": "add 1 and 2"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Adding."},
                    {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"a": 1, "b": 2}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "3"}
                ]},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_2", "name": "fail", "input": {}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_2",
                     "content": "boom", "is_error": true}
                ]}
            ]
        }));
        let parsed = parsed(&body);
        let messages = &parsed.input.messages;
        assert_eq!(messages.len(), 5);
        assert!(matches!(messages[1].role, Role::Assistant));
        assert_eq!(messages[1].content.as_deref(), Some("Adding."));
        assert_eq!(
            serde_json::to_value(&messages[1].tool_calls).expect("tool calls"),
            json!([{"id": "toolu_1", "name": "add", "arguments": "{\"a\":1,\"b\":2}"}])
        );
        assert!(matches!(messages[2].role, Role::Tool));
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("toolu_1"));
        assert_eq!(messages[2].content.as_deref(), Some("3"));
        assert!(matches!(messages[4].role, Role::Tool));
        assert_eq!(messages[4].content.as_deref(), Some("Error: boom"));
    }

    /// The rollback gate pinned against the real Flint template: Flint's
    /// grammar never strips at the template, and its jinja renders
    /// `reasoning_content` as a visible `<think>` block — so without the
    /// gate, flags off included, an Anthropic thinking block would change
    /// Flint's bytes. With the gate, the render equals the no-thinking
    /// render exactly.
    #[test]
    fn the_gate_keeps_flint_bytes_free_of_resent_reasoning() {
        use crate::tokenizer::{ChatTemplate, Thinking};
        let template = ChatTemplate::from_str(include_str!(
            "../../testdata/flint/tokenizer/chat_template.jinja"
        ))
        .expect("real flint template");
        let request = parsed(&serde_json::json!({
            "engine": "e",
            "model": "m",
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "plan"},
                    {"type": "text", "text": "done"}
                ]}
            ]
        }));
        let mut messages = request.input.messages;
        // Without the gate the flint render would carry the think block.
        let ungated = template
            .render(&messages, &[], Thinking::On)
            .expect("ungated flint render");
        assert!(
            ungated.contains("<think>"),
            "flint renders reasoning_content when it reaches the template:\n{ungated}"
        );
        // The gate drops the field before the render.
        for message in &mut messages {
            message.reasoning_content = None;
        }
        let gated = template
            .render(&messages, &[], Thinking::On)
            .expect("gated flint render");
        // The byte-equality pin: the gated render equals an independently
        // constructed no-thinking render of the same conversation.
        let baseline_request = parsed(&serde_json::json!({
            "engine": "e",
            "model": "m",
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "done"}
                ]}
            ]
        }));
        let baseline = template
            .render(&baseline_request.input.messages, &[], Thinking::On)
            .expect("baseline flint render");
        assert_eq!(
            gated, baseline,
            "the gated render equals the no-thinking render byte-for-byte"
        );
    }

    /// The gate wiring, exercised through the routes. The count_tokens post
    /// covers count_reply's gate; the messages twin below covers reply()'s —
    /// the generation path that drives decode and the prefix cache. An app
    /// whose entry serves the REAL Flint template (its grammar never strips
    /// reasoning at the template) must see the same tokens with and without
    /// a resent thinking block. If either gate call is removed, the matching
    /// pin fails — the ungated Flint render carries a visible <think> block
    /// and counts more.
    #[tokio::test]
    async fn count_tokens_gates_resent_reasoning_on_the_flint_bundle() {
        let engine = testkit::engine(testkit::script("ok"));
        let mut entry = testkit::entry("e1", "model-a", &engine, 4_096);
        entry.bundle = testkit::bundle_with_flint_template();
        let app = router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])));
        let base = json!({
            "model": "model-a",
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "done"}
                ]},
                {"role": "user", "content": "again"}
            ]
        });
        let with_thinking = json!({
            "model": "model-a",
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "plan it through", "signature": "s"},
                    {"type": "text", "text": "done"}
                ]},
                {"role": "user", "content": "again"}
            ]
        });
        let (plain_status, plain_body) = post_count_tokens(&app, &base).await;
        assert_eq!(plain_status, StatusCode::OK, "{plain_body}");
        let (think_status, think_body) = post_count_tokens(&app, &with_thinking).await;
        assert_eq!(think_status, StatusCode::OK, "{think_body}");
        assert_eq!(
            plain_body["input_tokens"], think_body["input_tokens"],
            "the gate drops resent reasoning before the flint render"
        );
    }

    /// The reply() half of the gate wiring: the same Flint app, the messages
    /// route, and equal usage.input_tokens with and without a resent
    /// thinking block. Removing the gate call in reply() alone fails this
    /// pin: the ungated render would carry a visible <think> block and count
    /// more input tokens on the generation path.
    #[tokio::test]
    async fn messages_gate_drops_resent_reasoning_on_the_flint_bundle() {
        // The fake script serves one job, so each post gets its own app on
        // the same bundle bytes.
        let app_of = || {
            let engine = testkit::engine(testkit::script("ok"));
            let mut entry = testkit::entry("e1", "model-a", &engine, 4_096);
            entry.bundle = testkit::bundle_with_flint_template();
            router(Arc::new(AppState::new(testkit::API_KEY, vec![entry])))
        };
        let base = json!({
            "model": "model-a",
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "done"}
                ]},
                {"role": "user", "content": "again"}
            ]
        });
        let with_thinking = json!({
            "model": "model-a",
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "plan it through", "signature": "s"},
                    {"type": "text", "text": "done"}
                ]},
                {"role": "user", "content": "again"}
            ]
        });
        let (plain_status, plain_body) = post(&app_of(), &base).await;
        assert_eq!(plain_status, StatusCode::OK, "{plain_body}");
        let (think_status, think_body) = post(&app_of(), &with_thinking).await;
        assert_eq!(think_status, StatusCode::OK, "{think_body}");
        assert_eq!(
            plain_body["usage"]["input_tokens"], think_body["usage"]["input_tokens"],
            "reply()'s gate drops resent reasoning before the flint render"
        );
    }

    #[test]
    fn assistant_thinking_blocks_parse_with_their_content_without_replay() {
        let parsed = parsed(&request(json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "let me think", "signature": "sig"},
                    {"type": "text", "text": "hello"}
                ]},
                {"role": "user", "content": "again"}
            ]
        })));
        let messages = &parsed.input.messages;
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1].content.as_deref(), Some("hello"));
        assert!(messages[1].tool_calls.is_empty());
    }

    #[test]
    fn images_and_documents_become_notes() {
        let parsed = parsed(&request(json!({
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "xx"}},
                {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "doc text"}},
                {"type": "document", "source": {"type": "url", "url": "https://example.com/doc.pdf"}}
            ]}]
        })));
        assert_eq!(
            message(&parsed, 0).content.as_deref(),
            Some(
                "[image omitted: this model reads text only]\n\ndoc text\n\n[document omitted: this model reads text only]"
            )
        );
    }

    #[test]
    fn mid_conversation_system_is_kept_in_place_and_unknown_roles_fail() {
        let parsed = parsed(&request(json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "stay brief"},
                {"role": "user", "content": "go"}
            ]
        })));
        assert!(matches!(message(&parsed, 1).role, Role::System));
        assert_eq!(message(&parsed, 1).content.as_deref(), Some("stay brief"));

        let error = parse_error(&request(json!({
            "messages": [{"role": "tool", "content": "3"}]
        })));
        bad_request(&error);
        assert!(error.message().contains("role"));
    }

    // ── parse: tools and tool choice ───────────────────────────────────────

    #[test]
    fn tools_wrap_into_the_template_shape() {
        let parsed = parsed(&request(json!({
            "tools": [{"name": "get_weather", "description": "Get the weather",
                       "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}]
        })));
        assert_eq!(
            serde_json::to_value(&parsed.input.tools).expect("tools"),
            json!([{"type": "function", "function": {
                "name": "get_weather", "description": "Get the weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}])
        );
    }

    #[test]
    fn tool_choice_matrix_maps_or_fails() {
        let tools = json!([{"name": "get_weather", "input_schema": {"type": "object"}}]);
        let auto = parsed(&request(
            json!({"tools": tools, "tool_choice": {"type": "auto"}}),
        ));
        assert_eq!(auto.input.tools.len(), 1);
        let none = parsed(&request(
            json!({"tools": tools, "tool_choice": {"type": "none"}}),
        ));
        assert!(none.input.tools.is_empty());
        let parallel = parsed(&request(json!({
            "tools": tools, "tool_choice": {"type": "auto", "disable_parallel_tool_use": true}
        })));
        assert_eq!(parallel.input.tools.len(), 1);
        for choice in [
            json!({"type": "any"}),
            json!({"type": "tool", "name": "get_weather"}),
        ] {
            let error = parse_error(&request(json!({"tools": tools, "tool_choice": choice})));
            bad_request(&error);
            assert!(error.message().contains("tool_choice"));
        }
    }

    // ── parse: thinking and unknown fields ─────────────────────────────────

    #[test]
    fn thinking_modes_map_and_bad_types_fail() {
        assert!(matches!(
            parsed(&request(json!({}))).input.thinking,
            Thinking::Default
        ));
        let on = request(json!({"thinking": {"type": "enabled", "budget_tokens": 1_024}}));
        assert!(matches!(parsed(&on).input.thinking, Thinking::On));
        let off = request(json!({"thinking": {"type": "disabled"}}));
        assert!(matches!(parsed(&off).input.thinking, Thinking::Off));
        let adaptive = request(json!({"thinking": {"type": "adaptive", "display": "omitted"}}));
        assert!(matches!(parsed(&adaptive).input.thinking, Thinking::On));
        let error = parse_error(&request(json!({"thinking": {"type": "sometimes"}})));
        bad_request(&error);
        assert!(error.message().contains("thinking"));
    }

    #[test]
    fn unknown_fields_are_dropped_and_structural_errors_name_the_field() {
        let parsed = parsed(&request(json!({
            "metadata": {"user_id": "u1"}, "an_future_field": true
        })));
        assert!(!parsed.input.messages.is_empty());

        for body in [
            json!({"max_tokens": 4, "messages": "not an array"}),
            json!({"max_tokens": 4, "messages": []}),
            json!({"model": "m", "max_tokens": 4,
                   "messages": [{"role": "user", "content": [{"type": "video", "video": "x"}]}]}),
        ] {
            let error = parse_error(&body);
            bad_request(&error);
        }
    }

    #[test]
    fn the_current_claude_client_field_set_parses() {
        let parsed = parsed(&request(json!({
            "system": [
                {"type": "text", "text": "You are Claude Code.",
                 "cache_control": {"type": "ephemeral"}}
            ],
            "thinking": {"type": "enabled", "budget_tokens": 2_048},
            "tool_choice": {"type": "auto"},
            "metadata": {"user_id": "user_abc"},
            "tools": [{"name": "Bash", "description": "Run a command",
                       "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
                       "cache_control": {"type": "ephemeral"}}]
        })));
        assert!(matches!(parsed.input.thinking, Thinking::On));
        assert_eq!(parsed.input.tools.len(), 1);
        assert!(matches!(message(&parsed, 0).role, Role::System));
    }

    // ── route: non-streaming replies ───────────────────────────────────────

    #[tokio::test]
    async fn a_non_streaming_reply_has_the_message_shape() {
        let engine = testkit::engine(testkit::script("Hello!"));
        let (status, body) = post(&app(&engine, 4_096), &request(json!({}))).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["type"], "message");
        assert_eq!(body["role"], "assistant");
        assert_eq!(body["model"], "model-a");
        assert_eq!(body["id"], "msg_0000000000000001");
        assert_eq!(body["content"], json!([{"type": "text", "text": "Hello!"}]));
        assert_eq!(body["stop_reason"], "end_turn");
        assert_eq!(body["stop_sequence"], Value::Null);
        assert_eq!(body["usage"]["input_tokens"], 21);
        assert_eq!(body["usage"]["output_tokens"], 7);
    }

    #[tokio::test]
    async fn think_text_and_a_tool_call_become_blocks() {
        let text = format!("<think>plan</think>Adding.{CALL}");
        let engine = testkit::engine(testkit::script(&text));
        let body = request(json!({"tools": [
            {"name": "add", "input_schema": {"type": "object", "properties": {
                "a": {"type": "integer"}, "b": {"type": "number"}}}}]}));
        let (status, reply) = post(&app(&engine, 4_096), &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        assert_eq!(
            reply["content"],
            json!([
                {"type": "thinking", "thinking": "plan", "signature": ""},
                {"type": "text", "text": "Adding."},
                {"type": "tool_use", "id": "call_0000000000000001_0",
                 "name": "add", "input": {"a": 1, "b": 2}}
            ])
        );
        assert_eq!(reply["stop_reason"], "tool_use");
    }

    #[tokio::test]
    async fn the_reply_matches_the_contract_schema() {
        let text = format!("<think>plan</think>Adding.{CALL}");
        let engine = testkit::engine(testkit::script(&text));
        let body = request(json!({"tools": [
            {"name": "add", "input_schema": {"type": "object", "properties": {
                "a": {"type": "integer"}, "b": {"type": "number"}}}}]}));
        let (status, reply) = post(&app(&engine, 4_096), &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        crate::server::contract::assert_schema("Message", &reply);
    }

    #[tokio::test]
    async fn an_error_reply_matches_the_contract_schema() {
        let engine = testkit::engine(Vec::new());
        let body = json!({"model": "model-a",
            "messages": [{"role": "user", "content": "hi"}]});
        let (status, reply) = post(&app(&engine, 4_096), &body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
        crate::server::contract::assert_schema("AnthropicError", &reply);
    }

    #[tokio::test]
    async fn a_stop_sequence_reports_stop_sequence() {
        let engine = testkit::engine(testkit::script("Hello END tail"));
        let body = request(json!({"stop_sequences": ["END"]}));
        let (status, reply) = post(&app(&engine, 4_096), &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        assert_eq!(reply["content"], json!([{"type": "text", "text": "Hello"}]));
        assert_eq!(reply["stop_reason"], "stop_sequence");
        assert_eq!(reply["stop_sequence"], "END");
    }

    #[tokio::test]
    async fn max_tokens_reports_max_tokens() {
        let engine = testkit::engine(testkit::script("abcdefgh"));
        let (status, reply) = post(&app(&engine, 4_096), &request(json!({"max_tokens": 3}))).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        assert_eq!(reply["content"], json!([{"type": "text", "text": "abc"}]));
        assert_eq!(reply["stop_reason"], "max_tokens");
    }

    // ── route: auth and transport ──────────────────────────────────────────

    #[tokio::test]
    async fn the_auth_matrix_accepts_both_credential_headers() {
        let engine = testkit::engine(testkit::script("Hello!"));
        let (status, _) = post_with_api_key(&app(&engine, 4_096), &request(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        // A fresh engine: the first request consumed the script of the other one.
        let engine = testkit::engine(testkit::script("Hello!"));
        let (status, _) = post(&app(&engine, 4_096), &request(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        let app = app(&engine, 4_096);

        let wrong = testkit::post_bytes(
            "/v1/messages",
            Some("sk-wrong"),
            request(json!({})).to_string().into_bytes(),
        );
        let (status, _headers, bytes) = testkit::send(&app, wrong).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(
            error_type(&testkit::json_of(&bytes)),
            "authentication_error"
        );

        let none = testkit::post_bytes(
            "/v1/messages",
            None,
            request(json!({})).to_string().into_bytes(),
        );
        let (status, _headers, bytes) = testkit::send(&app, none).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(
            error_type(&testkit::json_of(&bytes)),
            "authentication_error"
        );
    }

    #[tokio::test]
    async fn the_beta_query_parameter_is_accepted() {
        let engine = testkit::engine(testkit::script("Hello!"));
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/messages?beta=true")
            .header("x-api-key", testkit::API_KEY)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(request(json!({})).to_string()))
            .expect("request");
        let (status, _headers, _bytes) = testkit::send(&app(&engine, 4_096), request).await;
        assert_eq!(status, StatusCode::OK);
    }

    // ── route: errors ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn missing_max_tokens_is_invalid_request_error() {
        let engine = testkit::engine(Vec::new());
        let body = json!({"model": "model-a",
            "messages": [{"role": "user", "content": "hi"}]});
        let (status, reply) = post(&app(&engine, 4_096), &body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
        assert_eq!(reply["type"], "error");
        assert_eq!(error_type(&reply), "invalid_request_error");
        crate::server::contract::assert_schema("AnthropicError", &reply);
        assert!(
            reply["error"]["message"]
                .as_str()
                .expect("message")
                .contains("max_tokens")
        );
    }

    #[tokio::test]
    async fn the_context_limit_is_invalid_request_error() {
        // The rendered prompt of "hi" is 21 tokens, so an entry limit of 21 refuses
        // it at submit. The Messages shape reports the refusal as 400, not 409.
        // At the exact boundary the wire stays true: the gate refuses at >= 21,
        // so the largest admissible prompt is 20.
        let engine = testkit::engine(testkit::script("abcdefgh"));
        let (status, reply) = post(&app(&engine, 21), &request(json!({}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
        assert_eq!(error_type(&reply), "invalid_request_error");
        assert_eq!(
            reply["error"]["message"],
            "prompt is too long: 21 tokens > 20 maximum"
        );
    }

    #[tokio::test]
    async fn the_context_limit_message_is_the_over_length_form() {
        // Claude Code auto-compacts only on the over-length form, so the
        // configured-limit refusal must carry it too — a session crossing a
        // configured limit below what Claude Code assumes would otherwise die
        // on an opaque 400 exactly like the pool denials did.
        let engine = testkit::engine(testkit::script("abcdefgh"));
        let (status, reply) = post(&app(&engine, 20), &request(json!({}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
        assert_eq!(error_type(&reply), "invalid_request_error");
        assert_eq!(
            reply["error"]["message"],
            "prompt is too long: 21 tokens > 19 maximum"
        );
    }

    #[tokio::test]
    async fn pool_exhaustion_is_the_anthropic_over_length_error() {
        // A 118,930-token prompt, streamed and not streamed, once met a pool
        // denial of session growth ("session growth requires 11141120 bytes,
        // available 7696432"), and the opaque 400 killed the Claude Code session
        // instead of letting it compact. The engine reports the denial typed,
        // and this route must answer with the exact Anthropic over-length form
        // Claude Code matches to auto-compact: prompt is too long: N tokens > M
        // maximum.
        let engine = testkit::refusing_engine(EngError::PromptTooLong {
            prompt_tokens: 118_930,
            limit_tokens: 110_000,
        });
        let (status, reply) = post(&app(&engine, 262_144), &request(json!({}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
        assert_eq!(error_type(&reply), "invalid_request_error");
        assert_eq!(
            reply["error"]["message"],
            "prompt is too long: 118930 tokens > 110000 maximum"
        );
    }

    #[tokio::test]
    async fn a_fresh_session_denied_admission_is_the_over_length_error() {
        // A fresh 17,479-token session could not admit because a dead session's
        // cached prefixes pinned the pool. After eviction, a denial that still
        // cannot be served reports the same over-length form. The effective
        // ceiling sat below even this fresh prompt.
        let engine = testkit::refusing_engine(EngError::PromptTooLong {
            prompt_tokens: 17_479,
            limit_tokens: 16_384,
        });
        let (status, reply) = post(&app(&engine, 262_144), &request(json!({}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
        assert_eq!(error_type(&reply), "invalid_request_error");
        assert_eq!(
            reply["error"]["message"],
            "prompt is too long: 17479 tokens > 16384 maximum"
        );
    }

    #[tokio::test]
    async fn an_unknown_model_is_not_found_error() {
        let engine = testkit::engine(testkit::script("Hello!"));
        let body = request(json!({"model": "nope"}));
        let (status, reply) = post(&app(&engine, 4_096), &body).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{reply}");
        assert_eq!(error_type(&reply), "not_found_error");
    }

    // ── review fixes: error table, route isolation, validation edges ──────

    #[tokio::test]
    async fn every_error_maps_to_the_anthropic_table() {
        use super::error_response;
        let rows = [
            (
                EngError::BadRequest("m".into()).into(),
                400,
                "invalid_request_error",
            ),
            (ServerError::Unauthorized, 401, "authentication_error"),
            (ServerError::NotFound("m".into()), 404, "not_found_error"),
            (
                EngError::MemoryPlan("m".into()).into(),
                400,
                "invalid_request_error",
            ),
            (
                EngError::ContextLimit("m".into()).into(),
                400,
                "invalid_request_error",
            ),
            (
                EngError::PromptTooLong {
                    prompt_tokens: 118_930,
                    limit_tokens: 110_000,
                }
                .into(),
                400,
                "invalid_request_error",
            ),
            (
                EngError::Checkpoint("/models/tier/map.json: unreadable".into()).into(),
                500,
                "api_error",
            ),
            (ServerError::Overloaded("m".into()), 503, "overloaded_error"),
            (
                ServerError::Eng(EngError::Backend {
                    operation: "decode",
                    message: "secret diagnostic".into(),
                }),
                500,
                "api_error",
            ),
        ];
        for (error, status, kind) in rows {
            let response = error_response(&error);
            assert_eq!(response.status().as_u16(), status, "status of {kind}");
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("body");
            let body = testkit::json_of(&bytes);
            assert_eq!(body["error"]["type"], kind, "type for status {status}");
        }
        // The Backend diagnostic never reaches the client; the auth message names
        // no specific header form.
        let backend = error_response(&ServerError::Eng(EngError::Backend {
            operation: "decode",
            message: "secret diagnostic".into(),
        }));
        let bytes = axum::body::to_bytes(backend.into_body(), usize::MAX)
            .await
            .expect("body");
        let text = String::from_utf8_lossy(&bytes).into_owned();
        assert!(!text.contains("secret"), "diagnostic leaked: {text}");
        // A checkpoint failure carries a server path in its detail; the client
        // sees only the generic api_error message.
        let checkpoint = error_response(&ServerError::Eng(EngError::Checkpoint(
            "/models/tier/map.json: unreadable".to_string(),
        )));
        let bytes = axum::body::to_bytes(checkpoint.into_body(), usize::MAX)
            .await
            .expect("body");
        let text = String::from_utf8_lossy(&bytes).into_owned();
        assert!(!text.contains("map.json"), "path leaked: {text}");
        assert_eq!(
            testkit::json_of(&bytes)["error"]["message"],
            "Model execution failed."
        );
        let unauthorized = error_response(&ServerError::Unauthorized);
        let bytes = axum::body::to_bytes(unauthorized.into_body(), usize::MAX)
            .await
            .expect("body");
        assert_eq!(
            testkit::json_of(&bytes)["error"]["message"],
            "missing or invalid credential"
        );
    }

    #[tokio::test]
    async fn the_api_key_header_only_works_on_the_messages_route() {
        let engine = testkit::engine(testkit::script("Hello!"));
        let app = app(&engine, 4_096);
        let chat = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("x-api-key", testkit::API_KEY)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({"model": "model-a", "messages": [{"role": "user", "content": "hi"}]})
                    .to_string(),
            ))
            .expect("request");
        let (status, _headers, _bytes) = testkit::send(&app, chat).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let models = axum::http::Request::builder()
            .method("GET")
            .uri("/v1/models")
            .header("x-api-key", testkit::API_KEY)
            .body(Body::empty())
            .expect("request");
        let (status, _headers, _bytes) = testkit::send(&app, models).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn duplicate_authorization_headers_are_rejected_on_messages() {
        let engine = testkit::engine(testkit::script("Hello!"));
        let app = app(&engine, 4_096);
        let credential = format!("Bearer {}", testkit::API_KEY);
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("authorization", &credential)
            .header("authorization", &credential)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(request(json!({})).to_string()))
            .expect("request");
        let (status, _headers, _bytes) = testkit::send(&app, request).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn message_content_is_required() {
        for body in [
            json!({"model": "m", "max_tokens": 4, "messages": [{"role": "user"}]}),
            json!({"model": "m", "max_tokens": 4,
                   "messages": [{"role": "user", "content": null}]}),
            json!({"model": "m", "max_tokens": 4,
                   "messages": [{"role": "user", "content": []}]}),
            json!({"model": "m", "max_tokens": 4,
                   "messages": [{"role": "user", "content": ""}]}),
            json!({"model": "m", "max_tokens": 4,
                   "messages": [{"role": "user", "content": [
                       {"type": "text", "text": ""}]}]}),
            json!({"model": "m", "max_tokens": 4,
                   "messages": [{"role": "assistant", "content": null}]}),
            json!({"model": "m", "max_tokens": 4,
                   "messages": [{"role": "assistant", "content": []}]}),
            json!({"model": "m", "max_tokens": 4,
                   "messages": [{"role": "assistant", "content": ""}]}),
            json!({"model": "m", "max_tokens": 4,
                   "messages": [{"role": "assistant", "content": [
                       {"type": "thinking", "thinking": "x", "signature": "s"}]}]}),
            json!({"model": "m", "max_tokens": 4,
                   "messages": [{"role": "assistant", "content": [
                       {"type": "text", "text": ""}]}]}),
            json!({"model": "m", "max_tokens": 4,
                   "messages": [{"role": "assistant", "content": [
                       {"type": "text", "text": ""},
                       {"type": "text", "text": ""}]}]}),
            json!({"model": "m", "max_tokens": 4,
                   "messages": [{"role": "assistant", "content": [
                       {"type": "thinking", "thinking": "x", "signature": "s"},
                       {"type": "text", "text": ""}]}]}),
            json!({"model": "m", "max_tokens": 4,
                   "messages": [{"role": "assistant", "content": [
                       {"type": "redacted_thinking", "data": "x"},
                       {"type": "text", "text": ""}]}]}),
            json!({"model": "m", "max_tokens": 4,
                   "messages": [{"role": "user", "content": [
                       {"type": "document",
                        "source": {"type": "text", "data": ""}}]}]}),
            json!({"model": "m", "max_tokens": 4,
                   "messages": [{"role": "user", "content": [
                       {"type": "document",
                        "source": {"type": "text", "data": ""}},
                       {"type": "text", "text": ""}]}]}),
        ] {
            let error = parse_error(&body);
            bad_request(&error);
            assert!(
                error.message().contains("content"),
                "message names content: {}",
                error.message()
            );
        }
    }

    #[test]
    fn nested_nulls_count_as_omitted() {
        let body = json!({"model": "m", "max_tokens": 4,
            "tools": [{"name": "t", "description": null,
                       "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": null},
            "thinking": {"type": "disabled", "budget_tokens": null},
            "messages": [
                {"role": "user", "content": "run it"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "t",
                     "input": null}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1",
                     "content": null, "is_error": null}]},
                {"role": "user", "content": "again"}]});
        let parsed =
            parse_anthropic(&serde_json::to_vec(&body).expect("body")).expect("nulls parse");
        assert_eq!(parsed.input.tools.len(), 1);
    }

    #[test]
    fn empty_text_between_blocks_joins_nothing() {
        let body = json!({"model": "m", "max_tokens": 4, "messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": "r1"},
                {"type": "text", "text": ""},
                {"type": "tool_result", "tool_use_id": "toolu_2",
                 "content": "r2"}]},
            {"role": "user", "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": ""},
                {"type": "text", "text": "b"}]}]});
        let parsed = parse_anthropic(&serde_json::to_vec(&body).expect("body")).expect("parses");
        assert_eq!(parsed.input.messages.len(), 3);
        assert!(matches!(parsed.input.messages[0].role, Role::Tool));
        assert!(matches!(parsed.input.messages[1].role, Role::Tool));
        assert!(matches!(parsed.input.messages[2].role, Role::User));
        assert_eq!(parsed.input.messages[2].content.as_deref(), Some("a\n\nb"));
    }

    #[test]
    fn empty_system_text_joins_nothing() {
        let body = json!({"model": "m", "max_tokens": 4,
            "system": [
                {"type": "text", "text": ""},
                {"type": "text", "text": "Be brief."}],
            "messages": [{"role": "user", "content": "hi"}]});
        let parsed = parse_anthropic(&serde_json::to_vec(&body).expect("body")).expect("parses");
        let system = parsed
            .input
            .messages
            .iter()
            .find(|message| matches!(message.role, Role::System))
            .expect("system stays");
        assert_eq!(system.content.as_deref(), Some("Be brief."));
    }

    #[tokio::test]
    async fn only_length_limit_failures_count_as_too_large() {
        // The cap is what turns a read failure into 413; a body of one byte
        // against a limit of zero is the smallest real trigger.
        let over = axum::body::to_bytes(Body::from("x"), 0)
            .await
            .expect_err("over the cap");
        assert!(is_length_limit(&over));
        let read = axum::Error::new(std::io::Error::other("read failed"));
        assert!(!is_length_limit(&read));
    }

    #[test]
    fn explicit_nulls_count_as_omitted_optional_fields() {
        let body = json!({"model": "m", "max_tokens": 4,
            "stream": null, "tools": null, "tool_choice": null,
            "thinking": null, "stop_sequences": null, "temperature": null,
            "top_p": null, "top_k": null, "system": null, "metadata": null,
            "messages": [{"role": "user", "content": "hi"}]});
        let parsed =
            parse_anthropic(&serde_json::to_vec(&body).expect("body")).expect("nulls parse");
        assert!(!parsed.input.stream);
        assert!(parsed.input.tools.is_empty());
        assert!(parsed.input.stops.is_empty());
        assert!(
            parsed
                .input
                .messages
                .iter()
                .all(|message| !matches!(message.role, Role::System))
        );
    }

    #[test]
    fn redacted_thinking_blocks_are_dropped_from_history() {
        let body = json!({"model": "m", "max_tokens": 4, "messages": [
            {"role": "assistant", "content": [
                {"type": "redacted_thinking", "data": "x"},
                {"type": "text", "text": "done"}]},
            {"role": "user", "content": "again"}]});
        let parsed = parse_anthropic(&serde_json::to_vec(&body).expect("body")).expect("parses");
        assert_eq!(parsed.input.messages[0].content.as_deref(), Some("done"));
    }

    #[tokio::test]
    async fn a_body_over_the_cap_is_request_too_large() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine, 4_096);
        let big = format!(
            "{{\"model\":\"model-a\",\"max_tokens\":1,\"messages\":[{{\"role\":\"user\",\"content\":\"{}\"}}]}}",
            "x".repeat(17 * 1024 * 1024)
        );
        let (status, _headers, bytes) = testkit::send(
            &app,
            testkit::post_bytes("/v1/messages", Some(testkit::API_KEY), big.into_bytes()),
        )
        .await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(error_type(&testkit::json_of(&bytes)), "request_too_large");
    }

    #[tokio::test]
    async fn text_after_a_tool_call_stays_after_the_tool_block() {
        let text = format!("before{CALL}after");
        let engine = testkit::engine(testkit::script(&text));
        let body = request(json!({"tools": [
            {"name": "add", "input_schema": {"type": "object"}}]}));
        let (status, reply) = post(&app(&engine, 4_096), &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        assert_eq!(
            reply["content"],
            json!([
                {"type": "text", "text": "before"},
                {"type": "tool_use", "id": "call_0000000000000001_0",
                 "name": "add", "input": {"a": 1, "b": 2}},
                {"type": "text", "text": "after"},
            ])
        );
    }

    // ── the SSE event stream ─────────────────────────────────────────────

    fn stream_request(extra: Value) -> Value {
        let mut body = json!({
            "model": "model-a",
            "stream": true,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}]
        });
        for (key, value) in extra.as_object().expect("object").iter() {
            body[key] = value.clone();
        }
        body
    }

    async fn stream(app: &Router, body: &Value) -> (StatusCode, String, Vec<SseEvent>) {
        let (status, headers, bytes) =
            testkit::send(app, testkit::post_json("/v1/messages", body)).await;
        let content_type = headers[header::CONTENT_TYPE]
            .to_str()
            .expect("type")
            .to_string();
        (status, content_type, testkit::sse_events(&bytes))
    }

    fn data(event: &SseEvent) -> Value {
        assert!(event.name.is_some(), "unnamed event: {event:?}");
        serde_json::from_str(&event.data).expect("event json")
    }

    /// The events of one name, parsed.
    fn of(events: &[SseEvent], name: &str) -> Vec<Value> {
        events
            .iter()
            .filter(|event| event.name.as_deref() == Some(name))
            .map(data)
            .collect()
    }

    /// The text deltas of one block, joined.
    fn block_text(events: &[SseEvent], index: u64) -> String {
        of(events, "content_block_delta")
            .iter()
            .filter(|delta| delta["index"] == index)
            .map(|delta| delta["delta"]["text"].as_str().expect("text delta"))
            .collect()
    }

    /// The blocks the stream opened, as (index, block type).
    fn blocks(events: &[SseEvent]) -> Vec<(u64, String)> {
        of(events, "content_block_start")
            .iter()
            .map(|start| {
                (
                    start["index"].as_u64().expect("index"),
                    start["content_block"]["type"]
                        .as_str()
                        .expect("type")
                        .to_string(),
                )
            })
            .collect()
    }

    #[tokio::test]
    async fn a_stream_replays_the_anthropic_event_sequence() {
        let engine = testkit::engine(testkit::script("Hi!"));
        let (status, content_type, events) =
            stream(&app(&engine, 4_096), &stream_request(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, "text/event-stream");
        let names: Vec<&str> = events
            .iter()
            .map(|event| event.name.as_deref().expect("named"))
            .collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ],
            "{events:?}"
        );
        let start = data(&events[0]);
        assert_eq!(start["message"]["id"], "msg_0000000000000001");
        assert_eq!(start["message"]["role"], "assistant");
        assert_eq!(start["message"]["model"], "model-a");
        // message_start reports the input count from admission; the output
        // count and the final totals arrive with message_delta.
        //
        // The streaming docs emit `usage: {"output_tokens": N}` in
        // `message_delta`, but the same page states those counts are
        // cumulative, and versioning allows additional fields. Real clients
        // may read the cumulative input accounting from the delta event, so
        // the engine intentionally keeps the superset with both
        // `input_tokens` and `output_tokens` — this test pins that contract.
        assert_eq!(start["message"]["usage"]["input_tokens"], 21);
        let block_start = data(&events[1]);
        assert_eq!(block_start["index"], 0);
        assert_eq!(
            block_start["content_block"],
            json!({"type": "text", "text": ""})
        );
        let texts: Vec<String> = events[2..5]
            .iter()
            .map(|event| {
                data(event)["delta"]["text"]
                    .as_str()
                    .expect("text")
                    .to_string()
            })
            .collect();
        assert_eq!(texts, ["H", "i", "!"]);
        for event in &events[2..5] {
            let delta = data(event);
            assert_eq!(delta["index"], 0);
            assert_eq!(delta["delta"]["type"], "text_delta");
        }
        assert_eq!(data(&events[5])["index"], 0);
        let message_delta = data(&events[6]);
        assert_eq!(message_delta["delta"]["stop_reason"], "end_turn");
        assert_eq!(message_delta["delta"]["stop_sequence"], Value::Null);
        assert_eq!(message_delta["usage"]["input_tokens"], 21);
        assert_eq!(message_delta["usage"]["output_tokens"], 4);
        assert_eq!(data(&events[7])["type"], "message_stop");
    }

    #[tokio::test]
    async fn thinking_streams_as_one_block_before_the_text() {
        let engine = testkit::engine(testkit::script("<think>plan</think>Done."));
        let (_, _, events) = stream(&app(&engine, 4_096), &stream_request(json!({}))).await;
        assert_eq!(
            blocks(&events),
            [(0, "thinking".to_string()), (1, "text".to_string())]
        );
        assert_eq!(
            of(&events, "content_block_start")[0]["content_block"],
            json!({"type": "thinking", "thinking": "", "signature": ""})
        );
        let stops: Vec<u64> = of(&events, "content_block_stop")
            .into_iter()
            .map(|event| event["index"].as_u64().expect("index"))
            .collect();
        assert_eq!(stops, [0, 1], "each block closes before the next opens");
        let thinking: String = of(&events, "content_block_delta")
            .iter()
            .filter(|delta| delta["delta"]["type"] == "thinking_delta")
            .map(|delta| delta["delta"]["thinking"].as_str().expect("thinking delta"))
            .collect();
        assert_eq!(thinking, "plan");
        assert_eq!(block_text(&events, 1), "Done.");
    }

    #[tokio::test]
    async fn a_tool_call_streams_as_one_tool_use_block() {
        let text = format!("Adding.{CALL}");
        let engine = testkit::engine(testkit::script(&text));
        let body = stream_request(json!({"tools": [
            {"name": "add", "input_schema": {"type": "object"}}],
            "max_tokens": 256}));
        let (status, _, events) = stream(&app(&engine, 4_096), &body).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            blocks(&events),
            [(0, "text".to_string()), (1, "tool_use".to_string())]
        );
        let tool_start = &of(&events, "content_block_start")[1];
        assert_eq!(
            tool_start["content_block"],
            json!({"type": "tool_use", "id": "call_0000000000000001_0",
                   "name": "add", "input": {}})
        );
        let json_delta = of(&events, "content_block_delta")
            .into_iter()
            .find(|delta| delta["delta"]["type"] == "input_json_delta")
            .expect("input_json_delta");
        assert_eq!(json_delta["delta"]["partial_json"], "{\"a\":1,\"b\":2}");
        let all: String = events.iter().map(|event| event.data.as_str()).collect();
        assert!(!all.contains("tool_call>"), "markup leaked: {all}");
        let message_delta = of(&events, "message_delta")
            .into_iter()
            .next()
            .expect("end");
        assert_eq!(message_delta["delta"]["stop_reason"], "tool_use");
    }

    #[tokio::test]
    async fn text_after_a_tool_call_opens_a_new_text_block() {
        let text = format!("before{CALL}after");
        let engine = testkit::engine(testkit::script(&text));
        let body = stream_request(json!({"tools": [
            {"name": "add", "input_schema": {"type": "object"}}],
            "max_tokens": 256}));
        let (_, _, events) = stream(&app(&engine, 4_096), &body).await;
        assert_eq!(
            blocks(&events),
            [
                (0, "text".to_string()),
                (1, "tool_use".to_string()),
                (2, "text".to_string()),
            ]
        );
        let stops: Vec<u64> = of(&events, "content_block_stop")
            .into_iter()
            .map(|event| event["index"].as_u64().expect("index"))
            .collect();
        assert_eq!(stops, [0, 1, 2], "each block closes before the next opens");
        assert_eq!(block_text(&events, 0), "before");
        assert_eq!(block_text(&events, 2), "after");
    }

    #[tokio::test]
    async fn a_stop_string_sets_stop_sequence_in_message_delta() {
        let engine = testkit::engine(testkit::script("one END two"));
        let body = stream_request(json!({"stop_sequences": ["END"]}));
        let (_, _, events) = stream(&app(&engine, 4_096), &body).await;
        assert_eq!(block_text(&events, 0), "one");
        let message_delta = of(&events, "message_delta")
            .into_iter()
            .next()
            .expect("end");
        assert_eq!(message_delta["delta"]["stop_reason"], "stop_sequence");
        assert_eq!(message_delta["delta"]["stop_sequence"], "END");
    }

    #[tokio::test]
    async fn max_tokens_ends_the_stream_with_max_tokens() {
        let engine = testkit::engine(testkit::script("abcdef"));
        let body = stream_request(json!({"max_tokens": 2}));
        let (_, _, events) = stream(&app(&engine, 4_096), &body).await;
        let message_delta = of(&events, "message_delta")
            .into_iter()
            .next()
            .expect("end");
        assert_eq!(message_delta["delta"]["stop_reason"], "max_tokens");
        assert_eq!(message_delta["usage"]["output_tokens"], 2);
        assert_eq!(
            of(&events, "message_stop").len(),
            1,
            "message_stop closes the stream"
        );
    }

    #[tokio::test]
    async fn a_prefill_failure_is_one_error_event_after_headers() {
        let engine = testkit::refusing_engine(EngError::MemoryPlan("no room".to_string()));
        let (status, content_type, events) =
            stream(&app(&engine, 4_096), &stream_request(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, "text/event-stream");
        let names: Vec<&str> = events
            .iter()
            .map(|event| event.name.as_deref().expect("named"))
            .collect();
        assert_eq!(names, ["message_start", "error"], "{events:?}");
        let error = &of(&events, "error")[0];
        assert_eq!(error["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn pool_exhaustion_on_a_stream_is_the_over_length_error_event() {
        // The stream has already sent its 200 headers when the pool denial
        // arrives, so the over-length report must reach Claude Code as the error
        // event — the exact form it matches to auto-compact mid-session.
        let engine = testkit::refusing_engine(EngError::PromptTooLong {
            prompt_tokens: 118_930,
            limit_tokens: 110_000,
        });
        let (status, content_type, events) =
            stream(&app(&engine, 262_144), &stream_request(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, "text/event-stream");
        let names: Vec<&str> = events
            .iter()
            .map(|event| event.name.as_deref().expect("named"))
            .collect();
        assert_eq!(names, ["message_start", "error"], "{events:?}");
        let error = &of(&events, "error")[0];
        assert_eq!(error["error"]["type"], "invalid_request_error");
        assert_eq!(
            error["error"]["message"],
            "prompt is too long: 118930 tokens > 110000 maximum"
        );
    }

    #[tokio::test]
    async fn a_decode_failure_sends_one_error_event_and_closes() {
        let failure = EngError::Backend {
            operation: "qmm",
            message: "Metal command buffer 7 failed".to_string(),
        };
        let engine = testkit::ending_engine("ab", Ending::Error(failure));
        let (status, _, events) = stream(&app(&engine, 4_096), &stream_request(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(of(&events, "error").len(), 1);
        assert_eq!(
            of(&events, "message_stop").len(),
            0,
            "no message_stop after an error"
        );
        let all: String = events.iter().map(|event| event.data.as_str()).collect();
        assert!(!all.contains("Metal"), "diagnostic leaked: {all}");
        let error = &of(&events, "error")[0];
        assert_eq!(error["error"]["type"], "api_error");
    }

    #[tokio::test]
    async fn a_checkpoint_failure_is_one_generic_error_event() {
        let failure = EngError::Checkpoint("tier mapping t8 did not load".to_string());
        let engine = testkit::ending_engine("ab", Ending::Error(failure));
        let (status, _, events) = stream(&app(&engine, 4_096), &stream_request(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(of(&events, "error").len(), 1);
        assert_eq!(
            of(&events, "message_stop").len(),
            0,
            "no message_stop after an error"
        );
        let all: String = events.iter().map(|event| event.data.as_str()).collect();
        assert!(!all.contains("t8"), "checkpoint detail leaked: {all}");
        let error = &of(&events, "error")[0];
        assert_eq!(error["error"]["type"], "api_error");
    }

    #[tokio::test]
    async fn a_cancelled_job_closes_the_stream_silently() {
        let engine = testkit::ending_engine("ab", Ending::Cancel);
        let (status, _, events) = stream(&app(&engine, 4_096), &stream_request(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(of(&events, "error").len(), 0, "no error event");
        assert_eq!(of(&events, "message_stop").len(), 0, "no message_stop");
        assert_eq!(of(&events, "message_delta").len(), 0, "no message_delta");
        assert_eq!(block_text(&events, 0), "ab");
    }

    #[tokio::test]
    async fn pre_header_errors_stay_json_even_when_stream_is_set() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine, 4_096);
        let unknown = stream_request(json!({"model": "nope"}));
        let (status, headers, bytes) =
            testkit::send(&app, testkit::post_json("/v1/messages", &unknown)).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(headers[header::CONTENT_TYPE], "application/json");
        assert_eq!(error_type(&testkit::json_of(&bytes)), "not_found_error");
    }

    // ── captured client replays ──────────────────────────────────────

    /// One fixture file of the captured-client directory, parsed.
    fn capture(name: &str, kind: &str) -> Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/server/testdata/anthropic")
            .join(format!("{name}.{kind}.json"));
        serde_json::from_str(&std::fs::read_to_string(path).expect("fixture"))
            .expect("fixture json")
    }

    /// Posts a captured request with its recorded headers plus the test key.
    /// The redacted credential headers of the capture are skipped.
    async fn replay_post(
        app: &Router,
        body: &Value,
        headers: &Value,
        query: &str,
    ) -> (StatusCode, String, Vec<SseEvent>, Value) {
        let uri = if query.is_empty() {
            "/v1/messages".to_string()
        } else {
            format!("/v1/messages?{query}")
        };
        let recorded_api_key = headers
            .as_object()
            .expect("recorded headers")
            .keys()
            .any(|name| name.eq_ignore_ascii_case("x-api-key"));
        let mut builder = axum::http::Request::builder()
            .method("POST")
            .uri(uri)
            .header(header::CONTENT_TYPE, "application/json");
        // The recorded credentials are redacted; the test key rides the same
        // credential header the capture used.
        if recorded_api_key {
            builder = builder.header("x-api-key", testkit::API_KEY);
        } else {
            builder = builder.header(
                header::AUTHORIZATION,
                format!("Bearer {}", testkit::API_KEY),
            );
        }
        for (name, value) in headers.as_object().expect("recorded headers") {
            // Credentials and content type are set once above.
            if ["x-api-key", "authorization", "content-type"]
                .iter()
                .any(|skip| name.eq_ignore_ascii_case(skip))
            {
                continue;
            }
            builder = builder.header(name.as_str(), value.as_str().expect("value"));
        }
        let request = builder.body(Body::from(body.to_string())).expect("request");
        let (status, reply_headers, bytes) = testkit::send(app, request).await;
        let content_type = reply_headers[header::CONTENT_TYPE]
            .to_str()
            .unwrap_or("")
            .to_string();
        let streaming = content_type.starts_with("text/event-stream");
        let events = if streaming {
            testkit::sse_events(&bytes)
        } else {
            Vec::new()
        };
        let json = if streaming || bytes.is_empty() {
            Value::Null
        } else {
            testkit::json_of(&bytes)
        };
        (status, content_type, events, json)
    }

    /// Every capture in the manifest replays to its expected contract
    /// shape. A capture joins by adding its four files and one manifest row;
    /// this test needs no change. The fixtures assert transport and mapping
    /// conformance only, never model behavior.
    #[tokio::test]
    async fn captured_client_requests_replay_to_the_contract_shapes() {
        let manifest_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/server/testdata/anthropic")
            .join("MANIFEST.json");
        let manifest: Vec<Value> =
            serde_json::from_str(&std::fs::read_to_string(manifest_path).expect("manifest"))
                .expect("manifest json");
        assert!(!manifest.is_empty(), "the manifest must carry rows");
        for row in &manifest {
            let name = row["name"].as_str().expect("name");
            let body = capture(name, "request");
            let headers = capture(name, "headers");
            let expected = capture(name, "expected");
            let query = expected["query"].as_str().unwrap_or("");
            let model = body["model"].as_str().expect("model").to_string();
            let engine = testkit::engine(testkit::script("Recorded."));
            let app = app_serving(&engine, 1_000_000, &model);
            let expected_status =
                StatusCode::from_u16(expected["status"].as_u64().expect("expected status") as u16)
                    .expect("known status");
            let (status, content_type, events, json) =
                replay_post(&app, &body, &headers, query).await;
            assert_eq!(status, expected_status, "{name}: {json}");
            let streaming = content_type.starts_with("text/event-stream");
            assert_eq!(
                expected["mode"],
                if streaming { "stream" } else { "message" },
                "{name}: mode matches the reply kind"
            );
            for check in expected["asserts"].as_array().expect("asserts") {
                match check.as_str().expect("assert") {
                    "event-sequence" => {
                        let names: Vec<&str> = events
                            .iter()
                            .map(|event| event.name.as_deref().expect("named"))
                            .collect();
                        assert_eq!(names.first(), Some(&"message_start"), "{name}: {names:?}");
                        assert_eq!(
                            names.get(1),
                            Some(&"content_block_start"),
                            "{name}: {names:?}"
                        );
                        assert_eq!(names.last(), Some(&"message_stop"), "{name}: {names:?}");
                        assert_eq!(
                            names[names.len() - 2],
                            "message_delta",
                            "{name}: exactly one message_delta closes the stream"
                        );
                        // The scripted reply holds one text block: every
                        // event between the block start and the block stop
                        // is a delta of that block.
                        assert!(
                            names[2..names.len() - 3]
                                .iter()
                                .all(|event| *event == "content_block_delta"),
                            "{name}: {names:?}"
                        );
                        assert!(!names.contains(&"error"), "{name}: {names:?}");
                        let indexes: Vec<u64> = ["content_block_start", "content_block_stop"]
                            .iter()
                            .flat_map(|kind| of(&events, kind))
                            .map(|event| event["index"].as_u64().expect("index"))
                            .collect();
                        assert_eq!(indexes, vec![0, 0], "{name}: block indexes match");
                        let deltas = of(&events, "content_block_delta");
                        assert!(
                            !deltas.is_empty(),
                            "{name}: the reply streams at least one delta"
                        );
                        assert!(
                            deltas
                                .iter()
                                .all(|event| event["index"].as_u64() == Some(0)),
                            "{name}: every delta belongs to the single block"
                        );
                    }
                    "usage-in-message-start" => {
                        let start = &of(&events, "message_start")[0];
                        let input = start["message"]["usage"]["input_tokens"]
                            .as_u64()
                            .expect("input tokens");
                        assert!(
                            input > 0,
                            "{name}: message_start carries the admission count"
                        );
                        // Replays are cold rows (a fresh engine per
                        // capture), so delta splits nothing: its
                        // input_tokens still equals the whole prompt. On a
                        // warm row delta carries the cache split instead —
                        // see the warm-stream test above.
                        let delta = &of(&events, "message_delta")[0];
                        assert_eq!(
                            delta["usage"]["input_tokens"].as_u64(),
                            Some(input),
                            "{name}: cold message_delta agrees on the whole input"
                        );
                        // The cache field follows the switch: Some(0)
                        // when the split is on, absent (Null) when the
                        // switch forces legacy — a replay is always a cold
                        // row, so a present field is always a zero.
                        let expected_cache_read = if crate::server::wire::cache_fields_on() {
                            serde_json::Value::from(0u64)
                        } else {
                            serde_json::Value::Null
                        };
                        assert_eq!(
                            delta["usage"]["cache_read_input_tokens"], expected_cache_read,
                            "{name}: a replayed cold row reports no cache read"
                        );
                    }
                    "message-shape" => {
                        assert_eq!(json["type"], "message", "{name}");
                        assert_eq!(json["role"], "assistant", "{name}");
                        assert!(json["content"].is_array(), "{name}");
                        assert!(json["stop_reason"].is_string(), "{name}: {json}");
                        assert!(
                            json["usage"]["input_tokens"].as_u64().unwrap_or(0) > 0,
                            "{name}: {json}"
                        );
                    }
                    other => panic!("{name}: unknown assert {other}"),
                }
            }
            for deviation in expected["deviations"].as_array().into_iter().flatten() {
                match deviation.as_str().expect("deviation") {
                    // The real body's thinking blocks must not reach the
                    // prompt content: the tool-call turn carries its call, no
                    // thinking text joined — the thinking rides on
                    // reasoning_content for the template's keep/drop rule.
                    "thinking-carried" => {
                        let parsed = parsed(&body);
                        let tool_turns: Vec<&PromptMessage> = parsed
                            .input
                            .messages
                            .iter()
                            .filter(|message| !message.tool_calls.is_empty())
                            .collect();
                        assert_eq!(tool_turns.len(), 1, "{name}: one tool call turn");
                        assert!(
                            tool_turns[0].content.as_deref().unwrap_or("").is_empty(),
                            "{name}: the turn holds only the tool call"
                        );
                        let thinking = body["messages"]
                            .as_array()
                            .expect("messages")
                            .iter()
                            .flat_map(|message| message["content"].as_array().into_iter().flatten())
                            .filter(|block| block["type"] == "thinking")
                            .map(|block| block["thinking"].as_str().unwrap_or(""))
                            // The same join the translation applies, so a
                            // future multi-block capture stays pinned.
                            .collect::<Vec<&str>>()
                            .join("\n\n");
                        assert!(
                            !thinking.is_empty(),
                            "{name}: the capture carries thinking text"
                        );
                        assert_eq!(
                            tool_turns[0].reasoning_content.as_deref(),
                            Some(thinking.as_str()),
                            "{name}: thinking rides as reasoning_content"
                        );
                    }
                    // The mid-conversation system message stays a system
                    // message in place: one past its source position, after
                    // the lifted top-level system.
                    "system-kept-in-place" => {
                        let source = body["messages"]
                            .as_array()
                            .expect("messages")
                            .iter()
                            .position(|message| message["role"] == "system")
                            .expect("a mid-conversation system message");
                        let parsed = parsed(&body);
                        assert!(
                            matches!(message(&parsed, source + 1).role, Role::System),
                            "{name}: system stays at {}",
                            source + 1
                        );
                    }
                    other => panic!("{name}: unknown deviation {other}"),
                }
            }
        }
    }

    /// Rebuilds the content blocks of a reply from its stream events: text
    /// deltas join per block, tool_use input joins from partial_json.
    fn rebuilt_blocks(events: &[SseEvent]) -> Vec<Value> {
        of(events, "content_block_start")
            .iter()
            .map(|start| {
                let index = start["index"].as_u64().expect("index");
                match start["content_block"]["type"].as_str().expect("type") {
                    "text" => json!({"type": "text", "text": block_text(events, index)}),
                    "tool_use" => {
                        let partials: String = of(events, "content_block_delta")
                            .iter()
                            .filter(|delta| delta["index"] == index)
                            .map(|delta| delta["delta"]["partial_json"].as_str().expect("partial"))
                            .collect();
                        let mut block = start["content_block"].clone();
                        block["input"] = serde_json::from_str(&partials).expect("joins");
                        block
                    }
                    other => panic!("unexpected block type {other}"),
                }
            })
            .collect()
    }

    /// The same captured request, answered non-streaming and streamed,
    /// rebuilds to the same message. The tool run adds a tool and a
    /// budget to the captured body so the scripted call completes: the
    /// tool_use rebuild and its partial_json join are exercised, not just
    /// text.
    #[tokio::test]
    async fn a_streamed_capture_rebuilds_the_non_streaming_message() {
        let body = capture("sdk-plain", "request");
        let headers = capture("sdk-plain", "headers");
        let model = body["model"].as_str().expect("model");
        for script in ["Hi!", &format!("before{CALL}after")] {
            let mut request_body = body.clone();
            let tool_run = script.contains(CALL);
            if tool_run {
                // The capture holds no tools and a 64-token budget; the
                // call needs both to complete.
                request_body["tools"] = json!([
                    {"name": "add", "input_schema": {"type": "object"}}
                ]);
                request_body["max_tokens"] = json!(256);
            }
            let engine = testkit::engine(testkit::script(script));
            let app = app_serving(&engine, 4_096, model);
            let (status, _, _, plain) = replay_post(&app, &request_body, &headers, "").await;
            assert_eq!(status, StatusCode::OK, "{plain}");
            assert_eq!(plain["type"], "message");
            if tool_run {
                let blocks = plain["content"].as_array().expect("blocks");
                assert_eq!(blocks.len(), 3, "{plain}");
                assert_eq!(blocks[0]["type"], "text", "{plain}");
                assert_eq!(blocks[1]["type"], "tool_use", "{plain}");
                assert_eq!(blocks[1]["name"], "add", "{plain}");
                assert_eq!(blocks[1]["input"], json!({"a": 1, "b": 2}), "{plain}");
                assert_eq!(blocks[2]["type"], "text", "{plain}");
                assert_eq!(plain["stop_reason"], "tool_use", "{plain}");
            }

            let engine = testkit::engine(testkit::script(script));
            let app = app_serving(&engine, 4_096, model);
            let mut streamed = request_body.clone();
            streamed["stream"] = json!(true);
            let (status, content_type, events, _) =
                replay_post(&app, &streamed, &headers, "").await;
            assert_eq!(status, StatusCode::OK);
            assert!(content_type.starts_with("text/event-stream"));

            assert_eq!(
                Value::Array(rebuilt_blocks(&events)),
                plain["content"],
                "stream rebuilds the blocks"
            );
            let delta = &of(&events, "message_delta")[0];
            assert_eq!(delta["delta"]["stop_reason"], plain["stop_reason"]);
            assert_eq!(
                delta["usage"], plain["usage"],
                "usage matches the JSON reply"
            );
        }
    }

    /// Asserts the three capture-name sets are equal and non-empty.
    ///
    /// Names appearing in some sets but not others are orphans — files on
    /// disk with no companion capture — and the assertion rejects them.
    fn assert_balanced_captures(
        requests: &[String],
        headers: &[String],
        expected: &[String],
        manifest_names: &[String],
    ) {
        let req_set: std::collections::BTreeSet<String> = requests.iter().cloned().collect();
        let hdr_set: std::collections::BTreeSet<String> = headers.iter().cloned().collect();
        let exp_set: std::collections::BTreeSet<String> = expected.iter().cloned().collect();

        for (label, actual) in [("headers", &hdr_set), ("expected", &exp_set)] {
            let missing: Vec<&String> = actual.difference(&req_set).collect();
            assert!(
                missing.is_empty(),
                "orphan {label} files with no matching capture: {missing:?}"
            );
        }
        let extra: Vec<&String> = req_set
            .difference(&hdr_set)
            .chain(req_set.difference(&exp_set))
            .collect();
        assert!(
            extra.is_empty(),
            "request files missing their headers/expected companions: {extra:?}"
        );
        assert!(!req_set.is_empty(), "fixture request set is empty");
        assert_eq!(
            req_set,
            manifest_names
                .iter()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>(),
            "fixture request names and MANIFEST rows diverged"
        );
        // Set equality alone would accept a duplicated MANIFEST row; the
        // length pin keeps the base's exact-list behavior.
        assert_eq!(
            manifest_names.len(),
            req_set.len(),
            "duplicate MANIFEST rows for one capture"
        );
    }

    fn assert_no_machine_context(name: &str, scanned: &str, text: &str) {
        // JSON-escaped slashes ("\/home\/dan") would hide every path needle
        // from a raw scan, so scan a de-escaped copy instead. Fixtures hold
        // JSON nested inside JSON strings, so escaping can stack; repeat the
        // replace until it stabilizes. Scanning the de-escaped text is safe:
        // replacing "\/" with "/" never hides a needle that was visible in
        // the raw text (no needle contains a backslash, and for the home-path
        // loop every delimiter set that contains '/' also contains '\\'),
        // and each pass is strictly stronger by the same argument.
        let mut deescaped = text.to_string();
        while deescaped.contains("\\/") {
            deescaped = deescaped.replace("\\/", "/");
        }
        let text = deescaped.as_str();
        for needle in [
            "/Users/",
            "-Users-",
            "/media/",
            "-media-",
            "nvme",
            "scrch",
            "Dan Campos",
            "OS Version:",
        ] {
            assert!(
                !text.contains(needle),
                "{name} {scanned} carries {needle:?}: regenerate sanitized fixtures"
            );
        }
        // Home paths may only appear in the neutral rewritten form;
        // anything else is capturing-host layout.
        let path_delims: &[char] = &['/', ' ', '`', '"', '\\'];
        let slug_delims: &[char] = &['-', ' ', '"', '\\'];
        for (prefix, neutral, delims) in [
            ("/home/", "/home/user", path_delims),
            ("-home-", "-home-user", slug_delims),
        ] {
            for (hit, _) in text.match_indices(prefix) {
                let rest = &text[hit..];
                assert!(
                    rest.starts_with(neutral)
                        && match rest.as_bytes().get(neutral.len()) {
                            None => true,
                            Some(&b) => delims.contains(&(b as char)),
                        },
                    "{name} {scanned} carries a non-neutral {prefix} path: regenerate sanitized fixtures"
                );
            }
        }
    }

    #[test]
    fn assert_no_machine_context_boundary() {
        // These must panic — non-neutral home paths, plain or JSON-escaped
        // ("\/home\/dan" must not slip past the scan).
        let failing = [
            "/home/username",
            "/home/user2",
            "-home-username",
            "\"path\": \"\\/home\\/dan\"",
            "\\/Users\\/dan",
            "\\\\/home\\\\/dan",
        ];
        for s in failing {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                assert_no_machine_context("boundary-fail", "body", s)
            }));
            assert!(result.is_err(), "should reject {s:?}");
        }
        // Neutral forms must pass — no panic — including JSON-escaped ones.
        for s in [
            "/home/user/inference-engine/x",
            "-home-user-inference-engine",
            "... at /home/user and run ...",
            "/home/user",
            "\"at \\/home\\/user again\"",
            "\\/home\\/user\\/inference-engine",
        ] {
            assert_no_machine_context("boundary-pass", "body", s);
        }
    }

    /// The committed captures carry no machine context of the capturing
    /// host — paths, user names, OS details, or client identifiers — and
    /// the fixture set stays whole: every capture has its headers file,
    /// expectation and manifest row, all Claude Code captures are
    /// present, and their mid-list system rows match the counts the
    /// demotion seam test relies on. The regenerator lives in scratch;
    /// this guard lives with the fixtures.
    #[test]
    fn captured_fixtures_carry_no_machine_context() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/server/testdata/anthropic");
        let manifest: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("MANIFEST.json")).expect("manifest"),
        )
        .expect("manifest JSON");
        let mut manifest_names: Vec<String> = manifest
            .as_array()
            .expect("manifest rows")
            .iter()
            .map(|row| row["name"].as_str().expect("manifest row name").to_string())
            .collect();
        manifest_names.sort();

        let mut request_names: Vec<String> = Vec::new();
        let mut header_names: Vec<String> = Vec::new();
        let mut expected_names: Vec<String> = Vec::new();
        let mut cc_checked = 0;
        for entry in std::fs::read_dir(&dir).expect("fixture dir") {
            let path = entry.expect("entry").path();
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if let Some(capture) = name.strip_suffix(".request.json") {
                request_names.push(capture.to_string());
                let is_cc = capture.starts_with("cc-");
                cc_checked += usize::from(is_cc);

                let text = std::fs::read_to_string(&path).expect("fixture");
                let headers = std::fs::read_to_string(dir.join(format!("{capture}.headers.json")))
                    .expect("headers fixture");
                assert!(
                    dir.join(format!("{capture}.expected.json")).exists(),
                    "{capture}: missing expected.json"
                );
                assert_no_machine_context(&name, "body", &text);
                assert_no_machine_context(&name, "headers", &headers);

                let expected_path = dir.join(format!("{capture}.expected.json"));
                let expected_text =
                    std::fs::read_to_string(&expected_path).expect("expected fixture");
                assert_no_machine_context(&name, "expected", &expected_text);

                // A client identifier may only appear as the redacted
                // placeholder, and an account id only empty or redacted.
                assert_eq!(
                    text.matches("device_id").count(),
                    text.matches("device_id\\\":\\\"redacted\\\"").count(),
                    "{name}: unredacted device_id"
                );
                assert_eq!(
                    text.matches("session_id").count(),
                    text.matches("session_id\\\":\\\"redacted\\\"").count(),
                    "{name}: unredacted session_id"
                );
                let account_total = text.matches("account_uuid").count();
                let account_empty = text.matches("account_uuid\\\":\\\"\\\"").count();
                let account_redacted = text.matches("account_uuid\\\":\\\"redacted\\\"").count();
                assert_eq!(
                    account_total,
                    account_empty + account_redacted,
                    "{name}: unredacted account_uuid"
                );

                // The mid-list system rows match the counts the demotion seam
                // test asserts: one on the first turn, two on later turns.
                if is_cc {
                    let body: Value = serde_json::from_str(&text).expect("fixture JSON");
                    let mid_systems = body["messages"]
                        .as_array()
                        .expect("messages")
                        .iter()
                        .skip(1)
                        .filter(|message| message["role"] == "system")
                        .count();
                    let expected = if capture == "cc-plain" { 1 } else { 2 };
                    assert_eq!(
                        mid_systems, expected,
                        "{name}: mid-list system row count changed"
                    );
                }
            }
            // Collect names from headers and expected files too, to detect
            // orphans — a .headers.json or .expected.json with no matching
            // .request.json was previously invisible.
            if let Some(capture) = name.strip_suffix(".headers.json") {
                header_names.push(capture.to_string());
            }
            if let Some(capture) = name.strip_suffix(".expected.json") {
                expected_names.push(capture.to_string());
            }
        }

        // No orphans: every request, header, and expected file shares the
        // same name set, and both match the manifest. The helper is the
        // same logic the orphan unit tests exercise with synthetic lists.
        request_names.sort();
        header_names.sort();
        expected_names.sort();
        assert_balanced_captures(
            &request_names,
            &header_names,
            &expected_names,
            &manifest_names,
        );

        let manifest_text = std::fs::read_to_string(dir.join("MANIFEST.json")).expect("manifest");
        assert_no_machine_context("manifest", "manifest", &manifest_text);

        assert!(cc_checked >= 5, "the claude-code captures must be present");
        assert!(
            request_names.len() >= 8,
            "the full capture set must be present"
        );
    }

    /// Orphan fixture files — .headers.json or .expected.json with no
    /// matching .request.json — are rejected by the guard logic.
    #[test]
    fn assert_balanced_captures_rejects_orphan_headers() {
        let manifest = vec!["alpha".to_string()];
        let requests = vec!["alpha".to_string()];
        // headers.json exists for "beta" but no beta.request.json.
        let headers = vec!["alpha".to_string(), "beta".to_string()];
        let expected = vec!["alpha".to_string()];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_balanced_captures(&requests, &headers, &expected, &manifest);
        }));
        assert!(result.is_err(), "orphan headers must panic");
    }

    #[test]
    fn assert_balanced_captures_rejects_orphan_expected() {
        let manifest = vec!["alpha".to_string()];
        let requests = vec!["alpha".to_string()];
        let headers = vec!["alpha".to_string()];
        // expected.json exists for "beta" but no beta.request.json.
        let expected = vec!["alpha".to_string(), "beta".to_string()];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_balanced_captures(&requests, &headers, &expected, &manifest);
        }));
        assert!(result.is_err(), "orphan expected must panic");
    }

    #[test]
    fn assert_balanced_captures_passes_balanced() {
        let names = vec!["alpha".to_string(), "beta".to_string()];
        let requests = names.clone();
        let headers = names.clone();
        let expected = names.clone();
        // Panics if the sets differ or don't match manifest.
        assert_balanced_captures(&requests, &headers, &expected, &names);
    }

    #[test]
    fn assert_balanced_captures_rejects_request_without_headers() {
        let manifest = vec!["alpha".to_string(), "beta".to_string()];
        // beta.request.json exists but beta.headers.json does not.
        let requests = vec!["alpha".to_string(), "beta".to_string()];
        let headers = vec!["alpha".to_string()];
        let expected = vec!["alpha".to_string(), "beta".to_string()];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_balanced_captures(&requests, &headers, &expected, &manifest);
        }));
        assert!(result.is_err(), "request without headers must panic");
    }

    #[test]
    fn assert_balanced_captures_rejects_duplicate_manifest_rows() {
        // A duplicated MANIFEST row collapses in the set comparison; the
        // length pin must still catch it.
        let manifest = vec!["alpha".to_string(), "alpha".to_string()];
        let requests = vec!["alpha".to_string()];
        let headers = vec!["alpha".to_string()];
        let expected = vec!["alpha".to_string()];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_balanced_captures(&requests, &headers, &expected, &manifest);
        }));
        assert!(result.is_err(), "duplicate MANIFEST rows must panic");
    }

    /// Fixture replays alone cannot catch a template rejection, which is
    /// how the captures missed the real Qwen3.6 template rejecting Claude
    /// Code's mid-list system messages. This pins the endpoint-to-template
    /// seam: every captured Claude Code body must parse and render under
    /// the real template (the tokenizer testdata copy is byte-identical to
    /// the model's), with each mid-list system row demoted to a marked
    /// user turn — exact marker counts prove rows were demoted, not
    /// dropped.
    #[test]
    fn captured_claude_code_bodies_render_with_the_real_qwen_template() {
        let mut template = crate::tokenizer::ChatTemplate::from_str(include_str!(
            "../../testdata/chat_template.jinja"
        ))
        .expect("real qwen template");
        // Forced off: this pin asserts the flags-off (no-wrap) marker counts
        // and must not read the ambient environment.
        template.force_flags_off();
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/server/testdata/anthropic");
        // Turn-1 captures carry one mid-list system row; turn-2+ carry two.
        let expected_markers = [
            ("cc-plain", 1),
            ("cc-tools", 2),
            ("cc-history", 2),
            ("cc-thinktools", 2),
            ("cc-midconv", 2),
        ];
        for (name, markers) in expected_markers {
            let bytes =
                std::fs::read(dir.join(format!("{name}.request.json"))).expect("fixture read");
            let request = parse_anthropic(&bytes).expect("fixture parses");
            let text = template
                .render(
                    &request.input.messages,
                    &request.input.tools,
                    request.input.thinking,
                )
                .unwrap_or_else(|error| panic!("{name} must render: {error:?}"));
            assert_eq!(
                text.matches("<|im_start|>user\n[system] ").count(),
                markers,
                "{name}: demoted marker count"
            );
        }
    }

    // ── count_tokens: integration tests ────────────────────────────────────

    /// Posts to the count_tokens endpoint using the testkit helpers.
    async fn post_count_tokens(app: &Router, body: &Value) -> (StatusCode, Value) {
        let (status, _headers, bytes) =
            testkit::send(app, testkit::post_json("/v1/messages/count_tokens", body)).await;
        let body = if bytes.is_empty() {
            Value::Null
        } else {
            testkit::json_of(&bytes)
        };
        (status, body)
    }

    /// Posts to the count_tokens endpoint with `x-api-key` instead of the
    /// bearer header.
    async fn post_count_tokens_with_api_key(app: &Router, body: &Value) -> (StatusCode, Value) {
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/messages/count_tokens")
            .header("x-api-key", testkit::API_KEY)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("request");
        let (status, _headers, bytes) = testkit::send(app, request).await;
        let body = if bytes.is_empty() {
            Value::Null
        } else {
            testkit::json_of(&bytes)
        };
        (status, body)
    }

    #[tokio::test]
    async fn count_tokens_without_max_tokens_returns_exact_token_count() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine, 4_096);
        // Build the same body count_tokens receives: model + messages only.
        let body = json!({"model": "model-a", "messages": [{"role": "user", "content": "hi"}]});
        let (status, reply) = post_count_tokens(&app, &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        crate::server::contract::assert_schema("CountTokensResponse", &reply);
        // Compute expected count by encoding the same messages.
        let entry = testkit::entry("e1", "model-a", &engine, 4_096);
        let bundle = entry.bundle;
        let messages: Vec<crate::tokenizer::PromptMessage> =
            vec![crate::tokenizer::PromptMessage {
                role: crate::tokenizer::Role::User,
                content: Some("hi".to_string()),
                reasoning_content: None,
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            }];
        let expected = bundle
            .encode_prompt(&messages, &[], crate::tokenizer::Thinking::Default)
            .expect("encode")
            .len() as u64;
        assert_eq!(reply["input_tokens"], expected);
    }

    #[tokio::test]
    async fn count_tokens_with_max_tokens_ignored_returns_same_count() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine, 4_096);
        let body = json!({
            "model": "model-a",
            "max_tokens": 999,
            "messages": [{"role": "user", "content": "hi"}]
        });
        let (status, reply) = post_count_tokens(&app, &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        crate::server::contract::assert_schema("CountTokensResponse", &reply);
        // Same body without max_tokens should give the same count.
        let body_no_max: Value =
            json!({"model": "model-a", "messages": [{"role": "user", "content": "hi"}]});
        let (status2, reply2) = post_count_tokens(&app, &body_no_max).await;
        assert_eq!(status, status2);
        assert_eq!(reply["input_tokens"], reply2["input_tokens"]);
    }

    #[tokio::test]
    async fn count_tokens_authenticates_with_x_api_key() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine, 4_096);
        let body = json!({"model": "model-a", "messages": [{"role": "user", "content": "hi"}]});
        let (status, _) = post_count_tokens_with_api_key(&app, &body).await;
        assert_eq!(status, StatusCode::OK);
        // Wrong key: 401.
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/messages/count_tokens")
            .header("x-api-key", "sk-wrong")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({"model": "model-a", "messages": [{"role": "user", "content": "hi"}]})
                    .to_string(),
            ))
            .expect("request");
        let (status, _headers, bytes) = testkit::send(&app, request).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let body = testkit::json_of(&bytes);
        assert_eq!(body["error"]["type"], "authentication_error");
        // No key: 401.
        let request = testkit::post_bytes(
            "/v1/messages/count_tokens",
            None,
            json!({"model": "model-a", "messages": [{"role": "user", "content": "hi"}]})
                .to_string()
                .into_bytes(),
        );
        let (status, _headers, bytes) = testkit::send(&app, request).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let body = testkit::json_of(&bytes);
        assert_eq!(body["error"]["type"], "authentication_error");
    }

    #[tokio::test]
    async fn count_tokens_unknown_model_gives_not_found() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine, 4_096);
        let body = json!({
            "model": "unknown-model",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let (status, reply) = post_count_tokens(&app, &body).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(reply["error"]["type"], "not_found_error");
        crate::server::contract::assert_schema("AnthropicError", &reply);
    }

    #[tokio::test]
    async fn count_tokens_empty_messages_returns_400() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine, 4_096);
        let body = json!({"model": "model-a", "messages": []});
        let (status, reply) = post_count_tokens(&app, &body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(reply["error"]["type"], "invalid_request_error");
        crate::server::contract::assert_schema("AnthropicError", &reply);
    }

    #[tokio::test]
    async fn count_tokens_stream_true_stays_a_json_reply() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine, 4_096);
        let body = json!({
            "model": "model-a",
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        });
        let (status, headers, bytes) =
            testkit::send(&app, testkit::post_json("/v1/messages/count_tokens", &body)).await;
        assert_eq!(status, StatusCode::OK);
        // count_tokens never streams: the reply stays JSON, never the SSE media type.
        let content_type = headers
            .get(header::CONTENT_TYPE)
            .expect("content type")
            .to_str()
            .expect("text");
        assert!(content_type.starts_with("application/json"));
        let reply = testkit::json_of(&bytes);
        crate::server::contract::assert_schema("CountTokensResponse", &reply);
    }

    #[tokio::test]
    async fn count_tokens_max_tokens_zero_is_validated_then_ignored() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine, 4_096);
        // A provided max_tokens is validated like on /v1/messages before it is
        // ignored: zero is under the minimum of 1.
        let body = json!({
            "model": "model-a",
            "max_tokens": 0,
            "messages": [{"role": "user", "content": "hi"}]
        });
        let (status, reply) = post_count_tokens(&app, &body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
        assert_eq!(reply["error"]["type"], "invalid_request_error");
        crate::server::contract::assert_schema("AnthropicError", &reply);
    }

    #[tokio::test]
    async fn count_tokens_body_over_the_cap_is_request_too_large() {
        let engine = testkit::engine(Vec::new());
        let app = app(&engine, 4_096);
        let big = format!(
            "{{\"model\":\"model-a\",\"messages\":[{{\"role\":\"user\",\"content\":\"{}\"}}]}}",
            "x".repeat(17 * 1024 * 1024)
        );
        let (status, _headers, bytes) = testkit::send(
            &app,
            testkit::post_bytes(
                "/v1/messages/count_tokens",
                Some(testkit::API_KEY),
                big.into_bytes(),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(error_type(&testkit::json_of(&bytes)), "request_too_large");
    }
}
