//! JSON shapes of the replies. The types are hand-written under decision 0043.
//!
//! Each type matches a schema of `api/openapi.yaml`, and the tests check the match.
//! The types are private to the server module.

use crate::sched::worker::EngineState;
use serde::Serialize;

/// The reason that a completion ended. It is the `finish_reason` of a choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FinishReason {
    /// End of sequence or a stop-sequence match.
    Stop,
    /// The `max_tokens` limit or the context limit.
    Length,
    /// The model completed at least one tool call.
    ToolCalls,
}

impl FinishReason {
    /// The wire name of the reason.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
        }
    }
}

/// The `Usage` schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
    reused_tokens: u32,
}

impl Usage {
    /// Builds the usage of one request. `total_tokens` is the saturating sum.
    pub(crate) fn new(prompt_tokens: u32, completion_tokens: u32, reused_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
            reused_tokens,
        }
    }

    /// The prompt tokens that stored state served.
    pub(crate) fn reused_tokens(&self) -> u32 {
        self.reused_tokens
    }
}

/// The `ToolCall` schema: one completed call of a non-streaming reply.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct ToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: ToolCallFunction,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct ToolCallFunction {
    name: String,
    arguments: String,
}

impl ToolCall {
    /// Builds a call. `arguments` is a JSON object as text.
    pub(crate) fn new(id: String, name: String, arguments: String) -> Self {
        Self {
            id,
            kind: "function",
            function: ToolCallFunction { name, arguments },
        }
    }
}

/// The parts of a finished reply.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct ReplyParts {
    /// The answer text, without think blocks and tool call markup.
    pub(crate) content: String,
    /// The text of the think blocks, without the tags.
    pub(crate) reasoning: String,
    /// The completed tool calls in order.
    pub(crate) tool_calls: Vec<ToolCall>,
}

#[derive(Serialize)]
struct ResponseMessage {
    role: &'static str,
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<ToolCall>,
}

#[derive(Serialize)]
struct Choice {
    index: u32,
    message: ResponseMessage,
    finish_reason: &'static str,
}

/// The `ChatCompletion` schema.
#[derive(Serialize)]
pub(crate) struct ChatCompletion {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    engine: String,
    choices: Vec<Choice>,
    usage: Usage,
}

impl ChatCompletion {
    /// Builds the reply of a non-streaming request. `content` is null when the reply holds tool
    /// calls and no text. `reasoning_content` is absent when the reply has no think block.
    pub(crate) fn new(
        head: &ReplyHead,
        parts: ReplyParts,
        finish: FinishReason,
        usage: Usage,
    ) -> Self {
        let content = if parts.content.is_empty() && !parts.tool_calls.is_empty() {
            None
        } else {
            Some(parts.content)
        };
        Self {
            id: head.id.clone(),
            object: "chat.completion",
            created: head.created,
            model: head.model.clone(),
            engine: head.engine.clone(),
            choices: vec![Choice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant",
                    content,
                    reasoning_content: (!parts.reasoning.is_empty()).then_some(parts.reasoning),
                    tool_calls: parts.tool_calls,
                },
                finish_reason: finish.as_str(),
            }],
            usage,
        }
    }
}

/// The fields that every reply and every chunk of one request share.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ReplyHead {
    pub(crate) id: String,
    pub(crate) created: u64,
    pub(crate) model: String,
    pub(crate) engine: String,
}

impl ReplyHead {
    /// The prefix of the tool call ids of this reply. It derives from the completion id.
    /// The tool-call parser appends the index of each call to it.
    pub(crate) fn call_prefix(&self) -> String {
        format!("call_{}_", self.id.trim_start_matches("chatcmpl-"))
    }
}

#[derive(Serialize)]
struct ToolCallDelta {
    index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    kind: Option<&'static str>,
    function: ToolCallFunctionDelta,
}

#[derive(Serialize)]
struct ToolCallFunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

#[derive(Default, Serialize)]
struct ChoiceDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<ToolCallDelta>,
}

#[derive(Serialize)]
struct ChunkChoice {
    index: u32,
    delta: ChoiceDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'static str>,
}

/// The `ChatCompletionChunk` schema.
#[derive(Serialize)]
pub(crate) struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    engine: String,
    choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

impl ChatCompletionChunk {
    /// The first chunk of a stream. It carries the role and sends at admission.
    pub(crate) fn role(head: &ReplyHead) -> Self {
        Self::build(
            head,
            ChoiceDelta {
                role: Some("assistant"),
                content: Some(String::new()),
                ..ChoiceDelta::default()
            },
            None,
            None,
        )
    }

    /// A chunk with answer text.
    pub(crate) fn content(head: &ReplyHead, text: String) -> Self {
        Self::build(
            head,
            ChoiceDelta {
                content: Some(text),
                ..ChoiceDelta::default()
            },
            None,
            None,
        )
    }

    /// A chunk with think text.
    pub(crate) fn reasoning(head: &ReplyHead, text: String) -> Self {
        Self::build(
            head,
            ChoiceDelta {
                reasoning_content: Some(text),
                ..ChoiceDelta::default()
            },
            None,
            None,
        )
    }

    /// The first entry of a tool call: `id`, `type`, and the function name.
    pub(crate) fn tool_call_start(head: &ReplyHead, index: u32, id: String, name: String) -> Self {
        Self::tool_call(
            head,
            ToolCallDelta {
                index,
                id: Some(id),
                kind: Some("function"),
                function: ToolCallFunctionDelta {
                    name: Some(name),
                    arguments: Some(String::new()),
                },
            },
        )
    }

    /// The argument entry of a tool call. `arguments` is a JSON object as text.
    pub(crate) fn tool_call_arguments(head: &ReplyHead, index: u32, arguments: String) -> Self {
        Self::tool_call(
            head,
            ToolCallDelta {
                index,
                id: None,
                kind: None,
                function: ToolCallFunctionDelta {
                    name: None,
                    arguments: Some(arguments),
                },
            },
        )
    }

    /// The final chunk. It has an empty delta, the finish reason, and the usage.
    pub(crate) fn last(head: &ReplyHead, finish: FinishReason, usage: Usage) -> Self {
        Self::build(
            head,
            ChoiceDelta::default(),
            Some(finish.as_str()),
            Some(usage),
        )
    }

    fn tool_call(head: &ReplyHead, call: ToolCallDelta) -> Self {
        Self::build(
            head,
            ChoiceDelta {
                tool_calls: vec![call],
                ..ChoiceDelta::default()
            },
            None,
            None,
        )
    }

    fn build(
        head: &ReplyHead,
        delta: ChoiceDelta,
        finish_reason: Option<&'static str>,
        usage: Option<Usage>,
    ) -> Self {
        Self {
            id: head.id.clone(),
            object: "chat.completion.chunk",
            created: head.created,
            model: head.model.clone(),
            engine: head.engine.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta,
                finish_reason,
            }],
            usage,
        }
    }
}

/// The `Engine` schema.
#[derive(Serialize)]
pub(crate) struct EngineInfo {
    pub(crate) id: String,
    pub(crate) model: String,
    pub(crate) state: &'static str,
}

/// The `Model` schema.
#[derive(Serialize)]
pub(crate) struct Model {
    pub(crate) id: String,
    object: &'static str,
    pub(crate) created: u64,
    owned_by: &'static str,
    pub(crate) engines: Vec<EngineInfo>,
}

impl Model {
    /// Builds a model entry. Every model of this server belongs to `local`.
    pub(crate) fn new(id: String, created: u64, engines: Vec<EngineInfo>) -> Self {
        Self {
            id,
            object: "model",
            created,
            owned_by: "local",
            engines,
        }
    }
}

/// The response of `GET /v1/models`.
#[derive(Serialize)]
pub(crate) struct ModelList {
    object: &'static str,
    pub(crate) data: Vec<Model>,
}

impl ModelList {
    /// Wraps the model entries.
    pub(crate) fn new(data: Vec<Model>) -> Self {
        Self {
            object: "list",
            data,
        }
    }
}

/// The `Engine.state` value of an engine state. The names follow `server.md`.
pub(crate) fn state_name(state: EngineState) -> &'static str {
    match state {
        EngineState::Loading => "loading",
        EngineState::Ready => "ready",
        EngineState::Busy => "busy",
        EngineState::Error => "error",
        EngineState::Stopped => "stopped",
        EngineState::NativeFailed => "native-failed",
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ChatCompletion, ChatCompletionChunk, EngineInfo, FinishReason, Model, ModelList, ReplyHead,
        ReplyParts, ToolCall, Usage, state_name,
    };
    use crate::sched::worker::EngineState;
    use crate::server::contract::{assert_pointer, assert_schema};
    use serde_json::{Value, json, to_value};

    fn head() -> ReplyHead {
        ReplyHead {
            id: "chatcmpl-0000000000000001".to_string(),
            created: 1_790_000_000,
            model: "fake-model".to_string(),
            engine: "e1".to_string(),
        }
    }

    fn value<T: serde::Serialize>(item: &T) -> Value {
        to_value(item).expect("serialize")
    }

    fn text_parts(text: &str) -> ReplyParts {
        ReplyParts {
            content: text.to_string(),
            ..ReplyParts::default()
        }
    }

    fn call() -> ToolCall {
        ToolCall::new(
            "call_1".to_string(),
            "add".to_string(),
            "{\"a\":1}".to_string(),
        )
    }

    #[test]
    fn the_tool_call_prefix_derives_from_the_completion_id() {
        assert_eq!(head().call_prefix(), "call_0000000000000001_");
    }

    #[test]
    fn usage_totals_prompt_and_completion_tokens() {
        let usage = value(&Usage::new(7, 5, 2));
        assert_eq!(
            usage,
            json!({"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12, "reused_tokens": 2})
        );
        assert_schema("Usage", &usage);
    }

    #[test]
    fn usage_total_saturates_instead_of_wrapping() {
        let usage = value(&Usage::new(u32::MAX, 1, 0));
        assert_eq!(usage["total_tokens"], u32::MAX);
    }

    #[test]
    fn a_completion_matches_the_contract_and_carries_the_text() {
        let completion = ChatCompletion::new(
            &head(),
            text_parts("hello"),
            FinishReason::Length,
            Usage::new(3, 1, 0),
        );
        let body = value(&completion);
        assert_schema("ChatCompletion", &body);
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["engine"], "e1");
        assert_eq!(body["choices"][0]["message"]["role"], "assistant");
        assert_eq!(body["choices"][0]["message"]["content"], "hello");
        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert_eq!(body["usage"]["total_tokens"], 4);
        let message = &body["choices"][0]["message"];
        assert!(message.get("reasoning_content").is_none(), "{message}");
        assert!(message.get("tool_calls").is_none(), "{message}");
    }

    #[test]
    fn a_completion_carries_the_reasoning_apart_from_the_content() {
        let parts = ReplyParts {
            content: "42".to_string(),
            reasoning: "think first".to_string(),
            tool_calls: Vec::new(),
        };
        let completion =
            ChatCompletion::new(&head(), parts, FinishReason::Stop, Usage::new(1, 1, 0));
        let body = value(&completion);
        assert_schema("ChatCompletion", &body);
        let message = &body["choices"][0]["message"];
        assert_eq!(message["content"], "42");
        assert_eq!(message["reasoning_content"], "think first");
    }

    #[test]
    fn a_completion_with_tool_calls_has_null_content_and_the_calls() {
        let parts = ReplyParts {
            content: String::new(),
            reasoning: String::new(),
            tool_calls: vec![call()],
        };
        let completion =
            ChatCompletion::new(&head(), parts, FinishReason::ToolCalls, Usage::new(1, 1, 0));
        let body = value(&completion);
        assert_schema("ChatCompletion", &body);
        let choice = &body["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls");
        assert_eq!(choice["message"]["content"], Value::Null);
        assert_eq!(
            choice["message"]["tool_calls"],
            json!([{"id": "call_1", "type": "function",
                    "function": {"name": "add", "arguments": "{\"a\":1}"}}])
        );
    }

    #[test]
    fn text_before_a_tool_call_stays_in_the_content() {
        let parts = ReplyParts {
            content: "Let me add.".to_string(),
            reasoning: String::new(),
            tool_calls: vec![call()],
        };
        let completion =
            ChatCompletion::new(&head(), parts, FinishReason::ToolCalls, Usage::new(1, 1, 0));
        assert_eq!(
            value(&completion)["choices"][0]["message"]["content"],
            "Let me add."
        );
    }

    #[test]
    fn the_role_chunk_carries_the_role_and_empty_content() {
        let chunk = value(&ChatCompletionChunk::role(&head()));
        assert_schema("ChatCompletionChunk", &chunk);
        assert_eq!(
            chunk["choices"][0]["delta"],
            json!({"role": "assistant", "content": ""})
        );
        assert_eq!(chunk["object"], "chat.completion.chunk");
        assert_eq!(chunk["id"], "chatcmpl-0000000000000001");
    }

    #[test]
    fn content_and_reasoning_chunks_omit_the_role_and_use_their_own_field() {
        let content = value(&ChatCompletionChunk::content(&head(), "llo".to_string()));
        let reasoning = value(&ChatCompletionChunk::reasoning(&head(), "hmm".to_string()));
        assert_schema("ChatCompletionChunk", &content);
        assert_schema("ChatCompletionChunk", &reasoning);
        assert_eq!(content["choices"][0]["delta"], json!({"content": "llo"}));
        assert_eq!(
            reasoning["choices"][0]["delta"],
            json!({"reasoning_content": "hmm"})
        );
    }

    #[test]
    fn a_text_chunk_has_neither_finish_reason_nor_usage() {
        let chunk = value(&ChatCompletionChunk::content(&head(), "x".to_string()));
        assert!(
            chunk["choices"][0].get("finish_reason").is_none(),
            "{chunk}"
        );
        assert!(chunk.get("usage").is_none(), "{chunk}");
    }

    #[test]
    fn a_tool_call_streams_as_a_start_entry_and_an_arguments_entry() {
        let start = value(&ChatCompletionChunk::tool_call_start(
            &head(),
            0,
            "call_1".to_string(),
            "add".to_string(),
        ));
        let arguments = value(&ChatCompletionChunk::tool_call_arguments(
            &head(),
            0,
            "{\"a\":1}".to_string(),
        ));
        assert_schema("ChatCompletionChunk", &start);
        assert_schema("ChatCompletionChunk", &arguments);
        assert_eq!(
            start["choices"][0]["delta"]["tool_calls"],
            json!([{"index": 0, "id": "call_1", "type": "function",
                    "function": {"name": "add", "arguments": ""}}])
        );
        assert_eq!(
            arguments["choices"][0]["delta"]["tool_calls"],
            json!([{"index": 0, "function": {"arguments": "{\"a\":1}"}}])
        );
    }

    #[test]
    fn the_last_chunk_has_an_empty_delta_the_finish_reason_and_the_usage() {
        let chunk = value(&ChatCompletionChunk::last(
            &head(),
            FinishReason::ToolCalls,
            Usage::new(3, 2, 0),
        ));
        assert_schema("ChatCompletionChunk", &chunk);
        assert_eq!(chunk["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(chunk["choices"][0]["delta"], json!({}));
        assert_eq!(chunk["usage"]["completion_tokens"], 2);
        assert_eq!(chunk["usage"]["total_tokens"], 5);
    }

    #[test]
    fn the_model_list_matches_the_models_response_schema() {
        let engines = vec![EngineInfo {
            id: "e1".to_string(),
            model: "fake-model".to_string(),
            state: state_name(EngineState::Ready),
        }];
        let list = ModelList::new(vec![Model::new(
            "fake-model".to_string(),
            1_790_000_000,
            engines,
        )]);
        let body = value(&list);
        assert_pointer(
            "/paths/~1v1~1models/get/responses/200/content/application~1json/schema",
            &body,
        );
        assert_eq!(body["object"], "list");
        assert_eq!(body["data"][0]["object"], "model");
        assert_eq!(body["data"][0]["owned_by"], "local");
        assert_eq!(body["data"][0]["engines"][0]["state"], "ready");
    }

    #[test]
    fn every_engine_state_has_a_name_that_the_contract_accepts() {
        let rows = [
            (EngineState::Loading, "loading"),
            (EngineState::Ready, "ready"),
            (EngineState::Busy, "busy"),
            (EngineState::Error, "error"),
            (EngineState::Stopped, "stopped"),
            (EngineState::NativeFailed, "native-failed"),
        ];
        for (state, name) in rows {
            assert_eq!(state_name(state), name);
            assert_schema("Engine", &json!({"id": "e1", "model": "m", "state": name}));
        }
    }
}
