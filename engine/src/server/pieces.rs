//! Maps the events of the tool-call and think-block parser to the parts of a reply.
//!
//! Contract: the server design ("Streaming format"). The parser in `tokenizer::toolcall`
//! (decision 0025) owns the grammar: the think and call markers, the trimming of sections, the
//! types of the arguments, and the check of a call body. This module owns only what the reply
//! does with each parser event:
//!
//! - Reasoning and content events become reasoning and content parts.
//! - A complete call becomes a call part. When the request lists no tool, the call becomes
//!   content text instead, because the request did not allow it. A call to a function that is
//!   not in the tool list of the request is still a call. A `number` argument written `2` stays
//!   `2` in the call arguments (decision 0025, TC1 behavior).
//! - A malformed call stays in the content as `<tool_call>`, the raw body, and `</tool_call>`.
//! - A call left open at the end stays in the content as `<tool_call>` and the raw body. The
//!   `length` limit drops it instead.

use crate::tokenizer::ToolSpec;
use crate::tokenizer::toolcall::{
    MalformedCall, Start, ToolCall, ToolCallError, ToolCallParser, ToolEvent, ToolSchemas,
};
use serde_json::Value;

/// One part of the reply, in order. The adapter never returns an empty text part.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Piece {
    /// Answer text.
    Content(String),
    /// Think text, without the tags.
    Reasoning(String),
    /// A completed tool call. Its id is the call prefix followed by its index. The function
    /// can be absent from the request's tool list. A `number` argument written `2` stays `2`.
    Call(ToolCall),
}

/// Feeds the decoded text of one reply to the parser and returns reply parts.
/// Feed the text of each step with `push` and end the reply with `finish`.
pub(crate) struct ReplyParser {
    parser: ToolCallParser,
    allow_calls: bool,
}

impl ReplyParser {
    /// Creates the parser of one reply. `start` is `Start::Reasoning` when the prompt ended
    /// inside a think block. `tools` are the tools of the request. `call_prefix` starts every
    /// call id.
    pub(crate) fn new(start: Start, tools: &[ToolSpec], call_prefix: &str) -> Self {
        Self {
            parser: ToolCallParser::new(start, ToolSchemas::from_tools(tools), call_prefix),
            allow_calls: !tools.is_empty(),
        }
    }

    /// Takes the next text of the reply and returns the parts that it completes.
    pub(crate) fn push(&mut self, text: &str) -> Vec<Piece> {
        let events = self.parser.push(text);
        self.convert(events, false)
    }

    /// Ends the reply and returns the parts that remain. `cut_by_length` is true when the output
    /// limit ended the reply: a call left open is then dropped.
    pub(crate) fn finish(&mut self, cut_by_length: bool) -> Vec<Piece> {
        let events = self.parser.finish();
        self.convert(events, cut_by_length)
    }

    fn convert(&self, events: Vec<ToolEvent>, cut_by_length: bool) -> Vec<Piece> {
        events
            .into_iter()
            .filter_map(|event| match event {
                ToolEvent::Reasoning(text) => Some(Piece::Reasoning(text)),
                ToolEvent::Content(text) => Some(Piece::Content(text)),
                ToolEvent::Call(call) if self.allow_calls => Some(Piece::Call(call)),
                ToolEvent::Call(call) => Some(Piece::Content(call_text(&call))),
                ToolEvent::Malformed(MalformedCall {
                    error: ToolCallError::Unterminated,
                    raw,
                }) => (!cut_by_length).then(|| Piece::Content(format!("<tool_call>{raw}"))),
                ToolEvent::Malformed(MalformedCall { raw, .. }) => {
                    Some(Piece::Content(format!("<tool_call>{raw}</tool_call>")))
                }
            })
            .collect()
    }
}

/// Writes a call as text in the form of the chat template. A request without tools allows no
/// call, so the reply keeps the call as content. The text can differ from the model output in
/// whitespace, because the parser keeps no raw text of a valid call. The template renders a
/// non-string value with `tojson`. This function writes the compact `serde_json` form, so the
/// separators of non-string values can differ from the template output too.
fn call_text(call: &ToolCall) -> String {
    let mut text = format!("<tool_call>\n<function={}>\n", call.name);
    if let Ok(Value::Object(arguments)) = serde_json::from_str::<Value>(&call.arguments) {
        for (key, value) in arguments {
            let value = match value {
                Value::String(value) => value,
                other => other.to_string(),
            };
            text.push_str(&format!("<parameter={key}>\n{value}\n</parameter>\n"));
        }
    }
    text.push_str("</function>\n</tool_call>");
    text
}

#[cfg(test)]
mod tests {
    use super::{Piece, ReplyParser};
    use crate::tokenizer::ToolSpec;
    use crate::tokenizer::toolcall::{Start, ToolCall};
    use serde_json::{Value, json};

    const PREFIX: &str = "call_0000000000000001_";

    fn tools() -> Vec<ToolSpec> {
        vec![ToolSpec(json!({"type": "function", "function": {
            "name": "add",
            "parameters": {"type": "object", "properties": {
                "a": {"type": "integer"}, "b": {"type": "number"},
                "note": {"type": "string"}}}}}))]
    }

    /// Feeds the pieces of text one by one, ends the reply, and joins neighbours of one kind.
    fn run(start: Start, tools: &[ToolSpec], pushes: &[&str], cut_by_length: bool) -> Vec<Piece> {
        let mut parser = ReplyParser::new(start, tools, PREFIX);
        let mut pieces = Vec::new();
        for text in pushes {
            pieces.extend(parser.push(text));
        }
        pieces.extend(parser.finish(cut_by_length));
        let mut joined: Vec<Piece> = Vec::new();
        for piece in pieces {
            match (joined.last_mut(), piece) {
                (Some(Piece::Content(last)), Piece::Content(text)) => last.push_str(&text),
                (Some(Piece::Reasoning(last)), Piece::Reasoning(text)) => last.push_str(&text),
                (_, piece) => joined.push(piece),
            }
        }
        joined
    }

    fn content(text: &str) -> Piece {
        Piece::Content(text.to_string())
    }

    fn reasoning(text: &str) -> Piece {
        Piece::Reasoning(text.to_string())
    }

    fn call(index: u32, name: &str, arguments: Value) -> Piece {
        Piece::Call(ToolCall {
            id: format!("{PREFIX}{index}"),
            name: name.to_string(),
            arguments: arguments.to_string(),
        })
    }

    const ADD: &str = "<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2.5\n</parameter>\n</function>\n</tool_call>";

    #[test]
    fn plain_text_passes_through_as_content() {
        let pieces = run(Start::Content, &tools(), &["Hello ", "world."], false);
        assert_eq!(pieces, vec![content("Hello world.")]);
    }

    #[test]
    fn a_reply_that_starts_inside_a_think_block_ends_it_at_the_close_marker() {
        let pieces = run(Start::Reasoning, &tools(), &["hmm</think>ok"], false);
        assert_eq!(pieces, vec![reasoning("hmm"), content("ok")]);
    }

    #[test]
    fn a_think_block_in_the_text_goes_to_the_reasoning_without_its_tags() {
        let pieces = run(
            Start::Content,
            &tools(),
            &["<think>hmm</think>answer"],
            false,
        );
        assert_eq!(pieces, vec![reasoning("hmm"), content("answer")]);
    }

    #[test]
    fn a_complete_call_becomes_a_call_with_a_prefixed_id_and_typed_arguments() {
        let pieces = run(Start::Content, &tools(), &[ADD], false);
        assert_eq!(pieces, vec![call(0, "add", json!({"a": 1, "b": 2.5}))]);
    }

    #[test]
    fn the_tools_of_the_request_type_the_arguments() {
        let body = "<tool_call>\n<function=add>\n<parameter=a>\n7\n</parameter>\n<parameter=note>\n42\n</parameter>\n</function>\n</tool_call>";
        let pieces = run(Start::Content, &tools(), &[body], false);
        assert_eq!(pieces, vec![call(0, "add", json!({"a": 7, "note": "42"}))]);
    }

    #[test]
    fn call_ids_count_up_in_the_order_of_the_calls() {
        let text = format!("{ADD}{ADD}");
        let pieces = run(Start::Content, &tools(), &[&text], false);
        assert_eq!(
            pieces,
            vec![
                call(0, "add", json!({"a": 1, "b": 2.5})),
                call(1, "add", json!({"a": 1, "b": 2.5}))
            ]
        );
    }

    #[test]
    fn a_call_split_over_many_pushes_is_sent_once_and_only_when_complete() {
        let mut parser = ReplyParser::new(Start::Content, &tools(), PREFIX);
        let mut seen = Vec::new();
        for chunk in ADD.as_bytes().chunks(7) {
            seen.push(parser.push(std::str::from_utf8(chunk).expect("ascii")));
        }
        let last = seen.pop().expect("chunks");
        assert!(
            seen.iter().all(Vec::is_empty),
            "a call leaked early: {seen:?}"
        );
        assert_eq!(last, vec![call(0, "add", json!({"a": 1, "b": 2.5}))]);
        assert_eq!(parser.finish(false), Vec::new());
    }

    #[test]
    fn text_around_a_call_stays_in_order_without_the_edge_whitespace() {
        let text = format!("Adding. {ADD} Then more.");
        let pieces = run(Start::Content, &tools(), &[&text], false);
        assert_eq!(
            pieces,
            vec![
                content("Adding."),
                call(0, "add", json!({"a": 1, "b": 2.5})),
                content("Then more.")
            ]
        );
    }

    #[test]
    fn a_partial_marker_waits_for_the_next_text() {
        let mut parser = ReplyParser::new(Start::Content, &tools(), PREFIX);
        assert_eq!(parser.push("a<"), vec![content("a")]);
        assert_eq!(parser.push("b"), vec![content("<b")]);
    }

    #[test]
    fn a_partial_marker_at_the_end_is_released_by_finish() {
        let pieces = run(Start::Content, &tools(), &["ok <thi"], false);
        assert_eq!(pieces, vec![content("ok <thi")]);
    }

    #[test]
    fn a_malformed_call_stays_in_the_content_with_its_exact_body() {
        let broken = "<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n</tool_call>";
        let pieces = run(Start::Content, &tools(), &[broken], false);
        assert_eq!(pieces, vec![content(broken)]);
    }

    #[test]
    fn a_terminated_malformed_call_stays_in_the_content_at_the_length_limit() {
        let broken = "<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n</tool_call>";
        let pieces = run(Start::Content, &tools(), &[broken], true);
        assert_eq!(pieces, vec![content(broken)]);
    }

    #[test]
    fn a_call_left_open_at_another_end_stays_in_the_content() {
        let open = "<tool_call>\n<function=add>\n<parameter=a>\n1";
        let pieces = run(Start::Content, &tools(), &[open], false);
        assert_eq!(pieces, vec![content(open)]);
    }

    #[test]
    fn a_call_left_open_at_the_length_limit_is_dropped() {
        let open = "text <tool_call>\n<function=add>\n<parameter=a>\n1";
        let pieces = run(Start::Content, &tools(), &[open], true);
        assert_eq!(pieces, vec![content("text")]);
    }

    #[test]
    fn a_request_without_tools_keeps_a_call_as_content() {
        let pieces = run(Start::Content, &[], &[ADD], false);
        assert_eq!(pieces, vec![content(ADD)]);
    }

    #[test]
    fn a_call_kept_as_content_writes_string_arguments_unquoted() {
        let body = "<tool_call>\n<function=add>\n<parameter=note>\nhi there\n</parameter>\n</function>\n</tool_call>";
        let pieces = run(Start::Content, &[], &[body], false);
        assert_eq!(pieces, vec![content(body)]);
    }

    #[test]
    fn a_call_kept_as_content_writes_other_values_in_compact_json() {
        let flags = vec![ToolSpec(json!({"type": "function", "function": {
            "name": "set",
            "parameters": {"type": "object", "properties": {
                "on": {"type": "boolean"}, "opts": {"type": "object"}}}}}))];
        let body = "<tool_call>\n<function=set>\n<parameter=on>\ntrue\n</parameter>\n<parameter=opts>\n{ \"k\" : [1, 2] }\n</parameter>\n</function>\n</tool_call>";
        let pieces = run(Start::Content, &flags, &[body], false);
        assert_eq!(
            pieces,
            vec![call(0, "set", json!({"on": true, "opts": {"k": [1, 2]}}))]
        );
        let pieces = run(Start::Content, &[], &[body], false);
        let written = "<tool_call>\n<function=set>\n<parameter=on>\ntrue\n</parameter>\n<parameter=opts>\n{\"k\":[1,2]}\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(pieces, vec![content(written)]);
    }

    #[test]
    fn a_tool_call_inside_a_think_block_is_reasoning_text() {
        let text = format!("<think>{ADD}</think>done");
        let pieces = run(Start::Content, &tools(), &[&text], false);
        assert_eq!(pieces, vec![reasoning(ADD), content("done")]);
    }
}
